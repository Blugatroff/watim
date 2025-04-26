from typing import List, Dict, Tuple, NoReturn, Sequence, assert_never
from dataclasses import dataclass
import os
import copy

from util import Ref, normalize_path
from format import Formattable, FormatInstr, named_record, format_seq, format_str, format_list, format_dict
from indexed_dict import IndexedDict
from lexer import Token
from parsing.types import I8, I32, I64, Bool, GenericType, HoleType
from parsing.parser import NumberWord, BreakWord
import parsing.parser as parser
from resolving.types import CustomTypeHandle, NamedType, Type, CustomTypeType, PtrType, FunctionType, TupleType, resolved_type_eq, resolved_types_eq, PtrType
from resolving.intrinsics import IntrinsicType, IntrinsicShr, IntrinsicEqual, IntrinsicStore, IntrinsicNot, IntrinsicUninit, IntrinsicSetStackSize, IntrinsicShr, IntrinsicRotr, IntrinsicGreater, IntrinsicGreaterEq, IntrinsicLess, IntrinsicLessEq, IntrinsicNotEqual, IntrinsicFlip, IntrinsicMemFill, INTRINSIC_TO_LEXEME, INTRINSICS, IntrinsicAdd, IntrinsicSub, IntrinsicDiv, IntrinsicDrop, IntrinsicMemGrow, IntrinsicMod, IntrinsicMul, IntrinsicAnd, IntrinsicOr, IntrinsicShl, IntrinsicWord, IntrinsicRotl, IntrinsicMemCopy
from resolving.words import CustomTypeHandle, ResolvedWord, FunctionHandle, StringWord, InitWord, RefWord, GetWord, StructFieldInitWord, CallWord, SizeofWord, UnnamedStructWord, StructWord, FunRefWord, StoreWord, LoadWord, MatchCase, MatchWord, VariantWord, LoopWord, CastWord, TupleMakeWord, TupleUnpackWord, GetFieldWord, IfWord, SetWord, BlockWord, IndirectCallWord, FieldAccess, Scope, ScopeId, GlobalId, LocalId
from resolving.top_items import Function, FunctionSignature, Import, CustomType, Global, Extern, Local, LocalName, ImportItem, Struct, Variant, VariantCase

@dataclass
class ResolverException(Exception):
    path: str
    file: str
    token: Token
    message: str

    def display(self) -> str:
        if self.token is None:
            lines = self.file.splitlines()
            line = len(lines) + 1
            column = len(lines[-1]) + 1 if len(lines) != 0 else 1
        else:
            line = self.token.line
            column = self.token.column
        return f"{self.path}:{line}:{column} {self.message}"

@dataclass
class Module(Formattable):
    path: str
    id: int
    imports: Dict[str, List[Import]]
    custom_types: IndexedDict[str, CustomType]
    globals: IndexedDict[str, Global]
    functions: IndexedDict[str, Function | Extern]
    data: bytes

    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Module", [
            ("imports", format_dict(self.imports, format_str, format_seq)),
            ("custom-types", self.custom_types.format_instrs(format_str)),
            ("globals", self.globals.format_instrs(format_str)),
            ("functions", self.functions.format_instrs(format_str))])

def determine_compilation_order(modules: Dict[str, List[parser.TopItem]]) -> IndexedDict[str, List[parser.TopItem]]:
    unprocessed = IndexedDict.from_items(modules.items())
    ordered: IndexedDict[str, List[parser.TopItem]] = IndexedDict()
    while len(unprocessed) > 0:
        i = 0
        while i < len(unprocessed):
            postpone = False
            module_path,top_items = list(unprocessed.items())[i]
            for top_item in top_items:
                if not isinstance(top_item, parser.Import):
                    continue
                imp: parser.Import = top_item
                if os.path.dirname(module_path) != "":
                    path = os.path.normpath(os.path.dirname(module_path) + "/" + imp.file_path.lexeme[1:-1])
                else:
                    path = os.path.normpath(imp.file_path.lexeme[1:-1])
                if "./"+path not in ordered.keys():
                    postpone = True
                    break
            if postpone:
                i += 1
                continue
            ordered[module_path] = top_items
            unprocessed.delete(i)
    return ordered

class Env:
    parent: 'Env | None'
    scope_counter: Ref[int]
    scope_id: ScopeId
    vars: Dict[str, List[Tuple[Local, LocalId]]]
    vars_by_id: Dict[LocalId, Local]

    def __init__(self, parent: 'Env | List[Local] | List[NamedType]'):
        if isinstance(parent, Env):
            self.parent = parent
        else:
            self.parent = None
        self.scope_counter = parent.scope_counter if isinstance(parent, Env) else Ref(0)
        self.scope_id = ScopeId(self.scope_counter.value)
        self.scope_counter.value += 1
        self.vars = {}
        self.vars_by_id = parent.vars_by_id if isinstance(parent, Env) else {}
        if isinstance(parent, list):
            for param in parent:
                if isinstance(param, Local):
                    self.insert(param)
                else:
                    self.insert(Local(LocalName(param.name), param.taip, False, True))


    def lookup(self, name: Token) -> Tuple[Local, LocalId] | None:
        if name.lexeme not in self.vars:
            if self.parent is not None:
                return self.parent.lookup(name)
            return None
        vars = self.vars[name.lexeme]
        if len(vars) == 0:
            if self.parent is not None:
                return self.parent.lookup(name)
            return None
        return vars[-1]

    def insert(self, var: Local) -> LocalId:
        if var.name.get() in self.vars:
            id = LocalId(var.name.get(), self.scope_id, len(self.vars[var.name.get()]))
            self.vars[var.name.get()].append((var, id))
            self.vars_by_id[id] = var
            return id
        id = LocalId(var.name.get(), self.scope_id, 0)
        self.vars[var.name.get()] = [(var, id)]
        self.vars_by_id[id] = var
        return id

    def get_var_type(self, id: LocalId) -> Type:
        return self.vars_by_id[id].taip

    def mark_var_as_reffed(self, id: LocalId):
        self.vars_by_id[id].was_reffed = True

    def child(self) -> 'Env':
        return Env(self)

@dataclass
class Stack:
    parent: 'Stack | None'
    stack: List[Type]
    negative: List[Type]

    @staticmethod
    def empty() -> 'Stack':
        return Stack(None, [], [])

    def append(self, taip: Type):
        self.push(taip)

    def push(self, taip: Type):
        self.stack.append(taip)

    def push_many(self, taips: List[Type]):
        for taip in taips:
            self.append(taip)

    def pop(self) -> Type | None:
        if len(self.stack) != 0:
            return self.stack.pop()
        if self.parent is None:
            return None
        taip = self.parent.pop()
        if taip is None:
            return None
        self.negative.append(taip)
        return taip

    def drop_n(self, n: int):
        for _ in range(n):
            self.pop()

    def pop_n(self, n: int) -> List[Type]:
        popped: List[Type] = []
        while n != 0:
            popped_type = self.pop()
            if popped_type is None:
                break
            popped.append(popped_type)
            n -= 1
        popped.reverse()
        return popped

    def clone(self) -> 'Stack':
        return Stack(self.parent.clone() if self.parent is not None else None, list(self.stack), list(self.negative))

    def dump(self) -> List[Type]:
        dump: List[Type] = []
        while True:
            t = self.pop()
            if t is None:
                dump.reverse()
                return dump
            dump.append(t)

    def make_child(self) -> 'Stack':
        return Stack(self.clone(), [], [])

    def apply(self, other: 'Stack'):
        for _ in other.negative:
            self.pop()
        for added in other.stack:
            self.append(added)

    def use(self, n: int):
        popped = []
        for _ in range(n):
            taip = self.pop()
            assert(taip is not None)
            popped.append(taip)
        for taip in reversed(popped):
            self.append(taip)

    def compatible_with(self, other: 'Stack') -> bool:
        if len(self) != len(other):
            return False
        self.use(len(other.stack))
        other.use(len(self.stack))
        negative_is_fine = resolved_types_eq(self.negative, other.negative)
        positive_is_fine = resolved_types_eq(self.stack, other.stack)
        return negative_is_fine and positive_is_fine

    def __len__(self) -> int:
        return len(self.stack) + (len(self.parent) if self.parent is not None else 0)

    def __getitem__(self, index: int) -> Type:
        if index > 0:
            return self.stack[index]
        if abs(index) <= len(self.stack):
            return self.stack[index]
        assert(self.parent is not None)
        return self.parent[index + len(self.stack)]

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, Stack):
            return False
        a = self
        ia = len(a.stack)
        ib = len(b.stack)
        while True:
            while ia == 0 and a.parent is not None:
                a = a.parent
                ia = len(a.stack)
            while ib == 0 and b.parent is not None:
                b = b.parent
                ib = len(b.stack)
            if (a is None) ^ (b is None):
                return False
            a_end = ia == 0 and a.parent is None
            b_end = ib == 0 and b.parent is None
            if a_end ^ b_end:
                return False
            if a_end and b_end:
                return True
            if not resolved_type_eq(a.stack[ia - 1], b.stack[ib - 1]):
                return False
            ib -= 1
            ia -= 1

    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Stack", [
            ("parent", self.parent),
            ("stack", format_list(self.stack)),
            ("negative", format_list(self.negative))])

@dataclass
class BreakStack:
    token: Token
    types: List[Type]
    reachable: bool

@dataclass
class StructLitContext:
    struct: CustomTypeHandle
    generic_arguments: List[Type]
    fields: Dict[str, Tuple[int, Type]]

@dataclass
class ResolveWordContext:
    env: Env
    break_stacks: List[BreakStack] | None
    block_returns: List[Type] | None
    reachable: bool
    struct_context: StructLitContext | None

    def with_env(self, env: Env) -> 'ResolveWordContext':
        return ResolveWordContext(env, self.break_stacks, self.block_returns, self.reachable, self.struct_context)

    def with_break_stacks(self, break_stacks: List[BreakStack], block_returns: List[Type] | None) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, break_stacks, block_returns, self.reachable, self.struct_context)

    def with_reachable(self, reachable: bool) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, self.break_stacks, self.block_returns, reachable, self.struct_context)

    def with_struct_context(self, struct_context: StructLitContext) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, self.break_stacks, self.block_returns, self.reachable, struct_context)

@dataclass
class BlockAnnotation:
    parameters: List[Type]
    returns: List[Type]


@dataclass
class TypeLookup:
    module: int
    types: List[CustomType]
    other_modules: List[List[CustomType]]

    def lookup(self, handle: CustomTypeHandle) -> CustomType:
        if handle.module == self.module:
            return self.types[handle.index]
        else:
            return self.other_modules[handle.module][handle.index]


    def types_pretty_bracketed(self, types: List[Type]) -> str:
        return f"[{self.types_pretty(types)}]"

    def types_pretty(self, types: List[Type]) -> str:
        s = ""
        for i, taip in enumerate(types):
            s += self.type_pretty(taip)
            if i + 1 < len(types):
                s += ", "
        return s

    def type_pretty(self, taip: Type) -> str:
        if isinstance(taip, I8):
            return "i8"
        if isinstance(taip, I32):
            return "i32"
        if isinstance(taip, I64):
            return "i64"
        if isinstance(taip, Bool):
            return "bool"
        if isinstance(taip, PtrType):
            return f".{self.type_pretty(taip.child)}"
        if isinstance(taip, CustomTypeType):
            s = self.lookup(taip.type_definition).name.lexeme
            if len(taip.generic_arguments) != 0:
                return f"{s}<{self.types_pretty(taip.generic_arguments)}>"
            return s
        if isinstance(taip, FunctionType):
            return f"({self.types_pretty(taip.parameters)} -> {self.types_pretty(taip.returns)})"
        if isinstance(taip, TupleType):
            return self.types_pretty_bracketed(taip.items)
        if isinstance(taip, GenericType):
            return taip.token.lexeme
        if isinstance(taip, HoleType):
            return taip.token.lexeme


@dataclass
class ResolveCtx:
    parsed_modules: IndexedDict[str, List[parser.TopItem]]
    resolved_modules: IndexedDict[str, Module]
    top_items: List[parser.TopItem]
    module_id: int
    static_data: bytearray

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolverException(self.parsed_modules.index_key(self.module_id), "", token, message)

    def resolve_imports(self) -> Dict[str, List[Import]]:
        resolved_imports: Dict[str, List[Import]] = {}
        module_path = list(self.parsed_modules.keys())[self.module_id]
        for top_item in self.top_items:
            if not isinstance(top_item, parser.Import):
                continue
            imp: parser.Import = top_item
            path = "" if module_path == "" else os.path.dirname(module_path)
            path = normalize_path(path + "/" + imp.file_path.lexeme[1:-1])
            imported_module_id = list(self.parsed_modules.keys()).index(path)
            items = self.resolve_import_items(imported_module_id, imp.items)
            if imp.qualifier.lexeme not in resolved_imports:
                resolved_imports[imp.qualifier.lexeme] = []
            resolved_imports[imp.qualifier.lexeme].append(Import(imp.token, path, imp.qualifier, imported_module_id, items))


        return resolved_imports

    def resolve_import_items(self, imported_module_id: int, items: Sequence[Token]) -> List[ImportItem]:
        imported_module = list(self.parsed_modules.values())[imported_module_id]
        def resolve_item(item_name: Token) -> ImportItem:
            item = ResolveCtx.lookup_item_in_module(imported_module, imported_module_id, item_name)
            if item is None:
                self.abort(item_name, "not found")
            return item
        return list(map(resolve_item, items))

    def resolve_custom_types(self, imports: Dict[str, List[Import]]) -> IndexedDict[str, CustomType]:
        resolved_custom_types: IndexedDict[str, CustomType] = IndexedDict()
        for top_item in self.top_items:
            if isinstance(top_item, parser.Struct):
                resolved_custom_types[top_item.name.lexeme] = self.resolve_struct(top_item, imports)
            if isinstance(top_item, parser.Variant):
                resolved_custom_types[top_item.name.lexeme] = self.resolve_variant(top_item, imports)
        return resolved_custom_types

    def resolve_struct(self, struct: parser.Struct, imports: Dict[str, List[Import]]) -> Struct:
        return Struct(
            struct.name,
            list(struct.generic_parameters),
            [self.resolve_named_type(imports, field) for field in struct.fields]
        )

    def resolve_variant(self, variant: parser.Variant, imports: Dict[str, List[Import]]) -> Variant:
        return Variant(
            variant.name,
            list(variant.generic_parameters),
            [VariantCase(
                case.name,
                None if case.taip is None else self.resolve_type(imports, case.taip)
            ) for case in variant.cases]
        )

    @staticmethod
    def lookup_item_in_module(module: List[parser.TopItem], module_id: int, name: Token) -> ImportItem | None:
        type_index = 0
        function_index = 0
        for top_item in module:
            if isinstance(top_item, parser.Global) or isinstance(top_item, parser.Import):
                continue
            if isinstance(top_item, parser.Struct) or isinstance(top_item, parser.Variant):
                if top_item.name.lexeme == name.lexeme:
                    return ImportItem(name, CustomTypeHandle(module_id, type_index))
                type_index += 1
                continue
            if isinstance(top_item, parser.Function):
                if top_item.signature.name.lexeme == name.lexeme:
                    return ImportItem(name, FunctionHandle(module_id, function_index))
                function_index += 1
                continue
            if isinstance(top_item, parser.Extern):
                if top_item.signature.name.lexeme == name.lexeme:
                    return ImportItem(name, FunctionHandle(module_id, function_index))
                function_index += 1
                continue
        return None

    def resolve_named_type(self, imports: Dict[str, List[Import]], named_type: parser.NamedType) -> NamedType:
        return NamedType(named_type.name, self.resolve_type(imports, named_type.taip))

    def resolve_named_types(self, imports: Dict[str, List[Import]], named_types: Sequence[parser.NamedType]) -> List[NamedType]:
        return [self.resolve_named_type(imports, named_type) for named_type in named_types]

    def resolve_type(self, imports: Dict[str, List[Import]], taip: parser.Type) -> Type:
        if isinstance(taip, parser.PtrType):
            return PtrType(self.resolve_type(imports, taip.child))
        if isinstance(taip, parser.CustomTypeType) or isinstance(taip, parser.ForeignType):
            return self.resolve_custom_type(imports, taip)
        if isinstance(taip, parser.FunctionType):
            return FunctionType(
                taip.token,
                self.resolve_types(imports, list(taip.parameters)),
                self.resolve_types(imports, list(taip.returns)),
            )
        if isinstance(taip, parser.TupleType):
            return TupleType(
                taip.token,
                self.resolve_types(imports, list(taip.items))
            )
        return taip

    def resolve_types(self, imports: Dict[str, List[Import]], types: Sequence[parser.Type]) -> List[Type]:
        return [self.resolve_type(imports, taip) for taip in types]

    def resolve_custom_type(self, imports: Dict[str, List[Import]], taip: parser.CustomTypeType | parser.ForeignType) -> CustomTypeType:
        type_index = 0
        generic_arguments = self.resolve_types(imports, list(taip.generic_arguments))
        if isinstance(taip, parser.CustomTypeType):
            for top_item in self.top_items:
                if isinstance(top_item, parser.Struct) or isinstance(top_item, parser.Variant):
                    if top_item.name.lexeme == taip.name.lexeme:
                        if len(top_item.generic_parameters) != len(generic_arguments):
                            self.generic_arguments_mismatch_error(taip.name, len(top_item.generic_parameters), len(generic_arguments))
                        return CustomTypeType(
                            taip.name,
                            CustomTypeHandle(self.module_id, type_index),
                            generic_arguments,
                        )
                    type_index += 1
            for imports_with_same_qualifier in imports.values():
                for imp in imports_with_same_qualifier:
                    for item in imp.items:
                        if isinstance(item.handle, CustomTypeHandle) and item.name.lexeme == taip.name.lexeme:
                            return CustomTypeType(taip.name, item.handle, generic_arguments)
            self.abort(taip.name, "type not found")
        if isinstance(taip, parser.ForeignType):
            for imp in imports[taip.module.lexeme]:
                module = self.resolved_modules.index(imp.module)
                for j,custom_type in enumerate(module.custom_types.values()):
                    if custom_type.name.lexeme == taip.name.lexeme:
                        assert(len(custom_type.generic_parameters) == len(generic_arguments))
                        return CustomTypeType(taip.name, CustomTypeHandle(imp.module, j), generic_arguments)
            assert(False)

    def resolve_globals(self, imports: Dict[str, List[Import]]) -> IndexedDict[str, Global]:
        globals: IndexedDict[str, Global] = IndexedDict()
        for top_item in self.top_items:
            if isinstance(top_item, parser.Global):
                globals[top_item.name.lexeme] = Global(
                    top_item.name,
                    self.resolve_type(imports, top_item.taip),
                    was_reffed=False,
                )
        return globals

    def resolve_signatures(self, imports: Dict[str, List[Import]]) -> List[FunctionSignature]:
        signatures = []
        for top_item in self.top_items:
            if not isinstance(top_item, parser.Function) and not isinstance(top_item, parser.Extern):
                continue
            signatures.append(self.resolve_signature(imports, top_item.signature))
        return signatures

    def resolve_signature(self, imports: Dict[str, List[Import]], signature: parser.FunctionSignature) -> FunctionSignature:
        return FunctionSignature(
            list(signature.generic_parameters),
            self.resolve_named_types(imports, signature.parameters),
            self.resolve_types(imports, signature.returns),
        )

    def forbid_directly_recursive_types(self, type_lookup: TypeLookup):
        for i in range(len(type_lookup.types)):
            handle = CustomTypeHandle(type_lookup.module, i)
            if self.is_directly_recursive(type_lookup, handle, []):
                token = type_lookup.lookup(handle).name
                self.abort(token, "structs and variants cannot be recursive")

    def is_directly_recursive(self, type_lookup: TypeLookup, handle: CustomTypeHandle, stack: List[CustomTypeHandle]) -> bool:
        if handle in stack:
            return True
        taip = type_lookup.lookup(handle)
        if isinstance(taip, Struct):
            for field in taip.fields:
                if isinstance(field.taip, CustomTypeType):
                    if self.is_directly_recursive(type_lookup, field.taip.type_definition, [handle] + stack):
                        return True
            return False
        if isinstance(taip, Variant):
            for case in taip.cases:
                if isinstance(case.taip, CustomTypeType):
                    if self.is_directly_recursive(type_lookup, case.taip.type_definition, [handle] + stack):
                        return True
            return False
        assert_never(taip)

    def generic_arguments_mismatch_error(self, token: Token, expected: int, actual: int):
        msg = f"expected {expected} generic arguments, not {actual}"
        self.abort(token, msg)

    def resolve_functions(
        self,
        imports: Dict[str, List[Import]],
        type_lookup: TypeLookup,
        signatures: List[FunctionSignature],
        globals: IndexedDict[str, Global]
    ) -> IndexedDict[str, Function | Extern]:
        functions: IndexedDict[str, Function | Extern] = IndexedDict()
        for top_item in self.top_items:
            if isinstance(top_item, parser.Function):
                function: parser.Function = top_item
                signature = self.resolve_signature(imports, function.signature)
                env = Env(list(map(Local.make_parameter, signature.parameters)))
                stack = Stack.empty()
                ctx = WordCtx(self, imports, env, type_lookup, signatures, globals)
                words, diverges = ctx.resolve_words(stack, list(function.body))
                if not diverges and not resolved_types_eq(stack.stack, signature.returns):
                    msg  = "unexpected return values:\n\texpected: "
                    msg += type_lookup.types_pretty_bracketed(signature.returns)
                    msg += "\n\tactual:   "
                    msg += type_lookup.types_pretty_bracketed(stack.stack)
                    self.abort(function.signature.name, msg)
                functions[function.signature.name.lexeme] = Function(
                    function.signature.name,
                    function.signature.export_name,
                    signature,
                    Scope(env.scope_id, words),
                    env.vars_by_id
                )
                continue
            if isinstance(top_item, parser.Extern):
                extern: parser.Extern = top_item
                functions[extern.signature.name.lexeme] = Extern(
                    extern.signature.name,
                    extern.module.lexeme,
                    extern.name.lexeme,
                    self.resolve_signature(imports, extern.signature),
                )
                continue
        return functions

    def allocate_static_data(self, data: bytes) -> int:
        offset = self.static_data.find(data)
        if offset == -1:
            offset = len(self.static_data)
            self.static_data.extend(data)
        return offset

@dataclass
class WordCtx:
    ctx: ResolveCtx
    imports: Dict[str, List[Import]]
    env: Env
    type_lookup: TypeLookup
    signatures: List[FunctionSignature]
    globals: IndexedDict[str, Global]
    struct_lit_ctx: StructLitContext | None = None
    break_stacks: List[BreakStack] | None = None
    block_returns: List[Type] | None = None
    reachable: bool = True

    def with_env(self, env: Env) -> 'WordCtx':
        new = copy.copy(self)
        new.env = env
        return new

    def with_break_stacks(self, break_stacks: List[BreakStack], block_returns: List[Type] | None) -> 'WordCtx':
        new = copy.copy(self)
        new.break_stacks = break_stacks
        new.block_returns = block_returns
        return new

    def with_struct_lit_ctx(self, ctx: StructLitContext) -> 'WordCtx':
        new = copy.copy(self)
        new.struct_lit_ctx = ctx
        return new

    def abort(self, token: Token, message: str) -> NoReturn:
        self.ctx.abort(token, message)

    def resolve_words(self, stack: Stack, remaining_words: List[parser.Word]) -> Tuple[List[ResolvedWord], bool]:
        diverges = False
        resolved: List[ResolvedWord] = []
        while len(remaining_words) != 0:
            parsed_word = remaining_words.pop(0)
            res = self.resolve_word(stack, remaining_words, parsed_word)
            if res is None:
                continue
            resolved_words,word_diverges = res
            diverges = diverges or word_diverges
            self.reachable = not diverges
            resolved.extend(resolved_words)
        return (resolved, diverges)

    def resolve_word(self, stack: Stack, remaining_words: List[parser.Word], word: parser.Word) -> Tuple[List[ResolvedWord], bool] | None:
        if isinstance(word, NumberWord):
            stack.push(I32())
            return ([word], False)
        if isinstance(word, parser.StringWord):
            stack.push(PtrType(I8()))
            stack.push(I32())
            offset = self.ctx.allocate_static_data(bytes(word.data))
            return ([StringWord(word.token, offset, len(word.data))], False)
        if isinstance(word, parser.GetWord):
            return self.resolve_get_local(stack, word)
        if isinstance(word, parser.RefWord):
            return self.resolve_ref_local(stack, word)
        if isinstance(word, parser.InitWord):
            return self.resolve_init_local(stack, word)
        if isinstance(word, parser.CallWord) or isinstance(word, parser.ForeignCallWord):
            return self.resolve_call(stack, word)
        if isinstance(word, parser.CastWord):
            return self.resolve_cast(stack, word)
        if isinstance(word, parser.SizeofWord):
            return self.resolve_sizeof(stack, word)
        if isinstance(word, parser.UnnamedStructWord):
            return self.resolve_make_struct(stack, word)
        if isinstance(word, parser.StructWord):
            return self.resolve_make_struct_named(stack, word)
        if isinstance(word, parser.FunRefWord):
            return self.resolve_fun_ref(stack, word)
        if isinstance(word, parser.IfWord):
            return self.resolve_if(stack, remaining_words, word)
        if isinstance(word, parser.LoopWord):
            return self.resolve_loop(stack, word)
        if isinstance(word, BreakWord):
            return self.resolve_break(stack, word.token)
        if isinstance(word, parser.SetWord):
            return self.resolve_set_local(stack, word)
        if isinstance(word, parser.BlockWord):
            return self.resolve_block(stack, word)
        if isinstance(word, parser.IndirectCallWord):
            return self.resolve_indirect_call(stack, word)
        if isinstance(word, parser.StoreWord):
            return self.resolve_store(stack, word)
        if isinstance(word, parser.LoadWord):
            return self.resolve_load(stack, word)
        if isinstance(word, parser.MatchWord):
            return self.resolve_match(stack, word)
        if isinstance(word, parser.VariantWord):
            return self.resolve_make_variant(stack, word)
        if isinstance(word, parser.GetFieldWord):
            return self.resolve_get_field(stack, word)
        if isinstance(word, parser.TupleMakeWord):
            return self.resolve_make_tuple(stack, word)
        if isinstance(word, parser.TupleUnpackWord):
            return self.resolve_unpack_tuple(stack, word)
        if isinstance(word, parser.StackAnnotation):
            self.resolve_stack_annotation(stack, word)
            return None
        if isinstance(word, parser.InlineRefWord):
            return self.resolve_ref(stack, word)

    def resolve_ref(self, stack: Stack, word: parser.InlineRefWord) -> Tuple[List[ResolvedWord], bool]:
        taip = stack.pop()
        assert(taip is not None)
        local = Local(LocalName("synth:ref"), taip, False, True)
        local_id = self.env.insert(local)
        stack.push(PtrType(taip))
        return ([
            InitWord(word.token, local_id, taip),
            RefWord(word.token, local_id, [])], False)

    def resolve_get_local(self, stack: Stack, word: parser.GetWord) -> Tuple[List[ResolvedWord], bool]:
        var_id,taip = self.resolve_var_name(word.ident)
        fields = self.resolve_field_accesses(taip, word.fields)
        resolved_type = taip if len(fields) == 0 else fields[-1].target_taip
        stack.push(resolved_type)
        return ([GetWord(word.ident, var_id, taip, fields, resolved_type)], False)

    def resolve_ref_local(self, stack: Stack, word: parser.RefWord) -> Tuple[List[ResolvedWord], bool]:
        local_and_id = self.env.lookup(word.ident)
        if local_and_id is not None:
            local,local_id = local_and_id
            def set_reffed():
                local.was_reffed = True
            taip = local.taip
            var_id: LocalId | GlobalId = local_id
        else:
            if word.ident.lexeme not in self.globals:
                self.abort(word.ident, f"var `{word.ident.lexeme}` not found")
            global_id = self.globals.index_of(word.ident.lexeme)
            globl = self.globals[word.ident.lexeme]
            def set_reffed():
                globl.was_reffed = True
            taip = globl.taip
            var_id = GlobalId(self.ctx.module_id, global_id)

        fields = self.resolve_field_accesses(taip, word.fields)

        i = 0
        while True:
            if i == len(word.fields):
                set_reffed()
                break
            if isinstance(fields[i].source_taip, PtrType):
                break
            i += 1

        result_type = taip if len(fields) == 0 else fields[-1].target_taip
        stack.push(PtrType(result_type))
        return ([RefWord(word.ident, var_id, fields)], False)

    def resolve_init_local(self, stack: Stack, word: parser.InitWord) -> Tuple[List[ResolvedWord], bool]:
        taip = stack.pop()
        assert(taip is not None)
        if self.struct_lit_ctx is not None:
            if word.ident.lexeme in self.struct_lit_ctx.fields:
                field_index,field_type = self.struct_lit_ctx.fields[word.ident.lexeme]
                field_type = self.insert_generic_arguments(self.struct_lit_ctx.generic_arguments, field_type)
                if not resolved_type_eq(field_type, taip):
                    self.abort(word.ident, "wrong type for field")
                del self.struct_lit_ctx.fields[word.ident.lexeme]
                return ([StructFieldInitWord(
                    word.ident,
                    self.struct_lit_ctx.struct,
                    field_type,
                    field_index,
                    self.struct_lit_ctx.generic_arguments)], False)

        local = Local(LocalName(word.ident), taip, False, False)
        local_id = self.env.insert(local)
        return ([InitWord(word.ident, local_id, taip)], False)

    def resolve_var_name(self, name: Token) -> Tuple[GlobalId | LocalId, Type]:
        local_and_id = self.env.lookup(name)
        if local_and_id is not None:
            local,local_id = local_and_id
            return (local_id, local.taip)
        if name.lexeme not in self.globals:
            self.abort(name, f"local {name.lexeme} not found")
        global_id = self.globals.index_of(name.lexeme)
        globl = self.globals[name.lexeme]
        return (GlobalId(self.ctx.module_id, global_id), globl.taip)

    def resolve_call(self, stack: Stack, word: parser.CallWord | parser.ForeignCallWord) -> Tuple[List[ResolvedWord], bool]:
        if word.ident.lexeme in INTRINSICS:
            resolved_generic_arguments = [self.ctx.resolve_type(self.imports, taip) for taip in word.generic_arguments]
            intrinsic = INTRINSICS[word.ident.lexeme]
            return ([self.resolve_intrinsic(word.ident, stack, intrinsic, resolved_generic_arguments)], False)
        resolved_word = self.resolve_call_word(word)
        signature = self.lookup_signature(resolved_word.function)
        args = stack.pop_n(len(signature.parameters))
        self.infer_generic_arguments_from_args(word.ident, args, signature.parameters, resolved_word.generic_arguments)
        self.push_returns(stack, signature.returns, resolved_word.generic_arguments)
        return ([resolved_word], False)

    def parameter_argument_mismatch_error(self, token: Token, arguments: List[Type], parameters: List[NamedType], generic_arguments: List[Type]) -> NoReturn:
        self.type_mismatch_error(
                token,
                [self.insert_generic_arguments(generic_arguments, parameter.taip) for parameter in parameters],
                arguments)

    def infer_generic_arguments_from_args(self, token: Token, arguments: List[Type], parameters: List[NamedType], generic_arguments: List[Type]) -> None:
        hole_mapping: Dict[Token, Type] = {}
        for i in range(1, len(parameters) + 1):
            if len(arguments) < i:
                self.parameter_argument_mismatch_error(token, arguments, parameters, generic_arguments)
            parameter = parameters[-i].taip
            argument = arguments[-i]
            parameter = self.insert_generic_arguments(generic_arguments, parameter)
            if not self.infer_holes(hole_mapping, token, argument, parameter):
                self.parameter_argument_mismatch_error(token, arguments, parameters, generic_arguments)

        for (i, generic_argument) in enumerate(generic_arguments):
            generic_arguments[i] = self.fill_holes(hole_mapping, generic_argument)

        # TODO: check that the generic arguments don't contain any holes now

    def infer_holes(self, mapping: Dict[Token, Type], token: Token, actual: Type, holey: Type) -> bool:
        assert not isinstance(actual, HoleType)
        match holey:
            case HoleType(hole):
                if hole in mapping and not resolved_type_eq(mapping[hole], actual):
                    msg = "Failed to infer type for hole, contradicting types inferred:\n"
                    msg += f"inferred now:        {self.type_lookup.type_pretty(actual)}\n"
                    msg += f"inferred previously: {self.type_lookup.type_pretty(mapping[hole])}\n"
                    self.abort(hole, msg)
                mapping[hole] = actual
                return True
            case Bool() | I8() | I32() | I64():
                return resolved_type_eq(actual, holey)
            case GenericType():
                return isinstance(actual, GenericType) and actual.generic_index == holey.generic_index
            case PtrType(holey):
                if not isinstance(actual, PtrType):
                    return False
                return self.infer_holes(mapping, token, actual.child, holey)
            case TupleType(_, holey_items):
                if not isinstance(actual, TupleType):
                    return False
                return self.infer_holes_all(mapping, token, actual.items, holey_items)
            case FunctionType():
                if not isinstance(actual, FunctionType):
                    return False
                return self.infer_holes_all(mapping, token, actual.parameters, holey.parameters) and self.infer_holes_all(mapping, token, actual.returns, holey.returns)
            case CustomTypeType():
                if not isinstance(actual, CustomTypeType):
                    return False
                if actual.type_definition != holey.type_definition:
                    return False
                return self.infer_holes_all(mapping, token, actual.generic_arguments, holey.generic_arguments)

    def infer_holes_all(self, mapping: Dict[Token, Type], token: Token, actual: List[Type], holey: List[Type]) -> bool:
        assert(len(actual) == len(holey))
        actuals_correct = True
        for (actual_t, holey_t) in zip(actual, holey):
            actuals_correct &= self.infer_holes(mapping, token, actual_t, holey_t)
        return actuals_correct


    def fill_holes(self, mapping: Dict[Token, Type], taip: Type) -> Type:
        match taip:
            case HoleType(hole):
                if hole not in mapping:
                    self.abort(hole, "failed to infer type for hole")
                return mapping[hole]
            case PtrType(child):
                return PtrType(self.fill_holes(mapping, child))
            case TupleType(token, items):
                return TupleType(token, [self.fill_holes(mapping, t) for t in items])
            case FunctionType(token, parameters, returns):
                return FunctionType(
                    token,
                    [self.fill_holes(mapping, t) for t in parameters],
                    [self.fill_holes(mapping, t) for t in returns])
            case CustomTypeType(name, type_definition, generic_arguments):
                return CustomTypeType(name, type_definition, [self.fill_holes(mapping, t) for t in generic_arguments])
            case other:
                return other

    def push_returns(self, stack: Stack, returns: List[Type], generic_arguments: List[Type] | None):
        for ret in returns:
            if generic_arguments is None:
                stack.push(ret)
            else:
                stack.push(self.insert_generic_arguments(generic_arguments, ret))

    def resolve_call_word(self, word: parser.CallWord | parser.ForeignCallWord) -> CallWord:
        if isinstance(word, parser.ForeignCallWord):
            resolved_generic_arguments = self.ctx.resolve_types(self.imports, word.generic_arguments)
            imports = self.imports[word.module.lexeme]
            for imp in imports:
                module = self.ctx.resolved_modules.index(imp.module)
                if word.ident.lexeme not in module.functions:
                    continue
                function_id = module.functions.index_of(word.ident.lexeme)
                signature = module.functions.index(function_id).signature
                if len(signature.generic_parameters) != len(resolved_generic_arguments):
                    self.ctx.generic_arguments_mismatch_error(word.ident, len(signature.generic_parameters), len(resolved_generic_arguments))
                return CallWord(word.ident, FunctionHandle(imp.module, function_id), resolved_generic_arguments)
            self.abort(word.ident, f"function `{word.ident.lexeme}` not found")
        resolved_generic_arguments = [self.ctx.resolve_type(self.imports, taip) for taip in word.generic_arguments]
        function = self.find_function(word.ident)
        if function is None:
            self.abort(word.ident, f"function `{word.ident.lexeme}` not found")
        assert(function is not None)
        signature = self.lookup_signature(function)
        if len(signature.generic_parameters) != len(resolved_generic_arguments):
            self.ctx.generic_arguments_mismatch_error(word.ident, len(signature.generic_parameters), len(resolved_generic_arguments))
        return CallWord(word.ident, function, resolved_generic_arguments)

    def resolve_cast(self, stack: Stack, word: parser.CastWord) -> Tuple[List[ResolvedWord], bool]:
        src = stack.pop()
        if src is None:
            self.abort(word.token, "cast expected a value, got []")
        dst = self.ctx.resolve_type(self.imports, word.taip)
        stack.push(dst)
        return ([CastWord(word.token, src, dst)], False)

    def resolve_sizeof(self, stack: Stack, word: parser.SizeofWord) -> Tuple[List[ResolvedWord], bool]:
        stack.push(I32())
        return ([SizeofWord(word.token, self.ctx.resolve_type(self.imports, word.taip))], False)

    def resolve_make_struct(self, stack: Stack, word: parser.UnnamedStructWord) -> Tuple[List[ResolvedWord], bool]:
        struct_type = self.ctx.resolve_custom_type(self.imports, word.taip)
        struc = self.type_lookup.lookup(struct_type.type_definition)
        assert(not isinstance(struc, Variant))
        args = stack.pop_n(len(struc.fields))
        self.infer_generic_arguments_from_args(word.token, args, struc.fields, struct_type.generic_arguments)
        # self.expect_arguments(stack, word.token, struct_type.generic_arguments, struc.fields)
        stack.push(struct_type)
        return ([UnnamedStructWord(word.token, struct_type)], False)

    def resolve_make_struct_named(self, stack: Stack, word: parser.StructWord) -> Tuple[List[ResolvedWord], bool]:
        struct_type = self.ctx.resolve_custom_type(self.imports, word.taip)
        struct = self.type_lookup.lookup(struct_type.type_definition)
        if isinstance(struct, Variant):
            self.abort(word.token, "can only make struct types, not variants")
        env = self.env.child()
        struct_lit_ctx = StructLitContext(
            struct_type.type_definition,
            struct_type.generic_arguments,
            { field.name.lexeme: (i,field.taip) for i,field in enumerate(struct.fields) })
        ctx = self.with_struct_lit_ctx(struct_lit_ctx).with_env(env)
        words,diverges = ctx.resolve_words(stack, list(word.words))
        if len(struct_lit_ctx.fields) != 0:
            error_message = "missing fields in struct literal:"
            for field_name,(_,field_type) in struct_lit_ctx.fields.items():
                error_message += f"\n\t{field_name}: {ctx.type_lookup.type_pretty(field_type)}"
            ctx.abort(word.token, error_message)
        stack.push(struct_type)
        return ([StructWord(word.token, struct_type, Scope(env.scope_id, words))], diverges)

    def resolve_fun_ref(self, stack: Stack, word: parser.FunRefWord) -> Tuple[List[ResolvedWord], bool]:
        call = self.resolve_call_word(word.call)
        signature = self.lookup_signature(call.function)
        parameters = [parameter.taip for parameter in signature.parameters]
        stack.push(FunctionType(call.name, parameters, signature.returns))
        return ([FunRefWord(call)], False)

    def resolve_if(self, stack: Stack, remaining_words: List[parser.Word], word: parser.IfWord) -> Tuple[List[ResolvedWord], bool]:
        if not isinstance(stack.pop(), Bool):
            self.abort(word.token, "expected a bool for `if`")
        true_env = self.env.child()
        true_stack = stack.make_child()
        true_ctx = self.with_env(true_env)

        false_env = self.env.child()
        false_stack = stack.make_child()
        false_ctx = self.with_env(false_env)

        true_words, true_words_diverge = true_ctx.resolve_words(true_stack, list(word.true_words.words))
        true_parameters = true_stack.negative

        if true_words_diverge and (word.false_words is None or len(word.false_words.words) == 0):
            remaining_stack = stack.make_child()
            remaining_stack.use(len(true_parameters))

            remaining_ctx = false_ctx

            resolved_remaining_words,remaining_words_diverge = remaining_ctx.resolve_words(
                    remaining_stack, remaining_words)

            stack.drop_n(len(remaining_stack.negative))
            stack.push_many(remaining_stack.stack)

            diverges = remaining_words_diverge
            return ([IfWord(
                word.token,
                list(remaining_stack.negative),
                None if diverges else list(remaining_stack.stack),
                Scope(true_ctx.env.scope_id, true_words),
                Scope(remaining_ctx.env.scope_id, resolved_remaining_words),
                diverges)], diverges)
        false_words, false_words_diverge = false_ctx.resolve_words(false_stack, [] if word.false_words is None else list(word.false_words.words))
        if not true_words_diverge and not false_words_diverge:
            if not true_stack.compatible_with(false_stack):
                msg  = "stack mismatch between if and else branch:\n\tif   "
                msg += self.type_lookup.types_pretty_bracketed(true_stack.stack)
                msg += "\n\telse "
                msg += self.type_lookup.types_pretty_bracketed(false_stack.stack)
                self.abort(word.token, msg)
            parameters = list(true_stack.negative)
        else:
            # TODO: Check, that the parameters of both branches are compatible
            parameters = true_stack.negative if len(true_stack.negative) > len(false_stack.negative) else false_stack.negative

        if not true_words_diverge:
            returns = true_stack.stack
        elif not false_words_diverge:
            returns = false_stack.stack
        else:
            returns = None

        param_count = max(len(true_stack.negative), len(false_stack.negative))
        if true_words_diverge and false_words_diverge:
            stack.drop_n(param_count)
        else:
            self.expect(stack, word.token, parameters)
            if returns is not None:
                stack.push_many(returns)

        diverges = true_words_diverge and false_words_diverge
        return ([IfWord(
            word.token,
            parameters,
            returns,
            Scope(true_ctx.env.scope_id, true_words),
            Scope(false_ctx.env.scope_id, false_words),
            diverges)], diverges)

    def resolve_loop(self, stack: Stack, word: parser.LoopWord) -> Tuple[List[ResolvedWord], bool]:
        annotation = None if word.annotation is None else self.resolve_block_annotation(word.annotation)
        loop_break_stacks: List[BreakStack] = []

        loop_env = self.env.child()
        loop_stack = stack.make_child()
        loop_ctx = self.with_env(loop_env).with_break_stacks(
                loop_break_stacks,
                None if annotation is None else annotation.returns)

        words,_ = loop_ctx.resolve_words(loop_stack, list(word.words.words))
        diverges = len(loop_break_stacks) == 0
        parameters = loop_stack.negative if annotation is None else annotation.parameters

        if len(loop_break_stacks) != 0:
            first = loop_break_stacks[0]
            diverges = not first.reachable
            for break_stack in loop_break_stacks[1:]:
                if not break_stack.reachable:
                    break
                if not resolved_types_eq(first.types, break_stack.types):
                    self.break_stack_mismatch_error(word.token, loop_break_stacks)

        if not resolved_types_eq(parameters, loop_stack.stack):
            self.abort(word.token, "unexpected values remaining on stack at the end of loop")

        if annotation is not None:
            returns = annotation.returns
        elif len(loop_break_stacks) != 0:
            returns = loop_break_stacks[0].types
        else:
            returns = loop_stack.stack

        self.expect(stack, word.token, parameters)
        stack.push_many(returns)
        body = Scope(loop_ctx.env.scope_id, words)
        return ([LoopWord(word.token, body, parameters, returns, diverges)], diverges)

    def resolve_break(self, stack: Stack, token: Token) -> Tuple[List[ResolvedWord], bool]:
        if self.block_returns is None:
            dump = stack.dump()
        else:
            dump = stack.pop_n(len(self.block_returns))

        if self.break_stacks is None:
            self.abort(token, "`break` can only be used inside of blocks and loops")

        self.break_stacks.append(BreakStack(token, dump, self.reachable))
        return ([BreakWord(token)], True)

    def resolve_set_local(self, stack: Stack, word: parser.SetWord) -> Tuple[List[ResolvedWord], bool]:
        var_id,taip = self.resolve_var_name(word.ident)
        fields = self.resolve_field_accesses(taip, word.fields)
        if len(fields) == 0:
            resolved_type = taip
        else:
            resolved_type = fields[-1].target_taip
        self.expect(stack, word.ident, [resolved_type])
        return ([SetWord(word.ident, var_id, fields)], False)

    def resolve_block(self, stack: Stack, word: parser.BlockWord) -> Tuple[List[ResolvedWord], bool]:
        annotation = None if word.annotation is None else self.resolve_block_annotation(word.annotation)
        block_break_stacks: List[BreakStack] = []

        block_env = self.env.child()
        block_stack = stack.make_child()
        block_ctx = self.with_env(block_env).with_break_stacks(
                block_break_stacks,
                None if annotation is None else annotation.returns)

        words, diverges = block_ctx.resolve_words(block_stack, list(word.words.words))
        block_end_is_reached = not diverges

        parameters = block_stack.negative if annotation is None else annotation.parameters
        if len(block_break_stacks) != 0:
            first = block_break_stacks[0]
            diverges = not first.reachable
            for break_stack in block_break_stacks[1:]:
                if not break_stack.reachable:
                    diverges = True
                    break
                if not resolved_types_eq(first.types, break_stack.types):
                    if block_end_is_reached:
                        block_break_stacks.append(BreakStack(word.words.end, block_stack.stack, diverges))
                    self.break_stack_mismatch_error(word.token, block_break_stacks)
            if block_end_is_reached:
                if not resolved_types_eq(block_stack.stack, first.types):
                    block_break_stacks.append(BreakStack(word.words.end, block_stack.stack, diverges))
                    self.break_stack_mismatch_error(word.token, block_break_stacks)

        if annotation is not None:
            returns = annotation.returns
        elif len(block_break_stacks) != 0:
            returns = block_break_stacks[0].types
        else:
            returns = block_stack.stack

        self.expect(stack, word.token, parameters)
        stack.push_many(returns)
        body = Scope(block_ctx.env.scope_id, words)
        return ([BlockWord(word.token, body, parameters, returns)], diverges)

    def resolve_indirect_call(self, stack: Stack, word: parser.IndirectCallWord) -> Tuple[List[ResolvedWord], bool]:
        fun_type = stack.pop()
        if fun_type is None:
            self.abort(word.token, "`->` expected a function on the stack, got: []")
        if not isinstance(fun_type, FunctionType):
            self.abort(word.token, "TODO")
        self.expect(stack, word.token, fun_type.parameters)
        self.push_returns(stack, fun_type.returns, None)
        return ([IndirectCallWord(word.token, fun_type)], False)

    def resolve_store(self, stack: Stack, word: parser.StoreWord) -> Tuple[List[ResolvedWord], bool]:
        var_id, taip = self.resolve_var_name(word.ident)
        fields = self.resolve_field_accesses(taip, word.fields)
        expected_type = taip if len(fields) == 0 else fields[-1].target_taip
        if not isinstance(expected_type, PtrType):
            self.abort(word.ident, "`=>` can only store into ptr types")
        expected_type = expected_type.child
        self.expect(stack, word.ident, [expected_type])
        return ([StoreWord(word.ident, var_id, fields)], False)

    def resolve_load(self, stack: Stack, word: parser.LoadWord) -> Tuple[List[ResolvedWord], bool]:
        taip = stack.pop()
        if taip is None:
            self.abort(word.token, "`~` expected a ptr, got: []")
        if not isinstance(taip, PtrType):
            msg = f"`~` expected a ptr, got: [{taip}]"
            self.abort(word.token, msg)
        stack.push(taip.child)
        return ([LoadWord(word.token, taip.child)], False)

    def resolve_match(self, stack: Stack, word: parser.MatchWord) -> Tuple[List[ResolvedWord], bool]:
        match_diverges = True
        arg_item = stack.pop()
        if arg_item is None:
            self.abort(word.token, "expected a value to match on")
        by_ref = isinstance(arg_item, PtrType)
        arg = arg_item.child if isinstance(arg_item, PtrType) else arg_item
        if not isinstance(arg, CustomTypeType):
            self.abort(word.token, "can only match n variants")
        generic_arguments = arg.generic_arguments
        variant_type = arg
        variant = self.type_lookup.lookup(arg.type_definition)
        if not isinstance(variant, Variant):
            self.abort(word.token, "can only match on variants")
        remaining_cases: List[str] = [case.name.lexeme for case in variant.cases]
        case_stacks: List[Tuple[Stack, Token, bool]] = []
        visited_cases: List[Token] = []
        cases: List[MatchCase] = []
        for parsed_case in word.cases:
            tag: int | None = None
            for j, variant_case in enumerate(variant.cases):
                if variant_case.name.lexeme == parsed_case.name.lexeme:
                    tag = j
            if tag is None:
                self.abort(parsed_case.name, "not part of variant")

            case_type = variant.cases[tag].taip
            case_stack = stack.make_child()
            if case_type is not None:
                if by_ref:
                    case_type = PtrType(case_type)
                case_type = self.insert_generic_arguments(generic_arguments, case_type)
                case_stack.push(case_type)

            case_env = self.env.child()
            case_ctx = self.with_env(case_env)
            words, case_diverges = case_ctx.resolve_words(case_stack, list(parsed_case.words))
            match_diverges = match_diverges and case_diverges
            cases.append(MatchCase(case_type, tag, Scope(case_env.scope_id, words)))

            if parsed_case.name.lexeme not in remaining_cases:
                other = next(token for token in visited_cases if token.lexeme == parsed_case.name.lexeme)
                msg  = "duplicate case in match:"
                msg += f"\n\t{other.line}:{other.column} {other.lexeme}"
                msg += f"\n\t{parsed_case.name.line}:{parsed_case.name.column} {parsed_case.name.lexeme}"
                self.abort(word.token, msg)

            remaining_cases.remove(parsed_case.name.lexeme)

            case_stacks.append((case_stack, parsed_case.name, case_diverges))
            visited_cases.append(parsed_case.name)

        if word.default is None:
            if len(remaining_cases) != 0:
                msg = "missing case in match:"
                for case in remaining_cases:
                    msg += f"\n\t{case}"
                self.abort(word.token, msg)
            default_case = None
        else:
            def_stack = stack.make_child()
            def_env = self.env.child()
            def_ctx = self.with_env(def_env)
            def_stack.push(arg_item)
            words, default_diverges = def_ctx.resolve_words(def_stack, list(word.default.words))
            match_diverges = match_diverges and default_diverges
            case_stacks.append((def_stack, word.default.name, default_diverges))
            default_case = Scope(def_env.scope_id, words)

        first_non_diverging_case: Stack | None = None
        for case_stack,case_token,case_diverges in case_stacks:
            if not case_diverges:
                if first_non_diverging_case is None:
                    first_non_diverging_case = case_stack
                elif not first_non_diverging_case.compatible_with(case_stack):
                    msg = "arms of match case have different types:"
                    for case_stack, case_token, _ in case_stacks:
                        msg += f"\n\t{self.type_lookup.types_pretty_bracketed(case_stack.negative)}"
                        msg += f" -> {self.type_lookup.types_pretty_bracketed(case_stack.stack)}"
                        msg += f" in case {case_token.lexeme}"
                    self.abort(word.token, msg)

        if len(case_stacks) == 0:
            returns: List[Type] | None = []
            parameters = []
        else:
            most_params = case_stacks[0][0]
            for case_stack,_,_ in case_stacks[1:]:
                if len(case_stack.negative) > len(most_params.negative):
                    most_params = case_stack

            parameters = list(most_params.negative)
            parameters.reverse()

            returns = list(parameters)
            for case_stack,_,case_diverges in case_stacks:
                if not case_diverges:
                    del returns[len(returns) - len(case_stack.negative):]
                    returns.extend(case_stack.stack)
                    break

        self.expect(stack, word.token, parameters)
        stack.push_many(returns or [])
        if match_diverges:
            returns = None
        return ([MatchWord(word.token, variant_type, by_ref, cases, default_case, parameters, returns)], match_diverges)

    def resolve_make_variant(self, stack: Stack, word: parser.VariantWord) -> Tuple[List[ResolvedWord], bool]:
        variant_type = self.ctx.resolve_custom_type(self.imports, word.taip)
        variant = self.type_lookup.lookup(variant_type.type_definition)
        if not isinstance(variant, Variant):
            self.abort(word.token, "can not make this type")
        tag: None | int = None
        for i,case in enumerate(variant.cases):
            if case.name.lexeme == word.case.lexeme:
                tag = i
        if tag is None:
            self.abort(word.token, "case is not part of variant")
        case = variant.cases[tag]
        if case.taip is not None:
            expected = self.insert_generic_arguments(variant_type.generic_arguments, case.taip)
            self.expect(stack, word.token, [expected])
        stack.push(variant_type)
        return ([VariantWord(word.token, tag, variant_type)], False)

    def resolve_get_field(self, stack: Stack, word: parser.GetFieldWord) -> Tuple[List[ResolvedWord], bool]:
        taip = stack.pop()
        if taip is None:
            self.abort(word.token, "expected a value on the stack")
        fields = self.resolve_field_accesses(taip, word.fields)
        on_ptr = isinstance(taip, PtrType)
        taip = fields[-1].target_taip
        taip = PtrType(taip) if on_ptr else taip
        stack.push(taip)
        return ([GetFieldWord(word.token, fields, on_ptr)], False)

    def resolve_make_tuple(self, stack: Stack, word: parser.TupleMakeWord) -> Tuple[List[ResolvedWord], bool]:
        num_items = int(word.item_count.lexeme)
        items: List[Type] = []
        for _ in range(num_items):
            item = stack.pop()
            if item is None:
                self.abort(word.token, "expected more")
            items.append(item)
        items.reverse()
        taip = TupleType(word.token, items)
        stack.push(taip)
        return ([TupleMakeWord(word.token, taip)], False)

    def resolve_unpack_tuple(self, stack: Stack, word: parser.TupleUnpackWord) -> Tuple[List[ResolvedWord], bool]:
        taip = stack.pop()
        if taip is None or not isinstance(taip, TupleType):
            self.abort(word.token, "expected a tuple on the stack")
        stack.push_many(taip.items)
        return ([TupleUnpackWord(word.token, taip)], False)

    def stack_annotation_mismatch(self, stack: Stack, annotation: parser.StackAnnotation) -> NoReturn:
        expected = self.ctx.resolve_types(self.imports, annotation.types)
        msg = "Stack doesn't match annotation:\n"
        msg += "  actual:   ["
        if len(stack) > len(expected) + 1:
            msg += "..., "
        actual = stack.pop_n(len(expected) + 1)
        msg += self.type_lookup.types_pretty(actual)
        msg += "]\n"
        msg += "  expected: "
        msg += self.type_lookup.types_pretty_bracketed(expected)
        msg += "\n"
        self.abort(annotation.token, msg)

    def resolve_stack_annotation(self, stack: Stack, word: parser.StackAnnotation) -> None:
        if len(stack) < len(word.types):
            self.stack_annotation_mismatch(stack, word)
        for i, taip in enumerate(reversed(word.types)):
            expected = self.ctx.resolve_type(self.imports, taip)
            if not resolved_type_eq(stack[-i-1], expected):
                self.stack_annotation_mismatch(stack, word)
        return None

    def resolve_block_annotation(self, annotation: parser.BlockAnnotation) -> BlockAnnotation:
        return BlockAnnotation(
                self.ctx.resolve_types(self.imports, annotation.parameters),
                self.ctx.resolve_types(self.imports, annotation.returns))

    def expect_arguments(self, stack: Stack, token: Token, generic_arguments: List[Type], parameters: List[NamedType]):
        i = len(parameters)
        popped: List[Type] = []
        while i != 0:
            expected_type = self.insert_generic_arguments(generic_arguments, parameters[i - 1].taip)
            popped_type = stack.pop()
            error = popped_type is None or not resolved_type_eq(popped_type, expected_type)
            if popped_type is not None:
                popped.append(popped_type)
            if error:
                while True:
                    popped_type = stack.pop()
                    if popped_type is None:
                        break
                    popped.append(popped_type)
                expected = [self.insert_generic_arguments(generic_arguments, parameter.taip) for parameter in parameters]
                popped.reverse()
                self.type_mismatch_error(token, expected, popped)
            assert(popped_type is not None)
            i -= 1

    def expect(self, stack: Stack, token: Token, expected: List[Type]):
        i = len(expected)
        popped: List[Type] = []
        while i != 0:
            expected_type = expected[i - 1]
            popped_type = stack.pop()
            error = popped_type is None or not resolved_type_eq(popped_type, expected_type)
            if error:
                popped.reverse()
                self.type_mismatch_error(token, expected, popped)
            assert(popped_type is not None)
            popped.append(popped_type)
            i -= 1

    def type_mismatch_error(self, token: Token, expected: List[Type], actual: List[Type]) -> NoReturn:
        message  = "expected:\n\t" + self.type_lookup.types_pretty_bracketed(expected)
        message += "\ngot:\n\t" + self.type_lookup.types_pretty_bracketed(actual)
        self.abort(token, message)

    def resolve_intrinsic(self, token: Token, stack: Stack, intrinsic: IntrinsicType, generic_arguments: List[Type]) -> IntrinsicWord:
        match intrinsic:
            case IntrinsicType.ADD | IntrinsicType.SUB:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if isinstance(taip, PtrType):
                    narrow_type: PtrType | I8 | I32 | I64 = taip
                    if not isinstance(stack[-1], I32):
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, i32]")
                    stack.pop()
                elif isinstance(taip, I8):
                    narrow_type = I8()
                    popped = self.expect_stack(token, stack, [I8(), I8()])
                    stack.append(taip)
                elif isinstance(taip, I32):
                    narrow_type = I32()
                    popped = self.expect_stack(token, stack, [I32(), I32()])
                    stack.append(taip)
                elif isinstance(taip, I64):
                    narrow_type = I64()
                    self.expect_stack(token, stack, [I64(), I64()])
                    stack.append(taip)
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]} cannot add to {self.type_lookup.type_pretty(taip)}")
                if intrinsic == IntrinsicType.ADD:
                    return IntrinsicAdd(token, narrow_type)
                if intrinsic == IntrinsicType.SUB:
                    return IntrinsicSub(token, narrow_type)
            case IntrinsicType.DROP:
                if len(stack) == 0:
                    self.abort(token, "`drop` expected non empty stack")
                stack.pop()
                return IntrinsicDrop(token)
            case IntrinsicType.MOD | IntrinsicType.MUL | IntrinsicType.DIV:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if isinstance(taip, I32):
                    narrow_type = I32()
                    popped = self.expect_stack(token, stack, [I32(), I32()])
                elif isinstance(taip, I64):
                    narrow_type = I64()
                    popped = self.expect_stack(token, stack, [I64(), I64()])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i32, i32] or [i64, i64] on stack")
                stack.append(taip)
                if intrinsic == IntrinsicType.MOD:
                    return IntrinsicMod(token, narrow_type)
                if intrinsic == IntrinsicType.MUL:
                    return IntrinsicMul(token, narrow_type)
                if intrinsic == IntrinsicType.DIV:
                    return IntrinsicDiv(token, narrow_type)
            case IntrinsicType.AND | IntrinsicType.OR:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                match taip:
                    case I8():
                        popped = self.expect_stack(token, stack, [I8(), I8()])
                    case I32():
                        popped = self.expect_stack(token, stack, [I32(), I32()])
                    case I64():
                        popped = self.expect_stack(token, stack, [I64(), I64()])
                    case Bool():
                        popped = self.expect_stack(token, stack, [Bool(), Bool()])
                    case _:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` can only and i8, i32, i64 and bool")
                stack.append(popped[0])
                if intrinsic == IntrinsicType.AND:
                    return IntrinsicAnd(token, taip)
                if intrinsic == IntrinsicType.OR:
                    return IntrinsicOr(token, taip)
            case IntrinsicType.SHR | IntrinsicType.SHL | IntrinsicType.ROTR | IntrinsicType.ROTL:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if isinstance(taip, I8):
                    popped = self.expect_stack(token, stack, [I8(), I8()])
                elif isinstance(taip, I32):
                    popped = self.expect_stack(token, stack, [I32(), I32()])
                else:
                    popped = self.expect_stack(token, stack, [I64(), I64()])
                stack.append(popped[0])
                if intrinsic == IntrinsicType.SHL:
                    return IntrinsicShl(token, taip)
                if intrinsic == IntrinsicType.SHR:
                    return IntrinsicShr(token, taip)
                if intrinsic == IntrinsicType.ROTR:
                    return IntrinsicRotr(token, taip)
                if intrinsic == IntrinsicType.ROTL:
                    return IntrinsicRotl(token, taip)
            case IntrinsicType.GREATER | IntrinsicType.LESS | IntrinsicType.GREATER_EQ | IntrinsicType.LESS_EQ:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                if isinstance(taip, I8):
                    narrow_type = I8()
                    self.expect_stack(token, stack, [I8(), I8()])
                elif isinstance(taip, I32):
                    narrow_type = I32()
                    self.expect_stack(token, stack, [I32(), I32()])
                elif isinstance(taip, I64):
                    narrow_type = I64()
                    self.expect_stack(token, stack, [I64(), I64()])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i8, i8] or [i32, i32] or [i64, i64] on stack")
                stack.append(Bool())
                if intrinsic == IntrinsicType.GREATER:
                    return IntrinsicGreater(token, narrow_type)
                if intrinsic == IntrinsicType.LESS:
                    return IntrinsicLess(token, narrow_type)
                if intrinsic == IntrinsicType.GREATER_EQ:
                    return IntrinsicGreaterEq(token, narrow_type)
                if intrinsic == IntrinsicType.LESS_EQ:
                    return IntrinsicLessEq(token, narrow_type)
            case IntrinsicType.MEM_COPY:
                self.expect_stack(token, stack, [PtrType(I8()), PtrType(I8()), I32()])
                return IntrinsicMemCopy(token)
            case IntrinsicType.MEM_FILL:
                self.expect_stack(token, stack, [PtrType(I32()), I32(), I32()])
                return IntrinsicMemFill(token)
            case IntrinsicType.NOT_EQ | IntrinsicType.EQ:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if not resolved_type_eq(stack[-1], stack[-2]):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [a, a] for any a")
                taip = stack[-1]
                stack.pop()
                stack.pop()
                stack.append(Bool())
                if intrinsic == IntrinsicType.EQ:
                    return IntrinsicEqual(token, taip)
                if intrinsic == IntrinsicType.NOT_EQ:
                    return IntrinsicNotEqual(token, taip)
            case IntrinsicType.FLIP:
                a = stack.pop()
                b = stack.pop()
                if a is None or b is None:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                stack.push_many([a, b])
                return IntrinsicFlip(token, b, a)
            case IntrinsicType.MEM_GROW:
                self.expect_stack(token, stack, [I32()])
                stack.append(I32())
                return IntrinsicMemGrow(token)
            case IntrinsicType.STORE:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                ptr_type = stack[-2]
                if not isinstance(ptr_type, PtrType) or not resolved_type_eq(ptr_type.child, stack[-1]):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, a]")
                taip = stack[-1]
                stack.pop()
                stack.pop()
                return IntrinsicStore(token, taip)
            case IntrinsicType.NOT:
                taip = stack[-1]
                if len(stack) == 0 or (not isinstance(taip, I64) and not isinstance(taip, I32) and not isinstance(taip, Bool)):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected an i64, i32 or bool on the stack")
                return IntrinsicNot(token, taip)
            case IntrinsicType.UNINIT:
                if len(generic_arguments) != 1:
                    self.abort(token, "uninit only accepts one generic argument")
                stack.append(generic_arguments[0])
                return IntrinsicUninit(token, generic_arguments[0])
            case IntrinsicType.SET_STACK_SIZE:
                self.expect_stack(token, stack, [I32()])
                return IntrinsicSetStackSize(token)

    def expect_stack(self, token: Token, stack: Stack, expected: List[Type]) -> List[Type]:
        popped: List[Type] = []
        def abort() -> NoReturn:
            stackdump = stack.dump() + list(reversed(popped))
            self.abort(token, f"expected:\n\t{self.type_lookup.types_pretty_bracketed(expected)}\ngot:\n\t{self.type_lookup.types_pretty_bracketed(stackdump)}")
        for expected_type in reversed(expected):
            top = stack.pop()
            if top is None:
                abort()
            popped.append(top)
            if not resolved_type_eq(expected_type, top):
                abort()
        return list(reversed(popped))

    def break_stack_mismatch_error(self, token: Token, break_stacks: List[BreakStack]):
        msg = "break stack mismatch:"
        for break_stack in break_stacks:
            msg += f"\n\t{break_stack.token.line}:{break_stack.token.column} {self.type_lookup.types_pretty_bracketed(break_stack.types)}"
        self.abort(token, msg)

    def find_function(self, name: Token) -> FunctionHandle | None:
        function_index = 0
        for top_item in self.ctx.top_items:
            if isinstance(top_item, parser.Function) or isinstance(top_item, parser.Extern):
                if top_item.signature.name.lexeme == name.lexeme:
                    return FunctionHandle(self.ctx.module_id, function_index)
                function_index += 1
        for imps in self.imports.values():
            for imp in imps:
                for item in imp.items:
                    if item.name.lexeme == name.lexeme:
                        if isinstance(item.handle, FunctionHandle):
                            return item.handle
        return None

    def lookup_signature(self, function: FunctionHandle) -> FunctionSignature:
        if function.module == self.ctx.module_id:
            return self.signatures[function.index]
        return self.ctx.resolved_modules.index(function.module).functions.index(function.index).signature

    def resolve_field_accesses(self, taip: Type, fields: Sequence[Token]) -> List[FieldAccess]:
        resolved = []
        if len(fields) == 0:
            return []
        for field_name in fields:
            assert((isinstance(taip, PtrType) and isinstance(taip.child, CustomTypeType)) or isinstance(taip, CustomTypeType))
            unpointered = taip.child if isinstance(taip, PtrType) else taip
            assert(isinstance(unpointered, CustomTypeType))
            custom_type: CustomTypeType = unpointered
            type_definition = self.type_lookup.lookup(custom_type.type_definition)
            if isinstance(type_definition, Variant):
                self.abort(field_name, "variants do not have fields")
            struct: Struct = type_definition
            found_field = False
            for field_index,struct_field in enumerate(struct.fields):
                if struct_field.name.lexeme == field_name.lexeme:
                    target_type = self.insert_generic_arguments(custom_type.generic_arguments, struct_field.taip)
                    resolved.append(FieldAccess(field_name, taip, target_type, field_index))
                    taip = target_type
                    found_field = True
                    break
            if not found_field:
                self.abort(field_name, "field not found")
        return resolved

    def insert_generic_arguments(self, generics: List[Type], taip: Type) -> Type:
        if isinstance(taip, PtrType):
            return PtrType(self.insert_generic_arguments(generics, taip.child))
        if isinstance(taip, CustomTypeType):
            return CustomTypeType(taip.name, taip.type_definition, self.insert_generic_arguments_all(generics, taip.generic_arguments))
        if isinstance(taip, FunctionType):
            return FunctionType(
                taip.token,
                self.insert_generic_arguments_all(generics, taip.parameters),
                self.insert_generic_arguments_all(generics, taip.returns),
            )
        if isinstance(taip, TupleType):
            return TupleType(
                taip.token,
                self.insert_generic_arguments_all(generics, taip.items),
            )
        if isinstance(taip, GenericType):
            return generics[taip.generic_index]
        return taip

    def insert_generic_arguments_all(self, generics: List[Type], types: List[Type]) -> List[Type]:
        return [self.insert_generic_arguments(generics, taip) for taip in types]
