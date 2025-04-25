#!/usr/bin/env python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple, NoReturn, Dict, Set, Sequence, Literal, TypeGuard, assert_never
import sys
import os
import unittest
import copy

import parser
from util import Ref, Lazy, sys_stdin, normalize_path, uhex, intercalate, listtostr
from indexed_dict import IndexedDict
from format import Formattable, FormatInstr, unnamed_record, named_record, format_seq, format_str, format_optional, format_dict, format_list, format
from parser import Parser, PrimitiveType, GenericType, HoleType, I8, I32, I64, Bool, NumberWord, BreakWord, ParserException
from lexer import Token, TokenLocation, Lexer
import resolver
from resolver import determine_compilation_order, ResolveCtx, TypeLookup, LocalName, ScopeId, GlobalId, LocalId, IntrinsicDrop, IntrinsicMemCopy, IntrinsicMemFill, IntrinsicSetStackSize, NumberWord, IntrinsicMemGrow, StringWord, ROOT_SCOPE, ResolverException

def load_recursive(
        modules: Dict[str, parser.Module],
        path: str,
        path_location: TokenLocation | None,
        stdin: str | None = None,
        import_stack: List[str]=[]):
    if path == "-":
        file = stdin if stdin is not None else sys_stdin.get()
    else:
        try:
            with open(path, 'r') as reader:
                file = reader.read()
        except FileNotFoundError:
            raise ParserException(path_location, f"File not found: ./{path}")

    tokens = Lexer(file).lex()
    module = Parser(path, file, tokens).parse()
    modules[path] = module
    for imp in module.imports:
        if os.path.dirname(path) != "":
            p = os.path.normpath(os.path.dirname(path) + "/" + imp.file_path.lexeme[1:-1])
        else:
            p = os.path.normpath(imp.file_path.lexeme[1:-1])
        if p in import_stack:
            error_message = "Module import cycle detected: "
            for a in import_stack:
                error_message += f"{a} -> "
            raise ParserException(TokenLocation(path, imp.file_path.line, imp.file_path.column), error_message)
        if p in modules:
            continue
        import_stack.append(p)
        load_recursive(
            modules,
            p,
            TokenLocation(path, imp.file_path.line, imp.file_path.column),
            stdin,
            import_stack,
        )
        import_stack.pop()

def resolve_modules(modules_unordered: Dict[str, List[parser.TopItem]]) -> IndexedDict[str, resolver.Module]:
    modules = determine_compilation_order({
        ("./" + path if path != "-" else path): module
        for path, module in modules_unordered.items()
    })
    resolved_modules: IndexedDict[str, resolver.Module] = IndexedDict()
    other_module_types: List[List[resolver.CustomType]] = []
    for id,(module_path,top_items) in enumerate(modules.items()):
        ctx = ResolveCtx(modules, resolved_modules, top_items, id, bytearray())
        imports = ctx.resolve_imports()
        custom_types = ctx.resolve_custom_types(imports)
        globals = ctx.resolve_globals(imports)
        signatures = ctx.resolve_signatures(imports)

        type_lookup = TypeLookup(module=id, types=list(custom_types.values()), other_modules=other_module_types)
        functions = ctx.resolve_functions(imports, type_lookup, signatures, globals)

        ctx.forbid_directly_recursive_types(type_lookup)

        other_module_types.append(list(custom_types.values()))
        resolved_modules[module_path] = resolver.Module(
            module_path,
            id,
            imports,
            custom_types,
            globals,
            functions,
            bytes(ctx.static_data),
        )
    return resolved_modules

@dataclass
class PtrType(Formattable):
    child: 'Type'
    def __str__(self) -> str:
        return f"PtrType(child={str(self.child)})"

    def size(self) -> int:
        return 4

    def can_live_in_reg(self) -> bool:
        return True

@dataclass
class NamedType(Formattable):
    name: Token
    taip: 'Type'
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("NamedType", [self.name, self.taip])

@dataclass
class FunctionType(Formattable):
    token: Token
    parameters: List['Type']
    returns: List['Type']

    def size(self) -> int:
        return 4

    def can_live_in_reg(self) -> bool:
        return True

@dataclass
class Struct:
    name: Token
    fields: Lazy[List[NamedType]]
    generic_parameters: List['Type']

    def __str__(self) -> str:
        return f"Struct(name={str(self.name)})"

    def size(self) -> int:
        field_sizes = [field.taip.size() for field in self.fields.get()]
        size = 0
        largest_field = 0
        for i, field_size in enumerate(field_sizes):
            largest_field = max(largest_field, field_size)
            size += field_size
            if i + 1 < len(field_sizes):
                next_field_size = field_sizes[i + 1]
                size = align_to(size, min(next_field_size, 4))
        return align_to(size, largest_field)

    def field_offset(self, field_index: int) -> int:
        fields = self.fields.get()
        offset = 0
        for i in range(0, field_index):
            field_size = fields[i].taip.size()
            offset += field_size
            if i + 1 < len(fields):
                next_field_size = fields[i + 1].taip.size()
                offset = align_to(offset, min(next_field_size, 4))
        return offset

@dataclass
class VariantCase:
    name: Token
    taip: 'Type | None'

@dataclass
class Variant:
    name: Token
    cases: Lazy[List[VariantCase]]
    generic_arguments: List['Type']

    def size(self) -> int:
        return 4 + max((t.taip.size() for t in self.cases.get() if t.taip is not None), default=0)

TypeDefinition = Struct | Variant

@dataclass
class StructHandle(Formattable):
    module: int
    index: int
    instance: int

@dataclass
class StructType(Formattable):
    name: Token
    struct: StructHandle
    _size: Lazy[int]
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("StructType", [
            ("name", self.name),
            ("struct", self.struct)])

    def can_live_in_reg(self) -> bool:
        return self.size() <= 8

    def size(self):
        return self._size.get()

    @staticmethod
    def dummy(name: str, size: int) -> 'StructType':
        return StructType(Token.dummy(name), StructHandle(0, 0, 0), Lazy(lambda: size))

@dataclass
class TupleType(Formattable):
    token: Token
    items: List['Type']

    def can_live_in_reg(self) -> bool:
        return self.size() <= 8

    def size(self):
        return sum(t.size() for t in self.items)

Type = PrimitiveType | PtrType | StructType | FunctionType | TupleType

def type_eq(a: Type, b: Type) -> bool:
    if isinstance(a, Bool) and isinstance(b, Bool):
        return True
    if isinstance(a, I8) and isinstance(b, I8):
        return True
    if isinstance(a, I32) and isinstance(b, I32):
        return True
    if isinstance(a, I64) and isinstance(b, I64):
        return True
    if isinstance(a, PtrType) and isinstance(b, PtrType):
        return type_eq(a.child, b.child)
    if isinstance(a, StructType) and isinstance(b, StructType):
        return a.struct.module == b.struct.module and a.struct.index == b.struct.index and a.struct.instance == b.struct.instance
    if isinstance(a, FunctionType) and isinstance(b, FunctionType):
        return types_eq(a.parameters, b.parameters) and types_eq(a.returns, b.returns)
    if isinstance(a, TupleType) and isinstance(b, TupleType):
        return types_eq(a.items, b.items)
    return False

def types_eq(a: List[Type], b: List[Type]) -> bool:
    if len(a) != len(b):
        return False
    for i in range(0, len(a)):
        if not type_eq(a[i], b[i]):
            return False
    return True

def format_type(a: Type) -> str:
    match a:
        case PtrType(child):
            return f".{format_type(child)}"
        case StructType(name):
            return name.lexeme 
        case FunctionType(_, parameters, returns):
            s = "("
            for param in parameters:
                s += format_type(param) + ", "
            s = s[:-2] + " -> "
            if len(a.returns) == 0:
                return s[:-1] + ")"
            for ret in returns:
                s += format_type(ret) + ", "
            return s[:-2] + ")"
        case TupleType(_, items):
            return "[" + intercalate(", ", map(format_type, items)) + "]"
        case Bool():
            return "bool"
        case I8():
            return "i8"
        case I32():
            return "i32"
        case I64():
            return "i64"
        case other:
            assert_never(other)

@dataclass
class ParameterLocal:
    name: LocalName
    taip: Type
    _lives_in_memory: bool

    def size(self) -> int:
        return self.taip.size()

    def lives_in_memory(self) -> bool:
        return self._lives_in_memory

    def needs_moved_into_memory(self) -> bool:
        return self.lives_in_memory() and self.taip.can_live_in_reg()

    def can_be_abused_as_ref(self) -> bool:
        return not self.taip.can_live_in_reg() or self.taip.size() <= 4

@dataclass
class InitLocal:
    name: LocalName
    taip: Type
    _lives_in_memory: bool

    def size(self) -> int:
        return self.taip.size()

    def lives_in_memory(self) -> bool:
        return self._lives_in_memory

type Local = ParameterLocal | InitLocal

@dataclass
class FunctionSignature:
    generic_arguments: List[Type]
    parameters: List[NamedType]
    returns: List[Type]

    def returns_any_struct(self) -> bool:
        return any(isinstance(ret, StructType) for ret in self.returns)

@dataclass
class Global:
    name: Token
    taip: Type
    was_reffed: bool

@dataclass
class Extern:
    name: Token
    extern_module: str
    extern_name: str
    signature: FunctionSignature

@dataclass
class Scope:
    id: ScopeId
    words: List['Word']

@dataclass
class ConcreteFunction:
    name: Token
    export_name: Token | None
    signature: FunctionSignature
    body: Scope
    locals_copy_space: int
    max_struct_ret_count: int
    locals: Dict[LocalId, Local]

@dataclass
class GenericFunction:
    instances: Dict[int, ConcreteFunction]

Function = ConcreteFunction | GenericFunction

@dataclass(frozen=True, eq=True)
class FunctionHandle:
    module: int
    index: int
    instance: int | None

@dataclass(frozen=True, eq=True)
class ExternHandle:
    module: int
    index: int

@dataclass
class CallWord:
    name: Token
    function: 'FunctionHandle | ExternHandle'
    return_space_offset: int

@dataclass
class CastWord:
    token: Token
    source: Type
    taip: Type

@dataclass
class LoadWord:
    token: Token
    taip: Type
    copy_space_offset: int | None

@dataclass
class IfWord:
    token: Token
    parameters: List[Type]
    returns: List[Type] | None
    true_branch: Scope
    false_branch: Scope
    diverges: bool

@dataclass
class IndirectCallWord:
    token: Token
    taip: FunctionType
    return_space_offset: int

@dataclass
class FunRefWord:
    call: CallWord
    table_index: int

@dataclass
class LoopWord:
    token: Token
    body: Scope
    parameters: List[Type]
    returns: List[Type]
    diverges: bool

@dataclass
class BlockWord:
    token: Token
    body: Scope
    parameters: List[Type]
    returns: List[Type]

@dataclass
class SizeofWord:
    token: Token
    taip: Type

@dataclass
class FieldAccess:
    name: Token
    source_taip: StructType | PtrType
    target_taip: Type
    offset: int

@dataclass
class Offset:
    offset: int

    def __str__(self) -> str:
        return f"i32.const {self.offset} i32.add"

@dataclass
class OffsetLoad:
    offset: int
    taip: Type

    def __str__(self) -> str:
        size = self.taip.size()
        if size == 1:
            return f"i32.load8_u offset={self.offset}" if self.offset != 0 else "i32.load8_u"
        if size <= 4:
            return f"i32.load offset={self.offset}" if self.offset != 0 else "i32.load"
        if size <= 8:
            return f"i64.load offset={self.offset}" if self.offset != 0 else "i64.load"
        if self.offset == 0:
            return f"i32.const {size} memory.copy"
        else:
            return f"i32.const {self.offset} i32.add i32.const {size} memory.copy"

@dataclass
class I32InI64:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.wrap_i64"
        return f"i64.const {self.offset * 8} i64.shr_u i32.wrap_i64"

@dataclass
class I16InI32:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.const 0xFFFF i32.and"
        return f"i32.const {self.offset * 8} i32.shr_u i32.const 0xFFFF i32.and"

@dataclass
class I8InI32:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.const 0xFF i32.and"
        return f"i32.const {self.offset * 8} i32.shr_u i32.const 0xFF i32.and"

@dataclass
class I8InI64:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.wrap_i64 i32.const 0xFF i32.and"
        return f"i64.const {self.offset * 8} i64.shr_u i32.wrap_i64 i32.const 0xFF i32.and"

type Load = Offset | OffsetLoad | I32InI64 | I8InI32 | I16InI32 | I8InI64

def is_bitshift(load) -> TypeGuard[I8InI64 | I32InI64 | I8InI32 | I16InI32]:
    return isinstance(load, I8InI64) or isinstance(load, I32InI64) or isinstance(load, I8InI32) or isinstance(load, I16InI32)

@dataclass
class GetFieldWord:
    token: Token
    target_taip: Type
    loads: List[Load]
    on_ptr: bool
    copy_space_offset: int | None

@dataclass
class SetWord:
    token: Token
    local_id: LocalId | GlobalId
    target_taip: Type
    loads: List[Load]
    var_lives_in_memory: bool

@dataclass
class InitWord:
    token: Token
    local_id: LocalId
    taip: Type
    var_lives_in_memory: bool

@dataclass
class GetWord:
    ident: Token
    local_id: LocalId | GlobalId
    target_taip: Type
    loads: List[Load]
    copy_space_offset: int | None
    var_lives_in_memory: bool

@dataclass
class RefWord:
    ident: Token
    local_id: LocalId | GlobalId
    loads: List[Load]

@dataclass
class IntrinsicAdd:
    token: Token
    taip: PtrType | I8 | I32 | I64

@dataclass
class IntrinsicSub:
    token: Token
    taip: PtrType | I8 | I32 | I64

@dataclass
class IntrinsicMul:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicDiv:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicMod:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicEqual:
    token: Token
    taip: Type

@dataclass
class IntrinsicNotEqual:
    token: Token
    taip: Type

@dataclass
class IntrinsicFlip:
    token: Token
    lower: Type
    upper: Type

@dataclass
class IntrinsicAnd:
    token: Token
    taip: PrimitiveType

@dataclass
class IntrinsicNot:
    token: Token
    taip: PrimitiveType

@dataclass
class IntrinsicGreaterEq:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicLessEq:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicGreater:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicLess:
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicShl:
    token: Token
    taip: Type

@dataclass
class IntrinsicShr:
    token: Token
    taip: Type

@dataclass
class IntrinsicRotl:
    token: Token
    taip: Type

@dataclass
class IntrinsicRotr:
    token: Token
    taip: Type

@dataclass
class IntrinsicOr:
    token: Token
    taip: Type

@dataclass
class IntrinsicStore:
    token: Token
    taip: Type

@dataclass
class IntrinsicUninit:
    token: Token
    taip: Type
    copy_space_offset: int

@dataclass
class StoreWord:
    token: Token
    local: LocalId | GlobalId
    taip: Type
    loads: List[Load]

@dataclass
class StructWord:
    token: Token
    taip: StructType
    copy_space_offset: int
    body: Scope

@dataclass
class UnnamedStructWord:
    token: Token
    taip: StructType
    copy_space_offset: int

@dataclass
class StructFieldInitWord:
    token: Token
    taip: Type
    copy_space_offset: int

@dataclass
class VariantWord:
    token: Token
    tag: int
    variant: StructHandle
    copy_space_offset: int

@dataclass
class MatchCase:
    tag: int
    body: Scope

@dataclass
class MatchWord:
    token: Token
    variant: StructHandle
    by_ref: bool
    cases: List[MatchCase]
    default: Scope | None
    parameters: List[Type]
    returns: List[Type] | None

@dataclass
class TupleMakeWord:
    token: Token
    taip: TupleType
    copy_space_offset: int

@dataclass
class TupleUnpackWord:
    token: Token
    item: List[Type]
    copy_space_offset: int

type IntrinsicWord = (
      IntrinsicAdd
    | IntrinsicSub
    | IntrinsicEqual
    | IntrinsicNotEqual
    | IntrinsicAnd
    | IntrinsicDrop
    | IntrinsicGreaterEq
    | IntrinsicLessEq
    | IntrinsicMul
    | IntrinsicMod
    | IntrinsicDiv
    | IntrinsicGreater
    | IntrinsicLess
    | IntrinsicFlip
    | IntrinsicShl
    | IntrinsicShr
    | IntrinsicRotl
    | IntrinsicRotr
    | IntrinsicOr
    | IntrinsicStore
    | IntrinsicMemCopy
    | IntrinsicMemFill
    | IntrinsicMemGrow
    | IntrinsicNot
    | IntrinsicUninit
    | IntrinsicSetStackSize
)

type Word = NumberWord | StringWord | CallWord | GetWord | InitWord | CastWord | SetWord | LoadWord | IntrinsicWord | IfWord | RefWord | IndirectCallWord | StoreWord | FunRefWord | LoopWord | BreakWord | SizeofWord | BlockWord | GetFieldWord | StructWord | StructFieldInitWord | UnnamedStructWord | VariantWord | MatchWord | InitWord | TupleMakeWord | TupleUnpackWord

@dataclass
class Module:
    id: int
    type_definitions: Dict[int, List[TypeDefinition]]
    externs: Dict[int, Extern]
    globals: List[Global]
    functions: Dict[int, Function]
    data: bytes

@dataclass
class Monomizer:
    modules: IndexedDict[str, resolver.Module]
    type_definitions: Dict[resolver.CustomTypeHandle, List[Tuple[List[Type], TypeDefinition]]] = field(default_factory=dict)
    externs: Dict[resolver.FunctionHandle, Extern] = field(default_factory=dict)
    globals: Dict[GlobalId, Global] = field(default_factory=dict)
    functions: Dict[resolver.FunctionHandle, Function] = field(default_factory=dict)
    signatures: Dict[resolver.FunctionHandle, FunctionSignature | List[FunctionSignature]] = field(default_factory=dict)
    function_table: Dict[FunctionHandle | ExternHandle, int] = field(default_factory=dict)

    def monomize(self) -> Tuple[Dict[FunctionHandle | ExternHandle, int], Dict[int, Module]]:
        self.externs = {
            resolver.FunctionHandle(m, i): self.monomize_extern(f)
            for m,module in self.modules.indexed_values()
            for i,f in enumerate(module.functions.values())
            if isinstance(f, resolver.Extern)
        }
        self.globals = {
            GlobalId(m, i): self.monomize_global(globl)
            for m,module in self.modules.indexed_values()
            for i,globl in enumerate(module.globals.values())
        }
        for id in range(len(self.modules)):
            module = self.modules.index(id)
            for index, function in enumerate(module.functions.values()):
                if isinstance(function, resolver.Extern):
                    continue
                if function.export_name is not None:
                    assert(len(function.signature.generic_parameters) == 0)
                    handle = resolver.FunctionHandle(id, index)
                    self.monomize_function(handle, [])

        mono_modules = {}
        for module_id,module in enumerate(self.modules.values()):
            externs: Dict[int, Extern] = { handle.index: extern for (handle,extern) in self.externs.items() if handle.module == module_id }
            globals: List[Global] = [globl for id, globl in self.globals.items() if id.module == module_id]
            type_definitions: Dict[int, List[TypeDefinition]] = { handle.index: [taip for _,taip in monomorphizations] for handle,monomorphizations in self.type_definitions.items() if handle.module == module_id }
            functions = { handle.index: function for handle,function in self.functions.items() if handle.module == module_id }
            mono_modules[module_id] = Module(module_id, type_definitions, externs, globals, functions, self.modules.index(module_id).data)
        return self.function_table, mono_modules

    def monomize_locals(self, locals: Dict[LocalId, resolver.Local], generics: List[Type]) -> Dict[LocalId, Local]:
        res: Dict[LocalId, Local] = {}
        for id, local in locals.items():
            taip = self.monomize_type(local.taip, generics)
            lives_in_memory = local.was_reffed or not taip.can_live_in_reg()
            if local.is_parameter:
                res[id] = ParameterLocal(local.name, taip, lives_in_memory)
            else:
                res[id] = InitLocal(local.name, taip, lives_in_memory)
            continue
        return res

    def monomize_concrete_signature(self, signature: resolver.FunctionSignature) -> FunctionSignature:
        assert(len(signature.generic_parameters) == 0)
        return self.monomize_signature(signature, [])

    def monomize_function(self, function: resolver.FunctionHandle, generics: List[Type]) -> ConcreteFunction:
        f = self.modules.index(function.module).functions.index(function.index)
        assert(isinstance(f, resolver.Function))
        if len(generics) == 0:
            assert(len(f.signature.generic_parameters) == 0)
        signature = self.monomize_signature(f.signature, generics)
        if len(f.signature.generic_parameters) == 0:
            self.signatures[function] = signature
            generic_index = None
        else:
            if function not in self.signatures:
                self.signatures[function] = []
            instances = self.signatures[function]
            assert(isinstance(instances, list))
            generic_index = len(instances)
            instances.append(signature)
        copy_space_offset = Ref(0)
        max_struct_ret_count = Ref(0)
        monomized_locals = self.monomize_locals(f.locals, generics)
        body = Scope(
            f.body.id,
            self.monomize_words(
                f.body.words,
                generics,
                copy_space_offset,
                max_struct_ret_count,
                monomized_locals,
                None))
        concrete_function = ConcreteFunction(
            f.name,
            f.export_name,
            signature,
            body,
            copy_space_offset.value,
            max_struct_ret_count.value,
            monomized_locals)
        if len(f.signature.generic_parameters) == 0:
            assert(len(generics) == 0)
            assert(function not in self.functions)
            self.functions[function] = concrete_function
            return concrete_function
        assert(generic_index is not None)
        if function not in self.functions:
            self.functions[function] = GenericFunction({})
        generic_function = self.functions[function]
        assert(isinstance(generic_function, GenericFunction))
        assert(generic_index not in generic_function.instances)
        generic_function.instances[generic_index] = concrete_function
        return concrete_function

    def monomize_signature(self, signature: resolver.FunctionSignature, generics: List[Type]) -> FunctionSignature:
        parameters = list(map(lambda t: self.monomize_named_type(t, generics), signature.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), signature.returns))
        return FunctionSignature(generics, parameters, returns)

    def monomize_global(self, globl: resolver.Global) -> Global:
        return Global(globl.name, self.monomize_type(globl.taip, []), globl.was_reffed)

    def monomize_scope(self, scope: resolver.Scope, generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], locals: Dict[LocalId, Local], struct_space: int | None) -> Scope:
        return Scope(scope.id, self.monomize_words(scope.words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space))

    def monomize_words(self, words: List[resolver.ResolvedWord], generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], locals: Dict[LocalId, Local], struct_space: int | None) -> List[Word]:
        return list(map(lambda w: self.monomize_word(w, generics, copy_space_offset, max_struct_ret_count, locals, struct_space), words))

    def monomize_word(self, word: resolver.ResolvedWord, generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], locals: Dict[LocalId, Local], struct_space: int | None) -> Word:
        match word:
            case NumberWord():
                return word
            case StringWord():
                return word
            case resolver.CallWord():
                return self.monomize_call_word(word, copy_space_offset, max_struct_ret_count, generics)
            case resolver.IndirectCallWord(token, taip):
                monomized_function_taip = self.monomize_function_type(taip, generics)
                local_copy_space_offset = copy_space_offset.value
                copy_space = sum(taip.size() for taip in monomized_function_taip.returns if not taip.can_live_in_reg())
                copy_space_offset.value += copy_space
                if copy_space != 0:
                    max_struct_ret_count.value = max(max_struct_ret_count.value, len(monomized_function_taip.returns))
                return IndirectCallWord(token, monomized_function_taip, local_copy_space_offset)
            case resolver.GetWord(token, local_id, var_taip, resolved_fields, taip):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                monomized_taip = self.monomize_type(taip, generics)
                if not monomized_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += monomized_taip.size()
                else:
                    offset = None
                lives_in_memory = self.does_var_live_in_memory(local_id, locals)
                monomized_var_taip = self.monomize_type(var_taip, generics)
                loads = determine_loads(fields, just_ref=False, base_in_mem=lives_in_memory)
                target_taip = fields[-1].target_taip if len(fields) != 0 else monomized_var_taip
                return GetWord(token, local_id, target_taip, loads, offset, lives_in_memory)
            case resolver.InitWord(token, local_id, taip):
                return InitWord(token, local_id, self.monomize_type(taip, generics), self.does_var_live_in_memory(local_id, locals))
            case resolver.SetWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                lives_in_memory = self.does_var_live_in_memory(local_id, locals)
                target_lives_in_memory = lives_in_memory or any(isinstance(field.source_taip, resolver.PtrType) for field in resolved_fields)
                monomized_var_taip = self.lookup_var_taip(local_id, locals)
                loads = determine_loads(fields, just_ref=target_lives_in_memory, base_in_mem=lives_in_memory)
                monomized_taip = fields[-1].target_taip if len(fields) != 0 else monomized_var_taip
                return SetWord(token, local_id, monomized_taip, loads, target_lives_in_memory)
            case resolver.RefWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                loads = determine_loads(fields, just_ref=True)
                return RefWord(token, local_id, loads)
            case resolver.StoreWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                loads = determine_loads(fields)
                monomized_taip = fields[-1].target_taip if len(fields) != 0 else self.lookup_var_taip(local_id, locals)
                if len(fields) == 0 and isinstance(local_id, LocalId):
                    assert(isinstance(monomized_taip, PtrType))
                    monomized_taip = monomized_taip.child
                return StoreWord(token, local_id, monomized_taip, loads)
            case resolver.IntrinsicAdd(token, taip):
                return IntrinsicAdd(token, self.monomize_addable_type(taip, generics))
            case resolver.IntrinsicSub(token, taip):
                return IntrinsicSub(token, self.monomize_addable_type(taip, generics))
            case resolver.IntrinsicMul(token, taip):
                return IntrinsicMul(token, taip)
            case resolver.IntrinsicMod(token, taip):
                return IntrinsicMod(token, taip)
            case resolver.IntrinsicDiv(token, taip):
                return IntrinsicDiv(token, taip)
            case resolver.IntrinsicEqual(token, taip):
                return IntrinsicEqual(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicNotEqual(token, taip):
                return IntrinsicNotEqual(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicGreaterEq(token, taip):
                return IntrinsicGreaterEq(token, taip)
            case resolver.IntrinsicGreater(token, taip):
                return IntrinsicGreater(token, taip)
            case resolver.IntrinsicLess(token, taip):
                return IntrinsicLess(token, taip)
            case resolver.IntrinsicLessEq(token, taip):
                return IntrinsicLessEq(token, taip)
            case resolver.IntrinsicAnd(token, taip):
                return IntrinsicAnd(token, taip)
            case resolver.IntrinsicNot(token, taip):
                return IntrinsicNot(token, taip)
            case IntrinsicDrop():
                return word
            case BreakWord():
                return word
            case resolver.IntrinsicFlip(token, lower, upper):
                return IntrinsicFlip(token, self.monomize_type(lower, generics), self.monomize_type(upper, generics))
            case resolver.IntrinsicShl(token, taip):
                return IntrinsicShl(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicShr(token, taip):
                return IntrinsicShr(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicRotl(token, taip):
                return IntrinsicRotl(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicRotr(token, taip):
                return IntrinsicRotr(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicOr(token, taip):
                return IntrinsicOr(token, self.monomize_type(taip, generics))
            case resolver.IntrinsicStore(token, taip):
                return IntrinsicStore(token, self.monomize_type(taip, generics))
            case IntrinsicMemCopy():
                return word
            case IntrinsicMemFill():
                return word
            case IntrinsicMemGrow():
                return word
            case IntrinsicSetStackSize():
                return word
            case resolver.IntrinsicUninit(token, taip):
                monomized_taip = self.monomize_type(taip, generics)
                offset = copy_space_offset.value
                if not monomized_taip.can_live_in_reg():
                    copy_space_offset.value += monomized_taip.size()
                return IntrinsicUninit(token, monomized_taip, offset)
            case resolver.LoadWord(token, taip):
                monomized_taip = self.monomize_type(taip, generics)
                if not monomized_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += monomized_taip.size()
                else:
                    offset = None
                return LoadWord(token, monomized_taip, offset)
            case resolver.CastWord(token, source, taip):
                return CastWord(token, self.monomize_type(source, generics), self.monomize_type(taip, generics))
            case resolver.IfWord(token, resolved_parameters, resolved_returns, resolved_if_words, resolved_else_words, diverges):
                true_branch = self.monomize_scope(resolved_if_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                false_branch = self.monomize_scope(resolved_else_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = None if resolved_returns is None else list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return IfWord(token, parameters, returns, true_branch, false_branch, diverges)
            case resolver.FunRefWord(call):
                # monomize_call_word increments the copy_space, but if we're just taking the pointer
                # of the function, then we're not actually calling it and no space should be allocated.
                cso = copy_space_offset.value
                msrc = max_struct_ret_count.value
                call_word = self.monomize_call_word(call, copy_space_offset, max_struct_ret_count, generics)
                # So restore the previous values of copy_space_offset and max_struct_ret_count afterwards.
                # TODO: extract those parts of monomize_call_word which are common to both actual calls and just FunRefs.
                copy_space_offset.value = cso
                max_struct_ret_count.value = msrc
                table_index = self.insert_function_into_table(call_word.function)
                return FunRefWord(call_word, table_index)
            case resolver.LoopWord(token, resolved_body, resolved_parameters, resolved_returns, diverges):
                body = self.monomize_scope(resolved_body, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return LoopWord(token, body, parameters, returns, diverges)
            case resolver.SizeofWord(token, taip):
                return SizeofWord(token, self.monomize_type(taip, generics))
            case resolver.BlockWord(token, resolved_body, resolved_parameters, resolved_returns):
                body = self.monomize_scope(resolved_body, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return BlockWord(token, body, parameters, returns)
            case resolver.GetFieldWord(token, resolved_fields, on_ptr):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                target_taip = fields[-1].target_taip
                offset = None
                if not on_ptr and not target_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += target_taip.size()
                loads = determine_loads(fields, just_ref=on_ptr)
                return GetFieldWord(token, target_taip, loads, on_ptr, offset)
            case resolver.StructWord(token, taip, resolved_body):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                copy_space_offset.value += monomized_taip.size()
                body = self.monomize_scope(resolved_body, generics, copy_space_offset, max_struct_ret_count, locals, offset)
                return StructWord(token, monomized_taip, offset, body)
            case resolver.UnnamedStructWord(token, taip):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                if not monomized_taip.can_live_in_reg():
                    copy_space_offset.value += monomized_taip.size()
                return UnnamedStructWord(token, monomized_taip, offset)
            case resolver.StructFieldInitWord(token, struct, taip, _, generic_arguments):
                generics_here = list(map(lambda t: self.monomize_type(t, generics), generic_arguments))
                (_,monomized_struct) = self.monomize_struct(struct, generics_here)
                if isinstance(monomized_struct, Variant):
                    assert(False)
                assert(struct_space is not None)
                field_copy_space_offset: int = struct_space
                for i,field in enumerate(monomized_struct.fields.get()):
                    if field.name.lexeme == token.lexeme:
                        field_copy_space_offset += monomized_struct.field_offset(i)
                        break
                return StructFieldInitWord(token, self.monomize_type(taip, generics), field_copy_space_offset)
            case resolver.VariantWord(token, case, resolved_variant_type):
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                (variant_handle, variant) = self.monomize_struct(resolved_variant_type.type_definition, this_generics)
                offset = copy_space_offset.value
                if variant.size() > 8:
                    copy_space_offset.value += variant.size()
                return VariantWord(token, case, variant_handle, offset)
            case resolver.MatchWord(token, resolved_variant_type, by_ref, cases, default_case, resolved_parameters, resolved_returns):
                monomized_cases: List[MatchCase] = []
                for resolved_case in cases:
                    body = self.monomize_scope(resolved_case.body, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                    monomized_cases.append(MatchCase(resolved_case.tag, body))
                monomized_default_case = None if default_case is None else self.monomize_scope(default_case, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                monomized_variant = self.monomize_struct(resolved_variant_type.type_definition, this_generics)[0]
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = None if resolved_returns is None else list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return MatchWord(token, monomized_variant, by_ref, monomized_cases, monomized_default_case, parameters, returns)
            case resolver.TupleMakeWord(token, tupl):
                offset = copy_space_offset.value
                mono_tupl = TupleType(tupl.token, list(map(lambda t: self.monomize_type(t, generics), tupl.items)))
                offset = copy_space_offset.value
                if mono_tupl.size() > 4:
                    copy_space_offset.value += mono_tupl.size()
                return TupleMakeWord(token, mono_tupl, offset)
            case resolver.TupleUnpackWord(token, resolver.TupleType(_, items)):
                offset = copy_space_offset.value
                mono_items = list(map(lambda t: self.monomize_type(t, generics), items))
                copy_space_offset.value += sum(item.size() for item in mono_items if not item.can_live_in_reg())
                return TupleUnpackWord(token, mono_items, offset)
            case other:
                assert_never(other)

    def lookup_var_taip(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, Local]) -> Type:
        if isinstance(local_id, LocalId):
            return locals[local_id].taip
        return self.globals[local_id].taip

    def does_var_live_in_memory(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, Local]) -> bool:
        if isinstance(local_id, LocalId):
            return locals[local_id].lives_in_memory()
        globl = self.globals[local_id]
        return globl.was_reffed or not globl.taip.can_live_in_reg()

    def insert_function_into_table(self, function: FunctionHandle | ExternHandle) -> int:
        if function not in self.function_table:
            self.function_table[function] = len(self.function_table)
        return self.function_table[function]

    def monomize_field_accesses(self, fields: List[resolver.FieldAccess], generics: List[Type]) -> List[FieldAccess]:
        if len(fields) == 0:
            return []

        field = fields[0]

        if isinstance(field.source_taip, resolver.CustomTypeType):
            source_taip: PtrType | StructType = self.monomize_struct_type(field.source_taip, generics)
            resolved_struct = field.source_taip.type_definition
            generic_arguments = field.source_taip.generic_arguments
        else:
            assert(isinstance(field.source_taip.child, resolver.CustomTypeType))
            source_taip = PtrType(self.monomize_type(field.source_taip.child, generics))
            resolved_struct = field.source_taip.child.type_definition
            generic_arguments = field.source_taip.child.generic_arguments
        [_, struct] = self.monomize_struct(resolved_struct, list(map(lambda t: self.monomize_type(t, generics), generic_arguments)))
        assert(not isinstance(struct, Variant))
        target_taip = self.monomize_type(field.target_taip, struct.generic_parameters)
        offset = struct.field_offset(field.field_index)
        return [FieldAccess(field.name, source_taip, target_taip, offset)] + self.monomize_field_accesses(fields[1:], struct.generic_parameters)

    def monomize_call_word(self, word: resolver.CallWord, copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], generics: List[Type]) -> CallWord:
        if word.function in self.externs:
            signature = self.externs[word.function].signature
            offset = copy_space_offset.value
            copy_space = sum(taip.size() for taip in signature.returns if isinstance(taip, StructType))
            max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
            copy_space_offset.value += copy_space
            return CallWord(word.name, ExternHandle(word.function.module, word.function.index), offset)
        generics_here = list(map(lambda t: self.monomize_type(t, generics), word.generic_arguments))
        if word.function in self.signatures:
            signatures = self.signatures[word.function]
            if isinstance(signatures, FunctionSignature):
                signature = signatures
                assert(len(word.generic_arguments) == 0)
                offset = copy_space_offset.value
                copy_space = sum(taip.size() for taip in signature.returns if not taip.can_live_in_reg())
                max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
                copy_space_offset.value += copy_space
                return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, None), offset)
            for instance_index, signature in enumerate(signatures):
                if types_eq(signature.generic_arguments, generics_here):
                    offset = copy_space_offset.value
                    copy_space = sum(taip.size() for taip in signature.returns if not taip.can_live_in_reg())
                    max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
                    copy_space_offset.value += copy_space
                    return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, instance_index), offset)
        self.monomize_function(word.function, generics_here)
        return self.monomize_call_word(word, copy_space_offset, max_struct_ret_count, generics) # the function instance should now exist, try monomorphizing this CallWord again

    def lookup_struct(self, struct: resolver.CustomTypeHandle, generics: List[Type]) -> Tuple[StructHandle, TypeDefinition] | None:
        if struct not in self.type_definitions:
            return None
        for instance_index, (genics, instance) in enumerate(self.type_definitions[struct]):
            if types_eq(genics, generics):
                return StructHandle(struct.module, struct.index, instance_index), instance
        return None

    def add_struct(self, handle: resolver.CustomTypeHandle, taip: TypeDefinition, generics: List[Type]) -> StructHandle:
        if handle not in self.type_definitions:
            self.type_definitions[handle] = []
        instance_index = len(self.type_definitions[handle])
        self.type_definitions[handle].append((generics, taip))
        return StructHandle(handle.module, handle.index, instance_index)

    def monomize_struct(self, struct: resolver.CustomTypeHandle, generics: List[Type]) -> Tuple[StructHandle, TypeDefinition]:
        handle_and_instance = self.lookup_struct(struct, generics)
        if handle_and_instance is not None:
            return handle_and_instance
        s = self.modules.index(struct.module).custom_types.index(struct.index)
        if isinstance(s, resolver.Variant):
            def cases() -> List[VariantCase]:
                return [VariantCase(c.name, self.monomize_type(c.taip, generics) if c.taip is not None else None) for c in s.cases]

            variant_instance = Variant(s.name, Lazy(cases), generics)
            handle = self.add_struct(struct, variant_instance, generics)
            return handle, variant_instance

        def fields() -> List[NamedType]:
            return list(map(lambda t: self.monomize_named_type(t, generics), s.fields))

        struct_instance = Struct(s.name, Lazy(fields), generics)
        handle = self.add_struct(struct, struct_instance, generics)
        return handle, struct_instance

    def monomize_named_type(self, taip: resolver.NamedType, generics: List[Type]) -> NamedType:
        return NamedType(taip.name, self.monomize_type(taip.taip, generics))

    def monomize_type(self, taip: resolver.Type, generics: List[Type]) -> Type:
        match taip:
            case resolver.PtrType():
                return PtrType(self.monomize_type(taip.child, generics))
            case resolver.GenericType(_, generic_index):
                return generics[generic_index]
            case resolver.CustomTypeType():
                return self.monomize_struct_type(taip, generics)
            case resolver.FunctionType():
                return self.monomize_function_type(taip, generics)
            case resolver.TupleType(token, items):
                return TupleType(token, list(map(lambda item: self.monomize_type(item, generics), items)))
            case resolver.HoleType():
                assert(False)
            case other:
                return other

    def monomize_addable_type(self, taip: resolver.PtrType | I8 | I32 | I64, generics: List[Type]) -> PtrType | I8 | I32 | I64:
        match taip:
            case I8() | I32() | I64():
                return taip
            case resolver.PtrType():
                return PtrType(self.monomize_type(taip.child, generics))
            case other:
                assert_never(other)

    def monomize_struct_type(self, taip: resolver.CustomTypeType, generics: List[Type]) -> StructType:
        this_generics = list(map(lambda t: self.monomize_type(t, generics), taip.generic_arguments))
        handle,struct = self.monomize_struct(taip.type_definition, this_generics)
        return StructType(taip.name, handle, Lazy(lambda: struct.size()))

    def monomize_function_type(self, taip: resolver.FunctionType, generics: List[Type]) -> FunctionType:
        parameters = list(map(lambda t: self.monomize_type(t, generics), taip.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), taip.returns))
        return FunctionType(taip.token, parameters, returns)

    def monomize_extern(self, extern: resolver.Extern) -> Extern:
        signature = self.monomize_concrete_signature(extern.signature)
        return Extern(extern.name, extern.extern_module, extern.extern_name, signature)

def align_to(n: int, to: int) -> int:
    if to == 0:
        return n
    return n + (to - (n % to)) * ((n % to) > 0)


def merge_locals_module(module: Module):
    for function in module.functions.values():
        if isinstance(function, ConcreteFunction):
            merge_locals_function(function)
            return
        for instance in function.instances.values():
            merge_locals_function(instance)

@dataclass
class Disjoint:
    scopes: Set[ScopeId]
    reused: Set[LocalId]
    substitutions: Dict[LocalId, LocalId]

    def fixup_var(self, var: LocalId | GlobalId) -> LocalId | GlobalId:
        if isinstance(var, GlobalId):
            return var
        if var not in self.substitutions:
            return var
        return self.substitutions[var]

def merge_locals_function(function: ConcreteFunction):
    disjoint = Disjoint(set(), set(), {})
    merge_locals_scope(function.body, function.locals, disjoint)

def merge_locals_scope(scope: Scope, locals: Dict[LocalId, Local], disjoint: Disjoint):
    for word in scope.words:
        merge_locals_word(word, locals, disjoint, scope.id)

def merge_locals_word(word: Word, locals: Dict[LocalId, Local], disjoint: Disjoint, scope: ScopeId):
    if isinstance(word, InitWord):
        reused_local = find_disjoint_local(locals, disjoint, locals[word.local_id])
        if reused_local is None:
            return
        del locals[word.local_id]
        disjoint.substitutions[word.local_id] = reused_local
        word.local_id = reused_local
        return
    if isinstance(word, GetWord):
        word.local_id = disjoint.fixup_var(word.local_id)
        return
    if isinstance(word, SetWord):
        word.local_id = disjoint.fixup_var(word.local_id)
        return
    if isinstance(word, RefWord):
        word.local_id = disjoint.fixup_var(word.local_id)
        return
    if isinstance(word, StoreWord):
        word.local = disjoint.fixup_var(word.local)
        return
    if isinstance(word, IfWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.true_branch, locals, disjoint)
        disjoint.reused = outer_reused

        disjoint.scopes.add(word.true_branch.id)
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.false_branch, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.false_branch.id)
        return
    if isinstance(word, BlockWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.body, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.body.id)
        return
    if isinstance(word, LoopWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.body, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.body.id)
        return
    if isinstance(word, StructWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.body, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.body.id)
        return
    if isinstance(word, MatchWord):
        for cays in word.cases:
            outer_reused = disjoint.reused.copy()
            merge_locals_scope(cays.body, locals, disjoint)
            disjoint.reused = outer_reused
            disjoint.scopes.add(cays.body.id)
        if word.default is not None:
            outer_reused = disjoint.reused.copy()
            merge_locals_scope(word.default, locals, disjoint)
            disjoint.reused = outer_reused
            disjoint.scopes.add(word.default.id)

def find_disjoint_local(locals: Dict[LocalId, Local], disjoint: Disjoint, to_be_replaced: Local) -> LocalId | None:
    local_size = to_be_replaced.taip.size()
    if len(disjoint.scopes) == 0:
        return None
    for local_id, local in locals.items():
        if local.lives_in_memory() != to_be_replaced.lives_in_memory():
            continue
        if local.taip.size() != local_size:
            continue
        if local_id.scope not in disjoint.scopes:
            continue
        if local_id in disjoint.reused:
            continue
        disjoint.reused.add(local_id)
        return local_id
    return None

class DetermineLoadsToValueTests(unittest.TestCase):
    def test_no_fields_returns_empty(self) -> None:
        loads = determine_loads([])
        self.assertTrue(len(loads) == 0)

    def test_by_value_on_struct(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 12), I32(), 4)
        ])
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == OffsetLoad(4, I32()))

    def test_by_value_on_struct_packed(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 8), I32(), 0)
        ])
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == I32InI64(0))

    def test_packed_value_on_struct_in_mem(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 8), I32(), 0)
        ], base_in_mem=True)
        self.assertEqual(loads, [OffsetLoad(0, I32())])

    def test_by_value_on_struct_packed_small_noop(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 4), I32(), 0)
        ])
        self.assertEqual(loads, [])

    def test_by_value_on_nested_struct(self) -> None:
        foo = StructType.dummy("Foo", 12)
        v2 = StructType.dummy("V2", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("v"), foo, v2               , 4),
            FieldAccess(Token.dummy("x"), v2 , I32(), 0),
        ])
        self.assertEqual(loads, [OffsetLoad(4, I32())])

    def test_by_value_through_ptr(self) -> None:
        node = StructType.dummy("Node", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , node         , PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), I32(), 0),
        ])
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == OffsetLoad(4, I32()))

    def test_by_value_through_two_ptrs(self) -> None:
        node = StructType.dummy("Node", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , node         , PtrType(node)    , 4),
            FieldAccess(Token.dummy("next") , PtrType(node), PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), I32(), 0),
        ])
        self.assertEqual(loads, [
            OffsetLoad(8, PtrType(node)),
            OffsetLoad(0, I32())])

    def test_get_sub_struct_by_value(self) -> None:
        bar = StructType.dummy("Bar", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("v"), StructType.dummy("Foo", 16), bar, 4)
        ])
        self.assertEqual(loads, [OffsetLoad(4, bar)])

    def test_get_subsub_struct_by_value(self) -> None:
        bar = StructType.dummy("Bar", 12)
        inner = StructType.dummy("BarInner", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("v")    , StructType.dummy("Foo", 16), bar  , 4),
            FieldAccess(Token.dummy("inner"), bar                        , inner, 0),
        ])
        self.assertEqual(loads, [OffsetLoad(4, inner)])

    def test_by_ref_get_on_large_value(self) -> None:
        node = StructType.dummy("Node", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , PtrType(node), PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), I32(), 0),
        ])
        self.assertEqual(loads, [
            OffsetLoad(4, PtrType(node)),
            OffsetLoad(0, I32())])

    def test_by_ref_get_on_packed_value(self) -> None:
        node = StructType.dummy("Node", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , PtrType(node), PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), I32(), 0),
        ])
        self.assertTrue(len(loads) == 2)
        self.assertTrue(loads[0] == OffsetLoad(4, PtrType(node)))
        self.assertTrue(loads[1] == OffsetLoad(0, I32()))

    def test_get_through_bitshift(self) -> None:
        node = StructType.dummy("Node", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , node         , PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), I32(), 0),
        ])
        self.assertTrue(len(loads) == 2)
        self.assertTrue(loads[0] == I32InI64(4))
        self.assertTrue(loads[1] == OffsetLoad(0, I32()))

    def test_ref_field_in_big_struct(self) -> None:
        ctx = StructType.dummy("Ctx", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("counter"), ctx, I32(), 8),
        ], just_ref=True)
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == Offset(8))

    def test_ref_subfield_in_big_struct(self) -> None:
        ctx = StructType.dummy("Ctx", 20)
        word_ctx = StructType.dummy("WordCtx", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("word-ctx"), ctx              , word_ctx         , 4),
            FieldAccess(Token.dummy("counter") , PtrType(word_ctx), I32(), 8),
        ], just_ref=True)
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == Offset(12))

    def test_ref_field_in_big_struct_at_offset_0(self) -> None:
        ctx = StructType.dummy("Ctx", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("counter"), ctx, I32(), 0),
        ], just_ref=True)
        self.assertTrue(len(loads) == 0)

    def test_ignored_wrapper_struct(self) -> None:
        allocator = StructType.dummy("PageAllocator", 4)
        page = StructType.dummy("Page", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("free-list"), allocator    , PtrType(page)    , 0),
            FieldAccess(Token.dummy("foo")      , PtrType(page), I32(), 4),
        ], just_ref=True)
        self.assertEqual(loads, [Offset(4)])

    def test_no_unnecessary_bitshift_on_ptr(self) -> None:
        foo = StructType.dummy("Foo", 12)
        v2 = StructType.dummy("V2", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("v")  , foo, v2               , 8),
            FieldAccess(Token.dummy("foo"), v2 , I32(), 4),
        ])
        self.assertEqual(loads, [OffsetLoad(12, I32())])

    def test_set_field_on_ptr(self) -> None:
        small = StructType.dummy("Small", 4)
        loads = determine_loads([
            FieldAccess(Token.dummy("value"), PtrType(small), I32(), 0),
        ], just_ref=True)
        self.assertEqual(loads, [])

    def test_get_value_on_packed_but_reffed_struct(self) -> None:
        prestat = StructType.dummy("Prestat", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("path_len"), prestat, I32(), 4)
        ], base_in_mem=True)
        self.assertEqual(loads, [OffsetLoad(4, I32())])

    def test_get_struct_through_ptr(self) -> None:
        named_type = StructType.dummy("NamedType", 16)
        taip = StructType.dummy("Type", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("type"), PtrType(named_type), taip, 4),
        ])
        self.assertEqual(loads, [OffsetLoad(4, taip)])

    def test_packed_value_field_through_ptr(self) -> None:
        token = StructType.dummy("Token", 8)
        immediate_string = StructType.dummy("ImmediateString", 4)
        loads = determine_loads([
            FieldAccess(Token.dummy("lexeme"), PtrType(token), PtrType(immediate_string), 4),
            FieldAccess(Token.dummy("len"), PtrType(immediate_string), I32(), 0),
        ])
        self.assertEqual(loads, [
            OffsetLoad(4, PtrType(immediate_string)),
            OffsetLoad(0, I32())])

    def test_set_field_through_bitshift(self) -> None:
        token = StructType.dummy("Token", 8)
        immediate_string = StructType.dummy("ImmediateString", 4)
        loads = determine_loads([
            FieldAccess(Token.dummy("lexeme"), token, PtrType(immediate_string), 4),
            FieldAccess(Token.dummy("len"), PtrType(immediate_string), I32(), 0),
        ], just_ref=True)
        self.assertEqual(loads, [I32InI64(4)])

def merge_loads(loads: List[Load]) -> List[Load]:
    if len(loads) <= 1:
        return loads
    a = loads[0]
    b = loads[1]
    if isinstance(a, OffsetLoad) and is_bitshift(b):
        return [OffsetLoad(a.offset + b.offset, I32())] + loads[2:]
    if isinstance(a, Offset) and isinstance(b, Offset):
        return [Offset(a.offset + b.offset)] + loads[2:]
    if isinstance(a, Offset) and isinstance(b, OffsetLoad):
        return [OffsetLoad(a.offset + b.offset, b.taip)] + loads[2:]
    if isinstance(a, I32InI64) and isinstance(b, I8InI32):
        return [I8InI64(a.offset + b.offset)] + loads[2:]
    if isinstance(a, I32InI64) and isinstance(b, I8InI32):
        return [I8InI32(a.offset + b.offset)] + loads[2:]
    if isinstance(a, I16InI32) and isinstance(b, I8InI32):
        return [I8InI32(a.offset + b.offset)] + loads[2:]
    return loads

# Returns the loads necessary to get the value of the final field on the stack.
def determine_loads(fields: List[FieldAccess], just_ref: bool = False, base_in_mem: bool = False) -> List[Load]:
    if len(fields) == 0:
        return []
    field = fields[0]
    if isinstance(field.source_taip, StructType):
        offset = field.offset
        if base_in_mem or field.source_taip.size() > 8:
            if len(fields) > 1 or just_ref:
                load: Load = Offset(offset)
            else:
                load = OffsetLoad(offset, field.target_taip)
            if load == Offset(0):
                return determine_loads(fields[1:], just_ref, base_in_mem=True)
            return merge_loads([load] + determine_loads(fields[1:], just_ref, base_in_mem=True))

        source_type_size = field.source_taip.size()
        target_type_size = field.target_taip.size()
        if source_type_size > 4: # source_taip is between >=4 and <=8 bytes
            if target_type_size == 1:
                load = I8InI64(offset)
            elif target_type_size == 4:
                load = I32InI64(offset)
            else:
                assert(False) # TODO
            return merge_loads([load] + determine_loads(fields[1:], just_ref, base_in_mem))

        if target_type_size != source_type_size:
            if target_type_size == 1:
                return merge_loads([I8InI32(offset)] + determine_loads(fields[1:], just_ref, base_in_mem))
            if target_type_size == 2:
                return merge_loads([I16InI32(offset)] + determine_loads(fields[1:], just_ref, base_in_mem))
            assert(False) # TODO

        assert(field.source_taip.size() == 4) # alternative is TODO
        return determine_loads(fields[1:], just_ref, base_in_mem)

    if isinstance(field.source_taip, PtrType):
        if (just_ref and len(fields) == 1) or (not field.target_taip.can_live_in_reg() and len(fields) != 1):
            # Instead of actually loading the value, we just ref it, since this is
            # the last field access in the chain and `just_ref` is set.
            if field.offset == 0:
                return determine_loads(fields[1:], just_ref, base_in_mem)
            return merge_loads([Offset(field.offset)] + determine_loads(fields[1:], just_ref, base_in_mem))

        return merge_loads([OffsetLoad(field.offset, field.target_taip)] + determine_loads(fields[1:], just_ref, base_in_mem))

    assert_never(field.source_taip)

@dataclass
class WatGenerator:
    modules: Dict[int, Module]
    function_table: Dict[FunctionHandle | ExternHandle, int]
    guard_stack: bool
    chunks: List[str] = field(default_factory=list)
    indentation: int = 0
    globals: Dict[GlobalId, Global]= field(default_factory=dict)
    module_data_offsets: Dict[int, int] = field(default_factory=dict)

    pack_i32s_used: bool = False
    unpack_i32s_used: bool = False
    flip_i64_i32_used: bool = False
    flip_i32_i64_used: bool = False
    flip_i64_i64_used: bool = False
    dup_i64_used: bool = False

    def write(self, s: str) -> None:
        self.chunks.append(s)

    def write_indent(self) -> None:
        self.chunks.append("\t" * self.indentation)

    def write_line(self, line: str) -> None:
        self.write_indent()
        self.write(line)
        self.write("\n")

    def indent(self) -> None:
        self.indentation += 1

    def dedent(self) -> None:
        self.indentation -= 1

    def lookup_type_definition(self, handle: StructHandle) -> TypeDefinition:
        return self.modules[handle.module].type_definitions[handle.index][handle.instance]

    def lookup_extern(self, handle: ExternHandle) -> Extern:
        return self.modules[handle.module].externs[handle.index]

    def lookup_function(self, handle: FunctionHandle) -> ConcreteFunction:
        function = self.modules[handle.module].functions[handle.index]
        if isinstance(function, GenericFunction):
            assert(handle.instance is not None)
            return function.instances[handle.instance]
        return function

    def write_wat_module(self) -> str:
        assert(len(self.chunks) == 0)
        self.write_line("(module")
        self.indent()
        for module in self.modules.values():
            for extern in module.externs.values():
                self.write_extern(module.id, extern)
                self.write("\n")
            for i, globl in enumerate(module.globals):
                self.globals[GlobalId(module.id, i)] = globl

        self.write_line("(memory 1 65536)")
        self.write_line("(export \"memory\" (memory 0))")

        all_data: bytes = b""
        for id in sorted(self.modules):
            self.module_data_offsets[id] = len(all_data)
            all_data += self.modules[id].data

        self.write_function_table()

        data_end = align_to(len(all_data), 4)
        global_mem = self.write_globals(data_end)
        stack_start = align_to(global_mem, 4)
        self.write_line(f"(global $stac:k (mut i32) (i32.const {stack_start}))")
        if self.guard_stack:
            self.write_line("(global $stack-siz:e (mut i32) (i32.const 65536))")

        self.write_data(all_data)

        for module_id in sorted(self.modules):
            module = self.modules[module_id]
            for function in sorted(module.functions.keys()):
                self.write_function(module_id, module.functions[function])

        self.write_intrinsics()

        self.dedent()
        self.write(")")
        return ''.join(self.chunks)

    def write_function(self, module: int, function: Function, instance_id: int | None = None) -> None:
        if isinstance(function, GenericFunction):
            for (id, instance) in function.instances.items():
                self.write_function(module, instance, id)
            return
        self.write_indent()
        self.write("(")
        self.write_signature(module, function.name, function.export_name, function.signature, instance_id, function.locals)
        if len(function.signature.generic_arguments) > 0:
            self.write(" ;;")
            for taip in function.signature.generic_arguments:
                self.write(" ")
                self.write_type_human(taip)
        self.write("\n")
        self.indent()
        self.write_locals(function.locals)
        for i in range(0, function.max_struct_ret_count):
            self.write_indent()
            self.write(f"(local $s{i}:a i32)\n")
        if function.locals_copy_space != 0:
            self.write_indent()
            self.write("(local $locl-copy-spac:e i32)\n")

        uses_stack = function.locals_copy_space != 0 or any(local.lives_in_memory() for local in function.locals.values())
        if uses_stack:
            self.write_indent()
            self.write("(local $stac:k i32)\n")
            self.write_indent()
            self.write("global.get $stac:k local.set $stac:k\n")

        if function.locals_copy_space != 0:
            self.write_mem("locl-copy-spac:e", function.locals_copy_space, ROOT_SCOPE, 0)
        self.write_structs(function.locals)
        if uses_stack and self.guard_stack:
            self.write_line("call $stack-overflow-guar:d")
        self.write_words(module, { id: local.name.get() for id, local in function.locals.items() }, function.body.words)
        if uses_stack:
            self.write_line("local.get $stac:k global.set $stac:k")
        self.dedent()
        self.write_line(")")

    def write_mem(self, name: str, size: int, scope: ScopeId, shadow: int) -> None:
        self.write_indent()
        self.write(f"global.get $stac:k global.get $stac:k i32.const {align_to(size, 4)} i32.add global.set $stac:k local.set ${name}")
        if scope != ROOT_SCOPE or shadow != 0:
            self.write(f":{scope}:{shadow}")
        self.write("\n")

    def write_structs(self, locals: Dict[LocalId, Local]) -> None:
        for local_id, local in locals.items():
            if not isinstance(local, ParameterLocal) and local.lives_in_memory():
                self.write_mem(local.name.get(), local.taip.size(), local_id.scope, local_id.shadow)
            if isinstance(local, ParameterLocal) and local.needs_moved_into_memory():
                self.write_indent()
                self.write("global.get $stac:k global.get $stac:k local.get $")
                if not local.can_be_abused_as_ref():
                    self.write("v:")
                self.write(f"{local.name.get()} ")
                self.write_type(local.taip)
                self.write(f".store local.tee ${local.name.get()} i32.const {align_to(local.taip.size(), 4)} i32.add global.set $stac:k\n")

    def write_locals(self, locals: Dict[LocalId, Local]) -> None:
        for local_id, local in locals.items():
            if isinstance(local, ParameterLocal):
                if local.needs_moved_into_memory() and not local.can_be_abused_as_ref():
                    self.write_line(f"(local ${local.name.get()} i32)")
                continue
            local = locals[local_id]
            self.write_indent()
            self.write(f"(local ${local.name.get()}")
            if local_id.scope != ROOT_SCOPE or local_id.shadow != 0:
                self.write(f":{local_id.scope}:{local_id.shadow}")
            self.write(" ")
            if local.lives_in_memory():
                self.write("i32")
            else:
                self.write_type(local.taip)
            self.write(")\n")

    def write_words(self, module: int, locals: Dict[LocalId, str], words: List[Word]) -> None:
        for word in words:
            self.write_word(module, locals, word)

    def write_local_ident(self, locals: Dict[LocalId, str], local: LocalId) -> None:
        if local.scope != ROOT_SCOPE or local.shadow != 0:
            self.write(f"${locals[local]}:{local.scope}:{local.shadow}")
        else:
            self.write(f"${locals[local]}")

    def write_word(self, module: int, locals: Dict[LocalId, str], word: Word) -> None:
        match word:
            case NumberWord(token):
                self.write_line(f"i32.const {token.lexeme}")
            case GetWord(token, local_id, target_taip, loads, copy_space_offset, var_lives_in_memory):
                self.write_indent()
                if not target_taip.can_live_in_reg():
                    # set up the address to store the result in
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                else:
                    self.write("local.get ")
                    self.write_local_ident(locals, local_id)
                # at this point, either the value itself or a pointer to it is on the stack
                for i, load in enumerate(loads):
                    self.write(" ")
                    self.write(str(load))
                if len(loads) == 0:
                    if target_taip.can_live_in_reg():
                        if var_lives_in_memory:
                            self.write(" ")
                            self.write_type(target_taip)
                            self.write(".load\n")
                            return
                    else:
                        self.write(f" i32.const {target_taip.size()} memory.copy")
                self.write("\n")
            case GetFieldWord(token, target_taip, loads, on_ptr, copy_space_offset):
                if len(loads) == 0:
                    self.write_line(";; GetField was no-op")
                    return
                self.write_indent()
                if not on_ptr and not target_taip.can_live_in_reg():
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                for load in loads:
                    self.write(str(load))
                self.write("\n")
            case RefWord(token, local_id, loads):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                if isinstance(local_id, LocalId):
                    self.write("local.get ")
                    self.write_local_ident(locals, local_id)
                for i, load in enumerate(loads):
                    self.write(f" {load}")
                self.write("\n")
            case SetWord(token, local_id, target_taip, loads, target_lives_in_memory):
                self.write_set(local_id, locals, target_lives_in_memory, target_taip, loads)
            case InitWord(name, local_id, taip, var_lives_in_memory):
                self.write_set(local_id, locals, var_lives_in_memory, taip, [])
            case CallWord(name, function_handle, return_space_offset):
                self.write_indent()
                match function_handle:
                    case ExternHandle():
                        extern = self.lookup_extern(function_handle)
                        signature = extern.signature
                        self.write(f"call ${function_handle.module}:{name.lexeme}")
                    case FunctionHandle():
                        function = self.lookup_function(function_handle)
                        signature = function.signature
                        self.write(f"call ${function_handle.module}:{function.name.lexeme}")
                        if function_handle.instance is not None and function_handle.instance != 0:
                            self.write(f":{function_handle.instance}")
                    case other:
                        assert_never(other)
                self.write_return_struct_receiving(return_space_offset, signature.returns)
            case IndirectCallWord(token, taip, return_space_offset):
                self.write_indent()
                self.write("(call_indirect")
                self.write_parameters(taip.parameters)
                self.write_returns(taip.returns)
                self.write(")")
                self.write_return_struct_receiving(return_space_offset, taip.returns)
            case IntrinsicStore(token, taip):
                self.write_indent()
                self.write_store(taip)
                self.write("\n")
            case IntrinsicAdd(token, taip):
                if isinstance(taip, PtrType) or isinstance(taip, I32) or isinstance(taip, I8):
                    self.write_line("i32.add")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.add")
                    return
                assert_never(taip)
            case IntrinsicSub(token, taip):
                if isinstance(taip, PtrType) or isinstance(taip, I32) or isinstance(taip, I8):
                    self.write_line("i32.sub")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.sub")
                    return
                assert_never(taip)
            case IntrinsicMul(_, taip):
                self.write_line(f"{'i64' if isinstance(taip, I64) else 'i32'}.mul")
            case IntrinsicDrop():
                self.write_line("drop")
            case IntrinsicOr(_, taip):
                self.write_indent()
                self.write_type(taip)
                self.write(".or\n")
            case IntrinsicEqual(_, taip):
                if taip.size() > 4:
                    self.write_line("i64.eq")
                    return
                assert(taip.can_live_in_reg())
                self.write_line("i32.eq")
            case IntrinsicNotEqual(_, taip):
                if isinstance(taip, I64):
                    self.write_line("i64.ne")
                    return
                assert(taip.can_live_in_reg())
                self.write_line("i32.ne")
            case IntrinsicGreaterEq(_, taip):
                if isinstance(taip, I32) or isinstance(taip, I8):
                    self.write_line("i32.ge_u")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.ge_u")
                    return
                assert_never(taip)
            case IntrinsicGreater(_, taip):
                if isinstance(taip, I32) or isinstance(taip, I8):
                    self.write_line("i32.gt_u")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.gt_u")
                    return
                assert_never(taip)
            case IntrinsicLessEq(_, taip):
                if isinstance(taip, I32) or isinstance(taip, I8):
                    self.write_line("i32.le_u")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.le_u")
                    return
                assert_never(taip)
            case IntrinsicLess(_, taip):
                if isinstance(taip, I32) or isinstance(taip, I8):
                    self.write_line("i32.lt_u")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.lt_u")
                    return
                assert_never(taip)
            case IntrinsicFlip(_, lower, upper):
                lower_type = "i32" if lower.size() <= 4 or lower.size() > 8 else "i64"
                upper_type = "i32" if upper.size() <= 4 or upper.size() > 8 else "i64"
                if lower_type == "i32" and upper_type == "i64":
                    self.flip_i32_i64_used = True
                    self.write_line("call $intrinsic:flip-i32-i64")
                    return
                if lower_type == "i64" and upper_type == "i32":
                    self.flip_i64_i32_used = True
                    self.write_line("call $intrinsic:flip-i64-i32")
                    return
                if lower_type == "i32":
                    self.write_line("call $intrinsic:flip")
                    return
                self.flip_i64_i64_used = True
                self.write_line("call $intrinsic:flip-i64-i64")
            case IntrinsicShl(token, taip):
                if isinstance(taip, I64):
                    self.write_line("i64.rotl")
                else:
                    self.write_line("i32.shl")
            case IntrinsicShr(token, taip):
                if isinstance(taip, I64):
                    self.write_line("i64.shr")
                else:
                    self.write_line("i32.shr_u")
            case IntrinsicRotl(token, taip):
                if isinstance(taip, I64):
                    self.write_line("i64.rotl")
                else:
                    self.write_line("i32.rotl")
            case IntrinsicRotr(token, taip):
                if isinstance(taip, I64):
                    self.write_line("i64.rotr")
                else:
                    self.write_line("i32.rotr")
            case IntrinsicAnd(_, taip):
                if isinstance(taip, I32) or isinstance(taip, Bool) or isinstance(taip, I8):
                    self.write_line("i32.and")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.and")
                    return
                assert_never(taip)
            case IntrinsicNot(_, taip):
                if isinstance(taip, Bool):
                    self.write_line("i32.const 1 i32.and i32.const 1 i32.xor i32.const 1 i32.and")
                    return
                if isinstance(taip, I8):
                    self.write_line("i32.const 255 i32.and i32.const 255 i32.xor i32.const 255 i32.and")
                    return
                if isinstance(taip, I32):
                    self.write_line("i32.const -1 i32.xor")
                    return
                if isinstance(taip, I64):
                    self.write_line("i64.const -1 i64.xor")
                    return
                assert_never(taip)
            case IntrinsicMod(_, taip):
                match taip:
                    case I8() | I32():
                        self.write_line("i32.rem_u")
                    case I64():
                        self.write_line("i64.rem_u")
                    case _:
                        assert_never(taip)
            case IntrinsicDiv(_, taip):
                match taip:
                    case I32() | I8():
                        self.write_line("i32.div_u")
                    case I64():
                        self.write_line("i64.div_u")
                    case _:
                        assert_never(taip)
            case IntrinsicMemCopy():
                self.write_line("memory.copy")
            case IntrinsicMemFill():
                self.write_line("memory.fill")
            case IntrinsicMemGrow():
                self.write_line("memory.grow")
            case IntrinsicSetStackSize():
                if self.guard_stack:
                    self.write_line("global.set $stack-siz:e")
                else:
                    self.write_line("drop")
            case IntrinsicUninit(_, taip, copy_space_offset):
                if taip.size() <= 4:
                    self.write_line("i32.const 0")
                elif taip.size() <= 8:
                    self.write_line("i64.const 0")
                else:
                    self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add")
            case CastWord(_, source, taip):
                if (isinstance(source, I32) or isinstance(source, PtrType)) and isinstance(taip, PtrType):
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if isinstance(source, I32) and isinstance(taip, FunctionType):
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if isinstance(source, I32) and isinstance(taip, StructType) and taip.size() == 4:
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if (isinstance(source, Bool) or isinstance(source, I32)) and isinstance(taip, I64):
                    self.write_line("i64.extend_i32_u ;; cast to i64")
                    return
                if (isinstance(source, Bool) or isinstance(source, I32)) and isinstance(taip, I8):
                    self.write_line(f"i32.const 0xFF i32.and ;; cast to {format_type(taip)}")
                    return
                if isinstance(source, I64) and isinstance(taip, I8): 
                    self.write_line(f"i64.const 0xFF i64.and i32.wrap_i64 ;; cast to {format_type(taip)}")
                    return
                if isinstance(source, I64) and not isinstance(taip, I64):
                    self.write_line(f"i32.wrap_i64 ;; cast to {format_type(taip)}")
                    return
                if source.can_live_in_reg() and source.size() <= 4 and isinstance(taip, I32):
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if source.can_live_in_reg() and source.size() <= 8 and isinstance(taip, I64):
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                self.write_line(f"UNSUPPORTED Cast from {format_type(source)} to {format_type(taip)}")
            case StringWord(_, offset, string_len):
                self.write_line(f"i32.const {self.module_data_offsets[module] + offset} i32.const {string_len}")
            case SizeofWord(_, taip):
                self.write_line(f"i32.const {taip.size()}")
            case FunRefWord(_, table_index):
                self.write_line(f"i32.const {table_index + 1}")
            case StoreWord(token, local_id, target_taip, loads):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                else:
                    self.write("local.get ")
                    self.write_local_ident(locals, local_id)
                for load in loads:
                    # self.write(f" i32.load offset={offset}")
                    self.write(f" {load}")
                self.write(" call $intrinsic:flip ")
                self.write_store(target_taip)
                self.write("\n")
            case LoadWord(_, taip, copy_space_offset):
                self.write_indent()
                if not taip.can_live_in_reg():
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset}")
                    self.write(f" i32.add call $intrinsic:dupi32 call $intrinsic:rotate-left i32.const {word.taip.size()} memory.copy\n")
                elif taip.size() == 1:
                    self.write("i32.load8_u\n")
                else:
                    self.write_type(taip)
                    self.write(".load\n")
            case BreakWord():
                self.write_line("br $block")
            case BlockWord(token, body, parameters, returns):
                self.write_indent()
                self.write("(block $block")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_words(module, locals, body.words)
                self.dedent()
                self.write_line(")")
            case LoopWord(_, body, parameters, returns, diverges):
                self.write_indent()
                self.write("(block $block ")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_indent()
                self.write("(loop $loop ")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_words(module, locals, body.words)
                self.write_line("br $loop")
                self.dedent()
                self.write_line(")")
                self.dedent()
                self.write_line(")")
                if diverges:
                    self.write_line("unreachable")
            case IfWord(_, parameters, returns, true_branch, false_branch, diverges):
                self.write_indent()
                self.write("(if")
                self.write_parameters(parameters)
                self.write_returns(returns or [])
                self.write("\n")
                self.indent()
                self.write_line("(then")
                self.indent()
                self.write_words(module, locals, true_branch.words)
                self.dedent()
                self.write_line(")")
                if len(false_branch.words) > 0:
                    self.write_line("(else")
                    self.indent()
                    self.write_words(module, locals, false_branch.words)
                    self.dedent()
                    self.write_line(")")
                self.dedent()
                self.write_line(")")
                if diverges:
                    self.write_line("unreachable")
            case StructWord(_, taip, copy_space_offset, body):
                self.write_indent()
                struct = self.lookup_type_definition(taip.struct)
                assert(not isinstance(struct, Variant))
                if taip.size() == 0:
                    for field in struct.fields.get():
                        self.write("drop ")
                    self.write(f"i32.const 0 ;; make {format_type(taip)}\n")
                    return
                self.write(f";; make {format_type(taip)}\n")
                self.indent()
                self.write_words(module, locals, body.words)
                self.dedent()
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ")
                if taip.size() <= 4:
                    self.write("i32.load ")
                elif taip.size() <= 8:
                    self.write("i64.load ")
                self.write(f";; make {format_type(taip)} end\n")
            case UnnamedStructWord(_, taip, copy_space_offset):
                self.write_indent()
                struct = self.lookup_type_definition(taip.struct)
                assert(not isinstance(struct, Variant))
                if taip.can_live_in_reg():
                    fields = struct.fields.get()
                    if taip.size() == 0:
                        for field in fields:
                            self.write("drop ")
                        self.write(f"i32.const 0 ;; make {format_type(taip)}\n")
                        return
                    if taip.size() <= 8:
                        for i in range(len(fields), 0, -1):
                            offset = struct.field_offset(i - 1)
                            if i != len(fields) and (offset != 0 or taip.size() > 4):
                                if taip.size() <= 4:
                                    self.write("call $intrinsic:flip ")
                                else:
                                    self.write("call $intrinsic:flip-i32-i64 ")
                                    self.flip_i32_i64_used = True
                            if taip.size() > 4:
                                self.write("i64.extend_i32_u ")
                            if offset != 0:
                                if taip.size() <= 4:
                                    self.write(f"i32.const {offset * 8} i32.shl ")
                                else:
                                    self.write(f"i64.const {offset * 8} i64.shl ")
                            if i != len(fields):
                                if taip.size() <= 4:
                                    self.write("i32.or ")
                                else:
                                    self.write("i64.or ")
                        self.write(f";; make {format_type(taip)}\n")
                        return
                self.write(f";; make {format_type(taip)}\n")
                self.indent()
                for i in reversed(range(0, len(struct.fields.get()))):
                    field = struct.fields.get()[i]
                    field_offset = struct.field_offset(i)
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + field_offset} i32.add ")
                    if field.taip.size() > 4 and field.taip.size() <= 8:
                        self.write("call $intrinsic:flip-i64-i32 ")
                        self.flip_i64_i32_used = True
                    else:
                        self.write("call $intrinsic:flip ")
                    self.write_store(field.taip)
                    self.write("\n")
                self.dedent()
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {format_type(taip)} end")
            case StructFieldInitWord(_, taip, copy_space_offset):
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call ")
                if taip.size() <= 4 or taip.size() > 8:
                    self.write("$intrinsic:flip ")
                else:
                    self.write("$intrinsic:flip-i64-i32 ")
                    self.flip_i64_i32_used = True
                self.write_store(taip)
                self.write("\n")
            case MatchWord(_, variant_handle, by_ref, cases, default, parameters, returns):
                variant = self.lookup_type_definition(variant_handle)
                def go(remaining_cases: List[MatchCase]):
                    if len(remaining_cases) == 0:
                        if default is None:
                            if len(cases) != 0:
                                self.write("unreachable")
                            return
                        self.write_words(module, locals, default.words)
                        return
                    case = remaining_cases[0]
                    assert(isinstance(variant, Variant))
                    case_taip = variant.cases.get()[case.tag].taip
                    if variant.size() > 8 or by_ref:
                        self.write(f"call $intrinsic:dupi32 i32.load i32.const {case.tag} i32.eq (if")
                    elif variant.size() > 4 and variant.size() <= 8:
                        self.write(f"call $intrinsic:dupi64 i32.wrap_i64 i32.const {case.tag} i32.eq (if")
                        self.dup_i64_used = True
                    else:
                        self.write(f"call $intrinsic:dupi32 i32.const {case.tag} i32.eq (if")
                    self.write_parameters(parameters)
                    variant_inhabits_i64 = variant.size() <= 8 and variant.size() > 4 and not by_ref
                    if variant_inhabits_i64:
                        self.write(" (param i64)")
                    else:
                        self.write(" (param i32)")

                    if returns is not None:
                        self.write_returns(returns)
                    self.write("\n")
                    self.write_line("(then")
                    self.indent()
                    if case_taip is None:
                        self.write_line("drop")
                    elif case_taip.size() != 0:
                        self.write_indent()
                        if by_ref or variant.size() > 8:
                            self.write("i32.const 4 i32.add")
                            if not by_ref and case_taip.can_live_in_reg():
                                if case_taip.size() <= 8 and case_taip.size() > 4:
                                    self.write(" i64.load")
                                else:
                                    self.write(" i32.load")
                        else:
                            self.write("i64.const 32 i64.shr_u i32.wrap_i64")
                        self.write("\n")
                    elif variant_inhabits_i64:
                        self.write_line("i32.wrap_i64")
                    self.write_words(module, locals, case.body.words)
                    self.dedent()
                    self.write_line(")")
                    self.write_indent()
                    if len(remaining_cases) == 1 and default is not None:
                        self.write("(else\n")
                        self.indent()
                        go(remaining_cases[1:])
                        self.dedent()
                        self.write_indent()
                        self.write("))")
                    else:
                        self.write("(else ")
                        go(remaining_cases[1:])
                        self.write("))")
                self.write_line(f";; match on {variant.name.lexeme}")
                self.write_indent()
                go(cases)
                if returns is None:
                    if len(cases) != 0 or default is not None:
                        self.write("\n")
                        self.write_indent()
                    self.write("unreachable")
                self.write("\n")
            case VariantWord(_, tag, variant_handle, copy_space_offset):
                variant = self.lookup_type_definition(variant_handle)
                assert(isinstance(variant, Variant))
                case_taip = variant.cases.get()[tag].taip
                if variant.size() <= 4:
                    assert(variant.size() == 4)
                    self.write_indent()
                    if case_taip is not None:
                        self.write("drop ")
                    self.write(f"i32.const {tag} ;; store tag {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}\n")
                    return
                if variant.size() <= 8:
                    if case_taip is None:
                        self.write_line(f"i64.const {tag} ;; make {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}")
                    else:
                        self.write_line("i64.extend_i32_u i64.const 32 i64.shl ;; store value")
                        self.write_line(f"i64.const {tag} ;; store tag")
                        self.write_line(f"i64.or ;; make {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}")
                    return
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add i32.const {tag} i32.store ;; store tag")
                if case_taip is not None:
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + 4} i32.add ")
                    if case_taip.size() > 4 and case_taip.size() <= 8:
                        self.flip_i64_i32_used = True
                        self.write("call $intrinsic:flip-i64-i32 ")
                    else:
                        self.write("call $intrinsic:flip ")
                    self.write_store(case_taip)
                    self.write(" ;; store value\n")
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}")
            case TupleMakeWord(token, tupl, copy_space_offset):
                self.write_indent()
                if tupl.can_live_in_reg():
                    if tupl.size() == 0:
                        for item in tupl.items:
                            self.write("drop ")
                        self.write("i32.const 0 ")
                    if tupl.size() <= 4:
                        pass
                    else:
                        # alternative is TODO
                        assert(len(tupl.items) == 2 and tupl.items[0].size() == 4 and tupl.items[1].size() == 4)
                        self.write("call $intrinsic:pack-i32s")
                        self.pack_i32s_used = True
                    self.write(f";; make {format_type(tupl)}\n")
                    return
                self.write(f";; make {format_type(tupl)}\n")
                item_offset = tupl.size()
                self.indent()
                for item in reversed(tupl.items):
                    item_offset -= item.size()
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + item_offset} i32.add ")
                    if item.size() > 4 and item.size() <= 8:
                        self.write("call $intrinsic:flip-i64-i32 ")
                        self.flip_i64_i32_used = True
                    else:
                        self.write("call $intrinsic:flip ")
                    self.write_store(item)
                    self.write("\n")
                self.dedent()
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {format_type(tupl)} end\n")
            case TupleUnpackWord(token, items, copy_space_offset):
                if len(items) == 2 and items[0].size() == 4 and items[1].size() == 4:
                    self.unpack_i32s_used = True
                    self.write_line(f"call $intrinsic:unpack-i32s ;; unpack {listtostr(items, format_type)}")
                    return
                if len(items) == 0:
                    self.write_line(f"drop ;; unpack {listtostr(items, format_type)}")
                    return
                self.write_line(f";; unpack {listtostr(items, format_type)}")
                self.indent()
                offset = 0
                for i, item in enumerate(items):
                    self.write_indent()
                    if item.size() == 0:
                        self.write("i32.const ")
                        continue
                    if i + 1 != len(items):
                        self.write("call $intrinsic:dupi32 ")
                    self.write(f"i32.const {offset} i32.add ")
                    if not item.can_live_in_reg():
                        self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                        self.write("call $intrinsic:rotate-left ")
                        self.write(f"i32.const {item.size()} memory.copy")
                        copy_space_offset += item.size()
                    else:
                        self.write("i32.load")
                    if i + 1 != len(items):
                        self.write(" call $intrinsic:flip")
                    self.write("\n")
                    offset += item.size()
                self.dedent()
            case other:
                assert_never(other)

    def write_store(self, taip: Type):
        if not taip.can_live_in_reg():
            self.write(f"i32.const {taip.size()} memory.copy")
        elif taip.size() == 1:
            self.write("i32.store8")
        elif taip.size() > 4:
            self.write("i64.store")
        else:
            self.write("i32.store")

    def write_set(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, str], target_lives_in_memory: bool, target_taip: Type, loads: List[Load]):
        self.write_indent()
        def write_ident():
            match local_id:
                case LocalId():
                    self.write_local_ident(locals, local_id)
                    return
                case GlobalId():
                    globl = self.globals[local_id]
                    self.write(f"${globl.name.lexeme}:{local_id.module}")
                    return
                case other:
                    assert_never(other)
        if not target_lives_in_memory and len(loads) == 0:
            if isinstance(local_id, LocalId):
                self.write("local.set ")
            else:
                self.write("global.set ")
            write_ident()
            self.write("\n")
            return
        if isinstance(local_id, LocalId):
            self.write("local.get ")
        else:
            self.write("global.get ")
        write_ident()
        if len(loads) == 0:
            if target_taip.size() > 4 and target_taip.size() <= 8:
                self.write(" call $intrinsic:flip-i64-i32 ")
                self.flip_i64_i32_used = True
            else:
                self.write(" call $intrinsic:flip ")
            self.write_store(target_taip)
            self.write("\n")
            return
        last_load = loads[-1]
        if not target_lives_in_memory and is_bitshift(last_load):
            for i, load in enumerate(loads):
                if all(is_bitshift(load) for load in loads[i:]):
                    break
                self.write(f" {load}")
            if isinstance(last_load, I32InI64) or isinstance(last_load, I8InI64):
                self.write(f" i64.const {uhex(0xFFFFFFFF_FFFFFFFF ^ (0xFFFFFFFF << (last_load.offset * 8)))} i64.and ")
            if isinstance(last_load, I8InI32):
                self.write(f" i32.const {uhex(0xFF << (last_load.offset * 8))} i32.and ")

            self.write("call $intrinsic:flip-i32-i64 i64.extend_i32_u ")
            self.flip_i32_i64_used = True
            if isinstance(last_load, I32InI64) and last_load.offset != 0:
                self.write(f"i64.const {last_load.offset * 8} i64.shl ")
            if isinstance(last_load, I8InI32) and last_load.offset != 0:
                self.write(f"i32.const {last_load.offset * 8} i32.shl ")
            self.write("i32.or " if isinstance(last_load, I8InI32) else "i64.or ")
            if not target_lives_in_memory:
                if isinstance(local_id, LocalId):
                    self.write("local.set ")
                else:
                    self.write("global.set ")
                write_ident()
                self.write("\n")
                return
            else:
                if isinstance(local_id, LocalId):
                    self.write("local.get ")
                else:
                    self.write("global.get ")
                write_ident()
                self.write(" call $intrinsic:flip-i64-i32 ")
                self.flip_i64_i32_used = True
                self.write_store(target_taip)
                self.write("\n")
                return

        for i, load in enumerate(loads):
            self.write(f" {load}")
        if target_taip.size() > 4 and target_taip.size() <= 8:
            self.write(" call $intrinsic:flip-i64-i32 ")
        else:
            self.write(" call $intrinsic:flip ")
        self.write_store(target_taip)
        self.write("\n")
        return


    def write_return_struct_receiving(self, offset: int, returns: List[Type]) -> None:
        self.write("\n")
        if all(t.can_live_in_reg() for t in returns):
            return
        for i in range(0, len(returns)):
            self.write_line(f"local.set $s{i}:a")
        for i in range(len(returns), 0, -1):
            ret = returns[len(returns) - i]
            if not ret.can_live_in_reg():
                self.write_line(f"local.get $locl-copy-spac:e i32.const {offset} i32.add call $intrinsic:dupi32 local.get $s{i - 1}:a i32.const {ret.size()} memory.copy")
                offset += ret.size()
            else:
                self.write_line(f"local.get $s{i - 1}:a")

    def write_signature(self, module: int, name: Token, export_name: Token | None, signature: FunctionSignature, instance_id: int | None, locals: Dict[LocalId, Local]) -> None:
        self.write(f"func ${module}:{name.lexeme}")
        if instance_id is not None and instance_id != 0:
            self.write(f":{instance_id}")
        if export_name is not None:
            self.write(f" (export {export_name.lexeme})")
        for parameter in signature.parameters:
            self.write(" (param $")
            for local in locals.values():
                if isinstance(local, ParameterLocal):
                    if local.name.get() == parameter.name.lexeme:
                        if local.lives_in_memory() and local.taip.can_live_in_reg() and local.taip.size() > 4:
                            self.write("v:")
                        break
            self.write(f"{parameter.name.lexeme} ")
            self.write_type(parameter.taip)
            self.write(")")
        self.write_returns(signature.returns)

    def write_type_human(self, taip: Type) -> None:
        self.write(format_type(taip))

    def write_parameters(self, parameters: Sequence[NamedType | Type]) -> None:
        for parameter in parameters:
            if isinstance(parameter, NamedType):
                self.write(f" (param ${parameter.name.lexeme} ")
                self.write_type(parameter.taip)
                self.write(")")
                continue
            self.write(" (param ")
            self.write_type(parameter)
            self.write(")")

    def write_returns(self, returns: List[Type]) -> None:
        for taip in returns:
            self.write(" (result ")
            self.write_type(taip)
            self.write(")")

    def write_intrinsics(self) -> None:
        self.write_line("(func $intrinsic:flip (param $a i32) (param $b i32) (result i32 i32) local.get $b local.get $a)")
        if self.flip_i32_i64_used:
            self.write_line("(func $intrinsic:flip-i32-i64 (param $a i32) (param $b i64) (result i64 i32) local.get $b local.get $a)")
        if self.flip_i64_i32_used:
            self.write_line("(func $intrinsic:flip-i64-i32 (param $a i64) (param $b i32) (result i32 i64) local.get $b local.get $a)")
        if self.flip_i64_i64_used:
            self.write_line("(func $intrinsic:flip-i64-i64 (param $a i64) (param $b i64) (result i64 i64) local.get $b local.get $a)")
        self.write_line("(func $intrinsic:dupi32 (param $a i32) (result i32 i32) local.get $a local.get $a)")
        if self.dup_i64_used:
            self.write_line("(func $intrinsic:dupi64 (param $a i64) (result i64 i64) local.get $a local.get $a)")
        self.write_line("(func $intrinsic:rotate-left (param $a i32) (param $b i32) (param $c i32) (result i32 i32 i32) local.get $b local.get $c local.get $a)")
        if self.pack_i32s_used:
            self.write_line("(func $intrinsic:pack-i32s (param $a i32) (param $b i32) (result i64) local.get $a i64.extend_i32_u local.get $b i64.extend_i32_u i64.const 32 i64.shl i64.or)")
        if self.unpack_i32s_used:
            self.write_line("(func $intrinsic:unpack-i32s (param $a i64) (result i32) (result i32) local.get $a i32.wrap_i64 local.get $a i64.const 32 i64.shr_u i32.wrap_i64)")
        if self.guard_stack:
            self.write_line("(func $stack-overflow-guar:d i32.const 1 global.get $stac:k global.get $stack-siz:e i32.lt_u i32.div_u drop)")

    def write_function_table(self) -> None:
        if len(self.function_table) == 0:
            self.write_line("(table funcref (elem))")
            return
        self.write_line("(table funcref (elem $intrinsic:flip")
        self.indent()
        self.write_indent()
        functions = list(self.function_table.items())
        functions.sort(key=lambda kv: kv[1])
        for i, (handle, _) in enumerate(functions):
            module = self.modules[handle.module]
            if isinstance(handle, FunctionHandle):
                function = module.functions[handle.index]
                if isinstance(function, GenericFunction):
                    assert(handle.instance is not None)
                    function = function.instances[handle.instance]
                    name = f"${handle.module}:{function.name.lexeme}:{handle.instance}"
                else:
                    name = f"${handle.module}:{function.name.lexeme}"
            else:
                name = "TODO"
            self.write(f"{name}")
            if i + 1 != len(functions):
                self.write(" ")
        self.write("))\n")
        self.dedent()

    def write_globals(self, ptr: int) -> int:
        for global_id, globl in self.globals.items():
            self.write_indent()
            size = globl.taip.size()
            lives_in_memory = globl.was_reffed or not globl.taip.can_live_in_reg()
            initial_value = ptr if lives_in_memory else 0
            taip = "i64" if not lives_in_memory and size > 4 and size <= 8 else "i32"
            self.write(f"(global ${globl.name.lexeme}:{global_id.module} {taip if lives_in_memory else f"(mut {taip})"} ({taip}.const {initial_value}))\n")
            if not lives_in_memory:
                continue
            ptr += globl.taip.size()
        return ptr

    def write_data(self, data: bytes) -> None:
        self.write_indent()
        self.write("(data (i32.const 0) \"")
        def escape_char(char: int) -> str:
            if char == b"\\"[0]:
               return "\\\\"
            if char == b"\""[0]:
                return "\\\""
            if char == b"\t"[0]:
               return "\\t"
            if char == b"\r"[0]:
               return "\\r"
            if char == b"\n"[0]:
               return "\\n"
            if char >= 32 and char <= 126:
               return chr(char)
            hex_digits = "0123456789abcdef"
            return f"\\{hex_digits[char >> 4]}{hex_digits[char & 15]}"
        for char in data:
            self.write(escape_char(char))
        self.write("\")\n")

    def write_extern(self, module_id: int, extern: Extern) -> None:
        self.write_indent()
        self.write("(import ")
        self.write(extern.extern_module)
        self.write(" ")
        self.write(extern.extern_name)
        self.write(" (")
        self.write_signature(module_id, extern.name, None, extern.signature, None, {})
        self.write("))")

    def write_type(self, taip: Type) -> None:
        size = taip.size()
        if size > 4 and size <= 8:
            self.write("i64")
        else:
            self.write("i32")

Mode = Literal["lex"] | Literal["parse"] | Literal["check"] | Literal["monomize"] | Literal["compile"] | Literal["inference-tree"]

def run(path: str, mode: Mode, guard_stack: bool, stdin: str | None = None) -> str:
    if path == "-":
        file = stdin if stdin is not None else sys_stdin.get()
    else:
        with open(path, 'r') as reader:
            file = reader.read()
    tokens = Lexer(file).lex()
    if mode == "lex":
        return "\n".join([format(token.format_instrs()) for token in tokens])
    if mode == "parse":
        module = Parser(path, file, tokens).parse()
        return format(module.format_instrs())
    modules: Dict[str, parser.Module] = {}
    load_recursive(modules, os.path.normpath(path), None, stdin)
    resolved_modules = resolve_modules({ k: m.top_items for k,m in modules.items()})
    if mode == "check":
        return format(resolved_modules.format_instrs(format_str))
    function_table, mono_modules = Monomizer(resolved_modules).monomize()
    if mode == "monomize":
        return "TODO"
    if mode == "inference-tree":
        return ""
    for mono_module in mono_modules.values():
        merge_locals_module(mono_module)
    return WatGenerator(mono_modules, function_table, guard_stack).write_wat_module()

help = """The native Watim compiler

Usage: watim <command> <watim-source-file> [options]
Commands:
  lex       [path]   Lex code and print the Tokens.
  parse     [path]   Parse code and print the AST
  check     [path]   Typecheck and print the AST
  monomize  [path]   Monomize the entire program
  optimize  [path]   Optimize the entire program
  compile   [path]   Compile the entire program
Options:
  -q, --quiet  Don't print any logs to stderr
"""

@dataclass
class CliArgException(Exception):
    message: str

def main(argv: List[str], stdin: str | None = None) -> str:
    argv = [arg for arg in argv if arg != "-q"]
    if len(argv) == 1:
        raise CliArgException(help)
    if argv[1] == "units":
        suite = unittest.TestSuite()
        classes = [DetermineLoadsToValueTests]
        for klass in classes:
            for method in dir(klass):
                if method.startswith("test_"):
                    suite.addTest(klass(method))
        runner = unittest.TextTestRunner()
        runner.run(suite)
        return ""
    mode: Mode = "compile"
    if len(argv) >= 2 and argv[1] == "lex":
        mode = "lex"
        path = argv[2] if len(argv) > 2 else "-"
    elif len(argv) >= 2 and argv[1] == "parse":
        mode = "parse"
        path = argv[2] if len(argv) > 2 else "-"
    elif len(argv) > 2 and argv[1] == "check":
        mode = "check"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "monomize":
        mode = "monomize"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "compile":
        mode = "compile"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "inference-tree":
        mode = "inference-tree"
        path = argv[2]
    else:
        path = argv[1]
    return run(path, mode, "--guard-stack" in argv, stdin)

if __name__ == "__main__":
    try:
        print(main(sys.argv))
    except CliArgException as e:
        print(e.message, file=sys.stderr)
        exit(1)
    except ParserException as e:
        print(e.display(), file=sys.stderr)
        exit(1)
    except ResolverException as e:
        print(e.display(), file=sys.stderr)
        exit(1)

