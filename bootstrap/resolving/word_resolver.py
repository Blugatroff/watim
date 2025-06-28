from typing import Sequence, Tuple, Dict, NoReturn
from dataclasses import dataclass
import copy

from indexed_dict import IndexedDict

import parsing.words as parsing
from lexer import Token
from resolving.words import Word, IntrinsicWord, FunctionHandle, GlobalId
from resolving.module import Module, ResolveException
import resolving.words as resolved
from resolving.intrinsics import INTRINSICS
from resolving.types import with_generics
from resolving.type_resolver import TypeResolver, TypeLookup
from resolving.top_items import Import, Local, LocalName, CustomTypeHandle, Variant, Struct, Global, LocalId, FunctionSignature, VariantImport
from resolving.env import Env

@dataclass
class StructLiteralEnv:
    struct: CustomTypeHandle
    all_fields: Dict[str, int]
    remaining_fields: Dict[str, int]

    @staticmethod
    def of_struct(struct: Struct, handle: CustomTypeHandle) -> 'StructLiteralEnv':
        return StructLiteralEnv(
            handle,
            { field.name.lexeme: field_index for field_index, field in enumerate(struct.fields) },
            { field.name.lexeme: field_index for field_index, field in enumerate(struct.fields) })

@dataclass
class WordResolver:
    module_id: int
    module_path: str
    imports: Dict[str, Tuple[Import, ...]]
    globals: IndexedDict[str, Global]
    signatures: IndexedDict[str, FunctionSignature]
    type_resolver: TypeResolver
    modules: IndexedDict[str, Module]
    type_lookup: TypeLookup
    env: Env
    struct_literal_env: StructLiteralEnv | None = None

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolveException(self.module_path, token, message)

    def with_env(self, env: Env) -> 'WordResolver':
        self = copy.copy(self)
        self.env = env
        return self

    def with_struct_literal_env(self, env: StructLiteralEnv) -> 'WordResolver':
        self = copy.copy(self)
        self.struct_literal_env = env
        return self

    def without_struct_literal_env(self) -> 'WordResolver':
        self = copy.copy(self)
        self.struct_literal_env = None
        return self

    def resolve_words(self, words: Sequence[parsing.Word]) -> Tuple[Word, ...]:
        return tuple(resolved_word for word in words for resolved_word in self.resolve_word(word))

    def resolve_word(self, word: parsing.Word) -> Sequence[Word]:
        match word:
            case parsing.NumberWord():
                return word,
            case parsing.StringWord():
                return word,
            case parsing.BreakWord():
                return word,
            case parsing.CallWord():
                call = self.resolve_call(word)
                if call is None:
                    if word.ident.lexeme in INTRINSICS:
                        return IntrinsicWord(
                                word.ident,
                                INTRINSICS[word.ident.lexeme],
                                self.type_resolver.resolve_types(word.generic_arguments)),
                    self.abort(word.ident, f"function `{word.ident.lexeme}` not found")
                return call,
            case parsing.ForeignCallWord():
                return self.resolve_foreign_call(word),
            case parsing.CastWord():
                return resolved.CastWord(word.token, self.type_resolver.resolve_type(word.taip)),
            case parsing.InitWord():
                return self.resolve_init_local(word),
            case parsing.GetWord():
                return resolved.GetWord(word.ident, self.lookup_variable(word.ident), word.fields),
            case parsing.SetWord():
                return resolved.SetWord(word.ident, self.lookup_variable(word.ident), word.fields),
            case parsing.RefWord():
                return resolved.RefWord(word.ident, self.lookup_variable(word.ident), word.fields),
            case parsing.StoreWord():
                return self.resolve_store_local(word),
            case parsing.LoadWord():
                return word,
            case parsing.SizeofWord():
                return resolved.SizeofWord(word.token, self.type_resolver.resolve_type(word.taip)),
            case parsing.VariantWord():
                return self.resolve_make_variant(word),
            case parsing.IndirectCallWord():
                return word,
            case parsing.FunRefWord():
                call = self.resolve_call(word.call)
                if call is None:
                    self.abort(word.call.ident, f"function `{word.call.ident.lexeme}` not found")
                return resolved.FunRefWord(call),
            case parsing.MakeTupleWord():
                return word,
            case parsing.GetFieldWord():
                return word,
            case parsing.TupleUnpackWord():
                return word,
            case parsing.InlineRefWord():
                return self.resolve_inline_ref_word(word)
            case parsing.StackAnnotation():
                return resolved.StackAnnotation(word.token, self.type_resolver.resolve_types(word.types)),
            case parsing.StructWord():
                return resolved.StructWord(word.token, self.type_resolver.resolve_custom_type(word.taip)),
            case parsing.StructWordNamed():
                return self.resolve_struct_word_named(word),
            case parsing.MatchWord():
                return self.resolve_match_word(word),
            case parsing.IfWord():
                return resolved.IfWord(
                    word.token,
                    self.resolve_scope(word.true_words.words),
                    self.resolve_scope(word.false_words.words if word.false_words is not None else ())),
            case parsing.LoopWord():
                return resolved.LoopWord(word.token, self.resolve_scope(word.words.words), self.resolve_block_annotation(word.annotation)),
            case parsing.BlockWord():
                return resolved.BlockWord(
                    word.token,
                    word.words.end,
                    self.resolve_scope(word.words.words),
                    self.resolve_block_annotation(word.annotation)),

    def resolve_inline_ref_word(self, word: parsing.InlineRefWord) -> Sequence[resolved.Word]:
        local = Local(LocalName("synth:ref"), None)
        local_id = self.env.insert(local)
        return (resolved.InitWord(word.token, local_id), resolved.RefWord(word.token, local_id, ()))

    def resolve_call(self, word: parsing.CallWord | parsing.ForeignCallWord) -> resolved.CallWord | None:
        if isinstance(word, parsing.ForeignCallWord):
            return self.resolve_foreign_call(word)
        if word.ident.lexeme not in self.signatures:
            for imports in self.imports.values():
                for imp in imports:
                    for item in imp.items:
                        if isinstance(item.handle, FunctionHandle) and item.name.lexeme == word.ident.lexeme:
                            return resolved.CallWord(
                                word.ident,
                                item.handle,
                                self.type_resolver.resolve_types(word.generic_arguments))
            return None

        function_handle = FunctionHandle(self.module_id, self.signatures.index_of(word.ident.lexeme))
        return resolved.CallWord(
            word.ident,
            function_handle,
            self.type_resolver.resolve_types(word.generic_arguments))

    def resolve_foreign_call(self, word: parsing.ForeignCallWord) -> resolved.CallWord:
        if word.module.lexeme not in self.imports:
            self.abort(word.module, "module not found")
        for imp in self.imports[word.module.lexeme]:
            module = self.modules.index(imp.module)
            item = module.lookup_item(word.ident)
            if isinstance(item, FunctionHandle):
                return resolved.CallWord(word.ident, item, self.type_resolver.resolve_types(word.generic_arguments))
        self.abort(word.ident, f"function `{word.ident.lexeme}` not found")

    def lookup_variable(self, name: Token) -> LocalId | GlobalId:
        local_id = self.env.lookup(name)
        if local_id is not None:
            return local_id
        if name.lexeme in self.globals:
            return GlobalId(self.module_id, self.globals.index_of(name.lexeme))
        self.abort(name, "variable not found")

    def resolve_init_local(self, word: parsing.InitWord) -> resolved.InitWord | resolved.StructFieldInitWord:
        if self.struct_literal_env is not None:
            field_name = word.ident.lexeme
            if field_name in self.struct_literal_env.all_fields:
                del self.struct_literal_env.remaining_fields[field_name]
                field_index = self.struct_literal_env.all_fields[field_name]
                return resolved.StructFieldInitWord(word.token, self.struct_literal_env.struct, field_index)
        local_id = self.env.insert(Local(LocalName(word.ident), None))
        return resolved.InitWord(word.ident, local_id)

    def resolve_scope(self, parsed_words: Tuple[parsing.Word, ...], keep_struct_literal_env=False) -> resolved.Scope:
        env = self.env.child()
        if not keep_struct_literal_env:
            self = self.without_struct_literal_env()
        words = self.with_env(env).resolve_words(parsed_words)
        return resolved.Scope(env.scope_id, words)

    def resolve_store_local(self, word: parsing.StoreWord) -> resolved.StoreWord:
        local_id = self.env.lookup(word.ident)
        if local_id is None:
            self.abort(word.ident, "local not found")
        return resolved.StoreWord(word.ident, local_id, word.fields)

    def resolve_block_annotation(self, annotation: parsing.BlockAnnotation | None) -> resolved.BlockAnnotation | None:
        if annotation is None:
            return None
        return resolved.BlockAnnotation(
                self.type_resolver.resolve_types(annotation.parameters),
                self.type_resolver.resolve_types(annotation.returns))

    def resolve_make_variant(self, word: parsing.VariantWord) -> resolved.VariantWord:
        variant_type = self.type_resolver.resolve_custom_type(word.taip)
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
        return resolved.VariantWord(word.token, tag, variant_type)

    def resolve_match_word(self, word: parsing.MatchWord) -> resolved.MatchWord:
        variant_handle = self.infer_match_type(word)
        variant = self.type_lookup.lookup(variant_handle)
        assert(isinstance(variant, Variant))
        cases = tuple(self.resolve_match_case(variant, cays) for cays in word.cases)
        default = resolved.DefaultCase(
                word.default.name,
                self.resolve_scope(word.default.words)) if word.default is not None else None
        return resolved.MatchWord(
                word.token,
                variant_handle,
                cases,
                default)

    def infer_match_type(self, match: parsing.MatchWord) -> CustomTypeHandle:
        inferred: CustomTypeHandle | None = None
        for case in match.cases:
            if case.module is None:
                if case.variant is None:
                    handle,_ = self.lookup_constructor(case.name)
                    if inferred is not None and inferred != handle:
                        self.abort(case.name, "constructors belong to different variants")
                    inferred = handle
                else:
                    handle = self.type_resolver.resolve_custom_type_name(case.variant)
                    if not isinstance(self.type_lookup.lookup(handle), Variant):
                        self.abort(case.variant, "expected variant")
                    if inferred is not None and inferred != handle:
                        self.abort(case.name, "constructors belong to different variants")
                    inferred = handle
            else:
                assert(case.variant is not None)
                if case.module.lexeme not in self.imports:
                    self.abort(case.module, "module not found")
                for qualifier,imports in self.imports.items():
                    if qualifier != case.module.lexeme:
                        continue
                    found = False
                    for imp in imports:
                        module = self.modules.index(imp.module)
                        if case.variant.lexeme in module.type_definitions:
                            handle = CustomTypeHandle(imp.module, module.type_definitions.index_of(case.variant.lexeme))
                            if inferred is not None and inferred != handle:
                                self.abort(case.name, "foo constructors belong to different variants")
                            inferred = handle
                            found = True
                    if not found:
                        self.abort(case.variant, "variant not found")

        if inferred is None:
            self.abort(match.token, "could not determine type of match")
        return inferred

    def lookup_constructor(self, name: Token) -> Tuple[CustomTypeHandle, int]:
        for imports in self.imports.values():
            for imp in imports:
                for item in imp.items:
                    if isinstance(item, VariantImport):
                        variant = self.type_lookup.lookup(item.handle)
                        assert(not isinstance(variant, Struct))
                        for constructor in item.constructors:
                            if variant.cases[constructor].name.lexeme == name.lexeme:
                                return item.handle, constructor
                    pass
        for index, type_definition in enumerate(self.type_lookup.type_definitions.values()):
            if not isinstance(type_definition, Struct):
                for tag, case in enumerate(type_definition.cases):
                    if case.name.lexeme == name.lexeme:
                        return CustomTypeHandle(self.module_id, index), tag
        self.abort(name, f"constructor {name.lexeme} not found")

    def resolve_match_case(self, variant: Variant, cays: parsing.MatchCase) -> resolved.MatchCase:
        tag = None
        for i, variant_case in enumerate(variant.cases):
            if variant_case.name.lexeme == cays.name.lexeme:
                tag = i
        if tag is None:
            self.abort(cays.name, "case not found")
        return resolved.MatchCase(tag, cays.name, self.resolve_scope(cays.words))

    def resolve_struct_word_named(self, word: parsing.StructWordNamed) -> resolved.StructWordNamed:
        taip = self.type_resolver.resolve_custom_type(word.taip)
        struct = self.type_lookup.lookup(taip.type_definition)
        if isinstance(struct, Variant):
            self.abort(taip.name, "expected struct")
        struct_literal_env = StructLiteralEnv.of_struct(struct, taip.type_definition)
        words = self.with_struct_literal_env(struct_literal_env).resolve_scope(word.words, keep_struct_literal_env=True)
        if len(struct_literal_env.remaining_fields) != 0:
            error_message = "missing fields in struct literal:"
            for field_name,field_index in struct_literal_env.remaining_fields.items():
                field_taip = with_generics(struct.fields[field_index].taip, taip.generic_arguments)
                error_message += f"\n\t{field_name}: {self.type_lookup.type_pretty(field_taip)}"
            self.abort(word.token, error_message)
        return resolved.StructWordNamed(
            word.token,
            taip,
            words)

