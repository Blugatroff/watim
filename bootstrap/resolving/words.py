from typing import List, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq, named_record, format_optional, format_list, format_str
from lexer import Token

from resolving.intrinsics import IntrinsicWord
from resolving.types import CustomTypeType, PtrType, Type, FunctionType, CustomTypeHandle, TupleType
from parsing.words import BreakWord, NumberWord

@dataclass(frozen=True, eq=True)
class ScopeId(Formattable):
    raw: int
    def format_instrs(self) -> List[FormatInstr]:
        return [self.raw]
ROOT_SCOPE: ScopeId = ScopeId(0)

@dataclass
class Scope(Formattable):
    id: ScopeId
    words: List['ResolvedWord']
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Scope", [self.id, format_seq(self.words, multi_line=True)])


@dataclass(frozen=True, eq=True)
class GlobalId(Formattable):
    module: int
    index: int
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Globalid", [self.module, self.index])

@dataclass(frozen=True, eq=True)
class LocalId(Formattable):
    name: str
    scope: ScopeId
    shadow: int
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("LocalId", [
            format_str(self.name),
            self.scope,
            self.shadow])

@dataclass
class FieldAccess(Formattable):
    name: Token
    source_taip: CustomTypeType | PtrType
    target_taip: Type
    field_index: int
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FieldAccess", [
            self.name, self.source_taip, self.target_taip, self.field_index])

@dataclass
class StringWord(Formattable):
    token: Token
    offset: int
    len: int

@dataclass
class LoadWord(Formattable):
    token: Token
    taip: Type

@dataclass
class InitWord(Formattable):
    name: Token
    local_id: LocalId
    taip: Type

@dataclass
class GetWord(Formattable):
    token: Token
    local_id: LocalId | GlobalId
    var_taip: Type
    fields: Tuple[FieldAccess, ...]
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GetLocal", [
            self.token,
            self.local_id,
            self.var_taip,
            self.taip,
            format_seq(self.fields, multi_line=True)])

@dataclass
class RefWord(Formattable):
    token: Token
    local_id: LocalId | GlobalId
    fields: Tuple[FieldAccess, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("RefLocal", [self.token, self.local_id, format_seq(self.fields, multi_line=True)])

@dataclass
class SetWord(Formattable):
    token: Token
    local_id: LocalId | GlobalId
    fields: Tuple[FieldAccess, ...]

@dataclass
class StoreWord(Formattable):
    token: Token
    local: LocalId | GlobalId
    fields: Tuple[FieldAccess, ...]

@dataclass
class CallWord(Formattable):
    name: Token
    function: 'FunctionHandle'
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Call", [self.name, self.function, format_seq(self.generic_arguments)])

@dataclass(frozen=True, eq=True)
class FunctionHandle(Formattable):
    module: int
    index: int
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunctionHandle", [self.module, self.index])

@dataclass
class FunRefWord(Formattable):
    call: CallWord
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunRef", [self.call])

@dataclass
class IfWord(Formattable):
    token: Token
    parameters: List[Type]
    returns: List[Type] | None
    true_branch: Scope
    false_branch: Scope
    diverges: bool
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("If", [
            ("token", self.token),
            ("parameters", format_seq(self.parameters)),
            ("returns", format_optional(self.returns, format_list)),
            ("true-branch", self.true_branch),
            ("false-branch", self.false_branch)])

@dataclass
class LoopWord(Formattable):
    token: Token
    body: Scope
    parameters: Tuple[Type, ...]
    returns: Tuple[Type, ...]
    diverges: bool
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Loop", [
            ("token", self.token),
            ("parameters", format_seq(self.parameters)),
            ("returns", format_optional(None if self.diverges else self.returns, format_seq)),
            ("body", self.body)])


@dataclass
class BlockWord(Formattable):
    token: Token
    body: Scope
    parameters: Tuple[Type, ...]
    returns: Tuple[Type, ...]

@dataclass
class CastWord(Formattable):
    token: Token
    source: Type
    taip: Type

@dataclass
class SizeofWord(Formattable):
    token: Token
    taip: Type

@dataclass
class GetFieldWord(Formattable):
    token: Token
    fields: Tuple[FieldAccess, ...]
    on_ptr: bool

@dataclass
class IndirectCallWord(Formattable):
    token: Token
    taip: FunctionType
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("IndirectCall", [self.token, self.taip])

@dataclass
class StructFieldInitWord(Formattable):
    token: Token
    struct: CustomTypeHandle
    taip: Type
    field_index: int

    generic_arguments: Tuple[Type, ...]

@dataclass
class StructWord(Formattable):
    token: Token
    taip: CustomTypeType
    body: Scope
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("StructWordNamed", [
            ("token", self.token),
            ("type", self.taip),
            ("body", self.body)])

@dataclass
class UnnamedStructWord(Formattable):
    token: Token
    taip: CustomTypeType
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("StructWord", [
            ("token", self.token),
            ("type", self.taip)])

@dataclass
class VariantWord(Formattable):
    token: Token
    tag: int
    variant: CustomTypeType

@dataclass
class MatchCase(Formattable):
    taip: Type | None
    tag: int
    body: Scope

@dataclass
class MatchWord(Formattable):
    token: Token
    variant: CustomTypeType
    by_ref: bool
    cases: List[MatchCase]
    default: Scope | None
    parameters: List[Type]
    returns: List[Type] | None

@dataclass
class TupleMakeWord(Formattable):
    token: Token
    taip: TupleType

@dataclass
class TupleUnpackWord(Formattable):
    token: Token
    items: TupleType

ResolvedWord = (
      NumberWord
    | StringWord
    | CallWord
    | GetWord
    | RefWord
    | SetWord
    | StoreWord
    | FunRefWord
    | IfWord
    | LoadWord
    | LoopWord
    | BlockWord
    | BreakWord
    | CastWord
    | SizeofWord
    | GetFieldWord
    | IndirectCallWord
    | IntrinsicWord
    | InitWord
    | StructFieldInitWord
    | StructWord
    | UnnamedStructWord
    | VariantWord
    | MatchWord
    | TupleMakeWord
    | TupleUnpackWord
)
