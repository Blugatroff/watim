from typing import List, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq, named_record, format_optional, format_list
from lexer import Token

from parsing.words import BreakWord, NumberWord
from resolving.types import CustomTypeType, PtrType, Type, FunctionType, TupleType
from resolving.words import ScopeId, LocalId, GlobalId, FunctionHandle
from checking import intrinsics as intrinsics
from checking.intrinsics import IntrinsicWord

@dataclass
class Scope(Formattable):
    id: ScopeId
    words: List['Word']
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Scope", [self.id, format_seq(self.words, multi_line=True)])

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
    function: FunctionHandle
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Call", [self.name, self.function, format_seq(self.generic_arguments)])

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
    field_index: int

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
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("VariantWord", [
            ("token", self.token),
            ("tag", self.tag),
            ("type", self.variant)])

@dataclass
class MatchCase(Formattable):
    taip: Type | None
    tag: int
    body: Scope
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("MatchCase", [format_optional(self.taip), self.tag, self.body])

@dataclass
class MatchWord(Formattable):
    token: Token
    variant: CustomTypeType
    by_ref: bool
    cases: Tuple[MatchCase, ...]
    default: Scope | None
    parameters: Tuple[Type, ...]
    returns: Tuple[Type, ...] | None
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Match", [
            ("token", self.token),
            ("variant", self.variant),
            ("by-ref", self.by_ref),
            ("cases", format_seq(self.cases, multi_line=True)),
            ("default", format_optional(self.default)),
            ("parameters", format_seq(self.parameters)),
            ("returns", format_optional(self.returns, format_seq))])

@dataclass
class TupleMakeWord(Formattable):
    token: Token
    taip: TupleType

@dataclass
class TupleUnpackWord(Formattable):
    token: Token
    items: TupleType

type Word = (
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
