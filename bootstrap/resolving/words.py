from typing import List, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq, named_record, format_str, format_optional
from lexer import Token

from resolving.intrinsics import IntrinsicType
from resolving.types import CustomTypeType, Type, CustomTypeHandle
from parsing.words import (
    BreakWord as BreakWord,
    NumberWord as NumberWord,
    StringWord as StringWord,
    LoadWord as LoadWord,
    IndirectCallWord as IndirectCallWord,
    MakeTupleWord as MakeTupleWord,
    GetFieldWord as GetFieldWord,
    TupleUnpackWord as TupleUnpackWord,
)

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
    | StructWordNamed
    | StructWord
    | VariantWord
    | MatchWord
    | MakeTupleWord
    | TupleUnpackWord
    | StackAnnotation
)

@dataclass(frozen=True, eq=True)
class ScopeId(Formattable):
    raw: int
    def format_instrs(self) -> List[FormatInstr]:
        return [self.raw]
ROOT_SCOPE: ScopeId = ScopeId(0)

@dataclass
class Scope(Formattable):
    id: ScopeId
    words: Tuple[Word, ...]
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
class InitWord(Formattable):
    token: Token
    local_id: LocalId

@dataclass
class GetWord(Formattable):
    token: Token
    local_id: LocalId | GlobalId
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GetLocal", [
            self.token,
            self.local_id,
            format_seq(self.fields, multi_line=True)])

@dataclass
class RefWord(Formattable):
    token: Token
    local_id: LocalId | GlobalId
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("RefLocal", [self.token, self.local_id, format_seq(self.fields, multi_line=True)])

@dataclass
class SetWord(Formattable):
    token: Token
    local_id: LocalId | GlobalId
    fields: Tuple[Token, ...]

@dataclass
class StoreWord(Formattable):
    token: Token
    local: LocalId | GlobalId
    fields: Tuple[Token, ...]

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
    true_branch: Scope
    false_branch: Scope
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("If", [
            ("token", self.token),
            ("true-branch", self.true_branch),
            ("false-branch", self.false_branch)])

@dataclass
class BlockAnnotation(Formattable):
    parameters: Tuple[Type, ...]
    returns: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("BlockAnnotation", [
            format_seq(self.parameters),
            format_seq(self.returns)])

@dataclass
class LoopWord(Formattable):
    token: Token
    body: Scope
    annotation: BlockAnnotation | None
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Loop", [
            ("token", self.token),
            ("body", self.body),
            ("annotation", format_optional(self.annotation))])


@dataclass
class BlockWord(Formattable):
    token: Token
    end: Token
    body: Scope
    annotation: BlockAnnotation | None

@dataclass
class CastWord(Formattable):
    token: Token
    taip: Type

@dataclass
class SizeofWord(Formattable):
    token: Token
    taip: Type

@dataclass
class StructFieldInitWord(Formattable):
    token: Token
    struct: CustomTypeHandle
    field_index: int

@dataclass
class StructWordNamed(Formattable):
    token: Token
    taip: CustomTypeType
    body: Scope
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("StructWordNamed", [
            ("token", self.token),
            ("type", self.taip),
            ("body", self.body)])

@dataclass
class StructWord(Formattable):
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
        return named_record("MakeVariant", [
            ("token", self.token),
            ("tag", self.tag),
            ("type", self.variant)])

@dataclass
class MatchCase(Formattable):
    tag: int
    name: Token
    body: Scope
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("MatchCase", [self.tag, self.name, self.body])

@dataclass
class MatchWord(Formattable):
    token: Token
    variant: CustomTypeHandle
    cases: Tuple[MatchCase, ...]
    default: Scope | None
    underscore: Token | None
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Match", [
            ("token", self.token),
            ("variant", self.variant),
            ("cases", format_seq(self.cases, multi_line=True)),
            ("default", format_optional(self.default))])

@dataclass
class StackAnnotation(Formattable):
    token: Token
    types: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("StackAnnotation", [
            self.token,
            format_seq(self.types)])

@dataclass(frozen=True)
class IntrinsicWord(Formattable):
    token: Token
    ty: IntrinsicType
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Intrinsic", [self.token, self.ty, format_seq(self.generic_arguments)])

