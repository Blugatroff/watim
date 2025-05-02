from typing import List, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq, format_optional
from lexer import Token
from parsing.types import Type, CustomTypeType, ForeignType

@dataclass
class NumberWord(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Number", [self.token])

@dataclass
class BreakWord(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Break", [self.token])

type Word = 'NumberWord | StringWord | CallWord | GetWord | RefWord | SetWord | StoreWord | InitWord | CallWord | ForeignCallWord | FunRefWord | IfWord | LoadWord | LoopWord | BlockWord | BreakWord | CastWord | SizeofWord | GetFieldWord | IndirectCallWord | StructWord | StructWordNamed | MatchWord | VariantWord | TupleUnpackWord | MakeTupleWord | StackAnnotation | InlineRefWord'
@dataclass
class StringWord(Formattable):
    token: Token
    data: bytearray
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("StringWord", [self.token])

@dataclass
class GetWord(Formattable):
    token: Token
    ident: Token
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GetLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class RefWord(Formattable):
    token: Token
    ident: Token
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("RefLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class SetWord(Formattable):
    token: Token
    ident: Token
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("SetLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class InlineRefWord(Formattable):
    token: Token

@dataclass
class StoreWord(Formattable):
    token: Token
    ident: Token
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("StoreLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class InitWord(Formattable):
    token: Token
    ident: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("InitLocal", [self.token, self.ident])

@dataclass
class ForeignCallWord(Formattable):
    module: Token
    ident: Token
    generic_arguments: Tuple[Type, ...]

@dataclass
class CallWord(Formattable):
    ident: Token
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("LocalCall", [
            self.ident,
            format_seq(self.generic_arguments, multi_line=True)])

@dataclass
class FunRefWord(Formattable):
    start: Token
    call: CallWord | ForeignCallWord
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunRef", [self.start, self.call])

@dataclass
class IfWord(Formattable):
    token: Token
    true_words: 'Words'
    false_words: 'Words | None'
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("If", [
            self.token,
            self.true_words,
            format_optional(self.false_words)])

@dataclass
class LoadWord(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Load", [self.token])

@dataclass
class BlockAnnotation(Formattable):
    parameters: List[Type]
    returns: List[Type]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("BlockAnnotation", [
            format_seq(self.parameters),
            format_seq(self.returns)])

@dataclass
class Words(Formattable):
    words: Tuple[Word, ...]
    end: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Words", [
            format_seq(self.words, multi_line=True),
            self.end])

@dataclass
class LoopWord(Formattable):
    token: Token
    words: Words
    annotation: BlockAnnotation | None
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Loop", [
            self.token,
            format_optional(self.annotation),
            self.words])

@dataclass
class BlockWord(Formattable):
    token: Token
    words: Words
    annotation: BlockAnnotation | None
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Block", [
            self.token,
            format_optional(self.annotation),
            self.words])

@dataclass
class CastWord(Formattable):
    token: Token
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Cast", [
            self.token, self.taip])

@dataclass
class SizeofWord(Formattable):
    token: Token
    taip: Type

@dataclass
class GetFieldWord(Formattable):
    token: Token
    fields: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GetField", [self.token, format_seq(self.fields)])

@dataclass
class IndirectCallWord(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("IndirectCall", [self.token])

@dataclass
class StructWordNamed(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType
    words: Tuple[Word, ...]

@dataclass
class StructWord(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType

@dataclass
class VariantWord(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType
    case: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("MakeVariant", [self.token, self.taip, self.case])

@dataclass
class MatchCase(Formattable):
    case: Token
    name: Token
    words: Tuple[Word, ...]

@dataclass
class MatchWord(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType | None
    cases: Tuple[MatchCase, ...]
    default: MatchCase | None

@dataclass
class TupleUnpackWord(Formattable):
    token: Token

@dataclass
class MakeTupleWord(Formattable):
    token: Token
    items: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("MakeTuple", [self.token, self.items])

@dataclass
class StackAnnotation(Formattable):
    token: Token
    types: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("StackAnnotation", [
            self.token,
            format_seq(self.types)])
