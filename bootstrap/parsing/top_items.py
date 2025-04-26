from typing import List, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq, format_optional
from lexer import Token
from parsing.words import Word
from parsing.types import Type, NamedType

type TypeDefinition = 'Struct | Variant'

type TopItem = 'Import | TypeDefinition | Global | Function | Extern'

@dataclass(frozen=True, eq=True)
class Import(Formattable):
    token: Token
    file_path: Token
    qualifier: Token
    items: Tuple[Token, ...]

@dataclass
class FunctionSignature(Formattable):
    export_name: Token | None
    name: Token
    generic_parameters: Tuple[Token, ...]
    parameters: Tuple[NamedType, ...]
    returns: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Signature", [
            format_seq(self.generic_parameters),
            format_seq(self.parameters),
            format_seq(self.returns)])

@dataclass
class Extern(Formattable):
    token: Token
    module: Token
    name: Token
    signature: FunctionSignature
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Extern", [
            self.token,
            self.module,
            self.name,
            self.signature.name,
            self.signature])

@dataclass
class Global(Formattable):
    token: Token
    name: Token
    taip: Type

@dataclass
class Function(Formattable):
    token: Token
    signature: FunctionSignature
    body: Tuple[Word, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Function", [
            self.token,
            self.signature.name,
            format_optional(self.signature.export_name),
            self.signature,
            format_seq(self.body, multi_line=True)])

@dataclass
class Struct(Formattable):
    token: Token
    name: Token
    fields: Tuple[NamedType, ...]
    generic_parameters: Tuple[Token, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Struct", [
            self.token,
            self.name,
            format_seq(self.generic_parameters),
            format_seq(self.fields, multi_line=True)])

@dataclass
class VariantCase:
    name: Token
    taip: Type | None

@dataclass
class Variant(Formattable):
    name: Token
    generic_parameters: Tuple[Token, ...]
    cases: List[VariantCase]
