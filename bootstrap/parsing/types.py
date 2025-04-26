from typing import List
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq
from lexer import Token

@dataclass
class I8(Formattable):
    def size(self) -> int:
        return 1
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["I8"]

@dataclass
class I32(Formattable):
    def size(self) -> int:
        return 4
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["I32"]

@dataclass
class I64(Formattable):
    def size(self) -> int:
        return 8
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["I64"]

@dataclass
class Bool(Formattable):
    def size(self) -> int:
        return 4
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["Bool"]

type PrimitiveType = I8 | I32 | I64 | Bool

@dataclass
class GenericType(Formattable):
    token: Token
    generic_index: int

    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GenericType", [self.token, self.generic_index])

type Type = 'PrimitiveType | PtrType | TupleType | GenericType | ForeignType | CustomTypeType | FunctionType | HoleType'

@dataclass
class PtrType(Formattable):
    child: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Ptr", [self.child])

@dataclass
class ForeignType(Formattable):
    module: Token
    name: Token
    generic_arguments: List[Type]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("ForeignCustomType", [
            self.module,
            self.name,
            format_seq(self.generic_arguments)])

@dataclass
class TupleType(Formattable):
    token: Token
    items: List[Type]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("TupleType", [
            self.token,
            format_seq(self.items)])

@dataclass
class CustomTypeType(Formattable):
    name: Token
    generic_arguments: List[Type]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("LocalCustomType", [
            self.name,
            format_seq(self.generic_arguments)])

@dataclass
class FunctionType(Formattable):
    token: Token
    parameters: List[Type]
    returns: List[Type]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunType", [
            self.token,
            format_seq(self.parameters),
            format_seq(self.returns)])

@dataclass
class HoleType(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return ["Hole"]

@dataclass
class NamedType(Formattable):
    name: Token
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("NamedType", [self.name, self.taip])
