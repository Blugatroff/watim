from typing import List, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq
from lexer import Token

@dataclass(frozen=True, eq=True)
class I8(Formattable):
    def size(self) -> int:
        return 1
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["I8"]

@dataclass(frozen=True, eq=True)
class I32(Formattable):
    def size(self) -> int:
        return 4
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["I32"]

@dataclass(frozen=True, eq=True)
class I64(Formattable):
    def size(self) -> int:
        return 8
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["I64"]

@dataclass(frozen=True, eq=True)
class Bool(Formattable):
    def size(self) -> int:
        return 4
    def can_live_in_reg(self) -> bool:
        return True
    def format_instrs(self) -> List[FormatInstr]:
        return ["Bool"]

type PrimitiveType = I8 | I32 | I64 | Bool

@dataclass(frozen=True)
class GenericType(Formattable):
    token: Token
    generic_index: int

    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GenericType", [self.token, self.generic_index])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GenericType) and other.generic_index == self.generic_index

type Type = 'PrimitiveType | PtrType | TupleType | GenericType | ForeignType | CustomTypeType | FunctionType | HoleType'

@dataclass(frozen=True, eq=True)
class PtrType(Formattable):
    child: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Ptr", [self.child])

@dataclass(frozen=True, eq=True)
class ForeignType(Formattable):
    module: Token
    name: Token
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("ForeignCustomType", [
            self.module,
            self.name,
            format_seq(self.generic_arguments)])

@dataclass(frozen=True, eq=True)
class TupleType(Formattable):
    token: Token
    items: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("TupleType", [
            self.token,
            format_seq(self.items)])

@dataclass(frozen=True, eq=True)
class CustomTypeType(Formattable):
    name: Token
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("LocalCustomType", [
            self.name,
            format_seq(self.generic_arguments, multi_line=True)])

@dataclass(frozen=True, eq=True)
class FunctionType(Formattable):
    token: Token
    parameters: Tuple[Type, ...]
    returns: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunType", [
            self.token,
            format_seq(self.parameters, multi_line=True),
            format_seq(self.returns, multi_line=True)])

@dataclass
class HoleType(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return ["Hole"]

@dataclass(frozen=True)
class NamedType(Formattable):
    name: Token
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("NamedType", [self.name, self.taip])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamedType):
            return False
        return self.name.lexeme == other.name.lexeme and self.taip == other.taip
