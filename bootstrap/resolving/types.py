from typing import List, Tuple
from dataclasses import dataclass, field

from parsing.types import PrimitiveType, GenericType, HoleType
from format import Formattable, FormatInstr, unnamed_record, format_seq
from lexer import Token

type Type = PrimitiveType | PtrType | TupleType | GenericType | CustomTypeType | FunctionType | HoleType

def with_generics(taip: Type, generics: Tuple[Type, ...]) -> Type:
    match taip:
        case PtrType(child):
            return PtrType(with_generics(child, generics))
        case CustomTypeType(name, type_definition, generic_arguments):
            return CustomTypeType(
                name,
                type_definition,
                tuple(with_generics(arg, generics) for arg in generic_arguments))
        case TupleType(token, items):
            return TupleType(
                token,
                tuple(with_generics(item, generics) for item in items))
        case FunctionType(token, parameters, returns):
            return FunctionType(
                token,
                tuple(with_generics(param, generics) for param in parameters),
                tuple(with_generics(ret, generics)   for ret   in returns))
        case GenericType(token, index):
            return generics[index]
        case other:
            return other

@dataclass(frozen=True, eq=True)
class PtrType(Formattable):
    child: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Ptr", [self.child])

@dataclass(frozen=True, eq=True)
class CustomTypeHandle(Formattable):
    module: int
    index: int
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("CustomTypeHandle", [self.module, self.index])

@dataclass(eq=True, frozen=True)
class CustomTypeType(Formattable):
    name: Token = field(compare=False)
    type_definition: CustomTypeHandle
    generic_arguments: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("CustomType", [
            self.type_definition.module,
            self.type_definition.index,
            format_seq(self.generic_arguments)])

@dataclass(frozen=True, eq=True)
class TupleType(Formattable):
    token: Token = field(compare=False)
    items: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("TupleType", [self.token, format_seq(self.items)])

@dataclass(frozen=True, eq=True)
class FunctionType(Formattable):
    token: Token = field(compare=False)
    parameters: Tuple[Type, ...]
    returns: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunType", [self.token, format_seq(self.parameters), format_seq(self.returns)])

@dataclass(frozen=True)
class NamedType(Formattable):
    name: Token
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("NamedType", [self.name, self.taip])
    def __eq__(self, other: object) -> bool:
        return isinstance(other, NamedType) and other.name.lexeme == self.name.lexeme and other.taip == self.taip

