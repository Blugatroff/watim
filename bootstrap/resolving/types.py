from typing import List
from dataclasses import dataclass

from parsing.types import I8, I32, I64, Bool, PrimitiveType, GenericType, HoleType
from format import Formattable, FormatInstr, unnamed_record, format_seq
from lexer import Token

@dataclass
class PtrType(Formattable):
    child: 'Type'
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Ptr", [self.child])

@dataclass(frozen=True, eq=True)
class CustomTypeHandle(Formattable):
    module: int
    index: int
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("CustomTypeHandle", [self.module, self.index])

@dataclass
class CustomTypeType(Formattable):
    name: Token
    type_definition: CustomTypeHandle
    generic_arguments: List['Type']
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("CustomType", [
            self.type_definition.module,
            self.type_definition.index,
            format_seq(self.generic_arguments)])

@dataclass
class TupleType(Formattable):
    token: Token
    items: List['Type']
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("TupleType", [self.token, format_seq(self.items)])

@dataclass
class FunctionType(Formattable):
    token: Token
    parameters: List['Type']
    returns: List['Type']
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunType", [self.token, format_seq(self.parameters), format_seq(self.returns)])

type Type = PrimitiveType | PtrType | TupleType | GenericType | CustomTypeType | FunctionType | HoleType

def resolved_type_eq(a: Type, b: Type):
    if isinstance(a, Bool) and isinstance(b, Bool):
        return a == b
    if isinstance(a, I8) and isinstance(b, I8):
        return a == b
    if isinstance(a, I32) and isinstance(b, I32):
        return a == b
    if isinstance(a, I64) and isinstance(b, I64):
        return a == b
    if isinstance(a, PtrType) and isinstance(b, PtrType):
        return resolved_type_eq(a.child, b.child)
    if isinstance(a, CustomTypeType) and isinstance(b, CustomTypeType):
        module_eq = a.type_definition.module == b.type_definition.module
        index_eq  = a.type_definition.index == b.type_definition.index
        return resolved_types_eq(a.generic_arguments, b.generic_arguments) if module_eq and index_eq else False
    if isinstance(a, FunctionType) and isinstance(b, FunctionType):
        if len(a.parameters) != len(b.parameters) or len(a.returns) != len(b.returns):
            return False
        for c,d in zip(a.parameters, b.parameters):
            if not resolved_type_eq(c, d):
                return False
        for c,d in zip(a.parameters, b.parameters):
            if not resolved_type_eq(c, d):
                return False
        return True
    if isinstance(a, TupleType) and isinstance(b, TupleType):
        return resolved_types_eq(a.items, b.items)
    if isinstance(a, GenericType) and isinstance(b, GenericType):
        return a.generic_index == b.generic_index
    return False

def resolved_types_eq(a: List[Type], b: List[Type]) -> bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not resolved_type_eq(a[i], b[i]):
            return False
    return True

@dataclass
class NamedType(Formattable):
    name: Token
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("NamedType", [self.name, self.taip])

