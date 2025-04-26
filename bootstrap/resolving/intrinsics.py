from typing import List
from dataclasses import dataclass
from enum import Enum

from format import Formattable, FormatInstr, unnamed_record
from resolving.types import I8, I32, I64, PrimitiveType, PtrType, Type
from lexer import Token

class IntrinsicType(str, Enum):
    ADD = "ADD"
    STORE = "STORE"
    DROP = "DROP"
    SUB = "SUB"
    EQ = "EQ"
    NOT_EQ = "NOT_EQ"
    MOD = "MOD"
    DIV = "DIV"
    AND = "AND"
    NOT = "NOT"
    OR = "OR"
    LESS = "LESS"
    GREATER = "GREATER"
    LESS_EQ = "LESS_EQ"
    GREATER_EQ = "GREATER_EQ"
    MUL = "MUL"
    SHL = "SHL"
    SHR = "SHR"
    ROTL = "ROTL"
    ROTR = "ROTR"
    MEM_GROW = "MEM_GROW"
    MEM_COPY = "MEM_COPY"
    MEM_FILL = "MEM_FILL"
    FLIP = "FLIP"
    UNINIT = "UNINIT"
    SET_STACK_SIZE = "SET_STACK_SIZE"

INTRINSICS: dict[str, IntrinsicType] = {
        "drop": IntrinsicType.DROP,
        "flip": IntrinsicType.FLIP,
        "+": IntrinsicType.ADD,
        "lt": IntrinsicType.LESS,
        "gt": IntrinsicType.GREATER,
        "=": IntrinsicType.EQ,
        "le": IntrinsicType.LESS_EQ,
        "ge": IntrinsicType.GREATER_EQ,
        "not": IntrinsicType.NOT,
        "mem-grow": IntrinsicType.MEM_GROW,
        "-": IntrinsicType.SUB,
        "and": IntrinsicType.AND,
        "%": IntrinsicType.MOD,
        "/": IntrinsicType.DIV,
        "/=": IntrinsicType.NOT_EQ,
        "*": IntrinsicType.MUL,
        "mem-copy": IntrinsicType.MEM_COPY,
        "mem-fill": IntrinsicType.MEM_FILL,
        "shl": IntrinsicType.SHL,
        "shr": IntrinsicType.SHR,
        "rotl": IntrinsicType.ROTL,
        "rotr": IntrinsicType.ROTR,
        "or": IntrinsicType.OR,
        "store": IntrinsicType.STORE,
        "uninit": IntrinsicType.UNINIT,
        "set-stack-size": IntrinsicType.SET_STACK_SIZE,
}
INTRINSIC_TO_LEXEME: dict[IntrinsicType, str] = {v: k for k, v in INTRINSICS.items()}

@dataclass
class IntrinsicAdd(Formattable):
    token: Token
    taip: PtrType | I8 | I32 | I64
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Intrinsic", [
            self.token, unnamed_record("Add", [self.taip])])

@dataclass
class IntrinsicSub(Formattable):
    token: Token
    taip: PtrType | I8 | I32 | I64

@dataclass
class IntrinsicDrop(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Drop", [self.token])

@dataclass
class IntrinsicMod(Formattable):
    token: Token
    taip: I32 | I64
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Intrinsic", [
            self.token, unnamed_record("Mod", [self.taip])])

@dataclass
class IntrinsicMul(Formattable):
    token: Token
    taip: I32 | I64

@dataclass
class IntrinsicDiv(Formattable):
    token: Token
    taip: I32 | I64

@dataclass
class IntrinsicAnd(Formattable):
    token: Token
    taip: PrimitiveType

@dataclass
class IntrinsicOr(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicShl(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicShr(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicRotr(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicRotl(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicGreater(Formattable):
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicLess(Formattable):
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicGreaterEq(Formattable):
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicLessEq(Formattable):
    token: Token
    taip: I8 | I32 | I64

@dataclass
class IntrinsicMemCopy(Formattable):
    token: Token

@dataclass
class IntrinsicMemFill(Formattable):
    token: Token

@dataclass
class IntrinsicEqual(Formattable):
    token: Token
    taip: Type
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Intrinsic", [
            self.token,
            unnamed_record("Eq", [self.taip])])

@dataclass
class IntrinsicNotEqual(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicFlip(Formattable):
    token: Token
    lower: Type
    upper: Type

@dataclass
class IntrinsicMemGrow(Formattable):
    token: Token

@dataclass
class IntrinsicSetStackSize(Formattable):
    token: Token

@dataclass
class IntrinsicStore(Formattable):
    token: Token
    taip: Type

@dataclass
class IntrinsicNot(Formattable):
    token: Token
    taip: PrimitiveType

@dataclass
class IntrinsicUninit(Formattable):
    token: Token
    taip: Type

IntrinsicWord = (
      IntrinsicAdd
    | IntrinsicSub
    | IntrinsicDrop
    | IntrinsicMod
    | IntrinsicMul
    | IntrinsicDiv
    | IntrinsicAnd
    | IntrinsicOr
    | IntrinsicShl
    | IntrinsicShr
    | IntrinsicRotl
    | IntrinsicRotr
    | IntrinsicGreater
    | IntrinsicLess
    | IntrinsicGreaterEq
    | IntrinsicLessEq
    | IntrinsicMemCopy
    | IntrinsicMemFill
    | IntrinsicEqual
    | IntrinsicNotEqual
    | IntrinsicFlip
    | IntrinsicMemGrow
    | IntrinsicStore
    | IntrinsicNot
    | IntrinsicUninit
    | IntrinsicSetStackSize
)
