from typing import List, Dict, Tuple
from dataclasses import dataclass

from format import Formattable, FormatInstr, Instrs, unnamed_record, format_str, format_seq, named_record, format_optional, format_dict
from lexer import Token
from resolving.words import FunctionHandle, Scope, LocalId
from resolving.types import CustomTypeHandle, Type, NamedType

type TopItem = Import | Struct | Variant | Extern | Function
type TypeDefinition = Struct | Variant

@dataclass
class StructImport(Formattable):
    name: Token
    handle: CustomTypeHandle
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("StructImport", [self.name, self.handle])

@dataclass
class VariantImport(Formattable):
    name: Token
    handle: CustomTypeHandle
    constructors: Tuple[int, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("VariantImport", [
            self.name,
            self.handle,
            format_seq([Instrs([constructor]) for constructor in self.constructors])])

@dataclass
class FunctionImport(Formattable):
    name: Token
    handle: FunctionHandle
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("FunctionImport", [self.name, self.handle])

type ImportItem = VariantImport | FunctionImport | StructImport

@dataclass
class Import(Formattable):
    token: Token
    file_path: str
    qualifier: Token
    module: int
    items: Tuple[ImportItem, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Import", [
            self.token,
            self.module,
            format_str(self.file_path),
            self.qualifier,
            format_seq(self.items, multi_line=True)])

@dataclass
class Struct(Formattable):
    name: Token
    generic_parameters: Tuple[Token, ...]
    fields: Tuple[NamedType, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Struct", [
            ("name", self.name),
            ("generic-parameters", format_seq(self.generic_parameters)),
            ("fields", format_seq(self.fields, multi_line=True))])

@dataclass
class VariantCase(Formattable):
    name: Token
    taip: Type | None
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("VariantCase", [self.name, format_optional(self.taip)])

@dataclass
class Variant(Formattable):
    name: Token
    generic_parameters: Tuple[Token, ...]
    cases: Tuple[VariantCase, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Variant", [
            ("name", self.name),
            ("generic-parameters", format_seq(self.generic_parameters)),
            ("cases", format_seq(self.cases, multi_line=True))])

@dataclass
class FunctionSignature(Formattable):
    generic_parameters: Tuple[Token, ...]
    parameters: Tuple[NamedType, ...]
    returns: Tuple[Type, ...]
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Signature", [
            ("generic-parameters", format_seq(self.generic_parameters)),
            ("parameters", format_seq(self.parameters)),
            ("returns", format_seq(self.returns))])

@dataclass
class Global(Formattable):
    name: Token
    taip: Type
    was_reffed: bool = False
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Global", [self.name, self.taip, self.was_reffed])


@dataclass(frozen=True, eq=True)
class LocalName(Formattable):
    name: Token | str
    def format_instrs(self) -> List[FormatInstr]:
        return [self.name]

    def get(self) -> str:
        return self.name if isinstance(self.name, str) else self.name.lexeme

@dataclass
class Local(Formattable):
    name: LocalName
    parameter: Type | None # if this local is a parameter, then this will be non-None

    @staticmethod
    def make(taip: NamedType) -> 'Local':
        return Local(LocalName(taip.name), None)

    @staticmethod
    def make_parameter(taip: NamedType) -> 'Local':
        return Local(LocalName(taip.name), taip.taip)

    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Local", [self.name, format_optional(self.parameter)])

@dataclass
class Function(Formattable):
    name: Token
    export_name: Token | None
    signature: FunctionSignature
    body: Scope
    locals: Dict[LocalId, Local]
    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Function", [
            ("name", self.name),
            ("export", format_optional(self.export_name)),
            ("signature", self.signature),
            ("locals", format_dict(self.locals)),
            ("body", self.body)])

@dataclass
class Extern(Formattable):
    name: Token
    extern_module: str
    extern_name: str
    signature: FunctionSignature
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Extern", [
            self.name,
            self.extern_module,
            self.extern_name,
            self.signature])
