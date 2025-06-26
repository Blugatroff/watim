from typing import Dict, Tuple, List
from dataclasses import dataclass

from format import Formattable, FormatInstr, named_record, format_dict, format_str, format_seq
from indexed_dict import IndexedDict
from lexer import Token
from resolving.top_items import Import, TypeDefinition, Function, Extern, Global, FunctionHandle, CustomTypeHandle

@dataclass
class Module(Formattable):
    path: str
    id: int
    imports: Dict[str, Tuple[Import, ...]]
    type_definitions: IndexedDict[str, TypeDefinition]
    globals: IndexedDict[str, Global]
    functions: IndexedDict[str, Function | Extern]

    def format_instrs(self) -> List[FormatInstr]:
        return named_record("Module", [
            ("imports", format_dict(self.imports, format_str, format_seq)),
            ("type-definitions", self.type_definitions.format_instrs(format_str)),
            ("globals", self.globals.format_instrs(format_str)),
            ("functions", self.functions.format_instrs(format_str))])

    def lookup_item(self, name: Token) -> FunctionHandle | CustomTypeHandle | None:
        if name.lexeme in self.functions:
            return FunctionHandle(self.id, self.functions.index_of(name.lexeme))
        if name.lexeme in self.type_definitions:
            return CustomTypeHandle(self.id, self.type_definitions.index_of(name.lexeme))
        return None

@dataclass
class ResolveException(Exception):
    path: str
    token: Token
    message: str

    def display(self) -> str:
        line = self.token.line
        column = self.token.column
        return f"{self.path}:{line}:{column} {self.message}"

