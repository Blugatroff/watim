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

    def lookup_item(self, module_id: int, name: Token) -> FunctionHandle | CustomTypeHandle | None:
        for function_index, function in enumerate(self.functions.values()):
            if function.name.lexeme == name.lexeme:
                return FunctionHandle(module_id, function_index)
        for type_index, type_definition in enumerate(self.type_definitions.values()):
            if type_definition.name.lexeme == name.lexeme:
                return CustomTypeHandle(module_id, type_index)
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

