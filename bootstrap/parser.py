from typing import List, Tuple, Optional, NoReturn, TypeGuard, Callable
from dataclasses import dataclass

from format import Formattable, FormatInstr, unnamed_record, format_seq, format_optional
from lexer import Token, TokenType, TokenLocation

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
class NumberWord(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Number", [self.token])

@dataclass
class BreakWord(Formattable):
    token: Token
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Break", [self.token])

type Word = 'NumberWord | StringWord | CallWord | GetWord | RefWord | SetWord | StoreWord | InitWord | CallWord | ForeignCallWord | FunRefWord | IfWord | LoadWord | LoopWord | BlockWord | BreakWord | CastWord | SizeofWord | GetFieldWord | IndirectCallWord | StructWord | UnnamedStructWord | MatchWord | VariantWord | TupleUnpackWord | TupleMakeWord | StackAnnotation | InlineRefWord'

type TypeDefinition = 'Struct | Variant'

type TopItem = 'Import | TypeDefinition | Global | Function | Extern'

@dataclass
class ParserException(Exception):
    location: TokenLocation | Tuple[str, str] | None
    message: str

    def display(self) -> str:
        if self.location is None:
            return self.message
        if isinstance(self.location, TokenLocation):
            file_path = self.location.file_path
            line = self.location.line
            column = self.location.column
        else:
            file_path = self.location[0]
            lines = self.location[1].splitlines()
            line = len(lines) + 1
            column = len(lines[-1]) + 1 if len(lines) != 0 else 1
        return f"{file_path}:{line}:{column} {self.message}"


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

# ========================================
# Datatypes for parsed words
# ========================================
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
    fields: List[Token]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("GetLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class RefWord(Formattable):
    token: Token
    ident: Token
    fields: List[Token]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("RefLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class SetWord(Formattable):
    token: Token
    ident: Token
    fields: List[Token]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("SetLocal", [self.token, self.ident, format_seq(self.fields)])

@dataclass
class InlineRefWord(Formattable):
    token: Token

@dataclass
class StoreWord(Formattable):
    token: Token
    ident: Token
    fields: List[Token]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("Store", [self.token, self.ident, format_seq(self.fields)])

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
    generic_arguments: List[Type]

@dataclass
class CallWord(Formattable):
    ident: Token
    generic_arguments: List[Type]
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("LocalCall", [
            self.ident,
            format_seq(self.generic_arguments, multi_line=True)])

@dataclass
class FunRefWord(Formattable):
    call: CallWord | ForeignCallWord

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
    words: List[Word]
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
    fields: List[Token]

@dataclass
class IndirectCallWord(Formattable):
    token: Token

@dataclass
class StructWord(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType
    words: List[Word]

@dataclass
class UnnamedStructWord(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType

@dataclass
class VariantWord(Formattable):
    token: Token
    taip: CustomTypeType | ForeignType
    case: Token

@dataclass
class MatchCase(Formattable):
    case: Token
    name: Token
    words: List[Word]

@dataclass
class MatchWord(Formattable):
    token: Token
    cases: List[MatchCase]
    default: MatchCase | None

@dataclass
class TupleUnpackWord(Formattable):
    token: Token

@dataclass
class TupleMakeWord(Formattable):
    token: Token
    item_count: Token

@dataclass
class StackAnnotation(Formattable):
    token: Token
    types: List['Type']
    def format_instrs(self) -> List[FormatInstr]:
        return unnamed_record("StackAnnotation", [
            self.token,
            format_seq(self.types)])

# ========================================
# Datatypes for parsed TopItems
# ========================================
@dataclass
class Import(Formattable):
    token: Token
    file_path: Token
    qualifier: Token
    items: List[Token]

@dataclass
class FunctionSignature(Formattable):
    export_name: Optional[Token]
    name: Token
    generic_parameters: List[Token]
    parameters: List[NamedType]
    returns: List[Type]
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
    body: List[Word]
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
    fields: List[NamedType]
    generic_parameters: List[Token]
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
    generic_parameters: List[Token]
    cases: List[VariantCase]


# ========================================
# A parsed module, output of the parser
# ========================================
@dataclass
class Module(Formattable):
    path: str
    file: str
    top_items: List[TopItem]
    imports: List[Import]
    type_definitions: List[TypeDefinition]
    globals: List[Global]
    functions: List[Function | Extern]

    def format_instrs(self) -> List[FormatInstr]:
        return format_seq(self.top_items, multi_line=True)

@dataclass
class Parser:
    file_path: str
    file: str
    tokens: List[Token]
    cursor: int = 0

    # ========================================
    # Utility functions for the parser
    # ========================================
    def peek(self, skip_ws: bool = False) -> Token | None:
        i = self.cursor
        while True:
            if i >= len(self.tokens):
                return None
            token = self.tokens[i]
            if skip_ws and token.ty == TokenType.SPACE:
                i += 1
                continue
            return token

    def advance(self, skip_ws: bool = False):
        while True:
            if self.cursor >= len(self.tokens):
                return None
            token = self.tokens[self.cursor]
            self.cursor += 1
            if skip_ws and token.ty == TokenType.SPACE:
                continue
            return token

    def retreat(self):
        assert(self.cursor > 0)
        self.cursor -= 1

    def abort(self, message: str) -> NoReturn:
        if self.cursor < len(self.tokens):
            current = self.tokens[self.cursor]
            raise ParserException(TokenLocation(self.file_path, current.line, current.column), message)
        else:
            raise ParserException((self.file_path, self.file), message)


    # ========================================
    # Parsing routines
    # ========================================
    def parse(self) -> Module:
        top_items: List[Import | TypeDefinition | Global | Function | Extern] = []
        while len(self.tokens) != 0:
            token = self.advance(skip_ws=True)
            if token is None:
                break
            if token.ty == TokenType.IMPORT:
                file_path = self.advance(skip_ws=True)
                if file_path is None or file_path.ty != TokenType.STRING:
                    self.abort("Expected file path")

                ass = self.advance(skip_ws=True)
                if ass is None or ass.ty != TokenType.AS:
                    self.abort("Expected `as`")

                module_qualifier = self.advance(skip_ws=True)
                if module_qualifier is None or module_qualifier.ty != TokenType.IDENT:
                    self.abort("Expected an identifier as module qualifier")

                paren = self.peek(skip_ws=True)
                items = []
                if paren is not None and paren.ty == TokenType.LEFT_PAREN:
                    self.advance(skip_ws=True) # skip LEFT_PAREN
                    while True:
                        item = self.advance(skip_ws=True)
                        if item is None:
                            self.abort("expected a function or type to import")
                        if item.ty == TokenType.RIGHT_PAREN:
                            break
                        items.append(item)
                        comma = self.advance(skip_ws=True)
                        if comma is None or comma.ty == TokenType.RIGHT_PAREN:
                            break
                        if comma.ty != TokenType.COMMA:
                            self.abort("expected `)`")
                top_items.append(Import(token, file_path, module_qualifier, items))
                continue

            if token.ty == TokenType.FN:
                top_items.append(self.parse_function(token))
                continue

            if token.ty == TokenType.EXTERN:
                module = self.advance(skip_ws=True)
                if module is None or module.ty != TokenType.STRING:
                    self.abort("Expected string as extern function module name")
                name = self.advance(skip_ws=True)
                if name is None or name.ty != TokenType.STRING:
                    self.abort("Expected string as extern function name")
                fn = self.advance(skip_ws=True)
                if fn is None or fn.ty != TokenType.FN:
                    self.abort("Expected `fn`")
                signature = self.parse_function_signature()
                top_items.append(Extern(token, module, name, signature))
                continue

            if token.ty == TokenType.STRUCT:
                name = self.advance(skip_ws=True)
                if name is None or name.ty != TokenType.IDENT:
                    self.abort("Expected identifier as struct name")
                generic_parameters = self.parse_generic_parameters()
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.LEFT_BRACE:
                    self.abort("Expected `{`")
                fields = []
                while True:
                    next = self.advance(skip_ws=True)
                    if next is not None and next.ty == TokenType.RIGHT_BRACE:
                        break
                    field_name = next
                    if field_name is None or field_name.ty != TokenType.IDENT:
                        self.abort("Expected identifier as struct field name")
                    colon = self.advance(skip_ws=True)
                    if colon is None or colon.ty != TokenType.COLON:
                        self.abort("Expected `:` after field name")
                    taip = self.parse_type(generic_parameters)
                    fields.append(NamedType(field_name, taip))
                top_items.append(Struct(token, name, fields, generic_parameters))
                continue

            if token.ty == TokenType.VARIANT:
                name = self.advance(skip_ws=True)
                if name is None:
                    self.abort("Expected an identifier")
                generic_parameters = self.parse_generic_parameters()
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.LEFT_BRACE:
                    self.abort("Expected `{`")
                cases: List[VariantCase] = []
                while True:
                    next = self.peek(skip_ws=True)
                    if next is None or next.ty == TokenType.RIGHT_BRACE:
                        self.advance(skip_ws=True)
                        break
                    case = self.advance(skip_ws=True)
                    if case is None or case.ty != TokenType.CASE:
                        self.abort("expected `case`")
                    ident = self.advance(skip_ws=True)
                    if ident is None or ident.ty != TokenType.IDENT:
                        self.abort("expected an identifier")
                    arrow = self.peek(skip_ws=True)
                    if arrow is None or arrow.ty != TokenType.ARROW:
                        cases.append(VariantCase(ident, None))
                        continue
                    self.advance(skip_ws=True)
                    cases.append(VariantCase(ident, self.parse_type(generic_parameters)))
                top_items.append(Variant(name, generic_parameters, cases))
                continue

            if token.ty == TokenType.GLOBAL:
                name = self.advance(skip_ws=True)
                if name is None:
                    self.abort("expected an identifier")
                colon = self.advance(skip_ws=True)
                if colon is None or colon.ty != TokenType.COLON:
                    self.abort("Expected `:`")
                taip = self.parse_type([])
                top_items.append(Global(token, name, taip))
                continue

            self.abort("Expected function import or struct definition")
        def is_import(obj: object) -> TypeGuard[Import]:
            return isinstance(obj, Import)
        def is_type_definition(obj: object) -> TypeGuard[TypeDefinition]:
            return isinstance(obj, Struct) or isinstance(obj, Variant)
        def is_global(obj: object) -> TypeGuard[Global]:
           return isinstance(obj, Global)
        imports: List[Import] = list(filter(is_import, top_items))
        type_definitions: List[TypeDefinition] = list(filter(is_type_definition, top_items))
        globals: List[Global] = list(filter(is_global, top_items))
        functions: List[Function | Extern] = [f for f in top_items if isinstance(f, Function) or isinstance(f, Extern)]
        return Module(self.file_path, self.file, top_items, imports, type_definitions, globals, functions)

    def parse_function(self, start: Token) -> Function:
        signature = self.parse_function_signature()
        token = self.advance(skip_ws=True)
        if token is None or token.ty != TokenType.LEFT_BRACE:
            self.abort("Expected `{`")
        body = self.parse_words(signature.generic_parameters)
        return Function(start, signature, body.words)

    def parse_words(self, generic_parameters: List[Token]) -> Words:
        words: List[Word] = []
        while True:
            token = self.peek(skip_ws=True)
            if token is not None and token.ty == TokenType.RIGHT_BRACE:
                self.advance(skip_ws=True)
                return Words(words, token)
            words.append(self.parse_word(generic_parameters))

    def parse_word(self, generic_parameters: List[Token]) -> Word:
        token = self.advance(skip_ws=True)
        if token is None:
            self.abort("Expected a word")
        if token.ty == TokenType.NUMBER:
            return NumberWord(token)
        if token.ty == TokenType.STRING:
            string = bytearray()
            i = 1
            while i < len(token.lexeme) - 1:
                if token.lexeme[i] != "\\":
                    string.extend(token.lexeme[i].encode('utf-8'))
                    i += 1
                    continue
                if token.lexeme[i + 1] == "\"":
                    string.extend(b"\"")
                elif token.lexeme[i + 1] == "n":
                    string.extend(b"\n")
                elif token.lexeme[i + 1] == "t":
                    string.extend(b"\t")
                elif token.lexeme[i + 1] == "r":
                    string.extend(b"\r")
                elif token.lexeme[i + 1] == "\\":
                    string.extend(b"\\")
                else:
                    assert(False)
                i += 2
            return StringWord(token, string)
        if token.ty == TokenType.AMPERSAND:
            next = self.peek(skip_ws=False)
            if next is not None and next.ty == TokenType.SPACE:
                return InlineRefWord(token)
        if token.ty in [TokenType.DOLLAR, TokenType.AMPERSAND, TokenType.HASH, TokenType.DOUBLE_ARROW]:
            indicator_token = token
            name = self.advance(skip_ws=True)
            if name is None or name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as variable name")
            token = self.peek(skip_ws=True)
            def construct(name: Token, fields: List[Token]) -> Word:
                match indicator_token.ty:
                    case TokenType.DOLLAR:
                        return GetWord(indicator_token, name, fields)
                    case TokenType.AMPERSAND:
                        return RefWord(indicator_token, name, fields)
                    case TokenType.HASH:
                        return SetWord(indicator_token, name, fields)
                    case TokenType.DOUBLE_ARROW:
                        return StoreWord(indicator_token, name, fields)
                    case _:
                        assert(False)
            if token is None or token.ty == TokenType.SPACE:
                return construct(name, [])
            fields = self.parse_field_accesses()
            return construct(name, fields)
        if token.ty == TokenType.AT:
            ident = self.advance(skip_ws=False)
            if ident is None or ident.ty != TokenType.IDENT:
                self.abort("Expected an identifier as variable name")
            return InitWord(token, ident)
        if token.ty == TokenType.IDENT:
            return self.parse_call_word(generic_parameters, token)
        if token.ty == TokenType.BACKSLASH:
            token = self.advance(skip_ws=True) # skip `\`
            assert(token is not None)
            return FunRefWord(self.parse_call_word(generic_parameters, token))
        if token.ty == TokenType.IF:
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            if_words = self.parse_words(generic_parameters)
            next = self.peek(skip_ws=True)
            if next is None or next.ty != TokenType.ELSE:
                return IfWord(token, if_words, None)
            self.advance(skip_ws=True) # skip `else`
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            else_words = self.parse_words(generic_parameters)
            return IfWord(token, if_words, else_words)
        if token.ty == TokenType.TILDE:
            return LoadWord(token)
        if token.ty == TokenType.LOOP or token.ty == TokenType.BLOCK:
            brace = self.advance(skip_ws=True)
            if brace is None:
                self.abort("Expected `{`")
            if brace.ty == TokenType.LEFT_BRACE:
                parameters = None
                returns = None
            else:
                parameters = []
                while True:
                    next = self.peek(skip_ws=True)
                    if next is None or next.ty == TokenType.ARROW:
                       self.advance(skip_ws=True) # skip `->`
                       break
                    parameters.append(self.parse_type(generic_parameters))
                    comma = self.peek(skip_ws=True)
                    if comma is None or comma.ty == TokenType.ARROW:
                        self.advance(skip_ws=True) # skip `->`
                        break
                    if comma.ty != TokenType.COMMA:
                        self.abort("Expected `,`")
                    self.advance(skip_ws=True)
                returns = []
                while True:
                    next = self.peek(skip_ws=True)
                    if next is None or next.ty == TokenType.RIGHT_PAREN:
                        self.advance(skip_ws=True) # skip `)`
                        break
                    returns.append(self.parse_type(generic_parameters))
                    comma = self.advance(skip_ws=True)
                    if comma is None or comma.ty == TokenType.RIGHT_PAREN:
                        break
                    if comma.ty != TokenType.COMMA:
                        self.abort("Expected `,`")
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.LEFT_BRACE:
                    self.abort("Expected `{`")
            annotation = None if parameters is None and returns is None else BlockAnnotation(parameters or [], returns or [])
            words = self.parse_words(generic_parameters)
            if token.ty == TokenType.LOOP:
                return LoopWord(token, words, annotation)
            if token.ty == TokenType.BLOCK:
                return BlockWord(token, words, annotation)
        if token.ty == TokenType.BREAK:
            return BreakWord(token)
        if token.ty == TokenType.BANG:
            return CastWord(token, self.parse_type(generic_parameters))
        if token.ty == TokenType.SIZEOF:
            paren = self.advance(skip_ws=True)
            if paren is None or paren.ty != TokenType.LEFT_PAREN:
                self.abort("Expected `(`")
            taip = self.parse_type(generic_parameters)
            paren = self.advance(skip_ws=True)
            if paren is None or paren.ty != TokenType.RIGHT_PAREN:
                self.abort("Expected `)`")
            return SizeofWord(token, taip)
        if token.ty == TokenType.DOT:
            self.retreat()
            return GetFieldWord(token, self.parse_field_accesses())
        if token.ty == TokenType.ARROW:
            return IndirectCallWord(token)
        if token.ty == TokenType.MAKE:
            struct_name_token = self.advance(skip_ws=True)
            taip = self.parse_struct_type(struct_name_token, generic_parameters)
            dot = self.peek(skip_ws=False)
            if dot is not None and dot.ty == TokenType.DOT:
                self.advance(skip_ws=False)
                case_name = self.advance(skip_ws=False)
                if case_name is None or case_name.ty != TokenType.IDENT:
                    self.abort("expected an identifier")
                return VariantWord(token, taip, case_name)
            brace = self.peek(skip_ws=True)
            if brace is not None and brace.ty == TokenType.LEFT_BRACE:
                brace = self.advance(skip_ws=True)
                words = self.parse_words(generic_parameters)
                return StructWord(token, taip, words.words)
            return UnnamedStructWord(token, taip)
        if token.ty == TokenType.MATCH:
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            cases: List[MatchCase] = []
            while True:
                next = self.peek(skip_ws=True)
                if next is None or next.ty == TokenType.RIGHT_BRACE:
                    self.advance(skip_ws=True)
                    return MatchWord(token, cases, None)
                case = self.advance(skip_ws=True)
                if case is None or case.ty != TokenType.CASE:
                    self.abort("expected `case`")
                case_name = self.advance(skip_ws=True)
                if case_name is None or (case_name.ty != TokenType.IDENT and case_name.ty != TokenType.UNDERSCORE):
                    self.abort("Expected an identifier")
                arrow = self.advance(skip_ws=True)
                if arrow is None or arrow.ty != TokenType.ARROW:
                    self.abort("Expected `->`")
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.LEFT_BRACE:
                    self.abort("Expected `{`")
                words = self.parse_words(generic_parameters)
                if case_name.ty == TokenType.UNDERSCORE:
                    brace = self.advance(skip_ws=True)
                    if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                        self.abort("Expected `}`")
                    return MatchWord(token, cases, MatchCase(next, case_name, words.words))
                cases.append(MatchCase(next, case_name, words.words))
        if token.ty == TokenType.LEFT_BRACKET:
            comma = self.advance(skip_ws=True)
            if comma is None or comma.ty != TokenType.COMMA:
                self.abort("Expected `,`")
            number_or_close = self.advance(skip_ws=True)
            if number_or_close is None or (number_or_close.ty != TokenType.NUMBER and number_or_close.ty != TokenType.RIGHT_BRACKET):
                self.abort("Expected `,` or `]`")
            if number_or_close.ty == TokenType.RIGHT_BRACKET:
                return TupleUnpackWord(token)
            close = self.advance(skip_ws=True)
            if close is None or close.ty != TokenType.RIGHT_BRACKET:
                self.abort("Expected `]`")
            return TupleMakeWord(token, number_or_close)
        if token.ty == TokenType.COLON:
            next = self.advance(skip_ws=True)
            if next is None or next.ty != TokenType.LEFT_PAREN:
                self.abort("Expected `(`")
            types: List[Type] = []
            while True:
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.RIGHT_PAREN:
                    self.advance(skip_ws=True)
                    return StackAnnotation(token, [])
                types.append(self.parse_type(generic_parameters))
                next = self.advance(skip_ws=True)
                if next is None:
                    self.abort("Expected `,` or `)`")
                if next.ty == TokenType.RIGHT_PAREN:
                    return StackAnnotation(token, types)
                if next.ty != TokenType.COMMA:
                    self.abort("Expected `,` or `)`")
        self.abort("Expected word")

    def parse_call_word(self, generic_parameters: List[Token], token: Token) -> CallWord | ForeignCallWord:
        next = self.peek(skip_ws=False)
        if next is not None and next.ty == TokenType.COLON:
            module = token
            self.advance(skip_ws=False) # skip the `:`
            name = self.advance(skip_ws=False)
            if name is None or name.ty != TokenType.IDENT:
                self.abort("Expected an identifier")
            next = self.peek()
            generic_arguments = self.parse_generic_arguments(generic_parameters) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []
            return ForeignCallWord(module, name, generic_arguments)
        name = token
        generic_arguments = self.parse_generic_arguments(generic_parameters) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []
        return CallWord(name, generic_arguments)

    def parse_field_accesses(self) -> List[Token]:
        fields = []
        while True:
            token = self.peek(skip_ws=False)
            if token is None or token.ty != TokenType.DOT:
                break
            self.advance(skip_ws=False) # skip the `.`
            token = self.advance(skip_ws=False)
            if token is None or token.ty != TokenType.IDENT:
                self.abort("Expected an identifier as field name")
            fields.append(token)
        return fields

    def parse_function_signature(self) -> FunctionSignature:
        function_ident = self.advance(skip_ws=True)
        if function_ident is None or function_ident.ty != TokenType.IDENT:
            self.abort("Expected identifier as function name")

        token = self.peek(skip_ws=True)
        if token is None:
            self.abort("Expected `<` or `(`")
        if token.ty == TokenType.LEFT_TRIANGLE:
            generic_parameters = self.parse_generic_parameters()
        else:
            generic_parameters = []

        token = self.advance(skip_ws=True)
        if token is None or token.ty not in [TokenType.LEFT_PAREN, TokenType.STRING]:
            self.abort("Expected either `(` or a string as name of an exported function")

        if token.ty == TokenType.STRING:
            function_export_name = token
            token = self.advance(skip_ws=True)
            if token is None or token.ty != TokenType.LEFT_PAREN:
                self.abort("Expected `(`)")
        else:
            function_export_name = None

        parameters = []
        while True:
            token = self.advance(skip_ws=True)
            if token is not None and token.ty == TokenType.RIGHT_PAREN:
                break
            if token is None or token.ty != TokenType.IDENT:
                self.abort("Expected `)` or an identifier as a function parameter name")
            parameter_name = token
            token = self.advance(skip_ws=True)
            if token is None or token.ty != TokenType.COLON:
                self.abort("Expected `:` after function parameter name")

            parameter_type = self.parse_type(generic_parameters)
            parameters.append(NamedType(parameter_name, parameter_type))
            token = self.advance(skip_ws=True)
            if token is not None and token.ty == TokenType.RIGHT_PAREN:
                break
            if token is None or token.ty != TokenType.COMMA:
                self.abort("Expected `,` after function parameter")

        returns = []
        token = self.peek(skip_ws=True)
        if token is not None and token.ty == TokenType.ARROW:
            self.advance(skip_ws=True) # skip the `->`
            while True:
                taip = self.parse_type(generic_parameters)
                returns.append(taip)
                token = self.peek(skip_ws=True)
                if token is None or token.ty != TokenType.COMMA:
                    break
                self.advance(skip_ws=True) # skip the `,`

        return FunctionSignature(function_export_name, function_ident, generic_parameters, parameters, returns)

    def parse_triangle_listed[T](self, elem: Callable[['Parser'], T]) -> List[T]:
        token = self.advance(skip_ws=True)
        if token is None or token.ty != TokenType.LEFT_TRIANGLE:
            self.abort("Expected `<`")
        items = []
        while True:
            token = self.peek(skip_ws=True)
            if token is None:
                self.abort("Expected `>` or an identifier")
            if token.ty == TokenType.RIGHT_TRIANGLE:
                self.advance(skip_ws=True) # skip `>`
                break
            items.append(elem(self))
            token = self.advance(skip_ws=True)
            if token is None or token.ty == TokenType.RIGHT_TRIANGLE:
                break
            if token.ty != TokenType.COMMA:
                self.abort("Expected `,`")
        return items

    def parse_generic_arguments(self, generic_parameters: List[Token]) -> List[Type]:
        next = self.peek(skip_ws=False)
        return self.parse_triangle_listed(lambda self: self.parse_type(generic_parameters)) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []

    def parse_generic_parameters(self) -> List[Token]:
        def parse_ident(self):
            token = self.advance(skip_ws=True)
            if token is None or token.ty != TokenType.IDENT:
                self.abort("Expected an identifier as generic paramter")
            return token
        next = self.peek(skip_ws=False)
        return self.parse_triangle_listed(parse_ident) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []

    def parse_struct_type(self, token: Token | None, generic_parameters: List[Token]) -> CustomTypeType | ForeignType:
        if token is None or token.ty != TokenType.IDENT:
            self.abort("Expected an identifer as struct name")
        next = self.peek(skip_ws=True)
        if next is not None and next.ty == TokenType.COLON:
            self.advance(skip_ws=True) # skip the `:`
            module = token
            struct_name = self.advance(skip_ws=True)
            if struct_name is None or struct_name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as struct name")
            return ForeignType(module, struct_name, self.parse_generic_arguments(generic_parameters))
        else:
            struct_name = token
            if struct_name is None or struct_name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as struct name")
            return CustomTypeType(struct_name, self.parse_generic_arguments(generic_parameters))

    def parse_type(self, generic_parameters: List[Token]) -> Type:
        token = self.advance(skip_ws=True)
        if token is None:
            self.abort("Expected a type")
        if token.ty == TokenType.I8:
            return I8()
        if token.ty == TokenType.I32:
            return I32()
        if token.ty == TokenType.I64:
            return I64()
        if token.ty == TokenType.Bool:
            return Bool()
        if token.ty == TokenType.DOT:
            return PtrType(self.parse_type(generic_parameters))
        if token.ty == TokenType.UNDERSCORE:
            return HoleType(token)
        if token.ty == TokenType.IDENT:
            for generic_index, lexeme in enumerate(map(lambda t: t.lexeme, generic_parameters)):
                if lexeme == token.lexeme:
                    return GenericType(token, generic_index)
            return self.parse_struct_type(token, generic_parameters)
        if token.ty == TokenType.LEFT_PAREN:
            args = []
            while True:
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.ARROW:
                    self.advance(skip_ws=True) # skip `=>`
                    break
                args.append(self.parse_type(generic_parameters))
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.ARROW:
                    self.advance(skip_ws=True) # skip `=>`
                    break
                comma = self.advance(skip_ws=True)
                if comma is None or comma.ty != TokenType.COMMA:
                    self.abort("Expected `,` in argument list of function type.")
            rets = []
            while True:
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.RIGHT_PAREN:
                    self.advance(skip_ws=True) # skip `)`
                    break
                rets.append(self.parse_type(generic_parameters))
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.RIGHT_PAREN:
                    self.advance(skip_ws=True) # skip `)`
                    break
                comma = self.advance(skip_ws=True)
                if comma is None or comma.ty != TokenType.COMMA:
                    self.abort("Expected `,` in return list of function type.")
            return FunctionType(token, args, rets)
        if token.ty == TokenType.LEFT_BRACKET:
            items = []
            while True:
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.RIGHT_BRACKET:
                    self.advance(skip_ws=True) # skip `]`
                    break
                items.append(self.parse_type(generic_parameters))
                next = self.advance(skip_ws=True)
                if next is None or next.ty == TokenType.RIGHT_BRACKET:
                    break
                comma = next
                if comma is None or comma.ty != TokenType.COMMA:
                    self.abort("Expected `,` in tuple type.")
            return TupleType(token, items)
        self.abort("Expected type")
