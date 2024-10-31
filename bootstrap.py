#!/usr/bin/env python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeVar, Callable, List, Tuple, NoReturn, Dict, Sequence, Literal, Iterator, TypeGuard, TypeAlias, assert_never
from functools import reduce
import sys
import os
import unittest

# =============================================================================
#  Utils
# =============================================================================

def listtostr[T](seq: Sequence[T], tostr: Callable[[T], str] | None = None, multi_line: bool = False) -> str:
    if len(seq) == 0:
        return "[]"
    s = "[\n" if multi_line else "["
    for e in seq:
        v = str(e) if tostr is None else tostr(e)
        s += indent(v) if multi_line else v
        s += ",\n" if multi_line else ", "
    return s[0:-2] + "]" if multi_line else s[0:-2] + "]"

@dataclass
class Lazy[T]:
    produce: Callable[[], T]
    inner: T | None = None

    def get(self) -> T:
        if self.inner is not None:
            return self.inner
        v = self.produce()
        self.inner = v
        self.produce = lambda: v
        return self.inner

    def has_value(self) -> bool:
        return self.inner is not None

sys_stdin = Lazy(lambda: sys.stdin.read())

@dataclass
class Ref[T]:
    value: T

# =============================================================================
#  Lexer
# =============================================================================

class TokenType(str, Enum):
    NUMBER = "NUMBER"
    FN = "FN"
    IMPORT = "IMPORT"
    IDENT = "IDENT"
    STRING = "STRING"
    LEFT_TRIANGLE = "LEFT_TRIANGLE"
    RIGHT_TRIANGLE = "RIGHT_TRIANGLE"
    LEFT_PAREN = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    LEFT_BRACE = "LEFT_BRACE"
    RIGHT_BRACE = "RIGHT_BRACE"
    COLON = "COLON"
    DOT = "DOT"
    COMMA = "COMMA"
    DOLLAR = "DOLLAR"
    AMPERSAND = "AMPERSAND"
    HASH = "HASH"
    AT = "AT"
    BANG = "BANG"
    TILDE = "TILDE"
    BACKSLASH = "BACKSLASH"
    LEFT_BRACKET = "LEFT_BRACKET"
    RIGHT_BRACKET = "RIGHT_BRACKET"
    GLOBAL = "GLOBAL"
    SIZEOF = "SIZEOF"
    AS = "AS"
    STRUCT = "STRUCT"
    BLOCK = "BLOCK"
    BREAK = "BREAK"
    LOOP = "LOOP"
    IF = "IF"
    ELSE = "ELSE"
    EXTERN = "EXTERN"
    BOOL = "BOOL"
    I8 = "I8"
    I32 = "I32"
    I64 = "I64"
    ARROW = "ARROW"
    DOUBLE_ARROW = "DOUBLE_ARROW"
    SPACE = "SPACE"
    MAKE = "MAKE"
    VARIANT = "VARIANT"
    MATCH = "MATCH"
    CASE = "CASE"
    UNDERSCORE = "UNDERSCORE"

@dataclass(frozen=True)
class Token:
    ty: TokenType
    line: int
    column: int
    lexeme: str

    @staticmethod
    def space(line: int, column: int) -> 'Token':
        return Token(TokenType.SPACE, line, column, " ")

    @staticmethod
    def keyword(ty: TokenType, line: int, column: int) -> 'Token':
        return Token(ty, line, column, Token.keyword_lexeme(ty))

    @staticmethod
    def keyword_lexeme(ty: TokenType) -> str:
        return TYPE_LEXEME_DICT[ty]

    def __str__(self) -> str:
        return f"({self.ty.value} {self.lexeme} {str(self.line)} {str(self.column)})"

    @staticmethod
    def dummy(lexeme: str) -> 'Token':
        return Token(TokenType.STRING, 0, 0, lexeme)

LEXEME_TYPE_DICT: dict[str, TokenType] = {
    "fn":     TokenType.FN,
    "import": TokenType.IMPORT,
    "as":     TokenType.AS,
    "global": TokenType.GLOBAL,
    "struct": TokenType.STRUCT,
    "block":  TokenType.BLOCK,
    "break":  TokenType.BREAK,
    "loop":   TokenType.LOOP,
    "if":     TokenType.IF,
    "else":   TokenType.ELSE,
    "extern": TokenType.EXTERN,
    "bool":   TokenType.BOOL,
    "i8":     TokenType.I8,
    "i32":    TokenType.I32,
    "i64":    TokenType.I64,
    "sizeof": TokenType.SIZEOF,
    "->":     TokenType.ARROW,
    "=>":     TokenType.DOUBLE_ARROW,
    " ":      TokenType.SPACE,
    "<":      TokenType.LEFT_TRIANGLE,
    ">":      TokenType.RIGHT_TRIANGLE,
    "(":      TokenType.LEFT_PAREN,
    ")":      TokenType.RIGHT_PAREN,
    "{":      TokenType.LEFT_BRACE,
    "}":      TokenType.RIGHT_BRACE,
    "[":      TokenType.LEFT_BRACKET,
    "]":      TokenType.RIGHT_BRACKET,
    ":":      TokenType.COLON,
    ".":      TokenType.DOT,
    ",":      TokenType.COMMA,
    "$":      TokenType.DOLLAR,
    "&":      TokenType.AMPERSAND,
    "#":      TokenType.HASH,
    "@":      TokenType.AT,
    "!":      TokenType.BANG,
    "~":      TokenType.TILDE,
    "\\":     TokenType.BACKSLASH,
    "make":   TokenType.MAKE,
    "variant":TokenType.VARIANT,
    "match":  TokenType.MATCH,
    "case":   TokenType.CASE,
    "_":      TokenType.UNDERSCORE,
}
TYPE_LEXEME_DICT: dict[TokenType, str] = {v: k for k, v in LEXEME_TYPE_DICT.items()}

@dataclass
class LexerException(Exception):
    message: str
    line: int
    column: int

@dataclass
class Lexer:
    input: str
    cursor: int = 0
    line: int = 1
    column: int = 1
    tokens: List[Token] = field(default_factory=list)

    def current(self) -> str:
        return self.input[self.cursor]

    def eof(self) -> bool:
        return self.cursor == len(self.input)

    def last_char(self) -> bool:
        return self.cursor + 1 == len(self.input)

    def peek(self) -> str:
        return self.input[self.cursor + 1]

    def advance(self, n: int = 1):
        self.cursor += n
        self.column += n

    def add_space(self):
        if len(self.tokens) == 0 or self.tokens[-1].ty != TokenType.SPACE:
            self.tokens.append(Token.space(self.line, self.column))
        pass

    def lex(self) -> List[Token]:
        while not self.eof():
            if not self.last_char() and self.current() == '/' and self.peek() == '/':
                while not self.eof() and self.current() != '\n':
                    self.advance()
                continue

            if self.current() == '\n':
                self.add_space()
                self.advance()
                self.column = 1
                self.line += 1
                continue

            if self.current() == '\r':
                self.add_space()
                self.advance()
                continue

            if self.current() == ' ':
                self.add_space()
                self.advance()
                continue

            if self.current() == '\t':
                self.add_space()
                self.advance()
                continue

            if self.current() == '-':
                if not self.last_char() and self.peek() == '>':
                    self.tokens.append(Token.keyword(TokenType.ARROW, self.line, self.column))
                    self.advance(2)
                    continue

            if self.current() == '=':
                if not self.last_char() and self.peek() == '>':
                    self.tokens.append(Token.keyword(TokenType.DOUBLE_ARROW, self.line, self.column))
                    self.advance(2)
                    continue

            one_char_tokens = "<>(){}:.,$&#@!~\\_[]"
            if self.current() in one_char_tokens:
                ty = TokenType(LEXEME_TYPE_DICT[self.current()])
                self.tokens.append(Token(ty, self.line, self.column, self.current()))
                self.advance()
                continue

            if self.current() == '"':
                start = self.cursor
                start_line = self.line
                start_column = self.column

                self.advance()
                while True:
                    if self.eof():
                        raise LexerException("Unterminated String", self.line, self.column)
                    char = self.current()
                    self.advance()
                    if char == '"':
                        break

                    if char == '\\':
                        if self.eof():
                            raise LexerException("Unterminated String", self.line, self.column)
                        if self.current() in "ntr\\\"":
                            self.advance()
                lexeme = self.input[start:self.cursor]
                self.tokens.append(Token(TokenType.STRING, start_line, start_column, lexeme))
                continue

            if self.current().isdigit():
                start = self.cursor
                start_column = self.column
                while not self.eof() and self.current().isdigit():
                    self.advance()
                lexeme = self.input[start:self.cursor]
                self.tokens.append(Token(TokenType.NUMBER, self.line, start_column, lexeme))
                continue

            if Lexer.allowed_in_ident(self.current()):
                start_column = self.column
                start = self.cursor
                while not self.eof() and Lexer.allowed_in_ident(self.current()):
                    self.advance()
                lexeme = self.input[start:self.cursor]
                try:
                    self.tokens.append(Token(LEXEME_TYPE_DICT[lexeme], self.line, start_column, lexeme))
                except KeyError:
                    self.tokens.append(Token(TokenType.IDENT, self.line, start_column, lexeme))
                continue
            raise LexerException("Unexpected character encountered: " + self.current(), self.line, self.column)

        return self.tokens

    @staticmethod
    def allowed_in_ident(char: str) -> bool:
        return char not in "#${}()<> \t\n:&~,.[]"

# =============================================================================
#  Parser
# =============================================================================

class PrimitiveType(str, Enum):
    I8 = "TYPE_I8"
    I32 = "TYPE_I32"
    I64 = "TYPE_I64"
    BOOL = "TYPE_BOOL"

    def __str__(self) -> str:
        if self == PrimitiveType.I8:
            return "I8"
        if self == PrimitiveType.I32:
            return "I32"
        if self == PrimitiveType.I64:
            return "I64"
        if self == PrimitiveType.BOOL:
            return "Bool"
        assert_never(self)

    def pretty(self) -> str:
        if self == PrimitiveType.I8:
            return "i8"
        if self == PrimitiveType.I32:
            return "i32"
        if self == PrimitiveType.I64:
            return "i64"
        if self == PrimitiveType.BOOL:
            return "bool"
        assert_never(self)

    def size(self) -> int:
        if self == PrimitiveType.I8:
            return 1
        if self == PrimitiveType.I32:
            return 4
        if self == PrimitiveType.I64:
            return 8
        if self == PrimitiveType.BOOL:
            return 4
        assert_never(self)

    def can_live_in_reg(self) -> bool:
        return True

@dataclass
class GenericType:
    token: Token
    generic_index: int

    def __str__(self) -> str:
        return f"(GenericType {self.token} {self.generic_index})"

type ParsedType = 'PrimitiveType | Parser.PtrType | Parser.TupleType | GenericType | Parser.ForeignType | Parser.StructType | Parser.FunctionType'

@dataclass
class NumberWord:
    token: Token

    def __str__(self) -> str:
        return f"(Number {self.token})"

@dataclass
class BreakWord:
    token: Token

    def __str__(self) -> str:
        return f"(Break {self.token})"

type ParsedWord = 'NumberWord | Parser.StringWord | Parser.CallWord | Parser.GetWord | Parser.RefWord | Parser.SetWord | Parser.StoreWord | Parser.InitWord | Parser.CallWord | Parser.ForeignCallWord | Parser.FunRefWord | Parser.IfWord | Parser.LoadWord | Parser.LoopWord | Parser.BlockWord | BreakWord | Parser.CastWord | Parser.SizeofWord | Parser.GetFieldWord | Parser.IndirectCallWord | Parser.StructWord | Parser.UnnamedStructWord | Parser.MatchWord | Parser.VariantWord | Parser.TupleUnpackWord | Parser.TupleMakeWord'

type ParsedTypeDefinition = 'Parser.Struct | Parser.Variant'

type ParsedTopItem = 'Parser.Import | ParsedTypeDefinition | Parser.Global | Parser.Function | Parser.Extern'

@dataclass
class ParserException(Exception):
    file_path: str
    file: str
    token: Token | None
    message: str

    def display(self) -> str:
        if self.token is None:
            lines = self.file.splitlines()
            line = len(lines) + 1
            column = len(lines[-1]) + 1 if len(lines) != 0 else 1
        else:
            line = self.token.line
            column = self.token.column
        return f"{self.file_path}:{line}:{column} {self.message}"

@dataclass
class Parser:
    file_path: str
    file: str
    tokens: List[Token]
    cursor: int = 0

    # ========================================
    # Datatypes for parsed types
    # ========================================
    @dataclass
    class PtrType:
        child: ParsedType

    @dataclass
    class ForeignType:
        module: Token
        name: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class TupleType:
        token: Token
        items: List[ParsedType]

    @dataclass
    class StructType:
        name: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class FunctionType:
        token: Token
        args: List[ParsedType]
        rets: List[ParsedType]

    @dataclass
    class NamedType:
        name: Token
        taip: ParsedType

        def __str__(self) -> str:
            return f"(NamedType {self.name} {self.taip})"


    # ========================================
    # Datatypes for parsed words
    # ========================================
    @dataclass
    class StringWord:
        token: Token
        string: bytearray

    @dataclass
    class GetWord:
        token: Token
        fields: List[Token]

    @dataclass
    class RefWord:
        token: Token
        fields: List[Token]

    @dataclass
    class SetWord:
        token: Token
        fields: List[Token]

    @dataclass
    class StoreWord:
        token: Token
        fields: List[Token]

    @dataclass
    class InitWord:
        name: Token

    @dataclass
    class ForeignCallWord:
        module: Token
        name: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class CallWord:
        name: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class FunRefWord:
        call: 'Parser.CallWord | Parser.ForeignCallWord'

    @dataclass
    class IfWord:
        token: Token
        if_words: List['ParsedWord']
        else_words: List['ParsedWord']

    @dataclass
    class LoadWord:
        token: Token

    @dataclass
    class BlockAnnotation:
        parameters: List[ParsedType]
        returns: List[ParsedType]

        def __str__(self) -> str:
            return f"(BlockAnnotation {listtostr(self.parameters)} {listtostr(self.returns)})"

    @dataclass
    class Words:
        words: List['ParsedWord']
        end: Token

        def __str__(self) -> str:
            return f"(Words {listtostr(self.words)} {self.end})"

    @dataclass
    class LoopWord:
        token: Token
        words: 'Parser.Words'
        annotation: 'Parser.BlockAnnotation | None'

        def __str__(self) -> str:
            return f"(Loop {self.token} {format_maybe(self.annotation)} {self.words})"

    @dataclass
    class BlockWord:
        token: Token
        words: 'Parser.Words'
        annotation: 'Parser.BlockAnnotation | None'

        def __str__(self) -> str:
            return f"(Block {self.token} {format_maybe(self.annotation)} {self.words})"

    @dataclass
    class CastWord:
        token: Token
        taip: ParsedType

    @dataclass
    class SizeofWord:
        token: Token
        taip: ParsedType

    @dataclass
    class GetFieldWord:
        token: Token
        fields: List[Token]

    @dataclass
    class IndirectCallWord:
        token: Token

    @dataclass
    class StructWord:
        token: Token
        taip: 'Parser.StructType | Parser.ForeignType'
        words: List['ParsedWord']

    @dataclass
    class UnnamedStructWord:
        token: Token
        taip: 'Parser.StructType | Parser.ForeignType'

    @dataclass
    class VariantWord:
        token: Token
        taip: 'Parser.StructType | Parser.ForeignType'
        case: Token

    @dataclass
    class MatchCase:
        token: Token
        case: Token
        words: List[ParsedWord]

    @dataclass
    class MatchWord:
        token: Token
        cases: List['Parser.MatchCase']
        default: List[ParsedWord] | None

    @dataclass
    class TupleUnpackWord:
        token: Token

    @dataclass
    class TupleMakeWord:
        token: Token
        item_count: Token


    # ========================================
    # Datatypes for parsed TopItems
    # ========================================
    @dataclass
    class Import:
        token: Token
        file_path: Token
        qualifier: Token
        items: List[Token]

    @dataclass
    class FunctionSignature:
        export_name: Optional[Token]
        name: Token
        generic_parameters: List[Token]
        parameters: List['Parser.NamedType']
        returns: List[ParsedType]

        def __str__(self) -> str:
            return f"(Signature {listtostr(self.generic_parameters)} {listtostr(self.parameters)} {listtostr(self.returns)})"

    @dataclass
    class Extern:
        module: Token
        name: Token
        signature: 'Parser.FunctionSignature'

    @dataclass
    class Global:
        token: Token
        name: Token
        taip: ParsedType

    @dataclass
    class Function:
        token: Token
        signature: 'Parser.FunctionSignature'
        body: List[ParsedWord]

        def __str__(self) -> str:
            return f"(Function {self.token} {self.signature.name} {self.signature} {listtostr(self.body, multi_line=True)})"

    @dataclass
    class Struct:
        token: Token
        name: Token
        fields: List['Parser.NamedType']
        generic_parameters: List[Token]

        def __str__(self) -> str:
            return f"(Struct {self.token} {self.name} {listtostr(self.generic_parameters)} {listtostr(self.fields, multi_line=True)})"

    @dataclass
    class VariantCase:
        name: Token
        taip: ParsedType | None

    @dataclass
    class Variant:
        name: Token
        generic_parameters: List[Token]
        cases: List['Parser.VariantCase']


    # ========================================
    # A parsed module, output of the parser
    # ========================================
    @dataclass
    class Module:
        path: str
        file: str
        top_items: List[ParsedTopItem]
        imports: List['Parser.Import']
        type_definitions: List['ParsedTypeDefinition']
        globals: List['Parser.Global']
        functions: List['Parser.Function | Parser.Extern']

        def __str__(self) -> str:
            return listtostr(self.top_items, multi_line=True)


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
        raise ParserException(self.file_path, self.file, self.tokens[self.cursor] if self.cursor < len(self.tokens) else None, message)


    # ========================================
    # Parsing routines
    # ========================================
    def parse(self) -> 'Parser.Module':
        top_items: List[Parser.Import | ParsedTypeDefinition | Parser.Global | Parser.Function | Parser.Extern] = []
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
                top_items.append(Parser.Import(token, file_path, module_qualifier, items))
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
                top_items.append(Parser.Extern(module, name, signature))
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
                    fields.append(Parser.NamedType(field_name, taip))
                top_items.append(Parser.Struct(token, name, fields, generic_parameters))
                continue

            if token.ty == TokenType.VARIANT:
                name = self.advance(skip_ws=True)
                if name is None:
                    self.abort("Expected an identifier")
                generic_parameters = self.parse_generic_parameters()
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.LEFT_BRACE:
                    self.abort("Expected `{`")
                cases: List[Parser.VariantCase] = []
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
                        cases.append(Parser.VariantCase(ident, None))
                        continue
                    self.advance(skip_ws=True)
                    cases.append(Parser.VariantCase(ident, self.parse_type(generic_parameters)))
                top_items.append(Parser.Variant(name, generic_parameters, cases))
                continue

            if token.ty == TokenType.GLOBAL:
                name = self.advance(skip_ws=True)
                if name is None:
                    self.abort("expected an identifier")
                colon = self.advance(skip_ws=True)
                if colon is None or colon.ty != TokenType.COLON:
                    self.abort("Expected `:`")
                taip = self.parse_type([])
                top_items.append(Parser.Global(token, name, taip))
                continue

            self.abort("Expected function import or struct definition")
        def is_import(obj: object) -> TypeGuard[Parser.Import]:
            return isinstance(obj, Parser.Import)
        def is_type_definition(obj: object) -> TypeGuard[ParsedTypeDefinition]:
            return isinstance(obj, Parser.Struct) or isinstance(obj, Parser.Variant)
        def is_global(obj: object) -> TypeGuard[Parser.Global]:
           return isinstance(obj, Parser.Global)
        imports: List[Parser.Import] = list(filter(is_import, top_items))
        type_definitions: List[ParsedTypeDefinition] = list(filter(is_type_definition, top_items))
        globals: List[Parser.Global] = list(filter(is_global, top_items))
        functions: List[Parser.Function | Parser.Extern] = [f for f in top_items if isinstance(f, Parser.Function) or isinstance(f, Parser.Extern)]
        return Parser.Module(self.file_path, self.file, top_items, imports, type_definitions, globals, functions)

    def parse_function(self, start: Token) -> 'Parser.Function':
        signature = self.parse_function_signature()
        token = self.advance(skip_ws=True)
        if token is None or token.ty != TokenType.LEFT_BRACE:
            self.abort("Expected `{`")

        body = self.parse_words(signature.generic_parameters)

        token = self.advance(skip_ws=True)
        assert(token is not None and token.ty == TokenType.RIGHT_BRACE)
        return Parser.Function(start, signature, body)

    def parse_words(self, generic_parameters: List[Token]) -> List[ParsedWord]:
        words = []
        while True:
            token = self.peek(skip_ws=True)
            if token is not None and token.ty == TokenType.RIGHT_BRACE:
                break
            words.append(self.parse_word(generic_parameters))
        return words

    def parse_word(self, generic_parameters: List[Token]) -> ParsedWord:
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
            return Parser.StringWord(token, string)
        if token.ty in [TokenType.DOLLAR, TokenType.AMPERSAND, TokenType.HASH, TokenType.DOUBLE_ARROW]:
            indicator_token = token
            name = self.advance(skip_ws=True)
            if name is None or name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as variable name")
            token = self.peek(skip_ws=True)
            def construct(name: Token, fields: List[Token]) -> ParsedWord:
                match indicator_token.ty:
                    case TokenType.DOLLAR:
                        return Parser.GetWord(name, fields)
                    case TokenType.AMPERSAND:
                        return Parser.RefWord(name, fields)
                    case TokenType.HASH:
                        return Parser.SetWord(name, fields)
                    case TokenType.DOUBLE_ARROW:
                        return Parser.StoreWord(name, fields)
                    case _:
                        assert(False)
            if token is None or token.ty == TokenType.SPACE:
                return construct(name, [])
            fields = self.parse_field_accesses()
            return construct(name, fields)
        if token.ty == TokenType.AT:
            token = self.advance(skip_ws=False)
            if token is None or token.ty != TokenType.IDENT:
                self.abort("Expected an identifier as variable name")
            return Parser.InitWord(token)
        if token.ty == TokenType.IDENT:
            return self.parse_call_word(generic_parameters, token)
        if token.ty == TokenType.BACKSLASH:
            token = self.advance(skip_ws=True) # skip `\`
            assert(token is not None)
            return Parser.FunRefWord(self.parse_call_word(generic_parameters, token))
        if token.ty == TokenType.IF:
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            if_words = self.parse_words(generic_parameters)
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                self.abort("Expected `}`")
            next = self.peek(skip_ws=True)
            if next is None or next.ty != TokenType.ELSE:
                return Parser.IfWord(token, if_words, [])
            self.advance(skip_ws=True) # skip `else`
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            else_words = self.parse_words(generic_parameters)
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                self.abort("Expected `}`")
            return Parser.IfWord(token, if_words, else_words)
        if token.ty == TokenType.TILDE:
            return Parser.LoadWord(token)
        if token.ty == TokenType.LOOP or token.ty == TokenType.BLOCK:
            brace = self.advance(skip_ws=True)
            if brace is None:
                self.abort("Expected `{`")
            if brace.ty == TokenType.LEFT_BRACE:
                parameters = None
                returns = None
            else:
                if brace.ty == TokenType.LEFT_PAREN:
                    parameters = []
                    while True:
                        next = self.peek(skip_ws=True)
                        if next is None or next.ty == TokenType.RIGHT_PAREN:
                           self.advance(skip_ws=True) # skip `)`
                           break
                        parameters.append(self.parse_type(generic_parameters))
                        comma = self.peek(skip_ws=True)
                        if comma is None or comma.ty == TokenType.RIGHT_PAREN:
                            self.advance(skip_ws=True) # skip `)`
                            break
                        if comma.ty != TokenType.COMMA:
                            self.abort("Expected `,`")
                        self.advance(skip_ws=True)
                    arrow = self.peek(skip_ws=True)
                    if arrow is None or (arrow.ty != TokenType.ARROW and arrow.ty != TokenType.LEFT_BRACE):
                        self.abort("Expected `->` or `{`")
                    if arrow.ty == TokenType.ARROW:
                        self.advance(skip_ws=True)
                else:
                    parameters = None
                returns = []
                while True:
                    next = self.peek(skip_ws=True)
                    if next is None or next.ty == TokenType.LEFT_BRACE:
                        self.advance(skip_ws=True)
                        break
                    returns.append(self.parse_type(generic_parameters))
                    comma = self.advance(skip_ws=True)
                    if comma is None or comma.ty == TokenType.LEFT_BRACE:
                        break
                    if comma.ty != TokenType.COMMA:
                        self.abort("Expected `,`")
            annotation = None if parameters is None and returns is None else Parser.BlockAnnotation(parameters or [], returns or [])
            words = self.parse_words(generic_parameters)
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                self.abort("Expected `}`")
            if token.ty == TokenType.LOOP:
                return Parser.LoopWord(token, Parser.Words(words, brace), annotation)
            if token.ty == TokenType.BLOCK:
                return Parser.BlockWord(token, Parser.Words(words, brace), annotation)
        if token.ty == TokenType.BREAK:
            return BreakWord(token)
        if token.ty == TokenType.BANG:
            return Parser.CastWord(token, self.parse_type(generic_parameters))
        if token.ty == TokenType.SIZEOF:
            paren = self.advance(skip_ws=True)
            if paren is None or paren.ty != TokenType.LEFT_PAREN:
                self.abort("Expected `(`")
            taip = self.parse_type(generic_parameters)
            paren = self.advance(skip_ws=True)
            if paren is None or paren.ty != TokenType.RIGHT_PAREN:
                self.abort("Expected `)`")
            return Parser.SizeofWord(token, taip)
        if token.ty == TokenType.DOT:
            self.retreat()
            return Parser.GetFieldWord(token, self.parse_field_accesses())
        if token.ty == TokenType.ARROW:
            return Parser.IndirectCallWord(token)
        if token.ty == TokenType.MAKE:
            struct_name_token = self.advance(skip_ws=True)
            taip = self.parse_struct_type(struct_name_token, generic_parameters)
            dot = self.peek(skip_ws=False)
            if dot is not None and dot.ty == TokenType.DOT:
                self.advance(skip_ws=False)
                case_name = self.advance(skip_ws=False)
                if case_name is None or case_name.ty != TokenType.IDENT:
                    self.abort("expected an identifier")
                return Parser.VariantWord(token, taip, case_name)
            brace = self.peek(skip_ws=True)
            if brace is not None and brace.ty == TokenType.LEFT_BRACE:
                brace = self.advance(skip_ws=True)
                words = self.parse_words(generic_parameters)
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                    self.abort("Expected `}`")
                return Parser.StructWord(token, taip, words)
            return Parser.UnnamedStructWord(token, taip)
        if token.ty == TokenType.MATCH:
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            cases: List[Parser.MatchCase] = []
            while True:
                next = self.peek(skip_ws=True)
                if next is None or next.ty == TokenType.RIGHT_BRACE:
                    self.advance(skip_ws=True)
                    return Parser.MatchWord(token, cases, None)
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
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                    self.abort("Expected `}`")
                if case_name.ty == TokenType.UNDERSCORE:
                    brace = self.advance(skip_ws=True)
                    if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                        self.abort("Expected `}`")
                    return Parser.MatchWord(token, cases, words)
                cases.append(Parser.MatchCase(next, case_name, words))
        if token.ty == TokenType.LEFT_BRACKET:
            comma = self.advance(skip_ws=True)
            if comma is None or comma.ty != TokenType.COMMA:
                self.abort("Expected `,`")
            number_or_close = self.advance(skip_ws=True)
            if number_or_close is None or (number_or_close.ty != TokenType.NUMBER and number_or_close.ty != TokenType.RIGHT_BRACKET):
                self.abort("Expected `,` or `]`")
            if number_or_close.ty == TokenType.RIGHT_BRACKET:
                return Parser.TupleUnpackWord(token)
            close = self.advance(skip_ws=True)
            if close is None or close.ty != TokenType.RIGHT_BRACKET:
                self.abort("Expected `]`")
            return Parser.TupleMakeWord(token, number_or_close)
        self.abort("Expected word")

    def parse_call_word(self, generic_parameters: List[Token], token: Token) -> 'Parser.CallWord | Parser.ForeignCallWord':
        next = self.peek(skip_ws=False)
        if next is not None and next.ty == TokenType.COLON:
            module = token
            self.advance(skip_ws=False) # skip the `:`
            name = self.advance(skip_ws=False)
            if name is None or name.ty != TokenType.IDENT:
                self.abort("Expected an identifier")
            next = self.peek()
            generic_arguments = self.parse_generic_arguments(generic_parameters) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []
            return Parser.ForeignCallWord(module, name, generic_arguments)
        name = token
        generic_arguments = self.parse_generic_arguments(generic_parameters) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []
        return Parser.CallWord(name, generic_arguments)

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

    def parse_function_signature(self) -> 'Parser.FunctionSignature':
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
            parameters.append(Parser.NamedType(parameter_name, parameter_type))
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

        return Parser.FunctionSignature(function_export_name, function_ident, generic_parameters, parameters, returns)

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

    def parse_generic_arguments(self, generic_parameters: List[Token]) -> List[ParsedType]:
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

    def parse_struct_type(self, token: Token | None, generic_parameters: List[Token]) -> 'Parser.StructType | Parser.ForeignType':
        if token is None or token.ty != TokenType.IDENT:
            self.abort("Expected an identifer as struct name")
        next = self.peek(skip_ws=True)
        if next is not None and next.ty == TokenType.COLON:
            self.advance(skip_ws=True) # skip the `:`
            module = token
            struct_name = self.advance(skip_ws=True)
            if struct_name is None or struct_name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as struct name")
            return Parser.ForeignType(module, struct_name, self.parse_generic_arguments(generic_parameters))
        else:
            struct_name = token
            if struct_name is None or struct_name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as struct name")
            return Parser.StructType(struct_name, self.parse_generic_arguments(generic_parameters))

    def parse_type(self, generic_parameters: List[Token]) -> ParsedType:
        token = self.advance(skip_ws=True)
        if token is None:
            self.abort("Expected a type")
        if token.ty == TokenType.I8:
            return PrimitiveType.I8
        if token.ty == TokenType.I32:
            return PrimitiveType.I32
        if token.ty == TokenType.I64:
            return PrimitiveType.I64
        if token.ty == TokenType.BOOL:
            return PrimitiveType.BOOL
        if token.ty == TokenType.DOT:
            return Parser.PtrType(self.parse_type(generic_parameters))
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
            return Parser.FunctionType(token, args, rets)
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
            return Parser.TupleType(token, items)
        self.abort("Expected type")

def load_recursive(modules: Dict[str, Parser.Module], path: str, stdin: str | None = None, import_stack: List[str]=[]):
    if path == "-":
        file = stdin if stdin is not None else sys_stdin.get()
    else:
        with open(path, 'r') as reader:
            file = reader.read()
    tokens = Lexer(file).lex()
    module = Parser(path, file, tokens).parse()
    modules[path] = module
    for imp in module.imports:
        if os.path.dirname(path) != "":
            p = os.path.normpath(os.path.dirname(path) + "/" + imp.file_path.lexeme[1:-1])
        else:
            p = os.path.normpath(imp.file_path.lexeme[1:-1])
        if p in import_stack:
            error_message = "Module import cycle detected: "
            for a in import_stack:
                error_message += f"{a} -> "
            raise ParserException(path, file, imp.file_path, error_message)
        if p in modules:
            continue
        import_stack.append(p)
        load_recursive(modules, p, stdin, import_stack)
        import_stack.pop()

# =============================================================================
#  Resolved Types
# =============================================================================

@dataclass
class ResolvedPtrType:
    child: 'ResolvedType'

    def __str__(self) -> str:
        return f"(Ptr {str(self.child)})"

@dataclass(frozen=True, eq=True)
class ResolvedCustomTypeHandle:
    module: int
    index: int

    def __str__(self) -> str:
        return f"(CustomTypeHandle {str(self.module)} {str(self.index)})"

@dataclass
class ResolvedStructType:
    name: Token
    struct: ResolvedCustomTypeHandle
    generic_arguments: List['ResolvedType']

    def __str__(self) -> str:
        return f"(CustomType {self.struct.module} {self.struct.index} {listtostr(self.generic_arguments)})"

    def pretty(self) -> str:
        s = self.name.lexeme
        if len(self.generic_arguments) == 0:
            return s
        s += "<"
        for i in range(0, len(self.generic_arguments)):
            s += str(self.generic_arguments[i])
            if i + 1 != len(self.generic_arguments):
                s += ", "
        return s + ">"

@dataclass
class ResolvedTupleType:
    token: Token
    items: List['ResolvedType']

    def __str__(self) -> str:
        return f"(TupleType {self.token} {listtostr(self.items)})"

@dataclass
class ResolvedFunctionType:
    token: Token
    parameters: List['ResolvedType']
    returns: List['ResolvedType']

    def __str__(self) -> str:
        return f"(FunType {self.token} {listtostr(self.parameters)} {listtostr(self.returns)})"

ResolvedType = PrimitiveType | ResolvedPtrType | ResolvedTupleType | GenericType | ResolvedStructType | ResolvedFunctionType

def resolved_type_eq(a: ResolvedType, b: ResolvedType):
    if isinstance(a, PrimitiveType):
        return a == b
    if isinstance(a, ResolvedPtrType) and isinstance(b, ResolvedPtrType):
        return resolved_type_eq(a.child, b.child)
    if isinstance(a, ResolvedStructType) and isinstance(b, ResolvedStructType):
        return a.struct.module == b.struct.module and a.struct.index == b.struct.index and resolved_types_eq(a.generic_arguments, b.generic_arguments)
    if isinstance(a, ResolvedFunctionType) and isinstance(b, ResolvedFunctionType):
        if len(a.parameters) != len(b.parameters) or len(a.returns) != len(b.returns):
            return False
        for c,d in zip(a.parameters, b.parameters):
            if not resolved_type_eq(c, d):
                return False
        for c,d in zip(a.parameters, b.parameters):
            if not resolved_type_eq(c, d):
                return False
        return True
    if isinstance(a, ResolvedTupleType) and isinstance(b, ResolvedTupleType):
        return resolved_types_eq(a.items, b.items)
    if isinstance(a, GenericType) and isinstance(b, GenericType):
        return a.generic_index == b.generic_index
    return False

def resolved_types_eq(a: List[ResolvedType], b: List[ResolvedType]) -> bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not resolved_type_eq(a[i], b[i]):
            return False
    return True

def format_resolved_type(a: ResolvedType) -> str:
    match a:
        case PrimitiveType():
            return a.pretty()
        case ResolvedPtrType(child):
            return f".{format_resolved_type(child)}"
        case ResolvedStructType(name, _, generic_arguments):
            if len(generic_arguments) == 0:
                return name.lexeme
            s = name.lexeme + "<"
            for i,arg in enumerate(generic_arguments):
                s += format_resolved_type(arg)
                if i + 1 != len(generic_arguments):
                    s += ", "
            return s + ">"
        case ResolvedFunctionType(_, parameters, returns):
            s = "("
            for param in parameters:
                s += format_resolved_type(param) + ", "
            s = s[:-2] + " -> "
            if len(returns) == 0:
                return s[:-1] + ")"
            for ret in returns:
                s += format_resolved_type(ret) + ", "
            return s[:-2] + ")"
        case ResolvedTupleType(token, items):
            return listtostr(items, format_resolved_type)
        case GenericType(token, _):
            return token.lexeme
        case other:
            assert_never(other)

@dataclass
class ResolvedNamedType:
    name: Token
    taip: ResolvedType

    def __str__(self) -> str:
        return f"(NamedType {str(self.name)} {str(self.taip)})"


# =============================================================================
#  Resolved TopItems
# =============================================================================

@dataclass
class ImportItem:
    name: Token
    handle: 'ResolvedFunctionHandle | ResolvedCustomTypeHandle'

    def __str__(self) -> str:
        return f"(ImportItem {self.name} {self.handle})"

@dataclass
class Import:
    token: Token
    file_path: str
    qualifier: Token
    module: int
    items: List[ImportItem]

    def __str__(self) -> str:
        return f"(Import {self.token} {self.module} {self.file_path} {self.qualifier} {listtostr(self.items, multi_line=True)})"

@dataclass
class ResolvedStruct:
    name: Token
    fields: List['ResolvedNamedType']
    generic_parameters: List[Token]

    def __str__(self) -> str:
        return "(Struct\n" + indent(f"name={self.name},\ngeneric-parameters={listtostr(self.generic_parameters)},\nfields={listtostr(self.fields, multi_line=True)}") + ")"

def format_maybe(v, format = None) -> str:
    return "None" if v is None else (f"(Some {v})" if format is None else f"(Some {format(v)})")

@dataclass
class ResolvedVariantCase:
    name: Token
    taip: 'ResolvedType | None'

    def __str__(self) -> str:
        return f"(VariantCase {self.name} {format_maybe(self.taip)})"

@dataclass
class ResolvedVariant:
    name: Token
    cases: List[ResolvedVariantCase]
    generic_parameters: List[Token]

    def __str__(self) -> str:
        return "(Variant\n" + indent(f"name={self.name},\ngeneric-parameters={indent_non_first(listtostr(self.generic_parameters))},\ncases={listtostr(self.cases, multi_line=True)}") + ")"

ResolvedTypeDefinition = ResolvedStruct | ResolvedVariant

@dataclass
class ResolvedFunctionSignature:
    export_name: Optional[Token]
    name: Token
    generic_parameters: List[Token]
    parameters: List[ResolvedNamedType]
    returns: List[ResolvedType]

    def __str__(self) -> str:
        s = "(Signature\n"
        s += f"  generic-parameters={listtostr(self.generic_parameters)},\n"
        s += f"  parameters={listtostr(self.parameters)},\n"
        s += f"  returns={listtostr(self.returns)}"
        return s + ")"

@dataclass
class ResolvedGlobal:
    taip: ResolvedNamedType
    was_reffed: bool = False

    def __str__(self) -> str:
        return f"(Global {self.taip.name} {self.taip.taip} {self.was_reffed})"

@dataclass(frozen=True, eq=True)
class GlobalId:
    module: int
    index: int

@dataclass(frozen=True, eq=True)
class LocalId:
    name: str
    scope: int
    shadow: int

    def __str__(self) -> str:
        return f"(LocalId \"{self.name}\" {self.scope} {self.shadow})"
@dataclass
class ResolverException(Exception):
    path: str
    file: str
    token: Token
    message: str

    def display(self) -> str:
        if self.token is None:
            lines = self.file.splitlines()
            line = len(lines) + 1
            column = len(lines[-1]) + 1 if len(lines) != 0 else 1
        else:
            line = self.token.line
            column = self.token.column
        return f"{self.path}:{line}:{column} {self.message}"

@dataclass
class ResolvedLocal:
    name: Token
    taip: ResolvedType
    is_parameter: bool
    was_reffed: bool = False

    @staticmethod
    def make(taip: ResolvedNamedType) -> 'ResolvedLocal':
        return ResolvedLocal(taip.name, taip.taip, False)

    @staticmethod
    def make_parameter(taip: ResolvedNamedType) -> 'ResolvedLocal':
        return ResolvedLocal(taip.name, taip.taip, True)

    def __str__(self) -> str:
        return f"(Local {self.name} {self.taip} {self.was_reffed} {self.is_parameter})"

@dataclass
class ResolvedBody:
    words: List['ResolvedWord']
    locals: Dict[LocalId, ResolvedLocal]

@dataclass
class ResolvedFunction:
    signature: ResolvedFunctionSignature
    body: ResolvedBody

    def __str__(self) -> str:
        s = "(Function\n"
        s += f"  name={self.signature.name},\n"
        s += f"  export={format_maybe(self.signature.export_name)},\n"
        s += f"  signature={indent_non_first(str(self.signature))},\n"
        s += f"  locals={indent_non_first(format_dict(self.body.locals))},\n"
        s += f"  words={indent_non_first(listtostr(self.body.words, multi_line=True))}"
        return s + ")"

@dataclass
class ResolvedExtern:
    module: Token
    name: Token
    signature: ResolvedFunctionSignature

    def __str__(self) -> str:
        return f"(Extern {self.signature.name} {self.module.lexeme} {self.name.lexeme} {str(self.signature)})"

# =============================================================================
#  Resolved Words
# =============================================================================

@dataclass
class ResolvedFieldAccess:
    name: Token
    source_taip: ResolvedStructType | ResolvedPtrType
    target_taip: ResolvedType
    field_index: int

    def __str__(self) -> str:
        return f"(FieldAccess {self.name} {self.source_taip} {self.target_taip} {self.field_index})"

@dataclass
class StringWord:
    token: Token
    offset: int
    len: int

@dataclass
class ResolvedLoadWord:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedInitWord:
    name: Token
    local_id: LocalId
    taip: ResolvedType

@dataclass
class ResolvedGetWord:
    token: Token
    local_id: LocalId | GlobalId
    var_taip: ResolvedType
    fields: List[ResolvedFieldAccess]
    taip: ResolvedType

    def __str__(self) -> str:
        return f"(GetLocal {self.token} {self.local_id} {self.var_taip} {self.taip} {listtostr(self.fields, multi_line=True)})"

@dataclass
class ResolvedRefWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[ResolvedFieldAccess]

    def __str__(self) -> str:
        return f"(RefLocal {self.token} {self.local_id} {listtostr(self.fields, multi_line=True)})"

@dataclass
class ResolvedSetWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedStoreWord:
    token: Token
    local: LocalId | GlobalId
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedCallWord:
    name: Token
    function: 'ResolvedFunctionHandle'
    generic_arguments: List[ResolvedType]

    def __str__(self) -> str:
        return f"(Call {self.name} {self.function} {listtostr(self.generic_arguments)})"

@dataclass(frozen=True, eq=True)
class ResolvedFunctionHandle:
    module: int
    index: int

    def __str__(self) -> str:
        return f"(FunctionHandle {self.module} {self.index})"

@dataclass
class ResolvedFunRefWord:
    call: ResolvedCallWord

@dataclass
class ResolvedIfWord:
    token: Token
    parameters: List[ResolvedType]
    returns: List[ResolvedType]
    if_words: List['ResolvedWord']
    else_words: List['ResolvedWord']
    diverges: bool

    def __str__(self) -> str:
        s = "(If\n"
        s += f"  token={self.token},\n"
        s += f"  parameters={listtostr(self.parameters)},\n"
        s += f"  returns={format_maybe(None if self.diverges else self.returns, listtostr)},\n"
        s += f"  true-words={indent_non_first(listtostr(self.if_words, multi_line=True))},\n"
        s += f"  false-words={indent_non_first(listtostr(self.else_words, multi_line=True))}"
        return s + ")"

@dataclass
class ResolvedLoopWord:
    token: Token
    words: List['ResolvedWord']
    parameters: List[ResolvedType]
    returns: List[ResolvedType]
    diverges: bool

    def __str__(self) -> str:
        s = "(Loop\n"
        s += f"  token={self.token},\n"
        s += f"  parameters={listtostr(self.parameters)},\n"
        s += f"  returns={format_maybe(None if self.diverges else self.returns, listtostr)},\n"
        s += f"  words={indent_non_first(listtostr(self.words, multi_line=True))}"
        return s + ")"


@dataclass
class ResolvedBlockWord:
    token: Token
    words: List['ResolvedWord']
    parameters: List[ResolvedType]
    returns: List[ResolvedType]

@dataclass
class ResolvedCastWord:
    token: Token
    source: ResolvedType
    taip: ResolvedType

@dataclass
class ResolvedSizeofWord:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedGetFieldWord:
    token: Token
    fields: List[ResolvedFieldAccess]
    on_ptr: bool

@dataclass
class ResolvedIndirectCallWord:
    token: Token
    taip: ResolvedFunctionType

@dataclass
class ResolvedStructFieldInitWord:
    token: Token
    struct: ResolvedCustomTypeHandle
    generic_arguments: List[ResolvedType]
    taip: ResolvedType

@dataclass
class ResolvedStructWord:
    token: Token
    taip: ResolvedStructType
    words: List['ResolvedWord']

@dataclass
class ResolvedUnnamedStructWord:
    token: Token
    taip: ResolvedStructType

@dataclass
class ResolvedVariantWord:
    token: Token
    tag: int
    variant: ResolvedStructType

@dataclass
class ResolvedMatchCase:
    taip: ResolvedType | None
    tag: int
    words: List['ResolvedWord']

@dataclass
class ResolvedMatchWord:
    token: Token
    variant: ResolvedStructType
    by_ref: bool
    cases: List[ResolvedMatchCase]
    default: List['ResolvedWord'] | None
    parameters: List[ResolvedType]
    returns: List[ResolvedType]

@dataclass
class ResolvedTupleMakeWord:
    token: Token
    taip: ResolvedTupleType

@dataclass
class ResolvedTupleUnpackWord:
    token: Token
    items: List[ResolvedType]


# =============================================================================
#  Resolved Intrinsics
# =============================================================================

class IntrinsicType(str, Enum):
    ADD = "ADD"
    STORE = "STORE"
    STORE8 = "STORE8"
    LOAD8 = "LOAD8"
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
    ROTR = "ROTR"
    ROTL = "ROTL"
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
        "store8": IntrinsicType.STORE8,
        "load8": IntrinsicType.LOAD8,
        "%": IntrinsicType.MOD,
        "/": IntrinsicType.DIV,
        "/=": IntrinsicType.NOT_EQ,
        "*": IntrinsicType.MUL,
        "mem-copy": IntrinsicType.MEM_COPY,
        "mem-fill": IntrinsicType.MEM_FILL,
        "rotl": IntrinsicType.ROTL,
        "rotr": IntrinsicType.ROTR,
        "or": IntrinsicType.OR,
        "store": IntrinsicType.STORE,
        "uninit": IntrinsicType.UNINIT,
        "set-stack-size": IntrinsicType.SET_STACK_SIZE,
}
INTRINSIC_TO_LEXEME: dict[IntrinsicType, str] = {v: k for k, v in INTRINSICS.items()}

@dataclass
class ResolvedIntrinsicAdd:
    token: Token
    taip: ResolvedPtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

    def __str__(self) -> str:
        return f"(Intrinsic {self.token} (Add {self.taip}))"

@dataclass
class ResolvedIntrinsicSub:
    token: Token
    taip: ResolvedPtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicDrop:
    token: Token

    def __str__(self) -> str:
        return f"(Drop {self.token})"

@dataclass
class ResolvedIntrinsicMod:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicMul:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicDiv:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicAnd:
    token: Token
    taip: PrimitiveType

@dataclass
class ResolvedIntrinsicOr:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicRotr:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicRotl:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicGreater:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicLess:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicGreaterEq:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicLessEq:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicLoad8:
    token: Token

@dataclass
class IntrinsicStore8:
    token: Token

@dataclass
class IntrinsicMemCopy:
    token: Token

@dataclass
class IntrinsicMemFill:
    token: Token

@dataclass
class ResolvedIntrinsicEqual:
    token: Token
    taip: ResolvedType

    def __str__(self) -> str:
        return f"(Intrinsic {self.token} (Eq {self.taip}))"

@dataclass
class ResolvedIntrinsicNotEqual:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicFlip:
    token: Token
    lower: ResolvedType
    upper: ResolvedType

@dataclass
class IntrinsicMemGrow:
    token: Token

@dataclass
class IntrinsicSetStackSize:
    token: Token

@dataclass
class ResolvedIntrinsicStore:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicNot:
    token: Token
    taip: PrimitiveType

@dataclass
class ResolvedIntrinsicUninit:
    token: Token
    taip: ResolvedType

ResolvedIntrinsicWord = ResolvedIntrinsicAdd | ResolvedIntrinsicSub | IntrinsicDrop | ResolvedIntrinsicMod | ResolvedIntrinsicMul | ResolvedIntrinsicDiv | ResolvedIntrinsicAnd | ResolvedIntrinsicOr | ResolvedIntrinsicRotr | ResolvedIntrinsicRotl | ResolvedIntrinsicGreater | ResolvedIntrinsicLess | ResolvedIntrinsicGreaterEq | ResolvedIntrinsicLessEq | IntrinsicStore8 | IntrinsicLoad8 | IntrinsicMemCopy | IntrinsicMemFill | ResolvedIntrinsicEqual | ResolvedIntrinsicNotEqual | ResolvedIntrinsicFlip | IntrinsicMemGrow | ResolvedIntrinsicStore | ResolvedIntrinsicNot | ResolvedIntrinsicUninit | IntrinsicSetStackSize

ResolvedWord = NumberWord | StringWord | ResolvedCallWord | ResolvedGetWord | ResolvedRefWord | ResolvedSetWord | ResolvedStoreWord | ResolvedCallWord | ResolvedCallWord | ResolvedFunRefWord | ResolvedIfWord | ResolvedLoadWord | ResolvedLoopWord | ResolvedBlockWord | BreakWord | ResolvedCastWord | ResolvedSizeofWord | ResolvedGetFieldWord | ResolvedIndirectCallWord | ResolvedIntrinsicWord | ResolvedInitWord | ResolvedStructFieldInitWord | ResolvedStructWord | ResolvedUnnamedStructWord | ResolvedVariantWord | ResolvedMatchWord | ResolvedTupleMakeWord | ResolvedTupleUnpackWord

# =============================================================================
#  Resolver / Typechecker
# =============================================================================

@dataclass
class ResolvedModule:
    path: str
    id: int
    imports: Dict[str, List[Import]]
    type_definitions: List[ResolvedTypeDefinition]
    globals: List[ResolvedGlobal]
    functions: List[ResolvedFunction | ResolvedExtern]
    data: bytes

    def __str__(self):
        type_definitions = { d.name.lexeme: d for d in self.type_definitions }
        globals = { g.taip.name.lexeme: g for g in self.globals }
        functions = { f.signature.name.lexeme: f for f in self.functions }
        return f"(Module\n  imports={indent_non_first(format_dict(self.imports))},\n  custom-types={indent_non_first(format_dict(type_definitions))},\n  globals={indent_non_first(format_dict(globals))},\n  functions={indent_non_first(format_dict(functions))})"

def determine_compilation_order(unprocessed: List[Parser.Module]) -> List[Parser.Module]:
    ordered: List[Parser.Module] = []
    while len(unprocessed) > 0:
        i = 0
        while i < len(unprocessed):
            module = unprocessed[i]
            postpone = False
            for imp in module.imports:
                if os.path.dirname(module.path) != "":
                    path = os.path.normpath(os.path.dirname(module.path) + "/" + imp.file_path.lexeme[1:-1])
                else:
                    path = os.path.normpath(imp.file_path.lexeme[1:-1])
                if path not in map(lambda m: m.path, ordered):
                    postpone = True
                    break
            if postpone:
                i += 1
                continue
            ordered.append(unprocessed.pop(i))
    return ordered

class Env:
    parent: 'Env | None'
    scope_counter: Ref[int]
    scope_id: int
    vars: Dict[str, List[Tuple[ResolvedLocal, LocalId]]]
    vars_by_id: Dict[LocalId, ResolvedLocal]

    def __init__(self, parent: 'Env | List[ResolvedLocal]'):
        if isinstance(parent, Env):
            self.parent = parent
        else:
            self.parent = None
        self.scope_counter = parent.scope_counter if isinstance(parent, Env) else Ref(0)
        self.scope_id = self.scope_counter.value
        self.scope_counter.value += 1
        self.vars = {}
        self.vars_by_id = parent.vars_by_id if isinstance(parent, Env) else {}
        if isinstance(parent, list):
            for param in parent:
                self.insert(param)

    def lookup(self, name: Token) -> Tuple[ResolvedLocal, LocalId] | None:
        if name.lexeme not in self.vars:
            if self.parent is not None:
                return self.parent.lookup(name)
            return None
        vars = self.vars[name.lexeme]
        if len(vars) == 0:
            if self.parent is not None:
                return self.parent.lookup(name)
            return None
        return vars[-1]

    def insert(self, var: ResolvedLocal) -> LocalId:
        if var.name.lexeme in self.vars:
            id = LocalId(var.name.lexeme, self.scope_id, len(self.vars[var.name.lexeme]))
            self.vars[var.name.lexeme].append((var, id))
            self.vars_by_id[id] = var
            return id
        id = LocalId(var.name.lexeme, self.scope_id, 0)
        self.vars[var.name.lexeme] = [(var, id)]
        self.vars_by_id[id] = var
        return id

    def get_var_type(self, id: LocalId) -> ResolvedType:
        return self.vars_by_id[id].taip

    def mark_var_as_reffed(self, id: LocalId):
        self.vars_by_id[id].was_reffed = True

@dataclass
class Stack:
    parent: 'Stack | None'
    stack: List[ResolvedType]
    negative: List[ResolvedType]

    @staticmethod
    def empty() -> 'Stack':
        return Stack(None, [], [])

    def append(self, taip: ResolvedType):
        self.stack.append(taip)

    def extend(self, taips: List[ResolvedType]):
        for taip in taips:
            self.append(taip)

    def pop(self) -> ResolvedType | None:
        if len(self.stack) != 0:
            return self.stack.pop()
        if self.parent is None:
            return None
        taip = self.parent.pop()
        if taip is None:
            return None
        self.negative.append(taip)
        return taip

    def clone(self) -> 'Stack':
        return Stack(self.parent.clone() if self.parent is not None else None, list(self.stack), list(self.negative))

    def dump(self) -> List[ResolvedType]:
        dump: List[ResolvedType] = []
        while True:
            t = self.pop()
            if t is None:
                dump.reverse()
                return dump
            dump.append(t)

    def make_child(self) -> 'Stack':
        return Stack(self.clone(), [], [])

    def apply(self, other: 'Stack'):
        for _ in other.negative:
            self.pop()
        for added in other.stack:
            self.append(added)

    def lift(self, n: int):
        popped = []
        for _ in range(n):
            taip = self.pop()
            assert(taip is not None)
            popped.append(taip)
        for taip in reversed(popped):
            self.append(taip)

    def compatible_with(self, other: 'Stack') -> bool:
        if len(self) != len(other):
            return False
        self.lift(len(other.stack))
        other.lift(len(self.stack))
        negative_is_fine = resolved_types_eq(self.negative, other.negative)
        positive_is_fine = resolved_types_eq(self.stack, other.stack)
        return negative_is_fine and positive_is_fine

    def __len__(self) -> int:
        return len(self.stack) + (len(self.parent) if self.parent is not None else 0)

    def __getitem__(self, index: int) -> ResolvedType:
        if index > 0:
            return self.stack[index]
        if abs(index) <= len(self.stack):
            return self.stack[index]
        assert(self.parent is not None)
        return self.parent[index + len(self.stack)]

    def __eq__(self, b: object) -> bool:
        if not isinstance(b, Stack):
            return False
        a = self
        ia = len(a.stack)
        ib = len(b.stack)
        while True:
            while ia == 0 and a.parent is not None:
                a = a.parent
                ia = len(a.stack)
            while ib == 0 and b.parent is not None:
                b = b.parent
                ib = len(b.stack)
            if (a is None) ^ (b is None):
                return False
            if a is None and b is None:
                return True
            a_end = ia == 0 and a.parent is None
            b_end = ib == 0 and b.parent is None
            if a_end ^ b_end:
                return False
            if a_end and b_end:
                return True
            if not resolved_type_eq(a.stack[ia - 1], b.stack[ib - 1]):
                return False
            ib -= 1
            ia -= 1

    def __str__(self) -> str:
        return f"Stack(parent={self.parent}, stack={listtostr(self.stack)}, negative={listtostr(self.negative)})"

@dataclass
class BreakStack:
    token: Token
    types: List[ResolvedType]
    reachable: bool

@dataclass
class StructLitContext:
    struct: ResolvedCustomTypeHandle
    generic_arguments: List[ResolvedType]
    fields: Dict[str, ResolvedType]

@dataclass
class ResolveWordContext:
    env: Env
    break_stacks: List[BreakStack] | None
    block_returns: List[ResolvedType] | None
    reachable: bool
    struct_context: StructLitContext | None

    def with_env(self, env: Env) -> 'ResolveWordContext':
        return ResolveWordContext(env, self.break_stacks, self.block_returns, self.reachable, self.struct_context)

    def with_break_stacks(self, break_stacks: List[BreakStack], block_returns: List[ResolvedType] | None) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, break_stacks, block_returns, self.reachable, self.struct_context)

    def with_reachable(self, reachable: bool) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, self.break_stacks, self.block_returns, reachable, self.struct_context)

    def with_struct_context(self, struct_context: StructLitContext) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, self.break_stacks, self.block_returns, self.reachable, struct_context)

@dataclass
class BlockAnnotation:
    parameters: List[ResolvedType]
    returns: List[ResolvedType]

@dataclass
class FunctionResolver:
    module_resolver: 'ModuleResolver'
    signatures: List[ResolvedFunctionSignature]
    type_definitions: List[ResolvedTypeDefinition]
    function: Parser.Function
    signature: ResolvedFunctionSignature

    def abort(self, token: Token, message: str) -> NoReturn:
        self.module_resolver.abort(token, message)

    def resolve(self) -> ResolvedFunction:
        env = Env(list(map(ResolvedLocal.make_parameter, self.signature.parameters)))
        stack: Stack = Stack.empty()
        context = ResolveWordContext(env, None, None, True, None)
        (words, diverges) = self.resolve_words(context, stack, self.function.body)
        if not diverges:
            if not resolved_types_eq(stack.stack, self.signature.returns):
                msg  =  "unexpected return values:\n"
                msg += f"\texpected: {listtostr(self.signature.returns, format_resolved_type)}\n"
                msg += f"\tactual:   {listtostr(stack.stack, format_resolved_type)}"
                self.abort(self.signature.name, msg)
        body = ResolvedBody(words, env.vars_by_id)
        return ResolvedFunction(self.signature, body)

    def resolve_words(self, context: ResolveWordContext, stack: Stack, parsed_words: List[ParsedWord]) -> Tuple[List[ResolvedWord], bool]:
        parsed_words.reverse()
        diverges = False
        words: List[ResolvedWord] = []
        while len(parsed_words) != 0:
            parsed_word = parsed_words.pop()
            (word, word_diverges) = self.resolve_word(context.with_reachable(not diverges), stack, parsed_word, parsed_words)
            diverges = diverges or word_diverges
            words.append(word)
        return (words, diverges)

    def resolve_word(self, context: ResolveWordContext, stack: Stack, word: ParsedWord, remaining_words: List[ParsedWord]) -> Tuple[ResolvedWord, bool]:
        match word:
            case NumberWord():
                stack.append(PrimitiveType.I32)
                return (word, False)
            case Parser.StringWord(token, string):
                stack.append(ResolvedPtrType(PrimitiveType.I32))
                stack.append(PrimitiveType.I32)
                offset = self.module_resolver.data.find(string)
                if offset == -1:
                    offset = len(self.module_resolver.data)
                    self.module_resolver.data.extend(string)
                return (StringWord(token, offset, len(string)), False)
            case Parser.CastWord(token, parsed_taip):
                source_type = stack.pop()
                if source_type is None:
                    self.abort(token, "expected a non-empty stack")
                resolved_type = self.module_resolver.resolve_type(parsed_taip)
                stack.append(resolved_type)
                return (ResolvedCastWord(token, source_type, resolved_type), False)
            case Parser.IfWord(token, parsed_if_words, parsed_else_words):
                if len(stack) == 0 or stack[-1] != PrimitiveType.BOOL:
                    self.abort(token, "expected a boolean on stack")
                stack.pop()
                if_env = Env(context.env)
                if_stack = stack.make_child()
                else_env = Env(context.env)
                else_stack = stack.make_child()
                (if_words, if_words_diverge) = self.resolve_words(context.with_env(if_env), if_stack, parsed_if_words)
                (else_words, else_words_diverge) = self.resolve_words(context.with_env(else_env), else_stack, parsed_else_words)
                if_parameters = if_stack.negative
                if len(else_words) == 0 and if_words_diverge:
                    remaining_words.reverse()
                    remaining_stack = stack.make_child()
                    remaining_stack.lift(len(if_parameters))
                    (remaining_resolved_words, remaining_words_diverge) = self.resolve_words(context.with_env(else_env), remaining_stack, remaining_words)
                    stack.apply(remaining_stack)
                    remaining_words_parameters = list(remaining_stack.negative)
                    if_returns = remaining_stack.stack
                    if len(if_parameters) >= len(remaining_words_parameters):
                        parameters = if_parameters
                    else:
                        parameters = remaining_words_parameters
                    return (ResolvedIfWord(token, parameters, if_returns, if_words, remaining_resolved_words, remaining_words_diverge), remaining_words_diverge)
                if not if_words_diverge:
                    if_returns = if_stack.stack
                elif not else_words_diverge:
                    if_returns = else_stack.stack
                else:
                    if_returns = []
                if not if_words_diverge and not else_words_diverge:
                    if if_stack != else_stack:
                        error_message = f"stack mismatch between if and else branch:\n\tif   {listtostr(if_stack.stack, format_resolved_type)}\n\telse {listtostr(else_stack.stack, format_resolved_type)}"
                        self.abort(word.token, error_message)
                    stack.apply(if_stack)
                elif if_words_diverge and else_words_diverge:
                    for _ in range(max(len(if_stack.negative), len(else_stack.negative))):
                        stack.pop()
                elif not if_words_diverge:
                    stack.apply(if_stack)
                else:
                    assert(not else_words_diverge)
                    stack.apply(else_stack)
                diverges = if_words_diverge and else_words_diverge
                return (ResolvedIfWord(token, if_parameters, if_returns, if_words, else_words, diverges), diverges)
            case Parser.LoopWord(token, parsed_words, parsed_annotation):
                loop_stack = stack.make_child()
                loop_env = Env(context.env)
                loop_break_stacks: List[BreakStack] = []
                annotation = None if parsed_annotation is None else self.resolve_block_annotation(parsed_annotation)
                loop_context = context.with_env(loop_env).with_break_stacks(loop_break_stacks, annotation.returns if annotation is not None else None)
                (words, _) = self.resolve_words(loop_context, loop_stack, parsed_words.words)
                diverges = len(loop_break_stacks) == 0
                parameters = annotation.parameters if annotation is not None else loop_stack.negative
                if len(loop_break_stacks) != 0:
                    diverges = diverges or not loop_break_stacks[0].reachable
                    for i in range(1, len(loop_break_stacks)):
                        if not resolved_types_eq(loop_break_stacks[0].types, loop_break_stacks[i].types):
                            self.abort(token, self.break_stack_mismatch_error(loop_break_stacks))
                if not resolved_types_eq(parameters, loop_stack.stack):
                    self.abort(token, "unexpected items remaining on stack at the end of loop")
                if len(loop_break_stacks) != 0:
                    returns = annotation.returns if annotation is not None else loop_break_stacks[0].types
                    for _ in range(len(parameters)):
                        stack.pop()
                    stack.extend(returns)
                else:
                    returns = annotation.returns if annotation is not None else loop_stack.stack
                    stack.apply(loop_stack)
                return (ResolvedLoopWord(token, words, parameters, returns, diverges), diverges)
            case Parser.BlockWord(token, parsed_words, parsed_annotation):
                block_stack = stack.make_child()
                block_env = Env(context.env)
                block_break_stacks: List[BreakStack] = []
                annotation = None if parsed_annotation is None else self.resolve_block_annotation(parsed_annotation)
                block_context = context.with_env(block_env).with_break_stacks(block_break_stacks, annotation.returns if annotation is not None else None)
                (words, diverges) = self.resolve_words(block_context, block_stack, parsed_words.words)
                parameters = annotation.parameters if annotation is not None else stack.clone().dump() # TODO: check that the annotation matches the inferred params
                for _ in range(len(parameters)):
                    stack.pop()
                block_end_is_reached = not diverges
                if len(block_break_stacks) != 0:
                    diverges = diverges and not block_break_stacks[0].reachable
                    def on_error():
                        error_message = self.break_stack_mismatch_error(block_break_stacks)
                        end = parsed_words.end
                        error_message += f"\n\t{end.line}:{end.column} {listtostr(block_stack.stack, format_resolved_type)}"
                        self.abort(token, error_message)
                    for i in range(1, len(block_break_stacks)):
                        if not resolved_types_eq(block_break_stacks[0].types, block_break_stacks[i].types):
                            on_error()
                    if block_end_is_reached:
                        if not resolved_types_eq(block_break_stacks[0].types, block_stack.clone().dump()):
                            on_error()
                    returns = block_break_stacks[0].types
                    for _ in range(len(block_stack.negative)):
                        stack.pop()
                    stack.extend(returns)
                else:
                    returns = block_stack.stack
                    stack.apply(block_stack)
                return (ResolvedBlockWord(token, words, parameters, returns), diverges)
            case Parser.CallWord(name):
                if name.lexeme in INTRINSICS:
                    intrinsic = INTRINSICS[name.lexeme]
                    resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
                    return (self.resolve_intrinsic(name, stack, intrinsic, resolved_generic_arguments), False)
                resolved_call_word = self.resolve_call_word(word)
                signature = self.module_resolver.get_signature(resolved_call_word.function)
                self.type_check_call(stack, resolved_call_word.name, resolved_call_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
                return (resolved_call_word, False)
            case Parser.ForeignCallWord(name):
                resolved_word = self.resolve_foreign_call_word(word)
                signature = self.module_resolver.get_signature(resolved_word.function)
                self.type_check_call(stack, name, resolved_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
                return (resolved_word, False)
            case Parser.GetWord(token, fields):
                (var_taip, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_taip, fields)
                resolved_taip = var_taip if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                stack.append(resolved_taip)
                return (ResolvedGetWord(token, local, var_taip, resolved_fields, resolved_taip), False)
            case Parser.InitWord(name):
                if context.struct_context is not None and name.lexeme in context.struct_context.fields:
                    field = context.struct_context.fields.pop(name.lexeme)
                    field = FunctionResolver.resolve_generic(context.struct_context.generic_arguments)(field)
                    self.expect_stack(name, stack, [field])
                    return (ResolvedStructFieldInitWord(name, context.struct_context.struct, context.struct_context.generic_arguments, field), False)
                taip = stack.pop()
                if taip is None:
                    self.abort(name, "expected a non-empty stack")
                named_taip = ResolvedNamedType(name, taip)
                local_id = context.env.insert(ResolvedLocal.make(named_taip))
                return (ResolvedInitWord(name, local_id, taip), False)
            case Parser.RefWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                if all(not isinstance(f.source_taip, ResolvedPtrType) for f in resolved_fields):
                    if isinstance(local, LocalId):
                        context.env.mark_var_as_reffed(local)
                    else:
                        if self.module_resolver.id == local.module:
                            globl = self.module_resolver.globals[local.index]
                        else:
                            globl = self.module_resolver.resolved_modules[local.module].globals[local.index]
                        globl.was_reffed = True
                res_type = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                stack.append(ResolvedPtrType(res_type))
                return (ResolvedRefWord(token, local, resolved_fields), False)
            case Parser.SetWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                expected_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                self.expect_stack(token, stack, [expected_taip])
                return (ResolvedSetWord(token, local, resolved_fields), False)
            case Parser.StoreWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                expected_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                if not isinstance(expected_taip, ResolvedPtrType):
                    self.abort(word.token, "`=>` can only store into ptr types")
                self.expect_stack(token, stack, [expected_taip.child])
                return (ResolvedStoreWord(token, local, resolved_fields), False)
            case Parser.FunRefWord(call):
                match call:
                    case Parser.CallWord(name, _):
                        resolved_call_word = self.resolve_call_word(call)
                        signature = self.module_resolver.get_signature(resolved_call_word.function)
                        stack.append(ResolvedFunctionType(name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                        return (ResolvedFunRefWord(resolved_call_word), False)
                    case Parser.ForeignCallWord(name):
                        resolved_foreign_call_word = self.resolve_foreign_call_word(call)
                        signature = self.module_resolver.get_signature(resolved_foreign_call_word.function)
                        stack.append(ResolvedFunctionType(name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                        return (ResolvedFunRefWord(resolved_foreign_call_word), False)
                    case other:
                        assert_never(other)
            case Parser.LoadWord(token):
                if len(stack) == 0:
                    self.abort(token, "expected a non-empty stack")
                top = stack.pop()
                if not isinstance(top, ResolvedPtrType):
                    self.abort(token, "expected a pointer on the stack")
                stack.append(top.child)
                return (ResolvedLoadWord(token, top.child), False)
            case BreakWord(token):
                if context.block_returns is None:
                    dump = stack.dump()
                else:
                    dump = []
                    for _ in context.block_returns:
                        t = stack.pop()
                        assert(t is not None)
                        dump.append(t)
                    dump.reverse()
                if context.break_stacks is None:
                    self.abort(token, "`break` can only be used inside of blocks and loops")
                context.break_stacks.append(BreakStack(token, dump, context.reachable))
                return (word, True)
            case Parser.SizeofWord(token, parsed_taip):
                stack.append(PrimitiveType.I32)
                return (ResolvedSizeofWord(token, self.module_resolver.resolve_type(parsed_taip)), False)
            case Parser.GetFieldWord(token, fields):
                taip = stack.pop()
                if taip is None:
                    self.abort(word.token, "GetField expected a struct on the stack")
                resolved_fields = self.resolve_fields(taip, fields)
                on_ptr = isinstance(taip, ResolvedPtrType)
                if on_ptr:
                    stack.append(ResolvedPtrType(resolved_fields[-1].target_taip))
                else:
                    stack.append(resolved_fields[-1].target_taip)
                return (ResolvedGetFieldWord(token, resolved_fields, on_ptr), False)
            case Parser.IndirectCallWord(token):
                if len(stack) == 0:
                    self.abort(token, "`->` expected a function on the stack")
                function_type = stack.pop()
                if not isinstance(function_type, ResolvedFunctionType):
                    self.abort(token, "`->` expected a function on the stack")
                self.type_check_call(stack, token, None, function_type.parameters, function_type.returns)
                return (ResolvedIndirectCallWord(token, function_type), False)
            case Parser.StructWord(token, taip, parsed_words):
                resolved_struct_taip = self.module_resolver.resolve_struct_type(taip)
                struct = self.module_resolver.get_type_definition(resolved_struct_taip.struct)
                if isinstance(struct, ResolvedVariant):
                    self.abort(token, "can only make struct types, not variants")
                lit_context = StructLitContext(resolved_struct_taip.struct, resolved_struct_taip.generic_arguments, { field.name.lexeme: field.taip for field in struct.fields })
                (words, diverges) = self.resolve_words(context.with_struct_context(lit_context), stack, parsed_words)
                if len(lit_context.fields) != 0:
                    error_message = "missing fields in struct literal:"
                    for field_name, field_taip in lit_context.fields.items():
                        error_message += f"\n\t{field_name}: {format_resolved_type(field_taip)}"
                    self.abort(token, error_message)
                stack.append(resolved_struct_taip)
                return (ResolvedStructWord(token, resolved_struct_taip, words), diverges)
            case Parser.UnnamedStructWord(token, taip):
                resolved_struct_taip = self.module_resolver.resolve_struct_type(taip)
                struct = self.module_resolver.get_type_definition(resolved_struct_taip.struct)
                if isinstance(struct, ResolvedVariant):
                    self.abort(token, "expected a struct")
                struct_field_types = list(map(FunctionResolver.resolve_generic(resolved_struct_taip.generic_arguments), map(lambda f: f.taip, struct.fields)))
                self.expect_stack(token, stack, struct_field_types)
                stack.append(resolved_struct_taip)
                return (ResolvedUnnamedStructWord(token, resolved_struct_taip), False)
            case Parser.VariantWord(token, taip, case_name):
                resolved_variant_taip = self.module_resolver.resolve_struct_type(taip)
                variant = self.module_resolver.get_type_definition(resolved_variant_taip.struct)
                assert(isinstance(variant, ResolvedVariant))
                tag = None
                for i, case in enumerate(variant.cases):
                    if case.name.lexeme == case_name.lexeme:
                        tag = i
                        break
                if tag is None:
                    self.abort(token, "unexpected argument to variant make")
                case_taip = variant.cases[tag].taip
                if case_taip is not None:
                    self.expect_stack(token, stack, [FunctionResolver.resolve_generic(resolved_variant_taip.generic_arguments)(case_taip)])
                stack.append(resolved_variant_taip)
                return (ResolvedVariantWord(token, tag, resolved_variant_taip), False)
            case Parser.MatchWord(token, cases, default):
                resolved_cases: List[ResolvedMatchCase] = []
                match_diverges = True
                arg = stack.pop()
                if arg is None:
                    self.abort(token, "expected an item on stack")
                if not isinstance(arg, ResolvedStructType) and not (isinstance(arg, ResolvedPtrType) and isinstance(arg.child, ResolvedStructType)):
                    self.abort(token, "can only match on variants")
                if isinstance(arg, ResolvedPtrType):
                    assert(isinstance(arg.child, ResolvedStructType))
                    by_ref = True
                    arg = arg.child
                else:
                    by_ref = False
                variant = self.module_resolver.get_type_definition(arg.struct)
                if isinstance(variant, ResolvedStruct):
                    self.abort(token, "can only match on variants")
                remaining_cases = list(map(lambda c: c.name.lexeme, variant.cases))
                visited_cases: Dict[str, Token] = {}
                case_stacks: List[Tuple[Stack, str, bool]] = []
                for parsed_case in cases:
                    tag = None
                    for i, vc in enumerate(variant.cases):
                        if vc.name.lexeme == parsed_case.case.lexeme:
                            tag = i
                    if tag is None:
                        self.abort(parsed_case.token, "type is not part of variant")
                    case_stack = stack.make_child()
                    taip = variant.cases[tag].taip
                    if taip is not None:
                        if by_ref:
                            taip = ResolvedPtrType(taip)
                        case_stack.append(FunctionResolver.resolve_generic(arg.generic_arguments)(taip))
                    case_context = context.with_env(Env(context.env))
                    (resolved_words, diverges) = self.resolve_words(case_context, case_stack, parsed_case.words)
                    match_diverges = match_diverges and diverges
                    resolved_cases.append(ResolvedMatchCase(taip, tag, resolved_words))
                    if parsed_case.case.lexeme in visited_cases:
                        error_message = "duplicate case in match:"
                        for occurence in [visited_cases[parsed_case.case.lexeme], parsed_case.case]:
                            error_message += f"\n\t{occurence.line}:{occurence.column} {occurence.lexeme}"
                        self.abort(token, error_message)
                    remaining_cases.remove(parsed_case.case.lexeme)
                    visited_cases[parsed_case.case.lexeme] = parsed_case.case
                    case_stacks.append((case_stack, parsed_case.case.lexeme, diverges))
                if default is not None:
                    default_case_stack = stack.make_child()
                    default_case_stack.append(arg if not by_ref else ResolvedPtrType(arg))
                    case_context = context.with_env(Env(context.env))
                    (resolved_words, diverges) = self.resolve_words(case_context, default_case_stack, default)
                    match_diverges = match_diverges and diverges
                    default_case = resolved_words
                    case_stacks.append((default_case_stack, "_", diverges))
                else:
                    default_case = None
                non_diverging_case_stacks = list(map(lambda t: (t[0], t[1]), filter(lambda t: not t[2], case_stacks)))
                if len(non_diverging_case_stacks) != 0:
                    for i in range(1, len(non_diverging_case_stacks)):
                        (case_stack, _) = non_diverging_case_stacks[i]
                        if not case_stack.compatible_with(non_diverging_case_stacks[0][0]):
                            error_message = "arms of match case have different types:"
                            for case_stack, case_name_str in non_diverging_case_stacks:
                                error_message += f"\n\t{listtostr(case_stack.negative, format_resolved_type)} -> {listtostr(case_stack.stack, format_resolved_type)} in case {case_name_str}"
                            self.abort(token, error_message)

                if len(case_stacks) == 0:
                    parameters = []
                    returns = []
                else:
                    most_params = case_stacks[0][0]
                    for i in range(1, len(case_stacks)):
                        if len(most_params.negative) < len(case_stacks[i][0].negative):
                            most_params = case_stacks[i][0]
                    parameters = list(reversed(most_params.negative))
                    returns = list(parameters)
                    if len(non_diverging_case_stacks) != 0:
                        for _ in non_diverging_case_stacks[0][0].negative:
                            returns.pop()
                        returns.extend(non_diverging_case_stacks[0][0].stack)
                    for _ in parameters:
                        stack.pop()
                    for t in returns:
                        stack.append(t)
                if len(remaining_cases) != 0 and default is None:
                    error_message = "missing case in match:"
                    for case_name_str in remaining_cases:
                        error_message += f"\n\t{case_name_str}"
                    self.abort(token, error_message)
                match_diverges = match_diverges
                return (ResolvedMatchWord(token, arg, by_ref, resolved_cases, default_case, parameters, returns), match_diverges)
            case Parser.TupleMakeWord(token, num_items_token):
                num_items = int(num_items_token.lexeme)
                items = []
                for _ in range(num_items):
                    item = stack.pop()
                    if item is None:
                        self.abort(token, f"expected {num_items} of values")
                    items.append(item)
                items.reverse()
                stack.append(ResolvedTupleType(token, items))
                return (ResolvedTupleMakeWord(token, ResolvedTupleType(token, items)), False)
            case Parser.TupleUnpackWord(token):
                tupl = stack.pop()
                if tupl is None or not isinstance(tupl, ResolvedTupleType):
                    self.abort(token, "expected a tuple on the stack")
                stack.extend(tupl.items)
                return (ResolvedTupleUnpackWord(token, tupl.items), False)
            case other:
                assert_never(other)

    def break_stack_mismatch_error(self, break_stacks: List[BreakStack]):
        error_message = "break stack mismatch:"
        for break_stack in break_stacks:
            error_message += f"\n\t{break_stack.token.line}:{break_stack.token.column} {listtostr(break_stack.types, format_resolved_type)}"
        return error_message

    def resolve_block_annotation(self, annotation: Parser.BlockAnnotation) -> BlockAnnotation:
        return BlockAnnotation(
            list(map(self.module_resolver.resolve_type, annotation.parameters)),
            list(map(self.module_resolver.resolve_type, annotation.returns)),
        )

    def resolve_intrinsic(self, token: Token, stack: Stack, intrinsic: IntrinsicType, generic_arguments: List[ResolvedType]) -> ResolvedIntrinsicWord:
        match intrinsic:
            case IntrinsicType.ADD | IntrinsicType.SUB:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if isinstance(taip, ResolvedPtrType):
                    narrow_type: ResolvedPtrType | Literal[PrimitiveType.I32, PrimitiveType.I64] = taip
                    if stack[-1] != PrimitiveType.I32:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, i32]")
                    stack.pop()
                elif taip == PrimitiveType.I32:
                    narrow_type = PrimitiveType.I32
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    stack.append(taip)
                elif taip == PrimitiveType.I64:
                    narrow_type = PrimitiveType.I64
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    stack.append(taip)
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]} cannot add to {format_resolved_type(taip)}")
                if intrinsic == IntrinsicType.ADD:
                    return ResolvedIntrinsicAdd(token, narrow_type)
                if intrinsic == IntrinsicType.SUB:
                    return ResolvedIntrinsicSub(token, narrow_type)
            case IntrinsicType.DROP:
                if len(stack) == 0:
                    self.abort(token, "`drop` expected non empty stack")
                stack.pop()
                return IntrinsicDrop(token)
            case IntrinsicType.MOD | IntrinsicType.MUL | IntrinsicType.DIV:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if taip == PrimitiveType.I32:
                    narrow_type = PrimitiveType.I32
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                elif taip == PrimitiveType.I64:
                    narrow_type = PrimitiveType.I64
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i32, i32] or [i64, i64] on stack")
                stack.append(taip)
                if intrinsic == IntrinsicType.MOD:
                    return ResolvedIntrinsicMod(token, narrow_type)
                if intrinsic == IntrinsicType.MUL:
                    return ResolvedIntrinsicMul(token, narrow_type)
                if intrinsic == IntrinsicType.DIV:
                    return ResolvedIntrinsicDiv(token, narrow_type)
            case IntrinsicType.AND | IntrinsicType.OR:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                match taip:
                    case PrimitiveType.I32:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    case PrimitiveType.I64:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    case PrimitiveType.BOOL:
                        popped = self.expect_stack(token, stack, [PrimitiveType.BOOL, PrimitiveType.BOOL])
                    case _:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` can only add i32, i64 and bool")
                stack.append(popped[0])
                if intrinsic == IntrinsicType.AND:
                    return ResolvedIntrinsicAnd(token, taip)
                if intrinsic == IntrinsicType.OR:
                    return ResolvedIntrinsicOr(token, taip)
            case IntrinsicType.ROTR | IntrinsicType.ROTL:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if taip == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I32])
                stack.append(popped[0])
                if intrinsic == IntrinsicType.ROTR:
                    return ResolvedIntrinsicRotr(token, taip)
                if intrinsic == IntrinsicType.ROTL:
                    return ResolvedIntrinsicRotl(token, taip)
            case IntrinsicType.GREATER | IntrinsicType.LESS | IntrinsicType.GREATER_EQ | IntrinsicType.LESS_EQ:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                if taip == PrimitiveType.I32:
                    narrow_type = PrimitiveType.I32
                    self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                elif taip == PrimitiveType.I64:
                    narrow_type = PrimitiveType.I64
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i32, i32] or [i64, i64] on stack")
                stack.append(PrimitiveType.BOOL)
                if intrinsic == IntrinsicType.GREATER:
                    return ResolvedIntrinsicGreater(token, narrow_type)
                if intrinsic == IntrinsicType.LESS:
                    return ResolvedIntrinsicLess(token, narrow_type)
                if intrinsic == IntrinsicType.GREATER_EQ:
                    return ResolvedIntrinsicGreaterEq(token, narrow_type)
                if intrinsic == IntrinsicType.LESS_EQ:
                    return ResolvedIntrinsicLessEq(token, narrow_type)
            case IntrinsicType.LOAD8:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32)])
                stack.append(PrimitiveType.I32)
                return IntrinsicLoad8(token)
            case IntrinsicType.STORE8:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32), PrimitiveType.I32])
                return IntrinsicStore8(token)
            case IntrinsicType.MEM_COPY:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32), ResolvedPtrType(PrimitiveType.I32), PrimitiveType.I32])
                return IntrinsicMemCopy(token)
            case IntrinsicType.MEM_FILL:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32), PrimitiveType.I32, PrimitiveType.I32])
                return IntrinsicMemFill(token)
            case IntrinsicType.NOT_EQ | IntrinsicType.EQ:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if not resolved_type_eq(stack[-1], stack[-2]):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [a, a] for any a")
                taip = stack[-1]
                stack.pop()
                stack.pop()
                stack.append(PrimitiveType.BOOL)
                if intrinsic == IntrinsicType.EQ:
                    return ResolvedIntrinsicEqual(token, taip)
                if intrinsic == IntrinsicType.NOT_EQ:
                    return ResolvedIntrinsicNotEqual(token, taip)
            case IntrinsicType.FLIP:
                a = stack.pop()
                b = stack.pop()
                if a is None or b is None:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                stack.extend([a, b])
                return ResolvedIntrinsicFlip(token, b, a)
            case IntrinsicType.MEM_GROW:
                self.expect_stack(token, stack, [PrimitiveType.I32])
                stack.append(PrimitiveType.I32)
                return IntrinsicMemGrow(token)
            case IntrinsicType.STORE:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                ptr_type = stack[-2]
                if not isinstance(ptr_type, ResolvedPtrType) or not resolved_type_eq(ptr_type.child, stack[-1]):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, a]")
                taip = stack[-1]
                stack.pop()
                stack.pop()
                return ResolvedIntrinsicStore(token, taip)
            case IntrinsicType.NOT:
                taip = stack[-1]
                if len(stack) == 0 or (taip != PrimitiveType.I32 and taip != PrimitiveType.BOOL) or not isinstance(taip, PrimitiveType):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected a i32 or bool on the stack")
                return ResolvedIntrinsicNot(token, taip)
            case IntrinsicType.UNINIT:
                if len(generic_arguments) != 1:
                    self.abort(token, "uninit only accepts one generic argument")
                stack.append(generic_arguments[0])
                return ResolvedIntrinsicUninit(token, generic_arguments[0])
            case IntrinsicType.SET_STACK_SIZE:
                self.expect_stack(token, stack, [PrimitiveType.I32])
                return IntrinsicSetStackSize(token)
            case _:
                assert_never(intrinsic)

    def expect_stack(self, token: Token, stack: Stack, expected: List[ResolvedType]) -> List[ResolvedType]:
        popped: List[ResolvedType] = []
        def abort() -> NoReturn:
            stackdump = stack.dump() + list(reversed(popped))
            self.abort(token, f"expected:\n\t{listtostr(expected, format_resolved_type)}\ngot:\n\t{listtostr(stackdump, format_resolved_type)}")
        for expected_type in reversed(expected):
            top = stack.pop()
            if top is None:
                abort()
            popped.append(top)
            if not resolved_type_eq(expected_type, top):
                abort()
        return list(reversed(popped))

    def resolve_call_word(self, word: Parser.CallWord) -> ResolvedCallWord:
        resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
        function = self.module_resolver.resolve_function_name(word.name)
        signature = self.module_resolver.get_signature(function)
        if len(resolved_generic_arguments) != len(signature.generic_parameters):
            self.module_resolver.abort(word.name, f"expected {len(signature.generic_parameters)} generic arguments, not {len(resolved_generic_arguments)}")
        return ResolvedCallWord(word.name, function, resolved_generic_arguments)

    def type_check_call(self, stack: Stack, token: Token, generic_arguments: None | List[ResolvedType], parameters: List[ResolvedType], returns: List[ResolvedType]):
        conrete_parameters = list(map(FunctionResolver.resolve_generic(generic_arguments), parameters)) if generic_arguments is not None else parameters
        self.expect_stack(token, stack, conrete_parameters)
        concrete_return_types = list(map(FunctionResolver.resolve_generic(generic_arguments), returns)) if generic_arguments is not None else returns
        stack.extend(concrete_return_types)

    @staticmethod
    def resolve_generic(generic_arguments: None | List[ResolvedType]) -> Callable[[ResolvedType], ResolvedType]:
        def inner(taip: ResolvedType) -> ResolvedType:
            if generic_arguments is None:
                return taip
            match taip:
                case GenericType():
                    return generic_arguments[taip.generic_index]
                case ResolvedPtrType(child):
                    return ResolvedPtrType(inner(child))
                case ResolvedStructType(name, struct, gen_args):
                    return ResolvedStructType(name, struct, list(map(inner, gen_args)))
                case ResolvedFunctionType(token, parameters, returns):
                    return ResolvedFunctionType(token, list(map(inner, parameters)), list(map(inner, returns)))
                case ResolvedTupleType(token, items):
                    return ResolvedTupleType(token, list(map(inner, items)))
                case other:
                    return other
        return inner

    def resolve_foreign_call_word(self, word: Parser.ForeignCallWord) -> ResolvedCallWord:
        resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
        for (qualifier, imps) in self.module_resolver.imports.items():
            if qualifier == word.module.lexeme:
                for imp in imps:
                    module = self.module_resolver.resolved_modules[imp.module]
                    for index, f in enumerate(module.functions):
                        if f.signature.name.lexeme == word.name.lexeme:
                            if len(word.generic_arguments) != len(f.signature.generic_parameters):
                                self.module_resolver.abort(word.name, f"expected {len(f.signature.generic_parameters)} generic arguments, not {len(word.generic_arguments)}")
                            return ResolvedCallWord(word.name, ResolvedFunctionHandle(module.id, index), resolved_generic_arguments)
                self.abort(word.name, f"function {word.name.lexeme} not found")
        self.abort(word.name, f"module {word.module.lexeme} not found")

    def resolve_var_name(self, env: Env, name: Token) -> Tuple[ResolvedType, LocalId | GlobalId]:
        var = env.lookup(name)
        if var is None:
            for index, globl in enumerate(self.module_resolver.globals):
                if globl.taip.name.lexeme == name.lexeme:
                    return (globl.taip.taip, GlobalId(self.module_resolver.id, index))
            self.abort(name, f"local {name.lexeme} not found")
        return (var[0].taip, var[1])

    def resolve_fields(self, taip: ResolvedType, fields: List[Token]) -> List[ResolvedFieldAccess]:
        resolved_fields: List[ResolvedFieldAccess] = []
        while len(fields) > 0:
            field_name = fields[0]
            def inner(source_taip: ResolvedStructType | ResolvedPtrType, fields: List[Token]) -> ResolvedType:
                if isinstance(source_taip, ResolvedStructType):
                    struct = self.module_resolver.get_type_definition(source_taip.struct)
                    generic_arguments = source_taip.generic_arguments
                else:
                    assert(isinstance(source_taip.child, ResolvedStructType))
                    struct = self.module_resolver.get_type_definition(source_taip.child.struct)
                    generic_arguments = source_taip.child.generic_arguments
                if isinstance(struct, ResolvedVariant):
                    self.abort(field_name, "can not access fields of a variant")
                for field_index, struct_field in enumerate(struct.fields):
                    if struct_field.name.lexeme == field_name.lexeme:
                        target_taip = FunctionResolver.resolve_generic(generic_arguments)(struct_field.taip)
                        resolved_fields.append(ResolvedFieldAccess(field_name, source_taip, target_taip, field_index))
                        fields.pop(0)
                        return target_taip
                self.abort(field_name, f"field not found {field_name.lexeme}")
            if isinstance(taip, ResolvedStructType):
                taip = inner(taip, fields)
                continue
            if isinstance(taip, ResolvedPtrType):
                taip = inner(taip, fields)
                continue
            else:
                self.abort(field_name, f"field not found {field_name.lexeme} WTF?")
        return resolved_fields

K = TypeVar('K')
V = TypeVar('V')
def bag(items: Iterator[Tuple[K, V]]) -> Dict[K, List[V]]:
    bag: Dict[K, List[V]] = {}
    for k,v in items:
        if k in bag:
            bag[k].append(v)
        else:
            bag[k] = [v]
    return bag

@dataclass
class ModuleResolver:
    resolved_modules: Dict[int, ResolvedModule]
    resolved_modules_by_path: Dict[str, ResolvedModule]
    module: Parser.Module
    id: int
    imports: Dict[str, List[Import]] = field(default_factory=dict)
    resolved_type_definitions: List[ResolvedTypeDefinition] = field(default_factory=list)
    globals: List[ResolvedGlobal] = field(default_factory=list)
    data: bytearray = field(default_factory=bytearray)
    signatures: List[ResolvedFunctionSignature] = field(default_factory=list)

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolverException(self.module.path, self.module.file, token, message)

    def get_signature(self, function: ResolvedFunctionHandle) -> ResolvedFunctionSignature:
        if self.id == function.module:
            return self.signatures[function.index]
        else:
            return self.resolved_modules[function.module].functions[function.index].signature

    def get_type_definition(self, struct: ResolvedCustomTypeHandle) -> ResolvedTypeDefinition:
        if struct.module == self.id:
            return self.resolved_type_definitions[struct.index]
        return self.resolved_modules[struct.module].type_definitions[struct.index]

    def resolve(self) -> ResolvedModule:
        resolved_imports = bag(map(lambda imp: (imp.qualifier.lexeme, self.resolve_import(imp)), self.module.imports))
        self.imports = resolved_imports
        resolved_type_definitions = list(map(self.resolve_type_definition, self.module.type_definitions))
        self.resolved_type_definitions = resolved_type_definitions
        for i, type_definition in enumerate(self.resolved_type_definitions):
            if isinstance(type_definition, ResolvedStruct) or isinstance(type_definition, ResolvedVariant):
                if self.is_struct_recursive(ResolvedCustomTypeHandle(self.id, i)):
                    self.abort(type_definition.name, "structs and variants cannot be recursive")
        self.globals = list(map(self.resolve_global, self.module.globals))
        resolved_signatures = list(map(lambda f: self.resolve_function_signature(f.signature), self.module.functions))
        self.signatures = resolved_signatures
        resolved_functions = list(map(lambda f: self.resolve_function(f[0], f[1]), zip(resolved_signatures, self.module.functions)))
        return ResolvedModule(self.module.path, self.id, resolved_imports, resolved_type_definitions, self.globals, resolved_functions, self.data)

    def resolve_function(self, signature: ResolvedFunctionSignature, function: Parser.Function | Parser.Extern) -> ResolvedFunction | ResolvedExtern:
        if isinstance(function, Parser.Extern):
            return ResolvedExtern(function.module, function.name, self.resolve_function_signature(function.signature))
        return FunctionResolver(self, self.signatures, self.resolved_type_definitions, function, signature).resolve()

    def resolve_function_name(self, name: Token) -> ResolvedFunctionHandle:
        for index, signature in enumerate(self.signatures):
            if signature.name.lexeme == name.lexeme:
                return ResolvedFunctionHandle(self.id, index)
        for _, imps in self.imports.items():
            for imp in imps:
                for item in imp.items:
                    if item.name.lexeme == name.lexeme and not isinstance(item.handle, ResolvedCustomTypeHandle):
                        return item.handle
        self.abort(name, f"function {name.lexeme} not found")

    def resolve_global(self, globl: Parser.Global) -> ResolvedGlobal:
        return ResolvedGlobal(ResolvedNamedType(globl.name, self.resolve_type(globl.taip)))

    def resolve_import(self, imp: Parser.Import) -> Import:
        if os.path.dirname(self.module.path) != "":
            path = os.path.normpath(os.path.dirname(self.module.path) + "/" + imp.file_path.lexeme[1:-1])
        else:
            path = os.path.normpath(imp.file_path.lexeme[1:-1])
        imported_module = self.resolved_modules_by_path[path]
        resolved_items: List[ImportItem] = []
        for item in imp.items:
            resolved_item = None
            for struct_id, type_definition in enumerate(imported_module.type_definitions):
                if type_definition.name.lexeme == item.lexeme:
                    resolved_item = ImportItem(item, ResolvedCustomTypeHandle(imported_module.id, struct_id))
                    break
            if resolved_item is not None:
                resolved_items.append(resolved_item)
                continue
            for fun_id, function in enumerate(imported_module.functions):
                if function.signature.name.lexeme == item.lexeme:
                    resolved_item = ImportItem(item, ResolvedFunctionHandle(imported_module.id, fun_id))
                    break
            if resolved_item is not None:
                resolved_items.append(resolved_item)
                continue
            if resolved_item is not None:
                resolved_items.append(resolved_item)
                continue
            self.abort(item, "not found")
        return Import(imp.token, imp.file_path.lexeme, imp.qualifier, imported_module.id, resolved_items)

    def resolve_named_type(self, named_type: Parser.NamedType) -> ResolvedNamedType:
        return ResolvedNamedType(named_type.name, self.resolve_type(named_type.taip))

    def resolve_type(self, taip: ParsedType) -> ResolvedType:
        match taip:
            case PrimitiveType():
                return taip
            case Parser.PtrType(child):
                return ResolvedPtrType(self.resolve_type(child))
            case GenericType():
                return taip
            case Parser.FunctionType(token, parsed_args, parsed_rets):
                args = list(map(self.resolve_type, parsed_args))
                rets = list(map(self.resolve_type, parsed_rets))
                return ResolvedFunctionType(token, args, rets)
            case Parser.TupleType(token, items):
                return ResolvedTupleType(token, list(map(self.resolve_type, items)))
            case struct_type:
                return self.resolve_struct_type(struct_type)

    def resolve_struct_type(self, taip: Parser.StructType | Parser.ForeignType) -> ResolvedStructType:
        match taip:
            case Parser.StructType(name, generic_arguments):
                resolved_generic_arguments = list(map(self.resolve_type, generic_arguments))
                struct_handle = self.resolve_struct_name(name)
                if struct_handle.module == self.id:
                    generic_parameters = self.module.type_definitions[struct_handle.index].generic_parameters
                else:
                    generic_parameters = self.resolved_modules[struct_handle.module].type_definitions[struct_handle.index].generic_parameters
                if len(generic_parameters) != len(generic_arguments):
                    self.abort(name, f"expected {len(generic_parameters)} generic arguments, not {len(generic_arguments)}")
                return ResolvedStructType(name, struct_handle, resolved_generic_arguments)
            case Parser.ForeignType(module, name, generic_arguments):
                resolved_generic_arguments = list(map(self.resolve_type, generic_arguments))
                for qualifier, imps in self.imports.items():
                    if qualifier == taip.module.lexeme:
                        for imp in imps:
                            for index, struct in enumerate(self.resolved_modules[imp.module].type_definitions):
                                if struct.name.lexeme == name.lexeme:
                                    return ResolvedStructType(taip.name, ResolvedCustomTypeHandle(imp.module, index), resolved_generic_arguments)
                self.abort(taip.module, f"struct {module.lexeme}:{name.lexeme} not found")
            case other:
                assert_never(other)

    def resolve_struct_name(self, name: Token) -> ResolvedCustomTypeHandle:
        for index, struct in enumerate(self.module.type_definitions):
            if struct.name.lexeme == name.lexeme:
                return ResolvedCustomTypeHandle(self.id, index)
        for _, imps in self.imports.items():
            for imp in imps:
                for item in imp.items:
                    if item.name.lexeme == name.lexeme and isinstance(item.handle, ResolvedCustomTypeHandle):
                        return item.handle
        self.abort(name, f"struct {name.lexeme} not found")

    def resolve_type_definition(self, definition: ParsedTypeDefinition) -> ResolvedTypeDefinition:
        match definition:
            case Parser.Struct():
                return self.resolve_struct(definition)
            case Parser.Variant():
                return self.resolve_variant(definition)
            case other:
                assert_never(other)

    def resolve_struct(self, struct: Parser.Struct) -> ResolvedStruct:
        return ResolvedStruct(struct.name, list(map(self.resolve_named_type, struct.fields)), struct.generic_parameters)

    def is_struct_recursive(self, struct_handle: ResolvedCustomTypeHandle, stack: List[ResolvedStruct | ResolvedVariant] = []) -> bool:
        struct = self.get_type_definition(struct_handle)
        if struct in stack:
            return True
        if isinstance(struct, ResolvedStruct):
            return any(isinstance(field.taip, ResolvedStructType) and self.is_struct_recursive(field.taip.struct, stack + [struct]) for field in struct.fields)
        if isinstance(struct, ResolvedVariant):
            return any(isinstance(case.taip, ResolvedStructType) and self.is_struct_recursive(case.taip.struct, stack + [struct]) for case in struct.cases)
        assert_never(struct)

    def resolve_variant(self, variant: Parser.Variant) -> ResolvedVariant:
        return ResolvedVariant(variant.name, list(map(lambda t: ResolvedVariantCase(t.name, self.resolve_type(t.taip) if t.taip is not None else None), variant.cases)), variant.generic_parameters)

    def resolve_function_signature(self, signature: Parser.FunctionSignature) -> ResolvedFunctionSignature:
        parameters = list(map(self.resolve_named_type, signature.parameters))
        rets = list(map(self.resolve_type, signature.returns))
        return ResolvedFunctionSignature(signature.export_name, signature.name, signature.generic_parameters, parameters, rets)

@dataclass
class PtrType:
    child: 'Type'
    def __str__(self) -> str:
        return f"PtrType(child={str(self.child)})"

    def size(self) -> int:
        return 4

    def can_live_in_reg(self) -> bool:
        return True

@dataclass
class NamedType:
    name: Token
    taip: 'Type'

    def __str__(self) -> str:
        return f"NamedType(name={str(self.name)}, taip={str(self.taip)})"

@dataclass
class FunctionType:
    token: Token
    parameters: List['Type']
    returns: List['Type']

    def size(self) -> int:
        return 4

    def can_live_in_reg(self) -> bool:
        return True

@dataclass
class Struct:
    name: Token
    fields: Lazy[List[NamedType]]
    generic_parameters: List['Type']

    def __str__(self) -> str:
        return f"Struct(name={str(self.name)})"

    def size(self) -> int:
        field_sizes = [field.taip.size() for field in self.fields.get()]
        size = 0
        for i, field_size in enumerate(field_sizes):
            size += field_size
            if field_size % 4 != 0 and i + 1 < len(field_sizes) and field_sizes[i + 1] >= 4:
                size = align_to(size, 4)
        return size

    def field_offset(self, field_index: int) -> int:
        fields = self.fields.get()
        offset = 0
        for i in range(0, field_index):
            field_size = fields[i].taip.size()
            offset += field_size
            if field_size % 4 != 0 and i + 1 < len(fields) and fields[i + 1].taip.size() >= 4:
                offset = align_to(offset, 4)
        return offset

@dataclass
class VariantCase:
    name: Token
    taip: 'Type | None'

@dataclass
class Variant:
    name: Token
    cases: Lazy[List[VariantCase]]
    generic_arguments: List['Type']

    def size(self) -> int:
        return 4 + max((t.taip.size() for t in self.cases.get() if t.taip is not None), default=0)

TypeDefinition = Struct | Variant

@dataclass
class StructHandle:
    module: int
    index: int
    instance: int

    def __str__(self) -> str:
        return f"ResolvedCustomTypeHandle(module={str(self.module)}, index={str(self.index)}, instance={str(self.instance)})"

@dataclass
class StructType:
    name: Token
    struct: StructHandle
    _size: Lazy[int]

    def __str__(self) -> str:
        return f"StructType(name={str(self.name)}, struct={str(self.struct)})"

    def can_live_in_reg(self) -> bool:
        return self.size() <= 8

    def size(self):
        return self._size.get()

    @staticmethod
    def dummy(name: str, size: int) -> 'StructType':
        return StructType(Token.dummy(name), StructHandle(0, 0, 0), Lazy(lambda: size))

@dataclass
class TupleType:
    token: Token
    items: List['Type']

    def can_live_in_reg(self) -> bool:
        return self.size() <= 8

    def size(self):
        return sum(t.size() for t in self.items)

Type = PrimitiveType | PtrType | StructType | FunctionType | TupleType

def type_eq(a: Type, b: Type) -> bool:
    if isinstance(a, PrimitiveType) and isinstance(b, PrimitiveType):
        return a == b
    if isinstance(a, PtrType) and isinstance(b, PtrType):
        return type_eq(a.child, b.child)
    if isinstance(a, StructType) and isinstance(b, StructType):
        return a.struct.module == b.struct.module and a.struct.index == b.struct.index and a.struct.instance == b.struct.instance
    if isinstance(a, FunctionType) and isinstance(b, FunctionType):
        return types_eq(a.parameters, b.parameters) and types_eq(a.returns, b.returns)
    if isinstance(a, TupleType) and isinstance(b, TupleType):
        return types_eq(a.items, b.items)
    return False

def types_eq(a: List[Type], b: List[Type]) -> bool:
    if len(a) != len(b):
        return False
    for i in range(0, len(a)):
        if not type_eq(a[i], b[i]):
            return False
    return True

def format_type(a: Type) -> str:
    match a:
        case PrimitiveType():
            return a.pretty()
        case PtrType(child):
            return f".{format_type(child)}"
        case StructType(name):
            return name.lexeme 
        case FunctionType(_, parameters, returns):
            s = "("
            for param in parameters:
                s += format_type(param) + ", "
            s = s[:-2] + " -> "
            if len(a.returns) == 0:
                return s[:-1] + ")"
            for ret in returns:
                s += format_type(ret) + ", "
            return s[:-2] + ")"
        case TupleType(_, items):
            return listtostr(items, format_type)
        case other:
            assert_never(other)

@dataclass
class ParameterLocal:
    name: Token
    taip: Type
    _lives_in_memory: bool

    def size(self) -> int:
        return self.taip.size()

    def lives_in_memory(self) -> bool:
        return self._lives_in_memory

    def needs_moved_into_memory(self) -> bool:
        return self.lives_in_memory() and self.taip.can_live_in_reg()

    def can_be_abused_as_ref(self) -> bool:
        return not self.taip.can_live_in_reg() or self.taip.size() <= 4

@dataclass
class InitLocal:
    name: Token
    taip: Type
    _lives_in_memory: bool

    def size(self) -> int:
        return self.taip.size()

    def lives_in_memory(self) -> bool:
        return self._lives_in_memory

Local = ParameterLocal | InitLocal

@dataclass
class Body:
    words: List['Word']
    locals_copy_space: int
    max_struct_ret_count: int
    locals: Dict[LocalId, Local]

@dataclass
class FunctionSignature:
    export_name: Optional[Token]
    name: Token
    generic_arguments: List[Type]
    parameters: List[NamedType]
    returns: List[Type]

    def returns_any_struct(self) -> bool:
        return any(isinstance(ret, StructType) for ret in self.returns)

@dataclass
class Global:
    taip: NamedType
    was_reffed: bool

@dataclass
class Extern:
    module: Token
    name: Token
    signature: FunctionSignature

@dataclass
class ConcreteFunction:
    signature: FunctionSignature
    body: Body

@dataclass
class GenericFunction:
    instances: Dict[int, ConcreteFunction]

Function = ConcreteFunction | GenericFunction

@dataclass(frozen=True, eq=True)
class FunctionHandle:
    module: int
    index: int
    instance: int | None

@dataclass(frozen=True, eq=True)
class ExternHandle:
    module: int
    index: int

@dataclass
class CallWord:
    name: Token
    function: 'FunctionHandle | ExternHandle'
    return_space_offset: int

@dataclass
class CastWord:
    token: Token
    source: Type
    taip: Type

@dataclass
class LoadWord:
    token: Token
    taip: Type
    copy_space_offset: int | None

@dataclass
class IfWord:
    token: Token
    parameters: List[Type]
    returns: List[Type]
    if_words: List['Word']
    else_words: List['Word']
    diverges: bool

@dataclass
class IndirectCallWord:
    token: Token
    taip: FunctionType
    return_space_offset: int

@dataclass
class FunRefWord:
    call: CallWord
    table_index: int

@dataclass
class LoopWord:
    token: Token
    words: List['Word']
    parameters: List[Type]
    returns: List[Type]
    diverges: bool

@dataclass
class BlockWord:
    token: Token
    words: List['Word']
    parameters: List[Type]
    returns: List[Type]

@dataclass
class SizeofWord:
    token: Token
    taip: Type

@dataclass
class FieldAccess:
    name: Token
    source_taip: StructType | PtrType
    target_taip: Type
    offset: int

@dataclass
class Offset:
    offset: int

    def __str__(self) -> str:
        return f"i32.const {self.offset} i32.add"

@dataclass
class OffsetLoad:
    offset: int
    taip: Type

    def __str__(self) -> str:
        size = self.taip.size()
        if size <= 4:
            return f"i32.load offset={self.offset}" if self.offset != 0 else "i32.load"
        if size <= 8:
            return f"i64.load offset={self.offset}" if self.offset != 0 else "i64.load"
        if self.offset == 0:
            return f"i32.const {size} memory.copy"
        else:
            return f"i32.const {self.offset} i32.add i32.const {size} memory.copy"

class BitShiftLoad(Enum):
    Upper32 = 0
    Lower32 = 1

    def __str__(self) -> str:
        if self == BitShiftLoad.Upper32:
            return "i64.const 32 i64.shr_u i32.wrap_i64"
        if self == BitShiftLoad.Lower32:
            return "i32.wrap_i64"

type Load = Offset | OffsetLoad | BitShiftLoad

@dataclass
class GetFieldWord:
    token: Token
    target_taip: Type
    loads: List[Load]
    on_ptr: bool
    copy_space_offset: int | None

@dataclass
class SetWord:
    token: Token
    local_id: LocalId | GlobalId
    target_taip: Type
    loads: List[Load]
    var_lives_in_memory: bool

@dataclass
class InitWord:
    token: Token
    local_id: LocalId
    taip: Type
    var_lives_in_memory: bool

@dataclass
class GetWord:
    token: Token
    local_id: LocalId | GlobalId
    target_taip: Type
    loads: List[Load]
    copy_space_offset: int | None
    var_lives_in_memory: bool

@dataclass
class RefWord:
    token: Token
    local_id: LocalId | GlobalId
    loads: List[Load]

@dataclass
class IntrinsicAdd:
    token: Token
    taip: PtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicSub:
    token: Token
    taip: PtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicMul:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicDiv:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicMod:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicEqual:
    token: Token
    taip: Type

@dataclass
class IntrinsicNotEqual:
    token: Token
    taip: Type

@dataclass
class IntrinsicFlip:
    token: Token
    lower: Type
    upper: Type

@dataclass
class IntrinsicAnd:
    token: Token
    taip: PrimitiveType

@dataclass
class IntrinsicNot:
    token: Token
    taip: PrimitiveType

@dataclass
class IntrinsicGreaterEq:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicLessEq:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicGreater:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicLess:
    token: Token
    taip: Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicRotl:
    token: Token
    taip: Type

@dataclass
class IntrinsicRotr:
    token: Token
    taip: Type

@dataclass
class IntrinsicOr:
    token: Token
    taip: Type

@dataclass
class IntrinsicStore:
    token: Token
    taip: Type

@dataclass
class IntrinsicUninit:
    token: Token
    taip: Type
    copy_space_offset: int

@dataclass
class StoreWord:
    token: Token
    local: LocalId | GlobalId
    taip: Type
    loads: List[Load]

@dataclass
class StructWord:
    token: Token
    taip: StructType
    words: List['Word']
    copy_space_offset: int

@dataclass
class UnnamedStructWord:
    token: Token
    taip: StructType
    copy_space_offset: int

@dataclass
class StructFieldInitWord:
    token: Token
    taip: Type
    copy_space_offset: int

@dataclass
class VariantWord:
    token: Token
    tag: int
    variant: StructHandle
    copy_space_offset: int

@dataclass
class MatchCase:
    tag: int
    words: List['Word']

@dataclass
class MatchWord:
    token: Token
    variant: StructHandle
    by_ref: bool
    cases: List[MatchCase]
    default: List['Word'] | None
    parameters: List[Type]
    returns: List[Type]

@dataclass
class TupleMakeWord:
    token: Token
    taip: TupleType
    copy_space_offset: int

@dataclass
class TupleUnpackWord:
    token: Token
    item: List[Type]
    copy_space_offset: int

IntrinsicWord = IntrinsicAdd | IntrinsicSub | IntrinsicEqual | IntrinsicNotEqual | IntrinsicAnd | IntrinsicDrop | IntrinsicLoad8 | IntrinsicStore8 | IntrinsicGreaterEq | IntrinsicLessEq | IntrinsicMul | IntrinsicMod | IntrinsicDiv | IntrinsicGreater | IntrinsicLess | IntrinsicFlip | IntrinsicRotl | IntrinsicRotr | IntrinsicOr | IntrinsicStore | IntrinsicMemCopy | IntrinsicMemFill | IntrinsicMemGrow | IntrinsicNot | IntrinsicUninit | IntrinsicSetStackSize

Word = NumberWord | StringWord | CallWord | GetWord | InitWord | CastWord | SetWord | LoadWord | IntrinsicWord | IfWord | RefWord | IndirectCallWord | StoreWord | FunRefWord | LoopWord | BreakWord | SizeofWord | BlockWord | GetFieldWord | StructWord | StructFieldInitWord | UnnamedStructWord | VariantWord | MatchWord | InitWord | TupleMakeWord | TupleUnpackWord

@dataclass
class Module:
    id: int
    type_definitions: Dict[int, List[TypeDefinition]]
    externs: Dict[int, Extern]
    globals: List[Global]
    functions: Dict[int, Function]
    data: bytes

@dataclass
class Monomizer:
    modules: Dict[int, ResolvedModule]
    type_definitions: Dict[ResolvedCustomTypeHandle, List[Tuple[List[Type], TypeDefinition]]] = field(default_factory=dict)
    externs: Dict[ResolvedFunctionHandle, Extern] = field(default_factory=dict)
    globals: Dict[GlobalId, Global] = field(default_factory=dict)
    functions: Dict[ResolvedFunctionHandle, Function] = field(default_factory=dict)
    signatures: Dict[ResolvedFunctionHandle, FunctionSignature | List[FunctionSignature]] = field(default_factory=dict)
    function_table: Dict[FunctionHandle | ExternHandle, int] = field(default_factory=dict)

    def monomize(self) -> Tuple[Dict[FunctionHandle | ExternHandle, int], Dict[int, Module]]:
        self.externs = { ResolvedFunctionHandle(m, i): self.monomize_extern(f) for m,module in self.modules.items() for i,f in enumerate(module.functions) if isinstance(f, ResolvedExtern) }
        self.globals = { GlobalId(m, i): self.monomize_global(globl) for m,module in self.modules.items() for i,globl in enumerate(module.globals) }
        for id in sorted(self.modules):
            module = self.modules[id]
            for index, function in enumerate(module.functions):
                if isinstance(function, ResolvedExtern):
                    continue
                if function.signature.export_name is not None:
                    assert(len(function.signature.generic_parameters) == 0)
                    handle = ResolvedFunctionHandle(id, index)
                    self.monomize_function(handle, [])

        mono_modules = {}
        for module_id in self.modules:
            module = self.modules[module_id]
            externs: Dict[int, Extern] = { handle.index: extern for (handle,extern) in self.externs.items() if handle.module == module_id }
            globals: List[Global] = [globl for id, globl in self.globals.items() if id.module == module_id]
            type_definitions: Dict[int, List[TypeDefinition]] = { handle.index: [taip for _,taip in monomorphizations] for handle,monomorphizations in self.type_definitions.items() if handle.module == module_id }
            functions = { handle.index: function for handle,function in self.functions.items() if handle.module == module_id }
            mono_modules[module_id] = Module(module_id, type_definitions, externs, globals, functions, self.modules[module_id].data)
        return self.function_table, mono_modules

    def monomize_locals(self, locals: Dict[LocalId, ResolvedLocal], generics: List[Type]) -> Dict[LocalId, Local]:
        res: Dict[LocalId, Local] = {}
        for id, local in locals.items():
            taip = self.monomize_type(local.taip, generics)
            lives_in_memory = local.was_reffed or not taip.can_live_in_reg()
            if local.is_parameter:
                res[id] = ParameterLocal(local.name, taip, lives_in_memory)
            else:
                res[id] = InitLocal(local.name, taip, lives_in_memory)
            continue
        return res

    def monomize_concrete_signature(self, signature: ResolvedFunctionSignature) -> FunctionSignature:
        assert(len(signature.generic_parameters) == 0)
        return self.monomize_signature(signature, [])

    def monomize_function(self, function: ResolvedFunctionHandle, generics: List[Type]) -> ConcreteFunction:
        f = self.modules[function.module].functions[function.index]
        assert(isinstance(f, ResolvedFunction))
        if len(generics) == 0:
            assert(len(f.signature.generic_parameters) == 0)
        signature = self.monomize_signature(f.signature, generics)
        if len(f.signature.generic_parameters) == 0:
            self.signatures[function] = signature
            generic_index = None
        else:
            if function not in self.signatures:
                self.signatures[function] = []
            instances = self.signatures[function]
            assert(isinstance(instances, list))
            generic_index = len(instances)
            instances.append(signature)
        copy_space_offset = Ref(0)
        max_struct_ret_count = Ref(0)
        monomized_locals = self.monomize_locals(f.body.locals, generics)
        body = Body(
            self.monomize_words(
                f.body.words,
                generics,
                copy_space_offset,
                max_struct_ret_count,
                monomized_locals,
                None),
            copy_space_offset.value,
            max_struct_ret_count.value,
            monomized_locals)
        concrete_function = ConcreteFunction(signature, body)
        if len(f.signature.generic_parameters) == 0:
            assert(len(generics) == 0)
            assert(function not in self.functions)
            self.functions[function] = concrete_function
            return concrete_function
        assert(generic_index is not None)
        if function not in self.functions:
            self.functions[function] = GenericFunction({})
        generic_function = self.functions[function]
        assert(isinstance(generic_function, GenericFunction))
        assert(generic_index not in generic_function.instances)
        generic_function.instances[generic_index] = concrete_function
        return concrete_function

    def monomize_signature(self, signature: ResolvedFunctionSignature, generics: List[Type]) -> FunctionSignature:
        parameters = list(map(lambda t: self.monomize_named_type(t, generics), signature.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), signature.returns))
        return FunctionSignature(signature.export_name, signature.name, generics, parameters, returns)

    def monomize_global(self, globl: ResolvedGlobal) -> Global:
        return Global(self.monomize_named_type(globl.taip, []), globl.was_reffed)

    def monomize_words(self, words: List[ResolvedWord], generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], locals: Dict[LocalId, Local], struct_space: int | None) -> List[Word]:
        return list(map(lambda w: self.monomize_word(w, generics, copy_space_offset, max_struct_ret_count, locals, struct_space), words))

    def monomize_word(self, word: ResolvedWord, generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], locals: Dict[LocalId, Local], struct_space: int | None) -> Word:
        match word:
            case NumberWord():
                return word
            case StringWord():
                return word
            case ResolvedCallWord():
                return self.monomize_call_word(word, copy_space_offset, max_struct_ret_count, generics)
            case ResolvedIndirectCallWord(token, taip):
                monomized_function_taip = self.monomize_function_type(taip, generics)
                local_copy_space_offset = copy_space_offset.value
                copy_space = sum(taip.size() for taip in monomized_function_taip.returns if not taip.can_live_in_reg())
                copy_space_offset.value += copy_space
                if copy_space != 0:
                    max_struct_ret_count.value = max(max_struct_ret_count.value, len(monomized_function_taip.returns))
                return IndirectCallWord(token, monomized_function_taip, local_copy_space_offset)
            case ResolvedGetWord(token, local_id, var_taip, resolved_fields, taip):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                monomized_taip = self.monomize_type(taip, generics)
                if not monomized_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += monomized_taip.size()
                else:
                    offset = None
                lives_in_memory = self.does_var_live_in_memory(local_id, locals)
                monomized_var_taip = self.monomize_type(var_taip, generics)
                loads = determine_loads(fields, just_ref=False, base_in_mem=lives_in_memory)
                target_taip = fields[-1].target_taip if len(fields) != 0 else monomized_var_taip
                return GetWord(token, local_id, target_taip, loads, offset, lives_in_memory)
            case ResolvedInitWord(token, local_id, taip):
                return InitWord(token, local_id, self.monomize_type(taip, generics), self.does_var_live_in_memory(local_id, locals))
            case ResolvedSetWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                lives_in_memory = self.does_var_live_in_memory(local_id, locals)
                target_lives_in_memory = lives_in_memory or any(isinstance(field.source_taip, ResolvedPtrType) for field in resolved_fields)
                monomized_var_taip = self.lookup_var_taip(local_id, locals)
                loads = determine_loads(fields, just_ref=target_lives_in_memory, base_in_mem=lives_in_memory)
                monomized_taip = fields[-1].target_taip if len(fields) != 0 else monomized_var_taip
                return SetWord(token, local_id, monomized_taip, loads, target_lives_in_memory)
            case ResolvedRefWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                loads = determine_loads(fields, just_ref=True)
                return RefWord(token, local_id, loads)
            case ResolvedStoreWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                loads = determine_loads(fields)
                monomized_taip = fields[-1].target_taip if len(fields) != 0 else self.lookup_var_taip(local_id, locals)
                if len(fields) == 0 and isinstance(local_id, LocalId):
                    assert(isinstance(monomized_taip, PtrType))
                    monomized_taip = monomized_taip.child
                return StoreWord(token, local_id, monomized_taip, loads)
            case ResolvedIntrinsicAdd(token, taip):
                return IntrinsicAdd(token, self.monomize_addable_type(taip, generics))
            case ResolvedIntrinsicSub(token, taip):
                return IntrinsicSub(token, self.monomize_addable_type(taip, generics))
            case ResolvedIntrinsicMul(token, taip):
                return IntrinsicMul(token, taip)
            case ResolvedIntrinsicMod(token, taip):
                return IntrinsicMod(token, taip)
            case ResolvedIntrinsicDiv(token, taip):
                return IntrinsicDiv(token, taip)
            case ResolvedIntrinsicEqual(token, taip):
                return IntrinsicEqual(token, self.monomize_type(taip, generics))
            case ResolvedIntrinsicNotEqual(token, taip):
                return IntrinsicNotEqual(token, self.monomize_type(taip, generics))
            case ResolvedIntrinsicGreaterEq(token, taip):
                return IntrinsicGreaterEq(token, taip)
            case ResolvedIntrinsicGreater(token, taip):
                return IntrinsicGreater(token, taip)
            case ResolvedIntrinsicLess(token, taip):
                return IntrinsicLess(token, taip)
            case ResolvedIntrinsicLessEq(token, taip):
                return IntrinsicLessEq(token, taip)
            case ResolvedIntrinsicAnd(token, taip):
                return IntrinsicAnd(token, taip)
            case ResolvedIntrinsicNot(token, taip):
                return IntrinsicNot(token, taip)
            case IntrinsicDrop():
                return word
            case BreakWord():
                return word
            case ResolvedIntrinsicFlip(token, lower, upper):
                return IntrinsicFlip(token, self.monomize_type(lower, generics), self.monomize_type(upper, generics))
            case IntrinsicLoad8():
                return word
            case IntrinsicStore8():
                return word
            case ResolvedIntrinsicRotl(token, taip):
                return IntrinsicRotl(token, self.monomize_type(taip, generics))
            case ResolvedIntrinsicRotr(token, taip):
                return IntrinsicRotr(token, self.monomize_type(taip, generics))
            case ResolvedIntrinsicOr(token, taip):
                return IntrinsicOr(token, self.monomize_type(taip, generics))
            case ResolvedIntrinsicStore(token, taip):
                return IntrinsicStore(token, self.monomize_type(taip, generics))
            case IntrinsicMemCopy():
                return word
            case IntrinsicMemFill():
                return word
            case IntrinsicMemGrow():
                return word
            case IntrinsicSetStackSize():
                return word
            case ResolvedIntrinsicUninit(token, taip):
                monomized_taip = self.monomize_type(taip, generics)
                offset = copy_space_offset.value
                if not monomized_taip.can_live_in_reg():
                    copy_space_offset.value += monomized_taip.size()
                return IntrinsicUninit(token, monomized_taip, offset)
            case ResolvedLoadWord(token, taip):
                monomized_taip = self.monomize_type(taip, generics)
                if not monomized_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += monomized_taip.size()
                else:
                    offset = None
                return LoadWord(token, monomized_taip, offset)
            case ResolvedCastWord(token, source, taip):
                return CastWord(token, self.monomize_type(source, generics), self.monomize_type(taip, generics))
            case ResolvedIfWord(token, resolved_parameters, resolved_returns, resolved_if_words, resolved_else_words, diverges):
                if_words = self.monomize_words(resolved_if_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                else_words = self.monomize_words(resolved_else_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return IfWord(token, parameters, returns, if_words, else_words, diverges)
            case ResolvedFunRefWord(call):
                # monomize_call_word increments the copy_space, but if we're just taking the pointer
                # of the function, then we're not actually calling it and no space should be allocated.
                cso = copy_space_offset.value
                msrc = max_struct_ret_count.value
                call_word = self.monomize_call_word(call, copy_space_offset, max_struct_ret_count, generics)
                # So restore the previous values of copy_space_offset and max_struct_ret_count afterwards.
                # TODO: extract those parts of monomize_call_word which are common to both actual calls and just FunRefs.
                copy_space_offset.value = cso
                max_struct_ret_count.value = msrc
                table_index = self.insert_function_into_table(call_word.function)
                return FunRefWord(call_word, table_index)
            case ResolvedLoopWord(token, resolved_words, resolved_parameters, resolved_returns, diverges):
                words = self.monomize_words(resolved_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return LoopWord(token, words, parameters, returns, diverges)
            case ResolvedSizeofWord(token, taip):
                return SizeofWord(token, self.monomize_type(taip, generics))
            case ResolvedBlockWord(token, resolved_words, resolved_parameters, resolved_returns):
                words = self.monomize_words(resolved_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return BlockWord(token, words, parameters, returns)
            case ResolvedGetFieldWord(token, resolved_fields, on_ptr):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                target_taip = fields[-1].target_taip
                offset = None
                if not on_ptr and not target_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += target_taip.size()
                loads = determine_loads(fields, just_ref=on_ptr)
                return GetFieldWord(token, target_taip, loads, on_ptr, offset)
            case ResolvedStructWord(token, taip, resolved_words):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                copy_space_offset.value += monomized_taip.size()
                words = self.monomize_words(resolved_words, generics, copy_space_offset, max_struct_ret_count, locals, offset)
                return StructWord(token, monomized_taip, words, offset)
            case ResolvedUnnamedStructWord(token, taip):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                if not monomized_taip.can_live_in_reg():
                    copy_space_offset.value += monomized_taip.size()
                return UnnamedStructWord(token, monomized_taip, offset)
            case ResolvedStructFieldInitWord(token, struct, generic_arguments, taip):
                generics_here = list(map(lambda t: self.monomize_type(t, generics), generic_arguments))
                (_,monomized_struct) = self.monomize_struct(struct, generics_here)
                if isinstance(monomized_struct, Variant):
                    assert(False)
                assert(struct_space is not None)
                field_copy_space_offset: int = struct_space
                for i,field in enumerate(monomized_struct.fields.get()):
                    if field.name.lexeme == token.lexeme:
                        field_copy_space_offset += monomized_struct.field_offset(i)
                        break
                return StructFieldInitWord(token, self.monomize_type(taip, generics), field_copy_space_offset)
            case ResolvedVariantWord(token, case, resolved_variant_type):
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                (variant_handle, variant) = self.monomize_struct(resolved_variant_type.struct, this_generics)
                offset = copy_space_offset.value
                if variant.size() > 8:
                    copy_space_offset.value += variant.size()
                return VariantWord(token, case, variant_handle, offset)
            case ResolvedMatchWord(token, resolved_variant_type, by_ref, cases, default_case, resolved_parameters, resolved_returns):
                monomized_cases: List[MatchCase] = []
                for resolved_case in cases:
                    words = self.monomize_words(resolved_case.words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                    monomized_cases.append(MatchCase(resolved_case.tag, words))
                monomized_default_case = None if default_case is None else self.monomize_words(default_case, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                monomized_variant = self.monomize_struct(resolved_variant_type.struct, this_generics)[0]
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return MatchWord(token, monomized_variant, by_ref, monomized_cases, monomized_default_case, parameters, returns)
            case ResolvedTupleMakeWord(token, tupl):
                offset = copy_space_offset.value
                mono_tupl = TupleType(tupl.token, list(map(lambda t: self.monomize_type(t, generics), tupl.items)))
                offset = copy_space_offset.value
                if mono_tupl.size() > 4:
                    copy_space_offset.value += mono_tupl.size()
                return TupleMakeWord(token, mono_tupl, offset)
            case ResolvedTupleUnpackWord(token, items):
                offset = copy_space_offset.value
                mono_items = list(map(lambda t: self.monomize_type(t, generics), items))
                copy_space_offset.value += sum(item.size() for item in mono_items if not item.can_live_in_reg())
                return TupleUnpackWord(token, mono_items, offset)
            case other:
                assert_never(other)

    def lookup_var_taip(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, Local]) -> Type:
        if isinstance(local_id, LocalId):
            return locals[local_id].taip
        return self.globals[local_id].taip.taip

    def does_var_live_in_memory(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, Local]) -> bool:
        if isinstance(local_id, LocalId):
            return locals[local_id].lives_in_memory()
        globl = self.globals[local_id]
        return globl.was_reffed or not globl.taip.taip.can_live_in_reg()

    def insert_function_into_table(self, function: FunctionHandle | ExternHandle) -> int:
        if function not in self.function_table:
            self.function_table[function] = len(self.function_table)
        return self.function_table[function]

    def monomize_field_accesses(self, fields: List[ResolvedFieldAccess], generics: List[Type]) -> List[FieldAccess]:
        if len(fields) == 0:
            return []

        field = fields[0]

        if isinstance(field.source_taip, ResolvedStructType):
            source_taip: PtrType | StructType = self.monomize_struct_type(field.source_taip, generics)
            resolved_struct = field.source_taip.struct
            generic_arguments = field.source_taip.generic_arguments
        else:
            assert(isinstance(field.source_taip.child, ResolvedStructType))
            source_taip = PtrType(self.monomize_type(field.source_taip.child, generics))
            resolved_struct = field.source_taip.child.struct
            generic_arguments = field.source_taip.child.generic_arguments
        [_, struct] = self.monomize_struct(resolved_struct, list(map(lambda t: self.monomize_type(t, generics), generic_arguments)))
        assert(not isinstance(struct, Variant))
        target_taip = self.monomize_type(field.target_taip, struct.generic_parameters)
        offset = struct.field_offset(field.field_index)
        return [FieldAccess(field.name, source_taip, target_taip, offset)] + self.monomize_field_accesses(fields[1:], struct.generic_parameters)

    def monomize_call_word(self, word: ResolvedCallWord, copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], generics: List[Type]) -> CallWord:
        if word.function in self.externs:
            signature = self.externs[word.function].signature
            offset = copy_space_offset.value
            copy_space = sum(taip.size() for taip in signature.returns if isinstance(taip, StructType))
            max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
            copy_space_offset.value += copy_space
            return CallWord(word.name, ExternHandle(word.function.module, word.function.index), offset)
        generics_here = list(map(lambda t: self.monomize_type(t, generics), word.generic_arguments))
        if word.function in self.signatures:
            signatures = self.signatures[word.function]
            if isinstance(signatures, FunctionSignature):
                signature = signatures
                assert(len(word.generic_arguments) == 0)
                offset = copy_space_offset.value
                copy_space = sum(taip.size() for taip in signature.returns if not taip.can_live_in_reg())
                max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
                copy_space_offset.value += copy_space
                return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, None), offset)
            for instance_index, signature in enumerate(signatures):
                if types_eq(signature.generic_arguments, generics_here):
                    offset = copy_space_offset.value
                    copy_space = sum(taip.size() for taip in signature.returns if not taip.can_live_in_reg())
                    max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
                    copy_space_offset.value += copy_space
                    return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, instance_index), offset)
        self.monomize_function(word.function, generics_here)
        return self.monomize_call_word(word, copy_space_offset, max_struct_ret_count, generics) # the function instance should now exist, try monomorphizing this CallWord again

    def lookup_struct(self, struct: ResolvedCustomTypeHandle, generics: List[Type]) -> Tuple[StructHandle, TypeDefinition] | None:
        if struct not in self.type_definitions:
            return None
        for instance_index, (genics, instance) in enumerate(self.type_definitions[struct]):
            if types_eq(genics, generics):
                return StructHandle(struct.module, struct.index, instance_index), instance
        return None

    def add_struct(self, handle: ResolvedCustomTypeHandle, taip: TypeDefinition, generics: List[Type]) -> StructHandle:
        if handle not in self.type_definitions:
            self.type_definitions[handle] = []
        instance_index = len(self.type_definitions[handle])
        self.type_definitions[handle].append((generics, taip))
        return StructHandle(handle.module, handle.index, instance_index)

    def monomize_struct(self, struct: ResolvedCustomTypeHandle, generics: List[Type]) -> Tuple[StructHandle, TypeDefinition]:
        handle_and_instance = self.lookup_struct(struct, generics)
        if handle_and_instance is not None:
            return handle_and_instance
        s = self.modules[struct.module].type_definitions[struct.index]
        if isinstance(s, ResolvedVariant):
            def cases() -> List[VariantCase]:
                return [VariantCase(c.name, self.monomize_type(c.taip, generics) if c.taip is not None else None) for c in s.cases]

            variant_instance = Variant(s.name, Lazy(cases), generics)
            handle = self.add_struct(struct, variant_instance, generics)
            return handle, variant_instance

        def fields() -> List[NamedType]:
            return list(map(lambda t: self.monomize_named_type(t, generics), s.fields))

        struct_instance = Struct(s.name, Lazy(fields), generics)
        handle = self.add_struct(struct, struct_instance, generics)
        return handle, struct_instance

    def monomize_named_type(self, taip: ResolvedNamedType, generics: List[Type]) -> NamedType:
        return NamedType(taip.name, self.monomize_type(taip.taip, generics))

    def monomize_type(self, taip: ResolvedType, generics: List[Type]) -> Type:
        match taip:
            case PrimitiveType():
                return taip
            case ResolvedPtrType():
                return PtrType(self.monomize_type(taip.child, generics))
            case GenericType(_, generic_index):
                return generics[generic_index]
            case ResolvedStructType():
                return self.monomize_struct_type(taip, generics)
            case ResolvedFunctionType():
                return self.monomize_function_type(taip, generics)
            case ResolvedTupleType(token, items):
                return TupleType(token, list(map(lambda item: self.monomize_type(item, generics), items)))
            case other:
                assert_never(other)

    def monomize_addable_type(self, taip: ResolvedPtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64], generics: List[Type]) -> PtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]:
        match taip:
            case PrimitiveType.I32 | PrimitiveType.I64:
                return taip
            case ResolvedPtrType():
                return PtrType(self.monomize_type(taip.child, generics))
            case other:
                assert_never(other)

    def monomize_struct_type(self, taip: ResolvedStructType, generics: List[Type]) -> StructType:
        this_generics = list(map(lambda t: self.monomize_type(t, generics), taip.generic_arguments))
        handle,struct = self.monomize_struct(taip.struct, this_generics)
        return StructType(taip.name, handle, Lazy(lambda: struct.size()))

    def monomize_function_type(self, taip: ResolvedFunctionType, generics: List[Type]) -> FunctionType:
        parameters = list(map(lambda t: self.monomize_type(t, generics), taip.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), taip.returns))
        return FunctionType(taip.token, parameters, returns)

    def monomize_extern(self, extern: ResolvedExtern) -> Extern:
        signature = self.monomize_concrete_signature(extern.signature)
        return Extern(extern.module, extern.name, signature)

def align_to(n: int, to: int) -> int:
    return n + (to - (n % to)) * ((n % to) > 0)


class DetermineLoadsToValueTests(unittest.TestCase):
    def test_no_fields_returns_empty(self) -> None:
        loads = determine_loads([])
        self.assertTrue(len(loads) == 0)

    def test_by_value_on_struct(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 12), PrimitiveType.I32, 4)
        ])
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == OffsetLoad(4, PrimitiveType.I32))

    def test_by_value_on_struct_packed(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 8), PrimitiveType.I32, 0)
        ])
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == BitShiftLoad.Lower32)

    def test_packed_value_on_struct_in_mem(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 8), PrimitiveType.I32, 0)
        ], base_in_mem=True)
        self.assertEqual(loads, [OffsetLoad(0, PrimitiveType.I32)])

    def test_by_value_on_struct_packed_small_noop(self) -> None:
        loads = determine_loads([
            FieldAccess(Token.dummy("x"), StructType.dummy("Foo", 4), PrimitiveType.I32, 0)
        ])
        self.assertEqual(loads, [])

    def test_by_value_on_nested_struct(self) -> None:
        foo = StructType.dummy("Foo", 12)
        v2 = StructType.dummy("V2", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("v"), foo, v2               , 4),
            FieldAccess(Token.dummy("x"), v2 , PrimitiveType.I32, 0),
        ])
        self.assertEqual(loads, [OffsetLoad(4, PrimitiveType.I32)])

    def test_by_value_through_ptr(self) -> None:
        node = StructType.dummy("Node", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , node         , PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), PrimitiveType.I32, 0),
        ])
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == OffsetLoad(4, PrimitiveType.I32))

    def test_by_value_through_two_ptrs(self) -> None:
        node = StructType.dummy("Node", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , node         , PtrType(node)    , 4),
            FieldAccess(Token.dummy("next") , PtrType(node), PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), PrimitiveType.I32, 0),
        ])
        self.assertEqual(loads, [
            OffsetLoad(8, PtrType(node)),
            OffsetLoad(0, PrimitiveType.I32)])

    def test_get_sub_struct_by_value(self) -> None:
        bar = StructType.dummy("Bar", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("v"), StructType.dummy("Foo", 16), bar, 4)
        ])
        self.assertEqual(loads, [OffsetLoad(4, bar)])

    def test_get_subsub_struct_by_value(self) -> None:
        bar = StructType.dummy("Bar", 12)
        inner = StructType.dummy("BarInner", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("v")    , StructType.dummy("Foo", 16), bar  , 4),
            FieldAccess(Token.dummy("inner"), bar                        , inner, 0),
        ])
        self.assertEqual(loads, [OffsetLoad(4, inner)])

    def test_by_ref_get_on_large_value(self) -> None:
        node = StructType.dummy("Node", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , PtrType(node), PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), PrimitiveType.I32, 0),
        ])
        self.assertEqual(loads, [
            OffsetLoad(4, PtrType(node)),
            OffsetLoad(0, PrimitiveType.I32)])

    def test_by_ref_get_on_packed_value(self) -> None:
        node = StructType.dummy("Node", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , PtrType(node), PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), PrimitiveType.I32, 0),
        ])
        self.assertTrue(len(loads) == 2)
        self.assertTrue(loads[0] == OffsetLoad(4, PtrType(node)))
        self.assertTrue(loads[1] == OffsetLoad(0, PrimitiveType.I32))

    def test_get_through_bitshift(self) -> None:
        node = StructType.dummy("Node", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("next") , node         , PtrType(node)    , 4),
            FieldAccess(Token.dummy("value"), PtrType(node), PrimitiveType.I32, 0),
        ])
        self.assertTrue(len(loads) == 2)
        self.assertTrue(loads[0] == BitShiftLoad.Upper32)
        self.assertTrue(loads[1] == OffsetLoad(0, PrimitiveType.I32))

    def test_ref_field_in_big_struct(self) -> None:
        ctx = StructType.dummy("Ctx", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("counter"), ctx, PrimitiveType.I32, 8),
        ], just_ref=True)
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == Offset(8))

    def test_ref_subfield_in_big_struct(self) -> None:
        ctx = StructType.dummy("Ctx", 20)
        word_ctx = StructType.dummy("WordCtx", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("word-ctx"), ctx              , word_ctx         , 4),
            FieldAccess(Token.dummy("counter") , PtrType(word_ctx), PrimitiveType.I32, 8),
        ], just_ref=True)
        self.assertTrue(len(loads) == 1)
        self.assertTrue(loads[0] == Offset(12))

    def test_ref_field_in_big_struct_at_offset_0(self) -> None:
        ctx = StructType.dummy("Ctx", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("counter"), ctx, PrimitiveType.I32, 0),
        ], just_ref=True)
        self.assertTrue(len(loads) == 0)

    def test_ignored_wrapper_struct(self) -> None:
        allocator = StructType.dummy("PageAllocator", 4)
        page = StructType.dummy("Page", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("free-list"), allocator    , PtrType(page)    , 0),
            FieldAccess(Token.dummy("foo")      , PtrType(page), PrimitiveType.I32, 4),
        ], just_ref=True)
        self.assertEqual(loads, [Offset(4)])

    def test_no_unnecessary_bitshift_on_ptr(self) -> None:
        foo = StructType.dummy("Foo", 12)
        v2 = StructType.dummy("V2", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("v")  , foo, v2               , 8),
            FieldAccess(Token.dummy("foo"), v2 , PrimitiveType.I32, 4),
        ])
        self.assertEqual(loads, [OffsetLoad(12, PrimitiveType.I32)])

    def test_set_field_on_ptr(self) -> None:
        small = StructType.dummy("Small", 4)
        loads = determine_loads([
            FieldAccess(Token.dummy("value"), PtrType(small), PrimitiveType.I32, 0),
        ], just_ref=True)
        self.assertEqual(loads, [])

    def test_get_value_on_packed_but_reffed_struct(self) -> None:
        prestat = StructType.dummy("Prestat", 8)
        loads = determine_loads([
            FieldAccess(Token.dummy("path_len"), prestat, PrimitiveType.I32, 4)
        ], base_in_mem=True)
        self.assertEqual(loads, [OffsetLoad(4, PrimitiveType.I32)])

    def test_get_struct_through_ptr(self) -> None:
        named_type = StructType.dummy("NamedType", 16)
        taip = StructType.dummy("Type", 12)
        loads = determine_loads([
            FieldAccess(Token.dummy("type"), PtrType(named_type), taip, 4),
        ])
        self.assertEqual(loads, [OffsetLoad(4, taip)])

    def test_packed_value_field_through_ptr(self) -> None:
        token = StructType.dummy("Token", 8)
        immediate_string = StructType.dummy("ImmediateString", 4)
        loads = determine_loads([
            FieldAccess(Token.dummy("lexeme"), PtrType(token), PtrType(immediate_string), 4),
            FieldAccess(Token.dummy("len"), PtrType(immediate_string), PrimitiveType.I32, 0),
        ])
        self.assertEqual(loads, [
            OffsetLoad(4, PtrType(immediate_string)),
            OffsetLoad(0, PrimitiveType.I32)])

    def test_set_field_through_bitshift(self) -> None:
        token = StructType.dummy("Token", 8)
        immediate_string = StructType.dummy("ImmediateString", 4)
        loads = determine_loads([
            FieldAccess(Token.dummy("lexeme"), token, PtrType(immediate_string), 4),
            FieldAccess(Token.dummy("len"), PtrType(immediate_string), PrimitiveType.I32, 0),
        ], just_ref=True)
        self.assertEqual(loads, [BitShiftLoad.Upper32])

def merge_loads(loads: List[Load]) -> List[Load]:
    if len(loads) <= 1:
        return loads
    if isinstance(loads[0], OffsetLoad) and isinstance(loads[1], BitShiftLoad):
        if loads[1] == BitShiftLoad.Lower32:
            return [OffsetLoad(loads[0].offset + 0, PrimitiveType.I32)] + loads[2:]
        if loads[0] == BitShiftLoad.Upper32:
            return [OffsetLoad(loads[0].offset + 4, PrimitiveType.I32)] + loads[2:]
    if isinstance(loads[0], Offset) and isinstance(loads[1], Offset):
        return [Offset(loads[0].offset + loads[1].offset)] + loads[2:]
    if isinstance(loads[0], Offset) and isinstance(loads[1], OffsetLoad):
        return [OffsetLoad(loads[0].offset + loads[1].offset, loads[1].taip)] + loads[2:]
    return loads

# Returns the loads necessary to get the value of the final field on the stack.
def determine_loads(fields: List[FieldAccess], just_ref: bool = False, base_in_mem: bool = False) -> List[Load]:
    if len(fields) == 0:
        return []
    field = fields[0]
    if isinstance(field.source_taip, StructType):
        offset = field.offset
        if base_in_mem or field.source_taip.size() > 8:
            if len(fields) > 1 or just_ref:
                load: Load = Offset(offset)
            else:
                load = OffsetLoad(offset, field.target_taip)
            if load == Offset(0):
                return determine_loads(fields[1:], just_ref, base_in_mem=True)
            return merge_loads([load] + determine_loads(fields[1:], just_ref, base_in_mem=True))

        if field.source_taip.size() > 4: # source_taip is between >=4 and <=8 bytes
            assert(offset == 4 or offset == 0) # alternative is TODO
            if offset == 0:
               load = BitShiftLoad.Lower32
            elif offset == 4:
               load = BitShiftLoad.Upper32
            return merge_loads([load] + determine_loads(fields[1:], just_ref, base_in_mem))

        assert(field.source_taip.size() == 4) # alternative is TODO
        return determine_loads(fields[1:], just_ref, base_in_mem)

    if isinstance(field.source_taip, PtrType):
        if (just_ref and len(fields) == 1) or (not field.target_taip.can_live_in_reg() and len(fields) != 1):
            # Instead of actually loading the value, we just ref it, since this is
            # the last field access in the chain and `just_ref` is set.
            if field.offset == 0:
                return determine_loads(fields[1:], just_ref, base_in_mem)
            return merge_loads([Offset(field.offset)] + determine_loads(fields[1:], just_ref, base_in_mem))

        return merge_loads([OffsetLoad(field.offset, field.target_taip)] + determine_loads(fields[1:], just_ref, base_in_mem))

    assert_never(field.source_taip)

@dataclass
class WatGenerator:
    modules: Dict[int, Module]
    function_table: Dict[FunctionHandle | ExternHandle, int]
    guard_stack: bool
    chunks: List[str] = field(default_factory=list)
    indentation: int = 0
    globals: Dict[GlobalId, Global]= field(default_factory=dict)
    module_data_offsets: Dict[int, int] = field(default_factory=dict)

    pack_i32s_used: bool = False
    unpack_i32s_used: bool = False
    flip_i64_i32_used: bool = False
    flip_i32_i64_used: bool = False
    flip_i64_i64_used: bool = False
    dup_i64_used: bool = False

    def write(self, s: str) -> None:
        self.chunks.append(s)

    def write_indent(self) -> None:
        self.chunks.append("\t" * self.indentation)

    def write_line(self, line: str) -> None:
        self.write_indent()
        self.write(line)
        self.write("\n")

    def indent(self) -> None:
        self.indentation += 1

    def dedent(self) -> None:
        self.indentation -= 1

    def lookup_type_definition(self, handle: StructHandle) -> TypeDefinition:
        return self.modules[handle.module].type_definitions[handle.index][handle.instance]

    def lookup_extern(self, handle: ExternHandle) -> Extern:
        return self.modules[handle.module].externs[handle.index]

    def lookup_function(self, handle: FunctionHandle) -> ConcreteFunction:
        function = self.modules[handle.module].functions[handle.index]
        if isinstance(function, GenericFunction):
            assert(handle.instance is not None)
            return function.instances[handle.instance]
        return function

    def write_wat_module(self) -> str:
        assert(len(self.chunks) == 0)
        self.write_line("(module")
        self.indent()
        for module in self.modules.values():
            for extern in module.externs.values():
                self.write_extern(module.id, extern)
                self.write("\n")
            for i, globl in enumerate(module.globals):
                self.globals[GlobalId(module.id, i)] = globl

        self.write_line("(memory 1 65536)")
        self.write_line("(export \"memory\" (memory 0))")

        all_data: bytes = b""
        for id in sorted(self.modules):
            self.module_data_offsets[id] = len(all_data)
            all_data += self.modules[id].data

        self.write_function_table()

        data_end = align_to(len(all_data), 4)
        global_mem = self.write_globals(data_end)
        stack_start = align_to(global_mem, 4)
        self.write_line(f"(global $stac:k (mut i32) (i32.const {stack_start}))")
        if self.guard_stack:
            self.write_line("(global $stack-siz:e (mut i32) (i32.const 65536))")

        self.write_data(all_data)

        for module_id in sorted(self.modules):
            module = self.modules[module_id]
            for function in sorted(module.functions.keys()):
                self.write_function(module_id, module.functions[function])

        self.write_intrinsics()

        self.dedent()
        self.write(")")
        return ''.join(self.chunks)

    def write_function(self, module: int, function: Function, instance_id: int | None = None) -> None:
        if isinstance(function, GenericFunction):
            for (id, instance) in function.instances.items():
                self.write_function(module, instance, id)
            return
        self.write_indent()
        self.write("(")
        self.write_signature(module, function.signature, instance_id, function.body.locals)
        if len(function.signature.generic_arguments) > 0:
            self.write(" ;;")
            for taip in function.signature.generic_arguments:
                self.write(" ")
                self.write_type_human(taip)
        self.write("\n")
        self.indent()
        self.write_locals(function.body)
        for i in range(0, function.body.max_struct_ret_count):
            self.write_indent()
            self.write(f"(local $s{i}:a i32)\n")
        if function.body.locals_copy_space != 0:
            self.write_indent()
            self.write("(local $locl-copy-spac:e i32)\n")

        uses_stack = function.body.locals_copy_space != 0 or any(local.lives_in_memory() for local in function.body.locals.values())
        if uses_stack:
            self.write_indent()
            self.write("(local $stac:k i32)\n")
            self.write_indent()
            self.write("global.get $stac:k local.set $stac:k\n")

        if function.body.locals_copy_space != 0:
            self.write_mem("locl-copy-spac:e", function.body.locals_copy_space, 0, 0)
        self.write_structs(function.body.locals)
        if uses_stack and self.guard_stack:
            self.write_line("call $stack-overflow-guar:d")
        self.write_body(module, function.body)
        if uses_stack:
            self.write_line("local.get $stac:k global.set $stac:k")
        self.dedent()
        self.write_line(")")

    def write_mem(self, name: str, size: int, scope: int, shadow: int) -> None:
        self.write_indent()
        self.write(f"global.get $stac:k global.get $stac:k i32.const {align_to(size, 4)} i32.add global.set $stac:k local.set ${name}")
        if scope != 0 or shadow != 0:
            self.write(f":{scope}:{shadow}")
        self.write("\n")

    def write_structs(self, locals: Dict[LocalId, Local]) -> None:
        for local_id, local in locals.items():
            if not isinstance(local, ParameterLocal) and local.lives_in_memory():
                self.write_mem(local.name.lexeme, local.taip.size(), local_id.scope, local_id.shadow)
            if isinstance(local, ParameterLocal) and local.needs_moved_into_memory():
                self.write_indent()
                self.write("global.get $stac:k global.get $stac:k local.get $")
                if not local.can_be_abused_as_ref():
                    self.write("v:")
                self.write(f"{local.name.lexeme} ")
                self.write_type(local.taip)
                self.write(f".store local.tee ${local.name.lexeme} i32.const {local.taip.size()} i32.add global.set $stac:k\n")

    def write_locals(self, body: Body) -> None:
        for local_id, local in body.locals.items():
            if isinstance(local, ParameterLocal):
                if local.needs_moved_into_memory() and not local.can_be_abused_as_ref():
                    self.write_line(f"(local ${local.name.lexeme} i32)")
                continue
            local = body.locals[local_id]
            self.write_indent()
            self.write(f"(local ${local.name.lexeme}")
            if local_id.scope != 0 or local_id.shadow != 0:
                self.write(f":{local_id.scope}:{local_id.shadow}")
            self.write(" ")
            if local.lives_in_memory():
                self.write("i32")
            else:
                self.write_type(local.taip)
            self.write(")\n")

    def write_body(self, module: int, body: Body) -> None:
        self.write_words(module, { id: local.name.lexeme for id, local in body.locals.items() }, body.words)

    def write_words(self, module: int, locals: Dict[LocalId, str], words: List[Word]) -> None:
        for word in words:
            self.write_word(module, locals, word)

    def write_local_ident(self, name: str, local: LocalId) -> None:
        if local.scope != 0 or local.shadow != 0:
            self.write(f"${name}:{local.scope}:{local.shadow}")
        else:
            self.write(f"${name}")

    def write_word(self, module: int, locals: Dict[LocalId, str], word: Word):
        match word:
            case NumberWord(token):
                self.write_line(f"i32.const {token.lexeme}")
            case GetWord(token, local_id, target_taip, loads, copy_space_offset, var_lives_in_memory):
                self.write_indent()
                if not target_taip.can_live_in_reg():
                    # set up the address to store the result in
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                else:
                    self.write("local.get ")
                    self.write_local_ident(token.lexeme, local_id)
                # at this point, either the value itself or a pointer to it is on the stack
                for i, load in enumerate(loads):
                    self.write(" ")
                    self.write(str(load))
                if len(loads) == 0:
                    if target_taip.can_live_in_reg():
                        if var_lives_in_memory:
                            self.write(" ")
                            self.write_type(target_taip)
                            self.write(".load\n")
                            return
                    else:
                        self.write(f" i32.const {target_taip.size()} memory.copy")
                self.write("\n")
            case GetFieldWord(token, target_taip, loads, on_ptr, copy_space_offset):
                if len(loads) == 0:
                    self.write_line(";; GetField was no-op")
                    return
                self.write_indent()
                if not on_ptr and not target_taip.can_live_in_reg():
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                for load in loads:
                    self.write(str(load))
                self.write("\n")
            case RefWord(token, local_id, loads):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                if isinstance(local_id, LocalId):
                    self.write("local.get ")
                    self.write_local_ident(token.lexeme, local_id)
                for i, load in enumerate(loads):
                    self.write(f" {load}")
                self.write("\n")
            case SetWord(token, local_id, target_taip, loads, target_lives_in_memory):
                self.write_set(local_id, locals, target_lives_in_memory, target_taip, loads)
            case InitWord(name, local_id, taip, var_lives_in_memory):
                self.write_set(local_id, locals, var_lives_in_memory, taip, [])
            case CallWord(name, function_handle, return_space_offset):
                self.write_indent()
                match function_handle:
                    case ExternHandle():
                        extern = self.lookup_extern(function_handle)
                        signature = extern.signature
                        self.write(f"call ${function_handle.module}:{name.lexeme}")
                    case FunctionHandle():
                        function = self.lookup_function(function_handle)
                        signature = function.signature
                        self.write(f"call ${function_handle.module}:{function.signature.name.lexeme}")
                        if function_handle.instance is not None and function_handle.instance != 0:
                            self.write(f":{function_handle.instance}")
                    case other:
                        assert_never(other)
                self.write_return_struct_receiving(return_space_offset, signature.returns)
            case IndirectCallWord(token, taip, return_space_offset):
                self.write_indent()
                self.write("(call_indirect")
                self.write_parameters(taip.parameters)
                self.write_returns(taip.returns)
                self.write(")")
                self.write_return_struct_receiving(return_space_offset, taip.returns)
            case IntrinsicStore(token, taip):
                self.write_indent()
                self.write_store(taip)
                self.write("\n")
            case IntrinsicAdd(token, taip):
                if isinstance(taip, PtrType) or taip == PrimitiveType.I32:
                    self.write_line("i32.add")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.add")
                    return
                assert_never(taip)
            case IntrinsicSub(token, taip):
                if isinstance(taip, PtrType) or taip == PrimitiveType.I32:
                    self.write_line("i32.sub")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.sub")
                    return
                assert_never(taip)
            case IntrinsicMul(_, taip):
                self.write_line(f"{'i64' if taip == PrimitiveType.I64 else 'i32'}.mul")
            case IntrinsicDrop():
                self.write_line("drop")
            case IntrinsicOr(_, taip):
                self.write_indent()
                self.write_type(taip)
                self.write(".or\n")
            case IntrinsicEqual(_, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.eq")
                    return
                assert(taip.can_live_in_reg())
                self.write_line("i32.eq")
            case IntrinsicNotEqual(_, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.ne")
                    return
                assert(taip.can_live_in_reg())
                self.write_line("i32.ne")
            case IntrinsicGreaterEq(_, taip):
                if taip == PrimitiveType.I32:
                    self.write_line("i32.ge_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.ge_u")
                    return
                assert_never(taip)
            case IntrinsicGreater(_, taip):
                if taip == PrimitiveType.I32:
                    self.write_line("i32.gt_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.gt_u")
                    return
                assert_never(taip)
            case IntrinsicLessEq(_, taip):
                if taip == PrimitiveType.I32:
                    self.write_line("i32.le_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.le_u")
                    return
                assert_never(taip)
            case IntrinsicLess(_, taip):
                if taip == PrimitiveType.I32:
                    self.write_line("i32.lt_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.lt_u")
                    return
                assert_never(taip)
            case IntrinsicFlip(_, lower, upper):
                lower_type = "i32" if lower.size() <= 4 or lower.size() > 8 else "i64"
                upper_type = "i32" if upper.size() <= 4 or upper.size() > 8 else "i64"
                if lower_type == "i32" and upper_type == "i64":
                    self.flip_i32_i64_used = True
                    self.write_line("call $intrinsic:flip-i32-i64")
                    return
                if lower_type == "i64" and upper_type == "i32":
                    self.flip_i64_i32_used = True
                    self.write_line("call $intrinsic:flip-i64-i32")
                    return
                if lower_type == "i32":
                    self.write_line("call $intrinsic:flip")
                    return
                self.flip_i64_i64_used = True
                self.write_line("call $intrinsic:flip-i64-i64")
            case IntrinsicRotl(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.extend_i32_u i64.rotl")
                else:
                    self.write_line("i32.rotl")
            case IntrinsicRotr(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.extend_i32_u i64.rotr")
                else:
                    self.write_line("i32.rotr")
            case IntrinsicAnd(_, taip):
                if taip == PrimitiveType.I32 or taip == PrimitiveType.BOOL or taip == PrimitiveType.I8:
                    self.write_line("i32.and")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.and")
                    return
                assert_never(taip)
            case IntrinsicNot(_, taip):
                if taip == PrimitiveType.BOOL:
                    self.write_line("i32.const 1 i32.and i32.const 1 i32.xor i32.const 1 i32.and")
                    return
                if taip == PrimitiveType.I8:
                    self.write_line("i32.const 255 i32.and i32.const 255 i32.xor i32.const 255 i32.and")
                    return
                if taip == PrimitiveType.I32:
                    self.write_line("i32.const -1 i32.xor")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.const -1 i64.xor")
                    return
                assert_never(taip)
            case IntrinsicLoad8():
                self.write_line("i32.load8_u")
            case IntrinsicStore8():
                self.write_line("i32.store8")
            case IntrinsicMod(_, taip):
                match taip:
                    case PrimitiveType.I32:
                        self.write_line("i32.rem_u")
                    case PrimitiveType.I64:
                        self.write_line("i64.rem_u")
                    case _:
                        assert_never(taip)
            case IntrinsicDiv(_, taip):
                match taip:
                    case PrimitiveType.I32:
                        self.write_line("i32.div_u")
                    case PrimitiveType.I64:
                        self.write_line("i64.div_u")
                    case _:
                        assert_never(taip)
            case IntrinsicMemCopy():
                self.write_line("memory.copy")
            case IntrinsicMemFill():
                self.write_line("memory.fill")
            case IntrinsicMemGrow():
                self.write_line("memory.grow")
            case IntrinsicSetStackSize():
                if self.guard_stack:
                    self.write_line("global.set $stack-siz:e")
                else:
                    self.write_line("drop")
            case IntrinsicUninit(_, taip, copy_space_offset):
                if taip.size() <= 4:
                    self.write_line("i32.const 0")
                elif taip.size() <= 8:
                    self.write_line("i64.const 0")
                else:
                    self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add")
            case CastWord(_, source, taip):
                if (source == PrimitiveType.I32 or isinstance(source, PtrType)) and isinstance(taip, PtrType):
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if source == PrimitiveType.I32 and isinstance(taip, FunctionType):
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if source == PrimitiveType.I32 and isinstance(taip, StructType) and taip.size() == 4:
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if (source == PrimitiveType.BOOL or source == PrimitiveType.I32) and taip == PrimitiveType.I64: 
                    self.write_line("i64.extend_i32_u")
                    return
                if (source == PrimitiveType.BOOL or source == PrimitiveType.I32) and taip == PrimitiveType.I8: 
                    self.write_line(f"i32.const 255 i32.and ;; cast to {format_type(taip)}")
                    return
                if source == PrimitiveType.I64 and taip == PrimitiveType.I8: 
                    self.write_line(f"i64.const 255 i64.and i32.wrap_i64 ;; cast to {format_type(taip)}")
                    return
                if source == PrimitiveType.I64 and taip != PrimitiveType.I64:
                    self.write_line(f"i32.wrap_i64 ;; cast to {format_type(taip)}")
                    return
                if source.can_live_in_reg() and source.size() <= 4 and taip == PrimitiveType.I32:
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                if source.can_live_in_reg() and source.size() <= 8 and taip == PrimitiveType.I64:
                    self.write_line(f";; cast to {format_type(taip)}")
                    return
                self.write_line(f"UNSUPPORTED Cast from {format_type(source)} to {format_type(taip)}")
            case StringWord(_, offset, string_len):
                self.write_line(f"i32.const {self.module_data_offsets[module] + offset} i32.const {string_len}")
            case SizeofWord(_, taip):
                self.write_line(f"i32.const {taip.size()}")
            case FunRefWord(_, table_index):
                self.write_line(f"i32.const {table_index + 1}")
            case StoreWord(token, local_id, target_taip, loads):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                else:
                    self.write("local.get ")
                    self.write_local_ident(token.lexeme, local_id)
                for load in loads:
                    # self.write(f" i32.load offset={offset}")
                    self.write(f" {load}")
                self.write(" call $intrinsic:flip ")
                self.write_store(target_taip)
                self.write("\n")
            case LoadWord(_, taip, copy_space_offset):
                self.write_indent()
                if not taip.can_live_in_reg():
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset}")
                    self.write(f" i32.add call $intrinsic:dupi32 call $intrinsic:rotate-left i32.const {word.taip.size()} memory.copy\n")
                elif taip == PrimitiveType.I8:
                    self.write("i32.load8\n")
                else:
                    self.write_type(taip)
                    self.write(".load\n")
            case BreakWord():
                self.write_line("br $block")
            case BlockWord(token, words, parameters, returns):
                self.write_indent()
                self.write("(block $block")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_words(module, locals, words)
                self.dedent()
                self.write_line(")")
            case LoopWord(_, words, parameters, returns, diverges):
                self.write_indent()
                self.write("(block $block ")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_indent()
                self.write("(loop $loop ")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_words(module, locals, words)
                self.write_line("br $loop")
                self.dedent()
                self.write_line(")")
                self.dedent()
                self.write_line(")")
                if diverges:
                    self.write_line("unreachable")
            case IfWord(_, parameters, returns, if_words, else_words, diverges):
                self.write_indent()
                self.write("(if")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_line("(then")
                self.indent()
                self.write_words(module, locals, if_words)
                self.dedent()
                self.write_line(")")
                if len(else_words) > 0:
                    self.write_line("(else")
                    self.indent()
                    self.write_words(module, locals, else_words)
                    self.dedent()
                    self.write_line(")")
                self.dedent()
                self.write_line(")")
                if diverges:
                    self.write_line("unreachable")
            case StructWord(_, taip, words, copy_space_offset):
                self.write_indent()
                struct = self.lookup_type_definition(taip.struct)
                assert(not isinstance(struct, Variant))
                if taip.size() == 0:
                    for field in struct.fields.get():
                        self.write("drop ")
                    self.write(f"i32.const 0 ;; make {format_type(taip)}\n")
                    return
                self.write(f";; make {format_type(taip)}\n")
                self.indent()
                self.write_words(module, locals, words)
                self.dedent()
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ")
                if taip.size() <= 4:
                    self.write("i32.load ")
                elif taip.size() <= 8:
                    self.write("i64.load ")
                self.write(f";; make {format_type(taip)} end\n")
            case UnnamedStructWord(_, taip, copy_space_offset):
                self.write_indent()
                struct = self.lookup_type_definition(taip.struct)
                assert(not isinstance(struct, Variant))
                if taip.can_live_in_reg():
                    fields = struct.fields.get()
                    if taip.size() == 0:
                        for field in fields:
                            self.write("drop ")
                        self.write(f"i32.const 0 ;; make {format_type(taip)}\n")
                        return
                    if taip.size() <= 4:
                        assert(len(fields) == 1) # alternative is TODO
                        self.write(f";; make {format_type(taip)}\n")
                        return
                    if taip.size() <= 8:
                        assert(len(fields) == 1 or len(fields) == 2) # alternative is TODO
                        if len(fields) == 1:
                            self.write(f";; make {format_type(taip)}\n")
                            return
                        if len(fields) == 2:
                            assert(fields[0].taip.size() == 4 and fields[1].taip.size() == 4) # alternative is TODO
                            self.write(f"call $intrinsic:pack-i32s ;; make {format_type(taip)}\n")
                            self.pack_i32s_used = True
                            return
                        assert(False) # TODO
                self.write(f";; make {format_type(taip)}\n")
                self.indent()
                for i in reversed(range(0, len(struct.fields.get()))):
                    field = struct.fields.get()[i]
                    field_offset = struct.field_offset(i)
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + field_offset} i32.add ")
                    if field.taip.size() > 4 and field.taip.size() <= 8:
                        self.write("call $intrinsic:flip-i64-i32 ")
                        self.flip_i64_i32_used = True
                    else:
                        self.write("call $intrinsic:flip ")
                    self.write_store(field.taip)
                    self.write("\n")
                self.dedent()
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {format_type(taip)} end")
            case StructFieldInitWord(_, taip, copy_space_offset):
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call ")
                if taip.size() <= 4 or taip.size() > 8:
                    self.write("$intrinsic:flip ")
                else:
                    self.write("$intrinsic:flip-i64-i32 ")
                    self.flip_i64_i32_used = True
                self.write_store(taip)
                self.write("\n")
            case MatchWord(_, variant_handle, by_ref, cases, default, parameters, returns):
                variant = self.lookup_type_definition(variant_handle)
                def go(cases: List[MatchCase]):
                    if len(cases) == 0:
                        if default is None:
                            self.write("unreachable")
                            return
                        # if variant.size() <= 4 and by_ref:
                        #     self.write_indent()
                        #     self.write("i32.load")
                        #     self.write("\n")
                        self.write_words(module, locals, default)
                        return
                    case = cases[0]
                    assert(isinstance(variant, Variant))
                    case_taip = variant.cases.get()[case.tag].taip
                    if variant.size() > 8 or by_ref:
                        self.write(f"call $intrinsic:dupi32 i32.load i32.const {case.tag} i32.eq (if")
                    elif variant.size() > 4 and variant.size() <= 8:
                        self.write(f"call $intrinsic:dupi64 i32.wrap_i64 i32.const {case.tag} i32.eq (if")
                        self.dup_i64_used = True
                    else:
                        self.write(f"call $intrinsic:dupi32 i32.const {case.tag} i32.eq (if")
                    self.write_parameters(parameters)
                    if variant.size() > 8 or by_ref or variant.size() <= 4:
                        self.write(" (param i32)")
                    else:
                        self.write(" (param i64)")

                    self.write_returns(returns)
                    self.write("\n")
                    self.write_line("(then")
                    self.indent()
                    if case_taip is None:
                        self.write_line("drop")
                    elif case_taip.size() != 0:
                        self.write_indent()
                        if by_ref or variant.size() > 8:
                            self.write("i32.const 4 i32.add")
                            if not by_ref and case_taip.can_live_in_reg():
                                if case_taip.size() <= 8 and case_taip.size() > 4:
                                    self.write(" i64.load")
                                else:
                                    self.write(" i32.load")
                        else:
                            self.write("i64.const 32 i64.shr_u i32.wrap_i64")
                        self.write("\n")
                    self.write_words(module, locals, case.words)
                    self.dedent()
                    self.write_line(")")
                    self.write_indent()
                    if len(cases) == 1 and default is not None:
                        self.write("(else\n")
                        self.indent()
                        go(cases[1:])
                        self.dedent()
                        self.write_indent()
                        self.write("))")
                    else:
                        self.write("(else ")
                        go(cases[1:])
                        self.write("))")
                self.write_line(f";; match on {variant.name.lexeme}")
                self.write_indent()
                go(cases)
                self.write("\n")
            case VariantWord(_, tag, variant_handle, copy_space_offset):
                variant = self.lookup_type_definition(variant_handle)
                assert(isinstance(variant, Variant))
                case_taip = variant.cases.get()[tag].taip
                if variant.size() <= 4:
                    assert(variant.size() == 4)
                    self.write_indent()
                    if case_taip is not None:
                        self.write("drop ")
                    self.write(f"i32.const {tag} ;; store tag {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}\n")
                    return
                if variant.size() <= 8:
                    if case_taip is None:
                        self.write_line(f"i64.const {tag} ;; make {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}")
                    else:
                        self.write_line("i64.extend_i32_u i64.const 32 i64.shl ;; store value")
                        self.write_line(f"i64.const {tag} ;; store tag")
                        self.write_line(f"i64.or ;; make {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}")
                    return
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add i32.const {tag} i32.store ;; store tag")
                if case_taip is not None:
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + 4} i32.add ")
                    if case_taip.size() > 4 and case_taip.size() <= 8:
                        self.write("call $intrinsic:flip-i64-i32 ")
                    else:
                        self.write("call $intrinsic:flip ")
                    self.write_store(case_taip)
                    self.write(" ;; store value\n")
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {variant.name.lexeme}.{variant.cases.get()[tag].name.lexeme}")
            case TupleMakeWord(token, tupl, copy_space_offset):
                self.write_indent()
                if tupl.can_live_in_reg():
                    if tupl.size() == 0:
                        for item in tupl.items:
                            self.write("drop ")
                        self.write("i32.const 0 ")
                    if tupl.size() <= 4:
                        pass
                    else:
                        # alternative is TODO
                        assert(len(tupl.items) == 2 and tupl.items[0].size() == 4 and tupl.items[1].size() == 4)
                        self.write("call $intrinsic:pack-i32s")
                        self.pack_i32s_used = True
                    self.write(f";; make {format_type(tupl)}\n")
                    return
                self.write(f";; make {format_type(tupl)}\n")
                item_offset = tupl.size()
                self.indent()
                for item in reversed(tupl.items):
                    item_offset -= item.size()
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + item_offset} i32.add ")
                    if item.size() > 4 and item.size() <= 8:
                        self.write("call $intrinsic:flip-i64-i32 ")
                        self.flip_i64_i32_used = True
                    else:
                        self.write("call $intrinsic:flip ")
                    self.write_store(item)
                    self.write("\n")
                self.dedent()
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {format_type(tupl)} end\n")
            case TupleUnpackWord(token, items, copy_space_offset):
                if len(items) == 2 and items[0].size() == 4 and items[1].size() == 4:
                    self.unpack_i32s_used = True
                    self.write_line(f"call $intrinsic:unpack-i32s ;; unpack {listtostr(items, format_type)}")
                    return
                self.write_line(f";; unpack {listtostr(items, format_type)}")
                self.indent()
                offset = 0
                for i, item in enumerate(items):
                    self.write_indent()
                    if item.size() == 0:
                        self.write("i32.const ")
                        continue
                    if i + 1 != len(items):
                        self.write("call $intrinsic:dupi32 ")
                    self.write(f"i32.const {offset} i32.add ")
                    if not item.can_live_in_reg():
                        self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                        self.write("call $intrinsic:rotate-left ")
                        self.write(f"i32.const {item.size()} memory.copy")
                        copy_space_offset += item.size()
                    else:
                        self.write("i32.load")
                    if i + 1 != len(items):
                        self.write(" call $intrinsic:flip")
                    self.write("\n")
                    offset += item.size()
                self.dedent()
            case other:
                print(other, file=sys.stderr)
                assert_never(other)
        return False

    def write_store(self, taip: Type):
        if not taip.can_live_in_reg():
            self.write(f"i32.const {taip.size()} memory.copy")
        elif taip == PrimitiveType.I8:
            self.write("i32.store8")
        elif taip.size() > 4:
            self.write("i64.store")
        else:
            self.write("i32.store")

    def write_set(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, str], target_lives_in_memory: bool, target_taip: Type, loads: List[Load]):
        self.write_indent()
        def write_ident():
            match local_id:
                case LocalId():
                    local = locals[local_id]
                    self.write_local_ident(local, local_id)
                    return
                case GlobalId():
                    globl = self.globals[local_id]
                    self.write(f"${globl.taip.name.lexeme}:{local_id.module}")
                    return
                case other:
                    assert_never(other)
        if not target_lives_in_memory and len(loads) == 0:
            # The local does not live in linear memory, so the target_taip must also already be on the stack unpacked.
            assert(target_taip.can_live_in_reg())
            if isinstance(local_id, LocalId):
                self.write("local.set ")
            else:
                self.write("global.set ")
            write_ident()
            self.write("\n")
            return
        if isinstance(local_id, LocalId):
            self.write("local.get ")
        else:
            self.write("global.get ")
        write_ident()
        if len(loads) == 0:
            if target_taip.size() > 4 and target_taip.size() <= 8:
                self.write(" call $intrinsic:flip-i64-i32 ")
                self.flip_i64_i32_used = True
            else:
                self.write(" call $intrinsic:flip ")
            self.write_store(target_taip)
            self.write("\n")
            return
        if not target_lives_in_memory and isinstance(loads[-1], BitShiftLoad):
            for i, load in enumerate(loads):
                if all(isinstance(load, BitShiftLoad) for load in loads[i:]):
                    break
                self.write(f" {load}")
            if loads[-1] == BitShiftLoad.Upper32:
                self.write(" i64.const 0xFFFFFFFF i64.and")
            if loads[-1] == BitShiftLoad.Lower32:
                self.write(" i64.const 0xFFFFFFFF00000000 i64.and")
            self.write(" call $intrinsic:flip-i32-i64 i64.extend_i32_u ")
            self.flip_i32_i64_used = True
            if loads[-1] == BitShiftLoad.Upper32:
                self.write("i64.const 32 i64.shl ")
            self.write("i64.or ")
            if not target_lives_in_memory:
                if isinstance(local_id, LocalId):
                    self.write("local.set ")
                else:
                    self.write("global.set ")
                write_ident()
                self.write("\n")
                return
            else:
                if isinstance(local_id, LocalId):
                    self.write("local.get ")
                else:
                    self.write("global.get ")
                write_ident()
                self.write(" call $intrinsic:flip-i64-i32 ")
                self.flip_i64_i32_used = True
                if isinstance(loads[-1], BitShiftLoad):
                    # In case of a value being packed into an i64, we need to store
                    # the modified i64 back, not the target type.
                    self.write_store(PrimitiveType.I64)
                else:
                    self.write_store(target_taip)
                self.write("\n")
                return

        for i, load in enumerate(loads):
            self.write(f" {load}")
        if target_taip.size() > 4 and target_taip.size() <= 8:
            self.write(" call $intrinsic:flip-i64-i32 ")
        else:
            self.write(" call $intrinsic:flip ")
        self.write_store(target_taip)
        self.write("\n")
        return


    def write_return_struct_receiving(self, offset: int, returns: List[Type]) -> None:
        self.write("\n")
        if all(t.can_live_in_reg() for t in returns):
            return
        for i in range(0, len(returns)):
            self.write_line(f"local.set $s{i}:a")
        for i in range(len(returns), 0, -1):
            ret = returns[len(returns) - i]
            if not ret.can_live_in_reg():
                self.write_line(f"local.get $locl-copy-spac:e i32.const {offset} i32.add call $intrinsic:dupi32 local.get $s{i - 1}:a i32.const {ret.size()} memory.copy")
                offset += ret.size()
            else:
                self.write_line(f"local.get $s{i - 1}:a")

    def write_signature(self, module: int, signature: FunctionSignature, instance_id: int | None, locals: Dict[LocalId, Local]) -> None:
        self.write(f"func ${module}:{signature.name.lexeme}")
        if instance_id is not None and instance_id != 0:
            self.write(f":{instance_id}")
        if signature.export_name is not None:
            self.write(f" (export {signature.export_name.lexeme})")
        for parameter in signature.parameters:
            self.write(" (param $")
            for local in locals.values():
                if isinstance(local, ParameterLocal):
                    if local.name.lexeme == parameter.name.lexeme:
                        if local.lives_in_memory() and local.taip.can_live_in_reg() and local.taip.size() > 4:
                            self.write("v:")
                        break
            self.write(f"{parameter.name.lexeme} ")
            self.write_type(parameter.taip)
            self.write(")")
        self.write_returns(signature.returns)

    def write_type_human(self, taip: Type) -> None:
        self.write(format_type(taip))

    def write_parameters(self, parameters: Sequence[NamedType | Type]) -> None:
        for parameter in parameters:
            if isinstance(parameter, NamedType):
                self.write(f" (param ${parameter.name.lexeme} ")
                self.write_type(parameter.taip)
                self.write(")")
                continue
            self.write(" (param ")
            self.write_type(parameter)
            self.write(")")

    def write_returns(self, returns: List[Type]) -> None:
        for taip in returns:
            self.write(" (result ")
            self.write_type(taip)
            self.write(")")

    def write_intrinsics(self) -> None:
        self.write_line("(func $intrinsic:flip (param $a i32) (param $b i32) (result i32 i32) local.get $b local.get $a)")
        if self.flip_i64_i32_used:
            self.write_line("(func $intrinsic:flip-i64-i32 (param $a i64) (param $b i32) (result i32 i64) local.get $b local.get $a)")
        if self.flip_i32_i64_used:
            self.write_line("(func $intrinsic:flip-i32-i64 (param $a i32) (param $b i64) (result i64 i32) local.get $b local.get $a)")
        if self.flip_i64_i64_used:
            self.write_line("(func $intrinsic:flip-i64-i64 (param $a i64) (param $b i64) (result i64 i64) local.get $b local.get $a)")
        self.write_line("(func $intrinsic:dupi32 (param $a i32) (result i32 i32) local.get $a local.get $a)")
        if self.dup_i64_used:
            self.write_line("(func $intrinsic:dupi64 (param $a i64) (result i64 i64) local.get $a local.get $a)")
        self.write_line("(func $intrinsic:rotate-left (param $a i32) (param $b i32) (param $c i32) (result i32 i32 i32) local.get $b local.get $c local.get $a)")
        if self.pack_i32s_used:
            self.write_line("(func $intrinsic:pack-i32s (param $a i32) (param $b i32) (result i64) local.get $a i64.extend_i32_u local.get $b i64.extend_i32_u i64.const 32 i64.shl i64.or)")
        if self.unpack_i32s_used:
            self.write_line("(func $intrinsic:unpack-i32s (param $a i64) (result i32) (result i32) local.get $a i32.wrap_i64 local.get $a i64.const 32 i64.shr_u i32.wrap_i64)")
        if self.guard_stack:
            self.write_line("(func $stack-overflow-guar:d i32.const 1 global.get $stac:k global.get $stack-siz:e i32.lt_u i32.div_u drop)")

    def write_function_table(self) -> None:
        if len(self.function_table) == 0:
            self.write_line("(table funcref (elem))")
            return
        self.write_line("(table funcref (elem $intrinsic:flip")
        self.indent()
        self.write_indent()
        functions = list(self.function_table.items())
        functions.sort(key=lambda kv: kv[1])
        for i, (handle, _) in enumerate(functions):
            module = self.modules[handle.module]
            if isinstance(handle, FunctionHandle):
                function = module.functions[handle.index]
                if isinstance(function, GenericFunction):
                    assert(handle.instance is not None)
                    function = function.instances[handle.instance]
                    name = f"${handle.module}:{function.signature.name.lexeme}:{handle.instance}"
                else:
                    name = f"${handle.module}:{function.signature.name.lexeme}"
            else:
                name = "TODO"
            self.write(f"{name}")
            if i + 1 != len(functions):
                self.write(" ")
        self.write("))\n")
        self.dedent()

    def write_globals(self, ptr: int) -> int:
        for global_id, globl in self.globals.items():
            self.write_indent()
            size = globl.taip.taip.size()
            lives_in_memory = globl.was_reffed or not globl.taip.taip.can_live_in_reg()
            initial_value = ptr if lives_in_memory else 0
            taip = "i64" if not lives_in_memory and size > 4 and size <= 8 else "i32"
            self.write(f"(global ${globl.taip.name.lexeme}:{global_id.module} (mut {taip}) ({taip}.const {initial_value}))\n")
            if not lives_in_memory:
                continue
            ptr += globl.taip.taip.size()
        return ptr

    def write_data(self, data: bytes) -> None:
        self.write_indent()
        self.write("(data (i32.const 0) \"")
        def escape_char(char: int) -> str:
            if char == b"\\"[0]:
               return "\\\\"
            if char == b"\""[0]:
                return "\\\""
            if char == b"\t"[0]:
               return "\\t"
            if char == b"\r"[0]:
               return "\\r"
            if char == b"\n"[0]:
               return "\\n"
            if char >= 32 and char <= 126:
               return chr(char)
            hex_digits = "0123456789abcdef"
            return f"\\{hex_digits[char >> 4]}{hex_digits[char & 15]}"
        for char in data:
            self.write(escape_char(char))
        self.write("\")\n")

    def write_extern(self, module_id: int, extern: Extern) -> None:
        self.write_indent()
        self.write("(import ")
        self.write(extern.module.lexeme)
        self.write(" ")
        self.write(extern.name.lexeme)
        self.write(" (")
        self.write_signature(module_id, extern.signature, None, {})
        self.write("))")

    def write_type(self, taip: Type) -> None:
        size = taip.size()
        if size > 4 and size <= 8:
            self.write("i64")
        else:
            self.write("i32")

def fmt(a: object) -> str:
    if isinstance(a, str):
        return f"\"{a}\""
    if isinstance(a, list):
        return listtostr(a)
    return str(a)
def format_dict(dictionary: dict) -> str:
    if len(dictionary) == 0:
        return "(Map)"
    return "(Map\n" + indent(reduce(lambda a,b: a+",\n"+b, map(lambda kv: f"{fmt(kv[0])}={fmt(kv[1])}", dictionary.items()))) + ")"

def indent_non_first(s: str) -> str:
    return reduce(lambda a,b: f"{a}  {b}", map(lambda s: f"{s}", s.splitlines(keepends=True)))

def indent(s: str) -> str:
    return reduce(lambda a,b: f"{a}{b}", map(lambda s: f"  {s}", s.splitlines(keepends=True)))

Mode = Literal["lex"] | Literal["parse"] | Literal["check"] | Literal["monomize"] | Literal["compile"]

def run(path: str, mode: Mode, guard_stack: bool, stdin: str | None = None) -> str:
    if path == "-":
        file = stdin if stdin is not None else sys_stdin.get()
    else:
        with open(path, 'r') as reader:
            file = reader.read()
    tokens = Lexer(file).lex()
    if mode == "lex":
        out = ""
        for token in tokens:
            out += str(token) + "\n"
        return out
    if mode == "parse":
        module = Parser(path, file, tokens).parse()
        return str(module)
    modules: Dict[str, Parser.Module] = {}
    load_recursive(modules, os.path.normpath(path), stdin)

    resolved_modules: Dict[int, ResolvedModule] = {}
    resolved_modules_by_path: Dict[str, ResolvedModule] = {}
    for id, module in enumerate(determine_compilation_order(list(modules.values()))):
        resolved_module = ModuleResolver(resolved_modules, resolved_modules_by_path, module, id).resolve()
        resolved_modules[id] = resolved_module
        resolved_modules_by_path[module.path] = resolved_module
    if mode == "check":
        return format_dict({ (f"./{k}" if k != "-" else k): v for k,v in resolved_modules_by_path.items() })
    function_table, mono_modules = Monomizer(resolved_modules).monomize()
    if mode == "monomize":
        return "TODO"
    return WatGenerator(mono_modules, function_table, guard_stack).write_wat_module()

def main(argv: List[str], stdin: str | None = None) -> str:
    if len(argv) > 0 and argv[1] == "units":
        suite = unittest.TestSuite()
        classes = [DetermineLoadsToValueTests]
        for klass in classes:
            for method in dir(klass):
                if method.startswith("test_"):
                    suite.addTest(klass(method))
        runner = unittest.TextTestRunner()
        runner.run(suite)
        return ""
    mode: Mode = "compile"
    if len(argv) >= 2 and argv[1] == "lex":
        mode = "lex"
        path = argv[2] if len(argv) > 2 else "-"
    elif len(argv) >= 2 and argv[1] == "parse":
        mode = "parse"
        path = argv[2] if len(argv) > 2 else "-"
    elif len(argv) > 2 and argv[1] == "check":
        mode = "check"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "monomize":
        mode = "monomize"
        path = argv[2]
    elif len(argv) > 2 and argv[1] == "compile":
        mode = "compile"
        path = argv[2]
    else:
        path = argv[1]
    return run(path, mode, "--guard-stack" in argv, stdin)

if __name__ == "__main__":
    try:
        print(main(sys.argv))
    except ParserException as e:
        print(e.display(), file=sys.stderr)
        exit(1)
    except ResolverException as e:
        print(e.display(), file=sys.stderr)
        exit(1)

