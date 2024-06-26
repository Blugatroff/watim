#!/usr/bin/env python
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Any, TypeVar, Callable, Generic, List, Tuple, NoReturn, Dict, Sequence, Literal, assert_never
import subprocess
import glob
import sys
import os

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
    MEMORY = "MEMORY"
    SIZEOF = "SIZEOF"
    LOCAL = "LOCAL"
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
        return f"{self.ty.value} \"{self.lexeme}\" {str(self.line)} {str(self.column)}"

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
        if len(self.tokens) == 0 or self.tokens[-1].ty != "Foo":
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

            one_char_tokens = "<>(){}:.,$&#@!~\\"
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
                except:
                    self.tokens.append(Token(TokenType.IDENT, self.line, start_column, lexeme))
                continue
            raise LexerException("Unexpected character encountered: " + self.current(), self.line, self.column)

        return self.tokens

    @staticmethod
    def allowed_in_ident(char: str) -> bool:
        return char not in "#${}()<> \t\n:&~,."

LEXEME_TYPE_DICT: dict[str, TokenType] = {
    "fn":     TokenType.FN,
    "import": TokenType.IMPORT,
    "as":     TokenType.AS,
    "memory": TokenType.MEMORY,
    "local":  TokenType.LOCAL,
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
}
TYPE_LEXEME_DICT: dict[TokenType, str] = {v: k for k, v in LEXEME_TYPE_DICT.items()}

class PrimitiveType(str, Enum):
    I32 = "TYPE_I32"
    I64 = "TYPE_I64"
    BOOL = "TYPE_BOOL"

    def __str__(self) -> str:
        if self == PrimitiveType.I32:
            return "i32"
        if self == PrimitiveType.I64:
            return "i64"
        if self == PrimitiveType.BOOL:
            return "bool"

    def size(self) -> int:
        if self == PrimitiveType.I32:
            return 4
        if self == PrimitiveType.I64:
            return 8
        if self == PrimitiveType.BOOL:
            return 4

@dataclass
class ParsedPtrType:
    child: 'ParsedType'

@dataclass
class GenericType:
    token: Token
    generic_index: int

@dataclass
class ParsedForeignType:
    module: Token
    name: Token
    generic_arguments: List['ParsedType']

@dataclass
class ParsedStructType:
    name: Token
    generic_arguments: List['ParsedType']

@dataclass
class ParsedFunctionType:
    token: Token
    args: List['ParsedType']
    rets: List['ParsedType']

ParsedType = PrimitiveType | ParsedPtrType | GenericType | ParsedForeignType | ParsedStructType | ParsedFunctionType


@dataclass
class NumberWord:
    token: Token

@dataclass
class ParsedStringWord:
    token: Token
    string: bytearray

@dataclass
class ParsedGetWord:
    token: Token
    fields: List[Token]

@dataclass
class ParsedRefWord:
    token: Token
    fields: List[Token]

@dataclass
class ParsedSetWord:
    token: Token
    fields: List[Token]

@dataclass
class ParsedStoreWord:
    token: Token
    fields: List[Token]

@dataclass
class ParsedInitWord:
    name: Token

@dataclass
class ParsedForeignCallWord:
    module: Token
    name: Token
    generic_arguments: List[ParsedType]

@dataclass
class ParsedCallWord:
    name: Token
    generic_arguments: List[ParsedType]

@dataclass
class ParsedFunRefWord:
    call: ParsedCallWord | ParsedForeignCallWord

@dataclass
class ParsedIfWord:
    token: Token
    if_words: List['ParsedWord']
    else_words: List['ParsedWord']

@dataclass
class ParsedLoadWord:
    token: Token

@dataclass
class ParsedLoopWord:
    token: Token
    words: List['ParsedWord']

@dataclass
class ParsedBlockWord:
    token: Token
    words: List['ParsedWord']
    end_token: Token

@dataclass
class BreakWord:
    token: Token

@dataclass
class ParsedCastWord:
    token: Token
    taip: ParsedType

@dataclass
class ParsedSizeofWord:
    token: Token
    taip: ParsedType

@dataclass
class ParsedGetFieldWord:
    token: Token
    fields: List[Token]

@dataclass
class ParsedIndirectCallWord:
    token: Token

@dataclass
class ParsedStructWord:
    token: Token
    taip: ParsedStructType | ParsedForeignType
    words: List['ParsedWord']

@dataclass
class ParsedUnnamedStructWord:
    token: Token
    taip: ParsedStructType | ParsedForeignType

@dataclass
class ParsedVariantWord:
    token: Token
    taip: ParsedStructType | ParsedForeignType
    case: Token

@dataclass
class ParsedMatchCase:
    token: Token
    case: Token
    words: List['ParsedWord']

@dataclass
class ParsedMatchWord:
    token: Token
    cases: List[ParsedMatchCase]

ParsedWord = NumberWord | ParsedStringWord | ParsedCallWord | ParsedGetWord | ParsedRefWord | ParsedSetWord | ParsedStoreWord | ParsedInitWord | ParsedCallWord | ParsedForeignCallWord | ParsedFunRefWord | ParsedIfWord | ParsedLoadWord | ParsedLoopWord | ParsedBlockWord | BreakWord | ParsedCastWord | ParsedSizeofWord | ParsedGetFieldWord | ParsedIndirectCallWord | ParsedStructWord | ParsedUnnamedStructWord | ParsedMatchWord | ParsedVariantWord

@dataclass
class ParsedNamedType:
    name: Token
    taip: ParsedType

@dataclass
class ParsedImport:
    file_path: Token
    module_qualifier: Token

@dataclass
class ParsedFunctionSignature:
    export_name: Optional[Token]
    name: Token
    generic_parameters: List[Token]
    parameters: List[ParsedNamedType]
    returns: List[ParsedType]

@dataclass
class ParsedExtern:
    module: Token
    name: Token
    signature: ParsedFunctionSignature

@dataclass
class ParsedMemory:
    name: Token
    taip: ParsedType
    size: Token | None

@dataclass
class ParsedFunction:
    signature: ParsedFunctionSignature
    memories: List[ParsedMemory]
    locals: List[ParsedNamedType]
    body: List[ParsedWord]

@dataclass
class ParsedStruct:
    name: Token
    fields: List[ParsedNamedType]

@dataclass
class ParsedVariantCase:
    name: Token
    taip: ParsedType | None

@dataclass
class ParsedVariant:
    name: Token
    generic_parameters: List[Token]
    cases: List[ParsedVariantCase]

ParsedTypeDefinition = ParsedStruct | ParsedVariant

@dataclass
class ParsedModule:
    path: str
    file: str
    imports: List[ParsedImport]
    type_definitions: List[ParsedTypeDefinition]
    memories: List[ParsedMemory]
    functions: List[ParsedFunction]
    externs: List[ParsedExtern]

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

T = TypeVar('T')

@dataclass
class Parser:
    file_path: str
    file: str
    tokens: List[Token]
    cursor: int = 0

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

    def retreat(self, token: Token):
        assert(self.cursor > 0)
        self.cursor -= 1

    def abort(self, message: str) -> NoReturn:
        raise ParserException(self.file_path, self.file, self.tokens[self.cursor] if self.cursor < len(self.tokens) else None, message)

    def parse(self) -> ParsedModule:
        imports: List[ParsedImport] = []
        type_definitions: List[ParsedTypeDefinition] = []
        memories: List[ParsedMemory] = []
        functions: List[ParsedFunction] = []
        externs: List[ParsedExtern] = []
        while len(self.tokens) != 0:
            token = self.advance(skip_ws=True)
            if token is None:
                break
            if token.ty == TokenType.IMPORT:
                file_path = self.advance(skip_ws=True)
                if file_path is None or file_path.ty != TokenType.STRING:
                    self.abort("Expected file path")

                token = self.advance(skip_ws=True)
                if token is None or token.ty != TokenType.AS:
                    self.abort("Expected `as`")

                module_qualifier = self.advance(skip_ws=True)
                if module_qualifier is None or module_qualifier.ty != TokenType.IDENT:
                    self.abort("Expected an identifier as module qualifier")

                imports.append(ParsedImport(file_path, module_qualifier))
                continue

            if token.ty == TokenType.FN:
                functions.append(self.parse_function())
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
                externs.append(ParsedExtern(module, name, signature))
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
                    fields.append(ParsedNamedType(field_name, taip))
                type_definitions.append(ParsedStruct(name, fields))
                continue

            if token.ty == TokenType.VARIANT:
                name = self.advance(skip_ws=True)
                generic_parameters = self.parse_generic_parameters()
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.LEFT_BRACE:
                    self.abort("Expected `{`")
                cases: List[ParsedVariantCase] = []
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
                        cases.append(ParsedVariantCase(ident, None))
                        continue
                    self.advance(skip_ws=True)
                    cases.append(ParsedVariantCase(ident, self.parse_type(generic_parameters)))
                type_definitions.append(ParsedVariant(name, generic_parameters, cases))
                continue

            if token.ty == TokenType.MEMORY:
                self.retreat(token)
                memories.append(self.parse_memory([]))
                continue

            self.abort("Expected function import or struct definition")
        return ParsedModule(self.file_path, self.file, imports, type_definitions, memories, functions, externs)

    def parse_function(self) -> ParsedFunction:
        signature = self.parse_function_signature()
        token = self.advance(skip_ws=True)
        if token is None or token.ty != TokenType.LEFT_BRACE:
            self.abort("Expected `{`")

        memories = []
        while True:
            token = self.peek(skip_ws=True)
            if token is None or token.ty != TokenType.MEMORY:
                break
            memories.append(self.parse_memory(signature.generic_parameters))

        locals = []
        while True:
            token = self.peek(skip_ws=True)
            if token is None or token.ty != TokenType.LOCAL:
                break
            self.advance(skip_ws=True) # skip `local`
            name = self.advance(skip_ws=True)
            token = self.advance(skip_ws=True)
            if token is None or token.ty != TokenType.COLON:
                self.abort("Expected `:`")
            taip = self.parse_type(signature.generic_parameters)
            locals.append(ParsedNamedType(name, taip))

        body = self.parse_words(signature.generic_parameters)

        token = self.advance(skip_ws=True)
        assert(token.ty == TokenType.RIGHT_BRACE)
        return ParsedFunction(signature, memories, locals, body)

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
            return ParsedStringWord(token, string)
        if token.ty in [TokenType.DOLLAR, TokenType.AMPERSAND, TokenType.HASH, TokenType.DOUBLE_ARROW]:
            indicator_token = token
            name = self.advance(skip_ws=True)
            if name is None or name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as variable name")
            token = self.peek(skip_ws=True)
            def construct(name: Token, fields: List[Token]) -> ParsedWord:
                match indicator_token.ty:
                    case TokenType.DOLLAR:
                        return ParsedGetWord(name, fields)
                    case TokenType.AMPERSAND:
                        return ParsedRefWord(name, fields)
                    case TokenType.HASH:
                        return ParsedSetWord(name, fields)
                    case TokenType.DOUBLE_ARROW:
                        return ParsedStoreWord(name, fields)
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
            return ParsedInitWord(token)
        if token.ty == TokenType.IDENT:
            return self.parse_call_word(generic_parameters, token)
        if token.ty == TokenType.BACKSLASH:
            token = self.advance(skip_ws=True) # skip `\`
            return ParsedFunRefWord(self.parse_call_word(generic_parameters, token))
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
                return ParsedIfWord(token, if_words, [])
            self.advance(skip_ws=True) # skip `else`
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            else_words = self.parse_words(generic_parameters)
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                self.abort("Expected `}`")
            return ParsedIfWord(token, if_words, else_words)
        if token.ty == TokenType.TILDE:
            return ParsedLoadWord(token)
        if token.ty == TokenType.LOOP or token.ty == TokenType.BLOCK:
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            words = self.parse_words(generic_parameters)
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                self.abort("Expected `}`")
            if token.ty == TokenType.LOOP:
                return ParsedLoopWord(token, words)
            if token.ty == TokenType.BLOCK:
                return ParsedBlockWord(token, words, brace)
        if token.ty == TokenType.BREAK:
            return BreakWord(token)
        if token.ty == TokenType.BANG:
            return ParsedCastWord(token, self.parse_type(generic_parameters))
        if token.ty == TokenType.SIZEOF:
            paren = self.advance(skip_ws=True)
            if paren is None or paren.ty != TokenType.LEFT_PAREN:
                self.abort("Expected `(`")
            taip = self.parse_type(generic_parameters)
            paren = self.advance(skip_ws=True)
            if paren is None or paren.ty != TokenType.RIGHT_PAREN:
                self.abort("Expected `)`")
            return ParsedSizeofWord(token, taip)
        if token.ty == TokenType.DOT:
            self.retreat(token)
            return ParsedGetFieldWord(token, self.parse_field_accesses())
        if token.ty == TokenType.ARROW:
            return ParsedIndirectCallWord(token)
        if token.ty == TokenType.MAKE:
            struct_name_token = self.advance(skip_ws=True)
            taip = self.parse_struct_type(struct_name_token, generic_parameters)
            dot = self.peek(skip_ws=False)
            if dot is not None and dot.ty == TokenType.DOT:
                self.advance(skip_ws=False)
                case_name = self.advance(skip_ws=False)
                if case_name is None or case_name.ty != TokenType.IDENT:
                    self.abort("expected an identifier")
                return ParsedVariantWord(token, taip, case_name)
            brace = self.peek(skip_ws=True)
            if brace is not None and brace.ty == TokenType.LEFT_BRACE:
                brace = self.advance(skip_ws=True)
                words = self.parse_words(generic_parameters)
                brace = self.advance(skip_ws=True)
                if brace is None or brace.ty != TokenType.RIGHT_BRACE:
                    self.abort("Expected `}`")
                return ParsedStructWord(token, taip, words)
            return ParsedUnnamedStructWord(token, taip)
        if token.ty == TokenType.MATCH:
            brace = self.advance(skip_ws=True)
            if brace is None or brace.ty != TokenType.LEFT_BRACE:
                self.abort("Expected `{`")
            cases: List[ParsedMatchCase] = []
            while True:
                next = self.peek(skip_ws=True)
                if next is None or next.ty == TokenType.RIGHT_BRACE:
                    self.advance(skip_ws=True)
                    return ParsedMatchWord(token, cases)
                case = self.advance(skip_ws=True)
                if case is None or case.ty != TokenType.CASE:
                    self.abort("expected `case`")
                case_name = self.advance(skip_ws=True)
                if case_name is None or case_name.ty != TokenType.IDENT:
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
                cases.append(ParsedMatchCase(next, case_name, words))
        self.abort("Expected word")

    def parse_call_word(self, generic_parameters: List[Token], token: Token) -> ParsedCallWord | ParsedForeignCallWord:
        next = self.peek(skip_ws=False)
        if next is not None and next.ty == TokenType.COLON:
            module = token
            self.advance(skip_ws=False) # skip the `:`
            name = self.advance(skip_ws=False)
            if name is None or name.ty != TokenType.IDENT:
                self.abort("Expected an identifier")
            next = self.peek()
            generic_arguments = self.parse_generic_arguments(generic_parameters) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []
            return ParsedForeignCallWord(module, name, generic_arguments)
        name = token
        generic_arguments = self.parse_generic_arguments(generic_parameters) if next is not None and next.ty == TokenType.LEFT_TRIANGLE else []
        return ParsedCallWord(name, generic_arguments)

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

    def parse_memory(self, generic_parameters: List[Token]) -> ParsedMemory:
        token = self.advance(skip_ws=True)
        if token is None or token.ty != TokenType.MEMORY:
            self.abort("Expected `memory`")
        name = self.advance(skip_ws=True)
        if name is None or name.ty != TokenType.IDENT:
            self.abort("Expected an identifer as memory name")
        token = self.advance(skip_ws=True)
        if token is None or token.ty != TokenType.COLON:
            self.abort("Expected `:`")
        taip = self.parse_type(generic_parameters)
        size = self.peek(skip_ws=True)
        if size is not None and size.ty == TokenType.NUMBER:
            size = self.advance(skip_ws=True)
        else:
            size = None
        return ParsedMemory(name, taip, size)

    def parse_function_signature(self) -> ParsedFunctionSignature:
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
            parameters.append(ParsedNamedType(parameter_name, parameter_type))
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

        return ParsedFunctionSignature(function_export_name, function_ident, generic_parameters, parameters, returns)

    def parse_triangle_listed(self, elem: Callable[['Parser'], T]) -> List[T]:
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

    def parse_struct_type(self, token: Token | None, generic_parameters: List[Token]) -> ParsedStructType | ParsedForeignType:
        if token is None or token.ty != TokenType.IDENT:
            self.abort("Expected an identifer as struct name")
        next = self.peek(skip_ws=True)
        if next is not None and next.ty == TokenType.COLON:
            self.advance(skip_ws=True) # skip the `:`
            module = token
            struct_name = self.advance(skip_ws=True)
            if struct_name is None or struct_name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as struct name")
            return ParsedForeignType(module, struct_name, self.parse_generic_arguments(generic_parameters))
        else:
            struct_name = token
            if struct_name is None or struct_name.ty != TokenType.IDENT:
                self.abort("Expected an identifier as struct name")
            return ParsedStructType(struct_name, self.parse_generic_arguments(generic_parameters))

    def parse_type(self, generic_parameters: List[Token]) -> ParsedType:
        token = self.advance(skip_ws=True)
        if token is None:
            self.abort("Expected a type")
        if token.ty == TokenType.I32:
            return PrimitiveType.I32
        if token.ty == TokenType.I64:
            return PrimitiveType.I64
        if token.ty == TokenType.BOOL:
            return PrimitiveType.BOOL
        if token.ty == TokenType.DOT:
            return ParsedPtrType(self.parse_type(generic_parameters))
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
            return ParsedFunctionType(token, args, rets)
        self.abort("Expected type")

@dataclass
class Import:
    file_path: Token
    qualifier: Token
    module: int

    def __str__(self) -> str:
        return f"Import(file_path={str(self.file_path)}, qualifier={str(self.qualifier)})"

@dataclass
class ResolvedPtrType:
    child: 'ResolvedType'

    def __str__(self) -> str:
        return f"ResolvedPtrType(child={str(self.child)})"

def listtostr(l: Sequence[T], tostr: Callable[[T], str] | None = None) -> str:
    if len(l) == 0:
        return "[]"
    s = "["
    for e in l:
        if tostr is None:
            s += str(e) + ", "
        else:
            s += tostr(e) + ", "
    return s[0:-2] + "]"

@dataclass
class ResolvedStructHandle:
    module: int
    index: int

    def __str__(self) -> str:
        return f"ResolvedStructHandle(module={str(self.module)}, index={str(self.index)})"

@dataclass
class ResolvedStructType:
    name: Token
    struct: ResolvedStructHandle
    generic_arguments: List['ResolvedType']

    def __str__(self) -> str:
        return f"ResolvedStructType(name={str(self.name)}, struct={str(self.struct)}, generic_arguments={listtostr(self.generic_arguments)})"

@dataclass
class ResolvedFunctionType:
    token: Token
    parameters: List['ResolvedType']
    returns: List['ResolvedType']

@dataclass
class ResolvedStruct:
    name: Token
    fields: List['ResolvedNamedType']

    def __str__(self) -> str:
        return f"ResolvedStruct(name={str(self.name)})"

@dataclass
class ResolvedVariantCase:
    name: Token
    taip: 'ResolvedType | None'

@dataclass
class ResolvedVariant:
    name: Token
    cases: List[ResolvedVariantCase]

ResolvedTypeDefinition = ResolvedStruct | ResolvedVariant

ResolvedType = PrimitiveType | ResolvedPtrType | GenericType | ResolvedStructType | ResolvedFunctionType

def resolved_type_eq(a: ResolvedType, b: ResolvedType):
    if isinstance(a, PrimitiveType):
        return a == b
    if isinstance(a, ResolvedPtrType) and isinstance(b, ResolvedPtrType):
        return resolved_type_eq(a.child, b.child)
    if isinstance(a, ResolvedStructType) and isinstance(b, ResolvedStructType):
        return a.struct.module == a.struct.module and a.struct.index == b.struct.index and resolved_types_eq(a.generic_arguments, b.generic_arguments)
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
            return str(a)
        case ResolvedPtrType(child):
            return f".{format_resolved_type(a.child)}"
        case ResolvedStructType(name, _, generic_arguments):
            if len(generic_arguments) == 0:
                return name.lexeme
            s = name.lexeme + "<"
            for arg in generic_arguments:
                s += format_resolved_type(arg) + ", "
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
        case GenericType(token, _):
            return token.lexeme
        case other:
            assert_never(other)

@dataclass
class ResolvedNamedType:
    name: Token
    taip: ResolvedType

    def __str__(self) -> str:
        return f"ResolvedNamedType(name={str(self.name)}, taip={str(self.taip)})"

@dataclass
class ResolvedFunctionSignature:
    export_name: Optional[Token]
    name: Token
    generic_parameters: List[Token]
    parameters: List[ResolvedNamedType]
    returns: List[ResolvedType]

@dataclass
class ResolvedMemory:
    taip: ResolvedNamedType
    size: Token | None

@dataclass
class ResolvedFieldAccess:
    name: Token
    source_taip: ResolvedStructType
    target_taip: ResolvedType
    field_index: int

@dataclass
class ResolvedBody:
    words: List['ResolvedWord']
    locals: Dict['LocalId', 'ResolvedLocal']

@dataclass
class StringWord:
    token: Token
    offset: int
    len: int

@dataclass
class ResolvedLoadWord:
    token: Token
    taip: ResolvedType

@dataclass(frozen=True, eq=True)
class GlobalId:
    module: int
    index: int

@dataclass(frozen=True, eq=True)
class LocalId:
    name: str
    scope: int
    shadow: int

@dataclass
class InitWord:
    name: Token
    local_id: LocalId

@dataclass
class ResolvedGetWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[ResolvedFieldAccess]
    taip: ResolvedType

@dataclass
class ResolvedRefWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[ResolvedFieldAccess]

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
    function: 'ResolvedFunctionHandle | ResolvedExternHandle'
    generic_arguments: List[ResolvedType]

@dataclass
class ResolvedFunctionHandle:
    module: int
    index: int

@dataclass
class ResolvedExternHandle:
    module: int
    index: int

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

@dataclass
class ResolvedLoopWord:
    token: Token
    words: List['ResolvedWord']
    parameters: List[ResolvedType]
    returns: List[ResolvedType]

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

@dataclass
class ResolvedIndirectCallWord:
    token: Token
    taip: ResolvedFunctionType

@dataclass
class ResolvedStructFieldInitWord:
    token: Token
    struct: ResolvedStructHandle
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
    parameters: List[ResolvedType]
    returns: List[ResolvedType]

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
    FLIP = "FLIP"
    UNINIT = "UNINIT"

@dataclass
class ParsedIntrinsicWord:
    ty: IntrinsicType
    token: Token

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
        "rotl": IntrinsicType.ROTL,
        "rotr": IntrinsicType.ROTR,
        "or": IntrinsicType.OR,
        "store": IntrinsicType.STORE,
        "uninit": IntrinsicType.UNINIT,
}
INTRINSIC_TO_LEXEME: dict[IntrinsicType, str] = {v: k for k, v in INTRINSICS.items()}

@dataclass
class ResolvedIntrinsicAdd:
    token: Token
    taip: ResolvedPtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicSub:
    token: Token
    taip: ResolvedPtrType | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicDrop:
    token: Token

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
class ResolvedIntrinsicEqual:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicNotEqual:
    token: Token
    taip: ResolvedType

@dataclass
class IntrinsicFlip:
    token: Token

@dataclass
class IntrinsicMemGrow:
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

ResolvedIntrinsicWord = ResolvedIntrinsicAdd | ResolvedIntrinsicSub | IntrinsicDrop | ResolvedIntrinsicMod | ResolvedIntrinsicMul | ResolvedIntrinsicDiv | ResolvedIntrinsicAnd | ResolvedIntrinsicOr | ResolvedIntrinsicRotr | ResolvedIntrinsicRotl | ResolvedIntrinsicGreater | ResolvedIntrinsicLess | ResolvedIntrinsicGreaterEq | ResolvedIntrinsicLessEq | IntrinsicStore8 | IntrinsicLoad8 | IntrinsicMemCopy | ResolvedIntrinsicEqual | ResolvedIntrinsicNotEqual |IntrinsicFlip | IntrinsicMemGrow | ResolvedIntrinsicStore | ResolvedIntrinsicNot | ResolvedIntrinsicUninit

ResolvedWord = NumberWord | StringWord | ResolvedCallWord | ResolvedGetWord | ResolvedRefWord | ResolvedSetWord | ResolvedStoreWord | InitWord | ResolvedCallWord | ResolvedCallWord | ResolvedFunRefWord | ResolvedIfWord | ResolvedLoadWord | ResolvedLoopWord | ResolvedBlockWord | BreakWord | ResolvedCastWord | ResolvedSizeofWord | ResolvedGetFieldWord | ResolvedIndirectCallWord | ResolvedIntrinsicWord | InitWord | ResolvedStructFieldInitWord | ResolvedStructWord | ResolvedUnnamedStructWord | ResolvedVariantWord | ResolvedMatchWord

@dataclass
class ResolvedFunction:
    signature: ResolvedFunctionSignature
    memories: List[ResolvedMemory]
    locals: List[ResolvedNamedType]
    body: ResolvedBody

@dataclass
class ResolvedExtern:
    module: Token
    name: Token
    signature: ResolvedFunctionSignature

@dataclass
class ResolvedModule:
    path: str
    id: int
    imports: List[Import]
    type_definitions: List[ResolvedTypeDefinition]
    externs: List[ResolvedExtern]
    memories: List[ResolvedMemory]
    functions: List[ResolvedFunction]
    data: bytes

@dataclass
class Lazy(Generic[T]):
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

def load_recursive(modules: Dict[str, ParsedModule], path: str, stdin: str | None = None, import_stack: List[str]=[]):
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

def determine_compilation_order(unprocessed: List[ParsedModule]) -> List[ParsedModule]:
    ordered: List[ParsedModule] = []
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

class LocalType(str, Enum):
    PARAMETER = "PARAMETER"
    MEMORY = "MEMORY"
    LOCAL = "LOCAL"

@dataclass
class ResolvedParameterLocal:
    name: Token
    taip: ResolvedType

    @staticmethod
    def make(taip: ResolvedNamedType) -> 'ResolvedParameterLocal':
        return ResolvedParameterLocal(taip.name, taip.taip)

@dataclass
class ResolvedMemoryLocal:
    name: Token
    taip: ResolvedType
    size: int | None

@dataclass
class ResolvedDeclaredLocal:
    name: Token
    taip: ResolvedType

    @staticmethod
    def make(taip: ResolvedNamedType) -> 'ResolvedDeclaredLocal':
        return ResolvedDeclaredLocal(taip.name, taip.taip)

@dataclass
class ResolvedInitLocal:
    name: Token
    taip: ResolvedType

    @staticmethod
    def make(taip: ResolvedNamedType) -> 'ResolvedInitLocal':
        return ResolvedInitLocal(taip.name, taip.taip)

ResolvedLocal = ResolvedParameterLocal | ResolvedMemoryLocal | ResolvedDeclaredLocal | ResolvedInitLocal

@dataclass
class Ref(Generic[T]):
    value: T

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

@dataclass
class Stack:
    parent: 'Stack | None'
    stack: List[ResolvedType]
    negative: List[ResolvedType]
    drained: bool = False # True in case of non-termination because of break

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

    def drain(self):
        self.drained = True
        if self.parent is not None:
            self.parent.drain()

    def make_child(self) -> 'Stack':
        return Stack(self.clone(), [], [])

    def apply(self, other: 'Stack'):
        for removed in other.negative:
            self.pop()
        for added in other.stack:
            self.append(added)

    def __len__(self) -> int:
        return len(self.stack) + (len(self.parent) if self.parent is not None else 0)

    def __getitem__(self, index: int) -> ResolvedType:
        return self.stack[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Stack):
            return False
        if len(self.stack) != len(other.stack):
            return False
        if len(self.negative) != len(other.negative):
            return False
        for i in range(len(self.stack)):
            if not resolved_type_eq(self.stack[i], other.stack[i]):
                return False
        for i in range(len(self.negative)):
            if not resolved_type_eq(self.negative[i], other.negative[i]):
                return False
        return True

    def __str__(self) -> str:
        return f"Stack(drained={str(self.drained)}, parent={self.parent}, stack={listtostr(self.stack)}, negative={listtostr(self.negative)})"

@dataclass
class BreakStack:
    token: Token
    types: List[ResolvedType]
    reachable: bool

@dataclass
class StructLitContext:
    struct: ResolvedStructHandle
    generic_arguments: List[ResolvedType]
    fields: Dict[str, ResolvedType]

@dataclass
class ResolveWordContext:
    env: Env
    break_stacks: List[BreakStack]
    reachable: bool
    struct_context: StructLitContext | None

    def with_env(self, env: Env) -> 'ResolveWordContext':
        return ResolveWordContext(env, self.break_stacks, self.reachable, self.struct_context)

    def with_break_stacks(self, break_stacks: List[BreakStack]) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, break_stacks, self.reachable, self.struct_context)

    def with_reachable(self, reachable: bool) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, self.break_stacks, reachable, self.struct_context)

    def with_struct_context(self, struct_context: StructLitContext) -> 'ResolveWordContext':
        return ResolveWordContext(self.env, self.break_stacks, self.reachable, struct_context)

@dataclass
class FunctionResolver:
    module_resolver: 'ModuleResolver'
    externs: List[ResolvedExtern]
    signatures: List[ResolvedFunctionSignature]
    type_definitions: List[ResolvedTypeDefinition]
    function: ParsedFunction
    signature: ResolvedFunctionSignature

    def abort(self, token: Token, message: str) -> NoReturn:
        self.module_resolver.abort(token, message)

    def resolve(self) -> ResolvedFunction:
        memories = list(map(self.module_resolver.resolve_memory, self.function.memories))
        locals = list(map(self.module_resolver.resolve_named_type, self.function.locals))
        env = Env(list(map(ResolvedParameterLocal.make, self.signature.parameters)))
        for memory in memories:
            env.insert(ResolvedMemoryLocal(memory.taip.name, memory.taip.taip, int(memory.size.lexeme) if memory.size is not None else None))
        for local in locals:
            env.insert(ResolvedInitLocal(local.name, local.taip))
        stack: Stack = Stack.empty()
        context = ResolveWordContext(env, [], True, None)
        (words, diverges) = self.resolve_words(context, stack, self.function.body)
        if not diverges:
            self.expect_stack(self.signature.name, stack, self.signature.returns)
            if len(stack) != 0:
                self.abort(self.signature.name, "items left on stack at end of function")
        body = ResolvedBody(words, env.vars_by_id)
        return ResolvedFunction(self.signature, memories, locals, body)

    def resolve_word(self, context: ResolveWordContext, stack: Stack, word: ParsedWord) -> Tuple[ResolvedWord, bool]:
        match word:
            case NumberWord():
                stack.append(PrimitiveType.I32)
                return (word, False)
            case ParsedStringWord(token, string):
                stack.append(ResolvedPtrType(PrimitiveType.I32))
                stack.append(PrimitiveType.I32)
                offset = len(self.module_resolver.data)
                self.module_resolver.data.extend(string)
                return (StringWord(token, offset, len(string)), False)
            case ParsedCastWord(token, parsed_taip):
                source_type = stack.pop()
                if source_type is None:
                    self.abort(token, "expected a non-empty stack")
                resolved_type = self.module_resolver.resolve_type(parsed_taip)
                stack.append(resolved_type)
                return (ResolvedCastWord(token, source_type, resolved_type), False)
            case ParsedIfWord(token, parsed_if_words, parsed_else_words):
                if len(stack) == 0 or stack[-1] != PrimitiveType.BOOL:
                    self.abort(token, "expected a boolean on stack")
                stack.pop()
                if_env = Env(context.env)
                if_stack = stack.make_child()
                else_env = Env(context.env)
                else_stack = stack.make_child()
                (if_words, if_words_diverge) = self.resolve_words(context.with_env(if_env), if_stack, parsed_if_words)
                (else_words, else_words_diverge) = self.resolve_words(context.with_env(else_env), else_stack, parsed_else_words)
                parameters = if_stack.negative
                if not if_stack.drained:
                    returns = if_stack.stack
                elif not else_stack.drained:
                    returns = else_stack.stack
                else:
                    returns = []
                if not if_stack.drained and not else_stack.drained:
                    if if_stack != else_stack:
                        self.abort(word.token, "Type mismatch in if branches")
                    stack.apply(if_stack)
                elif if_stack.drained and else_stack.drained:
                    stack.drain()
                elif not if_stack.drained:
                    stack.apply(if_stack)
                else:
                    assert(not else_stack.drained)
                    stack.apply(else_stack)
                diverges = if_words_diverge and else_words_diverge
                return (ResolvedIfWord(token, parameters, returns, if_words, else_words), diverges)
            case ParsedLoopWord(token, parsed_words):
                loop_stack = stack.make_child()
                loop_env = Env(context.env)
                loop_break_stacks: List[BreakStack] = []
                (words, diverges) = self.resolve_words(context.with_env(loop_env).with_break_stacks(loop_break_stacks), loop_stack, parsed_words)
                parameters = loop_stack.negative
                if len(loop_break_stacks) != 0:
                    returns = loop_break_stacks[0].types
                    stack.extend(returns)
                else:
                    returns = loop_stack.stack
                    stack.apply(loop_stack)
                if len(loop_break_stacks) == 0:
                    diverges = True
                return (ResolvedLoopWord(token, words, parameters, returns), diverges)
            case ParsedBlockWord(token, parsed_words, end_token):
                block_stack = stack.make_child()
                block_env = Env(context.env)
                block_break_stacks: List[BreakStack] = []
                (words, diverges) = self.resolve_words(context.with_env(block_env).with_break_stacks(block_break_stacks), block_stack, parsed_words)
                parameters = block_stack.negative
                if len(block_break_stacks) != 0:
                    diverges = diverges and not block_break_stacks[0].reachable
                    for i in range(1, len(block_break_stacks)):
                        if not resolved_types_eq(block_break_stacks[0].types, block_break_stacks[i].types):
                            error_message = "break stack mismatch:"
                            for break_stack in block_break_stacks:
                                error_message += f"\n\t{break_stack.token.line}:{break_stack.token.column} {listtostr(break_stack.types)}"
                            error_message += f"\n\t{end_token.line}:{end_token.column} {listtostr(block_stack.stack)}"
                            self.abort(token, error_message)
                    if not block_stack.drained and not resolved_types_eq(block_break_stacks[0].types, block_stack.stack):
                        self.abort(word.token, "the items remaining on the stack at the end of the block don't match the break statements")
                    returns = block_break_stacks[0].types
                    stack.extend(returns)
                else:
                    returns = block_stack.stack
                    stack.apply(block_stack)
                return (ResolvedBlockWord(token, words, parameters, returns), diverges)
            case ParsedCallWord(name, generic_arguments):
                if name.lexeme in INTRINSICS:
                    intrinsic = INTRINSICS[name.lexeme]
                    resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
                    return (self.resolve_intrinsic(name, stack, intrinsic, resolved_generic_arguments), False)
                resolved_call_word = self.resolve_call_word(context.env, word)
                signature = self.module_resolver.get_signature(resolved_call_word.function)
                self.type_check_call(stack, resolved_call_word.name, resolved_call_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
                return (resolved_call_word, False)
            case ParsedForeignCallWord(module, name, generic_arguments):
                resolved_word = self.resolve_foreign_call_word(context.env, word)
                signature = self.module_resolver.get_signature(resolved_word.function)
                self.type_check_call(stack, name, resolved_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
                return (resolved_word, False)
            case ParsedGetWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                resolved_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                stack.append(resolved_taip)
                return (ResolvedGetWord(token, local, resolved_fields, resolved_taip), False)
            case ParsedInitWord(name):
                if context.struct_context is not None and name.lexeme in context.struct_context.fields:
                    field = context.struct_context.fields.pop(name.lexeme)
                    field = FunctionResolver.resolve_generic(context.struct_context.generic_arguments)(field)
                    self.expect_stack(name, stack, [field])
                    return (ResolvedStructFieldInitWord(name, context.struct_context.struct, context.struct_context.generic_arguments, field), False)
                taip = stack.pop()
                if taip is None:
                    self.abort(name, "expected a non-empty stack")
                named_taip = ResolvedNamedType(name, taip)
                local_id = context.env.insert(ResolvedInitLocal(named_taip.name, named_taip.taip))
                return (InitWord(name, local_id), False)
            case ParsedRefWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                stack.append(ResolvedPtrType(var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip))
                return (ResolvedRefWord(token, local, resolved_fields), False)
            case ParsedSetWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                expected_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                self.expect_stack(token, stack, [expected_taip])
                return (ResolvedSetWord(token, local, resolved_fields), False)
            case ParsedStoreWord(token, fields):
                (var_type, local) = self.resolve_var_name(context.env, token)
                resolved_fields = self.resolve_fields(var_type, fields)
                expected_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
                if not isinstance(expected_taip, ResolvedPtrType):
                    self.abort(word.token, "`=>` can only store into ptr types")
                self.expect_stack(token, stack, [expected_taip.child])
                return (ResolvedStoreWord(token, local, resolved_fields), False)
            case ParsedFunRefWord(call):
                match call:
                    case ParsedCallWord(name, _):
                        resolved_call_word = self.resolve_call_word(context.env, call)
                        signature = self.module_resolver.get_signature(resolved_call_word.function)
                        stack.append(ResolvedFunctionType(name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                        return (ResolvedFunRefWord(resolved_call_word), False)
                    case ParsedForeignCallWord(name):
                        resolved_foreign_call_word = self.resolve_foreign_call_word(context.env, call)
                        signature = self.module_resolver.get_signature(resolved_foreign_call_word.function)
                        stack.append(ResolvedFunctionType(name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                        return (ResolvedFunRefWord(resolved_foreign_call_word), False)
                    case other:
                        assert_never(other)
            case ParsedLoadWord(token):
                if len(stack) == 0:
                    self.abort(token, "expected a non-empty stack")
                top = stack.pop()
                if not isinstance(top, ResolvedPtrType):
                    self.abort(token, "expected a pointer on the stack")
                stack.append(top.child)
                return (ResolvedLoadWord(token, top.child), False)
            case BreakWord(token):
                dump = stack.dump()
                context.break_stacks.append(BreakStack(token, dump, context.reachable))
                stack.drain()
                return (word, False)
            case ParsedSizeofWord(token, parsed_taip):
                stack.append(PrimitiveType.I32)
                return (ResolvedSizeofWord(token, self.module_resolver.resolve_type(parsed_taip)), False)
            case ParsedGetFieldWord(token, fields):
                taip = stack.pop()
                if taip is None:
                    self.abort(word.token, "GetField expected a struct on the stack")
                resolved_fields = self.resolve_fields(taip, fields)
                stack.append(ResolvedPtrType(resolved_fields[-1].target_taip))
                return (ResolvedGetFieldWord(token, resolved_fields), False)
            case ParsedIndirectCallWord(token):
                if len(stack) == 0:
                    self.abort(token, "`->` expected a function on the stack")
                function_type = stack.pop()
                if not isinstance(function_type, ResolvedFunctionType):
                    self.abort(token, "`->` expected a function on the stack")
                self.type_check_call(stack, token, None, function_type.parameters, function_type.returns)
                return (ResolvedIndirectCallWord(token, function_type), False)
            case ParsedStructWord(token, taip, parsed_words):
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
            case ParsedUnnamedStructWord(token, taip):
                resolved_struct_taip = self.module_resolver.resolve_struct_type(taip)
                struct = self.module_resolver.get_type_definition(resolved_struct_taip.struct)
                assert(not isinstance(struct, ResolvedVariant))
                struct_field_types = list(map(FunctionResolver.resolve_generic(resolved_struct_taip.generic_arguments), map(lambda f: f.taip, struct.fields)))
                self.expect_stack(token, stack, struct_field_types)
                stack.append(resolved_struct_taip)
                return (ResolvedUnnamedStructWord(token, resolved_struct_taip), False)
            case ParsedVariantWord(token, taip, case_name):
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
            case ParsedMatchWord(token, cases):
                resolved_cases: List[ResolvedMatchCase] = []
                match_diverges = False
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
                case_stacks: List[Stack] = []
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
                        foo = FunctionResolver.resolve_generic(arg.generic_arguments)(taip)
                        case_stack.append(foo)
                    (resolved_words, diverges) = self.resolve_words(context, case_stack, parsed_case.words)
                    match_diverges = match_diverges or diverges
                    resolved_cases.append(ResolvedMatchCase(taip, tag, resolved_words))
                    if parsed_case.case.lexeme in visited_cases:
                        error_message = "duplicate case in match:"
                        for occurence in [visited_cases[parsed_case.case.lexeme], parsed_case.case]:
                            error_message += f"\n\t{occurence.line}:{occurence.column} {occurence.lexeme}"
                        self.abort(token, error_message)
                    remaining_cases.remove(parsed_case.case.lexeme)
                    visited_cases[parsed_case.case.lexeme] = parsed_case.case
                    case_stacks.append(case_stack)
                if len(case_stacks) != 0:
                    for i in range(1, len(case_stacks)):
                        negative_is_fine = resolved_types_eq(case_stacks[i].negative, case_stacks[0].negative)
                        positive_is_fine = resolved_types_eq(case_stacks[i].stack, case_stacks[0].stack)
                        if not negative_is_fine or not positive_is_fine:
                            error_message = "arms of match case have different types:"
                            for case_stack in case_stacks:
                                error_message += f"\n\t{listtostr(case_stack.negative)} -> {listtostr(case_stack.stack)}"
                            self.abort(token, error_message)
                    parameters = case_stacks[0].negative
                    returns = case_stacks[0].stack
                    stack.apply(case_stacks[0])
                else:
                    parameters = []
                    returns = []
                if len(remaining_cases) != 0:
                    error_message = "missing case in match:"
                    for case_name_str in remaining_cases:
                        error_message += f"\n\t{case_name_str}"
                    self.abort(token, error_message)
                match_diverges = match_diverges or len(cases) == 0
                return (ResolvedMatchWord(token, arg, by_ref, resolved_cases, parameters, returns), match_diverges)
            case other:
                assert_never(other)

    def resolve_words(self, context: ResolveWordContext, stack: Stack, parsed_words: List[ParsedWord]) -> Tuple[List[ResolvedWord], bool]:
        diverges = False
        words: List[ResolvedWord] = []
        for parsed_word in parsed_words:
            (word, word_diverges) = self.resolve_word(context.with_reachable(not diverges), stack, parsed_word)
            diverges = diverges or word_diverges
            words.append(word)
        return (words, diverges)

    def resolve_intrinsic(self, token: Token, stack: Stack, intrinsic: IntrinsicType, generic_arguments: List[ResolvedType]) -> ResolvedIntrinsicWord:
        match intrinsic:
            case IntrinsicType.ADD | IntrinsicType.SUB:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if isinstance(taip, ResolvedPtrType):
                    if stack[-1] != PrimitiveType.I32:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, i32]")
                    stack.pop()
                elif taip == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    stack.append(taip)
                elif taip == PrimitiveType.I64:
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    stack.append(taip)
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]} cannot add to {format_resolved_type(taip)}")
                if intrinsic == IntrinsicType.ADD:
                    return ResolvedIntrinsicAdd(token, taip)
                if intrinsic == IntrinsicType.SUB:
                    return ResolvedIntrinsicSub(token, taip)
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
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                elif taip == PrimitiveType.I64:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i32, i32] or [i64, i64] on stack")
                stack.append(taip)
                if intrinsic == IntrinsicType.MOD:
                    return ResolvedIntrinsicMod(token, taip)
                if intrinsic == IntrinsicType.MUL:
                    return ResolvedIntrinsicMul(token, taip)
                if intrinsic == IntrinsicType.DIV:
                    return ResolvedIntrinsicDiv(token, taip)
                assert_never(intrinsic)
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
                    case other:
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
                    self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                elif taip == PrimitiveType.I64:
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i32, i32] or [i64, i64] on stack")
                stack.append(PrimitiveType.BOOL)
                if intrinsic == IntrinsicType.GREATER:
                    return ResolvedIntrinsicGreater(token, taip)
                if intrinsic == IntrinsicType.LESS:
                    return ResolvedIntrinsicLess(token, taip)
                if intrinsic == IntrinsicType.GREATER_EQ:
                    return ResolvedIntrinsicGreaterEq(token, taip)
                if intrinsic == IntrinsicType.LESS_EQ:
                    return ResolvedIntrinsicLessEq(token, taip)
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
                return IntrinsicFlip(token)
            case IntrinsicType.MEM_GROW:
                self.expect_stack(token, stack, [PrimitiveType.I32])
                stack.append(PrimitiveType.I32)
                return IntrinsicMemGrow(token)
            case IntrinsicType.STORE:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if not isinstance(stack[-2], ResolvedPtrType) or not resolved_type_eq(stack[-2].child, stack[-1]):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, a]")
                taip = stack[-1]
                stack.pop()
                stack.pop()
                return ResolvedIntrinsicStore(token, taip)
            case IntrinsicType.NOT:
                if len(stack) == 0 or (stack[-1] != PrimitiveType.I32 and stack[-1] != PrimitiveType.BOOL):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected a i32 or bool on the stack")
                return ResolvedIntrinsicNot(token, stack[-1])
            case IntrinsicType.UNINIT:
                if len(generic_arguments) != 1:
                    self.abort(token, "uninit only accepts one generic argument")
                stack.append(generic_arguments[0])
                return ResolvedIntrinsicUninit(token, generic_arguments[0])
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

    def resolve_call_word(self, env: Env, word: ParsedCallWord) -> ResolvedCallWord:
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
                case other:
                    return other
        return inner

    def resolve_foreign_call_word(self, env: Env, word: ParsedForeignCallWord) -> ResolvedCallWord:
        resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
        for imp in self.module_resolver.imports:
            if imp.qualifier.lexeme == word.module.lexeme:
                module = self.module_resolver.resolved_modules[imp.module]
                for index, f in enumerate(module.functions):
                    if f.signature.name.lexeme == word.name.lexeme:
                        return ResolvedCallWord(word.name, ResolvedFunctionHandle(module.id, index), resolved_generic_arguments)
                for index, extern in enumerate(module.externs):
                    if extern.signature.name.lexeme == word.name.lexeme:
                        return ResolvedCallWord(word.name, ResolvedExternHandle(module.id, index), resolved_generic_arguments)
                self.abort(word.name, f"function {word.name.lexeme} not found")
        self.abort(word.name, f"module {word.module.lexeme} not found")

    def resolve_var_name(self, env: Env, name: Token) -> Tuple[ResolvedType, LocalId | GlobalId]:
        var = env.lookup(name)
        if var is None:
            for index, memory in enumerate(self.module_resolver.memories):
                if memory.taip.name.lexeme == name.lexeme:
                    return (memory.taip.taip, GlobalId(self.module_resolver.id, index))
            self.abort(name, f"local {name.lexeme} not found")
        return (var[0].taip, var[1])

    def resolve_fields(self, taip: ResolvedType, fields: List[Token]) -> List[ResolvedFieldAccess]:
        resolved_fields: List[ResolvedFieldAccess] = []
        while len(fields) > 0:
            field_name = fields[0]
            def inner(source_taip: ResolvedStructType, fields: List[Token]) -> ResolvedType:
                struct = self.module_resolver.get_type_definition(source_taip.struct)
                if isinstance(struct, ResolvedVariant):
                    self.abort(field_name, "can not access fields of a variant")
                for field_index, field in enumerate(struct.fields):
                    if field.name.lexeme == field_name.lexeme:
                        target_taip = FunctionResolver.resolve_generic(source_taip.generic_arguments)(field.taip)
                        resolved_fields.append(ResolvedFieldAccess(field_name, source_taip, target_taip, field_index))
                        fields.pop(0)
                        return target_taip
                self.abort(field_name, f"field not found {field_name.lexeme}")
            if isinstance(taip, ResolvedStructType):
                taip = inner(taip, fields)
                continue
            if isinstance(taip, ResolvedPtrType) and not isinstance(taip.child, ResolvedPtrType):
                taip = taip.child
                continue
            else:
                self.abort(field_name, f"field not found {field_name.lexeme} WTF?")
        return resolved_fields


@dataclass
class ModuleResolver:
    resolved_modules: Dict[int, ResolvedModule]
    resolved_modules_by_path: Dict[str, ResolvedModule]
    module: ParsedModule
    id: int
    imports: List[Import] = field(default_factory=list)
    resolved_type_definitions: List[ResolvedTypeDefinition] = field(default_factory=list)
    externs: List[ResolvedExtern] = field(default_factory=list)
    memories: List[ResolvedMemory] = field(default_factory=list)
    data: bytearray = field(default_factory=bytearray)
    signatures: List[ResolvedFunctionSignature] = field(default_factory=list)

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolverException(self.module.path, self.module.file, token, message)

    def get_signature(self, function: ResolvedFunctionHandle | ResolvedExternHandle) -> ResolvedFunctionSignature:
        if isinstance(function, ResolvedFunctionHandle):
            if self.id == function.module:
                return self.signatures[function.index]
            else:
                return self.resolved_modules[function.module].functions[function.index].signature
        if self.id == function.module:
            return self.externs[function.index].signature
        return self.resolved_modules[function.module].externs[function.index].signature

    def get_type_definition(self, struct: ResolvedStructHandle) -> ResolvedTypeDefinition:
        if struct.module == self.id:
            return self.resolved_type_definitions[struct.index]
        return self.resolved_modules[struct.module].type_definitions[struct.index]

    def resolve(self) -> ResolvedModule:
        resolved_imports = list(map(self.resolve_import, self.module.imports))
        self.imports = resolved_imports
        resolved_type_definitions = list(map(self.resolve_type_definition, self.module.type_definitions))
        self.resolved_type_definitions = resolved_type_definitions
        self.memories = list(map(self.resolve_memory, self.module.memories))
        resolved_externs = list(map(self.resolve_extern, self.module.externs))
        self.externs = resolved_externs
        resolved_signatures = list(map(lambda f: self.resolve_function_signature(f.signature), self.module.functions))
        self.signatures = resolved_signatures
        resolved_functions = list(map(lambda f: self.resolve_function(f[0], f[1]), zip(resolved_signatures, self.module.functions)))
        return ResolvedModule(self.module.path, self.id, resolved_imports, resolved_type_definitions, resolved_externs, self.memories, resolved_functions, self.data)

    def resolve_function(self, signature: ResolvedFunctionSignature, function: ParsedFunction) -> ResolvedFunction:
        return FunctionResolver(self, self.externs, self.signatures, self.resolved_type_definitions, function, signature).resolve()

    def resolve_function_name(self, name: Token) -> ResolvedFunctionHandle | ResolvedExternHandle:
        for index, signature in enumerate(self.signatures):
            if signature.name.lexeme == name.lexeme:
                return ResolvedFunctionHandle(self.id, index)
        for index, extern in enumerate(self.externs):
            if extern.signature.name.lexeme == name.lexeme:
                return ResolvedExternHandle(self.id, index)
        self.abort(name, f"function {name.lexeme} not found")

    def resolve_memory(self, memory: ParsedMemory) -> ResolvedMemory:
        return ResolvedMemory(ResolvedNamedType(memory.name, ResolvedPtrType(self.resolve_type(memory.taip))), memory.size)

    def resolve_extern(self, extern: ParsedExtern) -> ResolvedExtern:
        return ResolvedExtern(extern.module, extern.name, self.resolve_function_signature(extern.signature))

    def resolve_import(self, imp: ParsedImport) -> Import:
        if os.path.dirname(self.module.path) != "":
            path = os.path.normpath(os.path.dirname(self.module.path) + "/" + imp.file_path.lexeme[1:-1])
        else:
            path = os.path.normpath(imp.file_path.lexeme[1:-1])
        imported_module = self.resolved_modules_by_path[path]
        return Import(imp.file_path, imp.module_qualifier, imported_module.id)

    def resolve_named_type(self, named_type: ParsedNamedType) -> ResolvedNamedType:
        return ResolvedNamedType(named_type.name, self.resolve_type(named_type.taip))

    def resolve_type(self, taip: ParsedType) -> ResolvedType:
        match taip:
            case PrimitiveType():
                return taip
            case ParsedPtrType(child):
                return ResolvedPtrType(self.resolve_type(child))
            case GenericType():
                return taip
            case ParsedFunctionType(token, parsed_args, parsed_rets):
                args = list(map(self.resolve_type, parsed_args))
                rets = list(map(self.resolve_type, parsed_rets))
                return ResolvedFunctionType(token, args, rets)
            case struct_type:
                return self.resolve_struct_type(struct_type)

    def resolve_struct_type(self, taip: ParsedStructType | ParsedForeignType) -> ResolvedStructType:
        match taip:
            case ParsedStructType(name, generic_arguments):
                resolved_generic_arguments = list(map(self.resolve_type, generic_arguments))
                return ResolvedStructType(name, self.resolve_struct_name(name), resolved_generic_arguments)
            case ParsedForeignType(module, name, generic_arguments):
                resolved_generic_arguments = list(map(self.resolve_type, generic_arguments))
                for imp in self.imports:
                    if imp.qualifier.lexeme == taip.module.lexeme:
                        for index, struct in enumerate(self.resolved_modules[imp.module].type_definitions):
                            if struct.name.lexeme == name.lexeme:
                                return ResolvedStructType(taip.name, ResolvedStructHandle(imp.module, index), resolved_generic_arguments)
                self.abort(taip.module, f"struct {module.lexeme}:{name.lexeme} not found")
            case other:
                assert_never(other)

    def resolve_struct_name(self, name: Token) -> ResolvedStructHandle:
        for index, struct in enumerate(self.module.type_definitions):
            if struct.name.lexeme == name.lexeme:
                return ResolvedStructHandle(self.id, index)
        self.abort(name, f"struct {name.lexeme} not found")

    def resolve_type_definition(self, definition: ParsedTypeDefinition) -> ResolvedTypeDefinition:
        match definition:
            case ParsedStruct():
                return self.resolve_struct(definition)
            case ParsedVariant():
                return self.resolve_variant(definition)
            case other:
                assert_never(other)

    def resolve_struct(self, struct: ParsedStruct) -> ResolvedStruct:
        return ResolvedStruct(struct.name, list(map(self.resolve_named_type, struct.fields)))

    def resolve_variant(self, variant: ParsedVariant) -> ResolvedVariant:
        return ResolvedVariant(variant.name, list(map(lambda t: ResolvedVariantCase(t.name, self.resolve_type(t.taip) if t.taip is not None else None), variant.cases)))

    def resolve_function_signature(self, signature: ParsedFunctionSignature) -> ResolvedFunctionSignature:
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

@dataclass
class Struct:
    name: Token
    fields: List['NamedType']
    generic_parameters: List['Type']

    def __str__(self) -> str:
        return f"Struct(name={str(self.name)})"

    def size(self) -> int:
        size = 0
        for field in self.fields:
            size += field.taip.size()
        return size

    def field_offset(self, field_index: int) -> int:
        return sum(self.fields[i].taip.size() for i in range(0, field_index))

@dataclass
class VariantCase:
    name: Token
    taip: 'Type | None'

@dataclass
class Variant:
    name: Token
    cases: List[VariantCase]
    generic_arguments: List['Type']

    def size(self) -> int:
        return 4 + max((t.taip.size() for t in self.cases if t.taip is not None), default=0)

TypeDefinition = Struct | Variant

@dataclass
class StructHandle:
    module: int
    index: int
    instance: int

    def __str__(self) -> str:
        return f"ResolvedStructHandle(module={str(self.module)}, index={str(self.index)}, instance={str(self.instance)})"

@dataclass
class StructType:
    name: Token
    struct: StructHandle
    _size: int

    def __str__(self) -> str:
        return f"StructType(name={str(self.name)}, struct={str(self.struct)})"

    def size(self) -> int:
        return self._size

Type = PrimitiveType | PtrType | StructType | FunctionType

def type_eq(a: Type, b: Type) -> bool:
    if isinstance(a, PrimitiveType) and isinstance(b, PrimitiveType):
        return a == b
    if isinstance(a, PtrType) and isinstance(b, PtrType):
        return type_eq(a.child, b.child)
    if isinstance(a, StructType) and isinstance(b, StructType):
        return a.struct.module == b.struct.module and a.struct.index == b.struct.index and a.struct.instance == b.struct.instance
    if isinstance(a, FunctionType) and isinstance(b, FunctionType):
        return types_eq(a.parameters, b.parameters) and types_eq(a.returns, b.returns)
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
            return str(a)
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
        case other:
            assert_never(other)

@dataclass
class ParameterLocal:
    name: Token
    taip: Type

    def size(self) -> int:
        return self.taip.size()

@dataclass
class MemoryLocal:
    name: Token
    taip: Type
    annotated_size: int | None = None

    def size(self) -> int:
        return self.annotated_size if self.annotated_size is not None else self.taip.size()

@dataclass
class DeclaredLocal:
    name: Token
    taip: Type

    def size(self) -> int:
        return self.taip.size()

@dataclass
class InitLocal:
    name: Token
    taip: Type

    def size(self) -> int:
        return self.taip.size()

Local = ParameterLocal | MemoryLocal | DeclaredLocal | InitLocal

@dataclass
class Body:
    words: List['Word']
    locals: Dict[LocalId, Local]
    locals_copy_space: int
    max_struct_ret_count: int

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
class Memory:
    taip: NamedType
    size: Token | None

@dataclass
class Extern:
    module: Token
    name: Token
    signature: FunctionSignature

@dataclass
class ConcreteFunction:
    signature: FunctionSignature
    memories: List[Memory]
    locals: List[NamedType]
    body: Lazy[Body]

@dataclass
class GenericFunction:
    instances: List[Tuple[List[Type], ConcreteFunction]]

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
    source_taip: StructType
    target_taip: Type
    offset: int

@dataclass
class GetFieldWord:
    token: Token
    fields: List[FieldAccess]

@dataclass
class SetWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[FieldAccess]

@dataclass
class GetWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[FieldAccess]
    copy_space_offset: int | None

@dataclass
class RefWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[FieldAccess]

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
    copy_space_offset: int

@dataclass
class StoreWord:
    token: Token
    local: LocalId | GlobalId
    fields: List[FieldAccess]

@dataclass
class StructWord:
    token: Token
    taip: Type
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
    parameters: List[Type]
    returns: List[Type]

IntrinsicWord = IntrinsicAdd | IntrinsicSub | IntrinsicEqual | IntrinsicNotEqual | IntrinsicAnd | IntrinsicDrop | IntrinsicLoad8 | IntrinsicStore8 | IntrinsicGreaterEq | IntrinsicLessEq | IntrinsicMul | IntrinsicMod | IntrinsicDiv | IntrinsicGreater | IntrinsicLess | IntrinsicFlip | IntrinsicRotl | IntrinsicRotr | IntrinsicOr | IntrinsicStore | IntrinsicMemCopy | IntrinsicMemGrow | IntrinsicNot | IntrinsicUninit

Word = NumberWord | StringWord | CallWord | GetWord | InitWord | CastWord | SetWord | LoadWord | IntrinsicWord | IfWord | RefWord | IndirectCallWord | StoreWord | FunRefWord | LoopWord | BreakWord | SizeofWord | BlockWord | GetFieldWord | StructWord | StructFieldInitWord | UnnamedStructWord | VariantWord | MatchWord

@dataclass
class Module:
    id: int
    type_definitions: Dict[int, List[TypeDefinition]]
    externs: List[Extern]
    memories: List[Memory]
    functions: Dict[int, Function]
    data: bytes

@dataclass
class Monomizer:
    modules: Dict[int, ResolvedModule]
    type_definitions: Dict[int, Dict[int, List[Tuple[List[Type], TypeDefinition]]]] = field(default_factory=dict)
    externs: Dict[int, List[Extern]] = field(default_factory=dict)
    functions: Dict[int, Dict[int, Function]] = field(default_factory=dict)
    function_table: Dict[FunctionHandle | ExternHandle, int] = field(default_factory=dict)
    struct_word_copy_space_offset: int = 0

    def monomize(self) -> Tuple[Dict[FunctionHandle | ExternHandle, int], Dict[int, Module]]:
        for id in sorted(self.modules):
            module = self.modules[id]
            functions: List[Function] = []
            self.externs[id] = list(map(self.monomize_extern, module.externs))
            for index, function in enumerate(module.functions):
                if function.signature.export_name is not None:
                    assert(len(function.signature.generic_parameters) == 0)
                    signature = self.monomize_concrete_signature(function.signature)
                    memories = list(map(lambda m: self.monomize_memory(m, []), function.memories))
                    locals = list(map(lambda t: self.monomize_named_type(t, []), function.locals))
                    copy_space_offset = Ref(0)
                    max_struct_ret_count = Ref(0)
                    body = Lazy(lambda: Body(self.monomize_words(function.body.words, [], copy_space_offset, max_struct_ret_count), self.monomize_locals(function.body.locals, []), copy_space_offset.value, max_struct_ret_count.value))
                    f = ConcreteFunction(signature, memories, locals, body)
                    if id not in self.functions:
                        self.functions[id] = {}
                    self.functions[id][index] = f
                    body.get()

        all_have = False
        while not all_have:
            all_have = True
            for module_functions in self.functions.values():
                for ffs in module_functions.values():
                    if isinstance(ffs, ConcreteFunction):
                        if ffs.body.has_value():
                            continue
                        ffs.body.get()
                        all_have = False
                        break
                    for (_, instance) in ffs.instances:
                        if instance.body.has_value():
                            continue
                        instance.body.get()
                        all_have = False
                        break
                    if not all_have:
                        break
                if not all_have:
                    break

        mono_modules = {}
        for module_id in self.modules:
            if module_id not in self.type_definitions:
                self.type_definitions[module_id] = {}
            if module_id not in self.functions:
                self.functions[module_id] = {}
            module = self.modules[module_id]
            externs: List[Extern] = self.externs[module_id]
            type_definitions: Dict[int, List[TypeDefinition]] = { k: [t[1] for t in v] for k, v in self.type_definitions[module_id].items() }
            memories = list(map(lambda m: self.monomize_memory(m, []), module.memories))
            mono_modules[module_id] = Module(module_id, type_definitions, externs, memories, self.functions[module_id], self.modules[module_id].data)
        return self.function_table, mono_modules

    def monomize_locals(self, locals: Dict[LocalId, ResolvedLocal], generics: List[Type]) -> Dict[LocalId, Local]:
        res: Dict[LocalId, Local] = {}
        for id, local in locals.items():
            taip = self.monomize_type(local.taip, generics)
            match local:
                case ResolvedParameterLocal(name):
                    res[id] = ParameterLocal(local.name, taip)
                    continue
                case ResolvedMemoryLocal(name):
                    res[id] = MemoryLocal(local.name, taip, local.size)
                    continue
                case ResolvedInitLocal(name):
                    res[id] = InitLocal(local.name, taip)
                    continue
                case ResolvedDeclaredLocal(name):
                    res[id] = DeclaredLocal(local.name, taip)
                    continue
                case other:
                    assert_never(other)
        return res

    def monomize_concrete_signature(self, signature: ResolvedFunctionSignature) -> FunctionSignature:
        assert(len(signature.generic_parameters) == 0)
        return self.monomize_signature(signature, [])

    def monomize_function(self, function: ResolvedFunctionHandle, generics: List[Type]) -> ConcreteFunction:
        f = self.modules[function.module].functions[function.index]
        if len(generics) == 0:
            assert(len(f.signature.generic_parameters) == 0)
        signature = self.monomize_signature(f.signature, generics)
        memories = list(map(lambda m: self.monomize_memory(m, generics), f.memories))
        locals = list(map(lambda t: self.monomize_named_type(t, generics), f.locals))
        copy_space_offset = Ref(0)
        max_struct_ret_count = Ref(0)
        body = Lazy(lambda: Body(list(map(lambda w: self.monomize_word(w, generics, copy_space_offset, max_struct_ret_count), f.body.words)), self.monomize_locals(f.body.locals, generics), copy_space_offset.value, max_struct_ret_count.value))
        concrete_function = ConcreteFunction(signature, memories, locals, body)
        if function.module not in self.functions:
            self.functions[function.module] = {}
        if len(f.signature.generic_parameters) == 0:
            assert(len(generics) == 0)
            assert(function.index not in self.functions[function.module])
            self.functions[function.module][function.index] = concrete_function
            return concrete_function
        if function.index not in self.functions[function.module]:
            self.functions[function.module][function.index] = GenericFunction([])
        generic_function = self.functions[function.module][function.index]
        assert(isinstance(generic_function, GenericFunction))
        generic_function.instances.append((generics, concrete_function))
        return concrete_function

    def monomize_signature(self, signature: ResolvedFunctionSignature, generics: List[Type]) -> FunctionSignature:
        parameters = list(map(lambda t: self.monomize_named_type(t, generics), signature.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), signature.returns))
        return FunctionSignature(signature.export_name, signature.name, generics, parameters, returns)

    def monomize_memory(self, memory: ResolvedMemory, generics: List[Type]) -> Memory:
        return Memory(self.monomize_named_type(memory.taip, generics), memory.size)

    def monomize_words(self, words: List[ResolvedWord], generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int]) -> List[Word]:
        return list(map(lambda w: self.monomize_word(w, generics, copy_space_offset, max_struct_ret_count), words))

    def monomize_word(self, word: ResolvedWord, generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int]) -> Word:
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
                copy_space_offset.value += sum(taip.size() for taip in monomized_function_taip.returns)
                max_struct_ret_count.value = max(max_struct_ret_count.value, len(monomized_function_taip.returns))
                return IndirectCallWord(token, monomized_function_taip, local_copy_space_offset)
            case ResolvedGetWord(token, local_id, resolved_fields, taip):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                monomized_taip = self.monomize_type(taip, generics)
                if isinstance(monomized_taip, StructType):
                    offset = copy_space_offset.value
                    copy_space_offset.value += monomized_taip.size()
                else:
                    offset = None
                return GetWord(token, local_id, fields, offset)
            case InitWord():
                return word
            case ResolvedSetWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                return SetWord(token, local_id, fields)
            case ResolvedRefWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                return RefWord(token, local_id, fields)
            case ResolvedStoreWord(token, local_id, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                return StoreWord(token, local_id, fields)
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
            case IntrinsicFlip():
                return word
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
            case IntrinsicMemGrow():
                return word
            case ResolvedIntrinsicUninit(token, taip):
                monomized_taip = self.monomize_type(taip, generics)
                offset = copy_space_offset.value
                copy_space_offset.value += monomized_taip.size()
                return IntrinsicUninit(token, offset)
            case ResolvedLoadWord(token, taip):
                monomized_taip = self.monomize_type(taip, generics)
                if isinstance(monomized_taip, StructType):
                    offset = copy_space_offset.value
                    copy_space_offset.value += monomized_taip.size()
                else:
                    offset = None
                return LoadWord(token, monomized_taip, offset)
            case ResolvedCastWord(token, source, taip):
                return CastWord(token, self.monomize_type(source, generics), self.monomize_type(taip, generics))
            case ResolvedIfWord(token, resolved_parameters, resolved_returns, resolved_if_words, resolved_else_words):
                if_words = self.monomize_words(resolved_if_words, generics, copy_space_offset, max_struct_ret_count)
                else_words = self.monomize_words(resolved_else_words, generics, copy_space_offset, max_struct_ret_count)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return IfWord(token, parameters, returns, if_words, else_words)
            case ResolvedFunRefWord(call):
                call_word = self.monomize_call_word(call, copy_space_offset, max_struct_ret_count, generics)
                table_index = self.insert_function_into_table(call_word.function)
                return FunRefWord(call_word, table_index)
            case ResolvedLoopWord(token, resolved_words, parameters, returns):
                words = self.monomize_words(resolved_words, generics, copy_space_offset, max_struct_ret_count)
                parameters = list(map(lambda t: self.monomize_type(t, generics), parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), returns))
                return LoopWord(token, words, parameters, returns)
            case ResolvedSizeofWord(token, taip):
                return SizeofWord(token, self.monomize_type(taip, generics))
            case ResolvedBlockWord(token, resolved_words, resolved_parameters, resolved_returns):
                words = self.monomize_words(resolved_words, generics, copy_space_offset, max_struct_ret_count)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return BlockWord(token, words, parameters, returns)
            case ResolvedGetFieldWord(token, resolved_fields):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                return GetFieldWord(token, fields)
            case ResolvedStructWord(token, taip, resolved_words):
                monomized_taip = self.monomize_type(taip, generics)
                offset = copy_space_offset.value
                copy_space_offset.value += monomized_taip.size()
                previous_struct_word_copy_space_offset = self.struct_word_copy_space_offset
                self.struct_word_copy_space_offset = offset
                words = self.monomize_words(resolved_words, generics, copy_space_offset, max_struct_ret_count)
                self.struct_word_copy_space_offset = previous_struct_word_copy_space_offset
                return StructWord(token, monomized_taip, words, offset)
            case ResolvedUnnamedStructWord(token, taip):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                self.struct_word_copy_space_offset = offset
                copy_space_offset.value += monomized_taip.size()
                return UnnamedStructWord(token, monomized_taip, offset)
            case ResolvedStructFieldInitWord(token, struct, generic_arguments, taip):
                field_copy_space_offset = self.struct_word_copy_space_offset
                generics_here = list(map(lambda t: self.monomize_type(t, generics), generic_arguments))
                (struct_handle, monomized_struct) = self.monomize_struct(struct, generics_here)
                if isinstance(monomized_struct, Variant):
                    assert(False)

                for field in monomized_struct.fields:
                    if field.name.lexeme == token.lexeme:
                        break
                    field_copy_space_offset += field.taip.size()
                return StructFieldInitWord(token, self.monomize_type(taip, generics), field_copy_space_offset)
            case ResolvedVariantWord(token, case, resolved_variant_type):
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                (variant_handle, variant) = self.monomize_struct(resolved_variant_type.struct, this_generics)
                offset = copy_space_offset.value
                copy_space_offset.value += variant.size()
                return VariantWord(token, case, variant_handle, offset)
            case ResolvedMatchWord(token, resolved_variant_type, by_ref, cases, parameters, returns):
                monomized_cases: List[MatchCase] = []
                for resolved_case in cases:
                    words = self.monomize_words(resolved_case.words, generics, copy_space_offset, max_struct_ret_count)
                    monomized_cases.append(MatchCase(resolved_case.tag, words))
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                monomized_variant = self.monomize_struct(resolved_variant_type.struct, this_generics)[0]
                return MatchWord(token, monomized_variant, by_ref, monomized_cases, parameters, returns)
            case other:
                assert_never(other)

    def insert_function_into_table(self, function: FunctionHandle | ExternHandle) -> int:
        if function not in self.function_table:
            self.function_table[function] = len(self.function_table)
        return self.function_table[function]

    def monomize_field_accesses(self, fields: List[ResolvedFieldAccess], generics: List[Type]) -> List[FieldAccess]:
        if len(fields) == 0:
            return []

        field = fields[0]
        assert(isinstance(field.source_taip, ResolvedPtrType) or isinstance(field.source_taip, ResolvedStructType))

        source_taip = self.monomize_struct_type(field.source_taip, generics)
        handle_and_struct = self.monomize_struct(field.source_taip.struct, list(map(lambda t: self.monomize_type(t, generics), field.source_taip.generic_arguments)))
        struct = handle_and_struct[1]
        assert(not isinstance(struct, Variant))
        target_taip = self.monomize_type(field.target_taip, struct.generic_parameters)
        offset = struct.field_offset(field.field_index)
        return [FieldAccess(field.name, source_taip, target_taip, offset)] + self.monomize_field_accesses(fields[1:], struct.generic_parameters)

    def monomize_call_word(self, word: ResolvedCallWord, copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], generics: List[Type]) -> CallWord:
        if isinstance(word.function, ResolvedExternHandle):
            signature = self.externs[word.function.module][word.function.index].signature
            offset = copy_space_offset.value
            copy_space = sum(taip.size() for taip in signature.returns if isinstance(taip, StructType))
            max_struct_ret_count.value = max(max_struct_ret_count.value, len(signature.returns) if copy_space > 0 else 0)
            copy_space_offset.value += copy_space
            return CallWord(word.name, ExternHandle(word.function.module, word.function.index), offset)
        generics_here = list(map(lambda t: self.monomize_type(t, generics), word.generic_arguments))
        if word.function.module not in self.functions:
            self.functions[word.function.module] = {}
        if word.function.index in self.functions[word.function.module]:
            function = self.functions[word.function.module][word.function.index]
            if isinstance(function, ConcreteFunction):
                assert(len(word.generic_arguments) == 0)
                offset = copy_space_offset.value
                copy_space = sum(taip.size() for taip in function.signature.returns if isinstance(taip, StructType))
                max_struct_ret_count.value = max(max_struct_ret_count.value, len(function.signature.returns) if copy_space > 0 else 0)
                copy_space_offset.value += copy_space
                return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, None), offset)
            for instance_index, (instance_generics, instance) in enumerate(function.instances):
                if types_eq(instance_generics, generics_here):
                    offset = copy_space_offset.value
                    copy_space = sum(taip.size() for taip in instance.signature.returns if isinstance(taip, StructType))
                    max_struct_ret_count.value = max(max_struct_ret_count.value, len(instance.signature.returns) if copy_space > 0 else 0)
                    copy_space_offset.value += copy_space
                    return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, instance_index), offset)
        self.monomize_function(word.function, generics_here)
        return self.monomize_call_word(word, copy_space_offset, max_struct_ret_count, generics) # the function instance should now exist, try monomorphizing this CallWord again

    def lookup_struct(self, struct: ResolvedStructHandle, generics: List[Type]) -> Tuple[StructHandle, TypeDefinition] | None:
        if struct.module not in self.type_definitions:
            self.type_definitions[struct.module] = {}
        if struct.index not in self.type_definitions[struct.module]:
            return None
        for instance_index, (genics, instance) in enumerate(self.type_definitions[struct.module][struct.index]):
            if types_eq(genics, generics):
                return StructHandle(struct.module, struct.index, instance_index), instance
        return None

    def add_struct(self, module: int, index: int, taip: TypeDefinition, generics: List[Type]) -> StructHandle:
        if module not in self.type_definitions:
            self.type_definitions[module] = {}
        if index not in self.type_definitions[module]:
            self.type_definitions[module][index] = []
        instance_index = len(self.type_definitions[module][index])
        self.type_definitions[module][index].append((generics, taip))
        return StructHandle(module, index, instance_index)

    def monomize_struct(self, struct: ResolvedStructHandle, generics: List[Type]) -> Tuple[StructHandle, TypeDefinition]:
        handle_and_instance = self.lookup_struct(struct, generics)
        if handle_and_instance is not None:
            return handle_and_instance
        s = self.modules[struct.module].type_definitions[struct.index]
        if isinstance(s, ResolvedVariant):
            cases: List[VariantCase] = []
            variant_instance = Variant(s.name, cases, generics)
            handle = self.add_struct(struct.module, struct.index, variant_instance, generics)
            for case in map(lambda c: VariantCase(c.name, self.monomize_type(c.taip, generics) if c.taip is not None else None), s.cases):
                cases.append(case)
            return handle, variant_instance
        fields: List[NamedType] = []
        struct_instance = Struct(s.name, fields, generics)
        handle = self.add_struct(struct.module, struct.index, struct_instance, generics)
        for field in map(lambda t: self.monomize_named_type(t, generics), s.fields):
            fields.append(field)
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
        return StructType(taip.name, handle, struct.size())

    def monomize_function_type(self, taip: ResolvedFunctionType, generics: List[Type]) -> FunctionType:
        parameters = list(map(lambda t: self.monomize_type(t, generics), taip.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), taip.returns))
        return FunctionType(taip.token, parameters, returns)

    def monomize_extern(self, extern: ResolvedExtern) -> Extern:
        signature = self.monomize_concrete_signature(extern.signature)
        return Extern(extern.module, extern.name, signature)

def align_to(n: int, to: int) -> int:
    return n + (to - (n % to)) * ((n % to) > 0)

@dataclass
class WatGenerator:
    modules: Dict[int, Module]
    function_table: Dict[FunctionHandle | ExternHandle, int]
    chunks: List[str] = field(default_factory=list)
    indentation: int = 0
    globals: Dict[GlobalId, Memory]= field(default_factory=dict)
    module_data_offsets: Dict[int, int] = field(default_factory=dict)

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
            return function.instances[handle.instance][1]
        return function

    def write_wat_module(self) -> str:
        assert(len(self.chunks) == 0)
        self.write_line("(module")
        self.indent()
        for module in self.modules.values():
            for extern in module.externs:
                self.write_extern(module.id, extern)
                self.write("\n")
            for i, memory in enumerate(module.memories):
                self.globals[GlobalId(module.id, i)] = memory

        self.write_line("(memory 1 65536)")
        self.write_line("(export \"memory\" (memory 0))")

        all_data: bytes = b""
        for id in sorted(self.modules):
            self.module_data_offsets[id] = len(all_data)
            all_data += self.modules[id].data

        self.write_intrinsics()

        self.write_function_table()

        data_end = align_to(len(all_data), 4)
        global_mem = self.write_globals(data_end)
        stack_start = align_to(data_end + global_mem, 4)
        self.write_line(f"(global $stac:k (mut i32) (i32.const {stack_start}))")

        self.write_data(all_data)

        for module_id in sorted(self.modules):
            module = self.modules[module_id]
            for function in module.functions.values():
                self.write_function(module_id, function)
        self.dedent()
        self.write(")")
        return ''.join(self.chunks)

    def write_function(self, module: int, function: Function, instance_id: int | None = None) -> None:
        if isinstance(function, GenericFunction):
            for (id, (_, instance)) in enumerate(function.instances):
                self.write_function(module, instance, id)
            return
        body = function.body.get()
        self.write_indent()
        self.write("(")
        self.write_signature(module, function.signature, instance_id)
        if len(function.signature.generic_arguments) > 0:
            self.write(" ;;")
            for taip in function.signature.generic_arguments:
                self.write(" ")
                self.write_type_human(taip)
        self.write("\n")
        self.indent()
        self.write_locals(function.body.get())
        for i in range(0, body.max_struct_ret_count):
            self.write_indent()
            self.write(f"(local $s{i}:a i32)\n")
        if body.locals_copy_space != 0:
            self.write_indent()
            self.write("(local $locl-copy-spac:e i32)\n")

        uses_stack = body.locals_copy_space != 0 or any(isinstance(local.taip, StructType) or isinstance(local, MemoryLocal) for local in function.body.get().locals.values())
        if uses_stack:
            self.write_indent()
            self.write("(local $stac:k i32)\n")
            self.write_indent()
            self.write("global.get $stac:k local.set $stac:k\n")

        for local_id, local in function.body.get().locals.items():
            if isinstance(local, MemoryLocal):
                self.write_mem(local.name.lexeme, local.size(), local_id.scope, local_id.shadow)
        if body.locals_copy_space != 0:
            self.write_mem("locl-copy-spac:e", body.locals_copy_space, 0, 0)
        self.write_structs(body.locals)
        self.write_body(module, body)
        if uses_stack:
            self.write_indent()
            self.write("local.get $stac:k global.set $stac:k\n")
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
            if not isinstance(local, ParameterLocal) and isinstance(local.taip, StructType):
                self.write_mem(local.name.lexeme, local.taip.size(), local_id.scope, local_id.shadow)

    def write_locals(self, body: Body) -> None:
        for local_id, local in body.locals.items():
            if isinstance(local, ParameterLocal):
                continue
            local = body.locals[local_id]
            self.write_indent()
            if local.taip == PrimitiveType.I64:
                ty = "i64"
            else:
                ty = "i32"
            self.write(f"(local ${local.name.lexeme}")
            if local_id.scope != 0 or local_id.shadow != 0:
                self.write(f":{local_id.scope}:{local_id.shadow}")
            self.write(f" {ty})\n")

    def write_body(self, module: int, body: Body) -> None:
        self.write_words(module, body.locals, body.words)

    def write_words(self, module: int, locals: Dict[LocalId, Local], words: List[Word]) -> None:
        for word in words:
            self.write_word(module, locals, word)

    def write_local_ident(self, name: str, local: LocalId) -> None:
        if local.scope != 0 or local.shadow != 0:
            self.write(f"${name}:{local.scope}:{local.shadow}")
        else:
            self.write(f"${name}")

    def write_word(self, module: int, locals: Dict[LocalId, Local], word: Word) -> None:
        match word:
            case NumberWord(token):
                self.write_line(f"i32.const {token.lexeme}")
            case GetWord(token, local_id, fields, copy_space_offset):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    target_taip = fields[-1].target_taip if len(fields) > 0 else self.globals[local_id].taip.taip
                if isinstance(local_id, LocalId):
                    local = locals[local_id]
                    target_taip = fields[-1].target_taip if len(fields) > 0 else local.taip
                if isinstance(target_taip, StructType):
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:dupi32 ")
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                else:
                    self.write("local.get ")
                    self.write_local_ident(token.lexeme, local_id)
                loads = self.determine_loads(fields)
                if len(loads) == 0 and isinstance(target_taip, StructType):
                    self.write(f" i32.const {target_taip.size()} memory.copy")
                for i, load in enumerate(loads):
                    if i + 1 < len(loads) or not isinstance(target_taip, StructType):
                        self.write(f" i32.load offset={load}")
                    else:
                        self.write(f" i32.const {load} i32.add i32.const {target_taip.size()} memory.copy")
                self.write("\n")
            case GetFieldWord(token, fields):
                assert(fields != 0)
                loads = self.determine_loads(fields)
                for i, load in enumerate(loads):
                    if i + 1 == len(loads):
                        self.write(f" i32.const {load} i32.add")
                    else:
                        self.write(f" i32.load offset={load}")
                self.write("\n")
            case RefWord(token, local_id, fields):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                if isinstance(local_id, LocalId):
                    self.write(f"local.get ")
                    self.write_local_ident(token.lexeme, local_id)
                loads = self.determine_loads(fields)
                for i, load in enumerate(loads):
                    if i + 1 == len(loads):
                        self.write(f" i32.const {load} i32.add")
                    else:
                        self.write(f" i32.load offset={load}")
                self.write("\n")
            case SetWord(token, local_id, fields):
                if isinstance(local_id, GlobalId):
                    print("SetWord on Global: TODO", file=sys.stderr)
                    assert(False)
                    return
                self.write_set(local_id, locals, fields)
            case InitWord(name, local_id):
                self.write_set(local_id, locals, [])
            case CallWord(name, function_handle, return_space_offset):
                self.write_indent()
                if isinstance(function_handle, ExternHandle):
                    extern = self.lookup_extern(function_handle)
                    signature = extern.signature
                    self.write(f"call ${function_handle.module}:{name.lexeme}")
                if isinstance(function_handle, FunctionHandle):
                    function = self.lookup_function(function_handle)
                    signature = function.signature
                    self.write(f"call ${function_handle.module}:{function.signature.name.lexeme}")
                    if function_handle.instance is not None:
                        self.write(f":{function_handle.instance}")
                self.write_return_struct_receiving(return_space_offset, signature.returns)
            case IndirectCallWord(token, taip, return_space_offset):
                self.write_indent()
                self.write("(call_indirect")
                self.write_parameters(taip.parameters)
                self.write_returns(taip.returns)
                self.write(")")
                self.write_return_struct_receiving(return_space_offset, taip.returns)
            case IntrinsicStore(token, taip):
                if isinstance(taip, StructType):
                    self.write_line(f"i32.const {taip.size()} memory.copy")
                else:
                    self.write_line("i32.store")
            case IntrinsicAdd(token, taip):
                if isinstance(taip, PtrType) or taip == PrimitiveType.I32:
                    self.write_line(f"i32.add")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line(f"i64.add")
                    return
                assert_never(word.taip)
            case IntrinsicSub(token, taip):
                if isinstance(taip, PtrType) or taip == PrimitiveType.I32:
                    self.write_line(f"i32.sub")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line(f"i64.sub")
                    return
                assert_never(word.taip)
            case IntrinsicMul(_, taip):
                self.write_line(f"{str(taip)}.mul")
            case IntrinsicDrop():
                self.write_line("drop")
            case IntrinsicOr(_, taip):
                self.write_indent()
                self.write_type(taip)
                self.write(".or\n")
            case IntrinsicEqual(_, taip):
                if taip == PrimitiveType.I32 or taip == PrimitiveType.BOOL or isinstance(taip, FunctionType) or isinstance(taip, PtrType):
                    self.write_line("i32.eq")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.eq")
                    return
                if isinstance(taip, StructType):
                    print("IntrinsicEqual for struct type TODO", file=sys.stderr)
                    assert(False)
                assert_never(taip)
            case IntrinsicNotEqual(_, taip):
                if isinstance(taip, StructType):
                    print("IntrinsicNotEqual for struct type TODO", file=sys.stderr)
                    assert(False)
                if taip == PrimitiveType.I64:
                    self.write_line("i64.eq")
                    return
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
            case IntrinsicFlip():
                self.write_line("call $intrinsic:flip")
            case IntrinsicRotl(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.extend_i32_s i64.rotl")
                else:
                    self.write_line("i32.rotl")
            case IntrinsicRotr(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.extend_i32_s i64.rotr")
                else:
                    self.write_line("i32.rotr")
            case IntrinsicAnd(_, taip):
                if taip == PrimitiveType.I32 or taip == PrimitiveType.BOOL:
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
                    case other:
                        assert_never(other)
            case IntrinsicDiv(_, taip):
                match taip:
                    case PrimitiveType.I32:
                        self.write_line("i32.div_u")
                    case PrimitiveType.I64:
                        self.write_line("i64.div_u")
                    case other:
                        assert_never(other)
            case IntrinsicMemCopy():
                self.write_line("memory.copy")
            case IntrinsicMemGrow():
                self.write_line("memory.grow")
            case IntrinsicUninit(_, copy_space_offset):
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add")
            case CastWord(_, source, taip):
                if (source == PrimitiveType.BOOL or source == PrimitiveType.I32) and taip == PrimitiveType.I64: 
                    self.write_line(f"i64.extend_i32_s ;; cast to {format_type(taip)}")
                    return
                if source == PrimitiveType.I64 and taip != PrimitiveType.I64:
                    self.write_line(f"i32.wrap_i64 ;; cast to {format_type(taip)}")
                    return
                self.write_line(f";; cast to {format_type(taip)}")
            case StringWord(_, offset, string_len):
                self.write_line(f"i32.const {self.module_data_offsets[module] + offset} i32.const {string_len}")
            case SizeofWord(_, taip):
                self.write_line(f"i32.const {taip.size()}")
            case FunRefWord(_, table_index):
                self.write_line(f"i32.const {table_index}")
            case StoreWord(token, local_id, fields):
                self.write_indent()
                if isinstance(local_id, GlobalId):
                    self.write(f"global.get ${token.lexeme}:{local_id.module}")
                    target_type = fields[-1].target_taip if len(fields) > 0 else self.globals[local_id].taip.taip
                else:
                    self.write(f"local.get ")
                    self.write_local_ident(token.lexeme, local_id)
                    target_type = locals[local_id].taip
                    assert(isinstance(target_type, PtrType))
                    target_type = fields[-1].target_taip if len(fields) > 0 else target_type.child
                loads = self.determine_loads(fields)
                for offset in loads:
                    self.write(f" i32.load offset={offset}")
                self.write(" call $intrinsic:flip ")
                if isinstance(target_type, StructType):
                    self.write(f" i32.const {target_type.size()} memory.copy\n")
                else:
                    self.write_type(target_type)
                    self.write(".store\n")
            case LoadWord(_, taip, copy_space_offset):
                if isinstance(taip, StructType):
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset}")
                    self.write(f" i32.add call $intrinsic:dupi32 call $intrinsic:rotate-left i32.const {word.taip.size()} memory.copy\n")
                else:
                    self.write_indent()
                    self.write_type(taip)
                    self.write(".load\n")
            case BreakWord():
                self.write_line("br $block")
            case BlockWord(token, words, parameters, returns):
                self.write_indent()
                self.write("(block $block ")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_words(module, locals, words)
                self.dedent()
                self.write_indent()
                self.write(")\n")
            case LoopWord(_, words, parameters, returns):
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
            case IfWord(_, parameters, returns, if_words, else_words):
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
            case StructWord(_, taip, words, copy_space_offset):
                self.write_indent()
                self.write(f";; make {format_type(taip)}\n")
                self.indent()
                self.write_words(module, locals, words)
                self.dedent()
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {format_type(taip)} end\n")
            case UnnamedStructWord(_, taip, copy_space_offset):
                self.write_indent()
                self.write(f";; make {format_type(taip)}\n")
                struct = self.lookup_type_definition(taip.struct)
                assert(not isinstance(struct, Variant))
                field_offset = taip.size()
                self.indent()
                for field in reversed(struct.fields):
                    field_offset -= field.taip.size()
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + field_offset} i32.add call $intrinsic:flip ")
                    if isinstance(field.taip, StructType):
                        self.write(f"i32.const {field.taip.size()} memory.copy\n")
                    else:
                        self.write("i32.store\n")
                self.dedent()
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add ;; make {format_type(taip)} end\n")
            case StructFieldInitWord(_, taip, copy_space_offset):
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add call $intrinsic:flip ")
                if isinstance(taip, StructType):
                    self.write(f"i32.const {taip.size()} memory.copy\n")
                else:
                    self.write("i32.store\n")
            case MatchWord(_, variant_handle, by_ref, cases, parameters, returns):
                variant = self.lookup_type_definition(variant_handle)
                def go(cases: List[MatchCase]):
                    if len(cases) == 0:
                        self.write("unreachable")
                        return
                    case = cases[0]
                    assert(isinstance(variant, Variant))
                    case_taip = variant.cases[case.tag].taip
                    self.write(f"call $intrinsic:dupi32 i32.load i32.const {case.tag} i32.eq (if (param i32)")
                    self.write_parameters(parameters)
                    self.write_returns(returns)
                    self.write("\n")
                    self.write_line("(then")
                    self.indent()
                    if case_taip is None:
                        self.write_line("drop")
                    else:
                        self.write_indent()
                        self.write("i32.const 4 i32.add")
                        if case_taip == PrimitiveType.I64 and not by_ref:
                            self.write(" i64.load")
                        elif not isinstance(case_taip, StructType) and not by_ref:
                            self.write(" i32.load")
                        self.write("\n")
                    self.write_words(module, locals, case.words)
                    self.dedent()
                    self.write_line(")")
                    self.write_indent()
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
                case_taip = variant.cases[tag].taip
                self.write_indent()
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add i32.const {tag} i32.store ;; store tag\n")
                if case_taip is not None:
                    self.write_indent()
                    self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset + 4} i32.add call $intrinsic:flip ")
                    if isinstance(case_taip, StructType):
                        self.write(f"i32.const {case_taip.size()} memory.copy ;; store value\n")
                    elif case_taip == PrimitiveType.I64:
                        self.write(f"i64.store ;; store value\n")
                    else:
                        self.write(f"i32.store ;; store value\n")
                self.write_line(f"local.get $locl-copy-spac:e i32.const {copy_space_offset} i32.add")
            case other:
                print(other, file=sys.stderr)
                assert_never(other)

    def write_set(self, local_id: LocalId, locals: Dict[LocalId, Local], fields: List[FieldAccess]):
        local = locals[local_id]
        loads = self.determine_loads(fields)
        target_taip = fields[-1].target_taip if len(fields) != 0 else local.taip
        self.write_indent()
        if not isinstance(target_taip, StructType) and len(loads) == 0:
            self.write(f"local.set ")
            self.write_local_ident(local.name.lexeme, local_id)
            self.write("\n")
            return
        self.write(f"local.get ")
        self.write_local_ident(local.name.lexeme, local_id)
        if isinstance(target_taip, StructType) and len(loads) == 0:
            self.write(f" call $intrinsic:flip i32.const {target_taip.size()} memory.copy\n")
            return
        if not isinstance(target_taip, StructType):
            for i, load in enumerate(loads):
                self.write(f" i32.const {load} i32.add ")
                if i + 1 == len(loads):
                    self.write("call $intrinsic:flip ")
                    self.write_type(local.taip)
                    self.write(".store")
                else:
                    self.write("i32.load")
            self.write("\n")
            return
        for i, load in enumerate(loads):
            self.write(f" i32.const {load} i32.add ")
            if i + 1 == len(loads):
                self.write(f"call $intrinsic:flip i32.const {target_taip.size()} memory.copy")
            else:
                self.write("i32.load")
        self.write("\n")
        return


    def write_return_struct_receiving(self, offset: int, returns: List[Type]) -> None:
        self.write("\n")
        if not any(isinstance(t, StructType) for t in returns):
            return
        for i in range(0, len(returns)):
            self.write_line(f"local.set $s{i}:a")
        for i in range(len(returns), 0, -1):
            ret = returns[len(returns) - i]
            if isinstance(ret, StructType):
                self.write_line(f"local.get $locl-copy-spac:e i32.const {offset} i32.add call $intrinsic:dupi32 local.get $s{i - 1}:a i32.const {ret.size()} memory.copy")
                offset += 1
            else:
                self.write_line(f"local.get $s{i - 1}:a")

    def determine_loads(self, fields: List[FieldAccess]) -> List[int]:
        loads: List[int] = []
        i = 0
        while i < len(fields):
            taip = fields[i].source_taip
            assert(isinstance(taip, StructType) or (isinstance(taip, PtrType) and isinstance(taip.child, StructType)))

            offset = 0
            while i < len(fields): # loop for every field access which can be reduced to one load with offset
                field = fields[i]
                offset += field.offset
                i += 1
                if not isinstance(field.target_taip, StructType):
                    break
            loads.append(offset)
        return loads

    def write_signature(self, module: int, signature: FunctionSignature, instance_id: int | None = None) -> None:
        self.write(f"func ${module}:{signature.name.lexeme}")
        if instance_id is not None:
            self.write(f":{instance_id}")
        if signature.export_name is not None:
            self.write(f" (export {signature.export_name.lexeme})")
        self.write_parameters(signature.parameters)
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
            self.write(f" (param ")
            self.write_type(parameter)
            self.write(")")

    def write_returns(self, returns: List[Type]) -> None:
        for taip in returns:
            self.write(f" (result ")
            self.write_type(taip)
            self.write(")")

    def write_intrinsics(self) -> None:
        self.write_line("(func $intrinsic:flip (param $a i32) (param $b i32) (result i32 i32) local.get $b local.get $a)")
        self.write_line("(func $intrinsic:dupi32 (param $a i32) (result i32 i32) local.get $a local.get $a)")
        self.write_line("(func $intrinsic:rotate-left (param $a i32) (param $b i32) (param $c i32) (result i32 i32 i32) local.get $b local.get $c local.get $a)")

    def write_function_table(self) -> None:
        if len(self.function_table) == 0:
            self.write_line("(table funcref (elem))")
            return
        self.write_line("(table funcref (elem")
        self.indent()
        functions = list(self.function_table.items())
        functions.sort(key=lambda kv: kv[1])
        for handle, _ in functions:
            module = self.modules[handle.module]
            if isinstance(handle, FunctionHandle):
                function = module.functions[handle.index]
                if isinstance(function, GenericFunction):
                    function = function.instances[0][1]
                    name = f"${handle.module}:{function.signature.name.lexeme}:{handle.instance}"
                else:
                    name = f"${handle.module}:{function.signature.name.lexeme}"
            else:
                name = "TODO"
            self.write(f"{name} ")
        self.dedent()
        self.write_line("))")

    def write_globals(self, ptr: int) -> int:
        for global_id, globl in self.globals.items():
            self.write_indent()
            self.write(f"(global ${globl.taip.name.lexeme}:{global_id.module} (mut i32) (i32.const {ptr}))\n")
            ptr += globl.taip.taip.size() if globl.size is None else int(globl.size.lexeme)
        return ptr

    def write_data(self, data: bytes) -> None:
        self.write_indent()
        self.write("(data (i32.const 0) \"")
        def escape_char(char: int) -> str:
            if char == b"\\"[0]:
               return "\\\\"
            if char == b"\""[0]:
                return "\\\""
            if char >= 32 and char <= 126:
               return chr(char)
            if char == b"\t"[0]:
               return "\\t"
            if char == "\r"[0]:
               return "\\r"
            if char == b"\n"[0]:
               return "\\n"
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
        self.write_signature(module_id, extern.signature)
        self.write("))")

    def write_type(self, taip: Type) -> None:
        if taip == PrimitiveType.I64:
            self.write("i64")
        else:
            self.write("i32")

Mode = Literal["lex"] | Literal["compile"]

def run(path: str, mode: Mode, stdin: str | None = None) -> str:
    if mode == "lex":
        if path == "-":
            file = stdin if stdin is not None else sys_stdin.get()
        else:
            with open(path, 'r') as reader:
                file = reader.read()
        tokens = Lexer(file).lex()
        out = ""
        for token in tokens:
            out += str(token) + "\n"
        return out
    modules: Dict[str, ParsedModule] = {}
    load_recursive(modules, os.path.normpath(path), stdin)

    resolved_modules: Dict[int, ResolvedModule] = {}
    resolved_modules_by_path: Dict[str, ResolvedModule] = {}
    for id, module in enumerate(determine_compilation_order(list(modules.values()))):
        resolved_module = ModuleResolver(resolved_modules, resolved_modules_by_path, module, id).resolve()
        resolved_modules[id] = resolved_module
        resolved_modules_by_path[module.path] = resolved_module
    function_table, mono_modules = Monomizer(resolved_modules).monomize()
    return WatGenerator(mono_modules, function_table).write_wat_module()

def main(argv: List[str], stdin: str | None = None) -> str:
    mode: Literal["compile"] | Literal["lex"] = "compile"
    if len(argv) > 2 and argv[2] == "--lex":
        mode = "lex"
    return run(argv[1], mode, stdin)

if __name__ == "__main__":
    try:
        print(main(sys.argv))
    except ParserException as e:
        print(e.display(), file=sys.stderr)
        exit(1)
    except ResolverException as e:
        print(e.display(), file=sys.stderr)
        exit(1)

