#!/usr/bin/env python
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Any, TypeVar, Callable, Generic, List, Tuple, NoReturn, Dict, Sequence, assert_never
from abc import ABC, abstractmethod
from shutil import copyfile
from functools import reduce
import subprocess
import glob
import json
import pathlib
import sys
import os

from termcolor import colored

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
        return f"\"{self.lexeme}:{str(self.line)}:{str(self.column)}\""

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
            self.tokens.append(Token.space(self.cursor, self.column))
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
class DerefWord:
    token: Token

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

ParsedWord = NumberWord | ParsedStringWord | ParsedCallWord | DerefWord | ParsedGetWord | ParsedRefWord | ParsedSetWord | ParsedStoreWord | ParsedInitWord | ParsedCallWord | ParsedForeignCallWord | ParsedFunRefWord | ParsedIfWord | ParsedLoadWord | ParsedLoopWord | ParsedBlockWord | BreakWord | ParsedCastWord | ParsedSizeofWord | ParsedGetFieldWord | ParsedIndirectCallWord

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
class ParsedModule:
    path: str
    imports: List[ParsedImport]
    structs: List[ParsedStruct]
    memories: List[ParsedMemory]
    functions: List[ParsedFunction]
    externs: List[ParsedExtern]

@dataclass
class ParserException(Exception):
    token: Token | None
    message: str

T = TypeVar('T')

@dataclass
class Parser:
    file_path: str
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
        assert(False)

    def parse(self) -> ParsedModule:
        imports: List[ParsedImport] = []
        structs: List[ParsedStruct] = []
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
                structs.append(ParsedStruct(name, fields))
                continue

            if token.ty == TokenType.MEMORY:
                self.retreat(token)
                memories.append(self.parse_memory([]))
                continue

            self.abort("Expected function import or struct definition")
        return ParsedModule(self.file_path, imports, structs, memories, functions, externs)

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
                return ParsedBlockWord(token, words)
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

def listtostr(l: Sequence[object]) -> str:
    if len(l) == 0:
        return "[]"
    s = "["
    for e in l:
        s += str(e) + ", "
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

ResolvedType = PrimitiveType | ResolvedPtrType | GenericType | ResolvedStructType | ResolvedFunctionType

def resolved_type_eq(a: ResolvedType, b: ResolvedType):
    if isinstance(a, PrimitiveType):
        return a == b
    if isinstance(a, ResolvedPtrType) and isinstance(b, ResolvedPtrType):
        return resolved_type_eq(a.child, b.child)
    if isinstance(a, ResolvedStructType) and isinstance(b, ResolvedStructType):
        return a.struct.module == a.struct.module and a.struct.index == b.struct.index
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
    if isinstance(a, PrimitiveType):
        return str(a)
    if isinstance(a, ResolvedPtrType):
        return f".{format_resolved_type(a.child)}"
    if isinstance(a, ResolvedStructType):
        if len(a.generic_arguments) == 0:
            return a.name.lexeme
        s = a.name.lexeme + "<"
        for arg in a.generic_arguments:
            s += format_resolved_type(arg) + ", "
        return s + ">"
    if isinstance(a, ResolvedFunctionType):
        s = "("
        for param in a.parameters:
            s += format_resolved_type(param) + ", "
        s = s[:-2] + " -> "
        if len(a.returns) == 0:
            return s[:-1] + ")"
        for ret in a.returns:
            s += format_resolved_type(ret) + ", "
        return s[:-2] + ")"
    if isinstance(a, GenericType):
        return a.token.lexeme
    assert_never(a)

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
    base_taip: ResolvedType
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedIndirectCallWord:
    token: Token
    taip: ResolvedFunctionType

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
        }
INTRINSIC_TO_LEXEME: dict[IntrinsicType, str] = {v: k for k, v in INTRINSICS.items()}

@dataclass
class ResolvedIntrinsicAdd:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicSub:
    token: Token
    taip: ResolvedType

@dataclass
class IntrinsicDrop:
    token: Token

@dataclass
class ResolvedIntrinsicMod:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicMul:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicDiv:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicAnd:
    token: Token
    taip: ResolvedType

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
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicLess:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicGreaterEq:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicLessEq:
    token: Token
    taip: ResolvedType

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
    taip: ResolvedType

ResolvedIntrinsicWord = ResolvedIntrinsicAdd | ResolvedIntrinsicSub | IntrinsicDrop | ResolvedIntrinsicMod | ResolvedIntrinsicMul | ResolvedIntrinsicDiv | ResolvedIntrinsicAnd | ResolvedIntrinsicOr | ResolvedIntrinsicRotr | ResolvedIntrinsicRotl | ResolvedIntrinsicGreater | ResolvedIntrinsicLess | ResolvedIntrinsicGreaterEq | ResolvedIntrinsicLessEq | IntrinsicStore8 | IntrinsicLoad8 | IntrinsicMemCopy | ResolvedIntrinsicEqual | ResolvedIntrinsicNotEqual |IntrinsicFlip | IntrinsicMemGrow | ResolvedIntrinsicStore | ResolvedIntrinsicNot

ResolvedWord = NumberWord | StringWord | ResolvedCallWord | DerefWord | ResolvedGetWord | ResolvedRefWord | ResolvedSetWord | ResolvedStoreWord | InitWord | ResolvedCallWord | ResolvedCallWord | ResolvedFunRefWord | ResolvedIfWord | ResolvedLoadWord | ResolvedLoopWord | ResolvedBlockWord | BreakWord | ResolvedCastWord | ResolvedSizeofWord | ResolvedGetFieldWord | ResolvedIndirectCallWord | ResolvedIntrinsicWord | InitWord

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
    structs: List[ResolvedStruct]
    externs: List[ResolvedExtern]
    memories: List[ResolvedMemory]
    functions: List[ResolvedFunction]
    data: bytes

def load_recursive(modules: Dict[str, ParsedModule], path: str, import_stack: List[str]=[]):
    with open(path, 'r') as reader:
        file = reader.read()
        tokens = Lexer(file).lex()
        module = Parser(path, tokens).parse()
        modules[path] = module
        for imp in module.imports:
            if os.path.dirname(path) != "":
                p = os.path.normpath(os.path.dirname(path) + "/" + imp.file_path.lexeme[1:-1])
            else:
                p = os.path.normpath(imp.file_path.lexeme[1:-1])
            if p in import_stack:
                print("Module import cycle detected: ", end="")
                for a in import_stack:
                    print(f"{a} -> ", end="")
                print(p)
                exit(1)
            if p in modules:
                continue
            import_stack.append(p)
            load_recursive(modules, p, import_stack)
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
    token: Token
    message: str

class LocalType(str, Enum):
    PARAMETER = "PARAMETER"
    MEMORY = "MEMORY"
    LOCAL = "LOCAL"

@dataclass
class ResolvedLocal:
    name: Token
    taip: ResolvedType
    ty: LocalType
    size: int | None = None # only used in case of self.ty == LocalType.MEMORY

    @staticmethod
    def param(param: ResolvedNamedType) -> 'ResolvedLocal':
        return ResolvedLocal(param.name, param.taip, LocalType.PARAMETER)

    @staticmethod
    def memory(param: ResolvedNamedType, size: int | None) -> 'ResolvedLocal':
        return ResolvedLocal(param.name, param.taip, LocalType.MEMORY, size)

    @staticmethod
    def local(param: ResolvedNamedType) -> 'ResolvedLocal':
        return ResolvedLocal(param.name, param.taip, LocalType.LOCAL)

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
        # parent_dump = self.parent.dump() if self.parent is not None else []
        # return parent_dump + self.stack

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
class FunctionResolver:
    module_resolver: 'ModuleResolver'
    externs: List[ResolvedExtern]
    signatures: List[ResolvedFunctionSignature]
    structs: List[ResolvedStruct]
    function: ParsedFunction
    signature: ResolvedFunctionSignature

    def abort(self, token: Token, message: str) -> NoReturn:
        self.module_resolver.abort(token, message)

    def resolve(self) -> ResolvedFunction:
        memories = list(map(self.module_resolver.resolve_memory, self.function.memories))
        locals = list(map(self.module_resolver.resolve_named_type, self.function.locals))
        env = Env(list(map(ResolvedLocal.param, self.signature.parameters)))
        for memory in memories:
            env.insert(ResolvedLocal.memory(memory.taip, int(memory.size.lexeme) if memory.size is not None else None))
        for local in locals:
            env.insert(ResolvedLocal.local(local))
        stack: Stack = Stack.empty()
        words = list(map(lambda w: self.resolve_word(env, stack, [], w), self.function.body))
        self.expect_stack(self.signature.name, stack, self.signature.returns)
        if len(stack) != 0:
            self.abort(self.signature.name, "items left on stack at end of function")
        body = ResolvedBody(words, env.vars_by_id)
        return ResolvedFunction(self.signature, memories, locals, body)

    def resolve_word(self, env: Env, stack: Stack, break_stacks: List[List[ResolvedType]], word: ParsedWord) -> ResolvedWord:
        if isinstance(word, NumberWord):
            stack.append(PrimitiveType.I32)
            return word
        if isinstance(word, ParsedStringWord):
            word.token
            stack.append(ResolvedPtrType(PrimitiveType.I32))
            stack.append(PrimitiveType.I32)
            offset = len(self.module_resolver.data)
            self.module_resolver.data.extend(word.string)
            return StringWord(word.token, offset, len(word.string))
        if isinstance(word, ParsedCastWord):
            source_type = stack.pop()
            if source_type is None:
                self.abort(word.token, "expected a non-empty stack")
            resolved_type = self.module_resolver.resolve_type(word.taip)
            stack.append(resolved_type)
            return ResolvedCastWord(word.token, source_type, resolved_type)
        if isinstance(word, ParsedIfWord):
            if len(stack) == 0 or stack[-1] != PrimitiveType.BOOL:
                self.abort(word.token, "expected a boolean on stack")
            stack.pop()
            if_env = Env(env)
            if_stack = stack.make_child()
            def resolve_if_word(word: ParsedWord):
                return self.resolve_word(if_env, if_stack, break_stacks, word)
            else_env = Env(env)
            else_stack = stack.make_child()
            def resolve_else_word(word: ParsedWord):
                return self.resolve_word(else_env, else_stack, break_stacks, word)
            if_words = list(map(resolve_if_word, word.if_words))
            else_words = list(map(resolve_else_word, word.else_words))
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
            return ResolvedIfWord(word.token, parameters, returns, if_words, else_words)
        if isinstance(word, ParsedLoopWord):
            loop_stack = stack.make_child()
            loop_env = Env(env)
            loop_break_stacks: List[List[ResolvedType]] = []
            def resolve_loop_word(word: ParsedWord):
                return self.resolve_word(loop_env, loop_stack, loop_break_stacks, word)
            words = list(map(resolve_loop_word, word.words))
            parameters = loop_stack.negative
            if len(loop_break_stacks) != 0:
                returns = loop_break_stacks[0]
                stack.extend(loop_break_stacks[0])
            else:
                returns = loop_stack.stack
                stack.apply(loop_stack)
            return ResolvedLoopWord(word.token, words, parameters, returns)
        if isinstance(word, ParsedBlockWord):
            block_stack = stack.make_child()
            block_env = Env(env)
            block_break_stacks: List[List[ResolvedType]] = []
            def resolve_block_word(word: ParsedWord):
                return self.resolve_word(block_env, block_stack, block_break_stacks, word)
            words = list(map(resolve_block_word, word.words))
            parameters = block_stack.negative
            if len(block_break_stacks) != 0:
                for i in range(1, len(block_break_stacks)):
                    if not resolved_types_eq(block_break_stacks[0], block_break_stacks[i]):
                        self.abort(word.token, "break stack mismatch")
                if not block_stack.drained and not resolved_types_eq(block_break_stacks[0], block_stack.stack):
                    self.abort(word.token, "the items remaining on the stack at the end of the block don't match the break statements")
                returns = block_break_stacks[0]
                stack.extend(block_break_stacks[0])
            else:
                returns = block_stack.stack
                stack.apply(block_stack)
            return ResolvedBlockWord(word.token, words, parameters, returns)
        if isinstance(word, ParsedCallWord):
            if word.name.lexeme in INTRINSICS:
                intrinsic = INTRINSICS[word.name.lexeme]
                return self.resolve_intrinsic(word.name, stack, intrinsic)
            resolved_call_word = self.resolve_call_word(env, word)
            signature = self.module_resolver.get_signature(resolved_call_word.function)
            self.type_check_call(stack, resolved_call_word.name, resolved_call_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
            return resolved_call_word
        if isinstance(word, ParsedForeignCallWord):
            resolved_word = self.resolve_foreign_call_word(env, word)
            signature = self.module_resolver.get_signature(resolved_word.function)
            self.type_check_call(stack, word.name, resolved_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
            return resolved_word
        if isinstance(word, DerefWord):
            assert(False)
            return word
        if isinstance(word, ParsedGetWord):
            (var_type, local) = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var_type, word.fields)
            stack.append(var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip)
            return ResolvedGetWord(word.token, local, resolved_fields)
        if isinstance(word, ParsedInitWord):
            taip = stack.pop()
            if taip is None:
                self.abort(word.name, "expected a non-empty stack")
            named_taip = ResolvedNamedType(word.name, taip)
            local_id = env.insert(ResolvedLocal.local(named_taip))
            return InitWord(word.name, local_id)
        if isinstance(word, ParsedRefWord):
            (var_type, local) = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var_type, word.fields)
            stack.append(ResolvedPtrType(var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip))
            return ResolvedRefWord(word.token, local, resolved_fields)
        if isinstance(word, ParsedSetWord):
            (var_type, local) = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var_type, word.fields)
            expected_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
            self.expect_stack(word.token, stack, [expected_taip])
            return ResolvedSetWord(word.token, local, resolved_fields)
        if isinstance(word, ParsedStoreWord):
            (var_type, local) = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var_type, word.fields)
            expected_taip = var_type if len(resolved_fields) == 0 else resolved_fields[-1].target_taip
            if not isinstance(expected_taip, ResolvedPtrType):
                self.abort(word.token, "`=>` can only store into ptr types")
            self.expect_stack(word.token, stack, [expected_taip.child])
            return ResolvedStoreWord(word.token, local, resolved_fields)
        if isinstance(word, ParsedFunRefWord):
            if isinstance(word.call, ParsedCallWord):
                resolved_call_word = self.resolve_call_word(env, word.call)
                signature = self.module_resolver.get_signature(resolved_call_word.function)
                stack.append(ResolvedFunctionType(word.call.name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                return ResolvedFunRefWord(resolved_call_word)
            if isinstance(word.call, ParsedForeignCallWord):
                resolved_foreign_call_word = self.resolve_foreign_call_word(env, word.call)
                signature = self.module_resolver.get_signature(resolved_foreign_call_word.function)
                stack.append(ResolvedFunctionType(word.call.name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                return ResolvedFunRefWord(resolved_foreign_call_word)
            assert_never(word.call)
        if isinstance(word, ParsedLoadWord):
            if len(stack) == 0:
                self.abort(word.token, "expected a non-empty stack")
            top = stack.pop()
            if not isinstance(top, ResolvedPtrType):
                self.abort(word.token, "expected a pointer on the stack")
            stack.append(top.child)
            return ResolvedLoadWord(word.token, top.child)
        if isinstance(word, BreakWord):
            dump = stack.dump()
            break_stacks.append(dump)
            stack.drain()
            return word
        if isinstance(word, ParsedSizeofWord):
            stack.append(PrimitiveType.I32)
            return ResolvedSizeofWord(word.token, self.module_resolver.resolve_type(word.taip))
        if isinstance(word, ParsedGetFieldWord):
            taip = stack.pop()
            if taip is None:
                self.abort(word.token, "GetField expected a struct on the stack")
            resolved_fields = self.resolve_fields(taip, word.fields)
            stack.append(ResolvedPtrType(resolved_fields[-1].target_taip))
            return ResolvedGetFieldWord(word.token, taip, resolved_fields)
        if isinstance(word, ParsedIndirectCallWord):
            if len(stack) == 0:
                self.abort(word.token, "`->` expected a function on the stack")
            function_type = stack.pop()
            if not isinstance(function_type, ResolvedFunctionType):
                self.abort(word.token, "`->` expected a function on the stack")
            self.type_check_call(stack, word.token, None, function_type.parameters, function_type.returns)
            return ResolvedIndirectCallWord(word.token, function_type)
        assert_never(word)

    def resolve_intrinsic(self, token: Token, stack: Stack, intrinsic: IntrinsicType) -> ResolvedIntrinsicWord:
        match intrinsic:
            case IntrinsicType.ADD | IntrinsicType.SUB:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if isinstance(stack[-2], ResolvedPtrType):
                    if stack[-1] != PrimitiveType.I32:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, i32]")
                    stack.pop()
                if stack[-1] == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    stack.append(popped[0])
                if stack[-1] == PrimitiveType.I64:
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    stack.append(PrimitiveType.I64)
                if intrinsic == IntrinsicType.ADD:
                    return ResolvedIntrinsicAdd(token, stack[-1])
                if intrinsic == IntrinsicType.SUB:
                    return ResolvedIntrinsicSub(token, stack[-1])
            case IntrinsicType.DROP:
                if len(stack) == 0:
                    self.abort(token, "`drop` expected non empty stack")
                stack.pop()
                return IntrinsicDrop(token)
            case IntrinsicType.MOD | IntrinsicType.MUL | IntrinsicType.DIV:
                if isinstance(stack[-2], ResolvedPtrType):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if stack[-1] == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                stack.append(popped[0])
                if intrinsic == IntrinsicType.MOD:
                    return ResolvedIntrinsicMod(token, stack[-1])
                if intrinsic == IntrinsicType.MUL:
                    return ResolvedIntrinsicMul(token, stack[-1])
                if intrinsic == IntrinsicType.DIV:
                    return ResolvedIntrinsicDiv(token, stack[-1])
            case IntrinsicType.AND | IntrinsicType.OR:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                match taip:
                    case PrimitiveType.I32:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    case PrimitiveType.I64:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    case _:
                        popped = self.expect_stack(token, stack, [PrimitiveType.BOOL, PrimitiveType.BOOL])
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
                if isinstance(stack[-2], ResolvedPtrType):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                if taip == PrimitiveType.I32:
                    self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
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
            case _:
                assert_never(intrinsic)
                self.abort(token, "TODO")


    def expect_stack(self, token: Token, stack: Stack, expected: List[ResolvedType]) -> List[ResolvedType]:
        popped: List[ResolvedType] = []
        for expected_type in reversed(expected):
            top = stack.pop()
            if top is None:
                self.abort(token, "expected: " + format_resolved_type(expected_type))
            popped.append(top)
            if not resolved_type_eq(expected_type, top):
                self.abort(token, "expected: " + format_resolved_type(expected_type) + "\ngot: " + format_resolved_type(top))
        return list(reversed(popped))

    def resolve_call_word(self, env: Env, word: ParsedCallWord) -> ResolvedCallWord:
        resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
        function = self.module_resolver.resolve_function_name(word.name)
        return ResolvedCallWord(word.name, function, resolved_generic_arguments)

    def type_check_call(self, stack: Stack, token: Token, generic_arguments: None | List[ResolvedType], parameters: List[ResolvedType], returns: List[ResolvedType]):
        conrete_parameters = list(map(FunctionResolver.resolve_generic(generic_arguments), parameters)) if generic_arguments is not None else parameters
        self.expect_stack(token, stack, conrete_parameters)
        concrete_return_types = list(map(FunctionResolver.resolve_generic(generic_arguments), returns)) if generic_arguments is not None else returns
        stack.extend(concrete_return_types)

    @staticmethod
    def resolve_generic(generic_arguments: None | List[ResolvedType]) -> Callable[[ResolvedType], ResolvedType]:
        def inner(taip: ResolvedType):
            if generic_arguments is None:
                return taip
            if isinstance(taip, GenericType):
                return generic_arguments[taip.generic_index]
            if isinstance(taip, ResolvedPtrType):
                return ResolvedPtrType(inner(taip.child))
            if isinstance(taip, ResolvedStructType):
                return ResolvedStructType(taip.name, taip.struct, list(map(inner, taip.generic_arguments)))
            if isinstance(taip, ResolvedFunctionType):
                return ResolvedFunctionType(taip.token, list(map(inner, taip.parameters)), list(map(inner, taip.returns)))
            return taip
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
                for field_index, field in enumerate(self.module_resolver.get_struct(source_taip.struct).fields):
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
    structs: List[ResolvedStruct] = field(default_factory=list)
    externs: List[ResolvedExtern] = field(default_factory=list)
    memories: List[ResolvedMemory] = field(default_factory=list)
    data: bytearray = field(default_factory=bytearray)
    signatures: List[ResolvedFunctionSignature] = field(default_factory=list)

    def abort(self, token: Token, message: str) -> NoReturn:
        print(token, self.module.path + " " + message)
        assert(False)
        raise ResolverException(token, self.module.path + " " + message)

    def get_signature(self, function: ResolvedFunctionHandle | ResolvedExternHandle) -> ResolvedFunctionSignature:
        if isinstance(function, ResolvedFunctionHandle):
            if self.id == function.module:
                return self.signatures[function.index]
            else:
                return self.resolved_modules[function.module].functions[function.index].signature
        if self.id == function.module:
            return self.externs[function.index].signature
        return self.resolved_modules[function.module].externs[function.index].signature

    def get_struct(self, struct: ResolvedStructHandle) -> ResolvedStruct:
        if struct.module == self.id:
            return self.structs[struct.index]
        return self.resolved_modules[struct.module].structs[struct.index]

    def resolve(self) -> ResolvedModule:
        resolved_imports = list(map(self.resolve_import, self.module.imports))
        self.imports = resolved_imports
        resolved_structs = list(map(self.resolve_struct, self.module.structs))
        self.structs = resolved_structs
        self.memories = list(map(self.resolve_memory, self.module.memories))
        resolved_externs = list(map(self.resolve_extern, self.module.externs))
        self.externs = resolved_externs
        resolved_signatures = list(map(lambda f: self.resolve_function_signature(f.signature), self.module.functions))
        self.signatures = resolved_signatures
        resolved_functions = list(map(lambda f: self.resolve_function(f[0], f[1]), zip(resolved_signatures, self.module.functions)))
        return ResolvedModule(self.module.path, self.id, resolved_imports, resolved_structs, resolved_externs, self.memories, resolved_functions, self.data)

    def resolve_function(self, signature: ResolvedFunctionSignature, function: ParsedFunction) -> ResolvedFunction:
        return FunctionResolver(self, self.externs, self.signatures, self.structs, function, signature).resolve()

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
        if isinstance(taip, PrimitiveType):
            return taip
        if isinstance(taip, ParsedPtrType):
            return ResolvedPtrType(self.resolve_type(taip.child))
        if isinstance(taip, ParsedStructType):
            resolved_generic_arguments = list(map(self.resolve_type, taip.generic_arguments))
            return ResolvedStructType(taip.name, self.resolve_struct_name(taip.name), resolved_generic_arguments)
        if isinstance(taip, GenericType):
            return taip
        if isinstance(taip, ParsedForeignType):
            resolved_generic_arguments = list(map(self.resolve_type, taip.generic_arguments))
            for imp in self.imports:
                if imp.qualifier.lexeme == taip.module.lexeme:
                    for index, struct in enumerate(self.resolved_modules[imp.module].structs):
                        if struct.name.lexeme == taip.name.lexeme:
                            return ResolvedStructType(taip.name, ResolvedStructHandle(imp.module, index), resolved_generic_arguments)
            self.abort(taip.module, f"struct {taip.module.lexeme}:{taip.name.lexeme} not found")
        if isinstance(taip, ParsedFunctionType):
            args = list(map(self.resolve_type, taip.args))
            rets = list(map(self.resolve_type, taip.rets))
            return ResolvedFunctionType(taip.token, args, rets)
        return assert_never(taip)

    def resolve_struct_name(self, name: Token) -> ResolvedStructHandle:
        for index, struct in enumerate(self.module.structs):
            if struct.name.lexeme == name.lexeme:
                return ResolvedStructHandle(self.id, index)
        self.abort(name, f"struct {name.lexeme} not found")

    def resolve_struct(self, struct: ParsedStruct) -> ResolvedStruct:
        return ResolvedStruct(struct.name, list(map(self.resolve_named_type, struct.fields)))

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
    if isinstance(a, PrimitiveType):
        return str(a)
    if isinstance(a, PtrType):
        return f".{format_type(a.child)}"
    if isinstance(a, StructType):
        return a.name.lexeme 
    if isinstance(a, FunctionType):
        s = "("
        for param in a.parameters:
            s += format_type(param) + ", "
        s = s[:-2] + " -> "
        if len(a.returns) == 0:
            return s[:-1] + ")"
        for ret in a.returns:
            s += format_type(ret) + ", "
        return s[:-2] + ")"
    assert_never(a)

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

@dataclass
class Local:
    name: Token
    taip: Type
    ty: LocalType
    _size: int = 0 # only used in case of self.ty == LocalType.MEMORY

    def size(self) -> int:
        if self.ty != LocalType.MEMORY:
            return self.taip.size()
        return self._size

@dataclass
class Body:
    words: List['Word']
    locals: Dict[LocalId, Local]

@dataclass
class FunctionSignature:
    export_name: Optional[Token]
    name: Token
    generic_arguments: List[Type]
    parameters: List[NamedType]
    returns: List[Type]

    def returns_any_struct(self) -> bool:
        for ret in self.returns:
            if isinstance(ret, StructType):
                return True
        return False

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

@dataclass
class CastWord:
    token: Token
    source: Type
    taip: Type

@dataclass
class LoadWord:
    token: Token
    taip: Type

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
    base_taip: Type
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

@dataclass
class RefWord:
    token: Token
    local_id: LocalId | GlobalId
    fields: List[FieldAccess]

@dataclass
class IntrinsicAdd:
    token: Token
    taip: Type

@dataclass
class IntrinsicSub:
    token: Token
    taip: Type

@dataclass
class IntrinsicMul:
    token: Token
    taip: Type

@dataclass
class IntrinsicDiv:
    token: Token
    taip: Type

@dataclass
class IntrinsicMod:
    token: Token
    taip: Type

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
    taip: Type

@dataclass
class IntrinsicNot:
    token: Token
    taip: Type

@dataclass
class IntrinsicGreaterEq:
    token: Token
    taip: Type

@dataclass
class IntrinsicLessEq:
    token: Token
    taip: Type

@dataclass
class IntrinsicGreater:
    token: Token
    taip: Type

@dataclass
class IntrinsicLess:
    token: Token
    taip: Type

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
class StoreWord:
    token: Token
    local: LocalId | GlobalId
    fields: List[FieldAccess]

IntrinsicWord = IntrinsicAdd | IntrinsicSub | IntrinsicEqual | IntrinsicNotEqual | IntrinsicAnd | IntrinsicDrop | IntrinsicLoad8 | IntrinsicStore8 | IntrinsicGreaterEq | IntrinsicLessEq | IntrinsicMul | IntrinsicMod | IntrinsicDiv | IntrinsicGreater | IntrinsicLess | IntrinsicFlip | IntrinsicRotl | IntrinsicRotr | IntrinsicOr | IntrinsicStore | IntrinsicMemCopy | IntrinsicMemGrow | IntrinsicNot

Word = NumberWord | StringWord | CallWord | GetWord | InitWord | CastWord | SetWord | LoadWord | IntrinsicWord | IfWord | RefWord | IndirectCallWord | StoreWord | FunRefWord | LoopWord | BreakWord | SizeofWord | BlockWord | GetFieldWord

@dataclass
class Module:
    id: int
    # imports: List[Import]
    structs: Dict[int, List[Struct]]
    externs: List[Extern]
    memories: List[Memory]
    functions: Dict[int, Function]
    data: bytes

@dataclass
class Monomizer:
    modules: Dict[int, ResolvedModule]
    structs: Dict[int, Dict[int, List[Tuple[List[Type], Struct]]]] = field(default_factory=dict)
    functions: Dict[int, Dict[int, Function]] = field(default_factory=dict)
    function_table: Dict[FunctionHandle | ExternHandle, int] = field(default_factory=dict)

    def monomize(self) -> Tuple[Dict[FunctionHandle | ExternHandle, int], Dict[int, Module]]:
        for id in sorted(self.modules):
            module = self.modules[id]
            functions: List[Function] = []
            for index, function in enumerate(module.functions):
                if function.signature.export_name is not None:
                    assert(len(function.signature.generic_parameters) == 0)
                    signature = self.monomize_concrete_signature(function.signature)
                    memories = list(map(lambda m: self.monomize_memory(m, []), function.memories))
                    locals = list(map(lambda t: self.monomize_named_type(t, []), function.locals))
                    body = Lazy(lambda: Body(self.monomize_words(function.body.words, []), self.monomize_locals(function.body.locals, [])))
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
            if module_id not in self.structs:
                self.structs[module_id] = {}
            if module_id not in self.functions:
                self.functions[module_id] = {}
            module = self.modules[module_id]
            externs: List[Extern] = list(map(self.monomize_extern, module.externs))
            structs: Dict[int, List[Struct]] = { k: [t[1] for t in v] for k, v in self.structs[module_id].items() }
            memories = list(map(lambda m: self.monomize_memory(m, []), module.memories))
            mono_modules[module_id] = Module(module_id, structs, externs, memories, self.functions[module_id], self.modules[module_id].data)
        return self.function_table, mono_modules

    def monomize_locals(self, locals: Dict[LocalId, ResolvedLocal], generics: List[Type]) -> Dict[LocalId, Local]:
        res = {}
        for id, local in locals.items():
            taip = self.monomize_type(local.taip, generics)
            res[id] = Local(local.name, taip, local.ty, local.size or taip.size())
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
        body = Lazy(lambda: Body(list(map(lambda w: self.monomize_word(w, generics), f.body.words)), self.monomize_locals(f.body.locals, generics)))
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

    def monomize_words(self, words: List[ResolvedWord], generics: List[Type]) -> List[Word]:
        return list(map(lambda w: self.monomize_word(w, generics), words))

    def monomize_word(self, word: ResolvedWord, generics: List[Type]) -> Word:
        if isinstance(word, NumberWord):
            return word
        if isinstance(word, StringWord):
            return word
        if isinstance(word, ResolvedCallWord):
            return self.monomize_call_word(word, generics)
        if isinstance(word, ResolvedGetWord):
            fields = self.monomize_field_accesses(word.fields, generics)
            return GetWord(word.token, word.local_id, fields)
        if isinstance(word, InitWord):
            return word
        if isinstance(word, ResolvedSetWord):
            fields = self.monomize_field_accesses(word.fields, generics)
            return SetWord(word.token, word.local_id, fields)
        if isinstance(word, ResolvedRefWord):
            fields = self.monomize_field_accesses(word.fields, generics)
            return RefWord(word.token, word.local_id, fields)
        if isinstance(word, ResolvedStoreWord):
            fields = self.monomize_field_accesses(word.fields, generics)
            return StoreWord(word.token, word.local, fields)
        if isinstance(word, ResolvedIntrinsicAdd):
            return IntrinsicAdd(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicSub):
            return IntrinsicSub(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicMul):
            return IntrinsicMul(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicMod):
            return IntrinsicMod(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicDiv):
            return IntrinsicDiv(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicEqual):
            return IntrinsicEqual(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicNotEqual):
            return IntrinsicNotEqual(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicGreaterEq):
            return IntrinsicGreaterEq(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicGreater):
            return IntrinsicGreater(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicLess):
            return IntrinsicLess(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicLessEq):
            return IntrinsicLessEq(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicAnd):
            return IntrinsicAnd(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicNot):
            return IntrinsicNot(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, IntrinsicDrop):
            return word
        if isinstance(word, BreakWord):
            return word
        if isinstance(word, IntrinsicFlip):
            return word
        if isinstance(word, IntrinsicLoad8):
            return word
        if isinstance(word, IntrinsicStore8):
            return word
        if isinstance(word, ResolvedIntrinsicRotl):
            return IntrinsicRotl(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicRotr):
            return IntrinsicRotr(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicOr):
            return IntrinsicOr(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIntrinsicStore):
            return IntrinsicStore(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, IntrinsicMemCopy):
            return word
        if isinstance(word, IntrinsicMemGrow):
            return word
        if isinstance(word, ResolvedLoadWord):
            return LoadWord(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedCastWord):
            return CastWord(word.token, self.monomize_type(word.source, generics), self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedIfWord):
            if_words = self.monomize_words(word.if_words, generics)
            else_words = self.monomize_words(word.else_words, generics)
            parameters = list(map(lambda t: self.monomize_type(t, generics), word.parameters))
            returns = list(map(lambda t: self.monomize_type(t, generics), word.returns))
            return IfWord(word.token, parameters, returns, if_words, else_words)
        if isinstance(word, ResolvedIndirectCallWord):
            return IndirectCallWord(word.token, self.monomize_function_type(word.taip, generics))
        if isinstance(word, ResolvedFunRefWord):
            call_word = self.monomize_call_word(word.call, generics)
            table_index = self.insert_function_into_table(call_word.function)
            return FunRefWord(call_word, table_index)
        if isinstance(word, ResolvedLoopWord):
            words = self.monomize_words(word.words, generics)
            parameters = list(map(lambda t: self.monomize_type(t, generics), word.parameters))
            returns = list(map(lambda t: self.monomize_type(t, generics), word.returns))
            return LoopWord(word.token, words, parameters, returns)
        if isinstance(word, ResolvedSizeofWord):
            return SizeofWord(word.token, self.monomize_type(word.taip, generics))
        if isinstance(word, ResolvedBlockWord):
            words = self.monomize_words(word.words, generics)
            parameters = list(map(lambda t: self.monomize_type(t, generics), word.parameters))
            returns = list(map(lambda t: self.monomize_type(t, generics), word.returns))
            return BlockWord(word.token, words, parameters, returns)
        if isinstance(word, ResolvedGetFieldWord):
            fields = self.monomize_field_accesses(word.fields, generics)
            return GetFieldWord(word.token, self.monomize_type(word.base_taip, generics), fields)
        # assert_never(word)
        print(word, file=sys.stderr)
        assert(False)

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
        target_taip = self.monomize_type(field.target_taip, struct.generic_parameters)
        offset = struct.field_offset(field.field_index)
        return [FieldAccess(field.name, source_taip, target_taip, offset)] + self.monomize_field_accesses(fields[1:], struct.generic_parameters)

    def monomize_call_word(self, word: ResolvedCallWord, generics: List[Type]) -> CallWord:
        if isinstance(word.function, ResolvedExternHandle):
            return CallWord(word.name, ExternHandle(word.function.module, word.function.index))
        generics_here = list(map(lambda t: self.monomize_type(t, generics), word.generic_arguments))
        if word.function.module not in self.functions:
            self.functions[word.function.module] = {}
        if word.function.index in self.functions[word.function.module]:
            function = self.functions[word.function.module][word.function.index]
            if isinstance(function, ConcreteFunction):
                assert(len(word.generic_arguments) == 0)
                return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, None))
            for instance_index, (instance_generics, instance) in enumerate(function.instances):
                if types_eq(instance_generics, generics_here):
                    return CallWord(word.name, FunctionHandle(word.function.module, word.function.index, instance_index))
        self.monomize_function(word.function, generics_here)
        return self.monomize_call_word(word, generics) # the function instance should now exist, try monomorphizing this CallWord again

    def lookup_struct(self, struct: ResolvedStructHandle, generics: List[Type]) -> Tuple[StructHandle, Struct] | None:
        if struct.module not in self.structs:
            self.structs[struct.module] = {}
        if struct.index not in self.structs[struct.module]:
            return None
        for instance_index, (genics, instance) in enumerate(self.structs[struct.module][struct.index]):
            if types_eq(genics, generics):
                return StructHandle(struct.module, struct.index, instance_index), instance
        return None

    def add_struct(self, module: int, index: int, struct: Struct, generics: List[Type]) -> StructHandle:
        if module not in self.structs:
            self.structs[module] = {}
        if index not in self.structs[module]:
            self.structs[module][index] = []
        instance_index = len(self.structs[module][index])
        self.structs[module][index].append((generics, struct))
        return StructHandle(module, index, instance_index)

    def monomize_struct(self, struct: ResolvedStructHandle, generics: List[Type]) -> Tuple[StructHandle, Struct]:
        handle_and_instance = self.lookup_struct(struct, generics)
        if handle_and_instance is not None:
            return handle_and_instance
        s = self.modules[struct.module].structs[struct.index]
        fields: List[NamedType] = []
        struct_instance = Struct(s.name, fields, generics)
        handle = self.add_struct(struct.module, struct.index, struct_instance, generics)
        for field in map(lambda t: self.monomize_named_type(t, generics), s.fields):
            fields.append(field)
        return handle, struct_instance

    def monomize_named_type(self, taip: ResolvedNamedType, generics: List[Type]) -> NamedType:
        return NamedType(taip.name, self.monomize_type(taip.taip, generics))

    def monomize_type(self, taip: ResolvedType, generics: List[Type]) -> Type:
        if isinstance(taip, PrimitiveType):
            return taip
        if isinstance(taip, ResolvedPtrType):
            return PtrType(self.monomize_type(taip.child, generics))
        if isinstance(taip, GenericType):
            return generics[taip.generic_index]
        if isinstance(taip, ResolvedStructType):
            return self.monomize_struct_type(taip, generics)
        if isinstance(taip, ResolvedFunctionType):
            return self.monomize_function_type(taip, generics)
        assert_never(taip)

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

    def lookup_struct(self, handle: StructHandle) -> Struct:
        return self.modules[handle.module].structs[handle.index][handle.instance]

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

        self.write_line("(memory 1 65536)\n")
        self.write_line("(export \"memory\" (memory 0))\n")

        all_data: bytes = b""
        for id in sorted(self.modules):
            self.module_data_offsets[id] = len(all_data)
            all_data += self.modules[id].data

        self.write_intrinsics()

        self.write_function_table()

        data_end = align_to(len(all_data), 4)
        global_mem = self.write_globals(data_end)
        stack_start = align_to(data_end + global_mem, 4)
        self.write_line(f"(global $stac:k (mut i32) (i32.const {stack_start}))\n")

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
        self.write_indent()
        self.write("(")
        self.write_signature(module, function.signature, instance_id)
        if len(function.signature.generic_arguments) > 0:
            self.write(" ;; ")
        for taip in function.signature.generic_arguments:
            self.write_type_human(taip)
            self.write(" ")
        self.write_line("")
        self.indent()
        self.write_locals(function.body.get())
        struct_return_space, max_struct_arg_count = self.measure_struct_return_space(function.body.get().words)
        if struct_return_space != 0:
            for i in range(0, max_struct_arg_count):
                self.write_indent()
                self.write(f"(local $s{i}:a i32)\n")
        locals_copy_space = self.measure_locals_copy_space(function.body.get().locals, function.body.get().words)
        if locals_copy_space != 0:
            self.write_indent()
            self.write("(local $locl-copy-spac:e i32)\n")

        uses_stack = struct_return_space != 0 or locals_copy_space != 0 or any(isinstance(local.taip, StructType) or local.ty == LocalType.MEMORY for local in function.body.get().locals.values())
        if uses_stack:
            self.write_indent()
            self.write("(local $stac:k i32)\n")
            if struct_return_space > 0:
                self.write_indent()
                self.write("(local $struc-return-spac:e i32)\n")
                self.write_indent()
                self.write("global.get $stac:k local.set $stac:k\n")
                self.write_mem("struc-return-spac:e", struct_return_space, 0, 0)
            else:
                self.write_indent()
                self.write("global.get $stac:k local.set $stac:k\n")

        for local_id, local in function.body.get().locals.items():
            if local.ty == LocalType.MEMORY:
                self.write_mem(local.name.lexeme, local.size(), local_id.scope, local_id.shadow)
        if locals_copy_space != 0:
            self.write_mem("locl-copy-spac:e", locals_copy_space, 0, 0)
        self.write_structs(function.body.get().locals)
        self.write_body(module, Ref(0), Ref(0), function.body.get())
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
            if local.ty != LocalType.PARAMETER and isinstance(local.taip, StructType):
                self.write_mem(local.name.lexeme, local.taip.size(), local_id.scope, local_id.shadow)

    def measure_struct_return_space(self, words: List[Word]) -> Tuple[int, int]:
        size = 0
        max_struct_arg_count = 0
        for word in words:
            if isinstance(word, CallWord):
                if isinstance(word.function, FunctionHandle):
                    returns = self.lookup_function(word.function).signature.returns
                else:
                    returns = self.lookup_extern(word.function).signature.returns

                for ret in returns:
                    if isinstance(ret, StructType):
                        size += ret.size()
                        max_struct_arg_count = max(max_struct_arg_count, len(returns))
            if isinstance(word, LoopWord):
                s, sc = self.measure_struct_return_space(word.words)
                size += s
                max_struct_arg_count = max(max_struct_arg_count, sc)
            if isinstance(word, BlockWord):
                s, sc = self.measure_struct_return_space(word.words)
                size += s
                max_struct_arg_count = max(max_struct_arg_count, sc)
            if isinstance(word, IfWord):
                s, sc = self.measure_struct_return_space(word.if_words)
                size += s
                max_struct_arg_count = max(max_struct_arg_count, sc)
                s, sc = self.measure_struct_return_space(word.else_words)
                size += s
                max_struct_arg_count = max(max_struct_arg_count, sc)
        return size, max_struct_arg_count

    def measure_locals_copy_space(self, locals: Dict[LocalId, Local], words: List[Word]) -> int:
        size = 0
        for word in words:
            if isinstance(word, GetWord):
                if isinstance(word.local_id, GlobalId):
                    target_taip = word.fields[-1].target_taip if len(word.fields) > 0 else self.globals[word.local_id].taip.taip
                else:
                    local = locals[word.local_id]
                    target_taip = word.fields[-1].target_taip if len(word.fields) > 0 else local.taip
                if isinstance(target_taip, StructType):
                    size += target_taip.size()
                    continue
            if isinstance(word, LoadWord):
                size += word.taip.size()
                continue
            if isinstance(word, LoopWord):
                size += self.measure_locals_copy_space(locals, word.words)
                continue
            if isinstance(word, BlockWord):
                size += self.measure_locals_copy_space(locals, word.words)
                continue
            if isinstance(word, IfWord):
                size += self.measure_locals_copy_space(locals, word.if_words)
                size += self.measure_locals_copy_space(locals, word.else_words)
                continue
        return size

    def write_locals(self, body: Body) -> None:
        for local_id, local in body.locals.items():
            if local.ty == LocalType.PARAMETER: # is a parameter
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

    def write_body(self, module: int, copy_space_offset: Ref[int], struct_return_space: Ref[int], body: Body) -> None:
        self.write_words(module, body.locals, copy_space_offset, struct_return_space, body.words)

    def write_words(self, module: int, locals: Dict[LocalId, Local], copy_space_offset: Ref[int], struct_return_space: Ref[int], words: List[Word]) -> None:
        for word in words:
            self.write_indent()
            self.write_word(module, locals, copy_space_offset, struct_return_space, word)
            self.write("\n")

    def write_word(self, module: int, locals: Dict[LocalId, Local], copy_space_offset: Ref[int], struct_return_space: Ref[int], word: Word) -> None:
        if isinstance(word, NumberWord):
            self.write(f"i32.const {word.token.lexeme}")
            return
        if isinstance(word, GetWord):
            if isinstance(word.local_id, GlobalId):
                target_taip = word.fields[-1].target_taip if len(word.fields) > 0 else self.globals[word.local_id].taip.taip
            if isinstance(word.local_id, LocalId):
                local = locals[word.local_id]
                target_taip = word.fields[-1].target_taip if len(word.fields) > 0 else local.taip
            if isinstance(target_taip, StructType):
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset.value} i32.add call $intrinsic:dupi32\n")
                self.write_indent()
                copy_space_offset.value += target_taip.size()
            if isinstance(word.local_id, GlobalId):
                self.write(f"global.get ${word.token.lexeme}:{word.local_id.module}")
            else:
                self.write(f"local.get ${word.token.lexeme}")
                if word.local_id.scope != 0 or word.local_id.shadow != 0:
                    self.write(f":{word.local_id.scope}:{word.local_id.shadow}")
            loads = self.determine_loads(word.fields)
            if len(loads) == 0 and isinstance(target_taip, StructType):
                self.write(f" i32.const {target_taip.size()} memory.copy")

            for i, load in enumerate(loads):
                if i + 1 < len(loads) or not isinstance(target_taip, StructType):
                    self.write(f" i32.load offset={load} ")
                else:
                    self.write(f" i32.const {load} i32.add i32.const {target_taip.size()} memory.copy")
            return
            assert_never(word.local_id)
        if isinstance(word, GetFieldWord):
            assert(word.fields != 0)
            loads = self.determine_loads(word.fields)
            for i, load in enumerate(loads):
                if i + 1 == len(loads):
                    self.write(f" i32.const {load} i32.add")
                else:
                    self.write(f" i32.load offset={load}")
            return
        if isinstance(word, RefWord):
            if isinstance(word.local_id, GlobalId):
                self.write(f"global.get ${word.token.lexeme}:{word.local_id.module}")
            if isinstance(word.local_id, LocalId):
                self.write(f"local.get ${word.token.lexeme}")
                if word.local_id.scope != 0 or word.local_id.shadow != 0:
                    self.write(f":{word.local_id.scope}:{word.local_id.shadow}")
            loads = self.determine_loads(word.fields)
            for i, load in enumerate(loads):
                if i + 1 == len(loads):
                    self.write(f" i32.const {load} i32.add")
                else:
                    self.write(f" i32.load offset={load}")
            return
            assert_never(word.local_id)
        if isinstance(word, SetWord):
            if isinstance(word.local_id, GlobalId):
                self.write(str(word.__class__))
                return
            self.write_set(word.local_id, locals, word.fields)
            return
        if isinstance(word, InitWord):
            self.write_set(word.local_id, locals, [])
            return
        if isinstance(word, CallWord):
            if isinstance(word.function, ExternHandle):
                extern = self.lookup_extern(word.function)
                self.write(f"call ${word.function.module}:{word.name.lexeme}")
                if extern.signature.returns_any_struct():
                    self.write("CallWord which returns Struct TODO")
                    return
                return
            if isinstance(word.function, FunctionHandle):
                function = self.lookup_function(word.function)
                if isinstance(function, GenericFunction):
                    assert(word.function.instance is not None)
                    function = function.instances[word.function.instance][1]
                self.write(f"call ${word.function.module}:{function.signature.name.lexeme}")
                if word.function.instance is not None:
                    self.write(f":{word.function.instance}")
                self.write_return_struct_receiving(struct_return_space, function.signature.returns)
                return
        if isinstance(word, IndirectCallWord):
            self.write("(call_indirect ")
            self.write_parameters(word.taip.parameters)
            self.write_returns(word.taip.returns)
            self.write(")\n")
            self.write_return_struct_receiving(struct_return_space, word.taip.returns)
            return
        if isinstance(word, IntrinsicStore):
            if isinstance(word.taip, StructType):
                self.write(f"i32.const {word.taip.size()} memory.copy")
            else:
                self.write("i32.store")
            return
        if isinstance(word, IntrinsicAdd):
            if isinstance(word.taip, PtrType) or word.taip == PrimitiveType.I32:
                self.write(f"i32.add")
                return
            if word.taip == PrimitiveType.I64:
                self.write(f"i64.add")
                return
            self.write(str(word))
            return
        if isinstance(word, IntrinsicSub):
            self.write(f"{str(word.taip)}.sub")
            return
        if isinstance(word, IntrinsicMul):
            self.write(f"{str(word.taip)}.mul")
            return
        if isinstance(word, IntrinsicDrop):
            self.write("drop")
            return
        if isinstance(word, IntrinsicOr):
            self.write_type(word.taip)
            self.write(".or")
            return
        if isinstance(word, IntrinsicEqual):
            if isinstance(word.taip, StructType):
                self.write("IntrinsicEqual for struct type TODO")
                return
            if word.taip == PrimitiveType.I64:
                self.write("i64.eq")
                return
            self.write("i32.eq")
            return
        if isinstance(word, IntrinsicNotEqual):
            if isinstance(word.taip, StructType):
                self.write("IntrinsicNotEqual for struct type TODO")
                return
            if word.taip == PrimitiveType.I64:
                self.write("i64.eq")
                return
            self.write("i32.ne")
            return
        if isinstance(word, IntrinsicGreaterEq):
            if word.taip == PrimitiveType.I32:
                self.write("i32.ge_u")
                return
            self.write(str(word.__class__))
            return
        if isinstance(word, IntrinsicGreater):
            if word.taip == PrimitiveType.I32:
                self.write("i32.gt_u")
                return
            self.write(str(word.__class__))
            return
        if isinstance(word, IntrinsicLessEq):
            if word.taip == PrimitiveType.I32:
                self.write("i32.le_u")
                return
            self.write(str(word.__class__))
            return
        if isinstance(word, IntrinsicLess):
            if word.taip == PrimitiveType.I32:
                self.write("i32.lt_u")
                return
            self.write(str(word.__class__))
            return
        if isinstance(word, IntrinsicFlip):
            self.write("call $intrinsic:flip")
            return
        if isinstance(word, IntrinsicRotl):
            if word.taip == PrimitiveType.I64:
                self.write("i64.extend_i32_s i64.rotl")
            else:
                self.write("i32.rotl")
            return
        if isinstance(word, IntrinsicRotr):
            if word.taip == PrimitiveType.I64:
                self.write("i64.extend_i32_s i64.rotr")
            else:
                self.write("i32.rotr")
            return
        if isinstance(word, IntrinsicAnd):
            if word.taip == PrimitiveType.I32 or word.taip == PrimitiveType.BOOL:
                self.write("i32.and")
                return
            if word.taip == PrimitiveType.I64:
                self.write("i64.and")
                return
            self.write("IntrinsicAnd for non i64 or i32 type TODO")
            return
        if isinstance(word, IntrinsicNot):
            if word.taip == PrimitiveType.BOOL:
                self.write("i32.const 1 i32.and i32.const 1 i32.xor i32.const 1 i32.and")
                return
            if word.taip == PrimitiveType.I32:
                self.write("i32.const -1 i32.xor")
                return
            if word.taip == PrimitiveType.I64:
                self.write("i64.const -1 i64.xor")
                return
            assert(False)
        if isinstance(word, IntrinsicLoad8):
            self.write("i32.load8_u")
            return
        if isinstance(word, IntrinsicStore8):
            self.write("i32.store8")
            return
        if isinstance(word, IntrinsicMod):
            self.write("i32.rem_u")
            return
        if isinstance(word, IntrinsicDiv):
            self.write("i32.div_u")
            return
        if isinstance(word, IntrinsicMemCopy):
            self.write("memory.copy")
            return
        if isinstance(word, IntrinsicMemGrow):
            self.write("memory.grow")
            return
        if isinstance(word, CastWord):
            if (word.source == PrimitiveType.BOOL or word.source == PrimitiveType.I32) and word.taip == PrimitiveType.I64:
                self.write(f"i64.extend_i32_s ;; cast to {format_type(word.taip)}")
                return
            if word.source == PrimitiveType.I64 and word.taip != PrimitiveType.I64:
                self.write(f"i32.wrap_i64 ;; cast to {format_type(word.taip)}")
                return
            self.write(f";; cast to {format_type(word.taip)}")
            return
        if isinstance(word, StringWord):
            self.write(f"i32.const {self.module_data_offsets[module] + word.offset} i32.const {word.len}")
            return
        if isinstance(word, SizeofWord):
            self.write(f"i32.const {word.taip.size()}")
            return
        if isinstance(word, FunRefWord):
            self.write(f"i32.const {word.table_index}")
            return
        if isinstance(word, StoreWord):
            if isinstance(word.local, GlobalId):
                self.write(f"global.get ${word.token.lexeme}:{word.local.module}")
                target_type = word.fields[-1].target_taip if len(word.fields) > 0 else self.globals[word.local].taip.taip
            else:
                self.write(f"local.get ${word.token.lexeme}")
                if word.local.scope != 0 or word.local.shadow != 0:
                    self.write(f":{word.local.scope}:{word.local.shadow}")
                target_type = locals[word.local].taip
                assert(isinstance(target_type, PtrType))
                target_type = word.fields[-1].target_taip if len(word.fields) > 0 else target_type.child
            loads = self.determine_loads(word.fields)
            for offset in loads:
                self.write(f" i32.load offset={offset}")
            self.write(" call $intrinsic:flip ")
            if isinstance(target_type, StructType):
                self.write(f" i32.const {target_type.size()} memory.copy")
            else:
                self.write_type(target_type)
                self.write(".store")
            return
        if isinstance(word, LoadWord):
            if isinstance(word.taip, StructType):
                self.write(f"local.get $locl-copy-spac:e i32.const {copy_space_offset.value}")
                copy_space_offset.value += word.taip.size()
                self.write(f" i32.add call $intrinsic:dupi32 call $intrinsic:rotate-left i32.const {word.taip.size()} memory.copy")
            else:
                self.write_type(word.taip)
                self.write(".load")
            return
        if isinstance(word, BreakWord):
            self.write("br $block\n")
            return
        if isinstance(word, BlockWord):
            self.write("(block $block ")
            self.write_returns(word.returns)
            self.write("\n")
            self.indent()
            self.write_words(module, locals, copy_space_offset, struct_return_space, word.words)
            self.dedent()
            self.write_indent()
            self.write(")\n")
            return
        if isinstance(word, LoopWord):
            self.write("(block $block ")
            self.write_returns(word.returns)
            self.write("\n")
            self.indent()
            self.write_indent()
            self.write("(loop $loop ")
            self.write_returns(word.returns)
            self.write("\n")
            self.indent()
            self.write_words(module, locals, copy_space_offset, struct_return_space, word.words)
            self.write_indent()
            self.write("br $loop\n")
            self.dedent()
            self.write_indent()
            self.write(")\n")
            self.dedent()
            self.write_indent()
            self.write(")\n")
            return
        if isinstance(word, IfWord):
            self.write("(if ")
            self.write_parameters(word.parameters)
            self.write_returns(word.returns)
            self.write("\n")
            self.indent()
            self.write_indent()
            self.write("(then\n")
            self.indent()
            self.write_words(module, locals, copy_space_offset, struct_return_space, word.if_words)
            self.dedent()
            self.write_indent()
            self.write(")\n")
            if len(word.else_words) > 0:
                self.write_indent()
                self.write("(else\n")
                self.indent()
                self.write_words(module, locals, copy_space_offset, struct_return_space, word.else_words)
                self.dedent()
                self.write_indent()
                self.write(")")
            self.write("\n")
            self.dedent()
            self.write_indent()
            self.write(")\n")
            return
        assert_never(word)

    def write_set(self, local_id: LocalId, locals: Dict[LocalId, Local], fields: List[FieldAccess]):
        local = locals[local_id]
        loads = self.determine_loads(fields)
        target_taip = fields[-1].target_taip if len(fields) != 0 else local.taip
        if not isinstance(target_taip, StructType) and len(loads) == 0:
            self.write(f"local.set ${local.name.lexeme}")
            if local_id.scope != 0 or local_id.shadow != 0:
                self.write(f":{local_id.scope}:{local_id.shadow}")
                return
            return
        if isinstance(target_taip, StructType) and len(loads) == 0:
            self.write(f"local.get ${local.name.lexeme}")
            if local_id.scope != 0 or local_id.shadow != 0:
                self.write(f":{local_id.scope}:{local_id.shadow}")
            self.write(f" call $intrinsic:flip i32.const {target_taip.size()} memory.copy")
            return
        self.write(f"local.get ${local.name.lexeme}")
        if local_id.scope != 0 or local_id.shadow != 0:
            self.write(f":{local_id.scope}:{local_id.shadow}")
        if not isinstance(target_taip, StructType):
            for i, load in enumerate(loads):
                self.write(f" i32.const {load} i32.add ")
                if i + 1 == len(loads):
                    self.write(" call $intrinsic:flip ")
                    self.write_type(local.taip)
                    self.write(".store")
                else:
                    self.write("i32.load")
            return
        for i, load in enumerate(loads):
            self.write(f" i32.const {load} i32.add ")
            if i + 1 == len(loads):
                self.write(f" call $intrinsic:flip i32.const {target_taip.size()} memory.copy")
            else:
                self.write("i32.load")
        return


    def write_return_struct_receiving(self, struct_return_space: Ref[int], returns: List[Type]) -> None:
        if not any(isinstance(t, StructType) for t in returns):
            return
        for i in range(0, len(returns)):
            self.write_indent()
            self.write(f"local.set $s{i}:a\n")
        self.write_indent()
        for i in range(len(returns), 0, -1):
            ret = returns[len(returns) - i]
            if isinstance(ret, StructType):
                self.write(f"local.get $struc-return-spac:e i32.const {struct_return_space.value} i32.add call $intrinsic:dupi32 local.get $s{i - 1}:a i32.const {ret.size()} memory.copy\n")
                struct_return_space.value += ret.size()
            else:
                self.write(f"local.get $s{i - 1}:a\n")

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
        self.write(" ")
        self.write_parameters(signature.parameters)
        self.write(" ")
        self.write_returns(signature.returns)

    def write_type_human(self, taip: Type) -> None:
        self.write(format_type(taip))

    def write_parameters(self, parameters: Sequence[NamedType | Type]) -> None:
        for parameter in parameters:
            if isinstance(parameter, NamedType):
                self.write(f"(param ${parameter.name.lexeme} ")
                self.write_type(parameter.taip)
                self.write(") ")
                continue
            self.write(f"(param ")
            self.write_type(parameter)
            self.write(") ")

    def write_returns(self, returns: List[Type]) -> None:
        for taip in returns:
            self.write(f"(result ")
            self.write_type(taip)
            self.write(") ")

    def write_intrinsics(self) -> None:
        self.write_line("(func $intrinsic:flip (param $a i32) (param $b i32) (result i32 i32) local.get $b local.get $a)")
        self.write_line("(func $intrinsic:dupi32 (param $a i32) (result i32 i32) local.get $a local.get $a)")
        self.write_line("(func $intrinsic:rotate-left (param $a i32) (param $b i32) (param $c i32) (result i32 i32 i32) local.get $b local.get $c local.get $a)")
        self.write_line("(func $intrinsic:rotate-right (param $a i32) (param $b i32) (param $c i32) (result i32 i32 i32) local.get $c local.get $a local.get $b)")

    def write_function_table(self) -> None:
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


def main() -> None:
    modules: Dict[str, ParsedModule] = {}
    load_recursive(modules, os.path.normpath(sys.argv[1]))

    resolved_modules: Dict[int, ResolvedModule] = {}
    resolved_modules_by_path: Dict[str, ResolvedModule] = {}
    try:
        for id, module in enumerate(determine_compilation_order(list(modules.values()))):
            resolved_module = ModuleResolver(resolved_modules, resolved_modules_by_path, module, id).resolve()
            resolved_modules[id] = resolved_module
            resolved_modules_by_path[module.path] = resolved_module
        function_table, mono_modules = Monomizer(resolved_modules).monomize()
        print(WatGenerator(mono_modules, function_table).write_wat_module())
    except ResolverException as e:
        print(e.token, file=sys.stderr)
        print(e.message, file=sys.stderr)
        exit(1)


if __name__ == "__main__":
    main()
    exit(0)
    for path in sys.argv[1:]:
        with open(path, 'r') as reader:
            input = reader.read()
            tokens = Lexer(input).lex()
            module = Parser(path, tokens).parse()
            for function in module.functions:
                print(function.signature)
                for word in function.body:
                    print(word)
            for struct in module.structs:
                print(struct)
            for memory in module.memories:
                print(memory)

