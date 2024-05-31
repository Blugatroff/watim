#!/usr/bin/env python
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Any, TypeVar, Callable, Generic, List, Tuple, NoReturn, Dict, assert_never
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
class StringWord:
    token: Token

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
class InitWord:
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
class LoadWord:
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
class IndirectCallWord:
    token: Token

ParsedWord = NumberWord | StringWord | ParsedCallWord | DerefWord | ParsedGetWord | ParsedRefWord | ParsedSetWord | ParsedStoreWord | InitWord | ParsedCallWord | ParsedForeignCallWord | ParsedFunRefWord | ParsedIfWord | LoadWord | ParsedLoopWord | ParsedBlockWord | BreakWord | ParsedCastWord | ParsedSizeofWord | ParsedGetFieldWord | IndirectCallWord

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
        raise ParserException(self.tokens[self.cursor - 1], message)

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
            return StringWord(token)
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
            return InitWord(token)
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
            return LoadWord(token)
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
            return IndirectCallWord(token)
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
class ResolvedImport:
    file_path: Token
    qualifier: Token
    module: 'ResolvedModule'

@dataclass
class ResolvedPtrType:
    child: 'ResolvedType'

@dataclass
class ResolvedForeignType:
    module_token: Token
    name: Token
    struct: 'ResolvedStruct'
    generic_arguments: List['ResolvedType']

@dataclass
class ResolvedStructType:
    module: int
    name: Token
    struct: 'Callable[[], ResolvedStruct]'
    generic_arguments: List['ResolvedType']

@dataclass
class ResolvedFunctionType:
    token: Token
    parameters: List['ResolvedType']
    returns: List['ResolvedType']

@dataclass
class ResolvedStruct:
    name: Token
    fields: List['ResolvedNamedType']

@dataclass
class ResolvedStructHandle:
    module: int
    index: int

ResolvedType = PrimitiveType | ResolvedPtrType | GenericType | ResolvedForeignType | ResolvedStructType | ResolvedFunctionType

def resolved_type_eq(a: ResolvedType, b: ResolvedType):
    if isinstance(a, PrimitiveType):
        return a == b
    if isinstance(a, ResolvedPtrType) and isinstance(b, ResolvedPtrType):
        return resolved_type_eq(a.child, b.child)
    if isinstance(a, ResolvedStructType) and isinstance(b, ResolvedStructType):
        return a.struct() == b.struct()
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
    if isinstance(a, ResolvedForeignType) and isinstance(b, ResolvedForeignType):
        return a.struct == b.struct
    if isinstance(a, ResolvedStructType) and isinstance(b, ResolvedForeignType):
        return a.struct() == b.struct
    if isinstance(a, ResolvedForeignType) and isinstance(b, ResolvedStructType):
        return resolved_type_eq(b, a)
    if isinstance(a, GenericType) and isinstance(b, GenericType):
        return a.generic_index == b.generic_index
    return False

def format_type(a: ResolvedType) -> str:
    if isinstance(a, PrimitiveType):
        return str(a)
    if isinstance(a, ResolvedPtrType):
        return f".{format_type(a.child)}"
    if isinstance(a, ResolvedStructType):
        if len(a.generic_arguments) == 0:
            return a.name.lexeme
        s = a.name.lexeme + "<"
        for arg in a.generic_arguments:
            s += format_type(arg) + ", "
        return s + ">"
    if isinstance(a, ResolvedFunctionType):
        s = "("
        for param in a.parameters:
            s += format_type(param) + ", "
        s = s[:-2] + " -> "
        if len(a.returns) == 0:
            return s[:-1] + ")"
        for ret in a.returns:
            s += format_type(ret) + ", "
        return s[:-2] + ")"
    if isinstance(a, GenericType):
        return a.token.lexeme
    if isinstance(a, ResolvedForeignType):
        return a.name.lexeme
    assert_never(a)

@dataclass
class ResolvedNamedType:
    name: Token
    taip: ResolvedType

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
    struct: ResolvedStruct
    taip: ResolvedType

@dataclass
class ResolvedGetWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedRefWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedSetWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedStoreWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[ResolvedFieldAccess]

@dataclass
class ResolvedInitWord:
    name: Token
    taip: ResolvedNamedType

@dataclass
class ResolvedForeignCallWord:
    module_token: Token
    module: 'ResolvedModule'
    name: Token
    function: 'ResolvedFunction | ResolvedExtern'
    generic_arguments: List[ResolvedType]

    def signature(self) -> ResolvedFunctionSignature:
        if isinstance(self.function, ResolvedFunction):
            return self.function.signature
        if isinstance(self.function, ResolvedExtern):
            return self.function.signature
        assert_never(self.function)

@dataclass
class ResolvedCallWord:
    name: Token
    function: 'ResolvedFunctionHandle | ResolvedExtern'
    generic_arguments: List[ResolvedType]

    def signature(self, signatures: List[ResolvedFunctionSignature]) -> ResolvedFunctionSignature:
        if isinstance(self.function, ResolvedFunctionHandle):
            return signatures[self.function.index]
        if isinstance(self.function, ResolvedExtern):
            return self.function.signature
        assert_never(self.function)

@dataclass
class ResolvedFunctionHandle:
    index: int

@dataclass
class ResolvedForeignFunctionHandle:
    index: int

@dataclass
class ResolvedFunRefWord:
    call: ResolvedCallWord | ResolvedForeignCallWord

@dataclass
class ResolvedIfWord:
    token: Token
    if_words: List['ResolvedWord']
    else_words: List['ResolvedWord']

@dataclass
class ResolvedLoopWord:
    token: Token
    words: List['ResolvedWord']

@dataclass
class ResolvedBlockWord:
    token: Token
    words: List['ResolvedWord']

@dataclass
class ResolvedCastWord:
    token: Token
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
class IntrinsicWord:
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

ResolvedWord = NumberWord | StringWord | ResolvedCallWord | DerefWord | ResolvedGetWord | ResolvedRefWord | ResolvedSetWord | ResolvedStoreWord | InitWord | ResolvedCallWord | ResolvedForeignCallWord | ResolvedFunRefWord | ResolvedIfWord | LoadWord | ResolvedLoopWord | ResolvedBlockWord | BreakWord | ResolvedCastWord | ResolvedSizeofWord | ResolvedGetFieldWord | IndirectCallWord | IntrinsicWord | ResolvedInitWord

@dataclass
class ResolvedFunction:
    signature: ResolvedFunctionSignature
    memories: List[ResolvedMemory]
    locals: List[ResolvedNamedType]
    body: List[ResolvedWord]

@dataclass
class ResolvedExtern:
    module: Token
    name: Token
    signature: ResolvedFunctionSignature

@dataclass
class ResolvedModule:
    path: str
    id: int
    imports: List[ResolvedImport]
    structs: List[ResolvedStruct]
    externs: List[ResolvedExtern]
    memories: List[ResolvedMemory]
    functions: List[ResolvedFunction]

def load_recursive(modules: Dict[str, ParsedModule], path: str, import_stack: List[str]=[]):
    with open(path, 'r') as reader:
        file = reader.read()
        tokens = Lexer(file).lex()
        module = Parser(path, tokens).parse()
        modules[path] = module
        for imp in module.imports:
            p = os.path.normpath(os.path.dirname(path) + "/" + imp.file_path.lexeme[1:-1])
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
                path = os.path.normpath(os.path.dirname(module.path) + "/" + imp.file_path.lexeme[1:-1])
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

@dataclass
class Env:
    parent: 'Env | None'
    vars: Dict[str, List[ResolvedNamedType]] = field(default_factory=dict)

    def lookup(self, name: Token) -> ResolvedNamedType | None:
        if name.lexeme not in self.vars:
            if self.parent is not None:
                return self.parent.lookup(name)
            return None
        vars = self.vars[name.lexeme]
        if len(vars) == 0:
            if self.parent is not None:
                return self.parent.lookup(name)
            return None
        return vars[0]

    def insert(self, var: ResolvedNamedType):
        if var.name.lexeme in self.vars:
            self.vars[var.name.lexeme].insert(0, var)
            return
        self.vars[var.name.lexeme] = [var]

@dataclass
class Stack:
    parent: 'Stack | None'
    stack: List[ResolvedType]
    negative_depth: int
    negative: List[ResolvedType]

    @staticmethod
    def empty() -> 'Stack':
        return Stack(None, [], 0, [])

    def append(self, taip: ResolvedType):
        self.stack.append(taip)
        if self.negative_depth > 0:
            self.negative_depth -= 1

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
        if self.negative_depth == len(self.negative):
            self.negative.append(taip)
        return taip

    def clone(self) -> 'Stack':
        return Stack(self.parent.clone() if self.parent is not None else None, list(self.stack), self.negative_depth, list(self.negative))

    def make_child(self) -> 'Stack':
        return Stack(self.clone(), [], 0, [])

    def apply(self, other: 'Stack'):
        for removed in other.negative:
            self.pop()
        for added in other.stack:
            self.append(added)

    def __len__(self) -> int:
        return len(self.stack) + (len(self.parent) if self.parent is not None else 0)

    def __getitem__(self, index: int) -> ResolvedType:
        return self.stack[index]


@dataclass
class FunctionResolver:
    module_resolver: 'ModuleResolver'
    signatures: List[ResolvedFunctionSignature]
    structs: List[ResolvedStruct]
    function: ParsedFunction
    signature: ResolvedFunctionSignature

    def resolve(self) -> ResolvedFunction:
        memories = list(map(self.module_resolver.resolve_memory, self.function.memories))
        locals = list(map(self.module_resolver.resolve_named_type, self.function.locals))
        env = Env(None)
        for param in self.signature.parameters:
            env.insert(param)
        for local in locals:
            env.insert(local)
        for memory in memories:
            env.insert(memory.taip)
        stack: Stack = Stack.empty()
        body = list(map(lambda w: self.resolve_word(env, stack, [], w), self.function.body))
        return ResolvedFunction(self.signature, memories, locals, body)

    def resolve_word(self, env: Env, stack: Stack, break_stacks: List[Stack], word: ParsedWord) -> ResolvedWord:
        if isinstance(word, NumberWord):
            stack.append(PrimitiveType.I32)
            return word
        if isinstance(word, StringWord):
            stack.append(ResolvedPtrType(PrimitiveType.I32))
            stack.append(PrimitiveType.I32)
            return word
        if isinstance(word, ParsedCastWord):
            if len(stack) == 0:
                self.module_resolver.abort(word.token, "expected a non-empty stack")
            stack.pop()
            resolved_type = self.module_resolver.resolve_type(word.taip)
            stack.append(resolved_type)
            return ResolvedCastWord(word.token, resolved_type)
        if isinstance(word, ParsedIfWord):
            if len(stack) == 0 or stack[-1] != PrimitiveType.BOOL:
                print(stack)
                self.module_resolver.abort(word.token, "expected a boolean on stack")
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
            stack.apply(if_stack)
            return ResolvedIfWord(word.token, if_words, else_words)
        if isinstance(word, ParsedLoopWord):
            loop_stack = stack.make_child()
            loop_env = Env(env)
            loop_break_stacks: List[Stack] = []
            def resolve_loop_word(word: ParsedWord):
                return self.resolve_word(loop_env, loop_stack, loop_break_stacks, word)
            words = list(map(resolve_loop_word, word.words))
            stack.apply(loop_break_stacks[0] if len(loop_break_stacks) != 0 else loop_stack)
            return ResolvedLoopWord(word.token, words)
        if isinstance(word, ParsedBlockWord):
            block_stack = stack.make_child()
            block_env = Env(env)
            block_break_stacks: List[Stack] = []
            def resolve_block_word(word: ParsedWord):
                return self.resolve_word(block_env, block_stack, block_break_stacks, word)
            words = list(map(resolve_block_word, word.words))
            if len(block_break_stacks) != 0:
                stack.apply(block_break_stacks[0])
            else:
                stack.apply(block_stack)
            return ResolvedBlockWord(word.token, words)
        if isinstance(word, ParsedCallWord):
            if word.name.lexeme in INTRINSICS:
                intrinsic = INTRINSICS[word.name.lexeme]
                self.type_check_intrinsic(word.name, stack, intrinsic)
                return IntrinsicWord(intrinsic, word.name)
            resolved_call_word = self.resolve_call_word(env, word)
            if isinstance(resolved_call_word.function, ResolvedFunctionHandle):
                signature = self.signatures[resolved_call_word.function.index]
            else:
                signature = resolved_call_word.function.signature
            self.type_check_call(stack, resolved_call_word.name, resolved_call_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
            return resolved_call_word
        if isinstance(word, ParsedForeignCallWord):
            resolved_word = self.resolve_foreign_call_word(env, word)
            signature = resolved_word.signature()
            self.type_check_call(stack, word.name, resolved_word.generic_arguments, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns)
            return resolved_word
        if isinstance(word, DerefWord):
            return word
        if isinstance(word, ParsedGetWord):
            var = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var.taip, word.fields)
            stack.append(var.taip if len(resolved_fields) == 0 else resolved_fields[-1].taip)
            return ResolvedGetWord(word.token, var, resolved_fields)
        if isinstance(word, InitWord):
            taip = stack.pop()
            if taip is None:
                self.module_resolver.abort(word.name, "expected a non-empty stack")
            named_taip = ResolvedNamedType(word.name, taip)
            env.insert(named_taip)
            return ResolvedInitWord(word.name, named_taip)
        if isinstance(word, ParsedRefWord):
            var = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var.taip, word.fields)
            stack.append(ResolvedPtrType(var.taip if len(resolved_fields) == 0 else resolved_fields[-1].taip))
            return ResolvedRefWord(word.token, var, resolved_fields)
        if isinstance(word, ParsedSetWord):
            var = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var.taip, word.fields)
            expected_taip = var.taip if len(resolved_fields) == 0 else resolved_fields[-1].taip
            self.expect_stack(word.token, stack, [expected_taip])
            return ResolvedSetWord(word.token, var, resolved_fields)
        if isinstance(word, ParsedStoreWord):
            var = self.resolve_var_name(env, word.token)
            resolved_fields = self.resolve_fields(var.taip, word.fields)
            stack.append(var.taip if len(resolved_fields) == 0 else resolved_fields[-1].taip)
            return ResolvedStoreWord(word.token, var, resolved_fields)
        if isinstance(word, ParsedFunRefWord):
            if isinstance(word.call, ParsedCallWord):
                resolved_call_word = self.resolve_call_word(env, word.call)
                signature = resolved_call_word.signature(self.module_resolver.signatures)
                stack.append(ResolvedFunctionType(word.call.name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                return ResolvedFunRefWord(resolved_call_word)
            if isinstance(word.call, ParsedForeignCallWord):
                resolved_foreign_call_word = self.resolve_foreign_call_word(env, word.call)
                signature = resolved_foreign_call_word.signature()
                stack.append(ResolvedFunctionType(word.call.name, list(map(lambda nt: nt.taip, signature.parameters)), signature.returns))
                return ResolvedFunRefWord(resolved_foreign_call_word)
            assert_never(word.call)
        if isinstance(word, LoadWord):
            if len(stack) == 0:
                self.module_resolver.abort(word.token, "expected a non-empty stack")
            top = stack.pop()
            if not isinstance(top, ResolvedPtrType):
                print(stack)
                self.module_resolver.abort(word.token, "expected a pointer on the stack")
            stack.append(top.child)
            return word
        if isinstance(word, BreakWord):
            return word
        if isinstance(word, ParsedSizeofWord):
            stack.append(PrimitiveType.I32)
            return ResolvedSizeofWord(word.token, self.module_resolver.resolve_type(word.taip))
        if isinstance(word, ParsedGetFieldWord):
            taip = stack.pop()
            if taip is None:
                self.module_resolver.abort(word.token, "GetField expected a struct on the stack")
            resolved_fields = self.resolve_fields(taip, word.fields)
            stack.append(ResolvedPtrType(resolved_fields[-1].taip))
            return ResolvedGetFieldWord(word.token, taip, resolved_fields)
        if isinstance(word, IndirectCallWord):
            if len(stack) == 0:
                self.module_resolver.abort(word.token, "`->` expected a function on the stack")
            function_type = stack.pop()
            if not isinstance(function_type, ResolvedFunctionType):
                self.module_resolver.abort(word.token, "`->` expected a function on the stack")
            self.type_check_call(stack, word.token, None, function_type.parameters, function_type.returns)
            return word
        assert_never(word)

    def type_check_intrinsic(self, token: Token, stack: Stack, intrinsic: IntrinsicType):
        match intrinsic:
            case IntrinsicType.ADD | IntrinsicType.SUB:
                if len(stack) < 2:
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if isinstance(stack[-2], ResolvedPtrType):
                    if stack[-1] != PrimitiveType.I32:
                        self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, i32]")
                    stack.pop()
                if stack[-1] == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    stack.append(popped[0])
                if stack[-1] == PrimitiveType.I64:
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    stack.append(PrimitiveType.I64)
            case IntrinsicType.DROP:
                if len(stack) == 0:
                    print(stack)
                    self.module_resolver.abort(token, "`drop` expected non empty stack")
                stack.pop()
            case IntrinsicType.MOD | IntrinsicType.MUL | IntrinsicType.DIV:
                if isinstance(stack[-2], ResolvedPtrType):
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if stack[-1] == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                stack.append(popped[0])
            case IntrinsicType.AND | IntrinsicType.OR:
                if len(stack) < 2:
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                match stack[-1]:
                    case PrimitiveType.I32:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    case PrimitiveType.I64:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    case _:
                        popped = self.expect_stack(token, stack, [PrimitiveType.BOOL, PrimitiveType.BOOL])
                stack.append(popped[0])
            case IntrinsicType.ROTR | IntrinsicType.ROTL:
                if isinstance(stack[-2], ResolvedPtrType):
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if stack[-2] == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I32])
                stack.append(popped[0])
            case IntrinsicType.GREATER | IntrinsicType.LESS | IntrinsicType.GREATER_EQ | IntrinsicType.LESS_EQ:
                if isinstance(stack[-2], ResolvedPtrType):
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if stack[-1] == PrimitiveType.I32:
                    self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                stack.append(PrimitiveType.BOOL)
            case IntrinsicType.LOAD8:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32)])
                stack.append(PrimitiveType.I32)
            case IntrinsicType.STORE8:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32), PrimitiveType.I32])
            case IntrinsicType.MEM_COPY:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I32), ResolvedPtrType(PrimitiveType.I32), PrimitiveType.I32])
            case IntrinsicType.NOT_EQ | IntrinsicType.EQ:
                if len(stack) < 2:
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if not resolved_type_eq(stack[-1], stack[-2]):
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [a, a] for any a")
                stack.pop()
                stack.pop()
                stack.append(PrimitiveType.BOOL)
            case IntrinsicType.FLIP:
                a = stack.pop()
                b = stack.pop()
                if a is None or b is None:
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                stack.extend([a, b])
            case IntrinsicType.MEM_GROW:
                self.expect_stack(token, stack, [PrimitiveType.I32])
                stack.append(PrimitiveType.I32)
            case IntrinsicType.STORE:
                if len(stack) < 2:
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                if not isinstance(stack[-2], ResolvedPtrType) or not resolved_type_eq(stack[-2].child, stack[-1]):
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, a]")
                stack.pop()
                stack.pop()
            case IntrinsicType.NOT:
                if len(stack) == 0 or (stack[-1] != PrimitiveType.I32 and stack[-1] != PrimitiveType.BOOL):
                    self.module_resolver.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected a i32 or bool on the stack")
            case _:
                self.module_resolver.abort(token, "TODO")


    def expect_stack(self, token: Token, stack: Stack, expected: List[ResolvedType]) -> List[ResolvedType]:
        popped: List[ResolvedType] = []
        for expected_type in reversed(expected):
            top = stack.pop()
            if top is None:
                self.module_resolver.abort(token, "expected: " + format_type(expected_type))
            popped.append(top)
            if not resolved_type_eq(expected_type, top):
                self.module_resolver.abort(token, "expected: " + format_type(expected_type) + "\ngot: " + format_type(top))
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
                if isinstance(taip.struct, ResolvedStruct):
                    struct = taip.struct
                else:
                    struct = taip.struct()
                concrete_struct = struct
                assert(not isinstance(taip.struct, ResolvedStruct))
                return ResolvedStructType(taip.module, taip.name, lambda: concrete_struct, list(map(inner, taip.generic_arguments)))
            if isinstance(taip, ResolvedFunctionType):
                return ResolvedFunctionType(taip.token, list(map(inner, taip.parameters)), list(map(inner, taip.returns)))
            return taip
        return inner

    def resolve_foreign_call_word(self, env: Env, word: ParsedForeignCallWord) -> ResolvedForeignCallWord:
        resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
        for imp in self.module_resolver.imports:
            if imp.qualifier.lexeme == word.module.lexeme:
                module = imp.module
                for f in module.functions:
                    if f.signature.name.lexeme == word.name.lexeme:
                        return ResolvedForeignCallWord(word.module, module, word.name, f, resolved_generic_arguments)
                for extern in module.externs:
                    if extern.signature.name.lexeme == word.name.lexeme:
                        return ResolvedForeignCallWord(word.module, module, word.name, extern, resolved_generic_arguments)
                self.module_resolver.abort(word.name, f"function {word.name.lexeme} not found")
        self.module_resolver.abort(word.name, f"module {word.module.lexeme} not found")

    def resolve_var_name(self, env: Env, name: Token) -> ResolvedNamedType:
        var = env.lookup(name)
        if var is None:
            for memory in self.module_resolver.memories:
                if memory.taip.name.lexeme == name.lexeme:
                    return memory.taip
            self.module_resolver.abort(name, f"local {name.lexeme} not found")
        return var

    def resolve_fields(self, taip: ResolvedType, fields: List[Token]) -> List[ResolvedFieldAccess]:
        resolved_fields: List[ResolvedFieldAccess] = []
        while len(fields) > 0:
            field_name = fields[0]
            def inner(struct: ResolvedStruct, generic_arguments: List[ResolvedType], fields: List[Token]) -> ResolvedType:
                for field in struct.fields:
                    if field.name.lexeme == field_name.lexeme:
                        taip = FunctionResolver.resolve_generic(generic_arguments)(field.taip)
                        resolved_fields.append(ResolvedFieldAccess(field_name, struct, taip))
                        fields.pop(0)
                        return taip
                self.module_resolver.abort(field_name, f"field not found {field_name.lexeme}")
            if isinstance(taip, ResolvedStructType):
                struct = taip.struct()
                taip = inner(taip.struct(), taip.generic_arguments, fields)
                continue
            if isinstance(taip, ResolvedForeignType):
                taip = inner(taip.struct, taip.generic_arguments, fields)
                continue
            if isinstance(taip, ResolvedPtrType) and not isinstance(taip.child, ResolvedPtrType):
                taip = taip.child
                continue
            else:
                self.module_resolver.abort(field_name, f"field not found {field_name.lexeme} WTF?")
        return resolved_fields


@dataclass
class ModuleResolver:
    resolved_modules: Dict[int, ResolvedModule]
    resolved_modules_by_path: Dict[str, ResolvedModule]
    module: ParsedModule
    id: int
    imports: List[ResolvedImport] = field(default_factory=list)
    structs: List[ResolvedStruct] = field(default_factory=list)
    externs: List[ResolvedExtern] = field(default_factory=list)
    memories: List[ResolvedMemory] = field(default_factory=list)

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolverException(token, self.module.path + " " + message)

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
        return ResolvedModule(self.module.path, self.id, resolved_imports, resolved_structs, resolved_externs, self.memories, resolved_functions)

    def resolve_function(self, signature: ResolvedFunctionSignature, function: ParsedFunction) -> ResolvedFunction:
        return FunctionResolver(self, self.signatures, self.structs, function, signature).resolve()

    def resolve_function_name(self, name: Token) -> ResolvedFunctionHandle | ResolvedExtern:
        for index, signature in enumerate(self.signatures):
            if signature.name.lexeme == name.lexeme:
                return ResolvedFunctionHandle(index)
        for extern in self.externs:
            if extern.signature.name.lexeme == name.lexeme:
                return extern
        self.abort(name, f"function {name.lexeme} not found")

    def resolve_memory(self, memory: ParsedMemory) -> ResolvedMemory:
        return ResolvedMemory(ResolvedNamedType(memory.name, ResolvedPtrType(self.resolve_type(memory.taip))), memory.size)

    def resolve_extern(self, extern: ParsedExtern) -> ResolvedExtern:
        return ResolvedExtern(extern.module, extern.name, self.resolve_function_signature(extern.signature))

    def resolve_import(self, imp: ParsedImport) -> ResolvedImport:
        path = os.path.normpath(os.path.dirname(self.module.path) + "/" + imp.file_path.lexeme[1:-1])
        imported_module = self.resolved_modules_by_path[path]
        return ResolvedImport(imp.file_path, imp.module_qualifier, imported_module)

    def resolve_named_type(self, named_type: ParsedNamedType) -> ResolvedNamedType:
        return ResolvedNamedType(named_type.name, self.resolve_type(named_type.taip))

    def resolve_type(self, taip: ParsedType) -> ResolvedType:
        if isinstance(taip, PrimitiveType):
            return taip
        if isinstance(taip, ParsedPtrType):
            return ResolvedPtrType(self.resolve_type(taip.child))
        if isinstance(taip, ParsedStructType):
            resolved_generic_arguments = list(map(self.resolve_type, taip.generic_arguments))
            return ResolvedStructType(self.id, taip.name, self.resolve_struct_name(taip.name), resolved_generic_arguments)
        if isinstance(taip, GenericType):
            return taip
        if isinstance(taip, ParsedForeignType):
            resolved_generic_arguments = list(map(self.resolve_type, taip.generic_arguments))
            for imp in self.imports:
                if imp.qualifier.lexeme == taip.module.lexeme:
                    for struct in imp.module.structs:
                        if struct.name.lexeme == taip.name.lexeme:
                            return ResolvedForeignType(taip.module, taip.name, struct, resolved_generic_arguments)
            self.abort(taip.module, f"struct {taip.module.lexeme}:{taip.name.lexeme} not found")
        if isinstance(taip, ParsedFunctionType):
            args = list(map(self.resolve_type, taip.args))
            rets = list(map(self.resolve_type, taip.rets))
            return ResolvedFunctionType(taip.token, args, rets)
        return assert_never(taip)

    def resolve_struct_name(self, name: Token) -> Callable[[], ResolvedStruct]:
        for index, struct in enumerate(self.module.structs):
            if struct.name.lexeme == name.lexeme:
                return lambda: self.structs[index]
        self.abort(name, f"struct {name.lexeme} not found")

    def resolve_struct(self, struct: ParsedStruct) -> ResolvedStruct:
        return ResolvedStruct(struct.name, list(map(self.resolve_named_type, struct.fields)))

    def resolve_function_signature(self, signature: ParsedFunctionSignature) -> ResolvedFunctionSignature:
        parameters = list(map(self.resolve_named_type, signature.parameters))
        rets = list(map(self.resolve_type, signature.returns))
        return ResolvedFunctionSignature(signature.export_name, signature.name, signature.generic_parameters, parameters, rets)

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
        print(len(resolved_modules))
    except ResolverException as e:
        print(e.token)
        print(e.message)


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

