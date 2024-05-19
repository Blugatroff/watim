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
                        self.advance()
                        break

                    if char == '\\':
                        if self.eof():
                            raise LexerException("Unterminated String", self.line, self.column)
                        if self.current() in "ntr\\\"":
                            self.advance()
                lexeme = self.input[start:self.cursor-1]
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
class GetFieldWord:
    token: Token
    fields: List[Token]

@dataclass
class IndirectCallWord:
    token: Token

ParsedWord = NumberWord | StringWord | ParsedCallWord | DerefWord | ParsedGetWord | ParsedRefWord | ParsedSetWord | ParsedStoreWord | InitWord | ParsedCallWord | ParsedForeignCallWord | ParsedFunRefWord | ParsedIfWord | LoadWord | ParsedLoopWord | ParsedBlockWord | BreakWord | ParsedCastWord | ParsedSizeofWord | GetFieldWord | IndirectCallWord

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
            return GetFieldWord(token, self.parse_field_accesses())
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
    module: 'ResolvedModule'
    taip: 'ResolvedStruct'
    generic_arguments: List['ResolvedType']

@dataclass
class ResolvedStructType:
    name: Token
    struct: 'ResolvedStructHandle'
    generic_arguments: List['ResolvedType']

@dataclass
class ResolvedFunctionType:
    token: Token
    args: List['ResolvedType']
    rets: List['ResolvedType']

@dataclass
class ResolvedStruct:
    name: Token
    fields: List['ResolvedNamedType']

@dataclass
class ResolvedStructHandle:
    index: int

ResolvedType = PrimitiveType | ResolvedPtrType | GenericType | ResolvedForeignType | ResolvedStructType | ResolvedFunctionType

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
class ResolvedGetWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[Token]

@dataclass
class ResolvedRefWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[Token]

@dataclass
class ResolvedSetWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[Token]

@dataclass
class ResolvedStoreWord:
    token: Token
    local: ResolvedNamedType | InitWord | ResolvedMemory
    fields: List[Token]

@dataclass
class ResolvedForeignCallWord:
    module_token: Token
    module: 'ResolvedModule'
    name: Token
    function: 'ResolvedFunction | ResolvedExtern'
    generic_arguments: List[ResolvedType]

@dataclass
class ResolvedCallWord:
    name: Token
    function: 'ResolvedFunctionHandle'
    generic_arguments: List[ResolvedType]

@dataclass
class ResolvedFunctionHandle:
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

ResolvedWord = NumberWord | StringWord | ResolvedCallWord | DerefWord | ResolvedGetWord | ResolvedRefWord | ResolvedSetWord | ResolvedStoreWord | InitWord | ResolvedCallWord | ResolvedForeignCallWord | ResolvedFunRefWord | ResolvedIfWord | LoadWord | ResolvedLoopWord | ResolvedBlockWord | BreakWord | ResolvedCastWord | ResolvedSizeofWord | GetFieldWord | IndirectCallWord | IntrinsicWord

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
    id: int
    imports: List[ResolvedImport]
    structs: List[ResolvedStruct]
    externs: List[ResolvedExtern]
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
    vars: Dict[str, List[ResolvedNamedType | InitWord]] = field(default_factory=dict)

    def lookup(self, name: Token) -> ResolvedNamedType | InitWord | None:
        if name.lexeme not in self.vars:
            return None
        vars = self.vars[name.lexeme]
        if len(vars) == 0:
            return None
        return vars[0]

    def insert(self, var: ResolvedNamedType | InitWord):
        if var.name.lexeme in self.vars:
            self.vars[var.name.lexeme].insert(0, var)
            return
        self.vars[var.name.lexeme] = [var]

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
        env = Env()
        for param in self.signature.parameters:
            env.insert(param)
        for local in locals:
            env.insert(local)
        for memory in memories:
            env.insert(memory.taip)
        body = list(map(lambda w: self.resolve_word(env, w), self.function.body))
        return ResolvedFunction(self.signature, memories, locals, body)

    def resolve_word(self, env: Env, word: ParsedWord) -> ResolvedWord:
        def resolve_word(word: ParsedWord):
            return self.resolve_word(env, word)
        if isinstance(word, NumberWord):
            return word
        if isinstance(word, StringWord):
            return word
        if isinstance(word, ParsedCastWord):
            return ResolvedCastWord(word.token, self.module_resolver.resolve_type(word.taip))
        if isinstance(word, ParsedIfWord):
            if_words = list(map(resolve_word, word.if_words))
            else_words = list(map(resolve_word, word.else_words))
            return ResolvedIfWord(word.token, if_words, else_words)
        if isinstance(word, ParsedLoopWord):
            words = list(map(resolve_word, word.words))
            return ResolvedLoopWord(word.token, words)
        if isinstance(word, ParsedBlockWord):
            words = list(map(resolve_word, word.words))
            return ResolvedBlockWord(word.token, words)
        if isinstance(word, ParsedCallWord):
            if word.name.lexeme in INTRINSICS:
                return IntrinsicWord(INTRINSICS[word.name.lexeme], word.name)
            return self.resolve_call_word(env, word)
        if isinstance(word, ParsedForeignCallWord):
            return self.resolve_foreign_call_word(env, word)
        if isinstance(word, DerefWord):
            return word
        if isinstance(word, ParsedGetWord):
            var = self.resolve_var_name(env, word.token)
            return ResolvedGetWord(word.token, var, word.fields)
        if isinstance(word, InitWord):
            env.insert(word)
            return word
        if isinstance(word, ParsedRefWord):
            var = self.resolve_var_name(env, word.token)
            return ResolvedRefWord(word.token, var, word.fields)
        if isinstance(word, ParsedSetWord):
            var = self.resolve_var_name(env, word.token)
            return ResolvedSetWord(word.token, var, word.fields)
        if isinstance(word, ParsedStoreWord):
            var = self.resolve_var_name(env, word.token)
            return ResolvedStoreWord(word.token, var, word.fields)
        if isinstance(word, ParsedFunRefWord):
            if isinstance(word.call, ParsedCallWord):
                return ResolvedFunRefWord(self.resolve_call_word(env, word.call))
            if isinstance(word.call, ParsedForeignCallWord):
                return ResolvedFunRefWord(self.resolve_foreign_call_word(env, word.call))
            assert_never(word.call)
        if isinstance(word, LoadWord):
            return word
        if isinstance(word, BreakWord):
            return word
        if isinstance(word, ParsedSizeofWord):
            return ResolvedSizeofWord(word.token, self.module_resolver.resolve_type(word.taip))
        if isinstance(word, GetFieldWord):
            return word
        if isinstance(word, IndirectCallWord):
            return word
        assert_never(word)

    def resolve_call_word(self, env: Env, word: ParsedCallWord) -> ResolvedCallWord:
        resolved_generic_arguments = list(map(self.module_resolver.resolve_type, word.generic_arguments))
        function = self.module_resolver.resolve_function_name(word.name)
        return ResolvedCallWord(word.name, function, resolved_generic_arguments)

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
        return ResolvedCallWord(word.name, function, resolved_generic_arguments)

    def resolve_var_name(self, env: Env, name: Token) -> ResolvedNamedType | InitWord | ResolvedMemory:
        var = env.lookup(name)
        if var is None:
            for memory in self.module_resolver.memories:
                if memory.taip.name.lexeme == name.lexeme:
                    return memory
            self.module_resolver.abort(name, f"local {name.lexeme} not found")
        return var

    def resolve_fields(self, taip: ResolvedType, fields: List[Token]) -> List[Tuple[ResolvedStruct, Token]]:
        resolved_fields = []
        for field_name in fields:
            if isinstance(taip, ResolvedStructType):
                struct = self.structs[taip.struct.index]
                for field in struct.fields:
                    if field.name.lexeme == field_name.lexeme:
                        resolved_fields.append((struct, field_name))
                self.module_resolver.abort(field_name, f"field not found {field_name.lexeme}")
        return resolved_fields


@dataclass
class ModuleResolver:
    resolved_modules: Dict[str, ResolvedModule]
    module: ParsedModule
    imports: List[ResolvedImport] = field(default_factory=list)
    structs: List[ResolvedStruct] = field(default_factory=list)
    externs: List[ResolvedExtern] = field(default_factory=list)
    memories: List[ResolvedMemory] = field(default_factory=list)

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolverException(token, self.module.path + " " + message)

    def resolve(self, id: int) -> ResolvedModule:
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
        return ResolvedModule(id, resolved_imports, resolved_structs, resolved_externs, resolved_functions)

    def resolve_function(self, signature: ResolvedFunctionSignature, function: ParsedFunction) -> ResolvedFunction:
        return FunctionResolver(self, self.signatures, self.structs, function, signature).resolve()

    def resolve_function_name(self, name: Token) -> ResolvedFunctionHandle:
        function_names = list(map(lambda s: s.name.lexeme, self.signatures)) + list(map(lambda s: s.signature.name.lexeme, self.externs))
        for index, signature in enumerate(function_names):
            if signature == name.lexeme:
                return ResolvedFunctionHandle(index)
        self.abort(name, f"function {name.lexeme} not found")

    def resolve_memory(self, memory: ParsedMemory) -> ResolvedMemory:
        return ResolvedMemory(ResolvedNamedType(memory.name, self.resolve_type(memory.taip)), memory.size)

    def resolve_extern(self, extern: ParsedExtern) -> ResolvedExtern:
        return ResolvedExtern(extern.module, extern.name, self.resolve_function_signature(extern.signature))

    def resolve_import(self, imp: ParsedImport) -> ResolvedImport:
        path = os.path.normpath(os.path.dirname(self.module.path) + "/" + imp.file_path.lexeme[1:-1])
        imported_module = self.resolved_modules[path]
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
            return ResolvedStructType(taip.name, self.resolve_struct_name(taip.name), resolved_generic_arguments)
        if isinstance(taip, GenericType):
            return taip
        if isinstance(taip, ParsedForeignType):
            resolved_generic_arguments = list(map(self.resolve_type, taip.generic_arguments))
            for imp in self.imports:
                if imp.qualifier.lexeme == taip.module.lexeme:
                    for struct in imp.module.structs:
                        if struct.name.lexeme == taip.name.lexeme:
                            return ResolvedForeignType(taip.module, taip.name, imp.module, struct, resolved_generic_arguments)
            self.abort(taip.module, f"struct {taip.module.lexeme}:{taip.name.lexeme} not found")
        if isinstance(taip, ParsedFunctionType):
            args = list(map(self.resolve_type, taip.args))
            rets = list(map(self.resolve_type, taip.rets))
            return ResolvedFunctionType(taip.token, args, rets)
        return assert_never(taip)

    def resolve_struct_name(self, name: Token) -> ResolvedStructHandle:
        for index, struct in enumerate(self.module.structs):
            if struct.name.lexeme == name.lexeme:
                return ResolvedStructHandle(index)
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

    resolved_modules: Dict[str, ResolvedModule] = {}
    for id, module in enumerate(determine_compilation_order(list(modules.values()))):
        resolved_modules[module.path] = ModuleResolver(resolved_modules, module).resolve(id)

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

