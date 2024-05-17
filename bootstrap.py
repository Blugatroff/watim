#!/usr/bin/env python
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Any, TypeVar, Callable, Generic, List, Tuple, NoReturn, assert_never
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
    PTR = "TYPE_PTR"

@dataclass
class ParsedPtrType:
    child: 'ParsedType'

@dataclass
class ParsedGenericType:
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

ParsedType = PrimitiveType | ParsedPtrType | ParsedGenericType | ParsedForeignType | ParsedStructType | ParsedFunctionType


@dataclass
class ParsedNumberWord:
    token: Token

@dataclass
class ParsedStringWord:
    token: Token

@dataclass
class ParsedDerefWord:
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
    token: Token

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
class ParsedBreakWord:
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

ParsedWord = ParsedNumberWord | ParsedStringWord | ParsedCallWord | ParsedDerefWord | ParsedGetWord | ParsedRefWord | ParsedSetWord | ParsedStoreWord | ParsedInitWord | ParsedCallWord | ParsedForeignCallWord | ParsedFunRefWord | ParsedIfWord | ParsedLoadWord | ParsedLoopWord | ParsedBlockWord | ParsedBreakWord | ParsedCastWord | ParsedSizeofWord | ParsedGetFieldWord | ParsedIndirectCallWord

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
    last_processed_token: Token | None = None

    def peek(self, skip_ws: bool = False) -> Token | None:
        i = 0
        while True:
            if i >= len(self.tokens):
                return None
            token = self.tokens[i]
            if skip_ws and token.ty == TokenType.SPACE:
                i += 1
                continue
            self.last_processed_token = token
            return token

    def advance(self, skip_ws: bool = False):
        while True:
            if len(self.tokens) == 0:
                return None
            token = self.tokens[0]
            self.tokens = self.tokens[1:]
            if skip_ws and token.ty == TokenType.SPACE:
                continue
            self.last_processed_token = token
            return token

    def retreat(self, token: Token):
        self.tokens.insert(0, token)

    def abort(self, message: str) -> NoReturn:
        raise ParserException(self.last_processed_token, message)

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
                print(len(memories))
                continue

            self.abort("Expected function import or struct definition")
        return ParsedModule(imports, structs, memories, functions, externs)

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
            return ParsedNumberWord(token)
        if token.ty == TokenType.STRING:
            return ParsedStringWord(token)
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
            return ParsedBreakWord(token)
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
                    return ParsedGenericType(token, generic_index)
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

