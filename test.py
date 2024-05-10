#!/usr/bin/env python
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Optional, Any, TypeVar, Callable, Generic, List, assert_never
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
    TOKEN_NUMBER = "TOKEN_NUMBER"
    TOKEN_FN = "TOKEN_FN"
    TOKEN_IMPORT = "TOKEN_IMPORT"
    TOKEN_IDENT = "TOKEN_IDENT"
    TOKEN_STRING = "TOKEN_STRING"
    TOKEN_LEFT_TRIANGLE = "TOKEN_LEFT_TRIANGLE"
    TOKEN_RIGHT_TRIANGLE = "TOKEN_RIGHT_TRIANGLE"
    TOKEN_LEFT_PAREN = "TOKEN_LEFT_PAREN"
    TOKEN_RIGHT_PAREN = "TOKEN_RIGHT_PAREN"
    TOKEN_LEFT_BRACE = "TOKEN_LEFT_BRACE"
    TOKEN_RIGHT_BRACE = "TOKEN_RIGHT_BRACE"
    TOKEN_COLON = "TOKEN_COLON"
    TOKEN_DOT = "TOKEN_DOT"
    TOKEN_COMMA = "TOKEN_COMMA"
    TOKEN_DOLLAR = "TOKEN_DOLLAR"
    TOKEN_AMPERSAND = "TOKEN_AMPERSAND"
    TOKEN_HASH = "TOKEN_HASH"
    TOKEN_AT = "TOKEN_AT"
    TOKEN_BANG = "TOKEN_BANG"
    TOKEN_TILDE = "TOKEN_TILDE"
    TOKEN_BACKSLASH = "TOKEN_BACKSLASH"
    TOKEN_MEMORY = "TOKEN_MEMORY"
    TOKEN_LOCAL = "TOKEN_LOCAL"
    TOKEN_AS = "TOKEN_AS"
    TOKEN_STRUCT = "TOKEN_STRUCT"
    TOKEN_BLOCK = "TOKEN_BLOCK"
    TOKEN_BREAK = "TOKEN_BREAK"
    TOKEN_LOOP = "TOKEN_LOOP"
    TOKEN_IF = "TOKEN_IF"
    TOKEN_ELSE = "TOKEN_ELSE"
    TOKEN_EXTERN = "TOKEN_EXTERN"
    TOKEN_BOOL = "TOKEN_BOOL"
    TOKEN_I8 = "TOKEN_I8"
    TOKEN_I32 = "TOKEN_I32"
    TOKEN_I64 = "TOKEN_I64"
    TOKEN_ARROW = "TOKEN_ARROW"
    TOKEN_DOUBLE_ARROW = "TOKEN_DOUBLE_ARROW"
    TOKEN_SPACE = "TOKEN_SPACE"


@dataclass(frozen=True)
class Token:
    ty: TokenType
    line: int
    column: int
    lexeme: str

    @staticmethod
    def space(line: int, column: int) -> 'Token':
        return Token(TokenType.TOKEN_SPACE, line, column, " ")
    
    @staticmethod
    def keyword(ty: TokenType, line: int, column: int) -> 'Token':
        return Token(ty, line, column, Token.keyword_lexeme(ty))
    
    @staticmethod
    def keyword_lexeme(ty: TokenType) -> str:
        return TOKEN_TYPE_LEXEME_DICT[ty]

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
        # if len(self.tokens) == 0 or self.tokens[-1].ty != "Foo":
        #     self.tokens.append(Token.space(self.cursor, self.column))
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
                    self.tokens.append(Token.keyword(TokenType.TOKEN_ARROW, self.line, self.column))
                    self.advance(2)
                    continue
            
            if self.current() == '=':
                if not self.last_char() and self.peek() == '>':
                    self.tokens.append(Token.keyword(TokenType.TOKEN_DOUBLE_ARROW, self.line, self.column))
                    self.advance(2)
                    continue
            
            one_char_tokens = "<>(){}:.,$&#@!~\\"
            if self.current() in one_char_tokens:
                ty = TokenType(LEXEME_TOKEN_TYPE_DICT[self.current()])
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
                self.tokens.append(Token(TokenType.TOKEN_STRING, start_line, start_column, lexeme))
                continue
            
            if self.current().isdigit():
                start = self.cursor
                start_column = self.column
                while not self.eof() and self.current().isdigit():
                    self.advance()
                lexeme = self.input[start:self.cursor]
                self.tokens.append(Token(TokenType.TOKEN_NUMBER, self.line, start_column, lexeme))
                continue
            
            if Lexer.allowed_in_ident(self.current()):
                start_column = self.column
                start = self.cursor
                while not self.eof() and Lexer.allowed_in_ident(self.current()):
                    self.advance()
                lexeme = self.input[start:self.cursor]
                try:
                    self.tokens.append(Token(LEXEME_TOKEN_TYPE_DICT[lexeme], self.line, start_column, lexeme))
                except:
                    self.tokens.append(Token(TokenType.TOKEN_IDENT, self.line, start_column, lexeme))
                continue
            raise LexerException("Unexpected character encountered: " + self.current(), self.line, self.column)

        return self.tokens

    @staticmethod
    def allowed_in_ident(char: str) -> bool:
        return char not in "#${}()<> \t\n:&~,."

LEXEME_TOKEN_TYPE_DICT: dict[str, TokenType] = {
    "fn":     TokenType.TOKEN_FN,
    "import": TokenType.TOKEN_IMPORT,
    "as":     TokenType.TOKEN_AS,
    "memory": TokenType.TOKEN_MEMORY,
    "local":  TokenType.TOKEN_LOCAL,
    "struct": TokenType.TOKEN_STRUCT,
    "block":  TokenType.TOKEN_BLOCK,
    "break":  TokenType.TOKEN_BREAK,
    "loop":   TokenType.TOKEN_LOOP,
    "if":     TokenType.TOKEN_IF,
    "else":   TokenType.TOKEN_ELSE,
    "extern": TokenType.TOKEN_EXTERN,
    "bool":   TokenType.TOKEN_BOOL,
    "i8":     TokenType.TOKEN_I8,
    "i32":    TokenType.TOKEN_I32,
    "i64":    TokenType.TOKEN_I64,
    "->":     TokenType.TOKEN_ARROW,
    "=>":     TokenType.TOKEN_DOUBLE_ARROW,
    " ":      TokenType.TOKEN_SPACE,
    "<":      TokenType.TOKEN_LEFT_TRIANGLE,
    ">":      TokenType.TOKEN_RIGHT_TRIANGLE,
    "(":      TokenType.TOKEN_LEFT_PAREN,
    ")":      TokenType.TOKEN_RIGHT_PAREN,
    "{":      TokenType.TOKEN_LEFT_BRACE,
    "}":      TokenType.TOKEN_RIGHT_BRACE,
    ":":      TokenType.TOKEN_COLON,
    ".":      TokenType.TOKEN_DOT,
    ",":      TokenType.TOKEN_COMMA,
    "$":      TokenType.TOKEN_DOLLAR,
    "&":      TokenType.TOKEN_AMPERSAND,
    "#":      TokenType.TOKEN_HASH,
    "@":      TokenType.TOKEN_AT,
    "!":      TokenType.TOKEN_BANG,
    "~":      TokenType.TOKEN_TILDE,
    "\\":     TokenType.TOKEN_BACKSLASH,
}
TOKEN_TYPE_LEXEME_DICT: dict[TokenType, str] = {v: k for k, v in LEXEME_TOKEN_TYPE_DICT.items()}

@dataclass
class WatimException(Exception):
    message: str

def watim_lex(input: str) -> List[dict]:
    watim = subprocess.run(args=["wasmtime", "--dir=.", "--", "./watim.wasm", "-q", "--lex", "-"], input=input.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = watim.stdout.decode('utf-8').strip()
    if watim.returncode != 0:
        raise WatimException(watim.stderr.decode('utf-8').strip())
    return json.loads(stdout)

def diff_side_by_side(expected: str, actual: str):
    a = expected.splitlines()
    b = actual.splitlines()
    width_a = max(max(map(len, a)), len("EXPECTED: "))
    width_b = max(map(len, b))
    
    res = "EXPECTED:" + " " * (width_a - len("EXPECTED")) + "  ACTUAL:"
    for i in range(0, max(len(a), len(b))):
        res += '\n'
        if i < len(a) and i < len(b):
            res += a[i] + " " * (width_a - len(a[i])) + " # " + b[i]
        elif i < len(a):
            res += a[i]
        elif i < len(b):
            res += " " * width_a + " # " + b[i]
    return res

def indent(s: str):
    return reduce(lambda a, b: f"{a}\n  {b}", s.splitlines(), "")[1:]

with open("./v2/parser.watim", 'r') as reader:
    input = reader.read()
    watim_output = watim_lex(input)
    py_output = list(map(asdict, Lexer(input).lex()))
    if watim_output != py_output:
        print(diff_side_by_side(json.dumps(py_output, indent="  "), json.dumps(watim_output, indent="  ")))
        for i in range(0, min(len(watim_output), len(py_output))):
            if watim_output[i] != py_output[i]:
                print("first difference: ")
                print(diff_side_by_side(json.dumps(py_output[i], indent="  "), json.dumps(watim_output[i], indent="  ")))
                break
exit(0)

@dataclass
class NumberWord:
    ty: str
    number: int
    def __init__(self, number: int):
        self.ty = "WORD_NUMBER"
        self.number = number

@dataclass
class CallWord:
    ty: str
    ident: str
    def __init__(self, ident: str):
        self.ty = "WORD_CALL"
        self.ident = ident

@dataclass
class WordDeref:
    ty: str = "WORD_DEREF"

Word = NumberWord | CallWord | WordDeref

@dataclass
class ParsedFunction:
    ty: str
    ident: str
    export: Optional[str]
    locals: List[int]
    body: List[Word]
    def __init__(self, ident: str, export: Optional[str], locals: List[int], body: List[Word]):
        self.ty = "TOP_ITEM_FN"
        self.ident = ident
        self.export = export
        self.locals = locals
        self.body = body
    
TopItem = ParsedFunction

################################################################################
#                                                                              #
################################################################################


T = TypeVar('T')

@dataclass(frozen=True)
class Success(Generic[T]):
    value: T

Result = Success[T] | str

@dataclass(frozen=True)
class TestFailure:
    expected: str
    actual: str

class TestSuccess:
    pass

TestResult = TestSuccess | TestFailure

class Test(ABC):
    name: Optional[str]
    @abstractmethod
    def run(self) -> TestResult:
        pass

@dataclass(frozen=True)
class LexerTest(Test):
    input: str
    expected: Result[List[Token]]
    name: Optional[str] = None

    def run(self) -> TestResult:
        watim = subprocess.run(args=["wasmtime", "--dir=.", "--", "./watim.wasm", "-q", "--lex", "-"], input=self.input.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = watim.stdout.decode('utf-8').strip()
        match self.expected:
            case str() as error:
                if watim.stderr == error:
                    return TestSuccess()
                else:
                    return TestFailure(expected=error, actual=watim.stderr.decode('utf-8'))
            case Success(value=tokens):
                expected = json.dumps(list(map(asdict, tokens)), indent="  ")
                try:
                    output = json.loads(stdout)
                    if output == json.loads(expected):
                        return TestSuccess()
                    else:
                        return TestFailure(expected=expected, actual=json.dumps(output, indent="  "))
                except json.decoder.JSONDecodeError:
                    return TestFailure(expected=expected, actual=stdout)                
            case _ as unreachable:
                assert_never(unreachable)

@dataclass(frozen=True)
class ParserTest(Test):
    input: str
    expected: Result[List[TopItem]]
    name: Optional[str] = None

    def run(self) -> TestResult:
        watim = subprocess.run(args=["wasmtime", "--dir=.", "--", "./watim.wasm", "-q", "--parse-ast", "-"], input=self.input.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = watim.stdout.decode('utf-8').strip()
        match self.expected:
            case str() as error:
                if watim.stderr == error:
                    return TestSuccess()
                else:
                    return TestFailure(expected=error, actual=watim.stderr.decode('utf-8'))
            case Success(value=top_items):
                expected = json.dumps(list(map(asdict, top_items)), indent="  ")
                try:
                    output = json.loads(stdout)
                    if output == json.loads(expected):
                        return TestSuccess()
                    else:
                        return TestFailure(expected=expected, actual=json.dumps(output, indent="  "))
                except json.decoder.JSONDecodeError:
                    return TestFailure(expected=expected, actual=stdout)                
            case _ as unreachable:
                assert_never(unreachable)

tests = [
    LexerTest("fn", Success([Token(TokenType.TOKEN_FN, 1, 1, "fn")])),
    LexerTest("0 some-function .foo", Success([
        Token(TokenType.TOKEN_NUMBER, 1, 1, "0"),
        Token(TokenType.TOKEN_IDENT, 1, 3, "some-function"),
        Token(TokenType.TOKEN_DOT, 1, 17, "."), 
        Token(TokenType.TOKEN_IDENT, 1, 18, "foo")
    ])),
    ParserTest("fn foo() { }", Success([ParsedFunction("foo", None, [], [])])),
    ParserTest("fn foo() { 0 some-function .foo }", Success([
        ParsedFunction("foo", None, [], [
            NumberWord(0),
            CallWord("some-function"),
            WordDeref()
        ]),
    ])),
]

some_test_failed = False
for (i, test) in enumerate(tests):
    print(f"running test {i} {test.name or ''}", end='')
    result = test.run()
    match result:
        case TestSuccess():
            print(colored("passed", 'green'))
        case TestFailure(expected, actual):
            some_test_failed = True
            print(colored("failed", 'red'))
            print(indent(diff_side_by_side(expected, actual)))
        case _ as unreachable:
            assert_never(unreachable)

if some_test_failed:
    exit(1)
