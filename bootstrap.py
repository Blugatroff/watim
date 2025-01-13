#!/usr/bin/env python
from dataclasses import dataclass, field
from enum import Enum
from collections.abc import Iterable
from typing import Optional, Callable, List, Tuple, NoReturn, Dict, Set, Sequence, Literal, Iterator, TypeGuard, assert_never
from functools import reduce
import sys
import os
import unittest
import copy

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

def bag[K, V](items: Iterator[Tuple[K, V]]) -> Dict[K, List[V]]:
    bag: Dict[K, List[V]] = {}
    for k,v in items:
        if k in bag:
            bag[k].append(v)
        else:
            bag[k] = [v]
    return bag

def normalize_path(path: str) -> str:
    if not path.startswith("./"):
        path = "./" + path
    path = path.replace("//", "/").replace("/./", "/")
    splits = path.split("/")
    outsplits = []
    i = 0
    while i < len(splits):
        split = splits[i]
        if i + 1 != len(splits) and splits[i + 1] == ".." and split != "." and split != "..":
            i += 2
            continue
        outsplits.append(split)
        i += 1

    out = "/".join(outsplits)
    if out != path:
        return normalize_path(out)

    return out

def uhex(n: int) -> str:
    return "0x" + hex(n)[2:].upper()

class IndexedDict[K, V]:
    inner: Dict[K, Ref[Tuple[V, int]]]
    pairs: List[Tuple[K, Ref[Tuple[V, int]]]]

    def __init__(self, inner: Dict[K, Ref[Tuple[V, int]]] | None = None, pairs: List[Tuple[K, Ref[Tuple[V, int]]]] | None = None):
        assert((inner is None) == (pairs is None))
        self.inner = inner or {}
        self.pairs = pairs or []

    @staticmethod
    def from_values(values: Iterable[V], key: Callable[[V], K]) -> 'IndexedDict[K, V]':
        inner = { key(value): Ref((value, i)) for i,value in enumerate(values) }
        pairs = list(inner.items())
        return IndexedDict(inner, pairs)

    @staticmethod
    def from_items(items: Iterable[Tuple[K, V]]) -> 'IndexedDict[K, V]':
        inner = { key: Ref((value, i)) for i,(key,value) in enumerate(items) }
        pairs = list(inner.items())
        return IndexedDict(inner, pairs)

    def index(self, index: int) -> V:
        assert(len(self.inner) == len(self.pairs))
        return self.pairs[index][1].value[0]

    def index_key(self, index: int) -> K:
        assert(len(self.inner) == len(self.pairs))
        return self.pairs[index][0]

    def index_of(self, key: K) -> int:
        return self.inner[key].value[1]

    def __contains__(self, key: K) -> bool:
        return key in self.inner

    def __iter__(self) -> Iterable[K]:
        for k in self.inner:
            yield k

    def __getitem__(self, key: K) -> V:
        return self.inner[key].value[0]

    def __setitem__(self, key: K, value: V):
        if key in self.inner:
            pair = self.inner[key].value
            self.inner[key].value = (value, pair[1])
        else:
            ref = Ref((value, len(self.pairs)))
            self.inner[key] = ref
            self.pairs.append((key, ref))

    def keys(self) -> Iterable[K]:
        return self.inner.keys()

    def values(self) -> Iterable[V]:
        return map(lambda ref: ref.value[0], self.inner.values())

    def items(self) -> Iterable[Tuple[K, V]]:
        return map(lambda kv: (kv[0], kv[1].value[0]), self.inner.items())

    def indexed_values(self) -> Iterable[Tuple[int, V]]:
        return enumerate(map(lambda kv: kv[1].value[0], self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def delete(self, index: int):
        del self.inner[self.pairs.pop(index)[0]]

    def __str__(self) -> str:
        if len(self) == 0:
            return "(Map)"
        return "(Map\n" + indent(reduce(
            lambda a,b: a+",\n"+b,
            map(lambda kv: f"{fmt(kv[0])}={fmt(kv[1])}", self.items()))) + ")"

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

type ParsedType = 'PrimitiveType | Parser.PtrType | Parser.TupleType | GenericType | Parser.ForeignType | Parser.CustomTypeType | Parser.FunctionType'

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

type ParsedWord = 'NumberWord | Parser.StringWord | Parser.CallWord | Parser.GetWord | Parser.RefWord | Parser.SetWord | Parser.StoreWord | Parser.InitWord | Parser.CallWord | Parser.ForeignCallWord | Parser.FunRefWord | Parser.IfWord | Parser.LoadWord | Parser.LoopWord | Parser.BlockWord | BreakWord | Parser.CastWord | Parser.SizeofWord | Parser.GetFieldWord | Parser.IndirectCallWord | Parser.StructWord | Parser.UnnamedStructWord | Parser.MatchWord | Parser.VariantWord | Parser.TupleUnpackWord | Parser.TupleMakeWord | Parser.StackAnnotation'

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
    class CustomTypeType:
        name: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class FunctionType:
        token: Token
        parameters: List[ParsedType]
        returns: List[ParsedType]

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
        data: bytearray

    @dataclass
    class GetWord:
        ident: Token
        fields: List[Token]

    @dataclass
    class RefWord:
        ident: Token
        fields: List[Token]

    @dataclass
    class SetWord:
        ident: Token
        fields: List[Token]

    @dataclass
    class StoreWord:
        ident: Token
        fields: List[Token]

    @dataclass
    class InitWord:
        ident: Token

    @dataclass
    class ForeignCallWord:
        module: Token
        ident: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class CallWord:
        ident: Token
        generic_arguments: List[ParsedType]

    @dataclass
    class FunRefWord:
        call: 'Parser.CallWord | Parser.ForeignCallWord'

    @dataclass
    class IfWord:
        token: Token
        true_words: List['ParsedWord']
        false_words: List['ParsedWord']

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
        taip: 'Parser.CustomTypeType | Parser.ForeignType'
        words: List['ParsedWord']

    @dataclass
    class UnnamedStructWord:
        token: Token
        taip: 'Parser.CustomTypeType | Parser.ForeignType'

    @dataclass
    class VariantWord:
        token: Token
        taip: 'Parser.CustomTypeType | Parser.ForeignType'
        case: Token

    @dataclass
    class MatchCase:
        case: Token
        name: Token
        words: List[ParsedWord]

    @dataclass
    class MatchWord:
        token: Token
        cases: List['Parser.MatchCase']
        default: 'Parser.MatchCase | None'

    @dataclass
    class TupleUnpackWord:
        token: Token

    @dataclass
    class TupleMakeWord:
        token: Token
        item_count: Token

    @dataclass
    class StackAnnotation:
        token: Token
        types: List['ParsedType']

        def __str__(self) -> str:
            return f"(StackAnnotation {self.token} {listtostr(self.types)})"

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
                    return Parser.MatchWord(token, cases, Parser.MatchCase(next, case_name, words))
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
        if token.ty == TokenType.COLON:
            next = self.advance(skip_ws=True)
            if next is None or next.ty != TokenType.LEFT_PAREN:
                self.abort("Expected `(`")
            types: List[ParsedType] = []
            while True:
                next = self.peek(skip_ws=True)
                if next is not None and next.ty == TokenType.RIGHT_PAREN:
                    self.advance(skip_ws=True)
                    return Parser.StackAnnotation(token, [])
                types.append(self.parse_type(generic_parameters))
                next = self.advance(skip_ws=True)
                if next is None:
                    self.abort("Expected `,` or `)`")
                if next.ty == TokenType.RIGHT_PAREN:
                    return Parser.StackAnnotation(token, types)
                if next.ty != TokenType.COMMA:
                    self.abort("Expected `,` or `)`")
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

    def parse_struct_type(self, token: Token | None, generic_parameters: List[Token]) -> 'Parser.CustomTypeType | Parser.ForeignType':
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
            return Parser.CustomTypeType(struct_name, self.parse_generic_arguments(generic_parameters))

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
class ResolvedCustomTypeType:
    name: Token
    type_definition: ResolvedCustomTypeHandle
    generic_arguments: List['ResolvedType']

    def __str__(self) -> str:
        return f"(CustomType {self.type_definition.module} {self.type_definition.index} {listtostr(self.generic_arguments)})"

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

ResolvedType = PrimitiveType | ResolvedPtrType | ResolvedTupleType | GenericType | ResolvedCustomTypeType | ResolvedFunctionType

def resolved_type_eq(a: ResolvedType, b: ResolvedType):
    if isinstance(a, PrimitiveType):
        return a == b
    if isinstance(a, ResolvedPtrType) and isinstance(b, ResolvedPtrType):
        return resolved_type_eq(a.child, b.child)
    if isinstance(a, ResolvedCustomTypeType) and isinstance(b, ResolvedCustomTypeType):
        module_eq = a.type_definition.module == b.type_definition.module
        index_eq  = a.type_definition.index == b.type_definition.index
        return resolved_types_eq(a.generic_arguments, b.generic_arguments) if module_eq and index_eq else False
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
        return f"(Import {self.token} {self.module} \"{self.file_path}\" {self.qualifier} {listtostr(self.items, multi_line=True)})"

@dataclass
class ResolvedStruct:
    name: Token
    generic_parameters: List[Token]
    fields: List['ResolvedNamedType']

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
    generic_parameters: List[Token]
    cases: List[ResolvedVariantCase]

    def __str__(self) -> str:
        return "(Variant\n" + indent(f"name={self.name},\ngeneric-parameters={indent_non_first(listtostr(self.generic_parameters))},\ncases={listtostr(self.cases, multi_line=True)}") + ")"

ResolvedCustomType = ResolvedStruct | ResolvedVariant

@dataclass
class ResolvedFunctionSignature:
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
    name: Token
    taip: ResolvedType
    was_reffed: bool = False

    def __str__(self) -> str:
        return f"(Global {self.name} {self.taip} {self.was_reffed})"

@dataclass(frozen=True, eq=True)
class ScopeId:
    raw: int
    def __str__(self) -> str:
        return str(self.raw)
ROOT_SCOPE: ScopeId = ScopeId(0)

@dataclass(frozen=True, eq=True)
class GlobalId:
    module: int
    index: int

@dataclass(frozen=True, eq=True)
class LocalId:
    name: str
    scope: ScopeId
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
class ResolvedScope:
    id: ScopeId
    words: List['ResolvedWord']

    def __str__(self) -> str:
        return f"(Scope {self.id} {indent_non_first(listtostr(self.words, multi_line=True))})"

@dataclass
class ResolvedFunction:
    name: Token
    export_name: Optional[Token]
    signature: ResolvedFunctionSignature
    body: ResolvedScope
    locals: Dict[LocalId, ResolvedLocal]

    def __str__(self) -> str:
        s = "(Function\n"
        s += f"  name={self.name},\n"
        s += f"  export={format_maybe(self.export_name)},\n"
        s += f"  signature={indent_non_first(str(self.signature))},\n"
        s += f"  locals={indent_non_first(format_dict(self.locals))},\n"
        s += f"  body={str(self.body)}"
        return s + ")"

@dataclass
class ResolvedExtern:
    name: Token
    extern_module: str
    extern_name: str
    signature: ResolvedFunctionSignature

    def __str__(self) -> str:
        return f"(Extern {self.name} {self.extern_module} {self.extern_name} {str(self.signature)})"

# =============================================================================
#  Resolved Words
# =============================================================================

@dataclass
class ResolvedFieldAccess:
    name: Token
    source_taip: ResolvedCustomTypeType | ResolvedPtrType
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
    returns: List[ResolvedType] | None
    true_branch: ResolvedScope
    false_branch: ResolvedScope
    diverges: bool

    def __str__(self) -> str:
        s = "(If\n"
        s += f"  token={self.token},\n"
        s += f"  parameters={listtostr(self.parameters)},\n"
        s += f"  returns={format_maybe(None if self.diverges else self.returns, listtostr)},\n"
        s += f"  true-branch={self.true_branch},\n"
        s += f"  false-branch={self.false_branch}"
        return s + ")"

@dataclass
class ResolvedLoopWord:
    token: Token
    body: ResolvedScope
    parameters: List[ResolvedType]
    returns: List[ResolvedType]
    diverges: bool

    def __str__(self) -> str:
        s = "(Loop\n"
        s += f"  token={self.token},\n"
        s += f"  parameters={listtostr(self.parameters)},\n"
        s += f"  returns={format_maybe(None if self.diverges else self.returns, listtostr)},\n"
        s += f"  body={self.body}"
        return s + ")"


@dataclass
class ResolvedBlockWord:
    token: Token
    body: ResolvedScope
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
    taip: ResolvedType
    field_index: int

    generic_arguments: List[ResolvedType]

@dataclass
class ResolvedStructWord:
    token: Token
    taip: ResolvedCustomTypeType
    body: ResolvedScope

@dataclass
class ResolvedUnnamedStructWord:
    token: Token
    taip: ResolvedCustomTypeType

@dataclass
class ResolvedVariantWord:
    token: Token
    tag: int
    variant: ResolvedCustomTypeType

@dataclass
class ResolvedMatchCase:
    taip: ResolvedType | None
    tag: int
    body: ResolvedScope

@dataclass
class ResolvedMatchWord:
    token: Token
    variant: ResolvedCustomTypeType
    by_ref: bool
    cases: List[ResolvedMatchCase]
    default: ResolvedScope | None
    parameters: List[ResolvedType]
    returns: List[ResolvedType] | None

@dataclass
class ResolvedTupleMakeWord:
    token: Token
    taip: ResolvedTupleType

@dataclass
class ResolvedTupleUnpackWord:
    token: Token
    items: ResolvedTupleType


# =============================================================================
#  Resolved Intrinsics
# =============================================================================

class IntrinsicType(str, Enum):
    ADD = "ADD"
    STORE = "STORE"
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
    SHL = "SHL"
    SHR = "SHR"
    ROTL = "ROTL"
    ROTR = "ROTR"
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
        "%": IntrinsicType.MOD,
        "/": IntrinsicType.DIV,
        "/=": IntrinsicType.NOT_EQ,
        "*": IntrinsicType.MUL,
        "mem-copy": IntrinsicType.MEM_COPY,
        "mem-fill": IntrinsicType.MEM_FILL,
        "shl": IntrinsicType.SHL,
        "shr": IntrinsicType.SHR,
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
    taip: ResolvedPtrType | Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

    def __str__(self) -> str:
        return f"(Intrinsic {self.token} (Add {self.taip}))"

@dataclass
class ResolvedIntrinsicSub:
    token: Token
    taip: ResolvedPtrType | Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

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
class ResolvedIntrinsicShl:
    token: Token
    taip: ResolvedType

@dataclass
class ResolvedIntrinsicShr:
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
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicLess:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicGreaterEq:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class ResolvedIntrinsicLessEq:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

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

ResolvedIntrinsicWord = (
      ResolvedIntrinsicAdd
    | ResolvedIntrinsicSub
    | IntrinsicDrop
    | ResolvedIntrinsicMod
    | ResolvedIntrinsicMul
    | ResolvedIntrinsicDiv
    | ResolvedIntrinsicAnd
    | ResolvedIntrinsicOr
    | ResolvedIntrinsicShl
    | ResolvedIntrinsicShr
    | ResolvedIntrinsicRotl
    | ResolvedIntrinsicRotr
    | ResolvedIntrinsicGreater
    | ResolvedIntrinsicLess
    | ResolvedIntrinsicGreaterEq
    | ResolvedIntrinsicLessEq
    | IntrinsicMemCopy
    | IntrinsicMemFill
    | ResolvedIntrinsicEqual
    | ResolvedIntrinsicNotEqual
    | ResolvedIntrinsicFlip
    | IntrinsicMemGrow
    | ResolvedIntrinsicStore
    | ResolvedIntrinsicNot
    | ResolvedIntrinsicUninit
    | IntrinsicSetStackSize
)

ResolvedWord = (
      NumberWord
    | StringWord
    | ResolvedCallWord
    | ResolvedGetWord
    | ResolvedRefWord
    | ResolvedSetWord
    | ResolvedStoreWord
    | ResolvedCallWord
    | ResolvedCallWord
    | ResolvedFunRefWord
    | ResolvedIfWord
    | ResolvedLoadWord
    | ResolvedLoopWord
    | ResolvedBlockWord
    | BreakWord
    | ResolvedCastWord
    | ResolvedSizeofWord
    | ResolvedGetFieldWord
    | ResolvedIndirectCallWord
    | ResolvedIntrinsicWord
    | ResolvedInitWord
    | ResolvedStructFieldInitWord
    | ResolvedStructWord
    | ResolvedUnnamedStructWord
    | ResolvedVariantWord
    | ResolvedMatchWord
    | ResolvedTupleMakeWord
    | ResolvedTupleUnpackWord
)

# =============================================================================
#  Resolver / Typechecker
# =============================================================================

@dataclass
class ResolvedModule:
    path: str
    id: int
    imports: Dict[str, List[Import]]
    custom_types: IndexedDict[str, ResolvedCustomType]
    globals: IndexedDict[str, ResolvedGlobal]
    functions: IndexedDict[str, ResolvedFunction | ResolvedExtern]
    data: bytes

    def __str__(self):
        imports = indent_non_first(format_dict(self.imports))
        custom_types = indent_non_first(str(self.custom_types))
        globals = indent_non_first(str(self.globals))
        functions = indent_non_first(str(self.functions))
        return f"(Module\n  imports={imports},\n  custom-types={custom_types},\n  globals={globals},\n  functions={functions})"

def determine_compilation_order(modules: Dict[str, List[ParsedTopItem]]) -> IndexedDict[str, List[ParsedTopItem]]:
    unprocessed = IndexedDict.from_items(modules.items())
    ordered: IndexedDict[str, List[ParsedTopItem]] = IndexedDict()
    while len(unprocessed) > 0:
        i = 0
        while i < len(unprocessed):
            postpone = False
            module_path,top_items = list(unprocessed.items())[i]
            for top_item in top_items:
                if not isinstance(top_item, Parser.Import):
                    continue
                imp: Parser.Import = top_item
                if os.path.dirname(module_path) != "":
                    path = os.path.normpath(os.path.dirname(module_path) + "/" + imp.file_path.lexeme[1:-1])
                else:
                    path = os.path.normpath(imp.file_path.lexeme[1:-1])
                if "./"+path not in ordered.keys():
                    postpone = True
                    break
            if postpone:
                i += 1
                continue
            ordered[module_path] = top_items
            unprocessed.delete(i)
    return ordered

class Env:
    parent: 'Env | None'
    scope_counter: Ref[int]
    scope_id: ScopeId
    vars: Dict[str, List[Tuple[ResolvedLocal, LocalId]]]
    vars_by_id: Dict[LocalId, ResolvedLocal]

    def __init__(self, parent: 'Env | List[ResolvedLocal] | List[ResolvedNamedType]'):
        if isinstance(parent, Env):
            self.parent = parent
        else:
            self.parent = None
        self.scope_counter = parent.scope_counter if isinstance(parent, Env) else Ref(0)
        self.scope_id = ScopeId(self.scope_counter.value)
        self.scope_counter.value += 1
        self.vars = {}
        self.vars_by_id = parent.vars_by_id if isinstance(parent, Env) else {}
        if isinstance(parent, list):
            for param in parent:
                if isinstance(param, ResolvedLocal):
                    self.insert(param)
                else:
                    self.insert(ResolvedLocal(param.name, param.taip, False, True))


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

    def child(self) -> 'Env':
        return Env(self)

@dataclass
class Stack:
    parent: 'Stack | None'
    stack: List[ResolvedType]
    negative: List[ResolvedType]

    @staticmethod
    def empty() -> 'Stack':
        return Stack(None, [], [])

    def append(self, taip: ResolvedType):
        self.push(taip)

    def push(self, taip: ResolvedType):
        self.stack.append(taip)

    def push_many(self, taips: List[ResolvedType]):
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

    def drop_n(self, n: int):
        for _ in range(n):
            self.pop()

    def pop_n(self, n: int) -> List[ResolvedType]:
        popped: List[ResolvedType] = []
        while n != 0:
            popped_type = self.pop()
            if popped_type is None:
                break
            popped.append(popped_type)
            n -= 1
        popped.reverse()
        return popped

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

    def use(self, n: int):
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
        self.use(len(other.stack))
        other.use(len(self.stack))
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
    fields: Dict[str, Tuple[int, ResolvedType]]

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
class TypeLookup:
    module: int
    types: List[ResolvedCustomType]
    other_modules: List[List[ResolvedCustomType]]

    def lookup(self, handle: ResolvedCustomTypeHandle) -> ResolvedCustomType:
        if handle.module == self.module:
            return self.types[handle.index]
        else:
            return self.other_modules[handle.module][handle.index]


    def types_pretty_bracketed(self, types: List[ResolvedType]) -> str:
        return f"[{self.types_pretty(types)}]"

    def types_pretty(self, types: List[ResolvedType]) -> str:
        s = ""
        for i, taip in enumerate(types):
            s += self.type_pretty(taip)
            if i + 1 < len(types):
                s += ", "
        return s

    def type_pretty(self, taip: ResolvedType) -> str:
        if taip == PrimitiveType.I8:
            return "i8"
        if taip == PrimitiveType.I32:
            return "i32"
        if taip == PrimitiveType.I64:
            return "i64"
        if taip == PrimitiveType.BOOL:
            return "bool"
        if isinstance(taip, ResolvedPtrType):
            return f".{self.type_pretty(taip.child)}"
        if isinstance(taip, ResolvedCustomTypeType):
            return self.lookup(taip.type_definition).name.lexeme
        if isinstance(taip, ResolvedFunctionType):
            return f"({self.types_pretty(taip.parameters)} -> {self.types_pretty(taip.returns)})"
        if isinstance(taip, ResolvedTupleType):
            return self.types_pretty_bracketed(taip.items)
        if isinstance(taip, GenericType):
            return taip.token.lexeme
        assert_never(taip)


@dataclass
class ResolveCtx:
    parsed_modules: IndexedDict[str, List[ParsedTopItem]]
    resolved_modules: IndexedDict[str, ResolvedModule]
    top_items: List[ParsedTopItem]
    module_id: int
    static_data: bytearray

    def abort(self, token: Token, message: str) -> NoReturn:
        raise ResolverException(self.parsed_modules.index_key(self.module_id), "", token, message)

    def resolve_imports(self) -> Dict[str, List[Import]]:
        resolved_imports: Dict[str, List[Import]] = {}
        module_path = list(self.parsed_modules.keys())[self.module_id]
        for i,top_item in enumerate(self.top_items):
            if not isinstance(top_item, Parser.Import):
                continue
            imp: Parser.Import = top_item
            path = "" if module_path == "" else os.path.dirname(module_path)
            path = normalize_path(path + "/" + imp.file_path.lexeme[1:-1])
            imported_module_id = list(self.parsed_modules.keys()).index(path)
            items = self.resolve_import_items(imported_module_id, imp.items)
            if imp.qualifier.lexeme not in resolved_imports:
                resolved_imports[imp.qualifier.lexeme] = []
            resolved_imports[imp.qualifier.lexeme].append(Import(imp.token, path, imp.qualifier, imported_module_id, items))


        return resolved_imports

    def resolve_import_items(self, imported_module_id: int, items: List[Token]) -> List[ImportItem]:
        imported_module = list(self.parsed_modules.values())[imported_module_id]
        def resolve_item(item_name: Token) -> ImportItem:
            item = ResolveCtx.lookup_item_in_module(imported_module, imported_module_id, item_name)
            if item is None:
                self.abort(item_name, "not found")
            return item
        return list(map(resolve_item, items))

    def resolve_custom_types(self, imports: Dict[str, List[Import]]) -> IndexedDict[str, ResolvedCustomType]:
        resolved_custom_types: IndexedDict[str, ResolvedCustomType] = IndexedDict()
        for top_item in self.top_items:
            if isinstance(top_item, Parser.Struct):
                resolved_custom_types[top_item.name.lexeme] = self.resolve_struct(top_item, imports)
            if isinstance(top_item, Parser.Variant):
                resolved_custom_types[top_item.name.lexeme] = self.resolve_variant(top_item, imports)
        return resolved_custom_types

    def resolve_struct(self, struct: Parser.Struct, imports: Dict[str, List[Import]]) -> ResolvedStruct:
        return ResolvedStruct(
            struct.name,
            struct.generic_parameters,
            [self.resolve_named_type(imports, field) for field in struct.fields]
        )

    def resolve_variant(self, variant: Parser.Variant, imports: Dict[str, List[Import]]) -> ResolvedVariant:
        return ResolvedVariant(
            variant.name,
            variant.generic_parameters,
            [ResolvedVariantCase(
                case.name,
                None if case.taip is None else self.resolve_type(imports, case.taip)
            ) for case in variant.cases]
        )

    @staticmethod
    def lookup_item_in_module(module: List[ParsedTopItem], module_id: int, name: Token) -> ImportItem | None:
        type_index = 0
        function_index = 0
        for top_item in module:
            if isinstance(top_item, Parser.Global) or isinstance(top_item, Parser.Import):
                continue
            if isinstance(top_item, Parser.Struct) or isinstance(top_item, Parser.Variant):
                if top_item.name.lexeme == name.lexeme:
                    return ImportItem(name, ResolvedCustomTypeHandle(module_id, type_index))
                type_index += 1
                continue
            if isinstance(top_item, Parser.Function):
                if top_item.signature.name.lexeme == name.lexeme:
                    return ImportItem(name, ResolvedFunctionHandle(module_id, function_index))
                function_index += 1
                continue
            if isinstance(top_item, Parser.Extern):
                if top_item.signature.name.lexeme == name.lexeme:
                    return ImportItem(name, ResolvedFunctionHandle(module_id, function_index))
                function_index += 1
                continue
        return None

    def resolve_named_type(self, imports: Dict[str, List[Import]], named_type: Parser.NamedType) -> ResolvedNamedType:
        return ResolvedNamedType(named_type.name, self.resolve_type(imports, named_type.taip))

    def resolve_named_types(self, imports: Dict[str, List[Import]], named_types: List[Parser.NamedType]) -> List[ResolvedNamedType]:
        return [self.resolve_named_type(imports, named_type) for named_type in named_types]

    def resolve_type(self, imports: Dict[str, List[Import]], taip: ParsedType) -> ResolvedType:
        if isinstance(taip, PrimitiveType):
            return taip
        if isinstance(taip, GenericType):
            return taip
        if isinstance(taip, Parser.PtrType):
            return ResolvedPtrType(self.resolve_type(imports, taip.child))
        if isinstance(taip, Parser.CustomTypeType) or isinstance(taip, Parser.ForeignType):
            return self.resolve_custom_type(imports, taip)
        if isinstance(taip, Parser.FunctionType):
            return ResolvedFunctionType(
                taip.token,
                self.resolve_types(imports, taip.parameters),
                self.resolve_types(imports, taip.returns),
            )
        if isinstance(taip, Parser.TupleType):
            return ResolvedTupleType(
                taip.token,
                self.resolve_types(imports, taip.items)
            )
        assert_never(taip)

    def resolve_types(self, imports: Dict[str, List[Import]], types: List[ParsedType]) -> List[ResolvedType]:
        return [self.resolve_type(imports, taip) for taip in types]

    def resolve_custom_type(self, imports: Dict[str, List[Import]], taip: Parser.CustomTypeType | Parser.ForeignType) -> ResolvedCustomTypeType:
        type_index = 0
        generic_arguments = self.resolve_types(imports, taip.generic_arguments)
        if isinstance(taip, Parser.CustomTypeType):
            for top_item in self.top_items:
                if isinstance(top_item, Parser.Struct) or isinstance(top_item, Parser.Variant):
                    if top_item.name.lexeme == taip.name.lexeme:
                        if len(top_item.generic_parameters) != len(generic_arguments):
                            self.generic_arguments_mismatch_error(taip.name, len(top_item.generic_parameters), len(generic_arguments))
                        return ResolvedCustomTypeType(
                            taip.name,
                            ResolvedCustomTypeHandle(self.module_id, type_index),
                            generic_arguments,
                        )
                    type_index += 1
            for imports_with_same_qualifier in imports.values():
                for imp in imports_with_same_qualifier:
                    for item in imp.items:
                        if isinstance(item.handle, ResolvedCustomTypeHandle) and item.name.lexeme == taip.name.lexeme:
                            return ResolvedCustomTypeType(taip.name, item.handle, generic_arguments)
            self.abort(taip.name, "type not found")
        if isinstance(taip, Parser.ForeignType):
            for i,imp in enumerate(imports[taip.module.lexeme]):
                module = self.resolved_modules.index(imp.module)
                for j,custom_type in enumerate(module.custom_types.values()):
                    if custom_type.name.lexeme == taip.name.lexeme:
                        assert(len(custom_type.generic_parameters) == len(generic_arguments))
                        return ResolvedCustomTypeType(taip.name, ResolvedCustomTypeHandle(imp.module, j), generic_arguments)
            assert(False)
        assert_never(taip)

    def resolve_globals(self, imports: Dict[str, List[Import]]) -> IndexedDict[str, ResolvedGlobal]:
        globals: IndexedDict[str, ResolvedGlobal] = IndexedDict()
        for top_item in self.top_items:
            if isinstance(top_item, Parser.Global):
                globals[top_item.name.lexeme] = ResolvedGlobal(
                    top_item.name,
                    self.resolve_type(imports, top_item.taip),
                    was_reffed=False,
                )
        return globals

    def resolve_signatures(self, imports: Dict[str, List[Import]]) -> List[ResolvedFunctionSignature]:
        signatures = []
        for top_item in self.top_items:
            if not isinstance(top_item, Parser.Function) and not isinstance(top_item, Parser.Extern):
                continue
            signatures.append(self.resolve_signature(imports, top_item.signature))
        return signatures

    def resolve_signature(self, imports: Dict[str, List[Import]], signature: Parser.FunctionSignature) -> ResolvedFunctionSignature:
        return ResolvedFunctionSignature(
            signature.generic_parameters,
            self.resolve_named_types(imports, signature.parameters),
            self.resolve_types(imports, signature.returns),
        )

    def forbid_directly_recursive_types(self, type_lookup: TypeLookup):
        for i in range(len(type_lookup.types)):
            handle = ResolvedCustomTypeHandle(type_lookup.module, i)
            if self.is_directly_recursive(type_lookup, handle, []):
                token = type_lookup.lookup(handle).name
                self.abort(token, "structs and variants cannot be recursive")

    def is_directly_recursive(self, type_lookup: TypeLookup, handle: ResolvedCustomTypeHandle, stack: List[ResolvedCustomTypeHandle]) -> bool:
        if handle in stack:
            return True
        taip = type_lookup.lookup(handle)
        if isinstance(taip, ResolvedStruct):
            for field in taip.fields:
                if isinstance(field.taip, ResolvedCustomTypeType):
                    if self.is_directly_recursive(type_lookup, field.taip.type_definition, [handle] + stack):
                        return True
            return False
        if isinstance(taip, ResolvedVariant):
            for case in taip.cases:
                if isinstance(case.taip, ResolvedCustomTypeType):
                    if self.is_directly_recursive(type_lookup, case.taip.type_definition, [handle] + stack):
                        return True
            return False
        assert_never(taip)

    def generic_arguments_mismatch_error(self, token: Token, expected: int, actual: int):
        msg = f"expected {expected} generic arguments, not {actual}"
        self.abort(token, msg)

    def resolve_functions(
        self,
        imports: Dict[str, List[Import]],
        type_lookup: TypeLookup,
        signatures: List[ResolvedFunctionSignature],
        globals: IndexedDict[str, ResolvedGlobal]
    ) -> IndexedDict[str, ResolvedFunction | ResolvedExtern]:
        functions: IndexedDict[str, ResolvedFunction | ResolvedExtern] = IndexedDict()
        for top_item in self.top_items:
            if isinstance(top_item, Parser.Function):
                function: Parser.Function = top_item
                signature = self.resolve_signature(imports, function.signature)
                env = Env(list(map(ResolvedLocal.make_parameter, signature.parameters)))
                stack = Stack.empty()
                ctx = WordCtx(self, imports, env, type_lookup, signatures, globals)
                words, diverges = ctx.resolve_words(stack, function.body)
                if not diverges and not resolved_types_eq(stack.stack, signature.returns):
                    msg  = "unexpected return values:\n\texpected: "
                    msg += type_lookup.types_pretty_bracketed(signature.returns)
                    msg += "\n\tactual:   "
                    msg += type_lookup.types_pretty_bracketed(stack.stack)
                    self.abort(function.signature.name, msg)
                functions[function.signature.name.lexeme] = ResolvedFunction(
                    function.signature.name,
                    function.signature.export_name,
                    signature,
                    ResolvedScope(env.scope_id, words),
                    env.vars_by_id
                )
                continue
            if isinstance(top_item, Parser.Extern):
                extern: Parser.Extern = top_item
                functions[extern.signature.name.lexeme] = ResolvedExtern(
                    extern.signature.name,
                    extern.module.lexeme,
                    extern.name.lexeme,
                    self.resolve_signature(imports, extern.signature),
                )
                continue
        return functions

    def allocate_static_data(self, data: bytes) -> int:
        offset = self.static_data.find(data)
        if offset == -1:
            offset = len(self.static_data)
            self.static_data.extend(data)
        return offset

@dataclass
class WordCtx:
    ctx: ResolveCtx
    imports: Dict[str, List[Import]]
    env: Env
    type_lookup: TypeLookup
    signatures: List[ResolvedFunctionSignature]
    globals: IndexedDict[str, ResolvedGlobal]
    struct_lit_ctx: StructLitContext | None = None
    break_stacks: List[BreakStack] | None = None
    block_returns: List[ResolvedType] | None = None
    reachable: bool = True

    def with_env(self, env: Env) -> 'WordCtx':
        new = copy.copy(self)
        new.env = env
        return new

    def with_break_stacks(self, break_stacks: List[BreakStack], block_returns: List[ResolvedType] | None) -> 'WordCtx':
        new = copy.copy(self)
        new.break_stacks = break_stacks
        new.block_returns = block_returns
        return new

    def with_struct_lit_ctx(self, ctx: StructLitContext) -> 'WordCtx':
        new = copy.copy(self)
        new.struct_lit_ctx = ctx
        return new

    def abort(self, token: Token, message: str) -> NoReturn:
        self.ctx.abort(token, message)

    def resolve_words(self, stack: Stack, remaining_words: List[ParsedWord]) -> Tuple[List[ResolvedWord], bool]:
        diverges = False
        resolved: List[ResolvedWord] = []
        while len(remaining_words) != 0:
            parsed_word = remaining_words.pop(0)
            res = self.resolve_word(stack, remaining_words, parsed_word)
            if res is None:
                continue
            resolved_word,word_diverges = res
            diverges = diverges or word_diverges
            self.reachable = not diverges
            resolved.append(resolved_word)
        return (resolved, diverges)

    def resolve_word(self, stack: Stack, remaining_words: List[ParsedWord], word: ParsedWord) -> Tuple[ResolvedWord, bool] | None:
        if isinstance(word, NumberWord):
            stack.push(PrimitiveType.I32)
            return (word, False)
        if isinstance(word, Parser.StringWord):
            stack.push(ResolvedPtrType(PrimitiveType.I8))
            stack.push(PrimitiveType.I32)
            offset = self.ctx.allocate_static_data(word.data)
            return (StringWord(word.token, offset, len(word.data)), False)
        if isinstance(word, Parser.GetWord):
            return self.resolve_get_local(stack, word)
        if isinstance(word, Parser.RefWord):
            return self.resolve_ref_local(stack, word)
        if isinstance(word, Parser.InitWord):
            return self.resolve_init_local(stack, word)
        if isinstance(word, Parser.CallWord) or isinstance(word, Parser.ForeignCallWord):
            return self.resolve_call(stack, word)
        if isinstance(word, Parser.CastWord):
            return self.resolve_cast(stack, word)
        if isinstance(word, Parser.SizeofWord):
            return self.resolve_sizeof(stack, word)
        if isinstance(word, Parser.UnnamedStructWord):
            return self.resolve_make_struct(stack, word)
        if isinstance(word, Parser.StructWord):
            return self.resolve_make_struct_named(stack, word)
        if isinstance(word, Parser.FunRefWord):
            return self.resolve_fun_ref(stack, word)
        if isinstance(word, Parser.IfWord):
            return self.resolve_if(stack, remaining_words, word)
        if isinstance(word, Parser.LoopWord):
            return self.resolve_loop(stack, word)
        if isinstance(word, BreakWord):
            return self.resolve_break(stack, word.token)
        if isinstance(word, Parser.SetWord):
            return self.resolve_set_local(stack, word)
        if isinstance(word, Parser.BlockWord):
            return self.resolve_block(stack, word)
        if isinstance(word, Parser.IndirectCallWord):
            return self.resolve_indirect_call(stack, word)
        if isinstance(word, Parser.StoreWord):
            return self.resolve_store(stack, word)
        if isinstance(word, Parser.LoadWord):
            return self.resolve_load(stack, word)
        if isinstance(word, Parser.MatchWord):
            return self.resolve_match(stack, word)
        if isinstance(word, Parser.VariantWord):
            return self.resolve_make_variant(stack, word)
        if isinstance(word, Parser.GetFieldWord):
            return self.resolve_get_field(stack, word)
        if isinstance(word, Parser.TupleMakeWord):
            return self.resolve_make_tuple(stack, word)
        if isinstance(word, Parser.TupleUnpackWord):
            return self.resolve_unpack_tuple(stack, word)
        if isinstance(word, Parser.StackAnnotation):
            self.resolve_stack_annotation(stack, word)
            return None
        assert_never(word)

    def resolve_get_local(self, stack: Stack, word: Parser.GetWord) -> Tuple[ResolvedWord, bool]:
        var_id,taip = self.resolve_var_name(word.ident)
        fields = self.resolve_field_accesses(taip, word.fields)
        resolved_type = taip if len(fields) == 0 else fields[-1].target_taip
        stack.push(resolved_type)
        return (ResolvedGetWord(word.ident, var_id, taip, fields, resolved_type), False)

    def resolve_ref_local(self, stack: Stack, word: Parser.RefWord) -> Tuple[ResolvedWord, bool]:
        local_and_id = self.env.lookup(word.ident)
        if local_and_id is not None:
            local,local_id = local_and_id
            def set_reffed():
                local.was_reffed = True
            taip = local.taip
            var_id: LocalId | GlobalId = local_id
        else:
            if word.ident.lexeme not in self.globals:
                self.abort(word.ident, f"var `{word.ident.lexeme}` not found")
            global_id = self.globals.index_of(word.ident.lexeme)
            globl = self.globals[word.ident.lexeme]
            def set_reffed():
                globl.was_reffed = True
            taip = globl.taip
            var_id = GlobalId(self.ctx.module_id, global_id)

        fields = self.resolve_field_accesses(taip, word.fields)

        i = 0
        while True:
            if i == len(word.fields):
                set_reffed()
                break
            if isinstance(fields[i].source_taip, ResolvedPtrType):
                break
            i += 1

        result_type = taip if len(fields) == 0 else fields[-1].target_taip
        stack.push(ResolvedPtrType(result_type))
        return (ResolvedRefWord(word.ident, var_id, fields), False)

    def resolve_init_local(self, stack: Stack, word: Parser.InitWord) -> Tuple[ResolvedWord, bool]:
        taip = stack.pop()
        assert(taip is not None)
        if self.struct_lit_ctx is not None:
            if word.ident.lexeme in self.struct_lit_ctx.fields:
                field_index,field_type = self.struct_lit_ctx.fields[word.ident.lexeme]
                field_type = self.resolve_generic(self.struct_lit_ctx.generic_arguments, field_type)
                if not resolved_type_eq(field_type, taip):
                    self.abort(word.ident, "wrong type for field")
                del self.struct_lit_ctx.fields[word.ident.lexeme]
                return (ResolvedStructFieldInitWord(
                    word.ident,
                    self.struct_lit_ctx.struct,
                    field_type,
                    field_index,
                    self.struct_lit_ctx.generic_arguments), False)

        local = ResolvedLocal(word.ident, taip, False, False)
        local_id = self.env.insert(local)
        return (ResolvedInitWord(word.ident, local_id, taip), False)

    def resolve_var_name(self, name: Token) -> Tuple[GlobalId | LocalId, ResolvedType]:
        local_and_id = self.env.lookup(name)
        if local_and_id is not None:
            local,local_id = local_and_id
            return (local_id, local.taip)
        if name.lexeme not in self.globals:
            self.abort(name, f"local {name.lexeme} not found")
        global_id = self.globals.index_of(name.lexeme)
        globl = self.globals[name.lexeme]
        return (GlobalId(self.ctx.module_id, global_id), globl.taip)

    def resolve_call(self, stack: Stack, word: Parser.CallWord | Parser.ForeignCallWord) -> Tuple[ResolvedWord, bool]:
        if word.ident.lexeme in INTRINSICS:
            resolved_generic_arguments = [self.ctx.resolve_type(self.imports, taip) for taip in word.generic_arguments]
            intrinsic = INTRINSICS[word.ident.lexeme]
            return (self.resolve_intrinsic(word.ident, stack, intrinsic, resolved_generic_arguments), False)
        resolved_word = self.resolve_call_word(word)
        signature = self.lookup_signature(resolved_word.function)
        self.type_check_call(stack, word.ident, resolved_word.generic_arguments, signature.parameters, signature.returns)
        return (resolved_word, False)

    def type_check_call(self, stack: Stack, token: Token, generic_arguments: List[ResolvedType], parameters: List[ResolvedNamedType], returns: List[ResolvedType]):
        self.expect_arguments(stack, token, generic_arguments, parameters)
        self.push_returns(stack, returns, generic_arguments)

    def type_check_call_unnamed(self, stack: Stack, token: Token, generic_arguments: List[ResolvedType] | None, parameters: List[ResolvedType], returns: List[ResolvedType]):
        self.expect(stack, token, parameters)
        self.push_returns(stack, returns, generic_arguments)

    def push_returns(self, stack: Stack, returns: List[ResolvedType], generic_arguments: List[ResolvedType] | None):
        for ret in returns:
            if generic_arguments is None:
                stack.push(ret)
            else:
                stack.push(self.resolve_generic(generic_arguments, ret))

    def resolve_call_word(self, word: Parser.CallWord | Parser.ForeignCallWord) -> ResolvedCallWord:
        if isinstance(word, Parser.ForeignCallWord):
            resolved_generic_arguments = self.ctx.resolve_types(self.imports, word.generic_arguments)
            imports = self.imports[word.module.lexeme]
            for imp in imports:
                module = self.ctx.resolved_modules.index(imp.module)
                function_id = module.functions.index_of(word.ident.lexeme)
                signature = module.functions.index(function_id).signature
                if len(signature.generic_parameters) != len(resolved_generic_arguments):
                    self.ctx.generic_arguments_mismatch_error(word.ident, len(signature.generic_parameters), len(resolved_generic_arguments))
                return ResolvedCallWord(word.ident, ResolvedFunctionHandle(imp.module, function_id), resolved_generic_arguments)
            self.abort(word.ident, f"function `{word.ident.lexeme}` not found")
        resolved_generic_arguments = [self.ctx.resolve_type(self.imports, taip) for taip in word.generic_arguments]
        function = self.find_function(word.ident)
        if function is None:
            self.abort(word.ident, f"function `{word.ident.lexeme}` not found")
        assert(function is not None)
        signature = self.lookup_signature(function)
        if len(signature.generic_parameters) != len(resolved_generic_arguments):
            self.ctx.generic_arguments_mismatch_error(word.ident, len(signature.generic_parameters), len(resolved_generic_arguments))
        return ResolvedCallWord(word.ident, function, resolved_generic_arguments)

    def resolve_cast(self, stack: Stack, word: Parser.CastWord) -> Tuple[ResolvedWord, bool]:
        src = stack.pop()
        if src is None:
            self.abort(word.token, "cast expected a value, got []")
        dst = self.ctx.resolve_type(self.imports, word.taip)
        stack.push(dst)
        return (ResolvedCastWord(word.token, src, dst), False)

    def resolve_sizeof(self, stack: Stack, word: Parser.SizeofWord) -> Tuple[ResolvedWord, bool]:
        stack.push(PrimitiveType.I32)
        return (ResolvedSizeofWord(word.token, self.ctx.resolve_type(self.imports, word.taip)), False)

    def resolve_make_struct(self, stack: Stack, word: Parser.UnnamedStructWord) -> Tuple[ResolvedWord, bool]:
        struct_type = self.ctx.resolve_custom_type(self.imports, word.taip)
        struc = self.type_lookup.lookup(struct_type.type_definition)
        assert(not isinstance(struc, ResolvedVariant))
        self.expect_arguments(stack, word.token, struct_type.generic_arguments, struc.fields)
        stack.push(struct_type)
        return (ResolvedUnnamedStructWord(word.token, struct_type), False)

    def resolve_make_struct_named(self, stack: Stack, word: Parser.StructWord) -> Tuple[ResolvedWord, bool]:
        struct_type = self.ctx.resolve_custom_type(self.imports, word.taip)
        struct = self.type_lookup.lookup(struct_type.type_definition)
        if isinstance(struct, ResolvedVariant):
            self.abort(word.token, "can only make struct types, not variants")
        env = self.env.child()
        struct_lit_ctx = StructLitContext(
            struct_type.type_definition,
            struct_type.generic_arguments,
            { field.name.lexeme: (i,field.taip) for i,field in enumerate(struct.fields) })
        ctx = self.with_struct_lit_ctx(struct_lit_ctx).with_env(env)
        words,diverges = ctx.resolve_words(stack, word.words)
        if len(struct_lit_ctx.fields) != 0:
            error_message = "missing fields in struct literal:"
            for field_name,(_,field_type) in struct_lit_ctx.fields.items():
                error_message += f"\n\t{field_name}: {ctx.type_lookup.type_pretty(field_type)}"
            ctx.abort(word.token, error_message)
        stack.push(struct_type)
        return (ResolvedStructWord(word.token, struct_type, ResolvedScope(env.scope_id, words)), diverges)

    def resolve_fun_ref(self, stack: Stack, word: Parser.FunRefWord) -> Tuple[ResolvedWord, bool]:
        call = self.resolve_call_word(word.call)
        signature = self.lookup_signature(call.function)
        parameters = [parameter.taip for parameter in signature.parameters]
        stack.push(ResolvedFunctionType(call.name, parameters, signature.returns))
        return (ResolvedFunRefWord(call), False)

    def resolve_if(self, stack: Stack, remaining_words: List[ParsedWord], word: Parser.IfWord) -> Tuple[ResolvedWord, bool]:
        if stack.pop() != PrimitiveType.BOOL:
            self.abort(word.token, "expected a bool for `if`")
        true_env = self.env.child()
        true_stack = stack.make_child()
        true_ctx = self.with_env(true_env)

        false_env = self.env.child()
        false_stack = stack.make_child()
        false_ctx = self.with_env(false_env)

        true_words, true_words_diverge = true_ctx.resolve_words(true_stack, word.true_words)
        true_parameters = true_stack.negative

        if true_words_diverge and len(word.false_words) == 0:
            remaining_stack = stack.make_child()
            remaining_stack.use(len(true_parameters))

            remaining_ctx = false_ctx

            resolved_remaining_words,remaining_words_diverge = remaining_ctx.resolve_words(
                    remaining_stack, remaining_words)

            stack.drop_n(len(remaining_stack.negative))
            stack.push_many(remaining_stack.stack)

            diverges = remaining_words_diverge
            return (ResolvedIfWord(
                word.token,
                list(remaining_stack.negative),
                None if diverges else list(remaining_stack.stack),
                ResolvedScope(true_ctx.env.scope_id, true_words),
                ResolvedScope(remaining_ctx.env.scope_id, resolved_remaining_words),
                diverges), diverges)
        false_words, false_words_diverge = false_ctx.resolve_words(false_stack, word.false_words)
        if not true_words_diverge and not false_words_diverge:
            if not true_stack.compatible_with(false_stack):
                msg  = "stack mismatch between if and else branch:\n\tif   "
                msg += self.type_lookup.types_pretty_bracketed(true_stack.stack)
                msg += "\n\telse "
                msg += self.type_lookup.types_pretty_bracketed(false_stack.stack)
                self.abort(word.token, msg)
            parameters = list(true_stack.negative)
        else:
            # TODO: Check, that the parameters of both branches are compatible
            parameters = true_stack.negative if len(true_stack.negative) > len(false_stack.negative) else false_stack.negative

        if not true_words_diverge:
            returns = true_stack.stack
        elif not false_words_diverge:
            returns = false_stack.stack
        else:
            returns = None

        param_count = max(len(true_stack.negative), len(false_stack.negative))
        if true_words_diverge and false_words_diverge:
            stack.drop_n(param_count)
        else:
            self.expect(stack, word.token, parameters)
            if returns is not None:
                stack.push_many(returns)

        diverges = true_words_diverge and false_words_diverge
        return (ResolvedIfWord(
            word.token,
            parameters,
            returns,
            ResolvedScope(true_ctx.env.scope_id, true_words),
            ResolvedScope(false_ctx.env.scope_id, false_words),
            diverges), diverges)

    def resolve_loop(self, stack: Stack, word: Parser.LoopWord) -> Tuple[ResolvedWord, bool]:
        annotation = None if word.annotation is None else self.resolve_block_annotation(word.annotation)
        loop_break_stacks: List[BreakStack] = []

        loop_env = self.env.child()
        loop_stack = stack.make_child()
        loop_ctx = self.with_env(loop_env).with_break_stacks(
                loop_break_stacks,
                None if annotation is None else annotation.returns)

        words,_ = loop_ctx.resolve_words(loop_stack, word.words.words)
        diverges = len(loop_break_stacks) == 0
        parameters = loop_stack.negative if annotation is None else annotation.parameters

        if len(loop_break_stacks) != 0:
            first = loop_break_stacks[0]
            diverges = not first.reachable
            for break_stack in loop_break_stacks[1:]:
                if not break_stack.reachable:
                    break
                if not resolved_types_eq(first.types, break_stack.types):
                    self.break_stack_mismatch_error(word.token, loop_break_stacks)

        if not resolved_types_eq(parameters, loop_stack.stack):
            self.abort(word.token, "unexpected values remaining on stack at the end of loop")

        if annotation is not None:
            returns = annotation.returns
        elif len(loop_break_stacks) != 0:
            returns = loop_break_stacks[0].types
        else:
            returns = loop_stack.stack

        self.expect(stack, word.token, parameters)
        stack.push_many(returns)
        body = ResolvedScope(loop_ctx.env.scope_id, words)
        return (ResolvedLoopWord(word.token, body, parameters, returns, diverges), diverges)

    def resolve_break(self, stack: Stack, token: Token) -> Tuple[ResolvedWord, bool]:
        if self.block_returns is None:
            dump = stack.dump()
        else:
            dump = stack.pop_n(len(self.block_returns))

        if self.break_stacks is None:
            self.abort(token, "`break` can only be used inside of blocks and loops")

        self.break_stacks.append(BreakStack(token, dump, self.reachable))
        return (BreakWord(token), True)

    def resolve_set_local(self, stack: Stack, word: Parser.SetWord) -> Tuple[ResolvedWord, bool]:
        var_id,taip = self.resolve_var_name(word.ident)
        fields = self.resolve_field_accesses(taip, word.fields)
        if len(fields) == 0:
            resolved_type = taip
        else:
            resolved_type = fields[-1].target_taip
        self.expect(stack, word.ident, [resolved_type])
        return (ResolvedSetWord(word.ident, var_id, fields), False)

    def resolve_block(self, stack: Stack, word: Parser.BlockWord) -> Tuple[ResolvedWord, bool]:
        annotation = None if word.annotation is None else self.resolve_block_annotation(word.annotation)
        block_break_stacks: List[BreakStack] = []

        block_env = self.env.child()
        block_stack = stack.make_child()
        block_ctx = self.with_env(block_env).with_break_stacks(
                block_break_stacks,
                None if annotation is None else annotation.returns)

        words, diverges = block_ctx.resolve_words(block_stack, word.words.words)
        block_end_is_reached = not diverges

        parameters = block_stack.negative if annotation is None else annotation.parameters
        if len(block_break_stacks) != 0:
            first = block_break_stacks[0]
            diverges = not first.reachable
            for break_stack in block_break_stacks[1:]:
                if not break_stack.reachable:
                    diverges = True
                    break
                if not resolved_types_eq(first.types, break_stack.types):
                    if block_end_is_reached:
                        block_break_stacks.append(BreakStack(word.words.end, block_stack.stack, diverges))
                    self.break_stack_mismatch_error(word.token, block_break_stacks)
            if block_end_is_reached:
                if not resolved_types_eq(block_stack.stack, first.types):
                    block_break_stacks.append(BreakStack(word.words.end, block_stack.stack, diverges))
                    self.break_stack_mismatch_error(word.token, block_break_stacks)

        if annotation is not None:
            returns = annotation.returns
        elif len(block_break_stacks) != 0:
            returns = block_break_stacks[0].types
        else:
            returns = block_stack.stack

        self.expect(stack, word.token, parameters)
        stack.push_many(returns)
        body = ResolvedScope(block_ctx.env.scope_id, words)
        return (ResolvedBlockWord(word.token, body, parameters, returns), diverges)

    def resolve_indirect_call(self, stack: Stack, word: Parser.IndirectCallWord) -> Tuple[ResolvedWord, bool]:
        fun_type = stack.pop()
        if fun_type is None:
            self.abort(word.token, "`->` expected a function on the stack, got: []")
        if not isinstance(fun_type, ResolvedFunctionType):
            self.abort(word.token, "TODO")
        self.type_check_call_unnamed(stack, word.token, None, fun_type.parameters, fun_type.returns)
        return (ResolvedIndirectCallWord(word.token, fun_type), False)

    def resolve_store(self, stack: Stack, word: Parser.StoreWord) -> Tuple[ResolvedWord, bool]:
        var_id, taip = self.resolve_var_name(word.ident)
        fields = self.resolve_field_accesses(taip, word.fields)
        expected_type = taip if len(fields) == 0 else fields[-1].target_taip
        if not isinstance(expected_type, ResolvedPtrType):
            self.abort(word.ident, "`=>` can only store into ptr types")
        expected_type = expected_type.child
        self.expect(stack, word.ident, [expected_type])
        return (ResolvedStoreWord(word.ident, var_id, fields), False)

    def resolve_load(self, stack: Stack, word: Parser.LoadWord) -> Tuple[ResolvedWord, bool]:
        taip = stack.pop()
        if taip is None:
            self.abort(word.token, "`~` expected a ptr, got: []")
        if not isinstance(taip, ResolvedPtrType):
            msg = f"`~` expected a ptr, got: [{taip}]"
            self.abort(word.token, msg)
        stack.push(taip.child)
        return (ResolvedLoadWord(word.token, taip.child), False)

    def resolve_match(self, stack: Stack, word: Parser.MatchWord) -> Tuple[ResolvedWord, bool]:
        match_diverges = True
        arg_item = stack.pop()
        if arg_item is None:
            self.abort(word.token, "expected a value to match on")
        by_ref = isinstance(arg_item, ResolvedPtrType)
        arg = arg_item.child if isinstance(arg_item, ResolvedPtrType) else arg_item
        if not isinstance(arg, ResolvedCustomTypeType):
            self.abort(word.token, "can only match n variants")
        generic_arguments = arg.generic_arguments
        variant_type = arg
        variant = self.type_lookup.lookup(arg.type_definition)
        if not isinstance(variant, ResolvedVariant):
            self.abort(word.token, "can only match on variants")
        remaining_cases: List[str] = [case.name.lexeme for case in variant.cases]
        case_stacks: List[Tuple[Stack, Token, bool]] = []
        visited_cases: List[Token] = []
        cases: List[ResolvedMatchCase] = []
        for parsed_case in word.cases:
            tag: int | None = None
            for j, variant_case in enumerate(variant.cases):
                if variant_case.name.lexeme == parsed_case.name.lexeme:
                    tag = j
            if tag is None:
                self.abort(parsed_case.name, "not part of variant")

            case_type = variant.cases[tag].taip
            case_stack = stack.make_child()
            if case_type is not None:
                if by_ref:
                    case_type = ResolvedPtrType(case_type)
                case_type = self.resolve_generic(generic_arguments, case_type)
                case_stack.push(case_type)

            case_env = self.env.child()
            case_ctx = self.with_env(case_env)
            words, case_diverges = case_ctx.resolve_words(case_stack, parsed_case.words)
            match_diverges = match_diverges and case_diverges
            cases.append(ResolvedMatchCase(case_type, tag, ResolvedScope(case_env.scope_id, words)))

            if parsed_case.name.lexeme not in remaining_cases:
                other = next(token for token in visited_cases if token.lexeme == parsed_case.name.lexeme)
                msg  = "duplicate case in match:"
                msg += f"\n\t{other.line}:{other.column} {other.lexeme}"
                msg += f"\n\t{parsed_case.name.line}:{parsed_case.name.column} {parsed_case.name.lexeme}"
                self.abort(word.token, msg)

            remaining_cases.remove(parsed_case.name.lexeme)

            case_stacks.append((case_stack, parsed_case.name, case_diverges))
            visited_cases.append(parsed_case.name)

        if word.default is None:
            if len(remaining_cases) != 0:
                msg = "missing case in match:"
                for case in remaining_cases:
                    msg += f"\n\t{case}"
                self.abort(word.token, msg)
            default_case = None
        else:
            def_stack = stack.make_child()
            def_env = self.env.child()
            def_ctx = self.with_env(def_env)
            def_stack.push(arg_item)
            words, default_diverges = def_ctx.resolve_words(def_stack, word.default.words)
            match_diverges = match_diverges and default_diverges
            case_stacks.append((def_stack, word.default.name, default_diverges))
            default_case = ResolvedScope(def_env.scope_id, words)

        first_non_diverging_case: Stack | None = None
        for case_stack,case_token,case_diverges in case_stacks:
            if not case_diverges:
                if first_non_diverging_case is None:
                    first_non_diverging_case = case_stack
                elif not first_non_diverging_case.compatible_with(case_stack):
                    msg = "arms of match case have different types:"
                    for case_stack, case_token, _ in case_stacks:
                        msg += f"\n\t{self.type_lookup.types_pretty_bracketed(case_stack.negative)}"
                        msg += f" -> {self.type_lookup.types_pretty_bracketed(case_stack.stack)}"
                        msg += f" in case {case_token.lexeme}"
                    self.abort(word.token, msg)

        if len(case_stacks) == 0:
            returns: List[ResolvedType] | None = []
            parameters = []
        else:
            most_params = case_stacks[0][0]
            for case_stack,_,_ in case_stacks[1:]:
                if len(case_stack.negative) > len(most_params.negative):
                    most_params = case_stack

            parameters = list(most_params.negative)
            parameters.reverse()

            returns = list(parameters)
            for case_stack,_,case_diverges in case_stacks:
                if not case_diverges:
                    del returns[len(returns) - len(case_stack.negative):]
                    returns.extend(case_stack.stack)
                    break

        self.expect(stack, word.token, parameters)
        stack.push_many(returns or [])
        if match_diverges:
            returns = None
        return (ResolvedMatchWord(word.token, variant_type, by_ref, cases, default_case, parameters, returns), match_diverges)

    def resolve_make_variant(self, stack: Stack, word: Parser.VariantWord) -> Tuple[ResolvedWord, bool]:
        variant_type = self.ctx.resolve_custom_type(self.imports, word.taip)
        variant = self.type_lookup.lookup(variant_type.type_definition)
        if not isinstance(variant, ResolvedVariant):
            self.abort(word.token, "can not make this type")
        tag: None | int = None
        for i,case in enumerate(variant.cases):
            if case.name.lexeme == word.case.lexeme:
                tag = i
        if tag is None:
            self.abort(word.token, "case is not part of variant")
        case = variant.cases[tag]
        if case.taip is not None:
            expected = self.resolve_generic(variant_type.generic_arguments, case.taip)
            self.expect(stack, word.token, [expected])
        stack.push(variant_type)
        return (ResolvedVariantWord(word.token, tag, variant_type), False)

    def resolve_get_field(self, stack: Stack, word: Parser.GetFieldWord) -> Tuple[ResolvedWord, bool]:
        taip = stack.pop()
        if taip is None:
            self.abort(word.token, "expected a value on the stack")
        fields = self.resolve_field_accesses(taip, word.fields)
        on_ptr = isinstance(taip, ResolvedPtrType)
        taip = fields[-1].target_taip
        taip = ResolvedPtrType(taip) if on_ptr else taip
        stack.push(taip)
        return (ResolvedGetFieldWord(word.token, fields, on_ptr), False)

    def resolve_make_tuple(self, stack: Stack, word: Parser.TupleMakeWord) -> Tuple[ResolvedWord, bool]:
        num_items = int(word.item_count.lexeme)
        items: List[ResolvedType] = []
        for _ in range(num_items):
            item = stack.pop()
            if item is None:
                self.abort(word.token, "expected more")
            items.append(item)
        items.reverse()
        taip = ResolvedTupleType(word.token, items)
        stack.push(taip)
        return (ResolvedTupleMakeWord(word.token, taip), False)

    def resolve_unpack_tuple(self, stack: Stack, word: Parser.TupleUnpackWord) -> Tuple[ResolvedWord, bool]:
        taip = stack.pop()
        if taip is None or not isinstance(taip, ResolvedTupleType):
            self.abort(word.token, "expected a tuple on the stack")
        stack.push_many(taip.items)
        return (ResolvedTupleUnpackWord(word.token, taip), False)

    def resolve_stack_annotation(self, stack: Stack, word: Parser.StackAnnotation) -> None:
        if len(stack) < len(word.types):
            self.abort(word.token, "stack annotation doesn't match reality")
        for i, taip in enumerate(reversed(word.types)):
            expected = self.ctx.resolve_type(self.imports, taip)
            if not resolved_type_eq(stack[-i], expected):
                self.abort(word.token, "stack annotation doesn't match reality")
        return None

    def resolve_block_annotation(self, annotation: Parser.BlockAnnotation) -> BlockAnnotation:
        return BlockAnnotation(
                self.ctx.resolve_types(self.imports, annotation.parameters),
                self.ctx.resolve_types(self.imports, annotation.returns))

    def expect_arguments(self, stack: Stack, token: Token, generic_arguments: List[ResolvedType], parameters: List[ResolvedNamedType]):
        i = len(parameters)
        popped: List[ResolvedType] = []
        while i != 0:
            expected_type = self.resolve_generic(generic_arguments, parameters[i - 1].taip)
            popped_type = stack.pop()
            error = popped_type is None or not resolved_type_eq(popped_type, expected_type)
            if popped_type is not None:
                popped.append(popped_type)
            if error:
                while True:
                    popped_type = stack.pop()
                    if popped_type is None:
                        break
                    popped.append(popped_type)
                expected = [parameter.taip for parameter in parameters]
                popped.reverse()
                self.type_mismatch_error(token, expected, popped)
            assert(popped_type is not None)
            i -= 1

    def expect(self, stack: Stack, token: Token, expected: List[ResolvedType]):
        i = len(expected)
        popped: List[ResolvedType] = []
        while i != 0:
            expected_type = expected[i - 1]
            popped_type = stack.pop()
            error = popped_type is None or not resolved_type_eq(popped_type, expected_type)
            if error:
                popped.reverse()
                self.type_mismatch_error(token, expected, popped)
            assert(popped_type is not None)
            popped.append(popped_type)
            i -= 1

    def type_mismatch_error(self, token: Token, expected: List[ResolvedType], actual: List[ResolvedType]):
        message  = "expected:\n\t" + self.type_lookup.types_pretty_bracketed(expected)
        message += "\ngot:\n\t" + self.type_lookup.types_pretty_bracketed(actual)
        self.abort(token, message)

    def resolve_intrinsic(self, token: Token, stack: Stack, intrinsic: IntrinsicType, generic_arguments: List[ResolvedType]) -> ResolvedIntrinsicWord:
        match intrinsic:
            case IntrinsicType.ADD | IntrinsicType.SUB:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if isinstance(taip, ResolvedPtrType):
                    narrow_type: ResolvedPtrType | Literal[PrimitiveType.I8, PrimitiveType.I32, PrimitiveType.I64] = taip
                    if stack[-1] != PrimitiveType.I32:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [.a, i32]")
                    stack.pop()
                elif taip == PrimitiveType.I8:
                    narrow_type = PrimitiveType.I8
                    popped = self.expect_stack(token, stack, [PrimitiveType.I8, PrimitiveType.I8])
                    stack.append(taip)
                elif taip == PrimitiveType.I32:
                    narrow_type = PrimitiveType.I32
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    stack.append(taip)
                elif taip == PrimitiveType.I64:
                    narrow_type = PrimitiveType.I64
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    stack.append(taip)
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]} cannot add to {self.type_lookup.type_pretty(taip)}")
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
                    case PrimitiveType.I8:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I8, PrimitiveType.I8])
                    case PrimitiveType.I32:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                    case PrimitiveType.I64:
                        popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                    case PrimitiveType.BOOL:
                        popped = self.expect_stack(token, stack, [PrimitiveType.BOOL, PrimitiveType.BOOL])
                    case _:
                        self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` can only and i8, i32, i64 and bool")
                stack.append(popped[0])
                if intrinsic == IntrinsicType.AND:
                    return ResolvedIntrinsicAnd(token, taip)
                if intrinsic == IntrinsicType.OR:
                    return ResolvedIntrinsicOr(token, taip)
            case IntrinsicType.SHR | IntrinsicType.SHL | IntrinsicType.ROTR | IntrinsicType.ROTL:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-2]
                if taip == PrimitiveType.I8:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I8, PrimitiveType.I8])
                elif taip == PrimitiveType.I32:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                else:
                    popped = self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                stack.append(popped[0])
                if intrinsic == IntrinsicType.SHL:
                    return ResolvedIntrinsicShl(token, taip)
                if intrinsic == IntrinsicType.SHR:
                    return ResolvedIntrinsicShr(token, taip)
                if intrinsic == IntrinsicType.ROTR:
                    return ResolvedIntrinsicRotr(token, taip)
                if intrinsic == IntrinsicType.ROTL:
                    return ResolvedIntrinsicRotl(token, taip)
            case IntrinsicType.GREATER | IntrinsicType.LESS | IntrinsicType.GREATER_EQ | IntrinsicType.LESS_EQ:
                if len(stack) < 2:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected two items on stack")
                taip = stack[-1]
                if taip == PrimitiveType.I8:
                    narrow_type = PrimitiveType.I8
                    self.expect_stack(token, stack, [PrimitiveType.I8, PrimitiveType.I8])
                elif taip == PrimitiveType.I32:
                    narrow_type = PrimitiveType.I32
                    self.expect_stack(token, stack, [PrimitiveType.I32, PrimitiveType.I32])
                elif taip == PrimitiveType.I64:
                    narrow_type = PrimitiveType.I64
                    self.expect_stack(token, stack, [PrimitiveType.I64, PrimitiveType.I64])
                else:
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected [i8, i8] or [i32, i32] or [i64, i64] on stack")
                stack.append(PrimitiveType.BOOL)
                if intrinsic == IntrinsicType.GREATER:
                    return ResolvedIntrinsicGreater(token, narrow_type)
                if intrinsic == IntrinsicType.LESS:
                    return ResolvedIntrinsicLess(token, narrow_type)
                if intrinsic == IntrinsicType.GREATER_EQ:
                    return ResolvedIntrinsicGreaterEq(token, narrow_type)
                if intrinsic == IntrinsicType.LESS_EQ:
                    return ResolvedIntrinsicLessEq(token, narrow_type)
            case IntrinsicType.MEM_COPY:
                self.expect_stack(token, stack, [ResolvedPtrType(PrimitiveType.I8), ResolvedPtrType(PrimitiveType.I8), PrimitiveType.I32])
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
                stack.push_many([a, b])
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
                if len(stack) == 0 or (taip != PrimitiveType.I64 and taip != PrimitiveType.I32 and taip != PrimitiveType.BOOL) or not isinstance(taip, PrimitiveType):
                    self.abort(token, f"`{INTRINSIC_TO_LEXEME[intrinsic]}` expected an i64, i32 or bool on the stack")
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
            self.abort(token, f"expected:\n\t{self.type_lookup.types_pretty_bracketed(expected)}\ngot:\n\t{self.type_lookup.types_pretty_bracketed(stackdump)}")
        for expected_type in reversed(expected):
            top = stack.pop()
            if top is None:
                abort()
            popped.append(top)
            if not resolved_type_eq(expected_type, top):
                abort()
        return list(reversed(popped))

    def break_stack_mismatch_error(self, token: Token, break_stacks: List[BreakStack]):
        msg = "break stack mismatch:"
        for break_stack in break_stacks:
            msg += f"\n\t{break_stack.token.line}:{break_stack.token.column} {self.type_lookup.types_pretty_bracketed(break_stack.types)}"
        self.abort(token, msg)

    def find_function(self, name: Token) -> ResolvedFunctionHandle | None:
        function_index = 0
        for i, top_item in enumerate(self.ctx.top_items):
            if isinstance(top_item, Parser.Function) or isinstance(top_item, Parser.Extern):
                if top_item.signature.name.lexeme == name.lexeme:
                    return ResolvedFunctionHandle(self.ctx.module_id, function_index)
                function_index += 1
        for imps in self.imports.values():
            for imp in imps:
                for item in imp.items:
                    if item.name.lexeme == name.lexeme:
                        if isinstance(item.handle, ResolvedFunctionHandle):
                            return item.handle
        return None

    def lookup_signature(self, function: ResolvedFunctionHandle) -> ResolvedFunctionSignature:
        if function.module == self.ctx.module_id:
            return self.signatures[function.index]
        return self.ctx.resolved_modules.index(function.module).functions.index(function.index).signature

    def resolve_field_accesses(self, taip: ResolvedType, fields: List[Token]) -> List[ResolvedFieldAccess]:
        resolved = []
        if len(fields) == 0:
            return []
        for field_name in fields:
            assert((isinstance(taip, ResolvedPtrType) and isinstance(taip.child, ResolvedCustomTypeType)) or isinstance(taip, ResolvedCustomTypeType))
            unpointered = taip.child if isinstance(taip, ResolvedPtrType) else taip
            assert(isinstance(unpointered, ResolvedCustomTypeType))
            custom_type: ResolvedCustomTypeType = unpointered
            type_definition = self.type_lookup.lookup(custom_type.type_definition)
            if isinstance(type_definition, ResolvedVariant):
                self.abort(field_name, "variants do not have fields")
            struct: ResolvedStruct = type_definition
            found_field = False
            for field_index,struct_field in enumerate(struct.fields):
                if struct_field.name.lexeme == field_name.lexeme:
                    target_type = self.resolve_generic(custom_type.generic_arguments, struct_field.taip)
                    resolved.append(ResolvedFieldAccess(field_name, taip, target_type, field_index))
                    taip = target_type
                    found_field = True
                    break
            if not found_field:
                self.abort(field_name, "field not found")
        return resolved

    def resolve_generic(self, generics: List[ResolvedType], taip: ResolvedType) -> ResolvedType:
        if isinstance(taip, PrimitiveType):
            return taip
        if isinstance(taip, ResolvedPtrType):
            return ResolvedPtrType(self.resolve_generic(generics, taip.child))
        if isinstance(taip, ResolvedCustomTypeType):
            return ResolvedCustomTypeType(taip.name, taip.type_definition, self.resolve_generic_many(generics, taip.generic_arguments))
        if isinstance(taip, ResolvedFunctionType):
            return ResolvedFunctionType(
                taip.token,
                self.resolve_generic_many(generics, taip.parameters),
                self.resolve_generic_many(generics, taip.returns),
            )
        if isinstance(taip, ResolvedTupleType):
            return ResolvedTupleType(
                taip.token,
                self.resolve_generic_many(generics, taip.items),
            )
        if isinstance(taip, GenericType):
            return generics[taip.generic_index]
        assert_never(taip)

    def resolve_generic_many(self, generics: List[ResolvedType], types: List[ResolvedType]) -> List[ResolvedType]:
        return [self.resolve_generic(generics, taip) for taip in types]


def resolve_modules(modules_unordered: Dict[str, List[ParsedTopItem]]) -> IndexedDict[str, ResolvedModule]:
    modules = determine_compilation_order({
        ("./" + path if path != "-" else path): module
        for path, module in modules_unordered.items()
    })
    resolved_modules: IndexedDict[str, ResolvedModule] = IndexedDict()
    other_module_types: List[List[ResolvedCustomType]] = []
    for id,(module_path,top_items) in enumerate(modules.items()):
        ctx = ResolveCtx(modules, resolved_modules, top_items, id, bytearray())
        imports = ctx.resolve_imports()
        custom_types = ctx.resolve_custom_types(imports)
        globals = ctx.resolve_globals(imports)
        signatures = ctx.resolve_signatures(imports)

        type_lookup = TypeLookup(module=id, types=list(custom_types.values()), other_modules=other_module_types)
        functions = ctx.resolve_functions(imports, type_lookup, signatures, globals)

        ctx.forbid_directly_recursive_types(type_lookup)

        other_module_types.append(list(custom_types.values()))
        resolved_modules[module_path] = ResolvedModule(
            module_path,
            id,
            imports,
            custom_types,
            globals,
            functions,
            ctx.static_data
        )
    return resolved_modules

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
        largest_field = 0
        for i, field_size in enumerate(field_sizes):
            largest_field = max(largest_field, field_size)
            size += field_size
            if i + 1 < len(field_sizes):
                next_field_size = field_sizes[i + 1]
                size = align_to(size, min(next_field_size, 4))
        return align_to(size, largest_field)

    def field_offset(self, field_index: int) -> int:
        fields = self.fields.get()
        offset = 0
        for i in range(0, field_index):
            field_size = fields[i].taip.size()
            offset += field_size
            if i + 1 < len(fields):
                next_field_size = fields[i + 1].taip.size()
                offset = align_to(offset, min(next_field_size, 4))
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
class FunctionSignature:
    generic_arguments: List[Type]
    parameters: List[NamedType]
    returns: List[Type]

    def returns_any_struct(self) -> bool:
        return any(isinstance(ret, StructType) for ret in self.returns)

@dataclass
class Global:
    name: Token
    taip: Type
    was_reffed: bool

@dataclass
class Extern:
    name: Token
    extern_module: str
    extern_name: str
    signature: FunctionSignature

@dataclass
class Scope:
    id: ScopeId
    words: List['Word']

@dataclass
class ConcreteFunction:
    name: Token
    export_name: Token | None
    signature: FunctionSignature
    body: Scope
    locals_copy_space: int
    max_struct_ret_count: int
    locals: Dict[LocalId, Local]

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
    returns: List[Type] | None
    true_branch: Scope
    false_branch: Scope
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
    body: Scope
    parameters: List[Type]
    returns: List[Type]
    diverges: bool

@dataclass
class BlockWord:
    token: Token
    body: Scope
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

    def is_bitshift(self) -> bool:
        return False

@dataclass
class OffsetLoad:
    offset: int
    taip: Type

    def __str__(self) -> str:
        size = self.taip.size()
        if size == 1:
            return f"i32.load8_u offset={self.offset}" if self.offset != 0 else "i32.load8_u"
        if size <= 4:
            return f"i32.load offset={self.offset}" if self.offset != 0 else "i32.load"
        if size <= 8:
            return f"i64.load offset={self.offset}" if self.offset != 0 else "i64.load"
        if self.offset == 0:
            return f"i32.const {size} memory.copy"
        else:
            return f"i32.const {self.offset} i32.add i32.const {size} memory.copy"

    def is_bitshift(self) -> bool:
        return False

@dataclass
class I32InI64:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.wrap_i64"
        return f"i64.const {self.offset * 8} i64.shr_u i32.wrap_i64"

    def is_bitshift(self) -> bool:
        return True

@dataclass
class I8InI32:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.const 0xFF i32.and"
        return f"i32.const {self.offset * 8} i32.shr_u i32.const 0xFF i32.and"

    def is_bitshift(self) -> bool:
        return True

@dataclass
class I8InI64:
    offset: int

    def __str__(self) -> str:
        if self.offset == 0:
            return "i32.wrap_i64 i32.const 0xFF i32.and"
        return f"i64.const {self.offset * 8} i64.shr_u i32.wrap_i64 i32.const 0xFF i32.and"

    def is_bitshift(self) -> bool:
        return True

type Load = Offset | OffsetLoad | I32InI64 | I8InI32 | I8InI64

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
    ident: Token
    local_id: LocalId | GlobalId
    target_taip: Type
    loads: List[Load]
    copy_space_offset: int | None
    var_lives_in_memory: bool

@dataclass
class RefWord:
    ident: Token
    local_id: LocalId | GlobalId
    loads: List[Load]

@dataclass
class IntrinsicAdd:
    token: Token
    taip: PtrType | Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicSub:
    token: Token
    taip: PtrType | Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicMul:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicDiv:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicMod:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

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
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicLessEq:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicGreater:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicLess:
    token: Token
    taip: Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]

@dataclass
class IntrinsicShl:
    token: Token
    taip: Type

@dataclass
class IntrinsicShr:
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
    copy_space_offset: int
    body: Scope

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
    body: Scope

@dataclass
class MatchWord:
    token: Token
    variant: StructHandle
    by_ref: bool
    cases: List[MatchCase]
    default: Scope | None
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

IntrinsicWord = (
      IntrinsicAdd
    | IntrinsicSub
    | IntrinsicEqual
    | IntrinsicNotEqual
    | IntrinsicAnd
    | IntrinsicDrop
    | IntrinsicGreaterEq
    | IntrinsicLessEq
    | IntrinsicMul
    | IntrinsicMod
    | IntrinsicDiv
    | IntrinsicGreater
    | IntrinsicLess
    | IntrinsicFlip
    | IntrinsicShl
    | IntrinsicShr
    | IntrinsicRotl
    | IntrinsicRotr
    | IntrinsicOr
    | IntrinsicStore
    | IntrinsicMemCopy
    | IntrinsicMemFill
    | IntrinsicMemGrow
    | IntrinsicNot
    | IntrinsicUninit
    | IntrinsicSetStackSize
)

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
    modules: IndexedDict[str, ResolvedModule]
    type_definitions: Dict[ResolvedCustomTypeHandle, List[Tuple[List[Type], TypeDefinition]]] = field(default_factory=dict)
    externs: Dict[ResolvedFunctionHandle, Extern] = field(default_factory=dict)
    globals: Dict[GlobalId, Global] = field(default_factory=dict)
    functions: Dict[ResolvedFunctionHandle, Function] = field(default_factory=dict)
    signatures: Dict[ResolvedFunctionHandle, FunctionSignature | List[FunctionSignature]] = field(default_factory=dict)
    function_table: Dict[FunctionHandle | ExternHandle, int] = field(default_factory=dict)

    def monomize(self) -> Tuple[Dict[FunctionHandle | ExternHandle, int], Dict[int, Module]]:
        self.externs = {
            ResolvedFunctionHandle(m, i): self.monomize_extern(f)
            for m,module in self.modules.indexed_values()
            for i,f in enumerate(module.functions.values())
            if isinstance(f, ResolvedExtern)
        }
        self.globals = {
            GlobalId(m, i): self.monomize_global(globl)
            for m,module in self.modules.indexed_values()
            for i,globl in enumerate(module.globals.values())
        }
        for id in range(len(self.modules)):
            module = self.modules.index(id)
            for index, function in enumerate(module.functions.values()):
                if isinstance(function, ResolvedExtern):
                    continue
                if function.export_name is not None:
                    assert(len(function.signature.generic_parameters) == 0)
                    handle = ResolvedFunctionHandle(id, index)
                    self.monomize_function(handle, [])

        mono_modules = {}
        for module_id,module in enumerate(self.modules.values()):
            externs: Dict[int, Extern] = { handle.index: extern for (handle,extern) in self.externs.items() if handle.module == module_id }
            globals: List[Global] = [globl for id, globl in self.globals.items() if id.module == module_id]
            type_definitions: Dict[int, List[TypeDefinition]] = { handle.index: [taip for _,taip in monomorphizations] for handle,monomorphizations in self.type_definitions.items() if handle.module == module_id }
            functions = { handle.index: function for handle,function in self.functions.items() if handle.module == module_id }
            mono_modules[module_id] = Module(module_id, type_definitions, externs, globals, functions, self.modules.index(module_id).data)
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
        f = self.modules.index(function.module).functions.index(function.index)
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
        monomized_locals = self.monomize_locals(f.locals, generics)
        body = Scope(
            f.body.id,
            self.monomize_words(
                f.body.words,
                generics,
                copy_space_offset,
                max_struct_ret_count,
                monomized_locals,
                None))
        concrete_function = ConcreteFunction(
            f.name,
            f.export_name,
            signature,
            body,
            copy_space_offset.value,
            max_struct_ret_count.value,
            monomized_locals)
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
        return FunctionSignature(generics, parameters, returns)

    def monomize_global(self, globl: ResolvedGlobal) -> Global:
        return Global(globl.name, self.monomize_type(globl.taip, []), globl.was_reffed)

    def monomize_scope(self, scope: ResolvedScope, generics: List[Type], copy_space_offset: Ref[int], max_struct_ret_count: Ref[int], locals: Dict[LocalId, Local], struct_space: int | None) -> Scope:
        return Scope(scope.id, self.monomize_words(scope.words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space))

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
            case ResolvedIntrinsicShl(token, taip):
                return IntrinsicShl(token, self.monomize_type(taip, generics))
            case ResolvedIntrinsicShr(token, taip):
                return IntrinsicShr(token, self.monomize_type(taip, generics))
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
                true_branch = self.monomize_scope(resolved_if_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                false_branch = self.monomize_scope(resolved_else_words, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = None if resolved_returns is None else list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return IfWord(token, parameters, returns, true_branch, false_branch, diverges)
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
            case ResolvedLoopWord(token, resolved_body, resolved_parameters, resolved_returns, diverges):
                body = self.monomize_scope(resolved_body, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return LoopWord(token, body, parameters, returns, diverges)
            case ResolvedSizeofWord(token, taip):
                return SizeofWord(token, self.monomize_type(taip, generics))
            case ResolvedBlockWord(token, resolved_body, resolved_parameters, resolved_returns):
                body = self.monomize_scope(resolved_body, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                parameters = list(map(lambda t: self.monomize_type(t, generics), resolved_parameters))
                returns = list(map(lambda t: self.monomize_type(t, generics), resolved_returns))
                return BlockWord(token, body, parameters, returns)
            case ResolvedGetFieldWord(token, resolved_fields, on_ptr):
                fields = self.monomize_field_accesses(resolved_fields, generics)
                target_taip = fields[-1].target_taip
                offset = None
                if not on_ptr and not target_taip.can_live_in_reg():
                    offset = copy_space_offset.value
                    copy_space_offset.value += target_taip.size()
                loads = determine_loads(fields, just_ref=on_ptr)
                return GetFieldWord(token, target_taip, loads, on_ptr, offset)
            case ResolvedStructWord(token, taip, resolved_body):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                copy_space_offset.value += monomized_taip.size()
                body = self.monomize_scope(resolved_body, generics, copy_space_offset, max_struct_ret_count, locals, offset)
                return StructWord(token, monomized_taip, offset, body)
            case ResolvedUnnamedStructWord(token, taip):
                monomized_taip = self.monomize_struct_type(taip, generics)
                offset = copy_space_offset.value
                if not monomized_taip.can_live_in_reg():
                    copy_space_offset.value += monomized_taip.size()
                return UnnamedStructWord(token, monomized_taip, offset)
            case ResolvedStructFieldInitWord(token, struct, taip, _, generic_arguments):
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
                (variant_handle, variant) = self.monomize_struct(resolved_variant_type.type_definition, this_generics)
                offset = copy_space_offset.value
                if variant.size() > 8:
                    copy_space_offset.value += variant.size()
                return VariantWord(token, case, variant_handle, offset)
            case ResolvedMatchWord(token, resolved_variant_type, by_ref, cases, default_case, resolved_parameters, resolved_returns):
                monomized_cases: List[MatchCase] = []
                for resolved_case in cases:
                    body = self.monomize_scope(resolved_case.body, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                    monomized_cases.append(MatchCase(resolved_case.tag, body))
                monomized_default_case = None if default_case is None else self.monomize_scope(default_case, generics, copy_space_offset, max_struct_ret_count, locals, struct_space)
                this_generics = list(map(lambda t: self.monomize_type(t, generics), resolved_variant_type.generic_arguments))
                monomized_variant = self.monomize_struct(resolved_variant_type.type_definition, this_generics)[0]
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
            case ResolvedTupleUnpackWord(token, ResolvedTupleType(_, items)):
                offset = copy_space_offset.value
                mono_items = list(map(lambda t: self.monomize_type(t, generics), items))
                copy_space_offset.value += sum(item.size() for item in mono_items if not item.can_live_in_reg())
                return TupleUnpackWord(token, mono_items, offset)
            case other:
                assert_never(other)

    def lookup_var_taip(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, Local]) -> Type:
        if isinstance(local_id, LocalId):
            return locals[local_id].taip
        return self.globals[local_id].taip

    def does_var_live_in_memory(self, local_id: LocalId | GlobalId, locals: Dict[LocalId, Local]) -> bool:
        if isinstance(local_id, LocalId):
            return locals[local_id].lives_in_memory()
        globl = self.globals[local_id]
        return globl.was_reffed or not globl.taip.can_live_in_reg()

    def insert_function_into_table(self, function: FunctionHandle | ExternHandle) -> int:
        if function not in self.function_table:
            self.function_table[function] = len(self.function_table)
        return self.function_table[function]

    def monomize_field_accesses(self, fields: List[ResolvedFieldAccess], generics: List[Type]) -> List[FieldAccess]:
        if len(fields) == 0:
            return []

        field = fields[0]

        if isinstance(field.source_taip, ResolvedCustomTypeType):
            source_taip: PtrType | StructType = self.monomize_struct_type(field.source_taip, generics)
            resolved_struct = field.source_taip.type_definition
            generic_arguments = field.source_taip.generic_arguments
        else:
            assert(isinstance(field.source_taip.child, ResolvedCustomTypeType))
            source_taip = PtrType(self.monomize_type(field.source_taip.child, generics))
            resolved_struct = field.source_taip.child.type_definition
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
        s = self.modules.index(struct.module).custom_types.index(struct.index)
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
            case ResolvedCustomTypeType():
                return self.monomize_struct_type(taip, generics)
            case ResolvedFunctionType():
                return self.monomize_function_type(taip, generics)
            case ResolvedTupleType(token, items):
                return TupleType(token, list(map(lambda item: self.monomize_type(item, generics), items)))
            case other:
                assert_never(other)

    def monomize_addable_type(self, taip: ResolvedPtrType | Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64], generics: List[Type]) -> PtrType | Literal[PrimitiveType.I8] | Literal[PrimitiveType.I32] | Literal[PrimitiveType.I64]:
        match taip:
            case PrimitiveType.I8 | PrimitiveType.I32 | PrimitiveType.I64:
                return taip
            case ResolvedPtrType():
                return PtrType(self.monomize_type(taip.child, generics))
            case other:
                assert_never(other)

    def monomize_struct_type(self, taip: ResolvedCustomTypeType, generics: List[Type]) -> StructType:
        this_generics = list(map(lambda t: self.monomize_type(t, generics), taip.generic_arguments))
        handle,struct = self.monomize_struct(taip.type_definition, this_generics)
        return StructType(taip.name, handle, Lazy(lambda: struct.size()))

    def monomize_function_type(self, taip: ResolvedFunctionType, generics: List[Type]) -> FunctionType:
        parameters = list(map(lambda t: self.monomize_type(t, generics), taip.parameters))
        returns = list(map(lambda t: self.monomize_type(t, generics), taip.returns))
        return FunctionType(taip.token, parameters, returns)

    def monomize_extern(self, extern: ResolvedExtern) -> Extern:
        signature = self.monomize_concrete_signature(extern.signature)
        return Extern(extern.name, extern.extern_module, extern.extern_name, signature)

def align_to(n: int, to: int) -> int:
    if to == 0:
        return n
    return n + (to - (n % to)) * ((n % to) > 0)


def merge_locals_module(module: Module):
    for function in module.functions.values():
        if isinstance(function, ConcreteFunction):
            merge_locals_function(function)
            return
        for instance in function.instances.values():
            merge_locals_function(instance)

@dataclass
class Disjoint:
    scopes: Set[ScopeId]
    reused: Set[LocalId]
    substitutions: Dict[LocalId, LocalId]

    def fixup_var(self, var: LocalId | GlobalId) -> LocalId | GlobalId:
        if isinstance(var, GlobalId):
            return var
        if var not in self.substitutions:
            return var
        return self.substitutions[var]

def merge_locals_function(function: ConcreteFunction):
    disjoint = Disjoint(set(), set(), {})
    merge_locals_scope(function.body, function.locals, disjoint)

def merge_locals_scope(scope: Scope, locals: Dict[LocalId, Local], disjoint: Disjoint):
    for word in scope.words:
        merge_locals_word(word, locals, disjoint, scope.id)

def merge_locals_word(word: Word, locals: Dict[LocalId, Local], disjoint: Disjoint, scope: ScopeId):
    if isinstance(word, InitWord):
        reused_local = find_disjoint_local(locals, disjoint, locals[word.local_id])
        if reused_local is None:
            return
        del locals[word.local_id]
        disjoint.substitutions[word.local_id] = reused_local
        word.local_id = reused_local
        return
    if isinstance(word, GetWord):
        word.local_id = disjoint.fixup_var(word.local_id)
        return
    if isinstance(word, SetWord):
        word.local_id = disjoint.fixup_var(word.local_id)
        return
    if isinstance(word, RefWord):
        word.local_id = disjoint.fixup_var(word.local_id)
        return
    if isinstance(word, StoreWord):
        word.local = disjoint.fixup_var(word.local)
        return
    if isinstance(word, IfWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.true_branch, locals, disjoint)
        disjoint.reused = outer_reused

        disjoint.scopes.add(word.true_branch.id)
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.false_branch, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.false_branch.id)
        return
    if isinstance(word, BlockWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.body, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.body.id)
        return
    if isinstance(word, LoopWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.body, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.body.id)
        return
    if isinstance(word, StructWord):
        outer_reused = disjoint.reused.copy()
        merge_locals_scope(word.body, locals, disjoint)
        disjoint.reused = outer_reused
        disjoint.scopes.add(word.body.id)
        return
    if isinstance(word, MatchWord):
        for cays in word.cases:
            outer_reused = disjoint.reused.copy()
            merge_locals_scope(cays.body, locals, disjoint)
            disjoint.reused = outer_reused
            disjoint.scopes.add(cays.body.id)
        if word.default is not None:
            outer_reused = disjoint.reused.copy()
            merge_locals_scope(word.default, locals, disjoint)
            disjoint.reused = outer_reused
            disjoint.scopes.add(word.default.id)

def find_disjoint_local(locals: Dict[LocalId, Local], disjoint: Disjoint, to_be_replaced: Local) -> LocalId | None:
    local_size = to_be_replaced.taip.size()
    if len(disjoint.scopes) == 0:
        return None
    for local_id, local in locals.items():
        if local.lives_in_memory() != to_be_replaced.lives_in_memory():
            continue
        if local.taip.size() != local_size:
            continue
        if local_id.scope not in disjoint.scopes:
            continue
        if local_id in disjoint.reused:
            continue
        disjoint.reused.add(local_id)
        return local_id
    return None

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
        self.assertTrue(loads[0] == I32InI64(0))

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
        self.assertTrue(loads[0] == I32InI64(4))
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
        self.assertEqual(loads, [I32InI64(4)])

def merge_loads(loads: List[Load]) -> List[Load]:
    if len(loads) <= 1:
        return loads
    if isinstance(loads[0], OffsetLoad) and (isinstance(loads[1], I32InI64) or isinstance(loads[1], I8InI32) or isinstance(loads[1], I8InI64)):
        return [OffsetLoad(loads[0].offset + loads[1].offset, PrimitiveType.I32)] + loads[2:]
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

        source_type_size = field.source_taip.size()
        target_type_size = field.target_taip.size()
        if source_type_size > 4: # source_taip is between >=4 and <=8 bytes
            if target_type_size == 1:
                load = I8InI64(offset)
            elif target_type_size == 4:
                load = I32InI64(offset)
            else:
                assert(False) # TODO
            return merge_loads([load] + determine_loads(fields[1:], just_ref, base_in_mem))

        if target_type_size != source_type_size:
            if target_type_size == 1:
                return merge_loads([I8InI32(offset)] + determine_loads(fields[1:], just_ref, base_in_mem))
            assert(False) # TODO

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
        self.write_signature(module, function.name, function.export_name, function.signature, instance_id, function.locals)
        if len(function.signature.generic_arguments) > 0:
            self.write(" ;;")
            for taip in function.signature.generic_arguments:
                self.write(" ")
                self.write_type_human(taip)
        self.write("\n")
        self.indent()
        self.write_locals(function.locals)
        for i in range(0, function.max_struct_ret_count):
            self.write_indent()
            self.write(f"(local $s{i}:a i32)\n")
        if function.locals_copy_space != 0:
            self.write_indent()
            self.write("(local $locl-copy-spac:e i32)\n")

        uses_stack = function.locals_copy_space != 0 or any(local.lives_in_memory() for local in function.locals.values())
        if uses_stack:
            self.write_indent()
            self.write("(local $stac:k i32)\n")
            self.write_indent()
            self.write("global.get $stac:k local.set $stac:k\n")

        if function.locals_copy_space != 0:
            self.write_mem("locl-copy-spac:e", function.locals_copy_space, ROOT_SCOPE, 0)
        self.write_structs(function.locals)
        if uses_stack and self.guard_stack:
            self.write_line("call $stack-overflow-guar:d")
        self.write_words(module, { id: local.name.lexeme for id, local in function.locals.items() }, function.body.words)
        if uses_stack:
            self.write_line("local.get $stac:k global.set $stac:k")
        self.dedent()
        self.write_line(")")

    def write_mem(self, name: str, size: int, scope: ScopeId, shadow: int) -> None:
        self.write_indent()
        self.write(f"global.get $stac:k global.get $stac:k i32.const {align_to(size, 4)} i32.add global.set $stac:k local.set ${name}")
        if scope != ROOT_SCOPE or shadow != 0:
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
                self.write(f".store local.tee ${local.name.lexeme} i32.const {align_to(local.taip.size(), 4)} i32.add global.set $stac:k\n")

    def write_locals(self, locals: Dict[LocalId, Local]) -> None:
        for local_id, local in locals.items():
            if isinstance(local, ParameterLocal):
                if local.needs_moved_into_memory() and not local.can_be_abused_as_ref():
                    self.write_line(f"(local ${local.name.lexeme} i32)")
                continue
            local = locals[local_id]
            self.write_indent()
            self.write(f"(local ${local.name.lexeme}")
            if local_id.scope != ROOT_SCOPE or local_id.shadow != 0:
                self.write(f":{local_id.scope}:{local_id.shadow}")
            self.write(" ")
            if local.lives_in_memory():
                self.write("i32")
            else:
                self.write_type(local.taip)
            self.write(")\n")

    def write_words(self, module: int, locals: Dict[LocalId, str], words: List[Word]) -> None:
        for word in words:
            self.write_word(module, locals, word)

    def write_local_ident(self, locals: Dict[LocalId, str], local: LocalId) -> None:
        if local.scope != ROOT_SCOPE or local.shadow != 0:
            self.write(f"${locals[local]}:{local.scope}:{local.shadow}")
        else:
            self.write(f"${locals[local]}")

    def write_word(self, module: int, locals: Dict[LocalId, str], word: Word) -> None:
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
                    self.write_local_ident(locals, local_id)
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
                    self.write_local_ident(locals, local_id)
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
                        self.write(f"call ${function_handle.module}:{function.name.lexeme}")
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
                if isinstance(taip, PtrType) or taip == PrimitiveType.I32 or taip == PrimitiveType.I8:
                    self.write_line("i32.add")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.add")
                    return
                assert_never(taip)
            case IntrinsicSub(token, taip):
                if isinstance(taip, PtrType) or taip == PrimitiveType.I32 or taip == PrimitiveType.I8:
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
                if taip == PrimitiveType.I32 or taip == PrimitiveType.I8:
                    self.write_line("i32.ge_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.ge_u")
                    return
                assert_never(taip)
            case IntrinsicGreater(_, taip):
                if taip == PrimitiveType.I32 or taip == PrimitiveType.I8:
                    self.write_line("i32.gt_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.gt_u")
                    return
                assert_never(taip)
            case IntrinsicLessEq(_, taip):
                if taip == PrimitiveType.I32 or taip == PrimitiveType.I8:
                    self.write_line("i32.le_u")
                    return
                if taip == PrimitiveType.I64:
                    self.write_line("i64.le_u")
                    return
                assert_never(taip)
            case IntrinsicLess(_, taip):
                if taip == PrimitiveType.I32 or taip == PrimitiveType.I8:
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
            case IntrinsicShl(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.rotl")
                else:
                    self.write_line("i32.shl")
            case IntrinsicShr(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.shr")
                else:
                    self.write_line("i32.shr_u")
            case IntrinsicRotl(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.rotl")
                else:
                    self.write_line("i32.rotl")
            case IntrinsicRotr(token, taip):
                if taip == PrimitiveType.I64:
                    self.write_line("i64.rotr")
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
            case IntrinsicMod(_, taip):
                match taip:
                    case PrimitiveType.I8 | PrimitiveType.I32:
                        self.write_line("i32.rem_u")
                    case PrimitiveType.I64:
                        self.write_line("i64.rem_u")
                    case _:
                        assert_never(taip)
            case IntrinsicDiv(_, taip):
                match taip:
                    case PrimitiveType.I32 | PrimitiveType.I8:
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
                    self.write_line("i64.extend_i32_u ;; cast to i64")
                    return
                if (source == PrimitiveType.BOOL or source == PrimitiveType.I32) and taip == PrimitiveType.I8: 
                    self.write_line(f"i32.const 0xFF i32.and ;; cast to {format_type(taip)}")
                    return
                if source == PrimitiveType.I64 and taip == PrimitiveType.I8: 
                    self.write_line(f"i64.const 0xFF i64.and i32.wrap_i64 ;; cast to {format_type(taip)}")
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
                    self.write_local_ident(locals, local_id)
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
                    self.write("i32.load8_u\n")
                else:
                    self.write_type(taip)
                    self.write(".load\n")
            case BreakWord():
                self.write_line("br $block")
            case BlockWord(token, body, parameters, returns):
                self.write_indent()
                self.write("(block $block")
                self.write_parameters(parameters)
                self.write_returns(returns)
                self.write("\n")
                self.indent()
                self.write_words(module, locals, body.words)
                self.dedent()
                self.write_line(")")
            case LoopWord(_, body, parameters, returns, diverges):
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
                self.write_words(module, locals, body.words)
                self.write_line("br $loop")
                self.dedent()
                self.write_line(")")
                self.dedent()
                self.write_line(")")
                if diverges:
                    self.write_line("unreachable")
            case IfWord(_, parameters, returns, true_branch, false_branch, diverges):
                self.write_indent()
                self.write("(if")
                self.write_parameters(parameters)
                self.write_returns(returns or [])
                self.write("\n")
                self.indent()
                self.write_line("(then")
                self.indent()
                self.write_words(module, locals, true_branch.words)
                self.dedent()
                self.write_line(")")
                if len(false_branch.words) > 0:
                    self.write_line("(else")
                    self.indent()
                    self.write_words(module, locals, false_branch.words)
                    self.dedent()
                    self.write_line(")")
                self.dedent()
                self.write_line(")")
                if diverges:
                    self.write_line("unreachable")
            case StructWord(_, taip, copy_space_offset, body):
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
                self.write_words(module, locals, body.words)
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
                    if taip.size() <= 8:
                        for i in range(len(fields), 0, -1):
                            offset = struct.field_offset(i - 1)
                            if i != len(fields) and (offset != 0 or taip.size() > 4):
                                if taip.size() <= 4:
                                    self.write("call $intrinsic:flip ")
                                else:
                                    self.write("call $intrinsic:flip-i32-i64 ")
                                    self.flip_i32_i64_used = True
                            if taip.size() > 4:
                                self.write("i64.extend_i32_u ")
                            if offset != 0:
                                if taip.size() <= 4:
                                    self.write(f"i32.const {offset * 8} i32.shl ")
                                else:
                                    self.write(f"i64.const {offset * 8} i64.shl ")
                            if i != len(fields):
                                if taip.size() <= 4:
                                    self.write("i32.or ")
                                else:
                                    self.write("i64.or ")
                        self.write(f";; make {format_type(taip)}\n")
                        return
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
                def go(remaining_cases: List[MatchCase]):
                    if len(remaining_cases) == 0:
                        if default is None:
                            if len(cases) != 0:
                                self.write("unreachable")
                            return
                        self.write_words(module, locals, default.words)
                        return
                    case = remaining_cases[0]
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
                    variant_inhabits_i64 = variant.size() <= 8 and variant.size() > 4 and not by_ref
                    if variant_inhabits_i64:
                        self.write(" (param i64)")
                    else:
                        self.write(" (param i32)")

                    if returns is not None:
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
                    elif variant_inhabits_i64:
                        self.write_line("i32.wrap_i64")
                    self.write_words(module, locals, case.body.words)
                    self.dedent()
                    self.write_line(")")
                    self.write_indent()
                    if len(remaining_cases) == 1 and default is not None:
                        self.write("(else\n")
                        self.indent()
                        go(remaining_cases[1:])
                        self.dedent()
                        self.write_indent()
                        self.write("))")
                    else:
                        self.write("(else ")
                        go(remaining_cases[1:])
                        self.write("))")
                self.write_line(f";; match on {variant.name.lexeme}")
                self.write_indent()
                go(cases)
                if returns is None:
                    if len(cases) != 0 or default is not None:
                        self.write("\n")
                        self.write_indent()
                    self.write("unreachable")
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
                        self.flip_i64_i32_used = True
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
                assert_never(other)

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
                    self.write_local_ident(locals, local_id)
                    return
                case GlobalId():
                    globl = self.globals[local_id]
                    self.write(f"${globl.name.lexeme}:{local_id.module}")
                    return
                case other:
                    assert_never(other)
        if not target_lives_in_memory and len(loads) == 0:
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
        last_load = loads[-1]
        if not target_lives_in_memory and last_load.is_bitshift():
            for i, load in enumerate(loads):
                if all(load.is_bitshift() for load in loads[i:]):
                    break
                self.write(f" {load}")
            if isinstance(last_load, I32InI64) or isinstance(last_load, I8InI64):
                self.write(f" i64.const {uhex(0xFFFFFFFF_FFFFFFFF ^ (0xFFFFFFFF << (last_load.offset * 8)))} i64.and ")
            if isinstance(last_load, I8InI32):
                self.write(f" i32.const {uhex(0xFF << (last_load.offset * 8))} i32.and ")

            self.write("call $intrinsic:flip-i32-i64 i64.extend_i32_u ")
            self.flip_i32_i64_used = True
            if isinstance(last_load, I32InI64) and last_load.offset != 0:
                self.write(f"i64.const {last_load.offset * 8} i64.shl ")
            if isinstance(last_load, I8InI32) and last_load.offset != 0:
                self.write(f"i32.const {last_load.offset * 8} i32.shl ")
            self.write("i32.or " if isinstance(last_load, I8InI32) else "i64.or ")
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

    def write_signature(self, module: int, name: Token, export_name: Token | None, signature: FunctionSignature, instance_id: int | None, locals: Dict[LocalId, Local]) -> None:
        self.write(f"func ${module}:{name.lexeme}")
        if instance_id is not None and instance_id != 0:
            self.write(f":{instance_id}")
        if export_name is not None:
            self.write(f" (export {export_name.lexeme})")
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
        if self.flip_i32_i64_used:
            self.write_line("(func $intrinsic:flip-i32-i64 (param $a i32) (param $b i64) (result i64 i32) local.get $b local.get $a)")
        if self.flip_i64_i32_used:
            self.write_line("(func $intrinsic:flip-i64-i32 (param $a i64) (param $b i32) (result i32 i64) local.get $b local.get $a)")
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
                    name = f"${handle.module}:{function.name.lexeme}:{handle.instance}"
                else:
                    name = f"${handle.module}:{function.name.lexeme}"
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
            size = globl.taip.size()
            lives_in_memory = globl.was_reffed or not globl.taip.can_live_in_reg()
            initial_value = ptr if lives_in_memory else 0
            taip = "i64" if not lives_in_memory and size > 4 and size <= 8 else "i32"
            self.write(f"(global ${globl.name.lexeme}:{global_id.module} (mut {taip}) ({taip}.const {initial_value}))\n")
            if not lives_in_memory:
                continue
            ptr += globl.taip.size()
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
        self.write(extern.extern_module)
        self.write(" ")
        self.write(extern.extern_name)
        self.write(" (")
        self.write_signature(module_id, extern.name, None, extern.signature, None, {})
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
    resolved_modules = resolve_modules({ k: m.top_items for k,m in modules.items()})
    if mode == "check":
        return format_dict({ k: v for k,v in resolved_modules.items() })
    function_table, mono_modules = Monomizer(resolved_modules).monomize()
    if mode == "monomize":
        return "TODO"
    for mono_module in mono_modules.values():
        merge_locals_module(mono_module)
    return WatGenerator(mono_modules, function_table, guard_stack).write_wat_module()

def main(argv: List[str], stdin: str | None = None) -> str:
    if len(argv) == 1:
        print("provide a command")
        exit(1)
    if argv[1] == "units":
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

