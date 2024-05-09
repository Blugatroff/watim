#!/usr/bin/env python
from dataclasses import dataclass, asdict
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
class Token:
    ty: str
    line: int
    column: int
    lexeme: str


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
    expected: Result[List[dict]]
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
                expected = json.dumps(top_items, indent="  ")
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
    LexerTest("fn", Success([Token("TOKEN_FN", 1, 1, "fn")])),
    LexerTest(".foo", Success([Token("TOKEN_DOT", 1, 1, "."), Token("TOKEN_IDENT", 1, 2, "foo")])),
    ParserTest("fn foo() { }", Success([{ "ty": "function", "ident": "foo", "export": None, "body": [], "locals": []}])),
]

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
