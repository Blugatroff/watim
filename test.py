#!/usr/bin/env python
from dataclasses import dataclass
from typing import Any, TypeVar, Callable
import subprocess
import glob
import json
import pathlib

from termcolor import colored

def lookup(dictionary: dict, key: str) -> Any | None:
    if key in dictionary:
        return dictionary[key]
    else:
        return None

def check_if_string(s: Any, name: str) -> str:
    if isinstance(s, str):
        return s
    else:
        raise ValueError(name + ' was not a string ' + s)

def check_if_int(i: int, name: str) -> int:
    if isinstance(i, int):
        return i
    else:
        raise ValueError(name + ' was not an int ' + i)

def check_if_string_or_none(s: Any, name: str) -> str | None:
    if s == None:
        return None
    else:
        return check_if_string(s, name)

@dataclass
class Spec:
    code: int
    stdout: str | None
    stderr: str | None
    stdin: str | None

    @staticmethod
    def parse(spec: dict):
        codeany = lookup(spec, 'code')
        if codeany == None:
            raise ValueError('spec.code is missing')
        code = check_if_int(codeany, 'spec.code')
        stdout = check_if_string_or_none(lookup(spec, 'stdout'), 'spec.stdout')
        stderr = check_if_string_or_none(lookup(spec, 'stderr'), 'spec.stderr')
        stdin = check_if_string_or_none(lookup(spec, 'stdin'), 'spec.stdin')
        return Spec(code=code, stdout=stdout, stderr=stderr, stdin=stdin)

@dataclass
class TestSpec:
    compilation: Spec
    runtime: Spec | None

    @staticmethod
    def parse(spec: dict):
        compilation_any = lookup(spec, 'compilation')
        if compilation_any == None:
            raise ValueError('compilation is missing')
        compilation = Spec.parse(compilation_any)
        runtime_any = lookup(spec, 'runtime')
        if runtime_any == None:
            runtime = None
        else:
            runtime = Spec.parse(runtime_any)
        return TestSpec(compilation=compilation, runtime=runtime)

tests = glob.glob("./tests/*.watim")

def compile(path, onCmd: Callable[[str], None]):
    cmd = "wasmtime --dir=. ./watim.wasm -- -q " + path
    onCmd(cmd)
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run(onCmd: Callable[[str], None], wat: bytes, stdin: str):
    with open('./out.wat', 'wb') as outwat:
        outwat.write(wat)
    cmd = "wasmtime ./out.wat"
    onCmd(cmd)
    return subprocess.run(cmd, shell=True, input=bytes(stdin, 'ASCII'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

T = TypeVar('T')
def check_equal(expected: T, got: T, name: str):
    if expected != got:
        print(name)
        print("    expected: ", expected)
        print("    got:      ", got)
        print("    failed")
        exit(1)

for path in tests:
    print(colored("Running test " + path, 'cyan'))
    onCmd= lambda cmd: print("  + " + colored(cmd, 'magenta'))
    compilation_output = compile(path, onCmd)

    with open(pathlib.Path(path).with_suffix('.json'), 'r') as reader:
        spec = json.load(reader)

        spec = TestSpec.parse(spec)

        check_equal(spec.compilation.code, compilation_output.returncode, 'compilation exit code')

        if spec.compilation.stdout != None:
            check_equal(bytes(spec.compilation.stdout, 'ASCII'), compilation_output.stdout, 'compilation stdout')

        if spec.compilation.stderr != None:
            check_equal(bytes(spec.compilation.stderr, 'ASCII'), compilation_output.stderr, 'compilation stderr')

        if spec.runtime != None:
            stdin: str = spec.runtime.stdin or ''
            runtime_output = run(onCmd, compilation_output.stdout, stdin)
            check_equal(spec.runtime.code, runtime_output.returncode, 'runtime exit code')
            if spec.runtime.stdout != None:
                check_equal(bytes(spec.runtime.stdout, 'ASCII'), runtime_output.stdout, 'runtime stdout')

            if spec.runtime.stderr != None:
                check_equal(bytes(spec.runtime.stderr, 'ASCII'), runtime_output.stderr, 'runtime sterr')
    print(colored('PASSED', 'green'))
    print()

