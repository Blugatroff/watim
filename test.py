#!/usr/bin/env python
from dataclasses import dataclass, asdict
from typing import Any, TypeVar, Callable
from shutil import copyfile
import subprocess
import glob
import json
import pathlib
import sys
import os

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


default_code = '''extern "wasi_unstable" "proc_exit" fn proc_exit(code: i32)

fn main "_start" () {
    0 proc_exit
}
'''
default_spec = TestSpec(
    compilation=Spec(code=0, stdout=None, stderr="", stdin=None), 
    runtime=Spec(code=0, stdout="", stderr="", stdin=None)
)

if len(sys.argv) > 1:
    path_watim = lambda name: './tests/' + name + '.watim'
    path_json = lambda name: './tests/' + name + '.json'
    if sys.argv[1] == 'new':
        if len(sys.argv) < 3:
            print('./test.py new <name>')
            exit(1)
        name = sys.argv[2]
        if os.path.exists(path_watim(name)) or os.path.exists(path_json(name)):
            print(colored('the tests files already exist', 'red'))
            exit(1)

        with open(path_watim(name), 'w') as file:
            file.write(default_code)
        with open(path_json(name), 'w') as file:
            file.write(json.dumps(
                asdict(default_spec),
                indent=4
            ))
        exit(0)
    if sys.argv[1] == 'copy':
        if len(sys.argv) < 4:
            print('./test.py copy <template> <name>')
            exit(1)
        template = sys.argv[2]
        name = sys.argv[3]
        copyfile(path_watim(template), path_watim(name))
        copyfile(path_json(template), path_json(name))
        exit(0)

tests = glob.glob('./tests/*.watim')

def compile(path, onCmd: Callable[[str], None]):
    cmd = 'wasmtime --dir=. ./watim.wasm -- -q ' + path
    onCmd(cmd)
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run(onCmd: Callable[[str], None], wat: bytes, stdin: str):
    with open('./out.wat', 'wb') as outwat:
        outwat.write(wat)
    cmd = 'wasmtime ./out.wat'
    onCmd(cmd)
    return subprocess.run(cmd, shell=True, input=bytes(stdin, 'ASCII'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

T = TypeVar('T')
def check_equal(expected: T, got: T, name: str) -> bool:
    if expected != got:
        print(colored('  ' + name + ':', 'red'))
        print(colored('    expected: ', 'cyan'), colored(str(expected), 'green'))
        print(colored('    got:      ', 'cyan'), colored(str(got), 'red'))
        return True
    return False

quiet = True

def run_test(path: str, onCmd) -> bool:
    compilation_output = compile(path, onCmd)

    with open(pathlib.Path(path).with_suffix('.json'), 'r') as reader:
        spec = json.load(reader)

        spec = TestSpec.parse(spec)

        if check_equal(spec.compilation.code, compilation_output.returncode, 'compilation exit code'):
            return True

        if spec.compilation.stdout != None:
            if check_equal(bytes(spec.compilation.stdout, 'ASCII'), compilation_output.stdout, 'compilation stdout'):
                return True

        if spec.compilation.stderr != None:
            if check_equal(bytes(spec.compilation.stderr, 'ASCII'), compilation_output.stderr, 'compilation stderr'):
                return True

        if spec.runtime != None:
            stdin: str = spec.runtime.stdin or ''
            runtime_output = run(onCmd, compilation_output.stdout, stdin)
            if check_equal(spec.runtime.code, runtime_output.returncode, 'runtime exit code'):
                return True
            if spec.runtime.stdout != None:
                if check_equal(bytes(spec.runtime.stdout, 'ASCII'), runtime_output.stdout, 'runtime stdout'):
                    return True

            if spec.runtime.stderr != None:
                if check_equal(bytes(spec.runtime.stderr, 'ASCII'), runtime_output.stderr, 'runtime stderr'):
                    return True
    return False

some_test_failed = False
for path in tests:
    print(colored('Running test ' + path + ' ', 'cyan'))
    def onCmd(cmd):
        if not quiet:
            print("  + " + colored(cmd, 'magenta'))
    
    failed = run_test(path, onCmd)
    if not failed:
        print(colored('PASSED', 'green'))
        if not quiet:
            print()
        continue
    some_test_failed = True

if some_test_failed:
    exit(1)
