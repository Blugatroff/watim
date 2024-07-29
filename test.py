#!/usr/bin/env python
from dataclasses import dataclass, asdict
from typing import Any, TypeVar, Callable, List
from shutil import copyfile
import subprocess
import glob
import json
import pathlib
import sys
import os

from termcolor import colored

from bootstrap import main, ParserException, ResolverException

if subprocess.run(f"python bootstrap.py ./test.watim > test.wat", shell=True).returncode != 0:
    exit(1)

def parse_test_file(path: str):
    output = subprocess.run(f"wasmtime --dir=. -- test.wat {path}", shell=True, stdout=subprocess.PIPE)
    if output.returncode != 0:
        return None
    return json.loads(output.stdout)

@dataclass
class CompilerOutput:
    returncode: int
    stdout: str
    stderr: str

already_compiled = False
def run_native_compiler(args: List[str], stdin: str):
    global already_compiled
    if not already_compiled:
        already_compiled = True
        if subprocess.run(f"python bootstrap.py ./v2/main.watim > watim.wat", shell=True).returncode != 0:
            exit(1)
    compiler = subprocess.run(["wasmtime", "--dir=.", "--", "./watim.wat"] + args + ["-"], input=bytes(test["compiler-stdin"], 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return CompilerOutput(compiler.returncode, compiler.stdout.decode("UTF-8").strip(), compiler.stderr.decode("UTF-8").strip())

def run_bootstrap_compiler(args: List[str], stdin: str):
    # compiler = subprocess.run(["python", "./bootstrap.py", "-"] + args, input=bytes(test["compiler-stdin"], 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # return CompilerOutput(compiler.returncode, compiler.stdout.decode("UTF-8").strip(), compiler.stderr.decode("UTF-8").strip())
    try:
        stdout = main([sys.argv[0], "-"] + args, stdin)
        return CompilerOutput(0, stdout.strip(), "")
    except ParserException as e:
        return CompilerOutput(1, "", e.display().strip())
    except ResolverException as e:
        return CompilerOutput(1, "", e.display().strip())

if len(sys.argv) > 2 and sys.argv[1] == "--native":
    pattern = sys.argv[2]
elif len(sys.argv) > 1:
    pattern = sys.argv[1]
else:
    pattern = "./tests/*.watim"
tests = glob.glob(pattern)
failed = False
for path in tests:
    test = parse_test_file(path)
    previous_failed = failed
    failed = True
    if test is None:
        print(f"{path}: failed to parse test file", file=sys.stderr)
        exit(1)
    if test["compiler-stdin"] is None:
        print(f"{path}: compiler-stdin ist missing", file=sys.stderr)
        continue
    if "--native" in sys.argv:
        compiler = run_native_compiler(test['compiler-args'] if test['compiler-args'] is not None else [], test['compiler-stdin'])
    else:
        compiler = run_bootstrap_compiler(test['compiler-args'] if test['compiler-args'] is not None else [], test['compiler-stdin'])
    if test['compiler-status'] is not None and compiler.returncode != test['compiler-status']:
        print(f"{path}: expected different compiler status:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-status']}", file=sys.stderr)
        print(f"Actual:\n{compiler.returncode}", file=sys.stderr)
        print(f"compiler-stderr was: {compiler.stderr}", file=sys.stderr)
        continue
    if test['compiler-stderr'] is not None and compiler.stderr != test['compiler-stderr'].strip():
        print(f"{path}: expected different compiler stderr:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-stderr']}", file=sys.stderr)
        print(f"Actual:\n{compiler.stderr}", file=sys.stderr)
        continue
    if test['compiler-stdout'] is not None and compiler.stdout != test['compiler-stdout'].strip():
        print(f"{path}: expected different compiler stdout:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-stdout']}", file=sys.stderr)
        print(f"Actual:\n{compiler.stdout}", file=sys.stderr)
        continue
    with open('./out.wat', 'wb') as outwat:
        outwat.write(compiler.stdout.encode("UTF-8"))
        cmd = 'wasmtime ./out.wat'
    if compiler.returncode == 0 and test['status'] is not None:
        program = subprocess.run(["wasmtime", "./out.wat"], input=bytes(test["stdin"] or "", 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if test['stderr'] is not None and program.stderr.strip() != test['stderr'].encode('UTF-8').strip():
            print(f"{path}: expected different stderr:", file=sys.stderr)
            print(f"Expected:\n{test['stderr']}", file=sys.stderr)
            print(f"Actual:\n{program.stderr.decode('UTF-8')}", file=sys.stderr)
            continue
        if test['stdout'] is not None and program.stdout.strip() != test['stdout'].encode('UTF-8').strip():
            print(f"{path}: expected different stdout:", file=sys.stderr)
            print(f"Expected:\n{test['stdout']}", file=sys.stderr)
            print(f"Actual:\n{program.stdout.decode('UTF-8')}", file=sys.stderr)
            continue
        if test['status'] is not None and program.returncode != test['status']:
            print(f"{path}: expected different status:", file=sys.stderr)
            print(f"Expected:\n{test['status']}", file=sys.stderr)
            print(f"Actual:\n{program.returncode}", file=sys.stderr)
            if test['stderr'] is None:
                print(f"stderr was: {program.stderr.decode('UTF-8')}", file=sys.stderr)
            continue
    failed = previous_failed
    print(f"{path} passed")

if failed:
    exit(1)
