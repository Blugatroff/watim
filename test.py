#!/usr/bin/env python
from dataclasses import dataclass
from typing import List
import subprocess
import glob
import json
import sys
import os
import difflib

from bootstrap import main, ParserException, ResolverException

if not os.path.isfile("test.wat") and subprocess.run("python bootstrap.py ./test.watim > test.wat", shell=True).returncode != 0:
    exit(1)

def parse_test_file(path: str):
    output = subprocess.run(f"wasmtime --dir=. -- test.wat read {path}", shell=True, stdout=subprocess.PIPE)
    if output.returncode != 0:
        return None
    return json.loads(output.stdout)

def write_test_file(path: str, test: dict):
    output = subprocess.run(["wasmtime", "--dir=.", "--", "./test.wat", "write", path], input=bytes(json.dumps(test), 'UTF-8'))
    if output.returncode != 0:
        print(output, file=sys.stderr)
        return exit(1)

@dataclass
class CompilerOutput:
    returncode: int
    stdout: str
    stderr: str

watim_bin_path = None
if "--native" in sys.argv:
    if subprocess.run("python bootstrap.py ./v2/main.watim > watim.wat", shell=True).returncode != 0:
        exit(1)
    watim_bin_path = os.path.realpath("./watim.wat")

def run_native_compiler(args: List[str] | None, stdin: str):
    compiler = subprocess.run(["wasmtime", "--dir=.", "--", watim_bin_path] + (args or ["-"]), input=bytes(test["compiler-stdin"], 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return CompilerOutput(compiler.returncode, compiler.stdout.decode("UTF-8").strip(), compiler.stderr.decode("UTF-8").strip())

def run_bootstrap_compiler(args: List[str] | None, stdin: str):
    # compiler = subprocess.run(["python", "./bootstrap.py", "-"] + args, input=bytes(test["compiler-stdin"], 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # return CompilerOutput(compiler.returncode, compiler.stdout.decode("UTF-8").strip(), compiler.stderr.decode("UTF-8").strip())
    try:
        stdout = main([sys.argv[0]] + (args or ["-"]), stdin)
        return CompilerOutput(0, stdout.strip(), "")
    except ParserException as e:
        return CompilerOutput(1, "", e.display().strip())
    except ResolverException as e:
        return CompilerOutput(1, "", e.display().strip())
    except Exception as e:
        return CompilerOutput(1, "", str(e))

if len(sys.argv) > 2 and sys.argv[1] == "accept":
    path = sys.argv[2]
    test = parse_test_file(path)
    os.chdir("./tests/fixtures")
    if "--native" in sys.argv:
        compiler = run_native_compiler(test['compiler-args'], test['compiler-stdin'])
    else:
        compiler = run_bootstrap_compiler(test['compiler-args'], test['compiler-stdin'])
    os.chdir("../..")
    stdout = None
    stderr = None
    status = None
    if compiler.returncode == 0 and test['status'] is not None:
        with open('./out.wat', 'wb') as outwat:
            outwat.write(compiler.stdout.encode("UTF-8"))
        program = subprocess.run(["wasmtime", "./out.wat"], input=bytes(test["stdin"] or "", 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = None if test['stdout'] is None else program.stdout
        stderr = None if test['stderr'] is None else program.stderr
        status = None if test['status'] is None else program.returncode
    write_test_file(path, {
        "compiler-stdin": test['compiler-stdin'],
        "compiler-args": test['compiler-args'],
        "compiler-stdout": None if test['compiler-stdout'] is None else compiler.stdout,
        "compiler-stderr": None if test['compiler-stderr'] is None else compiler.stderr,
        "compiler-status": None if test['compiler-status'] is None else compiler.returncode,
        "stdin": test['stdin'],
        "stdout": stdout,
        "stderr": stderr,
        "status": status,
    })
    exit(0)

if len(sys.argv) > 2 and sys.argv[1] == "--native":
    pattern = sys.argv[2]
elif len(sys.argv) == 2 and sys.argv[1] == "--native":
    pattern = "./tests/*.watim"
elif len(sys.argv) > 1:
    pattern = sys.argv[1]
else:
    pattern = "./tests/*.watim"
tests = glob.glob(pattern)

def print_mismatch(expected: str, actual: str):
    for line in difflib.unified_diff(expected.splitlines(), actual.splitlines(), fromfile='expected', tofile='actual'):
        print(line, file=sys.stderr)

@dataclass
class Test:
    compiler_stdin: str | None
    compiler_status: int | None
    compiler_stdout: str | None
    compiler_stderr: str | None
    stdin: str | None
    status: int | None
    stdout: str | None
    stderr: str | None

native_tests = list(map(lambda p: f"./tests/{p}.watim", [
    "lex-make",
    "import-struct",
    "import-list",
    "resolve-struct",
    "resolve-global",
    "resolve-extern",
    "import-list-unnamed",
    "parse-generic-type",
    "resolve-function",
    "parse-annotation-no-returns",
]))

failed = False
for path in tests:
    if "--native" in sys.argv and path not in native_tests:
        continue
    test = parse_test_file(path)
    previous_failed = failed
    failed = True
    if test is None:
        print(f"{path}: failed to parse test file", file=sys.stderr)
        exit(1)
    if test["compiler-stdin"] is None:
        print(f"{path}: compiler-stdin ist missing", file=sys.stderr)
        continue
    os.chdir("./tests/fixtures")
    if "--native" in sys.argv:
        compiler = run_native_compiler(test['compiler-args'], test['compiler-stdin'])
    else:
        compiler = run_bootstrap_compiler(test['compiler-args'], test['compiler-stdin'])
    os.chdir("../..")
    if test['compiler-status'] is not None and compiler.returncode != test['compiler-status']:
        print(f"{path}: expected different compiler status:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-status']}", file=sys.stderr)
        print(f"Actual:\n{compiler.returncode}", file=sys.stderr)
        print(f"compiler-stderr was: {compiler.stderr}", file=sys.stderr)
        continue
    if test['compiler-stderr'] is not None and compiler.stderr != test['compiler-stderr'].strip():
        print(f"{path}: expected different compiler stderr:", file=sys.stderr)
        print_mismatch(test['compiler-stderr'], compiler.stderr)
        continue
    if test['compiler-stdout'] is not None and compiler.stdout != test['compiler-stdout'].strip():
        print(f"{path}: expected different compiler stdout:", file=sys.stderr)
        print_mismatch(test['compiler-stdout'], compiler.stdout)
        print(f"stderr was: {compiler.stderr}", file=sys.stderr)
        continue
    with open('./out.wat', 'wb') as outwat:
        outwat.write(compiler.stdout.encode("UTF-8"))
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
