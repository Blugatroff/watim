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

if subprocess.run(f"python bootstrap.py ./test.watim > test.wat", shell=True).returncode != 0:
    exit(1)

def parse_test_file(path: str):
    output = subprocess.run(f"wasmtime --dir=. -- test.wat {path}", shell=True, stdout=subprocess.PIPE)
    if output.returncode != 0:
        return None
    return json.loads(output.stdout)

tests = glob.glob('./tests/*.watim')
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
    compiler = subprocess.run(["python", "./bootstrap.py", "-"], input=bytes(test["compiler-stdin"], 'UTF-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if test['compiler-stderr'] is not None and compiler.stderr.strip() != test['compiler-stderr'].encode('UTF-8').strip():
        print(f"{path}: expected different compiler stderr:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-stderr']}", file=sys.stderr)
        print(f"Actual:\n{compiler.stderr.decode('UTF-8')}", file=sys.stderr)
        continue
    if test['compiler-stdout'] is not None and compiler.stdout.strip() != test['compiler-stdout'].encode('UTF-8').strip():
        print(f"{path}: expected different compiler stdout:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-stdout']}", file=sys.stderr)
        print(f"Actual:\n{compiler.stdout.decode('UTF-8')}", file=sys.stderr)
        continue
    if test['compiler-status'] is not None and compiler.returncode != test['compiler-status']:
        print(f"{path}: expected different compiler status:", file=sys.stderr)
        print(f"Expected:\n{test['compiler-status']}", file=sys.stderr)
        print(f"Actual:\n{compiler.returncode}", file=sys.stderr)
        if test['compiler-stderr'] is None:
            print(f"compiler-stderr was: {compiler.stderr.decode('UTF-8')}", file=sys.stderr)
        continue
    with open('./out.wat', 'wb') as outwat:
        outwat.write(compiler.stdout)
        cmd = 'wasmtime ./out.wat'
    if compiler.returncode == 0:
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
            continue
    failed = previous_failed
    print(f"{path} passed")

if failed:
    exit(1)
