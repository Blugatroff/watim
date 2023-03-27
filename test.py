#!/usr/bin/env python
import subprocess
import glob
import json
import pathlib

tests = glob.glob("./tests/*.watim")

def compile(path):
    cmd = "wasmtime --dir=. ./watim.wasm -- -q " + path
    print("+", cmd)
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def run(wat: str, stdin: str):
    cmd = "wasmtime ./out.wat"
    with open('./out.wat', 'wb') as outwat:
        outwat.write(wat)
    return subprocess.run(cmd, shell=True, input=bytes(stdin, 'ASCII'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def check_equal(expected, got):
    if expected != got:
        print("expected: ", expected)
        print("got:      ", got)
        print("failed")
        exit(1)

for path in tests:
    compilation_output = compile(path)

    with open(pathlib.Path(path).with_suffix('.json'), 'r') as reader:
        spec = json.load(reader)

        if 'stdout' in spec['compilation']:
            check_equal(bytes(spec['compilation']['stdout'], 'ASCII'), compilation_output.stdout)

        if 'stderr' in spec['compilation']:
            check_equal(bytes(spec['compilation']['stderr'], 'ASCII'), compilation_output.stderr)

        if 'runtime' in spec:
            runtime_spec = spec['runtime']
            if 'stdin' in runtime_spec:
                stdin = runtime_spec['stdin'] 
            else:
                stdin = ''
            run_output = run(compilation_output.stdout, stdin)
            if 'stdout' in runtime_spec:
                check_equal(bytes(runtime_spec['stdout'], 'ASCII'), run_output.stdout)

            if 'stderr' in runtime_spec:
                check_equal(bytes(runtime_spec['stderr'], 'ASCII'), run_output.stderr)
