#!/usr/bin/env sh
set -xe

# Use the python bootstrapping compiler to compile the code of the native compiler.
python ./bootstrap.py compile ./native/main.watim > bootstrapped.wat

# Let the bootstrapped native compiler recompile itself.
wasmtime -C cache=n --dir=. bootstrapped.wat compile ./native/main.watim > watim.wat

# Compile WAT into a binary
wat2wasm watim.wat -o watim.wasm

rm ./bootstrapped.wat ./watim.wat
