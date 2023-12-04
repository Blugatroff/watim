#!/usr/bin/env bash
set -xe

wasmtime -C cache=n --dir=. ./watim.wasm ./native/main.watim > out.wat && wasmtime -C cache=n --dir=. -- ./out.wat ./native/main.watim > watim.wat
wat2wasm ./watim.wat -o ./watim.wasm
# rm ./watim.wat

