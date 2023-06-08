#!/usr/bin/env bash
set -xe

wasmtime --dir=. ./watim.wasm ./native/main.watim > out.wat && wasmtime --dir=. ./out.wat -- ./native/main.watim > watim.wat
wat2wasm ./watim.wat -o ./watim.wasm
rm ./watim.wat

