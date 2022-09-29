#!/usr/bin/sh
wasmtime --dir=. ./watim.wasm $1 > out.wat && wasmtime --dir=. ./out.wat -- "${@:2}"
