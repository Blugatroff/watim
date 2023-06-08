#!/usr/bin/env sh
wasmtime --dir=. ./watim.wasm $1 -- -q > out.wat && wasmtime --dir=. ./out.wat -- "${@:2}"
