#!/usr/bin/sh
wasmtime --dir=. ./out.wat $1 > a.wat && wasmtime --dir=. ./a.wat "${@:2}"
