#!/usr/bin/env sh
./recompile-compiler.sh
hyperfine --warmup 5 "wasmtime --dir=. ./watim.wasm ./native/main.watim"

