#!/usr/bin/env sh
./recompile-compiler.sh
hyperfine "wasmtime --dir=. ./watim.wasm ./native/main.watim"

