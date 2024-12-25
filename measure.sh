#!/usr/bin/env sh
set -xeu
./bootstrap-native.sh
hyperfine --warmup 5 "wasmtime --dir=. ./watim.wasm compile ./native/main.watim"

