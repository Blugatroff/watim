#!/usr/bin/sh
#set -o xtrace
cargo run --quiet -- com $1 1024 > out.wat \
    && wat2wasm out.wat && wasm-opt --enable-multivalue -O3 ./out.wasm -o ./opt.wasm \
    && wasmtime --dir=. out.wat "${@:2}"
