#!/usr/bin/sh
set -o xtrace
#cargo run --quiet -- sim $1
cargo run --quiet -- com $1 > out.wat && wat2wasm out.wat && wasm-opt --enable-multivalue -O3 ./out.wasm -o ./opt.wasm && wasm3 opt.wasm
