#!/usr/bin/sh

# cargo run --quiet $1 > out.wat && wasmtime run out.wat
#cargo run --quiet sim $1
cargo run --quiet com $1 > out.wat && wat2wasm out.wat && wasmtime out.wasm