#!/usr/bin/sh

# cargo run --quiet $1 > out.wat && wasmtime run out.wat
cargo run --quiet $1 > out.wat && wat2wasm out.wat && wasm3 out.wasm