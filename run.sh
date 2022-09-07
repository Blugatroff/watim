#!/usr/bin/sh
#set -o xtrace
cargo run --quiet -- com $1 1024 > out.wat \
    && wasmtime --dir=. out.wat "${@:2}"
