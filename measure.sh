#!/usr/bin/env sh
(time (wasmtime --dir=. ./watim.wasm ./native/main.watim > /dev/null 2>&1)) |& grep user | rg 'm([0-9,]*)' -o -r '$1' | sed 's/,/./g'
