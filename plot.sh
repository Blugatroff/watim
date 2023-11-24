#!/usr/bin/env bash

wasmtime --dir=. ./watim.wasm ./native/main.watim -q > watim.wat 
(wasmtime --dir=. ./watim.wat ./native/main.watim --arena-graphs 3>&2 2>&1 1>&3) 2>/dev/null | gnuplot -p

