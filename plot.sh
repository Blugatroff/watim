#!/usr/bin/env bash

if ! grep -q ArenaGraph:enable ./native/main.watim; then
    echo "This script only works if the ArenaGraph module is enabled!"
    echo "To enable it, use ArenaGraph:enable before creating the first Arena"
    exit
fi

wasmtime --dir=. ./watim.wasm ./native/main.watim -q > watim.wat 
(wasmtime --dir=. ./watim.wat ./native/main.watim --arena-graph 3>&2 2>&1 1>&3) 2>/dev/null | gnuplot -p

