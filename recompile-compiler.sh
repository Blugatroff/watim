#!/usr/bin/bash
set -xe
./run.sh ./native/main.watim ./native/main.watim > watim.wat
wat2wasm ./watim.wat -o ./watim.wasm
rm ./watim.wat
