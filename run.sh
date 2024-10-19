#!/usr/bin/env sh

set -eu

if [ ! -f $(dirname "$0")/watim.wasm ]; then
    ./bootstrap-native.sh
    echo "Bootstrapping done" >> /dev/stderr
fi

wasmtime --dir=. -- $(dirname "$0")/watim.wasm compile $1 -q > out.wat
echo "Compiled successfully" >> /dev/stderr

echo "Running $1:" >> /dev/stderr
wasmtime --dir=. -- ./out.wat "${@:2}"

rm out.wat

