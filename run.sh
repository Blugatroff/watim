#!/usr/bin/env sh

set -eu

if [ ! -f $(dirname "$0")/watim.wasm ]; then
    ./bootstrap-native.sh
    echo "Bootstrapping done"
fi

wasmtime --dir=. -- $(dirname "$0")/watim.wasm compile $1 -q > out.wat
echo "Compiled successfully"

echo "Running $1:"
wasmtime --dir=. -- ./out.wat "${@:2}"

rm out.wat

