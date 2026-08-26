#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
assemblyscript_dir="$workspace_dir/minds/assemblyscript_wait_mind"
go_dir="$workspace_dir/minds/go_wait_mind"
tinygo_bin="${TINYGO_BIN:-tinygo}"
go_bin="${GO_BIN:-go}"
expected_tinygo_version="0.41.1"
wasmopt_bin="$assemblyscript_dir/node_modules/binaryen/bin/wasm-opt"

npm ci --prefix "$assemblyscript_dir"
npm run --prefix "$assemblyscript_dir" build

if ! command -v "$tinygo_bin" >/dev/null 2>&1; then
    echo "TinyGo is required to build the non-WASI Go conformance Mind." >&2
    echo "Install TinyGo or set TINYGO_BIN to its executable path." >&2
    exit 1
fi
tinygo_version="$("$tinygo_bin" version)"
if [[ "$tinygo_version" != "tinygo version $expected_tinygo_version "* ]]; then
    echo "TinyGo $expected_tinygo_version is required; found: $tinygo_version" >&2
    exit 1
fi
if ! command -v "$go_bin" >/dev/null 2>&1; then
    echo "Go is required by TinyGo; install it or set GO_BIN to its executable path." >&2
    exit 1
fi
go_version="$("$go_bin" version)"
if [[ ! "$go_version" =~ go1\.(19|2[0-6])([.[:space:]]|$) ]]; then
    echo "TinyGo $expected_tinygo_version requires Go 1.19 through 1.26; found: $go_version" >&2
    echo "Set GO_BIN to a compatible Go executable." >&2
    exit 1
fi
go_bin_dir="$(cd "$(dirname "$go_bin")" && pwd)"

mkdir -p "$go_dir/build"
mkdir -p "$go_dir/build/cache"
(
    cd "$go_dir"
    PATH="$go_bin_dir:$PATH" GOCACHE="$go_dir/build/cache" WASMOPT="$wasmopt_bin" "$tinygo_bin" build \
        -target=wasm-unknown \
        -buildmode=c-shared \
        -no-debug \
        -o build/go_wait_mind.wasm \
        .
)
