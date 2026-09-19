#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_dir"

cargo build --release --target wasm32-unknown-unknown --locked -p blob_web

expected_bindgen_version="0.2.100"
if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "blob_web.wasm was built, but wasm-bindgen CLI is required to generate browser glue." >&2
  echo "Install wasm-bindgen-cli ${expected_bindgen_version}, then rerun this script." >&2
  exit 1
fi

actual_bindgen_version="$(wasm-bindgen --version | awk '{print $2}')"
if [[ "$actual_bindgen_version" != "$expected_bindgen_version" ]]; then
  echo "wasm-bindgen CLI ${actual_bindgen_version} does not match crate ${expected_bindgen_version}." >&2
  exit 1
fi

wasm-bindgen \
  --target web \
  --out-dir blob_web/pkg \
  --out-name blob_web \
  target/wasm32-unknown-unknown/release/blob_web.wasm

cp blob_web/web/blob_web.js blob_web/pkg/index.js
