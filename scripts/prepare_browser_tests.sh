#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_dir"

bash scripts/build_browser_package.sh
cargo run --locked -p blob_web --example browser_fixture -- browser-tests/fixtures/generated
