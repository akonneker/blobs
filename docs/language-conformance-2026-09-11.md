# Fresh guest conformance — 2026-09-11

The maintained Rust, AssemblyScript and Go guests were rebuilt locally and all
18 guest/runtime gates from `scripts/conformance.sh` passed: 23 tests, including
the three normally ignored language-artifact tests. Every gate required a
positive executed-test count and rejected `Skipping:` output. This supplements
the earlier workspace tests with explicit evidence for the generated artifacts.

## Build identity

All five maintained Rust Minds and both ABI canaries were built with the locked
workspace for `wasm32-unknown-unknown` in release mode. AssemblyScript used its
locked npm dependencies (compiler 0.27.37, Extism PDK 1.0.0). The Go canary used
TinyGo 0.41.1, Go 1.25.1 and Extism Go PDK 1.1.1, matching the versions pinned
in the full-conformance workflow. The existing system Go 1.27 was outside the
build script's allowed range, so the compatible tools were installed only under
`/private/tmp/blob-conformance-tools-20260911` with temporary dependency caches.

The downloaded toolchains were verified against the SHA-256 values published
by the [TinyGo release](https://github.com/tinygo-org/tinygo/releases/tag/v0.41.1)
and [Go downloads](https://go.dev/dl/#go1.25.1):

| Archive | SHA-256 |
| --- | --- |
| tinygo0.41.1.darwin-arm64.tar.gz | `c684d154d89a452cc9c7fc5dc5fc80cb6a42445b3e44b3c12ed048692de0f341` |
| go1.25.1.darwin-arm64.tar.gz | `68deebb214f39d542e518ebb0598a406ab1b5a22bba8ec9ade9f55fb4dd94a6c` |

Exact build/test commands, source hashes, nine WASM artifact hashes/sizes and
per-gate logs are retained under `training-output/language-conformance-2026-09-11`.
`run_guest_gates.py` selects the guest-specific gates from a saved copy of the
conformance script and enforces a local execution deadline.

## Executed contracts and limits

The gates check structural language-PDK admission; Extism versus the compatible
executor; fresh-instance/private-memory behavior; worker/hash invariance;
required export and forbidden-import rejection; maintained Mind execution;
checkpoint/replay continuation; server re-execution; and the AS/Go canaries'
canonical Wait-with-retained-memory behavior. Five compatible-executor unit
tests are included.

This is deterministic fixture conformance. It does not qualify arbitrary
executable Wasm mutation, deterministic fuel policy or learned-policy export.
The complete `scripts/conformance.sh` was not rerun here: unrelated workspace,
schema and browser gates already have separate current evidence. The existing
weekly/manual full-conformance workflow remains the routine cross-language gate.
