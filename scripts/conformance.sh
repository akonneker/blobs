#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_dir"

cargo fmt --all -- --check
cargo test -p blob_interface
cargo test -p blob_engine --lib
cargo test -p blob_engine \
  --test checkpoint \
  --test canonical_hashing \
  --test replay_bundle \
  --test replay_segments \
  --test replay_store \
  --test replay_driver \
  --test server_verification \
  --test online_service \
  --test parallel_resolution \
  --test reference_mind_boundary \
  --test reference_resolution \
  --test resolution_conformance \
  --test state_delta \
  --test replay_chain \
  --test replay_conformance
cargo test -p blob_rl
cargo test -p blob_web

cargo clippy -p blob_engine --lib --no-deps \
  --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p blob_web --lib --no-deps \
  --target wasm32-unknown-unknown -- -D warnings
cargo build --release --target wasm32-unknown-unknown -p blob_web
node --check blob_web/web/blob_web.js
bash -n scripts/build_browser_package.sh
bash -n scripts/build_language_minds.sh
bash scripts/build_language_minds.sh
cargo test -p blob_engine --test language_pdk_conformance -- --ignored --nocapture

# Build every maintained reference Mind and both ABI canaries. This prevents
# optional artifact lookup in unit tests from silently skipping WASM execution.
cargo build --release --target wasm32-unknown-unknown \
  -p simple_mind -p aggressive_mind -p defensive_mind -p explorer_mind \
  -p isolation_canary -p invalid_mind_canary
cargo test -p blob_game \
  game::tests::test_sequential_vs_parallel_consistency -- --exact
cargo test -p blob_game \
  game::tests::test_stateful_wasm_is_pristine_and_worker_invariant -- --exact
cargo test -p blob_game \
  game::tests::test_reference_wasm_path_is_worker_and_hash_invariant -- --exact
cargo test -p blob_game \
  game::tests::test_reference_native_wasm_is_pristine_memory_explicit_and_worker_invariant -- --exact
cargo test -p blob_game \
  game::tests::test_reference_native_wasm_requires_the_explicit_export -- --exact
cargo test -p blob_game \
  game::tests::test_maintained_minds_execute_through_the_reference_abi -- --exact
cargo test -p blob_game \
  game::tests::test_reference_checkpoint_restore_preserves_explicit_memory -- --exact
cargo test -p blob_game \
  game::tests::test_game_checkpoint_rejects_previous_versions -- --exact
cargo test -p blob_game \
  game::tests::test_reference_wasm_path_exposes_server_verification_tick -- --exact
cargo test -p blob_game \
  game::tests::test_server_verifier_reexecutes_the_submitted_wasm -- --exact
cargo test -p blob_game \
  game::tests::test_reference_game_checkpoint_resumes_with_identical_hashes -- --exact
cargo test -p blob_game \
  game::tests::test_reference_replay_stream_checkpoint_resumes_exactly -- --exact
cargo test -p blob_game extism_compat::tests:: -- --nocapture
cargo test -p blob_game \
  game::tests::test_extism_compat_executor_matches_extism_and_keeps_pristine_guests -- --exact
cargo test -p blob_game \
  game::tests::test_all_maintained_minds_match_the_extism_compat_executor -- --exact
cargo test -p blob_game \
  game::tests::test_language_pdk_minds_match_both_executors -- --exact --ignored
cargo test -p blob_game \
  game::tests::test_extism_compat_is_worker_invariant_and_rejects_extra_import_surfaces -- --exact
