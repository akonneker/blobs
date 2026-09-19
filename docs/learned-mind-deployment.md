# Learned Mind deployment

The [composite utility follow-up](feeding-composite-deployment-2026-09-12.md)
adds an explicitly versioned sampled Move utility to this frozen parent. It
preserves the old greedy format and qualifies the first matched control/treatment
WASM pair; feeding ecology remains the next gate.

The [sampled target decoder](sampled-target-deployment-2026-09-12.md) now has a
separate portable contract and an ordinary-ABI canary. The frozen learned policy
described here still uses its explicit greedy execution contract; no sampled
learned checkpoint has yet been exported.

A verified behavior-cloning checkpoint can now be frozen into an ordinary
Extism PDK WASM Mind. The artifact runs through the existing game host and server
re-execution path, using the same anonymous local observation and explicit
cell-private memory as maintained Minds. No training process or shared policy
state is needed at runtime.

This is a deployment milestone. The exported parent still has the feeding and
contention limitations described in the [handoff](project-handoff.md); exporting
it does not promote it or establish robust gameplay competence.

## Build and verify

Run from the workspace root. These commands reproduce the retained artifact;
choose new output directory names when rerunning them.
Rust's `wasm32-unknown-unknown` target and Cap'n Proto compiler must be installed.
The following source is the frozen schema-34 effort-control parent used in the
September 11 experiments; its current migration initializes the residual to zero.

```sh
cargo run --release --locked -p blob_rl --no-default-features --features ndarray \
  --bin export-policy -- \
  --artifact training-output/feeding-dagger-round5-v1/effort-control-dose-lr005-steps10-v1 \
  --artifact-sha256 fc66244380552e63d23908973eb50f3c34ef13212498575e27a52064504633a7 \
  --output training-output/learned-mind-2026-09-12/frozen-final

python3 scripts/build_learned_mind.py \
  training-output/learned-mind-2026-09-12/frozen-final

RAYON_NUM_THREADS=4 cargo run --locked -p blob_game --example verify_learned_mind -- \
  training-output/learned-mind-2026-09-12/frozen-final \
  training-output/learned-mind-2026-09-12/verification-final
```

The build directory contains `weights.bin`, `export.json`, `mind.wasm` and
`mind-build.json`. The two manifests bind the source metadata/model, inherited
seed information, export execution identity, actual weight bytes, compiler,
deployment source files and compiled WASM. The builder snapshots weight bytes
privately and checks source stability during compilation. Neither exporter nor
builder overwrites an existing completed output.

Load `mind.wasm` as an ordinary game argument. For example, this runs a bounded
local match between two copies of the exported policy:

```sh
cargo run --locked -p blob_game -- \
  training-output/learned-mind-2026-09-12/frozen-final/mind.wasm \
  training-output/learned-mind-2026-09-12/frozen-final/mind.wasm \
  --config-path blob_game/config/sample.toml \
  --width 12 --height 12 --seed 72101 --steps 64
```

Add `--gui` to inspect a local match interactively. Native snapshot evaluation
and the exported guest remain distinct execution identities.

## Contracts and numerical qualification

`blob_policy` owns the shared observation projection, legal action masks,
greedy tie-breaking and `BRM1` signed-16-bit private-memory codec. Existing RL
module paths re-export those implementations. Its portable inference uses a
fixed order of scalar f32 multiply/add operations and pinned `libm` functions;
the native deployment oracle and WASM use the same runtime. It includes all
inference adapters and target residuals. Value and learned-gate diagnostic heads
are omitted because they cannot affect the deployed greedy decision.

The weight format has a fixed version and layer order. Dimensions are bounded
to 1–512, shape-derived byte lengths are checked before allocating layers, the
file is capped at 16 MiB, and malformed, trailing and non-finite weights fail
closed. Model-record input to export is capped at 64 MiB and metadata at 1 MiB.
Export first copies input bytes to a private temporary directory and verifies
that snapshot before the recorder opens it; a producer-path replacement cannot
change the verified model subsequently loaded by this exporter.

Burn's optimized reductions and transcendental functions need not produce the
same last floating-point bits as the portable runtime. The deployment execution
contract is explicitly `blob.policy.scalar-f32-libm.v1`. Export requires zero
action/signal mismatches over at least 100 sampled decisions and at most one
quantization unit of private-memory difference. Its fixed development seeds
72101–72104 are exposed diagnostics, never fresh confirmation seeds. Export and
replay qualification reject these seeds if reserved by the source ledger or
source training configuration. Subsequent
feeding qualification must evaluate the exported runtime under its own identity.

The first parent export checked 686 decisions: all actions/signals agreed; 85
memory updates differed, each by at most one quantization unit. A separate test
activates every inference adapter and compares every output head and recurrent
value with Burn in 48 inputs spanning all three routing contexts, using a
2e-5 relative-plus-absolute float tolerance. These checks are bounded evidence,
not a universal claim of identical Burn trajectories.

WASM qualification requires **exact** native-deployment/WASM action, signal and
private-memory agreement. The first parent run passed 2,010 decisions across ten
12×12 fixtures: two seeds in each of line, checkerboard, ring, loose-random and
random layouts, up to 64 resolver batches each. Every batch matches the native
state hash; the server re-executes the actual guest with four workers against
one-worker client replays, then signs and verifies the completed replay manifest.
Thirty-two sampled decisions also match stock Extism, and intervening unrelated
invocations cannot alter their output. Malformed input is explicitly rejected.
Replay segments, manifests, fixture attestations and the exact report are retained
under `training-output/learned-mind-2026-09-12/`.

The signing seed in this example is public fixture data. These checks use the
existing 64 MiB memory limit and two-second wall-clock invocation timeout; they
do not define deterministic fuel exhaustion or constitute online admission to a
production service. The bounded fixtures do not assert a gameplay winner.

## Maintenance and next steps

The ordinary CI workflow generates a clearly labeled synthetic nonzero network,
builds its WASM, executes the same parity/replay qualification and retains the
artifacts. It does not require private training-output files or silently skip
missing guests. Workspace tests separately compare exported weights with Burn.
Changes to inference semantics, weights format or numerical policy must update
the corresponding registered version/identity and rerun these gates.

Next: improve target-choice behavior on fresh development seeds, qualify feeding
and combat retention using the portable policy, and define deterministic guest
execution budgets. Long-horizon gameplay, more seeds, cross-host parity and direct
PPO-artifact export remain separate work; the current export command accepts the
shared loader's supported behavior-cloning schemas.

## September 12 validation

The final trained bundle is
`training-output/learned-mind-2026-09-12/frozen-final/mind.wasm` (1,499,218 bytes),
SHA-256 `632d8526805fe0a3b251e3837b4857be235f3064bf7102e83043b697b47d4b2c`.
Its final verifier again passes all 2,010 decisions and ten fixtures. The synthetic
CI fixture passes 508 decisions across the same ten fixture definitions. The
ordinary game CLI also loads two copies through stock Extism and completes 64
steps on a 12×12 board with 12 starting cells per team. This is a local execution
smoke, not a feeding/combat qualification.

The full workspace suite passed 607 tests (33 ignored). After adding the native
confirmation-reservation helper, all 16 portable tests, the Burn/export fidelity
test, workspace all-target/all-feature Clippy and portable/guest WASM Clippy pass.
All 63 schema constants/mirrors agree. Explicit negative integration checks reject
wrong source hashes, altered WASM and confirmation-seed collisions in both export
and replay qualification without creating a result directory. Rebuilding an
existing artifact is rejected without changing it.

Commands, logs and reports are under `training-output/learned-mind-2026-09-12/`.
CI wiring was exercised locally using its synthetic fixture; no remote CI run or
commit is claimed. Existing unrelated edits remain in the working tree.
