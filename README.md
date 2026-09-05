# Blob Game

Blob Game is a deterministic, mass-energy-conserving programming game and
multi-agent reinforcement-learning testbed. Teams are colonies of cells in a
2D world. Every ready cell receives its own bounded local observation and asks
a WebAssembly **Mind** for one action; the resolver then commits simultaneous
interactions in a canonical order.

The project is built around one operational principle: cells controlled by the
same Mind do not gain hidden shared state. A cell has no public identity, team
identifier, absolute coordinates, global clock, shared random seed, or host
communication channel. Coordination must emerge through inherited private
memory, visible behavior, environmental changes, and four anonymous conserved
signal fields.

The simulation, replay/verification kernel, RL pipeline, local observatory, and
remote training images are implemented and extensively tested. Learned policy
quality and the default physics profile are still research work: current
policies acquire basic competencies but do not yet retain robust long-horizon
combat behavior. The hosted submission service and public leaderboard are not
production deployments yet.

## What is implemented

- A fixed-point event-time resolver with immutable completion snapshots,
  deterministic conflict adjudication, reversible deltas, and worker-count
  invariant outcomes.
- Conserved mass-energy across cells, digestion, plants, loose environmental
  energy, attacks, regurgitation, terrain material, and signals.
- Ten primary actions: Wait, Move, Attack, Guard, Consume, Split, Regurgitate,
  Signal, Excavate, and DepositTerrain. Ordinary actions may also emit one
  variable-strength signal; the explicit Signal action writes all four
  channels independently.
- Configurable local observation and action neighborhoods without exposing
  global state.
- Mind ABI v8, encoded with Cap'n Proto, with explicit cell-private memory and
  engine-derived private random bytes.
- Fresh-instance WASM isolation through stock Extism and a restricted
  Extism-compatible Wasmtime executor. Maintained AssemblyScript and TinyGo
  artifacts exercise the cross-language PDK boundary.
- Canonical checkpoints, replay archives and segments, incremental hashes,
  server re-execution of submitted artifacts, Ed25519 attestations, and signed
  score publications.
- Recurrent PPO and behavior cloning, deterministic resume, curricula,
  competency/promotion gates, rated self-play, specialist distillation,
  viability sweeps, ecological characterization, and large-world scale gates.
- CPU and Vulkan/WGPU training backends, reproducible Docker runners, and a
  local browser observatory for telemetry and complete match histories.

## Workspace map

| Path | Responsibility |
|---|---|
| `blob_engine` | Canonical simulation, resolution, hashing, replay, verification, and online service primitives |
| `blob_interface` | Mind ABI v8 types, Cap'n Proto schema, bounded converters, and structural WASM admission |
| `blob_game` | CLI/GUI host, Mind execution pools, checkpoint envelopes, and projected renderer state |
| `blob_rl` | Native environment, policy models, training, evaluation, calibration, sweeps, and telemetry |
| `blob_web` | Browser replay WASM package and dependency-free local observatory components |
| `minds/` | Behavioral controls, the stateful colony prototype, and runtime/isolation canaries |
| `sweeps/` | Versioned experiment manifests, compact results, and interpretation notes |
| `docs/` | Design contracts, performance history, RL methodology, deployment, and test roadmaps |

`blob_engine` is authoritative. `blob_game` and `blob_rl` drive that same
resolver rather than maintaining alternate physics implementations.

## Build and test

Install a current Rust toolchain plus the `wasm32-unknown-unknown` target. The
complete local gate also expects Cap'n Proto, Node.js, TinyGo, and the
AssemblyScript toolchain described by the language-Mind build script.

```sh
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

Run the complete cross-language and browser-source conformance gate with:

```sh
scripts/conformance.sh
```

Run the real Chromium match-explorer suite with:

```sh
npm ci --prefix browser-tests
npm test --prefix browser-tests
```

Pull requests also run bounded native, WASM, schema, Chromium, and fuzz-smoke
jobs. See [`schemas/README.md`](schemas/README.md) before changing an artifact
version and [`fuzz/README.md`](fuzz/README.md) for the pinned local fuzz commands.

Build two example Minds and launch a local match:

```sh
cargo build --release --target wasm32-unknown-unknown \
  -p simple_mind -p aggressive_mind

cargo run --release -p blob_game -- --gui \
  target/wasm32-unknown-unknown/release/simple_mind.wasm \
  target/wasm32-unknown-unknown/release/aggressive_mind.wasm
```

Omit `--gui` for a headless match. Use `cargo run -p blob_game -- --help` for
world, seed, configuration, step, and checkpoint options.

## Writing a Mind

A Mind is an Extism-PDK module exporting `reference_mind_function`. The host
calls it independently for each ready cell with canonical Mind ABI v8 bytes.
The result contains one primary action, an optional anonymous signal sidecar,
and an explicit private-memory operation.

Mind authors may use any language whose PDK can satisfy the deterministic
profile. The submitted module must avoid WASI, HTTP, configuration, variables,
logging, custom host functions, and any other ambient capability. Structural
admission rejects modules outside `extism_pdk_deterministic_v1`; the stock
Extism host is the compatibility oracle.

The Rust examples in `minds/` are the easiest starting points:

- `simple_mind`, `aggressive_mind`, `defensive_mind`, and `explorer_mind` are
  deliberately limited experimental controls.
- `colony_mind` is a heterogeneous heuristic prototype with feeders,
  explorers, builders, defenders, attackers, lineage-local maps, and local
  signaling. It does not currently embed learned RL weights.
- `isolation_canary` and `invalid_mind_canary` test the host boundary and are
  not playable strategies.

See [the Mind audit and colony roadmap](docs/mind-audit-and-colony-roadmap.md)
for the behavioral limits of each maintained artifact.

## RL, calibration, and large worlds

The default RL feature is WGPU. For a portable CPU run, select the ndarray
backend explicitly:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin train -- --config blob_rl/config/default.toml --seed 42
```

The repository includes commands for non-learning viability checks, physics
calibration, ecological micro-characterization, behavior-cloning datasets,
checkpoint promotion, contact/feeding/combat evaluation, scale qualification,
and bounded sweep execution. The methodology and artifact contracts are in
[RL training and rules tuning](docs/rl-training-and-rules-tuning.md).

Large-world profiles cover 256x256, 512x512, and 1024x1024 simulations. Start
with calibration and scale qualification before committing a long training
run; a configuration that merely fits in memory is not evidence of a useful
learning horizon.

For remote jobs, `Dockerfile.training` provides separate portable CPU and
NVIDIA Vulkan/WGPU images. See [Dockerized remote training](docs/docker-training.md)
for immutable image identity, resource limits, checkpoints, and GPU preflight.

## Local match explorer

Start the dependency-free observatory:

```sh
scripts/serve_telemetry_viewer.sh
```

Then open `http://localhost:4173/?view=match`. The match explorer can pan,
zoom, follow individual cells, step or seek through a complete sparse history,
inspect resolution outcomes, switch field layers, and plot team population,
energy compartments, and selected-cell history. Other local views cover Mind
controls, telemetry, and adjudication studies.

The UI components use Shadow DOM and are intended to be embedded later in
`tinker.ninja`. Browser-loaded JSON is deliberately labeled local and
unverified. The browser replay package can verify a server attestation, but it
cannot turn a locally generated score into an authoritative result.

See [the browser package README](blob_web/README.md) for replay generation,
the presentation sidecar boundary, and embedding details.

## Trust model and online roadmap

The intended online flow is:

1. A server admits and stores the exact submitted WASM artifact.
2. The server derives private randomness and runs the match under a frozen
   ruleset, ABI, and runtime capability profile.
3. Canonical replay hashes and the terminal result are signed.
4. The browser verifies the replay and attestation with a trusted public key.

This kernel exists. A production service still needs authenticated uploads,
sandboxed compilation where source submissions are accepted, durable queues
and workers, quotas, object storage and retention, season configuration,
signing-key operations, leaderboard indexing/APIs, and site integration.

## Design and performance references

- [Current project handoff and next slice](docs/project-handoff.md)
- [Simulation and resolution design](docs/simulation-resolution-design.md)
- [Large-game optimization roadmap](docs/large-game-optimization-roadmap.md)
- [Measured performance baseline](docs/performance-baseline.md)
- [Testing and fuzzing roadmap](docs/testing-and-fuzzing-roadmap.md)
- [Physics, interface, and training diagnosis](docs/physics-vs-training-diagnostics.md)
- [Format and schema registry](schemas/README.md)
- [Maintained fuzz harnesses](fuzz/README.md)

Recorded optimized native profiles range from hundreds of thousands to low
millions of cell actions per second depending on workload, population, host,
integrity mode, and Mind execution path. Treat the checked-in measurements as
comparative evidence, not a universal throughput promise.
