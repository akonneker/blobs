# Maintained fuzz harnesses

The fuzz crate is intentionally outside the release workspace. It exercises
untrusted byte boundaries with tight allocation limits:

- `replay_formats` covers replay frames, archives, checkpoints, bundles,
  segments, manifests, match manifests, and attestations.
- `mind_abi` covers canonical Cap'n Proto Mind inputs and decisions and asserts
  semantic round trips for every accepted value.
- `resolver_commands` generates bounded worlds and action/clock/checkpoint
  sequences. It checks conservation, occupancy, canonical ordering, full versus
  incremental hashes, forward/backward deltas, reversed commitment order, and
  serial versus parallel resolution and passive updates. One branch restores
  checkpoints while the other retains its derived indexes.
  Header bit 5 additionally seeds sparse historical keys with a next-cell key of
  1,024; later births cross pages and mutations exercise shared Merkle branches.
  Header bit 6 chooses mixed per-actor actions for both permuted commits and
  atomic ordered batches. Ordered admission is compared with individual commits
  on a disposable clone; fatal memory/signal/actor errors must leave the real
  branches untouched, while semantic rejections commit their private memory.
- `wasm_admission` checks arbitrary bytes, repaired module headers, and generated
  valid core modules with known profile-admission outcomes. It covers every
  allowed PDK import signature, forbidden capabilities/import kinds, reference
  and initializer signatures, start functions, memory/table counts and memory
  modes. Appending an inert custom section must preserve accepted inspection.
  This target never compiles or executes guest code.
- `neighborhood` covers topology admission and masks, every compiled target,
  coordinate round trips and extreme offsets against a separate i128 geometry
  oracle. Ordinary worlds are at most 8x8 with up to 33 slots (32 allowed); huge
  virtual-world cases have zero slots and allocate no per-tile neighbor entries.
- `observations` checks actual masked Mind fields, action masks, ABI round trips,
  direct/batch projection, polluted scratch reuse and invariance under host-key
  reassignment or other cells' private-memory changes. Worlds have at most 25
  tiles, eight slots and 16 private bytes per cell. All ten action families,
  rejected commitments and exact early/middle/late progress boundaries are
  checked without advancing past the first event.
- `movement` compares one resolver Move with an independent geometry, occupancy,
  elevation and diagonal-corner oracle, including forbidden/invalid slots and
  wrapped self targets. It also checks conservation, hashes and delta round trips.
  Inputs have at most 64 bytes; worlds have at most 25 tiles and eight slots.

Install the pinned tool and run a target locally:

```sh
rustup toolchain install nightly-2026-08-01 --profile minimal
cargo +nightly-2026-08-01 install cargo-fuzz --version 0.13.2 --locked
cargo +nightly-2026-08-01 fuzz run replay_formats
python3 fuzz/generate_resolver_seeds.py
RAYON_NUM_THREADS=2 cargo +nightly-2026-08-01 fuzz run resolver_commands -- -max_total_time=300 -max_len=516 -rss_limit_mb=1024 -timeout=10
cargo run --manifest-path fuzz/Cargo.toml --example wasm_admission_seeds
cargo +nightly-2026-08-01 fuzz run wasm_admission -- -max_total_time=300 -max_len=8192 -rss_limit_mb=1024 -timeout=10
python3 fuzz/generate_neighborhood_seeds.py
cargo +nightly-2026-08-01 fuzz run neighborhood -- -max_total_time=300 -max_len=256 -rss_limit_mb=1024 -timeout=10
python3 fuzz/generate_observation_seeds.py
RAYON_NUM_THREADS=2 cargo +nightly-2026-08-01 fuzz run observations -- -max_total_time=300 -max_len=256 -rss_limit_mb=1024 -timeout=10
python3 fuzz/generate_movement_seeds.py
RAYON_NUM_THREADS=2 cargo +nightly-2026-08-01 fuzz run movement -- -max_total_time=300 -max_len=64 -rss_limit_mb=1024 -timeout=10
```

Run the shared resolver harness against 128 deterministic 64-command sequences
with `RAYON_NUM_THREADS=2 cargo test --manifest-path fuzz/Cargo.toml --lib`.
The resolver input has a four-byte world/profile header followed by eight-byte
commands; only the first 64 commands are executed. Commands choose individual
decisions, permuted commits, events, clock
advances, checkpoint restores or atomic ordered batches. Reversed/duplicate
actors, unknown/busy actors, sidecar signals and retained/replaced/oversized
memory exercise atomic rejection. Worlds have at most 25 tiles
and decisions have at most 17 bytes of private memory (16 allowed; 17 exercises
rejection). Generated seeds cover every action family, both boundary modes,
equal and variable action durations, and enabled/disabled passive processes.
The same test command checks 12,480 generated Wasm admission cases; a separate
Wasm validator first confirms that negative admission cases are valid modules.
The seed example writes raw modules and selectors for mutation, preserving any
previous fuzz discoveries. Runtime allocation ownership, traps, deadlines,
pristine reset and differential executor fuzzing remain separate work.
Neighborhood checks add 3,888 deterministic topology/mask combinations and 128
arbitrary selectors. Raw offsets, zero costs, invalid masks and excessive slot
counts must fail admission; accepted geometry must match the independent oracle.
Resolver checks additionally cover 640 mixed/atomic sequences and five fatal
batch fixtures with valid continuation, alongside the original arbitrary inputs.
Observation tests cover 128 permission combinations across four world profiles,
plus 256 arbitrary inputs. Movement tests exhaust 512 occupancy patterns under
all three corner policies and both boundaries, plus 512 arbitrary inputs.
An additional 110 observation fixtures assert every activity, all ten accepted
actions at six progress boundaries, idle guards and forbidden/invalid targets.
The generator supplies these fixtures alongside the 512 visibility seeds.

CI runs short bounded smoke jobs. Longer scheduled campaigns should retain
their corpus and crash directories as CI artifacts. A minimized finding must
become a deterministic regression test before it is considered resolved.
