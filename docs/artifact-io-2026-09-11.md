# Bounded RL artifact reads — 2026-09-11

Several artifact loaders checked path metadata and then called `fs::read`, which
opened the path again and read it without a byte cap. Growth or path replacement
between those operations could bypass the advertised limit before decoding or
hash verification. Fixed-size JSON/MessagePack limits therefore did not reliably
bound the bytes read.

`blob_rl/src/artifact_io.rs` now owns the common read primitive. It opens once,
checks metadata from that handle, and uses `Read::take(limit + 1)` on the same
handle. The extra byte detects growth past the limit; oversized input is rejected
before a decoder sees it. Exact-limit and empty inputs remain valid. A limit of
`u64::MAX` is rejected rather than overflowing the probe length.

The helper serves 23 RL modules: checkpoint/resume and cloning metadata,
demonstration manifests/payloads, sweep controls, feeding/checkpoint/combat
evaluations, counterfactual evidence, competency frontiers, calibration,
diagnosis, viability and scale qualification/transition/promotion. Existing byte
limits, artifact formats, hashes, schemas and semantic validation remain intact.
Cloning metadata already used a bounded stream; it now shares the same primitive.

Three deterministic regressions cover exact limits and one extra byte, an endless
reader that must supply only `limit + 1` bytes, a real file grown after inspection,
and path replacement after opening. The replacement test proves the opened
handle still reads the original inode. The initial RL library run passes 294
tests (four ignored). Final workspace validation passes 604 tests (33 ignored),
all-target/all-feature Clippy, formatting and 59 schema constants/mirrors.
Evidence is retained under
`training-output/artifact-io-2026-09-11` with source hashes and exact commands.

This fixes the read boundary. It is not a total decoded-heap budget, a deadline
for blocking filesystem I/O, or an atomic snapshot of an entire multi-file
artifact. JSON/MessagePack structural-allocation fuzzing and model-record
verification/loading races remain separate review work; immutable artifacts
should retain their existing publication and verification discipline.
