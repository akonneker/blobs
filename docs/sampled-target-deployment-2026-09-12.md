# Portable sampled target decoder — 2026-09-12

The sampled target decoder now has a separate portable contract and an ordinary
Mind-input adapter. Its untrained ABI canary produces **9,223 exactly matching
native/WASM decisions and quantized weight vectors**, including all candidate
counts through 32 and exact cumulative-probability boundaries. The selected
learned-utility probe still passes **all nine gates** with this shared decoder;
its validation results and loss checkpoints exactly match the preceding fit.

This closes the decoder arithmetic and ABI gate from the
[relational comparison](target-relational-2026-09-12.md). It does not transfer the
learned utility network to production observations, export new learned weights,
qualify feeding ecology, or change the existing frozen greedy Mind.

## Contract

`blob_policy::sampling::TARGET_SAMPLING_CONTRACT` is
`blob.policy.target-categorical-f64-libm-q48-dy-dx-word0.v1`.
It is separate from `blob.policy.scalar-f32-libm.v1`, the deployed greedy policy's
execution identity. The existing frozen weight format and greedy decoder remain
explicit; the new sampler is not silently enabled for old checkpoints.

For at most 32 logits, legal finite f32 values widen to f64 before subtraction of
the legal maximum. Portable `libm::exp` computes relative weights and
`libm::round` rounds them to integer multiples of 2^-48, with half ties away from
zero. Illegal candidates have zero weight, including those with nonfinite logits.
A nonfinite legal logit or mismatched/oversized shape is an error. Empty support
is valid and returns `None`, leaving fallback behavior to the calling policy.
Very small relative probabilities can round to zero; no epsilon is injected.

Every weight is at most 2^48 and the total at most 2^53. Selection computes
`floor(private_word * total / 2^64)` with u128 arithmetic, then selects the first
cumulative weight strictly above that point. There is no floating random
normalization or stochastic host call. Equal logits reproduce the exact u64
uniform-quantile boundaries. Endpoint, weighted-boundary, extreme-logit,
zero-support and malformed-input tests cover these rules.

`TargetDistribution::from_input` accepts logits indexed by ABI slot ID and an
explicit targeted action kind. It uses the shared input validator, exposed as
`validate_reference_mind_input`, before mask projection or indexing. The adapter
sorts slots by **(dy, dx)**, maps sampled positions back to ABI slot IDs, and rejects
duplicate geometry and the current tile `(0, 0)`. Geometry ordering matches the
synthetic Moore fixture and remains independent of ruleset slot numbering. It
need not reproduce an arbitrary ruleset's original slot-order quantile teacher.

Reachability, enabled target bits, supported efforts, payload and affordability
come from the existing policy action mask. The decoder supplies no food ranking,
occupancy preference or hidden identity; those belong in learned utilities.
`sample_target` consumes only private bytes 0–7 as a little-endian u64. Other
random words and valid private-memory contents cannot affect that draw. All four
targeted action kinds have separate mask checks in the tests.

## Ordinary ABI and WASM evidence

`blob_policy/examples/sampled_target_canary.rs` exports the ordinary
`reference_mind_function`. Its shared fixture uses an explicitly untrained
visible-food score and chooses a legal Move effort after sampling, or Wait when
there is no target. Diagnostic private memory contains the complete canonical
integer weight vector and slot mapping. This makes numerical differences
observable through the ordinary decision ABI, not just through matching choices.
The fixture is not a learned-policy deployment or ecological baseline.

`blob_game/examples/verify_sampled_target.rs` uses the engine's isolated Extism
compatibility executor, structural Mind admission, no host/network access, a
128-page memory limit and per-call timeout. It checks:

- 0–32 slots with equal weights, varying weights, disabled/unreachable targets,
  unaffordable moves and renumbered slots;
- every cumulative boundary and its adjacent u64 words, plus endpoints;
- all 255 nonempty eight-slot target masks;
- native/WASM action, signal and diagnostic memory equality on 9,223 invocations;
- 165 original/unrelated/original isolation sequences;
- rejection of malformed wire bytes, duplicate geometry and self geometry.

Every emitted action also passes the native policy commit-legality check. These
are ABI fixtures, not engine-generated trajectories or signed match replays.
There is no deterministic instruction-fuel or peak-RSS qualification here.
No engine seeds or reserved confirmation seeds were consumed.

Final evidence is `training-output/sampled-target-2026-09-12-v2/`: the pre-check
plan, complete report, supplied WASM copy and hashes of embedded native-verifier
source snapshots. The verifier hashes its compiled sources rather than reading
possibly changed source files at runtime. The supplied WASM has its own digest;
source snapshots describe the native verifier build. Revision 1 remains retained
as the earlier passing run before source-snapshot retention was added.

## Learned candidate retention

The research helper `blob_rl/examples/target_set_probe/sampling.rs` now delegates
to the portable `CategoricalDistribution`; it no longer has its own platform-exp
implementation. `--utility-only` rechecks the selected candidate across the same
three tasks and three initializations without retraining discarded controls.

`training-output/target-portable-2026-09-12-v1/` retains the nine completed fits,
exact executable, command, source copies, manifest/lockfile, plans and results.
All nine gates pass. Validation, loss checkpoints and applicable summary
diagnostics exactly match the nine corresponding arms in
`target-relational-2026-09-12-v5`. This is a decoder substitution check on exposed
development seeds, not fresh confirmation or learned 32-slot generalization.

## Reproduction

```
cargo test --locked -p blob_policy --test target_sampling
cargo build --release --locked -p blob_policy --example sampled_target_canary --target wasm32-unknown-unknown
cargo run --locked -p blob_game --example verify_sampled_target -- target/wasm32-unknown-unknown/release/examples/sampled_target_canary.wasm training-output/sampled-target-new-verification
cargo build --release --locked -p blob_rl --no-default-features --features ndarray --example target_set_probe
RAYON_NUM_THREADS=4 target/release/examples/target_set_probe --utility-only --output training-output/target-portable-new-run
python3 scripts/verify_target_probe.py training-output/target-portable-2026-09-12-v1 --unchanged-from training-output/target-relational-2026-09-12-v5
```

CI builds and verifies the sampling canary alongside the existing greedy learned
Mind parity gate. The native CI job also runs the focused target-probe tests.
Local validation passes 615 workspace tests (33 ignored), all 13 focused probe
tests, and workspace Clippy for all targets/features with warnings denied. The
schema gate still verifies 63 constants. Logs are retained with the final canary
report.

## Next gate

The [real-observation utility follow-up](feeding-utility-transfer-2026-09-12.md)
passes the conditional transfer and retention gate for all three initializations.
The [composite deployment](feeding-composite-deployment-2026-09-12.md) now binds
all six utilities to the frozen parent with an explicit sampled identity. Its
first matched control/treatment WASM pair passes native/WASM, inherited-output
and replay checks. Next evaluate frozen parent, learned control and treatment
in the actual five-layout feeding matrix. Visibility and full-capacity learned
competence remain open; reserved confirmation seeds remain untouched.
