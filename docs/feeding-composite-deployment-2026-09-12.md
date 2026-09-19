# Composite learned Mind deployment — 2026-09-12

The [passing feeding utilities](feeding-utility-transfer-2026-09-12.md) now have
an explicit deployment format and run with the frozen parent through the ordinary
Mind ABI. All six exported fits preserve **34,794 recorded sampled choices**
across the old and fresh development cohorts. The first initialization's matched
control and treatment WASM Minds pass **4,896 exact native/WASM decisions** and
**twenty signed replay fixtures**. Every checked parent output beyond Move target
is preserved exactly.

This closes the composite packaging and bounded runtime-fidelity gate. It does
not demonstrate better feeding survival, less contention, combat competence or
long-horizon robustness. The next gate is a matched feeding evaluation of the
frozen parent, learned control and treatment.

## Execution contract and artifacts

`blob_policy::composite` defines the execution identity
`blob.policy.move-utility-scalar-f32-libm-q48-signal-reserved.v1`.
The `BLCMP001` envelope contains a bounded parent byte length, the complete
unchanged `BLPOL001` parent weights and 5,105 utility f32 parameters. Utility
linear weights retain input-major order, followed by biases for encoder,
encoder2, fc and head. Loading checks exact lengths, finite values and derived
layer shapes. Old `BLPOL001` weights retain their original greedy semantics.

The utility reproduces the research architecture with explicit scalar f32
multiply/add order, portable `libm::tanhf`, sorted additive summaries and the
existing Q48 categorical decoder. Scores use only the 33 raw slot features and
the legal-target mask. Geometry is ordered by (dy, dx); the draw uses private
random word zero. No host identity, shared state or new observation fields are
introduced.

The combined Mind first computes the entire parent decision. For a Move, it
samples a destination legal at the **same effort while reserving the parent's
signal cost**, then changes only `target_slot`. This matters when a farther
destination is affordable alone but not together with the original signal.
Other actions, amounts, signal contents and memory updates pass through exactly.
The affordability regression explicitly exercises this boundary.

The final artifacts are under
`training-output/feeding-composite-2026-09-12-v2/exports/`. All three
initializations have control and treatment envelopes; initialization 1435400301
was chosen by its existing order for initial WASM qualification, without selecting
the best development score. Each envelope is 1,285,420 bytes.

| First-initialization artifact | SHA-256 |
| --- | --- |
| Treatment weights | `f05ec64249f7cfdf559d0c9d26bf4053c75fa968b0b5c54609f75702a6d2f89c` |
| Control weights | `dfc410c496271fd34e9a3c9aeb8ecbd2c30f9d068bf60cebb956917a05f77095` |
| Treatment WASM | `ebe8d5118ad5c2713e8c331a7b580347f640a5cff6951dcece632dc158cd652e` |
| Control WASM | `0e4b8336b5e003fbd2554322fce1c2485b12a20f5eafa0c4304855a779cd534a` |

The WASM files are `1435400301-treatment/mind.wasm` and
`1435400301-control/mind.wasm` within that export directory. Their build manifests
bind the exact WASM, envelope, export metadata, compiler, command and source
hashes. The v1 preliminary export remains retained; v2 adds the cumulative
architecture-study/deployment seed exposure and embedded exporter source copies.
All six weight envelopes are unchanged between those revisions.

## Recorded-frontier fidelity

`blob_rl/examples/export_feeding_composite.rs` hash-checks the archived study
report and each full-f32 utility record, loads a private record snapshot, and
re-verifies both demonstration-pair audits against the study. It checks the
parent export identity and retains its weight bytes exactly. Every reloaded
Burn model reproduces the previously archived evaluations before comparison
against portable execution. No parameters are trained during export.

For each of six fits, 1,959 old and 3,840 fresh development Move observations
produce exactly the same sampled targets. Maximum absolute legal-logit
differences range from 0.00000572 to 0.0000191 across cohorts/fits; logits are
not claimed bit-identical between Burn and portable execution. All original
agreement and retention gates remain passed.

This comparison uses the recorded parent's fixed-effort masks. The composite's
additional signal reservation is checked separately through unit and ABI/live
fixtures; the recorded-frontier result does not claim those masks are universally
identical. Changing targets also changes future observations, so per-invocation
output preservation does not imply identical long-term memory trajectories.

## Ordinary-ABI and replay checks

The shared learned-Mind builder and verifier now accept either explicit envelope
and reject an execution-identity/envelope mismatch. The verifier checks full
native/WASM decisions, including memory and signal, before returning the action
to the engine. For composites it also re-evaluates the parent on the same input,
restores only the old target in the resulting decision and requires exact
equality. Chosen actions must remain commit-legal with the inherited signal.

| Check | Treatment | Control |
| --- | ---: | ---: |
| Exact native/WASM decisions | 2,418 | 2,478 |
| ABI fixture decisions (0–32 slots) | 528 | 528 |
| Move decisions | 2,402 | 2,462 |
| Targets changed from parent | 623 | 463 |
| Non-Move decisions retained | 16 | 16 |
| Other inherited-output mismatches | 0 | 0 |
| Signed replay fixtures | 10 | 10 |

Each Mind also passes 32 stock-Extism comparisons, 32 isolation sequences and
malformed-input rejection. ABI fixtures include masked/hidden features,
renumbered slots, low energy and private-word endpoints. The five layouts use
development seeds 72101/72102, 64 event batches and worker counts one/four;
native and WASM state hashes agree, and server replay verification signs the
result. These small deployment fixtures are not the feeding competence matrix.

The learned control changes 463 parent targets across the combined ABI and
engine fixtures. Its near-perfect agreement on the training-distribution
development corpus is therefore not a global identity guarantee. Retain the
original parent as a separate ecological baseline. Sixteen non-Move decisions
per artifact are limited coverage, not a qualification of every action family.
Likewise, 32-slot ABI agreement proves runtime consistency, not learned success
at every neighborhood capacity or visibility setting.

## Provenance and verification

The exported ledger includes the parent's audited 52 exposed seeds, this
utility study's training/development/initialization seeds, the earlier
interaction/set/relational study seeds that informed architecture selection,
and deployment development seeds 72101–72104. All roles are disjoint. Reserved
confirmation seeds **1434999901/1434999902 remain untouched**.

The evidence directory retains exact exporter/verifier executables and hashes,
commands, status, compiled exporter source copies, six manifests and envelopes,
two WASM build manifests and replay attestations. Check consistency with:

```sh
python3 scripts/verify_feeding_composite.py training-output/feeding-composite-2026-09-12-v2
```

That script audits retained hashes, envelope contents, lineage and reported
fidelity/retention arithmetic. It does not rerun inference. CI now builds and
verifies both greedy and composite synthetic fixtures, separate from the trained
artifacts, alongside the categorical decoder canary.

Local verification passes **619 workspace tests (33 ignored)**, all-target and
all-feature workspace Clippy with warnings denied, formatting, the 63-constant
schema registry and diff whitespace checks. Four new runtime tests cover
artifact corruption, parent-output preservation, signal-only affordability
boundaries, slot permutation and masked competitors. Fresh greedy and composite
synthetic artifacts both build and pass the ordinary-ABI/replay verifier.
Logs and command statuses are retained with the final evidence.

## Next gate

The [feeding ecology follow-up](feeding-ecology-2026-09-12.md) now completes the
matrix below for all three initializations: 94.73–94.88% adjacent-food survival,
100% on-food survival and substantially reduced contention. The next gate is
sustained feeding past the inactive opponent's death, plus fresh development
coverage; confirmation remains reserved.


Add the frozen composite runtime to the feeding evaluation path. Freeze a
bounded three-arm plan—original parent, initialization-301 learned control and
treatment—using the same five-layout source configuration, development seeds,
world secrets and budgets. Measure target contention, food acquisition,
survival and retention on the actual resulting trajectories. Keep rules and
all weights fixed; do not spend final confirmation seeds until the candidate
and evaluation plan are frozen. Extend initialization and visibility/capacity
coverage only after this matched mechanistic gate succeeds.
