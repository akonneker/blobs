# Private target residual — 2026-09-11

Follow-up: the [CPU numerical investigation](backend-numerics-2026-09-12.md)
found biased batched gradients in the ARM backend used by earlier fits. The
measured residual behavior remains historical evidence, but failed fits do not
establish an architecture limit. Future training uses the corrected backend;
see the [latest target comparison](target-relational-2026-09-12.md).

The next slice from the [project handoff](project-handoff.md) adds a shared
32-unit ReLU scorer over each cell's 32 private-random features paired with one
raw 33-feature local slot. Its ten output channels contribute to the existing
action-kind-conditioned target logits. No slot index, cell identity, global state,
or batch aggregation enters the scorer. Each slot passes through the same weights.

The output layer starts at exact zero; the hidden layer uses ordinary random
initialization so gradients can learn the residual. The extra 2,442 parameters
bring the default model to 107,222 and the large profile to 346,582 parameters.
Global action-kind, effort, amount, signal, value, and recurrent heads receive no
residual output. The code is isolated in `blob_rl/src/model/target_residual.rs`.

`behavior-clone --target-residual-only` updates only this residual and restores
every inherited parameter from a frozen parent after each optimizer step. It
requires a verified parent and inherited/audited seed ledger, and is mutually
exclusive with the existing restricted adaptation modes. The CLI seeds the
backend before loading/migrating a parent so matched arms initialize new adapter
features identically, then training resets the seed as before.

Cloning schema 37 stores the new layout and adaptation mode. Schemas 34–36 load
through `PreTargetResidualPolicyValueNet` and acquire a zero-output residual.
Earlier supported schemas preserve their existing migrations and also receive a
zero residual. Schema 30 remains unsupported. Training schema 46 prevents old
optimizer/model records from being silently read as the new layout. Execution
identity, seed-ledger semantics, Mind ABI, and observation/action/memory encodings
are unchanged.

## Mechanistic gates

- Compare every output bit and inherited parameter bit between the pre-residual
  operation path and a migrated model, using varied observations and nonzero memory.
- Exercise the existing full-model slot-permutation gate with an active residual.
- Change one batch row's private randomness and verify the other row bit-for-bit.
- Verify active residual randomness changes relative target scores, and modifying
  one slot leaves every other slot's score unchanged.
- Perform eight adapter-only optimizer updates; check all inherited parameter bits
  after every step, improved target preference, and unchanged global/recurrent output.
- Exercise all admitted artifact layouts through the shared loader.

All 592 workspace tests passed (31 ignored), including the active slot-permutation
and all-schema import gates. Workspace all-target/all-feature Clippy, formatting,
whitespace checks, and all 59 schema constants/mirrors passed. This implementation
is not evidence of improved held-out target agreement or ecological survival.

## Experiment prerequisites

The selected parent is
`fc66244380552e63d23908973eb50f3c34ef13212498575e27a52064504633a7`,
locally at `training-output/feeding-dagger-round5-v1/effort-control-dose-lr005-steps10-v1`.
A metadata inventory found its complete 16-stage chain and all referenced corpora,
covering 52 exposed seeds. The five saved control/treatment pairs use seeds
1433000101 and 1433000202, disjoint from that lineage. The production loader must
still validate payload hashes and perform the explicit historical audit before
training. These seeds have already informed earlier architecture experiments;
they are development validation, not untouched final confirmation seeds.

The first lineage audit exposed two existing dataset-loader compatibility defects:
only the current schema was admitted for effort corrections, and schema 17 was
missing from the file loader's admission list. Both are fixed through a shared
schema predicate and schema-16-onward effort admission; private-memory and hash
checks remain enforced. The regression publishes/loads 16, 17, and 18 fixtures
and rejects schema 15, malformed memory, and tampered payloads. After this change,
291 RL library tests plus all-layout migration and seed-lineage integration tests
passed, as did RL all-target/all-feature Clippy. The failed pre-training attempt
is retained separately from the v2 run.

## First bounded probe

All five saved pairs passed verification, with 1,453 active treatment label changes.
The v2 run completed 22 optimizer updates per arm using the same seed, parent,
batch/unroll schedule, and frozen inherited parameters. The lineage audit verified
all ancestors and payloads; it recorded 53 training seeds, validation seed
1433000202, and untouched confirmation seeds 1434999901/1434999902.

| Held-out Move exact agreement (1,959 samples) | Initial | After 22 updates |
| --- | ---: | ---: |
| Control against its unchanged policy labels | 100% | 100% |
| Treatment against corrected target labels | 63.2466% | 63.2466% |

Treatment validation loss fell from 0.9414745 to 0.9414268. The dose fails the
mechanism gate; no ecological run or promotion follows from this result. The
control and treatment columns use their respective labels, so they are not a
direct comparison on a shared teacher-label test set. Data and exact commands:
`training-output/target-residual-2026-09-11-v2/comparison.json` and adjacent files.

## Bounded dose follow-up

Using the same immutable executables, original parent, optimizer seed, sampling
schedule, and verified paired datasets, both 88- and 256-update arms completed:

| Updates | Control Move exact | Treatment Move exact | Treatment validation loss |
| --- | ---: | ---: | ---: |
| 22 | 100% | 63.2466% | 0.9414268 |
| 88 | 100% | 63.2466% | 0.9409931 |
| 256 | 100% | 63.1955% | 0.9314124 |

No dose passes the mechanism gate. Stop this dose sweep; do not run an ecological
matrix. The residual is implemented and mechanically isolated, but this training
recipe does not improve the held-out decision boundary. Directional feature scale
(dx/dy are divided by 127) is a diagnostic hypothesis, not an established cause.
Any changed input interpretation must preserve existing schema-37 semantics or
introduce an explicit new version. Exact evidence and artifact hashes are retained
in `training-output/target-residual-2026-09-11-doses/comparison.json`.

The subsequent [margin diagnostic](target-margin-diagnostic-2026-09-11.md)
reproduces the validation count and shows positive but insufficient shifts on
every active correction. Offline gains up to 64× lose as many or more retained
choices than they fix; private-random rotation changes relative residual scores
only weakly on these observations. The next proposal needs a better conditioned
random/slot interaction gate, not simply more dose or gain on this fitted adapter.
