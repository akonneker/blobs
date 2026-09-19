# Target interaction capacity probe — 2026-09-12

Follow-up: the [CPU numerical investigation](backend-numerics-2026-09-12.md)
found biased batched gradients in the ARM backend used for these fits. Training
conclusions need a corrected rerun. The exact independent-score impossibility
argument remains valid. The [latest comparison](target-relational-2026-09-12.md)
uses the corrected backend.

Scaling local directions makes private-random target selection learnable in the
controlled fixed-pair task. The varying-candidate task also exposes a limitation
of independent slot scoring: the maintained teacher's quantile choice depends on
the candidate set. A teacher-informed set-context positive control passes both
tasks. This completes the controlled interaction gate from the
[margin diagnostic](target-margin-diagnostic-2026-09-11.md), without selecting or
promoting a new production policy.

The [learned set follow-up](target-set-2026-09-12.md) now tests that next gate,
identifies collisions in fitted max summaries, and evaluates an additive summary.
Neither learned representation passes the complete acceptance gate.

## Controlled experiment

`blob_rl/examples/target_interaction_probe.rs` trains small standalone scorers;
it does not load a checkpoint or run a simulation. Every arm has a shared 32-unit
ReLU hidden layer and a zero-initialized scalar output. This matches the structure
of one residual target channel, but omits inherited logits, memory, and other
action heads. Labels call the actual `choose_slot_by_quantile` helper used by
Simple's collision-aware feeding teacher.

There are four equally fed, equally distant cardinal destinations in the standard
Moore geometry order: north, west, east, south. One task always offers the first
two; the other samples all 15 nonempty subsets uniformly. These are synthetic
scorer inputs, not full ABI observations. Masked destinations do not participate
in the choice. Each example has an independently generated 32-byte private block.

The fixed budget was recorded before training: 8,192 training rows, 4,096 disjoint
development-validation rows, 512 AdamW updates, batch size 128, learning rate
0.005. All arms within a task use the same data and batch order. The three seeds
are 1435000101 (training), 1435000202 (development validation), and 1435000303
(initialization). These are now exposed development seeds; neither reserved final
confirmation seed was used. There is one initialization, not a robustness sweep.

The arms are:

- **Raw:** 32 byte/255 features plus the 33 slot features; dx/dy use units of 1/127.
- **Scaled:** identical inputs and parameter count, with dx/dy in local tile units.
- **Bilinear:** scaled inputs plus 64 centered-random-byte × direction products.
- **RankContext:** scaled inputs plus decoded random quantile, geometric rank
  midpoint within the candidate set, and reciprocal candidate count. This is a
  deliberately teacher-informed positive control. It changes multiple features
  together and does not isolate the contribution of each one.

The gate requires at least 95% exact validation targets and at least 90% jointly
correct original/intervened decisions on rows whose teacher target changes.
The intervention flips byte 7's high bit, shifting the little-endian random word's
quantile by one half. Each trained scorer also passes bit-exact score comparisons
under slot and batch-row permutation. Geometric rank stays attached to the physical
destination when tensor slots are permuted.

## Results

| Features | Parameters | Fixed-pair validation | Both correct after random intervention | Varying-subset validation | Both correct after random intervention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw | 2,145 | 50.24% | 0.00% | 55.71% | 0.00% |
| Scaled | 2,145 | 99.15% | 98.32% | 90.21% | 74.64% |
| Bilinear | 4,193 | 99.17% | 98.02% | 88.35% | 70.19% |
| RankContext | 2,241 | 99.39% | 98.90% | 96.36% | 90.38% |

All fixed-pair interventions change the teacher target. In the varying-subset
validation set, 3,056 of 4,096 do; the intervention percentages use those 3,056
rows. The positive control correctly answers both versions on 2,762 of them.

Raw scores never change their greedy choice under this intervention. Scaling alone
clears the fixed-pair gate with the same parameter count; the explicit products
provide no material advantage here. Independent scoring does not clear the
varying-subset gate. Only the set-context positive control passes both gates.

## Exact independent-scoring limit

For a fixed private random block, scores computed independently from each
destination induce a fixed ranking. The quantile teacher can contradict that
ranking: at random word `2^63`, it selects destination 1 from `[0, 1, 2]`, but
destination 2 from `[1, 2]`. Removing destination 0 changes the preference between
two unchanged destinations. A fixed per-destination ranking cannot satisfy both.

The diagnostic enumerates every one of the 24 rankings on four destinations and
every nonempty subset. It partitions all `2^64` random words at the exact integer
quantile boundaries for candidate counts 1–4. The optimal ranking gets
15, 14, 12, 12, 14, and 15 of the 15 subsets right in the six intervals. Weighting
by their exact word counts gives a maximum of
`255179959686315464022 / 276701161105643274240`, or **92.22%**.

This is an exact limit for the stated synthetic distribution and independent
scores with a fixed semantic tie break. It is not a bound on the real correction
corpora or the full policy: the inherited model has contextual pathways, and
adding a residual to those logits can produce context-dependent preferences.
It also does not prove that this limit caused the earlier 63.2% result.

## Next implementation gate

Use scaled geometry in the next candidate representation. Before changing model
schemas or collecting another matched treatment, test a learned, permutation-
equivariant candidate-set summary against the positive control, holding the
random input encoding fixed. Include varying candidate counts and tied-energy
subsets; distinguish learning a general set representation from supplying the
teacher's answer through engineered rank features. The existing independent
residual should not be expected to imitate arbitrary quantile subsets on its own.

A production change still needs exact-zero migration, frozen inherited outputs,
row isolation, portable/WASM parity, and improved target agreement in fresh
matched control/treatment data with inherited seed ledgers. Ecological evaluation
remains gated on that real-data improvement. This experiment does not demonstrate
better collision rates, feeding survival, combat, or deployment fitness.

## Reproduction and evidence

```sh
cargo test -p blob_rl --no-default-features --features ndarray --example target_interaction_probe
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --example target_interaction_probe -- --output /tmp/new-target-interaction-run
```

The output directory must not already exist. The run writes its plan and copied
source before training, progress after each arm, and a complete report only after
all arms finish. It binds the compiled diagnostic source, teacher helper source,
and Cargo lockfile hashes. It exports no deployable weights.

`training-output/target-interaction-2026-09-12-v1` retains the copied executable,
command/compiler identity, source, pre-run plan, all eight results, exact capacity
calculation, and logs. The full eight-arm run took 2.30 seconds after compilation.
The two diagnostic tests and focused Clippy pass. A separate Python enumeration
checks the capacity calculation, and a second execution checks report
reproducibility. Production inference and training semantics are unchanged.
