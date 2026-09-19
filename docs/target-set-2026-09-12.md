# Learned candidate-set probe — 2026-09-12

The fitted results below used a CPU backend later found to have biased batched
gradients on ARM. See the [numerical fix](backend-numerics-2026-09-12.md) and
[corrected relational comparison](target-relational-2026-09-12.md). The retained
fits and their exact collision witnesses remain evidence about those models;
their failed training gates are not clean architecture-limit evidence. The
implementation has since advanced; use the archived sources for this report.

The new learned set encoders use candidate context, but neither clears the full
target-selection gate. Max pooling discards some candidate-set distinctions needed
by the quantile teacher. An additive summary distinguishes every set in the tested
equal-energy slices and improves four- and eight-destination selection, but still
fails seven of nine task/initialization gates. No production architecture or policy
is selected from this result.

This follows the [direction-scaling probe](target-interaction-2026-09-12.md).
Implementation: `blob_rl/examples/target_set_probe.rs`. It trains standalone
synthetic scorers, not a checkpoint, residual against inherited logits, or full
Mind. There is no ecological evaluation or deployable weight export.

## Matched final comparison

The fixed budget is 16,384 training examples, 4,096 development-validation
examples, 1,024 AdamW updates, batch size 128, learning rate 0.005, and norm-1
gradient clipping. Each arm uses the same 5,009-parameter layout and identical
initial parameters for a given seed. Some parameters/input columns are unused
in the controls; equal nominal parameter count does not imply equal effective
capacity.

All arms receive relative dx/dy in tile units, food level divided by three, the
32 normalized private random bytes, and the decoded random-word quantile. A shared
3→16→16 tanh encoder feeds the slot scorer. The scorer has one 64-unit ReLU hidden
layer and a zero-initialized scalar output. There are no host identities or tensor
positions among its features.

- **Independent:** no set summary.
- **LearnedSet:** coordinate-wise max of encoded legal destinations.
- **LearnedSum:** sum of encoded legal destinations, divided by the fixed maximum
  slot count of eight. This preserves candidate-count information. Each feature's
  values are sorted before addition to make floating addition order canonical
  under slot permutation. Illegal slots contribute zero.
- **RankContext:** no learned set summary; the teacher-informed positive control
  receives best-energy membership, rank midpoint among the best candidates, and
  reciprocal best-candidate count. These engineered features remain zero in every
  learned arm.

The three tasks vary nonempty candidate subsets of four cardinal destinations,
eight Moore destinations with equal food, and eight Moore destinations with
independent food levels 1–3. The teacher selects uniformly by private quantile
among the highest-energy legal destinations, using the same helper as Simple's
collision-aware feeding policy. These are synthetic scorer inputs, not complete
ABI observations or a simulation of feeding costs.

Training seed 1435100101 and development-validation seed 1435100202 are distinct.
Initializations are 1435100301, 1435100302, and 1435100303. These five seeds are now
exposed development seeds. Revisions intentionally reuse the data for diagnosis;
the results are not untouched confirmation evidence. Reserved confirmation seeds
1434999901 and 1434999902 remain unused.

The predeclared acceptance gate requires **every** task and initialization to
reach 95% exact validation targets and 90% jointly correct original/intervened
choices on examples whose teacher target changes. The intervention flips the high
bit of byte 7, shifting the little-endian quantile by one half. Bit-exact slot and
batch-row permutation checks also run on every final model.

## Final results

Ranges below span the three initializations on the same development dataset.

| Task | Arm | Exact validation targets | Joint random-intervention accuracy | Gates passed / 3 |
| --- | --- | ---: | ---: | ---: |
| Four, equal food | Independent | 89.67–90.82% | 72.05–74.74% | 0 |
| Four, equal food | Learned max | 89.55–93.14% | 71.31–81.16% | 0 |
| Four, equal food | Learned sum | 95.56–97.92% | 88.99–94.74% | 2 |
| Four, equal food | Rank control | 98.17–98.78% | 95.01–97.07% | 3 |
| Eight, equal food | Independent | 71.85–72.58% | 48.01–49.02% | 0 |
| Eight, equal food | Learned max | 86.52–89.04% | 73.55–77.99% | 0 |
| Eight, equal food | Learned sum | 89.92–92.19% | 80.57–84.23% | 0 |
| Eight, equal food | Rank control | 97.44–98.19% | 94.65–95.76% | 3 |
| Eight, mixed food | Independent | 92.41–92.58% | 73.21–73.48% | 0 |
| Eight, mixed food | Learned max | 91.92–92.43% | 71.56–73.30% | 0 |
| Eight, mixed food | Learned sum | 89.53–92.60% | 64.40–73.74% | 0 |
| Eight, mixed food | Rank control | 98.39–98.54% | 94.48–95.15% | 3 |

The intervention denominators are 3,005, 3,962, and 2,247 for the three tasks.
Singleton best-candidate cases cannot change teacher target and are excluded from
that denominator. Every report also breaks accuracy down by tied-best candidate
count. All 36 models pass bit-exact score permutation checks.

Removing the learned summary after training lowers eight-equal accuracy to
32.01–55.42% for the max models and 58.18–60.89% for the additive models. This
establishes that those fitted models use context. In mixed-food cases, ablating
the additive summary has little benefit or harm depending on initialization;
there is no evidence of a reliable mixed-food improvement.

## Information lost by fitted max summaries

The diagnostic exhaustively enumerates all 255 nonempty equal-energy subsets of
eight destinations. The three fitted eight-equal max encoders produce only
159, 176, and 199 distinct summary bit patterns. Among sets sharing a summary,
156, 130, and 69 pairs respectively admit a private quantile for which the teacher
requires opposite preferences between two unchanged, shared destinations.

One retained witness uses masks 142 (`[1,2,3,7]`) and 158 (`[1,2,3,4,7]`) and random
word `3689348814741910324`. The teacher chooses slot 1 in the first set and slot 2
in the second. The fitted summary and every shared slot's actual score are
bit-identical; the model chooses slot 1 in both. A fixed ranking of those unchanged
scores cannot satisfy both labels. The diagnostic verifies this through the
trained scorer, not just a comparison of latent vectors.

Every additive encoder produces distinct summaries for all 255 subsets in each
tested equal-energy slice, with no such exact-collision witness. Mixed-food
models are inspected separately at common energy levels 1, 2, and 3; those slices
are not estimates of collision frequency in the mixed validation distribution.
Both summary types distinguish all 15 four-destination sets.

These are findings about the **fitted encoders**. They do not prove that all max
encoders must fail, nor that distinct additive vectors are sufficiently separated
or easy for the scorer to use. The additive arm's remaining errors show that
removing exact collisions alone does not solve selection.

## Training-control investigation

All earlier attempts remain retained, including failures:

1. Revision 1 used two ReLU scoring layers and an unbounded two-layer ReLU encoder.
   It failed even the teacher-informed positive control, so it was inconclusive
   about set representation quality.
2. Revision 2 restored a single hidden scoring layer and added loss traces.
   Several runs initially learned, then diverged; some recorded cross-entropies
   rose into the hundreds. This was not a successful model selected at an earlier
   update count.
3. Revision 3 added production behavior cloning's norm-1 gradient clipping.
   Clipping alone did not consistently stabilize the controls.
4. Revision 4 bounded both encoder layers with tanh. The positive control then
   passed all nine gates; the max summary passed none. A diagnostic re-execution
   added context ablation and collision witnesses without changing training.
5. Revision 5 added the matched additive arm to address those measured collisions.
   It retained the parameter layout, bounded encoder, clipping, seeds, and budget.

A numerical test independently compares batched linear gradients with flattened
slot-row gradients; it passes. This did not establish a backend defect. The
bounded encoder is associated with the restored controls, but the runs do not
by themselves prove a unique cause for the earlier divergence.

## Next gate

Test a learned **per-destination relational summary**: compare each candidate's
relative geometry and food with the other legal candidates before pooling. This
is a hypothesis motivated by the remaining rank/food-selection errors, not an
established remedy. Keep the additive arm and rank control as baselines, keep
random encoding consistent, predeclare the budget, and verify that the extra
pairwise computation is affordable at the maximum slot count.

Do not promote the best two four-destination fits, extend the old residual dose
sweep, or launch an ecological matrix from these synthetic results. A selected
representation still needs fresh matched real correction data, inherited seed
ledgers, exact-zero migration, frozen inherited behavior, and portable/WASM
parity before ecological qualification.

## Reproduction and checks

```sh
cargo test --locked -p blob_rl --no-default-features --features ndarray --example target_set_probe
cargo run --locked --release -p blob_rl --no-default-features --features ndarray \
  --example target_set_probe -- --output /tmp/new-target-set-run
```

The output directory must be new. The example writes its source/hash-bound plan
before training, progress after each arm, and a complete report at the end.
Evidence directories are `training-output/target-set-2026-09-12-v1` through `v5`,
`target-set-2026-09-12-v4-diagnostic`, and `target-set-2026-09-12-final`. They retain
copied binaries, sources, compiler/command identities, predeclared plans, all
results, logs, and bounded-run status. The final run reproduces revision 5 after
a lint-only diagnostic-loop cleanup and expansion of the isolation test.
The final run took 68.43 seconds. Its verifier confirms exact reproduction of all
36 results, unchanged results for all 27 earlier arms, matching compiled source/
teacher/lockfile hashes, gate arithmetic, and 36 independently checked teacher
witnesses. Run `python3 training-output/target-set-2026-09-12-final/verify.py` to
repeat that audit against the retained reports.

Four focused tests cover masked-slot and row isolation for both learned summaries,
exclusion of teacher-rank features, tied-contribution gradients through sorted
summation, batched gradient consistency, and mixed-food teacher semantics.
Focused Clippy, formatting, and the schema registry also pass. Production model,
training, inference, and checkpoint formats are unchanged.
