# Learned utility on feeding observations — 2026-09-12

The sampled utility now passes its **conditional offline transfer gate in all
three initializations**, on both retained and newly collected development
trajectories. Treatment target agreement on the new trajectories is
**97.99–98.83%**, with **98.03–99.12% retention** on unchanged Move labels.
All three matched controls also pass. Six full-precision diagnostic model records
reload with identical parameter bits and evaluation results.

This follows the [portable sampler](sampled-target-deployment-2026-09-12.md).
The utility is still a research example, not an exported composite Mind or a
promoted feeding policy. Non-Move outputs are held fixed by the offline setup;
live inherited behavior and ecological improvement remain unverified.

## Data and isolation

`blob_rl/examples/feeding_utility_probe.rs` uses the production demonstration
loader and correction-pair verifier. Across line, checkerboard, ring,
loose-random and random layouts, the retained pairs contain 10,240 observations,
3,880 parent Moves and 1,453 active target corrections. The old training seed
1433000101 supplies 1,921 Move rows; old development seed 1433000202 supplies
1,959, including 720 active and 1,239 unchanged labels.

Fresh pairs under `training-output/feeding-utility-fresh-2026-09-12-v2` use
development seeds **1435400201/1435400202**. They supply 3,840 Move rows, including
1,456 active and 2,384 unchanged labels. These seeds are evaluation-only for
gradient fitting but were used to develop and compare objectives; they are not
final confirmation. Random-layout active coverage is only six rows, so aggregate
success is not a strong per-layout qualification.

Each matched pair has identical observations, recurrent prefixes and non-action
labels. Only eligible Move targets change; kind and effort remain identical.
The utility uses the exact target mask for the recorded parent's effort.
Private bytes are recovered from the normalized observation with a bit-exact
re-encoding check, and the portable decoder consumes little-endian word zero.

The parent is
`training-output/feeding-dagger-round5-v1/effort-control-dose-lr005-steps10-v1`:

| Artifact | SHA-256 |
| --- | --- |
| Parent metadata | `fc66244380552e63d23908973eb50f3c34ef13212498575e27a52064504633a7` |
| Parent model | `e098ffa39ee1e247dca18dc959cdfbe8118d055260fed61b80980dcaef867bc2` |
| Source configuration | `4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9` |

The source configuration is `blob_rl/config/combat_warm_start_256.toml`.
Source/effective configuration and both rules hashes match the old corpus for
every layout. An initial collection accidentally used
`competitive_transfer_large.toml`; its partial directory
`feeding-utility-fresh-2026-09-12-v1` is retained and excluded from every fit and
reported evaluation. The mismatch was caught before fitting.

The production legacy audit checks 16 ancestor artifacts and 18 corpora,
inheriting 52 exposed seeds. All probe seeds are disjoint from that parent
ledger. Reserved confirmation seeds **1434999901/1434999902 remain untouched**.
Future integration must additionally inherit the old/fresh development seeds,
sampling seed 1435400101 and initializations 1435400301/0302/0303 from this study.

## Model and objective

The 5,105-parameter utility uses all 33 raw slot features, with dx/dy restored to
tile scale. A 33→16→16 tanh encoder feeds a sorted legal-slot sum divided by the
fixed capacity 32. Each slot's raw features, encoding and shared summary feed
a 65→64 ReLU→1 scoring head. The final head starts at zero. Random bytes,
headers, trajectory keys, labels and teacher ranks do not enter network features.
Scores use the existing Q48/u64 portable categorical decoder in (dy, dx) order.

Each fit uses 1,024 updates of 128 uniformly sampled training rows, AdamW at
0.005 and norm-one gradient clipping, with the corrected CPU backend. All arms
use the same batch-index stream. No parent parameters enter the optimizer.

Ordinary hard-label cross-entropy passes only one of three treatment fits.
Nearly every held-out error selects a target with the same visible food and
occupancy features as the label. This diagnostic compares five slot features;
it does not establish full teacher equivalence or rule out distance effects.

The succeeding objective trains the probability interval containing the
**recorded private quantile**. If L is cumulative probability before the labeled
slot, U is cumulative probability through it, and q is the recorded quantile,
the loss is `relu(L - q + 1e-6) + relu(q - U + 1e-6)`, averaged across rows.
The before/through masks are supervision only. This aligns fitting with the
sampled choice rather than pushing every observed label toward probability one.
The smooth loss uses f32 probabilities and quantiles; qualification still uses
exact u64 draws and the portable Q48 decoder.

## Matched results

Each fit must reach at least 95% overall Move agreement, 90% active agreement
and 95% unchanged-Move retention on **both** development cohorts. Every
evaluation also asserts finite scores, legal sampled targets and exact score
bits under slot and row reversal. There is no per-layout threshold in this gate.

| Initialization suffix / arm | Old overall | Old active | Old retention | Fresh overall | Fresh active | Fresh retention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 301 control | 99.74% | 99.86% | 99.68% | 99.84% | 99.59% | 100.00% |
| 301 treatment | 97.04% | 97.08% | 97.01% | 97.99% | 97.46% | 98.32% |
| 302 control | 99.59% | 99.58% | 99.60% | 99.69% | 99.45% | 99.83% |
| 302 treatment | 98.26% | 98.19% | 98.31% | 98.83% | 98.35% | 99.12% |
| 303 control | 99.59% | 99.44% | 99.68% | 99.82% | 99.59% | 99.96% |
| 303 treatment | 97.86% | 99.31% | 97.01% | 98.39% | 98.97% | 98.03% |

Evidence is retained under `training-output/`:

- `feeding-utility-2026-09-12-v1`: six cross-entropy fits; controls 3/3 and
  treatments 1/3 pass both cohorts.
- `feeding-utility-2026-09-12-v2`: twelve fits comparing both objectives.
  All six cross-entropy fits reproduce v1 exactly; interval fits pass 6/6.
- `feeding-utility-2026-09-12-v3`: six interval fits, unchanged from v2, with
  full-f32 `.mpk` records. Reloads preserve all 5,105 parameter bits and both
  development evaluations. These files are diagnostic records, not policy
  checkpoints; no policy schema or execution identity is assigned to them.

Each run retains its exact executable, command, status, source copies and hashes,
plan, per-layout results and loss checkpoints. The verifier checks archived
hashes, split provenance, grid completeness, metric arithmetic, gates, record
hashes and common-fit equality; it does not rerun inference:

```sh
python3 scripts/verify_feeding_utility.py training-output/feeding-utility-2026-09-12-v2 --unchanged-from training-output/feeding-utility-2026-09-12-v1
python3 scripts/verify_feeding_utility.py training-output/feeding-utility-2026-09-12-v3 --unchanged-from training-output/feeding-utility-2026-09-12-v2
```

Validation passes **615 workspace tests (33 ignored)**, the 13 existing target
probe tests and three new utility tests. The new tests check interval-loss
gradient direction, exclusion of private randomness/labels/host metadata from
network features, and isolation from illegal slots and other batch rows.
All-target/all-feature workspace Clippy passes with warnings denied, as do
formatting, the 63-constant schema registry and diff whitespace checks. Logs
and verifier outputs are retained in the v3 evidence directory. CI runs both
focused probe examples alongside its workspace checks.

## Next integration gate

The [composite deployment follow-up](feeding-composite-deployment-2026-09-12.md)
now closes the packaging and bounded runtime-fidelity work below. All six
utilities preserve recorded target choices, and the first matched pair passes
native/WASM and inherited-output checks. The next experiment is the three-arm
feeding matrix; ecological improvement remains unverified.


Bind an archived utility and the frozen parent into an explicit composite
artifact with a distinct sampled execution identity and cumulative lineage.
Replace only Move target selection while retaining the parent's kind, effort,
amount, signal and memory. Measure any Burn-to-portable numerical differences,
then require exact portable-native/WASM behavior and live retention before
running the five-layout ecological matrix. Full-capacity learned behavior and
visibility generalization remain open; 32-slot decoder coverage alone does not
qualify this learned utility. Keep final confirmation seeds reserved until the
candidate and evaluation plan are frozen.
