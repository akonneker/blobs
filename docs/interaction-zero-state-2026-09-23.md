# Full-corpus zero-state normalization — 2026-09-23

**The bounded normalization diagnostic is complete and negative: none of the
three normalized runs passes the unchanged joint training gate.** Zero-state
handling removes the measured range problem, but the hard-fixture fitting result
does not transfer to joint fit on the full training corpus at the frozen budget.
No candidate advances to development or deployment.

## Frozen comparison

The [roadmap's track 1B](learning-and-environment-roadmap.md) compares raw memory
with zero-state-preserving standardization on all 8,312 previously recorded
training rows: 890 combat and 7,422 feeding. All-zero incoming memory maps to a
positive-zero residual context; every other memory vector uses the unchanged
128-channel means/scales from the [hard-fixture study](interaction-standardized-2026-09-13.md).
The parent receives its original memory. There is no clipping or refitting of
normalizer statistics.

Architecture (7,514 parameters), labels, frozen parent scores, initializations
1435600301/302/303, sampling streams, optimizer and 2,048-update budget match the
[full-corpus context study](interaction-context-2026-09-13.md). Each batch draws
64 combat and 64 feeding rows uniformly with replacement. AdamW uses learning
rate 0.005, zero weight decay and norm clipping at 1. All six fits and all 18
checkpoints at 512/1,024/2,048 updates are retained. The final gate remains at
least 95% combat agreement, 90% attack agreement and 99% feeding agreement in
every target initialization.

The pre-fit range audit finds 2,146 zero-memory rows, all transformed to zero.
The largest absolute context value drops from 20,505.51 under unconditional
centering to 28.19. Of all rows, 2,099 exceed magnitude 4, 1,077 exceed 8, 621
exceed 16, and none exceeds 64. These are descriptive buckets, not fit gates.
Python independently reconstructs the f32 context bytes and parent-memory hash.

## All final results

Counts use the full corpus; attack counts use its 864 combat Attack labels.

| Initialization | Arm | Combat / 890 | Attack / 864 | Feeding / 7,422 | Feeding shortfall to 7,348 | Joint gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 301 | Raw | 869 | 864 | 7,264 | 84 | Fail |
| 301 | Zero-state | 869 | 863 | 7,282 | 66 | Fail |
| 302 | Raw | 872 | 864 | 7,262 | 86 | Fail |
| 302 | Zero-state | 871 | 863 | 7,274 | 74 | Fail |
| 303 | Raw | 870 | 864 | 7,283 | 65 | Fail |
| 303 | Zero-state | 871 | 864 | 7,270 | 78 | Fail |

Normalized final feeding agreement is 97.95–98.11%. Paired correct-count changes
are +18, +12 and −13 feeding rows, and 0, −1 and +1 combat rows. All final combat
and attack gates pass; feeding fails. None of the 18 retained checkpoints passes
the joint gate. Counts from correlated training rows are descriptive, not
independent episode-level evidence of generalization or action value.

The saved-weight scalar replay verifies **149,616 predictions**. Every checkpoint
has zero scalar/Burn kind mismatches. The nine raw checkpoint records reproduce
the prior study exactly, including losses, predictions, initial weights and
sampling identities; all nine saved weight files are byte-identical. This
supports attribution of the comparison to the declared input transform.

## Implementation and verification

The existing context probe now accepts the two-arm frozen plan and writes a
separate context audit. Its historical three-arm path remains available.
`scripts/verify_interaction_zero_state.py` checks the training-only source chain,
normalizer, context bytes, checkpoint coverage, saved predictions, legality,
count denominators, thresholds and exact raw-control regression. The existing
context verifier still returns exactly its archived baseline result.

The [harness](harness-usage.md) adds a `zero-state` study group with a count-based
joint-gate summary. It distinguishes successful execution and verification from
scientific rejection. `--require-scientific-pass` correctly exits 1 for this
negative study. Summary tests reject forged counts, rates, gates, thresholds,
duplicate checkpoints and wrong prediction/domain coverage. Six evidence
mutations reject changed budget, zero-state contract, parent/context hashes,
feeding count and raw-control loss.

Validation: 36 Rust tests pass through the `rl-probes` harness; 34 harness tests
pass; all six semantic mutations are rejected. RL all-target/all-feature Clippy,
formatting, the 66-constant schema registry and whitespace checks pass.
The archived probe, plan, source
snapshot, context range audit, six fits, replay, mutation checks and verification
remain under ignored `training-output/interaction-zero-state-2026-09-23-v1`.
Plan SHA-256: `dc8e244f9a02042b5ac11d0e55a3a7229c0ede2fd2215e2bc7c6647aab7d2aaf`.
The final reporting run is `training-output/harness/zero-state-study-2026-09-23-v2`;
v1 is the preliminary audit before ancillary evidence publication completed.
Rust coverage is in `training-output/harness/zero-state-rl-probes-2026-09-23-v1`.

No new simulation episodes, seeds, development predictions, runtime execution
contract or deployed export are introduced. Confirmation seeds 1434999901 and
1434999902 remain untouched. Raw artifacts and binaries are not committed.

## Decision and next work

Close track 1B without expanding architecture, optimizer or budget searches.
This result rejects sufficiency of the tested transform under this fixed
protocol; it does not establish that the labels are unlearnable or useful.

Next is track 1A's environment inventory, followed by fixed legal controls and
track 2's paired action-value checks. Source inspection already shows why the
inventory must identify effective environments separately: the base world has
resources and a 16,777,216-quanta deadline, while contact/skirmish evaluation
removes food, uses one/four cells per team and ends at 32,768/65,536 quanta.
Feeding qualification separately uses on/adjacent resources, a waiting opponent,
five layouts, a 262,144-quanta horizon and a 4,096-step safety cap. Those are
configuration facts, not new ecological measurements or profitability evidence.
Do not treat characterization of the base TOML alone as calibration of all assays.

Useful opportunities, legal observability and joint learning retention remain
unresolved requirements before development, autonomous combat and self-play.
