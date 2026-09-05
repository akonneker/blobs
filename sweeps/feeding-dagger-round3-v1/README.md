# Feeding DAgger round 3 V1

This slice follows the round-two foraging residual through the first complete
hierarchical correction. The maintained collision-aware forager remains a
valid target on the identical 256-cell horizon: across eight disjoint seeds it
retains 97.5% of on-food cells and 93.7% of adjacent-food cells, with 257.461
and 249.648 consumed energy per initial cell.

## Isolated physical heads

Behavior-cloning schema 27 adds verified-parent `--effort-head-only`
adaptation. Along with the existing foraging-adapter, action-kind-expert, and
target-query modes, it retains exactly one selected parameter family after
each optimizer update. The modes are mutually exclusive and do not alter the
Mind ABI, observation, physics, or runtime execution.

Component telemetry in `feeding-policy-corrections` now decomposes every flat
action disagreement into kind, target-only, effort-only, and target+effort
counts. The inherited round-two trajectory contained 196 kind errors, 760
effort-only errors, and 68 target+effort errors. Sixteen effort-only epochs at
`1e-3` changed held-out Move exact accuracy from 0% to 100% while preserving
every other output. Its new on-policy frontier was 784 Consume-to-Move kind
errors, 272 Move-to-Attack kind errors, and 48 target-only errors across 8,192
round-three samples.

The overlap auditor now reports conflicts within individual datasets as well
as across datasets. Round three has 35 exact observable states, 11 states after
projection to the eight raw foraging-adapter inputs, and zero conflicting
action-family labels under either representation. This rules out observation
aliasing for the Consume/Move boundary.

## Round-three correction

The inherited five-logit kind margin required a larger adapter dose than the
initial `1e-3` run. Sixteen adapter-only epochs at `1e-2` reach 100% held-out
Consume accuracy. Four interaction-head-only epochs then correct the late
Move-to-Attack frontier and reach 100% held-out action-kind accuracy. An
eight-epoch, flat-label-balanced target-head stage removes fresh kind errors
but does not improve exact target agreement; it is retained as the rollout
candidate because its physical behavior is substantially better.

On a fresh 4,096-decision policy trajectory the candidate agrees exactly on
3,977 decisions (97.1%); all 119 remaining errors are target-only. Three
disjoint rollout seeds give identical 33.6% on-food survival with 144.652
intake per initial cell. Adjacent-food survival is 58.2%, 59.0%, and 58.2%,
with 183.289, 185.992, and 183.289 intake. The preceding effort-only policy
had 0% survival and 57.000/33.844 intake. This is a real escape from collapse,
but it remains below the 80% survival promotion gates and is not promoted.

The next DAgger round should collect from the selected target-head artifact,
train target labels with direction-aware sampling or a smaller target-specific
adapter, and then correct the remaining late Consume boundary. It should stop
if fresh target agreement or survival regresses, and it must rerun combat
retention before promotion.

## Bound artifacts

| artifact | SHA-256 |
|---|---|
| round-three correction manifest | `beb6edd14dbe08e6d43eded030feea87fd3650f4efdaba2cc2e54226b306d557` |
| round-three correction payload | `8cd06ac28dd34c2906fb13472b74356c4f7296552fd6368f948ec8ad374ba8c4` |
| interaction-head metadata | `7aea5a3dd28a44313a7344bcc06e45fe5106848dfb693c46110917dcbe614492` |
| selected target-head metadata | `26b38d11ab6b5e63cd2fa7a7c6c97d91eff697623b42668f1823cc2895ba6f7e` |
| selected model | `07b93dbd87eee9a5e34334329c85deef51ab65cb3a3d91f2dff68de13014caeb` |

Local immutable evidence is retained under
`training-output/feeding-dagger-round3-v1` and
`training-output/feeding-foraging-context-adapter-v1`.
