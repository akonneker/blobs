# Specialist distillation paired sweep

This immutable six-run matrix compares the frequent four-stage curriculum with
and without stage-local frontier-teacher distillation. Seeds 42, 43, and 44 are
paired. Every run uses the same verified behavior clone, qualification artifact,
physics, rewards, PPO settings, curriculum cadence, evaluation suite, and
524,288-quanta-per-environment training target. The only experimental variable
is `[specialist_distillation]` at ecology/combat coefficients `0.10`/`0.10`.

The control qualified on 2/3 seeds; specialist distillation qualified on 3/3.
Seed 43 was already jointly qualified at update 16 in both variants. On seed 42,
distillation qualified at update 16 while the control required update 33. On the
diagnostic seed 44, the control never qualified and distillation qualified at
update 33 with 100% / 86.5% retained feeding survival, eight skirmish kills, and
1,216 skirmish damage.

The improvement costs inference throughput. Mean end-to-end collection rate was
309.9 actions/s for control and 274.8 actions/s for distillation, a paired mean
difference of -35.1 actions/s and a mean ratio of 0.887. Simulation throughput
fell from 52,176.6 to 46,115.3 quanta/s, a ratio of 0.884. With only three
replicates, the paired 95% Student-t interval for the action-rate difference is
wide (-35.1 +/- 36.6 actions/s), and the run order was control-first rather than
interleaved. Treat the approximately 11% cost as a useful engineering estimate,
not a stable benchmark.

`aggregate.json` contains the verified qualification-rate and throughput
summaries and paired differences. Each `result.json` contains the independently
verified `best.json` competency evidence for that run. The execution contract
hash-binds the trainer and initial-policy provenance arguments.
