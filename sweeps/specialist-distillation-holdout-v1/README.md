# Specialist-distillation holdout confirmation

This paired confirmation uses five training seeds (`45`–`49`) that were not
used to select the `0.05` coefficient. The control and distillation arms share
the same qualified initial policy, frequent four-stage curriculum, PPO setup,
fixed evaluation suite, and 524,288 simulation quanta per environment. Only
stage-local specialist distillation differs.

| Seed | Control | Coefficient 0.05 | Control best | 0.05 best |
| ---: | :---: | :---: | ---: | ---: |
| 45 | qualified | qualified | update 16 | update 16 |
| 46 | failed | failed | — | — |
| 47 | qualified | qualified | update 16 | update 32 |
| 48 | qualified | qualified | update 16 | update 16 |
| 49 | failed | failed | — | — |

Both arms qualified 3/5 runs. Seeds 45 and 48 qualified before a specialist
could affect later training and produced identical best evidence. Seed 47 also
qualified without distillation at update 16; the `0.05` arm did not publish its
joint checkpoint until update 32. There is therefore no holdout evidence that
`0.05` improves the probability or time of qualification.

The two failures expose different limits. Seed 46 had no kill-qualified combat
checkpoint at any evaluation boundary, so no combat teacher could be selected;
both variants followed the same trajectory and ended with qualified feeding
but zero skirmish kills. Seed 49 did retain update 16 as a combat teacher with
eight skirmish kills. Distillation improved the later ecology checkpoint's
skirmish damage from 1,544 to 1,960, but it still produced zero kills and never
jointly qualified. The mechanism can preserve some combat intensity without
reliably preserving the discrete behavior required by the gate.

Across the original seeds 42–44 and these untouched seeds, the observed rates
are 5/8 for control and 6/8 for `0.05`; the entire difference comes from seed
44. That is insufficient to enable specialist distillation by default or carry
it into large-world training as a presumed improvement. The `0.05` profile is
retained as an opt-in diagnostic while the next work should address reliable
combat-specialist availability and kill retention.

Mean sampled throughput was 370.4 actions/s for control and 331.9 actions/s for
`0.05`, but the control arm ran first and showed large late-run host variation.
These ordered wall-clock samples are recorded for diagnostics and are not used
as scientific evidence about the coefficient.
