# Micro-combat rehearsal ablation, 256×256 (v3)

This is the executable correction of V2. It retains the same two checkerboard
arms, three paired seeds, held-out suite, curriculum, training budget, and sole
scientific intervention:
`combat_curriculum.micro_combat.rollout_enabled`.

V3 additionally includes regression coverage proving that every curriculum
stage initializes for every run and that micro-combat evidence binds the
256×256 base-world compiled identity even though its hashed scenarios execute
on 7×7 worlds.

Every environment must reach 1,048,576 authoritative simulation quanta;
20,000,000 actions is only a safety ceiling. Execute serially with the faster,
lower-memory CPU backend:

```sh
cargo build --release -p blob_rl --bin train --bin rules-sweep-run \
  --no-default-features --features ndarray --locked
target/release/rules-sweep-run \
  sweeps/micro-combat-ablation-256-v3/manifest.json \
  --max-parallel 1
```

V1 and V2 remain immutable failed-attempt provenance and must not be retried.

## Result

All six runs completed successfully at the exact 1,048,576-quanta-per-env
budget. The treatment did not meet the promotion criterion and should not be
promoted:

| held-out metric | no rehearsal | balanced rehearsal | paired change (95% CI) |
| --- | ---: | ---: | ---: |
| final survival success | 2.08% | 4.17% | +2.08 pp ± 8.96 pp |
| best survival success | 3.13% | 5.21% | +2.08 pp ± 16.16 pp |
| final/best elimination success | 0% | 0% | 0 pp |
| micro-combat qualification | 0/3 | 0/3 | none |
| fixed-suite terminal win rate | 0% | 0% | 0 pp |

The small survival difference comes entirely from seed 256203: treatment
finished at 12.5%, control at 6.25%; the other two treatment seeds finished at
zero. Every terminal held-out micro scenario in every arm recorded zero
committed attacks, so the primary failure is action acquisition, not attack
damage or insufficient focus within an otherwise offensive policy. Five of six
terminal policies also failed both feeding tasks.

Training-distribution telemetry is descriptive but not a causal held-out
comparison because rehearsal deliberately substitutes small scenarios for
some ordinary rollout blocks. It nevertheless shows the treatment becoming
more spatially concentrated (mean spatial entropy −0.116 ± 0.011; encounter
edges −1.585 ± 0.083) without acquiring a greedy attack policy. The next
experiment should therefore target action acquisition and feeding retention,
not simply extend this rehearsal budget.
