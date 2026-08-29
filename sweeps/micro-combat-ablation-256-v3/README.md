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
