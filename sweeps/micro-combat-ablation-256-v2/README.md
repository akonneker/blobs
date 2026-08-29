# Micro-combat rehearsal ablation, 256×256 (v2)

V2 corrects the invalid dense-block starting assembly discovered when V1 was
launched. Both arms now use a checkerboard assembly, which preserves 256 cells
per team while guaranteeing every founder a local vacancy during adjacent-food
retention. An executable test constructs every curriculum stage for both arms
and all three paired seeds.

The scientific intervention remains only
`combat_curriculum.micro_combat.rollout_enabled` after normalizing artifact
destinations. Every environment must reach 1,048,576 authoritative simulation
quanta; 20,000,000 actions remains a safety ceiling.

Run serially with the CPU NdArray release build:

```sh
cargo build --release -p blob_rl --bin train --bin rules-sweep-run \
  --no-default-features --features ndarray --locked
target/release/rules-sweep-run \
  sweeps/micro-combat-ablation-256-v2/manifest.json \
  --max-parallel 1
```

V1 is retained as failed provenance and must not be retried.
