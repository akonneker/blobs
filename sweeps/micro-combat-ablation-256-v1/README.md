# Micro-combat rehearsal ablation, 256×256

This immutable six-run plan compares balanced named-scenario rehearsal against
an otherwise identical no-rehearsal control. Both arms use the same three
training seeds, six held-out scenarios, evaluation seeds, gates, physics,
rewards, PPO/model settings, 256×256 competitive world, and canonical-time
budget. After normalizing the required artifact destination, the only expanded
configuration difference within a seed pair is
`combat_curriculum.micro_combat.rollout_enabled`.

Each environment must reach 1,048,576 authoritative simulation quanta. The
20,000,000-action limit is a safety ceiling, not the experimental clock. Run
one trainer at a time on a single GPU:

```sh
cargo build --release -p blob_rl --bin train --bin rules-sweep-run
target/release/rules-sweep-run \
  sweeps/micro-combat-ablation-256-v1/manifest.json \
  --max-parallel 1
```

The executor resumes interrupted runs from verified checkpoints and publishes
`aggregate.json` only after all six hash-bound results pass verification. The
paired aggregate reports terminal and best survival/elimination rates, whether
the joint micro gate was ever reached, normalized time to first qualification,
simulation/action throughput, and peak host resident memory. Peak RSS excludes
GPU device allocations; compare device memory separately when selecting a GPU
deployment profile.

Execution was attempted on the CPU NdArray backend. All six runs failed before
their first update because the dense block contains interior cells with no
vacant neighbor, violating the adjacent-food retention setup. No learning
result was produced and this plan must not be retried. The durable failure
statuses and execution summary are retained as provenance. V2 changes both
arms to the checkerboard assembly and leaves the paired intervention intact.
