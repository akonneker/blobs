# Micro-combat action-acquisition A/B

This immutable six-run plan tests whether explicit, training-only Attack
sampling solves the zero-attack failure observed in the completed V3 rehearsal
sweep.

- Three paired seeds: `256301`, `256302`, `256303`.
- Both arms use named micro-combat rehearsal, the same 256×256 checkerboard
  competitive world, six held-out scenarios, rewards, schedule, and feeding
  gates.
- Both arms require at least `0.25` held-out committed attacks per evaluated
  scenario episode, 75% survival-objective success, and 25%
  elimination-objective success.
- Control uses only the ordinary uniform legal-kind exploration mixture.
- Treatment reserves 50% behavior-policy mass for Attack whenever Attack is
  legal in an assigned named training scenario.

The Attack mixture is recorded per transition and reconstructed exactly by
PPO. It is not applied in ordinary competitive rollouts or held-out greedy
evaluation, and it is not visible through the Mind ABI. Feeding qualification
remains an independent hard condition, so an aggressive policy that forgets
how to consume cannot qualify.

The generated [`manifest.json`](manifest.json) binds all six expanded configs
and hashes. Run it serially with the measured CPU backend:

```sh
cargo build --release -p blob_rl --bin train --bin rules-sweep-run \
  --no-default-features --features ndarray --locked
target/release/rules-sweep-run \
  sweeps/micro-combat-action-acquisition-256/manifest.json \
  --max-parallel 1
```

The primary acquisition endpoint is terminal held-out attack commitments per
episode. Survival, elimination, joint qualification, time to qualification,
feeding retention, fixed-match performance, throughput, and peak host RSS
remain required secondary evidence. Three pairs provide a directional result;
expand the seed set before policy selection if confidence intervals remain
wide.
