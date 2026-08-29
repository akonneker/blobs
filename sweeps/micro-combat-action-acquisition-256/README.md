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

## Result

All six runs completed successfully after every environment reached the
1,048,576-quanta authoritative world-time budget. Five stopped exactly at the
frontier; one control environment crossed it by one 256-quanta resolver batch.
The intervention changed training behavior
but did not produce a greedy combat policy and must not be promoted:

| held-out metric | uniform rehearsal | attack acquisition | paired change (95% CI) |
| --- | ---: | ---: | ---: |
| final survival success | 0% | 0% | 0 pp |
| best survival success | 0% | 1.04% | +1.04 pp ± 4.48 pp |
| final/best elimination success | 0% | 0% | 0 pp |
| final/best committed attacks per episode | 0 | 0 | 0 |
| micro-combat qualification | 0/3 | 0/3 | none |
| fixed-suite terminal win rate | 0% | 0% | 0 pp |

The training-time intervention itself worked. Across stride-sampled named
micro-combat episodes, control committed 62 attacks in 65 episodes, with 8
successful attacks, 95 applied damage, and no kills. Treatment committed 233
attacks in 92 episodes, with 36 successful attacks, 594 applied damage, and 5
kills. Because telemetry samples every eighth completed episode and the arms
have different episode counts, these totals characterize the mechanism rather
than serve as the held-out endpoint.

The failure is therefore not lack of attack exposure. A fixed 50% behavior
mixture generates attacks but weakens their policy-gradient attribution: when
the fixed component dominates an Attack sample, only the smaller learned
component responds to its return. Kind-only forcing also leaves the policy to
learn target, effort, timing, and survival together from sparse outcomes. At
the terminal boundary no run met the full feeding gate; only one control seed
showed successful consumption in both feeding scenarios, but its cells did not
meet the survival requirement.

Mean throughput was 2,463 actions/s for control and 2,142 actions/s for
treatment; the paired change was −321 ± 1,142 actions/s and is not resolved by
three pairs. Mean peak host RSS was approximately 641 MB and 625 MB,
respectively. The next experiment should use attack-dense supervised combat
demonstrations or a verified combat specialist as a warm start, then retain
feeding through multi-task demonstration coverage and functional anchoring.
