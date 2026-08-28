# Held-out combat-horizon sensitivity

This evaluation-only experiment applies three held-out contact/skirmish
horizon pairs to byte-identical frozen checkpoints from the schema-36
four-cycle source cohort. Training rollouts, weights, physics, reward, opponent
mix, and the eight evaluation seeds are identical within every checkpoint
comparison. Each JSON artifact binds the complete source checkpoint metadata
and model hashes, exact training configuration, seed suite, horizon pairs, raw
contact reports, and pass decisions.

The primary matrix covers update 16 and update 32 for seeds 45–49:

| Contact/skirmish quanta | Passed boundaries | Seed/update passes |
| ---: | ---: | :--- |
| 8,192 / 16,384 | 3/10 | 45/16, 47/16, 49/16 |
| 16,384 / 32,768 | 5/10 | 45/16, 45/32, 47/16, 48/16, 49/16 |
| 32,768 / 65,536 | 5/10 | 45/16, 45/32, 47/16, 48/16, 49/16 |

The maintained 16,384/32,768 gate is materially better than the half window:
it gives seed 45 update 32 and seed 48 update 16 enough canonical time to
produce the required skirmish kill. Doubling the maintained window produces no
additional pass. Several failed policies continue accumulating damage at the
longest window without converting it into skirmish kills, while others plateau
entirely. Their failure is therefore policy behavior, not a timeout artifact.

Qualification also decays sharply during continued PPO. Four of five update-16
policies pass at the maintained horizon, but only seed 45 still passes at
update 32. The next combat work should target retention/forgetting between
evaluation boundaries rather than lengthen the gate. The maintained horizon
remains the default; the half horizon is too censoring and the double horizon
adds evaluation cost without changing qualification in this cohort.

The five `seed-N.json` files are preliminary terminal-checkpoint aliases. The
ten `seed-N-update-U.json` artifacts are the balanced primary matrix.
