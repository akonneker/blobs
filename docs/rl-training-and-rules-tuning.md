# RL training and rules-tuning plan

## Learned-policy randomness semantics

Learned policies receive the same deterministic, cell-private 32-byte random
block as native and Wasm Minds. A fixed match seed reproduces every block, but
different cells and decision invocations receive independently derived values.
The RL host prepares each ready frontier exactly once and caches only the cell
routing handle plus that invocation's random bytes. Repeated observation reads
therefore cannot advance the private sequence, and exact checkpoints retain
and validate a pending frontier without exposing cell identity or random state
through any other cell's Mind input.

Zero-filled policy randomness was a historical evaluation simplification, not
a valid deployment mode. Feeding-evaluation schema 4 deliberately rejects
schema-3 results so those measurements cannot be resumed or promoted under the
corrected semantics. Greedy-policy correction collection now uses
demonstration schema 18, and counterfactual source trajectories plus frozen-
policy continuations use branch schema 4. Both policy and teacher see the same
canonical private block at a decision; fixed seeds remain exactly reproducible.
Diagnostics may still request zero randomness explicitly for a bound A/B
comparison. Micro-combat physical-state search may intentionally canonicalize
randomness away when policy behavior is not being evaluated.

Promotion evidence must be both seed-robust and layout-complete. The first
schema-4 control passed its declared checkerboard seeds at 80.57% adjacent
survival but failed a disjoint eight-seed confirmation at 79.88%. A schema-2
layout matrix then failed line and ring while passing checkerboard,
loose-random, and random. Near-threshold aggregate cell survival is diagnostic,
not sufficient promotion evidence: retain disjoint seed blocks, inspect
seed-cluster variation, and require every declared founder geometry to pass.

Status: event-time rollout correctness, PPO stability, deterministic held-out
evaluation, immutable artifacts, and exact update-boundary resume are
implemented. Typed baseline opponents and per-matchup held-out evaluation are
implemented. Bounded, rated snapshot self-play with exposure-aware selection is
implemented. The canonical serializable rules-profile boundary, immutable
replicated sweep planner, bounded/resumable process executor, paired
aggregation, and bounded authoritative gameplay telemetry are implemented.
An episode-boundary feeding curriculum and resolver-confirmed feeding
promotion gate are implemented.
Deterministic mirrored evaluation against the exact maintained simple,
aggressive, defensive, explorer, and colony Mind entrypoints is also
implemented as a local/unverified control matrix.

## Operational objective

Training must optimize the same anonymous, isolated Mind contract used by
native play, browser replay, and server verification. Experiment seeds,
ruleset hashes, environment configuration, reward configuration, model
configuration, and code revision must be sufficient to reproduce a result.
Rules should be tuned against gameplay metrics and multiple policy families,
not selected merely because one reward-shaped PPO run learns quickly.

Scenario failures must be attributed through the ordered analytical,
full-state, Mind-feasibility, maintained-teacher, on-policy recovery, learned
agreement, and held-out rollout ladder in
[`physics-vs-training-diagnostics.md`](physics-vs-training-diagnostics.md).
This prevents optimizer failure from being mistaken for bad physics and keeps
privileged planning evidence separate from controllers that obey the strictly
isolated per-cell Mind ABI. The schema-1 `scenario-diagnosis` artifact now
hash-binds explicit evidence claims; typed source-report adapters remain
required before it can authorize promotion automatically.

## Rollout semantics

Cells act on asynchronous event frontiers. One policy transition therefore
runs from a cell's decision until that same cell becomes ready again or dies;
resolver batches are not RL timesteps. The collector now:

- keeps one open transition per acting cell;
- accumulates rewards across intervening frontiers, including frontiers where
  another cell is the only actor;
- terminates a cell's trajectory on death and applies its own `cell_died`
  reward plus `team_loses` when the team is eliminated;
- bootstraps from the value at the same cell's next decision;
- discounts with `gamma ^ elapsed_simulation_time` and applies the same
  continuous-time convention to the GAE trace;
- discards action intervals still unresolved at an ordinary rollout boundary
  instead of treating them as zero-value terminals;
- at a simulation-time training boundary, stops opening new transitions and
  submits deterministic waits only until every already-open interval resolves,
  so the terminal policy update, evaluation, and immutable checkpoint cannot
  disappear behind censored tails.

Discarded-tail count and mean completed-transition duration are recorded in
`metrics.csv`. A later collector may carry tails across policy updates, but it
must bind each transition to its behavior-policy version; silently mixing
versions would violate PPO's on-policy assumption.

## Reproducibility and current metrics

`seed` now drives model initialization, environment generation, episode resets,
policy sampling, baseline-opponent randomness, and an independent PPO
minibatch-shuffle stream.
Configuration validation rejects empty workloads, invalid PPO coefficients,
impossible starting populations, and non-finite reward values.

Each training update records actions, completed transitions, discarded tails,
mean simulated transition time, action throughput, actual living training-cell
count, policy/value losses, entropy, approximate KL, policy clip fraction,
explained variance, optimizer steps, completed PPO epochs, KL early stopping,
episode outcomes, and reward summaries. `evaluation.csv` separately records
greedy-policy outcomes on an explicit fixed held-out seed suite. Wait, random,
and aggressive opponents are typed serialized profiles that execute through
the same anonymous Mind input as submitted code. Training selects one fixed
profile through `env.opponent`; evaluation runs every configured
`evaluation_opponents` profile independently on the same seeds. It can also run
integrity-checked immutable policy artifacts listed in `evaluation_snapshots`.
Both forms record per-matchup and suite rows. Self-play pool rotation is still
controlled separately by `[self_play]`; explicitly configured
`evaluation_snapshots` remain held-out opponents and are never silently added
to training rollouts.

## Evaluation and artifact semantics

Periodic evaluation is gradient-free and deterministic: it uses an inner
inference model, never consumes the training action or minibatch RNG streams,
chooses the highest-logit allowed action, and resolves ties toward the lower
canonical action index. The fixed suite starts at `evaluation_seed` and has
`eval_episodes` consecutive seeds. Its reward metric still includes configured
reward shaping, so win/loss/timeout outcomes remain the primary model-selection
measure and later unshaped evaluation must be reported separately.

Snapshot artifacts are resolved and fully verified once at training startup;
evaluation does not reload them per episode. Logs bind each snapshot matchup to
its checkpoint name, update, and model SHA-256. Opponent inference is exposed
through the same `ReferenceMind` interface as native and Wasm policies: each
pool worker owns an independent model clone, receives only the anonymous
canonical local observation, is reset at every invocation, and retains no
hidden policy state. Snapshot rollout and evaluation environments project each
canonical input independently, then execute the row-separable linear/ReLU
network once for the full opponent frontier. Each output row is decoded only
against its originating input. The trusted host owns the routing identifiers;
neither the tensor nor the policy contains cell identity or another cell's
features. Scalar and batched paths are regression-tested for identical
decisions, canonical hashes, host metadata, and private decision sequences.

### Maintained-Mind control matrix

The control matrix is separate from both synthetic RL `OpponentProfile`
baselines and learned checkpoint evaluation. It calls the same policy function
compiled into each maintained Mind, through `ReferenceMindInput` and
`ReferenceMindDecision`, and never provides team identity, host cell IDs, or
shared colony memory. Every two-team seed is played twice with the Minds
swapped between team-zero and team-one placements. The second game's outcomes,
cell counts, and host telemetry labels are normalized back to “colony” and
“control.” This pairing removes a fixed seat from the aggregate and makes seat
sensitivity visible; it does not make a three-seed suite statistically strong.

Create the immutable six-regime plan, then run the exact control matrix:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin rules-sweep -- blob_rl/config/colony_control_sweep.toml
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin control-matrix -- sweeps/colony-controls-event-frontier-v3/manifest.json \
  --max-parallel 4 \
  --output sweeps/colony-controls-event-frontier-v3/control-matrix-event-frontier.json \
  --live-output sweeps/colony-controls-event-frontier-v3/control-matrix-live.json
```

Publication uses create-new plus hard-link semantics and refuses to replace an
existing report. The report hashes the Mind ABI, verified sweep manifest,
semantic and compiled rulesets, scenario, telemetry configuration, profiles,
seed suite, and mirrored-seat policy. Its trust fields remain
`verification_scope=local_deterministic`, `server_verified=false`, and
`replay_committed=false`. The native/Wasm parity test guards implementation
drift, but an online leaderboard must run the submitted Wasm bytes in the
server verifier and bind its replay commitment before changing those labels.
The optional live file is a separate mutable progress schema, atomically
replaced after each matchup. It deliberately omits per-episode telemetry and
remains local/unverified; it is an operational UI feed, not a scientific or
leaderboard artifact.

Victory is currently immediate extermination followed by a draw at the first
resolved canonical event frontier at or after the configured simulated-time
deadline. The objective and deadline are part of `ScenarioProfile` and its
semantic hash. `max_episode_len` is different: it counts team-zero decision
frontiers and is only a fail-closed host guard. It remains in the complete
execution-config hash but is excluded from the scientific scenario hash and
cannot be changed by a scenario sweep. A control run that reaches it is an
error, not a timeout observation. Because the engine observes the deadline at
an event frontier, a deadline that is not itself scheduled may overshoot to the
next frontier. The current 1,048,576-quanta deadline aligns with a canonical
passive-field event, and all 19 draws terminate exactly at that time.

Control-matrix schema 6 records raw terminal energy compartments separately
for the colony and control. Core, assimilated, gut, carried material, and
payload escrow remain primitive evidence: the report and viewer do not yet
choose a gut discount or biomass adjudication formula. Constructed terrain is
not attributed to a team and is therefore not counted.

Four immutable sensitivity plans now isolate different questions:

- `colony_horizon_sweep.toml` varies only the canonical event-time deadline.
- `colony_population_fragmentation_sweep.toml` holds each team's initial
  cellular mass-energy at exactly 400 while splitting it among 1–16 cells.
- `colony_population_density_sweep.toml` scales population, board area, and
  resource sources together at nearly constant density.
- `colony_population_crowding_sweep.toml` changes population on a fixed board
  and resource field.

All four plans have now been executed against the four maintained control Minds
with three paired seeds and mirrored seats. These 120-episode population
matrices are smoke evidence, not selection-quality estimates; any rule choice
still needs a larger held-out seed suite and confidence intervals.

### Counterfactual deadline adjudication

The four-horizon control matrix has now been executed as 96 mirrored episodes:
23 extermination wins, 51 extermination losses, and 22 deadline draws for the
colony. Generate its immutable counterfactual analysis with:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin adjudication-sensitivity -- \
  sweeps/colony-horizon-v1/control-matrix-horizons.json \
  --policies blob_rl/config/adjudication_sensitivity.toml \
  --output sweeps/colony-horizon-v1/adjudication-sensitivity-v2.json
```

Each policy supplies integer basis-point weights for core, assimilated energy,
gut contents, carried material, and payload escrow, plus an optional minimum
winning margin. Checked `u128` arithmetic makes comparisons exact. The report
binds the exact source-report bytes, exact policy-file bytes, and a canonical
hash of the expanded policy definitions. It retains every deadline score and
normalized seat identity, validates score/outcome and aggregate accounting,
and is published immutably. It remains `local_counterfactual`, never
server-verified.

In this small suite, all 22 deadline draws become colony losses under every
tested policy: body-only, 25/50/100-percent gut, all cell-owned compartments,
and body-only with a ten-unit margin. Body-only deficits range from 11 to 189
mass-energy units (mean 86.4); fully counting gut shifts the mean deficit to
93.1. Thus gut weighting does not affect a label in this sample. The stronger
signal is that the surviving simple/defensive controls hold substantially more
cellular energy than the colony at the deadline.

That result does **not** establish biomass adjudication as desirable. It is
post-hoc scoring of policies that did not know the score while acting, and the
suite has only three maps per horizon. If biomass becomes public, both learned
and hand-authored Minds may adapt toward plant turtling. A rule-selection sweep
must therefore train or optimize against each candidate objective and measure
behavioral changes, not merely relabel fixed trajectories.

### Population-structure smoke results

The three population matrices expose effects that would be confounded in one
generic population sweep:

| Study | Canonical colony W-L-D | Body-only deadline W-L-D | Main observation |
| --- | ---: | ---: | --- |
| Fixed-energy fragmentation | 53-40-27 | 55-65-0 | Splitting 400 initial units changes both coordination and inertia; outcomes are non-monotonic. |
| Constant-density scale | 38-45-37 | 43-76-1 | Larger populations produce many more survivors at the deadline, but most body-score draws favor the controls. |
| Fixed-field crowding | 31-55-34 | 32-88-0 | Colony wins peak around four to eight initial cells and fall again at sixteen. |

Each study contains 24 mirrored episodes per variant. Under fixed-energy
fragmentation, the colony goes from 12-12-0 with one large cell to 13-1-10 with
sixteen small cells, while intermediate variants are not monotone. Under
constant-density scaling, the sixteen-cell case has no canonical extermination
losses (6-0-18), but body-only adjudication resolves those deadline draws as
five wins, twelve losses, and one remaining draw. This is a concrete warning
that “survived until timeout” and “led on biomass” measure different things.
Under fixed-field crowding, the canonical per-variant records are 1-23-0,
6-13-5, 11-9-4, 10-5-9, and 3-5-16 from one through sixteen cells.

Gut treatment is no longer label-neutral in these broader regimes. Fully
counting gut contents reverses four body-only deadline results: two against the
aggressive control in the fragmentation study and two against it in the
sixteen-cell density study. No such reversal appears in the crowding sample.
The raw compartments are therefore doing useful work: a public scoring rule
must state whether undigested energy is ownership, potential, or neither.

These are post-hoc comparisons of fixed policies and only three seeds. They do
not tell us how a Mind trained for biomass scoring would adapt, nor whether the
apparent high-population advantages survive broader maps. The next objective
experiment should keep extermination as an immediate terminal condition, train
separate policies under canonical-draw, body-only, and discounted-gut deadline
rewards, then measure plant occupancy, mobility, attacks, digestion, and wall
construction to detect turtling rather than inferring it from win labels.

### Objective-aware large-field training

`RewardConfig.deadline` now supplies an optional training-only terminal signal
at a canonical simulated-time draw. `none` is the default. The
`weighted_cell_energy` mode uses exact integer basis-point weights for core,
assimilated energy, gut contents, carried material, and payload escrow, plus an
optional mass-energy margin. Configuration validation rejects weights above
10,000 basis points, inert weights under `none`, all-zero weighted policies,
and worlds too large for exact `u128` accumulation. This setting remains in the
training configuration and checkpoint identity; it does not alter
`VictoryConfig`, `EpisodeOutcome`, replay verification, or leaderboard labels.

Three paired smoke plans exercise that boundary with no deadline signal,
body-only scoring, and 50-percent gut scoring:

- `objective_large_draw_sweep.toml`
- `objective_large_body_sweep.toml`
- `objective_large_gut_half_sweep.toml`

Each plan uses three matched seeds on 32, 64, 128, and 256-square fields.
Starting cells and resource sources scale with board area: 16, 64, 256, and
1,024 cells per team. All 36 runs completed and published paired aggregates.
The mean CPU collection throughput across the three objectives was about 4.16,
6.40, 7.36, and 7.63 thousand actions per second respectively. Larger batches
amortize model and dispatch overhead, so actions per second increase even while
the simulated population grows.

The 8,192-action smoke budget is intentionally not learning evidence. The
32-square runs completed about 32.7 episodes and the 64-square runs completed
eight; all were extermination losses. The 128- and 256-square runs completed no
training episode and retained four censored environments per run. Consequently
the three objective treatments produced identical behavior: no canonical
deadline was reached, so no candidate deadline reward could fire. This exposes
an important budget scaling rule: when population grows with area, a fixed
cell-action budget gives each starting cell 128, 32, 8, and 2 action samples
respectively. This does **not** mean scaling the timeout: every variant uses the
same global `sim_time_limit_quanta = 1,048,576`, checked against the one
authoritative simulation clock after every resolver batch. These historical
smokes used the legacy action-budget mode, where `total_timesteps` counts
collected cell transitions. Selection-grade configurations should instead set
`total_simulation_quanta_per_env`: every parallel environment must accrue that
much canonical world time, across episode resets, before training succeeds.
`total_timesteps` then becomes only a hard compute-safety ceiling and exhaustion
is a failed run. This makes exposure independent of population while leaving
the per-episode simulated-time deadline fixed. Metrics report minimum, maximum,
and total environment exposure plus simulation quanta per second.

### Large-world learning curriculum

Small boards are warm-start and debugging environments only. Selection-grade
training begins at 256x256, bridges through 512x512, and finishes with held-out
learning and evaluation at 1024x1024:

| Phase | Board | Cells/team | Plants | Episode horizon | Exposure/env |
|---|---:|---:|---:|---:|---:|
| minimum selection | 256x256 | 512 | 256 | 16,777,216 quanta | 67,108,864 quanta |
| bridge | 512x512 | 2,048 | 1,024 | 33,554,432 quanta | 134,217,728 quanta |
| scale qualification | 1024x1024 | 8,192 | 4,096 | 67,108,864 quanta | 268,435,456 quanta |

The profiles preserve population, scattered-energy, and plant density while
increasing both per-match horizon and total exposure with board width. They are
[`large_world_256.toml`](../blob_rl/config/large_world_256.toml),
[`large_world_512.toml`](../blob_rl/config/large_world_512.toml), and
[`large_world_1024.toml`](../blob_rl/config/large_world_1024.toml). Continue
from the previous phase's verified best checkpoint rather than relearning local
competence at each scale:

```sh
cargo run --release -p blob_rl --bin train -- \
  --config blob_rl/config/large_world_256.toml \
  --load-model /path/to/warm-start-checkpoint/model.mpk
cargo run --release -p blob_rl --bin train -- \
  --config blob_rl/config/large_world_512.toml \
  --load-model /path/to/verified-256-checkpoint/model.mpk
cargo run --release -p blob_rl --bin train -- \
  --config blob_rl/config/large_world_1024.toml \
  --load-model /path/to/verified-512-checkpoint/model.mpk
```

Use the `model.mpk` inside the verified checkpoint named by the preceding
phase's `best.json`.

All three large-world profiles use simulation-time-aware `gamma = 0.9995` and
`gae_lambda = 0.995`; the default 0.99 discount has a half-life of only about
69 nominal time units and would make distant consequences nearly invisible.
Recurrent backpropagation spans 64, 128, and 256 decisions respectively. This
does not by itself solve colony-scale credit assignment: feeding, discovery,
successful signaling, construction, defense, and combat telemetry should become
auxiliary value/prediction targets rather than relying on a rare match-terminal
reward to cross an entire large-world trajectory.

Curriculum and self-play activation use per-environment canonical simulation
time, not sampled action counts. More cells therefore increase work without
prematurely advancing the behavioral curriculum. Evaluation and checkpoint
intervals remain operational PPO-update cadences; the scientific budget and
match timeout are world-clock quantities.

Starting geometry is now an explicit, hash-bound scenario variable. Supported
layouts are uniform world-wide `random`, the former seeded `loose_random`
cluster, dense `block`, contiguous horizontal `line`, spaced `checkerboard`,
and square-perimeter `ring`. The geometric block, line, checkerboard, and ring
layouts receive seeded territorial anchors in opposed team pairs; loose-random
retains its prior seeded sequence. Layout generation either places the exact
requested population or rejects the scenario; it never silently starts with
fewer cells.
The three large-world profiles use `block`, so expansion begins from compact
territories rather than a pre-distributed colony.

[`starting_layout_sweep.toml`](../blob_rl/config/starting_layout_sweep.toml)
defines seven paired controls: one symmetric founder per team standing on a
plant, tight block, tight line, loose checkerboard, loose ring, loose random,
and global random. The single-founder treatment uses `on_all_cells`, which
places an identical plant under every team's starting cell; the curriculum-only
`on_training_cells` mode remains intentionally asymmetric. Minds receive no
layout name, team anchor, absolute coordinate, or other privileged input.
Plant and loose-energy geometry are independent, hash-bound scenario axes.
Each supports historical `uniform` placement, seeded `patches`, axis-aligned
`corridors`, separated `islands`, balanced starting-team `territories`, and an
explicit `favored_territory`. Structured generation places exactly the
requested unique sources or rejects the episode; it never spills excess
sources outside the selected geometry. Plants use an independent seed domain
and are placed first, so changing only loose-energy geometry does not move the
plants. Territorial ownership is derived from the actual anonymous starting
cell coordinates with a toroidal multi-source distance map; no ownership or
layout label enters a Mind input.

[`resource_layout_sweep.toml`](../blob_rl/config/resource_layout_sweep.toml)
defines eight paired controls over the same compact `block` start: historical
uniform placement, plant patches, separated plant islands, loose-energy
patches, loose-energy corridors, balanced plant territories, a deliberately
team-zero-favored plant territory, and crossed islands plus corridors. The
asymmetric treatment is explicit rather than masquerading as a fair matchup.
The 256/512/1024 learning profiles declare `uniform` for both families until
this sweep identifies viable large-world ecologies.

The resource-layout preflight now embeds one complete ecological
characterization per variant before running any baseline matchup. The report
retains movement-weighted plant and major-food distances for every team,
starting food occupancy, exclusively nearest resource shares, ties,
unreachable sources, canonical no-food travel endurance, and collapse flags.
Every characterization is bound to the matrix seed suite, scenario hash,
semantic and compiled rules hashes, and explicit micro-action cap. Dynamic
telemetry separately samples the number and rate of each side's live cells on
plants and on any major food; these fields are also carried into training-sweep
aggregates and paired differences.

[`resource_layout_gates.toml`](../blob_rl/config/resource_layout_gates.toml)
requires every starting cell to reach a plant, every team's p90 plant route to
fit within measured travel endurance, and symmetric variants to stay within
explicit distance, ownership-share, and starting-occupancy gaps. The named
`favored-team-zero-plants` treatment is exempt only from symmetry thresholds;
reachability and endurance still apply. The maintained forager must also show
nonzero dynamic plant occupancy and actual consumed energy in every matchup.

The immutable four-seed v2 preflight evaluated eight layouts, two candidate
proxies, and three opponent proxies (192 episodes) and passed all 99 checks.
Best no-food travel endurance was 44 movement units; the smallest team-p90
plant-distance margin was 26.86 units. Balanced plant territories reduced the
team mean-distance gap to 0.10 units and exclusive plant-share gap to zero. The
deliberately favored treatment produced a 14.58-unit mean-distance gap and a
1.0 exclusive share gap, confirming that the asymmetry detector sees the
intended intervention. Among symmetric variants, plant patches had the largest
p90 distance gap at 5.17 units, while loose corridors had the largest
major-food ownership gap at 0.233. The forager occupied plants in 10.5%–50.8%
of sampled live-cell states and consumed at least 2,944 energy per episode in
every matchup. These are coarse proxy results, not evidence that any geometry
will yield interesting learned strategies at 256–1024 scale.

Reproduce or resume the exact batch with:

```sh
target/release/viability-preflight \
  blob_rl/config/resource_layout_sweep.toml \
  --candidates forager,random \
  --opponents wait,forager,aggressive \
  --gates blob_rl/config/resource_layout_gates.toml \
  --baseline-variant uniform-control \
  --matrix-max-parallel 8 \
  --matrix-max-micro-actions 100000
```

The verified artifacts are
`sweeps/resource-layout-controls-v2/preflight/viability-matrix.json` and
`viability-decision.json`. The earlier v1 diagnostic remains immutable and is
not reused because v2 adds dynamic occupancy telemetry.

Large-world promotion is a separate bounded gate. The 256-square profile is
the minimum selection-grade board: complete maintained-policy and held-out
learned-policy matches run there before a checkpoint can advance. The 512 and
1024 profiles are bridge and final qualification scales. Routine preflight at
those sizes runs ecological characterization rather than a complete matchup
matrix; full matches are reserved for selected promoted checkpoints. This
keeps large-world evidence continuous without multiplying every PPO iteration
by a 67-million-quanta 1024 match.

[`large_world_scale_qualification.toml`](../blob_rl/config/large_world_scale_qualification.toml)
binds the three profile configs, common world seeds, and a 100,000-action host
safety cap. It requires strictly increasing boards beginning at 256, identical
semantic physics, complete plant reachability, and nonnegative p90 travel-
endurance margin for every team. It also enforces exact preservation of cells
per team per tile, plants per tile, loose-energy sources per tile, and canonical
deadline quanta per board-width unit. The latter makes match time depend on
traversal scale rather than population count.

Reproduce the immutable qualification with:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin scale-qualification -- \
  --spec blob_rl/config/large_world_scale_qualification.toml \
  --output sweeps/large-world-scale-qualification-v1.json
```

The three-seed v1 report passed all 24 checks. All profiles have 0.0078125
starting cells per team per tile, 0.00390625 plants per tile, 0.03125 loose-
energy sources per tile, and 65,536 deadline quanta per board-width unit. Every
starting cell could reach a plant. With 43 movement units of measured no-food
endurance, the worst team p90 route retained 29.17 units of margin at 256,
29.27 at 512, and 26.34 at 1024. The maximum team p90 distance disparity was
0.17, 0.24, and 3.17 units respectively; this is evidence of reachable
large-world food, not yet evidence of learned strategic competence.

The first 1024 qualification attempt also exposed an avoidable trusted-host
bottleneck: initial ecology inspection serialized a 164 MiB canonical
checkpoint and then hit the public 64 MiB decode limit. Characterization now
reads an immutable, minimal placement view directly from the in-process
canonical simulation. The public checkpoint decoder, replay verifier, server
submission limits, and Mind ABI are unchanged; only trusted offline diagnostics
bypass the redundant serialization.

Scale promotion uses evidence without redefining checkpoint validity.
[`large_world_promotion_policy.toml`](../blob_rl/config/large_world_promotion_policy.toml)
normally requires relevant ecological checks through the requested target,
held-out policy evaluation, and a passing feeding-competency report. Ecological
evidence can instead be marked `advisory`; a failure then produces an approved
decision with warnings. Required behavioral failures produce a rejected
promotion decision unless `--override-reason` records an explicit rationale.
The override cannot excuse corrupt checkpoint files, invalid hashes, different
semantic physics, an unrelated source scenario, or a target that does not
advance scale.

The promotion decision is a new immutable artifact beside, not inside, the
checkpoint. Rejection therefore does not prevent loading, replaying, exploring,
or continuing training from that checkpoint. It means only that the checkpoint
does not satisfy this named advancement policy. In particular, disabling a
feeding evaluation does not invalidate a checkpoint, but it also cannot satisfy
a policy that explicitly requires feeding evidence.

Create and independently re-verify a decision with:

```sh
cargo run --release -p blob_rl --bin checkpoint-promotion -- \
  --checkpoint checkpoints/source/checkpoint-00000100 \
  --qualification sweeps/large-world-scale-qualification-v1.json \
  --policy blob_rl/config/large_world_promotion_policy.toml \
  --target-profile selection-256 \
  --output promotions/checkpoint-00000100-to-256.json

cargo run --release -p blob_rl --bin checkpoint-promotion-verify -- \
  --decision promotions/checkpoint-00000100-to-256.json
```

An intentional exception adds, for example,
`--override-reason "sparse-food treatment; expected reachability failure"`.
The decision binds the verified checkpoint metadata and its model/optimizer/
resume hashes, the complete scale-qualification file and qualification hash,
the policy bytes, target profile and scenario, every derived check, and the
override text. Re-verification rereads all three source artifacts and rejects
substitution or later modification. If an archive is relocated, the verifier's
optional `--checkpoint`, `--qualification`, and `--policy` arguments accept new
locations while still requiring the originally bound bytes and hashes.

An approved decision is consumed by a separate scale-transition job rather
than by the checkpoint loader. `scale-transition plan` verifies the source
checkpoint, exact model shape and semantic physics, target configuration, and
promotion evidence before atomically publishing a portable job directory. It
copies the decision, qualification, and promotion policy into the job and
writes a fully expanded `target-config.toml` whose artifact path is relative to
the job. The source checkpoint remains external because its optimizer and exact
resume state can be large; `verify` and `run` accept `--source-checkpoint` when
that artifact has been relocated.

Normal scale advancement supplies a non-rejected decision:

```sh
cargo run --release -p blob_rl --bin scale-transition -- plan \
  --source-checkpoint checkpoints/source/checkpoint-00000100 \
  --target-config blob_rl/config/large_world_256.toml \
  --promotion-decision promotions/checkpoint-00000100-to-256.json \
  --output jobs/checkpoint-00000100-to-256
```

An unpromoted treatment must instead carry a nonempty rationale and is labeled
`experimental` in the hash-bound contract. It may optionally include a rejected
decision as evidence of which expectations it intentionally violates:

```sh
cargo run --release -p blob_rl --bin scale-transition -- plan \
  --source-checkpoint checkpoints/source/checkpoint-00000100 \
  --target-config blob_rl/config/large_world_256.toml \
  --experimental-reason "intentional sparse-food treatment" \
  --output jobs/sparse-food-256
```

Launch or audit the job with:

```sh
cargo run --release -p blob_rl --bin scale-transition -- run \
  jobs/checkpoint-00000100-to-256 --threads 16

cargo run --release -p blob_rl --bin scale-transition -- verify \
  jobs/checkpoint-00000100-to-256
```

Before launch, the runner publishes an immutable execution contract binding the
trainer binary hash and size, arguments, thread count, and optional OCI image
digest. The local runner executes the bound binary directly; a remote container
orchestrator is responsible for actually enforcing a supplied image digest.
The trainer imports only the verified full-precision model record, so optimizer,
rollout, RNG, and environment state correctly restart for the new scale. If an
execution is interrupted after publishing a target-scale checkpoint, rerunning
the same job resumes the newest verified update-boundary checkpoint. Completed
jobs have a hash-bound result and cannot be relaunched accidentally.

[`scale_transition_smoke.toml`](../blob_rl/config/scale_transition_smoke.toml)
is a deliberately experimental 32-to-64 pipeline probe. The maintained smoke
successfully imported a checkpoint policy, trained one target-scale update,
published its new checkpoint, and re-verified the complete job. It is not a
curriculum promotion or evidence of learned competence.

On the 2026-08-26 local release build, trusted on-demand integrity with full
host projection sustained about 431k actions/s at 256x256 and 2,048 cells,
422k actions/s at 512x512 and 8,192 cells, and 1.43M actions/s at 1024x1024 and
32,768 cells using the native wait Mind. A sparse 1024x1024/2,048-cell case
sustained 320k actions/s; 76% of elapsed time was prestate materialization.
These are engine capacity bounds, not PPO training throughput: neural inference,
rollout storage, optimization, active metabolism/diffusion, and evaluation will
lower end-to-end learning throughput. The sparse result identifies full-board
projection/cloning as the next large-world engine bottleneck.

A bounded CPU/ndarray smoke of the block-start 1024 profile also initialized the
1,048,576-tile world, collected 14,718 cell transitions, and completed one
four-epoch recurrent PPO update in 0.9 seconds of trainer time (16.0k sampled
actions/s; 1.72 seconds wall time including startup). The smoke used one
environment, 2,048 quanta of exposure, and disabled held-out evaluation and
checkpoint publication. It proves the profile's initialization, collection,
telemetry, and optimization path at 1024 scale; it is not evidence for the cost
of a 67,108,864-quanta match or for learned competence.

The smoke run also found and fixed a PPO singleton-minibatch failure. A final
one-sample minibatch previously removed both tensor axes with unrestricted
`squeeze`; dimension-specific squeezing now preserves the batch axis and has a
regression test. Failed immutable runs were retried through the executor and
all aggregates now verify.

The historical batch predates executable provenance and remains diagnostic: the
three singleton failures were retried after the trainer fix. New executions
publish an immutable `execution-contract.json` before launching trainers. It
binds the manifest, exact trainer-binary SHA-256 and size, trainer arguments,
per-run thread setting, and an optional OCI image digest. Every attempt,
recovery record, result, execution summary, and aggregate must match that
contract, so a mixed-build retry or substituted container fails before launch.

Before a full objective comparison, use a safe-policy curriculum or behavioral
cloning warm start. A randomly initialized policy is exterminated even by the
wait baseline before reaching the large-field deadline, so merely increasing
the action budget would mostly train against early self-destruction rather than
compare deadline incentives. The first full phase should imitate a maintained
forager/colony policy on all four field sizes, then fine-tune separate copies
under the three deadline signals with identical seed and evaluation suites.

Feeding is now treated as a prerequisite rather than left to emerge from a
random-field imitation mixture. Build and gate a candidate feeding-first warm
start with:

```sh
cargo build --release -p blob_rl --no-default-features --features ndarray \
  --bin demonstrations --bin behavior-clone --bin feeding-evaluation \
  --bin feeding-evaluation-merge --bin feeding-teacher-evaluation
bash scripts/build_feeding_warm_start.sh pretrained/feeding-simple-v1
```

The script collects separate on-food and adjacent-food datasets from the
maintained simple Mind. `demonstrations --feeding-stage` uses the same staged
environment constructor as PPO and held-out feeding evaluation; it alters only
initial resource placement and selects the wait prerequisite opponent. The
clone excludes any decision that does not round-trip through the learned action
catalog. It is evaluated on a disjoint fixed seed suite before PPO, and the
result is published as an immutable JSON artifact binding the exact config,
behavior-cloning metadata and model hashes, compiled rules, raw counters,
thresholds, and recomputed verdict. A failed report is useful diagnostic
evidence but is not a checkpoint-promotion substitute. The maintained script
uses `feeding-evaluation --require-pass`, so it preserves that evidence and
returns unsuccessfully rather than silently treating an unqualified model as a
warm start. Per-dataset action-label histograms and the artifact-bound
`--action-balancing {none,family,label}` plus weighting exponent expose catalog
imbalance for controlled follow-up sweeps. Balancing is computed from the exact
post-resampling samples presented in each epoch, not the raw source-corpus
sizes. Behavior-cloning artifacts record actual family presentations, weighted
loss mass, and initial/final action-kind and exact accuracy for every family on
both training and seed-held-out partitions.

The first bounded run found a structural learning seam. With no balancing the
flat action head predicts `Consume` on empty adjacent tiles: on-food evaluation
passes 100%, while adjacent-food movement and intake remain 0%. Full per-label
balancing reverses the failure—adjacent-food success reaches 100%—but causes
cells to move off food and miss the survival gate. Intermediate inverse-
frequency exponents show a sharp transition rather than a stable joint regime.
This evidence motivated replacing the flat action head with an exact
hierarchical decoder: action kind, then a kind-conditional target slot,
kind-conditional effort, and kind-conditional amount. The external catalog, legal-action masks,
demonstrations, replay records, and simulation semantics remain flat and
unchanged; only the policy parameterization and masked probability calculation
are factored. The main action output replaces 264 competing logits with 10 kind,
320 kind-target, 30 kind-effort, and 50 kind-amount logits. This prevents a
maximum Consume amount learned from feeding from also becoming a maximum Attack
payload. The maintained feeding warm-start uses
family-balanced labels because directional variants no longer overwhelm the
single Consume kind. Its default corpus retains about sixteen decisions per
cell; the former two-decision trace taught move -> consume but left later
recurrent states out of distribution. The first bounded hierarchical run on
eight disjoint seeds passed both stages at 100% episode success, with 99.0%
on-food survival, 88.5% adjacent-food survival, and respectively 140.7 and
126.8 consumed energy per starting cell. This qualifies the model as a PPO
warm start, not as a generally capable policy.

Feeding qualification ends on the configured simulation-clock deadline. The
`evaluation_max_episode_len` setting is only a host safety ceiling and defaults
to 4096 frontiers, comfortably above the maintained 65,536-quanta experiment;
the former ceiling of 64 could abort healthy many-cell episodes before world
time elapsed and thereby reintroduced population-dependent timeout behavior.

The initial behavior-cloning pipeline is now available. Demonstrations are
collected by dispatching each maintained teacher through the same anonymous
`ReferenceMindInput` boundary used by native and Wasm play. Each immutable
dataset contains fixed RL observations, legal-action masks, abstract action
labels, source simulation seeds, and an exact-round-trip flag. Collection
divides its cap across the configured seeds rather than allowing one long
episode to consume the dataset. Its manifest binds the teacher, seed
suite, normalized scenario, semantic and compiled rulesets, Mind ABI, source
config bytes, and MessagePack payload hash.

The RL policy now has a configurable recurrent vector (`model.recurrent_size`,
64 values by default). Computation uses f32 while canonical persistence uses a
deterministic signed-16-bit quantization after the bounded `tanh` update; the
default state occupies 136 bytes including its header. Each decision reads only
that cell's versioned canonical Mind private memory and writes its next vector
through the normal memory update. Split children inherit the newly computed
vector; new cells or malformed/version-mismatched memory start at zero. Batched
inference contains no attention, normalization, or cross-row aggregation, so
batching does not create a communication channel. Rejections still advance
private memory under the canonical resolver rule, and checkpoints/replays
include it automatically.

PPO and behavior cloning now use truncated backpropagation through time. PPO
groups completed rollout decisions by host-only `(environment, cell)` keys,
splits reused keys at terminal transitions, and forms graphs of at most
`ppo.recurrent_unroll_steps` decisions (16 by default). Each chunk begins with
the exact canonical private state recorded on-policy; later states are produced
by the current model with the same signed-16-bit canonicalization used at the
Mind boundary. Equal-length chunks are vectorized under the configured
minibatch decision budget and batches are shuffled across lengths.

Behavior cloning groups demonstrations by the host-only
`(source_seed, source_cell)` trajectory key, recurrent history is reconstructed
from zero at the start of each source trajectory, and gradients span up to
`recurrent_unroll_steps` consecutive decisions (16 by default). Chunk-boundary
state is detached and recomputed under the current model at the start of each
epoch. The same signed-16-bit canonicalization used by the Mind ABI is applied
between decisions, with a straight-through gradient estimator inside a chunk.
The trajectory key never enters the Mind observation or model input.

## Ecological seam characterization

Before committing a ruleset to a viability sweep or training run, publish its
deterministic micro-characterization:

```sh
cargo run --release -p blob_rl --bin ecological-characterization -- \
  --config blob_rl/config/default.toml \
  --seeds 700,701,702,703,704,705,706,707 \
  --max-micro-actions 1000 \
  --output characterizations/default-v4.json
```

The schema-v4 default-rules baseline is at
`sweeps/ecological-seams-default-v4.json`; compare future sweep variants
against its identity-bound distributions rather than a single random world.
The older v1/v2 files predate signal characterization, and v3 predates the
mass-shedding protocol; do not mix schemas in one comparison. A baseline
generated from an uncommitted tree carries a
`+dirty` code revision and should be regenerated from the clean release commit
before publication.

The immutable report embeds and hashes the complete rules and scenario, binds
the seed suite, and separately hashes its derived results. It measures:

- passive lifetime with no food;
- exact low/standard/high-effort travel in an empty bidirectional corridor,
  including action effort, inertia, duration, and metabolism;
- shortest move count and configured weighted movement distance from every
  initial team-zero cell to a plant and to any major initial food source across
  the requested generated worlds; and
- authoritative simultaneous full-payload attack volleys for all effort tiers,
  both unguarded and after a standard guard has been established;
- a maximum-bite consume cycle on a fixed synthetic plant, including consume
  effort, windup, digestion completion, gut state, and net assimilated energy;
- the least parent energy that authoritatively completes a minimum viable split
  after split effort and windup metabolism;
- an excavate/deposit round trip at scenario initial energy, including conserved
  material, elapsed time, final energy, and restoration of elevation;
- five signal impulses at one, two, four, eight, and sixteen emission quanta,
  including exact affordability, field strength after commitment and action
  completion, linear time-to-half and time-to-extinction boundaries, compiled
  neighbor visibility, directed observation edges, and a terrain-erasure
  probe;
- a duration-matched Wait control against one, two, four, eight, and sixteen
  quanta of mass shedding, followed by identical standard-effort corridor
  travel. It records post-preparation health/mass, metabolism, first-move
  effort and duration, total movement effort, action cadence, endurance, and
  stationary/travel death times; and
- a bounded plant siege for every locally possible attacker count. The defender
  establishes standard guard first and alternates guard and maximum available
  bites. High-effort attackers repeatedly commit one quarter of their currently
  expendable payload until the defender dies or no attacker can attack. Every
  attacker-count trial uses the scenario's simulation-clock deadline; the
  action count is only a separate host safety cap and never scales the
  scientific horizon with population.

Food distance is a topology measurement: it ignores transient occupancy but
uses the compiled movement target mask, boundary rule, and distance costs.
Combat places the victim on a configured plant, but plants do not heal cells
automatically. A cell must spend a consume action and wait for digestion, so an
instantaneous volley is intentionally unaffected by the plant beneath it. The
sustained siege separately exposes the alternating guard/consume schedule. It
is a deterministic protocol seam, not a claim that its quarter-payload attack
policy or strict alternation is optimal play. Change that protocol explicitly
and version the report before using another tactical assumption.

The report includes explicit collapse indicators for absent or unreachable
plants, median/p90 plant distance beyond the best measured travel endurance,
full-strength high-effort attacks that die during windup, initial cells that
cannot be killed by the maximum simultaneous local volley, failed full-bite
digestion, impossible minimum reproduction, failed terrain round trips, and a
plant defender that outlasts the bounded maximum local siege. Signal-specific
indicators flag disabled or unaffordable deposits, a minimum impulse that
expires before its own action completes, no neighbor-visible signal edges, and
an unavailable terrain-disruption probe. These are diagnostic seams, not
universal declarations that a ruleset is invalid.

For a deliberately coarse pass/fail decision, apply the separately versioned
behavior envelope after characterization:

```sh
cargo run --release -p blob_rl --bin physics-calibration -- \
  --characterization characterizations/default-v4.json \
  --spec blob_rl/config/physics_calibration.toml \
  --output characterizations/default-calibration-v1.json
```

The default envelope is intended to remove obviously bad regimes, not select
an optimum. It requires all sampled starts to reach major food and plants; a
20-time-unit stationary reaction window; at least 12 units of standard-effort
fasting travel; enough endurance for a p90 major-food round trip with 50%
reserve and a p90 plant round trip; positive and at least 25%-efficient net
energy from a fully digested bite; reproduction break-even no higher than
starting energy; an operational high attack; a one-attacker guard advantage;
and a plant defender that requires between two and four sustained attackers.
The initial terrain round trip must also complete. These thresholds live in
TOML precisely so later experiments can compare alternative notions of
"playable" without changing physics or characterization evidence.

The immutable calibration report binds the exact characterization and spec
bytes by SHA-256, repeats the characterization/rules/scenario identities, and
records each observed value, comparator, threshold, rationale, and verdict.
Verification revalidates the characterization and recomputes the entire report
from both bound inputs. A missing measurement fails closed. Calibration should
run before maintained-Mind viability matrices and PPO; passing it does not
demonstrate strategic balance or learnability.

The paired one-factor signal plan is
`blob_rl/config/signal_rule_sweep.toml`. It covers disabled and expensive
emission, persistent/slow/fast decay, current-tile-only and cardinal neighbor
visibility, expensive terrain disruption, and sparse/dense plants. Expand it
and run maintained-Mind controls before committing GPU time to PPO:

```sh
cargo run --release -p blob_rl --bin rules-sweep -- \
  blob_rl/config/signal_rule_sweep.toml
cargo run --release -p blob_rl --bin control-matrix -- \
  sweeps/signal-rules-v2/manifest.json \
  --max-parallel 4 \
  --output sweeps/signal-rules-v2/control-matrix.json \
  --live-output sweeps/signal-rules-v2/control-matrix-live.json
```

Treat `signal-disabled` as the negative control. A useful communication seam
should improve task outcomes over that control without merely maximizing field
energy, active channel-tiles, or persistence. Inspect selection and energy
cost, decay and terrain erasure, pattern diversity, and observable-field total
variation together; high field occupancy with no outcome improvement is signal
saturation, not evidence of communication.

Under the current defaults, a 16-energy bite completes digestion after 5,120
quanta and leaves 110 assimilated energy, minimum split commitment is 14 energy
but 15 is required to survive the windup, and an excavate/deposit cycle leaves
94 of the initial 100 energy. Three of eight possible adjacent attackers kill
the alternating plant defender under the bounded quarter-payload siege. These
figures are seams to interrogate in rules sweeps, not tuning targets.

The default one-quantum signal is affordable but reaches zero during its own
1,024-quanta action. The 2x/4x/8x/16x impulses leave 1/3/7/15 energy at that
frontier and have linear extinction times of 2,048/4,096/8,192/16,384 quanta.
In the synthetic terrain probe, the 16x impulse loses one additional unit while
excavation is pending and the successful edit erases the remaining 14 into
diffuse energy. This makes the minimum learned strength effectively an
ephemeral same-frontier marker under default decay, not a durable neighbor
beacon.

The schema-v4 shedding protocol explains why the cheap hidden-neighbor policy
can behave differently without communication. After the duration-matched Wait,
the default cell has mass 109, spends 65 movement energy, completes 31 standard
moves in 33,472 quanta, and dies at 35,840. A one-quantum emission leaves mass
108 and still completes the same 31 moves on the same schedule and death
frontier, but spends 64 movement energy: the one saved effort exactly offsets
the emitted quantum in this corridor. It does not reduce the first move's
3-energy effort or 1,216-quanta duration. At sixteen quanta the initial mass
falls to 93 and the first move crosses a discrete cost/duration boundary to 2
energy and 1,152 quanta, but endurance falls to 27 moves and travel death to
30,720. These threshold effects are genuine consequences of conserved
mass-energy and integer action rules, not evidence of message utility.

The maintained colony now uses durable semantic strengths of 4x plant, 8x
threat, 4x construction, and 2x frontier quanta. When multiple facts co-occur
it can spend a complete explicit four-channel Signal action; otherwise it
retains its primary action and selects the most urgent exactly affordable
sidecar. Shared Mind legality checks price the full sidecar amount rather than
assuming one quantum, and same-tile retention estimates use the actual emitted
amount.

This stronger heuristic is an integration probe, not a tuned strategy. In the
three-seed v2 matrix it produced 153 explicit actions and spent 8,222 signal
energy in baseline matches, which scored 5-18-1 versus 9-11-4 with signaling
disabled. Current-tile-only and cardinal visibility did not underperform the
fully visible baseline. The result shows that signal opportunity cost and
energy-as-health can dominate an unconditional beacon policy; it does not show
that communication lacks value. See `sweeps/signal-rules-v2/README.md` for the
full table. Subsequent learned comparisons must separately ablate deposited
strength, cadence, and dedicated-action opportunity cost.

The first fixed-rules policy ablation is published under
`sweeps/signal-policy-ablation-v1/`. Five hash-bound candidate identities run
the same 12 seeds in both seats against all four maintained controls: disabled,
one-quantum sidecar, semantic-strength sidecar, cadence-limited semantic
sidecar, and full explicit multichannel. The analyzer validates all five
control matrices and their 96 paired episode keys before publishing
`aggregate.json`.

One-quantum sidecars scored 51-23-22 versus 47-37-12 disabled and improved 16
paired outcomes while worsening 3. Semantic-strength sidecars fell to 39-55-2;
cadencing recovered only to 41-48-7. Adding 407 explicit multichannel actions
fell further to 34-60-2. This supports learned emission selection and strongly
rejects unconditional high-strength beacons. It does not yet prove the cheap
sidecar gain is communication: one-quantum energy transfer also changes health,
mass, inertia, and asynchronous timing. The next required control hides
neighbor signal observations while keeping those deposits physically intact.

That visibility control is published at
`sweeps/signal-visibility-ablation-v1/aggregate-v2.json`. Full, cardinal-only,
and hidden-neighbor observations score 51-23-22, 51-24-21, and 50-25-21. Hiding
every neighboring signal changes only 3 of 96 paired outcomes relative to full
visibility, but the hidden profile still improves 15 and worsens 5 relative to
the no-deposit colony. Most of the one-quantum advantage therefore survives
without neighbor communication. The likely seam is controlled transfer out of
assimilated health/mass, lower inertia, and altered event timing; the current
sample supports at most a small marginal communication effect.

Treat signal use as both a communication decision and a mass-energy action in
RL analysis. This dual use follows the intended physics, but attribution must
compare learned policies under full and hidden-neighbor visibility so mass
shedding is not mislabeled as communication. A separate unreadable energy-dump
action remains an optional configurable rule family if later experiments need
to make those behaviors independently selectable.

The frozen-policy attribution runner now performs that learned comparison
without retraining either arm:

```sh
cargo run -p blob_rl --bin learned-signal-attribution -- \
  artifacts/checkpoint-00000100 \
  --output sweeps/learned-signal-attribution-v1.json \
  --episodes 32 \
  --opponent wait --opponent random --opponent aggressive
```

It integrity-checks and loads one immutable checkpoint, runs identical ordered
opponent/seed pairs with all neighbor signal slots visible and then hidden, and
uses the same deterministic greedy decoder and isolated recurrent cell memory
as held-out checkpoint evaluation. Each variant retains exact learned-side
emissions and channel energy, global decay and terrain erasure, sampled field
energy and observable variation, reward, action count, episode length, and
outcome. The report binds the model, source checkpoint ruleset, scenario,
reward configuration, and both counterfactual semantic and compiled ruleset
hashes. `RewardConfig` contains no direct signal reward term; indirect reward
changes caused by signal physics remain visible rather than being subtracted.

This first evaluator deliberately reports its current evidence boundary: the
learned policy has the fixed team-zero seat used by the RL environment, the
report is locally deterministic rather than server verified, and no replay is
committed. It is therefore suitable for paired attribution during model
development, not yet a leaderboard-grade symmetric match. Seat-swapped learned
policy evaluation and replay-backed server verification are the next steps for
that production boundary.

The policy now chooses payload/amount parameters from five conditional tiers:
minimum, one-eighth, one-quarter, one-half, and the maximum legal commitment.
Split tiers interpolate from the minimum viable child allocation to the maximum
legal allocation. Physical action, action amount, signal pattern, and signal
strength heads are masked and trained as one joint decision without a
Cartesian-product output layer. The 4-bit pattern head represents every subset
of the four anonymous channels. Ordinary actions are restricted to no signal or
one selected channel; explicit Signal may write any nonempty subset. Its five
geometric strength tiers are one, two, four, eight, or sixteen emission quanta
per selected channel, bounded by exact commit affordability. The ABI permits
independent per-channel amounts, while the learned catalog deliberately uses
one strength across the selected pattern. The decoder
still cannot emit arbitrary split markers, exact values between tiers,
arbitrary signal vectors, or a maintained Mind's private-memory format. Consequently colony
demonstrations remain partially abstract rather than exact colony reproduction.
Dataset metadata reports exact round trips and memory replacements, and
`behavior-clone --exact-round-trip-only` can reject all lossy samples when an
exact-only experiment is required.

This changes the neural checkpoint, training-artifact, demonstration, and
behavior-cloning schemas. Pre-factored checkpoints and datasets are rejected
rather than silently interpreted under the new action semantics. Published
artifacts also contain an exact Git revision by default; locally modified trees
are marked with a `+dirty` suffix unless an explicit `BLOB_CODE_REVISION` is
provided by the build environment.

Generate the four density-matched field datasets and pretrain on their union:

```sh
cargo build --release -p blob_rl --bin demonstrations --bin behavior-clone
bash scripts/generate_behavior_curriculum.sh demonstrations/colony-fields-v1
target/release/behavior-clone \
  --config blob_rl/config/objective_large_draw_smoke.toml \
  --dataset demonstrations/colony-fields-v1/field-32 \
  --dataset demonstrations/colony-fields-v1/field-64 \
  --dataset demonstrations/colony-fields-v1/field-128 \
  --dataset demonstrations/colony-fields-v1/field-256 \
  --epochs 10 --minibatch-size 256 \
  --recurrent-unroll-steps 16 \
  --validation-fraction 0.1 --dataset-sampling balanced \
  --output pretrained/colony-fields-v1
```

The output is an immutable `model.mpk` plus `behavior-cloning.json`, which
binds every input dataset and reports initial/final masked training and
validation loss and accuracy. Validation holds out complete source seeds, so
adjacent decisions from a simulation can never leak across the split. Seed
ranking is deterministic and shared by datasets with the same seed suite.
Balanced sampling gives every field/population dataset the same number of
presentations per epoch, rotating and oversampling smaller training partitions
as needed; `proportional` instead presents every training sample once per
epoch. The artifact records the held-out seeds, unique partition sizes, per-
dataset presentations, total optimizer presentations, trajectory counts, and
the recurrent unroll length. The minibatch size is a decision budget: recurrent
chunks of length `L` are batched in groups of approximately
`minibatch_size / L`, and chunk batches are shuffled across lengths.
The command prints the metadata SHA-256. Put the directory and that digest in
the training configuration so sweeps bind and verify the warm start:

```toml
[initial_policy]
directory = "pretrained/colony-fields-v1"
artifact_sha256 = "<printed 64-character digest>"

[initial_policy.qualification]
path = "pretrained/colony-fields-v1/feeding-evaluation.json"
artifact_hash = "<feeding evaluation's 64-character artifact hash>"
```

Fresh PPO runs verify both metadata and model hashes before loading. Exact
checkpoint resume ignores initialization weights and restores the checkpointed
model, while retaining the initial-policy identity in its training config.
When the feeding gate is active, initialization additionally requires a
passing feeding-evaluation artifact. Its hash, behavior-clone metadata hash,
model shape, complete environment, compiled rules, and promotion thresholds
must match. Feeding-evaluation artifact schema 3 includes the rollout-stage
scheduling flag in its canonical identity.

The optional `[feeding_curriculum]` adds two prerequisite stages before normal
competitive rollout assignment. Newly reset environments first place a plant
under each training cell, then place one on a visible, reachable neighboring
vacancy. Both stages use the wait baseline; after the configured per-environment
canonical simulation-time frontiers, normal random resource placement and
opponent assignment resume. Population size therefore cannot compress the
curriculum into a handful of world updates merely by producing more actions at
each frontier.
Stage changes never rewrite an in-flight episode. They do not add observations,
cross-cell state, or privileged policy inputs: every decision still crosses the
ordinary anonymous cell-private Mind boundary and canonical resolver.
`rollout_stages_enabled = false` starts fresh environments directly in the
competitive stage without disabling held-out feeding evaluation when the
combat curriculum is disabled. An enabled combat curriculum supplies its own
cyclic stage assignment. In either case, staging and retention are separate
controls.

At every held-out evaluation boundary, the same greedy policy is separately
tested on fixed seeds for on-food extraction and adjacent-food move-then-extract
behavior. Success uses the resolver's exact `consumed_energy`, not requested
amount, reward, survival alone, or generic action payload. Configurable minimum
episode-success, survival, and intake-per-starting-cell thresholds are recorded
in `feeding-evaluation.csv`. When enabled, all checks must pass before a policy
can be labeled best or promoted into self-play; ordinary periodic checkpoints
remain available for debugging. `metrics.csv` records the rollout curriculum
stage for each update. Exact intake and sampled food occupancy are part of
telemetry schema 8. Detailed episode records also name the host curriculum
stage that produced them; this label is diagnostic metadata and is never
visible through the Mind ABI.

Training artifact schema 28 embeds the complete feeding report in every
evaluated immutable checkpoint. The report binds its seed suite and compiled
ruleset and records raw stage counters, thresholds, derived rates, individual
checks, and the verdict. Artifact verification recomputes all rates and checks
from the raw counters and configured thresholds. A modified verdict, threshold,
rate, seed suite, or ruleset identity is rejected. `best.json` is constructed
from verified checkpoint metadata and repeats the exact report; its verifier
requires every pointer field to match the named checkpoint. Self-play promotion
manifests likewise record the passing report for the newly promoted member.
If the scientific budget completes between periodic evaluation intervals, the
terminal update boundary is evaluated before checkpoint publication. Thus its
feeding verdict and matchup suite are embedded in the immutable checkpoint and
`best.json`, rather than existing only as mutable terminal CSV rows.

[`competitive_transfer.toml`](../blob_rl/config/competitive_transfer.toml) is
the bounded contact-transfer profile and
[`run_competitive_transfer.sh`](../scripts/run_competitive_transfer.sh) supplies
the four qualified-initial-policy identities without baking machine-local paths
into the profile. A seven-update, 65,536-quanta-per-environment characterization
found a sharp retention seam: learning rate `1e-4` preserved 100% immediate
feeding success but reduced on-food and adjacent survival to 47.9% and 49.0%,
so promotion failed. At `1e-5`, the exact same deterministic transfer retained
100% success, 91.7% on-food survival, and 89.6% adjacent survival, passing the
gate. The resulting policy beat Wait and Random in all 8 held-out matches but
drew all 8 short-horizon Aggressive matches, for 66.7% aggregate and 0% worst-
case win rate. This is a safe transfer baseline, not evidence that competitive
learning is mature; the next experiment must improve the Aggressive matchup
without crossing the measured retention boundary.

The transfer profile therefore applies a training-only 10% probability mixture
over the action kinds that are legal at the current decision. The conditional
target, effort, amount, and signal distributions remain learned. Both rollout
collection and PPO optimization use the same mixed distribution, so stored and
recomputed log probabilities agree. Greedy evaluation and exported minds do not
apply the mixture. This gives a feeding-specialized policy a bounded chance to
sample attacks and receive combat credit without changing the deployed Mind ABI
or injecting privileged information.

Combat reward shaping uses `reward.damage_enemy` only for damage the
authoritative resolver reports as actually applied to an opposing cell. It does
not credit requested payload, guard mitigation, overkill, friendly fire, or an
opponent's damage. Kill reward remains separate. This supplies denser combat
credit without altering match physics or exposing team ownership to a Mind.

The first conditional-head characterization rejected two tempting shortcuts.
A 50% uniform kind mixture produced 49 attacks but also many wasteful split and
terrain actions; only one attack landed. Broad imitation of the Aggressive
teacher retained only 27.1% of cells in both feeding stages and was rejected.
With the corrected kind-conditioned amount head, the feeding-only clone passed
at 100% success, 100% on-food survival, and 88.5% adjacent survival. A bounded
competitive transfer then sampled 25 attacks, landed two for six applied damage,
and reduced mean launched payload from roughly 63 to 17.4 mass while retaining
98.96% and 84.38% survival. It still drew every Aggressive evaluation. Extending
exposure fourfold raised applied damage to 19 but reduced adjacent survival to
70.8%, failing the gate without improving the matchup.

The optional `[combat_curriculum]` now makes local combat practice explicit. It
uses a repeating per-environment canonical-time cycle and changes scenarios
only when an episode resets. The maintained profile allocates 32,768 quanta to
on-food retention, 32,768 to adjacent-food retention, 65,536 to direct contact,
65,536 to a multi-cell skirmish, and the remaining 65,536 of a 262,144-quanta
cycle to ordinary competition. Contact episodes contain one cell per side on
adjacent paired tiles. Skirmishes contain four cells per side in two adjacent
opposed lines, so each cell has friendly neighbors and a local enemy front
without an interleaved `A B A B` placement. Both stages remove food, rotate
initial energies 60/100/180, and rotate ordinary aggressive and defensive
baseline Minds. The defensive Mind guards whenever a neighbor is present.
Neither baseline receives team labels or privileged state; all actions still
cross the same cell-private Mind input and canonical resolver. Curriculum stage
assignments, plus the active stage for each in-flight environment, are
checkpointed for exact resume.

Contact rollouts may set a higher legal-action-kind mixture than ordinary
rollouts. The exact mixture is stored per transition and reused by PPO, rather
than inferred later from a global setting. It remains training-only: greedy
evaluation and exported Minds are unchanged. At a 50% contact mixture, one
deterministic cycle produced 34 policy attacks in 28 completed contact episodes,
versus 4 attacks at the prior 10% mixture; one landed for 12 applied damage.
This successfully exposes the combat head but is not yet a competent combat
policy: there were no kills, the Aggressive held-out matchup remained 0%, and
adjacent-food survival fell to 50%, so the promotion gate correctly rejected
the checkpoint. The next experiment should pair a sweep of contact mixture and
cycle share with attack-focused demonstrations or an auxiliary contact
objective, while treating feeding retention as a hard constraint.

The attack-focused demonstration slice now records exact action-family counts
in schema-v9 manifests and can generate a paired-contact matrix with
`demonstrations --contact-energy ... --contact-opponent ...`. The aggressive
teacher produced attack-dense, exact-round-trip subsets across energy 60, 100,
and 180 against aggressive and defensive baselines. Behavior-cloning schema 14
adds two controls required by this experiment: a maximum rare-action weight
ratio, and a verified `--initial-behavior-clone` parent whose metadata hash is
stored in the child configuration. Supplying only a model or only a claimed
parent hash is rejected. Explicit `--dataset-weight` values can now interpolate
between natural-frequency and equal-dataset sampling while holding the total
presentations per epoch fixed.

`contact-evaluation` exercises the Cartesian product of configured contact
energies and opponents on held-out seeds through the ordinary greedy policy and
anonymous Mind boundary. It publishes model/config hashes plus commitments,
successful/frustrated/interrupted attacks, applied damage, kills, and outcomes.
This prevents opponent self-destruction from being misreported as learned
combat. In the first characterization the feeding parent and an eight-epoch
mixed clone both committed zero attacks even when their match outcomes changed.
A contact-only stage reached 260/260 successful attacks, damage in half the
episodes, and four kills, but collapsed adjacent-food survival to 16.7%.
Balanced consolidation retained attacks and damage in every contact variant,
but recovered adjacent survival only to 62.5%; explicit 20% and 10% contact
mixtures reached 72.9% and 69.8%, still below the 80% gate. These are diagnostic
candidates, not promotable warm starts.

`demonstration-overlap` removes the private randomness block before hashing
observations and can quantize normalized features to expose near-collisions.
The current four-dataset corpus has no exact cross-dataset action-family
conflicts. At 1/16 resolution it has 381 shared states and still no conflicts;
at 1/8 resolution, 92 shared states containing 3,035 samples have conflicting
families. Thus strict Mind separation is intact and the labels are not exactly
impossible, but feeding/contact representations overlap coarsely enough that
the current 128 -> recurrent-64 -> 64 shared trunk exhibits severe
interference. The next bounded experiment should compare additional model
capacity or an observation-conditioned mixture-of-experts trunk, then require
both feeding promotion and nonzero held-out contact damage. It must not add team
identity or any privileged colony channel.

That controlled capacity experiment now passes without changing the Mind
boundary. [`competitive_transfer_large.toml`](../blob_rl/config/competitive_transfer_large.toml)
keeps the environment, rules, rewards, curriculum, and evaluation seeds fixed,
but expands the row-separable policy from `128 -> recurrent 64 -> 64` to
`256 -> recurrent 128 -> 128`. Canonical private memory grows from 136 to 264
bytes per cell, still below the unchanged 2,048-byte rule limit. Parameter count
grows from 188,848 to 410,032 (2.17x), so this is a quality/throughput tradeoff,
not a free improvement. The release-mode NdArray capacity benchmark at a
512-cell batch measured 20 complete forward passes at 0.038 seconds for the
small model and 0.052 seconds for the large model, a 1.37x observed slowdown on
this machine. Run the ignored `benchmark_policy_capacity_inference` test on the
actual training/server hardware before setting population limits; parameter
ratio alone overstates the measured batched CPU penalty here.

The large feeding parent required an earlier stopping point: 128 cloning epochs
missed adjacent survival at 78.1%, while 96 epochs passed with 100% success,
85.4% on-food survival, and 80.2% adjacent survival. From that immutable parent,
128 contact-only epochs produced damage in every held-out contact variant. A
32-epoch consolidation with dataset weights `4,4,1,1`—40% on-food, 40%
adjacent-food, and 10% for each contact extreme—then passed both independent
checks: 88.5% on-food survival, 85.4% adjacent survival, 220/220 successful
held-out attacks, and applied damage in all six energy/opponent variants. The
maintained `build_contact_warm_start.sh` now encodes this three-stage lineage
and fails closed unless the final clone passes feeding promotion and deals
held-out contact damage. This makes a mixture-of-experts change unnecessary for
the initial warm start; it remains a possible efficiency experiment if the
2.17x parameter cost becomes material at large population frontiers.

`checkpoint-evaluation` applies both prerequisite suites to any immutable PPO
checkpoint, verifies the checkpoint sidecars first, and publishes a
self-hashed artifact bound to the exact metadata SHA-256, model SHA-256,
update, action count, config, rulesets, and seeds. Contact reports are also
validated for complete energy/opponent coverage, outcome totals, rates, and
aggregate counters. A typical independent check is:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin checkpoint_evaluation -- CHECKPOINT \
  --seeds 900000101,900000201,900000301,900000401,900000501,900000601,900000701,900000801 \
  --output checkpoint-evaluation.json --require-pass
```

This exposed a transfer instability that the terminal aggregate result hid.
At learning rate `1e-5`, survival was marginal after update 1, near 80% after
the on-food phase, and collapsed to 37.5% on-food / 35.4% adjacent by update 6,
before contact training. Contact damage remained active, so contact exploration
was not the cause. Rate `1e-6` passed its configured eight seeds at 82.3% for
both stages but failed the independent qualification suite at 75% adjacent
survival. An unanchored `1e-7` control passed both suites, demonstrating that
small enough updates could retain behavior, but that step size leaves little
room for useful competitive learning.

The maintained large transfer now uses functional initial-policy anchoring.
The verified clone runs once on each ordinary rollout observation and that
cell's private recurrent state. Its detached logits for all six masked policy
heads and its next private-memory vector are stored only in the host rollout
buffer. PPO reuses those targets across optimizer epochs and adds
`initial_policy_anchor_coeff * (policy KL + recurrent-state MSE)` to its loss.
No anchor value enters the Mind input, action ABI, simulation, or another
cell's row. A first implementation recomputed the parent in every minibatch and
fell from roughly 330 to 27 decisions/second; caching targets at collection
restored 295 decisions/second at coefficient `0.25`, an observed 11–17%
overhead relative to recent unanchored runs.

With anchoring, learning rate `1e-5` again survives the bounded transfer. The
calibrated `0.25` run passed the independent qualification suite at 93.8%
on-food and 84.4% adjacent survival, and the configured promotion suite at
87.5% and 88.5%. It retained damage in every contact episode. Fixed evaluation
remained 16 wins, 0 losses, and 8 aggressive timeouts. This establishes robust
retention at a useful optimizer step size, not competitive improvement; the
next target is turning the aggressive timeouts into wins without reopening
either prerequisite regression.

Checkpoint schema 29 makes that two-suite requirement part of training rather
than a manual post-check. When an initial-policy qualification is configured,
the trainer verifies the artifact and retains its exact seed list. At every
evaluation boundary it evaluates the normal configured seeds and, when they
differ, the initial qualification seeds. The reports are labeled `configured`
and `initial_qualification` in `feeding-evaluation.csv` and both complete
reports are embedded in checkpoint metadata, `best.json`, and rollout-pool
manifests. Best-checkpoint and league promotion require every present report to
pass; a valid failing checkpoint remains loadable for diagnosis but cannot be
promoted. Identical seed suites are evaluated and stored only once. The first
bounded schema-29 run reproduced the calibrated results exactly: configured
survival 87.5% / 88.5%, qualification survival 93.8% / 84.4%, active contact,
and fixed W/L/T 16/0/8.

[`competitive_transfer_large_long.toml`](../blob_rl/config/competitive_transfer_large_long.toml)
extends the calibrated profile to four complete 262,144-quanta curriculum
cycles without changing its parent, optimizer, anchor, rules, rewards, or
stage mixture. The characterization stopped at exactly 1,048,576 quanta per
environment after 57 updates and 18,050 learned actions. Every cycle-boundary
checkpoint passed both feeding suites, and the terminal checkpoint reached
100% on-food survival plus 83.3% independent adjacent-food survival. Full
matches nevertheless remained 16 wins, 0 losses, and 8 Aggressive timeouts.
The direct-contact trajectory exposed a selection failure hidden by aggregate
reward: update 16 dealt 1,176 damage and won all eight held-out 100-energy
Aggressive variants, while update 57 dealt 584 damage and won none. The old
best selector chose update 57 because its ordinary-match reward was higher.

Training artifact schema 30 therefore makes contact evidence first-class at
every combat-curriculum evaluation boundary. `contact-evaluation.csv` records
every energy/opponent variant, and the complete validated report is embedded
in checkpoints, `best.json`, rollout-pool manifests, and exact resume state.
An evaluated combat checkpoint cannot be published as best or enter self-play
without resolver-confirmed applied damage. Full-match worst-case and aggregate
win rates remain the primary ordering; when those tie, resolver-attributed
kills, local scenario wins, damage, and damaging-episode coverage are compared
before scalar reward. Kills precede local wins because an aggressive baseline
can exhaust itself without the learned policy receiving kill credit.
This preserves a robust micro-skill instead of allowing easier feeding reward
to mask combat forgetting. It does not change policy observations, private
memory, action resolution, or any other part of the Mind ABI.

Training artifact schema 31 extends that evidence across the distribution gap
between one-on-one contact and full colony matches. Contact-evaluation schema 2
records the stage and cells-per-team for every energy/opponent variant and runs
the same held-out seeds over both direct contact and four-versus-four
skirmishes. A checkpoint is combat-active only if authoritative resolution
reports a successful damaging attack in each stage, with no safety aborts.
One-on-one competence can therefore no longer satisfy the gate while the same
policy is inert in a local group fight. Skirmish rollouts use a 25% legal-kind
exploration mixture versus 50% in direct contact; their ordinary 0.25 functional
anchor is intentionally unchanged after stronger contact anchoring was rejected.
Scenario construction remains host-side setup, and the opposed-lines layout
adds no observation, team marker, shared memory, or resolver exception.
Under the maintained Moore-8 neighborhood, the direct-contact cell initially
observes one anonymous occupant; the four skirmish cells observe 3, 3, 5, and 5
anonymous occupants. This is the intended computational bridge: richer local
contention and support opportunities without any nonlocal colony information.

[`competitive_transfer_large_skirmish.toml`](../blob_rl/config/competitive_transfer_large_skirmish.toml)
then ran two complete schema-31 cycles from a freshly rebuilt qualified parent.
On the exact eight evaluation seeds, the parent dealt 1,176 direct-contact and
584 skirmish damage, but earned no kills or wins. Update 16 retained 992 contact
damage while raising skirmish damage to 1,160 and earning eight resolver-
attributed kills plus eight skirmish wins. Both feeding suites passed: configured
survival was 90.6% on-food and 87.5% adjacent, while independent qualification
survival was 91.7% and 81.2%. Full matches remained 16 wins, 0 losses, and 8
Aggressive timeouts.

The second cycle exposed continued local-skill forgetting. At terminal update
31, contact damage remained 1,016 but skirmish damage fell to 352 and all kills
disappeared. Eight nominal 60-energy Aggressive skirmish wins remained despite
only eight total damage and no kill credit because the baseline exhausted
itself. This evidence motivated the kill-first local tie-break above. The
schema-31 best pointer correctly retained update 16; terminal feeding still
passed at 94.8% / 80.2% independent survival, and full matches were unchanged.
The complete CPU NdArray run collected 9,767 actions in 47.4 seconds; exhaustive
held-out evaluation, rather than PPO collection, dominated wall time. The next
bounded experiment therefore made skirmish-kill retention an explicit
promotion threshold and tested whether a smaller rehearsal interval prevents
the second-cycle collapse before adding more optimizer exposure.

Training artifact schema 32 adds independent minimum contact- and
skirmish-kill thresholds to the combat gate. The maintained combat profiles
require at least one resolver-attributed skirmish kill; nominal wins caused by
opponent self-exhaustion no longer qualify a checkpoint. Checkpoint-evaluation
schema 3 binds those configured floors into immutable evaluation evidence,
best-checkpoint selection, and rollout-pool admission. The zero defaults keep
the mechanism usable for diagnostic profiles without silently changing their
gate.

The same slice corrected a previously unstated scheduler simplification.
Curriculum stages still rotate only at episode boundaries, but every new
episode's scientific-clock deadline is now clipped to the remaining quanta in
its current stage. A long competitive episode can therefore no longer skip the
next cycle's feeding-retention blocks. Final actions may complete slightly past
a deadline under the asynchronous physics; that bounded overshoot remains
visible in telemetry. In the old schema-31 run, only four on-food and four
adjacent-food episodes completed across four environments. With boundary
clipping, the corrected two-cycle schedule completed eight of each and the
four-cycle schedule completed sixteen of each.

The bounded cadence comparison used the same qualified clone, rules, rewards,
seed suites, per-stage total clock exposure, and 524,288-quanta budget per
environment. The corrected two-cycle profile selected update 16: 16 skirmish
kills, 968 skirmish damage, 936 contact damage, and independent feeding
survival of 92.7% on-food / 80.2% adjacent-food. The new
[`competitive_transfer_large_skirmish_frequent.toml`](../blob_rl/config/competitive_transfer_large_skirmish_frequent.toml)
uses four 131,072-quanta cycles and selected terminal update 33: eight skirmish
kills, 416 skirmish damage, 1,016 contact damage, and 100% / 86.5% independent
feeding survival. Both selected policies scored 16 wins, 0 losses, and eight
Aggressive timeouts in the fixed full-match suite; the frequent profile raised
average reward from 128.3 to 152.7. More frequent rehearsal is therefore kept
as a characterized alternative, not promoted to the default: it improved
ecological retention and ordinary-match reward but weakened direct skirmish
force on this single deterministic replicate. The next combat experiment
should add replicate seeds and tune rehearsal mixture or optimizer exposure
against the stage-specific kill/damage frontier rather than infer a cadence
winner from one run.

The follow-up paired sweep
[`combat_curriculum_cadence_sweep.toml`](../blob_rl/config/combat_curriculum_cadence_sweep.toml)
ran all three schedules at training seeds 42, 43, and 44 from the same qualified
clone. The immutable executor completed all nine runs without a retry. The
two-cycle balanced schedule produced a jointly qualifying best checkpoint on
3/3 seeds. Four-cycle balanced qualified on 2/3; four-cycle skirmish-heavy
qualified on 3/3. Conditional on a qualifying best, the two-cycle schedule
averaged 1,269 skirmish damage, 13.3 skirmish kills, and independent feeding
survival of 92.7% / 83.7%. Skirmish-heavy averaged 1,389 damage, 10.7 kills,
and 97.6% / 88.9% survival. Both retained the fixed full-match result of 16
wins, 0 losses, and eight Aggressive timeouts on every seed. With only three
paired replicates and opposing damage/kill movement, neither schedule dominates
the other statistically or behaviorally.

The unqualified four-cycle seed is more diagnostic than its aggregate mean.
At update 16 it delivered 16 resolver-attributed skirmish kills but independent
adjacent-food survival was only 69.8%; at update 32 survival recovered to 83.3%
while skirmish kills fell to zero. More frequent rehearsal therefore does not
by itself make the competencies coexist in one policy. The next optimization
target should be constrained multi-skill retention: archive the ecology/combat
Pareto frontier, measure stage regressions at bounded evaluation intervals, and
rehearse or distill the regressing skill without changing the Mind ABI. Longer
blind PPO exposure is not yet justified.

Training artifact schema 33 adds the prerequisite competency frontier without
weakening any promotion gate. At each held-out evaluation boundary, the host
derives a point from independent on-food/adjacent-food survival and intake,
configured and retention gate verdicts, direct-contact damage/kills,
four-versus-four skirmish damage/kills, and fixed-suite performance. An
evaluated checkpoint is retained when no existing point is at least as good in
every dimension. Newly dominant points remove the checkpoints they dominate;
incomparable ecology and combat specialists coexist. The mutable
`competency-frontier.json` index is capped at 32 entries, but every entry points
to an immutable full checkpoint and is revalidated against its embedded
reports before publication or loading. The complete frontier is also stored in
exact resume state, including source directories when a resumed run writes to a
new artifact root.

This archive is training infrastructure only. Frontier metrics, checkpoint
identity, curriculum stage, and specialist choice are never projected into a
Mind input, shared between cells, or consulted by the resolver. `best.json` and
self-play admission still require the full feeding and combat gates. A schema-33
rerun of four-cycle balanced seed 44 reproduced the interference and retained
two points: update 16 at 81.2% / 69.8% independent survival with 16 skirmish
kills and 1,552 skirmish damage, and update 32 at 87.5% / 83.3% survival with
zero kills and 1,248 damage. Neither was jointly qualified, so no best pointer
was published. The next slice can now load these exact immutable specialists
for stage-local distillation; it must prove that dynamic teacher selection is
checkpointed and behavior-policy-correct before it is enabled in maintained
profiles.

Training artifact schema 34 implements that bounded next step behind the
disabled-by-default `[specialist_distillation]` switch. The host selects an
ecology teacher only from frontier checkpoints that passed both configured and
independent-retention feeding gates, ranking worst-stage survival before
worst-stage intake. It independently selects a combat teacher only from
checkpoints that passed the resolver-attributed combat gate, ranking skirmish
kills, contact kills, and damage. On-food and adjacent-food transitions may use
only the ecology teacher; contact and skirmish transitions may use only the
combat teacher; competitive transitions retain the verified initial-policy
anchor. Before a qualifying specialist exists, the same initial policy is the
fallback on every stage. Teacher outputs are detached functional targets in
the PPO loss: they do not choose rollout actions, change recorded behavior
log-probabilities, enter observations, or alter canonical Mind memory.

The exact resume sidecar stores both chosen checkpoint identities and rejects
any identity absent from its checkpointed frontier. Loading then re-verifies
the immutable teacher artifact before inference. A frontier change forces a
checkpoint before a newly selected teacher can become active, so a teacher is
never loaded from an unpublished mutable model. The maintained profiles
remained unchanged for the first bounded experiment. The opt-in
[`competitive_transfer_large_specialist_distillation.toml`](../blob_rl/config/competitive_transfer_large_specialist_distillation.toml)
initially used coefficients `0.10`/`0.10` on the diagnostic seed-44 cadence as
the bounded experiment for deciding whether specialist rehearsal actually
creates a jointly qualified checkpoint. After the sensitivity sweep described
below, the maintained opt-in values are `0.05`/`0.05`.

The first CPU/NdArray run of that profile did create one. Update 16 exactly
reproduced the archived combat-side failure point (81.2% / 69.8% retention,
16 skirmish kills, 1,552 skirmish damage), activating only the combat teacher.
Update 32 produced the complementary qualified ecology point (100% / 88.5%
retention, zero skirmish kills), while the update-16 checkpoint remained the
combat teacher. The terminal update 33 then passed both feeding gates at 100%
/ 86.5% retention and the combat gate with eight skirmish kills and 1,216
skirmish damage. It published `best.json` after 12,473 sampled actions against
the 524,288-quanta-per-environment target. An exact continuation from the
update-32 checkpoint successfully reloaded the two distinct teacher artifacts
and published into a different output root. This is strong evidence that the
mechanism addresses the observed within-run interference, but one seed is not
enough to promote it into the maintained profile; the next comparison should
repeat paired control/distillation seeds and measure the extra teacher-inference
cost.

That paired comparison is now published under
[`sweeps/specialist-distillation-v1`](../sweeps/specialist-distillation-v1/README.md).
The sweep schema has a dedicated strict `specialist_distillation` override, so
the two variants cannot differ in PPO, rewards, model capacity, curriculum
cadence, physics, or evaluation identity. The execution-result schema also
verifies each run's `best.json` against its immutable checkpoint and aggregates
joint qualification as an exact 0/1 outcome. Seeds 42, 43, and 44 produced 2/3
jointly qualified controls and 3/3 distillation runs. Distillation moved seed 42
qualification from update 33 to update 16, left the already-qualified seed 43
best checkpoint unchanged, and converted the previously failing seed 44 into
the update-33 joint checkpoint described above.

The mechanism is not free. On the local four-thread CPU/NdArray executor,
control averaged 309.9 sampled actions/s and 52,176.6 simulation quanta/s;
distillation averaged 274.8 actions/s and 46,115.3 quanta/s. The paired mean
changes were -35.1 actions/s and -6,061.4 quanta/s, about 11.3% and 11.6% of the
control means. All three action-throughput pairs were slower, but at `n=3` the
Student-t half-width is 36.6 actions/s, and control-first execution can confound
small wall-clock differences. The result supports retaining the opt-in
mechanism and optimizing teacher inference next; it does not yet justify
silently enabling distillation in the maintained profile.

Teacher inference is now partitioned by active stage. The collector assigns
every observation row to exactly one verified teacher: ecology for on-food and
adjacent-food stages once qualified, combat for contact and skirmish once
qualified, and the initial policy for pre-frontier fallbacks and competitive
stages. It gathers compact disjoint observation/memory tensors for those routes
before the ordinary behavior-policy forward pass. The behavior-policy batch,
sampling order, recorded log probabilities, observations, and canonical Mind
memory are unchanged. A backend test compares every policy head and recurrent
target from partitioned and full-batch teacher evaluation.

The three-seed rerun is published under
[`sweeps/specialist-distillation-partitioned-inference-v1`](../sweeps/specialist-distillation-partitioned-inference-v1/README.md).
Every seed reproduced its previous best checkpoint exactly: update/action
identity, retained feeding survival, skirmish kills, and damage all match. Mean
throughput rose from 274.8 to 303.4 actions/s and from 46,115.3 to 50,905.0
simulation quanta/s, recovering about 10.4%. The remaining mean gap from the
earlier control is about 2.1%/2.4%, although the per-seed wall-clock results are
still noisy and were not interleaved. This removes teacher inference as a major
cost without changing the scientific result.

The follow-up coefficient sensitivity sweep is published under
[`sweeps/specialist-distillation-coefficient-v1`](../sweeps/specialist-distillation-coefficient-v1/README.md).
It moves the ecology and combat coefficients together through `0.05`, `0.10`,
and `0.20` while keeping the initial artifact, seeds, curriculum, evaluation,
world-time budget, and partitioned inference implementation fixed. Both `0.05`
and `0.10` jointly qualified 3/3 runs; `0.20` qualified only 2/3. On diagnostic
seed 44, `0.05` produced the joint checkpoint at update 32, one update before
`0.10`, while `0.20` again retained disjoint ecology and combat frontier points
without reconciling them. The lowest tested sufficient coefficient is therefore
`0.05`; stronger rehearsal is not monotonically safer. Reported wall-clock
rates are not used to rank the coefficients because the identical inference
shapes were run in order rather than interleaved and showed substantial host
load noise.

A five-seed confirmation on previously unused training seeds is published under
[`sweeps/specialist-distillation-holdout-v1`](../sweeps/specialist-distillation-holdout-v1/README.md).
It pairs a no-specialist control with coefficient `0.05` on seeds 45–49. Both
arms jointly qualified 3/5 runs. Seeds 45 and 48 qualified identically at the
first evaluation boundary; control seed 47 qualified at update 16 while its
distilled pair qualified only at update 32. Seed 46 never produced a
kill-qualified combat checkpoint, leaving the combat-teacher slot empty. Seed
49 did select an eight-kill combat teacher, and rehearsal raised the later
feeding specialist's skirmish damage, but did not preserve a kill. Across all
eight tested seeds, the observed qualification rates are therefore 5/8 for
control and 6/8 for `0.05`, with the entire difference attributable to seed 44.
The coefficient remains available in the opt-in profile but is not promoted to
the default or assumed to help large-world training. The next bounded work is
to make combat-specialist discovery and the discrete kill behavior more
reliable before paying to scale this mechanism to 256×256 worlds.

Training artifact schema 35 adds one explicit diagnostic for that failure mode:
`combat_precursor_min_skirmish_damage`. When configured, and only when no
checkpoint passes the combat gate, the deterministic frontier selector may use
a checkpoint meeting that resolver-attributed skirmish-damage threshold as a
temporary combat teacher. A gate-qualified combat checkpoint always takes
precedence. The threshold and exact selected identity are stored in immutable
checkpoint metadata and exact resume state; the console labels the fallback as
a precursor. This remains host-side training state and never enters Mind input,
cell memory, action resolution, or physics.

The paired test is published under
[`sweeps/specialist-distillation-precursor-v1`](../sweeps/specialist-distillation-precursor-v1/README.md).
Qualified-only and minimum-one-damage arms both qualified 3/5 on seeds 45–49,
with identical successful checkpoints. Seed 46 exercised the fallback: its
zero-kill update-16 checkpoint became the combat precursor, but later
checkpoints still had zero kills. Update-32 adjacent-food survival improved by
one percentage point while skirmish damage fell from 1,480 to 728; terminal
damage fell from 848 to 376. The precursor thus anchored a locally damaging but
strategically inadequate behavior and moved the policy away from the discrete
kill boundary. The option remains disabled by default as a reproducible
diagnostic; the result rejects threshold tuning or promotion. A better next
experiment is to revisit combat curriculum cadence on the five holdout seeds,
starting with the previously successful two-cycle balanced schedule, so the
learner discovers kill-capable behavior rather than imitating a zero-kill
proxy.

That holdout cadence work is published in two parts. The historical profile
bundle comparison under
[`sweeps/combat-curriculum-cadence-holdout-v2`](../sweeps/combat-curriculum-cadence-holdout-v2/README.md)
qualified 2/5 two-cycle runs and 3/5 four-cycle runs. It also exposed an unstated
coupling: the historical two-cycle profile doubles rollout retention/contact/
skirmish episode limits, while the contact and skirmish limits also governed
held-out combat evaluation. Feeding retention evaluation already used its own
promotion horizon. Seed 46 qualified only in that longer-horizon bundle, so its
apparent rescue could not be credited to cadence. The initial
`v1` launch is retained as a pre-simulation failure because its hash-bound
`/tmp` warm start had expired; `v2` uses a freshly rebuilt and independently
qualified artifact under a new immutable execution contract.
The rebuilt 90 MB warm start is also retained in the ignored local artifact
area at `training-output/warmstarts/skirmish-schema35`, with its metadata and
qualification hashes rechecked after copying, so future sweeps can bind a
durable path instead of `/tmp`.

The corrected comparison under
[`sweeps/combat-curriculum-cadence-isolated-v1`](../sweeps/combat-curriculum-cadence-isolated-v1/README.md)
holds all three stage-episode limits and all total stage exposure fixed. Only
cycle length changes. Four-cycle again qualifies 3/5 while two-cycle qualifies
2/5. Two-cycle rescues seed 49 but loses seeds 45 and 48, and seed 46 fails in
both arms. The earlier seed-46 rescue was therefore a horizon effect. This
rejects two-cycle cadence as the next maintained profile and leaves four-cycle
as the better tested schedule, although 3/5 is still too brittle for
large-world promotion. This motivated making rollout episode horizons and
held-out gate horizons separate explicit settings so neither can silently
stand in for the other.

Training artifact schema 36 implements that separation. Combat curriculum now
has independent `contact_evaluation_sim_time_limit_quanta` and
`skirmish_evaluation_sim_time_limit_quanta` fields alongside the existing
rollout episode limits. `environment_for_stage` uses only rollout limits;
`evaluation_environment_for_stage` replaces the final canonical horizon with
the held-out setting. Feeding retention remains governed independently by
`feeding_curriculum.promotion.evaluation_sim_time_limit_quanta`. Every
maintained combat profile and historical cadence override states the two new
gate horizons explicitly, preserving its prior behavior.

Contact-evaluation artifact schema 3 records both evaluation horizons and
rejects evidence when either differs from the verifying config. Training
checkpoints bind the expanded config under schema 36. A before/after
equivalence run on the preserved warm start, four contact seeds, and the
legacy-equivalent 32,768/65,536 horizons reproduced all 12 raw variants and
every aggregate count/rate exactly: 292 attacks committed, 252 succeeded, 880
damage, and zero kills. The new seam therefore changes provenance and
configurability without changing maintained-profile behavior.

Checkpoint-evaluation artifact schema 4 records its effective combat horizons
separately from the exact training configuration. The
`combat-horizon-evaluation` tool extends that separation into an efficient
matrix: it loads one frozen checkpoint once, evaluates repeatable
`CONTACT:SKIRMISH` pairs, and publishes one immutable artifact binding the
checkpoint metadata/model hashes, training config, seeds, raw reports, and
per-pair decisions. It does not rerun or reinterpret feeding evidence.

The first matrix is published under
[`sweeps/combat-evaluation-horizon-v1`](../sweeps/combat-evaluation-horizon-v1/README.md).
Across update 16 and 32 for five freshly rebuilt schema-36 policies, the half
8,192/16,384 horizon passes 3/10 checkpoints, the maintained 16,384/32,768
horizon passes 5/10, and the doubled 32,768/65,536 horizon still passes 5/10.
The maintained window rescues two delayed skirmish-kill policies; doubling it
adds damage but no qualification. Four update-16 policies pass the maintained
gate while only one update-32 policy does, identifying combat forgetting—not
an evaluation timeout—as the next training problem. The maintained horizon
therefore remains the default.

The collector also drains the final action frontier for simulation-time-limited
runs. Once every environment has reached its scientific clock budget it opens
no new training samples and submits only deterministic waits until actions that
began before the boundary have either reached their next decision or terminated.
The unavoidable final-action overshoot remains visible in the cumulative clock,
while `discarded_tails` is zero and the terminal PPO update is evaluated and
checkpointed. Action-count-limited runs retain the old exact stopping behavior.

Stage-local functional-anchor weights were then tested as a possible remedy for
contact forgetting. The coefficient is stored on each transition, so a mixed
rollout applies its exact contact or ordinary weight during PPO; no stage marker
or coefficient enters a Mind observation. The two-cycle control with the
ordinary `0.25` anchor previously found eight direct-contact wins, eight kills,
and 1,176 damage at update 16. Stronger contact coefficients of `1.0` and `0.5`
both finished with zero wins and zero kills, although they retained 1,064 and
1,008 damage respectively. This rejects stronger parent anchoring as the current
default: it slows combat forgetting, but also suppresses the policy departure
needed to turn damage into kills. The reproducible `0.5` profile is retained as
[`competitive_transfer_large_contact_anchor.toml`](../blob_rl/config/competitive_transfer_large_contact_anchor.toml),
explicitly as a diagnostic rather than a promotion candidate. The next combat
experiment should narrow the distribution gap with intermediate skirmishes,
not tighten this constraint further.

Training artifact schema 37 makes competency timing a first-class scientific
axis. Combat curricula may declare ordered
`competency_evaluation_frontiers_sim_time_quanta_per_cycle`; the trainer runs
the complete fixed, feeding-retention, and combat suites at the first PPO
boundary whose minimum per-environment clock crosses each periodic frontier.
The trigger is based on canonical simulation time, not population-dependent
action count. A rollout that crosses multiple frontiers has only one new model
to inspect, so it records the latest crossed frontier rather than duplicating
identical evidence. The exact requested frontier and observed minimum,
maximum, and total clocks are written to `competency-timeline.csv`.

Every world-time measurement publishes an immutable checkpoint even when it is
dominated and later omitted from the bounded frontier index. Checkpoint
metadata now exposes and hash-verifies the three world-time counters against
the exact resume sidecar; competency-frontier schema 2 repeats and verifies
them against the referenced checkpoint. Update-count evaluation remains
available and composes with the new trigger. World-time-only schedules are
also treated as evaluation-enabled by config validation, snapshot loading,
terminal-suite publication, and sweep aggregation rather than relying on the
old `eval_interval > 0` shortcut.

The paired diagnostic is published under
[`sweeps/combat-retention-stage-capture-v1`](../sweeps/combat-retention-stage-capture-v1/README.md).
It evaluated seeds 47–49 after the contact and skirmish portions of all four
cycles, yielding eight scheduled measurements plus the terminal policy per
run. All 3/3 controls and all 3/3 coefficient-0.05 distillation runs contained
a jointly qualified checkpoint. Control seed 49, which the former sparse
update cadence classified as a failure, qualified near 102,144 and 214,016
minimum simulation quanta. Sparse observation was therefore responsible for
at least one false negative; the maintained workflow should preserve qualified
world-time frontier checkpoints instead of trusting a terminal policy.

The finer trace also rejects stage-local distillation as a reliable retention
solution. Both arms produced eight jointly qualified measurements across 27
measurements. Distillation helped seed 47 reacquire kills in later cycles and
left seed 49 combat-qualified at terminal, but control alone reacquired combat
on seed 48. Terminal combat retention was 0/3 for control and 1/3 for
distillation. Coefficient 0.05 remains opt-in. The better next experiment is a
bounded cross-stage rehearsal buffer or post-update functional-regression
constraint using already qualified combat evidence, because a teacher applied
only on combat transitions cannot stop ecology or competitive updates from
overwriting the skill.

### Asymmetric micro-combat evaluation

The maintained
[`micro_combat_scenarios.toml`](../blob_rl/config/micro_combat_scenarios.toml)
suite isolates the local decisions that colony matches conflate. It contains
paired 1v1 survival against a deterministic aggressor and a private-random
stochastic aggressor, 1v2 and 1v3 survival, a 1v1 defender-breach objective,
and 2v1 elimination. Opposed-line placement makes every 1vN participant a
local neighbor at the start. Scenarios contain no plants or loose food, use
the canonical resolver and simulation clock, and may assign different initial
population and energy to the two teams before authoritative execution begins.
Those asymmetries are host-side initial conditions; they neither alter the
Mind ABI nor expose team identity, objective labels, opponent policy, or
privileged state to a cell.

`StochasticAggressive` is intentionally different from the generic random
baseline. It always closes on an enemy and ordinarily attacks an adjacent
enemy, but uses only the existing private random input block to vary waits and
attack payload. Identical inputs therefore remain deterministic and replayable.
The suite itself is strict, versioned, and semantically hashed; ambiguous
paired layouts, unbounded line populations, invalid energy, unsupported
opponents, duplicate labels, and unknown fields fail closed.

Run a frozen behavior clone with:

```sh
cargo run -p blob_rl --no-default-features --features ndarray \
  --bin micro-combat-evaluation -- \
  --config blob_rl/config/competitive_transfer_large_skirmish_stage_evaluation.toml \
  --scenarios blob_rl/config/micro_combat_scenarios.toml \
  --behavior-clone training-output/warmstarts/skirmish-schema35/behavior-clone \
  --seeds 1000000000,1000000001,1000000002,1000000003 \
  --output training-output/micro-combat/evidence.json
```

Publication is create-only and records the evaluator package version while
binding the exact training config, scenario bytes, scenario semantics,
behavior-clone metadata/model, compiled ruleset, and ordered seeds.
Per-scenario evidence reports outcome, alive-at-end and
scientific-survival rates, objective success, canonical survival time, final
population and assimilated-plus-gut energy, attacks, guards, attributed damage,
mitigation, and kills. Survival and elimination remain separate objectives:
an evasive survivor is not rejected for doing no damage, while a timed-out
turtle is not counted as an offensive success. Safety aborts never count as
scientific survival. `--require-all-objectives` is a coarse CI check, not a
training or promotion threshold.

The training integration is opt-in under
`[combat_curriculum.micro_combat]`. Its suite is stored inline in the training
configuration, even when it duplicates the standalone evaluation file, so an
exact checkpoint never depends on mutable external scenario bytes. Paired
1v1 scenarios rotate in declared order inside contact blocks; opposed-line
1vN scenarios rotate independently inside skirmish blocks. Each stage-local
round robin differs by at most one assignment, advances only when a new
episode is created, and remains capped by the current canonical-time stage
boundary. The two rotation cursors and every in-flight scenario index are
stored in exact-resume state. Restored environments reconstruct the same
asymmetric reset contract before importing canonical state.

Objective labels remain host-only: they change neither observations nor
physics and do not select a policy head. Ordinary rewards still arise from
survival, damage, kills, and outcomes. Telemetry labels the assigned scenario
for analysis, while the Mind sees only its own energy, local anonymous
neighbors, terrain/resources, private memory, and private randomness.

Held-out evaluation independently aggregates survival-objective and
elimination-objective success. Both configured minimum rates must pass, with
no safety aborts, before a checkpoint can become best or enter self-play; a
missing report fails closed. Best-policy tie-breaking ranks elimination rate
then survival rate after fixed-match and legacy contact evidence. Training
artifact schema 38, checkpoint-evaluation schema 5, and competency-frontier
schema 3 bind the report, exact suite, thresholds, rotation state, and the two
rates. The frontier's combat qualification is now the conjunction of legacy
contact/skirmish activity and the micro-combat gates.

The maintained
[`micro_combat_curriculum_256.toml`](../blob_rl/config/micro_combat_curriculum_256.toml)
profile embeds the exact standalone six-scenario suite and returns to a
256×256 competitive field outside the local rehearsal blocks. Launch it with
the ordinary trainer:

```sh
cargo run --release -p blob_rl --bin train -- \
  --config blob_rl/config/micro_combat_curriculum_256.toml
```

The initial thresholds are 75% survival and 25% elimination. They are explicit
calibration targets, not established optima. Held-out evaluation/gating and
rollout rehearsal have separate switches: `enabled = true` retains the exact
suite and gates, while `rollout_enabled` controls whether named scenarios enter
the contact/skirmish rollout rotation. Rehearsal is invalid unless evaluation
is enabled. This separation makes the no-rehearsal control scientifically
useful rather than exempting it from the test it is meant to compare.

The executable bounded A/B plan is published at
[`sweeps/micro-combat-ablation-256-v3`](../sweeps/micro-combat-ablation-256-v3/README.md).
It contains three paired seeds and two 256×256 arms. Every environment must
reach 1,048,576 authoritative simulation quanta across four full curriculum
cycles; 20 million actions is only the safety ceiling. Both arms retain the
same suite, thresholds, held-out seeds, rules, reward, model, PPO, population,
and schedule. After normalizing each run's artifact destination, the sole
within-pair intervention is `rollout_enabled`.

The original V1 launch failed before its first update: a 256-cell dense block
contains enclosed interior cells, while adjacent-food retention requires a
visible vacant neighbor for every founder. The failure is retained as durable
provenance. Configuration validation now rejects this combination, V2 uses a
checkerboard assembly in both arms, and a regression test constructs every
curriculum stage for every paired seed before a plan is accepted.

V2 then reached its first update and revealed that micro-combat evidence used
the 7×7 scenario's dimension-sensitive compiled hash where checkpoint metadata
requires the 256×256 base-world identity. The report now binds the base world;
the hashed suite continues to bind the small-world geometries. A regression
test checks this exact maintained configuration before V3 execution.

V3 completed all six bounded runs. Balanced rehearsal did not qualify: final
held-out survival was 4.17% versus 2.08% for control (paired change +2.08
percentage points with a ±8.96-point 95% interval), while both arms had zero
elimination success and zero terminal fixed-suite wins. More importantly,
every terminal held-out micro-combat report recorded zero committed attacks;
five of six terminal policies also failed both feeding tasks. This rejects
longer exposure to the same undifferentiated rehearsal as the immediate next
step. The next experiment should explicitly solve combat-action acquisition
while protecting feeding retention. Full evidence and caveats are recorded in
the V3 sweep README and aggregate.

The combat-action acquisition slice now makes that next experiment explicit.
`attack_action_kind_exploration_floor` is applied only to cells assigned to a
named micro-combat training scenario, and only when Attack is legal. The host
first forms the ordinary learned-plus-uniform legal-kind distribution, then
mixes the configured mass directly into Attack. Each transition records both
mixture weights and PPO reconstructs the exact behavior probability. This is
training scaffolding: it is absent from ordinary competitive rollouts,
held-out evaluation, Mind observations, and canonical physics.

Held-out qualification additionally requires
`min_attack_commitments_per_episode`, aggregated over the complete named suite.
The value is published in the competency frontier, timeline, immutable run
result, and paired sweep summaries. Thus passive survival cannot masquerade as
combat acquisition. Feeding remains a separate hard promotion condition and
the existing on-food/adjacent-food rehearsal blocks remain in the schedule;
the acquisition mechanism does not claim to prevent weight-level forgetting,
so the A/B must still demonstrate retained feeding.

The executable paired design is
[`micro_combat_action_acquisition_256.toml`](../blob_rl/config/micro_combat_action_acquisition_256.toml).
Both three-seed arms receive identical scenario rehearsal, rules, schedules,
rewards, and a held-out floor of 0.25 committed attacks per scenario episode.
The sole intervention is a 0.50 direct Attack mixture versus 0.0 in control.
This tests action acquisition before spending another full budget on reward or
physics tuning.

That A/B completed all six bounded-world-time runs. The 50% Attack mixture
substantially increased attacks inside sampled training scenarios (233 commits,
36 successes, 594 applied damage, and 5 kills versus 62, 8, 95, and 0), proving
that the behavior-policy intervention was active. It nevertheless produced
zero greedy held-out attack commitments at every evaluation boundary in both
arms, zero elimination success, zero terminal survival, and zero joint
qualifications. Best survival changed by only +1.04 percentage points with a
±4.48-point paired 95% interval. No treatment checkpoint met feeding
qualification either.

This rejects additional kind-only forced exploration as the next step. The
fixed Attack component can dominate sampled attacks while contributing only a
small learned-policy score gradient, and it does not teach target, effort, or
timing. The next acquisition path is an attack-dense supervised combat
specialist or multi-task warm start, followed by PPO with explicit feeding
retention and held-out greedy gates. The immutable aggregate and the sampled
training-mechanism caveat are recorded in the sweep README.

The supervised acquisition follow-up adds a prospective-teacher gate before
collecting labels. `feeding-teacher-evaluation` drives a maintained Mind
through independently prepared anonymous inputs and the ordinary resolver,
then applies the same full-population feeding thresholds used for learned
policies. A collision-aware Simple variant changes only equal-energy movement
ties: it maps cell-private random input to contiguous uniform target ranges,
so nearby cells disperse without shared identity/state and the mapping remains
learnable by the neural policy. On the eight-seed 256×256 checkerboard suite,
the teacher passes with 98.1% on-food survival and 93.4% adjacent-food
survival.

The initial sequential clone exposed a sampling error rather than a capacity
limit. Its feeding corpora contained 65,536 exact labels while the two exact
contact corpora contributed only 544; the former 4:1 feeding weights reduced
combat below one percent of effective examples, and a feeding-only intermediate
erased attacks before consolidation. A bounded four-epoch candidate recovered
both skills: it passes the merged eight-seed 256 feeding gate at 100% on-food
and 91.4% adjacent survival, attacks and damages in 100% of the 12
contact/skirmish variants (282 total damage), and makes 1.833 greedy attacks per
micro scenario episode. It still records no kills in the short elimination
probes, so it is an action-acquisition/retention warm start, not yet an
elimination-qualified combat policy.

The first multi-layout build then exposed a second weighting mistake. Explicit
behavior-cloning weights are direct dataset mixture shares; they do not
multiply each shard's sample count. Repeating weight 1 for ten feeding shards
and 64 for each contact shard therefore allocated 92.7% of every epoch to
combat, leaving only 419 presentations per feeding layout. That candidate
failed the ordinary adjacent-food gate at 78.9% survival. The corrected default
uses ten feeding shares of 1 and two contact shares of 2.5, allocating two
thirds of each epoch to feeding and one third to combat independent of shard
length.

That corrected global mixture still overvalues the easy stationary state: five
on-food shards plus post-arrival labels make exact Consume accuracy dominate,
while one wrong initial target can strand a cell. The maintained default keeps
the same two-thirds feeding/one-third combat split but assigns each layout 0.25
on-food and 1.75 adjacent-food shares. This makes the scarce, decision-critical
Move/target transition the primary feeding rehearsal without removing the
stationary Consume check.

Large feeding qualification is now resumable. Each seed publishes a complete
hash-bound shard; `feeding-evaluation-merge` validates common config, model,
rules, and unique seeds, sums only raw counters, and recomputes all rates,
gates, and the aggregate artifact hash. This also fixed an older validation
gap where `report_from_metrics` accepted caller-supplied derived rates instead
of deriving them from counters.

Cross-layout qualification now covers line, checkerboard, ring, loose-random,
and random founder geometries. `feeding-layout-evaluation` publishes the full
layout/seed Cartesian matrix, and its merger rejects missing, duplicate, or
mixed-policy shards. The original skill-balanced clone passes checkerboard but
fails the other four layouts on held-out seed 930000101: adjacent-food survival
is 75.0% for line, 76.6% for ring, 65.6% for loose-random, and 73.4% for random.
This confirms that repeated checkerboard seeds were measuring a specialized
policy rather than general feeding competence.

The first layout sweep also found a curriculum defect: adjacent plants were
chosen independently, so multiple cells could select the same vacant tile and
later writes silently replaced earlier plants. Resource initialization now
uses deterministic augmenting-path matching to assign one distinct visible,
reachable vacancy to every founder. A 256-cell regression covers all five
layouts. With corrected placement, the collision-aware teacher passes every
layout on the held-out seed, with adjacent survival from 91.8% to 99.6%.
Feeding-promotion schema 3 and demonstration schema 10 prevent pre-fix evidence
from being mistaken for post-fix evidence.

A later exact-seed schema-2 comparison confirms that this remains a training
seam after canonical private randomness was wired through learned rollouts.
Across two fresh seeds per layout, the collision-aware teacher passes all ten
trials. Mean teacher versus learned adjacent survival is 91.0% versus 74.8%
for line, 91.0% versus 76.0% for ring, 93.4% versus 80.7% for checkerboard,
93.2% versus 83.0% for loose-random, and 99.2% versus 98.2% for random.
Transition diagnostics show that off-plant cells with visible food choose
`Consume` 76–91% of the time; correctly targeted Moves occupy only 8–23% of
those decisions. Dense-layout contention then amplifies the error. This evidence
initially prioritized an adjacent-state action-kind correction before more
movement-effort or global mixture tuning.

The matched correction preflight instead found a runtime routing defect:
diffuse field energy, which cells cannot consume directly, selected the
foraging expert. The fixed anonymous-observation router uses current plant
capacity or loose energy as its consumable-food predicate; an exhausted plant
still routes to foraging, while an ordinary diffuse-only tile routes to
exploration. Under the corrected runtime the frozen clone makes no off-plant
`Consume` errors on line or ring and retains every cell reaching a plant.

The remaining failure is target coordination. Correct Moves are `Contested`
80.6% of the time on line and 85.4% on checkerboard. Exact-input correction
corpora show hundreds of target-only disagreements per dense layout and almost
no action-kind errors. The current A/B therefore keeps full recurrent prefixes,
changes only same-effort Move targets selected by the collision-aware teacher,
and updates only the target-query head. Control and treatment datasets are
independently hash-bound and must pass `feeding-correction-pair-verify` before
training. Collection seeds and final five-layout qualification seeds are
disjoint.

The target-query-only mechanism gate subsequently failed. At 22, 44, and 88
updates, treatment held-out Move exact accuracy stayed at 63.2%; at 256, 512,
and 1,024 updates it declined to 62.1%, 61.7%, and 60.0%, even as
cross-entropy fell. Matched controls remained 100% exact against the inherited
targets. This rules out a mere dose shortage. The teacher maps cell-private
randomness across equal-energy target slots, but the inherited shared
representation does not make that variation recoverable by changing only the
query head. A future target correction needs a zero-initialized,
permutation-equivariant adapter that consumes the raw private-random block and
each raw slot directly; its exact-input crossover remains a prerequisite for
ecological evaluation.

The warm-start builder therefore gates the teacher across all five layouts and
collects both on-food and adjacent-food demonstrations for each. The original
feeding sample budget is divided across layouts so broadening geometry does not
multiply the feeding corpus or its training cost. Demonstration manifests bind
both the unmodified source TOML and the validated effective configuration after
stage/layout overrides. The learned clone must then pass both the aggregate
feeding gate and the complete cross-layout gate before contact and micro-combat
qualification can promote it.

Per-layout sample division also constrains seed count. A 6,553-sample shard
spread over 32 seeds gives each seed only 205 labels, less than one 256-cell
decision frontier; adjacent-food corpora then contain initial Move labels but
no post-arrival Consume labels. The maintained build uses eight feeding seeds
per layout, retaining full-frontier coverage across several decisions while
keeping the 32-seed contact suite. Feeding evaluation remains on separate
held-out seeds.

The resulting bounded trials reject further blind mixture tuning. The first
multi-layout candidate accidentally devoted 92.7% of its epoch to combat and
failed aggregate adjacent feeding at 78.9%. Correct direct weights recovered
the ordinary checkerboard aggregate at 81.4%, but line, loose-random, and random
remained below the per-layout gate. Adding temporal post-arrival labels reduced
aggregate survival to 59.4% because easy repeated Consume decisions dominated
target accuracy. An adjacency-heavy continuation then passed checkerboard,
ring, loose-random, and random pilots (83.6%, 87.5%, 88.3%, and 99.2%) but line
remained at 75.0%; it also forgot combat completely, with zero attacks across
48 contact episodes and zero greedy attacks in the micro-combat suite.

The collision-aware teacher had one additional representability defect:
energy-tie choices used contiguous random quantiles, but no-food fallback used
integer modulo over the private random word. Because the policy observes
normalized random bytes, modulo creates a discontinuous target function. The
teacher now uses uniform contiguous quantiles in both paths and still passes
the complete two-seed layout gate (89.5%--98.8% adjacent survival). A fresh
clone nevertheless reached only 61.5% aggregate adjacent survival. Combined
with the combat-forgetting result, this is evidence that the flat shared MLP
and scalar mixture are the next constraint. The next model slice should use a
shared per-slot encoder/target scorer (or local attention) and keep explicit
feeding and combat heads or anchors, then rerun these exact immutable gates.

The shared per-slot architecture is now implemented. Every local slot passes
through the same 33-feature encoder; max-pooled slot features join the global
header before the cell-private recurrent transition, and each action kind emits
a query whose dot product with each slot embedding produces its 32 target
logits. Permuting complete slot records therefore leaves global outputs
unchanged and permutes every target head identically, which is enforced by a
model test. No operation combines batch rows, cells, teams, or hidden identity.
The default model shrinks from 188,848 to 84,848 parameters, although its
structured CPU matrix operations are slower per training epoch.

Because the model record changed, behavior-cloning schema 15 and training
artifact schema 41 reject flat-model checkpoints. Behavior-cloning schema 16
then adds effective-mixture family balancing and family-resolved diagnostics.
The warm-start builder now
starts without a parent by default and accepts an explicit compatible parent
only when supplied. A first eight-epoch scratch run was visibly undertrained
(72.7% exact validation accuracy and zero feeding survival). After 24 more
epochs it reached 90.3% exact validation accuracy: loose-random and random
adjacent survival improved to 90.6% and 100%, but line, checkerboard, and ring
remained 75.0%, 72.3%, and 79.7%. It also made zero attacks in both contact and
micro-combat evaluation. The architecture corrects slot generalization but
does not make exact-label likelihood equivalent to long-horizon competence.

The first competency-aware supervision pass found that the former family
balancer used raw corpus counts even when explicit dataset weights resampled a
very different mixture. In the maintained 12-corpus build, raw counts contained
only 224 Attack labels versus 44,559 Consume labels, while one effective epoch
presented about 10,919 Attack, 29,674 Consume, and 17,213 Move labels. The
corrected trainer derives weights from the actual deterministic epoch. With
full family balancing, its recorded cumulative weighted mass agrees across the
three families to within floating-point accumulation error.

That correction makes the interference visible but does not solve it. At eight
epochs the phase-balanced scratch model had 50% held-out Attack kind/exact
accuracy and, unlike the prior zero-attack model at the same budget, attacked
in all 48 contact episodes: 1,700 attempts, 1,312 damage, 12 kills, and 4.5
greedy micro attacks per scenario episode. It was still undertrained for
feeding, at 0% survival. After continuing to 32 total epochs, Consume reached
100%, Move reached 99.7% kind and 60.1% exact accuracy, but Attack collapsed to
0%. Rollouts matched: loose-random and random feeding passed while the three
dense layouts remained at 72.3%--75.0%, and contact/micro evaluation recorded
zero attacks. Equal loss mass therefore cannot prevent representational or
gradient interference in the shared action-kind head.

The phase-expert model slice is now implemented. Feeding and combat have
separate action-kind heads, while downstream target, effort, amount, signal,
value, and recurrent paths remain shared. Every demonstration dataset receives
an explicit feeding or combat phase in artifact-bound CLI order. Recurrent
chunks retain that label across shuffling and truncation; action-kind loss is
routed only to the assigned expert. Phase loss is independently balanced from
the exact post-resampling epoch, and artifacts report combined, routed-expert,
gate, and exact accuracy by action family plus phase-level gate accuracy.

Inference receives no phase label. A local gate mixes the two experts in
probability space from the same anonymous observation. The first shared-trunk
gate trial proved the experts worked—held-out Attack remained 100% in the
combat expert—but the gate classified every Attack sample as feeding after 32
epochs. A 4x auxiliary gate continuation recovered only 50% held-out Attack
routing and four attacks across 48 contact episodes. The final design gives the
gate its own shared-per-slot encoder and pooled feed-forward path, still without
batch-row, cell, team, or identity aggregation. It does not enlarge per-cell
memory or alter the Mind ABI. The default model is 96,444 parameters, versus
84,848 for the single-head slot model, but the second slot pass materially
increases CPU cloning and inference cost.

At 32 epochs the dedicated gate achieves held-out coexistence: Consume is 99.5%
combined/expert/exact, Move is 96.1% combined and 59.0% exact, Attack is 100%
combined/expert/gate and 75.0% exact, and phase gating is 99.9% feeding / 100%
combat. Actual rollouts nevertheless reject the candidate. Contact and micro
suites record zero attacks; dense-layout adjacent survival is 50.0% line,
66.8% checkerboard, and 68.4% ring, while loose-random and random pass at 84.0%
and 99.2%. The gap is now demonstrably distribution shift between held-out
teacher trajectories and policy-induced states, not shared-head forgetting.
The next slice should use direct rollout fine-tuning and/or deterministic
on-policy correction-label collection from failed states, retaining the expert
and rollout gates. This candidate is evidence, not a promotable warm start.

Behavior-cloning schema 17 and training-artifact schema 42 bind the expert
record, dataset phases, balanced phase loss, and diagnostics. Checkpoint-
evaluation schema 7 binds the split
evaluation/rehearsal and exact behavior-policy contracts. Sweep-execution
schema 10 reconstructs every complete micro-combat evaluation
boundary and binds terminal/best survival and elimination, joint-gate reach,
terminal/best attack commitment rate, and normalized actions to first
qualification into each immutable result and paired aggregate. Competency
frontier schema 4 preserves the same acquisition evidence. A run that never
qualifies is right-censored at its full budget rather than disappearing from
the learning-speed statistic. Trainer
metrics also report peak host resident-set bytes; aggregation takes the maximum
across resumed attempts. This intentionally excludes GPU device memory, which
must be characterized separately for deployment sizing. Throughput remains
reported as actions and authoritative simulation quanta per second.

The first rollout-distribution correction slice now uses deterministic
dataset aggregation rather than extracting isolated failure frames. A verified
greedy clone executes each contact episode, while the maintained aggressive
teacher labels every state the clone actually visits. Each cell's trajectory
is retained from its beginning, including intervening agreements, so recurrent
unrolls never splice across omitted decisions or initialize a mid-episode
hidden state as zero. Collection retains the canonical per-cell invocation
block used by held-out and deployed policy
execution. Greedy action selection remains deterministic for a fixed input and
seed. This remains a local Mind observation; the teacher and policy artifact
identity are host-side supervision metadata. Historical schema-16 correction
datasets retain their declared zero-randomness identity and remain readable,
but new correction collection publishes schema 18. Schema 18 additionally
binds the actual teacher action beside the frozen-policy action and can publish
a matched adjacent-empty-tile `Consume` control/treatment without dropping the
complete recurrent prefix.

Demonstration schema 11 distinguishes ordinary teacher rollouts from greedy
policy corrections and binds both the behavior-clone metadata and model hashes.
It also records total policy disagreements and the subset where the teacher
attacked but the rollout policy did not. Continuous teacher parameters that do
not exactly occur in the learned policy's discrete catalog are explicitly
projected to the nearest legal policy choice; the manifest records the number
of projected labels, and the stored label then round-trips exactly. This avoids
silently dropping the exact attack corrections that motivated collection.
Behavior-cloning schema 17 already
binds each dataset's complete manifest hash, so no clone-format change is
needed to carry the new collection identity. When correction is explicitly
enabled, the 256-cell warm-start builder produces a bootstrap clone, collects
separate 60-energy/aggressive and 180-energy/defensive correction trajectories
on disjoint seeds, and runs a bounded continuation with both the original
rehearsal corpus and corrections.
The immutable feeding, contact, and micro-combat gates still decide whether the
result is promotable; correction labels alone are not evidence of competence.

A bounded first correction run exposed two separate failure modes. Without
explicit projection, the 60-energy policy rollout produced 120 missed teacher
attacks, but exact-round-trip filtering removed every one because their payload
values fell between catalog tiers. Projected correction retains all 120. A
four-epoch, `1e-4` continuation then recovered 100% routed-expert Attack but
erased held-out Consume, so an enabled correction experiment defaults to one
epoch at `1e-5` and half the former correction share. That
conservative candidate retained 97.0% held-out Consume and 100% routed-expert
Attack, but its gate selected combat for only 66.7% of held-out Attack labels.
It committed 16 attacks across 48 contact episodes and about 0.021 attacks per
micro-combat episode, with zero elimination successes. It is rejected.

The correction data also reveals the next modeling seam: a dataset-wide
`combat` label is host scenario metadata, not necessarily inferable from a
later anonymous state after the policy has moved away from contact. Such Move
states can be locally indistinguishable from feeding Move states, forcing the
gate to learn contradictory labels. The next slice should route supervision
per decision using a locally derivable interaction predicate or the projected
action family, permit phase changes inside a recurrent trajectory, and verify
that identical observations cannot receive conflicting gate labels. Only then
should another multi-round correction schedule be tuned.

That routing correction is now implemented in behavior-cloning schema 18.
Attack and Guard decisions route to the combat expert; every other physical
action routes to the general/feeding expert. The label is recomputed for every
decision, so a single cell can Move, Attack, and Move again without assigning
the entire recurrent trajectory to a host scenario phase. Dataset manifests
remain responsible only for provenance and sampling weight. Artifact dataset
partitions now report eligible, training, and validation route counts instead
of one phase label. Before training, a domain-separated digest of each exact
bitwise observation fails closed if the same anonymous gate input has both
routes anywhere in the eligible corpus. The gate still receives neither cell
identity, team state,
host scenario metadata, nor recurrent private memory; Mind ABI separation is
unchanged.

On the reduced 1,638-sample-per-layout corpus, a 32-epoch scratch baseline
without policy-correction shards reached 76.0% held-out Consume, 99.6% Move,
100% Attack, 98.4% general-route gating, and 96.9% combat-route gating. More
importantly, action acquisition transferred to rollouts: all 48 contact
episodes attacked and dealt damage, totaling 1,332 committed attacks, 2,716
damage, and 28 kills. The micro suite averaged 8.67 attacks per episode. This
is not combat competence: one-versus-three survival was 0%, both elimination
objectives were 0%, and most contact variants still lost. The same routing run
with correction shards retained Attack but collapsed Consume, confirming that
the correction mixture—not local routing alone—is the immediate interference
source. Consequently, the maintained builder defaults correction continuation
off; `BLOB_POLICY_CORRECTION_ENABLED=1` makes it an explicit experiment. The
next training slice should improve combat target/effort sequencing and revisit
correction sampling without sacrificing the expanded full-size feeding corpus.

Demonstration schema 12 now connects that requirement to the observation-valid
combat oracle. `observation-policy-demonstrations` accepts an immutable search
report, replays its synthesized policy through the ordinary anonymous Mind-input
boundary, and publishes the visited observations plus complete target, effort,
amount, and signal choices in the existing recurrent behavior-cloning format.
The generator rechecks the policy hash, scenario and objective, every synthesis
continuation, complete policy coverage, and objective success. Only the report's
synthesis seeds are eligible; its sealed holdout seeds cannot enter the dataset.
The oracle's private-memory bytes are used to replay its table depth but are not
supervised as neural memory—the clone learns the sequence through its own
cell-private recurrent state.

The maintained 16-seed searching-pursuer control produced 112 exact samples:
64 Attack, 32 Move, and 16 Wait decisions. A one-epoch CPU/NdArray integration
smoke split them into 84 training and 28 seed-held-out validation samples and
published a normal behavior-cloning artifact. This is a transport and
provenance result, not learned competence; one epoch retained 0% exact action
accuracy. `build_combat_warm_start_256.sh` accepts an opt-in
`BLOB_OBSERVATION_POLICY_REPORT` and appends the resulting dataset with a
separate fixed-total
`BLOB_OBSERVATION_POLICY_CONSOLIDATION_WEIGHTS`, allowing a paired experiment
to reallocate combat supervision to oracle sequencing without changing the
two-thirds feeding share or the default builder.

```sh
cargo run -p blob_rl --no-default-features --features ndarray \
  --bin observation-policy-demonstrations -- \
  --config blob_rl/config/micro_combat_curriculum_256.toml \
  --scenarios blob_rl/config/micro_combat_pursuit_scenarios.toml \
  --policy-report training-output/micro-combat-observation-policy-cost-dense-metabolism-emission-1v1-searching-pursuer-h16-search16.json \
  --output training-output/demonstrations/searching-pursuer-oracle
```

The first paired experiment is published under
[`sweeps/observation-policy-warm-start-v1`](../sweeps/observation-policy-warm-start-v1/README.md).
The treatment kept 15 total mixture shares but moved half of the five-share
combat budget to the 112-sample searching-pursuer oracle. Contact/skirmish
attacks fell from 1,332 to 544 while damage rose from 2,716 to 5,952 and kills
from 28 to 56, direct evidence that the oracle teaches more effective target
and effort sequences. The candidate is nevertheless rejected: asymmetric
micro survival fell from 22/32 to 0/32, elimination stayed 0/16, and both arms
made zero feeding consumes across the eight-seed 256-cell gate.

The treatment also exposes the next representation error. Dataset-local
action-family routing sends the oracle's 32 Move and 16 Wait decisions to the
general/feeding expert, while repeated sampling of the narrow oracle corpus
drives held-out Consume accuracy from 76.0% to 1.1%. Attack remains 100% in the
combat expert, but the observation gate routes only 55% of treatment Attack
labels to it. The next bounded slice should audit a strictly Mind-visible
combat-context predicate, diversify oracle opponents/objectives, and add an
explicit fixed epoch-presentation budget before another mixture comparison.

Behavior-cloning schema 19 now closes the presentation-count confound with
the artifact-bound `epoch_sample_budget` (`behavior-clone --epoch-samples`).
The combat builder defaults to the measured 14,806-presentation control budget,
so appending a small treatment corpus reallocates a fixed amount of optimizer
work instead of increasing it.

It also adds the optional `visible-neighbor-context` expert router. Its label is
a pure function of the anonymous observation: any visible neighboring cell
selects the interaction/combat expert, regardless of the teacher action,
marker, team, or scenario. This is deliberately closer to an isolated/social
decomposition than a privileged feeding/combat label. Auditing the exact
13-corpus treatment mixture found zero contradictory routes among 839 distinct
observable states. The distribution also exposes the unavoidable semantic
cost: line and ring feeding samples all use the interaction expert,
checkerboard feeding samples all use the isolated expert, and mixed layouts
span both. This is desirable for preventing a combat oracle from erasing
social feeding, but means both experts must learn Consume.

The searching-pursuer oracle routes 80 of 112 decisions to the interaction
expert and 32 to the isolated expert. Those 32 are search moves made before an
opponent is visible. Routing them as combat would require either hidden
scenario identity or a stateful gate; the current gate is intentionally
stateless and receives only the current observation. The next experiment
should therefore use this audited router with the fixed budget, while adding
diverse search/exploration demonstrations to the isolated expert rather than
pretending unobservable combat context exists.

That diversity slice is complete and recorded in
[`sweeps/oracle-diversity-v1`](../sweeps/oracle-diversity-v1/README.md). Two
current-schema controls join the original 1-v-1 pursuit oracle: a 1-v-3
searching-pursuer survival policy that contributes 36 movement decisions, and
a branching 1-v-1 evader-elimination policy with 6 Move and 50 Attack
decisions. Together the three policies cover 204 exact decisions, two
objectives, two opponent behaviors, and one genuinely observation-dependent
branch. Each policy covered and achieved its objective on all 64 fresh,
disjoint holdout seeds.

Demonstration schema 13 now makes that holdout evidence an admission condition
rather than documentation. Oracle conversion requires a positive minimum
holdout count (32 by default), complete holdout coverage, and objective success
on every holdout; the manifest binds the requested threshold and all three
results. The warm-start builder
accepts one to three comma-separated reports through
`BLOB_OBSERVATION_POLICY_REPORTS`. Its three-oracle default preserves the
14,806-presentation epoch and the ten/five feeding/interaction split, assigning
one share each to the two contact teachers and three oracles. No tiny oracle
therefore exceeds one fifteenth of an epoch. The next bounded experiment can
now compare the maintained corpus against this diversified treatment without
changing routing, optimizer work, or aggregate feeding exposure.

The first paired run is recorded in
[`sweeps/oracle-diversity-ab-v1`](../sweeps/oracle-diversity-ab-v1/README.md).
Visible-neighbor routing is viable in the maintained control: held-out Consume
reached 98.0%, all eight on-food evaluations succeeded with 100% survival, and
all 32 asymmetric survival objectives passed. The adjacent-food stage still
retained only 46.5% of cells, so the control is not promotion-ready.

The diversified treatment is rejected. Consume fell to 1.5%; both feeding
stages had zero intake and zero survivors; micro survival fell to 0/32 and
elimination remained 0/16. It nevertheless made attacks far more efficient:
held-out contact/skirmish damage rose from 1,304 to 6,480 and kills from 24 to
72 while committed attacks fell from 1,380 to 284. This is useful evidence that
the oracle trajectories teach force selection, but the learned behavior spends
it suicidally rather than integrating it with metabolism or evasion.

That run exposed one remaining A/B confound. The fourfold action-weight cap was
applied after inverse-frequency balancing without restoring mean weight. The
control's total action-loss mass equaled its 473,792 presentations, while the
treatment's rare Wait family reduced total action-loss mass to 331,278 and
Consume mass from 157,931 to 87,843. Behavior-cloning schema 20 now normalizes
post-cap action weights over the actual presented labels, preserving their
relative cap while forcing total action-loss mass to equal the fixed sample
budget.

That exact rerun is recorded in
[`sweeps/oracle-diversity-ab-v2`](../sweeps/oracle-diversity-ab-v2/README.md).
Both arms now receive 473,792 total action-loss mass and 1,952 optimizer steps,
but the diversified treatment still falls to 1.1% held-out Consume accuracy,
zero feeding survivors, and 0/32 asymmetric survival objectives. The result
rules out the V1 loss-scale defect as the primary cause.

The treatment nevertheless demonstrates useful combat transfer: it deals
7,028 contact/skirmish damage versus the control's 1,304 and solves the local
two-versus-one elimination objective on all eight held-out seeds. The current
two-way observable routing is therefore too coarse, not merely incapable of
learning the oracle trajectories.

That replacement is implemented and audited in
[`sweeps/foraging-interaction-exploration-router-v1`](../sweeps/foraging-interaction-exploration-router-v1/README.md).
The model now has foraging, interaction, and exploration action-kind experts.
Its anonymous routing precedence is visible attack/guard activity, current-tile
food, any other visible neighbor, then exploration. Across all 17,128 exact V2
corpus decisions, the contexts receive 8,654, 4,590, and 3,884 samples with
zero contradictory routes. Behavior-cloning schema 21 and training-artifact
schema 43 fail closed on older two-head model records; demonstration datasets
remain reusable. The next experiment should repeat the exact normalized pair
with this router before changing any dataset weight or rollout gate.

That schema-21 pair is recorded in
[`sweeps/oracle-diversity-ab-v3`](../sweeps/oracle-diversity-ab-v3/README.md)
and rejects the learned soft gate. The control learns 99.6% held-out Consume in
the assigned foraging expert but reaches only 1.9% end to end because foraging
gate accuracy is 2.8%; the treatment reaches 99.0%, 20.2%, and 22.5%
respectively. Both arms make zero on-food consumes and lose every feeding cell.
The treatment retains 6,400 contact/skirmish damage and 72 kills but loses both
elimination objectives and all 32 survival episodes.

The authoritative rerun is recorded in
[`sweeps/oracle-diversity-ab-v4`](../sweeps/oracle-diversity-ab-v4/README.md).
Behavior-cloning schema 22 makes the anonymous predicate select the action-kind
expert directly; the learned gate is diagnostic-only. End-to-end and assigned
expert accuracy now match exactly. Training-artifact schema 44 prevents older
soft-routing checkpoints from being resumed under this changed inference
contract. The control improves from zero to 6,144
on-food consumes, proving that hard routing repairs the selector failure, but
stops after 48 energy intake per initial cell and still loses every cell. The
treatment retains 99.0% held-out Consume accuracy yet makes zero consumes on
fresh rollout seeds.

The remaining seam is teacher-trajectory covariate shift. The next slice should
collect immutable, clone-bound feeding corrections on disjoint seeds by labeling
the exact states reached by each frozen learned policy with the maintained
collision-aware forager. Retraining must keep total presentation count fixed
and compare against equal ordinary rehearsal. Promotion still depends on
held-out rollout survival, not static correction-set accuracy.

The causal precursor to that retraining is now implemented as the schema-1
`feeding-recovery` artifact. It forks exact non-initial states reached by a
verified frozen clone at bounded canonical-time intervals and lets the
memoryless collision-aware forager continue through the ordinary isolated Mind
boundary to the original deadline. It records whether each remaining movement,
consumption, and survival objective is still recoverable, without conflating
that stage-5 result with stage-6 action agreement. The artifact is the first
typed source understood by `scenario-diagnosis`: its reported verdict,
compiled-ruleset hash, and ordered feeding-scenario-suite hash are recomputed
and must match the diagnosis claim.

Stage 6 is now implemented as policy-correction datasets and a typed
`scenario-diagnosis` adapter. The first exact-input pilot is recorded in
[`sweeps/feeding-on-policy-agreement-pilot-v1`](../sweeps/feeding-on-policy-agreement-pilot-v1/README.md).
The control agrees with the maintained teacher on only 37.5% of on-food and
30.3% of adjacent-food decisions; the diversified model reaches 14.3% and 0.8%.
The dominant error is `Consume -> Attack`, followed by `Move -> Attack` in the
adjacent stage. Combined with the stage-5 teacher recovery result, this rules
out physical impossibility and strongly localizes the failure to learned
on-policy choices. The next causal experiment is therefore a fixed-presentation
correction-data versus ordinary-rehearsal A/B, followed by evaluation on a new
seed partition; no physics change is justified by this evidence.

That A/B is complete and recorded in
[`sweeps/feeding-correction-ab-v1`](../sweeps/feeding-correction-ab-v1/README.md).
The first orchestration dry run exposed an additional confound: equal sample
presentations yielded 61 versus 63 optimizer updates because recurrent chunks
are batched by length. Behavior-cloning schema 23 therefore adds an optional
exact `optimizer_steps_per_epoch` contract. It deterministically subdivides
equal-length chunk groups, rejects budgets below the minibatch-capacity minimum
or above one-update-per-chunk, and preserves recurrent sequences.

The corrected pair gives both arms 14,806 presentations and 64 optimizer
updates. A 14.3% correction share for one epoch at `1e-5` changes the model hash
but changes none of the paired greedy feeding behavior: agreement remains 37.5%
on-food and 30.0% adjacent-food, intake remains 48.000 and 23.242 energy per
initial cell, and survival remains zero. This rejects the dose, not the
correction method.

Demonstration schema 15 now measures teacher-action logit margins at that exact
correction boundary. Each hash-bound payload sample carries the selected policy
action and the nonnegative advantage of that action kind over the teacher kind
in integer micrologits. Validation recomputes family confusion, total and
maximum margin, fixed crossover bands, and per-family sums from the payload.
The paired probe shows two qualitatively different errors. On-food
`Consume -> Attack` averages 0.2971 logits in the rehearsal arm and 0.2783
after correction; adjacent `Consume -> Attack` moves from 0.4501 to 0.4322.
The weak treatment is moving Consume boundaries in the intended direction. In
contrast, adjacent teacher-`Move` errors retain wrong-kind advantages of
roughly 10.6--12.0 logits. The next bounded dose experiment should focus on
Consume; movement needs a separate representation/data diagnosis rather than
merely more of the mixed correction corpus.

The Consume-focused dose grid is complete and recorded in
[`sweeps/consume-correction-dose-grid-v1`](../sweeps/consume-correction-dose-grid-v1/README.md).
Two epochs at `5e-5` with a 20% pure on-food correction share cross the target
boundary: on-food survival rises from zero to 100%, intake rises from 57 to 249
energy per initial cell, and adjacent-food survival rises to 37.3%. Shares of
33.3% and 50% produce no additional rollout improvement, so the 20% arm is the
minimum tested effective treatment.

The improved trajectory exposes the next DAgger frontier. After consuming the
plant, the policy continues to Consume while the maintained teacher chooses
Move; the wrong Consume kind leads by 11.131 logits. Larger first-round shares
increase that error to 11.876 and 12.428 logits. Use the 20% arm as the parent,
collect complete-prefix post-depletion corrections, and pair the second-round
treatment against equal rehearsal. Do not call this candidate promotable until
adjacent survival and combat retention pass on new seeds.

That second round is complete and recorded in
[`sweeps/feeding-dagger-round2-v1`](../sweeps/feeding-dagger-round2-v1/README.md).
The normalized correction arm reduces the post-depletion Move deficit from
11.950 to 9.668 logits but regresses to the same zero-survival rollout as its
rehearsal control. A 50% correction share reaches about 7.5 logits and 6.4%
adjacent survival, which is still not promotable and is too inefficient to
justify further blind dose extrapolation.

Behavior-cloning schema 24 now supports verified parent-relative
`action_kind_expert_only` adaptation. Only one observation-routed action-kind
head survives each optimizer update; every shared/recurrent parameter and all
other heads remain at the parent. This is a host-side training constraint and
does not change the Mind ABI, observation, authoritative routing, or physics.
The experiment corrected an important unstated assumption: residual loose or
diffuse energy keeps post-plant states in the foraging context, so an
exploration-head treatment cannot consume this correction corpus.

Foraging-head correction-only training preserves the full Consume prefix for
eight epochs while reducing the Move deficit from 11.131 to 0.930 logits. One
additional epoch crosses the boundary but also changes 25% of prefix Consume
kinds to Move; fresh rollout intake falls to 48 and survival remains zero.
This sharp transition localizes the next representation seam. The foraging
head needs an observation-local residual or small adapter over raw current-food,
gut-capacity, and legality features so depletion can be distinguished without
rewriting the shared recurrent representation. The immutable round-two
trajectory is the acceptance test: retain every successful Consume-prefix
kind, select Move after depletion, then pass fresh feeding rollout gates before
running combat retention.

The next hierarchical correction is recorded in
[`sweeps/feeding-dagger-round3-v1`](../sweeps/feeding-dagger-round3-v1/README.md).
Behavior-cloning schema 27 adds verified-parent effort-head-only adaptation,
and correction telemetry now separates kind, target-only, effort-only, and
joint target/effort disagreements. The inherited trajectory contained 760
effort-only errors; the isolated effort stage reaches 100% held-out Move exact
accuracy without changing another output.

Round-three overlap analysis finds 35 exact observable states and no
within-dataset conflict. Even the eight raw adapter inputs retain 11 distinct,
non-conflicting states, so the Consume/Move failure was an optimization margin,
not missing local information. A larger adapter dose plus isolated interaction
and target stages reaches 97.1% exact agreement on a fresh 4,096-decision
trajectory. Three fresh rollout seeds improve on-food survival from 0% to
33.6% and adjacent survival from 0% to 58.2--59.0%; intake rises from
57.000/33.844 to 144.652/183.289--185.992 energy per initial cell. This is a
meaningful escape from collapse but remains below the 80% promotion gates.
Continue DAgger from this trajectory, emphasizing target-direction errors,
before combat-retention qualification.

Round four is recorded in
[`sweeps/feeding-dagger-round4-v1`](../sweeps/feeding-dagger-round4-v1/README.md).
Its two-seed, 16,384-decision frontier has no conflicting exact observable
labels, but direct target balancing regresses Move agreement. An isolated
foraging residual improves held-out Consume-kind accuracy from 88.9% to 98.9%
while retaining 100% Move-kind accuracy. The remaining Guard/Move boundary is
not separable by a safe linear-head dose: the first treatment that learns some
Guards also regresses Move.

Policy schema 28 therefore adds zero-initialized, phase-local interaction and
exploration action-kind residuals over the frozen recurrent state and the
non-random local observation header. The training constraint can update only
one selected residual, and legacy schema 25--27 foraging artifacts migrate
without changing behavior. The interaction treatment reduces the mean
Guard-to-Move margin from 19.9215 to 17.1222 logits but does not yet cross the
argmax boundary. Even so, the combined candidate improves on-food survival to
38.7% and adjacent-food survival to 61.3--62.5% on two fresh seeds, with
156.699 and 189.676--191.570 intake per initial cell. It remains below the 80%
promotion gates and is not promoted; require a targeted Guard crossover with
no Move regression before combat retention.

The bounded Guard treatment is recorded in
[`sweeps/feeding-dagger-round5-v1`](../sweeps/feeding-dagger-round5-v1/README.md).
Behavior-cloning schema 31 adds an optional legal action-kind hinge objective
and phase-resolved correction telemetry. That telemetry corrects the central
assumption from round four: on a fresh feeding trajectory, 137 of 168
Guard-to-Move errors route to exploration, while only 5 route to interaction
and 26 to foraging.

Schema 31 also adds a zero-initialized, permutation-invariant local-context
residual over the cell's non-random header and featurewise pooled raw neighbor
slots. It remains inside the Mind observation boundary and can be trained in
isolation. An eight-epoch exploration treatment reaches 76.5% held-out Guard
accuracy but regresses Move to 81.7%, so it is rejected. The selected staged
2+2+2 treatment preserves 100% held-out Move accuracy and every fresh greedy
choice while reducing mean Guard deficit from 17.1222 to 1.5967 logits; its
exploration subset averages 0.9765. It is diagnostic rather than promotable.
Behavior-cloning schema 32 adds exact total optimizer-step prefixes on top of
the existing fixed per-epoch update budget, with actual mid-epoch presentation
counts and completed-epoch telemetry. A coarse-to-fine search found no useful
checkpoint: update 505 has 0%/100% held-out Guard/Move accuracy, while the very
next update jumps to 76.5%/93.4%. On one shared fresh trajectory it converts 93
exploration Guard misses but creates 88 exploration Move-to-Guard errors,
reducing exact agreement from 86.9% to 85.9%. The next treatment should split
or scale that terminal update while retaining optimizer state; neither side of
the current boundary is promotable.

Schema 33 performs that controlled experiment by binding a fractional
learning-rate scale to only the final retained optimizer update. It preserves
the first 505 updates and their AdamW moments. The first held-out Move failures
occur before the first Guard success: dose 0.80625 loses three of 5,518 Moves
while fixing no Guards; dose 0.81250 fixes one of 170 Guards but loses 28
Moves. On a fresh 8,192-decision trajectory the latter fixes two exploration
Guard errors and creates nine exploration Move-to-Guard errors, reducing exact
agreement by 11. There is no useful scalar dose seam. Further work should
factor defensive readiness or threat-conditioned guarding in the
representation while preserving anonymous, local per-cell inputs.

Schema 34 tests that factorization directly. A reusable readiness diagnostic
ranks non-random header and permutation-invariant slot statistics inside one
authoritative routing context. Assimilated energy alone separates the
round-four exploration Guard/Move labels with 98.66% balanced accuracy: energy
bins 28--30 contain 202 Guards and no Moves, bin 31 is ambiguous, and bins
32--38 contain only Moves. A zero-initialized three-parameter readiness head
over assimilated and gut energy adds only to the exploration Guard logit and
can be trained while preserving every other model output path.

The isolated head confirms that energy evidence is usable but does not by
itself solve the inherited margin geometry. Natural-frequency cross-entropy
reduces the fresh Guard deficit from 17.1222 to 2.1169 logits without changing
an action. A capped-weight hinge reaches 0.9383 with no false Guards; its next
stage jumps to 76.5% held-out Guard and 93.6% Move. On a matched fresh seed it
fixes 220 exploration Guard errors, creates 62 exploration Move-to-Guard
errors, and reduces exact agreement from 86.0% to 84.9%. This rejects missing
visibility and excess adapter capacity as causes. The next objective should
explicitly constrain readiness to remain nonpositive on frozen-parent Move
examples while lifting only separable low-energy Guard states.

That proposed label-level objective is now superseded by exact counterfactual
branch evaluation. A first four-state, 256x256 adjacent-food probe on seed
`1420000202` compared the teacher Guard with the frozen policy Move at
256/1,024/4,096 simulation quanta under both frozen-policy and teacher
continuations. Survival tied in every pair. Guard retained more team stored
energy in all four states at 256 and 1,024 quanta, but under frozen-policy
continuation Move was better in all four at 4,096 quanta; under teacher
continuation the long horizon split two to two. The immutable artifact is
`training-output/feeding-dagger-round5-v1/counterfactual-guard-move-seed-1420000202-h4096.json`
(artifact hash
`171e94b40f5fbe77d1a339b90d8e7f11f5783d0042a02d58829ba55a122f445a`).

This is the important correction to the current strategy: low-energy Guard is
not yet a justified universal target. The action has a short-horizon resource
advantage at these states, while the preferred long-horizon action depends on
the continuation policy. Expand the state/seed sample, inspect the other
biological metrics, and train against a declared horizon/value objective only
after the resulting advantage is stable. Use all-legal short probes to discover
better nearby Guard/Move parameters and teacher-policy deep probes to confirm
them without multiplying full-world rollout cost.

Schema 2 then repeated the test on eight new discovery seeds, with two states
per seed, one state per simultaneous frontier, and a bounded 256-frontier source
window. All eight seeds supplied two qualifying states. Across the resulting 16
states, Guard retained more target and team stored energy at 256 and 1,024
quanta under both continuations. Survival and population tied everywhere. At
4,096 quanta under frozen-policy continuation, Move retained more team energy
in all 16 states and all eight seed-balanced comparisons, although Guard still
retained more target energy on seven of eight seed aggregates. Under teacher
continuation the long-horizon team preference split four seeds to four while
Guard retained more target energy on five of eight.

The compartment-complete multi-seed artifact is
`training-output/feeding-dagger-round5-v1/counterfactual-guard-move-multiseed-v2.json`
(artifact hash
`93b6311b11ea8ba9bb464bd1158230078167243634dccbad94398feacdfe320b`).
The reversal is therefore reproducible, not a one-seed anomaly. Do not derive
an unconditional Guard or Move correction set from these states. The next
modeling question is an explicit action-value objective—what horizon and whose
biological value the policy should optimize—followed by broader action
counterfactuals under that declared objective.

That objective is now explicit and sensitivity-tested. It uses survival or
population as a first lexicographic tier and compartment-weighted energy as a
second, with a conservative Pareto perspective that refuses cell/colony
tradeoffs and requires frozen-policy/teacher continuation agreement. Core and
assimilated energy receive full weight. At zero or quarter gut weight, all 16
states prefer Guard at 256/1,024 quanta; at 4,096 quanta eight prefer Guard and
eight are incomparable, while all eight seed aggregates prefer Guard. At half
gut weight the long-horizon conservative result falls to five Guard and eleven
incomparable states, with five Guard and three incomparable seeds. At full gut
weight it becomes one Move and fifteen incomparable states, with one Move and
seven incomparable seeds.

The value artifacts are `counterfactual-value-no-gut-v1.json`,
`counterfactual-value-quarter-gut-v1.json`,
`counterfactual-value-half-gut-v1.json`, and
`counterfactual-value-full-gut-v1.json` beneath the same round-five output
directory. Their hashes are respectively
`ea67013c8ed766a19c8c76a8df2e13a7ffa79e787ebda00ed1c66a0f66317b2f`,
`a62d3c6457135d9109d013559c467a6cccd7eabb89b88f462984d98784c39030`,
`d513ef44a74942274a403e9386506541e65f5ef5a94436df87d205164374ba60`,
and `f9a8008a4bad001c19972d96758b8c9045003b2964c5938fb3c0a267b463c3c`.
The action label is therefore not invariant to the declared biological value
of gut contents. No counterfactual supervision is promoted from this slice.

The next coarse-to-deep slice broadened the intervention surface rather than
continuing to compare two already suspect baselines. An eight-seed, one-state-
per-seed +256-quanta pass evaluated all 27 legal base Guard/Move choices under
both continuations. Every state and every tested gut valuation produced the
same ten-action frontier: low and medium Guard plus minimum-effort Move in each
of eight directions. Maximum Guard and every medium/high-effort Move were
robustly dominated. In particular, the learned policy's medium-effort Move is
dominated by the corresponding minimum-effort action.

A deep pilot on the first four seeds and an untouched confirmation on the
remaining four retained those ten candidates plus the learned baseline. At
1,024 quanta the same three target-free archetypes remain. At 4,096 quanta both
Guard efforts are dominated, leaving all eight minimum-effort directions but
only one target-free archetype: Move at minimum effort. The result is identical
for cell, colony, and conservative perspectives and for zero versus full gut
weight. Directional positions differ, but the measured biological outcomes do
not supply an observable basis for choosing among them. Direction must come
from visible geometry or the cell-private random block, never from the
privileged future branch.

The coarse branch artifact hash is
`7298bdd04ea1ce1768b1692a14c9ef9fc48494ceefc35d2251927c730dc2c764`.
The deep pilot/confirmation branch hashes are
`f68a95bf3c4a267d7e094f3ad889d82cb09a0aa3d0175c9cce2e1475d3fa7274`
and `445f5bc76de21c00e9caec61edf866ff6075532777d4f9a8f2c6a6b66add7e80`.
Their no-gut target-free value hashes are
`a16ef9f78fd6b636624b3345c0af96bae1ea4377703501ea4daaa032019a32ae`
and `a27a8eaf0ded62d5adb68593e257b26086076b039de9f6a0390b0d9fde100c8c`.
This is sufficient evidence for a target-free minimum-effort Move correction
experiment, but not for a direction label or general Move preference outside
the selected low-energy exploration context.

That experiment now has an exact-state active control. A replay enrichment pass
attaches the original anonymous observation, legal masks, and cell-private
recurrent memory only after matching the historical checkpoint and observation
hashes plus every state identity field. Each treatment label retains the
policy's target and changes only Move effort; `--effort-head-only` restores all
other parent parameters after every update. Across the eight states, learning
rate 0.005 crosses the held-out greedy-effort boundary at exactly ten optimizer
steps: nine steps remain at 0% corrected exact accuracy, while ten reach 100%.
The matched ten-step control remains unchanged and 100% correct. This proves
the correction mechanism, not ecological improvement. The attempted eight-seed
two-stage local CPU qualification was stopped without publication after fifteen
minutes; run full qualification as resumable shards or on the accelerated
remote runner. Exact hashes and commands are recorded in
`sweeps/feeding-dagger-round5-v1/README.md`.

The qualification path is seed-sharded and restart-safe. Each evaluator emits
one flushed progress record after every completed stage/seed episode. Passing
`--resume` never means "trust the path": it fully validates the existing
artifact and requires an exact match on the complete config, config-source
hash, clone-metadata hash, model hash, and ordered seed suite. The paired
`scripts/qualify_counterfactual_effort_ab.sh` runner preserves successful
shards across interruption, reports all shard failures, and only then merges
them in the declared order. This is the preferred operating path for remote or
preemptible qualification jobs.

The completed V3 evidence used the faster measured CPU NdArray backend and one
serial worker:

```sh
cargo build --release -p blob_rl --bin train --bin rules-sweep-run \
  --no-default-features --features ndarray --locked
target/release/rules-sweep-run \
  sweeps/micro-combat-ablation-256-v3/manifest.json \
  --max-parallel 1
```

Three pairs are a directional experiment, not a final selection study. Expand
the seed set if the paired confidence intervals overlap materially or if only
one arm reaches the joint gate.

After every rollout environment reaches
`self_play.start_after_sim_time_quanta_per_env`, promotion is considered every
`opponent_update_interval` PPO updates. The first eligible evaluated checkpoint
anchors the pool. Later candidates must remain within
`max_fixed_worst_case_regression` of the best fixed-suite result and achieve at
least `min_pool_win_rate` against every selectable pool member on the same
held-out seeds. Selection happens only at episode boundaries using an
independent checkpointed RNG; `baseline_probability` retains native-baseline
coverage. After fixed-baseline selection, snapshot probability is proportional
to
`exp((rating - max_rating) / rating_temperature) * (1 + selections)^(-exposure_exponent)`.
This favors strong opponents without repeatedly selecting the same member;
setting `exposure_exponent` to zero disables exposure balancing. Ratings use
held-out candidate-versus-incumbent results, count timeouts as half a win, and
apply the configured `elo_k_factor`. New candidates inherit the most recently
promoted rating as their provisional rating, or `initial_rating` for the first
member. Eviction keeps the anchor and newest members within
`max_opponent_pool`. An evicted member remains resident only while an existing
episode still uses it, then is released.

Every promotion is a complete immutable checkpoint, and
`opponent-pool/manifest-NNNNNNNN.json` records the ordered active pool with
checkpoint paths, update/action counters, model hashes, ruleset hash, and the
selection/gating configuration. It also records each member's rating, held-out
game count, and rollout-selection count. Continuation state binds those league
values, the selectable pool, temporarily retained opponents, per-environment
assignments, and opponent RNG. Promoted-checkpoint resume equivalence is
covered by the split-run regression.

Periodic and newly-best policies publish immutable `checkpoint-NNNNNNNN`
directories. Each contains full-precision Burn model and optimizer records, a
bounded MessagePack continuation record, and JSON metadata binding all three
SHA-256 digests, the training configuration, seed, compiled ruleset hash,
action/update counters, and evaluation result. The continuation record contains
every canonical environment, host-only cell dispatch state, episode counters
and reward accumulators, independent policy/minibatch RNG streams, counters,
and the best held-out result. Publication is a same-filesystem atomic directory
rename; `best.json` is an atomic pointer to an immutable checkpoint rather than
a mutable model copy.

`--resume checkpoint-NNNNNNNN` restores that exact clean boundary and appends
to existing metrics files. It rejects changed training dynamics and accepts
only a larger action target plus changes to evaluation/checkpoint cadence and
the output directory. `[initial_policy]` is the hash-verified scientific warm
start; `--load-model` remains the deliberately inexact ad hoc weights-only
path. Resume equivalence is tested by comparing the next model and full
continuation record with uninterrupted training.

## Maturity sequence

1. **Rollout correctness — implemented.** Gate all later work on deterministic
   event-time GAE, explicit terminal rewards, finite-loss smoke training, and
   resolver conformance.
2. **PPO mechanics — stability slice implemented.** PPO now uses independently
   seeded shuffled minibatches, clipped policy and value objectives, parameter-
   tensor norm clipping, approximate KL, clip fraction, explained variance,
   and configurable KL early stopping. `minibatch_size`, `max_grad_norm`,
   `value_clip_epsilon`, and `target_kl` are active. Burn applies the norm limit
   independently to each parameter tensor, not as one network-global norm.
   Learning-rate scheduling remains part of the later long-run tuning slice.
3. **Evaluation, artifacts, and resume — implemented.** Gradient-free
   deterministic evaluation, fixed held-out seed suites, atomic full-precision
   model/optimizer artifacts, integrity metadata, best-model selection, exact
   environment/RNG restoration, metrics continuation, and split-run
   equivalence are active.
4. **Opponent curriculum — rated bounded rotation implemented.** Wait, random,
   and aggressive profiles plus hash-bound policy snapshots are reproducible,
   configurable, and evaluated separately. Best-checkpoint selection maximizes
   worst-matchup win rate before aggregate win rate and reward, preventing one
   easy opponent from hiding a failed matchup. Deterministic episode-boundary
   sampling, baseline mixing, fixed-suite regression and incumbent-pool gates,
   anchor-preserving bounded eviction, exact resume, immutable pool manifests,
   Elo-style held-out ratings, exposure-aware weighted sampling, and batched
   snapshot-opponent inference are active. Behavioral diversity metrics (rather
   than exposure as a proxy) remain a possible later extension.
5. **Recurrent warm start — isolated-state foundation implemented.** Exact
   anonymous Mind inputs produce immutable, hash-bound maintained-Mind
   datasets; masked supervised optimization can combine field/population
   datasets using seed-disjoint validation and balanced per-dataset sampling,
   and publishes a model whose complete provenance is bound by fresh PPO
   configurations. Learned state is canonical, cell-private, replayable, and
   row-separable in batched execution. Behavior cloning and PPO both train
   bounded recurrent chunks with canonical state transitions, terminal-aware
   boundaries, and cell-private trajectory keys. PPO metrics record the unroll
   length and chunk count for every update.
6. **Feeding curriculum — implemented.** Deterministic on-food and adjacent-food
   initial conditions rotate only at episode boundaries, retain the unchanged
   Mind ABI and physics, and graduate into the ordinary competitive field.
   Fixed-seed greedy evaluation independently gates best-checkpoint and
   self-play promotion on actual extracted energy, movement, survival, and
   per-cell intake. Mutable CSV diagnostics and recomputed, tamper-detecting
   checkpoint, best-pointer, and self-play-promotion bindings are active.
7. **Local-combat curriculum — bridge implemented.** Canonical-time cycling
   interleaves feeding retention, paired one-on-one contact, opposed-line
   four-versus-four skirmishes, and ordinary competition. Stage-specific
   exploration is behavior-policy-correct, exact resume retains every active
   stage, and held-out evidence separates scenario, population, energy, and
   opponent. Stage-boundary clipping prevents long episodes from skipping later
   rehearsal blocks. Promotion now requires actual damage in both combat
   populations plus configurable resolver-attributed kill floors. One corrected
   paired cadence/mixture sweep found joint qualification on 3/3 two-cycle,
   2/3 four-cycle-balanced, and 3/3 four-cycle-skirmish-heavy seeds, but none
   converted the Aggressive full-match timeout. A bounded, verified competency
   frontier now preserves nondominated ecology and combat specialists without
   weakening promotion. The next evidence gap is stage-local specialist
   distillation with exact resume and regression controls; longer blind
   optimizer exposure remains deferred.
8. **Canonical rules sweeps — profile boundary implemented.** `[env.rules]`
   accepts strict partial overrides of the complete reference physics profile,
   expands omitted values before artifact publication, validates topology
   against the configured board, and binds the compiled semantic hash to every
   checkpoint and pool manifest. `Engine` preserves caller-supplied rules rather
   than rewriting neighborhood or mass thresholds. Rewards remain outside that
   hash and are stored separately in the full training config. The sweep
   planner expands named rule variants over at least three shared replicate
   seeds, rejects duplicate expanded rulesets, publishes full per-run configs,
   and records semantic, compiled, config-file, and normalized experiment
   hashes in one atomic manifest. The executor verifies that entire matrix,
   takes one OS-released manifest lock, bounds resident trainer processes,
   records each attempt atomically, resumes only from the newest fully verified
   update-boundary checkpoint, and refuses implicit retries after failures.
   Successful runs publish immutable terminal records; a complete matrix
   publishes per-variant and seed-paired candidate-minus-baseline summaries
   with small-sample Student's-t 95% intervals. Exact action outcomes and
   bounded ecology/energy time series are included in the run and paired sweep
   summaries.

Initial conditions are represented separately by a strict `ScenarioProfile`.
Sweep variants may use `[variants.scenario]` for board size, starting
population/energy, episode horizon, and initial resource/plant parameters.
These fields receive their own semantic hash; reward, PPO, opponent, and
resolver settings cannot be smuggled through the scenario override. A variant
is duplicate only when both its expanded rules and scenario match another
variant.

Before PPO, run anonymous native baselines directly through the canonical Mind
boundary. The `forager` profile consumes, moves toward locally visible
resources, and splits, but never attacks. For example:

```sh
cargo run -p blob_rl --no-default-features --features ndarray --bin viability -- \
  --config blob_rl/config/default.toml \
  --candidate forager --opponent forager \
  --seeds 101,202,303 \
  --output viability/forager-vs-forager.json
```

The immutable report binds code revision when available, semantic and compiled
rules hashes, a separate scenario hash, profiles, seeds, telemetry cadence,
per-episode resolver time/population/conservation totals, aggregate extinction
and outcome metrics, and the complete authoritative telemetry summary. Rewards
are zeroed and no model is constructed.

To evaluate every planned rules/scenario variant without training, run a
bounded profile matrix directly from the published sweep manifest:

```sh
cargo run -p blob_rl --no-default-features --features ndarray \
  --bin viability-matrix -- \
  sweeps/viability-v1/manifest.json \
  --candidates forager,random \
  --opponents wait,forager,aggressive \
  --baseline-variant baseline \
  --max-parallel 8 \
  --output viability/viability-v1-matrix.json
```

The executor verifies the manifest and every expanded config before running.
It restores results to canonical order after bounded parallel execution, so
`--max-parallel` does not alter report identity or bytes. Every non-baseline
variant is compared to the selected baseline episode-by-episode on identical
seeds and profile matchups. The report keeps loss, timeout, and win ordered but
distinct, and includes paired win, survivor-population, environment-step, and
resolver-time differences with uncertainty summaries. It refuses partial
matrices and immutable-report replacement.

Gate the completed matrix before allocating an RL job:

```sh
cargo run -p blob_rl --no-default-features --features ndarray \
  --bin viability-gate -- \
  viability/viability-v1-matrix.json \
  --gates blob_rl/config/viability_gates.toml \
  --output viability/viability-v1-decision.json
```

The strict TOML policy enables only thresholds that are explicitly present.
Absolute checks cover timeout, non-combat extinction, tracked mass-energy
drift, survival, reachable combat, commit rejection, and frustrated resolution
rates. Paired checks cover ordinal outcome regressions, candidate win-rate and
survivor-population drops, and timeout-rate increases relative to the selected
baseline variant. Thresholds are inclusive.

Before evaluating, the gate revalidates the complete nested matrix: seed and
profile grids, scenario and configuration hashes, episode-derived aggregates,
telemetry outcome totals, and all paired comparisons. Its immutable decision
binds SHA-256 hashes of the exact matrix and policy bytes plus the normalized
policy identity. A failed scientific gate still publishes its diagnostics and
exits with status 2, allowing a remote scheduler to block training while
retaining an auditable explanation. The supplied
`blob_rl/config/viability_gates.toml` is an explicit exploratory starting
policy, not a frozen balance specification.

Require that exact passing decision when launching the training sweep:

```sh
target/release/rules-sweep-run sweeps/viability-v1/manifest.json \
  --require-viability-gate viability/viability-v1-decision.json \
  --viability-matrix viability/viability-v1-matrix.json \
  --viability-gates blob_rl/config/viability_gates.toml \
  --max-parallel 1 --threads-per-run 8
```

All three gate inputs are mandatory together. Immediately before taking the
sweep lock or creating worker threads, the executor bounds and decodes the
inputs, revalidates the nested matrix, reruns the policy, requires exact
equality with the published decision, requires a passing result, and checks
that its manifest SHA-256 equals the sweep being launched. A stale policy,
edited matrix, copied decision from another sweep, failed check, or malformed
artifact therefore starts zero trainers and creates no run status. Retries
repeat the same verification. Use gate and executor binaries from the same
build: exact decision comparison intentionally includes package and source
revision identity.

For the normal restartable workflow, the preflight command composes planning,
matrix execution, gating, and optional training without weakening any stage:

```sh
cargo build --release -p blob_rl \
  --bin train --bin viability-preflight
target/release/viability-preflight \
  blob_rl/config/viability_sweep.toml \
  --candidates forager,random \
  --opponents wait,forager,aggressive \
  --gates blob_rl/config/viability_gates.toml \
  --baseline-variant baseline \
  --matrix-max-parallel 8
```

Add `--execute-training --training-max-parallel 1 --threads-per-run 8` to
continue directly into training after a pass. Without `--execute-training`,
preflight stops after publishing the decision. A failed scientific decision is
retained and exits with status 2 in either mode.

The deterministic stage paths are
`<sweep>/preflight/viability-matrix.json` and
`<sweep>/preflight/viability-decision.json`. On restart, the command verifies
and reuses each completed stage. It rejects changed sweep-spec or base-config
bytes, a changed profile matrix, edited artifacts, a changed policy behind an
existing decision, or a cross-build result rather than overwriting them. A per-sweep
preflight lock prevents concurrent matrix work; the training executor still
uses its own independent run lock and leases.

Plan the documented first viability sweep with:

```sh
cargo run -p blob_rl --bin rules-sweep -- blob_rl/config/viability_sweep.toml
```

The command refuses to replace an existing output plan. Each variant receives
the same ordered training seeds for paired analysis, and variants can alter
only `[variants.rules]` and `[variants.scenario]`; rewards, PPO settings,
evaluation partitions, opponent selection, and all other training dynamics
come from the shared base config. The resulting
`manifest.json` identifies every expanded `config.toml` and the planner prints
the corresponding executor command. This planning step performs no expensive
training itself.

Build the trainer and executor once, then run the published plan:

```sh
cargo build --release -p blob_rl --bin train --bin rules-sweep-run
target/release/rules-sweep-run sweeps/viability-v1/manifest.json \
  --max-parallel 1 --threads-per-run 8
```

For a containerized scientific run, also pass the deployed immutable identity,
for example `--trainer-container-digest sha256:...`. The executor always hashes
the trainer bytes itself and rechecks them immediately before each launch; the
OCI digest adds the surrounding runtime identity.

The ungated form remains available for exploratory or diagnostic training.
Production rule-tuning jobs should supply the required viability artifacts as
shown above.

For remote hosts, CPU and NVIDIA Vulkan/WGPU targets package the same trainer,
planner, and executor in `Dockerfile.training`; see
[Dockerized remote training](docker-training.md) for durable mounts, detached
execution, GPU prerequisites, resource bounds, and cross-host checkpoint
resume. Exact resume remains on the backend that created a checkpoint. Use
policy-only `--load-model` when deliberately transferring learned weights
between CPU and WGPU training.

`--max-parallel` bounds whole trainer processes; `--threads-per-run` sets each
child's `RAYON_NUM_THREADS`. The default process limit is one because multiple
WGPU trainers usually contend for the same GPU and replicate model/optimizer
memory. Increase it only for a CPU backend or explicitly partitioned devices.
Each attempt receives separate stdout/stderr logs. An interrupted run resumes
from its newest hash-verified immutable checkpoint; a failed run stays failed
until `--retry-failed` is supplied. The executor also recovers the narrow case
where training finished but it crashed before recording success by recognizing
a complete metrics tail. Each trainer holds its own OS run lease, so a trainer
orphaned by an executor crash is detected and never duplicated; a later pass
recovers its completed output or resumes after that lease is released.

When every run succeeds, `aggregate.json` is published automatically. It binds
the source manifest and execution contract, keeps evaluation metrics distinct
from shaped training metrics, and pairs variants by training seed before
calculating differences.
It will not silently drop failed or missing replicates. To re-verify completed
records and publish the aggregate without launching trainers, use:

```sh
target/release/rules-sweep-run sweeps/viability-v1/manifest.json --aggregate-only
```

The current aggregate uses each run's terminal training row and terminal
held-out `suite` evaluation row. If completion falls between periodic
evaluation boundaries, the trainer appends one deterministic final suite.
Training win/reward rows describe only the last PPO update and are diagnostics;
held-out evaluation comparisons are the scientific outcome. An enabled
evaluation that fails to publish a complete suite makes the run fail instead
of silently producing a training-only scientific result.

## Gameplay telemetry

Telemetry is host-only: it reads canonical resolver reports and immutable
simulation state after resolution, but is never placed in a Mind observation,
physics input, state hash, or replay. It has two cost classes:

- Every action commitment/completion is counted exactly by side and action
  family, including accepted, rejected, successful, frustrated, contested, and
  interrupted outcomes plus effort, payload, births, deaths, and attributed
  kills. Successful attacks additionally report raw, guard-mitigated, applied,
  and overkill damage for both the dealing and receiving side. Successful
  excavation/deposition reports exact elevation units and conserved terrain
  mass lifted or dumped. Accepted signal deposits are classified as explicit
  actions or sidecars and retain their exact four-bit channel pattern,
  per-channel deposit count, and conserved energy cost. Passive decay is
  reduced per channel inside the existing kernel; terrain erasure is read from
  the resolver's pre-edit tile delta and attributed to the manipulating side.
  This is linear in work and reports the resolver already produced and does
  not scan the board at each action frontier.
- Canonical cells and tiles are scanned periodically for population,
  mass-energy compartments, encounter edges, spatial entropy, and resource
  concentration. Signal samples additionally retain per-channel field energy,
  active tiles, spatial concentration, and total variation over the configured
  observable-signal graph. This is the configurable O(cells + tiles) portion.

The default profile is:

```toml
[telemetry]
enabled = true
state_sample_interval_steps = 32
max_state_samples_per_episode = 64
episode_log_stride = 16
```

Initial, periodic, episode-terminal, and training-terminal censored samples are
retained. The per-episode vector has a hard cap; if the configured cadence
exceeds it, the newest sample replaces the last retained periodic sample so the
terminal state is never lost. Increase the interval for very large boards.
Set `enabled = false` to remove both report accounting and state scans, or set
`episode_log_stride = 0` to retain the compact all-episode summary without
publishing detailed episode files.

Completed sampled episodes are immutable files under
`artifacts/telemetry/episodes/env-N/`. The compact
`artifacts/telemetry/summary.json` includes all episodes, including action
totals from currently censored episodes, and is carried through exact
checkpoint resume. This avoids duplicate episode rows when restarting from an
earlier update boundary.

Sweep `result.json` and `aggregate.json` derive comparable rates and paired
differences from that summary: outcome and attack rates, births/deaths/kills per
1,000 completed training actions, mean populations and assimilated energy,
plant/loose/diffuse environmental energy, encounter edges, training spatial
entropy, resource concentration, damage per successful attack, effective-damage
efficiency, guard mitigation, overkill, and terrain elevation/mass movement per
1,000 actions. Signal selection, deposited energy, passive decay, terrain
erasure, mean field energy, channel-active tiles, and observable field total
variation are included as first-class sweep metrics. Damage and signal effects
are taken directly from explicit resolver evidence rather than inferred from
ambiguous whole-cell or whole-board deltas.

Simultaneous attacks retain the existing aggregate physics: raw damage is
combined per victim, guard reduction is applied once, and actual damage is
capped by remaining assimilated energy. The resolver then attributes both the
post-guard and applied totals proportionally with largest-remainder rounding;
actor key breaks exact ties. Attribution therefore sums exactly to the physical
effect and is invariant to commit iteration and worker count. These effect
records remain host-only and never enter Mind observations.

## Rules-tuning protocol

Use at least three fixed seed partitions: training, validation, and a final
held-out test suite. For every candidate ruleset, first run non-learning
baselines to measure whether the game is viable, then train at least three
policy seeds and compare confidence intervals rather than best single runs.

The first sweep should measure these outcomes:

- win/loss/timeout rate and simulated time to termination;
- extinction rate without combat;
- living population, births, deaths, and energy per compartment over time;
- action choice, rejection, frustration, contention, and interruption rates;
- attack payload, raw/applied/mitigated/overkill damage, and kill counts plus
  consumption, regurgitation, movement, terrain lift/dump, and split success
  rates;
- spatial dispersion, encounter rate, and resource concentration;
- policy entropy, explained variance, approximate KL, and reward-component
  totals.

Tune coupled parameter families together:

1. **Viability:** metabolism, digestion, bite/gut capacity, starting energy,
   plant density, plant growth, and diffuse transport.
2. **Mobility and inertia:** move base duration, mass-duration slope,
   mass-per-effort, elevation cost, and effort-tier multipliers.
3. **Combat:** attack base duration, mass-duration slope, effort, payload and
   mass damage, guard reduction, and telegraph duration.
4. **Reproduction and ecology:** child core mass, minimum survival reserve,
   split duration/cost, regurgitation, and resource replenishment.

Do not begin with reward coefficients. Establish viable rule ranges using
outcome and conservation metrics, then use the sparsest reward that reliably
learns those rules. Report reward-shaped training performance separately from
unshaped evaluation performance.

## Micro-combat feasibility and policy oracles

A named micro-combat scenario is not a useful learning target merely because
its geometry and starting energies are valid. Before it enters a curriculum or
promotion gate, it must have a legal strategy witness under the exact compiled
ruleset. It should also fail a passive-degeneracy check: if `Wait` reliably
satisfies a survival objective, that objective measures timeout survival rather
than defensive control.

Run the maintained observation-legal witness audit with:

```sh
cargo run -p blob_rl --bin micro_combat_feasibility --release -- \
  --config blob_rl/config/micro_combat_ablation_256_v2_base.toml \
  --scenarios blob_rl/config/micro_combat_scenarios.toml \
  --seed-base 930000000 \
  --seed-count 128 \
  --reliability-target 0.95 \
  --output training-output/micro-combat-feasibility-128.json
```

Every witness uses the same anonymous `ReferenceMindInput` boundary as a
submitted Mind. Thus success is a realizable lower bound, not a privileged
simulator result. The first 128-seed audit found that the maintained 1v3
scenario is robustly survivable. Both evasion and continuous guarding survived
every seed, while waiting survived 78.125%. Evasion was the stronger
lexicographic witness: it retained 67.70 average stored energy and received
18.20 average damage, versus 44.34 energy and 39.66 damage under guarding. The
three easier survival scenarios were passively degenerate because waiting
survived every seed. Neither elimination scenario had a successful maintained
witness, so they are currently unqualified rather than presumed difficult.

The current aggressive profiles have no pursuit memory. Once an evader leaves
the observable neighborhood, they select an otherwise tied empty movement
target without remembering the last-seen direction. Report evasive results
against this opponent as current-rules evidence, not as a general pursuit
result. Scenario qualification should also include a private-memory pursuer so
the evasion seam tests escape against continued search rather than loss of
contact alone.

The maintained pursuer stores only the last-seen signed local direction and a
four-decision search lifetime in each cell's canonical private memory. On the
1v3 pursuit control, the original 8,192-quanta horizon remained passively
degenerate: waiting survived with only 5 stored energy. Waiting reached the
minimum-survival boundary at 12,288 quanta and failed by 16,384, while evasion
still survived every episode with 57 average stored energy. Continuous guarding
also survived but retained only 29. The 16,384-quanta case is therefore the
first maintained horizon witness with a reliable non-passive solution. Audit
final population as well as survival, because the forager can split and turn a
nominal single-founder scenario into a multi-cell survival strategy.

The single-cell planner branches from complete `BlobEnvCheckpoint`
continuations and hashes both canonical physics and host-private continuation
metadata, including the independent random streams needed to reproduce future
opponent decisions. Each successor is advanced by the authoritative resolver
to the next controlled-cell decision frontier. Explicit Signal and Split are
excluded: Signal is dominated for a solitary cell in the no-food probe, while
Split changes the problem into decentralized multi-cell control.

Run a bounded fixed-seed search with:

```sh
cargo run -p blob_rl --bin micro_combat_search --release -- \
  --config blob_rl/config/micro_combat_ablation_256_v2_base.toml \
  --scenarios blob_rl/config/micro_combat_pursuit_scenarios.toml \
  --scenario one-v-three-pursuer-survival-h16 \
  --seed 930100000 \
  --beam-width 64 \
  --max-depth 20 \
  --max-expansions 250000 \
  --output training-output/micro-combat-search-1v3-pursuer-h16-seed-930100000.json
```

The first run examined 88,862 transitions and 74,060 unique continuations. It
found and independently replayed a nine-Move sequence that reached the 16,384
quanta deadline with 35 stored energy. Beam pruning occurred, so the artifact
is explicitly `truncated: true` and `exhaustive: false`. The maintained evasive
controller retains 57 energy on the same matchup, meaning the search is a
physical-feasibility witness but neither a superior controller nor an
optimality certificate.

The planner now removes effort variants only when their exact effort cost and
bucketed completion time equal an earlier tier for the same action, target, and
payload. Do not use ordinary cost/time Pareto pruning here: spending additional
effort also sheds inertial mass and deposits energy at the origin, so a dearer
tier can change later movement and combat even when it is no faster now. On the
same seed, exact-equivalence pruning reduced the root from 154 to 153 choices
and the search from 88,862 to 88,285 expansions (0.65%), while reproducing the
same nine-Move timeout witness and 35 stored energy. The report records root
counts by action family, and `--disable-equivalent-effort-pruning` provides the
unpruned control. The modest win is preferable to building a faster oracle for
a simplified physics model.

The observation-policy planner lifts that fixed-seed sequence into a bounded
deterministic policy tree. Its table key is the canonical serialized Mind input
with the per-invocation random block zeroed; self state, local observations,
action-space limits, and private memory remain included. The generated policy
stores its decision depth in ordinary cell-private memory. It therefore cannot
branch on canonical coordinates, identity, team membership, global state, or a
shared seed. Every synthesized rule is replayed through the same anonymous
input and private-memory update path used by native and Wasm Minds. The trusted
offline search may score canonical particle outcomes, so reports label this as
privileged particle synthesis of an observation-legal policy, not as an
unprivileged learning algorithm.

Run synthesis seeds and disjoint held-out replays with:

```sh
cargo run -p blob_rl --bin micro_combat_observation_policy --release -- \
  --config blob_rl/config/micro_combat_ablation_256_v2_base.toml \
  --scenarios blob_rl/config/micro_combat_pursuit_scenarios.toml \
  --scenario one-v-three-pursuer-survival-h16 \
  --seed-base 930100000 \
  --seed-count 2 \
  --holdout-seed-base 930101000 \
  --holdout-seed-count 32 \
  --beam-width 8 \
  --max-decisions-per-particle 12 \
  --max-expansions 50000 \
  --output training-output/micro-combat-observation-policy-1v3-pursuer-h16-search2-holdout32.json
```

The first bounded run used 19,890 authoritative transitions to produce nine
observation rules. It survived both synthesis seeds and all 32 disjoint
holdouts to 16,384 quanta with 35 stored energy, and every replay had complete
table coverage. Beam pruning still makes this a witness rather than an
optimality certificate. All 34 runs followed the same nine-decision path and
ended with identical energy; that uniformity also reveals a limitation of this
control. The encountered pursuit trajectory does not expose meaningful hidden
seed variation. Future policy-oracle scenarios must force temporary loss of
contact or otherwise produce genuinely different hidden particles before
multi-seed success is treated as evidence about partial-observation planning.

The first attempt to create such a control exposed a trusted-host continuation
bug rather than a policy result. `BlobEnvCheckpoint` preserved cell random
lineages and decision counters but restored every engine with seed zero, so all
particles shared the same future private-random stream. Checkpoints now include
a redacted, trusted-host-only match-secret continuation. It is hashed as part
of the planner continuation, never enters the canonical replay or Mind ABI, and
must not be published while a match is accepting decisions. Regression tests
require identical restored checkpoints to reproduce random blocks and distinct
match seeds to remain distinct.

Two controls now make observation diversity explicit in the report:

- `one-v-three-searching-pursuer-survival-h16` is a negative control. Merely
  shortening pursuit memory does not create an information problem while the
  three-cell formation maintains continuous contact.
- `one-v-one-searching-pursuer-survival-h16` uses a 5x5 field and a pursuer that
  follows its last-seen direction once before switching to cell-private random
  search. Four synthesis seeds produced three distinct observation histories
  and an eight-rule policy branching at decisions 3 and 4. It won all four
  synthesis runs and 20 of 32 held-out runs. The other 12 encountered an unseen
  exact observation at decision 4, so the report correctly marks holdout table
  coverage incomplete rather than inventing a fallback action.

Increasing synthesis from four to sixteen seeds expanded the table from 8 to
32 rules and exposed branch points through decision 8, but held-out coverage
did not improve: 18 of 32 holdouts completed. The stochastic-aggressor positive
control branches at decisions 1 through 6 and gives zero complete holdout
replays from only four synthesis seeds. These are useful failures. Exact hashes
are appropriate for proving that no hidden state entered a decision, but an
exact lookup table is not a generalizing policy representation. The next
planner slice should keep exact hashes for audit while selecting actions through
a bounded observation abstraction or learned fallback, evaluated strictly on
disjoint holdout particles.

That generalizing layer is now implemented as observation-policy schema 2.
Exact Mind-input hashes remain training-time audit bindings and are never used
to select actions. The executable policy instead sparsely quantizes the fixed
RL observation tensor (omitting its private-random block) and binds the result
to the exact Mind-visible legal-choice catalog. A nearest-rule fallback may
cross quantized feature states, but only at the same private-memory decision
depth and with an identical legal catalog. Its sparse L1 radius is finite,
reported for every replay, and hashed into the policy artifact alongside the
quantization level. This makes an unseen-state action reusable without allowing
the fallback to cross an affordability, target-availability, or temporal
boundary. Missing partitions still fail closed.

Four synthesis particles were insufficient to cover deeper fallback
trajectories: the six-rule policy completed 22 of 32 validation seeds. Sixteen
synthesis particles produced 31 abstract rules and 33 exact audit bindings.
After fixing the fallback radius at 128, a fresh, previously unused 64-seed
holdout completed and survived 63 runs. Across 405 executed holdout decisions,
68 used an exact observation absent from the synthesis bindings; 54 of those
selected a bounded nearest rule, with maximum L1 distance 84, and the remainder
collapsed directly under quantization. The sole incomplete replay reached
decision 9 with a new 130-choice legal catalog, so the policy correctly refused
to borrow a rule trained in another legality partition. This is a useful
near-complete generalization witness, not an optimality claim: beam pruning was
active and the fallback radius was calibrated before the fresh holdout.

The auditable report is
`training-output/micro-combat-observation-policy-v2-1v1-searching-pursuer-h16-search16-fresh-holdout64.json`.

Counterexample-guided acquisition now searches for missing legality partitions
without relaxing that guard. Initial synthesis, acquisition, and final-holdout
seed ranges are pairwise disjoint. Each acquisition pass considers a bounded
number of the deepest legality partitions and may synthesize several alternative
representative seeds from each one. Every candidate is scored over the
same complete acquisition pool. Nondominated candidates are retained in a
bounded Pareto report over coverage, objective success, rule count, and maximum
fallback distance; policy advancement remains lexicographic with coverage
first. A lower-coverage candidate therefore remains inspectable when it offers
a complexity tradeoff but can never replace the incumbent. The final holdout
is evaluated only after this choice is frozen.

This monotonic check is necessary because bounded beam search is not monotonic
in particle count. A preliminary 1,024-seed experiment that appended failures
blindly fell from 1,021 covered acquisition seeds to 1,002 and achieved only
1,009 of 1,024 on its final range. With rollback enabled, the initial policy
again covered 1,021 of 1,024 acquisition seeds. Adding seed `930109558`, a
decision-9 counterexample with a 130-choice catalog, improved that fixed-pool
coverage to 1,022. A second candidate would have reduced it to 1,002 by creating
a new 138-choice tail, so it was rejected. The accepted policy has 33 abstract
rules and 35 exact audit bindings; it subsequently covered and survived all
128 untouched final seeds, making 104 bounded fallback selections with maximum
L1 distance 77. The complete run used 425,760 authoritative transitions.

The acquisition report is
`training-output/micro-combat-observation-policy-acquisition-monotonic-1v1-searching-pursuer-h16-pool1024-final128.json`.
This proves the acquisition machinery can repair a measured partition without
regressing its fixed discovery distribution; it does not prove global
improvement. Earlier paired 512-seed evaluations showed that a smaller
acquisition pool merely shifted rare failures and tied the baseline at 1,021 of
1,024 aggregate.

The schema-2 candidate tournament evaluated two independent representatives of
the same 130-choice decision-9 partition. The incumbent covered 1,021 of 1,024
acquisition seeds with 31 rules. Candidate `930109558` improved coverage to
1,022 with 33 rules and slightly fewer fallback selections; candidate
`930109942` regressed to 1,011 despite using only 32 rules. The bounded Pareto
frontier retained the accepted candidate and incumbent, selected the former,
and reported the rejected alternative. The frozen winner then covered and
survived all 128 new final seeds. This run used 416,274 authoritative
transitions and is recorded at
`training-output/micro-combat-observation-policy-acquisition-pareto-1v1-searching-pursuer-h16-pool1024-final128.json`.

Acquisition schema 3 now varies the search itself as well as the acquired
particle. A bounded, normalized beam-width grid is crossed with every proposed
counterexample seed. All candidates receive the same authoritative-transition
budget and are replayed over the same fixed acquisition pool. Candidate
summaries identify their beam width, and the selected search configuration is
carried into later acquisition rounds and the sealed final replay. The report
binds both the initially requested and ultimately selected configurations, so
a policy synthesized by a wider search cannot be mistaken for the base
configuration.

The first width tournament compared beams 8 and 16 for counterexample seed
`930109558` under a 200,000-transition candidate cap. Beam 8 reproduced the
1,022-of-1,024 acquisition result with 33 rules in 106,271 transitions. Beam 16
hit the full cap, produced 30 rules, and covered only 911 seeds. The fixed-budget
result does not mean wider beams are intrinsically worse: retaining twice as
many partial policies spent the budget before reaching an equally mature
controller. It does show that width 16 is a poor allocation at this compute
budget. The width-8 winner was frozen before evaluation and then covered and
survived all 128 untouched final seeds, using 128 nearest-rule fallbacks with
maximum L1 distance 74. The full tournament used 510,704 authoritative
transitions and is recorded at
`training-output/micro-combat-observation-policy-acquisition-beams-1v1-searching-pursuer-h16-pool1024-final128.json`.

The next planner comparison should vary deterministic tie-breaking at the same
beam and transition budget. If those candidates plateau, a small
information-set MCTS is the more meaningful algorithmic alternative. Beam
widths should only be revisited with an explicitly declared scaling experiment
that gives wider searches proportionally larger transition budgets; that is a
compute-quality curve, not a fixed-budget tournament.

Acquisition schema 4 makes that tie-break comparison explicit. The search
configuration carries an optional deterministic salt: `null` retains the
established canonical lexicographic ordering, while a numeric value hashes the
otherwise equal-scoring policy prefix under a domain-separated salt. This only
changes trusted offline frontier traversal. It is never provided to a Mind and
does not alter the policy ABI, observation abstraction, simulator randomness,
or fixed-pool scoring. Candidate identity and the selected search configuration
bind the optional salt. Canonical ordering is included by default, and the
cross-product of beam widths and tie-break strategies is capped at sixteen
configurations.

Keeping the canonical control was essential. A pilot that silently replaced it
with salted ordering made the apparent incumbent fall from 1,021 to 969 covered
acquisition seeds; that report was discarded before acceptance. In the valid
tournament, the untouched canonical incumbent again covered 1,021 of 1,024.
Adding counterexample `930109558` under canonical ordering covered 1,022. Four
salted beam-8 searches covered 969, 888, 834, and 851 seeds respectively. The
canonical candidate therefore won, reproducing policy hash
`44b70260a66228a05632dff1373959b6ee183547e9a22d3a593b56d63180f6b6`.
After selection was frozen, it covered and survived all 128 seeds in a new
final range, with 93 nearest-rule fallbacks and maximum L1 distance 76. The
six-entry tournament consumed 722,274 authoritative transitions and is
recorded at
`training-output/micro-combat-observation-policy-acquisition-ties-1v1-searching-pursuer-h16-pool1024-final128.json`.

This closes the cheap deterministic beam-search diversification seam: neither
additional width nor salted traversal improved the fixed-budget result. The
next algorithmic slice should prototype a small information-set MCTS/POMCP
search while preserving the same candidate identity, acquisition-pool, and
sealed-final-holdout contracts.

That bounded information-set MCTS prototype is now available through
`--algorithm information-set-mcts`. A tree node owns a partial deterministic
policy keyed only by Mind-visible observation abstractions and private-memory
depth. Hidden canonical states remain a trusted offline particle set used to
advance and score that policy; they never become a node key, action selector,
or runtime input. Expansion binds one legal action to the next unresolved
information set. Seeded UCT selects among expanded policy prefixes, and bounded
random rollouts add observation rules until they terminate, reach the rollout
depth, or exhaust the authoritative-transition budget. The best encountered
partial policy is still chosen by the established lexicographic scenario
objective and replayed through the ordinary Mind boundary.

The search configuration report-binds the algorithm, deterministic MCTS seed,
iteration cap, rollout-rule depth, integer-milli exploration coefficient, and
global simulator-transition cap. Search-report schema 4 records completed
algorithm iterations separately from transitions. A deterministic regression
test repeats the complete search and requires identical policy hashes,
expansion counts, and held-out replay records. Beam remains the default and its
ordering and scoring path are unchanged.

The first fixed-budget characterization used the same sixteen synthesis seeds,
the previously inspected 1,024-seed discovery pool, and a 200,000-transition
cap. MCTS completed 3,998 iterations and 199,986 transitions. It found a
seven-rule exact policy with no observation branching or nearest fallback: two
moves, a wait, and four attacks eliminate the searching pursuer at quanta 8,512.
The training cell survives with only 2 stored energy. That policy covered and
won all 1,024 discovery runs, improving on beam's 1,021 covered runs while using
far fewer executable rules. After those settings were frozen, the identical
policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`
eliminated the opponent in all 128 runs of a fresh final range, again with no
fallback and identical terminal state. Reports are stored at
`training-output/micro-combat-observation-policy-mcts-prototype-1v1-searching-pursuer-h16-search16-pool1024.json`
and
`training-output/micro-combat-observation-policy-mcts-prototype-1v1-searching-pursuer-h16-search16-final128.json`.

This is a strong legal witness, not an optimality result or yet a demonstration
of belief-dependent branching. The controller ends the fight before the
pursuer's hidden search stream can diversify observations, and its 2-energy
margin is thin. The next slice should integrate beam and MCTS as explicit
fixed-pool acquisition candidates and add terminal survival/energy margin to
candidate comparison before selecting between equally successful policies.

Acquisition schema 6 now performs that algorithm tournament. Candidate search
families are explicit and generate algorithm-specific configurations: beam
crosses its width and canonical/salted ordering grid, while information-set
MCTS crosses deterministic simulation seeds and inherits the declared
iteration, rollout, exploration, and transition limits. Candidate and Pareto
identities bind the complete search configuration rather than a beam-specific
subset. The combined grid is normalized, deduplicated, and capped at sixteen
configurations per proposed counterexample.

Fixed-pool summaries now include worst-case and aggregate final training energy
plus worst-case and aggregate simulated survival time. Coverage and objective
success remain the first two selection criteria. For a survival objective,
minimum survival time, minimum terminal energy, and then their aggregate values
rank otherwise equal candidates. For elimination, lower aggregate completion
time precedes retained energy. These margins also participate in Pareto
dominance, so a fragile policy cannot erase a robust policy merely by tying its
binary success count. Raw totals are safe here because every candidate is
replayed on the same complete acquisition pool.

The first direct tournament started from the canonical beam incumbent and
evaluated beam and MCTS after adding the same decision-9 counterexample
`930109558`. The incumbent covered 1,021 of 1,024 seeds; acquired beam covered
1,022 with 33 rules, 682 fallbacks, minimum energy 5, and maximum fallback L1
distance 84. Acquired MCTS covered and won all 1,024 with six exact rules, no
fallback, minimum energy 12, and identical quanta-7,168 eliminations. Its
policy moves twice and attacks four times. Coverage selected MCTS without
needing the new margin tie-break, while all three tradeoffs remain visible on
the Pareto frontier.

The selected configuration and counterexample were frozen before a new
128-seed final range was opened. Re-synthesis reproduced policy hash
`ccd53798de8df54faa11689e37efbcf58659003f2af5a2ffac36c9f5fc082625`;
all 128 replays eliminated the opponent at quanta 7,168 with one training cell,
12 stored energy, and no fallback. The complete tournament used 604,417
authoritative transitions and is recorded at
`training-output/micro-combat-observation-policy-acquisition-algorithms-1v1-searching-pursuer-h16-pool1024-final128.json`.

The immediate search-quality gap is now closed for this control. The next
planner slice should make MCTS memory and wall-clock costs observable, then
force genuinely divergent hidden observations so the oracle must learn a
branching policy rather than an open-loop elimination sequence.

That cost-observability slice is now complete. Search-report schema 5 records
search-only wall time, process peak resident memory, peak retained policy nodes,
particles, rules, and untried choices, a deterministic lower bound over owned
policy/checkpoint buffers, and policy-node/rollout clone counts. Acquisition
schema 7 copies the complete cost record into every candidate summary. Timing
and process RSS are host-dependent diagnostics; RSS is the process-wide high
water mark rather than an isolated allocation delta. Neither those values nor
the deterministic structural costs participate in candidate ordering, Pareto
dominance, or policy identity. A regression test varies all three kinds of cost
without changing candidate selection.

Separate release processes measured the canonical 16-seed, 200,000-transition
control so their RSS high-water marks remain comparable:

| search | wall time | peak RSS | retained-policy lower bound | peak nodes | peak particles | peak rules | peak untried choices | policy-node clones |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| beam, width 16 | 32.44 s | 762.55 MiB | 312.66 MiB | 2,448 | 39,168 | 68,544 | 0 | 66,347 |
| information-set MCTS | 118.16 s | 416.33 MiB | 367.89 MiB | 3,423 | 54,768 | 6,691 | 437,666 | 17,077 |

MCTS was 3.64 times slower at essentially the same transition budget, but used
45% less process peak RSS and made 74% fewer full policy-node clones. Its 3,998
rollouts account for 3,998 of those clones. Beam's largest transient candidate
layer, not its final width-16 frontier, explains its high RSS and clone count.
MCTS retains more checkpoints and a very large legal-choice frontier, so its
owned-buffer lower bound is slightly larger even though its measured process
RSS is lower; allocator reuse and the deliberately excluded tree/container
overhead mean the two memory measures answer different questions.

The MCTS run reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
confirming that observability did not perturb the search. The reports are
`training-output/micro-combat-observation-policy-cost-beam-1v1-searching-pursuer-h16-search16.json`
and
`training-output/micro-combat-observation-policy-cost-mcts-1v1-searching-pursuer-h16-search16.json`.
The next planner slice remains a forced hidden-observation divergence control.
After that witness exists, optimize cloning with shared immutable particle
checkpoints and avoid materializing beam children that cannot survive the
bounded top-k frontier.

The hidden-observation divergence control is now implemented as
`one-v-one-forking-evader-elimination-h2`. Its opponent is an ordinary
anonymous baseline Mind, not scenario-side privileged logic. At first contact
it consumes one cell-private random bit, moves to one of the two perpendicular
local tiles, writes only an `already forked` marker to its own private memory,
and waits thereafter. The controlled cell cannot observe the bit or the
opponent's memory. It can observe the resulting occupancy once the move has
resolved. A focused physics test starts from two seeds with an identical root
Mind input, confirms that their next Mind inputs differ, and proves that the
sets of immediately winning legal responses are both nonempty and disjoint.

Search-report schema 6 distinguishes mere observation multiplicity from
behavioral branching with `policy_choice_branching_depths`. A new diagnostic
`force_open_loop_actions` search constraint requires every observation at a
given private-memory depth to use the same exact action, target, effort, and
amount. The constraint is part of the complete reported search configuration
and candidate identity. It never changes the runtime ABI or exposes hidden
state. Acquisition schema 8 carries the new configuration and report field.

A matched information-set MCTS A/B used 16 synthesis seeds, 64 disjoint exact
holdouts, no nearest-rule fallback, 5,000 iterations, and a 100,000-transition
cap. Both searches completed 5,000 iterations and 58,320 authoritative
transitions:

| policy class | synthesis eliminations | holdout eliminations | executable rules | exact choice branches | holdout result |
| --- | ---: | ---: | ---: | --- | --- |
| observation-contingent | 16/16 | 64/64 | 3 | depth 1 | 64 wins at quanta 2,048 |
| forced open loop | 10/16 | 37/64 | 3 | none | 37 wins, 27 timeouts |

The successful policy uses one shared opening attack, then selects different
attack target/amount choices for the two locally observed fork positions. Its
policy hash is
`cc9243eb0a58dcfbe959a715e1d3d9d1259f7d426ccd16114ef768e4bef93b1e`.
The open-loop control sees the same two exact traces but must bind the same
depth-1 choice to both; it selects the majority branch and times out on the
other. Reports are stored at
`training-output/micro-combat-observation-policy-forking-evader-h2-branching-mcts-search16-holdout64.json`
and
`training-output/micro-combat-observation-policy-forking-evader-h2-open-loop-mcts-search16-holdout64.json`.

This is a constructive local branching witness and a matched bounded-search
gap, not an exhaustive proof that no longer open-loop program could exploit a
different opening under changed timing rules. The next planner work should add
an exact tiny-horizon open-loop certificate if that stronger claim is needed.
For implementation efficiency, the next slice is now shared immutable particle
checkpoints, followed by bounded top-k beam child generation.

Shared immutable particle checkpoints are now implemented. Each search
particle owns an `Arc` to an immutable `SingleCellSearchState`; cloning a policy
node copies only those references. An authoritative transition allocates a new
state for the affected particle and leaves every parent/sibling checkpoint
untouched. This is search-internal structural sharing and does not change
canonical engine checkpoints, replay serialization, Mind inputs, or isolation.
A regression test requires cloned nodes to retain pointer-identical states.

Search-report schema 7 and acquisition schema 9 add deterministic counts for
shared particle-state clones and newly allocated particle states. Retained-byte
estimation now deduplicates shared state allocations by pointer, so one
checkpoint is counted once per measured retained set. The original 16-seed,
200,000-transition controls reproduced both pre-optimization policy hashes and
transition counts:

| search | peak RSS before | peak RSS after | wall before | wall after | shared state clones | new state allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beam, width 16 | 762.55 MiB | 442.61 MiB | 32.44 s | 31.36 s | 1,061,552 | 184,212 |
| information-set MCTS | 416.33 MiB | 416.39 MiB | 118.16 s | 120.33 s | 273,232 | 200,002 |

Beam peak RSS fell 42% because its large transient candidate layer previously
deep-cloned checkpoints more than one million times. Its deterministic retained
lower bound remains about 313 MiB: at the dominant frontier, all sixteen
particles transition for each surviving child, so those resulting checkpoints
are genuinely distinct. MCTS memory and time were effectively unchanged in
this single-run measurement. Its persistent tree retains nearly every
transition result, while its rollouts generally transition all particles
immediately after cloning; pointer sharing removes copy work but not the
dominant retained states or simulation cost.

Reports are stored at
`training-output/micro-combat-observation-policy-cost-shared-checkpoints-beam-1v1-searching-pursuer-h16-search16.json`
and
`training-output/micro-combat-observation-policy-cost-shared-checkpoints-mcts-1v1-searching-pursuer-h16-search16.json`.
The next efficiency slice should bound beam child materialization with an
online top-k frontier. For MCTS, compact/lazy untried-choice representation is
now a higher-value memory seam than further checkpoint sharing.

Beam child materialization is now bounded online. Every generated child is
scored by the unchanged policy comparator and inserted into a preferred-first
frontier only if it belongs to the current best `beam_width` nodes. Inserting a
better child immediately evicts the current worst; a worse child is dropped
without ever joining the frontier. A regression test feeds the same candidates
to this selector and the former materialize-sort-truncate algorithm and
requires an identical ordered top-k result.

Search-report schema 8 and acquisition schema 10 record beam-generated and
beam-pruned child counts. The same isolated release control preserved policy
hash `a42c465f827af7d153fb70ca5419eeb164b0e6ffde6ec7db3d631929ac6c989f`,
all 184,196 transitions, 29 iterations, and the complete executable policy.
Resource use changed substantially:

| beam metric | materialized generation | shared checkpoints | online top-k |
| --- | ---: | ---: | ---: |
| peak RSS | 762.55 MiB | 442.61 MiB | 15.95 MiB |
| deterministic retained lower bound | 312.66 MiB | 312.72 MiB | 2.21 MiB |
| peak retained nodes | 2,448 | 2,448 | 16 |
| peak retained particles | 39,168 | 39,168 | 256 |
| peak retained rules | 68,544 | 68,544 | 448 |
| wall time | 32.44 s | 31.36 s | 30.44 s |

The search generated 66,233 children and pruned 65,785 during their respective
generation. Online retention therefore removes 96% of the post-sharing RSS and
98% relative to the original materialized implementation without reducing
simulator work or search quality. Wall time improves only 3% from the
checkpoint-sharing version because all candidate transitions are still
evaluated; the gain is deliberately memory-first.

The report is stored at
`training-output/micro-combat-observation-policy-cost-online-top-k-beam-1v1-searching-pursuer-h16-search16.json`.
Beam memory is no longer the immediate planner bottleneck. The next slice should
compact MCTS's 437,666 retained untried choices, ideally as a lazy deterministic
choice permutation/cursor rather than a `Vec<PolicyChoice>` per tree node.

That MCTS choice-frontier compaction is now complete. Each node represents the
unchanged deterministic `Vec::swap_remove` order as a remaining-choice count
plus sparse mappings from live positions to original catalog indices. A unit
test applies the same requested removal positions to a materialized vector and
the lazy representation and requires the complete removal order to match.
Legal-choice catalogs are interned globally within the search, so expanding a
node does not restore its simulator merely to regenerate a catalog. This keeps
the memory benefit without the wall-time regression measured for a pure
regeneration prototype.

Search-report schema 9 and acquisition schema 11 distinguish the logical
untried frontier from its physical representation. They report peak sparse
index overrides, a retained MCTS-choice storage lower bound, distinct interned
catalogs, and catalog reuse hits. The same isolated 16-seed, 200,000-transition
MCTS control preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
all 199,986 transitions, and all 3,998 completed iterations:

| MCTS metric | materialized per-node catalogs | lazy, interned catalogs |
| --- | ---: | ---: |
| wall time | 120.33 s | 118.12 s |
| peak RSS | 416.39 MiB | 394.30 MiB |
| logical untried choices | 437,666 | 437,666 |
| retained choice-storage lower bound | 13.43 MiB (retrospective) | 0.244 MiB |
| sparse index overrides | n/a | 2,725 |
| distinct/reused catalogs | n/a | 12 / 3,411 |

Choice-frontier storage therefore fell 98.2%, process RSS fell 5.3%, and wall
time improved 1.8%. The logical branching frontier is unchanged; only its
physical representation changed. The report is stored at
`training-output/micro-combat-observation-policy-cost-lazy-mcts-choices-1v1-searching-pursuer-h16-search16.json`.

MCTS's dominant retained cost is now the policy/checkpoint tree, whose lower
bound remains 385.84 MB, rather than choice storage. The next efficiency slice
should measure and remove duplicated material inside trusted planner
checkpoints, or introduce a compact planner-only state representation, while
preserving canonical replay checkpoints at trust and publication boundaries.

The planner-only checkpoint representation is now compact. A preliminary
attempt to strip duplicated host-cell projections was rejected after the
control showed that short policy/opponent memories made those fields less than
one megabyte of the retained tree. The dominant repetition was instead the
self-contained canonical checkpoint envelope: immutable compiled rules,
format metadata, and integrity fields were embedded alongside every encoded
state.

`ReferenceStateCheckpoint` now retains only the encoded canonical state plus
the hashes and dimensions needed to validate it. On restore, the trusted
planner engine supplies its already-compiled rules; semantic-ruleset,
compiled-ruleset, checkpoint-integrity, and state hashes must all match before
the state is installed. Export simultaneously reconstructs the ordinary
self-contained `ReferenceCheckpoint` bytes. Tests require those bytes to be
identical to the existing public format and require a compact restore to
re-export the complete RL environment checkpoint byte-for-byte. Replay,
server-verification, and artifact boundaries therefore remain unchanged.

The same isolated MCTS control again preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
all 199,986 transitions, and all 3,998 iterations:

| MCTS metric | full retained checkpoints | planner state checkpoints |
| --- | ---: | ---: |
| wall time | 118.12 s | 100.28 s |
| peak RSS | 394.30 MiB | 291.61 MiB |
| retained-policy lower bound | 367.97 MiB | 264.35 MiB |
| peak nodes / particles | 3,423 / 54,768 | 3,423 / 54,768 |

Peak RSS fell 26.0%, the deterministic retained lower bound fell 28.2%, and
wall time improved 15.1%. Relative to the original materialized-choice MCTS
control, RSS is now about 30% lower and runtime about 15% faster. The report is
stored at
`training-output/micro-combat-observation-policy-cost-rules-elided-planner-checkpoints-1v1-searching-pursuer-h16-search16.json`.

The remaining 264 MiB lower bound is primarily repeated full-board canonical
state. The next memory seam is therefore state deltas or chunk-level structural
sharing between parent and child checkpoints, with bounded reconstruction cost;
further envelope compaction will have diminishing returns.

Canonical planner state now uses bounded chunk-level structural sharing. Tiles
are stored in immutable eight-tile chunks; when a transition exports its child
checkpoint, every chunk equal to the corresponding parent chunk reuses the
parent allocation. In the first implementation, the complete canonical cell
table was likewise shared only when wholly unchanged. A changed tile chunk is
copied once. This deliberately avoids a chain of parent-relative deltas: every
checkpoint directly owns one fixed array of chunk references, and restore
always flattens exactly one board's chunks. Restore cost and retained-state
reachability therefore do not grow with search depth.

The public checkpoint remains the same self-contained canonical byte sequence.
Planner export constructs those bytes transiently, while the retained trusted
representation stores typed canonical state and its integrity hashes. Tests
require byte-identical public export, exact state restoration, reuse of all
unchanged chunks after a one-tile mutation, and cell-table reuse when cells did
not change. Search-report schema 10 and acquisition schema 12 add peak unique
allocation and reference counts for tile and cell chunks; these measurements
do not participate in policy identity or candidate selection.

The fixed MCTS control again preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
all 199,986 transitions, and all 3,998 iterations:

| MCTS metric | compact state checkpoints | shared state chunks |
| --- | ---: | ---: |
| wall time | 100.28 s | 107.82 s |
| peak RSS | 291.61 MiB | 201.19 MiB |
| retained-policy lower bound | 264.35 MiB | 149.42 MiB |
| unique / referenced tile chunks | n/a | 82,960 / 219,072 |
| unique / referenced cell tables | n/a | 54,768 / 54,768 |

Peak RSS fell another 31.0% and the retained lower bound fell 43.5%, at a 7.5%
wall-time cost from reconstructing and validating typed chunked state. Only
37.9% of referenced tile chunks were unique at peak. This control changes the
active cell state on every retained transition, so it could not reuse whole
cell-table allocations; cell chunking is a future seam for larger populations.
Relative to the original materialized-choice MCTS control, peak RSS is down
51.7% and wall time is down 8.8%. The report is stored at
`training-output/micro-combat-observation-policy-cost-shared-state-chunks-1v1-searching-pursuer-h16-search16.json`.

This is the preferred memory-bounded planner representation for now. The next
checkpoint optimization should be driven by large-board/population profiles:
consider a measured tile-chunk-size sweep and fixed-size cell chunks before a
more elaborate persistent map. Throughput-sensitive small searches may later
select the compact contiguous representation explicitly, but should not do so
implicitly or change search semantics.

Fixed-size cell chunks are now implemented as the next population-scale seam.
The canonical cell table uses immutable eight-cell chunks and reuses each
unchanged corresponding parent chunk. Empty and one-chunk tables are stored
inline, avoiding a separately allocated reference vector for the common tiny
combat case; larger populations use a flat vector of chunk references. As with
tile chunks, restore performs one bounded flatten and never walks parent state.
Adding, removing, or reordering a cell can invalidate later positional chunks,
but it cannot produce incorrect sharing because reuse requires exact chunk
equality.

A normal 18-cell engine fixture commits one action in the final chunk, requires
the first two chunks to remain pointer-identical, restores through the complete
host checkpoint path, and compares exact canonical state. The existing public
checkpoint-byte and deterministic-policy tests remain unchanged. Search-report
schema 11 and acquisition schema 13 now count actual cell-chunk references
rather than treating each whole cell table as one chunk.

An ignored reproducible characterization retains successive checkpoints while
one distinct cell commits `Wait` per transition:

| world / population / retained states | unshared lower bound | shared lower bound | reduction | unique / referenced cell chunks |
| --- | ---: | ---: | ---: | ---: |
| 256×256 / 8,192 / 257 | 2,541.94 MiB | 46.13 MiB | 98.2% | 1,280 / 263,168 |
| 1024×1024 / 32,768 / 5 | 745.31 MiB | 157.32 MiB | 78.9% | 4,100 / 20,480 |

The 256 case reused 1,023 of 1,024 cell chunks on every parent-to-child
transition. The short 1024 case reused 4,095 of 4,096 each time; its retained
lower bound is dominated by the one full board plus per-checkpoint tile-chunk
reference arrays, not repeated cell state. The manual test is
`characterize_planner_checkpoint_chunk_sharing`; its short 1024 chain is
deliberate so routine characterization does not recreate earlier memory
pressure.

The fixed two-cell MCTS control preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, and 3,998 iterations. Its retained lower bound remained
149.42 MiB and peak RSS was 200.63 MiB. The final inline one-chunk run took
105.42 seconds, 2.2% below the tile-only run's 107.82 seconds and 5.1% above the
compact-contiguous checkpoint control. This confirms that population-scale
sharing does not impose a material small-search penalty. The report is stored at
`training-output/micro-combat-observation-policy-cost-shared-cell-chunks-1v1-searching-pursuer-h16-search16.json`.

The per-checkpoint tile-reference profile and flat chunk-size sweep are now
complete. `characterize_flat_tile_chunk_sizes` evaluates the retained reference
array plus copied tile data for deterministic single-tile, 3×3, 5×5, and
dispersed-one-percent footprints:

| world | single-tile optimum | 3×3 optimum | 5×5 optimum | dispersed 1% optimum |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 64 tiles / 25.0 KiB | 64 / 43.6 KiB | 32 / 56.9 KiB | 8 / 866.0 KiB |
| 1024×1024 | 256 / 100.0 KiB | 256 / 172.6 KiB | 128 / 220.4 KiB | 8 / 13.52 MiB |

There is no robust single flat chunk size. Enlarging chunks helps local actions
by shrinking every checkpoint's reference array, but makes a spatially
dispersed passive-field update copy mostly unchanged tiles. Keeping eight-tile
chunks is correct for the latter and unnecessarily expensive for the former.

Planner tiles therefore now use a fixed-depth two-level representation. Fine
immutable chunks still contain eight tiles. Immutable pages contain up to 256
fine-chunk references, covering at most 2,048 row-major tiles, and checkpoints
retain only page references. An unchanged page is shared wholesale. A changed
page copies at most 4 KiB of fine references, then copies only the changed
eight-tile chunks. There are no parent pointers: restore always flattens one
page layer and one chunk layer, so reconstruction remains bounded independently
of search depth.

A 64×64 two-page conformance test changes one tile and requires one page and
511 of 512 fine chunks to remain pointer-identical, followed by exact public
checkpoint and restored-state equality. Search-report schema 12 and acquisition
schema 14 add unique/reference tile-page telemetry. Retained-byte accounting
deduplicates page reference arrays and fine tile allocations independently.

The large-population characterization changed as follows:

| world / population / retained states | flat fine chunks | hierarchical pages | additional reduction | unique / referenced pages |
| --- | ---: | ---: | ---: | ---: |
| 256×256 / 8,192 / 257 | 46.13 MiB | 14.25 MiB | 69.1% | 32 / 8,224 |
| 1024×1024 / 32,768 / 5 | 157.32 MiB | 149.36 MiB | 5.1% | 512 / 2,560 |

The 1024 result is now almost entirely the one 144 MiB canonical tile array;
only a five-state chain was retained. Longer large-world searches benefit more
because each additional unchanged checkpoint retains 512 page references
instead of 131,072 fine-chunk references.

The deliberately adverse 5×5 MCTS control has one changed page per particle
state and therefore cannot share pages. It nevertheless preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, and 3,998 iterations. Wall time was 103.68 seconds, peak
RSS 204.53 MiB, and the retained lower bound 150.26 MiB. Relative to the flat
cell-chunk run, runtime improved 1.6% while the extra one-page allocation per
state raised the lower bound by 0.56% and RSS by 1.95%. The report is
`training-output/micro-combat-observation-policy-cost-hierarchical-tile-chunks-1v1-searching-pursuer-h16-search16.json`.

Internal planner transitions no longer construct public checkpoint bytes.
Previously `ReferenceStateCheckpoint` encoded the complete canonical state,
rules, checkpoint body, and integrity trailer; `BlobEnv` then embedded those
bytes in JSON solely to hash a continuation identity. This repeated full-board
encoding even though the typed state chunks were retained separately and no
public artifact was being emitted.

Planner continuation identity schema 2 is a domain-versioned streamed JSON
digest over dimensions, semantic- and compiled-ruleset hashes, canonical state
hash, engine iteration, sorted host-cell continuation metadata, episode step,
and the host-private randomness checkpoint. Serialization writes directly into
SHA-256 and never allocates a canonical checkpoint or identity byte buffer.
The digest still binds every component required for exact future continuation,
including secrets that are deliberately absent from canonical replay.

The trusted typed checkpoint no longer stores a redundant checkpoint-body
hash. Restore validates the supplied semantic and compiled ruleset hashes,
reconstructs typed state, and recomputes the canonical state hash. Its fields
remain private and it never crosses a publication or untrusted-input boundary.
Ordinary `ReferenceCheckpoint` export, parsing, body-integrity validation,
replay, and server verification are unchanged. Tests restore the compact state
and require the resulting ordinary RL checkpoint to be byte-identical to the
one exported before compaction.

Bounded-search schema 2, observation-policy search schema 13, and acquisition
schema 15 identify the new continuation digest. The digest is used as a
deterministic deduplication key and, in bounded beam search, an arbitrary final
tie-break; consequently old and new final continuation hashes are not directly
comparable. It is never a Mind input or physics input. The fixed observation
policy control preserved its policy hash, all choices, 199,986 transitions,
3,998 iterations, and replay outcomes:

| MCTS metric | public-byte identity | streamed structured identity |
| --- | ---: | ---: |
| wall time | 103.68 s | 75.38 s |
| peak RSS | 204.53 MiB | 203.08 MiB |
| retained-policy lower bound | 150.26 MiB | 148.59 MiB |

Wall time fell 27.3%. Removing the redundant 32-byte checkpoint hash from each
retained state reduced the structural lower bound 1.1%; the main win is avoided
encoding bandwidth rather than retained memory. The report is
`training-output/micro-combat-observation-policy-cost-streaming-continuation-identity-1v1-searching-pursuer-h16-search16.json`.

The release large-world characterization improved even more because it exports
many checkpoints with large canonical tile arrays:

| world / population / retained states | encoded identity | streamed identity | improvement |
| --- | ---: | ---: | ---: |
| 256×256 / 8,192 / 257 | 5.543 s | 0.162 s | 97.1% / 34.2× |
| 1024×1024 / 32,768 / 5 | 1.687 s | 0.088 s | 94.8% / 19.2× |

Transient canonical-state materialization has now been removed from the planner
checkpoint path. `ReferenceStateCheckpoint::from_simulation` reads the immutable
live tile slice directly and compares it with parent pages/chunks. It traverses
the authoritative `CellStore` in canonical key order, using one reusable
eight-entry borrowed scratch buffer; unchanged chunks are shared without first
cloning their `CellState`s, while changed chunks clone only their own entries.
The checkpoint records the live clock and next-cell key through immutable
crate-private accessors.

This does not expose a new Mind or public-host capability. The views are used
only synchronously by the trusted engine while it holds the simulation
immutably. Canonical ordering remains the `CellStore`'s ordered active-key
iteration, the precomputed state hash is still bound into continuation
identity, and restore/public-export conformance tests remain byte-exact.

The release large-world checkpoint characterization improved by roughly
another factor of two without changing retained bytes or allocation sharing:

| world / population / retained states | cloned canonical views | direct live views | improvement |
| --- | ---: | ---: | ---: |
| 256×256 / 8,192 / 257 | 0.162 s | 0.083 s | 48.8% / 1.95× |
| 1024×1024 / 32,768 / 5 | 0.088 s | 0.047 s | 46.6% / 1.87× |

Relative to the former public-byte identity path, these workloads are now
about 66.8× and 35.9× faster respectively. The fixed 5×5 MCTS control was
neutral, as expected for 25 tiles and two cells: 75.35 seconds versus 75.38
seconds, with identical policy hash, 199,986 transitions, 3,998 iterations,
allocation telemetry, and 148.59 MiB retained lower bound. Peak RSS moved from
203.08 to 203.70 MiB, within normal host-run variance. The report is
`training-output/micro-combat-observation-policy-cost-direct-live-checkpoint-views-1v1-searching-pursuer-h16-search16.json`.

The remaining parent-comparison scan has now been removed from compatible
incremental checkpoints. A separate lazy checkpoint-mutation tracker records
changed tile chunks, changed cells, and the earliest cell-key structural edit.
It is independent of incremental hashing: a hash refresh cannot consume its
state, and checkpoint export cannot make a hash clean. Tracking is dormant for
ordinary simulation and activates only after a trusted planner root is
exported or restored.

Each tracker carries an opaque process-local mutation token. Cloned simulations
share their last common token, but the first mutation on either branch receives
a globally unique successor. A child can therefore use the sparse path only
when its proposed parent is the exact checkpoint base of that branch. Stale,
unrelated, and sibling-branch parents automatically use the full construction
path. Restoring a trusted checkpoint adopts its token so its next child remains
incremental. These tokens and summaries are host-only acceleration metadata;
they never enter canonical state, hashes, replay, serialized checkpoints, or a
Mind input.

For a compatible child, the tile builder clones only the small top-level page
reference vector and opens pages named by changed eight-tile chunks. The cell
builder locates changed packed chunks by canonical cell key. Ordinary cell
updates rebuild only those chunks; an insertion or deletion preserves the
unaffected prefix and rebuilds from the earliest structurally affected chunk,
which is necessary because packed canonical chunks can shift after a deletion.
Exact tests cover one-tile and one-cell edits, append-only births, continuation
after restore, stale parents, and divergent clone branches.

The manual release characterization now reports the exact chunks examined by
incremental construction. Its transition loop committed one cell action per
child, so all 260 incremental states visited one cell chunk and zero tile
chunks each:

| world / population / retained states | direct live scan | mutation summary | improvement |
| --- | ---: | ---: | ---: |
| 256×256 / 8,192 / 257 | 0.083 s | 0.010 s | 88.0% / 8.3× |
| 1024×1024 / 32,768 / 5 | 0.047 s | 0.030 s | 36.2% / 1.57× |

The 1024×1024 result is now dominated by its single unavoidable full root
snapshot; it retains only four incremental children. Relative to the former
public-byte identity path, the complete workloads are about 554× and 56×
faster respectively.

The fixed small-world MCTS control remains compute-bound rather than checkpoint
scan-bound. It reproduced the exact policy hash, all 199,986 transitions, 3,998
iterations, tree/allocation counters, and replay outcomes. Wall time was 77.67
seconds versus 75.35 seconds in the preceding run, within host variance; peak
RSS was 203.66 MiB. The one retained process-local token plus structure padding
adds 16 bytes per compact checkpoint, increasing the deterministic retained
lower bound by 0.84 MiB (0.56%) to 149.42 MiB. Per-construction visit counters
are test-only and add nothing to production checkpoints. The report is
`training-output/micro-combat-observation-policy-cost-checkpoint-mutation-summaries-1v1-searching-pursuer-h16-search16.json`.

Planner restoration is now explicitly profiled. Search-report schema 14 and
acquisition schema 16 record the exact restore count and canonical tile/cell
volume plus host-dependent nanoseconds for tile materialization, cell
materialization, resolver reconstruction, integrity validation, canonical
restore total, and complete environment restore total. The outer total includes
the canonical phases, blank environment construction, host metadata install,
and projection refresh; it must not be added to the nested canonical total.
Like wall time and RSS, phase timings do not participate in policy selection or
identity.

The release state-family matrix gives average canonical restore time:

| world / population | total | tile flatten | cell flatten | resolver reconstruction | integrity validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256×256 / 0 | 13.71 ms | 0.30 ms | <0.001 ms | 13.40 ms | 0.002 ms |
| 256×256 / 8,192 | 15.07 ms | 0.30 ms | 0.03 ms | 14.74 ms | 0.002 ms |
| 1024×1024 / 0 | 251.23 ms | 15.44 ms | <0.001 ms | 235.78 ms | 0.002 ms |
| 1024×1024 / 32,768 | 237.06 ms | 8.53 ms | 0.21 ms | 228.31 ms | 0.002 ms |

Resolver reconstruction accounts for 94–98% of these large-world restores;
cell materialization is negligible. This bucket validates canonical resources,
recompiles neighborhood/diffusion topology, rebuilds passive and due-action
indexes, allocates dense scratch storage, and initializes the incremental hash.
The weak population effect and strong board-size effect argue against beginning
with copy-on-write cell storage. Immutable topology/allocation reuse is the
more promising large-world seam.

The fixed MCTS control again reproduced the exact policy hash, 199,986
transitions, 3,998 iterations, and all deterministic tree/allocation counters.
It took 75.32 seconds and performed 1,790,334 complete environment restores:
8.95 restores per authoritative transition. Those restores processed exactly
44,758,350 tiles and 3,580,668 cells, confirming 25 tiles and two cells per
restore. Complete environment restoration consumed 40.68 seconds (54.0% of
search wall time); the nested canonical restore consumed 15.22 seconds (20.2%),
of which resolver reconstruction was 14.25 seconds (18.9%). Tile and cell
flattening together used only 0.30 seconds. Peak RSS was 202.55 MiB and retained
checkpoint bytes were unchanged. The report is
`training-output/micro-combat-observation-policy-cost-restore-profile-1v1-searching-pursuer-h16-search16.json`.

The immediate next slice should therefore consolidate observation, legality,
effort-pruning, and transition validation around one restored frontier view.
The current 8.95× restore amplification is a larger and safer target than a
chunk-backed mutable simulation. After that amplification is removed, reuse
the engine's already compiled immutable neighborhood/diffusion topology and
appropriately sized scratch allocations during canonical restore.

That frontier-consolidation slice is now complete. A single restored planner
frontier derives the anonymous zero-randomness Mind input, the exact legal
choice catalog used by observation abstraction, and the optionally
effort-pruned search catalog. Information-set selection retains the derived
search catalog for its selected particle set instead of restoring those
particles again. Bound-rule advancement and replay likewise derive observation
and legality together. The real observation-policy transition still prepares
the ordinary randomness-bearing Mind input, but now validates its choice from
that already prepared input rather than reconstructing the same checkpoint a
second time. No canonical coordinates, identities, opponent state, or shared
hidden data enter the reusable frontier, policy key, or action catalog.

The identical release MCTS control preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, and every deterministic tree/allocation
counter. Complete environment restores fell from 1,790,334 to 607,860: from
8.95 to 3.04 restores per authoritative transition, a 66.0% reduction. Search
wall time fell from 75.32 to 46.19 seconds (38.7%), complete environment
restore time from 40.68 to 14.08 seconds, and nested canonical restore time
from 15.22 to 5.25 seconds. Peak RSS and retained-policy structure were
effectively unchanged. The report is
`training-output/micro-combat-observation-policy-cost-frontier-consolidation-1v1-searching-pursuer-h16-search16.json`.

The remaining 3.04× is now structurally interpretable: frontier analysis and
each authoritative successor still need separate mutable environments, and
some search normalization revisits a state. The next safe optimization seam is
compiled immutable topology and scratch-allocation reuse during resolver
reconstruction. Cross-node frontier memoization should follow only with an
explicit memory bound, because the MCTS control already retains tens of
thousands of checkpoints and unbounded observation caching would trade the
recovered time for uncontrolled memory growth.

Compiled-topology reuse is now implemented. A reference simulation can export
an opaque, clone-cheap handle containing its compiled neighborhood targets and
forward/reverse diffusion adjacency. The handle contains no tiles, cells,
resources, clock, randomness, private memory, or pending actions. Reuse is
accepted only when its board dimensions and compiled-ruleset hash match the
restoring simulation. Compact engine restore automatically retains the current
handle, and the single-cell planner retains one handle per fixed scenario
simulator so even construction of its temporary environments avoids topology
recompilation.

The identical 16-seed MCTS control again preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, and every deterministic tree/allocation
counter. Restore count remains 607,860, as expected, while search wall time
fell from 46.19 to 41.82 seconds (9.5%). Complete environment restore time fell
from 14.08 to 9.91 seconds, nested canonical restore from 5.25 to 3.40 seconds,
and resolver reconstruction from 4.92 to 3.06 seconds. Relative to the
pre-consolidation 75.32-second control, the combined slices reduce wall time by
44.5%. The report is
`training-output/micro-combat-observation-policy-cost-shared-topology-1v1-searching-pursuer-h16-search16.json`.

Release restore characterization confirms that the seam matters more as worlds
grow:

| world / population | compiled each restore | shared topology | reduction |
| --- | ---: | ---: | ---: |
| 256×256 / 0 | 14.15 ms | 7.03 ms | 50.3% |
| 256×256 / 8,192 | 15.34 ms | 8.36 ms | 45.5% |
| 1024×1024 / 0 | 249.62 ms | 130.18 ms | 47.8% |
| 1024×1024 / 32,768 | 244.01 ms | 128.48 ms | 47.3% |

At 1024×1024, the remaining shared-topology resolver reconstruction still
costs roughly 117 ms. It now consists primarily of canonical validation,
derived passive/due-action index rebuilding, incremental-hash initialization,
and dense scratch initialization rather than geometry compilation. The next
restore slice should split those residual phases, then reuse bounded scratch
allocations or construct derived indexes directly from checkpoint summaries;
canonical validation must not be skipped merely because planner checkpoints
are trusted host objects.

That residual phase split is now complete. Observation-policy search-report
schema 15 and acquisition schema 17 add nested timings for topology validation,
cell validation/store construction, tile validation, tile passive indexes,
cell passive/due-action indexes, incremental-hash initialization, scratch
initialization, and metabolic-deadline indexing. The fields are nested within
resolver reconstruction and deliberately exclude unclassified orchestration;
they must not be added to the enclosing restore total. As with all prior timing
fields, they are host diagnostics and do not affect policy selection or
identity.

The 1024×1024 shared-topology matrix resolves the ambiguity:

| population | resolver total | hash initialization | tile passive indexes | tile validation | cell validation/store | scratch initialization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 119.58 ms | 104.86 ms (87.7%) | 8.95 ms | 5.74 ms | <0.001 ms | 0.03 ms |
| 32,768 | 120.57 ms | 103.85 ms (86.1%) | 9.06 ms | 4.51 ms | 2.08 ms | 0.27 ms |

The same result holds at 256×256: hash initialization consumes 6.34 of 6.78 ms
for an empty world and 7.20 of 8.30 ms with 8,192 cells. Scratch allocation is
not presently a useful target. The incremental hash builds page commitments
for every tile even though the trusted planner checkpoint already carries a
verified root hash; simply skipping that work would be incorrect because later
incremental mutations require the page tree and per-cell commitments.

The fixed MCTS control preserved policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, 607,860 restores, and all deterministic
tree counters. It took 41.91 seconds, within host variance of the 41.82-second
uninstrumented-topology run. Of 3.19 seconds of resolver reconstruction, hash
initialization used 2.22 seconds (69.5%) and repeated topology/rules validation
used 0.47 seconds (14.8%). The report is
`training-output/micro-combat-observation-policy-cost-restore-subphases-1v1-searching-pursuer-h16-search16.json`.

The next optimization should therefore make the planner checkpoint carry a
shareable, exact incremental-hash seed: page commitments/Merkle nodes and cell
memory commitments corresponding to its already verified state root. Restore
can then clone or copy-on-write that derived seed rather than hash the full
board. The seed must remain host-only, be tied to the checkpoint's compiled
ruleset and state hash, and retain a full recomputation oracle in tests. A
smaller follow-up can replace per-restore compiled-ruleset hashing with an
equality-checked topology provenance token.

That hash-seed slice is now complete with a deliberately narrower retained
certificate. Trusted planner checkpoints chunk and share tile-page
commitments, which eliminates the expensive full-board page hashing. Restore
rebuilds the small Merkle interior from those leaves before checking the root.
Cell leaves, private
memory commitments, cell page hashes, and the cell Merkle tree are rebuilt from
the canonical cells on restore: retaining those arrays initially cut a few more
microseconds but inflated the small MCTS retained lower bound from about 149
MiB to 395 MiB. The compact tile-only form preserves the large-world gain
without making policy-tree memory scale with per-cell commitment pages.

The seed is process-local derived data and is never emitted by the public
checkpoint, replay, or server-verification formats. Admission checks its
compiled-ruleset hash and tile/cell page geometry; normal restore
validation still checks the resulting state root against the checkpoint. Tests
restore independently changed sibling branches, compare each with a fresh full
canonical recomputation, mutate the restored caches again, and compare the new
roots with another full oracle. This is acceleration metadata, not a new trust
boundary or a source of Mind-visible state.

Seed retention is enabled only for worlds spanning at least two 256-tile hash
pages. A 5x5 combat search would otherwise retain a unique tiny tree for nearly
every state: the measured 4% wall-time gain cost about 11% more retained policy
memory. Small curricula therefore keep full hash reconstruction and essentially
their previous memory profile; larger worlds use the shared seed automatically.
Search-report schema 16 and acquisition schema 18 expose seed-materialization
time and retained seed chunk reference/unique counts.

The final 5x5 fixed MCTS control confirms the admission behavior: it retained
zero hash-seed chunks, reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
all 199,986 transitions, 3,998 iterations, and every deterministic allocation
counter. Its retained lower bound is byte-for-byte unchanged at 156,680,524
bytes (149.42 MiB), peak RSS was 203.27 MiB, and the 42.66-second wall time is
within host variance of the 41.91-second residual-profile control. The report
is `training-output/micro-combat-observation-policy-cost-admitted-tile-hash-seed-1v1-searching-pursuer-h16-search16.json`.

The final release restore matrix (including tile/cell flattening and all
canonical validation) is:

| world / population | before tile seed | tile-seeded restore | reduction | seed copy | hash initialization |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 / 0 | 7.03 ms | 0.77 ms | 89.1% | 0.002 ms | 0.024 ms |
| 256x256 / 8,192 | 8.36 ms | 2.14 ms | 74.4% | 0.001 ms | 0.68 ms |
| 1024x1024 / 0 | 130.18 ms | 28.40 ms | 78.2% | 0.014 ms | 0.36 ms |
| 1024x1024 / 32,768 | 128.48 ms | 26.67 ms | 79.2% | 0.012 ms | 2.10 ms |

The remaining large-world time is now canonical tile validation, passive-field
index construction, and tile flattening rather than hashing. Cell commitment
reconstruction remains population-proportional and accounts for the nonzero
hash phase in the 32,768-cell case. The next restore optimization should target
validated checkpoint-derived passive indexes or avoid flattening tile chunks;
either requires another explicit memory/correctness tradeoff rather than
skipping canonical validation.

The validation/index-fusion slice is now complete without adding checkpoint
metadata. Canonical cell validation records sorted digestion, metabolism,
zero-energy, and due-action keys while constructing the cell store, then
bulk-builds the ordered indexes. Canonical tile validation records growing
plants and active signal/diffuse tiles in the same traversal. Every previous
invariant and the full state-root check remain in place; this removes redundant
whole-state scans rather than trusting a derived checkpoint claim.

Search-report schema 17 and acquisition schema 19 replace the former separate
validation and passive-index timings with exact fused cell and tile phases.
The release matrix relative to the tile-seed slice is:

| world / population | tile seed only | fused validation/indexes | reduction | fused tile phase | tile flatten |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 / 0 | 0.77 ms | 0.56 ms | 27.3% | 0.21 ms | 0.32 ms |
| 256x256 / 8,192 | 2.14 ms | 1.87 ms | 12.6% | 0.21 ms | 0.32 ms |
| 1024x1024 / 0 | 28.40 ms | 20.03 ms | 29.5% | 6.12 ms | 13.48 ms |
| 1024x1024 / 32,768 | 26.67 ms | 21.29 ms | 20.2% | 5.13 ms | 10.91 ms |

The fixed 5x5 MCTS control again reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, all deterministic counters, and the
unchanged 149.42 MiB retained lower bound. Its 42.95-second wall time and 204.23
MiB peak RSS are within run variance of the preceding control. The report is
`training-output/micro-combat-observation-policy-cost-fused-restore-indexes-1v1-searching-pursuer-h16-search16.json`.

Tile flattening is now the largest individually measured restore phase: about
62% of the empty 1024x1024 restore and 51% of the 32,768-cell restore. The next
bounded experiment should compare serial and Rayon tile materialization above
an explicit board-size threshold. A chunk-backed mutable simulation remains a
larger architectural option only if parallel flattening cannot recover enough
of this cost.

The thresholded parallel tile-materialization slice is complete. The first
nested `flat_map` implementation was rejected: Rayon collection overhead made
it slower than the serial copy even at 1024x1024. The retained kernel instead
uses an indexed parallel range over canonical tile positions. Rayon can
therefore allocate the output once and write disjoint ordered ranges directly;
there is no unsafe initialization and no change to canonical row-major order.
A direct equivalence test covers tiny, page-aligned, and irregular multi-page
worlds with distinct terrain, loose energy, and all four signal channels.

Release crossover sweeps on this host found the indexed kernel faster by
384x384 with 16 workers. With only two workers it was approximately break-even
at 512x512 and 20% faster at 1024x1024, so the production threshold is the more
conservative 512x512. Worlds below it retain the serial path, and single-thread
hosts always use serial materialization. At 16 workers the isolated 1024x1024
copy fell from 4.42 ms to 3.52 ms (20.4%); the interleaved full restore matrix
showed the following end-to-end result:

| world / population | fused validation/indexes | parallel materialization | reduction | tile flatten after |
| --- | ---: | ---: | ---: | ---: |
| 256x256 / 0 | 0.56 ms | serial path | unchanged by policy | 0.39 ms in this run |
| 256x256 / 8,192 | 1.87 ms | serial path | unchanged by policy | 0.44 ms in this run |
| 1024x1024 / 0 | 20.03 ms | 18.69 ms | 6.7% | 10.72 ms |
| 1024x1024 / 32,768 | 21.29 ms | 19.99 ms | 6.1% | 7.49 ms |

The fixed 5x5 MCTS control remained below the threshold and again reproduced
policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic counter, and the
unchanged 149.42 MiB retained lower bound. Its 43.12-second wall time and
203.92 MiB peak RSS are within the existing control variance. The report is
`training-output/micro-combat-observation-policy-cost-parallel-tile-materialization-1v1-searching-pursuer-h16-search16.json`.

Parallel flattening recovers a real but intentionally bounded fraction of
large-world restore time. The remaining 1024x1024 cost is split between memory
materialization and validated tile/index construction. Removing substantially
more now requires either parallel canonical validation with deterministic error
selection and index assembly, or a chunk-backed mutable tile store that avoids
flattening entirely; the latter is the larger architectural step.

The parallel canonical tile-validation/index slice is now complete. Restore
divides worlds of at least 512x512 into ordered 16,384-tile partitions. Each
worker applies every canonical tile invariant in the original rule order and
builds local growing-plant, signal, and diffuse-energy index fragments. The
host collects partitions in row-major order, returns the first partition's
first error, and appends index fragments without sorting. Consequently both
valid derived indexes and malformed-state error selection match the serial
oracle exactly. WASM and smaller worlds retain the serial implementation.

The initial 2,048-tile partition size performed well on empty worlds but was
rejected after a dense-frontier check exposed excessive temporary-vector
allocation. At the retained 16,384-tile size, the final 1024x1024 isolated
release measurements are:

| workers / passive-index density | serial | parallel | reduction |
| --- | ---: | ---: | ---: |
| 16 / sparse | 3.86 ms | 1.31 ms | 66.0% |
| 16 / every tile active | 6.63 ms | 4.69 ms | 29.2% |
| 2 / sparse | 3.70 ms | 2.00 ms | 46.0% |
| 2 / every tile active | 6.26 ms | 5.34 ms | 14.7% |

The combined parallel-flattening and parallel-validation restore matrix is:

| world / population | parallel flatten only | parallel validation/indexes | reduction | final tile validation/index phase |
| --- | ---: | ---: | ---: | ---: |
| 256x256 / 0 | serial path | serial path | unchanged by policy | 0.22 ms |
| 256x256 / 8,192 | serial path | serial path | unchanged by policy | 0.23 ms |
| 1024x1024 / 0 | 18.69 ms | 13.57 ms | 27.4% | 1.76 ms |
| 1024x1024 / 32,768 | 19.99 ms | 13.11 ms | 34.4% | 1.48 ms |

Relative to the last wholly serial validation/index matrix, large-world restore
is down 32.3% empty and 38.4% at 32,768 cells. The exact-order regression test
uses sparse membership across all three derived indexes and injects different
invalid states into separate partitions to prove that parallel scheduling
cannot change the selected canonical error.

The fixed 5x5 MCTS control again remained below the threshold and reproduced
policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, all deterministic allocation counters,
and the unchanged 149.42 MiB retained lower bound. Its 42.97-second wall time
and 204.56 MiB peak RSS remain within control variance. The report is
`training-output/micro-combat-observation-policy-cost-parallel-tile-validation-1v1-searching-pursuer-h16-search16.json`.

Tile materialization is again the dominant 1024x1024 restore phase, at roughly
6-11 ms depending on cache state and population. Further substantial progress
requires the architectural chunk-backed mutable tile store, or a narrower
planner representation that can execute transitions without reconstructing a
fully flat canonical tile vector. Either change must preserve public flat
checkpoint/replay encodings and deterministic server reexecution.

That chunk-backed live tile-store slice is now complete. The authoritative
simulation stores tiles in canonical row-major pages of immutable eight-tile
chunks. Cloning a simulation or completion snapshot shares those allocations;
the first write detaches only its page directory and changed chunk. Trusted
planner restore consumes the checkpoint's existing tile pages directly rather
than flattening and then immediately rebuilding the same geometry. The public
canonical state, checkpoint, replay, hash, browser-WASM, and server-verification
formats remain flat and byte-for-byte unchanged. Hashing now reads through a
small storage-independent tile source, so the representation is host-only and
cannot alter Mind observations or authoritative commitments.

Tests cover irregular row-major geometry, exact page/chunk adoption, one-chunk
COW detachment, sibling isolation, unchanged allocation reuse after a restore,
full hash recomputation after branching, and all existing canonical golden
vectors. The complete all-feature workspace suite, warnings-denied native
Clippy, warnings-denied `wasm32-unknown-unknown` Clippy, and release `blob_web`
WASM build pass.

The release shared-topology restore matrix is:

| world / population | parallel validation | chunk-backed adoption | reduction | tile adoption | tile validation/index |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 / 0 | 0.56 ms | 0.268 ms | 52.1% | 0.005 ms | 0.236 ms |
| 256x256 / 8,192 | 1.87 ms | 1.577 ms | 15.7% | 0.006 ms | 0.237 ms |
| 1024x1024 / 0 | 13.57 ms | 2.180 ms | 83.9% | 0.066 ms | 1.725 ms |
| 1024x1024 / 32,768 | 13.11 ms | 6.620 ms | 49.5% | 0.087 ms | 1.727 ms |

The sparse completion-frontier diagnostic also falls from the last documented
2.708 ms to 0.131 ms per batch (20.7x); its complete tile prestate falls from
0.259 ms to 0.001 ms because the snapshot is now an allocation-sharing clone.
This is especially relevant to large online worlds with small simultaneous
completion frontiers.

The fixed 5x5 MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, and every deterministic tree/allocation
counter. Across 607,860 restores, measured tile materialization/adoption time
fell from 98.62 ms to 11.67 ms (88.2%); total planner restore time fell from
3.616 s to 3.491 s and wall time from 42.97 s to 42.24 s. The retained lower
bound remains byte-for-byte unchanged at 149.42 MiB. The report is
`training-output/micro-combat-observation-policy-cost-chunk-backed-live-tiles-1v1-searching-pursuer-h16-search16.json`.

The representation exposes a deliberate next seam. Eight-tile allocations
favor sparse planner branches and snapshot isolation, but the warm 512x512
dense passive benchmark is now 218 ms for eight combined steps versus the
previously documented 161 ms contiguous-vector run, while still retaining a
2.01x parallel-over-serial speedup. Before large-world training depends on
dense fields, add a page-base-plus-sparse-overlays representation (or another
measured compaction path) so dense kernels traverse contiguous pages while
sparse writes retain eight-tile COW granularity. Do not silently increase the
chunk size: that would trade the recovered locality for policy-tree memory and
must be evaluated against the same fixed MCTS retained-memory control.

The contiguous-page/sparse-overlay follow-up is now complete. Each logical
2,048-tile page owns one contiguous immutable base and a fixed logical map of
optional eight-tile replacements. A unique simulation writes directly into its
contiguous base. A sparse child of a shared snapshot clones only the page
header and written eight-tile overlay. Before a dense mutable pass, any sparse
overlays are folded once into a new contiguous base; the kernel then traverses
that page as an ordinary slice. Immutable pages without overlays likewise use
a direct slice iterator. Planner checkpoints adopt the same page objects, and
allocation accounting distinguishes shared base ranges from changed overlays.

The direct locality diagnostic performs 64 mutable passes over 262,144 tiles.
The page-overlay store measured 27.34 ms versus 32.61 ms for the flat-vector
oracle (0.84x in this run), demonstrating that the retained live abstraction no
longer imposes a dense traversal penalty. In the broader eight-step passive
suite, the three tile-heavy phases changed as follows relative to the initial
eight-allocation store:

| phase | eight-tile allocations | contiguous pages | reduction |
| --- | ---: | ---: | ---: |
| plant growth | 21.73 ms | 16.57 ms | 23.8% |
| signal decay | 79.72 ms | 62.80 ms | 21.2% |
| diffusion | 31.72 ms | 29.29 ms | 7.7% |
| complete passive suite | 218.32 ms | 197.46 ms | 9.6% |

The complete suite retains a 2.13x parallel-over-serial gain. Its remaining gap
from the older 160.9 ms historical measurement is not reproduced by the direct
tile-store comparison; metabolism and digestion account for most of the
current non-tile time and should be isolated before attributing that difference
to storage.

Sparse behavior is preserved. The 50,000-cell/eight-completion diagnostic is
0.115 ms per batch with a 0.001 ms prestate snapshot. Large trusted restores
remain about 2.15 ms for an empty 1024x1024 world and 6.43 ms with 32,768 cells.
The 257-state 256x256 checkpoint characterization retains all 2,097,152
unchanged logical tile-chunk references while reducing measured shared bytes
slightly from 14,998,392 to 14,932,600. The corresponding five-state 1024x1024
case retains all 524,288 parent tile-chunk references at 156,726,808 shared
bytes.

The fixed MCTS control again reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, and every deterministic tree/allocation
counter. The retained lower bound decreased from 156,680,524 to 156,242,380
bytes (0.28%). Its 42.84-second wall time and 216.28 MB peak RSS remain within
the established run variance. The report is
`training-output/micro-combat-observation-policy-cost-page-overlay-live-tiles-1v1-searching-pursuer-h16-search16.json`.

The next performance work should profile the remaining combined passive time
with tile storage held fixed, particularly metabolism, digestion preconditions,
and repeated cell/index work. The architectural tile-locality and sparse-COW
goals are now simultaneously satisfied; another tile representation change is
not justified without a new counterexample.

That cell-passive profiling and first optimization slice is complete. An
ignored release diagnostic now separates digestion overflow validation,
parallel digestion arithmetic, digestion-frontier retention, metabolic
deadline rebuild, serial/parallel metabolism arithmetic, metabolic tile
deposits, and diffuse-frontier maintenance. With 200,000 active cells on a
512x512 field, the original isolated costs were 1.15 ms for the digestion
overflow scan, 0.41 ms for parallel digestion arithmetic, 0.37 ms for its
frontier, 1.30 ms for deadline rebuild, 0.40 ms for serial metabolism
arithmetic, 3.00 ms for direct tile deposits, and 9.28 ms for extending the
diffuse `BTreeSet`. The previously unattributed cost was therefore primarily
the derived frontier, not cell arithmetic.

The retained native kernel activates only when every cell is metabolically
active, cell keys are sufficiently dense, both existing parallel thresholds
are crossed, and active cells occupy at least three quarters of the board. It
stores each cell's spent energy in the already-owned diffusion accumulator,
applies deposits in one row-major parallel tile pass, clears the touched
scratch slots, and rebuilds the exact diffuse frontier by row-major filtering.
The isolated gathered deposit costs 1.70 ms and the exact frontier scan costs
2.87 ms, replacing 3.00 ms of random page mutation plus 9.28 ms of tree
extension. Sparse populations and browser WASM retain canonical serial
execution, avoiding a full-board pass when occupancy is low.

Overflow behavior is explicitly preflighted. If any gathered deposit would
overflow its destination, the host falls back to canonical cell-key order so
both the returned error and partially advanced failure state match the serial
oracle. Full-occupancy worker-count tests cover the gathered normal path, and
the overflow regression compares cells, tiles, and cleared scratch storage
against serial execution. The digestion transfer precondition is now itself a
parallel read-only scan; this changes no failure ordering because the existing
serial fallback remains intact.

With the overflow preflight included, eight combined passive steps now measure
178.90 ms versus the preceding 197.46 ms page-overlay baseline, a 9.4%
reduction. The complete native path is 2.52x faster than its serial oracle;
metabolism is 43.87 ms versus 74.85 ms across the eight steps. The complete
all-feature workspace suite, warnings-denied native and WASM Clippy, and the
release `blob_web` WASM build pass.

The fixed MCTS control again reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. Wall time was 41.35
seconds and peak RSS was 216.65 MB, within the established run variance. The
report is
`training-output/micro-combat-observation-policy-cost-dense-metabolism-1v1-searching-pursuer-h16-search16.json`.

The next measured cell-side seam is digestion scheduling: its overflow
precondition and complete metabolic-deadline rebuild together exceed the
parallel digestion arithmetic itself. A bounded follow-up should calculate
deadlines alongside dense digestion and bulk-heapify precomputed events,
provided it preserves exact overflow selection and does not add retained
per-simulation scratch that would increase planner-tree memory.

That fused digestion/deadline slice is now complete. The subphase diagnostic
first rejected the obvious implementation: collecting a parallel vector of
200,000 schedule results cost 1.71 ms for fused digestion/calculation and a
further 1.13 ms for index assembly, versus about 0.49 ms for the existing
parallel digestion arithmetic plus 1.31 ms for its serial deadline rebuild.
Parallelizing arithmetic while introducing another materialized result was a
net loss.

The retained path is restricted to a hole-free dense cell keyspace where every
cell is digesting, already has assimilated energy, and crosses the existing
16,384-cell parallel threshold. One indexed read-only pass computes the exact
post-digestion transfer and deadline feasibility. When both fit, the resolver
reuses the existing exhaustion heap and position vectors: it zips mutable cell
slots directly with heap entries, applies digestion, writes each absolute
deadline into its final storage, then heapifies in place. No schedule vector or
new retained simulation scratch is allocated.

The fallbacks remain semantic rather than merely defensive. A digestion
transfer overflow uses canonical serial cell-key order and therefore preserves
the same partial failure state. A deadline overflow keeps parallel digestion
but uses the established rebuild, preserving the exact ordered overflow map.
Historical holes use the previous indexed-cell path. Browser WASM also retains
the serial path. Regressions cover full occupancy across one, two, and four
workers, transfer overflow, deadline overflow near `u64::MAX`, and a dense
active population whose key storage contains a removed historical cell.

On the 200,000-cell/512x512 benchmark, eight digestion phases fell from 18.03
ms to 9.85 ms, a 45.4% reduction. The complete eight-step passive suite fell
from 178.90 ms to 173.95 ms, another 2.8% reduction and 11.9% below the 197.46
ms page-overlay baseline. The final eight-worker suite is 2.53x faster than its
serial oracle. The fused digestion phase remains beneficial with fewer
workers: 25.61 ms with one worker and 16.88 ms with two, versus approximately
61 ms for the tree-traversing serial oracle in both controls.

The fixed MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. Wall time was 43.31
seconds and peak RSS was 216.40 MB, within established variance. The report is
`training-output/micro-combat-observation-policy-cost-fused-digestion-deadlines-1v1-searching-pursuer-h16-search16.json`.

The full all-feature workspace suite, warnings-denied native and WASM Clippy,
and release `blob_web` WASM build pass. Remaining dense passive cost is again
field-heavy: signal decay is about 67.34 ms, metabolism 45.25 ms, diffusion
30.29 ms, plant growth 21.21 ms, and digestion 9.85 ms across eight steps. A
next optimization should reprofile inside signal decay or metabolism rather
than further complicating the now-small digestion path.

That signal-decay profile found that the four-channel arithmetic itself costs
only about 0.5 ms per 512x512 field. Rebuilding the two ordered active-tile
frontiers cost roughly 5 ms, so merely parallelizing subtraction would have
missed the bottleneck. Direct parallel insertion into ordered sets was much
worse at about 22.6 ms, while collecting flat tile-index vectors was faster
but required multi-megabyte temporary allocations that would amplify planner
tree memory pressure.

The retained dense kernel processes fixed 2,048-tile pages in parallel. Each
page returns its four exact signal totals plus two 32-word bitmaps identifying
the surviving signal and diffusion frontiers. Results are collected in page
order, totals are checked with the canonical overflow semantics, and the small
bitmaps reconstruct the ordered frontiers deterministically. A 512x512 field
uses about 74 KiB of temporary page results rather than per-tile index vectors.
Sparse fields, browser WASM, below-threshold workloads, and any overflow-risk
case retain the established serial path. Regressions cover full occupancy with
one, two, and four workers, partial final pages, expiration and activation of
frontier entries, sparse fallback, and overflow-preserving failure state.

Across eight dense passive steps, signal decay fell from 67.34 ms to 48.09 ms,
a 28.6% reduction. The complete suite fell from 173.95 ms to 159.73 ms, another
8.2% reduction, 19.1% below the original 197.46 ms page-overlay baseline, and
2.72x faster than the serial oracle. The kernel remains useful with one worker
(68.36 ms signal, 246.26 ms complete) and two workers (56.91 ms signal, 183.51
ms complete). The fixed MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. It completed in 41.02
seconds with 216.74 MB peak RSS; the report is
`training-output/micro-combat-observation-policy-cost-fused-signal-frontiers-1v1-searching-pursuer-h16-search16.json`.

Metabolism is now the largest measured passive phase at about 49.18 ms across
eight steps, followed by signal decay at 48.09 ms, diffusion at 29.32 ms, plant
growth at 22.13 ms, and digestion at 11.01 ms. The next bounded optimization
should profile metabolism internally and retain only an allocation-conscious
dense path with explicit serial overflow fallbacks.

That metabolism slice found another derived-index cost rather than expensive
physics arithmetic. Advancing 200,000 cells costs about 0.9 ms serially or 0.3
ms in parallel, but the dense path rebuilt the complete ordered diffuse
frontier after every emission. In a steady-state field those cells deposit onto
tiles that are already active, so the rebuild repeatedly recreated an unchanged
set. Extending the set with every occupied position was worse still, at about
9.2 ms per step in the diagnostic.

The retained preflight now proves both deposit overflow safety and whether any
positive spend targets a previously zero diffuse tile. A hole-free cell store
uses its indexed slots directly; historical holes retain canonical active-key
traversal. When no tile is newly activated, the exact existing frontier is
preserved. If activation occurs, the established exact row-major rebuild still
runs. The ordinary serial and sparse paths likewise record only zero-to-positive
transitions rather than every deposit. Metabolism also skips its active-cell
set retention when no cell exhausted, because no membership can then have
changed. A one-worker crossover selects direct serial deposition instead of a
dense gather whose full tile pass is slower without parallel workers.

Regressions compare dense and serial behavior for mixed zero/nonzero diffuse
tiles, mixed surviving/exhausted cells, unchanged fully active frontiers, and a
removed historical cell key. Existing worker-count, index-consistency, and
overflow-partial-state regressions remain authoritative. No new retained
per-simulation scratch or canonical state was added; WASM retains the serial
path.

Across eight 200,000-cell/512x512 passive steps, metabolism fell from 49.18 ms
to 13.30 ms, a 73.0% reduction. The complete eight-step suite fell from 159.73
ms to 124.66 ms, another 22.0% reduction, 36.9% below the 197.46 ms page-overlay
baseline, and 3.06x faster than the current serial oracle. One worker now uses
direct deposits and completes the suite in 194.13 ms; two workers complete it
in 148.42 ms. The fixed MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. It completed in 45.50
seconds with 216.55 MB peak RSS; the report is
`training-output/micro-combat-observation-policy-cost-frontier-stable-metabolism-1v1-searching-pursuer-h16-search16.json`.

The largest measured phase is now signal decay at about 48.95 ms, followed by
diffusion at 29.83 ms, plant growth at 23.42 ms, metabolism at 13.30 ms, and
digestion at 9.14 ms. Signal remains largest despite its earlier frontier
optimization; a next slice should distinguish immutable page detachment,
four-channel arithmetic, page-result collection, and ordered frontier assembly
under the current implementation before changing it again.

That reprofile measured the complete signal wrapper at 5.80 ms for one
512x512 phase even though the fused page kernel itself was about 3.29 ms. The
wrapper cloned the complete dense ordered signal frontier before it knew that
the parallel path would not consume the clone. More importantly, the page
kernel rebuilt both ordered frontiers even when signal membership and diffuse
membership were unchanged. In the steady-state passive benchmark, every
signal-bearing tile remains signal-bearing and every diffuse tile is already
active, making both derived rebuilds redundant.

The dense preflight now calculates three facts in one deterministic parallel
reduction: whether every possible signal-to-diffuse transfer fits, whether any
tile loses its final nonzero signal channel, and whether any previously zero
diffuse tile receives positive decay. It evaluates exact per-channel decay for
the two membership predicates. The page kernel records and reconstructs a
frontier only when the corresponding predicate says membership can change;
otherwise the existing exact ordered set is retained. Dense-frontier cloning
is deferred until the serial fallback actually needs canonical keys.

This does not assume frontiers never change. Signal expiration and first
diffuse activation still use the established page bitmaps and deterministic
row-major reconstruction. Sparse fields, below-threshold workloads, browser
WASM, and overflow-risk cases retain the serial path and its ordered partial
failure state. A focused oracle compares both preflight predicates against
actual decay over zero and nonzero diffuse energy, four signal patterns, two
remainder patterns, and three elapsed intervals. The existing partial-page
case exercises simultaneous expiration and activation; worker-count and
overflow regressions remain unchanged.

Across eight dense passive steps, signal decay fell from 48.95 ms to 6.79 ms,
an 86.1% reduction. The complete suite fell from 124.66 ms to 81.91 ms, another
34.3% reduction, 58.5% below the 197.46 ms page-overlay baseline, and 4.69x
faster than the current serial oracle. With one worker the suite is 164.89 ms
and signal decay is 40.56 ms; with two workers the suite is 115.13 ms and
signal decay is 20.75 ms.

The fixed MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. It completed in 43.24
seconds with 216.96 MB peak RSS; the report is
`training-output/micro-combat-observation-policy-cost-frontier-stable-signal-1v1-searching-pursuer-h16-search16.json`.

Dense diffusion is now the largest measured passive phase at about 30.37 ms
across eight steps, followed by plant growth at 21.05 ms, metabolism at 13.92
ms, digestion at 9.79 ms, and signal decay at 6.79 ms. The next bounded slice
should reprofile diffusion source planning, reverse-neighbor gathering, page
mutation, and exact frontier maintenance under the current field density.

That diffusion profile found four full-field intermediates in the dense path:
parallel planning results, copied flux structs, parallel final-energy results,
and copied final energies. It also rebuilt the complete ordered diffuse
frontier after every synchronous step, even when every source and destination
remained positive. These costs dominated the actual fixed-degree neighbor
arithmetic.

The retained kernel stores only one temporary `u64` outgoing share per tile.
It gathers final energies directly into the simulation's existing reusable
dense accumulator, then commits tiles in parallel and clears that accumulator.
The tile-local diffusion remainder is recomputed during commit rather than
retaining the larger flux structure. Thus the temporary allocation falls to
one approximately 2 MiB vector for a 512x512 field, with no new retained
per-simulation scratch.

Each destination still uses its canonical ordered reverse-source list and the
same checked additions. Parallel workers atomically record the lowest failing
destination; after the read-only gather, that destination is recomputed to
produce the exact aggregation-versus-commit error. Scratch is cleared and no
canonical tile is mutated on failure. During a successful gather the kernel
also detects zero/nonzero membership transitions. An unchanged exact diffuse
frontier is preserved; a changed frontier is rebuilt canonically after all
final energies are known.

A directed east-only wrapped control shifts a 75%-dense frontier and matches
the serial synchronous oracle exactly. Separate deliberately overflowing
derived graphs exercise aggregation overflow, commit overflow, lowest-index
selection, unchanged canonical state, and cleared scratch. Existing Moore,
bounded-edge, sparse, one/two/four-worker, and energy-conservation regressions
remain in place. Browser WASM retains the serial path.

Across eight dense passive steps, diffusion fell from 30.37 ms to 12.44 ms, a
59.0% reduction. The complete suite fell from 81.91 ms to 62.19 ms, another
24.1% reduction, 68.5% below the 197.46 ms page-overlay baseline, and 6.24x
faster than the current serial oracle. One worker completes the suite in
161.10 ms and two workers in 104.04 ms; the compact gather does not regress the
one-worker control and reduces the two-worker diffusion phase to 23.08 ms.

The freshly rebuilt fixed MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. It completed in 43.25
seconds with 216.96 MB peak RSS; the report is
`training-output/micro-combat-observation-policy-cost-compact-diffusion-gather-1v1-searching-pursuer-h16-search16.json`.

Dense plant growth is now the largest measured passive phase at about 20.50 ms
across eight steps, followed by metabolism at 13.13 ms, diffusion at 12.44 ms,
digestion at 9.69 ms, and signal decay at 6.42 ms. The next slice should profile
growth arithmetic, page detachment, and diffuse-frontier removal, particularly
the common case where neither plant capacity nor diffuse exhaustion changes
membership.

That profile found that page-local mutation was not the dominant plant cost.
Every growth pass unconditionally retained the complete ordered diffuse
frontier, even though plant growth can only remove diffuse energy and the
steady-state control exhausted no tile. It also performed runtime `u128`
division and remainder operations for the default 1,024-quanta time scale on
every growing tile.

Plant growth now records whether any positive diffuse reservoir actually
reaches zero. Dense native passes preserve the existing exact frontier when no
membership changes; if exhaustion occurs they retain canonically after the
parallel mutation. Sparse and browser passes collect only the exhausted
growing tiles and remove those exact keys rather than scanning unrelated
diffuse tiles. A dirty externally mutated frontier is rebuilt before either
path, so this optimization does not assume that derived state is already
current.

The growth divisor is compiled once per pass. Power-of-two time denominators,
including the default 1,024, use exact shifts and masks; other denominators keep
the original `u128` quotient and remainder operations. An exhaustive boundary
control compares both operations against ordinary integer division for
power-of-two and non-power-of-two divisors through `u128::MAX`. Dense exhaustion
tests compare tile state and exact frontiers across one, two, and four workers,
while a sparse dirty-frontier control checks selective removal.

In the immediate eight-worker before/after control, plant growth fell from
18.84 ms to 3.01 ms across eight steps, an 84.0% reduction. The complete suite
fell from 59.27 ms to 45.05 ms, another 24.0% reduction, 77.2% below the 197.46
ms page-overlay baseline, and 8.11x faster than the current serial oracle. One
worker completes the suite in 144.32 ms with 14.21 ms of plant work; two workers
complete it in 89.26 ms with 7.42 ms of plant work.

The freshly rebuilt fixed MCTS control again reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. It completed in 43.12
seconds with 217.25 MB peak RSS; the report is
`training-output/micro-combat-observation-policy-cost-frontier-stable-plant-growth-1v1-searching-pursuer-h16-search16.json`.

Dense metabolism is now the largest measured passive phase at about 13.55 ms,
followed by diffusion at 11.78 ms, digestion at 10.43 ms, signal decay at 6.29
ms, and plant growth at 3.01 ms. The next slice should profile metabolism's
single-worker crossover and its exhaustion/frontier bookkeeping before changing
the arithmetic again.

That subphase profile showed that the gathered dense path still calculated
every metabolism spend twice: a parallel exact-overflow preflight performed the
full `u128` quotient, then a serial cell loop repeated the quotient while
mutating cells and assembling tile deposits. In the 200,000-cell control, the
same arithmetic took about 1.19 ms serially and 0.29 ms in parallel. Direct tile
deposition was already dominated by traversal and COW locality rather than the
addition itself.

The dense host path now performs only a conservative parallel overflow
preflight. Since actual spend cannot exceed assimilated energy, proving that a
tile can accept the cell's entire assimilated reservoir also proves the actual
deposit fits without calculating it. An inconclusive preflight falls back to
the unchanged canonical serial path, preserving its exact overflow and partial
failure behavior.

After a successful preflight, cell metabolism and deposit publication are
fused in one parallel pass. Each occupied tile has exactly one cell, so workers
publish into a reusable tile-indexed `AtomicU64` buffer with no contended
destination. A page-parallel tile pass commits and clears those deposits.
Relaxed atomics provide a safe Rust representation of the disjoint writes;
Rayon phase completion supplies the inter-pass synchronization. The buffer is
derived scratch, excluded from canonical state, hashes, checkpoints, replay,
and Mind inputs. Simulation clones intentionally receive an empty buffer, so
planner branches do not copy its retained allocation.

Zero-energy membership and diffuse activation are recorded during the fused
pass. The exact zero-energy and metabolism indexes are scanned or updated only
when a cell actually exhausts, and the diffuse frontier is rebuilt only when a
previously empty destination activates. Exhausted metabolism keys are removed
directly instead of retaining the complete ordered set. The common rate-divisor
helper also gives metabolism the exact power-of-two shift/mask path already
validated for plant growth.

The atomic gather is profitable from four workers upward; one- and two-worker
runs retain the serial cell/tile loop. Across eight dense steps at eight
workers, metabolism fell from 13.55 ms to 10.51 ms, a 22.4% reduction. The
complete suite fell from 45.05 ms to 41.90 ms, 78.8% below the 197.46 ms
page-overlay baseline and 8.87x faster than the current serial oracle. The
two-worker suite remains on the serial metabolism crossover at 90.27 ms total;
four workers complete it in 52.44 ms.

The fixed MCTS control reproduced policy hash
`f6bcf9836e2c7598770a512c9401f50e2e223570d368de3e5aee7bd9c0df9f47`,
199,986 transitions, 3,998 iterations, every deterministic allocation counter,
and the unchanged 156,242,380-byte retained lower bound. It completed in 42.87
seconds with 216.43 MB peak RSS; the report is
`training-output/micro-combat-observation-policy-cost-dense-metabolism-emission-1v1-searching-pursuer-h16-search16.json`.

Dense diffusion is again the largest measured passive phase at about 12.09 ms,
followed by metabolism at 10.51 ms, digestion at 9.90 ms, signal decay at 6.46
ms, and plant growth at 2.93 ms. A next slice should determine whether retaining
a compact diffusion remainder beside each outgoing share is cheaper than
recomputing the source plan during commit, without restoring the former large
full-field intermediates.

The planning oracle should preserve three distinct claims:

1. **Exact certificate:** exhaustive dynamic programming over all reachable
   canonical decision-frontier states proves the finite scenario optimum. This
   is expected to be practical only for very small deterministic cases.
2. **Privileged upper bound:** checkpoint-based A*/beam search or MCTS may read
   canonical state and a fixed future random stream. Failure is inconclusive;
   success proves physical feasibility but does not define a legal Mind.
3. **Observation-policy witness:** information-set MCTS/POMCP nodes are keyed by
   public Mind observations and private memory, with hidden-state particles
   sampled across scenario seeds. A resulting controller can be replayed
   through the Mind ABI and is a fair learned-policy comparator.

Search advances only between canonical ready-cell frontiers, so movement and
attack duration remain part of the transition model. A single-cell scenario is
the initial supported domain. Multi-cell planning must not choose a joint team
action from shared hidden state: each cell controller receives only its own
observation and memory, and coordination must occur through ordinary physical
or signal channels. A centralized multi-cell planner may be retained only as a
clearly labeled upper bound.

Use a lexicographic scenario objective so “optimal” is well-defined: maximize
objective success probability first, then terminal survival margin/weighted
biological energy, then scenario-specific efficiency such as damage per spent
mass or time to elimination. Scenario generation should sweep starting energy,
population, distance, horizon, and opponent profile, retaining cases with a
reliable legal witness, no passive solution, and a deliberate gap between a
simple baseline and the observation-policy oracle. This produces a curriculum
of solvable nontrivial tasks instead of hand-authored traps or timeout exploits.
