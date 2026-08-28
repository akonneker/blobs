# RL training and rules-tuning plan

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
  --bin demonstrations --bin behavior-clone --bin feeding-evaluation
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
imbalance for controlled follow-up sweeps.

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
