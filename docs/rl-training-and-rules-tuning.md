# RL training and rules-tuning plan

Status: event-time rollout correctness, PPO stability, deterministic held-out
evaluation, immutable artifacts, and exact update-boundary resume are
implemented. Typed baseline opponents and per-matchup held-out evaluation are
implemented. Bounded, rated snapshot self-play with exposure-aware selection is
implemented. The canonical serializable rules-profile boundary, immutable
replicated sweep planner, bounded/resumable process executor, paired
aggregation, and bounded authoritative gameplay telemetry are implemented.
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
- discards action intervals still unresolved at a rollout boundary instead of
  treating them as zero-value terminals.

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

Control-matrix schema 3 records raw terminal energy compartments separately
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
```

Fresh PPO runs verify both metadata and model hashes before loading. Exact
checkpoint resume ignores initialization weights and restores the checkpointed
model, while retaining the initial-policy identity in its training config.

After `self_play.start_after_timesteps`, promotion is considered every
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
6. **Canonical rules sweeps — profile boundary implemented.** `[env.rules]`
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
