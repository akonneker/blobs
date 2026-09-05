# Physics, interface, and training diagnosis

This project must not tune a learned policy around a scenario that is
physically impossible, nor tune the physics merely because one optimizer failed.
Every promoted learning scenario should therefore advance through the evidence
ladder below. The earliest failed or unsupported stage is the diagnosis.

The ladder preserves the project's core operational principle: a deployed Mind
is a collection of strictly separated cells. Each cell receives only its own
anonymous local observation, explicit random-input block, and private memory.
It has no host cell ID, team identity, canonical world state, shared hidden
memory, or privileged teammate channel. Full-state planners are useful only as
clearly labeled diagnostics; they never define a legal Mind or promotion
baseline.

## Evidence ladder

1. **Analytical bounds.** Check conservation, lifetime, travel, damage, intake,
   action-duration, and objective inequalities. Failure proves that the stated
   expected behavior cannot occur under the examined assumptions.
2. **Full-state feasibility.** Search canonical simulator states to establish a
   physical witness or an exact impossibility certificate. A successful
   privileged search proves physical feasibility. Failure of bounded A*, beam
   search, or MCTS is inconclusive; only exhaustive coverage can support a
   failed outcome.
3. **Mind feasibility.** Replay an information-set policy through the public
   Mind ABI. Its state key may contain only that cell's observation and private
   memory. For multiple cells, coordination must use ordinary physics or signal
   channels. Failure here after full-state success points to observation,
   action, memory, or timing semantics.
4. **Maintained teacher.** Test a deterministic, observation-legal teacher on
   held-out initial scenarios. This separates “a legal policy exists” from “our
   maintained policy construction can express it.”
5. **On-policy teacher recovery.** Freeze the learned policy, collect the exact
   states it visits, then ask the maintained teacher to continue from those
   states. If the teacher cannot recover, earlier policy errors have entered a
   different or irrecoverable state distribution. Static teacher-trajectory
   accuracy cannot diagnose this.
6. **Learned on-policy agreement.** Where the teacher can recover, compare the
   learned action, allowed-action mask, routed expert, and value estimate with
   the teacher label. Failure points to representation, routing, loss balance,
   optimization, or insufficient on-policy coverage.
7. **Held-out rollout competence.** Run complete, disjoint-seed rollouts using
   the actual promotion objective and time horizon. Failure after agreement
   points to compounding error, objective/reward mismatch, long-horizon credit,
   or remaining distributional gaps.

The order matters. Later success must not be used to conceal an earlier missing
test, and later stages stop after a failure. `not_applicable` is permitted only
for analytical bounds or full-state search and requires a reason; for example,
a directly replayed legal Mind witness already subsumes privileged physical
feasibility. An unattempted or computationally exhausted search is `not_run`,
not `failed`.

## Diagnostic meanings

| First failed stage | Current attribution | Appropriate next change |
|---|---|---|
| Analytical bounds or exact full-state feasibility | Physics or objective | Change rules, starting state, horizon, or victory criterion |
| Mind feasibility | Mind interface or action semantics | Change observable/action capability or scenario, while preserving cell isolation |
| Maintained teacher | Maintained policy gap | Improve the legal witness/controller |
| On-policy teacher recovery | On-policy distribution shift | Prevent or cover catastrophic off-teacher states |
| Learned on-policy agreement | Architecture or training | Fix representation, routing, data, loss, or optimization |
| Held-out rollout competence | Rollout competence | Inspect reward/objective alignment, compounding error, and longer horizons |
| No failed or missing stage | Competent | Eligible for the separate statistical promotion gates |

These are causal working labels, not permission to change the named subsystem
immediately. A rules change still requires the broad physics-calibration suite,
and a Mind ABI change still requires anonymity, isolation, parity, and
information-flow review.

## Machine-readable contract

`scenario-diagnosis` schema 1 accepts a canonical seven-entry TOML claim,
streams and SHA-256-binds every cited artifact, derives the attribution, and
publishes immutable JSON. Relative artifact paths are resolved from the TOML
file's directory. Passed and failed stages require an artifact; `not_run` and
`not_applicable` stages must not cite one. Once a stage fails or is not run, all
later stages must be `not_run`.

```toml
schema_version = 1
scenario_name = "adjacent-food retention / seed suite 20000..20031"
ruleset_hash = "<64 lowercase hex characters>"
scenario_hash = "<64 lowercase hex characters>"

[[evidence]]
stage = "analytical_bounds"
outcome = "passed"
artifact_kind = "physics_calibration_report"
artifact = "physics-calibration.json"
summary = "Travel, feeding, and stationary-lifetime bounds pass."

# Add the remaining six stages in canonical order.
```

Publish a descriptive diagnosis:

```sh
cargo run -p blob_rl --bin scenario-diagnosis -- \
  --spec path/to/diagnosis.toml \
  --output path/to/diagnosis.json
```

Add `--require-competent` when using it as a pipeline gate. The report contains
the spec hash, every evidence file hash and size, the derived attribution, the
first blocking stage, and a hash over the complete diagnosis identity.
Publication is create-new and refuses replacement.

Schema 1 validates the ladder and binds evidence bytes. Unknown artifact kinds
remain explicit evidence claims, while two artifact kinds are typed sources:

- `feeding_recovery_artifact` must appear at
  `on_policy_teacher_recovery`, pass its own complete validation, match the
  outer compiled ruleset and ordered scenario-suite hashes, and agree with the
  claimed outcome.
- `policy_correction_dataset` must appear at
  `learned_on_policy_agreement` and reference the dataset's `manifest.json`.
  The adapter validates the demonstration payload hash, model identity,
  compiled ruleset, scenario hash, confusion-matrix margins, exact-agreement
  rate, configured threshold, and derived verdict before accepting the claim.

Before this becomes a general promotion authority, add equivalent typed
adapters for physics calibration, exact/full-state search, Mind-feasibility,
maintained-teacher evaluation, and held-out rollouts. Each adapter must validate
the source artifact and recompute its outcome rather than trusting the TOML
label.

## Frozen-policy feeding recovery

`feeding-recovery` implements stage 5 for the on-food and adjacent-food
prerequisites. It follows a verified behavior clone greedily, excludes the
initial state already covered by teacher qualification, and forks the first
surviving policy-induced frontier plus later frontiers spaced by canonical
simulation time. Each exact checkpoint includes canonical physics, isolated
cell-private memory, and private-random continuation state. The memoryless
collision-aware forager then receives ordinary Mind inputs and continues to the
original episode deadline.

Recovery requires at least one surviving training cell and completion of only
the objective pieces not already achieved by the policy prefix: movement for
an adjacent-food state that has not moved, and positive consumption if the
prefix has not consumed. Frontiers too close to the deadline are excluded by
`min_remaining_time_quanta`. The gate separately requires per-seed checkpoint
coverage, a minimum recovery rate, and zero teacher or policy safety aborts.

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin feeding-recovery -- \
  --config path/to/training.toml \
  --behavior-clone path/to/behavior-clone \
  --seeds 20000,20001,20002,20003 \
  --checkpoint-interval-quanta 4096 \
  --max-checkpoints-per-seed 8 \
  --min-remaining-time-quanta 4096 \
  --min-recovery-rate 0.8 \
  --output path/to/feeding-recovery.json
```

## Exact-input agreement and correction data

`feeding-policy-corrections` implements stage 6 without substituting teacher
trajectories for the states the policy actually visits. At every decision it
gives the frozen learned policy and the maintained teacher the same anonymous
observation, action mask, private recurrent prefix, and canonical cell-private
random-input block. The learned policy controls the rollout; the teacher choice
becomes the label. Samples are retained as complete prefixes so recurrent
training cannot silently begin from an invented hidden state.

Demonstration schema 15 records exact full-choice agreement, teacher and policy
action-family margins, and the complete teacher-to-policy family confusion
matrix. It also stores the policy action and the selected-kind logit advantage
over the teacher kind beside every correction label. Aggregate mean/maximum
crossover distance, fixed threshold bands, and per-error-family sums are
recomputed from the payload during loading. Agreement uses integer parts per
million and margins use integer micrologits, avoiding floating-point identity
ambiguity in immutable artifacts. Schema-14 correction datasets remain
loadable but do not carry margin telemetry.

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin feeding-policy-corrections -- \
  --config path/to/training.toml \
  --behavior-clone path/to/behavior-clone \
  --stage adjacent-food \
  --seeds 950000101,950000202 \
  --max-samples 4096 \
  --minimum-policy-agreement-rate 0.8 \
  --output path/to/corrections
```

Correction seeds become training data as soon as their labels are consumed and
must never be reused for held-out evaluation. Policies that die before the
sample cap legitimately produce smaller datasets, so correction-versus-rehearsal
experiments must equalize the number of presented examples and optimizer steps,
not merely the number of files or epochs.

## Counterfactual action-value branches

Static teacher agreement does not establish that the teacher action has the
better consequence on a state visited by the learned policy.
`counterfactual-branch-evaluation` checkpoints an exact on-policy frontier,
changes one selected cell's physical action, and restores that checkpoint for
every paired branch. Other simultaneously ready cells retain their frozen
choices and recurrent updates. The selected cell retains the same
action-independent recurrent update. Each continuation then uses either the
frozen policy or the teacher for all cells, with elapsed simulation quanta—not
decision counts—as its horizon.

Use a short `all-legal` probe to inspect every legal base Guard and Move choice,
then a `teacher-policy` probe for long-horizon confirmation:

```sh
cargo run --release -p blob_rl --no-default-features --features ndarray \
  --bin counterfactual-branch-evaluation -- \
  --config path/to/training.toml \
  --behavior-clone path/to/behavior-clone \
  --stage adjacent-food \
  --seeds 1420000202 \
  --max-states 16 \
  --max-states-per-seed 2 \
  --max-states-per-frontier 1 \
  --max-source-frontiers-per-seed 256 \
  --candidate-scope teacher-policy \
  --horizon-quanta 256,1024,4096 \
  --continuations frozen-policy,teacher \
  --output path/to/counterfactual.json
```

The artifact binds the effective configuration, scenario and compiled ruleset,
Mind ABI, source model metadata/model hashes, source observation, trusted-host
checkpoint identity, options, raw branch metrics, and its own content hash. It
stores no secret checkpoint bytes and makes no privileged state visible to a
Mind. Results remain a vector of survival, population, target/team energy,
action-resolution, reward, intake, damage, birth/death, and position metrics;
the evaluator deliberately does not hide scientific value judgments in one
scalar score.

Schema 2 bounds both sides of the computation. Branch continuations have their
own frontier limit, and source-state discovery scans only the declared prefix
of each seed. Per-seed and per-simultaneous-frontier state caps prevent one busy
world or ready-cell batch from masquerading as independent replication. The
report preserves zero-hit seeds and recomputes both raw state-level preferences
and seed-balanced preferences; in the latter, all selected states from one seed
are summed before that seed casts one comparison vote. The global state cap must
cover every seed's per-seed allowance, so it cannot silently stop before later
seeds are scanned.

Schema 3 additionally preserves team core mass, assimilated energy, and gut
energy as separate exact compartments. The redundant stored- and total-energy
fields are recomputed during validation, so a downstream value analysis can
discount gut contents without rerunning or guessing canonical state.

`counterfactual-value-evaluation` interprets this evidence without a scalar
reward. Survival (cell perspective) or population (colony perspective) is the
first lexicographic tier. Explicitly weighted biological energy is the second.
The conservative perspective requires Pareto agreement between cell and colony
metrics. Selected continuation controllers must also agree; otherwise the
verdict is `incomparable` and is ineligible for a training label.

```sh
target/release/counterfactual-value-evaluation \
  --counterfactual path/to/schema-3-counterfactual.json \
  --perspectives cell,colony,conservative \
  --horizon-quanta 256,1024,4096 \
  --continuations frozen-policy,teacher \
  --core-weight-ppm 1000000 \
  --assimilated-weight-ppm 1000000 \
  --gut-weight-ppm 0 \
  --output path/to/value-analysis.json
```

Weights are integer parts per million and are hash-bound. The default counts
core and assimilated energy fully and gut contents at zero. Changing gut value
is a new immutable analysis, not an unrecorded interpretation of old results.

Value schema 3 also computes a robust Pareto frontier across every candidate
present in the source branch artifact. One candidate dominates another only
when it is non-worse under every selected continuation and strictly better
under at least one; controller or metric conflicts preserve both. Each frontier
is additionally projected to a target-free `(action kind, effort)` archetype.
This allows evidence to support “minimum-effort Move” without selecting a
direction from hidden future state that was absent from the cell's observation.

When converting that evidence into supervised data, retain the exact anonymous
observation, legal masks, and the cell's own private recurrent state. Give each
selected state an isolated host-only trajectory key. A correction label may
change only the evidenced target-free component; for Move effort, preserve the
frozen policy's observation-local target slot. Use an active control that
preserves the original effort and train both arms through effort-head-only
adaptation. Counterfactual futures remain host-side evidence and never become
direction, clock, identity, or shared-memory features in the Mind ABI.

Use `--candidate-scope all-legal` for the short discovery pass. A later deep
pass can use `--candidate-scope explicit --candidate-actions ...`; the exact
teacher and policy actions are included automatically as audit baselines. Every
declared action is still filtered through each source observation's legal mask.
The numeric ids are immutable catalog ids and must be recorded in the branch
artifact rather than reconstructed from prose.

These probes are causal evidence about particular visited states, not training
labels by themselves. Discovery seeds must remain outside final qualification,
and an apparent advantage should be reproduced across disjoint states and both
continuation controllers before it shapes an objective.

The feeding-recovery evaluator restores and completes one branch at a time; it does not retain
world checkpoints in the report. Each row stores a streaming SHA-256 identity,
simulation time, prefix progress, teacher suffix counters, and terminal result.
Peak memory is bounded by the main environment, one recovery branch, and the
single serialized checkpoint buffer used to create it, while total work remains
proportional to the configured number of forks.

## Required evaluation discipline

- Use disjoint seeds for demonstrations, correction collection, selection, and
  final rollout evaluation.
- Bind ruleset, scenario, ABI/model, teacher, and learned-policy identities in
  the underlying evidence, not only in the outer diagnosis.
- Match the production simulation clock and action-resolution semantics.
- Report per-layout and per-opponent failures; an aggregate mean cannot erase a
  collapsed regime.
- Treat success probability and confidence separately from causal diagnosis.
  Passing this ladder establishes competence evidence, not leaderboard-grade
  statistical confidence.
- Keep privileged planner traces out of learned-policy inputs. They may produce
  teacher labels only after those labels have been projected through the legal
  Mind observation and action boundary.

## Current feeding case study

The collision-aware maintained Mind and the frozen learned policy were run on
the same two held-out seeds for each of line, checkerboard, ring, loose-random,
and random founder layouts. The teacher passed all ten trials. Its mean
adjacent-food survival was 91.0% on line and ring, 93.4% on checkerboard, 93.2%
on loose-random, and 99.2% on random. The learned policy achieved 74.8%, 76.0%,
80.7%, 83.0%, and 98.2%, respectively. This is a legal-Mind feasibility
witness, not merely a privileged-state upper bound, so the bad line/ring result
is not attributable to an impossible physics regime.

Transition attribution localizes the learned failure. When a plant was visible
from an off-plant cell, the policy chose `Consume` on its empty current tile in
76.4–90.6% of decisions and a correctly targeted `Move` in only 8.2–23.4%.
Among those correct Moves, Frustrated-plus-Contested outcomes account for 31.1%
on line and 26.1% on ring, but less than 1% on random. Once on a plant, more
than 98.7% of decisions are `Consume` in every layout. The primary defect is
therefore the adjacent-state action-kind boundary; dense-layout contention is
a secondary amplifier, and plant retention is not the first correction target.

That original attribution exposed a routing bug rather than a weight defect:
the three-expert router treated diffuse environmental energy as directly
consumable food. Correct routing now selects foraging only on a plant tile
(including a temporarily exhausted plant, identified by capacity) or loose
energy. An ordinary diffuse-only tile selects exploration. This preserves the
Mind boundary because the decision uses only the anonymous observation.

With the frozen weights and corrected routing, off-plant visible-plant choices
contain no erroneous `Consume` actions on line and ring. Plant retention is
100% and survival rises to 92.8% on ring, but line remains 75.0% and
checkerboard falls to 75.2%. Transition evidence now attributes 80.6% of line
plant-directed Moves and 85.4% of checkerboard Moves to `Contested` outcomes.
Fresh exact-input teacher corpora localize nearly every remaining disagreement
to the target slot, not action kind or effort. This supersedes the proposed
adjacent `Consume` weight correction.

The active experiment therefore changes only same-effort `Move` target labels
on exact policy-induced recurrent prefixes. Control labels retain the frozen
policy target; treatment labels use the collision-aware teacher target when it
is a reachable unoccupied plant slot. Five layout pairs validate as trajectory
identical, with 401, 429, 395, 207, and 21 active labels for line,
checkerboard, ring, loose-random, and random. Training freezes every parameter
except the target-query head and qualification seeds are disjoint from
collection seeds.

That target-query-only mechanism failed. The 22/44/88-update treatments did
not change held-out greedy target accuracy; 256/512/1,024 updates made it
progressively worse despite lower cross-entropy. Matched controls retained
100% agreement with their original target labels. This disambiguates a model
capacity/path problem from physics and from simple undertraining: the fixed
query head cannot recover the teacher's private-random tie dispersion through
the inherited representation. Do not run ecological qualification for this
treatment. Add a separately trainable raw-random-plus-slot target adapter,
verify target crossover on disjoint exact inputs, and only then repeat the
layout matrix.
