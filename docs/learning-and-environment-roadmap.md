# Learning and environment calibration roadmap

Updated 2026-09-23. This is the current experiment order, superseding the task
order in older study reports. It is a plan, not new experimental evidence.
The [handoff](project-handoff.md) records implementation status and the
[experiment index](experiment-index.md) preserves historical dispositions.

## Objective and current uncertainty

Qualify an ordinary learned Mind that feeds, fights and recovers using only its
local observation, inherited private memory and cell-private randomness. Define
interesting behavior by useful responses to different conditions, not action
entropy, attack counts or visual activity alone. Richer coordination remains a
later claim requiring evidence that the available communication matters.

We have demonstrated improved feeding on development layouts and execution of
learned policies through the ordinary WASM interface. Seven frozen candidates
still make no attacks in the deployed combat assay; a maintained aggressor can
attack and kill. This establishes opportunity and mechanical feasibility, not
the strategic profitability of combat. Feeding against waiting opponents does
not establish competitive ecology.

The standardized residual fits all 128 hard training examples in all three
initializations. This establishes a bounded learning result, not useful combat
on autonomous trajectories. Both environmental incentives and learning remain
live explanations for the broader failure.

## Evidence ladder

| Question | Required evidence | Limit on interpretation |
| --- | --- | --- |
| Physically possible? | Energy/time bounds, scripted resolver cases and bounded full-state search | Failure of a finite search is not impossibility. |
| Worth doing? | Paired action branches and controller outcomes at declared world-time horizons | Legal attacks or damage alone do not establish benefit. |
| Locally identifiable? | A successful control using the exact anonymous Mind input and private state | Privileged search remains a host diagnostic, never a legal Mind witness. |
| Acquirable? | Matched fitting, numerical oracles and disjoint development evaluation | Training agreement does not establish rollout competence. |
| Stable in play? | Autonomous rollouts, recovery from learner-reached states and retained feeding | Teacher-prefix success does not qualify learner-generated trajectories. |

`blob_rl::scenario_diagnosis` already records an evidence ladder, but its first
schema binds explicit claims to artifacts. Add independently checked evidence
for profitability and observability; do not treat a declared stage as proof.

## Work order and dependencies

### 0. Make each run answer a bounded question

Implement the first reporting slice in the [harness analysis](harness-information-density.md):
retain full logs and produce a bounded verified summary with distinct execution,
verification and scientific outcomes. Use the next calibration and normalization
studies as pilots. Do not build a general experiment platform before these runs.

Before spending development seeds, freeze each study's question, variants,
controllers, metrics, thresholds, state/seed selection, horizons, safety limits
and analysis procedure. Preserve the baseline configuration. A rules variant is
a separately identified environment, never a silent change to the canonical game.

### 1A. Calibrate the environment without training

1. Inventory the current configuration with existing ecological characterization:
   food replenishment versus maintenance, travel cost, attack affordability and
   damage, guard cost, split affordability, encounter frequency, and time to the
   evaluation boundary. Check that reward components agree with the intended
   outcome; a shaped reward may favor behavior that loses the match.
2. Run fixed controls: Wait as a floor, collision-aware forager, aggressor,
   defender, explorer and the frozen learned candidate. Use existing legal
   maintained profiles where available; label adapters and any new conditional
   hybrid explicitly. Compare multiple opponent types, both team placements and
   several layouts. Record per-opponent results rather than only pooled wins.
3. Predeclare a small one-factor screening matrix around the current environment:
   resource supply, encounter spacing/density, combat cost/payoff and time horizon.
   Keep policy weights and training rewards fixed. Distinguish scenario changes,
   rules changes and host assessment horizons in the identities. Test selected
   interactions only in a separately declared follow-up; screening cannot exclude
   coupled effects. Choose exact values and run counts after the bounds audit.
4. Report survival, consumed/remaining energy, damage and kills, descendants,
   territorial/resource access, win/loss/draw and safety aborts. Where a metric is
   not instrumented, implement it or mark it unavailable, not zero. Include the
   number of actionable opportunities and policy decisions at those opportunities.

Exit artifact: a paired controller/configuration matrix with immutable identities,
full trial accounting and an explicit conclusion: useful regime found, inadequate
evidence, or measured incentive problem. A controller failure is not proof that
the environment forbids the behavior. A flat matrix may also indicate weak controls.

### 1B. Complete the bounded normalization diagnostic

This can proceed independently of 1A using the already frozen training corpus.
Predeclare zero incoming memory -> zero residual context; nonzero memory -> the
frozen standardization. Preserve the parent's actual memory. Audit transformed
ranges, then compare raw and normalized inputs on all 8,312 training rows using
the prior full-corpus architecture, labels, initializations, sampling and budget.
Retain every run and the unchanged joint combat/feeding thresholds. Do not expand
the architecture search simply because a threshold fails.

Exit artifact: independently replayed all-run training results. Passing means
these recorded labels can be learned; 1A/2 still determine whether they describe
useful behavior. Preserve the separate attack-activity diagnostic even if its
labels turn out to be strategically weak.

### 2. Measure action value and the information needed to recognize it

Extend the existing `counterfactual_branch` machinery, currently restricted to
Guard/Move, to relevant legal Attack/Guard/Move/feeding alternatives. Sample
states from both maintained-control and frozen-learner trajectories. Bind the
sampling rule before inspecting outcomes; do not select only states where an
attack later proves successful.

Fork identical checkpoints and intervene on a declared actor/action, preserving
other simultaneous decisions. Continue with both frozen-policy and maintained
control policies at short and longer world-time horizons. Bind the random-stream
semantics and initial state; diverging action histories need not receive identical
future draws. Store checkpoint, action, continuation and outcome identities.

Report paired outcome differences and opportunity frequency. Distinguish immediate
damage from net energy, survival and later team outcome. Label horizon truncation,
death and host aborts explicitly; never score incomplete branches as losses or zero.
Treat seeds/matches as independent units, not thousands of correlated cell decisions.
Retain uncertainty and mixed outcomes rather than forcing every state into a label.

Then test whether legal observation/history predicts the favorable action. Use
disjoint source episodes, compare observation-only with actual private memory,
and audit contradictory identical inputs. A full-state advantage establishes an
upper bound; failure of one observation-limited model is not proof of an information
limit. Check whether existing teacher labels agree with measured action value.

Exit artifact: profitable opportunities identifiable by a legal control, an
explicit information/incentive problem, or an inconclusive result requiring a
bounded follow-up. If labels change, create a new corpus/plan; do not reinterpret
the frozen normalization experiment as training on those new labels.

### 3. Join the tracks and qualify behavior

Only after useful opportunities and joint training retention are established,
evaluate the matched candidates on declared development data. Keep all-run gates;
do not pick the strongest initialization retrospectively. Existing development
seeds remain exposed; audit genuinely fresh seeds before using them for a new
generalization claim. Do not sweep environment settings on final confirmation.

Export passing candidates with a versioned transform/execution contract and exact
native/WASM checks. Run fixed-opponent combat, feeding retention, learner-state
recovery and held-out layouts/configurations. Success requires useful outcomes
and conditional behavior, not merely increased attack frequency. Preserve the
original environment as a transfer target if a curriculum environment was used.

Self-play follows this qualification. Communication/coordination claims then need
matched channel ablations and outcomes demonstrating benefit; diverse actions alone
are insufficient. Production hosting, deterministic guest budgets and broader
cross-host qualification remain distinct roadmap work.

## How results change the next action

| Observation | Next investigation |
| --- | --- |
| Passive/resource-only control dominates the tested matrix | Incentives, episode boundaries and opponent selection; confirm before changing rules. |
| Benefits appear only with privileged information | Legal observability, memory and a better bounded control. |
| Legal control benefits, but matched fitting fails | Representation, conditioning, labels and optimizer checks. |
| Fitting/development succeeds, autonomous outcomes fail | Learner-state coverage, compounding errors, exploration and recovery. |
| Learning works only in an easier regime | Curriculum and transfer, retaining the original benchmark. |
| Outcomes depend strongly on controller or horizon | Retain uncertainty; broaden those controls before making a causal claim. |

Keep cumulative seed ledgers and immutable binary/source evidence. Confirmation
seeds 1434999901/1434999902 remain reserved. Bound evaluation concurrency and history
retention; treat safety aborts as incomplete evidence. None of this roadmap changes
simulation semantics, qualifies a new Mind, or starts an experiment by itself.
