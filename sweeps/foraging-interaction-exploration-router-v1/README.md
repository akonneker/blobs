# Foraging/interaction/exploration router V1

This slice replaces the two-head action-kind mixture with three experts and
audits their supervision against the exact 15-dataset diversified-oracle V2
corpus before retraining. Routing is a pure function of the anonymous Mind
observation. It never receives teacher action, scenario, team, public identity,
or hidden world state.

## Observable routing rule

Precedence is deliberately asymmetric:

1. a visible neighbor in attack windup or guarding routes to `interaction`;
2. otherwise, food under the deciding cell routes to `foraging`;
3. otherwise, any visible neighboring cell routes to `interaction`;
4. otherwise the decision routes to `exploration`.

The threat override lets a cell react while standing on food. Current-tile
food protects peaceful on-food Consume behavior. Nearby food without a cell on
the current tile is treated as an exploration destination rather than as an
already-acquired resource.

An earlier candidate gave any observable food first priority. The pretraining
audit rejected it before rollout: 17,092 of 17,128 decisions, including every
contact-teacher decision, routed to foraging and exploration received no
samples. Contact scenarios commonly place cells on or within sight of food, so
mere visibility is not a useful behavioral boundary.

## Exact corpus audit

The final rule routes all 17,128 exact-round-trip samples with zero
contradictory route states and zero conflicting action families among the 144
observable states shared across datasets.

| context | samples | Wait labels | Consume labels | Move labels | Attack labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| foraging | 8,654 | 0 | 8,446 | 14 | 194 |
| interaction | 4,590 | 0 | 0 | 4,414 | 176 |
| exploration | 3,884 | 16 | 0 | 3,868 | 0 |

This is a useful decomposition rather than a proof of learnability. In
particular, all 384 exact samples from the energy-60 aggressive contact teacher
remain in `foraging`: its ready-state opponent exposes no attack-windup or
guard signal, and the cells stand on food. That ambiguity is real at the Mind
frontier. The foraging expert therefore must distinguish those attacks from
Consume using the rest of the anonymous observation. If the paired rollout
still collapses, the next candidate should use history-aware or learned latent
routing rather than hidden team/scenario labels.

The audit command now reports, per context, both demonstrated action-family
labels and observations where each catalog family was legal. Legal does not
mean useful: for example, Consume may be commit-legal on an empty tile and an
Attack may target empty space. Promotion decisions must continue to use held-out
rollouts rather than treating mask coverage as competence.

## Architecture and compatibility

The model adds one action-kind head and expands the local gate from two logits
to three. The default 128/64/64 model grows from 96,444 to 97,159 trainable
scalars (+715, 0.74%); recurrent Mind memory and all non-action-kind heads are
unchanged.

Because the recorded tensor shape changes, behavior-cloning schema 21 and
training-artifact schema 43 intentionally reject earlier model records instead
of failing later inside the tensor recorder. Demonstration datasets remain
compatible and are the source of this audit. The maintained 256-cell warm-start
builder now defaults to `foraging-interaction-exploration` routing.

## Paired experiment

The exact normalized rerun is recorded in
[`oracle-diversity-ab-v3`](../oracle-diversity-ab-v3/README.md). It rejects the
learned soft gate: assigned-expert Consume reaches 99.6% in the control and
99.0% in the treatment, but end-to-end Consume falls to 1.9% and 20.2% because
the gate rarely selects foraging. Both arms lose every feeding cell. The next
experiment should make this audited predicate authoritative rather than asking
a second model branch to approximate and potentially override it.

That authoritative schema-22 experiment is recorded in
[`oracle-diversity-ab-v4`](../oracle-diversity-ab-v4/README.md). It restores
exact agreement between assigned-expert and end-to-end accuracy and increases
the control from zero to 6,144 on-food consumes. Both arms nevertheless lose
every feeding cell, exposing teacher-trajectory covariate shift as the next
problem.
