# Feeding DAgger round 5 V1

This slice tests whether the remaining Guard/Move boundary can be crossed
without sacrificing valid movement. It adds a bounded action-kind margin
objective, exposes the routing context of every kind error, and corrects an
unstated representation assumption in the phase-local residuals.

## Margin supervision

Behavior-cloning schema 31 can replace action-kind cross-entropy with a
multiclass hinge against the strongest other legal kind. A requested margin of
zero pressures only tied or misclassified examples; a positive margin also
protects already-correct examples near a boundary. Conditional target, effort,
amount, signal, and phase-gate supervision remains unchanged. The margin and
its bounded positive weight must be enabled together, and evaluation reports
the same objective used for training.

The first zero-margin interaction treatment learned 9.4% of held-out Guards
but reduced Move-kind accuracy to 96.6%. Its fresh trajectory collapsed to
37.6% exact agreement: 3,722 teacher-Move states selected Guard, albeit by only
0.051 logits on average. A two-seed DAgger recovery corpus captured 10,522
post-crossover kind errors, but replaying it removed Guard without restoring a
clean Move boundary. These arms are rejected.

## The hidden routing assumption

Aggregate Guard-to-Move telemetry concealed which authoritative expert owned
the examples. The correction reporter now decomposes every kind error by the
same anonymous observation-only router used at inference. On the selected
fresh probe, the 168 Guard-to-Move errors are 26 foraging, 5 interaction, and
137 exploration. Most Guard supervision therefore belongs to exploration,
not interaction: the collision-aware teacher guards at low assimilated energy
only when there is no current or reachable food.

The prior interaction/exploration residual consumed recurrent features plus
the non-random header but trusted the frozen recurrent pooling to preserve all
neighbor evidence. Schema 31 adds a separate zero-initialized local-context
residual over the non-random header and a featurewise max-pooled summary of all
raw local slots. The pooling is slot-permutation invariant. It can express the
joint low-energy/no-food boundary while seeing only one cell's ordinary Mind
input—no identity, private randomness, other-cell memory, or host scenario
state. `--context-slot-adapter-only interaction|exploration` freezes every
other parameter, including the inherited recurrent/header residual.

The local-context treatment grows the large model from 341,493 to 344,137
parameters and the small profile from 102,133 to 104,777. The later scalar
readiness head adds three more parameters, yielding 344,140 and 104,780.
Schema-28/29 parents migrate
with exactly zero new residual output. The intermediate schema-30 slot-only
prototype omitted self state, was never promoted, and is deliberately rejected
by the lineage loader rather than silently interpreted as the corrected
architecture.

## Bounded exploration treatment

An uninterrupted eight-epoch exploration treatment crosses the boundary:
held-out Guard-kind accuracy reaches 76.5%. It also reduces Move-kind accuracy
to 81.7%, so it is rejected. A staged 2+2+2-epoch treatment, with optimizer
state deliberately reset at each immutable boundary, is the safe diagnostic
artifact. Its final stage reduces the configured validation objective from
1.9743 to 1.5815 while preserving 100% held-out Move-kind accuracy.

On a disjoint 8,192-decision trajectory, the safe artifact preserves the
parent's 93.6% exact agreement and all greedy action choices. The mean
Guard-to-Move deficit falls from the round-four candidate's 17.1222 logits to
1.5967; the 137 exploration-owned errors average 0.9765 logits. It does not yet
select Guard, so there is no physical rollout improvement and no promotion.
Running full feeding or combat qualification on an action-identical candidate
would add no evidence.

## Exact optimizer-step boundary

Behavior-cloning schema 32 adds an optional exact total optimizer-step budget.
It is legal only with the existing fixed per-epoch step budget, so every
artifact is a reproducible prefix of the same uninterrupted optimizer and data
order. Mid-epoch artifacts report actual sample, family, and phase
presentations plus the number of completely processed epochs; they no longer
masquerade as conventional epoch checkpoints.

A coarse-to-fine search localized the uninterrupted treatment's crossover to
one update. At update 505, held-out Guard/Move kind accuracy is 0%/100%; at
update 506 it is 76.5%/93.4%. On the same disjoint 8,192-decision seed, update
505 reaches 86.9% exact agreement. Update 506 converts 93 exploration-owned
Guard-to-Move errors, but creates 88 exploration-owned Move-to-Guard errors and
falls to 85.9% agreement. This is a symmetric boundary exchange, not a useful
checkpoint, so neither endpoint is promoted or sent to feeding qualification.

The schema-32 512-update reproduction has different recorder bytes from the
historical schema-31 artifact, which predates the corrected hinge evaluator.
It nevertheless produces the identical 8,192-sample payload and identical
full correction telemetry—including micrologit aggregates—on a second unseen
seed. The instrumentation is therefore behaviorally non-perturbing at the
published decision precision, but byte-identical weights are not claimed.

## Fractional terminal update

Behavior-cloning schema 33 binds a learning-rate scale for only the final
retained optimizer update. A non-unit scale requires an exact total step
budget, so it retains all AdamW moments and the exact first 505 updates while
fractionating update 506 without an optimizer restart.

The fractional search establishes an unfavorable threshold ordering. At dose
0.80000 all 170 held-out Guards remain wrong and all 5,518 Moves remain right.
At 0.80625 three Moves are already wrong while no Guard is right. At 0.81250
one Guard is right but 28 Moves are wrong. At 0.87500, 130 Guards are right but
346 Moves are wrong. Thus the first Move regression precedes the first Guard
gain; no useful argmax interval is hidden inside the full update.

On the same fresh seed used for the adjacent integer checkpoints, dose 0.81250
fixes two exploration Guard-to-Move errors but creates nine exploration
Move-to-Guard errors. Exact agreement falls from the step-505 control's
7,121/8,192 (86.9%) to 7,110/8,192 (86.8%). The nine wrong Guards lead by only
0.0017 logits on average, but shrinking the dose enough to remove them also
removes the Guard gains. This direction is rejected without physical rollout.

## Local readiness factor

The new `guard-readiness-diagnostics` tool ranks the non-random header and
permutation-invariant slot maximum, minimum, and mean features by balanced
Guard/Move separability within one authoritative routing context. On the
round-four correction corpus, exploration-owned labels are 260 Guard and 2,018
Move. Assimilated energy alone reaches 98.66% balanced accuracy: quantized
energy bins 28--30 contain 202 Guards and no Moves, bin 31 contains 58 Guards
and 54 Moves, and bins 32--38 contain only Moves. Gut energy is the next best
header feature at 82.56%. The required distinction is therefore present in
ordinary anonymous cell input; missing visibility is not the cause.

Behavior-cloning schema 34 adds a zero-initialized scalar exploration Guard
readiness residual over assimilated and gut energy. It adds only three
parameters and only to the exploration Guard logit. The dedicated adaptation
mode freezes every other parameter and its unit test verifies bit-identical
non-Guard logits, other experts, targets, and recurrent memory. Schema-31--33
models migrate with exactly zero residual output.

Natural-frequency cross-entropy over 1,024 updates preserves every held-out
Move and reduces the fresh exploration Guard deficit from 17.1222 to 2.1169
logits, but plateaus without selecting Guard. A margin treatment with Guard
weight capped at 4x reaches a 0.9383-logit deficit without Move-to-Guard errors.
Continuing it crosses to 76.5% held-out Guard but regresses Move to 93.6%.
On the matched fresh seed it converts 220 exploration Guard errors but creates
62 exploration Move-to-Guard errors; trajectory feedback increases Guard
errors in other contexts and exact agreement falls from 7,043/8,192 (86.0%)
to 6,951/8,192 (84.9%). It is rejected without physical rollout.

This rules out both excess adapter capacity and absent local energy evidence.
It initially suggested an explicitly constrained readiness objective, but exact
counterfactual branches now supersede that label-level treatment.

## Counterfactual action value

The schema-2 counterfactual survey changes one cell's action at an exact
on-policy checkpoint while holding every other simultaneous action and memory
update fixed. It selected two states from each of eight new seeds and at most
one cell from a ready frontier. Guard retained more team energy in every seed
at 256 and 1,024 simulation quanta. At 4,096 quanta, Move retained more team
energy in every seed under frozen-policy continuation, while teacher
continuation split four to four. Survival and population tied throughout.

The Guard/Move boundary is therefore a horizon- and continuation-dependent
value problem, not a missing static label. No readiness correction is promoted
from this evidence. The next experiment must declare whose biological value
and what horizon it optimizes before transforming branches into supervision.

That value layer now uses lexicographic survival/population and explicit
core/assimilated/gut weights. With core and assimilated energy fully valued,
the conservative 4,096-quanta result changes from eight Guard/eight
incomparable states at zero or quarter gut weight, to five/eleven at half gut
weight, to zero Guard, one Move, and fifteen incomparable at full gut weight.
All short-horizon states prefer Guard at every tested gut weight. This is a real
adjudication seam, so no branch is converted to supervision until the intended
gut valuation is chosen and then confirmed on broader actions and seeds.

## Coarse-to-deep action frontier

An all-legal short pass shows why teacher/policy comparison was insufficient.
Across eight seeds, the +256-quanta Pareto frontier contains low/medium Guard
and minimum-effort Move in every direction. The policy's medium-effort Move is
not on it. A four-seed deep pilot and disjoint four-seed confirmation both
collapse the +4,096-quanta frontier to the eight minimum-effort Moves. After
erasing target slot, all eight are one archetype. This holds for all three value
perspectives and zero/full gut value.

The next correction may supervise the exploration Move family and minimum
effort only. It must retain the ordinary target head or sample direction from
cell-private randomness; branch futures cannot label a direction that the Mind
could not infer from its observation.

## Bound artifacts

| artifact | SHA-256 |
|---|---|
| safe treatment metadata | `0ebb829dc04b83133bac8f80a4edbbe8891eb3212e55bdaad30330ee25dfe99a` |
| safe treatment model | `e132beff2066034bce9349d79fcc20e84b5356f0513990f138686173b4da16ad` |
| safe fresh-probe manifest | `2d9fb04fd6b79d7199ce0e5f7f60d12183360c6539db95e19223d87ede721f3c` |
| safe fresh-probe payload | `104b297984ca929450911c249612e29618d7d83c89393551c0ade683f517a6e9` |
| rejected crossover metadata | `60c550122eb41422c09f13229de83d879991f574fd9f18adaa2f7160099a1c52` |
| rejected crossover model | `78c4ac680be0f52428356393ba7e61d69b6247b1a7d329de4041ade256ea8c3d` |
| crossover-recovery manifest | `a9732563a1071bbcf16fc4eaa1ee64ed5108ff1d1b527ae303cae7ad261d1dbd` |
| crossover-recovery payload | `92905988a2120eb2279b008fcc094e5345b100a05b994526d35c3adadc5aa325` |
| step-505 metadata | `f1bb79db717e6743d704b8a1f9e25dcd150ea21652716ecd831d61c5d97541a2` |
| step-505 model | `d080f34565db871eddfba0a6fa26925f1547a47a05ec493d18ed2b5c9c535bd1` |
| step-505 fresh-probe manifest | `2bed1d27d7c39d021290975120771cc2c29f8ab2ed845489c02e2941ee5783ff` |
| step-505 fresh-probe payload | `158d4dad16e0de24c9a0812e12023f509ab2387b7927a3a854903ce0da00b1d1` |
| step-506 metadata | `232a8bd9b736c0f323fabe1c4d4fda6356986a6c3e681172f520fd49fdeef9d4` |
| step-506 model | `d9e24efc7434b4b65ade3e49babea49e9b5586ec884b765932d602c0a4729e60` |
| step-506 fresh-probe manifest | `36f1ec4c6fd4a7164cf3011ffd01dc5a7a89ffad382f28e2847336828ab07dd2` |
| step-506 fresh-probe payload | `96312cfc3b576471a7330a5e0dd17788eba577bc207d754d4186ec7809289c93` |
| step-512 reproduction metadata | `e6c0f20d6c31276c0bad074469228264c315d801434aa9f901f11f6f51cc9673` |
| step-512 reproduction model | `d448bbdd5a8b2af73f14cff1fd419baf7e3933f384b2a1000c2f8fcf28b65847` |
| historical/reproduction shared fresh payload | `3881a5381e278d4c5b2a3536646b2c052424924918945b4d7aa8edc3c13d1af3` |
| dose-0.80000 metadata | `cd983b52fa31cecd35d8a8141b3d06a71dbc578e7124e67723df8d566b97afc5` |
| dose-0.80000 model | `1ced744d0499a95a559c73f3b1694df437cb165c938a13bef1f97b08a7bda79b` |
| dose-0.80625 metadata | `4131a7e936e2979f6781e5caadceb20e82c7bee64229be5605d5d55d1825e553` |
| dose-0.80625 model | `9cf60e75e0ce729dd80792811f6e7bbeae520c4ca8e91801d0eb90aebaf40dbe` |
| dose-0.81250 metadata | `e00376b6df42f5777798ab6596caa0040bb70b6189188498b1dcfee0a699bb6b` |
| dose-0.81250 model | `c2a6d0e7abf95c155f2ced36dd6eed0bc4d2b48cc63106354e035848d325120f` |
| dose-0.81250 fresh-probe manifest | `9d351252b7b8fe02967585590d2848098116a61584d151d759751427b52a0b42` |
| dose-0.81250 fresh-probe payload | `80c36889f74b4bfac5c4d62b17e4ddfc581cb8be73556fcaee3fcf4663cd899a` |
| dose-0.87500 metadata | `c54f757e5a2d46b1fb46a5d080460b5f8604867639cf10205e798d3c46edf01e` |
| dose-0.87500 model | `a726e22a4e77af6ccb6eb25aa2ba198700308db7c9abac29582e07721e0850f2` |
| readiness cross-entropy metadata | `d2071e23acdc65e28506a79a6c7ca36b95d11645f78d4b295b51269fb3b48a5f` |
| readiness cross-entropy model | `2d6f492b02147462b570716681cc1bcc6772bb587531c274b6d886616e6c7137` |
| readiness margin-safe metadata | `77aa77d744ede86cace17e7ad848542765cbb535040884adfaadc0448c41f519` |
| readiness margin-safe model | `939bca6166f1b0b7c82b9a54aeb7db41259f65824edea5e0f686cf0134da7019` |
| readiness margin-safe fresh manifest | `c86ca71e67d70fd837f8b1c9ac3256688464ef5eb8e60ada72a96bd3dc8bbb09` |
| readiness margin-safe fresh payload | `1cfd6fedffc5f483373d850d7367128a1431da609365ab99e187a116563c75d1` |
| readiness crossed metadata | `72b0ed3daccd0177bb080a502c364f8b4d3a2f54abff8d0a0049ce4e418900fc` |
| readiness crossed model | `de46bca12b22ada2b4a49e25952f30b25724d76e18f5aedd07683eb6f89b55ab` |
| readiness crossed fresh manifest | `7846e0799c5a1b2ba793f13eb90eebda6936bad037eadd6965a81686bc72e686` |
| readiness crossed fresh payload | `d5ecaf789fce4c8ef0e288ecad74821743ec7ccc4a0ea4a002037652d5f4ab9e` |
| compartment-complete counterfactual artifact | `93b6311b11ea8ba9bb464bd1158230078167243634dccbad94398feacdfe320b` |
| counterfactual value, no gut | `ea67013c8ed766a19c8c76a8df2e13a7ffa79e787ebda00ed1c66a0f66317b2f` |
| counterfactual value, quarter gut | `a62d3c6457135d9109d013559c467a6cccd7eabb89b88f462984d98784c39030` |
| counterfactual value, half gut | `d513ef44a74942274a403e9386506541e65f5ef5a94436df87d205164374ba60` |
| counterfactual value, full gut | `f9a8008a4bad001c19972d96758b8c9045003b2964c5938fb3c0a267b463c3c` |
| all-legal short branch artifact | `7298bdd04ea1ce1768b1692a14c9ef9fc48494ceefc35d2251927c730dc2c764` |
| all-legal short no-gut frontier | `83de8b2982aa93c6d658ba260a6405f81788b1fad3100009aed12bdcfb14eafa` |
| deep frontier pilot | `f68a95bf3c4a267d7e094f3ad889d82cb09a0aa3d0175c9cce2e1475d3fa7274` |
| deep frontier pilot no-gut value | `a16ef9f78fd6b636624b3345c0af96bae1ea4377703501ea4daaa032019a32ae` |
| deep frontier confirmation | `445f5bc76de21c00e9caec61edf866ff6075532777d4f9a8f2c6a6b66add7e80` |
| deep confirmation no-gut value | `a27a8eaf0ded62d5adb68593e257b26086076b039de9f6a0390b0d9fde100c8c` |

Local immutable evidence is retained under
`training-output/feeding-dagger-round5-v1`.

## Target-free Move-effort correction A/B

Historical deep branch artifacts were enriched by replaying only their source
trajectories. Enrichment fails closed unless the checkpoint hash, anonymous
Mind-observation hash, frozen-policy choice, teacher choice, expert context,
cell, frontier, and simulation time all match. This recovered four pilot and
four confirmation observations plus each cell's exact private recurrent state
without recomputing the already verified branch futures.

At the conservative +4,096-quanta objective, all eight states again have one
target-free archetype: minimum-effort Move. The paired training corpora retain
the frozen policy's target slot. The control retains its original effort; the
treatment changes only effort to zero. Behavior cloning uses `--effort-head-only`,
so every inherited parameter outside that conditional head is restored from
the verified parent after every update.

A bounded dose probe at learning rate 0.005 found a sharp exact-state boundary:
nine optimizer steps reduced held-out loss from 20.4083 to 3.0052 without
changing the greedy effort, while ten steps reached 100% held-out exact
accuracy at loss 2.2900. The identically budgeted control remained at 100%
accuracy and loss 2.1096. This establishes mechanism, not ecological benefit.
An eight-seed, two-stage local CPU feeding evaluation was stopped unpublished
after fifteen minutes; qualification should run as resumable seed shards or on
the remote accelerated runner.

The reusable paired runner is
`scripts/run_counterfactual_effort_ab.sh`.

Ecological qualification is now restart-safe and visible at seed granularity.
`feeding-evaluation` reports each completed stage/seed episode to stderr and
accepts `--resume`; reuse succeeds only when the existing artifact validates
and its full training config, config hash, clone metadata hash, model hash, and
ordered seed list exactly match the requested run. A stale, corrupt, legacy,
or reordered result fails closed instead of being silently reused.

Run the paired qualification, preferably on an accelerated remote runner, with:

```sh
scripts/qualify_counterfactual_effort_ab.sh \
  blob_rl/config/combat_warm_start_256.toml \
  training-output/feeding-dagger-round5-v1/effort-control-dose-lr005-steps10-v1 \
  training-output/feeding-dagger-round5-v1/effort-correction-dose-lr005-steps10-v1 \
  training-output/feeding-dagger-round5-v1/effort-qualification-canonical-v1
```

On a Docker host, set `BLOB_EFFORT_AB_CONTAINER_IMAGE` to the immutable CPU
image and pass absolute versions of those four paths. The runner applies a
read-only, capability-free container contract around every shard and merge.
`BLOB_EFFORT_AB_MAX_PARALLEL` bounds concurrent shards and defaults to one;
each shard keeps its own adjacent log so concurrent progress never interleaves.

The runner publishes one immutable artifact per arm and seed, continues long
enough to account for every failed shard, and merges successful shards in the
declared seed order. Rerunning the same command verifies and immediately reuses
completed shards. `BLOB_EFFORT_AB_EVALUATION_SEEDS` overrides the seed list;
`BLOB_EFFORT_AB_REQUIRE_PASS=1` additionally makes a valid failed promotion
gate fail the job. No ecological result is claimed until both merged artifacts
exist and validate.

| artifact | SHA-256 |
|---|---|
| pilot enriched branch | `8554f7757755e56595b91e220856158ed10fde909a9c34a66e886428b781dbf2` |
| pilot enriched value | `ff39cd72533883e678e1b8a458824692f73b287ec487a5279867d6ee1a0e25e9` |
| confirmation enriched branch | `7cd81b2a91ced5e48f24d4362da1277b64ddfd680b54faffb6c6b7ba8c5837c2` |
| confirmation enriched value | `161e2fa01304f8884dcd56e39e8ac6d26dbfa107113deb5029081725e9178aff` |
| pilot control corpus manifest | `1ec441561f64ccd52815b63f3182e56fd92930829f9b53d5ea899ce35b511594` |
| pilot correction corpus manifest | `9ea214742c8d7ef7807e3d9f901d8aee905510adb8d9ffffe795d808d6260835` |
| confirmation control corpus manifest | `6c46a553c75e1e1c1c9e335dec5db9a47f2293c78e14ca169f3c78022c682bdb` |
| confirmation correction corpus manifest | `74728b3af0b337934bcc6f25b002c25fc0cb6c62d4aeb546ef9cc1f32d7915f3` |
| ten-step control metadata | `fc66244380552e63d23908973eb50f3c34ef13212498575e27a52064504633a7` |
| ten-step control model | `e098ffa39ee1e247dca18dc959cdfbe8118d055260fed61b80980dcaef867bc2` |
| ten-step correction metadata | `02d51ac85b59111ae047e1f1c64cb7551028065429651ad1fde40686ff1afc69` |
| ten-step correction model | `90ec3f4e10e36e8ef15909961badb06a5e147a8580925d6e7c90d8768f98063f` |

## Canonical private-randomness correction

The completed eight-seed qualification initially reported 81.64% on-food and
74.61% adjacent-food survival for the minimum-effort treatment. A transition
audit of the two worst adjacent seeds found that every cell reaching a plant
consumed, but only 33.92% of plant-directed Moves succeeded; 64.59% were
Frustrated or Contested. The audit then exposed that learned-policy rollouts
were projecting an all-zero private-randomness block even though ordinary
native/Wasm Mind execution and teacher rollouts receive canonical cell-private
randomness.

A bound A/B on those exact seeds changed only the projected randomness mode.
Canonical randomness raised survival from 351/512 (68.55%) to 406/512
(79.30%), plant reach from 426/512 (83.20%) to 488/512 (95.31%), and reduced
Frustrated-plus-Contested plant Moves from 6,082 to 194. The fixed match seeds
reproduce these blocks exactly; no shared seed, public identity, or cross-cell
state enters the Mind ABI.

The RL host now caches and checkpoints each pending invocation's host routing
handle and private random block, making repeated reads idempotent and tamper
detecting. Feeding-evaluation schema 4 invalidates the former zero-randomness
qualification. The historical schema-3 aggregate is retained only as evidence
of the harness artifact.

The fresh eight-seed schema-4 qualification reverses the apparent treatment
benefit. The unchanged control passes with 1,650/2,048 adjacent survivors
(80.57%), while the minimum-effort correction fails with 1,614/2,048 (78.81%).
Both arms have 100% on-food survival and 100% episode success in both stages.
The treatment produces 9,936 successful adjacent Moves versus 7,777 for the
control, but only 121,851 Consume successes versus 134,314; intake falls from
201.578 to 195.658 energy per initial cell. The target-free effort correction
is therefore rejected: under correct private randomness it increases
unproductive movement and lowers survival. The passing control remains the
comparison control, pending fresh-seed and cross-layout qualification.

Both imported aggregates were independently accepted by local `--resume`
validation against the exact config, ordered seeds, metadata, and model bytes.

Fresh-seed confirmation uses `scripts/qualify_feeding_policy.sh`. Its default
seeds `1431000101,1431000202,1431000303,1431000404,1431000505,1431000606,1431000707,1431000808`
were declared only after the schema-4 control/treatment decision and are
disjoint from that qualification. The runner has the same bounded parallelism,
read-only container, per-seed immutable shard, exact resume, and validated
aggregate contracts as the paired runner.

Cross-layout confirmation uses `scripts/qualify_feeding_layouts.sh` and schema
2 layout artifacts. Schema 1 predates canonical learned-policy randomness and
cannot resume. The runner defaults to line, checkerboard, ring, loose-random,
and random layouts on fresh seeds `1432000101,1432000202`, publishing one
resumable immutable shard per layout/seed pair before a complete matrix merge.

Both follow-up qualifications are now complete. The fresh eight-seed
checkerboard control fails narrowly with 1,636/2,048 adjacent survivors
(79.88%); on-food survival and both stage-success rates remain 100%. Across the
original and fresh checkerboard suites, the sixteen seed-level survival rates
average 80.22%, with sample standard deviation 2.02 percentage points and a
seed-cluster normal 95% interval of approximately 79.24–81.21%. A bare 80%
point-estimate threshold is therefore too unstable to establish robust
competence near this boundary.

The schema-2 cross-layout matrix fails overall and exposes a structured
geometry seam. Adjacent survival is 74.2–75.4% for line and 75.0–77.0% for
ring, but 80.5–80.9% for checkerboard, 82.8–83.2% for loose-random, and
97.7–98.8% for random. All ten on-food trials have 100% survival. The control
is not a promotable general feeding baseline. The next correction should focus
on locally crowded line/ring approach and retention states; a global movement-
effort correction is contradicted by the completed A/B.

| follow-up artifact | verdict | artifact hash | file SHA-256 |
|---|---|---|---|
| fresh checkerboard seeds | FAIL | `c5e11c9686fbb9bed9eae08de7f34c8f19a8282aed4685c5d1ad16d5c92f774b` | `42506fa0a9652b3277b20104dc5e62369031e21bab2f7f6180c726278484a9ac` |
| five-layout matrix | FAIL | `70ece2f95be855e1a8143ed9d7e43cb2a7b375a2643162f17ab324b37b62d198` | `136139c3be7325849c875b1cadafbce0f08c889c031ae7179d1619df46ebf64a` |

| canonical qualification artifact | artifact hash | file SHA-256 |
|---|---|---|
| control, PASS | `fcc4294f98e19bef5862214e72802d7997b60fa083e2ae3612e97669ad224b5b` | `28482527321b6f94a709133706714a6b9bdd421318ce62f6b5f66c909b6a77c5` |
| minimum-effort correction, FAIL | `9d7e426b87d0945e3a2464560bdb39b9f045bb5f96d6b037728aa4bfa0509e3e` | `58254f034acb77579f14afbb8e5af894db6bcdcd7d77f76c8762be0dbb29ad24` |

## Exact teacher feasibility and transition attribution

The schema-2 maintained-teacher matrix uses the same layouts, seeds, rules,
stage construction, and success gates as the frozen clone. It was sharded with
`scripts/qualify_feeding_layout_teacher.sh`; all ten collision-aware-forager
trials pass. The learned transition artifacts were produced with canonical
cell-private randomness by
`scripts/diagnose_feeding_layout_transitions.sh`. Both runners resume only after
validating the exact config, policy identity, layout, and ordered seeds.

| layout | teacher adjacent survival | learned adjacent survival | learned off-plant visible `Consume` | correct plant `Move` | failed correct `Move` (Frustrated + Contested) |
|---|---:|---:|---:|---:|---:|
| line | 91.02% | 74.80% | 90.56% | 8.25% | 31.07% |
| checkerboard | 93.36% | 80.66% | 88.53% | 8.91% | 17.86% |
| ring | 91.02% | 75.98% | 89.75% | 8.74% | 26.10% |
| loose-random | 93.16% | 83.01% | 86.04% | 10.89% | 17.47% |
| random | 99.22% | 98.24% | 76.36% | 23.42% | 0.92% |

The teacher is an observation-legal Mind, so this is direct evidence that the
physics admits successful behavior on the failed geometries. The clone is not
primarily leaving food after arrival: at least 98.7% of its on-plant decisions
are `Consume`. It is repeatedly trying to consume from the empty adjacent tile.
Line and ring then add a real but secondary contention penalty. This was the
bounded action-kind hypothesis before the router semantic was checked; the
later consumable-routing section supersedes it with direct evidence.

The remote CPU image was
`sha256:09a8d1fdd53a3215f824d680cf560255852a478d516e70b6bda89fdd9087f7bb`,
labeled
`f4c7f30e2803-working-tree-20260902-feeding-disambiguation-v1`.

| new evidence | artifact hash | file SHA-256 |
|---|---|---|
| collision-aware teacher matrix, PASS | `aed636aa684b6471d5840fa5ca97f07453d75256dde08aa681af8b9ee0d83778` | `59cb23fd2b51a64807132f756a331d3d02f09550b1a0597dd08d523fe28633e6` |
| learned line transitions | `9cec3d6eb37522f09f8359540a7c523f3988e86c8d5996756edf3e661f45eb96` | `943c884649f4fd1c3b3ed46f3ff9f46bbee17e42c5513e368bd88d0d91efeb8c` |
| learned checkerboard transitions | `f3f1c8768736e629b8b1f80431bca39816933ea77d9b14e10830db0d80139bf8` | `ece33c97265069be6f6cdd03287185b7a53531f7fab48586df2a5542c2864241` |
| learned ring transitions | `cfb15bc81100383590dd5911f494a85923bf1aaff082ee8b03e9a25d1ec3d4cc` | `2e8a4f550891ba869da291c7d0690bfb4c97f0fbab1dc87fde67ad530e89df95` |
| learned loose-random transitions | `384cb53ce5918961714006ed85bd2586ecf487d6b70b99905011d50461b01642` | `134c35217e0bc24f1200036e131bf85dc46df4947f0c2fee533f3cb8e2d79c74` |
| learned random transitions | `5137aa9c8dd43805ce24eaa8f374b9aa19944b51970a85047ee09e445b1d439d` | `844d299177839658d24efc9fb2363af91c3a365636d1069f59fd2411d032dae4` |

## Consumable-routing correction and target-only A/B

The preceding action-kind diagnosis found a semantic defect in the learned
model's three-expert router. Diffuse field energy feeds plants but cannot be
consumed by a cell; routing diffuse-only current tiles to foraging made the
frozen clone select `Consume` beside visible plants. The authoritative scalar
and batched-tensor routers now agree on this precedence:

1. visible Attack/Guard activity selects interaction;
2. plant capacity or loose energy on the current tile selects foraging;
3. any other visible cell selects interaction;
4. all remaining observations select exploration.

Plant capacity deliberately keeps an exhausted plant in foraging. The first
pilot routed only positive plant energy and was discarded because it made cells
leave temporarily exhausted plants.

The unchanged model was reevaluated under the corrected runtime on the same
five layouts and qualification seeds. All ten on-food trials retain 100% of
cells. Ring improves from 76.0% to 92.8% adjacent survival, loose-random from
83.0% to 84.8%, and random from 98.2% to 99.4%. Line remains 75.0%; checkerboard
falls from 80.7% to 75.2%. The matrix therefore still fails.

| layout | adjacent survival | correct visible-plant Moves | wrong target | non-Move | correct Moves contested |
|---|---:|---:|---:|---:|---:|
| line | 75.0% | 3,968 | 0 | 0 | 80.65% |
| checkerboard | 75.2% | 3,671 | 61 | 12 | 85.42% |
| ring | 92.8% | 1,610 | 52 | 0 | 54.35% |
| loose-random | 84.8% | 2,495 | 200 | 4 | 76.83% |
| random | 99.4% | 516 | 3 | 1 | 0.39% |

This reverses the causal ranking: action kind and plant retention are now
correct, while synchronized destination choice is the dominant dense-layout
failure. Full-teacher corpora on disjoint collection seeds confirm that line
has 401 disagreements, all target-only; ring has 433, all target-only;
checkerboard has 429 target-only disagreements and only 18 kind errors.

Schema-18 matched datasets isolate that seam. Each arm contains the same 2,048
policy-induced samples and complete recurrent prefixes. The control retains the
policy target; the treatment changes only same-effort `Move` targets to a
teacher-selected reachable, vacant plant slot. `feeding-correction-pair-verify`
accepts all five pairs: 401 line, 429 checkerboard, 395 ring, 207 loose-random,
and 21 random active label changes. The model dose response trains only the
target-query head for 22, 44, or 88 updates at learning rate 0.005; all other
parameters are restored from the verified parent after every update.

The mechanism gate rejected this architecture before ecological qualification.
At 22, 44, and 88 updates, held-out Move exact accuracy remained 63.2%: the
target-query loss fell but no useful greedy target boundary crossed. Larger
256, 512, and 1,024-update treatments reduced held-out Move exact accuracy to
62.1%, 61.7%, and 60.0%; their matched controls remained 100% exact against
their original labels. The first four expensive ecological trials at 22
updates were identical between arms, so the incomplete runs were stopped and
no report was published.

This is evidence against another dose sweep. The teacher deliberately uses the
cell-private random block to distribute equal-energy slots, while the frozen
shared representation was trained before canonical learned-policy randomness
was corrected and does not expose enough useful random variation to the sole
trainable query head. The next model slice should add a zero-initialized,
permutation-equivariant target residual over the raw private-random block and
each raw slot. Its control/treatment training can then update only that adapter.
It must cross a disjoint exact-input target-agreement gate before any full
five-layout run.

The remote correction image is
`sha256:1b54eb9200baa44822dcbd84e9575c2478b97ff4d28f74454b5e1819039a49bb`,
labeled
`f4c7f30e2803-working-tree-20260902-move-target-correction-v1`.

| new evidence | artifact hash |
|---|---|
| corrected-routing five-layout matrix | `a14a80c1880b8dd87925e7bc9322c569f6d83b67041f8524d87b9bb16c8839ad` |
| corrected-routing line transitions | `6540fb8fb4d610a78c48f406fc5b0f79a589c581dd8e127709e5c4fac1305aa5` |
| corrected-routing checkerboard transitions | `85d7b89232fc831bbbee241755e223101a899c03f25a2dc99ec58fe107f782c2` |
| corrected-routing ring transitions | `23be88d3c1bf1798186412ba27e54924f07ed14e0fcea986149e839cf6558d99` |
| corrected-routing loose-random transitions | `ce9f3416aba66ec68a42ab433ac246da797c070800822ea0c967e4b9107ab4ea` |
| corrected-routing random transitions | `1f33da52cf373cdef2c498f2b5d27a80b1eb11df40921b89737b4edbc756e88f` |
| line target-treatment manifest | `9c1416dd87aeb42d082e0ac2eb153b8e1fd4966e71ae369032760cf9ca4e4367` |
| checkerboard target-treatment manifest | `57c78ccb67181c19e50a6fd414a717a04519a8df72ae034c4f98b3b6f5770b22` |
| ring target-treatment manifest | `d25672292b073d5f5237f276b8ca1820db2724a046e5e989ad6a00b84a46840e` |
| loose-random target-treatment manifest | `80737aac9f65e10a6c0de51da17f836f86409a787caf555dd09c762676bcd1ea` |
| random target-treatment manifest | `dfdec3bf4727e103a98e00d0ab01218082543de22d549ff9850fb4879ccb7d29` |
| 1,024-update target-control metadata | `f76627fe1689405e4bc8ddd0b34f7ed37ee9faeec38ecbc6615e0a0aa537997c` |
| 1,024-update target-treatment metadata | `5b21988c22f071dfe425a1d0afffd697ffcd1956def853a50a4e3235f92d05c2` |
