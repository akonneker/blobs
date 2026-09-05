# Observation-policy oracle diversity V1

This bounded evidence set replaces a single repeatedly sampled combat oracle
with three Mind-observation-only policies spanning different opponent counts,
objectives, and action mixtures. Every report uses search-report schema 17,
keeps synthesis and holdout seeds disjoint, and covers and succeeds on all 64
sealed holdouts. Only synthesis trajectories enter behavior cloning.

| scenario | objective | synthesis | holdout | rules | decisions | action families |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1-v-1 searching pursuer h16 | survival | 16/16 | 64/64 | 7 | 112 | 16 Wait, 32 Move, 64 Attack |
| 1-v-3 searching pursuer h16 | survival | 4/4 | 64/64 | 9 | 36 | 36 Move |
| 1-v-1 forking evader h8 | elimination | 16/16 | 64/64 | 7 | 56 | 6 Move, 50 Attack |

The evader controller branches at decision one over two distinct observation
traces. The 1-v-3 controller supplies the missing out-of-contact movement
examples instead of teaching another attack-heavy prefix. Under the audited
visible-neighbor router, the three shards contribute 64 isolated and 140
social decisions with no contradictory routing observations.

Demonstration schema 13 binds the admission evidence into each manifest:
holdout seed count, the requested minimum, complete holdout coverage, and
universal holdout objective success. The converter defaults to requiring at
least 32 such seeds. Schema 11 and 12 datasets remain readable, but newly
generated oracle data cannot qualify without successful sealed holdouts.

The combat warm-start builder accepts the three report paths as a comma-separated
`BLOB_OBSERVATION_POLICY_REPORTS` value. Its default three-oracle mixture keeps
the fixed 14,806-presentation epoch and fifteen total shares: ten feeding, two
maintained contact, and one per oracle. Thus no 36-112-decision shard controls
more than one fifteenth of optimizer presentations.

## Bound evidence

- Forking-evader report: `273073e41744052168500c88b145532417aec0a4094723f22baa750789df3a00`
- Searching-pursuer report: `059c79276976e50d5e3c2ca7a15b410c77d5c3c28396d3c3fed0d2eaf2860c78`
- Three-pursuer report: `237b1a35dfff9e996eafaa8ba9004ae0bd5e74e1a962684793158d2ce68a113b`
- Forking-evader demonstration manifest: `fbf3ec82956a9462f5203d8fa37b94abcdc1c1bd40388eb4826e2712f8e73317`
- Searching-pursuer demonstration manifest: `dc413b4ccda4c840cb7ce816c6c9c14dc5129d369d06a9c5c77540f5876e4fbf`
- Three-pursuer demonstration manifest: `624730a90501811dbd0b4ddc97a27db79019b0a7b7c2c77125a32cdd1388bfde`
- One-epoch integration smoke artifact: `bdacfcf9d599b7612f02b1a052b22ba2121f3f3dd073286ee450e09d94d98887`

Reports and generated shards are retained under
`training-output/oracle-diversity-v1`. The next experiment is a paired
visible-neighbor-routing warm start: maintained control versus this diversified
three-oracle mixture, with feeding success retained as a hard acceptance gate.
The 204-presentation smoke artifact verifies schema-13 loading, three-way
sampling, schema-19 fixed budgeting, and visible-neighbor routing end to end;
its one epoch is not evidence of learned competence.
