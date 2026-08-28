# Combat-horizon frozen-policy source cohort

This one-arm, five-seed sweep reproduces the maintained four-cycle combat
curriculum under training artifact schema 36. It exists only to publish frozen
policies for the evaluation-only horizon matrix; no training-arm comparison is
made. Seeds 45–49 all completed their 524,288-quanta-per-environment budget.

The source policies are initialized from behavior-clone metadata SHA-256
`4244392609b76655df5a6884c8375494a8259ec8a19325f87871764f065eec45`.
Their passing feeding prerequisite was freshly republished after the schema-36
configuration expansion and has artifact hash
`5a6391a112ebba9fa0fab68a551c86c2c25ffa347818eb60d22a79f36dc8672c`.
The immutable execution contract records both identities and the trainer hash.

The preceding `combat-evaluation-horizon-source-v1` directory is retained as a
pre-simulation failure. Its old feeding artifact decoded with default values
for the new combat horizon fields, but canonical re-serialization no longer
matched its embedded hash. The verifier rejected all five launches before
model loading or simulation. No compatibility exception was added.
