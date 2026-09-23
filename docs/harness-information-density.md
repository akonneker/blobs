# Test harness information density and token spend

Analysis date: 2026-09-23. The [first pilot](harness-usage.md) now implements
named test groups, bounded summaries, study adapters and structured audit failures.
The analysis below records the design and measured baseline; cache reuse, baseline
diffs and adapters for future experiments remain proposals.
The objective is fewer model-visible tokens and fewer follow-up reads per correct
diagnosis, while preserving the complete scientific and regression evidence.

## Findings from this repository

| Surface inspected | Observed output or behavior | Opportunity |
| --- | --- | --- |
| Last publication core test log | 29,214 bytes, 480 lines, 32 suite summaries; 249 passed, 23 ignored | One suite manifest and aggregate result instead of successful test names. |
| Last publication RL test log | 44,144 bytes, 694 lines, 60 suite summaries; 328 passed, 4 ignored | Same; retain zero-test binaries and exclusions in coverage accounting. |
| Context study fit + scalar replay | 483,883 + 465,552 bytes, 27 checkpoints each | Read the independently checked result before prediction arrays. |
| Context study verification | 6,664 bytes | Existing summary is already about 99.3% smaller than those raw reports combined. |
| Standardized study verification | 11,241 bytes for 24 checkpoints | Default to matched-arm outcomes and gate margins; drill into checkpoint rows on demand. |
| `scripts/conformance.sh` | 23 `cargo test` commands, shell `set -e`, no aggregate completion manifest | Fail-fast leaves later gates unexecuted without one explicit coverage record. |
| Hard/standardized Python verifiers | 33 and 17 `assert` occurrences; many omit diagnostic messages | Report the exact invariant and first differing value instead of a bare assertion. |

Measurements use `/tmp/blobs-publish-{core,rl}-tests.log` and the retained
`training-output/interaction-{context,standardized}-2026-09-13-v1` artifacts.
The publication logs were already redirected, so these sizes are potential
exposure, not proof those bytes were consumed by a model. No tokenizer or billing
measurement was made. Byte reductions must not be presented as token/cost savings.

Raw arrays and full logs are valuable evidence. Keep them in ignored artifact
storage; reduce what enters conversation context. Minifying JSON alone preserves
most low-value repetition and makes failures harder to inspect.

## Priority 0: a bounded, trustworthy run report

Add a small shared reporting layer around existing commands, initially for one
conformance group and the next normalization/calibration study. Do not rewrite
the training models, simulation or immutable historical artifacts.

Each run should write full stdout/stderr, structured results and a compact human
view derived from the same checked object. Proposed defaults: at most 2 KiB for
a passing command group, at most 4 KiB for failure detail plus the bounded status
header, and at most five representative failures. These are pilot budgets, not
limits on evidence retained or on the checks performed. Always report omitted
failure counts and an exact path/selector for retrieving the rest.

The compact view must include:

- Run identity, exact command/working directory, source content identity including
  dirty inputs, toolchain/backend/features, configuration and input hashes. Keep
  full identities in the manifest; shortened display IDs must resolve to it.
- Requested and completed suites/arms/checkpoints, pass/fail/ignored/not-run counts,
  elapsed time and subprocess exit/signal. A zero exit with missing results, no
  matched tests, truncated output or absent completion marker is not a pass.
- Separate execution status, evidence verification status and scientific outcome.
  A valid negative experiment can complete successfully while rejecting a candidate.
  Failure of a required qualification gate must still fail that gate's command.
- Counts with denominators, thresholds and signed margins, worst relevant subgroup,
  matched-control deltas, and full accounting of every declared initialization.
  Report the actual worst subgroup; do not choose attractive checkpoint examples.
- The first failed evidence stage, bounded diagnostic witnesses, exact rerun argv,
  full-log/result paths and hashes, scope and claims that remain unqualified.

Map stable gate IDs to the predeclared next diagnostic so the result directly
supports a decision. Such a mapping must not silently change thresholds, select
a favorable seed, authorize a rules change or mark an uncertain cause as proven.

For calibration, separate legal opportunities, successful actions and profitable
actions; include benefit horizon, continuation policy and censored/aborted cases.
For normalization, separate numerical validity, corpus coverage, raw-control
regression, joint fit and development/deployment status. `complete: true` alone
cannot mean the model passed a scientific gate.

Parse structured producer output where available. Do not infer success from the
last `test result: ok` line or a global regex sum. A Cargo invocation may contain
many test binaries, zero-test targets or ignored runtime gates. Maintain explicit
suite identity and expected selection; preserve each subprocess exit status through
redirection/pipelines. Mark dependent suites not-run after prerequisite failures,
but independent suites may continue to provide a fuller failure picture.

## Priority 1: make failures explain themselves

Replace bare audit assertions incrementally with explicit checks carrying stable
invariant IDs and bounded structured fields. For example, a prediction mismatch
needs the seed, arm, checkpoint, corpus row ID, expected/actual kind, legal mask,
and source artifact locator. A numerical mismatch also needs error and tolerance;
a provenance mismatch needs the expected/actual digest and file.

Python `assert` checks disappear under `python -O`. Scientific admission checks
should use unconditional validation or explicitly reject optimized execution.
This is a correctness requirement as well as a reporting improvement. Preserve
existing semantic mutation tests while migrating the verifier implementation;
historical source snapshots and already retained reports stay unchanged.

Distinguish malformed evidence, violated invariant, failed learning threshold,
unsupported environment and incomplete execution. Group repeated failures by
invariant/cause with counts, but retain all instances. Hashes locate records;
they do not explain a failure or replace independent numerical/semantic oracles.

For numerical/resolver failures, save a minimal replayable witness: fixture or
checkpoint, legal input, private memory, backend, seed, action and first divergent
field/event. Minimize only if the same invariant still fails. For performance
failures, retain the original workload and distributions; a small reproducer may
remove the relevant scaling effect. Run IDs should make witnesses retrievable
without dumping a whole corpus or re-running a long campaign.

## Priority 2: reduce repeated work and repeated context

- Add named validation groups with an explicit coverage map. A local focused
  check should say which full integration gates it did not exercise. Do not weaken
  CI or qualification requirements to reduce tokens. In particular, example
  compilation is not equivalent to running its embedded unit tests.
- Show changes against a compatible baseline: newly failing invariants, changed
  gate margins, coverage changes and unresolved failures. Always include current
  totals and scope; an unchanged failure must not disappear from the summary.
- Cache reports only by complete execution/input identities, including source
  content, dependency lockfile, toolchain, features, backend, test selection,
  environment settings and artifact hashes. Label reused results and their age.
  A branch name or commit alone cannot identify a dirty build or runtime environment.
- Expose one status file updated at stage transitions and a bounded periodic
  heartbeat during long stages. Include last progress and log location; process
  liveness is not scientific progress. Avoid repeated reads of unchanged logs
  or a new model turn for every training update.
- Keep one current decision index linked to detailed historical evidence. Current
  handoff and study reports had conflicting next steps; repeated archaeology is
  token spend that smaller test output alone will not solve.

## Information gained per experiment

The strongest savings may come from avoiding ambiguous runs. Use the roadmap's
feasibility/value/observability/fit/rollout ladder and matched interventions. An
experiment should name the competing explanations it can distinguish, controls,
the result that would change the next action, and what a null result leaves open.

Cheap admission and numerical checks should precede expensive fitting or rollout.
This does not authorize stopping declared candidate runs after seeing a convenient
scientific result: retain all-run comparison rules. Stop dependent computation
on invalid provenance, invalid numerics or infrastructure failure and report the
unexecuted work. Avoid a broad parameter sweep when one discriminating control
would identify the failed stage.

## Pilot and acceptance criteria

1. Add the reporting wrapper and use it for one existing test group, then the
   two new roadmap tracks. Preserve producer reports and scientific gates.
2. Add structured failures to the hard/standardized verification path. Extend
   semantic mutation coverage to check the reported invariant, not only rejection.
3. Only after validating the pilot, add compatible-baseline diffs and selective
   reuse. Broad harness consolidation is a later decision.

Exercise the reporter with passing, failing, zero-match, ignored-only, timeout,
killed, truncated, missing-artifact, stale-identity and unknown-output cases.
Unknown or unparseable output must be reported as unverified, never passed. Verify
that raw logs remain complete and the wrapper preserves required-gate failure exits.

Compare the old and new views on the same frozen runs and injected failures. Record
model-visible bytes and, when a tokenizer/usage record is available, actual tokens;
also count tool reads, time to identify the correct failure and missed/misclassified
failures. Suggested pilot target: at least 90% less visible output for verbose
passing groups and no loss of coverage accounting or diagnostic correctness across
the injected cases. The 2 KiB target would be about 97% below the combined 73,358-byte
publication logs if one combined report suffices; this is a target, not a measured win.

Do not claim overall token savings from output size alone: reasoning, repeated
tool calls, source reads and recovery attempts also contribute. A slightly longer
first failure report is preferable if it removes several investigative round trips.
