"""Study adapter: verify old evidence, report fit separately from integrity."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from audit_checks import AuditFailure, equal, require
from .provenance import sha

PROTOCOL = "blob.harness.study.v1"


def inputs(root):
    # Top-level evidence and fit weights; historical provenance/readers also
    # verify the source-study chain and archived executable/source digests.
    paths = list(root.glob("*.json")) + list((root / "fits").glob("*.json"))
    paths += list((root / "fits").glob("*.f32"))
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def scientific(result, kind):
    if kind == "zero-state":
        from .full_corpus import scientific as full_corpus
        return full_corpus(result)
    rows = result["results"]
    require(bool(rows), "study.empty_results")
    keys = [(r["seed"], r["arm"], r["step"]) for r in rows]
    require(len(keys) == len(set(keys)), "study.duplicate_checkpoint")
    for row in rows:
        require(all(type(row[k]) is int and 0 <= row[k] <= 64 for k in ("combat", "feeding")), "study.invalid_counts")
        equal(row["exact_fit"], row["combat"] == row["feeding"] == 64, "study.exact_fit")
    last = max(r["step"] for r in rows)
    target = "standardized" if kind == "standardized" else "memory"
    final = [r for r in rows if r["step"] == last and r["arm"] == target]
    require(bool(final), "study.empty_target")
    arms = []
    for arm in dict.fromkeys(r["arm"] for r in rows):
        group = [r for r in rows if r["step"] == last and r["arm"] == arm]
        worst = min(((r[k], r["seed"], k) for r in group for k in ("combat", "feeding")))
        arms.append({"arm": arm, "runs": len(group), "passed": sum(r["exact_fit"] for r in group),
                     "worst": {"correct": worst[0], "rows": 64, "seed": worst[1], "domain": worst[2],
                               "threshold": 64, "margin": worst[0] - 64}})
    paired = []
    control = "memory" if kind == "standardized" else "observation"
    for r in final:
        c = next(x for x in rows if x["seed"] == r["seed"] and x["step"] == last and x["arm"] == control)
        paired.append({"seed": r["seed"], "combat_delta": r["combat"] - c["combat"],
                       "feeding_delta": r["feeding"] - c["feeding"]})
    return {"outcome": "passed" if all(r["exact_fit"] for r in final) else "rejected",
            "gate": "all_target_initializations_exact_at_final_checkpoint", "target_arm": target,
            "step": last, "arms": arms, "paired_correct_deltas": paired,
            "checkpoints_completed": len(rows), "checkpoints_requested": len(rows),
            "predictions": result["checkpoint_predictions"],
            "raw_control_regression": result.get("raw_twelve_checkpoints_and_weights_identical"),
            "not_qualified": ["full-corpus transfer", "development", "deployed combat", "self-play"],
            "scope": "Descriptive final-checkpoint training fit; no historical promotion gate is changed."}


def validate(envelope, kind, plan_sha256, study_root):
    equal(envelope["protocol"], PROTOCOL, "study.protocol")
    equal(envelope["kind"], kind, "study.kind")
    equal(envelope["plan_sha256"], plan_sha256, "study.plan_identity")
    equal(envelope["study_root"], str(Path(study_root).resolve()), "study.root_identity")
    require(envelope["verification"] == "passed", "study.verification", failure=envelope.get("failure"))
    equal(envelope["result"]["plan_sha256"], plan_sha256, "study.result_identity")
    require(envelope["result"]["complete"] is True, "study.incomplete")
    plan_path = Path(study_root) / "plan.json"
    equal(sha(plan_path), plan_sha256, "study.plan_identity")
    plan = json.loads(plan_path.read_text())
    expected = [(seed, arm, step) for seed in plan["sampling_seeds"] for arm in plan["arms"] for step in plan["checkpoints"]]
    equal([(r["seed"], r["arm"], r["step"]) for r in envelope["result"]["results"]], expected, "study.checkpoint_coverage")
    nrows = 128
    if kind == "zero-state":
        nrows = sum(r['combat'] + r['feeding'] for r in plan['training_counts'])
        equal(envelope['result']['training_rows'], nrows, 'study.training_coverage')
        equal(envelope['result']['thresholds'], plan['thresholds'], 'study.thresholds')
        for row in envelope['result']['results']:
            for domain in ('combat', 'feeding'):
                equal(row['counts'][domain]['rows'], sum(r[domain] for r in plan['training_counts']), 'study.domain_coverage')
    equal(envelope["result"]["checkpoint_predictions"], len(expected) * nrows, "study.prediction_coverage")
    equal(envelope["scientific"], scientific(envelope["result"], kind), "study.summary")
    require(bool(envelope["inputs"]), "study.missing_inputs")
    for path, digest in envelope["inputs"].items():
        equal(sha(path), digest, "study.stale_input", artifact=path)
    return envelope["scientific"]


def cli(kind, verify, preflight):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--harness-output", type=Path)
    parser.add_argument("--plan-sha256")
    a = parser.parse_args()
    if a.harness_output and (a.preflight or not a.plan_sha256):
        parser.error("--harness-output requires --plan-sha256 and a complete audit")
    root = a.root.resolve()
    envelope = {"protocol": PROTOCOL, "kind": kind, "plan_sha256": a.plan_sha256,
                "study_root": str(root),
                "verification": "failed", "scientific": {"outcome": "not_evaluated"}}
    try:
        before = inputs(root) if a.harness_output else None
        if a.plan_sha256:
            equal(sha(root / "plan.json"), a.plan_sha256, "study.plan_identity")
        if a.preflight:
            value = preflight(root)
            result = {"complete": True, "plan_sha256": sha(root / "plan.json"), "validation_rows_loaded": 0}
            if kind == 'zero-state':
                result = value[4]
            else:
                result.update({"context_audit": value[2]} if kind == "standardized" else {"fixture_rows": len(value[1])})
        else:
            result = verify(root)
        if a.harness_output:
            equal(inputs(root), before, "study.inputs_changed")
            envelope.update(verification="passed", result=result, inputs=before,
                            scientific=scientific(result, kind))
            with a.harness_output.open("x") as f:
                json.dump(envelope, f, indent=2, allow_nan=False)
                f.write("\n")
        else:
            print(json.dumps(result, indent=2, allow_nan=False))
    except Exception as error:
        failure = error.record() if isinstance(error, AuditFailure) else {
            "invariant": "study.legacy_assertion" if isinstance(error, AssertionError) else "study.invalid_evidence",
            "type": type(error).__name__, "message": str(error)[:500]}
        envelope["failure"] = failure
        if a.harness_output and not a.harness_output.exists():
            with a.harness_output.open("x") as f:
                json.dump(envelope, f, indent=2, allow_nan=False)
                f.write("\n")
        print(json.dumps(failure, allow_nan=False), file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
