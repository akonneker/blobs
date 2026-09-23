#!/usr/bin/env python3
"""Run a bounded reporting pilot; full logs stay under ignored training-output/."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import uuid

from harness.runner import run
from harness.provenance import sha

ROOT = Path(__file__).resolve().parent.parent


def group_steps(group, directory, study=None, plan_sha256=None):
    if group in ("core", "rl-probes"):
        cases = [(p, ["-p", p, "--lib"]) for p in ("blob_interface", "blob_engine", "blob_policy")] if group == "core" else [
            (name, ["-p", "blob_rl", "--no-default-features", "--features", "ndarray", kind, name])
            for kind, name in [("--test", "backend_numerics"), ("--test", "seed_lineage"),
                               ("--example", "context_separability_probe"), ("--example", "hard_fixture_probe"),
                               ("--example", "target_set_probe"), ("--example", "feeding_utility_probe")]]
        return [{"id": name, "adapter": "cargo", "expected_suites": [name],
                 "argv": ["cargo", "test", "--release", "--locked", *selection, "--", "--format", "pretty", "--color", "never"]}
                for name, selection in cases]
    artifact = directory / "study.json"
    return [{"id": group, "adapter": "study", "kind": group, "plan_sha256": plan_sha256, "study_root": str(study),
             "artifact": str(artifact), "argv": [sys.executable, "-B", str(ROOT / "scripts" / f"verify_interaction_{group}.py"),
             str(study), "--harness-output", str(artifact), "--plan-sha256", plan_sha256]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["core", "rl-probes", "hard", "standardized"], required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds per command")
    parser.add_argument("--require-scientific-pass", action="store_true")
    a = parser.parse_args()
    is_study = a.group in ("hard", "standardized")
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if is_study != bool(a.study) or (not is_study and (a.plan_sha256 or a.require_scientific_pass)):
        parser.error("study arguments and scientific gate apply only to hard/standardized groups; --study is required there")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = (a.out or ROOT / "training-output" / "harness" / (a.group + "-" + run_id)).resolve()
    if not directory.is_relative_to(ROOT / "training-output") or directory == ROOT / "training-output":
        parser.error("--out must be a new directory under this repository's training-output/")
    if a.study:
        a.study = a.study.resolve()
        if directory.is_relative_to(a.study):
            parser.error("report output must not modify the archived study")
        if a.plan_sha256 is None:
            a.plan_sha256 = sha(a.study / "plan.json")
    scope = {"core": "Native interface/engine/policy library tests only; integration, RL, WASM, browser and fuzz gates not run.",
             "rl-probes": "NdArray numerical/lineage and four example unit suites; full workspace, GPU, WASM and ecology not run.",
             "hard": "Audit existing hard-fixture evidence; no training, development or deployment qualification.",
             "standardized": "Audit existing normalization evidence; no new training, development or deployment qualification."}[a.group]
    steps = group_steps(a.group, directory, a.study, a.plan_sha256)
    return run(ROOT, a.group, directory, steps, scope, a.timeout, a.require_scientific_pass)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"HARNESS initialization failed: {error}", file=sys.stderr)
        raise SystemExit(2)
