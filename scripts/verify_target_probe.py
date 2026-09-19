#!/usr/bin/env python3
"""Audit retained target-probe provenance and report arithmetic, not fitted weights.

Pass the directory containing command.json, status.json, probe and run/. Older
runs did not copy Cargo.lock; supply their exact --historical-lock when needed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, historical_lock=None):
    checkout = Path(__file__).resolve().parents[1]
    run = directory / "run"
    plan = read(run / "plan.json")
    report = read(run / "report.json")
    assert report["complete"] and report["plan"] == plan
    assert read(directory / "status.json")["exit_code"] == 0
    command = read(directory / "command.json")
    assert digest(directory / "probe") == command["binary_sha256"]
    assert command["environment"]["RAYON_NUM_THREADS"] == "4"
    source_files = {
        "source_sha256": run / "probe.rs",
        "portable_sampling_source_sha256": run / "portable_sampling.rs",
        "relational_source_sha256": run / "target_set_probe/relational.rs",
        "benchmark_source_sha256": run / "target_set_probe/benchmark.rs",
        "sampling_source_sha256": run / "target_set_probe/sampling.rs",
        "dependency_manifest_sha256": run / "blob_rl.Cargo.toml",
        "teacher_source_sha256": checkout / "minds/blob_mind_utils/src/lib.rs",
        "cargo_lock_sha256": historical_lock or run / "Cargo.lock",
    }
    for key, path in source_files.items():
        if key in plan:
            assert digest(path) == plan[key], (key, str(path))
    expected = set(itertools.product(plan["tasks"], plan["modes"], plan["initialization_seeds"]))
    actual = {(r["task"], r["mode"], r["initialization_seed"]) for r in report["results"]}
    assert actual == expected and len(actual) == len(report["results"])
    gates = dict.fromkeys(plan["modes"], 0)
    for result in report["results"]:
        for evaluation in [result["validation"], result.get("initial_validation")]:
            if evaluation is None:
                continue
            correct = evaluation["correct"]
            rows = evaluation["rows"]
            both = evaluation["both_correct"]
            changed = evaluation["changed_labels"]
            groups = evaluation["by_tied_best_candidate_count_correct_total"]
            assert rows == plan["validation_rows"]
            assert 0 <= correct <= rows and 0 <= both <= changed <= rows
            assert all(0 <= c <= n for c, n in groups)
            assert sum(c for c, _ in groups) == correct
            assert sum(n for _, n in groups) == rows
            assert changed == rows - groups[0][1]
            assert evaluation["accuracy"] == correct / rows
            assert evaluation["intervention_accuracy"] == both / changed
            assert evaluation["slot_permutation_bit_exact"]
            assert evaluation["row_permutation_bit_exact"]
            assert evaluation["gate_pass"] == (correct / rows >= .95 and both / changed >= .90)
        gates[result["mode"]] += result["validation"]["gate_pass"]
    cost = report["native_cost"]["results"]
    cost_modes = {r["mode"] for r in cost}
    assert {(r["batch"], r["slots"], r["mode"]) for r in cost} == set(
        itertools.product([1, 32], [8, 32], cost_modes))
    assert len(cost) == 4 * len(cost_modes)
    for measurement in cost:
        assert 0 < measurement["median_ms"] <= measurement["p95_ms"]
        assert measurement["samples"] == 101
        expected_gate = measurement["p95_ms"] < 10 if (measurement["batch"], measurement["slots"]) == (1, 32) else None
        assert measurement["single_cell_32_slot_cost_gate"] == expected_gate
        pair_bytes = (measurement["batch"] * measurement["slots"] ** 2 * 16 * 4
                      if measurement["mode"] in {"Relational", "RelationalProduct"} else 0)
        assert measurement["pair_feature_tensor_bytes"] == pair_bytes
    return report, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--historical-lock", type=Path)
    parser.add_argument("--unchanged-from", type=Path,
                        help="Check common training results against a previous revision; omit across the backend fix.")
    args = parser.parse_args()
    report, gates = verify(args.directory, args.historical_lock)
    if args.unchanged_from:
        previous, _ = verify(args.unchanged_from, args.historical_lock)
        key = lambda r: (r["task"], r["mode"], r["initialization_seed"])
        by_key = {key(r): r for r in report["results"]}
        common = 0
        for old in previous["results"]:
            if key(old) not in by_key:
                continue
            common += 1
            new = by_key[key(old)]
            for field in ["validation", "loss_checkpoints", "context_diagnostic"]:
                assert new[field] == old[field], (key(old), field)
        assert common > 0, "no common training arms to compare"
    print(json.dumps({"verified_models": len(report["results"]), "gates_passed": gates,
                      "scope": "Archived source/binary hashes, report arithmetic, and optional retained-arm equality; no weight or timing reproduction."}, indent=2))


if __name__ == "__main__":
    main()
