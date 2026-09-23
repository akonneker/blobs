"""Execute named groups without dropping logs, exits, or unexecuted coverage."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from audit_checks import AuditFailure
from .cargo import parse
from .provenance import sha, snapshot, environment
from .reporting import render, write_json
from .study import validate


def stop(process):
    # Cargo and its test children share a new session. Kill the entire group,
    # including descendants if the immediate child has already exited.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            continue
        if sig == signal.SIGTERM:
            continue


def execute(step, directory, root, env, timeout, heartbeat):
    log = directory / (step["id"] + ".log")
    result = {**step, "execution": "not_run", "verification": "unverified",
              "scientific": {"outcome": "not_evaluated" if step["adapter"] == "study" else "not_applicable"},
              "log": str(log), "exit": None, "failures": []}
    start = time.monotonic()
    process = None
    try:
        with log.open("xb") as output:
            process = subprocess.Popen(step["argv"], cwd=root, env=env,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            result["execution"] = "running"
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0:
                    result["execution"] = "timeout"
                    stop(process)
                    break
                try:
                    process.wait(timeout=min(5, remaining))
                    result["execution"] = "passed" if process.returncode == 0 else "failed"
                    break
                except subprocess.TimeoutExpired:
                    heartbeat(step["id"], time.monotonic() - start)
    except KeyboardInterrupt:
        result["execution"] = "interrupted"
        if process:
            stop(process)
    except OSError as error:
        result["execution"] = "failed"
        result["failures"].append({"invariant": "runner.spawn", "message": str(error)})
    finally:
        if process is not None:
            if process.poll() is None:
                stop(process)
            result["exit"] = process.returncode
            result["signal"] = -process.returncode if process.returncode is not None and process.returncode < 0 else None
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    result["log_sha256"] = sha(log)
    result["log_bytes"] = log.stat().st_size
    try:
        if step["adapter"] == "cargo":
            if log.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("test log exceeds adapter's 32 MiB parse limit; full log retained")
            parsed = parse(log.read_text(errors="replace"), step["expected_suites"])
            result.update(parsed)
            result["verification"] = "passed"
            for suite in parsed["suites"]:
                for test in suite["tests"]:
                    if test["status"] == "FAILED":
                        result["failures"].append({"invariant": "test.failed", "suite": suite["id"],
                                                   "test": test["name"], "detail": test.get("detail", "See full log"),
                                                   "log": str(log), "line": test["line"]})
        elif step["adapter"] == "study":
            path = Path(step["artifact"])
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("study report exceeds 32 MiB parse limit")
            envelope = json.loads(path.read_text())
            if envelope.get("verification") == "failed":
                result["failures"].append(envelope.get("failure", {"invariant": "study.unexplained_failure"}))
            result["scientific"] = validate(envelope, step["kind"], step["plan_sha256"], step["study_root"])
            result["verification"] = "passed"
            result["artifact_sha256"] = sha(path)
        else:
            raise ValueError("unknown adapter")
    except Exception as error:
        result["verification"] = "failed"
        if isinstance(error, AuditFailure):
            witness = error.record()
        else:
            # Compiler/setup failures have no libtest completion record. Include
            # useful context without asking the reader to reopen a whole log.
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 8192))
                tail = stream.read().decode(errors="replace").splitlines()
            first = next((i for i, line in enumerate(tail) if line.startswith(("error", "Traceback"))), max(0, len(tail) - 12))
            witness = {"invariant": "runner.unverified_output", "type": type(error).__name__,
                       "message": str(error)[:700], "log": str(log),
                       "excerpt": "\n".join(tail[first:first + 12])[:1500]}
        result["failures"].append(witness)
        (directory / (step["id"] + ".adapter-error.log")).write_text(traceback.format_exc())
    if result["execution"] != "passed":
        result["failures"].append({"invariant": "runner.execution", "status": result["execution"],
                                   "exit": result["exit"], "log": str(log)})
    return result


def failed(step, require_scientific):
    return (step["execution"] != "passed" or step["verification"] != "passed"
            or bool(step.get("counts", {}).get("failed"))
            or (require_scientific and step["scientific"]["outcome"] != "passed"))


def run(root, group, directory, steps, scope, timeout, require_scientific=False):
    directory.mkdir(parents=True, exist_ok=False)
    before = snapshot(root)
    env = dict(os.environ, CARGO_INCREMENTAL="0", CARGO_TERM_COLOR="never",
               RUST_TEST_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    provenance = {"source": before, "environment": environment(root, env),
                  "cwd": str(root), "steps": steps, "timeout_per_step": timeout,
                  "required_scientific_pass": require_scientific, "cache_reused": False}
    write_json(directory / "manifest.json", provenance)
    start = time.monotonic()
    results = []

    def heartbeat(step, elapsed):
        write_json(directory / "status.json", {"status": "running", "step": step,
                   "step_elapsed_seconds": round(elapsed, 1), "completed": len(results),
                   "requested": len(steps), "updated_unix_seconds": time.time(),
                   "log": str(directory / (step + ".log")), "note": "Liveness, not scientific progress."})

    interrupted = False
    for step in steps:
        if interrupted:
            results.append({**step, "execution": "not_run", "verification": "unverified",
                            "scientific": {"outcome": "not_evaluated"}, "exit": None, "failures": []})
            continue
        heartbeat(step["id"], 0)
        result = execute(step, directory, root, env, timeout, heartbeat)
        results.append(result)
        interrupted = result["execution"] == "interrupted"
    after = snapshot(root)
    failures = [{"step": r["id"], **f} for r in results for f in r["failures"]]
    if after["sha256"] != before["sha256"]:
        failures.insert(0, {"invariant": "runner.source_changed", "expected": before["sha256"], "actual": after["sha256"]})
    for r in results:
        if require_scientific and r["verification"] == "passed" and r["scientific"]["outcome"] != "passed":
            failures.append({"step": r["id"], "invariant": "science.required_gate", "outcome": r["scientific"]["outcome"]})
    bad = bool(failures) or any(failed(r, require_scientific) for r in results)
    report = {"protocol": "blob.harness.run.v1", "group": group, "directory": str(directory),
              "scope": scope, "status": "failed" if bad else "passed", "steps": results,
              "coverage": {"requested": len(steps), "completed": sum(r["execution"] in ("passed", "failed") for r in results),
                           "not_run": sum(r["execution"] == "not_run" for r in results),
                           "incomplete": sum(r["execution"] in ("timeout", "interrupted") for r in results)},
              "elapsed_seconds": round(time.monotonic() - start, 3), "failures": failures,
              "manifest_sha256": sha(directory / "manifest.json"), "source_after_sha256": after["sha256"]}
    output = render(report)
    report["display_bytes"] = len(output.encode())
    report["raw_log_bytes"] = sum(r.get("log_bytes", 0) for r in results)
    write_json(directory / "report.json", report)
    (directory / "summary.txt").write_text(output)
    write_json(directory / "status.json", {"status": report["status"], "report": str(directory / "report.json"),
                                          "updated_unix_seconds": time.time()})
    print(output, end="")
    return 130 if interrupted else int(bad)
