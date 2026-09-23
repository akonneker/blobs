"""Bounded display derived only from the full run report."""
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def render(report):
    path = str(Path(report["directory"]) / "report.json")
    lines = [f"HARNESS {report['group']}: {report['status']} ({report['elapsed_seconds']:.1f}s)"]
    counts = report["coverage"]
    lines.append("Steps " + " ".join(f"{k}={v}" for k, v in counts.items()))
    totals = {k: sum(s.get("counts", {}).get(k, 0) for s in report["steps"])
              for k in ("passed", "failed", "ignored")}
    lines.append("Tests " + " ".join(f"{k}={v}" for k, v in totals.items()))
    for step in report["steps"]:
        lines.append(f"{step['id']}: execution={step['execution']} verification={step['verification']} "
                     f"science={step.get('scientific', {}).get('outcome', 'not_applicable')} exit={step.get('exit')}")
        scientific = step.get("scientific", {})
        if "arms" in scientific:
            for arm in scientific["arms"]:
                worst = arm["worst"]
                lines.append(f"  {arm['arm']} final exact={arm['passed']}/{arm['runs']}; "
                             f"worst={worst['correct']}/{worst['rows']} {worst['domain']} "
                             f"seed={worst['seed']} margin={worst['margin']}")
            lines.append(f"  checkpoints={scientific['checkpoints_completed']}/{scientific['checkpoints_requested']} "
                         f"predictions={scientific['predictions']} raw-regression={scientific['raw_control_regression']}")
            lines.append("  Not qualified: " + ", ".join(scientific["not_qualified"]))
    failures = report["failures"]
    details = []
    for failure in failures[:5]:
        # All witnesses live in report.json; never dump unbounded arrays here.
        value = json.dumps(failure, ensure_ascii=True)
        details.append(value[:650] + ("…" if len(value) > 650 else ""))
    if failures:
        lines.append(f"Failures={len(failures)}; shown={min(5, len(failures))}; omitted={max(0, len(failures)-5)}")
    header = "\n".join(lines)
    # Keep the artifact selector even when a future group has many steps.
    limit = 2048 if report["status"] == "passed" else 6144
    locator = path if len(path.encode()) <= 700 else "report.json in the requested output directory"
    scope = report['scope'][:256]
    footer = f"\nFull report/logs and exact rerun argv: {locator}\nScope: {scope}\n"
    body = header + ("\n" + "\n".join(details) if details else "")
    budget = limit - len(footer.encode()) - 40
    if len(body.encode()) > budget:
        body = body.encode()[:max(0, budget)].decode(errors="ignore") + "\n[display shortened; see full report]"
    return body + footer
