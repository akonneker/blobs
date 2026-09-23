"""Strict adapter for pinned stable Cargo/libtest text, with per-suite accounting."""
import re

HEADER = re.compile(r"^\s*Running (?:unittests .+|tests/\S+) \((.+)\)$")
START = re.compile(r"^running (\d+) tests?$")
TEST = re.compile(r"^test (\S+)(?: - should panic)? \.\.\. (ok|FAILED|ignored)(?:, .*)?$")
END = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out; finished in .+$")


class InvalidOutput(ValueError):
    pass


def parse(log, expected):
    suites = []
    current = None
    for line_number, line in enumerate(log.splitlines(), 1):
        header = HEADER.fullmatch(line)
        if header:
            if current is not None:
                raise InvalidOutput("new suite before prior completion")
            binary = header[1].replace("\\", "/").split("/")[-1]
            match = re.fullmatch(r"(.+)-[0-9a-f]+(?:\.exe)?", binary)
            if not match:
                raise InvalidOutput("unrecognized test binary identity")
            current = {"id": match[1], "binary": header[1], "tests": [],
                       "start_line": line_number}
        elif start := START.fullmatch(line):
            if current is None or "selected" in current:
                raise InvalidOutput("unexpected test selection")
            current["selected"] = int(start[1])
        elif test := TEST.fullmatch(line):
            if current is None or "selected" not in current:
                raise InvalidOutput("test outside selected suite")
            if any(t["name"] == test[1] for t in current["tests"]):
                raise InvalidOutput("duplicate test name")
            current["tests"].append({"name": test[1], "status": test[2], "line": line_number})
        elif end := END.fullmatch(line):
            if current is None or "selected" not in current:
                raise InvalidOutput("completion without selected suite")
            counts = dict(zip(("passed", "failed", "ignored", "measured", "filtered"),
                              map(int, end.groups()[1:])))
            observed = {key: sum(t["status"] == status for t in current["tests"])
                        for key, status in (("passed", "ok"), ("failed", "FAILED"), ("ignored", "ignored"))}
            if counts["measured"] or counts["filtered"]:
                raise InvalidOutput("unexpected benchmark or filtered tests")
            if any(counts[k] != v for k, v in observed.items()):
                raise InvalidOutput("test records disagree with completion counts")
            if sum(observed.values()) != current["selected"]:
                raise InvalidOutput("selected test count differs from completed records")
            if (end[1] == "FAILED") != bool(counts["failed"]):
                raise InvalidOutput("completion status disagrees with failed count")
            if counts["passed"] + counts["failed"] == 0:
                raise InvalidOutput("no executed tests (zero-match or ignored-only suite)")
            current.update(counts)
            current["end_line"] = line_number
            suites.append(current)
            current = None
        elif line.startswith("test result:") or line.lstrip().startswith("Running "):
            raise InvalidOutput("unknown suite/completion format")
        elif line.startswith("test ") and " ... " in line:
            raise InvalidOutput("unknown test record format")
    if current is not None:
        raise InvalidOutput("truncated suite without completion")
    ids = [s["id"] for s in suites]
    if sorted(ids) != sorted(expected) or len(ids) != len(set(ids)):
        raise InvalidOutput(f"expected suites {expected!r}; observed {ids!r}")
    for suite in suites:
        section = log.splitlines()[suite["start_line"] - 1:suite["end_line"]]
        for test in suite["tests"]:
            if test["status"] != "FAILED":
                continue
            marker = f"---- {test['name']} stdout ----"
            if marker in section:
                start = section.index(marker) + 1
                detail = []
                for line in section[start:]:
                    if line.startswith(("---- ", "failures:", "test result:")):
                        break
                    detail.append(line)
                test["detail"] = "\n".join(detail).strip()[:1500]
    return {"suites": suites, "counts": {k: sum(s[k] for s in suites)
            for k in ("passed", "failed", "ignored", "selected")}}
