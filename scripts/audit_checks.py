"""Unconditional audit checks and bounded, stable failure witnesses."""
import json
import sys


class AuditFailure(AssertionError):
    def __init__(self, invariant, **details):
        self.invariant = invariant
        self.details = details
        super().__init__(json.dumps(self.record(), sort_keys=True, allow_nan=False))

    def record(self):
        return {"invariant": self.invariant, **self.details}


def bounded(value):
    text = repr(value)
    return text if len(text) <= 240 else text[:240] + "…"


def require(condition, invariant, **context):
    if not condition:
        raise AuditFailure(invariant, **context)


def equal(actual, expected, invariant, **context):
    """Locate the first differing field without printing entire arrays/corpora."""
    if actual == expected:
        return
    path = []
    while True:
        if isinstance(actual, dict) and isinstance(expected, dict):
            if actual.keys() != expected.keys():
                actual, expected = sorted(actual), sorted(expected)
                path.append("keys")
                break
            key = next(k for k in expected if actual[k] != expected[k])
            path.append(str(key))
            actual, expected = actual[key], expected[key]
        elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                actual, expected = len(actual), len(expected)
                path.append("length")
                break
            index = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
            path.append(str(index))
            actual, expected = actual[index], expected[index]
        else:
            break
    raise AuditFailure(invariant, field="/".join(path), actual=bounded(actual),
                       expected=bounded(expected), **context)


# Upstream historical preflights still use assertions. Reject optimized Python
# even when this module is imported rather than executed as a CLI.
if not __debug__:
    sys.stderr.write('{"invariant":"audit.optimized_python_unsupported"}\n')
    raise SystemExit(2)
