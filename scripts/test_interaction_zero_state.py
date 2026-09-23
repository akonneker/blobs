#!/usr/bin/env python3
"""Semantic mutations of the full-corpus audit; archived evidence is read-only."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch

from audit_checks import AuditFailure, equal
import verify_interaction_context as shared
import verify_interaction_zero_state as verifier


def check(root):
    root = root.resolve()
    original = verifier.read

    @lru_cache(maxsize=None)
    def cached(path):
        return original(path)

    cases = [
        ('budget', 'plan.json', 'zero.inherited_plan'),
        ('zero_bypass', 'plan.json', 'zero.transform'),
        ('parent_memory', 'fits/context-audit.json', 'zero.context_audit'),
        ('context', 'fits/context-audit.json', 'zero.context_audit'),
        ('count', 'fits/report.json', 'context.metrics'),
        ('raw_loss', 'fits/report.json', 'zero.raw_record'),
    ]
    results = []
    for name, filename, invariant in cases:
        def mutated(path):
            path = path.resolve()
            value = cached(path)
            if path != root / filename:
                return value
            value = deepcopy(value)
            if name == 'budget': value['steps'] += 1
            elif name == 'zero_bypass': value['normalization']['zero_state'] = 'standardize all rows'
            elif name == 'parent_memory': value['parent_memory_sha256'] = '0' * 64
            elif name == 'context': value['transformed_context_sha256'] = '0' * 64
            elif name == 'count': value['results'][3]['metrics']['feeding']['correct'] -= 1
            elif name == 'raw_loss': value['results'][0]['losses'][0]['loss'] += 1
            return value

        with patch.object(verifier, 'read', side_effect=mutated), patch.object(shared, 'read', side_effect=mutated):
            try:
                verifier.verify(root)
            except AuditFailure as error:
                equal(error.invariant, invariant, 'mutation.wrong_invariant', case=name)
                results.append({'case': name, 'rejected': True, 'invariant': invariant})
            else:
                raise AssertionError(f'audit accepted {name}')
    return {'complete': True, 'checks': results, 'scope': 'Decoded semantic mutations beyond digest checks; no evidence files changed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(check(parser.parse_args().root), indent=2))
