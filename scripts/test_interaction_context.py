#!/usr/bin/env python3
"""Exercise context-diagnostic audit rejection without modifying saved evidence."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch
import verify_interaction_context as verifier


def check(root):
    root = root.resolve(); original = verifier.read
    @lru_cache(maxsize=None)
    def cached(path): return original(path)
    cases = [
        ('donor_membership', 'memory-donors.json'),
        ('validation_in_training', 'plan.json'),
        ('unmatched_batches', 'fits/report.json'),
        ('unmatched_initialization', 'fits/report.json'),
        ('forged_training_count', 'fits/report.json'),
        ('missing_checkpoint', 'fits/report.json'),
    ]
    results = []
    for name, filename in cases:
        def mutated(path):
            path = path.resolve(); value = cached(path)
            if path != root/filename: return value
            value = deepcopy(value)
            if name == 'donor_membership':
                d = value['donors']; d[0],d[1] = d[1],d[0]
            elif name == 'validation_in_training': value['training_seeds'].append(value['validation_seeds'][0])
            elif name == 'unmatched_batches': value['results'][3]['batch_stream_sha256'] = '0'*64
            elif name == 'unmatched_initialization': value['results'][3]['initial_weights_sha256'] = '0'*64
            elif name == 'forged_training_count': value['results'][0]['metrics']['feeding']['correct'] += 1
            elif name == 'missing_checkpoint': value['results'].pop()
            return value
        with patch.object(verifier,'read',side_effect=mutated):
            try: verifier.verify(root)
            except AssertionError: results.append({'case':name,'rejected':True})
            else: raise AssertionError(f'audit accepted {name}')
    return {'complete':True,'checks':results,'scope':'Decoded JSON mutations in memory exercise semantic checks beyond on-disk digests; evidence remains unchanged.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    print(json.dumps(check(parser.parse_args().root),indent=2))
