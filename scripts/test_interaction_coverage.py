#!/usr/bin/env python3
"""Exercise semantic audit rejection with in-memory mutations; never edit evidence."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch

import verify_interaction_head as verifier


def check(root):
    root = root.resolve()
    original = verifier.read

    @lru_cache(maxsize=None)
    def cached(path):
        return original(path)

    def reordered_partition(path, value):
        if path == root / 'plan.json':
            value['training_seeds'].reverse()

    def forged_training_count(path, value):
        if path == root / 'portable-replay.json':
            value['results'][1]['training']['feeding']['correct'] += 1

    def swapped_predictions(path, value):
        if path == root / 'portable-replay.json':
            pred = value['results'][1]['validation'][0]['combat']['predictions']
            j = next(i for i, k in enumerate(pred) if k != pred[0])
            pred[0], pred[j] = pred[j], pred[0]

    def forged_gate(path, value):
        if path == root / 'fits/report.json':
            value['results'][1]['offline_gate_pass'] = True
        if path == root / 'fits/1435600301-treatment/result.json':
            value['offline_gate_pass'] = True

    def forged_error_pool(path, value):
        if path == root / 'feeding-pool.json':
            value['hard_indices'].pop()

    def changed_update_budget(path, value):
        if path == root / 'plan.json':
            value['steps'] *= 2

    mutations = [reordered_partition, forged_training_count, swapped_predictions, forged_gate]
    if 'feeding_sampling' in original(root / 'plan.json'):
        mutations += [forged_error_pool, changed_update_budget]
    results = []
    for mutation in mutations:
        def mutated(path):
            path = path.resolve()
            value = deepcopy(cached(path))
            mutation(path, value)
            return value

        with patch.object(verifier, 'read', side_effect=mutated):
            try:
                verifier.audit(root)
            except AssertionError:
                results.append({'case': mutation.__name__, 'rejected': True})
            else:
                raise AssertionError(f'audit accepted {mutation.__name__}')
    return {'complete': True, 'checks': results,
            'scope': 'Mutations alter decoded JSON in memory while preserving on-disk digests, to exercise semantic checks beyond hash verification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(check(parser.parse_args().root), indent=2))
