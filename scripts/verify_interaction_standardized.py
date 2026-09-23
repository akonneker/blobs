#!/usr/bin/env python3
"""Audit an isolated, fixture-derived memory standardization intervention."""
from audit_checks import equal, require
import hashlib
import math
from pathlib import Path
import struct
from verify_feeding_ecology import read, sha
from verify_interaction_hard import preflight as prior_preflight, verify_fits


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def digest(values):
    return hashlib.sha256(b''.join(struct.pack('<f', x) for x in values)).hexdigest()


def preflight(root):
    p = read(root/'plan.json')
    source = Path(p['source_study'])
    require(source.resolve() != root.resolve(), 'normalization.source_cycle')
    equal(sha(source/'plan.json'), p['source_plan_sha256'], 'normalization.source_plan')
    equal(sha(source/'verification.json'), p['source_verification_sha256'], 'normalization.source_verification')
    old, rows = prior_preflight(source)
    equal(sorted(p), sorted(set(old) | {'normalization'}), 'normalization.plan_fields')
    changed = {'source_study', 'source_plan_sha256', 'source_verification_sha256', 'arms', 'scope'}
    for key in old:
        if key not in changed:
            equal(p[key], old[key], 'normalization.inherited_plan', key=key)
    equal(p['arms'], ['memory', 'standardized'], 'normalization.arms')
    for name, key in [('fixture.json', 'fixture_sha256'), ('seed-audit.json', 'seed_audit_sha256')]:
        for base in [root, source]:
            equal(sha(base/name), p[key], 'normalization.inherited_digest', artifact=str(base/name))
    equal(p['normalization']['file'], 'memory-normalizer.json', 'normalization.path')
    equal(sha(root/'memory-normalizer.json'), p['normalization']['sha256'], 'normalization.digest')
    normalizer = read(root/'memory-normalizer.json')
    equal(normalizer['fixture_sha256'], p['fixture_sha256'], 'normalization.fixture')
    equal(normalizer['rows'], 128, 'normalization.rows')
    means = [sum(row['memory'][j] for row in rows)/128 for j in range(128)]
    std = [math.sqrt(sum((row['memory'][j]-means[j])**2 for row in rows)/128) for j in range(128)]
    equal(normalizer['means'], [f32(v) for v in means], 'normalization.means')
    equal(normalizer['scales'], [f32(v or 1.) for v in std], 'normalization.scales')
    equal(normalizer['constant_channels'], [i for i, v in enumerate(std) if v == 0], 'normalization.constant_channels')
    transformed = [f32(f32(f32(x)-normalizer['means'][j])/normalizer['scales'][j]) for row in rows for j, x in enumerate(row['memory'])]
    audit = {'fixture_sha256': p['fixture_sha256'], 'rows': 128, 'context_width': 128,
             'parent_memory_sha256': digest(x for row in rows for x in row['memory']),
             'standardized_context_sha256': digest(transformed)}
    return p, rows, audit


def verify(root):
    p, rows, context_audit = preflight(root)
    actual_audit = read(root/'fits/context-audit.json')
    equal(sorted(actual_audit), sorted(context_audit), 'normalization.context_fields')
    for key, expected in context_audit.items():
        equal(actual_audit[key], expected, 'normalization.' + key)
    result = verify_fits(root, p, rows)
    source = Path(p['source_study'])
    old = read(source/'fits/report.json')['results']
    previous = {(r['seed'], r['step']): r for r in old if r['arm'] == 'memory'}
    actual = [r for r in read(root/'fits/report.json')['results'] if r['arm'] == 'memory']
    require(len(actual) == len(previous) == 12, 'normalization.raw_coverage')
    for r in actual:
        context = {'seed': r['seed'], 'step': r['step']}
        equal(r, previous[r['seed'], r['step']], 'normalization.raw_record', **context)
        equal((root/r['weights_file']).read_bytes(), (source/r['weights_file']).read_bytes(), 'normalization.raw_weights', **context)
    return {**result, 'raw_twelve_checkpoints_and_weights_identical': True, 'context_transform_verified_independently': True,
            'scope': 'Only residual memory standardization changes. Original parent memory, fixture, initialization, architecture, optimizer and update budget are retained. Training-only; no deployment or generalization qualification.'}


if __name__ == '__main__':
    from harness.study import cli
    cli('standardized', verify, preflight)
