#!/usr/bin/env python3
"""Recompute the deterministic training-only hard-example fixture and saved-fit checks."""
from audit_checks import equal, require
import hashlib
import math
import struct
from pathlib import Path
from diagnose_interaction_fit import ordinary
from verify_feeding_ecology import read, sha
from verify_interaction_context import preflight as source_preflight


def select(combat, feeding, hard, seed):
    selected = sorted(hard, key=lambda i: hashlib.sha256(seed.to_bytes(8, 'little') + i.to_bytes(8, 'little')).digest())[:64]
    features = {i: ordinary(r) for i, r in enumerate(combat) if r['teacher_kind'] == 4}
    chosen, distances = [], []
    for i in selected:
        target = ordinary(feeding[i])
        distance, index = min((sum((a-b)**2 for a, b in zip(target, feature)), j) for j, feature in features.items() if j not in chosen)
        chosen.append(index)
        distances.append(distance)
    return chosen, selected, distances


def preflight(root):
    p = read(root/'plan.json')
    source = Path(p['source_study'])
    equal(sha(source/'plan.json'), p['source_plan_sha256'], 'hard.source_plan', artifact=str(source/'plan.json'))
    equal(sha(source/'verification.json'), p['source_verification_sha256'], 'hard.source_verification')
    old, combat, feeding, _ = source_preflight(source)
    for key in ['parent', 'parent_manifest_sha256', 'parent_weights_sha256', 'training_seed', 'training_seeds', 'validation_seeds', 'sampling_seeds', 'seed_audit_sha256', 'parameters', 'learning_rate', 'optimizer']:
        equal(p[key], old[key], 'hard.inherited_plan', key=key)
    equal(sha(root/'seed-audit.json'), p['seed_audit_sha256'], 'hard.seed_audit')
    for key, expected in {'arms': ['observation', 'memory'], 'steps': 4096, 'checkpoints': [128, 512, 2048, 4096], 'batch': 128, 'parameters': 7514}.items():
        equal(p[key], expected, 'hard.plan', key=key)
    pool_path = Path(old['source_study'])/'feeding-pool.json'
    equal(sha(pool_path), p['source_pool_sha256'], 'hard.pool')
    equal(sha(root/'fixture.json'), p['fixture_sha256'], 'hard.fixture_digest')
    fixture = read(root/'fixture.json')
    c, f, d = select(combat, feeding, read(pool_path)['hard_indices'], p['sampling_seeds'][0])
    for key, expected in [('combat_indices', c), ('feeding_indices', f), ('squared_distances', d)]:
        equal(fixture[key], expected, 'hard.fixture_selection', key=key)
    rows = [combat[i] for i in c] + [feeding[i] for i in f]
    equal(fixture['rows'], rows, 'hard.fixture_rows')
    require(len(rows) == 128 and len(set(c)) == len(set(f)) == 64, 'hard.fixture_counts')
    keys = {}
    for i, row in enumerate(rows):
        label = row['teacher_kind'] if i < 64 else row['selected_kind']
        equal(label, 4 if i < 64 else 3, 'hard.fixture_label', row=i)
        key = ordinary(row) + tuple(row['memory']) + tuple(row['legal_kinds'])
        equal(label, keys.setdefault(key, label), 'hard.contradictory_input', row=i)
    return p, rows


def verify(root):
    p, rows = preflight(root)
    regression = root/'scalar-regression'
    require(read(regression/'verification.json')['all_prior_replay_records_identical'], 'hard.prior_replay')
    equal(sha(regression/'probe'), read(regression/'verification.json')['executable_sha256'], 'hard.prior_probe')
    equal(read(regression/'scalar-replay.json'), read(Path(p['source_study'])/'scalar-replay.json'), 'hard.prior_predictions')
    from diagnose_hard_separability import verify_saved
    equal(verify_saved(root), read(root/'linear-verification.json'), 'hard.linear_control')
    return verify_fits(root, p, rows)


def verify_fits(root, p, rows):
    report, replay = read(root/'fits/report.json'), read(root/'scalar-replay.json')
    prov = read(root/'provenance.json')
    equal(prov['plan_sha256'], sha(root/'plan.json'), 'fit.plan_digest')
    equal(prov['executable_sha256'], sha(root/'probe'), 'fit.probe_digest')
    for name, digest in prov['sources'].items():
        equal(sha(root/'sources'/name), digest, 'fit.source_digest', artifact=name)
    require(report['complete'] is True, 'fit.incomplete')
    equal(report['plan_sha256'], sha(root/'plan.json'), 'fit.report_plan')
    require(replay['complete'] is True, 'fit.replay_incomplete')
    equal(replay['fit_report_sha256'], sha(root/'fits/report.json'), 'fit.replay_digest')
    equal(read(root/'fits/decoder-sanity.json'), {'rows': 128, 'correct': 128, 'scope': 'Label-coded score control bypasses learning; verifies legal labels and inherited kind decoding only.'}, 'fit.decoder_sanity')
    expected = [(seed, arm, step) for seed in p['sampling_seeds'] for arm in p['arms'] for step in p['checkpoints']]
    for name, value in [('fit', report), ('replay', replay)]:
        equal([(r['seed'], r['arm'], r['step']) for r in value['results']], expected, 'fit.checkpoint_coverage', report=name)
    summary, initial, mismatches = [], {}, 0
    for fit, check in zip(report['results'], replay['results']):
        context = {k: fit[k] for k in ['seed', 'arm', 'step']}
        equal(fit['initial_weights_sha256'], initial.setdefault(fit['seed'], fit['initial_weights_sha256']), 'fit.initialization', **context)
        equal(fit['weights_file'], f"fits/{fit['seed']}-{fit['arm']}-{fit['step']}.f32", 'fit.weight_path', **context)
        weight = root/fit['weights_file']
        for digest in [fit['weights_sha256'], check['weights_sha256']]:
            equal(sha(weight), digest, 'fit.weight_digest', artifact=str(weight), **context)
        data = weight.read_bytes()
        equal(len(data), 7514*4, 'fit.weight_size', **context)
        require(all(math.isfinite(x) for x in struct.unpack('<7514f', data)), 'fit.nonfinite_weights', **context)
        equal(fit['activation'], check['activation'], 'fit.activation', **context)
        require(math.isfinite(fit['loss']) and math.isfinite(check['scalar_loss']) and abs(fit['loss']-check['scalar_loss']) < 1e-4,
                'fit.scalar_loss', actual=repr(fit['loss']), expected=repr(check['scalar_loss']), tolerance=1e-4, **context)
        metrics = {}
        for kind, subset in [('combat', rows[:64]), ('feeding', rows[64:])]:
            m, c = fit['metrics'][kind], check['metrics'][kind]
            ctx = {**context, 'domain': kind, 'artifact': str(root/'fits/report.json')}
            labels = [r['teacher_kind'] if kind == 'combat' else r['selected_kind'] for r in subset]
            equal(len(m['predictions']), 64, 'fit.prediction_count', **ctx)
            equal(len(c['predictions']), 64, 'fit.prediction_count', **ctx)
            for i, (row, actual, expected_kind) in enumerate(zip(subset, m['predictions'], c['predictions'])):
                row_id = i if kind == 'combat' else i + 64
                require(type(actual) is int and 0 <= actual < 10 and row['legal_kinds'][actual], 'fit.illegal_prediction', row=row_id, actual=repr(actual), legal_mask=row['legal_kinds'], **ctx)
                equal(actual, expected_kind, 'fit.prediction_mismatch', row=row_id, legal_mask=row['legal_kinds'], **ctx)
            correct = sum(k == label for k, label in zip(m['predictions'], labels))
            for key, expected_value in [('correct', correct), ('rows', 64), ('agreement', correct/64)]:
                equal(m[key], expected_value, 'fit.metrics', key=key, producer='fit', **ctx)
                equal(c[key], expected_value, 'fit.metrics', key=key, producer='replay', **ctx)
            require(type(m['scalar_burn_mismatches']) is int and m['scalar_burn_mismatches'] == 0, 'fit.scalar_burn_mismatch', actual=m['scalar_burn_mismatches'], **ctx)
            mismatches += m['scalar_burn_mismatches']
            metrics[kind] = correct
        summary.append({'seed': fit['seed'], 'arm': fit['arm'], 'step': fit['step'], **metrics, 'exact_fit': all(v == 64 for v in metrics.values()), 'activation': fit['activation'], 'loss': fit['loss']})
    return {'complete': True, 'plan_sha256': sha(root/'plan.json'), 'fixture_rows': 128, 'validation_rows_loaded': 0, 'checkpoint_predictions': len(expected)*128, 'scalar_burn_kind_mismatches': mismatches, 'initial_weights_match_across_arms': True, 'results': summary, 'scope': 'Small training-only overfit fixture, full batch, no deployment or generalization qualification.'}


if __name__ == '__main__':
    from harness.study import cli
    cli('hard', verify, preflight)
