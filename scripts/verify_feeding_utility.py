#!/usr/bin/env python3
"""Audit archived feeding-utility reports; optionally compare common fits."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluation(value, rows, active):
    assert (value['rows'], value['active_rows']) == (rows, active)
    assert value['retention_rows'] == rows - active
    for numerator, denominator, rate in [
        ('correct', 'rows', 'accuracy'),
        ('active_correct', 'active_rows', 'active_accuracy'),
        ('retention_correct', 'retention_rows', 'retention_accuracy'),
        ('common_treatment_correct', 'rows', 'common_treatment_accuracy'),
    ]:
        assert 0 <= value[numerator] <= value[denominator]
        assert value[rate] == value[numerator] / value[denominator]
    assert value['correct'] == value['active_correct'] + value['retention_correct']
    assert value['gate_pass'] == (value['accuracy'] >= .95
                                  and value['active_accuracy'] >= .90
                                  and value['retention_accuracy'] >= .95)
    assert value['slot_permutation_bit_exact'] and value['row_permutation_bit_exact']
    assert {v['layout'] for v in value['by_layout']} == {
        'line', 'checkerboard', 'ring', 'loose-random', 'random'}
    assert len(value['by_layout']) == 5
    for field in ['rows', 'correct', 'active_rows', 'active_correct',
                  'retention_rows', 'retention_correct']:
        assert sum(v[field] for v in value['by_layout']) == value[field]
    for v in value['by_layout']:
        assert v['accuracy'] == v['correct'] / v['rows']
        assert v['rows'] == v['active_rows'] + v['retention_rows']
        assert v['correct'] == v['active_correct'] + v['retention_correct']
        assert 0 <= v['active_correct'] <= v['active_rows']
        assert 0 <= v['retention_correct'] <= v['retention_rows']
    if 'same_visible_food_occupancy_errors' in value:
        assert 0 <= value['same_visible_food_occupancy_errors'] <= rows - value['correct']


def verify(directory):
    report = read(directory / 'run/report.json')
    plan = report['plan']
    command = read(directory / 'command.json')
    assert report['complete'] and read(directory / 'status.json')['exit_code'] == 0
    assert plan == read(directory / 'run/plan.json')
    assert report['results'] == read(directory / 'run/progress.json')
    assert digest(directory / 'probe') == command['binary_sha256']
    assert command['environment'] == {'RAYON_NUM_THREADS': '4'}
    assert Path(command['command'][0]).name == 'probe'
    assert Path(command['command'][command['command'].index('--output') + 1]).name == 'run'
    for name, expected in plan['source_sha256'].items():
        assert Path(name).name == name
        assert digest(directory / 'run' / name) == expected, name
    assert (plan['steps'], plan['batch'], plan['learning_rate']) == (1024, 128, .005)
    assert plan['initialization_seeds'] == [1435400301, 1435400302, 1435400303]
    assert plan['fresh_development_seeds'] == [1435400201, 1435400202]
    assert plan['reserved_confirmation_seeds'] == [1434999901, 1434999902]
    used = [plan['train_seed'], plan['old_development_seed'], plan['sampling_seed'],
            *plan['initialization_seeds'], *plan['fresh_development_seeds']]
    assert len(used) == len(set(used))
    inherited = plan['inherited_seed_ledger']
    assert inherited['artifact_sha256'] == plan['parent_sha256']
    exposed = set().union(*map(set, inherited['ledger'].values()))
    assert len(exposed) == 52
    assert not set(used) & (exposed | set(plan['reserved_confirmation_seeds']))
    assert len(inherited['legacy_audit']['artifacts']) == 16
    assert len(inherited['legacy_audit']['corpus_manifests']) == 18
    for name, count, active, seeds in [
        ('old_audit', 3880, 1453, [1433000101, 1433000202]),
        ('fresh_audit', 3840, 1456, [1435400201, 1435400202]),
    ]:
        audit = plan[name]
        assert audit['all_rows'] == 10240
        assert (audit['move_rows'], audit['active']) == (count, active)
        assert audit['non_target_rows_retained'] == 10240 - count
        assert audit['fixed_effort_masks_legal'] and audit['private_bytes_exactly_recovered']
        assert sum(g['move_rows'] for g in audit['groups']) == count
        assert sum(g['active'] for g in audit['groups']) == active
        assert len(audit['datasets']) == 5
        for dataset in audit['datasets']:
            assert dataset['samples'] == 2048 and dataset['seeds'] == seeds
            assert dataset['collection']['behavior_clone_metadata_sha256'] == plan['parent_sha256']
    for old, fresh in zip(plan['old_audit']['datasets'], plan['fresh_audit']['datasets']):
        for field in ['layout', 'source_config_sha256', 'effective_config_sha256',
                      'semantic_ruleset_hash', 'compiled_ruleset_hash']:
            assert old[field] == fresh[field]
    objectives = plan.get('objectives', ['CrossEntropy'])
    expected = set(itertools.product(plan['initialization_seeds'], ['control', 'treatment'], objectives))
    key = lambda r: (r['initialization_seed'], r['arm'], r.get('objective', 'CrossEntropy'))
    assert {key(r) for r in report['results']} == expected
    assert len(report['results']) == len(expected)
    gates = {}
    for result in report['results']:
        assert result['parameters'] == 5105
        for field, count, active in [('initial_old', 1959, 720),
                                     ('old_development', 1959, 720),
                                     ('fresh_development', 3840, 1456)]:
            evaluation(result[field], count, active)
        assert [v['step'] for v in result['loss_checkpoints']] == [0, 256, 512, 768, 1023]
        assert all(math.isfinite(v.get('objective_value', v.get('cross_entropy')))
                   for v in result['loss_checkpoints'])
        record = result.get('record')
        assert bool(record) == plan.get('save_full_precision_diagnostic_records', False)
        if record:
            assert Path(record['file']).name == record['file']
            path = directory / 'run' / record['file']
            assert digest(path) == record['sha256'] and path.stat().st_size == record['bytes']
            assert record['precision'] == 'f32'
            assert record['reload_evaluations_identical'] and record['reload_parameter_bits_identical']
        objective = key(result)[2]
        gates[objective] = gates.get(objective, 0) + int(
            result['old_development']['gate_pass'] and result['fresh_development']['gate_pass'])
    return report, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--unchanged-from', type=Path)
    args = parser.parse_args()
    report, gates = verify(args.directory)
    common = 0
    if args.unchanged_from:
        previous, _ = verify(args.unchanged_from)
        key = lambda r: (r['initialization_seed'], r['arm'], r.get('objective', 'CrossEntropy'))
        current = {key(r): r for r in report['results']}
        for old in previous['results']:
            if key(old) not in current:
                continue
            common += 1
            new = current[key(old)]
            for field in ['initial_old', 'old_development', 'fresh_development']:
                assert {k: new[field][k] for k in old[field]} == old[field], (key(old), field)
            normalize = lambda checkpoints: [
                (v['step'], v.get('objective_value', v.get('cross_entropy'))) for v in checkpoints]
            assert normalize(new['loss_checkpoints']) == normalize(old['loss_checkpoints']), key(old)
        assert common > 0
    print(json.dumps({'verified_models': len(report['results']), 'both_cohorts_passed': gates,
                      'unchanged_common_fits': common,
                      'scope': 'Archived hashes and reported arithmetic/equality; does not rerun inference or certify ecological performance.'}, indent=2))


if __name__ == '__main__':
    main()
