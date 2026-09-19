#!/usr/bin/env python3
"""Verify retained composite envelopes, provenance and matched WASM evidence."""
import argparse
import hashlib
import itertools
import json
import math
import struct
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    assert read(root / 'export-status.json')['exit_code'] == 0
    command = read(root / 'export-command.json')
    assert sha((root / 'exporter').read_bytes()) == command['binary_sha256']
    paths = sorted((root / 'exports').glob('*/export.json'))
    assert len(paths) == 6
    seen = set()
    rows = 0
    reports = {}
    for path in paths:
        manifest_bytes = path.read_bytes()
        manifest = json.loads(manifest_bytes)
        key = manifest['initialization_seed'], manifest['arm']
        assert key not in seen
        seen.add(key)
        assert manifest['schema_version'] == manifest['weight_format'] == 1
        assert manifest['execution_contract'] == 'blob.policy.move-utility-scalar-f32-libm-q48-signal-reserved.v1'
        weights = path.with_name('weights.bin').read_bytes()
        assert len(weights) == manifest['weight_bytes'] <= 16 * 1024 * 1024
        assert sha(weights) == manifest['weights_sha256'] and weights[:8] == b'BLCMP001'
        parent_length, = struct.unpack('<I', weights[8:12])
        assert len(weights) == 12 + parent_length + 5105 * 4
        assert weights[12:20] == b'BLPOL001'
        assert sha(weights[12:12 + parent_length]) == manifest['parent_weights_sha256']
        assert all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', weights[12 + parent_length:]))
        assert manifest['parent_parameters_bit_identical']
        for name, digest in manifest['export_source_sha256'].items():
            assert Path(name).name == name
            assert sha((root / 'exports/sources' / name).read_bytes()) == digest
        ledger = manifest['source_seed_ledger']
        training, validation, confirmation = (set(ledger[k]) for k in ['training', 'validation', 'confirmation'])
        assert not training & validation and not training & confirmation and not validation & confirmation
        assert confirmation == {1434999901, 1434999902}
        assert {1433000101, 1435400101, 1435400301, 1435400302, 1435400303} <= training
        assert {1433000202, 1435400201, 1435400202, 72101, 72102, 72103, 72104,
                1435000101, 1435000202, 1435000303,
                1435100101, 1435100202, 1435100301, 1435100302, 1435100303,
                1435200101, 1435200202, 1435200301, 1435200302, 1435200303} <= validation
        for field, count in [('old_fidelity', 1959), ('fresh_fidelity', 3840)]:
            fidelity = manifest[field]
            assert fidelity['rows'] == count and fidelity['sampled_choice_mismatches'] == 0
            assert 0 <= fidelity['max_legal_logit_absolute_difference'] < .0001
            evaluation = fidelity['burn_evaluation']
            assert evaluation['gate_pass'] and evaluation['rows'] == count
            for value in ['correct', 'active_correct', 'retention_correct']:
                assert fidelity[value] == evaluation[value]
            assert evaluation['correct'] / count >= .95
            assert evaluation['active_correct'] / evaluation['active_rows'] >= .90
            assert evaluation['retention_correct'] / evaluation['retention_rows'] >= .95
            rows += count
        if key[0] != 1435400301:
            continue
        arm = key[1]
        for stage in ['build', 'verification']:
            assert read(root / f'{arm}-{stage}-status.json')['exit_code'] == 0
        build = read(path.with_name('mind-build.json'))
        wasm = path.with_name('mind.wasm').read_bytes()
        assert build['export_manifest_sha256'] == sha(manifest_bytes)
        assert build['weights_sha256'] == sha(weights)
        assert build['wasm_sha256'] == sha(wasm) and build['wasm_bytes'] == len(wasm)
        verification = read(root / f'{arm}-verification/verification.json')
        assert verification['weights_sha256'] == sha(weights)
        assert verification['wasm_sha256'] == sha(wasm)
        assert verification['execution_contract'] == manifest['execution_contract']
        assert verification['native_wasm_decision_and_memory_mismatches'] == 0
        assert verification['abi_fixture_decisions'] == 528
        assert verification['decisions'] > 528 and verification['worker_counts'] == [1, 4]
        assert verification['stock_extism_checked_decisions'] == verification['isolation_checks'] == 32
        assert verification['malformed_input_rejected']
        retention = verification['inherited_retention']
        assert retention['other_output_mismatches'] == 0
        assert retention['move_decisions'] + retention['non_move_decisions'] == verification['decisions']
        assert 0 < retention['changed_move_targets'] <= retention['move_decisions']
        assert retention['non_move_decisions'] > 0
        assert len(verification['cases']) == 10
        assert {(case['layout'], case['seed']) for case in verification['cases']} == set(
            itertools.product(['line', 'checkerboard', 'ring', 'loose-random', 'random'], [72101, 72102]))
        verify_command = read(root / f'{arm}-verification-command.json')
        assert sha((root / 'verifier').read_bytes()) == verify_command['binary_sha256']
        reports[arm] = {'decisions': verification['decisions'], 'retention': retention}
    assert seen == set(itertools.product([1435400301, 1435400302, 1435400303], ['control', 'treatment']))
    return {'verified_exports': len(seen), 'unchanged_recorded_choices': rows,
            'wasm_evidence': reports, 'scope': 'Archived hashes and evidence consistency; no inference rerun or ecological qualification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
