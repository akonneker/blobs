#!/usr/bin/env python3
"""Freeze and run the seven-Mind canonical combat development comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from verify_feeding_ecology import read, sha
from verify_fresh_feeding import audit


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run(repo, output):
    assert not output.exists(), 'Use a new immutable evidence directory'
    feeding = repo / 'training-output/fresh-feeding-2026-09-12-v1'
    retained = audit(feeding)
    assert retained['all_initializations_pass']
    arms = {}
    for initialization in [1435400301, 1435400302, 1435400303]:
        plan = read(feeding / f'matrices/{initialization}/plan.json')
        for arm, identity in plan['arms'].items():
            name = 'parent' if arm == 'parent' else f'{initialization}-{arm}'
            arms[name] = identity
    output.mkdir(parents=True)
    (output / 'trials').mkdir()
    (output / 'sources').mkdir()
    for name in ['contact_deployed_evaluation', 'feeding_deployed_evaluation']:
        shutil.copy2(repo / 'target/release' / name, output / name)
    sources = ['blob_rl/src/contact_deployed.rs', 'blob_rl/src/contact_evaluation.rs',
               'blob_rl/src/deployed_artifact.rs', 'blob_rl/src/bin/contact_deployed_evaluation.rs',
               'blob_rl/src/bin/feeding_deployed_evaluation.rs', 'blob_rl/src/env.rs',
               'blob_rl/src/feeding_deployed.rs', 'blob_rl/src/config.rs', 'blob_rl/src/telemetry.rs',
               'blob_policy/src/composite.rs', 'blob_policy/src/runtime.rs', 'blob_policy/src/sampling.rs',
               'scripts/run_contact_deployed.py', 'scripts/verify_contact_deployed.py', 'Cargo.lock']
    for name in sources:
        target = output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, target)
    config = repo / 'blob_rl/config/combat_warm_start_256.toml'
    plan = {'scope': 'Frozen canonical combat on already exposed development seeds, not final confirmation.',
            'assessment_mode': 'canonical_match', 'arms': arms, 'seeds': [1435500201, 1435500202],
            'config': str(config), 'config_sha256': sha(config),
            'feeding_reference': {'path': str(feeding), 'summary_sha256': sha(feeding / 'summary.json'),
                                  'seed_ledger_sha256': sha(feeding / 'evaluation-seed-ledger.json')},
            'reserved_confirmation': [1434999901, 1434999902],
            'executable_sha256': {n: sha(output / n) for n in ['contact_deployed_evaluation', 'feeding_deployed_evaluation']},
            'source_sha256': {str(p.relative_to(output)): sha(p) for p in (output / 'sources').rglob('*') if p.is_file()},
            'process_concurrency': 1, 'environment': {'RAYON_NUM_THREADS': '4'}, 'timeout_seconds': 1200,
            'expected_episodes_per_arm': 24,
            'gate': {'all_treatments_pass_existing_combat_thresholds': True,
                     'no_illegal_decisions_or_safety_aborts': True,
                     'same_frozen_feeding_candidates': True},
            'feeding_sentinels': [
                {'arm': 'parent', 'layout': 'line', 'seed': 1435500201},
                {'arm': '1435400301-treatment', 'layout': 'checkerboard', 'seed': 1435500201}]}
    # Policy source must remain exactly the already-qualified portable runtime.
    old = read(feeding / 'matrices/1435400301/plan.json')
    for name in ['composite.rs', 'runtime.rs', 'sampling.rs']:
        key = f'sources/blob_policy/src/{name}'
        assert plan['source_sha256'][key] == old['source_sha256'][key]
    for identity in arms.values():
        assert sha(Path(identity['path']) / 'export.json') == identity['manifest_sha256']
        assert sha(Path(identity['path']) / 'weights.bin') == identity['weights_sha256']
    write(output / 'plan.json', plan)
    progress = []
    started = time.monotonic()
    for name, identity in arms.items():
        cmd = [str(output / 'contact_deployed_evaluation'), '--config', str(config),
               '--export', identity['path'], '--export-manifest-sha256', identity['manifest_sha256'],
               '--seeds', ','.join(map(str, plan['seeds'])), '--output', str(output / 'trials' / name)]
        write(output / f'{name}-command.json', cmd)
        write(output / 'current.json', {'arm': name, 'completed_arms': len(progress)})
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=repo, env={**os.environ, **plan['environment']}, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=plan['timeout_seconds'])
        progress.append({'arm': name, 'exit_code': result.returncode})
        write(output / 'progress.json', progress)
        assert result.returncode == 0, name
        print(name, read(output / 'trials' / name / 'report.json')['evaluation']['combat_thresholds_passed'], flush=True)
    (output / 'feeding-sentinels').mkdir()
    for sentinel in plan['feeding_sentinels']:
        name = f"{sentinel['layout']}-{sentinel['seed']}-{sentinel['arm']}"
        identity = arms[sentinel['arm']]
        cmd = [str(output / 'feeding_deployed_evaluation'), '--config', str(config), '--export', identity['path'],
               '--export-manifest-sha256', identity['manifest_sha256'], '--layout', sentinel['layout'],
               '--seed', str(sentinel['seed']), '--assessment-mode', 'sustained-feeding',
               '--output', str(output / 'feeding-sentinels' / name)]
        write(output / f'{name}-command.json', cmd)
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=repo, env={**os.environ, **plan['environment']}, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=plan['timeout_seconds'])
        assert result.returncode == 0, name
    write(output / 'status.json', {'complete': True, 'elapsed_seconds': time.monotonic() - started})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.output.resolve())
