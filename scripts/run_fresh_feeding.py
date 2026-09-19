#!/usr/bin/env python3
"""Run the fixed fresh-development feeding cohort on seven frozen Minds.

Writes all three immutable plans before evaluation, then runs one process at a
time. Failed biological gates are reported without skipping later candidates.
Infrastructure failures stop the run and retain all available evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from verify_feeding_ecology import summarize, validate_cohort

SEEDS = [1435500201, 1435500202]
INITIALIZATIONS = [1435400301, 1435400302, 1435400303]
SUFFIXES = ['v1', 'init1435400302', 'init1435400303']
RESERVED = [1434999901, 1434999902]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def prepare(repo, output):
    assert not output.exists(), 'Use a new immutable output directory'
    previous = [repo / f'training-output/sustained-feeding-2026-09-12-{s}' for s in SUFFIXES]
    old_plans = [read(p / 'plan.json') for p in previous]
    manifests = {}
    exposed = {1435400201, 1435400202}
    reserved = set(RESERVED)
    for plan in old_plans:
        for arm in plan['arms'].values():
            path = Path(arm['path']) / 'export.json'
            assert sha(path) == arm['manifest_sha256']
            assert sha(path.with_name('weights.bin')) == arm['weights_sha256']
            manifest = read(path)
            manifests[str(path)] = {'path': str(path), 'sha256': sha(path)}
            ledger = manifest.get('source_seed_ledger')
            if ledger:
                exposed.update(ledger['training'])
                exposed.update(ledger['validation'])
                reserved.update(ledger['confirmation'])
            exposed.update(manifest.get('development_seeds', []))
    assert not set(SEEDS) & (exposed | reserved)
    # Search historical local records before writing any new experiment record.
    cmd = ['rg', '--no-ignore', '-l', '-g', '*.json', '-g', '*.toml', '-g', '*.md',
           '|'.join(str(seed) for seed in SEEDS), 'training-output', 'docs']
    search = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    assert search.returncode == 1 and not search.stdout and not search.stderr, search
    output.mkdir(parents=True)
    audit = {'seeds': SEEDS, 'reserved_confirmation': RESERVED,
             'additional_exposed_seeds': [1435400201, 1435400202],
             'exposed_seeds': sorted(exposed), 'all_reserved_seeds': sorted(reserved),
             'manifests': list(manifests.values()),
             'prior_experiment_search': {'command': cmd, 'exit_code': search.returncode,
                                         'stdout': search.stdout, 'stderr': search.stderr},
             'scope': 'Disjoint from all six composite ledgers, parent development seeds and searched local records. Not a global registry of undeclared external work.'}
    write(output / 'seed-audit.json', audit)
    paths = []
    for initialization, old_root, old_plan in zip(INITIALIZATIONS, previous, old_plans):
        dest = output / 'matrices' / str(initialization)
        dest.mkdir(parents=True)
        (dest / 'trials').mkdir()
        shutil.copy2(old_root / 'evaluator', dest / 'evaluator')
        assert sha(dest / 'evaluator') == old_plan['evaluator_sha256']
        shutil.copytree(old_root / 'sources', dest / 'sources')
        shutil.copy2(Path(__file__), dest / 'sources/runner.py')
        shutil.copy2(repo / 'scripts/verify_feeding_ecology.py', dest / 'sources/verifier.py')
        plan = dict(old_plan)
        for key in ['canonical_prefix_trials', 'parent_trial_reuse']:
            plan.pop(key, None)
        plan.update({'cohort': 'fresh_development', 'seeds': SEEDS,
                     'scope': 'Prospective frozen-policy fresh-development feeding cohort. No training or confirmation.',
                     'boundary_verification': 'recorded_boundary_only',
                     'seed_audit': {'path': '../../seed-audit.json', 'sha256': sha(output / 'seed-audit.json')},
                     'source_sha256': {str(p.relative_to(dest)): sha(p) for p in (dest / 'sources').rglob('*') if p.is_file()},
                     'gate_note': 'Unchanged comparison criteria on ten layout/seed trials per arm. Safety aborts fail the gate; no cap adjustment within this cohort.'})
        if paths:
            plan['parent_trial_reuse'] = {
                f'{layout}-{seed}-parent': {
                    'source': str(paths[0] / 'trials' / f'{layout}-{seed}-parent'),
                    'source_plan_sha256': sha(paths[0] / 'plan.json')}
                for layout in plan['layouts'] for seed in SEEDS}
        write(dest / 'plan.json', plan)
        validate_cohort(dest, plan)
        paths.append(dest)
    write(output / 'cohort-plan.json', {
        'seeds': SEEDS, 'initializations': INITIALIZATIONS,
        'unique_trials': 70, 'unique_episodes': 140, 'process_concurrency': 1,
        'plans': {str(p.relative_to(output)): sha(p / 'plan.json') for p in paths},
        'policy_selection': 'All archived pairs, predeclared order; no selection from cohort outcomes.'})
    return paths


def run(repo, output, paths):
    start = time.monotonic()
    for dest in paths:
        plan = read(dest / 'plan.json')
        results = []
        begin_matrix = time.monotonic()
        for layout in plan['layouts']:
            for seed in plan['seeds']:
                for arm, identity in plan['arms'].items():
                    name = f'{layout}-{seed}-{arm}'
                    target = dest / 'trials' / name
                    write(output / 'current.json', {'initialization': plan['initialization_seed'],
                          'trial': name, 'completed_matrix_trials': len(results),
                          'elapsed_seconds': time.monotonic() - start})
                    begin = time.monotonic()
                    if name in plan.get('parent_trial_reuse', {}):
                        source = Path(plan['parent_trial_reuse'][name]['source'])
                        shutil.copytree(source, target)
                        assert sha(source / 'report.json') == sha(target / 'report.json')
                        status = {'trial': name, 'exit_code': 0, 'reused_from': str(source),
                                  'report_sha256': sha(target / 'report.json')}
                    else:
                        cmd = [str(dest / 'evaluator'), '--config', plan['config'],
                               '--export', identity['path'], '--export-manifest-sha256', identity['manifest_sha256'],
                               '--layout', layout, '--seed', str(seed),
                               '--assessment-mode', 'sustained-feeding', '--output', str(target)]
                        write(dest / f'{name}-command.json', cmd)
                        try:
                            with (dest / f'{name}.log').open('w') as log:
                                result = subprocess.run(cmd, cwd=repo, env={**os.environ, **plan['environment']},
                                                        stdout=log, stderr=subprocess.STDOUT,
                                                        timeout=plan['timeout_seconds_per_trial'])
                            status = {'trial': name, 'exit_code': result.returncode}
                        except subprocess.TimeoutExpired:
                            status = {'trial': name, 'timeout': True}
                    status['elapsed_seconds'] = time.monotonic() - begin
                    results.append(status)
                    write(dest / 'progress.json', results)
                    if status.get('exit_code') != 0:
                        raise RuntimeError(status)
                    print(plan['initialization_seed'], name, round(status['elapsed_seconds'], 1), flush=True)
        write(dest / 'status.json', {'complete': True, 'completed_trials': len(results),
              'reused_parent_trials': len(plan.get('parent_trial_reuse', {})),
              'elapsed_seconds': time.monotonic() - begin_matrix})
        summary = summarize(dest)
        write(dest / 'summary.json', summary)
        print('Gate:', summary['gate'], flush=True)
    write(output / 'status.json', {'complete': True, 'elapsed_seconds': time.monotonic() - start,
          'all_initializations_pass': all(read(p / 'summary.json')['gate']['passed'] for p in paths)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    run(repo, output, prepare(repo, output))
