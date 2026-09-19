#!/usr/bin/env python3
"""Audit all predeclared fresh-feeding matrices without treating reused parents as new trials."""
import argparse
from collections import Counter
import json
from pathlib import Path

from verify_feeding_ecology import read, sha, summarize


def audit(root):
    cohort = read(root / 'cohort-plan.json')
    assert cohort['initializations'] == [1435400301, 1435400302, 1435400303]
    assert cohort['process_concurrency'] == 1
    assert (cohort['unique_trials'], cohort['unique_episodes']) == (70, 140)
    assert set(cohort['plans']) == {f'matrices/{seed}' for seed in cohort['initializations']}
    matrices = {}
    unique = []
    report_hashes = {}
    signatures = {}
    for initialization in cohort['initializations']:
        relative = f'matrices/{initialization}'
        path = root / relative
        assert sha(path / 'plan.json') == cohort['plans'][relative]
        plan = read(path / 'plan.json')
        assert plan['initialization_seed'] == initialization
        assert plan['seeds'] == cohort['seeds']
        assert plan['gate']['maximum_on_food_survival_regression_vs_each_comparator'] == .01
        summary = summarize(path)
        matrices[str(initialization)] = summary
        for report_path in sorted((path / 'trials').glob('*/report.json')):
            name = report_path.parent.name
            report_hashes[str(report_path.relative_to(root))] = sha(report_path)
            report = read(report_path)
            trial = report['trial']
            arm = name.rsplit('-', 1)[1]
            for episode in trial['episodes']:
                key = trial['layout'], trial['seed'], episode['stage']
                signature = [episode[k] for k in ['initial_state_hash', 'first_frontier_sha256',
                                                   'effective_env_sha256', 'ruleset_hash']]
                assert signatures.setdefault(key, signature) == signature
                if initialization != cohort['initializations'][0] and arm == 'parent':
                    continue
                prefix = episode['opponent_extinction']
                unique.append({'initialization': initialization, 'trial': name,
                               'stage': episode['stage'], 'arm': arm,
                               'elapsed_quanta': episode['elapsed_quanta'], 'steps': episode['steps'],
                               'outcome': episode['outcome'], 'end_reason': episode['end_reason'],
                               'surviving_cells': episode['metrics']['surviving_cells'],
                               'illegal_decisions': episode['decisions']['illegal_decisions'],
                               'post_extinction_survivor_change': None if prefix is None else
                                   episode['metrics']['surviving_cells'] - prefix['training_cells']})
    assert len(unique) == cohort['unique_episodes']
    status = read(root / 'status.json')
    assert status['complete']
    passed = all(m['gate']['passed'] for m in matrices.values())
    assert status['all_initializations_pass'] == passed
    seed_audit = read(root / 'seed-audit.json')
    ledger = {role: set() for role in ['training', 'validation', 'confirmation']}
    for reference in seed_audit['manifests']:
        manifest = read(Path(reference['path']))
        for role, seeds in (manifest.get('source_seed_ledger') or {}).items():
            ledger[role].update(seeds)
        ledger['validation'].update(manifest.get('development_seeds', []))
    ledger['validation'].update(seed_audit['additional_exposed_seeds'])
    ledger['validation'].update(cohort['seeds'])
    ledger['confirmation'].update(seed_audit['reserved_confirmation'])
    assert not ledger['training'] & ledger['validation']
    assert not (ledger['training'] | ledger['validation']) & ledger['confirmation']
    return {'unique_trials': 70, 'unique_episodes': len(unique), 'all_initializations_pass': passed,
            'deadline_reached': sum(e['end_reason'] == 'Some(SimTimeDeadline)' for e in unique),
            'outcomes': dict(Counter(e['outcome'] for e in unique)),
            'illegal_decisions': sum(e['illegal_decisions'] for e in unique),
            'post_extinction_survivor_change': sum(e['post_extinction_survivor_change'] or 0 for e in unique),
            'episodes_with_extinction_boundary': sum(e['post_extinction_survivor_change'] is not None for e in unique),
            'max_steps_episode': max(unique, key=lambda e: e['steps']),
            'post_evaluation_seed_ledger': {role: sorted(seeds) for role, seeds in ledger.items()},
            'matrices': matrices, 'report_sha256': report_hashes, 'unique_episode_inventory': unique,
            'scope': 'Fresh relative to audited local lineage and records before this cohort. These seeds are now exposed development evidence; not final confirmation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
