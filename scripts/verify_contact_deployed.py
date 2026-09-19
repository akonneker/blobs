#!/usr/bin/env python3
"""Recompute canonical combat metrics, pairing and gates from episode evidence."""
import argparse
import itertools
import json
from pathlib import Path

from verify_feeding_ecology import read, sha
from verify_fresh_feeding import audit


def verify(root):
    plan = read(root / 'plan.json')
    assert plan['assessment_mode'] == 'canonical_match'
    assert plan['seeds'] == [1435500201, 1435500202]
    assert plan['reserved_confirmation'] == [1434999901, 1434999902]
    assert not set(plan['seeds']) & set(plan['reserved_confirmation'])
    assert plan['config_sha256'] == '4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9'
    for name, expected in {**plan['source_sha256'], **plan['executable_sha256']}.items():
        assert sha(root / name) == expected
    reference = plan['feeding_reference']
    feeding = Path(reference['path'])
    assert sha(feeding / 'summary.json') == reference['summary_sha256']
    assert sha(feeding / 'evaluation-seed-ledger.json') == reference['seed_ledger_sha256']
    feeding_summary = audit(feeding)
    assert feeding_summary['all_initializations_pass']
    arms = {'parent'} | {f'{i}-{a}' for i in [1435400301, 1435400302, 1435400303] for a in ['control', 'treatment']}
    assert set(plan['arms']) == arms
    signatures = {}
    results = {}
    hashes = {}
    for arm, identity in plan['arms'].items():
        parent = arm == 'parent'
        initialization, role = ('1435400301', 'parent') if parent else arm.split('-')
        old_plan = read(feeding / f'matrices/{initialization}/plan.json')
        assert identity == old_plan['arms'][role]
        manifest_path = Path(identity['path']) / 'export.json'
        assert sha(manifest_path) == identity['manifest_sha256']
        manifest = read(manifest_path)
        assert sha(manifest_path.with_name('weights.bin')) == identity['weights_sha256']
        path = root / 'trials' / arm / 'report.json'
        hashes[str(path.relative_to(root))] = sha(path)
        result = read(path)
        assert result['complete']
        metadata = result['plan']
        assert metadata == read(path.with_name('plan.json'))
        assert metadata['assessment_mode'] == 'canonical_match'
        assert metadata['seeds'] == plan['seeds']
        assert metadata['weights_sha256'] == identity['weights_sha256']
        assert metadata['export_manifest_sha256'] == identity['manifest_sha256']
        assert metadata['execution_contract'] == manifest['execution_contract']
        assert metadata['source_config_sha256'] == plan['config_sha256']
        config = metadata['config']['combat_curriculum']
        assert config['contact_initial_energies'] == [60, 100, 180]
        assert config['contact_opponents'] == ['aggressive', 'defensive']
        assert config['contact_cells_per_team'] == 1 and config['skirmish_cells_per_team'] == 4
        assert config['min_contact_kills_for_promotion'] == 0 and config['min_skirmish_kills_for_promotion'] == 1
        evaluation = result['evaluation']
        assert evaluation['schema_version'] == 1
        episodes = evaluation['episodes']
        assert len(episodes) == plan['expected_episodes_per_arm'] == 24
        grid = set(itertools.product(['contact', 'skirmish'], [60, 100, 180], ['aggressive', 'defensive'], plan['seeds']))
        seen = set()
        for e in episodes:
            key = e['stage'], e['initial_energy'], e['opponent'], e['seed']
            assert key in grid and key not in seen
            seen.add(key)
            signature = [e[k] for k in ['initial_state_hash', 'first_frontier_sha256', 'effective_env_sha256', 'ruleset_hash']]
            assert signatures.setdefault(key, signature) == signature
            assert e['horizon_quanta'] == (32768 if e['stage'] == 'contact' else 65536)
            assert e['cells_per_team'] == (1 if e['stage'] == 'contact' else 4)
            assert e['max_episode_steps'] == max(256, metadata['config']['env']['max_episode_len'])
            assert e['steps'] <= e['max_episode_steps']
            assert e['decisions']['total'] == sum(e['decisions']['action_kinds'].values())
            assert 0 <= e['decisions']['illegal'] <= e['decisions']['total']
            for family in ['attack', 'movement', 'consume']:
                counts = e[family]
                assert all(v >= 0 for v in counts.values())
                assert counts['committed'] == counts['accepted'] + counts['rejected_at_commit']
                assert counts['completed'] == sum(counts[k] for k in ['succeeded', 'rejected', 'frustrated', 'contested', 'interrupted'])
            outcome = e['outcome']
            if outcome == 'Some(Win)':
                assert e['opponent_cells'] == 0 and e['training_cells'] > 0 and e['end_reason'] == 'Some(Extermination)'
            elif outcome == 'Some(Loss)':
                assert e['training_cells'] == 0 and e['end_reason'] == 'Some(Extermination)'
            elif outcome == 'Some(Timeout)':
                assert e['elapsed_quanta'] >= e['horizon_quanta'] and e['end_reason'] == 'Some(SimTimeDeadline)'
            else:
                assert outcome == 'Some(SafetyAbort)' and e['steps'] == e['max_episode_steps']
                assert e['end_reason'] == 'Some(DecisionFrontierSafetyLimit)'
        assert seen == grid
        report = evaluation['report']
        assert report['schema_version'] == 3 and report['seeds'] == plan['seeds']
        assert report['contact_sim_time_limit_quanta'] == 32768 and report['skirmish_sim_time_limit_quanta'] == 65536
        assert all(e['ruleset_hash'] == report['ruleset_hash'] for e in episodes)
        variants = set()
        totals = dict.fromkeys(['episodes', 'attacking_episodes', 'damaging_episodes', 'attacks_committed', 'attacks_succeeded', 'damage_dealt', 'kills'], 0)
        for v in report['variants']:
            key = v['stage'], v['initial_energy'], v['opponent']
            assert key not in variants
            variants.add(key)
            selected = [e for e in episodes if (e['stage'], e['initial_energy'], e['opponent']) == key]
            assert len(selected) == 2 and v['cells_per_team'] == selected[0]['cells_per_team']
            raw = {'episodes': 2, 'attacking_episodes': sum(e['attack']['committed'] > 0 for e in selected),
                   'damaging_episodes': sum(e['damage_dealt'] > 0 for e in selected),
                   'attacks_committed': sum(e['attack']['committed'] for e in selected),
                   'attacks_succeeded': sum(e['attack']['succeeded'] for e in selected),
                   'damage_dealt': sum(e['damage_dealt'] for e in selected), 'kills': sum(e['kills'] for e in selected)}
            for name, value in raw.items():
                assert v[name] == value
                totals[name] += value
            for name, outcome in [('wins', 'Win'), ('losses', 'Loss'), ('timeouts', 'Timeout'), ('safety_aborts', 'SafetyAbort')]:
                assert v[name] == sum(e['outcome'] == f'Some({outcome})' for e in selected)
            for name in ['frustrated', 'interrupted']:
                assert v[f'attacks_{name}'] == sum(e['attack'][name] for e in selected)
            check_rates(v)
        assert len(variants) == 12
        assert all(report[k] == v for k, v in totals.items())
        check_rates(report)
        stages = {stage: [e for e in episodes if e['stage'] == stage] for stage in ['contact', 'skirmish']}
        active = all(e['outcome'] != 'Some(SafetyAbort)' for e in episodes) and all(
            sum(e['attack']['succeeded'] for e in group) > 0 and sum(e['damage_dealt'] for e in group) > 0 for group in stages.values())
        passed = active and sum(e['kills'] for e in stages['skirmish']) >= 1
        assert evaluation['combat_thresholds_passed'] == passed
        results[arm] = {**totals, 'combat_thresholds_passed': passed,
                        'illegal_decisions': sum(e['decisions']['illegal'] for e in episodes),
                        'safety_aborts': sum(e['outcome'] == 'Some(SafetyAbort)' for e in episodes),
                        'wins': sum(e['outcome'] == 'Some(Win)' for e in episodes),
                        'losses': sum(e['outcome'] == 'Some(Loss)' for e in episodes),
                        'action_kinds': {kind: sum(e['decisions']['action_kinds'].get(kind, 0) for e in episodes)
                                         for kind in sorted(set().union(*(e['decisions']['action_kinds'] for e in episodes)))}}
    for s in plan['feeding_sentinels']:
        name = f"{s['layout']}-{s['seed']}-{s['arm']}"
        current = root / 'feeding-sentinels' / name / 'report.json'
        old_arm = 'parent' if s['arm'] == 'parent' else 'treatment'
        old = feeding / f"matrices/1435400301/trials/{s['layout']}-{s['seed']}-{old_arm}/report.json"
        assert read(current)['trial'] == read(old)['trial']
        hashes[str(current.relative_to(root))] = sha(current)
    assert read(root / 'status.json')['complete']
    progress = read(root / 'progress.json')
    assert len(progress) == 7 and {p['arm'] for p in progress} == arms and all(p['exit_code'] == 0 for p in progress)
    gate = {'all_treatments_pass_existing_combat_thresholds': all(v['combat_thresholds_passed'] for k,v in results.items() if k.endswith('-treatment')),
            'no_illegal_decisions_or_safety_aborts': all(v['illegal_decisions'] == v['safety_aborts'] == 0 for v in results.values()),
            'same_frozen_feeding_candidates': True}
    return {'episodes': 168, 'feeding_sentinel_episodes': 4, 'arms': results,
            'gate': {'passed': all(gate.values()), 'checks': gate}, 'report_sha256': hashes,
            'scope': 'Canonical combat on exposed development seeds; existing activity/kill thresholds, not general combat mastery or final confirmation.'}


def check_rates(values):
    for field, numerator, denominator in [('attacking_episode_rate', 'attacking_episodes', 'episodes'),
                                           ('damaging_episode_rate', 'damaging_episodes', 'episodes'),
                                           ('attack_success_rate', 'attacks_succeeded', 'attacks_committed')]:
        expected = values[numerator] / values[denominator] if values[denominator] else 0
        assert abs(values[field] - expected) <= 1e-12


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
