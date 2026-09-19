#!/usr/bin/env python3
"""Verify occupied-target diagnostic provenance and unchanged combat trajectories."""
import argparse
import json
from pathlib import Path
from verify_feeding_ecology import read, sha
from verify_contact_deployed import verify


def audit(root):
    plan = read(root / 'plan.json')
    source = Path(plan['combat_root'])
    assert sha(source / 'plan.json') == plan['combat_plan_sha256']
    original = verify(source)
    assert sha(root / 'evaluator') == plan['evaluator_sha256']
    for name, expected in plan['sources'].items():
        assert sha(root / 'sources' / name) == expected
    assert plan['arms'] == {**read(source / 'plan.json')['arms'], 'aggressive-baseline': None}
    parent = read(source / 'trials/parent/report.json')['evaluation']['episodes']
    results = {}
    for arm, identity in plan['arms'].items():
        path = root / f'{arm}.json'
        result = read(path)
        assert result['complete'] and result['diagnostic_contract'] == 'blob.contact.occupied-target-opportunity.v2'
        assert result['seeds'] == plan['seeds'] == [1435500201, 1435500202]
        assert result['source_config_sha256'] == plan['config_sha256']
        if identity:
            assert result['identity'] == {'manifest_sha256': identity['manifest_sha256'], 'weights_sha256': identity['weights_sha256']}
            assert result['evaluation'] == read(source / 'trials' / arm / 'report.json')['evaluation']
        else:
            assert result['identity'] == {'baseline': 'aggressive'}
        episodes = result['evaluation']['episodes']
        counts = result['opportunities']
        assert len(episodes) == len(counts) == 24
        for e, o, paired in zip(episodes, counts, parent):
            for field in ['stage', 'seed', 'initial_energy', 'opponent', 'cells_per_team',
                          'initial_state_hash', 'first_frontier_sha256', 'effective_env_sha256', 'ruleset_hash']:
                assert e[field] == paired[field]
            assert o['decisions'] == e['decisions']['total']
            assert o['attacks_selected'] == e['decisions']['action_kinds'].get('Attack', 0)
            assert 0 <= o['legal_occupied_attack_available'] <= o['legal_attack_available'] <= o['decisions']
            assert 0 <= o['moves_despite_legal_occupied_attack'] <= o['legal_occupied_attack_available']
            assert o['moves_despite_legal_occupied_attack'] <= o['moves_despite_legal_attack'] <= e['decisions']['action_kinds'].get('Move', 0)
            assert e['decisions']['illegal'] == 0 and e['outcome'] != 'Some(SafetyAbort)'
        report = result['evaluation']['report']
        for field in ['damage_dealt', 'kills']:
            assert report[field] == sum(e[field] for e in episodes)
        assert report['attacks_committed'] == sum(e['attack']['committed'] for e in episodes)
        active = all(sum(e['attack']['succeeded'] for e in episodes if e['stage'] == stage) > 0
                     and sum(e['damage_dealt'] for e in episodes if e['stage'] == stage) > 0
                     for stage in ['contact', 'skirmish'])
        passed = active and sum(e['kills'] for e in episodes if e['stage'] == 'skirmish') >= 1
        assert passed == result['evaluation']['combat_thresholds_passed']
        results[arm] = {'totals': {k: sum(o[k] for o in counts) for k in counts[0]},
                        'occupied_opportunity_episodes': sum(o['legal_occupied_attack_available'] > 0 for o in counts),
                        'combat_thresholds_passed': passed, 'damage_dealt': report['damage_dealt'], 'kills': report['kills'],
                        'report_sha256': sha(path)}
    assert read(root / 'status.json')['complete']
    return {'original_combat_gate': original['gate'], 'arms': results,
            'repeated_frozen_episodes': 168, 'baseline_episodes': 24,
            'scope': 'Legal observable-occupant opportunities, not team identity or tactical value. All frozen-policy evaluation records exactly match the original failed comparison.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
