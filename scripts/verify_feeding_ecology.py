#!/usr/bin/env python3
"""Audit and summarize the immutable three-arm deployed feeding matrix."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cohort(root, plan):
    """Fresh development must be disjoint from every frozen candidate ledger."""
    if plan.get('cohort', 'exposed_development') == 'exposed_development':
        assert plan['seeds'] == [1435400201, 1435400202]
        return False
    assert plan['cohort'] == 'fresh_development'
    seeds = plan['seeds']
    assert seeds and len(seeds) == len(set(seeds))
    assert all(type(seed) is int and 0 <= seed < 2**64 for seed in seeds)
    reference = plan['seed_audit']
    path = root / reference['path']
    assert sha(path) == reference['sha256']
    audit = read(path)
    assert audit['seeds'] == seeds
    assert audit['reserved_confirmation'] == plan['reserved_confirmation']
    assert audit['prior_experiment_search']['exit_code'] == 1
    assert audit['prior_experiment_search']['stdout'] == ''
    assert audit['prior_experiment_search']['stderr'] == ''
    manifests = {}
    exposed = set(audit['additional_exposed_seeds'])
    reserved = set(plan['reserved_confirmation'])
    composite_pairs = set()
    for reference in audit['manifests']:
        path = Path(reference['path'])
        assert sha(path) == reference['sha256']
        manifest = read(path)
        manifests[reference['sha256']] = manifest
        ledger = manifest.get('source_seed_ledger')
        if ledger is not None:
            exposed.update(ledger['training'])
            exposed.update(ledger['validation'])
            reserved.update(ledger['confirmation'])
            composite_pairs.add((manifest['initialization_seed'], manifest['arm']))
        exposed.update(manifest.get('development_seeds', []))
    assert composite_pairs == set(itertools.product(
        [1435400301, 1435400302, 1435400303], ['control', 'treatment']))
    assert len(manifests) == 7
    assert not set(seeds) & (exposed | reserved)
    assert audit['exposed_seeds'] == sorted(exposed)
    assert audit['all_reserved_seeds'] == sorted(reserved)
    assert plan['initialization_seed'] in [1435400301, 1435400302, 1435400303]
    for arm, identity in plan['arms'].items():
        assert identity['manifest_sha256'] in manifests
        manifest = manifests[identity['manifest_sha256']]
        assert manifest['weights_sha256'] == identity['weights_sha256']
        if arm in ['control', 'treatment']:
            assert manifest['initialization_seed'] == plan['initialization_seed']
            assert manifest['arm'] == arm
        else:
            assert arm == 'parent' and 'initialization_seed' not in manifest
    assert plan['assessment_mode'] == 'sustained_feeding'
    assert plan['boundary_verification'] == 'recorded_boundary_only'
    assert 'canonical_prefix_trials' not in plan
    return True


def summarize(root, partial=False):
    plan = read(root / 'plan.json')
    sustained = plan.get('assessment_mode', 'canonical_match') == 'sustained_feeding'
    assert plan['source_config_sha256'] == '4e134f3395f55ec97342281aa76e8da561bdb083a5ac4de417d15b25f92565f9'
    assert sha(root / 'evaluator') == plan['evaluator_sha256']
    for name, expected in plan['source_sha256'].items():
        assert sha(root / name) == expected, name
    fresh = validate_cohort(root, plan)
    assert plan['layouts'] == ['line', 'checkerboard', 'ring', 'loose-random', 'random']
    assert set(plan['arms']) == {'parent', 'control', 'treatment'}
    assert plan['stages'] == ['on_food', 'adjacent_food']
    assert plan['reserved_confirmation'] == [1434999901, 1434999902]
    assert (plan['horizon_quanta'], plan['cells_per_team'], plan['max_episode_steps']) == (262144, 256, 4096)
    assert not set(plan['seeds']) & set(plan['reserved_confirmation'])
    grid = set(itertools.product(plan['layouts'], plan['seeds'], plan['arms']))
    complete = set()
    aggregates = {}
    paired = {}
    rows = []
    progress = {p['trial']: p for p in read(root / 'progress.json')} if (root / 'progress.json').exists() else {}
    for layout, seed, arm in sorted(grid):
        name = f'{layout}-{seed}-{arm}'
        path = root / 'trials' / name / 'report.json'
        if not path.exists():
            continue
        report = read(path)
        if name in plan.get('parent_trial_reuse', {}):
            reused = plan['parent_trial_reuse'][name]
            source = Path(reused['source'])
            if 'report_sha256' in reused:
                assert sha(path) == reused['report_sha256']
                assert sha(source / 'report.json') == reused['report_sha256']
            else:
                assert fresh and arm == 'parent'
                assert sha(source.parent.parent / 'plan.json') == reused['source_plan_sha256']
                assert sha(path) == sha(source / 'report.json')
        assert report['complete']
        identity = report['plan']
        trial = report['trial']
        assert identity == read(path.with_name('plan.json'))
        assert identity['source_config_sha256'] == plan['source_config_sha256']
        assert identity['weights_sha256'] == plan['arms'][arm]['weights_sha256']
        assert identity['export_manifest_sha256'] == plan['arms'][arm]['manifest_sha256']
        assert identity['seed'] == seed == trial['seed']
        assert identity['layout'] == trial['layout'] == layout.replace('-', '_')
        assert trial['schema_version'] in [1, 2]
        if trial['schema_version'] == 2:
            assert trial['assessment_mode'] == identity['assessment_mode'] == plan.get('assessment_mode', 'canonical_match')
        else:
            assert not sustained
        assert trial['execution_contract'] == identity['execution_contract']
        cfg = identity['config']
        assert cfg['env']['world_size'] == 256 and cfg['env']['cells_per_team'] == 256
        gate = cfg['feeding_curriculum']['promotion']
        assert gate['evaluation_sim_time_limit_quanta'] == 262144
        assert gate['evaluation_max_episode_len'] == 4096
        assert gate['min_on_food_episode_success_rate'] == .95
        assert gate['min_adjacent_episode_success_rate'] == gate['min_survival_rate'] == .8
        assert gate['min_consumed_energy_per_initial_cell'] == 1
        assert len(trial['episodes']) == 2
        promotion = trial['promotion']
        assert promotion['schema_version'] == 3 and promotion['seeds'] == [seed]
        assert promotion['stages'] == [e['metrics'] for e in trial['episodes']]
        assert promotion['passed'] == all(c['passed'] for c in promotion['checks'])
        expected_checks = {}
        for stage, metrics in zip(['on_food', 'adjacent_food'], promotion['stages']):
            expected_checks[f'{stage}_episode_success_rate'] = (
                metrics['episode_success_rate'],
                gate['min_on_food_episode_success_rate' if stage == 'on_food' else 'min_adjacent_episode_success_rate'])
            expected_checks[f'{stage}_no_safety_aborts'] = (float(metrics['safety_aborts'] == 0), 1.0)
            expected_checks[f'{stage}_survival_rate'] = (metrics['survival_rate'], gate['min_survival_rate'])
            expected_checks[f'{stage}_consumed_energy_per_initial_cell'] = (
                metrics['consumed_energy_per_initial_cell'], gate['min_consumed_energy_per_initial_cell'])
        assert len(promotion['checks']) == len(expected_checks)
        assert {c['name'] for c in promotion['checks']} == set(expected_checks)
        for check in promotion['checks']:
            assert (check['observed'], check['minimum']) == expected_checks[check['name']]
            assert check['passed'] == (check['observed'] >= check['minimum'])
        for stage, episode in zip(['on_food', 'adjacent_food'], trial['episodes']):
            metrics = episode['metrics']
            assert episode['stage'] == metrics['stage'] == stage
            assert episode['seed'] == seed
            assert metrics['initial_cells'] == 256 and metrics['episodes'] == 1
            assert 0 <= metrics['surviving_cells'] <= 256
            assert metrics['survival_rate'] == metrics['surviving_cells'] / 256
            assert metrics['consumed_energy_per_initial_cell'] == metrics['consumed_energy'] / 256
            assert metrics['episode_success_rate'] == metrics['successful_episodes']
            assert metrics['safety_aborts'] in [0, 1]
            assert metrics['safety_aborts'] == int(episode['outcome'] == 'Some(SafetyAbort)')
            assert metrics['successful_episodes'] == int(
                not metrics['safety_aborts'] and episode['consume']['succeeded'] > 0
                and episode['consume']['consumed_energy'] > 0
                and (stage == 'on_food' or episode['movement']['succeeded'] > 0))
            assert episode['elapsed_quanta'] >= 262144 or episode['outcome'] not in ['Some(Draw)']
            if sustained:
                assert episode['outcome'] in ['Some(Timeout)', 'Some(Loss)', 'Some(SafetyAbort)']
                if episode['outcome'] == 'Some(Timeout)':
                    assert episode['elapsed_quanta'] >= 262144
                    assert episode['end_reason'] == 'Some(SimTimeDeadline)'
                if episode['outcome'] == 'Some(Loss)':
                    assert metrics['surviving_cells'] == 0
                    assert episode['end_reason'] == 'Some(Extermination)'
                if episode['outcome'] == 'Some(SafetyAbort)':
                    assert episode['steps'] == 4096
                    assert episode['end_reason'] == 'Some(DecisionFrontierSafetyLimit)'
                prefix = episode['opponent_extinction']
                if prefix is not None:
                    assert prefix['elapsed_quanta'] == 102400 <= episode['elapsed_quanta']
                    assert 0 < prefix['steps'] <= episode['steps']
                    assert 0 <= prefix['training_cells'] <= 256
                    for family in ['movement', 'consume', 'decisions']:
                        assert set(prefix[family]) == set(episode[family])
                        assert all(0 <= value <= episode[family][key]
                                   for key, value in prefix[family].items())
                else:
                    assert episode['elapsed_quanta'] < 102400
            if sustained and not fresh:
                old_reference = plan['canonical_prefix_trials'][name]
                old_path = Path(old_reference['path'])
                assert sha(old_path) == old_reference['sha256']
                old_report = read(old_path)
                assert old_report['plan']['weights_sha256'] == identity['weights_sha256']
                old = next(e for e in old_report['trial']['episodes'] if e['stage'] == stage)
                for field in ['initial_state_hash', 'first_frontier_sha256', 'effective_env_sha256', 'ruleset_hash']:
                    assert episode[field] == old[field], (name, field)
                prefix = episode['opponent_extinction']
                assert prefix is not None
                assert prefix['elapsed_quanta'] == old['elapsed_quanta'] == 102400
                assert prefix['steps'] == old['steps']
                assert prefix['state_hash'] == old['final_state_hash']
                assert prefix['training_cells'] == old['metrics']['surviving_cells']
                for field in ['movement', 'consume', 'decisions']:
                    assert prefix[field] == old[field], (name, field)
            assert episode['steps'] <= 4096
            for family in ['movement', 'consume']:
                counts = episode[family]
                assert counts['committed'] == counts['accepted'] + counts['rejected_at_commit']
                assert counts['completed'] == sum(counts[k] for k in ['succeeded', 'rejected', 'frustrated', 'contested', 'interrupted'])
            assert metrics['movement_successes'] == episode['movement']['succeeded']
            assert metrics['consume_successes'] == episode['consume']['succeeded']
            assert metrics['consumed_energy'] == episode['consume']['consumed_energy']
            decisions = episode['decisions']
            assert decisions['off_plant_visible_food'] == sum(decisions[k] for k in ['moves_to_visible_food', 'moves_elsewhere', 'non_moves_with_visible_food'])
            assert decisions['moves_off_plant'] + decisions['consumes_on_plant'] <= decisions['on_plant']
            key = layout, seed, stage
            signature = [episode[k] for k in ['initial_state_hash', 'first_frontier_sha256', 'effective_env_sha256', 'ruleset_hash']]
            assert paired.setdefault(key, signature) == signature, key
            totals = aggregates.setdefault((arm, stage), dict(episodes=0, initial=0, surviving=0, consumed=0, moves=0, contested=0, succeeded=0, illegal=0, aborted=0, deadline_reached=0))
            totals['deadline_reached'] += int(episode['end_reason'] == 'Some(SimTimeDeadline)' and episode['elapsed_quanta'] >= 262144)
            if sustained and episode['opponent_extinction'] is not None:
                prefix = episode['opponent_extinction']
                for key, value in {
                    'opponent_extinction_episodes': 1,
                    'opponent_extinction_survivors': prefix['training_cells'],
                    'post_extinction_survivor_change': metrics['surviving_cells'] - prefix['training_cells'],
                    'post_extinction_consumed': episode['consume']['consumed_energy'] - prefix['consume']['consumed_energy'],
                    'post_extinction_decisions': decisions['decisions'] - prefix['decisions']['decisions'],
                }.items():
                    totals[key] = totals.get(key, 0) + value
            for key, value in {'episodes': 1, 'initial': 256, 'surviving': metrics['surviving_cells'], 'consumed': metrics['consumed_energy'], 'moves': episode['movement']['completed'], 'contested': episode['movement']['contested'], 'succeeded': episode['movement']['succeeded'], 'illegal': decisions['illegal_decisions'], 'aborted': metrics['safety_aborts']}.items():
                totals[key] += value
            rows.append({'layout': layout, 'seed': seed, 'arm': arm, 'stage': stage, 'survival': metrics['survival_rate'], 'consumed_per_cell': metrics['consumed_energy_per_initial_cell'], 'completed_moves': episode['movement']['completed'], 'contested_moves': episode['movement']['contested'], 'promotion_passed': promotion['passed']})
        if not partial:
            assert progress[name]['exit_code'] == 0
        complete.add((layout, seed, arm))
    for totals in aggregates.values():
        totals['survival_rate'] = totals['surviving'] / totals['initial']
        totals['consumed_per_cell'] = totals['consumed'] / totals['initial']
        totals['contested_fraction'] = totals['contested'] / totals['moves'] if totals['moves'] else None
    verdict = None
    if not partial:
        assert complete == grid
        status = read(root / 'status.json')
        assert status['complete'] and status['completed_trials'] == len(grid)
        checks = {'all_treatment_trial_promotion_checks_pass': all(r['promotion_passed'] for r in rows if r['arm'] == 'treatment'),
                  'no_illegal_decisions_or_safety_aborts': all(v['illegal'] == v['aborted'] == 0 for v in aggregates.values())}
        if sustained:
            checks['all_treatment_episodes_reach_deadline'] = all(
                aggregates['treatment', stage]['deadline_reached'] == len(plan['layouts']) * len(plan['seeds'])
                for stage in ['on_food', 'adjacent_food'])
        treatment = aggregates['treatment', 'adjacent_food']
        for arm in ['parent', 'control']:
            other = aggregates[arm, 'adjacent_food']
            checks[f'lower_contention_than_{arm}'] = (treatment['contested_fraction'] is not None and other['contested_fraction'] is not None and treatment['contested_fraction'] < other['contested_fraction'])
            checks[f'more_adjacent_survivors_than_{arm}'] = treatment['surviving'] > other['surviving']
            checks[f'on_food_retained_vs_{arm}'] = (aggregates['treatment', 'on_food']['survival_rate'] >= aggregates[arm, 'on_food']['survival_rate'] - plan['gate']['maximum_on_food_survival_regression_vs_each_comparator'])
        verdict = {'passed': all(checks.values()), 'checks': checks}
    return {'complete_trials': len(complete), 'expected_trials': len(grid), 'aggregates': {f'{arm}/{stage}': v for (arm, stage), v in sorted(aggregates.items())}, 'gate': verdict, 'rows': rows,
            'cohort': plan.get('cohort', 'exposed_development'),
            'scope': 'Full-population development ecology; rates describe this declared cohort and initialization, not final confirmation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.partial), indent=2))
