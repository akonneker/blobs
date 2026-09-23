"""Count-based summaries for the frozen full-corpus conditioning diagnostic."""
import math

from audit_checks import equal, require


def scientific(result):
    rows = result['results']
    require(bool(rows), 'study.empty_results')
    keys = [(r['seed'], r['arm'], r['step']) for r in rows]
    equal(len(keys), len(set(keys)), 'study.duplicate_checkpoint')
    thresholds = result['thresholds']
    equal(thresholds, {'combat': .95, 'attack': .9, 'feeding': .99}, 'study.thresholds')
    for row in rows:
        counts = row['counts']
        for domain in ('combat', 'feeding'):
            c = counts[domain]
            require(all(type(c[k]) is int for k in ('rows', 'correct', 'attack_labels', 'attack_correct')), 'study.invalid_counts')
            require(0 <= c['attack_correct'] <= c['attack_labels'] <= c['rows'] and
                    c['attack_correct'] <= c['correct'] <= c['rows'] and c['rows'] > 0, 'study.invalid_counts')
            equal(row[domain], c['correct'] / c['rows'], 'study.agreement')
        c = counts['combat']
        require(c['attack_labels'] > 0, 'study.no_attack_labels')
        equal(row['attack'], c['attack_correct'] / c['attack_labels'], 'study.agreement')
        equal(row['training_fit_thresholds_met'], all(row[k] >= v for k, v in thresholds.items()), 'study.joint_gate')
    last = max(r['step'] for r in rows)
    finals = [r for r in rows if r['step'] == last]
    arms = []
    for arm in ('memory', 'zero-state'):
        group = [r for r in finals if r['arm'] == arm]
        require(bool(group), 'study.empty_target')
        margins = []
        for row in group:
            for domain, threshold in thresholds.items():
                c = row['counts']['combat' if domain == 'attack' else domain]
                n, correct = (c['attack_labels'], c['attack_correct']) if domain == 'attack' else (c['rows'], c['correct'])
                minimum = math.ceil(threshold * n)
                margins.append({'correct': correct, 'rows': n, 'seed': row['seed'], 'domain': domain,
                                'threshold': minimum, 'margin': correct - minimum})
        arms.append({'arm': arm, 'runs': len(group), 'passed': sum(r['training_fit_thresholds_met'] for r in group),
                     'worst': min(margins, key=lambda v: v['margin'] / v['rows'])})
    paired = []
    for row in finals:
        if row['arm'] != 'zero-state':
            continue
        control = next(r for r in finals if r['arm'] == 'memory' and r['seed'] == row['seed'])
        paired.append({'seed': row['seed'], **{k + '_delta': row['counts'][k]['correct'] - control['counts'][k]['correct']
                                             for k in ('combat', 'feeding')}})
    target = arms[1]
    return {'outcome': 'passed' if target['passed'] == target['runs'] else 'rejected',
            'gate': 'all_target_initializations_joint_thresholds_at_final_checkpoint', 'gate_label': 'joint fit',
            'target_arm': 'zero-state', 'step': last, 'arms': arms, 'paired_correct_deltas': paired,
            'checkpoints_completed': len(rows), 'checkpoints_requested': len(rows),
            'predictions': result['checkpoint_predictions'],
            'raw_control_regression': result['raw_nine_checkpoints_and_weights_identical'],
            'not_qualified': ['action value', 'development', 'deployed combat', 'self-play'],
            'scope': 'Training-only joint fit; useful autonomous behavior remains unqualified.'}
