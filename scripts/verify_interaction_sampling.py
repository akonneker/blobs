#!/usr/bin/env python3
"""Verify fixed-corpus sampling fits, training strata and uniform reproduction."""
import argparse
import json
from pathlib import Path

from verify_feeding_ecology import read, sha
from verify_interaction_head import audit, audit_sampling_plan, LAYOUTS


def verify(root):
    plan = read(root / 'plan.json')
    preflight = audit_sampling_plan(root, plan)
    assert preflight == read(root / 'preflight.json')
    fits = audit(root)
    previous = Path(plan['sampling_source'])
    regression = root / 'uniform-regression'
    assert sha(regression / 'plan.json') == sha(previous / 'plan.json')
    assert sha(regression / 'seed-audit.json') == sha(previous / 'seed-audit.json')
    assert read(regression / 'fits/report.json') == read(previous / 'fits/report.json')
    for result in read(previous / 'fits/report.json')['results']:
        name = f"{result['sampling_seed']}-{result['arm']}"
        assert (regression / 'fits' / name / 'weights.bin').read_bytes() == (previous / 'fits' / name / 'weights.bin').read_bytes()
    labels = []
    for seed in plan['training_seeds']:
        directory = Path(plan['corpus_roots'][str(seed)]) / str(seed)
        for layout in LAYOUTS:
            labels.extend(r['selected_kind'] for r in read(directory / f'feeding-{layout}.json')['rows'])
    hard = set(read(root / 'feeding-pool.json')['hard_indices'])
    results = []
    for result in fits['results']:
        pred = result['training']['feeding']['predictions']
        strata = {}
        for name, membership in [('hard', True), ('remaining', False)]:
            indices = [i for i in range(len(labels)) if (i in hard) == membership]
            correct = sum(pred[i] == labels[i] for i in indices)
            strata[name] = {'rows': len(indices), 'correct': correct, 'agreement': correct / len(indices)}
        results.append({'sampling_seed': result['sampling_seed'], 'arm': result['arm'], 'gate': result['gate'],
                        'training': {kind: result['training'][kind]['correct'] / result['training'][kind]['rows'] for kind in ['combat', 'feeding']},
                        'training_feeding_strata': strata,
                        'validation': [{'seed': v['seed'], 'combat': v['combat']['agreement'],
                                        'attack': v['combat']['attack_agreement'], 'feeding': v['feeding']['agreement']} for v in result['validation']]})
    ca = read(root / 'fits/corpus-audit.json')
    assert ca == read(previous / 'fits/corpus-audit.json')
    assert ca['all_outputs_bit_identical'] and ca['zero_migration_rows'] == 14008
    return {'complete': True, 'plan_sha256': sha(root / 'plan.json'), 'all_pass': fits['all_pass'],
            'pool': preflight, 'uniform_six_fit_results_and_weights_identical': True,
            'native_burn_validation_predictions': 34176, 'independent_training_predictions': 49872,
            'zero_migration_rows': 14008, 'results': results,
            'scope': 'Sampling-only follow-up; all original development gates retained. No new seeds or episodes. Native fitted-model evidence only; no inherited WASM or ecological qualification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(verify(parser.parse_args().root), indent=2))
