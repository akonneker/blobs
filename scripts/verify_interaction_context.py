#!/usr/bin/env python3
"""Audit the training-only context diagnostic without loading development corpora."""
import argparse
from audit_checks import equal
import hashlib
import json
import math
import struct
from pathlib import Path
from verify_feeding_ecology import read, sha

LAYOUTS = ['Line', 'Checkerboard', 'Ring', 'LooseRandom', 'Random']


def preflight(root):
    p = read(root/'plan.json'); previous = Path(p['source_study'])
    for filename, key in [('plan.json','source_plan_sha256'), ('complete-verification.json','source_verification_sha256'), ('fits/corpus-audit.json','source_corpus_audit_sha256')]:
        assert sha(previous/filename) == p[key]
    old = read(previous/'plan.json')
    for key in ['parent','parent_manifest_sha256','parent_weights_sha256','training_seed','training_seeds','validation_seeds','sampling_seeds','corpus_roots','seed_audit_sha256']:
        assert p[key] == old[key]
    assert p['arms'] == ['observation','memory','shuffled']
    assert p['steps'] == 2048 and p['checkpoints'] == [512,1024,2048]
    assert p['batch'] == 128 and p['learning_rate'] == 0.005 and p['parameters'] == 7514
    assert p['thresholds'] == {'combat':0.95,'attack':0.9,'feeding':0.99}
    assert sha(root/'seed-audit.json') == p['seed_audit_sha256'] == sha(previous/'seed-audit.json')
    ledger = read(root/'seed-audit.json')['seed_ledger']
    assert set(p['training_seeds']+p['sampling_seeds']) <= set(ledger['training'])
    assert not set(ledger['training']) & (set(ledger['validation']) | set(ledger['confirmation']))
    assert ledger['confirmation'] == [1434999901,1434999902]
    assert sha(Path(p['parent'])/'export.json') == p['parent_manifest_sha256']
    assert sha(Path(p['parent'])/'weights.bin') == p['parent_weights_sha256']
    ca = read(previous/'fits/corpus-audit.json')['train']['partitions']
    assert [v['seed'] for v in ca] == p['training_seeds']
    combat, feeding, files, counts = [], [], [], []
    for seed, expected in zip(p['training_seeds'], ca):
        directory = Path(p['corpus_roots'][str(seed)])/str(seed)
        actual = {}; ncombat = nfeeding = 0
        for name in ['combat.json'] + [f'feeding-{layout}.json' for layout in LAYOUTS]:
            rows = read(directory/name)['rows']
            actual[name] = {'sha256':sha(directory/name), 'rows':len(rows)}
            assert actual[name] == expected['files'][name]
            for row in rows:
                assert row['context'] == 'Interaction'
                assert len(row['memory']) == 128 and len(row['observation']) == 1126
                assert all(math.isfinite(x) and -1 <= x <= 1 for x in row['memory'])
            if name == 'combat.json': combat.extend(rows); ncombat += len(rows)
            else: feeding.extend(rows); nfeeding += len(rows)
        files.append({'seed':seed,'files':actual})
        counts.append({'seed':seed,'combat':ncombat,'feeding':nfeeding})
    assert counts == p['training_counts']
    assert sha(root/'memory-donors.json') == p['donor_sha256']
    d = read(root/'memory-donors.json'); n = len(combat)+len(feeding)
    assert d['seed'] == p['sampling_seeds'][0] and d['rows'] == n
    donors = sorted(range(n), key=lambda i: hashlib.sha256(d['seed'].to_bytes(8,'little')+i.to_bytes(8,'little')).digest())
    fixed = [i for i,v in enumerate(donors) if i == v]
    if len(fixed) == 1:
        i = fixed[0]; j = (i+1)%n; donors[i], donors[j] = donors[j], donors[i]
    elif fixed:
        for i,j in zip(fixed,fixed[1:]+fixed[:1]): donors[i] = j
    assert donors == d['donors'] and all(i != v for i,v in enumerate(donors))
    return p, combat, feeding, files


def verify(root):
    p, combat, feeding, files = preflight(root)
    return verify_fits(root, p, combat, feeding, files)


def verify_fits(root, p, combat, feeding, files):
    assert read(root/'fits/corpus-audit.json') == {'training':files,'combat_rows':len(combat),'feeding_rows':len(feeding),'validation_rows_loaded':0}
    provenance = read(root/'provenance.json')
    assert provenance['plan_sha256'] == sha(root/'plan.json')
    assert provenance['executable_sha256'] == sha(root/'probe')
    for name,digest in provenance['sources'].items(): assert sha(root/'sources'/name) == digest
    report = read(root/'fits/report.json'); replay = read(root/'scalar-replay.json')
    assert report['complete'] and report['plan_sha256'] == sha(root/'plan.json')
    assert replay['complete'] and replay['fit_report_sha256'] == sha(root/'fits/report.json')
    expected = [(seed,arm,step) for seed in p['sampling_seeds'] for arm in p['arms'] for step in p['checkpoints']]
    assert [(r['seed'],r['arm'],r['step']) for r in report['results']] == expected
    assert [(r['seed'],r['arm'],r['step']) for r in replay['results']] == expected
    initial = {}; streams = {}; summary = []; mismatches = 0
    for fit, check in zip(report['results'], replay['results']):
        seed, arm, step = fit['seed'], fit['arm'], fit['step']
        assert fit['initial_weights_sha256'] == initial.setdefault(seed,fit['initial_weights_sha256'])
        assert fit['batch_stream_sha256'] == streams.setdefault((seed,step),fit['batch_stream_sha256'])
        assert fit['weights_file'] == f'fits/{seed}-{arm}-{step}.f32'
        weight = root/fit['weights_file']; assert sha(weight) == fit['weights_sha256'] == check['weights_sha256']
        assert len(weight.read_bytes()) == 7514*4 and all(math.isfinite(x) for x in struct.unpack('<7514f',weight.read_bytes()))
        metrics = {}
        for name, rows in [('combat',combat),('feeding',feeding)]:
            m = fit['metrics'][name]; c = check['metrics'][name]
            equal(m['predictions'], c['predictions'], 'context.predictions', seed=seed, arm=arm, step=step, domain=name)
            equal(len(m['predictions']), len(rows), 'context.prediction_coverage')
            assert all(0 <= k < 10 and row['legal_kinds'][k] for row,k in zip(rows,m['predictions']))
            labels = [r['teacher_kind'] if name == 'combat' else r['selected_kind'] for r in rows]
            correct = sum(k == label for k,label in zip(m['predictions'],labels)); attacks = labels.count(4)
            hits = sum(k == label == 4 for k,label in zip(m['predictions'],labels))
            for key,value in [('rows',len(rows)),('correct',correct),('attack_labels',attacks),('attack_correct',hits),('agreement',correct/len(rows)),('attack_agreement',hits/attacks if attacks else 1.)]:
                equal(m[key], value, 'context.metrics', seed=seed, arm=arm, step=step, domain=name, key=key)
                equal(c[key], value, 'context.replay_metrics', seed=seed, arm=arm, step=step, domain=name, key=key)
            assert isinstance(m['scalar_burn_mismatches'],int) and 0 <= m['scalar_burn_mismatches'] <= len(rows)
            mismatches += m['scalar_burn_mismatches']
            metrics[name] = m['agreement']
            if name == 'combat': metrics['attack'] = m['attack_agreement']
        met = all(metrics[k] >= p['thresholds'][k] for k in p['thresholds'])
        equal(fit['metrics']['training_fit_thresholds_met'], met, 'context.joint_gate')
        equal(check['metrics']['training_fit_thresholds_met'], met, 'context.replay_joint_gate')
        summary.append({'seed':seed,'arm':arm,'step':step,**metrics,'training_fit_thresholds_met':met})
    return {'complete':True,'plan_sha256':sha(root/'plan.json'),'training_rows':len(combat)+len(feeding),'validation_rows_loaded':0,'checkpoints':len(expected),'scalar_replayed_predictions':len(expected)*(len(combat)+len(feeding)),'scalar_burn_kind_mismatches':mismatches,'all_arms_initialization_and_batch_streams_match':True,'results':summary,'scope':'Training-only direct-context diagnostic. Parent scores already depend on actual private state. Equal nominal parameters; observation arm has zero context features. No deployment export, new seed, validation fit or promotion.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path); parser.add_argument('--preflight',action='store_true'); args = parser.parse_args()
    if args.preflight:
        p,c,f,_ = preflight(args.root)
        result = {'complete':True,'plan_sha256':sha(args.root/'plan.json'),'combat_rows':len(c),'feeding_rows':len(f),'validation_rows_loaded':0}
    else: result = verify(args.root)
    print(json.dumps(result,indent=2))
