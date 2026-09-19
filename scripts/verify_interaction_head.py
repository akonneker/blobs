#!/usr/bin/env python3
"""Audit frozen head-fit provenance, splits, saved predictions and retention gate."""
import argparse
from collections import Counter
import json
import math
import struct
from pathlib import Path
from verify_feeding_ecology import read, sha

LAYOUTS = ['Line','Checkerboard','Ring','LooseRandom','Random']


def audit(root):
    plan = read(root / 'plan.json')
    source = Path(plan.get('corpus_source', root))
    reused = source != root
    adapter = plan.get('intervention') == 'slot_adapter'
    relational = plan.get('intervention') == 'relational_residual'
    if reused:
        assert source.resolve() != root.resolve()
        assert sha(source / 'plan.json') == plan['corpus_source_plan_sha256']
        assert sha(source / 'verification.json') == plan['corpus_source_verification_sha256']
        audit(source)
    training_seeds = plan.get('training_seeds', [plan['training_seed']])
    validation_seeds = plan['validation_seeds']
    assert training_seeds and training_seeds[0] == plan['training_seed']
    assert len(set(training_seeds)) == len(training_seeds)
    assert validation_seeds and len(set(validation_seeds)) == len(validation_seeds)
    assert not set(training_seeds) & set(validation_seeds)
    seeds = training_seeds + validation_seeds
    new_training = plan.get('new_training_seeds', [])
    sampling_source = plan.get('sampling_source')
    if sampling_source:
        audit_sampling_plan(root, plan)
    seed_audit = read(root / 'seed-audit.json')
    assert sha(root / 'seed-audit.json') == plan['seed_audit_sha256']
    inherited = Path(seed_audit['inherited_ledger_path'])
    assert sha(inherited) == seed_audit['inherited_ledger_sha256']
    assert read(inherited)['seed_ledger'] == seed_audit['inherited_seed_ledger']
    old = seed_audit['inherited_seed_ledger']
    ledger = seed_audit['seed_ledger']
    if new_training:
        assert training_seeds == [plan['training_seed']] + new_training
        assert seed_audit['new_training_seeds'] == new_training
        assert not set(new_training) & set(sum(old.values(), []))
        assert ledger['training'] == sorted(old['training'] + new_training)
        assert ledger['validation'] == old['validation']
        assert ledger['confirmation'] == old['confirmation'] == [1434999901,1434999902]
        assert set(validation_seeds) <= set(old['validation'])
        assert set([plan['training_seed']] + plan['sampling_seeds']) <= set(old['training'])
        assert set(plan['corpus_roots']) == {str(s) for s in seeds}
        if not sampling_source:
            cp=read(root/'collection-provenance.json')
            assert cp['study_plan_sha256']==sha(root/'plan.json')
            assert cp['collector_sha256']==sha(root/'collector')
            original_cp=Path(cp['source_provenance_path'])
            assert sha(original_cp)==cp['source_provenance_sha256']
            assert read(original_cp)['collector_sha256']==cp['collector_sha256']
            for name,digest in read(original_cp)['sources'].items():
                assert sha(original_cp.parent/'sources'/name)==digest
            assert cp['seeds']==new_training and cp['process_concurrency']==1
            assert read(root/'collection-status.json')=={'complete':True,'seeds':new_training,'new_episodes':len(new_training)*22}
        prior=Path(plan['prior_study'])
        assert sha(prior/'plan.json')==plan['prior_study_plan_sha256']
        assert sha(prior/'complete-verification.json')==plan['prior_study_verification_sha256']
        unchanged=['parent','parent_manifest_sha256','parent_weights_sha256','steps','batch','learning_rate','optimizer','head','labels','validation_seeds','sampling_seeds','initialization','execution_contract','envelope','gate','source_config_sha256','intervention','collection','deployment_gate']
        assert all(plan[k]==read(prior/'plan.json')[k] for k in unchanged)
    else:
        assert ledger['training'] == sorted(old['training'] + [seeds[0]] + plan['sampling_seeds'])
        assert ledger['validation'] == sorted(old['validation'] + validation_seeds)
        assert ledger['confirmation'] == old['confirmation'] == [1434999901,1434999902]
        assert not set(seeds + plan['sampling_seeds']) & set(sum(old.values(),[]))
    assert not set(ledger['training']) & set(ledger['validation'])
    assert seed_audit['prior_local_search']['exit_code'] == 1
    parent = Path(plan['parent'])
    assert sha(parent / 'export.json') == plan['parent_manifest_sha256']
    assert sha(parent / 'weights.bin') == plan['parent_weights_sha256']
    provenance = [('fit-provenance.json','trainer')] if reused else [('collection-provenance.json','collector'),('fit-provenance.json','trainer')]
    for name, executable in provenance:
        p = read(root / name)
        assert p['plan_sha256'] == sha(root / 'plan.json')
        assert p[('collector' if executable == 'collector' else 'trainer') + '_sha256'] == sha(root / executable)
        for filename,digest in p['sources'].items():
            assert sha(root / 'sources' / filename) == digest
    corpora = {}
    ca = read(root / 'fits/corpus-audit.json')
    feeding_episodes = []
    for index,seed in enumerate(seeds):
        d = Path(plan.get('corpus_roots', {}).get(str(seed), source)) / str(seed)
        assert read(d / 'status.json')['complete']
        p = read(d / 'plan.json')
        assert p['seed'] == seed and p['manifest_sha256'] == plan['parent_manifest_sha256']
        assert p['weights_sha256'] == plan['parent_weights_sha256'] and p['config_sha256'] == plan['source_config_sha256']
        if len(training_seeds)>1:
            assert [p['seed'] for p in ca['train']['partitions']]==training_seeds
            expected = ca['train']['partitions'][index]['files'] if index < len(training_seeds) else ca['validation'][index-len(training_seeds)]
        else:
            expected = ca['train'] if index == 0 else ca['validation'][index-1]
        combat,feeding = [],[]
        for name in ['combat.json'] + [f'feeding-{layout}.json' for layout in LAYOUTS]:
            value = read(d / name)
            rows = value['rows']
            assert sha(d / name) == expected[name]['sha256'] and len(rows) == expected[name]['rows']
            for row in rows:
                assert row['context'] == 'Interaction' and len(row['hidden']) == len(row['memory']) == 128
                assert len(row['observation']) == 1126 and len(row['legal_kinds']) == 10
                assert row['legal_kinds'][row['selected_kind']] and row['legal_kinds'][row['teacher_kind']]
            assert all(n <= 2048 for n in Counter(r['episode'] for r in rows).values())
            if name == 'combat.json':
                assert all(e['seed'] == seed for e in value['evaluation']['episodes'])
                assert len(value['evaluation']['episodes']) == 12
                combat.extend(rows)
            else:
                assert value['trial']['seed'] == seed
                assert value['trial']['assessment_mode'] == 'sustained_feeding'
                assert len(value['trial']['episodes']) == 2
                feeding_episodes.extend(value['trial']['episodes'])
                feeding.extend(rows)
        corpora[seed] = {'combat':combat,'feeding':feeding}
    fit = read(root / 'fits/report.json')
    replay = read(root / 'portable-replay.json')
    rp = read(root / 'replay-provenance.json')
    assert rp['verifier_sha256'] == sha(root / 'portable-verifier')
    assert rp['source_sha256'] == sha(root / 'sources/verify_interaction_head.rs')
    assert rp['fit_report_sha256'] == replay['fit_report_sha256'] == sha(root / 'fits/report.json')
    assert fit['complete'] and replay['complete'] and fit['plan_sha256'] == sha(root / 'plan.json')
    expected = [(seed,arm) for seed in plan['sampling_seeds'] for arm in ['control','treatment']]
    assert [(r['sampling_seed'],r['arm']) for r in fit['results']] == expected
    assert [(r['sampling_seed'],r['arm']) for r in replay['results']] == expected
    results = []
    for result,record in zip(fit['results'],replay['results']):
        arm = result['arm']; seed = result['sampling_seed']
        d = root / 'fits' / f'{seed}-{arm}'
        assert result == read(d / 'result.json')
        manifest = read(d / 'export.json')
        assert sha(d / 'export.json') == record['manifest_sha256']
        assert sha(d / 'weights.bin') == result['weights_sha256'] == record['weights_sha256'] == manifest['weights_sha256']
        assert manifest['source_seed_ledger'] == ledger
        before = (parent / 'weights.bin').read_bytes()
        after = (d / 'weights.bin').read_bytes()
        if relational:
            assert before[:8] == b'BLCMP001' and after[:8] == b'BLCMR001'
            size = struct.unpack('<I',after[8:12])[0]
            assert size == len(before) and after[12:12+size] == before
            assert len(after) == 12+size+3418*4
            assert all(math.isfinite(x) for x in struct.unpack('<3418f',after[12+size:]))
            assert manifest['execution_contract'] == 'blob.policy.interaction-slot-encoder-f32-libm-move-q48.v1'
            assert result['added_parameters'] == 3418 and result['initialization_seed'] == seed
            assert record['parent_and_utility_bytes_identical'] and record['utility_identical']
        else:
            assert before[:8] == after[:8] == b'BLCMP001'
            assert before[12:20] == b'BLPOL001'
            assert struct.unpack('<III', before[20:32]) == (256,128,128)
            shapes = [(33,128),(198,256),(384,128),(128,128),(128,10),(136,16),(16,10)]
            if adapter:
                shapes += [(128,10),(166,16),(16,10)]
            parameters = 1322 if adapter else 1290
            start = 20 + 4*sum(i*o+o for i,o in shapes)
            end = start + parameters*4
            assert manifest['updated_parameter_range_in_parent'] == [start,end]
            assert len(before) == len(after)
            assert before[:12+start] == after[:12+start]
            assert before[12+end:] == after[12+end:]  # Includes every utility bit.
            assert result['updated_parameters'] == parameters and record['only_declared_head_changed'] and record['utility_identical']
        assert [v['seed'] for v in result['validation']] == validation_seeds
        assert [v['seed'] for v in record['validation']] == validation_seeds
        gates=[]
        for v,rv in zip(result['validation'],record['validation']):
            for kind in ['combat','feeding']:
                rows = corpora[v['seed']][kind]
                labels = [r['teacher_kind'] if arm=='treatment' and kind=='combat' else r['selected_kind'] for r in rows]
                predictions = rv[kind]['predictions']
                if relational:
                    assert predictions == v[kind]['predictions']
                assert len(predictions) == len(rows)
                assert all(0 <= p < 10 and r['legal_kinds'][p] for p,r in zip(predictions,rows))
                correct = sum(a==b for a,b in zip(labels,predictions))
                attacks = labels.count(4)
                hits = sum(a==b==4 for a,b in zip(labels,predictions))
                for key,value in [('rows',len(rows)),('correct',correct),('attack_labels',attacks),('attack_correct',hits)]:
                    assert v[kind][key] == rv[kind][key] == value
                assert v[kind]['agreement'] == correct/len(rows)
                assert v[kind]['attack_agreement'] == (hits/attacks if attacks else 1.)
            c,f = v['combat'],v['feeding']
            gates.append(c['agreement'] >= .95 and (arm=='control' or c['attack_agreement'] >= .9) and f['agreement'] >= .99 and c['portable_burn_kind_mismatches']==f['portable_burn_kind_mismatches']==0)
        assert result['offline_gate_pass'] == all(gates)
        results.append({'sampling_seed':seed,'arm':arm,'gate':all(gates),'validation':result['validation']})
        if relational:
            training=record['training']; assert training['seed'] == training_seeds[0]
            if len(training_seeds)>1: assert training['seeds']==training_seeds
            for kind in ['combat','feeding']:
                rows=[r for seed in training_seeds for r in corpora[seed][kind]]; pred=training[kind]['predictions']
                labels=[r['teacher_kind'] if arm=='treatment' and kind=='combat' else r['selected_kind'] for r in rows]
                assert len(pred)==len(rows)==training[kind]['rows']
                assert all(0 <= k < 10 and r['legal_kinds'][k] for k,r in zip(pred,rows))
                assert sum(k==label for k,label in zip(pred,labels))==training[kind]['correct']
                assert labels.count(4)==training[kind]['attack_labels']
                assert sum(k==label==4 for k,label in zip(pred,labels))==training[kind]['attack_correct']
            results[-1]['training']=training
    assert fit['all_pass'] == all(r['gate'] for r in results)
    sentinel = read(source / 'feeding-sentinel-verification.json')
    old_path = Path(sentinel['original'])
    assert sha(old_path) == sentinel['original_sha256']
    assert sha(source / 'feeding-sentinel/report.json') == sentinel['new_sha256']
    assert sha(source / 'feeding-evaluator') == sentinel['evaluator_sha256']
    assert read(old_path)['trial'] == read(source / 'feeding-sentinel/report.json')['trial']
    return {'complete':True,'all_pass':fit['all_pass'],'results':results,
            'corpus_rows':{s:{k:len(v) for k,v in rows.items()} for s,rows in corpora.items()},
            'frozen_collection_episodes':0 if sampling_source else len(new_training)*22 if new_training else (0 if reused else 66),'feeding_sentinel_episodes':0 if reused else 2,
            **({'reused_collection_episodes':110 if sampling_source else 66,'reused_sentinel_episodes':2} if reused else {}),
            'feeding_collection_safety_aborts':sum(e['outcome']=='Some(SafetyAbort)' for e in feeding_episodes),
            'scope':('Offline per-slot relational residual on reused development rows; no promotion. Native/Burn agreement and independently replayed predictions.' if relational else 'Offline local-adapter study on reused development rows, no ecological promotion or WASM qualification. Saved predictions independently regenerated through the portable runtime.' if adapter else 'Offline head-only study, no ecological promotion or WASM qualification. Saved predictions independently regenerated through the portable runtime.')}


def audit_sampling_plan(root, plan):
    previous = Path(plan['sampling_source'])
    assert previous.resolve() != root.resolve()
    assert sha(previous/'plan.json') == plan['sampling_source_plan_sha256']
    assert sha(previous/'portable-replay.json') == plan['sampling_source_replay_sha256']
    assert sha(previous/'complete-verification.json') == plan['sampling_source_verification_sha256']
    audit(previous)
    old = read(previous/'plan.json')
    added = {'sampling_source','sampling_source_plan_sha256','sampling_source_replay_sha256',
             'sampling_source_verification_sha256','feeding_sampling'}
    assert set(plan) == set(old) | added
    assert all(plan[k] == old[k] for k in old if k not in {'scope','sampling'})
    assert sha(root/'seed-audit.json') == sha(previous/'seed-audit.json')
    config = plan['feeding_sampling']
    assert config['hard_per_batch'] == 32
    assert config['pool_file'] == 'feeding-pool.json'
    assert sha(root/config['pool_file']) == config['pool_sha256']
    pool = read(root/config['pool_file'])
    assert pool['source_study'] == str(previous)
    assert pool['source_replay_sha256'] == plan['sampling_source_replay_sha256']
    assert pool['source_corpus_audit_sha256'] == sha(previous/'fits/corpus-audit.json')
    assert pool['training_seeds'] == plan['training_seeds']
    assert pool['sampling_seeds'] == plan['sampling_seeds']
    labels = []
    for seed in plan['training_seeds']:
        directory = Path(plan['corpus_roots'][str(seed)])/str(seed)
        for layout in LAYOUTS:
            labels.extend(r['selected_kind'] for r in read(directory/f'feeding-{layout}.json')['rows'])
    fits = [r for r in read(previous/'portable-replay.json')['results'] if r['arm']=='treatment']
    assert [r['sampling_seed'] for r in fits] == plan['sampling_seeds']
    predictions = [r['training']['feeding']['predictions'] for r in fits]
    assert all(len(v)==len(labels) for v in predictions)
    expected = [i for i,label in enumerate(labels) if any(v[i]!=label for v in predictions)]
    assert pool['hard_indices'] == expected
    assert pool['feeding_rows'] == len(labels)
    assert 0 < len(expected) < len(labels)
    return {'complete':True,'plan_sha256':sha(root/'plan.json'),
            'pool_sha256':config['pool_sha256'],'training_feeding_rows':len(labels),
            'hard_rows':len(expected),'remaining_rows':len(labels)-len(expected),
            'new_episodes':0,'new_seeds':0,
            'scope':'Pool recomputed only from all three source treatments training predictions; unchanged corpus, parent, architecture, optimizer, update budget and gates.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(audit(p.parse_args().root),indent=2))
