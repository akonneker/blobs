#!/usr/bin/env python3
"""Recompute the deterministic training-only hard-example fixture and saved-fit checks."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from diagnose_interaction_fit import ordinary
from verify_feeding_ecology import read, sha
from verify_interaction_context import preflight as source_preflight


def select(combat, feeding, hard, seed):
    selected = sorted(hard, key=lambda i:hashlib.sha256(seed.to_bytes(8,'little')+i.to_bytes(8,'little')).digest())[:64]
    features = {i:ordinary(r) for i,r in enumerate(combat) if r['teacher_kind']==4}
    chosen = []; distances = []
    for i in selected:
        target = ordinary(feeding[i])
        distance, index = min((sum((a-b)**2 for a,b in zip(target,feature)),j) for j,feature in features.items() if j not in chosen)
        chosen.append(index); distances.append(distance)
    return chosen, selected, distances


def preflight(root):
    p = read(root/'plan.json'); source = Path(p['source_study'])
    assert sha(source/'plan.json') == p['source_plan_sha256']
    assert sha(source/'verification.json') == p['source_verification_sha256']
    old,combat,feeding,_ = source_preflight(source)
    for key in ['parent','parent_manifest_sha256','parent_weights_sha256','training_seed','training_seeds','validation_seeds','sampling_seeds','seed_audit_sha256','parameters','learning_rate','optimizer']:
        assert p[key] == old[key]
    assert sha(root/'seed-audit.json') == p['seed_audit_sha256']
    assert p['arms'] == ['observation','memory'] and p['steps']==4096 and p['checkpoints']==[128,512,2048,4096]
    assert p['batch']==128 and p['parameters']==7514
    pool_path = Path(old['source_study'])/'feeding-pool.json'
    assert sha(pool_path) == p['source_pool_sha256']
    assert sha(root/'fixture.json') == p['fixture_sha256']
    fixture = read(root/'fixture.json')
    c,f,d = select(combat,feeding,read(pool_path)['hard_indices'],p['sampling_seeds'][0])
    assert fixture['combat_indices']==c and fixture['feeding_indices']==f and fixture['squared_distances']==d
    rows = [combat[i] for i in c]+[feeding[i] for i in f]
    assert fixture['rows']==rows and len(rows)==128 and len(set(c))==len(set(f))==64
    assert all(r['teacher_kind']==4 for r in rows[:64])
    assert all(r['selected_kind']==3 for r in rows[64:])
    keys={}
    for i,row in enumerate(rows):
        label=row['teacher_kind'] if i<64 else row['selected_kind']
        key=ordinary(row)+tuple(row['memory'])+tuple(row['legal_kinds'])
        assert keys.setdefault(key,label)==label
    return p,rows


def verify(root):
    p,rows=preflight(root)
    regression=root/'scalar-regression'
    assert read(regression/'verification.json')['all_prior_replay_records_identical']
    assert sha(regression/'probe')==read(regression/'verification.json')['executable_sha256']
    assert read(regression/'scalar-replay.json')==read(Path(p['source_study'])/'scalar-replay.json')
    from diagnose_hard_separability import verify_saved
    assert verify_saved(root)==read(root/'linear-verification.json')
    return verify_fits(root,p,rows)


def verify_fits(root,p,rows):
    report=read(root/'fits/report.json'); replay=read(root/'scalar-replay.json')
    prov=read(root/'provenance.json'); assert prov['plan_sha256']==sha(root/'plan.json') and prov['executable_sha256']==sha(root/'probe')
    for name,digest in prov['sources'].items(): assert sha(root/'sources'/name)==digest
    assert report['complete'] and report['plan_sha256']==sha(root/'plan.json')
    assert replay['complete'] and replay['fit_report_sha256']==sha(root/'fits/report.json')
    assert read(root/'fits/decoder-sanity.json')=={'rows':128,'correct':128,'scope':'Label-coded score control bypasses learning; verifies legal labels and inherited kind decoding only.'}
    expected=[(seed,arm,step) for seed in p['sampling_seeds'] for arm in p['arms'] for step in p['checkpoints']]
    assert [(r['seed'],r['arm'],r['step']) for r in report['results']]==expected
    assert [(r['seed'],r['arm'],r['step']) for r in replay['results']]==expected
    summary=[];initial={};mismatches=0
    for fit,check in zip(report['results'],replay['results']):
        assert fit['initial_weights_sha256']==initial.setdefault(fit['seed'],fit['initial_weights_sha256'])
        assert fit['weights_file']==f"fits/{fit['seed']}-{fit['arm']}-{fit['step']}.f32"
        weight=root/fit['weights_file'];assert sha(weight)==fit['weights_sha256']==check['weights_sha256']
        assert all(math.isfinite(x) for x in struct.unpack('<7514f',weight.read_bytes()))
        assert fit['activation']==check['activation']
        assert math.isfinite(fit['loss']) and abs(fit['loss']-check['scalar_loss'])<1e-4
        metrics={}
        for kind,subset in [('combat',rows[:64]),('feeding',rows[64:])]:
            m=fit['metrics'][kind]; c=check['metrics'][kind]; labels=[r['teacher_kind'] if kind=='combat' else r['selected_kind'] for r in subset]
            assert m['predictions']==c['predictions'] and len(m['predictions'])==64
            assert all(r['legal_kinds'][k] for r,k in zip(subset,m['predictions']))
            correct=sum(k==label for k,label in zip(m['predictions'],labels))
            assert m['correct']==c['correct']==correct and m['rows']==c['rows']==64 and m['agreement']==c['agreement']==correct/64
            mismatches+=m['scalar_burn_mismatches'];metrics[kind]=correct
        summary.append({'seed':fit['seed'],'arm':fit['arm'],'step':fit['step'],**metrics,'exact_fit':all(v==64 for v in metrics.values()),'activation':fit['activation'],'loss':fit['loss']})
    return {'complete':True,'plan_sha256':sha(root/'plan.json'),'fixture_rows':128,'validation_rows_loaded':0,'checkpoint_predictions':len(expected)*128,'scalar_burn_kind_mismatches':mismatches,'initial_weights_match_across_arms':True,'results':summary,'scope':'Small training-only overfit fixture, full batch, no deployment or generalization qualification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight:
        p,rows=preflight(a.root);result={'complete':True,'plan_sha256':sha(a.root/'plan.json'),'fixture_rows':len(rows),'validation_rows_loaded':0}
    else: result=verify(a.root)
    print(json.dumps(result,indent=2))
