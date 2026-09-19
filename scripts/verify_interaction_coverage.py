#!/usr/bin/env python3
"""Recompute fixed-architecture coverage results, inherited evidence and new episodes."""
import argparse
import json
from pathlib import Path
from verify_interaction_head import audit
from verify_feeding_ecology import read, sha
from verify_relational import verify as verify_prior


def verify(root):
    result=audit(root);p=read(root/'plan.json');prior=Path(p['prior_study'])
    verify_prior(prior)
    assert p['training_seeds']==[p['training_seed']]+p['new_training_seeds']
    assert len(p['new_training_seeds'])==2
    ledger=read(root/'seed-audit.json')['seed_ledger']
    assert len(ledger['training'])==63 and len(ledger['validation'])==24
    ca=read(root/'fits/corpus-audit.json')
    rows=result['corpus_rows']
    assert ca['zero_migration_rows']==sum(sum(partition.values()) for partition in rows.values())
    assert ca['all_outputs_bit_identical']
    new_episodes=[]
    for seed in p['new_training_seeds']:
        directory=Path(p['corpus_roots'][str(seed)])/str(seed)
        combat=read(directory/'combat.json')['evaluation']['episodes']
        new_episodes.extend(combat)
        for name in ['Line','Checkerboard','Ring','LooseRandom','Random']:
            new_episodes.extend(read(directory/f'feeding-{name}.json')['trial']['episodes'])
    assert len(new_episodes)==44
    validation_rows=sum(sum(rows[seed].values()) for seed in p['validation_seeds'])
    training_rows=sum(sum(rows[seed].values()) for seed in p['training_seeds'])
    return {'complete':True,'plan_sha256':sha(root/'plan.json'),'all_pass':result['all_pass'],
            'training_seeds':p['training_seeds'],'validation_seeds':p['validation_seeds'],
            'training_rows':training_rows,'validation_rows':validation_rows,
            'native_burn_validation_predictions':6*validation_rows,'independent_training_predictions':6*training_rows,
            'new_training_episodes':len(new_episodes),'new_safety_aborts':sum(e['outcome']=='Some(SafetyAbort)' for e in new_episodes),
            'results':[{'initialization':r['sampling_seed'],'arm':r['arm'],'gate':r['gate'],
                        'validation':[{'seed':v['seed'],'combat_agreement':v['combat']['agreement'],'attack_agreement':v['combat']['attack_agreement'],'feeding_retention':v['feeding']['agreement']} for v in r['validation']],
                        'training':{k:r['training'][k]['correct']/r['training'][k]['rows'] for k in ['combat','feeding']}}
                       for r in result['results']],
            'scope':'Same residual architecture, parent bytes, objective and update budget; larger audited training partition. Development rows remain excluded from training. New fits have native evidence only; prior WASM qualification is not inherited.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
