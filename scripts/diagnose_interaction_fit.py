#!/usr/bin/env python3
"""Local-feature aliasing and feeding false-Attack slices; no labels are changed."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from verify_feeding_ecology import read, sha
from verify_interaction_head import audit, LAYOUTS


def local(row):
    obs=row['observation']
    return tuple(obs[:38])+tuple(max(obs[j] for j in range(70+i,len(obs),33)) for i in range(33))


def ordinary(row):
    obs=row['observation']
    slots=sorted(tuple(obs[i:i+33]) for i in range(70,len(obs),33))
    return tuple(obs[:38])+tuple(value for slot in slots for value in slot)


def diagnose(root):
    audit(root)
    p=read(root/'plan.json');source=Path(p.get('corpus_source',root))
    training_seeds=p.get('training_seeds',[p['training_seed']])
    def corpus(seed):
        return Path(p.get('corpus_roots',{}).get(str(seed),source))/str(seed)
    all_rows=[];training_attacks=set();training_full_attacks=set()
    for seed in training_seeds+p['validation_seeds']:
        for name in ['combat.json']+[f'feeding-{l}.json' for l in LAYOUTS]:
            for row in read(corpus(seed)/name)['rows']:
                label=row['teacher_kind'] if name=='combat.json' else row['selected_kind']
                all_rows.append((row,label,seed,name))
                if seed in training_seeds and name=='combat.json' and label==4:
                    training_attacks.add(local(row));training_full_attacks.add(ordinary(row))
    aliases={}
    for name,key in [('local_max_summary',local),('complete_nonrandom_observation',ordinary)]:
        grouped=defaultdict(Counter)
        for row,label,_,_ in all_rows:grouped[key(row)][label]+=1
        conflicts=[v for v in grouped.values() if len(v)>1]
        aliases[name]={'groups':len(grouped),'conflicting_groups':len(conflicts),'conflicting_rows':sum(sum(v.values()) for v in conflicts),'minimum_label_disagreements_for_a_deterministic_function_of_only_these_features':sum(sum(v.values())-max(v.values()) for v in conflicts)}
    results=[]
    for arm in read(root/'portable-replay.json')['results']:
        if arm['arm']!='treatment':continue
        groups=[]
        for v in arm['validation']:
            start=0
            for layout in LAYOUTS:
                rows=read(corpus(v['seed'])/f'feeding-{layout}.json')['rows']
                predictions=v['feeding']['predictions'][start:start+len(rows)];start+=len(rows)
                errors=[(row,k) for row,k in zip(rows,predictions) if row['selected_kind']!=k]
                groups.append({'seed':v['seed'],'layout':layout,'rows':len(rows),'errors':len(errors),'predicted_error_kinds':dict(Counter(k for _,k in errors)),'errors_matching_training_attack_local_features':sum(local(row) in training_attacks for row,_ in errors),'errors_matching_training_attack_complete_nonrandom_observation':sum(ordinary(row) in training_full_attacks for row,_ in errors)})
        results.append({'sampling_seed':arm['sampling_seed'],'feeding':groups})
    return {'plan_sha256':sha(root/'plan.json'),'replay_sha256':sha(root/'portable-replay.json'),'rows':len(all_rows),'aliases':aliases,'treatments':results,'scope':'Post-failure development diagnosis. Feature-only lower bounds do not include frozen base/context offsets, legal masks or recurrent state, so they are not impossibility claims about the full deployed Mind. No training or tuning on validation rows.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(diagnose(p.parse_args().root),indent=2))
