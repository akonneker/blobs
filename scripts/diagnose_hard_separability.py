#!/usr/bin/env python3
"""Fixed-ridge linear feasibility check on the already frozen training fixture."""
import argparse
import json
import math
from pathlib import Path
from diagnose_interaction_fit import ordinary
from verify_feeding_ecology import read, sha


def solve(matrix, target):
    n=len(target);a=[row[:]+[y] for row,y in zip(matrix,target)]
    for i in range(n):
        pivot=max(range(i,n),key=lambda j:abs(a[j][i]));a[i],a[pivot]=a[pivot],a[i]
        assert abs(a[i][i])>1e-12
        for j in range(i+1,n):
            factor=a[j][i]/a[i][i]
            for k in range(i+1,n+1):a[j][k]-=factor*a[i][k]
            a[j][i]=0.
    result=[0.]*n
    for i in reversed(range(n)):result[i]=(a[i][n]-sum(a[i][j]*result[j] for j in range(i+1,n)))/a[i][i]
    return result


def diagnose(root):
    plan=read(root/'linear-plan.json');assert sha(root/'fixture.json')==plan['fixture_sha256']
    assert plan['arms']==['observation','memory'] and plan['ridge']==0.001
    rows=read(root/'fixture.json')['rows'];assert len(rows)==128
    assert all(r['legal_kinds'][3] and r['legal_kinds'][4] for r in rows)
    target=[1.]*64+[-1.]*64;results=[]
    for arm in plan['arms']:
        raw=[list(ordinary(r))+(r['memory'] if arm=='memory' else []) for r in rows];n=len(raw);width=len(raw[0])
        means=[sum(row[j] for row in raw)/n for j in range(width)]
        scales=[math.sqrt(sum((row[j]-means[j])**2 for row in raw)/n) for j in range(width)]
        x=[[(value-means[j])/(scales[j] or 1.) for j,value in enumerate(row)]+[1.] for row in raw]
        kernel=[[sum(a*b for a,b in zip(left,right))+(plan['ridge'] if i==j else 0.) for j,right in enumerate(x)] for i,left in enumerate(x)]
        alpha=solve(kernel,target)
        residual=max(abs(sum(a*b for a,b in zip(row,alpha))-y) for row,y in zip(kernel,target));assert residual<1e-6
        weights=[sum(row[j]*a for row,a in zip(x,alpha)) for j in range(width+1)]
        scores=[sum(w*v for w,v in zip(weights,row)) for row in x]
        predictions=[4 if score>0 else 3 for score in scores]
        record={'arm':arm,'features':width,'nonconstant_features':sum(s>0 for s in scales),'solver_max_residual':residual,'correct':sum(k==(4 if i<64 else 3) for i,k in enumerate(predictions)),'combat_correct':sum(k==4 for k in predictions[:64]),'feeding_correct':sum(k==3 for k in predictions[64:]),'minimum_signed_margin':min(s*y for s,y in zip(scores,target)),'means':means,'scales':scales,'weights':weights,'scores':scores,'predictions':predictions}
        # Serialize/reload before reconstructing the predictions from the feature input.
        saved=json.loads(json.dumps(record))
        replay=[sum(w*v for w,v in zip(saved['weights'],[(v-saved['means'][j])/(saved['scales'][j] or 1.) for j,v in enumerate(row)]+[1.])) for row in raw]
        assert replay==scores
        results.append(record)
    return {'complete':True,'plan_sha256':sha(root/'linear-plan.json'),'fixture_sha256':sha(root/'fixture.json'),'results':results,'scope':plan['scope']}


def verify_saved(root):
    plan=read(root/'linear-plan.json');report=read(root/'linear-separability.json')
    assert report['plan_sha256']==sha(root/'linear-plan.json')
    assert report['fixture_sha256']==plan['fixture_sha256']==sha(root/'fixture.json')
    assert [r['arm'] for r in report['results']]==plan['arms']==['observation','memory']
    rows=read(root/'fixture.json')['rows'];target=[1.]*64+[-1.]*64;checks=[]
    for fit in report['results']:
        raw=[list(ordinary(r))+(r['memory'] if fit['arm']=='memory' else []) for r in rows]
        width=len(raw[0]);assert fit['features']==len(fit['means'])==len(fit['scales'])==width
        assert len(fit['weights'])==width+1
        assert all(math.isfinite(v) for key in ['weights','means','scales'] for v in fit[key])
        assert all(s>=0 for s in fit['scales'])
        x=[[(v-fit['means'][j])/(fit['scales'][j] or 1.) for j,v in enumerate(row)]+[1.] for row in raw]
        scores=[math.fsum(w*v for w,v in zip(fit['weights'],row)) for row in x]
        assert all(abs(a-b)<1e-7 for a,b in zip(scores,fit['scores']))
        predictions=[4 if score>0 else 3 for score in scores]
        assert predictions==fit['predictions']
        correct=sum(k==(4 if i<64 else 3) for i,k in enumerate(predictions));assert correct==fit['correct']
        residual=max(abs(math.fsum(row[j]*(y-score) for row,y,score in zip(x,target,scores))-plan['ridge']*fit['weights'][j]) for j in range(width+1))
        assert residual<1e-5
        checks.append({'arm':fit['arm'],'correct':correct,'max_normal_equation_residual':residual})
    return {'complete':True,'artifact_sha256':sha(root/'linear-separability.json'),'checks':checks,'scope':'Recomputed inputs, predictions and ridge normal equations from saved coefficients, without refitting.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    print(json.dumps(verify_saved(args.root) if args.verify else diagnose(args.root),indent=2))
