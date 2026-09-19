#!/usr/bin/env python3
"""Audit an isolated, fixture-derived memory standardization intervention."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from verify_feeding_ecology import read, sha
from verify_interaction_hard import preflight as prior_preflight, verify_fits


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def digest(values):return hashlib.sha256(b''.join(struct.pack('<f',x) for x in values)).hexdigest()


def preflight(root):
    p=read(root/'plan.json');source=Path(p['source_study'])
    assert source.resolve()!=root.resolve()
    assert sha(source/'plan.json')==p['source_plan_sha256'] and sha(source/'verification.json')==p['source_verification_sha256']
    old,rows=prior_preflight(source)
    assert set(p)==set(old)|{'normalization'}
    changed={'source_study','source_plan_sha256','source_verification_sha256','arms','scope'}
    assert all(p[k]==old[k] for k in old if k not in changed)
    assert p['arms']==['memory','standardized']
    assert sha(root/'fixture.json')==sha(source/'fixture.json')==p['fixture_sha256']
    assert sha(root/'seed-audit.json')==sha(source/'seed-audit.json')==p['seed_audit_sha256']
    assert p['normalization']['file']=='memory-normalizer.json'
    assert sha(root/'memory-normalizer.json')==p['normalization']['sha256']
    normalizer=read(root/'memory-normalizer.json');assert normalizer['fixture_sha256']==p['fixture_sha256'] and normalizer['rows']==128
    means=[sum(row['memory'][j] for row in rows)/128 for j in range(128)]
    std=[math.sqrt(sum((row['memory'][j]-means[j])**2 for row in rows)/128) for j in range(128)]
    assert normalizer['means']==[f32(v) for v in means]
    assert normalizer['scales']==[f32(v or 1.) for v in std]
    assert normalizer['constant_channels']==[i for i,v in enumerate(std) if v==0]
    transformed=[f32(f32(f32(x)-normalizer['means'][j])/normalizer['scales'][j]) for row in rows for j,x in enumerate(row['memory'])]
    audit={'fixture_sha256':p['fixture_sha256'],'rows':128,'context_width':128,
           'parent_memory_sha256':digest(x for row in rows for x in row['memory']),
           'standardized_context_sha256':digest(transformed)}
    return p,rows,audit


def verify(root):
    p,rows,context_audit=preflight(root)
    assert read(root/'fits/context-audit.json')==context_audit
    result=verify_fits(root,p,rows)
    source=Path(p['source_study']);old=read(source/'fits/report.json')['results']
    previous={(r['seed'],r['step']):r for r in old if r['arm']=='memory'}
    actual=[r for r in read(root/'fits/report.json')['results'] if r['arm']=='memory']
    assert len(actual)==len(previous)==12
    for r in actual:
        assert r==previous[r['seed'],r['step']]
        assert (root/r['weights_file']).read_bytes()==(source/r['weights_file']).read_bytes()
    return {**result,'raw_twelve_checkpoints_and_weights_identical':True,'context_transform_verified_independently':True,
            'scope':'Only residual memory standardization changes. Original parent memory, fixture, initialization, architecture, optimizer and update budget are retained. Training-only; no deployment or generalization qualification.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight:
        p,_,context=preflight(a.root);result={'complete':True,'plan_sha256':sha(a.root/'plan.json'),'context_audit':context,'validation_rows_loaded':0}
    else:result=verify(a.root)
    print(json.dumps(result,indent=2))
