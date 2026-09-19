#!/usr/bin/env python3
"""Semantic mutation checks for the hard-example fixture audit."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch
import verify_interaction_hard as verifier


def check(root):
    root=root.resolve();original=verifier.read
    @lru_cache(maxsize=None)
    def cached(path):return original(path)
    cases=[('fixture_membership','fixture.json'),('validation_in_training','plan.json'),('decoder_sanity','fits/decoder-sanity.json'),('unmatched_initialization','fits/report.json'),('forged_loss','fits/report.json'),('forged_training_count','fits/report.json')]
    result=[]
    for name,filename in cases:
        def mutated(path):
            path=path.resolve();value=cached(path)
            if path!=root/filename:return value
            value=deepcopy(value)
            if name=='fixture_membership':value['feeding_indices'].reverse()
            elif name=='validation_in_training':value['training_seeds'].append(value['validation_seeds'][0])
            elif name=='decoder_sanity':value['correct']-=1
            elif name=='unmatched_initialization':value['results'][4]['initial_weights_sha256']='0'*64
            elif name=='forged_loss':value['results'][0]['loss']+=1
            elif name=='forged_training_count':value['results'][0]['metrics']['feeding']['correct']+=1
            return value
        with patch.object(verifier,'read',side_effect=mutated):
            try:verifier.verify(root)
            except AssertionError:result.append({'case':name,'rejected':True})
            else:raise AssertionError(f'audit accepted {name}')
    return {'complete':True,'checks':result,'scope':'Decoded JSON mutations in memory exercise semantic checks beyond file hashes; retained evidence is unchanged.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(check(parser.parse_args().root),indent=2))
