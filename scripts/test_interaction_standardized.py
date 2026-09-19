#!/usr/bin/env python3
"""Reject normalizer, context and metric mutations without editing evidence."""
import argparse
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
from unittest.mock import patch
import verify_interaction_standardized as verifier
import verify_interaction_hard as shared


def check(root):
    root=root.resolve();original=verifier.read
    @lru_cache(maxsize=None)
    def cached(path):return original(path)
    cases=[('changed_mean','memory-normalizer.json'),('changed_scale','memory-normalizer.json'),('changed_residual_context','fits/context-audit.json'),('changed_parent_memory','fits/context-audit.json'),('forged_standardized_count','fits/report.json'),('changed_budget','plan.json')]
    results=[]
    for name,filename in cases:
        def mutated(path):
            path=path.resolve();value=cached(path)
            if path!=root/filename:return value
            value=deepcopy(value)
            if name=='changed_mean':value['means'][0]+=0.1
            elif name=='changed_scale':value['scales'][0]*=2
            elif name=='changed_residual_context':value['standardized_context_sha256']='0'*64
            elif name=='changed_parent_memory':value['parent_memory_sha256']='0'*64
            elif name=='forged_standardized_count':value['results'][4]['metrics']['feeding']['correct']-=1
            elif name=='changed_budget':value['steps']+=1
            return value
        with patch.object(verifier,'read',side_effect=mutated), patch.object(shared,'read',side_effect=mutated):
            try:verifier.verify(root)
            except AssertionError:results.append({'case':name,'rejected':True})
            else:raise AssertionError(f'audit accepted {name}')
    return {'complete':True,'checks':results,'scope':'In-memory decoded JSON mutations exercise semantic checks beyond file digests; saved evidence is unchanged.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(check(parser.parse_args().root),indent=2))
