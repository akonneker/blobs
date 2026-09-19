#!/usr/bin/env python3
"""Training-only range audit before reusing fixture-derived memory scales."""
import argparse
import json
import math
from pathlib import Path
from verify_feeding_ecology import read,sha
from verify_interaction_standardized import preflight,f32
from verify_interaction_context import preflight as corpus_preflight


def diagnose(root):
    p,fixture,_=preflight(root);hard=read(Path(p['source_study'])/'plan.json');context=Path(hard['source_study'])
    _,combat,feeding,_=corpus_preflight(context);normalizer=read(root/'memory-normalizer.json')
    def ranges(rows):
        maxima=[];cold=[];nonzero=[]
        for row in rows:
            values=[f32(f32(f32(x)-normalizer['means'][j])/normalizer['scales'][j]) for j,x in enumerate(row['memory'])]
            assert all(math.isfinite(x) for x in values)
            maximum=max(map(abs,values));maxima.append(maximum)
            (nonzero if any(row['memory']) else cold).append(maximum)
        ordered=sorted(maxima)
        return {'rows':len(rows),'max_abs_context':max(maxima),'median_row_max_abs_context':ordered[len(ordered)//2],
                'rows_exceeding_abs_context':{str(t):sum(x>t for x in maxima) for t in [4,8,16,64,1024]},
                'zero_memory_rows':len(cold),'nonzero_memory_rows':len(nonzero),
                'max_abs_nonzero_memory_context':max(nonzero) if nonzero else None}
    return {'complete':True,'plan_sha256':sha(root/'plan.json'),'normalizer_sha256':sha(root/'memory-normalizer.json'),
            'fixture':ranges(fixture),'all_training':ranges(combat+feeding),'combat':ranges(combat),'feeding':ranges(feeding),
            'validation_rows_loaded':0,'scope':'Read-only training range audit, no clipping or model fitting. Reported thresholds are descriptive buckets, not qualification gates.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(diagnose(parser.parse_args().root),indent=2))
