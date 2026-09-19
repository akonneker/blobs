#!/usr/bin/env python3
"""Describe training-only private-memory variation and conflicting observations."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from diagnose_interaction_fit import ordinary
from verify_interaction_context import preflight
from verify_feeding_ecology import read, sha


def diagnose(root):
    p, combat, feeding, _ = preflight(root); rows = combat + feeding
    groups = defaultdict(list)
    for i,r in enumerate(rows):
        groups[ordinary(r)].append((i,r['teacher_kind'] if i < len(combat) else r['selected_kind']))
    conflicts = []
    for items in groups.values():
        if len({label for _,label in items}) < 2: continue
        pairs = []
        for a,(i,li) in enumerate(items):
            for j,lj in items[a+1:]:
                if li == lj: continue
                diffs = [abs(x-y) for x,y in zip(rows[i]['memory'],rows[j]['memory'])]
                pairs.append({'rows':[i,j], 'labels':[li,lj], 'max_memory_difference':max(diffs),
                              'rms_memory_difference':math.sqrt(sum(d*d for d in diffs)/128)})
        conflicts.append(pairs)
    means = [sum(r['memory'][j] for r in rows)/len(rows) for j in range(128)]
    std = [math.sqrt(sum((r['memory'][j]-means[j])**2 for r in rows)/len(rows)) for j in range(128)]
    source = Path(p['source_study'])
    assert sha(source/'feeding-pool.json') == read(source/'plan.json')['feeding_sampling']['pool_sha256']
    hard = read(source/'feeding-pool.json')['hard_indices']
    return {'plan_sha256':sha(root/'plan.json'), 'training_rows':len(rows),
            'zero_memory_rows':sum(not any(r['memory']) for r in rows),
            'hard_feeding_rows':len(hard),'hard_feeding_zero_memory_rows':sum(not any(feeding[i]['memory']) for i in hard),
            'channel_standard_deviation':std, 'constant_channels':sum(x == 0 for x in std),
            'saturated_component_fraction':sum(abs(x)>=0.999 for r in rows for x in r['memory'])/(len(rows)*128),
            'conflicting_observation_groups':conflicts,
            'scope':'Training-only memory statistics; exact distinguishability does not imply useful generalization.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(diagnose(parser.parse_args().root),indent=2))
