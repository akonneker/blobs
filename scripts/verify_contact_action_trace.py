#!/usr/bin/env python3
"""Recompute routed raw-score margins and verify unchanged combat evaluations."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from verify_feeding_ecology import read, sha
from verify_contact_deployed import verify


def audit(root):
    plan = read(root / 'plan.json')
    original = Path(plan['combat_root'])
    assert sha(original / 'plan.json') == plan['combat_plan_sha256']
    verify(original)
    assert sha(root / 'evaluator') == plan['evaluator_sha256']
    for name, digest in plan['sources'].items():
        assert sha(root / 'sources' / name) == digest
    assert plan['arms'] == read(original / 'plan.json')['arms']
    assert plan['seeds'] == [1435500201, 1435500202]
    result = {}
    for arm, identity in plan['arms'].items():
        report = read(root / arm / 'report.json')
        p = read(root / arm / 'plan.json')
        assert report['complete'] and report['plan'] == p
        for field in ['manifest_sha256', 'weights_sha256']:
            assert p[field] == identity[field]
        assert p['seeds'] == plan['seeds']
        assert p['source_config_sha256'] == sha(Path(plan['config']))
        old = read(original / 'trials' / arm / 'report.json')['evaluation']
        assert report['evaluation'] == old
        rows = report['rows']
        assert len(rows) == sum(e['decisions']['total'] for e in old['episodes'])
        for index, episode in enumerate(old['episodes']):
            selected = [r for r in rows if r['episode'] == index]
            assert len(selected) == episode['decisions']['total']
            counts = Counter(r['selected_kind'] for r in selected)
            names = ['Wait','Guard','Consume','Move','Attack','Split','Regurgitate','Excavate','DepositTerrain','Signal']
            assert {names[k]:v for k,v in counts.items()} == episode['decisions']['action_kinds']
        for r in rows:
            assert 0 <= r['episode'] < 24
            assert len(r['observation']) == 1126 and len(r['hidden']) == len(r['memory']) == 128
            for field in ['raw_kind','base_kind','context_delta','slot_delta']:
                assert len(r[field]) == 10 and all(math.isfinite(x) for x in r[field])
            assert len(r['legal_kinds']) == 10 and r['legal_kinds'][r['selected_kind']]
            assert r['legal_kinds'][r['teacher_kind']]
            for k in range(10):
                assert abs(r['raw_kind'][k]-sum(r[f][k] for f in ['base_kind','context_delta','slot_delta'])) < 1e-4
            if r['legal_occupied_attack']:
                assert r['context'] == 'Interaction' and r['legal_kinds'][4]
        slices = {}
        for name, selected in [('all',rows),('occupied',[r for r in rows if r['legal_occupied_attack']]),('teacher_attack',[r for r in rows if r['teacher_kind']==4])]:
            slices[name] = {'rows':len(selected), 'contexts':dict(Counter(r['context'] for r in selected)), 'teacher_kinds':dict(Counter(r['teacher_kind'] for r in selected)), 'move_minus_attack':{}}
            for field in ['base_kind','context_delta','slot_delta','raw_kind']:
                values = [r[field][3]-r[field][4] for r in selected]
                slices[name]['move_minus_attack'][field] = {'min':min(values),'mean':mean(values),'max':max(values)}
        result[arm] = {'report_sha256':sha(root / arm / 'report.json'), 'slices':slices}
    assert read(root / 'status.json')['complete']
    return {'complete':True,'unchanged_combat_episodes':168,'training_performed':False,'arms':result}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(audit(parser.parse_args().root),indent=2))
