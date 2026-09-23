#!/usr/bin/env python3
"""Audit fixed full-corpus raw/zero-state-preserving normalization fits."""
from audit_checks import equal, require
import hashlib
import math
from pathlib import Path
import struct
from verify_feeding_ecology import read, sha
from verify_interaction_context import preflight as corpus_preflight, verify_fits
from verify_interaction_standardized import preflight as normalizer_preflight, f32


def preflight(root):
    p = read(root/'plan.json')
    baseline = Path(p['baseline_study'])
    source = Path(p['normalizer_study'])
    for base, prefix in [(baseline, 'baseline'), (source, 'normalizer')]:
        equal(sha(base/'plan.json'), p[prefix+'_plan_sha256'], 'zero.source_plan', source=prefix)
        equal(sha(base/'verification.json'), p[prefix+'_verification_sha256'], 'zero.source_verification', source=prefix)
    old, combat, feeding, files = corpus_preflight(baseline)
    extra = {'baseline_study','baseline_plan_sha256','baseline_verification_sha256','normalizer_study','normalizer_plan_sha256','normalizer_verification_sha256','normalization'}
    equal(set(p), set(old)|extra, 'zero.plan_fields')
    for key in old:
        if key not in {'arms','context','scope','decision'}:
            equal(p[key],old[key],'zero.inherited_plan',key=key)
    equal(p['arms'],['memory','zero-state'],'zero.arms')
    normalizer_preflight(source)
    equal(p['normalization'], {'file':'memory-normalizer.json', 'sha256':sha(source/'memory-normalizer.json'),
          'zero_state':'all-zero incoming memory maps to positive zero residual context'}, 'zero.transform')
    for name in ['memory-normalizer.json','seed-audit.json','memory-donors.json']:
        origin = source if name=='memory-normalizer.json' else baseline
        equal(sha(root/name), sha(origin/name), 'zero.input_digest', artifact=name)
    n = read(root/'memory-normalizer.json')
    rows = combat+feeding
    original = hashlib.sha256()
    transformed = hashlib.sha256()
    maxima, cold = [], []
    for row in rows:
        memory = [f32(x) for x in row['memory']]
        values = [0.]*128 if not any(memory) else [f32(f32(x-n['means'][j])/n['scales'][j]) for j,x in enumerate(memory)]
        require(all(math.isfinite(x) for x in values),'zero.nonfinite_context')
        for x in memory: original.update(struct.pack('<f',x))
        for x in values: transformed.update(struct.pack('<f',x))
        maxima.append(max(map(abs,values)))
        if not any(memory): cold.append(maxima[-1])
    audit = {'rows':len(rows),'context_width':128,'zero_memory_rows':len(cold),
             'parent_memory_sha256':original.hexdigest(),'transformed_context_sha256':transformed.hexdigest()}
    ranges = {'complete':True,'plan_sha256':sha(root/'plan.json'),'validation_rows_loaded':0,
              'combat_rows':len(combat),'feeding_rows':len(feeding),'context_audit':audit,
              'max_abs_context':max(maxima),'max_abs_zero_memory_context':max(cold),
              'rows_exceeding_abs_context':{str(t):sum(x>t for x in maxima) for t in [4,8,16,64,1024]},
              'scope':'Pre-fit training-only range audit; buckets are descriptive, not new qualification gates.'}
    return p,combat,feeding,files,ranges


def verify(root):
    p,combat,feeding,files,ranges=preflight(root)
    equal(read(root/'range-audit.json'),ranges,'zero.range_audit')
    equal(read(root/'fits/context-audit.json'),ranges['context_audit'],'zero.context_audit')
    result=verify_fits(root,p,combat,feeding,files)
    equal(result['scalar_burn_kind_mismatches'],0,'zero.scalar_burn_mismatches')
    source=Path(p['baseline_study'])
    old={(r['seed'],r['step']):r for r in read(source/'fits/report.json')['results'] if r['arm']=='memory'}
    raw=[r for r in read(root/'fits/report.json')['results'] if r['arm']=='memory']
    equal(len(raw),9,'zero.raw_coverage')
    for r in raw:
        equal(r,old[r['seed'],r['step']],'zero.raw_record',seed=r['seed'],step=r['step'])
        equal((root/r['weights_file']).read_bytes(),(source/r['weights_file']).read_bytes(),'zero.raw_weights',seed=r['seed'],step=r['step'])
    # Retain count denominators for harness summaries, not rounded percentages.
    fits=read(root/'fits/report.json')['results']
    for summary, fit in zip(result['results'],fits):
        summary['counts']={k:{key:fit['metrics'][k][key] for key in ['rows','correct','attack_labels','attack_correct']} for k in ['combat','feeding']}
    return {**result,'checkpoint_predictions':result['scalar_replayed_predictions'],
            'raw_nine_checkpoints_and_weights_identical':True,'thresholds':p['thresholds'],
            'range_audit':ranges,'scope':'Full training-corpus conditioning diagnostic; no development predictions or deployment qualification.'}


if __name__=='__main__':
    from harness.study import cli
    cli('zero-state',verify,preflight)
