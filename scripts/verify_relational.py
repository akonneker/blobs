#!/usr/bin/env python3
"""Audit relational fits, exact-zero trajectories and bounded WASM qualification."""
import argparse
import json
import struct
from pathlib import Path
from verify_feeding_ecology import read, sha
from verify_interaction_head import audit


def verify(root):
    fits=audit(root);p=read(root/'plan.json');source=Path(p['corpus_source'])
    assert p['intervention']=='relational_residual'
    base=(Path(p['parent'])/'weights.bin').read_bytes()
    zero=root/'fits/zero';m=read(zero/'export.json');weights=(zero/'weights.bin').read_bytes()
    assert sha(zero/'weights.bin')==m['weights_sha256']
    assert weights[:8]==b'BLCMR001' and struct.unpack('<I',weights[8:12])[0]==len(base)
    assert weights[12:12+len(base)]==base and weights[12+len(base):]==bytes(3418*4)
    ca=read(root/'fits/corpus-audit.json');assert ca['zero_migration_rows']==sum(sum(row.values()) for row in fits['corpus_rows'].values())==8375
    assert ca['all_outputs_bit_identical']
    z=root/'zero-migration';zp=read(z/'plan.json');zv=read(z/'verification.json')
    assert zv['complete'] and zp['weights_sha256']==sha(zero/'weights.bin') and zp['manifest_sha256']==sha(zero/'export.json')
    for name,digest in zp['executables'].items():assert sha(z/name)==digest
    for name,digest in zv['reports'].items():assert sha(z/name)==digest
    for seed in p['validation_seeds']:
        assert read(z/str(seed)/'report.json')['evaluation']==read(source/str(seed)/'combat.json')['evaluation']
    before=read(source/'feeding-sentinel/report.json')['trial'];after=read(z/'feeding/report.json')['trial']
    assert before['execution_contract']=='blob.policy.move-utility-scalar-f32-libm-q48-signal-reserved.v1'
    assert after['execution_contract']==p['execution_contract']
    assert {k:v for k,v in before.items() if k!='execution_contract'}=={k:v for k,v in after.items() if k!='execution_contract'}
    w=read(root/'wasm-provenance.json');assert sha(root/'wasm-verifier')==w['verifier_sha256'];assert sha(root/'sources/verify_learned_mind.rs')==w['source_sha256']
    assert w['arms']==['1435600301-control','1435600301-treatment']
    results=[]
    for arm in w['arms']:
        export=root/'fits'/arm;report=read(root/f'wasm-{arm}/verification.json');build=read(export/'mind-build.json')
        assert build['export_manifest_sha256']==sha(export/'export.json')
        assert build['weights_sha256']==report['weights_sha256']==sha(export/'weights.bin')
        assert build['wasm_sha256']==report['wasm_sha256']==sha(export/'mind.wasm')
        assert report['execution_contract']==p['execution_contract']
        assert report['native_wasm_decision_and_memory_mismatches']==0
        assert report['stock_extism_checked_decisions']==report['isolation_checks']==32
        assert report['abi_fixture_decisions']==528 and len(report['cases'])==10
        assert not report['inherited_retention']['applicable']
        results.append({'arm':arm,'decisions':report['decisions'],'abi_fixture_decisions':report['abi_fixture_decisions'],'signed_replay_cases':len(report['cases']),'verification_sha256':sha(root/f'wasm-{arm}/verification.json')})
    return {'complete':True,'offline_gate':fits['all_pass'],'native_burn_validation_predictions':34176,'independent_training_predictions':16074,'zero_output_rows':8375,'zero_combat_episodes':24,'zero_feeding_episodes':2,'wasm_qualification':results,'scope':'No biological promotion. Two fitted Minds qualified on bounded deployment fixtures; other four fits have native evidence only. Zero migration repeats existing development episodes.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(verify(p.parse_args().root),indent=2))
