#!/usr/bin/env python3
"""Read-only training slices and input conflicts for the coverage experiment."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from diagnose_interaction_fit import ordinary
from verify_feeding_ecology import read, sha
from verify_interaction_head import LAYOUTS, audit


def diagnose(root):
    audit(root)
    plan = read(root / 'plan.json')
    training = plan['training_seeds']
    corpora = {}
    rows = []
    for seed in training + plan['validation_seeds']:
        directory = Path(plan['corpus_roots'][str(seed)]) / str(seed)
        for name in ['combat'] + [f'feeding-{layout}' for layout in LAYOUTS]:
            corpus = read(directory / f'{name}.json')['rows']
            corpora[seed, name] = corpus
            for index, row in enumerate(corpus):
                label = row['teacher_kind'] if name == 'combat' else row['selected_kind']
                rows.append((row, label, seed, name, index))
    conflicts = {}
    for partition, selected in [('training', set(training)), ('all_exposed', set(training + plan['validation_seeds']))]:
        conflicts[partition] = {}
        for name, key in [
            ('nonrandom_observation', ordinary),
            ('nonrandom_observation_and_memory', lambda r: ordinary(r) + tuple(r['memory'])),
            ('exact_policy_input', lambda r: tuple(r['observation']) + tuple(r['memory']) + tuple(r['legal_kinds'])),
        ]:
            groups = defaultdict(list)
            for row, label, seed, corpus, index in rows:
                if seed in selected:
                    groups[key(row)].append({'seed': seed, 'corpus': corpus, 'row': index, 'label': label})
            ambiguous = [g for g in groups.values() if len({r['label'] for r in g}) > 1]
            conflicts[partition][name] = {
                'conflicting_groups': len(ambiguous),
                'conflicting_rows': sum(map(len, ambiguous)),
                'minimum_disagreements_for_only_these_features': sum(len(g) - max(Counter(r['label'] for r in g).values()) for g in ambiguous),
                'groups': ambiguous,
            }
    slices = []
    for fit in read(root / 'portable-replay.json')['results']:
        if fit['arm'] != 'treatment':
            continue
        result = {'sampling_seed': fit['sampling_seed'], 'training': []}
        for kind in ['combat', 'feeding']:
            predictions = fit['training'][kind]['predictions']
            offset = 0
            for seed in training:
                for name in (['combat'] if kind == 'combat' else [f'feeding-{layout}' for layout in LAYOUTS]):
                    corpus = corpora[seed, name]
                    pred = predictions[offset:offset + len(corpus)]
                    offset += len(corpus)
                    assert len(pred) == len(corpus)
                    errors = Counter((r['teacher_kind'] if kind == 'combat' else r['selected_kind'], k)
                                     for r, k in zip(corpus, pred)
                                     if k != (r['teacher_kind'] if kind == 'combat' else r['selected_kind']))
                    result['training'].append({'seed': seed, 'corpus': name, 'rows': len(corpus),
                                               'errors': sum(errors.values()),
                                               'confusion': [{'label': a, 'predicted': b, 'count': n} for (a, b), n in sorted(errors.items())]})
            assert offset == len(predictions)
        slices.append(result)
    return {'plan_sha256': sha(root / 'plan.json'), 'replay_sha256': sha(root / 'portable-replay.json'),
            'rows': len(rows), 'conflicts': conflicts, 'treatments': slices,
            'scope': 'Post-failure diagnosis only. Observation-only conflicts omit private recurrent memory and frozen parent offsets. Exact-input uniqueness does not establish learnability or gameplay competence. No labels or fits changed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(diagnose(parser.parse_args().root), indent=2))
