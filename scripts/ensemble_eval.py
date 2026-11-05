#!/usr/bin/env python3
"""Average emotion probabilities over seeds, aligned by utterance key.

Each run saved with --save_probs holds, for its validation-selected epoch, the probabilities,
labels and keys ("<dialogue id>_<utterance index>") for the validation and test splits.
Rows are matched by key before averaging, so different per-seed row orders are handled.

usage: python3 scripts/ensemble_eval.py results/probs/ens_seed*.npz
Also needs results/experiments/<same name>.json to check each seed against its reported numbers.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score


def aligned(files, split):
    """Return keys, labels, and a (n_seeds, n_keys, classes) probability stack over the shared keys."""
    runs = []
    for f in files:
        d = np.load(f, allow_pickle=False)
        keys = [str(k) for k in d[f'{split}_keys']]
        runs.append((dict(zip(keys, range(len(keys)))), d[f'{split}_probs'], d[f'{split}_labels'], f))
    shared = set(runs[0][0])
    for idx, _, _, _ in runs[1:]:
        shared &= set(idx)
    order = sorted(shared)
    stack, labels = [], None
    for idx, probs, labs, f in runs:
        stack.append(probs[[idx[k] for k in order]])
        lab = labs[[idx[k] for k in order]]
        if labels is None:
            labels = lab
        elif not np.array_equal(labels, lab):
            raise SystemExit('label mismatch across seeds in %s split (%s)' % (split, f))
    return order, labels, np.stack(stack)


def report(name, labels, probs):
    pred = probs.argmax(1)
    acc = 100 * (pred == labels).mean()
    wf1 = 100 * f1_score(labels, pred, average='weighted')
    print('%-22s ACC %.2f  WF1 %.2f  (n=%d)' % (name, acc, wf1, len(labels)))


def main(files):
    files = sorted(files)
    _, t_labels, t_stack = aligned(files, 'test')
    for i, f in enumerate(files):
        j = json.load(open(os.path.join('results', 'experiments', os.path.basename(f).replace('.npz', '.json'))))
        report('seed %s' % j['seed'], t_labels, t_stack[i])
        print('   reported: ACC %.2f WF1 %.2f' % (j['test_acc_emo'], j['test_f1_emo']))
    report('ENSEMBLE test', t_labels, t_stack.mean(0))
    _, v_labels, v_stack = aligned(files, 'valid')
    report('ENSEMBLE valid', v_labels, v_stack.mean(0))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
