"""Summarise p6/p7 runs (clean and paper protocol): mean ± std per config.

Usage: python scripts/p6_summary.py ['results/experiments/p6_*.json' ...]
"""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

pats = sys.argv[1:] or ['results/experiments/p6_*.json', 'results/experiments/p7_*.json']
g = defaultdict(list)
for f in sorted(set(sum((glob.glob(p) for p in pats), []))):
    m = re.search(r'(p\w+?)_seed(\d+)\.json$', os.path.basename(f))
    if m:
        g[m.group(1)].append(json.load(open(f)))
ms = lambda xs: f"{st.mean(xs):.2f} ± {st.stdev(xs) if len(xs) > 1 else 0:.2f}" if xs else '-'
print('| config | n | valid WF1 | test WF1 | test Acc | macro F1 | Happy F1 |\n|---|---|---|---|---|---|---|')
for k, rs in sorted(g.items()):
    v = [r['valid_f1_emo'] for r in rs if 'valid_f1_emo' in r]
    print(f"| {k} | {len(rs)} | {ms(v)} | {ms([r['test_f1_emo'] for r in rs])} | {ms([r['test_acc_emo'] for r in rs])} | "
          f"{ms([100 * st.mean(r['test_per_class_f1']) for r in rs])} | {ms([100 * r['test_per_class_f1'][0] for r in rs])} |")
ex = {k: rs for k, rs in g.items() if all('test_exit_f1' in r for r in rs)}
if ex:
    print('\nTest WF1 per exit (layers given), at the valid-selected epoch:\n')
    for k, rs in sorted(ex.items()):
        cols = zip(*[r['test_exit_f1'] for r in rs])
        print(f"- {k}: " + ', '.join(f"L{l}: {st.mean(c):.2f}" for l, c in zip(rs[0]['exits'], cols)))
