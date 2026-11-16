"""Summarise p5 runs: mean +- std of valid WF1 (selection) and test WF1/Acc/Happy per config."""
import glob, json, re, statistics as st, sys
from collections import defaultdict
pat = sys.argv[1] if len(sys.argv) > 1 else 'results/experiments/p5_*.json'
g = defaultdict(list)
for f in sorted(glob.glob(pat)):
    m = re.search(r'(p5_.+)_seed\d+\.json$', f.replace('\', '/'))
    g[m.group(1)].append(json.load(open(f)))
ms = lambda xs: f"{st.mean(xs):.2f} ± {st.stdev(xs) if len(xs) > 1 else 0:.2f}"
print('| config | n | valid WF1 | test WF1 | test Acc | Happy F1 |\n|---|---|---|---|---|---|')
for k, rs in sorted(g.items()):
    print(f"| {k} | {len(rs)} | {ms([r['valid_f1_emo'] for r in rs])} | {ms([r['test_f1_emo'] for r in rs])} | "
          f"{ms([r['test_acc_emo'] for r in rs])} | {ms([100 * r['test_per_class_f1'][0] for r in rs])} |")
