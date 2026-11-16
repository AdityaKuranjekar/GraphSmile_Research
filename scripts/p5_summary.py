"""Summarise p5 runs: mean +- std of valid WF1 (selection) and test WF1/Acc/Happy per config."""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict
pats = sys.argv[1:] or ['results/experiments/p5_*.json']
g = defaultdict(list)
for f in sorted(sum((glob.glob(p) for p in pats), [])):
    m = re.search(r'(p\w+?)_seed\d+\.json$', os.path.basename(f))
    g[m.group(1)].append(json.load(open(f)))
ms = lambda xs: f"{st.mean(xs):.2f} ± {st.stdev(xs) if len(xs) > 1 else 0:.2f}"
print('| config | n | valid WF1 | test WF1 | test Acc | Happy F1 |\n|---|---|---|---|---|---|')
for k, rs in sorted(g.items()):
    print(f"| {k} | {len(rs)} | {ms([r['valid_f1_emo'] for r in rs])} | {ms([r['test_f1_emo'] for r in rs])} | "
          f"{ms([r['test_acc_emo'] for r in rs])} | {ms([100 * r['test_per_class_f1'][0] for r in rs])} |")
