"""Summarize results/logs/*.log into a markdown table.

The emotion numbers follow the repo's protocol: the best test epoch by emotion F1
(run.py prints "Acc: <max acc>, F-Score: <max f1>" at the end). The shift (MSAC)
numbers are taken from the same epoch as the best emotion F1.
"""
import glob
import os
import re

LOG_DIR = 'results/logs'
EPOCH_RE = re.compile(
    r'test_acc_emo: ([\d.]+), test_f1_emo: ([\d.]+), .*?'
    r'test_acc_sft: ([\d.]+), test_f1_sft: ([\d.]+)')
FINAL_RE = re.compile(r'^Acc: ([\d.]+), F-Score: ([\d.]+)', re.M)


def parse(path):
    text = open(path, encoding='utf-8', errors='replace').read()
    epochs = [tuple(map(float, m)) for m in EPOCH_RE.findall(text)]
    final = FINAL_RE.search(text)
    if not epochs or not final:
        return None
    best = max(epochs, key=lambda e: e[1])  # best test emotion F1
    return {
        'emo_acc': float(final.group(1)),
        'emo_f1': float(final.group(2)),
        'sft_acc': best[2],
        'sft_f1': best[3],
    }


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(LOG_DIR, '*.log'))):
        name = os.path.basename(path)[:-4]
        ds, _, tag = name.partition('_')
        res = parse(path)
        rows.append((ds, tag, res))

    print('| Dataset | Variant | Emo ACC | Emo WF1 | Shift ACC | Shift WF1 |')
    print('|---|---|---|---|---|---|')
    for ds, tag, res in rows:
        if res is None:
            print(f'| {ds} | {tag} | not finished | | | |')
        else:
            print(f"| {ds} | {tag} | {res['emo_acc']:.2f} | {res['emo_f1']:.2f} "
                  f"| {res['sft_acc']:.2f} | {res['sft_f1']:.2f} |")


if __name__ == '__main__':
    main()
