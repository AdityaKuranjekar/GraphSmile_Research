"""Compare results/logs/*.log with the numbers reported in the GraphSmile paper.

Protocol (matches run.py): the reported epoch is the one with the best test emotion F1.
Per-class F1 comes from the final classification report, which run.py prints for that epoch.
Sentiment (MSAC) numbers are the sentiment head (test_acc_sen / test_f1_sen) at that epoch.
Output is a markdown report on stdout:  python scripts/compare_to_paper.py > results/comparison.md
"""
import glob
import os
import re

LOG_DIR = 'results/logs'

# Paper values in %. Per-class lists follow the column order of the paper's tables.
PAPER_TABLE_IV = {  # GraphSmile row, MERC task
    'IEMOCAP': (['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated'],
                [63.09, 83.16, 71.07, 71.38, 79.66, 66.84], 72.77, 72.81),
    'IEMOCAP4': (['Happy', 'Sad', 'Neutral', 'Angry'],
                 [87.94, 88.67, 85.41, 84.73], 86.53, 86.52),
    'MELD': (['Neutral', 'Surprise', 'Fear', 'Sadness', 'Joy', 'Disgust', 'Anger'],
             [80.35, 59.11, 18.18, 42.46, 64.99, 32.43, 53.67], 67.70, 66.71),
    'CMUMOSEI7': (['HNeg', 'Neg', 'WNeg', 'Neu', 'WPos', 'Pos', 'HPos'],
                  [0.00, 28.79, 43.50, 39.19, 57.30, 35.95, 6.25], 46.82, 44.93),
}

# Table V, GraphSmile rows: (neg, neu, pos, acc, wf1). 'const' = Table III settings (†),
# 'opt' = hyperparameters optimized for MSAC (‡).
PAPER_TABLE_V = {
    'IEMOCAP': {'const': [86.87, 69.06, 89.21, 83.73, 83.66],
                'opt': [88.28, 72.89, 89.93, 84.97, 85.09]},
    'IEMOCAP4': {'const': [87.76, 82.93, 87.23, 85.79, 85.71],
                 'opt': [89.45, 85.53, 87.02, 87.49, 87.48]},
    'MELD': {'const': [79.15, 64.84, 70.08, 73.60, 73.40],  # columns: neu, pos, neg
             'opt': [79.35, 67.89, 70.72, 74.44, 74.31]},
    'CMUMOSEI7': {'const': [56.14, 39.80, 78.18, 66.83, 65.69],
                  'opt': [56.99, 41.02, 79.36, 67.73, 66.73]},
}

# Table VI: (MERC acc, MERC wf1, MSAC acc, MSAC wf1) per dataset, in the order
# IEMOCAP-6, IEMOCAP-4, MELD, CMU-MOSEI.
PAPER_TABLE_VI = {
    'full': [(72.77, 72.81, 84.97, 85.09), (86.53, 86.52, 87.49, 87.48),
             (67.70, 66.71, 74.44, 74.31), (46.82, 44.93, 67.73, 66.73)],
    'no_Ls': [(71.53, 71.52, 84.41, 84.53), (85.15, 85.15, 85.47, 85.40),
              (67.39, 66.30, 73.98, 73.84), (45.08, 43.37, 66.59, 65.91)],
    'no_Lo': [(71.41, 71.42, 84.10, 84.18), (84.52, 84.50, 85.26, 85.23),
              (67.62, 66.59, 74.18, 74.10), (45.20, 44.03, 66.71, 66.20)],
    'no_Ls_Lo': [(70.67, 70.49, 83.30, 83.54), (84.41, 84.40, 85.58, 85.55),
                 (67.55, 66.43, 73.91, 73.77), (44.29, 43.54, 66.54, 65.64)],
    'no_text': [(55.70, 54.23, 71.04, 70.42), (62.88, 62.22, 66.28, 60.76),
                (50.15, 42.27, 52.72, 52.10), (32.98, 28.28, 54.95, 47.80)],
    'no_visual': [(71.29, 71.23, 84.53, 84.69), (86.21, 86.20, 85.58, 85.57),
                  (67.39, 66.26, 73.83, 73.75), (45.32, 43.57, 66.30, 65.81)],
    'no_acoustic': [(68.45, 68.26, 83.24, 82.87), (83.35, 83.28, 84.31, 84.25),
                    (67.55, 66.40, 73.72, 73.62), (46.56, 44.57, 66.47, 65.57)],
}

# Table VII, GraphSmile row: (seconds per epoch, memory MB) per dataset.
PAPER_TABLE_VII = {
    'IEMOCAP': (1.12, 1840.89), 'IEMOCAP4': (0.67, 509.88),
    'MELD': (5.21, 513.33), 'CMUMOSEI7': (7.38, 467.13),
}

N_EMO_CLASSES = {'IEMOCAP': 6, 'IEMOCAP4': 4, 'MELD': 7, 'CMUMOSEI7': 7}

EPOCH_RE = re.compile(
    r'test_acc_emo: ([\d.]+), test_f1_emo: ([\d.]+), test_acc_sen: ([\d.]+), '
    r'test_f1_sen: ([\d.]+), test_acc_sft: [\d.]+, test_f1_sft: [\d.]+, '
    r'total time: ([\d.]+) sec')
ROW_RE = re.compile(r'^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+\s*$', re.M)
MEM_RE = re.compile(r'All Memory: ([\d.]+) MB')


def parse_log(path, ds):
    text = open(path, encoding='utf-8', errors='replace').read()
    epochs = [tuple(map(float, m)) for m in EPOCH_RE.findall(text)]
    if not epochs:
        return None
    # tuple: (acc_emo, f1_emo, acc_sen, f1_sen, seconds)
    best = max(epochs, key=lambda e: e[1])  # first best, same as run.py
    mem = MEM_RE.search(text)
    tail = text.split('Test performance..', 1)[-1]
    rows = [float(f) for _, f in ROW_RE.findall(tail)]
    n_emo = N_EMO_CLASSES[ds]
    emo_f1 = rows[:n_emo] if len(rows) >= n_emo else []
    sen_f1 = rows[n_emo:n_emo + 3] if len(rows) >= n_emo + 3 else []
    return {
        'acc_emo': best[0], 'f1_emo': best[1],
        'acc_sen': best[2], 'f1_sen': best[3],
        'emo_f1': emo_f1, 'sen_f1': sen_f1,
        'sec_per_epoch': sum(e[4] for e in epochs) / len(epochs),
        'mem_mb': float(mem.group(1)) if mem else None,
    }


def load(tag, ds):
    path = os.path.join(LOG_DIR, f'{ds}_{tag}.log')
    if not os.path.exists(path):
        return None
    return parse_log(path, ds)


def fmt(x, digits=2):
    return 'n/a' if x is None else f'{x:.{digits}f}'


def delta(ours, paper):
    return 'n/a' if ours is None else f'{ours - paper:+.2f}'


def table_iv():
    print('## Table IV: MERC (emotion), GraphSmile\n')
    for ds, (classes, paper_pc, paper_acc, paper_wf1) in PAPER_TABLE_IV.items():
        r = load('full', ds)
        print(f'### {ds}\n')
        print('| Class | Paper | Ours | Δ |')
        print('|---|---|---|---|')
        for i, name in enumerate(classes):
            ours = r['emo_f1'][i] * 100 if r and len(r['emo_f1']) > i else None
            print(f'| {name} | {paper_pc[i]:.2f} | {fmt(ours)} | {delta(ours, paper_pc[i])} |')
        ours_acc = r['acc_emo'] if r else None
        ours_wf1 = r['f1_emo'] if r else None
        print(f'| ACC | {paper_acc:.2f} | {fmt(ours_acc)} | {delta(ours_acc, paper_acc)} |')
        print(f'| WF1 | {paper_wf1:.2f} | {fmt(ours_wf1)} | {delta(ours_wf1, paper_wf1)} |\n')


def table_v():
    print('## Table V: MSAC (sentiment), GraphSmile with Table III settings (†)\n')
    print('| Dataset | Class | Paper | Ours | Δ |')
    print('|---|---|---|---|---|')
    for ds, vals in PAPER_TABLE_V.items():
        r = load('full', ds)
        if ds == 'MELD':
            # Paper columns are neu, pos, neg. Index mapping assumes 0=neg, 1=neu, 2=pos.
            order = {'neu': 1, 'pos': 2, 'neg': 0}
            pc = dict(zip(['neu', 'pos', 'neg'], vals['const'][:3]))
        else:
            order = {'neg': 0, 'neu': 1, 'pos': 2}
            pc = dict(zip(['neg', 'neu', 'pos'], vals['const'][:3]))
        for name in ['neg', 'neu', 'pos']:
            ours = r['sen_f1'][order[name]] * 100 if r and len(r['sen_f1']) == 3 else None
            print(f'| {ds} | {name} | {pc[name]:.2f} | {fmt(ours)} | {delta(ours, pc[name])} |')
        paper_acc, paper_wf1 = vals['const'][3], vals['const'][4]
        ours_acc = r['acc_sen'] if r else None
        ours_wf1 = r['f1_sen'] if r else None
        print(f'| {ds} | ACC | {paper_acc:.2f} | {fmt(ours_acc)} | {delta(ours_acc, paper_acc)} |')
        print(f'| {ds} | WF1 | {paper_wf1:.2f} | {fmt(ours_wf1)} | {delta(ours_wf1, paper_wf1)} |')
    print()


def table_vi():
    print('## Table VI: ablations (ACC / WF1)\n')
    print('MERC columns use emotion ACC/WF1; MSAC columns use sentiment ACC/WF1.\n')
    headers = ['IEMOCAP-6', 'IEMOCAP-4', 'MELD', 'CMU-MOSEI']
    print('| Variant | Dataset | Paper MERC | Ours MERC | Paper MSAC | Ours MSAC |')
    print('|---|---|---|---|---|---|')
    for tag, rows in PAPER_TABLE_VI.items():
        for ds, h, p in zip(PAPER_TABLE_IV.keys(), headers, rows):
            r = load(tag, ds)
            ours_m = f"{fmt(r['acc_emo'])} / {fmt(r['f1_emo'])}" if r else 'not run'
            ours_s = f"{fmt(r['acc_sen'])} / {fmt(r['f1_sen'])}" if r else 'not run'
            print(f'| {tag} | {h} | {p[0]:.2f} / {p[1]:.2f} | {ours_m} | '
                  f'{p[2]:.2f} / {p[3]:.2f} | {ours_s} |')
    print('\nNot run: No Res, No FC-Res, No Seg (the code has no switch for these yet).\n')


def table_vii():
    print('## Table VII: time and memory (GraphSmile)\n')
    print('Paper: RTX 3090. Ours: whatever GPU the job ran on. Time includes test evaluation.\n')
    print('| Dataset | Paper s/epoch | Ours s/epoch | Paper MB | Ours MB |')
    print('|---|---|---|---|---|')
    for ds, (p_sec, p_mem) in PAPER_TABLE_VII.items():
        r = load('full', ds)
        ours_sec = r['sec_per_epoch'] if r else None
        ours_mem = r['mem_mb'] if r else None
        print(f'| {ds} | {p_sec:.2f} | {fmt(ours_sec)} | {p_mem:.2f} | {fmt(ours_mem)} |')
    print()


def extra_runs():
    print('## Extra runs (not in the paper tables)\n')
    print('| Tag | Dataset | Ours ACC | Ours WF1 | Note |')
    print('|---|---|---|---|---|')
    for path in sorted(glob.glob(os.path.join(LOG_DIR, 'IEMOCAP_*.log'))):
        tag = os.path.basename(path)[len('IEMOCAP_'):-4]
        if tag in ('full', 'no_text', 'no_visual', 'no_acoustic', 'no_Ls', 'no_Lo', 'no_Ls_Lo'):
            continue
        r = parse_log(path, 'IEMOCAP')
        if r is None:
            print(f'| {tag} | IEMOCAP | not finished | | |')
        else:
            print(f'| {tag} | IEMOCAP | {r["acc_emo"]:.2f} | {r["f1_emo"]:.2f} | |')
    print()


def main():
    print('# GraphSmile: our results vs the paper\n')
    table_iv()
    table_v()
    table_vi()
    table_vii()
    extra_runs()


if __name__ == '__main__':
    main()
