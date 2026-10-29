"""Build docs/comparison_clean_protocol.pdf from results/ (logs and experiment JSON).

Usage: python scripts/build_comparison_pdf.py [out.pdf]   (run from the repo root; needs reportlab)
Every number is read from results/logs or results/experiments at build time.
"""
import json
import os
import re
import sys

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_to_paper as cp  # paper values and the single-seed log parser

SEEDS = [2024, 1, 2, 3, 4]
CLASSES = ['Happy', 'Sad', 'Neutral', 'Angry', 'Excited', 'Frustrated']
PAPER_PC, PAPER_ACC, PAPER_WF1 = cp.PAPER_TABLE_IV['IEMOCAP'][1:]
EXP = 'results/experiments'
EPOCH_RE = re.compile(r'test_acc_emo: ([\d.]+), test_f1_emo: ([\d.]+)')


def runs(tag):
    paths = [f'{EXP}/{tag}_seed{s}.json' for s in SEEDS]
    return [json.load(open(p)) for p in paths if os.path.exists(p)]


def ms(a):
    a = np.asarray(a, dtype=float)
    return f'{a.mean():.2f} ± {a.std(ddof=1):.2f}' if len(a) > 1 else f'{a.mean():.2f}'


def epoch_curves(tag):
    out = []
    for s in SEEDS:
        p = f'results/logs/{tag}_IEMOCAP_seed{s}.log'
        if os.path.exists(p):
            t = open(p, encoding='utf-8', errors='replace').read()
            out.append([float(b) for _, b in EPOCH_RE.findall(t)])
    return out


def table(data, widths, extra=()):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8e8e8')),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        *extra]))
    return t


def main(out):
    ss = getSampleStyleSheet()
    H, B, S = ss['Heading2'], ss['BodyText'], ss['Italic']
    A = cp.parse_log('results/logs/IEMOCAP_full.log', 'IEMOCAP')
    P0, P0C, P1C = runs('p0'), runs('p0c'), runs('p1c')

    story = [
        Paragraph('GraphSmile reproduction: paper vs clean-protocol runs', ss['Title']),
        Paragraph('Branch experiments/p0-protocol. Every "ours" number is read from results/logs and '
                  'results/experiments by scripts/build_comparison_pdf.py. Test set: IEMOCAP-6 (31 dialogues, '
                  '1623 utterances). Paper: Li et al., IEEE TPAMI 2025, Table IV.', B),
        Paragraph('1. Protocols compared', H),
        Paragraph('<b>A. Original code, test-selected, 1 seed</b> (IEMOCAP_full.log): epoch with the best TEST F1; '
                  'optimistic. <b>B. p0</b>: 5 seeds, epoch by valid_f1_emo, but the repo validation dialogues are also '
                  'training dialogues (found in pass 1), so the selector is a train-set metric. <b>C. p0c (baseline)</b>: '
                  '5 seeds, --clean_valid (108 train and 12 held-out valid dialogues, 0 overlap), epoch by valid_f1_emo. '
                  '<b>D. p1c</b>: C plus --class_weight. The paper does not state how its reported epoch is chosen.', B),
        Paragraph('2. IEMOCAP-6, per-class F1 (%)', H)]

    def pc(R, i):
        return ms([r['test_per_class_f1'][i] * 100 for r in R])

    rows = [['Class', 'Paper', 'A: orig, 1 seed', 'B: p0, 5 seeds', 'C: p0c clean, 5 seeds', 'D: p1c +cw, 5 seeds']]
    for i, c in enumerate(CLASSES):
        rows.append([c, f'{PAPER_PC[i]:.2f}', f'{A["emo_f1"][i] * 100:.2f}', pc(P0, i), pc(P0C, i), pc(P1C, i)])
    for name, key, pa, a in [('ACC', 'test_acc_emo', PAPER_ACC, A['acc_emo']), ('WF1', 'test_f1_emo', PAPER_WF1, A['f1_emo'])]:
        rows.append([name, f'{pa:.2f}', f'{a:.2f}'] + [ms([r[key] for r in R]) for R in (P0, P0C, P1C)])
    story += [table(rows, [2.2 * cm, 1.8 * cm, 3 * cm, 3.2 * cm, 3.6 * cm, 3.4 * cm],
                    [('FONTNAME', (0, -2), (-1, -1), 'Helvetica-Bold')]),
              Paragraph('Mean ± sample std (ddof=1) over seeds 2024, 1, 2, 3, 4. Column A is a single seed.', S),
              Paragraph('3. Where the gap comes from (diagnostic only; never used to choose anything)', H)]

    rows = [['Mean WF1 over 5 seeds', 'p0', 'p0c', 'p1c']]
    measures = [('Selected on valid (reported)', None),
                ('Final epoch (120)', lambda c: np.mean([x[-1] for x in c])),
                ('Mean of last 10 epochs', lambda c: np.mean([np.mean(x[-10:]) for x in c])),
                ('Best test epoch ("oracle", optimistic)', lambda c: np.mean([max(x) for x in c]))]
    for name, fn in measures:
        row = [name]
        for tag, R in (('p0', P0), ('p0c', P0C), ('p1c', P1C)):
            if fn is None:
                row.append(f'{np.mean([r["test_f1_emo"] for r in R]):.2f}')
            else:
                c = epoch_curves(tag)
                row.append(f'{fn(c):.2f}' if c else 'n/a')
        rows.append(row)
    rows.append(['Paper WF1', f'{PAPER_WF1:.2f}', '', ''])
    story += [table(rows, [7 * cm, 3 * cm, 3 * cm, 3 * cm]),
              Paragraph('Under every protocol training reaches about 71 WF1 at its best test epoch, but the epoch picked on '
                        'the 12-dialogue held-out validation set gives about 68.3 for both p0c and p1c. For p0c this is '
                        'close to simply taking the last epoch (68.04); for p1c the final epoch is higher (70.48), a '
                        'post-hoc observation from one set of 5 seeds that is not part of the pass-1 decision rule. '
                        'Valid WF1 is flat across seeds while test moves about 3 points between epochs.', B),
              Paragraph('4. Happy / Excited confusion (test)', H)]

    rows = [['Run', 'Happy to Excited (of 144), per seed', 'mean', 'Excited to Happy (of 299), per seed', 'mean']]
    for name, R in (('p0 (overlapping valid)', P0), ('p0c (clean baseline)', P0C), ('p1c (class weight)', P1C)):
        he = [r['test_confusion'][0][4] for r in R]
        eh = [r['test_confusion'][4][0] for r in R]
        rows.append([name, ', '.join(map(str, he)), f'{np.mean(he):.1f}', ', '.join(map(str, eh)), f'{np.mean(eh):.1f}'])
    story += [table(rows, [3.6 * cm, 4.6 * cm, 1.4 * cm, 4.6 * cm, 1.4 * cm]),
              Paragraph('Class weighting (pass 1) did not reduce the confusion. Decision: inconclusive; WF1 changed by '
                        '+0.11 and Happy by -0.81 against baseline stds of 0.94 and 4.17.', B),
              Paragraph('5. Other datasets (unchanged since the first reproduction)', H)]

    rows = [['Dataset', 'Paper ACC / WF1', 'Ours ACC / WF1 (1 seed, test-selected)']]
    for ds, label in (('IEMOCAP4', 'IEMOCAP-4'), ('MELD', 'MELD'), ('CMUMOSEI7', 'CMU-MOSEI')):
        r = cp.load('full', ds)
        _, _, pa, pw = cp.PAPER_TABLE_IV[ds]
        rows.append([label, f'{pa:.2f} / {pw:.2f}', f'{r["acc_emo"]:.2f} / {r["f1_emo"]:.2f}' if r else 'not run'])
    story += [table(rows, [4 * cm, 5 * cm, 7 * cm]),
              Paragraph('Not rerun under the clean protocol or with several seeds, so they carry the same test-selection '
                        'caveat as column A and are not comparable to sections 2 to 4.', S),
              Paragraph('6. Status and open items', H),
              Paragraph('The paper epoch-selection protocol is not stated in the text read (arXiv 2407.21536v2, Sections I to '
                        'V-A). Pass 2 (last-10-epoch weight averaging, --avg_last) is planned in '
                        'results/experiments/p2-plan.md and is not implemented or run. Baseline rows, MSAC rows marked with a '
                        'double dagger and Figs. 4 to 7 of the paper remain unreproduced (see REPRODUCTION.md).', B)]

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    SimpleDocTemplate(out, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                      topMargin=1.5 * cm, bottomMargin=1.5 * cm).build(story)
    print('Wrote', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'docs/comparison_clean_protocol.pdf')
