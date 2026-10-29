"""Build docs/paper_vs_reproduced_updated.pdf: the paper-style comparison with earlier and new rows.

Usage: python scripts/build_paper_vs_reproduced_pdf.py [out.pdf]   (run from the repo root; needs reportlab)

Rows per table: paper, ours (earlier = original single-seed, test-selected logs), ours (new = clean-protocol
5-seed runs, where they exist). Where nothing was rerun the new cell says "not rerun". All numbers are read
from results/logs and results/experiments at build time.
"""
import os
import sys

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_to_paper as cp  # noqa: E402
from build_comparison_pdf import epoch_curves, runs  # noqa: E402

NR = 'not rerun'
GREY, GREEN = colors.HexColor('#e8e8e8'), colors.HexColor('#e6f2e6')


def f2(x):
    return 'n/a' if x is None else f'{x:.2f}'


def dlt(a, b):
    return 'n/a' if a is None or b is None else f'{a - b:+.2f}'


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return f'{a.mean():.2f}±{a.std(ddof=1):.2f}'


def tbl(data, widths, new_rows=(), bold_rows=(), fs=7):
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [('FONTSIZE', (0, 0), (-1, -1), fs), ('BACKGROUND', (0, 0), (-1, 0), GREY),
             ('GRID', (0, 0), (-1, -1), 0.3, colors.grey), ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
             ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 2)]
    style += [('BACKGROUND', (0, i), (-1, i), GREEN) for i in new_rows]
    style += [('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold') for i in bold_rows]
    t.setStyle(TableStyle(style))
    return t


def main(out):
    ss = getSampleStyleSheet()
    title, body, small = ss['Title'], ParagraphStyle('b', parent=ss['BodyText'], fontSize=8.5, leading=11), \
        ParagraphStyle('s', parent=ss['BodyText'], fontSize=7, leading=9, textColor=colors.HexColor('#444444'))
    th = ParagraphStyle('th', parent=ss['Heading3'], alignment=1, spaceBefore=8, spaceAfter=0, fontSize=10)
    sub = ParagraphStyle('sub', parent=ss['Italic'], fontSize=7.5, leading=9, alignment=1, spaceAfter=3)
    hd = ParagraphStyle('hd', parent=ss['Heading4'], spaceBefore=4, spaceAfter=2, fontSize=8.5)

    P0C, P1C = runs('p0c'), runs('p1c')
    curves = epoch_curves('p0c')
    n = len(P0C)
    new_acc = ms([r['test_acc_emo'] for r in P0C])
    new_wf1 = ms([r['test_f1_emo'] for r in P0C])
    new_acc_m = np.mean([r['test_acc_emo'] for r in P0C])
    new_wf1_m = np.mean([r['test_f1_emo'] for r in P0C])
    new_pc = np.array([r['test_per_class_f1'] for r in P0C]) * 100

    story = [
        Paragraph('GraphSmile: paper tables and reproduced results', title),
        Paragraph('Updated comparison: paper vs ours (earlier) vs ours (new). Tracing intricate cues in dialogue '
                  '(IEEE TPAMI, 2025). Branch experiments/p0-protocol.', sub),
        Paragraph('<b>Earlier</b> = our first reproduction: one run (seed 2024), epoch chosen by best TEST F1, as the '
                  'original code does. <b>New</b> = clean protocol: 5 seeds (2024, 1, 2, 3, 4), the validation split is '
                  'held out (108 train / 12 valid dialogues, 0 overlap; the original code overlaps them), and the epoch is '
                  'chosen on validation F1 only. New values are mean ± sample std over seeds. New rows are shaded '
                  'green. They exist only for IEMOCAP-6 emotion (MERC); every other cell is marked "not rerun" and keeps '
                  'its earlier value. Δ is ours minus paper.', body),
        Paragraph('What changed and why it matters: the earlier numbers were optimistic because the epoch was picked on the test '
                  'set. The new numbers never use test for any choice, so they are lower, and they are the ones to trust. The '
                  'paper does not state how its reported epoch was chosen (arXiv 2407.21536v2, text read: Sections I to V-A).',
                  body),
        Spacer(1, 4)]

    # ---- Tables I to III (static, from the paper) ----
    story += [Paragraph('TABLE I: MERGING SCHEME OF EMOTION LABELS', th),
              Paragraph('Maps each dataset\'s emotion labels to negative, neutral and positive sentiment. Source: paper, Table I.', sub),
              tbl([['Sentiment', 'IEMOCAP-6', 'IEMOCAP-4', 'MELD', 'CMU-MOSEI'],
                   ['Negative', 'Sad, Angry, Frustrated', 'Sad, Angry', 'Negative', 'Highly Neg., Negative, Weakly Neg.'],
                   ['Neutral', 'Neutral', 'Neutral', 'Neutral', 'Neutral'],
                   ['Positive', 'Happy, Excited', 'Happy', 'Positive', 'Weakly Pos., Positive, Highly Pos.']],
                  [2 * cm, 4 * cm, 2.8 * cm, 2.4 * cm, 6 * cm]),
              Paragraph('TABLE II: STATISTICAL INFORMATION OF ALL DATASETS', th),
              Paragraph('Source: paper, Table II. Describes the data; not a result. 10% of the train+val dialogues form '
                        'the validation set.', sub),
              tbl([['Dataset', 'Conv. train+val', 'Conv. test', 'Utt. train+val', 'Utt. test'],
                   ['IEMOCAP-6', '120', '31', '5,810', '1,623'], ['IEMOCAP-4', '120', '31', '3,600', '943'],
                   ['MELD', '1,153', '280', '11,098', '2,610'], ['CMU-MOSEI', '2,549', '646', '18,198', '4,662']],
                  [3 * cm, 3.4 * cm, 3 * cm, 3.4 * cm, 3 * cm]),
              Paragraph('TABLE III: DESCRIPTION OF PARTIAL HYPERPARAMETERS', th),
              Paragraph('Source: paper, Table III. L2 = 1e-3 for all datasets; all our runs use these values.', sub),
              tbl([['Dataset', 'LR', 'BS', 'DR', 'L', 'P', 'B', 'λs', 'λo'],
                   ['IEMOCAP-6', '1e-4', '16', '0.2', '7', '17', '19', '1.0', '0.7'],
                   ['IEMOCAP-4', '3e-4', '16', '0.2', '4', '5', '10', '0.6', '0.6'],
                   ['MELD', '7e-5', '16', '0.2', '5', '3', '3', '0.5', '0.2'],
                   ['CMU-MOSEI', '8e-5', '32', '0.4', '2', '5', '2', '0.8', '1.0']],
                  [3 * cm] + [1.8 * cm] * 8),
              PageBreak()]

    # ---- Table IV ----
    story += [Paragraph('TABLE IV: MAIN RESULTS FOR THE MERC TASK ON ALL DATASETS', th),
              Paragraph('Emotion recognition, per-class F1 with ACC and weighted F1 (%). GraphSmile rows only; the paper\'s baseline '
                        'rows were not re-run.', sub)]
    for ds, label in (('IEMOCAP', 'IEMOCAP-6'), ('IEMOCAP4', 'IEMOCAP-4'), ('MELD', 'MELD'), ('CMUMOSEI7', 'CMU-MOSEI')):
        classes, ppc, pacc, pwf1 = cp.PAPER_TABLE_IV[ds]
        r = cp.load('full', ds)
        k = len(classes)
        hdr = ['Method'] + classes + ['ACC', 'WF1']
        ours = [r['emo_f1'][i] * 100 if r and len(r['emo_f1']) > i else None for i in range(k)]
        rows = [hdr, ['GraphSmile (paper)'] + [f2(v) for v in ppc] + [f2(pacc), f2(pwf1)],
                ['Ours (earlier)'] + [f2(v) for v in ours] + [f2(r['acc_emo']), f2(r['f1_emo'])],
                ['Δ earlier'] + [dlt(a, b) for a, b in zip(ours, ppc)] + [dlt(r['acc_emo'], pacc), dlt(r['f1_emo'], pwf1)]]
        new_idx, bold = [], []
        if ds == 'IEMOCAP':
            rows.append(['Ours (new, 5 seeds)'] + [f'{m:.2f}±{s:.2f}' for m, s in zip(new_pc.mean(0), new_pc.std(0, ddof=1))]
                        + [new_acc, new_wf1])
            rows.append(['Δ new (means)'] + [dlt(m, b) for m, b in zip(new_pc.mean(0), ppc)]
                        + [dlt(new_acc_m, pacc), dlt(new_wf1_m, pwf1)])
            new_idx, bold = [4, 5], [4]
            widths = [3.1 * cm] + [1.55 * cm] * 6 + [1.5 * cm] * 2
        else:
            rows.append(['Ours (new)'] + [NR] + [''] * (k + 1))
            widths = [3.1 * cm] + [1.3 * cm] * k + [1.2 * cm] * 2
            if k == 4:
                widths = [3.1 * cm] + [1.7 * cm] * 4 + [1.7 * cm] * 2
        story += [Paragraph(label, hd), tbl(rows, widths, new_idx, bold, fs=6.5)]
    story += [Paragraph('Earlier: best-test-F1 epoch, one seed. New (IEMOCAP-6): epoch by validation F1, 5 seeds, test read only at '
                        'that epoch. Class-weighted loss (pass 1, experimental flag, off by default) gave 68.38±0.62 WF1 and '
                        'Happy 44.80±4.34 and was inconclusive against the baseline, so it is not shown as a result. Sources: '
                        'paper Table IV; results/comparison.md; results/experiments/p0c_seed*.json.', small), PageBreak()]

    # ---- Table IV-b diagnostics ----
    sel = np.mean([r['test_f1_emo'] for r in P0C])
    story += [Paragraph('TABLE IV-b: WHERE THE IEMOCAP-6 GAP COMES FROM (new runs, diagnostic)', th),
              Paragraph('Mean test WF1 over 5 seeds for different ways of choosing the epoch. Only the first row is a valid result; '
                        'the others use or look at test curves and are for understanding only.', sub),
              tbl([['How the epoch is chosen', 'WF1 (%)', 'Valid result?'],
                   ['Best validation F1 (reported as "new")', f'{sel:.2f}', 'yes'],
                   ['Final epoch (120)', f'{np.mean([c[-1] for c in curves]):.2f}', 'yes (no selection)'],
                   ['Mean of last 10 epochs', f'{np.mean([np.mean(c[-10:]) for c in curves]):.2f}', 'yes (no selection)'],
                   ['Best test epoch ("oracle")', f'{np.mean([max(c) for c in curves]):.2f}', 'no, optimistic'],
                   ['Paper', f'{cp.PAPER_TABLE_IV["IEMOCAP"][3]:.2f}', 'protocol not stated']],
                  [8 * cm, 3 * cm, 4 * cm], bold_rows=[1]),
              Paragraph('Training reaches about 71 WF1 at its best test epoch, but the epoch picked on the 12-dialogue validation set '
                        'gives about 68. Valid WF1 is flat across seeds (std 0.45) while test moves about 3 points between epochs, '
                        'so epoch choice, not the model, explains most of the new-vs-earlier drop.', small)]

    he = [r['test_confusion'][0][4] for r in P0C]
    eh = [r['test_confusion'][4][0] for r in P0C]
    story += [tbl([['Confusion (new, per seed)', 'Seeds 2024, 1, 2, 3, 4', 'Mean'],
                   ['Happy predicted as Excited (of 144)', ', '.join(map(str, he)), f'{np.mean(he):.1f}'],
                   ['Excited predicted as Happy (of 299)', ', '.join(map(str, eh)), f'{np.mean(eh):.1f}']],
                  [6.5 * cm, 6.5 * cm, 2 * cm]),
              PageBreak()]

    # ---- Table V ----
    story += [Paragraph('TABLE V: MAIN RESULTS FOR THE MSAC TASK ON ALL DATASETS', th),
              Paragraph('Sentiment results for GraphSmile. † uses the Table III settings; ‡ settings optimized for MSAC. '
                        'New: not rerun for any dataset.', sub)]
    for ds, label in (('IEMOCAP', 'IEMOCAP-6'), ('IEMOCAP4', 'IEMOCAP-4'), ('MELD', 'MELD'), ('CMUMOSEI7', 'CMU-MOSEI')):
        v = cp.PAPER_TABLE_V[ds]
        cols = ['Neutral', 'Positive', 'Negative'] if ds == 'MELD' else ['Negative', 'Neutral', 'Positive']
        r = cp.load('full', ds)
        order = [0, 1, 2]  # MELD index order 0=neu, 1=pos, 2=neg matches the paper column order
        ours = [r['sen_f1'][i] * 100 for i in order] if r and len(r['sen_f1']) == 3 else [None] * 3
        rows = [['Method'] + cols + ['ACC', 'WF1'],
                ['GraphSmile† (paper)'] + [f2(x) for x in v['const']],
                ['GraphSmile† ours (earlier)'] + [f2(x) for x in ours] + [f2(r['acc_sen']), f2(r['f1_sen'])],
                ['Δ earlier'] + [dlt(a, b) for a, b in zip(ours, v['const'][:3])]
                + [dlt(r['acc_sen'], v['const'][3]), dlt(r['f1_sen'], v['const'][4])],
                ['GraphSmile† ours (new)', NR, '', '', '', ''],
                ['GraphSmile‡ (paper)'] + [f2(x) for x in v['opt']]]
        story += [Paragraph(label, hd), tbl(rows, [4.2 * cm] + [2.2 * cm] * 5, fs=6.5)]
    story += [Paragraph('The ‡ rows are paper values only (the paper does not give the optimized settings). Sources: paper Table V; '
                        'results/comparison.md.', small), PageBreak()]

    # ---- Table VI ----
    story += [Paragraph('TABLE VI: RESULTS FOR ABLATION STUDIES ON ALL DATASETS', th),
              Paragraph('ACC / WF1 (%). Ours (earlier) only, except the full GraphSmile IEMOCAP-6 MERC cell, which also has a new '
                        'value. No ablation was rerun under the clean protocol.', sub)]
    dsl = list(cp.PAPER_TABLE_IV.keys())
    heads = ['IEMOCAP-6', 'IEMOCAP-4', 'MELD', 'CMU-MOSEI']
    merc = [('GraphSmile', 'full'), ('No Ls', 'no_Ls'), ('No Lo', 'no_Lo'), ('No Ls & Lo', 'no_Ls_Lo'), ('No Res', 'no_Res'),
            ('No FC-Res', 'no_FCRes'), ('No Seg', 'no_Seg'), ('No Textual', 'no_text'), ('No Visual', 'no_visual'),
            ('No Acoustic', 'no_acoustic')]
    msac = [('GraphSmile', 'full'), ('No Le', 'no_Le'), ('No Lo', 'no_Lo'), ('No Le & Lo', 'no_Le_Lo'), ('No Res', 'no_Res'),
            ('No FC-Res', 'no_FCRes'), ('No Seg', 'no_Seg'), ('No Textual', 'no_text'), ('No Visual', 'no_visual'),
            ('No Acoustic', 'no_acoustic')]

    def block(items, metric):
        rows = [['Variant', 'Source'] + [f'{h} ACC' for h in heads] + [], ]
        rows = [['Variant', 'Source'] + sum([[f'{h[:9]} ACC', 'WF1'] for h in heads], [])]
        new_idx = []
        for name, tag in items:
            pap = cp.PAPER_TABLE_VI[tag]
            pr, orow = [name, 'paper'], ['', 'ours (earlier)']
            for j, ds in enumerate(dsl):
                p = pap[j]
                r = cp.load(tag, ds)
                if metric == 'merc':
                    pa, pw = p[0], p[1]
                    oa, ow = (r['acc_emo'], r['f1_emo']) if r else (None, None)
                    if tag in ('no_Le', 'no_Le_Lo'):
                        pa = pw = None
                else:
                    pa, pw = p[2], p[3]
                    oa, ow = (r['acc_sen'], r['f1_sen']) if r else (None, None)
                    if tag in ('no_Ls', 'no_Ls_Lo'):
                        pa = pw = None
                pr += [f2(pa), f2(pw)]
                orow += [f2(oa), f2(ow)]
            rows += [pr, orow]
            if metric == 'merc' and tag == 'full':
                rows.append(['', 'ours (new)', new_acc, new_wf1] + [NR, ''] * 3)
                new_idx.append(len(rows) - 1)
        return tbl(rows, [1.9 * cm, 2.1 * cm] + [1.35 * cm] * 8, new_idx, fs=5.8)

    story += [Paragraph('MERC task (emotion)', hd), block(merc, 'merc'), PageBreak(), Paragraph('MSAC task (sentiment)', hd),
              block(msac, 'msac'),
              Paragraph('MSAC paper values appear to use the optimized settings (‡); ours use the Table III settings (†), so they are not '
                        'like-for-like. No Le and No Le & Lo on IEMOCAP-6 and IEMOCAP-4 are far below the paper and were not verified '
                        '(may not have converged). Sources: paper Table VI; results/comparison.md.', small), PageBreak()]

    # ---- Table VII ----
    sec = [cp.parse_log(f'results/logs/p0c_IEMOCAP_seed{s}.log', 'IEMOCAP') for s in (2024, 1, 2, 3, 4)]
    rows = [['Dataset', 'Paper s/epoch', 'Ours earlier s/epoch', 'Ours new s/epoch', 'Paper MB', 'Ours earlier MB', 'Ours new MB']]
    for ds, label in (('IEMOCAP', 'IEMOCAP-6'), ('IEMOCAP4', 'IEMOCAP-4'), ('MELD', 'MELD'), ('CMUMOSEI7', 'CMU-MOSEI')):
        ps, pm = cp.PAPER_TABLE_VII[ds]
        r = cp.load('full', ds)
        if ds == 'IEMOCAP':
            ns = f'{np.mean([x["sec_per_epoch"] for x in sec]):.2f}'
            nm = f'{np.mean([x["mem_mb"] for x in sec]):.2f}'
        else:
            ns = nm = NR
        rows.append([label, f2(ps), f2(r['sec_per_epoch']), ns, f2(pm), f2(r['mem_mb']), nm])
    story += [Paragraph('TABLE VII: TIME AND MEMORY OVERHEAD FOR THE MERC TASK ON ALL DATASETS', th),
              Paragraph('Seconds per epoch (includes test evaluation) and allocated + reserved memory. Paper: RTX 3090. Ours: shared '
                        'A100 with other users\' jobs, so time is not like-for-like.', sub),
              tbl(rows, [2.6 * cm] + [2.4 * cm] * 6, fs=6.5),
              Paragraph('New IEMOCAP-6 values are the mean over the 5 clean-protocol runs. Source: paper Table VII; run logs.', small),
              Paragraph('Open items', hd),
              Paragraph('Not reproduced: paper baseline rows, MSAC ‡ rows, Figs. 4 to 7. Planned, not run: last-10-epoch weight averaging '
                        '(results/experiments/p2-plan.md). Details of every pass: results/experiments/*.md.', body)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 7)
        canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f'GraphSmile: paper vs reproduced (earlier and new)   |   page {doc.page}')
        canvas.restoreState()

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    SimpleDocTemplate(out, pagesize=A4, leftMargin=1.4 * cm, rightMargin=1.4 * cm, topMargin=1.3 * cm,
                      bottomMargin=1.6 * cm).build(story, onFirstPage=footer, onLaterPages=footer)
    print('Wrote', out, f'({n} seeds in new runs)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'docs/paper_vs_reproduced_updated.pdf')
