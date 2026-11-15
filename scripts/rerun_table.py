"""Full vs No-Textual drop table: paper, earlier runs (results/logs), rerun (results/rerun)."""
import os, sys
sys.path.insert(0, 'scripts')
import compare_to_paper as c
DS = [('IEMOCAP', 'IEMOCAP-6', 0), ('IEMOCAP4', 'IEMOCAP-4', 1), ('MELD', 'MELD', 2), ('CMUMOSEI7', 'CMU-MOSEI', 3)]
def get(d, ds, tag):
    return c.parse_log(os.path.join(d, f'{ds}_{tag}.log'), ds)
print('| Dataset | Source | Full ACC / WF1 | No-text ACC / WF1 | Drop ACC | Drop WF1 |')
print('|---|---|---|---|---|---|')
for ds, name, i in DS:
    pf, pn = c.PAPER_TABLE_VI['full'][i], c.PAPER_TABLE_VI['no_text'][i]
    rows = [('Paper', pf[0], pf[1], pn[0], pn[1])]
    for lab, d in [('Earlier', 'results/logs'), ('Rerun', 'results/rerun')]:
        f, n = get(d, ds, 'full'), get(d, ds, 'no_text')
        rows.append((lab, f['acc_emo'], f['f1_emo'], n['acc_emo'], n['f1_emo']))
    for lab, fa, ff, na, nf in rows:
        print(f'| {name} | {lab} | {fa:.2f} / {ff:.2f} | {na:.2f} / {nf:.2f} | {fa-na:.2f} | {ff-nf:.2f} |')
