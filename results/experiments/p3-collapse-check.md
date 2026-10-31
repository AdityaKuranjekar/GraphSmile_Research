# p3: collapse check on the no-L_e ablations

Branch `experiments/p3-collapse-check`, created from `origin/main` (ad738a6). No code was changed; `run.py` and existing scripts are untouched.

## Step 1: existing logs (no GPU)

Logs read: `results/logs/IEMOCAP_no_Le.log`, `IEMOCAP4_no_Le.log`, `IEMOCAP4_no_Le_Lo.log`. All three use the default protocol (`--select_on test`, `--classify emotion`).

**Selection detail (read from `run.py` ~L502-536).** With the default protocol, the epoch whose numbers are reported is the one with the best *test emotion F1*. The sentiment numbers in the final "Test performance" block are therefore taken at that epoch, not at the last one. Matching the final sentiment accuracy against the per-epoch rows:

| log | final reported sen acc | matching epoch row |
|---|---|---|
| IEMOCAP_no_Le | 54.59 | ep 2 (test_acc_sen 54.59, f1_sen 42.74) |
| IEMOCAP4_no_Le | 82.93 | ep 50 (test_acc_sen 82.93) |
| IEMOCAP4_no_Le_Lo | 44.01 | ep 1 and ep 2 (both 44.01) |

### IEMOCAP_no_Le (6 emotion classes, 120 epochs)
- Training loss falls every epoch from 1.4483 (ep 1) to 0.2879 (ep 120). Small bumps at ep 28 (0.5528) and ep 55 (0.4006), no divergence.
- Sentiment learns early: ep 1 acc 49.04 / f1 32.28; ep 3 72.89 / 69.19; ep 4 80.16 / 79.04; ep 20 82.50 / 82.66; peak ep 54 83.92 / 83.82; ep 120 83.36 / 83.47.
- Transient dips, not collapse: ep 31 81.15 / 81.67, ep 34 79.30 / 80.19, ep 56 79.17 / 79.25.
- Emotion head stays near chance (test_acc_emo 15-19 %); the final emotion matrix predicts only classes 0, 4 and 5.
- Verdict: the training did not fail and did not collapse. The reported sentiment (54.59 / 42.74) comes from ep 2, an under-trained epoch chosen by emotion F1.

### IEMOCAP4_no_Le (4 emotion classes, 120 epochs)
- Loss falls from 0.9824 (ep 1) to 0.2187 (ep 120); a slight rise after ep 117 (0.1946 to 0.2187). Not a divergence.
- Sentiment starts at the majority-class value: ep 1 and ep 2 both 44.01 / 26.90. Then it learns: ep 5 68.19 / 62.76; ep 8 78.58 / 77.94; ep 24 84.20 / 84.20.
- Dips: ep 18 73.49 / 70.51; ep 51 79.43 / 78.70; ep 81-83 82.08 to 83.88.
- Late values: ep 100 82.61 / 82.33; ep 114 84.52 / 84.38; ep 120 81.44 / 81.11.
- Verdict: learned, with no collapse. The reported epoch (ep 50, 82.93) is a normal sentiment value.

### IEMOCAP4_no_Le_Lo (`--loss_type sen`, 120 epochs)
- Loss falls from 1.0306 (ep 1) to 0.0014 (ep 120). It is near zero from about ep 60, so the model overfits strongly.
- Sentiment starts stuck at the majority class: ep 1 and ep 2 both 44.01 / 26.90, ep 3 49.52 / 37.68. It learns by ep 8 (80.49 / 80.08) and stays 82 to 84 through ep 120 (83.67 / 83.49).
- Dips: ep 17 74.76 / 74.65; ep 35 70.41 / 67.15.
- Verdict: the training learns and then does not collapse. The reported numbers collapse only because the selection picks ep 1 or 2, where sentiment predicts the majority class for every sample (final sentiment matrix: all 415 / 384 / 144 rows predicted class 0).

### Answer to the question
- Failed from the start (loss never falls): **none** of the three.
- Diverged later: **none**; losses decrease monotonically apart from small local bumps.
- Learned, then collapsed: **none** in the training curves. The collapse appears only in the reported number for IEMOCAP4_no_Le_Lo, because the default protocol selects by test emotion F1 and that epoch is the one where sentiment is still at the majority class.

## Step 2: rerun script

`scripts/run_collapse_check.sh`: 3 variants x 3 seeds (2024, 1, 2) = 9 runs, sequential. Ports 17001 to 17009. Logs go to `results/logs/p3_<dataset>_<variant>_seed<seed>.log`. The script prints the run count before it starts.

Limit: the script runs the default protocol only. The standing rule asks for a second number per run (`--clean_valid --select_on valid`). That is not in the requested run set, so it is not added here. It needs a separate decision.

## Step 3: pending
Sentiment ACC / WF1 per run, mean ± std per variant, and the Table VI comparison: waiting for the logs.
