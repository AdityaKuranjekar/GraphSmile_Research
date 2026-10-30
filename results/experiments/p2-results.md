# p2 results: sensitivity sweep, clean protocol, stage 1 (3 seeds)

Date: 2026-10-30. Source: JSON files in `server_pass2/experiments` (the JSON is the source of truth).

## Setup

- Protocol: clean (`--clean_valid --select_on valid`). Seeds 2024, 1, 2 for every setting.
- Baseline: p0c, same three seeds (2024, 1, 2). Its 5-seed numbers (68.26 ± 0.94) are not used here.
- Std: sample std (ddof=1) over the three seeds.
- WF1 = `test_f1_emo`. For all 27 runs it equals the support-weighted mean of the per-class F1 recomputed from `test_confusion`.
- Happy = per-class F1 index 0, Excited = index 4 (label order as in `p0-protocol.md`).
- Happy→Excited (H→E) = `test_confusion[0][4]`, out of 144 Happy test utterances.

## Completeness

All 24 p2 JSON files and the 3 p0c files used here have: a selected epoch, validation and test per-class F1 (6 values each), the test confusion matrix, and a `seed` field that matches the filename. **None are missing or incomplete.**

The first log, `p2_win_5_5_clean_seed2024.log`, was not diagnosed line by line. Its JSON is complete and is used as the source. Its tail matches the sibling logs.

## Results (test, mean ± sample std over seeds 2024, 1, 2)

| Setting | Valid WF1 (mean ± std) | Test WF1 | ΔWF1 vs p0c | Happy F1 | Excited F1 | H→E (of 144) | Selected epochs |
|---|---|---|---|---|---|---|---|
| **p0c baseline** | 73.37 ± 0.36 | **67.71 ± 0.64** | — | 47.75 ± 1.69 | 75.21 ± 0.14 | 49.3 ± 2.9 | 17, 16, 89 |
| heter_4 | 73.81 ± 0.45 | 68.10 ± 0.97 | +0.39 | 46.04 ± 4.64 | 73.03 ± 1.44 | 51.3 ± 10.6 | 42, 17, 36 |
| lambd_0.3 | 73.84 ± 0.55 | 67.88 ± 1.21 | +0.17 | 45.69 ± 4.02 | 75.20 ± 1.66 | 57.7 ± 8.7 | 60, 29, 12 |
| lambd_0.5 | 73.82 ± 0.47 | 67.92 ± 0.76 | +0.21 | 42.86 ± 1.17 | 74.65 ± 1.28 | 62.0 ± 9.8 | 55, 51, 16 |
| shift_10 | 73.70 ± 0.23 | 69.26 ± 1.22 | +1.55 | 49.53 ± 2.86 | 76.39 ± 0.94 | 51.7 ± 4.0 | 34, 31, 17 |
| shift_5 | 73.69 ± 0.23 | 69.14 ± 0.69 | +1.43 | 50.79 ± 1.94 | 75.06 ± 1.46 | 50.7 ± 5.5 | 35, 66, 112 |
| win_10_10 | 74.86 ± 0.28 | 67.88 ± 0.74 | +0.17 | 38.73 ± 5.79 | 75.71 ± 0.75 | 81.7 ± 12.4 | 49, 51, 56 |
| win_10_2 | 74.49 ± 0.44 | 66.52 ± 1.05 | −1.19 | 42.96 ± 3.71 | 70.30 ± 1.90 | 53.3 ± 4.5 | 91, 25, 113 |
| win_5_5 | 74.81 ± 0.54 | 68.38 ± 1.04 | +0.67 | 44.60 ± 12.18 | 74.91 ± 0.49 | 73.7 ± 22.1 | 55, 80, 15 |

Validation column: mean ± sample std (ddof=1) of `valid_f1_emo` over seeds 2024, 1, 2, from the clean-protocol JSON files. The baseline is `p0c_seed*.json` (73.39, 73.01, 73.72). The setting values are from `p2_<setting>_clean_seed*.json`. These 24 files were found in `D:\NotesSem 5\Research_Sem5_Rashmi\server_pass2\experiments\`, not in `results/experiments/` as requested, so the table above was computed from that folder without copying files.

### Validation vs test for shift_5 and shift_10

- **Validation:** shift_5 (73.69 ± 0.23) and shift_10 (73.70 ± 0.23) are each about +0.33 over the p0c baseline (73.37 ± 0.36). That gap is smaller than the baseline std, so neither clearly beats the baseline on validation. Both are essentially flat there.
- **Disagreement with test:** on test, shift_10 (+1.55) and shift_5 (+1.43) were the largest gains. On validation, the highest means are win_10_10 (74.86) and win_5_5 (74.81), and win_10_2 (74.49) is also above baseline. win_10_2 was the one setting rejected on test (−1.19). So validation and test disagree on the ranking. The shift gains on test are not confirmed by validation.
- **Weak signal:** the validation set has 12 dialogues, so these differences are noisy. Still, validation does not support the stage-2 promotion of shift_5 and shift_10 on the basis of their test gains. This should be weighed when stage 2 results come in.

No decision in this file was changed.

Per-seed test WF1 (2024 / 1 / 2):
- p0c: 67.96 / 68.19 / 66.98
- shift_10: 69.72 / 70.18 / 67.87
- shift_5: 68.51 / 69.88 / 69.02
- win_5_5: 69.22 / 67.22 / 68.71

## Decision rule

Keep only if the clean WF1 mean beats the p0c mean by more than the baseline std.
Threshold: 67.71 + 0.64 = **68.35**. Reject if the mean is worse by more than 0.64 (below 67.07). Otherwise mark inconclusive.

| Setting | Decision | Reason |
|---|---|---|
| shift_10 | **Keep** | +1.55 is above 0.64 |
| shift_5 | **Keep** | +1.43 is above 0.64 |
| win_5_5 | **Keep (marginal)** | +0.67 clears 0.64 by 0.03. Seed-paired differences are +1.26, −0.97, +1.73. Happy std is 12.18. Treat as fragile. |
| heter_4 | Inconclusive | +0.39 is within one std |
| lambd_0.3 | Inconclusive | +0.17 is within one std |
| lambd_0.5 | Inconclusive | +0.21 is within one std |
| win_10_10 | Inconclusive | +0.17 is within one std |
| win_10_2 | **Reject** | −1.19 is worse by more than one std |

## Stage 2 candidates

Under the stated rule, three settings pass to stage 2 (5 seeds, paper protocol):

1. **shift_10**
2. **shift_5**
3. **win_5_5**, marginal. Include only if you want to test a borderline pass.

Stage 2 must also report the paper-comparable number (`--select_on test`, default split), as required by the experiment rules. Stage 1 reports only the clean protocol, so no paper-comparable number exists yet for any setting here.

## Notes

- The baseline std (0.64) comes from three seeds, so it is itself a noisy estimate. The one-std margin is a coarse threshold.
- shift_5 gains through Happy (+3.0 over p0c, about 1.8 baseline Happy stds). Excited is flat (75.06 vs 75.21). shift_10 gains on both: Happy +1.8 and Excited +1.2.
- Happy→Excited counts vary widely across seeds in several settings (win_10_10 and win_5_5 especially). These confusion counts are not used in the decision.

## Stage 2 (promoted 2026-10-30)

Promoted: **shift_10** and **shift_5**. win_5_5 is not promoted.

Stage 2 needs, per the experiment rules:
- Paper protocol (default flags, `--select_on test`) for base, shift_5 and shift_10, seeds 2024, 1, 2, 3, 4.
- Clean protocol for shift_5 and shift_10, seeds 3 and 4. Clean base seeds 3 and 4 already exist as `p0c_seed3.json` and `p0c_seed4.json`.

Run on the user's server (`/DATA/rashmi_2521cs34/GraphSmile_Research`, conda env `graphsmile`), with the branch `experiments/p1b-provenance` checked out so `scripts/run_pass2.sh` is present:

```bash
SETTINGS="base shift_5 shift_10" PROTOCOLS="paper" SEEDS="2024 1 2 3 4" GPU=0 bash scripts/run_pass2.sh
```

```bash
SETTINGS="shift_5 shift_10" PROTOCOLS="clean" SEEDS="3 4" GPU=0 bash scripts/run_pass2.sh
```

That is 19 runs (15 paper, 4 clean). Results come back as `results/experiments/p2_<setting>_<protocol>_seed<seed>.json`.

Decision rule for stage 2: keep a setting only if its 5-seed clean WF1 mean beats the clean baseline by more than the baseline std, and its paper-comparable WF1 is not worse than the paper-protocol base. The outcome is in the stage 2 results section below: neither setting is kept.

## Stage 2 results (5 seeds: 2024, 1, 2, 3, 4)

Clean protocol (`--clean_valid --select_on valid`), mean ± sample std over 5 seeds. Baseline is p0c.

| Setting | Clean WF1 | ΔWF1 | Acc Δ | Happy Δ | Excited Δ | Decision |
|---|---|---|---|---|---|---|
| p0c baseline | 68.26 ± 0.94 | — | — | — | — | — |
| shift_5 | 68.52 ± 1.73 | +0.26 | +0.05 | +2.77 | −1.20 | Inconclusive |
| shift_10 | 68.35 ± 1.58 | +0.09 | −0.01 | +4.31 | +0.01 | Inconclusive |

Clean per-seed WF1 (2024 / 1 / 2 / 3 / 4):
- shift_5: 68.51 / 69.88 / 69.02 / 69.63 / 65.57
- shift_10: 69.72 / 70.18 / 67.87 / 67.61 / 66.36

Paper protocol (`--select_on test`, default), WF1 from each run's log `F-Score` line, 5 seeds:

| Setting | Paper WF1 | Std | Δ vs base |
|---|---|---|---|
| base | 71.61 | 0.36 | — |
| shift_5 | 71.48 | — | −0.13 |
| shift_10 | 71.49 | — | −0.11 |

**Decision:** neither setting is kept. Both clean gains are well inside the baseline std of 0.94. The stage 1 gain (+1.4 to +1.6 WF1 on 3 seeds) did not hold with five seeds. The paper-protocol numbers are slightly lower than base, within noise. Happy F1 rises for both settings, but shift_5 gives up Excited F1 to get it.
