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

| Setting | Test WF1 | ΔWF1 vs p0c | Happy F1 | Excited F1 | H→E (of 144) | Selected epochs |
|---|---|---|---|---|---|---|
| **p0c baseline** | **67.71 ± 0.64** | — | 47.75 ± 1.69 | 75.21 ± 0.14 | 49.3 ± 2.9 | 17, 16, 89 |
| heter_4 | 68.10 ± 0.97 | +0.39 | 46.04 ± 4.64 | 73.03 ± 1.44 | 51.3 ± 10.6 | 42, 17, 36 |
| lambd_0.3 | 67.88 ± 1.21 | +0.17 | 45.69 ± 4.02 | 75.20 ± 1.66 | 57.7 ± 8.7 | 60, 29, 12 |
| lambd_0.5 | 67.92 ± 0.76 | +0.21 | 42.86 ± 1.17 | 74.65 ± 1.28 | 62.0 ± 9.8 | 55, 51, 16 |
| shift_10 | 69.26 ± 1.22 | +1.55 | 49.53 ± 2.86 | 76.39 ± 0.94 | 51.7 ± 4.0 | 34, 31, 17 |
| shift_5 | 69.14 ± 0.69 | +1.43 | 50.79 ± 1.94 | 75.06 ± 1.46 | 50.7 ± 5.5 | 35, 66, 112 |
| win_10_10 | 67.88 ± 0.74 | +0.17 | 38.73 ± 5.79 | 75.71 ± 0.75 | 81.7 ± 12.4 | 49, 51, 56 |
| win_10_2 | 66.52 ± 1.05 | −1.19 | 42.96 ± 3.71 | 70.30 ± 1.90 | 53.3 ± 4.5 | 91, 25, 113 |
| win_5_5 | 68.38 ± 1.04 | +0.67 | 44.60 ± 12.18 | 74.91 ± 0.49 | 73.7 ± 22.1 | 55, 80, 15 |

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

Decision for stage 2 (to be filled in when results return): keep a setting only if its 5-seed clean WF1 mean beats the clean baseline by more than the baseline std, and its paper-comparable WF1 is not worse than the paper-protocol base.
