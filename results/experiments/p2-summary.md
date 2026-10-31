# p2 summary: IEMOCAP-6 sensitivity to window, shift window, depth and loss weights

Facts only. Where a number was not measured, it says so.

## 1. Identity

- **Branch:** `experiments/p1b-provenance`
- **Latest commit before this summary:** `f284518` ("p2: fix stale stage 2 wording in results"). `main` contains this branch's work as merge `ab136c2`.
- **Base commit:** `d12f026` (merge-base of the branch and `main` before the merge).
- **Pass id:** p2
- **What it tests:** whether IEMOCAP-6 results change when the window sizes, the shift window, the heterogeneous layer depth or the loss weights are varied from the paper configuration. Each setting changes one value from the paper baseline.

## 2. Code changes (`git diff d12f026..HEAD`)

Files changed on this branch:

| File | Change |
|---|---|
| `scripts/run_pass2.sh` | New, 92 lines. Launch script for the sweep. It passes existing flags only. |
| `results/experiments/p1b-provenance.md` | New, 236 lines. Data provenance checks. |
| `results/experiments/p2-results.md` | New, 126 lines. Stage 1 and stage 2 results and decisions. |

`run.py`, `trainer.py`, `model.py` and `scripts/run_all.sh` are **unchanged** from the base.

**New flags:** none in this pass. The sweep uses flags that already exist in `main`:

- `--clean_valid`: default off (`run.py` line 135).
- `--select_on`: default `test`, choices `test` or `valid` (`run.py` line 132).
- `--seed`: default 2024 (`run.py` line 130).
- `--out_json`: default empty (`run.py` line 140).
- `--shift_win`: default 12 in `run.py` (line 112).
- `--heter_n_layers`: default `[6, 6, 6]` in `run.py` (line 100).
- `--win`, `--lambd`: existing flags.

**Checks requested:**

- **(a) Every new behaviour is behind a flag that defaults off:** not applicable. No new behaviour was added in this pass.
- **(b) A run with default flags gives the same result as `main`:** not tested in this pass. No run with default flags was made on this branch.
- **(c) Hyperparameters in `scripts/run_all.sh` are unchanged:** confirmed. `run_all.sh` is not in the diff. Its IEMOCAP row is `17 17 | 7 7 7 | drop 0.2 | shift 19 | lambd 1.0 1.0 0.7`. `run_pass2.sh` base uses the same values.

**Note:** the `run.py` defaults for `--shift_win` (12) and `--heter_n_layers` (6, 6, 6) differ from the `run_all.sh` IEMOCAP values. Every pass 2 run passes its values explicitly, so the defaults did not affect these results.

## 3. Results

All clean-protocol numbers are test, from JSON files, mean ± sample std over seeds. The clean protocol selects the epoch on **valid** (`--clean_valid --select_on valid`).

The paper-protocol numbers are test-selected (`--select_on test`). They come from the `F-Score` line in each log, because paper runs write no JSON.

### 3a. Stage 1: clean protocol, 3 seeds (2024, 1, 2)

Files: `results/experiments/p2_<setting>_clean_seed{2024,1,2}.json` and `p0c_seed{2024,1,2}.json`.

| Setting | Seeds | Clean WF1 mean ± std | Paper WF1 | Happy F1 (mean ± std) | Excited F1 (mean ± std) |
|---|---|---|---|---|---|
| p0c baseline | 2024, 1, 2 | 67.71 ± 0.64 | not measured | 47.75 ± 1.69 | 75.21 ± 0.14 |
| heter_4 | 2024, 1, 2 | 68.10 ± 0.97 | not measured | 46.04 ± 4.64 | 73.03 ± 1.44 |
| lambd_0.3 | 2024, 1, 2 | 67.88 ± 1.21 | not measured | 45.69 ± 4.02 | 75.20 ± 1.66 |
| lambd_0.5 | 2024, 1, 2 | 67.92 ± 0.76 | not measured | 42.86 ± 1.17 | 74.65 ± 1.28 |
| shift_10 | 2024, 1, 2 | 69.26 ± 1.22 | not measured | 49.53 ± 2.86 | 76.39 ± 0.94 |
| shift_5 | 2024, 1, 2 | 69.14 ± 0.69 | not measured | 50.79 ± 1.94 | 75.06 ± 1.46 |
| win_10_10 | 2024, 1, 2 | 67.88 ± 0.74 | not measured | 38.73 ± 5.79 | 75.71 ± 0.75 |
| win_10_2 | 2024, 1, 2 | 66.52 ± 1.05 | not measured | 42.96 ± 3.71 | 70.30 ± 1.90 |
| win_5_5 | 2024, 1, 2 | 68.38 ± 1.04 | not measured | 44.60 ± 12.18 | 74.91 ± 0.49 |

### 3b. Stage 2: 5 seeds (2024, 1, 2, 3, 4)

Files: `results/experiments/p2_shift_{5,10}_clean_seed{2024,1,2,3,4}.json`, `p0c_seed{2024,1,2,3,4}.json`, and paper results from `results/logs/p2_{base,shift_5,shift_10}_paper_seed{...}.log`.

| Setting | Seeds | Clean WF1 mean ± std | Paper WF1 mean ± std | Happy F1 (clean) | Excited F1 (clean) |
|---|---|---|---|---|---|
| base (p0c clean, paper runs) | 2024, 1, 2, 3, 4 | 68.26 ± 0.94 | 71.61 ± 0.36 | 45.62 (mean only) | 76.10 (mean only) |
| shift_5 | 2024, 1, 2, 3, 4 | 68.52 ± 1.73 | 71.48 ± 0.21 | 48.39 (mean only) | 74.90 (mean only) |
| shift_10 | 2024, 1, 2, 3, 4 | 68.35 ± 1.58 | 71.49 ± 0.32 | 49.93 (mean only) | 76.10 (mean only) |

Std for the Happy and Excited columns was not measured in this report.

Per-seed clean WF1 (2024 / 1 / 2 / 3 / 4):
- shift_5: 68.51 / 69.88 / 69.02 / 69.63 / 65.57
- shift_10: 69.72 / 70.18 / 67.87 / 67.61 / 66.36

Per-seed paper WF1 (2024 / 1 / 2 / 3 / 4):
- base: 71.17 / 71.88 / 71.79 / 71.93 / 71.26
- shift_5: 71.55 / 71.34 / 71.76 / 71.51 / 71.22
- shift_10: 71.23 / 71.72 / 71.77 / 71.68 / 71.07

**Metric used to choose the setting:** valid, through the clean protocol (`--select_on valid`). The paper protocol uses test by design and is shown for comparison only.

## 4. Decision against the rule

**Rule (applied literally):** keep only if the 5-seed clean WF1 mean beats the clean baseline by more than the baseline std, **and** the paper-protocol WF1 is not worse.

Baseline clean std = 0.94. Baseline paper std = 0.36.

| Setting | Clean Δ WF1 | Clean test (Δ > 0.94?) | Paper Δ WF1 | Paper not worse? | Verdict |
|---|---|---|---|---|---|
| shift_5 | 68.52 − 68.26 = +0.26 | No (0.26 < 0.94) | 71.48 − 71.61 = −0.13 | Within noise (0.13 < 0.36) | **Inconclusive** |
| shift_10 | 68.35 − 68.26 = +0.09 | No (0.09 < 0.94) | 71.49 − 71.61 = −0.11 | Within noise (0.11 < 0.36) | **Inconclusive** |

Neither setting is kept. Neither is rejected, because neither is worse on clean by more than the std.

**Stage 1 verdicts (3 seeds, 3-seed baseline std 0.64), for reference:**
- shift_10 and shift_5 passed (+1.55 and +1.43).
- win_5_5 passed marginally (+0.67).
- win_10_2 was rejected (−1.19).
- heter_4, lambd_0.3, lambd_0.5 and win_10_10 were inconclusive (+0.17 to +0.39).

Stage 1 verdicts were overturned at stage 2 for shift_5 and shift_10 (+1.43 → +0.26 and +1.55 → +0.09). win_5_5 was not promoted to stage 2.

## 5. Deviations

**No kept change.** Nothing from this pass is kept.

The settings that were tested differ from the paper's Table III values in one parameter each, so they are **tuned variants, not paper settings**:
- shift_5 and shift_10 change `--shift_win` from 19 to 5 or 10.
- win_5_5, win_10_10 and win_10_2 change `--win`.
- heter_4 changes `--heter_n_layers` from 7 7 7 to 4 4 4.
- lambd_0.3 and lambd_0.5 change the third or second `--lambd` value.

## 6. Unverified or weak points

- **Paper-protocol numbers are not in JSON.** They come from the log line `Acc: …, F-Score: …`. The code that prints it (`run.py` line 586) prints `max(all_f1_emo)` when no validation selection is made, which is the best-epoch test F1. I read the print statement; I did not check how `all_f1_emo` is built line by line.
- **Paper per-class F1 for every setting was not measured.** Only WF1 is available.
- **Paper selection protocol not confirmed.** The paper's epoch-selection method is not stated in the pages read (see `p2-plan.md`). The paper-protocol baseline here is the repo's test-selected default, not a value taken from the paper.
- **Paper claim not verified:** the Stochastic Weight Averaging hypothesis in `p2-plan.md` was not tested in this pass. `--avg_last` was not used by `run_pass2.sh`.
- **Single-seed runs:** none. Every setting has at least 3 seeds.
- **Failed or incomplete runs:**
  - The first paper batch (06:48) failed in about one second because `torch` was missing in the base conda environment. No results came from it.
  - `p2_shift_5_paper_seed2024` was cut off during the SSH disconnect at about 16:15. Its log was overwritten by the restart, and it produced no result.
  - `p2_shift_5_paper_seed3` was cut off at the start of its run after another disconnect. It was rerun and finished.
  - `p2_win_5_5_clean_seed2024.log` was flagged as possibly damaged by a duplicate job. Its JSON is complete. The log was not checked line by line.
- **Valid and test disagree:** the clean baseline (68.26) is 3.35 below the paper-protocol baseline (71.61) for the same configuration. This gap is from epoch selection, not from the setting.
- **Stage 1 did not hold at stage 2.** The shift gains dropped from about +1.5 at 3 seeds to +0.1 to +0.3 at 5 seeds.
- **Happy F1 has high seed variance:** win_5_5 Happy std is 12.18 over 3 seeds.

## 7. Merge recommendation

**Already on `main` (merge `ab136c2`):**
- `scripts/run_pass2.sh`: safe. Only existing flags, no defaults changed.
- `results/experiments/p1b-provenance.md`: safe.
- `results/experiments/p2-results.md`: safe. It records the negative stage 2 result.

**Stays on the branch:**
- `results/experiments/p2-summary.md` (this file): hold until you decide whether to merge it.
- The untracked files `inspect_pkl.py`, `p0.patch`, `p1.patch` and `p1_clean.patch`: not committed. Their `run.py` changes are already in `main`.

**GPU runs needed before a keep decision:**
- **None for shift_5 or shift_10**, because both are already decided as not kept.
- **If a different setting were promoted,** each one needs 7 runs: 5 paper (seeds 2024, 1, 2, 3, 4) and 2 clean (seeds 3, 4). That is about 1 hour at 8.2 minutes per run.

Commands for a new setting `<name>` (see `results/experiments/p2-results.md` for the stage 2 commands used here):

```bash
SETTINGS="<name>" PROTOCOLS="paper" SEEDS="2024 1 2 3 4" GPU=0 bash scripts/run_pass2.sh
```

```bash
SETTINGS="<name>" PROTOCOLS="clean" SEEDS="3 4" GPU=0 bash scripts/run_pass2.sh
```
