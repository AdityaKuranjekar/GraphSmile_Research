# p3-collapse-check: summary report

Facts only. "not measured" means the number was not computed in this pass.

## 1. Identity

- Branch: `experiments/p3-collapse-check` (pushed to origin, tracking `origin/experiments/p3-collapse-check`).
- Latest commit before this report: `ca4b1fa`. Branch base: `ad738a6` (merge of protocol experiments). Note: `origin/main` has since moved to `ab136c2`; the diff below is taken against the branch base `ad738a6`, not current main.
- Pass id: p3-collapse-check.
- What it tests: whether the three no-L_e ablation runs (IEMOCAP no_Le, IEMOCAP4 no_Le, IEMOCAP4 no_Le_Lo) collapse (sentiment at the majority class), or fail to train, under the paper configuration with `--l2 0.001`, on seeds 2024, 1 and 2.

## 2. Code changes (`git diff ad738a6..HEAD`)

Two files added, no existing file modified:

- `scripts/run_collapse_check.sh` (new, 56 lines). Reruns 3 variants x 3 seeds = 9 runs, sequential, ports 17001 to 17009. Logs to `results/logs/p3_<dataset>_<variant>_seed<seed>.log`.
- `results/experiments/p3-collapse-check.md` (new, 49 lines). Step 1 readout of existing logs.

Confirmations:
- (a) Every new behaviour is behind a flag that defaults OFF: **not applicable**. There are no new flags. `run.py` is unchanged.
- (b) A run with default flags gives the same result as main: **not applicable as a code check**. `run.py` is unchanged. Evidence of reproduction: the seed 2024 runs reproduce the original logs exactly (IEMOCAP no_Le final sentiment ACC 54.59 in both; IEMOCAP4 no_Le final sentiment ACC 82.93 in both). These are the only seed-2024 matches checked.
- (c) Hyperparameters in `scripts/run_all.sh` (Table III) are unchanged: **confirmed by string comparison**. The dataset lines in `run_all.sh` (lines 14 and 15) and `run_remaining.sh` (lines 16 and 17) match the values hard-coded in `run_collapse_check.sh`. `run_all.sh` itself was not modified.

## 3. Results

Sentiment ACC and WF1 from the final test-report block of each log. Selection: default protocol (`--select_on test`), which picks the epoch with the best **test emotion F1**. Valid was not used for selection.

| Setting | Seeds | Clean WF1 mean±std | Paper-protocol sentiment WF1 mean±std (ACC mean±std) | Happy F1 | Excited F1 |
|---|---|---|---|---|---|
| IEMOCAP no_Le | 2024, 1, 2 | not measured | 68.10 ± 21.98 (ACC 72.07 ± 15.15) | not measured | not measured |
| IEMOCAP4 no_Le | 2024, 1, 2 | not measured | 63.12 ± 31.41 (ACC 68.86 ± 21.58) | not measured | not measured |
| IEMOCAP4 no_Le_Lo | 2024, 1, 2 | not measured | 62.26 ± 30.62 (ACC 68.12 ± 20.89) | not measured | not measured |

Per-run values (ACC / WF1):

| Setting | seed 2024 | seed 1 | seed 2 | Source file |
|---|---|---|---|---|
| IEMOCAP no_Le | 54.59 / 42.74 | 81.52 / 81.67 | 80.10 / 79.88 | `results/logs/p3_IEMOCAP_no_Le_seed{2024,1,2}.log` |
| IEMOCAP4 no_Le | 82.93 / 82.75 | 44.01 / 26.90 | 79.64 / 79.71 | `results/logs/p3_IEMOCAP4_no_Le_seed{2024,1,2}.log` |
| IEMOCAP4 no_Le_Lo | 44.01 / 26.90 | 80.70 / 80.17 | 79.64 / 79.71 | `results/logs/p3_IEMOCAP4_no_Le_Lo_seed{2024,1,2}.log` |

Baseline row: **not measured in this pass**. No full-model run with seeds 2024, 1, 2 under this script exists. The only full-model numbers in the repo are from earlier passes (p0, IEMOCAP-6 emotion, valid-selected, 5 seeds: WF1 70.82 ± 0.47). They are for a different metric and seed set and are not used here.

Std is sample std (n = 3).

Paper Table VI rows for these variants: **not in the files read for this report**. Not compared.

## 4. Decision against the rule

- Rule: keep only if the clean 5-seed mean beats the clean baseline by more than the baseline std, and the paper-protocol number is not worse.
- **Not applied.** This pass tests no candidate change. It is a diagnostic rerun of existing ablations. The clean protocol was not run, so the rule's first condition cannot be evaluated.
- Verdict per setting: **not applicable** (no keep/reject decision is made).

## 5. Deviations

- The kept change: **none**. Nothing is kept.
- Hyperparameters and architecture of the runs match Table III values (see section 2c). The runs use the paper configuration plus `--l2 0.001`, the same flag as `run_remaining.sh`. No tuned variant.

## 6. Unverified or weak points

- **Single protocol.** Only the paper-protocol (test-selected) numbers exist. The clean-protocol number is not measured.
- **Three seeds only.** Std values are very large (e.g. ±31 WF1 for IEMOCAP4 no_Le) because two of the nine runs sit at the majority-class value.
- **Collapse is selection-driven.** In IEMOCAP4 no_Le seed 1 and IEMOCAP4 no_Le_Lo seed 2024, the default protocol picks an epoch where sentiment predicts one class for every sample (ACC 44.01, WF1 26.90). The same runs learn sentiment to about 80 to 83 ACC at later epochs (step 1 readout). Valid and test were not compared, and the test-selected number is the one reported.
- **Emotion metrics not extracted.** The logs contain emotion reports (Happy, Excited and others), which were not read for this report.
- **Failed and overwritten runs.** The first attempt at 18:47 failed in all 9 runs with `ModuleNotFoundError: No module named 'torch'` (the `base` conda environment). Its output overwrote the earlier partial logs. A second attempt was killed at epoch 90 on the first run, after an SSH disconnect. The final 9 runs, all reported here, were run in the `graphsmile` environment (PyTorch 2.0.0+cu117, CUDA available).
- **Shared GPU.** Other users' jobs were running on the server during the sweep, so the runs were not timed in isolation.
- **Unchecked claims.** The step 1 readout identifies the selected epoch by matching accuracy values. This was inferred from matching numbers, not from a saved checkpoint.

## 7. Merge recommendation

- **Safe to merge into main** (no code path change, no flags):
  - `scripts/run_collapse_check.sh`: new script, only reads existing `run.py` arguments.
  - `results/experiments/p3-collapse-check.md` and this file: notes.
- **Stay on the branch:** nothing needs to stay. Nothing here is an extension or a tuned variant.
- **GPU runs still needed:** none for the diagnosis. For a clean-protocol number (the standing rule's second number), 9 runs are needed: the same 3 variants x 3 seeds with `--clean_valid --select_on valid` added to the `run.py` call. No such runs are scheduled.
