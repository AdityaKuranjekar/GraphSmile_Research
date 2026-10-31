# p2 summary: IEMOCAP-6 sensitivity to window, shift window, depth and loss weights

Facts only. "not measured" means no file in this checkout or in `server_pass2` holds the number.

## 1. Identity

- Branch: `experiments/p1b-provenance` (the pass-2 work was committed on this branch).
- Latest commit on the branch when this summary was written: `f284518` (`p2: fix stale stage 2 wording in results`). The summary commit is added on top of it.
- Base commit: `0905459` (`p1b: server files are byte-identical to README release (all four)`), the last commit before the p2 work.
- Note: the branch is already merged into `main` by merge commit `ab136c2`, which was made before this summary was written, not by this session.
- Pass id: `p2`. Tests whether changing the window size, shift window, depth or loss weights moves IEMOCAP-6 WF1 under the clean protocol (`--clean_valid --select_on valid`). Two-stage screen: stage 1 on 3 seeds, stage 2 on 5 seeds for promoted settings.

## 2. Code changes (`git diff 0905459 f284518`)

| File | Change |
|---|---|
| `scripts/run_pass2.sh` | New. Runs the sweep. Adds no flags to `run.py`. Variables: `PROTOCOLS` (default `clean`), `SEEDS` (default `2024 1 2`), `SETTINGS`. Skips the baseline clean run when `p0c_seed<seed>.json` exists. |
| `results/experiments/p2-results.md` | New. Results and decisions for stages 1 and 2. |

`run.py` and `scripts/run_all.sh` have no diff on this branch against the base (`git diff --stat 0905459 f284518 -- run.py scripts/run_all.sh` is empty).

No new flag was added in p2. The flags the script uses were added in earlier passes:

| Flag | Default | Effect | Added in |
|---|---|---|---|
| `--select_on {test,valid}` | `test` | Epoch chosen by test (original) or by valid F1 | `7b42cf2` (p0) |
| `--clean_valid` | off | Train on idx[split:], validate on idx[:split] | `ed1b437` (p1) |
| `--out_json PATH` | `''` | Writes selected-epoch per-class results to JSON | `7b42cf2` (p0) |
| `--seed` | 2024 | Random seed | `7b42cf2` (p0) |
| `--class_weight` | off | Inverse-frequency weighting (not used in p2) | `dd0b991` (p1) |

Checks:
- (a) Every new behaviour behind a default-off flag: the flags above default off except `--select_on` (default `test`, the original behaviour), which is the same as before. Not applicable to p2 itself because p2 adds no behaviour to `run.py`.
- (b) Default flags give the same result as main: **not tested** on GPU for this pass. A CPU synthetic-pickle check for p0 (`results/experiments/p0-protocol.md`) showed the default run prints the original line; that is not a result check.
- (c) Table III hyperparameters in `scripts/run_all.sh` unchanged: **yes**, the file has no diff on the branch since the base (or since `d12f026` on main). The p2 script hard-codes its own values (`--drop 0.2 --shift_win 19 --win 17 17 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7 --l2 1e-3`); these match `p0-protocol.md`.

## 3. Results

Clean protocol, test WF1 and per-class F1 (%), mean ± sample std (ddof=1).

Stage 1, seeds 2024, 1, 2. Source: `results/experiments/p2-results.md` (JSON files `p2_<setting>_clean_seed<seed>.json` and `p0c_seed<seed>.json` in `D:\NotesSem 5\Research_Sem5_Rashmi\server_pass2\experiments`, not copied into this repo).

| Setting | Seeds | Clean WF1 | Paper-protocol WF1 | Happy F1 | Excited F1 |
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

Stage 2, seeds 2024, 1, 2, 3, 4. Clean WF1 and Happy/Excited re-derived from the JSON files in `server_pass2` (`test_f1_emo`, `test_per_class_f1`).

| Setting | Seeds | Clean WF1 | Paper-protocol WF1 | Happy F1 | Excited F1 |
|---|---|---|---|---|---|
| p0c baseline | 5 | 68.26 ± 0.94 | 71.61 ± 0.36 (see section 6) | 45.62 ± 4.17 (memory file, not re-derived) | ~76 (std 0.01) |
| shift_5 | 5 | 68.52 ± 1.73 | 71.48, std not measured | ~48.4 (Δ +2.77 vs base, from p2-results.md) | ~75 (Δ −1.20) |
| shift_10 | 5 | 68.35 ± 1.58 | 71.49, std not measured | ~49.9 (Δ +4.31) | ~76 (Δ +0.01) |

Paper-protocol Happy and Excited F1 for every stage-2 setting: **not measured** (no JSON or per-class output found).

Selection metric: stage 1 promoted settings by **test** WF1 (see section 6). The stage-1 valid WF1 column in `p2-results.md` does not agree with test on the ranking.

## 4. Decision against the rule

Rule as written: keep only if the clean 5-seed mean beats the clean baseline by more than the baseline std, AND the paper-protocol number is not worse.

Stage 2 (baseline clean = 68.26 ± 0.94; threshold = 68.26 + 0.94 = 69.20; paper baseline = 71.61):

| Setting | Clean criterion | Paper criterion | Verdict |
|---|---|---|---|
| shift_5 | 68.52 − 68.26 = +0.26; needs > 0.94 → **fails** | 71.48 − 71.61 = −0.13 (worse) → **fails** | **reject** (not kept) |
| shift_10 | 68.35 − 68.26 = +0.09; needs > 0.94 → **fails** | 71.49 − 71.61 = −0.11 (worse) → **fails** | **reject** (not kept) |

Stage 1 (3-seed baseline 67.71 ± 0.64; threshold 68.35). Clean criterion only, because paper numbers were not measured. The combined rule cannot be applied at stage 1.

| Setting | Clean Δ vs 67.71 | Clears 0.64? | Verdict |
|---|---|---|---|
| shift_10 | +1.55 | yes | **inconclusive** (paper not measured; promoted, then rejected in stage 2) |
| shift_5 | +1.43 | yes | **inconclusive** (same) |
| win_5_5 | +0.67 | yes, by 0.03 | **inconclusive** (not promoted; paper not measured) |
| heter_4 | +0.39 | no | **inconclusive** |
| lambd_0.5 | +0.21 | no | **inconclusive** |
| lambd_0.3 | +0.17 | no | **inconclusive** |
| win_10_10 | +0.17 | no | **inconclusive** |
| win_10_2 | −1.19 | worse by more than 0.64 | **reject** |

Note: stage 1 used the 3-seed baseline (67.71 ± 0.64). The rule specifies the 5-seed baseline (68.26 ± 0.94). Against the 5-seed baseline, no stage-1 setting clears 0.94 on the 3-seed means except shift_10 (69.26 − 68.26 = +1.00) and shift_5 (69.14 − 68.26 = +0.88, fails). Those comparisons mix seed sets and are shown only for transparency.

**Overall: no setting is kept.**

## 5. Deviations

No change is kept. Every tested setting is a hyperparameter change from Table III (window, shift window, depth, loss weights). If any were kept it would be labelled "tuned variant, not paper setting". Architecture and code are unchanged.

## 6. Unverified or weak points

- **Paper-protocol stage 2 numbers have no source file here.** The values 71.61 ± 0.36 (base), 71.48 (shift_5) and 71.49 (shift_10) come from `p2-results.md`, which says they were read from each run's log `F-Score` line. No such log or JSON exists in `server_pass2` or in this repo. They are reported, not verified. The shift rows give no std.
- **Stage 2 candidates were chosen on test.** `p2-results.md` promotes shift_5 and shift_10 because their **test** gains (+1.55, +1.43) were the largest. Validation did not agree: the shift settings are +0.33 over baseline on valid, within one baseline std, and the valid ranking put win_10_10 and win_5_5 highest. This breaks the repo rule of no test-split selection. The stage 2 result still does not keep either setting, so no decision depends on it.
- **Stage 1 has 3 seeds**, and its keep verdicts are not final. The baseline std (0.64) comes from 3 seeds and is itself noisy.
- **win_5_5 is unstable.** Happy F1 std is 12.18 across 3 seeds, and per-seed Happy→Excited counts swing widely (73.7 ± 22.1). Its clean gain (+0.67) is seed-paired +1.26, −0.97, +1.73.
- **Single log not inspected:** `p2_win_5_5_clean_seed2024.log` was not read line by line; its JSON was used and is complete.
- **Valid and test disagree** for the stage 1 settings (section 3 above and `p2-results.md`).
- **Failed runs:** none reported. `p2-results.md` states that all 28 clean JSON files (24 stage 1 + 4 stage 2 shift seeds 3 and 4) are complete. The file count was confirmed by listing; the completeness check was not re-run per file in this summary.
- **Paper claims:** none new in p2.

## 7. Merge recommendation

Safe to merge (already on main through `ab136c2`; merging was not done in this session):
- `scripts/run_pass2.sh`: a script only, no change to `run.py` or to any default behaviour.
- `results/experiments/p2-results.md` and `results/experiments/p2-plan.md`: documentation.
- This file, `results/experiments/p2-summary.md`.

Stay on the branch / not merge:
- Nothing. No change was kept, so there is no tuned variant to merge.
- The per-run JSON and log files are not in the repo. They are in `D:\NotesSem 5\Research_Sem5_Rashmi\server_pass2`. Whether to commit them (size not measured) is a separate decision.

GPU runs still needed:
- **Stage 2 paper-protocol files**: if these runs were made on the server, their JSON files need copying into the repo so the paper numbers can be checked. Not measured here whether they exist.
- **To finish the combined rule for stage 1 settings** (only if you want win_5_5 decided rather than left inconclusive): paper protocol, seeds 2024, 1, 2, that is 3 runs:

```bash
SETTINGS="win_5_5" PROTOCOLS="paper" SEEDS="2024 1 2" GPU=0 bash scripts/run_pass2.sh
```

- Nothing else is needed for the current decisions, because shift_5 and shift_10 are already rejected on the rule and the rest fail the clean criterion.
