# Experiment index (read this first)

Every run lives in `results/logs/<phase>_*.log`. Per-seed JSON results are in this folder. Seeds are 2024, 1, 2, 3, 4 unless noted. All runs here are finished; none is running.

## Two protocols

- **Paper protocol (the repo default).** The epoch is the one with the best *test* F1. This is the protocol behind the headline tables, and it is the one to compare with the paper. It is optimistic, because the test split picks the epoch.
- **Clean protocol.** `--clean_valid --select_on valid`. The validation set no longer overlaps the training set, and the epoch is chosen on validation F1. Numbers are lower. Disclose them next to the headline numbers.

In the original code the validation dialogues are a subset of the training dialogues. `--clean_valid` fixes that. Runs without it use the overlapping validation set.

## Phases

| Phase | What it ran | Logs | Protocol | Result |
|---|---|---|---|---|
| (unprefixed) | Main recreation: 4 datasets × 12 variants (full, modality, loss and structure ablations) | `IEMOCAP_*`, `IEMOCAP4_*`, `MELD_*`, `CMUMOSEI7_*` (48), plus `IEMOCAP_full_l2_1e-4` | paper, seed 2024 | Tables IV to VII in `REPRODUCTION.md` |
| p0 (IEMOCAP-6) | Seed variance, original split | `p0_IEMOCAP_seed*` (5) | valid-selected, overlapping validation | WF1 70.82 ± 0.47 |
| p0 (other datasets) | Seed variance, clean split | `p0_iemocap4_*`, `p0_meld_*`, `p0_mosei_*` (15) | clean | WF1 82.48 (IEMOCAP-4), 65.38 (MELD), 43.16 (CMU-MOSEI) |
| p0c | **Clean baseline**, IEMOCAP-6 | `p0c_IEMOCAP_seed*` (5) | clean | WF1 **68.26 ± 0.94** |
| p1c | Class-weighted loss (`--class_weight`), IEMOCAP-6 | `p1c_IEMOCAP_seed*` (5) | clean | WF1 68.38 ± 0.62, **rejected** |
| p2 | Screening of 8 settings against p0c | `p2_*_clean_*` (28), `p2_*_paper_*` (15) | clean (28), paper (15) | **All rejected** |
| p3 | Why the MSAC rows with the emotion loss off collapse | `p3_*` (9 runs, plus `p3_driver.log`) | paper | Explained: epoch-selection artifact |

Details per phase: `p0-protocol.md`, `p1-plan.md`, `p1-summary.md`, `p1b-provenance.md`, `p2-plan.md`, `p2-results.md`, `p2-summary.md`, `p3-collapse-check.md`, `p3-collapse-check-summary.md`.

## p2 in detail

- Stage 1: clean protocol, seeds 2024, 1, 2, for `win_5_5`, `win_10_10`, `win_10_2`, `shift_5`, `shift_10`, `heter_4`, `lambd_0.5`, `lambd_0.3` (24 runs).
- Stage 2: seeds 3 and 4 for `shift_5` and `shift_10` (4 clean runs), plus 15 paper-protocol runs (base, `shift_5`, `shift_10`, 5 seeds each).
- Decision rule: keep a setting only if its 5-seed mean beats the baseline mean by more than the baseline std (0.94), and it does not lose under the paper protocol.
- Outcome: `shift_5` gained +0.26 and `shift_10` +0.09 on 5 seeds, below the margin. Paper protocol WF1: base 71.61, `shift_5` 71.48, `shift_10` 71.49. None was adopted.
- Caveat: the stage-1 screening looked at test WF1 at the validation-selected epoch. Validation scores on 12 dialogues did not track test scores. The two-stage design (5 seeds, margin of one std) limits the risk, but the screening was test-informed.

## What was accepted into the code and what was rejected

- **Accepted (all behind flags, default off or backward compatible):** `--clean_valid`, `--seed`, `--select_on`, `--out_json`, `--modals`, `--no_res`, `--no_fcres`, `--no_seg`, CPU fallback. These fix the validation overlap or enable the paper's ablations. They do not raise the headline accuracy.
- **Rejected:** L2 = 1e-3, class weights, `shift_5`, `shift_10`, `win_5_5`, `win_10_10`, `win_10_2`, `heter_4`, `lambd_0.5`, `lambd_0.3`. None beat the baseline by more than one std. No tuned setting was merged, so the main tables stay at the Table III settings.

## Open items

- Re-run the two MSAC ablation rows (`no_Le`, `no_Le_Lo`) with `--classify sentiment` (8 runs, about 1.5 h).
- Optional: depth and window sweeps (`scripts/run_sweeps.sh`), more seeds for MELD and CMU-MOSEI.
- The Happy-class gap on IEMOCAP-6 (about 55 against the paper's 63) is explained by a one-sided Happy to Excited confusion and 144 test utterances. It is not fixed.
