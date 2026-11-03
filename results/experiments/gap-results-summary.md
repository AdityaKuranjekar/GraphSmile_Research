# Gap plan: results summary (clean protocol, fixed code)

**Status:** Phase 0, Phase 0c (no-text), Phase 1.3 (modality-only) and Phase 4.1 (sentiment, MELD and MOSEI) are complete as of 3 Nov 2026. Still running: Phase 1.4 window 0 (3 seeds, MOSEI, GPU 2) and the IEMOCAP sentiment ablations (GPU 0). Source: JSON files in `results/experiments/` on the GPU server, summarised by `gap_summary.py` (`test_acc_emo`, `test_f1_emo` at the epoch chosen on validation, `--clean_valid --select_on valid`).

**Code:** branch `fix/loader-reshuffle-edge-weight`. The training order is reshuffled each epoch, and `edge_weight` is in the optimizer (see `gap-fix-loader-edge-weight.md`). All numbers here use the fixed code. Numbers from the earlier buggy code (`p0c_*`, `p0_*`, `gap-phase1-*`) are not comparable.

## 1. Phase 0: clean baseline on fixed code (5 seeds each)

| Dataset | ACC | WF1 | Paper (test-selected) ACC / WF1 |
|---|---|---|---|
| IEMOCAP-6 | 68.16 ± 0.55 | 68.09 ± 0.42 | 72.77 / 72.81 |
| IEMOCAP-4 | 81.82 ± 1.15 | 81.84 ± 1.14 | 86.53 / 86.52 |
| MELD | 66.67 ± 0.12 | 65.48 ± 0.13 | 67.70 / 66.71 |
| CMU-MOSEI | 46.50 ± 0.23 | 43.99 ± 0.47 | 46.82 / 44.93 |

Mean ± sample std over seeds 2024, 1, 2, 3, 4. Paper numbers are from the original code, test-selected, and not directly comparable (the MOSEI test split here has 4,184 utterances; the paper's has 4,662).

The earlier buggy clean baseline for MOSEI was 45.13 / 43.16. The new value is about 1.4 ACC higher. The two standard deviations overlap, so treat this as a small shift, not a confirmed gain. The buggy IEMOCAP-4 baseline was 82.48 ACC; the fixed one is 81.82, a small drop.

## 2. Phase 0c: no-text baseline (clean, fixed code)

| Dataset | Seeds | No-text ACC | No-text WF1 |
|---|---|---|---|
| IEMOCAP-6 | 3 | 54.45 ± 0.89 | 53.88 ± 1.14 |
| IEMOCAP-4 | 3 | 52.42 ± 0.22 | 51.22 ± 0.33 |
| MELD | 3 | 48.82 ± 0.21 | 43.92 ± 1.00 |
| CMU-MOSEI | 3 | 38.24 ± 0.15 | 28.97 ± 1.07 |

## 3. Text dominance, Δtext = full WF1 − no-text WF1 (clean, fixed code)

| Dataset | Full WF1 | No-text WF1 | Δtext | Target (cut of 6–7 points) |
|---|---|---|---|---|
| IEMOCAP-6 | 68.09 | 53.88 | 14.20 | about 7–8 |
| IEMOCAP-4 | 81.84 | 51.22 | 30.62 | about 24 |
| MELD | 65.48 | 43.92 | 21.56 | about 15 |
| CMU-MOSEI | 43.99 | 28.97 | 15.02 | about 8–9 |

Δtext is large on every dataset, so text dominance is real. No Phase 3 method has been run yet. The targets are the evaluator's 6–7 point cut applied to these clean values.

## 4. Phase 1.3: MOSEI modality contributions (clean, fixed code)

| Modalities (`--modals`) | ACC | WF1 | Seeds |
|---|---|---|---|
| text only (`t`) | 45.42 ± 0.50 | 43.57 ± 0.53 | 3 |
| text + audio (`ta`) | 46.14 ± 0.53 | 43.63 ± 0.53 | 3 |
| text + visual (`tv`) | 45.65 ± 0.81 | 43.47 ± 0.75 | 2 |
| all three (`tva`, full model) | 46.50 ± 0.23 | 43.99 ± 0.47 | 5 |

On MOSEI, text alone reaches 43.6 WF1, and the full model reaches 44.0. Audio and visual add no more than about 0.4 WF1 together, which is within seed noise. The model is not using audio or visual in any meaningful way on MOSEI. This matches the evaluator's point that audio and visual are underused.

## 5. Phase 4.1: sentiment ablations (test-selected, seed 2024, one seed each)

Sentiment values are the maximum per-epoch `test_acc_sen` and `test_f1_sen` from each run's log. These runs use the repo default protocol, which selects by test sentiment F1, so the numbers are optimistic. They are not clean-protocol numbers. The earlier 14–27% figures were emotion-level values from the final summary line, and they are withdrawn.

| Dataset | Ablation | Sentiment ACC | Sentiment WF1 | Paper MSAC (Table VI, ‡) ACC / WF1 |
|---|---|---|---|---|
| MELD | No L_e | 73.75 | 73.70 | 73.98 / 73.84 |
| MELD | No L_e & L_o | 73.72 | 73.67 | 73.91 / 73.77 |
| CMU-MOSEI | No L_e | 67.69 | 65.78 | 66.59 / 65.91 |
| CMU-MOSEI | No L_e & L_o | 67.73 | 65.97 | 66.54 / 65.64 |
| IEMOCAP-6 | No L_e | 83.24 | 83.33 | 84.41 / 84.53 |
| IEMOCAP-6 | No L_e & L_o | 82.87 | 82.91 | 83.30 / 83.54 |
| IEMOCAP-4 | No L_e | 85.68 | 85.70 | 85.47 / 85.40 |
| IEMOCAP-4 | No L_e & L_o | 84.73 | 84.70 | 85.58 / 85.55 |

All four IEMOCAP runs completed 120 epochs with no tracebacks. Across all four datasets, the replacement sentiment values are within about one point of the paper's MSAC ablation rows, except IEMOCAP-6 `No L_e`, which is about 1.2 ACC below the paper.

## 6. Still running

- Phase 1.4 window 0, 3 seeds, MOSEI, GPU 2 (queued after text+visual).

## 7. Blocked

- Phase 2 and Phase 3 code (`experiments/mosei-gap`, the flags for auxiliary regression, ordinal, focal, text dropout, unimodal heads and OGM) is not on GitHub. Phases 2 and 3 cannot start until it is pushed and rebased onto this branch.

## 8. Known limitations for the write-up

- The MOSEI test split differs from the paper's (4,184 vs 4,662 utterances), and its label distribution differs from train.
- No-text and modality rows use 3 seeds (MOSEI text+visual uses 2), so differences below about 1 WF1 point are noise.
- The sentiment ablation protocol selects by test sentiment F1, so those results are optimistic.
- The `optimizer params` log line overcounts because replaced `edge_weight` tensors stay in the optimizer. This does not affect training, but it should be fixed in a later code change.
