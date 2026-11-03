# Gap plan: results summary (clean protocol, fixed code)

**Status:** provisional. Numbers marked *n=1* or *n=3* are not final. Source: the JSON files in `results/experiments/` on the GPU server, summarised by a script that reads `test_acc_emo` and `test_f1_emo` at the epoch chosen on validation (`--clean_valid --select_on valid`). Written 2026-11-02, after the fixed Phase 0 runs finished.

**Code:** branch `fix/loader-reshuffle-edge-weight`. The training order is now reshuffled each epoch, and `edge_weight` is in the optimizer (see `gap-fix-loader-edge-weight.md`). All numbers here use the fixed code. Numbers from the earlier buggy code (`p0c_*`, `p0_*`, `gap-phase1-*`) are not comparable.

## 1. Phase 0: clean baseline on fixed code (5 seeds each)

| Dataset | ACC | WF1 | Paper (test-selected) ACC / WF1 |
|---|---|---|---|
| IEMOCAP-6 | 68.16 ± 0.55 | 68.09 ± 0.42 | 72.77 / 72.81 |
| IEMOCAP-4 | 81.82 ± 1.15 | 81.84 ± 1.14 | 86.53 / 86.52 |
| MELD | 66.67 ± 0.12 | 65.48 ± 0.13 | 67.70 / 66.71 |
| CMU-MOSEI | 46.50 ± 0.23 | 43.99 ± 0.47 | 46.82 / 44.93 |

Mean ± sample std over seeds 2024, 1, 2, 3, 4. Paper numbers are from the original code, test-selected, and not directly comparable (the MOSEI test split here has 4,184 utterances; the paper's has 4,662).

The earlier buggy clean baseline for MOSEI was 45.13 / 43.16. The new value is about 1.4 ACC higher. The two standard deviations overlap, so treat this as a small shift, not a confirmed gain.

## 2. Phase 0c: no-text baseline (clean, fixed code)

| Dataset | Seeds | No-text ACC | No-text WF1 |
|---|---|---|---|
| IEMOCAP-4 | 1 (n=1, provisional) | 52.60 | 50.93 |
| MELD | 3 | 48.82 ± 0.21 | 43.92 ± 1.00 |
| CMU-MOSEI | 3 | 38.24 ± 0.15 | 28.97 ± 1.07 |
| IEMOCAP-6 | 0 | not run yet | — |

## 3. Text dominance, Δtext = full WF1 − no-text WF1 (clean, fixed code)

| Dataset | Full WF1 | No-text WF1 | Δtext | Status |
|---|---|---|---|---|
| IEMOCAP-4 | 81.84 | 50.93 | 30.91 | provisional (n=1) |
| MELD | 65.48 | 43.92 | 21.56 | n=3 no-text |
| CMU-MOSEI | 43.99 | 28.97 | 15.02 | n=3 no-text |
| IEMOCAP-6 | 68.09 | — | — | not run yet |

**Target (from the gap plan):** reduce Δtext by 6–7 WF1 points on every dataset, while full-model WF1 stays at or above baseline beyond seed std. The target was written against the buggy numbers, so the per-dataset targets must be recalculated from these clean Δtext values once every no-text run is in.

## 4. Phase 1.3: MOSEI modality contributions (clean, fixed code)

| Modalities (`--modals`) | ACC | WF1 | Seeds |
|---|---|---|---|
| text only (`t`) | 45.42 ± 0.50 | 43.57 ± 0.53 | 3 |
| text + audio (`ta`) | 46.14 ± 0.53 | 43.63 ± 0.53 | 3 |
| text + visual (`tv`) | 46.22 | 44.00 | 1 (provisional) |
| all three (`tva`, full model) | 46.50 ± 0.23 | 43.99 ± 0.47 | 5 |

On MOSEI, text alone reaches about 43.6 WF1, and the full model is 44.0. Audio and visual together add about 0.4 WF1. The differences between rows are about the size of the seed noise, so they are not yet conclusive. This is consistent with the evaluator's point that audio and visual are underused on MOSEI.

## 5. Phase 4.1: sentiment ablations (test-selected, seed 2024, one seed each)

Sentiment values are the maximum per-epoch `test_acc_sen` and `test_f1_sen` from each run's log. These runs use the repo default protocol, which selects by test sentiment F1, so the numbers are optimistic. They are not clean-protocol numbers. The earlier 14–27% figures were emotion-level values from the final summary line, and they are withdrawn.

| Dataset | Ablation | Sentiment ACC | Sentiment WF1 | Paper MSAC (Table VI, ‡) ACC / WF1 |
|---|---|---|---|---|
| MELD | No L_e | 73.75 | 73.70 | 73.98 / 73.84 |
| MELD | No L_e & L_o | 73.72 | 73.67 | 73.91 / 73.77 |
| CMU-MOSEI | No L_e | 67.69 | 65.78 | 66.59 / 65.91 |
| CMU-MOSEI | No L_e & L_o | 67.73 | 65.97 | 66.54 / 65.64 |

The MELD and MOSEI values agree with the paper's collapsed-row replacements to within about 1 point. The IEMOCAP-6 and IEMOCAP-4 sentiment ablations are not yet run.

## 6. Still running or queued

Relaunched 2 Nov 18:05 after the earlier jobs stopped mid-run at about 17:05, with no traceback:
- IEMOCAP-4 no-text, seeds 1 and 2 (GPU 0). Seed 2024 is done.
- IEMOCAP-6 no-text, seeds 2024, 1, 2 (GPU 4)
- MOSEI text + visual, seeds 1 and 2 (GPU 2); then Phase 1.4 window 0, 3 seeds

Still to run: IEMOCAP-6 and IEMOCAP-4 sentiment ablations (no result JSON is written, so the skip check does not cover them).

## 7. Blocked

- Phase 2 and Phase 3 code (`experiments/mosei-gap`, the flags for auxiliary regression, ordinal, focal, text dropout, unimodal heads and OGM) is not on GitHub. Phases 2 and 3 cannot start until it is pushed and rebased onto this branch.

## 8. Known limitations for the write-up

- The test split for MOSEI differs from the paper's (4,184 vs 4,662 utterances), and its label distribution differs from train.
- Only 3 seeds for the no-text and modality runs, so differences below about 1 WF1 point are noise.
- The sentiment ablation protocol uses the repo default, which selects by test sentiment F1, so those results are optimistic.
- The `optimizer params` log line overcounts because replaced `edge_weight` tensors stay in the optimizer. It doesn't affect training, but it should be fixed in a later code change.
