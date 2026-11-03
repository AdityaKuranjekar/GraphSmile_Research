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

## 5. Phase 4.1: sentiment ablations (not yet valid)

Both MELD and MOSEI sentiment runs (`no_Le` and `no_Le_Lo`, seed 2024) finished, but their JSON files were not written. Their log numbers (MELD ACC 14–15, MOSEI ACC 24–27) look like emotion-level results rather than sentiment. **Do not report these numbers until the classification report in each log has been checked for 3 sentiment classes.**

## 6. Still running or queued

- IEMOCAP-4 no-text, seeds 1 and 2 (GPU 0)
- IEMOCAP-6 no-text, 5 seeds, then IEMOCAP sentiment ablations (GPU 0)
- MOSEI text + visual, seeds 1 and 2; then Phase 1.4 window 0 (GPU 2)

## 7. Blocked

- Phase 2 and Phase 3 code (`experiments/mosei-gap`, the flags for auxiliary regression, ordinal, focal, text dropout, unimodal heads and OGM) is not on GitHub. Phases 2 and 3 cannot start until it is pushed and rebased onto this branch.

## 8. Known limitations for the write-up

- The test split for MOSEI differs from the paper's (4,184 vs 4,662 utterances), and its label distribution differs from train.
- Only 3 seeds for the no-text and modality runs, so differences below about 1 WF1 point are noise.
- The sentiment ablation protocol uses the repo default, which selects by test sentiment F1, so those results are optimistic.
- The `optimizer params` log line overcounts because replaced `edge_weight` tensors stay in the optimizer. It doesn't affect training, but it should be fixed in a later code change.
