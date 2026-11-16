# Pass 5: branch self-distillation (KD idea A), IEMOCAP-6, full model (tva)

Branch `experiments/p5-branch-sd`. Flags (all off by default): `--branch_sd α` (six aux heads on the
directional streams t<-v, v<-t, t<-a, a<-t, v<-a, a<-v learn τ²·KL from the detached fused-H logits),
`--sd_tau τ`, `--branch_mkl γ` (symmetric KL between the tv/ta/va branch heads). Inference uses only H.

**Config note:** all p5 runs (baseline and SD) use the code defaults heter_n_layers 6 6 6, shift_win 12, lambd 1 1 1, not the paper IEMOCAP-6 config (7 7 7, 19, 1 1 0.7). Baseline and SD always share a config, so each comparison is like-for-like; for reference, the paper-config reproduction gives 71.17 WF1 (seed 2024, paper protocol) and 68.26 ± 0.94 (clean, 5 seeds).

Check: with flags off, per-epoch numbers match `main` exactly (seed 1, 4 epochs, clean protocol).

## Sweep (clean protocol, seeds 1, 2, 2024; selection on clean valid WF1)

| config | valid WF1 | test WF1 | test Acc | Happy F1 |
|---|---|---|---|---|
| baseline (matched config, same 3 seeds) | 74.12 ± 0.47 | 68.15 ± 1.16 | 68.48 ± 1.10 | 47.01 ± 4.23 |
| sd α=0.5 τ=2 | 73.78 ± 0.26 | 67.99 ± 1.18 | 68.21 ± 1.33 | 45.30 ± 1.37 |
| **sd α=1 τ=2 (selected)** | **74.37 ± 0.45** | 68.67 ± 0.54 | 68.82 ± 0.59 | 48.70 ± 6.26 |
| sd α=0.5 τ=4 | 73.85 ± 0.10 | 68.05 ± 0.64 | 68.23 ± 0.56 | 46.36 ± 2.88 |
| sd α=1 τ=4 | 74.13 ± 0.34 | 66.69 ± 0.67 | 67.59 ± 0.96 | 35.02 ± 2.36 |
| mkl γ=0.1 | 73.53 ± 0.37 | 68.04 ± 1.89 | 68.54 ± 1.56 | 46.73 ± 3.74 |
| mkl γ=0.5 | 73.58 ± 0.14 | 67.67 ± 0.96 | 67.84 ± 1.13 | 44.45 ± 0.81 |
| sd α=1 τ=2 + mkl γ=0.5 | 74.31 ± 0.23 | 67.92 ± 0.40 | 68.64 ± 0.54 | 39.78 ± 7.50 |

## Confirmation, 5 seeds (1, 2, 3, 4, 2024), same code, flags off vs α=1 τ=2

| | Paper (reported) | Baseline, paper protocol | SD, paper protocol | Baseline, clean | SD, clean |
|---|---|---|---|---|---|
| WF1 | 72.81 | 71.49 ± 0.40 | 70.09 ± 0.48 | 67.94 ± 1.68 | 68.51 ± 0.53 |
| Acc | 72.77 | 71.52 ± 0.38 | 70.30 ± 0.46 | 68.17 ± 1.84 | 68.85 ± 0.52 |
| Happy F1 | 63.09 | 59.01 ± 3.39 | 57.22 ± 3.42 | 45.93 ± 4.32 | 45.76 ± 6.02 |

Paper protocol = default split, best test epoch (`--select_on test`); clean = `--clean_valid --select_on valid`.
Paper-protocol numbers come from the "Test performance" line of each log (that mode writes no JSON).

## Decision: drop (do not keep)

- Clean (matched config): +0.57 WF1 and +0.39 valid WF1, inside seed noise (baseline std 1.68).
- Paper protocol: −1.40 WF1, consistent on every seed (all 5 SD seeds are below all 5 baseline seeds).
- Keep rule needs clean up AND paper not worse, so it fails.
- Upside: clean std fell (1.68 → 0.53), so SD makes runs more stable; the peak (best-test) epoch is lower, which fits the aux loss acting as a regulariser.
- Mutual KL between branches did not help at any weight tried and hurt Happy when combined.

## Not done
- Test under noisy or missing audio/video (PDF step 7). That is where this idea is expected to help; the full-model clean setting shows no gain.
