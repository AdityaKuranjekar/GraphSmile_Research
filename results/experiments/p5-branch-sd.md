# Pass 5: branch self-distillation (KD idea A), IEMOCAP-6, full model (tva)

Branch `experiments/p5-branch-sd`. Flags (all off by default): `--branch_sd α` (six aux heads on the
directional streams t<-v, v<-t, t<-a, a<-t, v<-a, a<-v learn τ²·KL from the detached fused-H logits),
`--sd_tau τ`, `--branch_mkl γ` (symmetric KL between the tv/ta/va branch heads). Inference uses only H.

Check: with flags off, per-epoch numbers match `main` exactly (seed 1, 4 epochs, clean protocol).

## Sweep (clean protocol, seeds 1, 2, 2024; selection on clean valid WF1)

| config | valid WF1 | test WF1 | test Acc | Happy F1 |
|---|---|---|---|---|
| baseline (p0c, same 3 seeds) | 73.37 ± 0.36 | 67.71 ± 0.64 | 67.82 ± 0.47 | 47.75 ± 1.69 |
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
| WF1 | 72.81 | 71.49 ± 0.40 | 70.09 ± 0.48 | 68.26 ± 0.94 | 68.51 ± 0.53 |
| Acc | 72.77 | 71.52 ± 0.38 | 70.30 ± 0.46 | 68.65 ± 1.21 | 68.85 ± 0.52 |
| Happy F1 | 63.09 | 59.01 ± 3.39 | 57.22 ± 3.42 | 45.62 ± 4.17 | 45.76 ± 6.02 |

Paper protocol = default split, best test epoch (`--select_on test`); clean = `--clean_valid --select_on valid`.
Paper-protocol numbers come from the "Test performance" line of each log (that mode writes no JSON).

## Decision: drop (do not keep)

- Clean: +0.25 WF1, well inside seed noise (std 0.5 to 0.9). The +1.0 seen on 3 seeds in the sweep shrank to +0.25 on 5 seeds.
- Paper protocol: −1.40 WF1, consistent on every seed (all 5 SD seeds are below all 5 baseline seeds).
- Keep rule needs clean up AND paper not worse, so it fails.
- Upside: clean std fell (0.94 → 0.53), so SD makes runs more stable; the peak (best-test) epoch is lower, which fits the aux loss acting as a regulariser.
- Mutual KL between branches did not help at any weight tried and hurt Happy when combined.

## Not done
- Test under noisy or missing audio/video (PDF step 7). That is where this idea is expected to help; the full-model clean setting shows no gain.
