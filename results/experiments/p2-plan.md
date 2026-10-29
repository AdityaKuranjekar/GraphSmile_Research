# p2 plan: epoch selection (NO CODE WRITTEN, awaiting approval)

Baseline to beat (p0c, clean split, valid-selected, 5 seeds): test WF1 68.26 ± 0.94, Happy 45.62 ± 4.17.
Reference points from the same logs (diagnostic, not used to choose anything): final epoch 68.04; mean of last 10
epochs 68.57; best-test-epoch 71.22.

## Literature and protocol facts, verified status
| Claim | Status |
|---|---|
| GraphSmile paper: validation = "10% of dialogues from the combination of training and validation sets" (IV-B); 120 train+val and 31 test dialogues for IEMOCAP | Verified verbatim (arXiv 2407.21536v2, via page text). Matches our 108 / 12 split. |
| The paper says how the reported number is chosen (best epoch, final epoch, mean of runs, seeds, std) | **Not stated** in the page text read (Sections I to V-A verbatim). Epochs, early stopping and schedule are also not stated. Sections V-D onward and the authors' repo were not read. |
| The authors' code selects by test or by validation | Not verified. This repo's `run.py` (inherited) selects by test F1 and its validation split overlaps train. Whether the authors' own runs used that is unknown. |
| Stochastic Weight Averaging (average weights along the SGD trajectory) improves generalisation | Verified as a published claim (Izmailov et al., arXiv 1803.05407), on image classification (CIFAR, ImageNet). **No verified evidence for ERC or for this model.** Treated as a hypothesis. |
| Random speaker-overlapping splits can inflate IEMOCAP scores | Verified as a claim of Antoniou et al. (ICASSP 2023) for speech emotion recognition. Whether our train/valid dialogue split shares speakers is not checked. |
| Averaging or k-fold choice is what the paper used | No source. Do not claim it. |

## Consequence for how we report
Since the paper's selection protocol is unknown, a fair comparison table needs three labelled lines, not one:
1. valid-selected (our clean protocol), 2. fixed-final-epoch, 3. best-test-epoch marked "oracle, optimistic, for comparison with an
unknown protocol only". Line 3 is computed from the logs after the fact and never used to choose a setting.

## Proposed pass 2: last-K-epoch weight averaging, no epoch selection
- Change: `--avg_last K` (default 0 = off). From epoch `epochs-K+1` to `epochs`, keep a running mean of the model
  parameters (and buffers, if the model has any, to be checked in code). After the last epoch, load the averaged weights
  and evaluate valid and test once. `--select_on` is ignored in this mode, so test is never used to choose.
- K = 10, fixed in advance (not tuned). K=10 is about 8% of the training length; no source says it is right.
- Protocol: p0c command plus `--avg_last 10`; seeds 2024, 1, 2, 3, 4; `--clean_valid`. The same run also logs the
  final-epoch result, so reference line 2 comes for free.
- Decision rule fixed now: **kept** if mean test WF1 >= 68.26 + 0.94 = 69.20, or Happy mean gain > 4.17 with WF1 not lower;
  **rejected** if mean WF1 drops by more than 0.94 with no Happy gain; otherwise **inconclusive**.
- Before GPU: CPU smoke test on the synthetic pickle: flag off reproduces the previous output exactly; with the flag on,
  the averaged weights equal an independent mean of the saved epoch weights.
- Expected honest limit: even the best-test-epoch line (71.22) is 1.6 below the paper, so selection fixes cannot by
  themselves close the whole gap, and cannot close the Happy gap (45.6 vs 63.1) unless Happy is mostly a selection effect.
  That is checkable first, for free: compute oracle Happy from the p0c logs.

## Not proposed (and why)
- 5-fold CV over the 120 train dialogues: sounder, but 5x the runs per seed and a retraining step; keep as pass 3 if averaging fails.
- Smoothed-valid selection: valid WF1 is flat (std 0.45) on 12 dialogues, so smoothing a flat signal helps little.
- Anything resembling picking K or the epoch from test curves.
