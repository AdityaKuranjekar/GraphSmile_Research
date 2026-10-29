# p1 plan (NO CODE WRITTEN, awaiting approval)

Baseline to beat (p0, 5 seeds, valid-selected): test WF1 70.82 ± 0.47, Happy 58.69 ± 3.18.

## Extra facts from the existing p0 logs (no GPU, no selection)
Best-test-epoch ("oracle", NOT used for any choice) per seed: 71.17, 71.88, 71.79, 71.93, 71.26 -> mean 71.61 ± 0.36.
So even with test selection we are about 1.2 below the paper (72.81). The remaining gap is not a selection effect.
Mean test WF1 over the last 10 epochs: 70.59, 67.49, 68.50, 71.39, 70.81. Late training is unstable for
some seeds, so epoch selection matters and a single epoch is noisy.

## Literature, what was and was not verified
| Claim | Status |
|---|---|
| DialogueRNN's IEMOCAP script uses inverse-frequency class weights in its loss, on by default | Verified in `declare-lab/conv-emotion/DialogueRNN/train_IEMOCAP.py` (weights 1/0.086747 ... 1/0.252668, `--class-weight` default True). Not verified: which index is which emotion, and how `MaskedNLLLoss` applies them. |
| Class weighting raises Happy F1 in GraphSmile | NOT verified. No source tested it on this model. It is a hypothesis. |
| GraphSmile's paper uses class weights | NOT checked. Our code uses plain `nn.NLLLoss`. |
| Happy/Excited are the most confused pair for contrastive ERC models | Verified in text of arXiv 2403.20289 (EACL): "similar emotions such as happy and excited are frequently misclassified as each other" (SPCL's confusion matrix). |
| EACL gives Happy 51.29 vs 43.96 for SPCL+CL (Excited 71.27 vs 66.72) | Verified from Table 3(a) of that paper, on their backbone and setup. Not comparable to our numbers. |
| EACL's excited/happy anchor similarity falls 0.77 -> 0.08 | Unverified (came from a model summary of the truncated page). Do not cite. |
| SPCL IEMOCAP WF1 69.74 | Unverified (aggregator site only). Do not cite. |
| Focal loss | Only the definition is verified (Lin et al., arXiv 1708.02002, an object-detection paper). No verified ERC evidence it helps Happy/Excited. |

## Proposed pass 1: inverse-frequency class-weighted NLL (single variant)
- Change: add `--class_weight` (default off). When on, the emotion loss uses `nn.NLLLoss(weight=w)` with
  `w_c = N / (K * n_c)` computed from the TRAIN split labels at runtime (not hard-coded). Sentiment and shift
  losses unchanged. Defaults and scripts untouched.
- Why this first: smallest change, directly targets the minority-class under-prediction, has a verified precedent
  in the model family we compare to (DialogueRNN), and is cheap to run.
- Risk: weighted F1 favours frequent classes, so Happy may rise while WF1 falls. We report both.
- Protocol: same command as p0 plus `--class_weight`; seeds 2024, 1, 2, 3, 4; `--select_on valid`; compare paired by seed.
- Decision rule fixed in advance: **kept** if mean test WF1 is >= 70.82 + 0.47 (one baseline std) OR Happy mean gain is
  > 3.18 with WF1 not lower than baseline; **rejected** if WF1 drops by more than 0.47 with no Happy gain; otherwise
  **inconclusive**. Only one variant is run, so test is read once and no variant is picked on test.
- Before GPU: CPU smoke test on the synthetic pickle (weights computed correctly, flag off reproduces p0 numbers
  bit-for-bit on the synthetic data).

## Not in pass 1
- EACL-style anchor/angle loss: needs the full paper and the authors' repo read first (appendix claims unverified).
- Focal loss, sqrt-inverse weights, label smoothing: possible later passes if pass 1 shows weighting matters.
- Looking for the remaining ~2-point gap in other classes (all classes are 0.8 to 4.4 below the paper).

## Status: approved and implemented (flag `--class_weight`, default off)
CPU smoke test (synthetic pickle, 3 epochs, seed 1; not a result):
- Flag off: output identical to the previous commit's `run.py` (only wall-clock time differs).
- Flag on: weights equal N / (K * n_c) from the train dialogues; they match an independent numpy computation.
- Decision: pending GPU results.

## IMPORTANT finding (found while implementing; affects p0 and p1)
`get_train_valid_sampler` builds `DistributedSampler(idx[split:])` and `DistributedSampler(idx[:split])`.
A DistributedSampler yields positions `0..len-1` of the list it is given, and `DataLoader` uses those as indices
into the full train set. So train draws dialogues `0..n-split-1` and valid draws `0..split-1`: **the validation
dialogues are a subset of the training dialogues.** Confirmed on the synthetic set (n=30: valid indices are contained in
train indices). This is existing paper-repo behaviour, not something we introduced.
Consequences:
- `valid_f1_emo` is a training-set metric, not held-out. That explains valid WF1 ~88 vs test ~71.
- `--select_on valid` therefore does not select on held-out data. It avoids selecting on test, but it is a weak
  and biased selector (it prefers fit to train). The p0 baseline (70.82 +/- 0.47) is still a valid
  "never looked at test" number, but its epoch choice is not a proper validation choice.
- Not yet verified on the real IEMOCAP pickle (n_train dialogues there); needs a one-line check on the GPU machine.
Proposed fix (NOT done, outside the approved scope): a `--clean_valid` flag, default off, that trains on
`idx[split:]` and validates on `idx[:split]`. Then rerun the baseline and pass 1 under it.

## Update: `--clean_valid` added (default off); plan reordered
- Change: with `--clean_valid`, train draws `idx[split:]` and valid draws `idx[:split]` (a small `DistributedSampler`
  subclass maps sampler positions back to dataset indices). `--class_weight` counts come from the dialogues the train
  loader actually draws, so it stays correct in both modes. Every run now logs one `split:` line with the number of
  valid dialogues that are also in train.
- CPU smoke test (synthetic, not a result): default output identical to the previous commit; default reports
  3 of 3 valid dialogues also in train; `--clean_valid` reports 0 of 3; class-weight counts under `--clean_valid`
  equal an independent numpy count over the train dialogues `idx[3:]`.
- Side observation (not changed): `setup_samplers` builds fresh samplers each epoch and calls `set_epoch` on them, but the
  `DataLoader`s keep their original samplers, so the shuffle order never changes between epochs. Same in both modes.
- New run order, all with `--clean_valid --select_on valid`, same 5 seeds, paper config, L2 1e-3:
  1. p0c: baseline (no class weight). This replaces p0 as the baseline to beat.
  2. p1c: same plus `--class_weight`. Decision rule as above, but with the p0c mean and std.
  p0's numbers (valid overlapped train) stay on record as the contaminated-selector baseline.

## Results: clean split (`--clean_valid --select_on valid`), 5 seeds, IEMOCAP-6, L2 1e-3
Real-data split check from the logs: 108 train dialogues, 12 valid dialogues, 0 valid dialogues also in train.
Commands: `run_clean.sh` as given (p0c = no `--class_weight`, p1c = `--class_weight`), GPU 1, seeds 2024 1 2 3 4.
Files: `results/experiments/p{0,1}c_seed*.json`, `results/logs/p{0,1}c_IEMOCAP_seed*.log`.
p1c class weights (train counts 459, 746, 1161, 854, 576, 1350): 1.869, 1.150, 0.739, 1.004, 1.489, 0.635
(Happy, Sad, Neutral, Angry, Excited, Frustrated).

Per seed, selected epoch / valid WF1 / test WF1 / Happy F1:

| Seed | p0c epoch | valid | test WF1 | Happy | p1c epoch | valid | test WF1 | Happy |
|---|---|---|---|---|---|---|---|---|
| 2024 | 17 | 73.39 | 67.96 | 46.3 | 37 | 74.29 | 68.53 | 44.6 |
| 1 | 16 | 73.01 | 68.19 | 47.4 | 36 | 73.92 | 68.71 | 44.9 |
| 2 | 89 | 73.72 | 66.98 | 49.6 | 16 | 73.48 | 67.89 | 50.0 |
| 3 | 59 | 74.10 | 68.64 | 38.6 | 31 | 74.65 | 67.61 | 38.0 |
| 4 | 71 | 73.98 | 69.54 | 46.3 | 87 | 73.83 | 69.14 | 46.4 |

Mean ± std (ddof=1), test, %:

| | p0c (baseline) | p1c (class weight) | Paper |
|---|---|---|---|
| Valid WF1 | 73.64 ± 0.45 | 74.03 ± 0.45 | |
| ACC | 68.65 ± 1.21 | 68.87 ± 0.63 | 72.77 |
| **WF1** | **68.26 ± 0.94** | **68.38 ± 0.62** | 72.81 |
| Happy | 45.62 ± 4.17 | 44.80 ± 4.34 | 63.09 |
| Sad | 79.89 ± 1.58 | 81.05 ± 1.24 | 83.16 |
| Neutral | 67.98 ± 2.22 | 68.60 ± 0.99 | 71.07 |
| Angry | 68.87 ± 1.56 | 68.20 ± 0.98 | 71.38 |
| Excited | 76.10 ± 1.24 | 76.36 ± 1.11 | 79.66 |
| Frustrated | 63.21 ± 2.47 | 62.73 ± 1.88 | 66.84 |

Happy->Excited (of 144) per seed: p0c 51, 46, 51, 83, 77 (mean 61.6); p1c 66, 65, 54, 83, 65 (mean 66.6).
Excited->Happy (of 299): p0c 34, 38, 24, 16, 15 (mean 25.4); p1c 24, 20, 51, 14, 19 (mean 25.6).

### Decision rule (fixed before the run, with p0c as baseline)
- Kept needs mean WF1 >= 68.26 + 0.94 = 69.20, or Happy gain > 4.17 with WF1 not lower. Observed WF1 +0.11 (paired
  per seed: +0.57, +0.52, +0.91, -1.03, -0.40); Happy -0.81 (paired: -1.6, -2.4, +0.4, -0.5, +0.1). Not met.
- Rejected needs a WF1 drop of more than 0.94 with no Happy gain. Observed +0.11. Not met.
- **Decision: inconclusive, with no evidence of a benefit.** The change is indistinguishable from zero on WF1 and Happy,
  Happy->Excited confusion did not fall (61.6 -> 66.6 on average), and the per-seed signs are mixed. The flag stays in
  the code, off by default. Inverse-frequency weighting is not a cure for the Happy/Excited confusion on this model.

### What the clean baseline shows (descriptive; test was not used to choose anything)
- **The honest baseline is much lower than before.** WF1 68.26 ± 0.94, Happy 45.62 ± 4.17, versus 70.82 ± 0.47 and
  58.69 with the overlapping "validation" in p0. The gap to the paper is now 4.5 WF1 and 17.5 Happy.
- **The loss is epoch selection, not training.** Best-test-epoch ("oracle", diagnostic only) is 71.22 for p0c and 71.01
  for p1c, close to p0's 71.61, and both runs train on 108 dialogues. The selected epoch gives 68.26. The final epoch
  alone gives 68.04 (p0c mean), so valid-based selection barely beats just taking epoch 120. In p0c, seeds 2024
  and 1 selected epochs 17 and 16, long before training converged.
- **The 12-dialogue validation set is too small to pick epochs.** Valid WF1 is 73.6 and flat across epochs and seeds
  (std 0.45), while test varies by about 3 points between epochs.
- **Implication:** the next lever is not another loss. The reported numbers depend mostly on how the epoch is
  chosen. Options for the next pass (all need your approval and need no test selection): average predictions or weights
  over the last K epochs, a larger validation share via k-fold on the train dialogues (select by pooled valid F1 over folds), or
  choosing the epoch by smoothed valid F1 over a window. Which one reproduces a protocol the paper could plausibly have
  used is unverified: the paper's selection protocol is not stated in anything I have checked.
