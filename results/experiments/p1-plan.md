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
