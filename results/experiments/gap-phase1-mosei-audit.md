# Gap plan, Phase 1.1 and 1.2: CMU-MOSEI class counts and feature audit

**Status:** complete (CPU only, no training).
**Script:** `scripts/gap_phase1_mosei_audit.py`
**Raw output:** `results/experiments/gap-phase1-mosei-audit.json`
**Pickle:** `CMUMOSEI7_path` in `run.py` (`/DATA/rashmi_2521cs34/data/CMUMOSEI7.pkl` on the GPU server)
**Run date:** 2026-11-01, on the GPU server, CPU only.

## 1.1 Class counts

| Class | Train+val count | Train+val share | Test count | Test share |
|---|---|---|---|---|
| Highly Negative | 799 | 4.2% | **22** | 0.5% |
| Negative | 2,093 | 11.0% | 159 | 3.8% |
| Weakly Negative | 3,007 | 15.8% | 602 | 14.4% |
| Neutral | 4,173 | 21.9% | 921 | 22.0% |
| Weakly Positive | 5,896 | 31.0% | 1,666 | 39.8% |
| Positive | 2,530 | 13.3% | 692 | 16.5% |
| Highly Positive | 552 | 2.9% | 122 | 2.9% |

Totals: train+val 19,050 utterances in 2,595 dialogues; test 4,184 utterances in 693 dialogues.

Sentiment, 3-way (neg / zero / pos):
- Train+val: 5,899 / 4,173 / 8,978 (31.0% / 21.9% / 47.1%)
- Test: 783 / 921 / 2,480 (18.7% / 22.0% / 59.3%)

### Findings
- **The test set is distributed differently from train.** Negative classes are 3 to 8 times rarer in test, and Weakly Positive is 9 points more common.
- **Highly Negative has only 22 test utterances.** A per-class F1 of 0.00 there is roughly "0 hits out of 22". Highly Positive (122 utterances) is also too small for a stable per-class F1. Per-class numbers need bootstrap confidence intervals.
- **The validation split is a random 10% of train**, so it has the train distribution, not the test one. Epoch selection on validation may reward different behaviour than the test set does. This is hypothesis H7, to be checked against the Phase 0 runs.

## 1.2 Feature audit

| Check | Text | Audio | Visual |
|---|---|---|---|
| Feature size (per utterance) | 1024 | 384 | 35 |
| Rows (utterances) | 23,234 | 23,234 | 23,234 |
| Missing dialogues | 0 | 0 | 0 |
| Rows with NaN | 0 | 0 | 0 |
| Rows with Inf | 0 | 0 | 0 |
| All-zero rows | 0 | 0 | **168 (0.7%)** |
| Dialogues whose length differs from labels | 0 | 0 | 0 |

Train and test dialogue IDs do not overlap (0 shared).

### Findings
- **Alignment is clean.** Every modality has one row per labelled utterance.
- **No NaN or Inf values** in any modality.
- **Visual has 168 all-zero rows.** Probably frames where face tracking failed. Small, but a real source of noise. Decide whether to mask or keep them.
- **Feature sizes:** text 1024 (consistent with RoBERTa-large), visual 35 (consistent with FACET). Audio 384 does not match COVAREP (74). The paper says librosa, but the file does not confirm the extractor. Check the dataset documentation before stating the audio source.

## Split size against the paper (Table II)

| | Paper | Our pickle | Difference |
|---|---|---|---|
| Train+val dialogues | 2,549 | 2,595 | +46 |
| Train+val utterances | 18,198 | 19,050 | +852 |
| Test dialogues | 646 | 693 | +47 |
| Test utterances | 4,662 | 4,184 | **-478** |

Our test set has fewer utterances than the paper's but more dialogues. **Our 45.10 is computed on a different test set than the paper's 46.82.** This must be stated as a limitation. The paper's exact split cannot be recovered from this pickle.

## Dialogue length (input to Phase 1.4)

- Train+val: mean 7.34, median 5, max 98; 11% of dialogues have one utterance.
- Test: mean 6.04, median 4, max 52; 13.9% have one utterance.
- A window of 5 already covers most dialogues. The flat window curve in the paper is therefore not explained by short dialogues. Phase 1.4 (GPU) decides whether context helps.

## Hypothesis status after Phase 1.1 and 1.2

| # | Hypothesis | Status |
|---|---|---|
| H1 | Minority-class collapse | **Partly supported.** Train imbalance is real. Test has only 22 Highly Negative utterances, so per-class F1 there cannot be interpreted. |
| H2 | Context does not help | **Unclear.** Dialogues are long enough to use context. Phase 1.4 decides. |
| H3 | Features broken or misaligned | **Mostly refuted.** Alignment and values are clean. Minor issues: 168 zero visual rows; split differs from paper. |
| H4 | Test-selected protocol inflates results | **Pending Phase 0.** |
| H5 | MOSEI under-tuned | **Untested.** |
| H6 | Continuous score discarded | **Confirmed by code** (`dataloader.py` bins the score; the raw value is not kept). |
| H7 | Train and test label distributions differ | **Supported by counts.** Affects epoch selection and class weighting. |

## Consequences for later phases
1. Focal loss and class weights are computed from train counts but judged on test. An improvement on Highly Negative may not show in test WF1 at all.
2. Every per-class number in the evaluator package needs a confidence interval, or it is dropped.
3. Limitations section: our test set differs from the paper's (4,184 vs 4,662 utterances), and the label distribution differs between train and test.
