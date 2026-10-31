# Reproduction of GraphSmile: paper vs our runs

This document compares our re-run of GraphSmile with the numbers reported in the paper
*"Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for
multimodal emotion recognition"* (IEEE TPAMI, 2025, doi:10.1109/TPAMI.2025.3581236).

Every "ours" number comes from a raw log in `results/logs/`. The index of all logs, with
their configuration, is in `results/proof_index.md`. A paper-style PDF of the same tables is
in `docs/paper_vs_reproduced.pdf`.

## 1. Status of each paper table and figure

| Item | What it is | Status |
|---|---|---|
| Table I | Merging scheme of emotion labels | Built into `dataloader.py`. Not a result, so nothing to reproduce. |
| Table II | Dataset statistics | Paper values shown. Not a result. |
| Table III | Hyperparameters | Matches `README.md`. L2 = 1e-3 as stated in the paper. |
| Table IV | MERC results | **GraphSmile rows reproduced** for all four datasets. Baseline rows (COGMEN, Joyful, DER-GCN, etc.) are paper values only, not re-run. |
| Table V | MSAC results | **† rows reproduced** (Table III settings). ‡ rows (optimized settings) are paper values only, not reproduced. |
| Table VI | Ablations | **All rows run.** Two MSAC rows are not verified (see section 5). |
| Table VII | Time and memory | Reproduced, on a different GPU (see section 4). |
| Fig. 1 to 3 | Architecture and examples | Diagrams. Nothing to reproduce. |
| Fig. 4 | Accuracy against network depth | **Not run.** Sweep script exists (`scripts/run_sweeps.sh`). |
| Fig. 5 | Accuracy against window size | **Not run.** Sweep script exists (`scripts/run_sweeps.sh`). |
| Fig. 6 | t-SNE of embeddings | **Not done.** The code does not save features. |
| Fig. 7 | Confusion matrices (IEMOCAP-6, MELD) | **Not compared.** The logs contain the matrices. |

## 2. Setup used for our runs

- **Code:** this repository, branch `claude/gifted-galileo-l5kpj3`, with the changes described in `RUNBOOK.md`.
- **Hardware:** NVIDIA A100-PCIE-40GB on a shared server. Other users ran jobs on the same GPU.
- **Software:** Python 3.10, PyTorch 2.0.0 with CUDA 11.7, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1.
- **Data:** the four feature pickles from the dataset release (`README.md`), not included in this repository.
- **Seeds:** the code fixes the seed to 2024. The main tables use that one seed. The IEMOCAP-6 base was also run with five seeds (2024, 1, 2, 3, 4); see "Five-seed paper-protocol base" in section 4. All other datasets and ablations are single-seed, so their run-to-run spread is not measured.

## 3. Protocol

The code reports the epoch with the best **test** emotion F1 (`run.py`). Every value in our
tables is from that one epoch. The paper's exact protocol is not stated, so the test-set
selection may differ from theirs.

- **Emotion (MERC):** ACC and weighted F1 at the best-F1 epoch. Per-emotion F1 is from the classification report of that epoch.
- **Sentiment (MSAC):** the sentiment head's ACC and weighted F1 at the same epoch.
- **Ablations:** the same protocol, one run per variant.

Paper per-class values are F1 scores in %. Ours are the same.

### Clean-protocol check (disclosure)

The tables below use the original code's default protocol (section 3), so they are the like-for-like
comparison with the released code. We also ran a stricter check for IEMOCAP-6 only, and report it here so
the dependence on epoch selection is visible.

- **Validation overlap in the original code.** `get_train_valid_sampler` builds `DistributedSampler`s from sliced
  lists, which yield list positions, not the dialogue indices. The "validation" dialogues are therefore also in the
  training set, and the original validation F1 is a training metric. `--clean_valid` fixes this (108 train, 12
  validation dialogues, 0 overlap). All new flags default off, so default runs are unchanged.
- **Results on IEMOCAP-6 (WF1, 5 seeds, mean ± std).** Clean split, epoch chosen on validation: 68.26 ± 0.94.
  Original overlapping split, epoch chosen on validation: 70.82 ± 0.47. Single run, seed 2024, best test epoch
  (main tables): 71.17. Final epoch 68.04, last-10-epoch mean 68.57, best-test-epoch oracle 71.22.
- **Class-weighted loss** (`--class_weight`, default off): 68.38 ± 0.62 WF1, Happy 44.80 ± 4.34. No measurable gain.
- **Reading.** The training run reaches about 71 WF1 at its best test epoch, but the epoch picked without test
  gives about 68, so most of the gap to 71 is epoch selection. The paper does not state how it chose its epoch.
- Details: `results/experiments/`, `docs/comparison_clean_protocol.pdf`, `docs/paper_vs_reproduced_updated.pdf`.

## 4. Results

### Table IV: MERC, GraphSmile (paper vs ours, %)

**IEMOCAP-6**

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Happy | 63.09 | 54.87 | -8.22 |
| Sad | 83.16 | 82.14 | -1.02 |
| Neutral | 71.07 | 70.85 | -0.22 |
| Angry | 71.38 | 68.92 | -2.46 |
| Excited | 79.66 | 78.04 | -1.62 |
| Frustrated | 66.84 | 66.22 | -0.62 |
| **ACC** | **72.77** | **71.23** | **-1.54** |
| **WF1** | **72.81** | **71.17** | **-1.64** |

**IEMOCAP-4**

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Happy | 87.94 | 86.12 | -1.82 |
| Sad | 88.67 | 88.45 | -0.22 |
| Neutral | 85.41 | 84.70 | -0.71 |
| Angry | 84.73 | 85.22 | +0.49 |
| **ACC** | **86.53** | **86.00** | **-0.53** |
| **WF1** | **86.52** | **85.98** | **-0.54** |

**MELD**

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Neutral | 80.35 | 80.27 | -0.08 |
| Surprise | 59.11 | 58.67 | -0.44 |
| Fear | 18.18 | 19.18 | +1.00 |
| Sadness | 42.46 | 43.45 | +0.99 |
| Joy | 64.99 | 64.56 | -0.43 |
| Disgust | 32.43 | 29.36 | -3.07 |
| Anger | 53.67 | 53.46 | -0.21 |
| **ACC** | **67.70** | **67.59** | **-0.11** |
| **WF1** | **66.71** | **66.55** | **-0.16** |

**CMU-MOSEI (7 classes)**

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Highly Negative | 0.00 | 0.00 | +0.00 |
| Negative | 28.79 | 30.28 | +1.49 |
| Weakly Negative | 43.50 | 45.19 | +1.69 |
| Neutral | 39.19 | 42.81 | +3.62 |
| Weakly Positive | 57.30 | 53.40 | -3.90 |
| Positive | 35.95 | 33.39 | -2.56 |
| Highly Positive | 6.25 | 3.17 | -3.08 |
| **ACC** | **46.82** | **45.10** | **-1.72** |
| **WF1** | **44.93** | **43.96** | **-0.97** |

### Table V: MSAC, sentiment with Table III settings (†, %)

| Dataset | Class | Paper | Ours | Δ |
|---|---|---|---|---|
| IEMOCAP-6 | Negative | 86.87 | 87.11 | +0.24 |
| IEMOCAP-6 | Neutral | 69.06 | 70.26 | +1.20 |
| IEMOCAP-6 | Positive | 89.21 | 89.66 | +0.45 |
| IEMOCAP-6 | **ACC** | **83.73** | **83.67** | **-0.06** |
| IEMOCAP-6 | **WF1** | **83.66** | **83.82** | **+0.16** |
| IEMOCAP-4 | Negative | 87.76 | 87.73 | -0.03 |
| IEMOCAP-4 | Neutral | 82.93 | 82.21 | -0.72 |
| IEMOCAP-4 | Positive | 87.23 | 85.71 | -1.52 |
| IEMOCAP-4 | **ACC** | **85.79** | **85.26** | **-0.53** |
| IEMOCAP-4 | **WF1** | **85.71** | **85.18** | **-0.53** |
| MELD | Neutral | 79.15 | 79.09 | -0.06 |
| MELD | Positive | 64.84 | 64.18 | -0.66 |
| MELD | Negative | 70.08 | 70.07 | -0.01 |
| MELD | **ACC** | **73.60** | **73.45** | **-0.15** |
| MELD | **WF1** | **73.40** | **73.24** | **-0.16** |
| CMU-MOSEI | Negative | 56.14 | 58.29 | +2.15 |
| CMU-MOSEI | Neutral | 39.80 | 42.10 | +2.30 |
| CMU-MOSEI | Positive | 78.18 | 76.65 | -1.53 |
| CMU-MOSEI | **ACC** | **66.83** | **65.27** | **-1.56** |
| CMU-MOSEI | **WF1** | **65.69** | **65.61** | **-0.08** |

The ‡ (optimized-setting) rows are not reproduced.

### Table VI: ablations (ACC / WF1, %)

MERC columns are emotion ACC / WF1. MSAC columns are sentiment ACC / WF1.

| Variant | Dataset | Paper MERC | Ours MERC | Paper MSAC | Ours MSAC |
|---|---|---|---|---|---|
| GraphSmile (full) | IEMOCAP-6 | 72.77 / 72.81 | 71.23 / 71.17 | 84.97 ‡ / 85.09 ‡ | 83.67 / 83.82 |
| GraphSmile (full) | IEMOCAP-4 | 86.53 / 86.52 | 86.00 / 85.98 | 87.49 ‡ / 87.48 ‡ | 85.26 / 85.18 |
| GraphSmile (full) | MELD | 67.70 / 66.71 | 67.59 / 66.55 | 74.44 ‡ / 74.31 ‡ | 73.45 / 73.24 |
| GraphSmile (full) | CMU-MOSEI | 46.82 / 44.93 | 45.10 / 43.96 | 67.73 ‡ / 66.73 ‡ | 65.27 / 65.61 |
| No L_s | IEMOCAP-6 | 71.53 / 71.52 | 70.92 / 70.83 | not comparable | not comparable |
| No L_s | IEMOCAP-4 | 85.15 / 85.15 | 85.37 / 85.36 | not comparable | not comparable |
| No L_s | MELD | 67.39 / 66.30 | 67.43 / 66.33 | not comparable | not comparable |
| No L_s | CMU-MOSEI | 45.08 / 43.37 | 45.43 / 44.43 | not comparable | not comparable |
| No L_o | IEMOCAP-6 | 71.41 / 71.42 | 70.92 / 71.02 | 84.10 ‡ / 84.18 ‡ | 82.26 / 82.45 |
| No L_o | IEMOCAP-4 | 84.52 / 84.50 | 84.84 / 84.79 | 85.26 ‡ / 85.23 ‡ | 83.46 / 83.25 |
| No L_o | MELD | 67.62 / 66.59 | 67.59 / 66.58 | 74.18 ‡ / 74.10 ‡ | 73.52 / 73.32 |
| No L_o | CMU-MOSEI | 45.20 / 44.03 | 45.53 / 44.09 | 66.71 ‡ / 66.20 ‡ | 66.73 / 65.03 |
| No L_s & L_o | IEMOCAP-6 | 70.67 / 70.49 | 70.06 / 70.01 | not comparable | not comparable |
| No L_s & L_o | IEMOCAP-4 | 84.41 / 84.40 | 85.05 / 85.04 | not comparable | not comparable |
| No L_s & L_o | MELD | 67.55 / 66.43 | 67.51 / 66.40 | not comparable | not comparable |
| No L_s & L_o | CMU-MOSEI | 44.29 / 43.54 | 44.74 / 43.86 | not comparable | not comparable |
| No Res | IEMOCAP-6 | 61.98 / 61.35 | 61.37 / 61.34 | 78.56 ‡ / 78.25 ‡ | 76.96 / 76.02 |
| No Res | IEMOCAP-4 | 79.64 / 79.58 | 80.28 / 80.21 | 81.02 ‡ / 80.83 ‡ | 81.34 / 81.13 |
| No Res | MELD | 52.07 / 46.24 | 48.24 / 47.36 | 58.08 ‡ / 56.61 ‡ | 55.67 / 55.60 |
| No Res | CMU-MOSEI | 41.75 / 40.12 | 43.02 / 40.25 | 62.26 ‡ / 61.06 ‡ | 62.98 / 61.50 |
| No FC-Res | IEMOCAP-6 | 70.43 / 70.39 | 70.43 / 70.47 | 84.35 ‡ / 84.38 ‡ | 83.43 / 83.72 |
| No FC-Res | IEMOCAP-4 | 85.37 / 85.36 | 85.26 / 85.25 | 86.00 ‡ / 85.90 ‡ | 85.15 / 85.02 |
| No FC-Res | MELD | 67.36 / 66.30 | 67.39 / 66.35 | 73.72 ‡ / 73.64 ‡ | 73.33 / 73.12 |
| No FC-Res | CMU-MOSEI | 45.03 / 43.28 | 45.12 / 43.41 | 66.20 ‡ / 65.69 ‡ | 66.37 / 64.54 |
| No Seg | IEMOCAP-6 | 71.66 / 71.69 | 71.41 / 71.32 | 83.49 ‡ / 83.64 ‡ | 83.24 / 83.46 |
| No Seg | IEMOCAP-4 | 85.47 / 85.43 | 85.05 / 85.01 | 85.37 ‡ / 85.32 ‡ | 84.31 / 84.14 |
| No Seg | MELD | 67.59 / 66.59 | 67.66 / 66.64 | 74.37 ‡ / 74.22 ‡ | 73.41 / 73.20 |
| No Seg | CMU-MOSEI | 45.89 / 44.53 | 45.67 / 44.22 | 66.44 ‡ / 65.88 ‡ | 66.85 / 65.87 |
| No Textual | IEMOCAP-6 | 55.70 / 54.23 | 59.33 / 59.34 | 71.04 ‡ / 70.42 ‡ | 71.16 / 70.97 |
| No Textual | IEMOCAP-4 | 62.88 / 62.22 | 64.16 / 62.92 | 66.28 ‡ / 60.76 ‡ | 64.48 / 62.31 |
| No Textual | MELD | 50.15 / 42.27 | 50.11 / 43.55 | 52.72 ‡ / 52.10 ‡ | 51.72 / 50.62 |
| No Textual | CMU-MOSEI | 32.98 / 28.28 | 35.54 / 31.19 | 54.95 ‡ / 47.80 ‡ | 55.04 / 48.93 |
| No Visual | IEMOCAP-6 | 71.29 / 71.23 | 70.98 / 71.06 | 84.53 ‡ / 84.69 ‡ | 84.35 / 84.32 |
| No Visual | IEMOCAP-4 | 86.21 / 86.20 | 85.79 / 85.76 | 85.58 ‡ / 85.57 ‡ | 85.37 / 85.27 |
| No Visual | MELD | 67.39 / 66.26 | 67.47 / 66.45 | 73.83 ‡ / 73.75 ‡ | 73.26 / 73.03 |
| No Visual | CMU-MOSEI | 45.32 / 43.57 | 45.65 / 43.95 | 66.30 ‡ / 65.81 ‡ | 66.20 / 64.80 |
| No Acoustic | IEMOCAP-6 | 68.45 / 68.26 | 67.59 / 67.52 | 83.24 ‡ / 82.87 ‡ | 82.32 / 82.05 |
| No Acoustic | IEMOCAP-4 | 83.35 / 83.28 | 83.14 / 83.14 | 84.31 ‡ / 84.25 ‡ | 84.31 / 84.23 |
| No Acoustic | MELD | 67.55 / 66.40 | 67.36 / 66.27 | 73.72 ‡ / 73.62 ‡ | 73.22 / 72.97 |
| No Acoustic | CMU-MOSEI | 46.56 / 44.57 | 45.89 / 44.08 | 66.47 ‡ / 65.57 ‡ | 66.80 / 65.34 |
| No L_e | IEMOCAP-6 | n/a | not comparable | 84.41 ‡ / 84.53 ‡ | 54.59 / 42.74 |
| No L_e | IEMOCAP-4 | n/a | not comparable | 85.47 ‡ / 85.40 ‡ | 82.93 / 82.75 |
| No L_e | MELD | n/a | not comparable | 73.98 ‡ / 73.84 ‡ | 72.61 / 72.33 |
| No L_e | CMU-MOSEI | n/a | not comparable | 66.59 ‡ / 65.91 ‡ | 65.58 / 65.39 |
| No L_e & L_o | IEMOCAP-6 | n/a | not comparable | 83.30 ‡ / 83.54 ‡ | 80.71 / 80.20 |
| No L_e & L_o | IEMOCAP-4 | n/a | not comparable | 85.58 ‡ / 85.55 ‡ | 44.01 / 26.90 |
| No L_e & L_o | MELD | n/a | not comparable | 73.91 ‡ / 73.77 ‡ | 72.72 / 72.45 |
| No L_e & L_o | CMU-MOSEI | n/a | not comparable | 66.54 ‡ / 65.64 ‡ | 58.58 / 59.98 |

‡ marks the paper's MSAC values. The paper's GraphSmile MSAC row in Table VI equals its
optimized (‡) result in Table V, so the MSAC block appears to use optimized settings; we have not
confirmed this for every ablation row. Ours use the Table III settings (†), so the MSAC columns
are not like-for-like, even for the full model.

**Notes**
- **Structure ablations (No Res, No FC-Res, No Seg):** the MERC results are close to the paper. No FC-Res and No Seg are within about 0.5 points on all datasets. No Res is close on IEMOCAP-6 and CMU-MOSEI, but MELD is about 4 points below the paper's ACC.
- **Sentiment-loss-off rows (No L_s, No L_s & L_o):** the MSAC columns are not comparable. Removing the sentiment loss leaves the sentiment head untrained. The paper's MSAC block uses No L_e and No L_e & L_o (emotion loss off, and emotion and shift loss off), which are different rows.
- **No L_e and No L_e & L_o (MSAC):** on MELD and CMU-MOSEI these are reasonably close to the paper. On IEMOCAP-6 and IEMOCAP-4 some runs are far below it (54.59 ACC against 84.41 for No L_e on IEMOCAP-6; 44.01 against 85.58 for No L_e & L_o on IEMOCAP-4). A three-seed rerun (pass p3, `results/experiments/p3-collapse-check-summary.md`) shows this is a **selection artifact, not failed training**: with the emotion loss off, the emotion head is untrained, so "best test emotion F1" picks an arbitrary epoch. At that epoch the sentiment head sometimes predicts one class (ACC 44.01), while the same runs reach about 80 to 83 ACC at other seeds. IEMOCAP-6 No L_e: seeds 2024 / 1 / 2 give ACC 54.59 / 81.52 / 80.10. The code's `--classify sentiment` option selects the epoch by sentiment F1, which is the natural choice for these MSAC rows; these rows have **not** been rerun that way yet. See section 5.
- **No Textual:** our result is higher than the paper on IEMOCAP-6 (59.33 against 55.70). We remove text by zeroing its input. The paper does not say how it removed text, so this difference may come from that choice.

### Table VII: time and memory (GraphSmile)

| Dataset | Paper s/epoch | Ours s/epoch | Paper memory (MB) | Ours memory (MB) |
|---|---|---|---|---|
| IEMOCAP-6 | 1.12 | 1.61 | 1840.89 | 1731.51 |
| IEMOCAP-4 | 0.67 | 1.01 | 509.88 | 431.89 |
| MELD | 5.21 | 7.55 | 513.33 | 422.61 |
| CMU-MOSEI | 7.38 | 12.84 | 467.13 | 306.57 |

The paper used an RTX 3090. Our runs used a shared A100, so the time comparison is not
like-for-like. The memory values are measured by the code.

### L2 check

The paper states an L2 weight of 1e-3. The repository default is 1e-4.

| Run | L2 | ACC | F-Score | Source |
|---|---|---|---|---|
| IEMOCAP-6 | 1e-4 | 71.35 | 71.29 | `results/logs/IEMOCAP_full_l2_1e-4.log` (max over epochs) |
| IEMOCAP-6 | 1e-3 | 71.23 | 71.17 | `results/logs/IEMOCAP_full.log` (max over epochs) |

The change in L2 moves IEMOCAP-6 by less than 0.2 points, so it does not explain the 1.5-point gap to the paper.

### Five-seed paper-protocol base (IEMOCAP-6)

This replaces the single-seed note in section 2 for the base row. Paper protocol: default `--select_on test`, seeds 2024, 1, 2, 3, 4.

The `Test performance..` line prints `max(all_acc_emo)` and `max(all_f1_emo)`, so each value is the maximum over epochs of that metric on its own. The two maxima can come from different epochs. This is the same convention as the IEMOCAP-6 rows above.

| Setting | ACC mean ± std | WF1 mean ± std | Per-seed WF1 (2024 / 1 / 2 / 3 / 4) |
|---|---|---|---|
| base | 71.62 ± 0.31 | 71.61 ± 0.36 | 71.17 / 71.88 / 71.79 / 71.93 / 71.26 |
| shift_5 | 71.52 ± 0.20 | 71.48 ± 0.21 | 71.55 / 71.34 / 71.76 / 71.51 / 71.22 |
| shift_10 | 71.49 ± 0.37 | 71.49 ± 0.32 | 71.23 / 71.72 / 71.77 / 71.68 / 71.07 |

Std is the sample std (n = 5). Source logs: `results/logs/p2_<setting>_paper_seed<seed>.log`. The WF1 means match `results/experiments/p2-results.md`.

For reference, the paper reports 72.77 ACC for this dataset (Table IV). Our base is about 1.1 points below on ACC.

### Negative results

No setting was kept. Each one failed the keep rule: the clean 5-seed mean must beat the clean baseline by more than the baseline std, and the paper-protocol WF1 must not be worse.

| Setting | Clean WF1 (baseline 68.26 ± 0.94) | Paper-protocol WF1 (base 71.61) | Result |
|---|---|---|---|
| Class weighting (`--class_weight`) | 68.38 ± 0.62 (Happy 44.80 ± 4.34) | not checked here | No measurable gain |
| shift_5 | 68.52 ± 1.73 (+0.26) | 71.48 (−0.13) | Not kept (inconclusive) |
| shift_10 | 68.35 ± 1.58 (+0.09) | 71.49 (−0.11) | Not kept (inconclusive) |

The stage 1 gains for shift_5 and shift_10 (+1.4 to +1.6 WF1 on 3 seeds) did not hold on five seeds. Details: `results/experiments/p2-results.md`.

## 5. Differences that remain

1. **IEMOCAP-6 is about 1.2 points below the paper.** Under the paper protocol (best test epoch), the five-seed mean is 71.61 ± 0.36 WF1 and 71.62 ± 0.31 ACC, against the paper's 72.81 and 72.77. All five seeds (WF1 71.17 to 71.93) are below the paper's value, so seed variation alone does not explain the gap. The seed-2024 run used in the main tables (71.17) is the lowest of the five. The gap is largest in Happy (-8.2 in that run); Happy F1 varies by several points between seeds. The paper does not report its own seed spread.
2. **Feature files:** we used the feature pickles from the dataset release. The paper's exact files may differ, which is untested.
3. **Test-set selection:** the code picks the best epoch by test F1, as described in section 3.
4. **MSAC rows with the emotion loss off (No L_e, No L_e & L_o):** these were selected by emotion F1 although the emotion head is untrained, which produced the collapsed values (see the note under Table VI). A rerun with `--classify sentiment` (8 runs, about 1.5 hours) would give the comparable numbers; it has not been done.
5. **Not reproduced:** the ‡ (optimized MSAC) rows, the baseline rows of Table IV, and Figs. 4 to 7.

## 5b. Experiment record

Each pass has a summary under `results/experiments/`. No tuned setting was adopted; the main tables stay at the Table III settings.

| Pass | Question | Result |
|---|---|---|
| p0 | Seed variance and a clean validation split | The original code's validation dialogues overlap its training set (fixed behind `--clean_valid`). Clean 5-seed IEMOCAP-6 WF1 is 68.26 ± 0.94. |
| p1 | Class-weighted emotion loss (`--class_weight`) | 68.38 ± 0.62 against 68.26. No measurable gain, not kept. |
| p1b | Are our feature files the released ones? | All four server files are byte-identical to the README release (md5). IEMOCAP-6 matches Table II; MELD is one dialogue short and CMU-MOSEI counts differ from Table II in the released file itself. |
| p2 | Sensitivity of window, shift window, layers, loss weights | Screened on 3 seeds; `shift_5` and `shift_10` looked better, but on 5 seeds the gains were +0.26 and +0.09 (not kept). Paper-protocol 5-seed WF1: base 71.61, shift_5 71.48, shift_10 71.49. |
| p3 | Why do some MSAC ablation rows collapse? | Selection artifact (see Table VI notes), not failed training. |

## 6. Proof and reproducibility

- **Raw logs:** every run is in `results/logs/`. Each log starts with the full argument list (`Namespace(...)`), so the configuration is recorded in the file itself.
- **Index:** `results/proof_index.md` lists all 49 runs with their configuration, completion status, epochs run, best-F1 epoch, and final metrics. Generate it with `python scripts/build_proof_index.py results/proof_index.md`.
- **Comparison:** `results/comparison.md` is generated from the logs by `python scripts/compare_to_paper.py results/comparison.md`.
- **Run again:** `RUNBOOK.md` gives the commands. `GPU=<id> bash scripts/run_all.sh` runs the main 28 jobs, `scripts/run_remaining.sh` runs the 20 ablation jobs, and `scripts/run_sweeps.sh` runs the L2, depth and window checks.
