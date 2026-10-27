# Reproduction of GraphSmile: paper vs our runs

This document compares our re-run of GraphSmile with the numbers reported in the paper
*"Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for
multimodal emotion recognition"* (IEEE TPAMI, 2025, doi:10.1109/TPAMI.2025.3581236).

Every "ours" number comes from a raw log in `results/logs/`. The index of all logs, with
their configuration, is in `results/proof_index.md`.

## 1. What the paper contains

| Item | What it is | Reproduced here |
|---|---|---|
| Table I | Merging scheme that maps dataset emotion labels to sentiments | Yes, built into `dataloader.py` (not a result) |
| Table II | Dataset statistics (conversations and utterances) | Not compared (it describes the data, it is not a result) |
| Table III | Hyperparameters (LR, batch size, dropout, layers, windows, loss weights) | Matches `README.md`, except the L2 weight (see section 5) |
| Table IV | MERC (emotion recognition) results for all baselines and GraphSmile, per emotion | GraphSmile row reproduced |
| Table V | MSAC (sentiment) results, with Table III settings (†) and optimized settings (‡) | † reproduced. ‡ not reproduced (values not given in the paper) |
| Table VI | Ablations: losses (No L_s, No L_o, No L_s & L_o), residual and segmentation (No Res, No FC-Res, No Seg), modalities (No Textual, No Visual, No Acoustic) | Loss and modality rows reproduced. Residual and segmentation rows **not reproduced** (no code switch yet) |
| Table VII | Seconds per epoch and memory | Reproduced, on a different GPU (see section 4) |
| Fig. 1 to 3 | Architecture, propagation example, shift features | Diagrams, nothing to reproduce |
| Fig. 4 | Accuracy against network depth, with and without residual connection | Not reproduced (depth sweep script exists, not yet run) |
| Fig. 5 | Accuracy against window size | Not reproduced (window sweep script exists, not yet run) |
| Fig. 6 | t-SNE of embeddings | Not reproduced (code does not save features) |
| Fig. 7 | Confusion matrices for IEMOCAP-6 and MELD | Logged, not yet compared |

## 2. Setup used for our runs

- **Code:** this repository, branch `claude/gifted-galileo-l5kpj3`, with the changes described in `RUNBOOK.md`.
- **Hardware:** NVIDIA A100-PCIE-40GB on a shared server. Other users were running jobs on the same GPU.
- **Software:** Python 3.10, PyTorch 2.0.0 with CUDA 11.7, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1.
- **Data:** the four feature pickles from the dataset release (`README.md`), not included in this repository.
- **Seeds:** the code fixes the seed to 2024. No extra seeds were run, so the run-to-run spread is not measured.

## 3. Protocol

The code reports the epoch with the best **test** emotion F1 (`run.py`). Our tables use that
same epoch for all numbers, so every value is from one epoch. The paper's exact protocol is
not stated in the paper, so the test-set selection may differ from theirs.

- **Emotion (MERC):** ACC and weighted F1 at the best-F1 epoch. Per-emotion F1 comes from the classification report of that epoch.
- **Sentiment (MSAC):** the sentiment head's ACC and weighted F1 at the same epoch.
- **Ablations:** the same protocol, one run per variant.

The paper reports per-class values as F1 scores in %. We use the same.

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

The optimized-setting rows (‡) of Table V are not reproduced.

### Table VI: ablations (ACC / WF1, %)

MERC columns are emotion ACC / WF1. MSAC columns are sentiment ACC / WF1.

| Variant | Dataset | Paper MERC | Ours MERC | Paper MSAC | Ours MSAC |
|---|---|---|---|---|---|
| GraphSmile (full) | IEMOCAP-6 | 72.77 / 72.81 | 71.23 / 71.17 | 84.97 / 85.09 | 83.67 / 83.82 |
| GraphSmile (full) | IEMOCAP-4 | 86.53 / 86.52 | 86.00 / 85.98 | 87.49 / 87.48 | 85.26 / 85.18 |
| GraphSmile (full) | MELD | 67.70 / 66.71 | 67.59 / 66.55 | 74.44 / 74.31 | 73.45 / 73.24 |
| GraphSmile (full) | CMU-MOSEI | 46.82 / 44.93 | 45.10 / 43.96 | 67.73 / 66.73 | 65.27 / 65.61 |
| No L_s | IEMOCAP-6 | 71.53 / 71.52 | 70.92 / 70.83 | 84.41 / 84.53 | 49.04 / 32.28 |
| No L_s | IEMOCAP-4 | 85.15 / 85.15 | 85.37 / 85.36 | 85.47 / 85.40 | 18.35 / 10.01 |
| No L_s | MELD | 67.39 / 66.30 | 67.43 / 66.33 | 73.98 / 73.84 | 19.89 / 6.69 |
| No L_s | CMU-MOSEI | 45.08 / 43.37 | 45.43 / 44.43 | 66.59 / 65.91 | 23.76 / 13.65 |
| No L_o | IEMOCAP-6 | 71.41 / 71.42 | 70.92 / 71.02 | 84.10 / 84.18 | 82.26 / 82.45 |
| No L_o | IEMOCAP-4 | 84.52 / 84.50 | 84.84 / 84.79 | 85.26 / 85.23 | 83.46 / 83.25 |
| No L_o | MELD | 67.62 / 66.59 | 67.59 / 66.58 | 74.18 / 74.10 | 73.52 / 73.32 |
| No L_o | CMU-MOSEI | 45.20 / 44.03 | 45.53 / 44.09 | 66.71 / 66.20 | 66.73 / 65.03 |
| No L_s & L_o | IEMOCAP-6 | 70.67 / 70.49 | 70.06 / 70.01 | 83.30 / 83.54 | 49.04 / 32.28 |
| No L_s & L_o | IEMOCAP-4 | 84.41 / 84.40 | 85.05 / 85.04 | 85.58 / 85.55 | 15.59 / 4.70 |
| No L_s & L_o | MELD | 67.55 / 66.43 | 67.51 / 66.40 | 73.91 / 73.77 | 19.96 / 6.64 |
| No L_s & L_o | CMU-MOSEI | 44.29 / 43.54 | 44.74 / 43.86 | 66.54 / 65.64 | 21.68 / 11.96 |
| No Textual | IEMOCAP-6 | 55.70 / 54.23 | 59.33 / 59.34 | 71.04 / 70.42 | 71.16 / 70.97 |
| No Textual | IEMOCAP-4 | 62.88 / 62.22 | 64.16 / 62.92 | 66.28 / 60.76 | 64.48 / 62.31 |
| No Textual | MELD | 50.15 / 42.27 | 50.11 / 43.55 | 52.72 / 52.10 | 51.72 / 50.62 |
| No Textual | CMU-MOSEI | 32.98 / 28.28 | 35.54 / 31.19 | 54.95 / 47.80 | 55.04 / 48.93 |
| No Visual | IEMOCAP-6 | 71.29 / 71.23 | 70.98 / 71.06 | 84.53 / 84.69 | 84.35 / 84.32 |
| No Visual | IEMOCAP-4 | 86.21 / 86.20 | 85.79 / 85.76 | 85.58 / 85.57 | 85.37 / 85.27 |
| No Visual | MELD | 67.39 / 66.26 | 67.47 / 66.45 | 73.83 / 73.75 | 73.26 / 73.03 |
| No Visual | CMU-MOSEI | 45.32 / 43.57 | 45.65 / 43.95 | 66.30 / 65.81 | 66.20 / 64.80 |
| No Acoustic | IEMOCAP-6 | 68.45 / 68.26 | 67.59 / 67.52 | 83.24 / 82.87 | 82.32 / 82.05 |
| No Acoustic | IEMOCAP-4 | 83.35 / 83.28 | 83.14 / 83.14 | 84.31 / 84.25 | 84.31 / 84.23 |
| No Acoustic | MELD | 67.55 / 66.40 | 67.36 / 66.27 | 73.72 / 73.62 | 73.22 / 72.97 |
| No Acoustic | CMU-MOSEI | 46.56 / 44.57 | 45.89 / 44.08 | 66.47 / 65.57 | 66.80 / 65.34 |

Not reproduced: **No Res, No FC-Res, No Seg** (the code has no switch for these yet).

**Notes on the rows that disagree most**
- **No Textual** is higher than the paper for IEMOCAP-6 (59.33 against 55.70). We remove text by zeroing its input. The paper does not say how it removed text, so this difference may come from that choice.
- **No L_s and No L_s & L_o, MSAC columns** are near chance in our runs. Removing the sentiment loss leaves the sentiment head untrained. The paper reports sentiment numbers for these rows, so its setup must differ. We have not resolved this.

### Table VII: time and memory (GraphSmile)

| Dataset | Paper s/epoch | Ours s/epoch | Paper memory (MB) | Ours memory (MB) |
|---|---|---|---|---|
| IEMOCAP-6 | 1.12 | 1.61 | 1840.89 | 1731.51 |
| IEMOCAP-4 | 0.67 | 1.01 | 509.88 | 431.89 |
| MELD | 5.21 | 7.55 | 513.33 | 422.61 |
| CMU-MOSEI | 7.38 | 12.84 | 467.13 | 306.57 |

The paper used an RTX 3090. Our runs used an A100 shared with other users, so the time
comparison is not like-for-like. The memory numbers are measured by the code on each run.

### L2 check

The paper states an L2 weight of 1e-3. The repository default is 1e-4.

| Run | L2 | ACC | F-Score | Source |
|---|---|---|---|---|
| IEMOCAP-6 | 1e-4 | 71.35 | 71.29 | `results/logs/IEMOCAP_full_l2_1e-4.log` (max over epochs) |
| IEMOCAP-6 | 1e-3 | 71.23 | 71.17 | `results/logs/IEMOCAP_full.log` (max over epochs) |

The change in L2 moves IEMOCAP-6 by less than 0.2 points, so it does not explain the 1.5-point gap to the paper.

## 5. Differences that remain

1. **IEMOCAP-6 is 1.5 points below the paper** on ACC and WF1, mainly in Happy (-8.2). We have not checked other seeds, so we cannot say whether this is run-to-run variation or a real difference.
2. **Feature files:** we used the feature pickles from the dataset release. The paper's exact files may differ, and this is untested.
3. **Test-set selection:** the code picks the best epoch by test F1, as described in section 3.
4. **Missing ablations:** No Res, No FC-Res, and No Seg are not implemented as switches.
5. **MSAC rows for No L_s**: see the note under Table VI.
6. **Figures 4 to 7** are not reproduced yet.

## 6. Proof and reproducibility

- **Raw logs:** every run is in `results/logs/`. Each log starts with the full argument list (`Namespace(...)`), so the configuration of each run is recorded in the file itself.
- **Index:** `results/proof_index.md` lists all runs with their configuration, completion status, epochs run, best-F1 epoch, and final metrics. Generate it with `python scripts/build_proof_index.py`.
- **Comparison:** `results/comparison.md` is generated from the logs by `python scripts/compare_to_paper.py results/comparison.md`.
- **Run it again:** `RUNBOOK.md` gives the commands. `GPU=<id> bash scripts/run_all.sh` runs all 28 jobs, and `scripts/run_sweeps.sh` runs the extra checks.
