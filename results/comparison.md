# GraphSmile: our results vs the paper

## Table IV: MERC (emotion), GraphSmile

### IEMOCAP

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Happy | 63.09 | 54.87 | -8.22 |
| Sad | 83.16 | 82.14 | -1.02 |
| Neutral | 71.07 | 70.85 | -0.22 |
| Angry | 71.38 | 68.92 | -2.46 |
| Excited | 79.66 | 78.04 | -1.62 |
| Frustrated | 66.84 | 66.22 | -0.62 |
| ACC | 72.77 | 71.23 | -1.54 |
| WF1 | 72.81 | 71.17 | -1.64 |

### IEMOCAP4

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Happy | 87.94 | 86.12 | -1.82 |
| Sad | 88.67 | 88.45 | -0.22 |
| Neutral | 85.41 | 84.70 | -0.71 |
| Angry | 84.73 | 85.22 | +0.49 |
| ACC | 86.53 | 86.00 | -0.53 |
| WF1 | 86.52 | 85.98 | -0.54 |

### MELD

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Neutral | 80.35 | 80.27 | -0.08 |
| Surprise | 59.11 | 58.67 | -0.44 |
| Fear | 18.18 | 19.18 | +1.00 |
| Sadness | 42.46 | 43.45 | +0.99 |
| Joy | 64.99 | 64.56 | -0.43 |
| Disgust | 32.43 | 29.36 | -3.07 |
| Anger | 53.67 | 53.46 | -0.21 |
| ACC | 67.70 | 67.59 | -0.11 |
| WF1 | 66.71 | 66.55 | -0.16 |

### CMUMOSEI7

| Class | Paper | Ours | Δ |
|---|---|---|---|
| HNeg | 0.00 | 0.00 | +0.00 |
| Neg | 28.79 | 30.28 | +1.49 |
| WNeg | 43.50 | 45.19 | +1.69 |
| Neu | 39.19 | 42.81 | +3.62 |
| WPos | 57.30 | 53.40 | -3.90 |
| Pos | 35.95 | 33.39 | -2.56 |
| HPos | 6.25 | 3.17 | -3.08 |
| ACC | 46.82 | 45.10 | -1.72 |
| WF1 | 44.93 | 43.96 | -0.97 |

## Table V: MSAC (sentiment), GraphSmile with Table III settings (†)

| Dataset | Class | Paper | Ours | Δ |
|---|---|---|---|---|
| IEMOCAP | neg | 86.87 | 87.11 | +0.24 |
| IEMOCAP | neu | 69.06 | 70.26 | +1.20 |
| IEMOCAP | pos | 89.21 | 89.66 | +0.45 |
| IEMOCAP | ACC | 83.73 | 83.67 | -0.06 |
| IEMOCAP | WF1 | 83.66 | 83.82 | +0.16 |
| IEMOCAP4 | neg | 87.76 | 87.73 | -0.03 |
| IEMOCAP4 | neu | 82.93 | 82.21 | -0.72 |
| IEMOCAP4 | pos | 87.23 | 85.71 | -1.52 |
| IEMOCAP4 | ACC | 85.79 | 85.26 | -0.53 |
| IEMOCAP4 | WF1 | 85.71 | 85.18 | -0.53 |
| MELD | neg | 70.08 | 70.07 | -0.01 |
| MELD | neu | 79.15 | 79.09 | -0.06 |
| MELD | pos | 64.84 | 64.18 | -0.66 |
| MELD | ACC | 73.60 | 73.45 | -0.15 |
| MELD | WF1 | 73.40 | 73.24 | -0.16 |
| CMUMOSEI7 | neg | 56.14 | 58.29 | +2.15 |
| CMUMOSEI7 | neu | 39.80 | 42.10 | +2.30 |
| CMUMOSEI7 | pos | 78.18 | 76.65 | -1.53 |
| CMUMOSEI7 | ACC | 66.83 | 65.27 | -1.56 |
| CMUMOSEI7 | WF1 | 65.69 | 65.61 | -0.08 |

## Table VI: ablations (ACC / WF1)

MERC columns use emotion ACC/WF1; MSAC columns use sentiment ACC/WF1.

| Variant | Dataset | Paper MERC | Ours MERC | Paper MSAC | Ours MSAC |
|---|---|---|---|---|---|
| full | IEMOCAP-6 | 72.77 / 72.81 | 71.23 / 71.17 | 84.97 / 85.09 | 83.67 / 83.82 |
| full | IEMOCAP-4 | 86.53 / 86.52 | 86.00 / 85.98 | 87.49 / 87.48 | 85.26 / 85.18 |
| full | MELD | 67.70 / 66.71 | 67.59 / 66.55 | 74.44 / 74.31 | 73.45 / 73.24 |
| full | CMU-MOSEI | 46.82 / 44.93 | 45.10 / 43.96 | 67.73 / 66.73 | 65.27 / 65.61 |
| no_Ls | IEMOCAP-6 | 71.53 / 71.52 | 70.92 / 70.83 | not comparable | not comparable |
| no_Ls | IEMOCAP-4 | 85.15 / 85.15 | 85.37 / 85.36 | not comparable | not comparable |
| no_Ls | MELD | 67.39 / 66.30 | 67.43 / 66.33 | not comparable | not comparable |
| no_Ls | CMU-MOSEI | 45.08 / 43.37 | 45.43 / 44.43 | not comparable | not comparable |
| no_Lo | IEMOCAP-6 | 71.41 / 71.42 | 70.92 / 71.02 | 84.10 / 84.18 | 82.26 / 82.45 |
| no_Lo | IEMOCAP-4 | 84.52 / 84.50 | 84.84 / 84.79 | 85.26 / 85.23 | 83.46 / 83.25 |
| no_Lo | MELD | 67.62 / 66.59 | 67.59 / 66.58 | 74.18 / 74.10 | 73.52 / 73.32 |
| no_Lo | CMU-MOSEI | 45.20 / 44.03 | 45.53 / 44.09 | 66.71 / 66.20 | 66.73 / 65.03 |
| no_Ls_Lo | IEMOCAP-6 | 70.67 / 70.49 | 70.06 / 70.01 | not comparable | not comparable |
| no_Ls_Lo | IEMOCAP-4 | 84.41 / 84.40 | 85.05 / 85.04 | not comparable | not comparable |
| no_Ls_Lo | MELD | 67.55 / 66.43 | 67.51 / 66.40 | not comparable | not comparable |
| no_Ls_Lo | CMU-MOSEI | 44.29 / 43.54 | 44.74 / 43.86 | not comparable | not comparable |
| no_text | IEMOCAP-6 | 55.70 / 54.23 | 59.33 / 59.34 | 71.04 / 70.42 | 71.16 / 70.97 |
| no_text | IEMOCAP-4 | 62.88 / 62.22 | 64.16 / 62.92 | 66.28 / 60.76 | 64.48 / 62.31 |
| no_text | MELD | 50.15 / 42.27 | 50.11 / 43.55 | 52.72 / 52.10 | 51.72 / 50.62 |
| no_text | CMU-MOSEI | 32.98 / 28.28 | 35.54 / 31.19 | 54.95 / 47.80 | 55.04 / 48.93 |
| no_visual | IEMOCAP-6 | 71.29 / 71.23 | 70.98 / 71.06 | 84.53 / 84.69 | 84.35 / 84.32 |
| no_visual | IEMOCAP-4 | 86.21 / 86.20 | 85.79 / 85.76 | 85.58 / 85.57 | 85.37 / 85.27 |
| no_visual | MELD | 67.39 / 66.26 | 67.47 / 66.45 | 73.83 / 73.75 | 73.26 / 73.03 |
| no_visual | CMU-MOSEI | 45.32 / 43.57 | 45.65 / 43.95 | 66.30 / 65.81 | 66.20 / 64.80 |
| no_acoustic | IEMOCAP-6 | 68.45 / 68.26 | 67.59 / 67.52 | 83.24 / 82.87 | 82.32 / 82.05 |
| no_acoustic | IEMOCAP-4 | 83.35 / 83.28 | 83.14 / 83.14 | 84.31 / 84.25 | 84.31 / 84.23 |
| no_acoustic | MELD | 67.55 / 66.40 | 67.36 / 66.27 | 73.72 / 73.62 | 73.22 / 72.97 |
| no_acoustic | CMU-MOSEI | 46.56 / 44.57 | 45.89 / 44.08 | 66.47 / 65.57 | 66.80 / 65.34 |
| no_Res | IEMOCAP-6 | 61.98 / 61.35 | 61.37 / 61.34 | 78.56 / 78.25 | 76.96 / 76.02 |
| no_Res | IEMOCAP-4 | 79.64 / 79.58 | 80.28 / 80.21 | 81.02 / 80.83 | 81.34 / 81.13 |
| no_Res | MELD | 52.07 / 46.24 | 48.24 / 47.36 | 58.08 / 56.61 | 55.67 / 55.60 |
| no_Res | CMU-MOSEI | 41.75 / 40.12 | 43.02 / 40.25 | 62.26 / 61.06 | 62.98 / 61.50 |
| no_FCRes | IEMOCAP-6 | 70.43 / 70.39 | 70.43 / 70.47 | 84.35 / 84.38 | 83.43 / 83.72 |
| no_FCRes | IEMOCAP-4 | 85.37 / 85.36 | 85.26 / 85.25 | 86.00 / 85.90 | 85.15 / 85.02 |
| no_FCRes | MELD | 67.36 / 66.30 | 67.39 / 66.35 | 73.72 / 73.64 | 73.33 / 73.12 |
| no_FCRes | CMU-MOSEI | 45.03 / 43.28 | 45.12 / 43.41 | 66.20 / 65.69 | 66.37 / 64.54 |
| no_Seg | IEMOCAP-6 | 71.66 / 71.69 | 71.41 / 71.32 | 83.49 / 83.64 | 83.24 / 83.46 |
| no_Seg | IEMOCAP-4 | 85.47 / 85.43 | 85.05 / 85.01 | 85.37 / 85.32 | 84.31 / 84.14 |
| no_Seg | MELD | 67.59 / 66.59 | 67.66 / 66.64 | 74.37 / 74.22 | 73.41 / 73.20 |
| no_Seg | CMU-MOSEI | 45.89 / 44.53 | 45.67 / 44.22 | 66.44 / 65.88 | 66.85 / 65.87 |
| no_Le | IEMOCAP-6 | n/a | not comparable | 84.41 / 84.53 | 54.59 / 42.74 |
| no_Le | IEMOCAP-4 | n/a | not comparable | 85.47 / 85.40 | 82.93 / 82.75 |
| no_Le | MELD | n/a | not comparable | 73.98 / 73.84 | 72.61 / 72.33 |
| no_Le | CMU-MOSEI | n/a | not comparable | 66.59 / 65.91 | 65.58 / 65.39 |
| no_Le_Lo | IEMOCAP-6 | n/a | not comparable | 83.30 / 83.54 | 80.71 / 80.20 |
| no_Le_Lo | IEMOCAP-4 | n/a | not comparable | 85.58 / 85.55 | 44.01 / 26.90 |
| no_Le_Lo | MELD | n/a | not comparable | 73.91 / 73.77 | 72.72 / 72.45 |
| no_Le_Lo | CMU-MOSEI | n/a | not comparable | 66.54 / 65.64 | 58.58 / 59.98 |

Notes:
- Paper Table VI lower block (MSAC) uses No L_e / No L_o / No L_e & L_o. Our `no_Ls` and `no_Ls_Lo` runs remove the sentiment loss, so they are not comparable with the MSAC rows.
- Paper Table VI MSAC values use the optimized sentiment settings (‡); our runs use the Table III settings (†).
- No Res, No FC-Res and No Seg are run with the flags --no_res, --no_fcres and --no_seg.

## Table VII: time and memory (GraphSmile)

Paper: RTX 3090. Ours: whatever GPU the job ran on. Time includes test evaluation.

| Dataset | Paper s/epoch | Ours s/epoch | Paper MB | Ours MB |
|---|---|---|---|---|
| IEMOCAP | 1.12 | 1.61 | 1840.89 | 1731.51 |
| IEMOCAP4 | 0.67 | 1.01 | 509.88 | 431.89 |
| MELD | 5.21 | 7.55 | 513.33 | 422.61 |
| CMUMOSEI7 | 7.38 | 12.84 | 467.13 | 306.57 |

## Extra runs (not in the paper tables)

| Tag | Dataset | Ours ACC | Ours WF1 | Note |
|---|---|---|---|---|
| full_l2_1e-4 | IEMOCAP | 71.16 | 71.29 | |
| no_FCRes | IEMOCAP | 70.43 | 70.47 | |
| no_Le | IEMOCAP | 21.50 | 18.77 | |
| no_Le_Lo | IEMOCAP | 16.64 | 12.97 | |
| no_Res | IEMOCAP | 61.37 | 61.34 | |
| no_Seg | IEMOCAP | 71.41 | 71.32 | |

