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
| Happy | 87.94 | n/a | n/a |
| Sad | 88.67 | n/a | n/a |
| Neutral | 85.41 | n/a | n/a |
| Angry | 84.73 | n/a | n/a |
| ACC | 86.53 | n/a | n/a |
| WF1 | 86.52 | n/a | n/a |

### MELD

| Class | Paper | Ours | Δ |
|---|---|---|---|
| Neutral | 80.35 | n/a | n/a |
| Surprise | 59.11 | n/a | n/a |
| Fear | 18.18 | n/a | n/a |
| Sadness | 42.46 | n/a | n/a |
| Joy | 64.99 | n/a | n/a |
| Disgust | 32.43 | n/a | n/a |
| Anger | 53.67 | n/a | n/a |
| ACC | 67.70 | n/a | n/a |
| WF1 | 66.71 | n/a | n/a |

### CMUMOSEI7

| Class | Paper | Ours | Δ |
|---|---|---|---|
| HNeg | 0.00 | n/a | n/a |
| Neg | 28.79 | n/a | n/a |
| WNeg | 43.50 | n/a | n/a |
| Neu | 39.19 | n/a | n/a |
| WPos | 57.30 | n/a | n/a |
| Pos | 35.95 | n/a | n/a |
| HPos | 6.25 | n/a | n/a |
| ACC | 46.82 | n/a | n/a |
| WF1 | 44.93 | n/a | n/a |

## Table V: MSAC (sentiment), GraphSmile with Table III settings (†)

| Dataset | Class | Paper | Ours | Δ |
|---|---|---|---|---|
| IEMOCAP | neg | 86.87 | 87.11 | +0.24 |
| IEMOCAP | neu | 69.06 | 70.26 | +1.20 |
| IEMOCAP | pos | 89.21 | 89.66 | +0.45 |
| IEMOCAP | ACC | 83.73 | 83.67 | -0.06 |
| IEMOCAP | WF1 | 83.66 | 83.82 | +0.16 |
| IEMOCAP4 | neg | 87.76 | n/a | n/a |
| IEMOCAP4 | neu | 82.93 | n/a | n/a |
| IEMOCAP4 | pos | 87.23 | n/a | n/a |
| IEMOCAP4 | ACC | 85.79 | n/a | n/a |
| IEMOCAP4 | WF1 | 85.71 | n/a | n/a |
| MELD | neg | 70.08 | n/a | n/a |
| MELD | neu | 79.15 | n/a | n/a |
| MELD | pos | 64.84 | n/a | n/a |
| MELD | ACC | 73.60 | n/a | n/a |
| MELD | WF1 | 73.40 | n/a | n/a |
| CMUMOSEI7 | neg | 56.14 | n/a | n/a |
| CMUMOSEI7 | neu | 39.80 | n/a | n/a |
| CMUMOSEI7 | pos | 78.18 | n/a | n/a |
| CMUMOSEI7 | ACC | 66.83 | n/a | n/a |
| CMUMOSEI7 | WF1 | 65.69 | n/a | n/a |

## Table VI: ablations (ACC / WF1)

MERC columns use emotion ACC/WF1; MSAC columns use sentiment ACC/WF1.

| Variant | Dataset | Paper MERC | Ours MERC | Paper MSAC | Ours MSAC |
|---|---|---|---|---|---|
| full | IEMOCAP-6 | 72.77 / 72.81 | 71.23 / 71.17 | 84.97 / 85.09 | 83.67 / 83.82 |
| full | IEMOCAP-4 | 86.53 / 86.52 | not run | 87.49 / 87.48 | not run |
| full | MELD | 67.70 / 66.71 | not run | 74.44 / 74.31 | not run |
| full | CMU-MOSEI | 46.82 / 44.93 | not run | 67.73 / 66.73 | not run |
| no_Ls | IEMOCAP-6 | 71.53 / 71.52 | not run | 84.41 / 84.53 | not run |
| no_Ls | IEMOCAP-4 | 85.15 / 85.15 | not run | 85.47 / 85.40 | not run |
| no_Ls | MELD | 67.39 / 66.30 | not run | 73.98 / 73.84 | not run |
| no_Ls | CMU-MOSEI | 45.08 / 43.37 | not run | 66.59 / 65.91 | not run |
| no_Lo | IEMOCAP-6 | 71.41 / 71.42 | not run | 84.10 / 84.18 | not run |
| no_Lo | IEMOCAP-4 | 84.52 / 84.50 | not run | 85.26 / 85.23 | not run |
| no_Lo | MELD | 67.62 / 66.59 | not run | 74.18 / 74.10 | not run |
| no_Lo | CMU-MOSEI | 45.20 / 44.03 | not run | 66.71 / 66.20 | not run |
| no_Ls_Lo | IEMOCAP-6 | 70.67 / 70.49 | not run | 83.30 / 83.54 | not run |
| no_Ls_Lo | IEMOCAP-4 | 84.41 / 84.40 | not run | 85.58 / 85.55 | not run |
| no_Ls_Lo | MELD | 67.55 / 66.43 | not run | 73.91 / 73.77 | not run |
| no_Ls_Lo | CMU-MOSEI | 44.29 / 43.54 | not run | 66.54 / 65.64 | not run |
| no_text | IEMOCAP-6 | 55.70 / 54.23 | 59.33 / 59.34 | 71.04 / 70.42 | 71.16 / 70.97 |
| no_text | IEMOCAP-4 | 62.88 / 62.22 | not run | 66.28 / 60.76 | not run |
| no_text | MELD | 50.15 / 42.27 | not run | 52.72 / 52.10 | not run |
| no_text | CMU-MOSEI | 32.98 / 28.28 | not run | 54.95 / 47.80 | not run |
| no_visual | IEMOCAP-6 | 71.29 / 71.23 | not run | 84.53 / 84.69 | not run |
| no_visual | IEMOCAP-4 | 86.21 / 86.20 | not run | 85.58 / 85.57 | not run |
| no_visual | MELD | 67.39 / 66.26 | not run | 73.83 / 73.75 | not run |
| no_visual | CMU-MOSEI | 45.32 / 43.57 | not run | 66.30 / 65.81 | not run |
| no_acoustic | IEMOCAP-6 | 68.45 / 68.26 | not run | 83.24 / 82.87 | not run |
| no_acoustic | IEMOCAP-4 | 83.35 / 83.28 | not run | 84.31 / 84.25 | not run |
| no_acoustic | MELD | 67.55 / 66.40 | not run | 73.72 / 73.62 | not run |
| no_acoustic | CMU-MOSEI | 46.56 / 44.57 | not run | 66.47 / 65.57 | not run |

Not run: No Res, No FC-Res, No Seg (the code has no switch for these yet).

## Table VII: time and memory (GraphSmile)

Paper: RTX 3090. Ours: whatever GPU the job ran on. Time includes test evaluation.

| Dataset | Paper s/epoch | Ours s/epoch | Paper MB | Ours MB |
|---|---|---|---|---|
| IEMOCAP | 1.12 | 1.61 | 1840.89 | 1731.51 |
| IEMOCAP4 | 0.67 | n/a | 509.88 | n/a |
| MELD | 5.21 | n/a | 513.33 | n/a |
| CMUMOSEI7 | 7.38 | n/a | 467.13 | n/a |

## Extra runs (not in the paper tables)

| Tag | Dataset | Ours ACC | Ours WF1 | Note |
|---|---|---|---|---|
| full_l2_1e-4 | IEMOCAP | 71.16 | 71.29 | |

