# GraphSmile: progress update (15 Nov 2026)

Written in plain, spoken-style English so it can be read out or forwarded as it is.

## 1. The short version

- We have reproduced the GraphSmile paper on all four datasets. The full model comes within about 0.1 to 1.7 points of the paper. The no-text model comes out close to the paper on IEMOCAP-4 and MELD, and higher than the paper on IEMOCAP-6 and CMU-MOSEI. The two tables are in section 2.
- We re-ran both settings (full and no-text) on the GPU server, and every epoch matched our earlier logs exactly. So the earlier numbers are real and the code has not drifted.
- The new goal, from the evaluator's note, is to raise the no-text accuracy and weighted F1, and if possible the overall numbers too. The plan for that is in section 3.
- **No new result for that goal has finished yet.** Runs were started on the server, but the connection to the server dropped, so I cannot see them. Section 4 says exactly what is known and what is not.

## 2. The reproduction: two tables

All values are in percent. "Drop" means the full-model number minus the no-text number. "Ours" is our rerun with the paper's hyperparameters and the paper's protocol (best test epoch, seed 2024). The earlier run and the rerun gave identical numbers.

**Table A: full model (text + visual + acoustic)**

| Dataset | Paper ACC | Paper WF1 | Ours ACC | Ours WF1 |
|---|---|---|---|---|
| IEMOCAP-6 | 72.77 | 72.81 | 71.23 | 71.17 |
| IEMOCAP-4 | 86.53 | 86.52 | 86.00 | 85.98 |
| MELD | 67.70 | 66.71 | 67.59 | 66.55 |
| CMU-MOSEI | 46.82 | 44.93 | 45.10 | 43.96 |

**Table B: no-text model (visual + acoustic only) and the drop**

| Dataset | Paper ACC | Paper WF1 | Ours ACC | Ours WF1 | Paper drop ACC / WF1 | Our drop ACC / WF1 |
|---|---|---|---|---|---|---|
| IEMOCAP-6 | 55.70 | 54.23 | 59.33 | 59.34 | 17.07 / 18.58 | 11.90 / 11.83 |
| IEMOCAP-4 | 62.88 | 62.22 | 64.16 | 62.92 | 23.65 / 24.30 | 21.84 / 23.06 |
| MELD | 50.15 | 42.27 | 50.11 | 43.55 | 17.55 / 24.44 | 17.48 / 23.00 |
| CMU-MOSEI | 32.98 | 28.28 | 35.54 | 31.19 | 13.84 / 16.65 | 9.56 / 12.77 |

How to read this, in plain words:

- Table A: our full model is slightly under the paper on IEMOCAP-6 and CMU-MOSEI (about 1.5 to 1.8 points of accuracy) and almost equal on IEMOCAP-4 and MELD.
- Table B: removing text hurts a lot on every dataset, which is the paper's point. On IEMOCAP-4 and MELD our no-text result is close to the paper's. On IEMOCAP-6 and CMU-MOSEI ours is higher than the paper's, so our drop is smaller.
- Honest caveat: these numbers pick the best epoch using the test set, which is how the paper does it, so they are a little optimistic. We also have "clean" numbers that pick the epoch on a separate validation set. They are lower. For the no-text model, clean WF1 is 53.88 (IEMOCAP-6), 51.22 (IEMOCAP-4), 43.92 (MELD) and 28.97 (CMU-MOSEI), from 3 seeds. The clean full-model WF1 is 68.09, 81.84, 65.48 and 43.99 from 5 seeds.

## 3. What we are planning to do (experiments and techniques)

Goal: raise the no-text accuracy and weighted F1 (call this T1), and check separately whether the full model also improves (T2). Better no-text does not automatically mean a better full model, so these are two separate claims.

Main technique, the headline experiment: **teach the no-text model using the full model.** The full model (it can see text) is the teacher. The no-text model is the student. The student learns to copy the teacher's soft predictions, on top of its normal loss. This is knowledge distillation. The evaluator's list does not contain this exact case, and it is the one aimed straight at the no-text number.

Other techniques from the evaluator's list, in the order we would try them:

1. **Branch self-distillation (idea A).** The six directional streams in the graph learn from the fused output and from each other. In the no-text model, four of the six streams carry blank text, so this may help.
2. **Text dropout during training.** Randomly blank the text while training the full model, so it learns not to lean on text only. This was tried on CMU-MOSEI earlier and was flat. It is untested on IEMOCAP and MELD.
3. **A no-text model that uses only the visual-acoustic stream.** Skip the streams that contain blank text. One small pilot.
4. **Imbalance-aware soft labels (idea F)** and **soft sentiment-shift targets plus a hierarchy-consistency term (idea C).** Later, on MELD and CMU-MOSEI.
5. Not for this goal: depth self-distillation (B), a causal student (D), a graph-free student (E). They aim at cost and deployment, not accuracy. An LLM teacher (G) is the costliest and riskiest, so it is last or out.

Already tried and rejected earlier, so we will not repeat them: class-weighted loss, the window, shift, depth and lambda sweeps, and the unimodal-head and OGM methods on CMU-MOSEI.

Rules every experiment follows:

- Each run reports two numbers: the paper-style number (best test epoch) and the clean number (epoch chosen on validation, with the validation overlap fixed).
- A change is kept only if the clean 5-seed mean improves and the paper-style number does not get worse. We never keep something because only the test number went up.
- Settings such as the distillation weight and temperature are chosen on validation only.
- Every new option is off by default. A check run with all options off must reproduce the old results exactly.
- Work is done on an experiments branch, nothing goes to main without approval, and we stop after each phase.

Phases and time (estimates, shared GPUs):

| Phase | What | Runs | Est. GPU time |
|---|---|---|---|
| P0 | Check patched code reproduces old no-text runs | 4 | ~1.5 h |
| P0b | Bring no-text clean baselines to 5 seeds | 8 | ~2.5 h |
| P1 | Train one full-model teacher per dataset | 4 | ~1.5 h |
| P2 | Teacher-to-student pilot (IEMOCAP-6, MELD) | 8 | ~3 h |
| P3 | Self-distillation, text dropout, va-only pilots | 8 | ~3 h |
| P4 | Winners only, 5 seeds on 4 datasets | up to 20 per variant | ~7 h per variant |
| P5 | Imbalance and shift ideas | ~8 | ~3 h |

Full detail is in `results/experiments/kd-notext-plan.md`.

## 4. What finished and what did not

Finished and saved:

- The full plan, pushed to GitHub on branch `experiments/p4-kd-notext`.
- The code for teacher-to-student distillation (four new options: `--save_teacher_logits`, `--teacher_logits`, `--kd_alpha`, `--kd_tau`, all off by default). It compiles. It lives on the GPU server in `GraphSmile_p4` and is not yet committed to git.
- A run script, `scripts/p4_run.sh`, and the patch script, `scripts/p4_patch.py`, saved in the repo.

Started on the server, status unknown:

- Four no-text baseline runs (IEMOCAP-6 seeds 3 and 4, MELD seeds 3 and 4). The last time I could see them, they were running at about 16 seconds per epoch (IEMOCAP-6 needs 120 epochs, so about 35 minutes if the GPU is not slowed down).
- Two teacher runs (full model, IEMOCAP-6 and MELD, seed 2024) on GPU 4.

Not done:

- **No result from these runs has been read yet.** I cannot say whether any accuracy went up. I have not claimed one.
- The student distillation runs were not launched, because they need a finished teacher.
- Why: the connection to the server (172.30.1.73) timed out, and a background wait on this PC was also stopped because the PC was low on memory. The server jobs were started with `nohup`, so they should keep going if the server is up, but I have not confirmed that.

## 5. Next steps

1. Check the server is reachable and the six runs are alive or finished.
2. If a teacher finished, launch the student runs (two settings each on IEMOCAP-6 and MELD), pick the setting on validation, and compare with the no-text baseline.
3. Commit the server code and the results to the branch and push.
4. Stop and report before starting the next phase.
