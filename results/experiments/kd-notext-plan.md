# Plan: raising the no-text (va) accuracy and WF1 with knowledge distillation

Status: plan only, nothing run yet. Branch `experiments/kd-notext-plan`. Source of ideas: `KD fix on graphsmile.pdf` (the evaluator's ideas A-G).

## 1. Goal and the two success criteria

The evaluator's end goal is a better no-text ablation, hoped to lift overall ACC and WF1 too. These are two separate claims, so they are reported separately.

| Target | Definition | Baseline to beat (earlier = rerun, %, ACC / WF1) |
|---|---|---|
| **T1: no-text** | Retrained `--modals va` model (Table VI style, like-for-like with our drop table) | IEMOCAP-6 59.33 / 59.34, IEMOCAP-4 64.16 / 62.92, MELD 50.11 / 43.55, MOSEI 35.54 / 31.19 |
| **T2: full** | `--modals tva` model | IEMOCAP-6 71.23 / 71.17, IEMOCAP-4 86.00 / 85.98, MELD 67.59 / 66.55, MOSEI 45.10 / 43.96 |

- "Better no-text gives better overall" is a hypothesis, not a given. Only some ideas (see section 3) can plausibly move the full model.
- A second no-text number exists: a `tva` model evaluated with text zeroed (modality-dropout robustness). It is a different metric. Label it "robustness" and never put it in the paper-comparison column.

## 2. Protocol (saved rules, apply to every run)

- Every run reports two numbers: (a) paper-comparable: `--select_on test`, default split; (b) clean: `--clean_valid --select_on valid`.
- Keep a change only if the clean 5-seed mean improves AND the paper-comparable number does not get worse.
- Tune alpha, tau, gamma and the dropout rate on valid only. Never choose anything on test.
- All new flags default off. Work on `experiments/<pass-id>` branches. No push to main without asking. Stop after each pass.
- **Step 0 of every pass (regression gate):** with all new flags off, the patched code must reproduce `results/rerun/*.log` epoch for epoch. Runs are deterministic (verified), so any difference means the patch is wrong. This is the "change the model cautiously" check.

## 3. Ranking of the PDF ideas

| Rank | Idea | Verdict |
|---|---|---|
| 1 | **H. Privileged-modality KD (new, not in the PDF)**: teacher = `tva`, student = `va` | The only idea aimed squarely at T1. None of the PDF's A-G has "teacher sees text, student does not". Headline experiment. |
| 2 | A. Branch self-distillation | The six directional streams teach each other. In `va` mode 4 of the 6 streams carry zeroed text, so it may help T1 and T2. Cheap, no external teacher. |
| 3 | Text-modality dropout while training `tva` (PDF point 7) | One model that survives missing text. Helps the robustness number and maybe T2. A few lines of code. |
| 4 | F. Imbalance-aware KD / logit adjustment | MELD no-text WF1 (43.55) is far below full (66.55), mostly minority classes. `--class_weight` already exists; try logit adjustment too. |
| 5 | C. Soft/relational SDP + hierarchy-consistency term | Mainly MOSEI. The hierarchy term (aggregate emotion probs through the Table I mapping, match the sentiment head) is cheap. |
| defer | B depth, D causal student, E graph-free student | These target cost and deployment, not accuracy. Not run for this goal. |
| last / out | G. LLM teacher | Highest cost and risk; the teacher's own accuracy must be checked first. Out of scope for now. |

Can move T2 (full): A, modality dropout, F, C. Aimed at T1 only: H.

## 4. Real insertion points (from the code)

- `model.py` `forward` returns `logit_emo, logit_sen, logit_shift, feat_fusion`. `feat_fusion` is the fused H (mean of six `modal_fusion` outputs over `featheter_tv[0|1]`, `featheter_ta[0|1]`, `featheter_va[0|1]`). `emo_output` and `sen_output` are linear heads on H.
- The six directional streams are those `featheter_*` pairs. For idea A, add flag-gated auxiliary linear heads on each `modal_fusion(featheter_x)` and return them as extra outputs.
- `trainer.py` builds `loss_emo`, `loss_sen`, `loss_sft` and combines them by `loss_type` (`emo_sen_sft`). The KD term goes here:
  `loss += alpha * tau^2 * KL(softmax(teacher / tau) || softmax(student_emo / tau))`, plus optional gamma-weighted branch KL and delta-weighted relational loss on `feat_fusion`.
- Text zeroing for `va` happens in `trainer.py` (`'t' not in modals` zeroes `textf0..3`). Modality dropout is the same zeroing applied with probability p during training only.
- `run.py` chooses the best epoch in the `--select_on` block (about lines 502-536). **The code saves no checkpoint at all** (no `state_dict` or save call), so a teacher must be produced first.

## 5. Practical KD pitfalls

1. **Getting teacher logits.** Add `--save_teacher_logits <path>`: train a `tva` model with `--clean_valid`, then at the clean-valid best epoch dump emotion logits for train/valid/test keyed by (dialogue id, utterance index). The student loads them with `--teacher_logits`. Only rank 0 writes (DDP and port setup unchanged).
2. **Teacher chosen on valid only**, never test, so the student inherits no test leakage.
3. **Train-set overconfidence.** A teacher fitted on train is near one-hot on train. Use tau in {2, 4}. If soft labels stay too sharp, use cross-fitted teacher logits (k-fold over train dialogues).
4. **Alignment under shuffling.** Index by dialogue id and position, and assert in the trainer that the batch labels equal the stored labels.
5. Teacher and student must use the identical `--clean_valid` split, so the teacher never saw the student's valid or test data.

## 6. Phases and budget

Reference: 8 runs (4 datasets x 2 variants) took about 2.5 h on a shared GPU, roughly 20 min per run. Estimates only.

| Phase | What | Runs | Est. GPU time |
|---|---|---|---|
| P0 | Regression gate on patched code, flags off (va, 4 datasets) | 4 | ~1.5 h |
| P1 | Teacher `tva`, clean-valid, save logits (IEMOCAP-6 and MELD first, 3 seeds) | 6 | ~2.5 h |
| P2 | **H pilot**: `va` student with KD, 1 seed, IEMOCAP-6 and MELD, grid alpha {0.5, 1} x tau {2, 4}, picked on valid | 8 | ~3 h |
| P3 | A (branch KD), modality dropout and the va-only-stream pilot, 1 seed, IEMOCAP-6 and MELD | 8 | ~3 h |
| P4 | Expand only variants that win the pilot on clean valid: 5 seeds x 4 datasets | up to 40 per variant | ~13 h per variant |
| P5 | F and C / hierarchy pilots (MELD, MOSEI) | ~8 | ~3 h |

Gate: nothing reaches P4 unless it wins its 1-seed clean-valid pilot. Stop and report after each phase.

## 7. Cautious model changes (accuracy only)

- Every architecture change is a flag, default off, and must pass the regression gate.
- Candidate for `va` only: skip `hetergconv_tv` and `hetergconv_ta` (they contain the zeroed text node, adding a constant vector to graph edges) and fuse only the `va` stream. One-seed pilot in P3. It changes capacity, so it is kept only under the combined keep rule.
- The full-model default path is not touched.

## 8. Reporting

Per variant: ACC / WF1 for T1 (and T2 where relevant), both protocols, mean and std over 5 seeds, drop versus full, and the paper's no-text number for reference (IEMOCAP-6 55.70 / 54.23, IEMOCAP-4 62.88 / 62.22, MELD 50.15 / 42.27, MOSEI 32.98 / 28.28). Output: `results/experiments/<pass-id>.md` plus per-run JSON via `--out_json`.

## 9. Honest expectations

- Our no-text numbers already beat the paper's on IEMOCAP-6 (59.3 vs 54.2) and MOSEI (31.2 vs 28.3). The target here is gain over our own baseline.
- A text teacher could plausibly help most on IEMOCAP (text is worth about 12 points) and MELD WF1 (about 23 points of headroom). MOSEI is uncertain.
- Improving the full model (T2) is a weaker claim and likely small. Report it separately, not as a consequence of T1.
