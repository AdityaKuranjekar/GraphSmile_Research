# Plan: raising the no-text (va) accuracy and WF1 with knowledge distillation

Status: plan only, nothing run yet. Pass id: **p4-kd** (continues p0 to p3 and the gap-plan phases in `results/experiments/README.md` and `gap-results-summary.md`). Branch `experiments/kd-notext-plan`. Source of ideas: `KD fix on graphsmile.pdf` (the evaluator's ideas A-G).

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

## 10. Corrections after reviewing earlier passes (these override sections 1, 4 and 6 where they differ)

### 10.1 Clean baselines (the keep rule compares these, not the test-selected numbers in section 1)

Source: `gap-results-summary.md`, clean protocol (`--clean_valid --select_on valid`), fixed code (branch `fix/loader-reshuffle-edge-weight`: per-epoch reshuffle, `edge_weight` in optimizer). Mean +- sample std, WF1/ACC in %.

| Dataset | T2 full ACC / WF1 (5 seeds) | T1 no-text ACC / WF1 (**3 seeds only**) | Delta text (WF1) |
|---|---|---|---|
| IEMOCAP-6 | 68.16 +- 0.55 / 68.09 +- 0.42 | 54.45 +- 0.89 / 53.88 +- 1.14 | 14.20 |
| IEMOCAP-4 | 81.82 +- 1.15 / 81.84 +- 1.14 | 52.42 +- 0.22 / 51.22 +- 0.33 | 30.62 |
| MELD | 66.67 +- 0.12 / 65.48 +- 0.13 | 48.82 +- 0.21 / 43.92 +- 1.00 | 21.56 |
| CMU-MOSEI | 46.50 +- 0.23 / 43.99 +- 0.47 | 38.24 +- 0.15 / 28.97 +- 1.07 | 15.02 |

The test-selected numbers in section 1 (our rerun on main) stay as the paper-comparable column only. Note the clean no-text numbers are much lower than the test-selected ones (for example IEMOCAP-4 51.2 vs 62.9 WF1), which shows how much epoch selection on test inflated them.

**New phase P0b (must run first):** T1 baseline `va` with `--clean_valid --select_on valid`, seeds 2024, 1, 2, 3, 4 on the 4 datasets, to bring the no-text baseline to 5 seeds (2 more seeds on top of the existing 3, so 8 new runs for seeds 3 and 4, about 2.5 h; the other seeds are reused from Phase 0c). Without 5 seeds the keep rule cannot pass or fail anything.

**Code base:** all p4 work branches from `fix/loader-reshuffle-edge-weight` (the fixed code), not main. The regression gate (section 2) is then: flags off must reproduce the Phase 0 / 0c JSON for seed 2024 exactly. The `results/rerun/` logs remain the reference for the main-branch code only.

### 10.2 What earlier passes already settled (do not repeat)

- Class-weighted loss (p1c, IEMOCAP-6): WF1 68.38 +- 0.62 vs 68.26, rejected. So idea F cannot rely on plain class weights; only logit adjustment or KD soft labels remain open, with low prior.
- p2 sweeps (window, shift, depth, lambda): all rejected on IEMOCAP-6. Not retried.
- Text dropout (`--text_drop 0.3`, MOSEI, 3 seeds): 44.17 +- 0.36 vs 44.12, flat. Unimodal heads (`--uni_aux`) and OGM lowered MOSEI WF1 (43.42, 43.61). These were run on MOSEI only; **IEMOCAP-6, IEMOCAP-4 and MELD are untested**, and the evaluator's 6 to 7 point cut in text dominance (target in `gap-results-summary.md` section 3) is still unmet. The p4 pilots therefore run modality dropout and A on IEMOCAP-6 and MELD, not MOSEI.
- No-text MOSEI WF1 (28.97) is lowest of all, because the model collapses to majority classes without text. A KD teacher may help here by giving soft class structure, but this is untested.

### 10.3 Known bug that affects the logit dump

`trainer.py` does `vids += data[-1]` once after the loop, so `vids` holds only the last batch's dialogue ids. A dump keyed by it would be wrong. The dump must collect ids inside the batch loop, from a non-shuffled evaluation pass over train, valid and test (the train loader reshuffles each epoch on the fixed code). This change is flag-gated (`--save_teacher_logits`) and covered by the regression gate.

### 10.4 How the best-epoch logits are obtained

No checkpoint exists, and training continues after the best epoch. Chosen approach: **in-memory snapshot**. Each time valid F1 improves, run one extra no-grad pass over train/valid/test (non-shuffled), keep the logits and ids in CPU memory, and write them to `--save_teacher_logits` at the end of training. The cost is about 3 extra eval passes per improving epoch (small against training). No `state_dict` file is written, so no large files enter git (rule: no pickles or logs over 1 MB in git; the logits go to the GPU server only).

### 10.5 Teacher budget (replaces the P1 row)

One fixed teacher per dataset (the clean-valid `tva` run with seed 2024) is shared by all student seeds. This is cheaper and keeps teacher variance out of the student comparison, but it means the student's seed std does not include teacher variance, so state that in the write-up. Teacher runs: 4 (one per dataset), about 1.5 h, replacing the 6-run P1. Optional later check: a second teacher seed on IEMOCAP-6 only, to see whether results depend on the teacher.

### 10.6 Updated phase list

| Phase | What | Runs | Est. GPU time |
|---|---|---|---|
| P0 | Regression gate, flags off, seed 2024, `va` on 4 datasets vs Phase 0c JSON | 4 | ~1.5 h |
| P0b | `va` clean baseline, seeds 3 and 4, 4 datasets (completes 5 seeds) | 8 | ~2.5 h |
| P1 | One `tva` teacher per dataset, clean-valid, logits saved (seed 2024) | 4 | ~1.5 h |
| P2 | **H pilot**: `va` KD student, 1 seed, IEMOCAP-6 and MELD, grid alpha {0.5, 1} x tau {2, 4}, chosen on valid | 8 | ~3 h |
| P3 | A (branch KD), modality dropout, va-only-stream pilots, 1 seed, IEMOCAP-6 and MELD | 8 | ~3 h |
| P4 | Winners only: 5 seeds x 4 datasets | up to 20 per variant | ~7 h per variant |
| P5 | F (logit adjustment) and C / hierarchy pilots, MELD and MOSEI | ~8 | ~3 h |

Stop for the supervisor's go-ahead after P0/P0b (baselines in hand) and again after each later phase. The first thing to run on approval is P0, then P0b.
