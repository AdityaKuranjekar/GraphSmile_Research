# Gap plan: raising CMU-MOSEI accuracy, cutting text dominance, and the ablations/limitations

This plan answers the evaluator's remark (A to D below). It is organised as phases, each split into numbered steps. Every step lists whether it needs a GPU, what it depends on, and its status.

**Evaluator asks**
- **(A)** Raise CMU-MOSEI ACC and WF1. MOSEI only.
- **(B)** Explain why CMU-MOSEI is lower than the other datasets.
- **(C)** Reduce text dominance by 6 to 7 WF1 points on every dataset.
- **(D)** Ablations and limitations.

**Protocol rule (applies everywhere):** every number that is compared against a baseline uses the **clean protocol**: `--clean_valid --select_on valid`, epoch chosen on validation. The paper's test-selected numbers are shown only as a reference column.

**Naming:** files for this plan use the `gap-` prefix. The existing `p0`, `p1`, `p2`, `p3` files belong to earlier passes and are unchanged.

## Status overview

| Phase | Step | What | Hardware | Depends on | Status |
|---|---|---|---|---|---|
| 0 | 0a | Clean baseline, 5 seeds, IEMOCAP-6 | GPU | none | Done earlier (`p0c_seed*.json`) |
| 0 | 0b | Clean baseline, 5 seeds, MOSEI, MELD, IEMOCAP-4 | GPU | none | **Running** (MOSEI seeds 2024 and 1 finished) |
| 0 | 0c | Clean no-text baseline, 3 seeds, all four datasets | GPU | 0b flags | Not started |
| 1 | 1.1 | Class counts, train vs test | CPU | none | **Done** (`gap-phase1-mosei-audit.md`) |
| 1 | 1.2 | Feature audit and split size vs paper | CPU | none | **Done** (`gap-phase1-mosei-audit.md`) |
| 1 | 1.3 | Modality-only runs: text only, text+audio, text+visual | GPU | 0b | Not started |
| 1 | 1.4 | Context: window 0 vs window 5 | GPU | 0b | Not started |
| 1 | 1.5 | Hypothesis table, supported / refuted / unclear | CPU | 1.1–1.4 | Partly done (1.1, 1.2 rows) |
| 2 | 2.1 | Auxiliary regression head on the raw score | GPU | 0b, 1.5 | Code not written |
| 2 | 2.2 | Ordinal soft labels (KL divergence) | GPU | 0b | Code not written |
| 2 | 2.3 | Focal loss with inverse-frequency weights | GPU | 0b | Code not written |
| 2 | 2.4 | Validation-only hyperparameter sweep | GPU | 0b | Not started |
| 2 | 2.5 | CORAL ordinal regression (optional) | GPU | 2.1–2.2 failed | Not started |
| 2 | 2.6 | Confirm the kept change on 5 seeds | GPU | 2.1–2.5 | Not started |
| 3 | 3.1 | Pre-fusion text dropout | GPU | 0c | Code not written |
| 3 | 3.2 | Unimodal auxiliary heads | GPU | 0c | Code not written |
| 3 | 3.3 | OGM-GE gradient modulation | GPU | 0c | Code not written |
| 3 | 3.4 | PMR prototype rebalancing (only if 3.1–3.3 miss the target) | GPU | 3.1–3.3 | Not started |
| 4 | 4.1 | Sentiment ablations, `--classify sentiment` | GPU | none (existing code) | Not started |
| 4 | 4.2 | Depth and window sweeps, MOSEI first | GPU | 0b | Not started |
| 4 | 4.3 | MOSEI to 5 seeds for the final model | GPU | 2.6 or 3.x | Not started |
| 4 | 4.4 | Limitations section | CPU | 1–4.3 | Not started |
| 5 | 5.1 | Evaluator package: before/after table | CPU | all | Not started |
| 5 | 5.2 | Negative results kept visible | CPU | all | Not started |
| 5 | 5.3 | Update `REPRODUCTION.md` and `RUNBOOK.md` | CPU | all | Not started |

## Phase 0: fix the measurement

Goal: a clean-protocol baseline that every later change is measured against.

- **0a. IEMOCAP-6, clean, 5 seeds.** Done earlier. Files: `results/experiments/p0c_seed*.json`. WF1 68.26 ± 0.94.
- **0b. MOSEI, MELD, IEMOCAP-4, clean, 5 seeds each.** GPU 0, started 13:54 IST on 1 Nov 2026 as one background job (`run_phase0.sh`).
  - MOSEI seed 2024 (clean): test ACC 44.36, WF1 42.62, epoch 44 chosen on validation. Compare the paper's test-selected 46.82 / 44.93 and our earlier 45.10 / 43.96.
  - MOSEI seed 1: finished 14:56, `exit=0`. Numbers to be read from `results/experiments/p0_mosei_seed1.json`.
  - Remaining: MOSEI seeds 2, 3, 4; MELD seeds 2024–4; IEMOCAP-4 seeds 2024–4. Estimated finish about 18:10 IST.
- **0c. No-text baseline, clean, 3 seeds, all four datasets.** Not in the original plan. It is needed for Δtext in Phase 3, because Δtext computed from test-selected numbers would not be accepted. Cost about 3 to 4 GPU hours. Run after 0b, or in parallel on a free GPU.
- **Deliverable:** a Phase 0 table with mean ± std across seeds, per dataset, paper protocol next to clean protocol.

## Phase 1: diagnose why MOSEI is low

Goal: decide which hypotheses H1 to H7 hold, before choosing a fix.

- **1.1. Class counts.** CPU. **Done.** Train and test distributions differ. See `gap-phase1-mosei-audit.md`.
- **1.2. Feature audit and split size.** CPU. **Done.** Alignment and values are clean. Visual has 168 zero rows. Our test set is 4,184 utterances; the paper's is 4,662.
- **1.3. Modality-only runs.** GPU, MOSEI, clean protocol, 3 seeds each: text only, text+audio, text+visual.
  - Existing ablations are "no X" only. These runs give the contribution of each modality on its own.
  - Check how `--modals` handles a single modality before running. Do not assume it.
  - Cost: 3 configs × 3 seeds × about 30 min = about 4.5 GPU hours. Runs can use separate GPUs.
- **1.4. Context.** GPU, MOSEI, clean protocol, 3 seeds: window 0 against window 5.
  - Dialogues have median length 5, so context is available. This run tests whether it is used.
  - Cost: 2 configs × 3 seeds ≈ 3 hours.
- **1.5. Hypothesis table.** CPU. Fill in H1 to H7 with supported / refuted / unclear. Rows for H1, H3, H6 and H7 can be filled now. H2 needs 1.4. H4 needs Phase 0. H5 needs Phase 2.4.

## Phase 2: raise MOSEI ACC and WF1 (MOSEI only, validation-selected)

Goal: a change that beats the Phase 0 MOSEI baseline by more than its seed std.

Each candidate is a flag that defaults to off, so existing runs reproduce exactly.

- **2.1. Auxiliary regression head on the raw score.** Keep the continuous score that `dataloader.py` currently discards. Add a regression loss. First candidate, because it adds ordinal information without changing the label set.
- **2.2. Ordinal soft labels.** Put mass on the adjacent classes; use KL divergence. Low risk.
- **2.3. Focal loss, γ = 1.0, with inverse-frequency weights.** Class weighting was tested only on IEMOCAP-6 (no gain), so MOSEI is untested. Judge on test WF1 with care: test has only 22 Highly Negative utterances.
- **2.4. Validation-only hyperparameter sweep.** Learning rate, dropout, hidden size, epochs. MOSEI has only ever run at the Table III defaults.
- **2.5. CORAL.** Optional. Only if 2.1 and 2.2 both fail.
- **2.6. Confirmation.** The kept change runs on 5 seeds. Keep it only if it beats the baseline by more than the baseline's seed std.

Cost: each candidate is 3 seeds × about 30 min = 1.5 GPU hours. Four candidates ≈ 6 GPU hours, or about 2 hours wall-clock on three GPUs.

## Phase 3: reduce text dominance (all four datasets)

Goal: Δtext = WF1(all modalities) − WF1(no text) falls by 6 to 7 points on every dataset, with full-model WF1 not lower than the clean baseline beyond its seed std.

Current Δtext (test-selected numbers, from `REPRODUCTION.md`): IEMOCAP-6 11.8, IEMOCAP-4 23.1, MELD 23.0, MOSEI 12.8. Phase 0c replaces these with clean-protocol numbers.

Target after the cut: about 5 on IEMOCAP-6, 16–17 on IEMOCAP-4 and MELD, 6 on MOSEI.

- **3.1. Pre-fusion text dropout**, p = 0.15–0.20, training only. Cheapest.
- **3.2. Unimodal auxiliary heads**: a linear head per modality, each with its own cross-entropy on the 7-class labels.
- **3.3. OGM-GE gradient modulation** (Peng et al., CVPR 2022). The original scales gradients with a discrepancy-based coefficient, not a fixed constant.
- **3.4. PMR** (prototype rebalancing), only if 3.1–3.3 leave Δtext above target.

Guardrail: a text-dominance cut can come from making text worse. Report Δtext next to full-model WF1 every time.

Cost: each method is 3 seeds × 4 datasets ≈ 3.5 GPU hours.

## Phase 4: ablations and limitations

- **4.1. Sentiment ablations.** No L_e and No L_e & L_o with `--classify sentiment` on all four datasets: 8 runs, about 1.5 GPU hours. Replaces the collapsed MSAC rows. This step has no code dependency, so it can run at any time.
- **4.2. Depth and window sweeps**, MOSEI first (paper Figs. 4 and 5). The window sweep overlaps with 1.4; reuse those results.
- **4.3. MOSEI to 5 seeds** for the final model only.
- **4.4. Limitations section (CPU).** Test-selected protocol in the paper; MOSEI test set differs from paper (4,184 vs 4,662); label distribution shift; feature size and source unclear for audio; zero visual rows; untuned baselines; monologue versus conversation; single-seed rows are noise below about 1 point.

## Phase 5: evaluator package (CPU)

- **5.1.** Before/after table: paper protocol and clean protocol, with each phase's gain and seed std.
- **5.2.** Keep the negative results visible: class weighting, shift windows, the L2 check, and any failed Phase 2 or 3 candidate.
- **5.3.** Update `REPRODUCTION.md` and `RUNBOOK.md` to match.

## GPU schedule

Server: five A100-40GB, shared with other users. Our Phase 0 job runs on GPU 0. GPUs 1 and 3 are nearly full. GPUs 2 and 4 have room.

| Order | Work | GPU | Wall-clock estimate |
|---|---|---|---|
| Now | Phase 0b (running) | 0 | until about 18:10 IST |
| Now | Phase 4.1 can start on a free GPU | 4 | about 2 hours |
| After 0b | Phase 0c no-text runs | 0 and 2 | about 2 hours on two GPUs |
| After 0b | Phase 1.3 and 1.4 | 0, 2 | about 3 hours on two GPUs |
| After 1.5 and code ready | Phase 2 candidates | 0, 2, 4 | about 2 hours per candidate |
| After 0c | Phase 3 | 0, 2, 4 | about 4 hours per method |
| After 2 and 3 | Phase 4.2 and 4.3 | any | about 4 hours |

The estimates assume the other users' load stays where it is now. Measure the GPU before starting a second job on the same GPU.

## Code rules

- New methods are flags, default off, one flag per method.
- Losses in `losses.py`. Modality changes in `module.py`. Do not fork `run.py`.
- Phase 2 and Phase 3 work on branch `experiments/mosei-gap`, merged after each phase.
- Every run's flags are recorded in its log (`Namespace(...)` line).
