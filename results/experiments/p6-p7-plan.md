# Passes 6 and 7: depth self-distillation (B) and soft/relational SDP (C)

Written 2026-11-17. Scope: **IEMOCAP-6 only, full model (tva)**. Goal: higher test Acc and WF1 for the
full-depth model. Branch `experiments/p6-depth-soft-sdp` (off p5). Server worktree `/DATA/rashmi_2521cs34/GS_p6`.
Every flag defaults to off; with all flags off the code matches p5/main exactly (checked, see below).

**Config:** all runs use the paper IEMOCAP-6 config (`--heter_n_layers 7 7 7 --shift_win 19 --lambd 1.0 1.0 0.7
--win 17 17`, lr 1e-4, bs 16, hidden 512, drop 0.2, l2 1e-3, 120 epochs). p5 used the code defaults by mistake.
A new same-code baseline is run at this config under both protocols, 5 seeds each.

## B. Depth self-distillation (pass 6)

With residuals on, each graph module outputs `x0 + Σ_{i≤L} FC_i(h_i)`. Exit k is the running sum after k layers,
taken from all three modules (tv, ta, va) and fused exactly like H (same `modal_fusion`, same order), then a
per-exit linear head. Teacher = the final emotion logits (detached).

- `--depth_sd α`: α·τ²·mean_k KL(p_H^τ ‖ p_exit_k^τ), τ = `--sd_tau` (default 2).
- `--depth_ce β`: β·mean_k CE(exit_k, y) (plain deep supervision, the ablation that separates "KD" from "extra labels").
- `--depth_exits`: which k (default 1..L−1 = 1..6). Inference still uses only H; exit WF1/Acc are logged per epoch
  and saved in the JSON (`test_exit_f1`), which gives the early-exit curve for free.
- Sweep (clean, seeds 1, 2, 2024): α=0.3; α=1; β=0.3 only; α=0.3+β=0.3. α starts below p5's 0.5 because idea A
  lost on the paper protocol at α≥0.5.

## C. Soft and relational SDP (pass 7)

A model cannot be its own relational teacher (the loss is zero by construction), so C uses a **separate teacher**:
the clean baseline run of the same seed. At its valid-selected epoch (never test-selected) it saves eval-mode
sentiment logits and H for all 120 train dialogues (`--dump_teacher`; edge weights are restored after the dump,
so the teacher run itself is unchanged). The student reads them by dialogue id (`--teacher_path`).

- Soft shift targets, `--sft_soft β`: shift target = (1−β)·hard + β·soft, with soft taken from the teacher's
  sentiment distribution p = softmax(z/τ), τ = `--sft_tau` (default 2):
  - `differ` (default): 1 − Σ_c p_i(c)p_j(c), the chance that two utterances' sentiments differ;
  - `ordinal`: |E_i[s] − E_j[s]| / 2 with s ∈ {neg 0, neu 1, pos 2}, so neg→pos is a full shift and neg→neu half.
  Loss is soft cross-entropy with mean reduction, so `lambd[2]` keeps its meaning. One-hot input reproduces the
  hard labels exactly (unit test).
- Relational KD, `--rel_kd δ`: per dialogue, MSE between the cosine-similarity matrices of student H and teacher H.
- Sweep (clean, seeds 1, 2, 2024, teacher = same-seed clean baseline): β=0.5; β=1; ordinal β=0.5; δ=1; δ=10.
- Known limit: the teacher's outputs on its own training dialogues are confident (it was fitted on them); τ softens
  this. Cross-fitted teachers (k-fold) are the fix if C shows promise but the targets look too hard.

## Protocol and decision rule (unchanged)

1. Sweep on clean protocol, 3 seeds; pick each pass's winner on **clean valid WF1** only.
2. Confirm each winner on 5 seeds (1, 2, 3, 4, 2024) under both protocols: paper (`--select_on test`, default
   split) and clean (`--clean_valid --select_on valid`). Report Acc, WF1 and Happy F1 against the same-code baseline.
3. Keep only if the clean mean improves AND the paper-protocol number is not worse.
4. If B and C each pass, one combined run (B+C) on 5 seeds × both protocols.

## Running B and C in parallel

One queue, 5 workers (runs need ~1.5 GB, GPUs are shared): `bash scripts/p6_queue.sh scripts/p6_sweep.jobs 5`
from the GS_p6 worktree. The first three jobs are the seed-1/2/2024 clean baselines, which also write the
teachers; B jobs start immediately, C jobs wait for their teacher file. Then the seed 3/4 teachers and the
5-seed paper-protocol baseline. Summary: `python scripts/p5_summary.py 'results/experiments/p6_*.json' 'results/experiments/p7_*.json'`.

## Checks done before launch

- CPU unit tests (`tests/test_p6_p7.py`, run on the server): flags-off output identical and still 4 outputs;
  B flags do not shift the base initialisation; exit at k=L equals H bit for bit; KD gradient reaches exit heads
  but not the teacher head; soft target from one-hot labels equals the hard shift labels for windows 19, −1, 4;
  ordinal target has the same zero pattern with values in {0, 0.5, 1}.
- GPU: 3-epoch flags-off parity p6 vs p5 code at the paper config (seed 1, clean); same run with `--dump_teacher`
  must match too; 2-epoch smoke runs of each flag and of the paper-protocol JSON.

## Teacher diagnostics (2026-11-17, `scripts/p7_teacher_diag.py`, seed 1 and 2 teachers)

- Alignment fine: teacher argmax on its 108 fitted dialogues 75.4 emo / 86.7 sen acc; on the 12 held-out 73.0 / 84.6.
  So the dumps are aligned and the teacher is not memorising its training dialogues.
- Soft shift targets are very soft at τ=2: mean "differ" target 0.36 on same-sentiment pairs vs 0.575 on
  different-sentiment pairs (τ=1: 0.20 vs 0.56). β=1 at τ=2 therefore nearly erases the shift signal. Added arm
  β=0.5, τ=1 (`p7_sft05t1`).
- Relational KD scale: teacher H is nearly collinear within a dialogue (mean off-diagonal cosine 0.963) and the
  cos-matrix MSE between two independent teachers is 3.2e-4, so δ=1 and δ=10 are effectively no-ops. Added δ=100
  and δ=1000. If neither moves anything, the fix is to centre H per dialogue before the cosine (code change, after
  this sweep).
