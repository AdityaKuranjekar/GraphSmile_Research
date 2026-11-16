# Next steps after pass 5 (written 2026-11-16)

Goal: raise IEMOCAP-6 full-model accuracy / WF1. Part A (branch self-distillation) was dropped; see `p5-branch-sd.md`.

## Where we stand (5 seeds, full model tva)
- Paper: 72.81 WF1. Our baseline, paper protocol (code-default config): 71.49. SD: 70.09. Clean: baseline 67.94, SD 68.51 (noise).
- Paper-config reproduction (L=7, shift_win=19, lambd 1 1 0.7): 71.17 paper protocol (seed 2024), 68.26 ± 0.94 clean.

## Next tasks, in order
1. **Use the paper config for all new runs.** p5 forgot `--heter_n_layers 7 7 7 --shift_win 19 --lambd 1.0 1.0 0.7 --win 17 17` (see `scripts/run_all.sh`). Add these to `scripts/p5_queue.sh` first.
2. **Optional quick recheck of A at paper config:** α=1 τ=2, 5 seeds × {paper, clean} = 10 runs (~4 h).
3. **A under noisy or missing modalities** (PDF step 7): add a test-time flag that zeroes or adds noise to audio/video, and compare baseline vs SD. This is where self-distillation is most likely to win.
4. **Idea B, depth self-distillation:** the deepest layer's output teaches the shallower layers. It reuses the same aux-head code pattern (heads built last, loss returned from forward for DDP).
5. **Idea F, imbalance-aware KD:** targets Happy (≈46 clean F1, the weakest class) and Happy/Excited confusion.

## How to run (lessons learned)
- Server: `ssh rashmi_2521cs34@172.30.1.73`; worktree `/DATA/rashmi_2521cs34/GS_p5` (IEMOCAP_path patched locally). Queue: `bash p5_queue.sh JOBS 3` (picks the freest GPU, retries on OOM). Launch with `setsid nohup ... < /dev/null & disown`.
- Paper-protocol runs write no JSON; read the last "Test performance" block of the log (best-test epoch).
- Never `pkill -f` a pattern that also matches the ssh command line.
- Rules: report paper + clean numbers; keep only if clean improves and paper does not get worse.
