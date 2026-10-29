# p0-protocol: validation-based selection, --seed, per-class output

**Decision: kept (tooling only; no model change, defaults unchanged).**

## Change
- `--seed` (default 2024). `trainer.set_seed` sets the value that `seed_everything()` uses.
- `--select_on {test,valid}` (default `test` = original behaviour). With `valid`, the epoch is
  the one with the best `valid_f1_emo`; test is only read at that epoch.
- `--out_json PATH`: writes epoch, valid ACC/F1, valid and test per-class F1, test confusion matrix.
  The log also gets one `RESULT_JSON {...}` line.
- Nothing else in `run.py` or the scripts changed. Defaults are the paper configuration.

## Verification (CPU, synthetic pickle, not a result)
Synthetic IEMOCAP-format pickle (40 dialogues, random features), 3 epochs, tiny model.
- Same seed twice gives identical selected epoch and metrics (seed 1: epoch 1, valid 4.48, test F1 2.11).
- Different seed gives a different result (seed 2: epoch 2, valid 7.03).
- Default run (no new flags) prints the original `Acc: ..., F-Score: ...` line.
- The first seed-1 attempt hit a pandas circular-import error on a cold start; it did not recur.

These numbers say nothing about model quality. They only show the plumbing works.

## Findings from the existing log (`results/logs/IEMOCAP_full.log`)
- Confusion matrix row 4 (Excited) has support 299, not 249. The 249 in the brief is the
  Frustrated diagonal. So 34 of **299** Excited are predicted Happy. Happy: 44 of 144 go to Excited (matches).
- The reported epoch is chosen on test F1 (`run.py`), so the current 71.17 is optimistically biased.
  Rerunning with `--select_on valid` will probably give a lower number than the paper-comparison tables.
- Validation is the first 10% of train dialogues (`get_train_valid_sampler`), a small and fixed set,
  so `valid_f1_emo` will be noisy. Not yet measured how small; check dialogue count on the GPU machine.
- RUNBOOK.md has no synthetic-pickle smoke-test recipe; the one used here is `mk.py` logic above
  (12-tuple IEMOCAP layout from `dataloader.py`).

## GPU request (baseline under the clean protocol)
```
for s in 2024 1 2 3 4; do
  python -u run.py --gpu 0 --port 1530 --classify emotion --dataset IEMOCAP --epochs 120 --textf_mode textf0 \
    --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 \
    --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7 --l2 1e-3 --seed $s --select_on valid \
    --out_json results/experiments/p0_seed$s.json 2>&1 | tee results/logs/p0_IEMOCAP_seed$s.log
done
```
(`--l2 1e-3` matches the earlier paper-setting runs.) Send back the JSON files and logs.
Seeds are 5 rather than 3 because the valid set is small. Mean and std over them is the baseline
every later pass must beat.

## Result: clean-protocol baseline (IEMOCAP-6, paper config, L2 1e-3, `--select_on valid`)
Command: the loop in the GPU request above (run via `run_p0.sh`, GPU 1, shared A100).
Files: `results/experiments/p0_seed{2024,1,2,3,4}.json`, `results/logs/p0_IEMOCAP_seed*.log`.
Std is the sample std (ddof=1) over 5 seeds.

| Seed | Epoch | Valid WF1 | Test ACC | Test WF1 | Happy | Sad | Neu | Ang | Exc | Fru |
|---|---|---|---|---|---|---|---|---|---|---|
| 2024 | 112 | 87.88 | 70.24 | 70.15 | 53.5 | 82.2 | 69.9 | 69.2 | 76.7 | 64.3 |
| 1 | 96 | 86.51 | 70.67 | 70.70 | 58.1 | 82.2 | 69.4 | 68.3 | 77.4 | 65.2 |
| 2 | 90 | 88.29 | 70.86 | 70.90 | 61.0 | 82.1 | 69.7 | 68.0 | 76.9 | 65.2 |
| 3 | 112 | 89.44 | 71.47 | 71.47 | 61.4 | 83.5 | 69.3 | 68.3 | 79.0 | 65.3 |
| 4 | 114 | 88.92 | 70.86 | 70.86 | 59.4 | 81.7 | 69.6 | 67.1 | 77.9 | 65.6 |
| **mean ± std** | | 88.21 ± 1.12 | 70.82 ± 0.44 | **70.82 ± 0.47** | **58.69 ± 3.18** | 82.34 ± 0.67 | 69.57 ± 0.23 | 68.19 ± 0.75 | 77.57 ± 0.92 | 65.12 ± 0.49 |
| paper | | | 72.77 | 72.81 | 63.09 | 83.16 | 71.07 | 71.38 | 79.66 | 66.84 |

Happy/Excited confusion (test, per seed): Happy->Excited 48, 37, 32, 36, 38 of 144;
Excited->Happy 33, 34, 42, 31, 32 of 299.

### What this shows
- Seed 2024 under valid selection gives Happy 53.5 / WF1 70.15. The earlier test-selected 71.17 was
  about 1 point optimistic for this seed.
- Happy varies a lot across seeds (std 3.18; range 53.5 to 61.4). The earlier 54.87 was a low draw.
  Happy mean is 58.69, so the gap to the paper is -4.4, not -8.2, and the paper's 63.09 is about 1.4 std above our mean.
- Overall WF1 is 70.82 ± 0.47 against 72.81 in the paper: a gap of 2.0, about 4 std. This gap is not seed noise.
  It is larger than before (71.17 vs 72.81) because the old number used test selection.
- Every class is below the paper by 0.8 to 4.4 points, so the shortfall is not only a Happy problem.
- Valid WF1 is about 88 while test WF1 is about 71. The valid split is the first 10% of the train
  dialogues and looks much easier than test (probably shared speakers with train; not verified). Valid is therefore a weak
  proxy for test and its seed-to-seed noise is similar in size to the gains we are looking for.
  Valid Happy F1 is 69.1 on average (per-class valid values are in the JSON files).

Decision: **kept**. This is the baseline for later passes: test WF1 70.82 ± 0.47, Happy 58.69 ± 3.18.
A later change counts as a gain only if the 5-seed mean beats this by more than the seed spread.
