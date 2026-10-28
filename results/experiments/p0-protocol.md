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
