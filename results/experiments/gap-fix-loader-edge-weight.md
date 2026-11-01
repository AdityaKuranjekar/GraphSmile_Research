# Gap fix: training order and edge_weight in the optimizer

**Branch:** `fix/loader-reshuffle-edge-weight` (from `706e371`).
**Scope:** two bugs in the baseline training loop. Both affect every dataset, including the Phase 0 baseline.
**Status:** fixed in code. CPU regression check passes. The clean-protocol baseline must be rerun on the fixed code before any number is reported.

## Bug 1: training order identical every epoch

**Where (baseline):** `run.py` `get_data_loaders` (lines 221–226) attaches the sampler to the training loader once. Each epoch, `setup_samplers` (lines 258–262, called at 407) built new samplers and called `set_epoch` on those. They were never attached to the loader.

**Effect:** the training loader used the same batch order in every epoch. Training is not reshuffled.

**Evidence (baseline, instrumented copy, synthetic data):**
```
train order epoch 0: [12, 29, 35, 10, 11]
train order epoch 1: [12, 29, 35, 10, 11]
```

**Fix:** `setup_samplers` now calls `set_epoch` on the loader's own sampler. The per-epoch reload of the dataset that existed only to feed `setup_samplers` is removed.

## Bug 2: edge_weight not in the optimizer

**Where (baseline):** `module.py` lines 39–41 create `edge_weight` lazily inside `forward`. `run.py` line 331 builds the AdamW optimizer from `model.parameters()` before any forward pass, so the parameter does not exist yet and is never added.

**Effect:** the 3 `edge_weight` tensors (one per multimodal graph) are never updated. They stay at 1.0. The paper describes these weights as trainable (Sec. III-B).

**Evidence (baseline, instrumented copy):**
```
optimizer vs model params: [('28', '31'), ('28', '31')]
```
The model has 31 parameters; the optimizer holds 28. The 3 missing are the `edge_weight` tensors.

**Fix:** `trainer.py` `sync_optimizer` adds any model parameter not yet in the optimizer. It is called after each training forward pass. It does not remove replaced parameters: AdamW skips parameters whose grad is None.

**Evidence (fixed code):**
```
optimizer vs model params: [('31', '31'), ('31', '31')]
EW_DRIFT 0.000112 ... 0.000121   (16 steps, lr 1e-5, synthetic data)
```

**Expected size of the effect:** Adam moves each weight by about the learning rate per step. At MOSEI's learning rate (8e-5) over a full run, the weights can move by a fraction of a unit, so the effect may be modest. Measure it on real runs.

## Regression check

`scripts/check_fixes.py`: runs the real `run.py` on a synthetic CMU-MOSEI pickle, in a temporary copy. It checks that the order changes between epochs and that the optimizer count equals the model count.

- Fixed code: `ALL CHECKS PASSED`.
- Baseline (instrumented): both checks fail, as described above.

Local environment: torch 2.5.1 CPU, sklearn 1.9.1, pandas 3.0.5, numpy 2.5.3. The server pins torch 2.0.0, numpy 1.26.4, pandas 2.2.3, sklearn 1.6.1. Run `scripts/check_fixes.py` on the server once before the rerun.

## Not fixed (found while reading)

- **`--loss_type auto_loss`** is broken. `run.py` builds an optimizer with the `AutomaticWeightedLoss` parameters for `auto_loss`, but `trainer.py` matches only `auto`. `auto_loss` matches no branch, so `loss` is never set. Not used by any configuration in the plan.
- **`--loss_type auto`** creates a new `AutomaticWeightedLoss` on every batch, so its learned weights reset each batch. Not used by any configuration in the plan.

## Consequences

1. **Rerun the clean-protocol baseline on the fixed code** for all four datasets (IEMOCAP-6, IEMOCAP-4, MELD, CMU-MOSEI), 5 seeds each. The old clean numbers (`p0c_*`, `p0_mosei_*`, and the in-progress Phase 0 run) were produced with both bugs and are not the new baseline.
2. **Keep the paper-protocol numbers from the original code** as the reference column, because the paper's own code has these bugs. Label them as such.
3. **Flag-branch merge:** the Phase 2/3 branch (`experiments/mosei-gap`) also edits `run.py` and `trainer.py`. Expect conflicts in the setup of samplers and the training loop. Rebase that branch onto this fix before merging.
4. **Do not change the server checkout while a Phase 0 job is running.** Each seed starts a new process that reads `run.py` from disk.
