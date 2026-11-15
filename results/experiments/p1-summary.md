# p1 summary: class-weighted loss under the clean protocol (with the p0 protocol baseline)

Facts only. Every number is read from the JSON or log files on this branch. "not measured" means no file exists for it.

## 1. Identity
- **Branch:** `experiments/p0-protocol`
- **Latest commit before this summary:** `d12f026` (origin head at the time of writing)
- **Base commit:** `16f12b6` on `claude/gifted-galileo-l5kpj3`, the state of the repo when this work started
- **Pass id:** p1. Tests `--class_weight`, an inverse-frequency weighted emotion loss, under the clean protocol (`--clean_valid --select_on valid`). The p0 protocol flags (`--seed`, `--select_on`, `--out_json`) and the clean split (`--clean_valid`) are included because p1 depends on them.
- **Not covered here:** the provenance checks (`results/experiments/p1b-provenance.md`) and the p2 results. The p1b branch's p2 commits were made by another session. I did not run or verify them.

## 2. Code changes (`16f12b6..d12f026`)
Files: `run.py` (+87 lines), `trainer.py` (+9), `scripts/build_comparison_pdf.py` (new), `scripts/build_paper_vs_reproduced_pdf.py` (new), `docs/*.pdf`, `results/experiments/*` (notes, JSON, logs). No model or architecture code changed.

| Flag | Default | What it does |
|---|---|---|
| `--seed` | 2024 | Sets the seed used by `seed_everything`. The base code fixed it at 2024. |
| `--select_on {test,valid}` | `test` | `test` = base behaviour (best test F1 epoch). `valid` = epoch chosen by `valid_f1_emo`, with test read only at that epoch, and a `RESULT_JSON` line printed. |
| `--out_json PATH` | `''` | Writes the selected-epoch JSON. Only when `--select_on valid`; under `test` it writes nothing. |
| `--clean_valid` | off | Trains on `idx[split:]` and validates on `idx[:split]`, mapping sampler positions back to dataset indices. Off = the base samplers. |
| `--class_weight` | off | Emotion NLL loss weighted by `N/(K·n_c)`, with counts taken from the train dialogues the loader actually draws. |

Also always printed (logging only, not a behaviour change): one `split:` line with the number of validation dialogues that also appear in train.

**(a) New behaviour behind a flag that defaults off:** yes for `--clean_valid` and `--class_weight`. `--seed` (2024) and `--select_on` (`test`) default to the base behaviour. `--out_json` defaults to nothing written.

**(b) Default flags reproduce the base:** tested, on CPU only. Synthetic IEMOCAP-format pickle (40 dialogues), 3 epochs, tiny model. `16f12b6` and `d12f026` were run with no new flags. The 7 lines of epoch, test and final `Acc` output are identical (`diff` empty). Not tested on GPU or on the real IEMOCAP pickle.

**(c) Table III hyperparameters unchanged in `scripts/run_all.sh`:** yes, the file is identical to `16f12b6` (`git diff --quiet`). The IEMOCAP-6 values in the GPU commands are the paper's Table III values: LR 1e-4, BS 16, DR 0.2, L 7, P 17, B 19, λs 1.0, λo 0.7. The GPU commands also use `--l2 1e-3`, the paper-text value. The repo default is 1e-4.

## 3. Results (IEMOCAP-6, emotion, % , sample std ddof=1)
Paper-protocol WF1 is the `F-Score` value on the final `Acc:` line of a run's log, which is the best-test-F1 epoch. It is **not measured** for any 5-seed row below, because those runs were clean or valid-selected.

| Setting | Seeds | Test WF1 (selection as stated) | Paper-protocol WF1 | Happy F1 | Excited F1 | Files | Chosen on |
|---|---|---|---|---|---|---|---|
| Original code, default split (`IEMOCAP_full`) | 1 (2024) | 71.17 | 71.17 | 54.87 | 78.04 | `results/logs/IEMOCAP_full.log` | test |
| p0: default split, valid-selected (valid overlaps train) | 5 | 70.82 ± 0.47 | not measured | 58.69 ± 3.18 | 77.57 ± 0.92 | `results/experiments/p0_seed*.json`, `results/logs/p0_IEMOCAP_seed*.log` | valid (contaminated) |
| **p0c baseline:** clean split, valid-selected | 5 | **68.26 ± 0.94** | not measured | 45.62 ± 4.17 | 76.10 ± 1.24 | `results/experiments/p0c_seed*.json`, `results/logs/p0c_IEMOCAP_seed*.log` | valid (clean) |
| **p1c:** clean split + `--class_weight`, valid-selected | 5 | **68.38 ± 0.62** | not measured | 44.80 ± 4.34 | 76.36 ± 1.11 | `results/experiments/p1c_seed*.json`, `results/logs/p1c_IEMOCAP_seed*.log` | valid (clean) |

Seeds for p0c and p1c are the same five: 2024, 1, 2, 3, 4. Diagnostic only, not paper protocol and not used to choose anything: best-test-epoch WF1 computed from the clean-split logs is 71.22 for p0c and 71.01 for p1c. Those logs are on the clean split, so they are not the paper's protocol.

Per-seed clean WF1: p0c 67.96, 68.19, 66.98, 68.64, 69.54; p1c 68.53, 68.71, 67.89, 67.61, 69.14. Selected epochs for p0c were 17, 16, 89, 59, 71; for p1c, 37, 36, 16, 31, 87.

## 4. Decision against the rule
Rule (as stated for this pass): keep only if the clean 5-seed mean beats the clean baseline by more than the baseline std, AND the paper-protocol number is not worse.

- **Condition A:** clean baseline mean 68.26, std 0.94. Threshold = 68.26 + 0.94 = **69.20**. p1c mean = **68.38**. Difference of means = 68.38 − 68.26 = **+0.12**. Paired per-seed mean difference = **+0.11**. 68.38 < 69.20, so **A is not met**.
- **Condition B:** p1c paper-protocol WF1 is **not measured**, so "not worse" cannot be shown. B is not established.
- **Verdict for `--class_weight`: NOT KEPT.** It fails A. B is unmeasured.

Not a model change, so no keep decision: the clean split (`--clean_valid`). It removes the validation overlap, and all clean numbers in this pass use it. Its effect on the reported WF1, relative to the overlapping-valid run (p0), is 70.82 → 68.26.

## 5. Deviations from paper settings
- **`--clean_valid` split (108 train / 12 valid dialogues, 0 overlap): tuned variant, not paper setting.** It matches the paper's stated 10% of train+val. The paper does not say whether its training excluded its validation dialogues.
- **`--class_weight`: tuned variant, not paper setting.** Verified only that DialogueRNN's IEMOCAP script uses inverse-frequency weights. It is not kept, so it is not part of any reproduction default.
- Kept changes: **none**. So no kept change departs from Table III.

## 6. Unverified or weak points
- **Paper's epoch-selection protocol: not stated** in the text read (arXiv 2407.21536v2, sections I to V-A verbatim; V-D onward only summarised). This is why condition B cannot be checked against a known protocol.
- **GraphSmile using class weights: not checked.**
- **DialogueRNN class weights:** verified in the repo script (`train_IEMOCAP.py`). Not checked: which index is which emotion, and how `MaskedNLLLoss` applies the weights.
- **EACL (arXiv 2403.20289):** Table 3(a) per-class numbers were read verbatim. The anchor-similarity claim (0.77 to 0.08) came from a model summary and is not verified. Not used.
- **SPCL 69.74 WF1:** from an aggregator site only. Not used.
- **Focal loss:** only its definition was checked. Not used.
- **Weight averaging (SWA):** verified only as an image-classification result. No evidence for emotion recognition in conversation. Not implemented.
- **Single-seed rows:** `IEMOCAP_full`, and all full-model runs on IEMOCAP-4, MELD and CMU-MOSEI (not rerun under the clean protocol).
- **Valid and test disagree:** clean validation WF1 is about 73.6 and test is 68.3. Validation is flat across seeds (std 0.45) on 12 dialogues (664 utterances, of which only 45 are Happy). Best-test-epoch (71.22) against valid-selected (68.26) differs by about 3 points, which is selection, not the model.
- **Failed or incomplete runs:** none in p0, p0c or p1c. All five seeds of each finished, and their JSON files exist.
- **Data:** server pickles are byte-identical to the README release (md5 checked). The released CMU-MOSEI file disagrees with paper Table II: 2,595 / 693 dialogues against 2,549 / 646, and 19,050 / 4,184 utterances against 18,198 / 4,662. MELD has one train+val dialogue fewer than Table II (1,152 against 1,153). So the CMU-MOSEI numbers are on that file, not the paper's exact test set.
- **Early-epoch selection:** p0c seeds 2024 and 1 picked epochs 17 and 16. Selection is unstable across seeds.
- **`--out_json` gap:** under `--select_on test`, `--out_json` writes nothing. Paper-protocol numbers can only come from logs. This is why no paper-protocol JSON exists.

## 7. Merge recommendation
**Repository state that needs your decision first.** Local `main` and `origin/main` are both at `ab136c2`. It contains two commits I did not make or push in this tab:
- `ad738a6` (author "Claude"): merges protocol experiments into `main`, with the message "keep default-protocol results as main reproduction".
- `ab136c2` (author AdityaKuranjekar): merges `experiments/p1b-provenance` into `main`, including p2 results. The message says "shift_5 and shift_10 are not kept".

Your rule says no push to `main` without your confirmation. I did not verify whether you approved these. I have not changed `main`.

**Safe to merge, once you confirm:**
- `run.py` and `trainer.py` flag changes. The default-equivalence check was done on CPU only.
- `docs/*.pdf` and the two build scripts.
- `results/experiments/*.md`, the JSON files and the logs (each log is about 800 lines and under 1 MB).

**Keep on the branch:**
- `--clean_valid` and `--class_weight`: experimental variants, not paper settings. `--class_weight` is not kept.
- `p1-plan.md` and `p2-plan.md`: plans.
- Untracked helper files on the laptop: `inspect_pkl.py` and the `.patch` files. Not for merge.

**GPU runs needed:**
- **For a keep decision on `--class_weight` as it stands: none.** Condition A already fails on existing data. More runs of the same setting cannot change that.
- **For condition B on a future candidate that passes A:** 10 runs, 5 seeds baseline plus 5 seeds with class weight, on the default split with test selection. Read the paper-protocol WF1 from the `F-Score` value on the final `Acc:` line of each log. No JSON is written under test selection.

```bash
for s in 2024 1 2 3 4; do python -u run.py --gpu 1 --port 1530 --classify emotion --dataset IEMOCAP --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7 --l2 1e-3 --seed $s > results/logs/p1paper_base_IEMOCAP_seed$s.log 2>&1; done
```

```bash
for s in 2024 1 2 3 4; do python -u run.py --gpu 1 --port 1530 --classify emotion --dataset IEMOCAP --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7 --l2 1e-3 --seed $s --class_weight > results/logs/p1paper_cw_IEMOCAP_seed$s.log 2>&1; done
```

Check the GPU with `nvidia-smi` first. Port and GPU index are examples, as before.
