#!/usr/bin/env bash
# Re-run only the two settings the supervisor asked for, on all 4 datasets:
#   full (text+visual+acoustic) and no_text (visual+acoustic).
# Same hyperparameters, seed (2024) and protocol as scripts/run_all.sh.
# Logs go to results/rerun/ so the earlier results/logs/ are not overwritten.
# Usage (repo root, conda env graphsmile):  GPU=0 bash scripts/run_rerun_full_notext.sh
set -uo pipefail
GPU="${GPU:-0}"
LOG_DIR=results/rerun
mkdir -p "$LOG_DIR"

DATASETS=(
  "IEMOCAP|120|1e-04|16|512|17 17|7 7 7|0.2|19|1.0 1.0 0.7"
  "IEMOCAP4|120|3e-04|16|256|5 5|4 4 4|0.2|10|1.0 0.6 0.6"
  "MELD|50|7e-05|16|384|3 3|5 5 5|0.2|3|1.0 0.5 0.2"
  "CMUMOSEI7|60|8e-05|32|256|5 5|2 2 2|0.4|2|1.0 0.8 1.0"
)
VARIANTS=("full|--modals tva" "no_text|--modals va")

PORT=16300
for ds_spec in "${DATASETS[@]}"; do
  IFS='|' read -r DS EPOCHS LR BS HID WIN HETER DROP SHIFT LAMBD <<< "$ds_spec"
  for var_spec in "${VARIANTS[@]}"; do
    IFS='|' read -r TAG FLAGS <<< "$var_spec"
    PORT=$((PORT + 1))
    echo "[$(date +%T)] $DS / $TAG"
    # shellcheck disable=SC2086
    python -u run.py --gpu "$GPU" --port "$PORT" --classify emotion --dataset "$DS" \
      --epochs "$EPOCHS" --textf_mode textf0 --lr "$LR" --batch_size "$BS" \
      --hidden_dim "$HID" --win $WIN --heter_n_layers $HETER --drop "$DROP" \
      --shift_win "$SHIFT" --lambd $LAMBD --l2 0.001 --loss_type emo_sen_sft $FLAGS \
      > "$LOG_DIR/${DS}_${TAG}.log" 2>&1 || echo "  FAILED: $DS/$TAG"
  done
done
echo "Done. Logs in $LOG_DIR"
