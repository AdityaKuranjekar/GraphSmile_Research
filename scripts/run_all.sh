#!/usr/bin/env bash
# Reproduce the main results (Table IV/V) and the runnable ablations (Table VI).
# Run from the repository root:  bash scripts/run_all.sh
# Set the GPU index with GPU=<id> (default 0). Fill in the dataset paths in run.py first.
set -uo pipefail

GPU="${GPU:-0}"
LOG_DIR=results/logs
mkdir -p "$LOG_DIR" results

# Dataset name, epochs, lr, batch, hidden, win_p win_f, heter layers, drop, shift_win, lambd
DATASETS=(
  "IEMOCAP|120|1e-04|16|512|17 17|7 7 7|0.2|19|1.0 1.0 0.7"
  "IEMOCAP4|120|3e-04|16|256|5 5|4 4 4|0.2|10|1.0 0.6 0.6"
  "MELD|50|7e-05|16|384|3 3|5 5 5|0.2|3|1.0 0.5 0.2"
  "CMUMOSEI7|60|8e-05|32|256|5 5|2 2 2|0.4|2|1.0 0.8 1.0"
)

# tag | extra flags. Modality ablations (Table VI "No Textual/Visual/Acoustic")
# and loss ablations ("No L_s" -> emo_sft, "No L_o" -> emo_sen, "No L_s&L_o" -> emo).
VARIANTS=(
  "full|--modals tva --loss_type emo_sen_sft"
  "no_text|--modals va --loss_type emo_sen_sft"
  "no_visual|--modals ta --loss_type emo_sen_sft"
  "no_acoustic|--modals tv --loss_type emo_sen_sft"
  "no_Ls|--modals tva --loss_type emo_sft"
  "no_Lo|--modals tva --loss_type emo_sen"
  "no_Ls_Lo|--modals tva --loss_type emo"
)

FAILED=()
PORT=15300

for ds_spec in "${DATASETS[@]}"; do
  IFS='|' read -r DS EPOCHS LR BS HID WIN HETER DROP SHIFT LAMBD <<< "$ds_spec"
  for var_spec in "${VARIANTS[@]}"; do
    IFS='|' read -r TAG FLAGS <<< "$var_spec"
    LOG="$LOG_DIR/${DS}_${TAG}.log"
    PORT=$((PORT + 1))
    echo "[$(date +%T)] $DS / $TAG -> $LOG"
    # shellcheck disable=SC2086
    python -u run.py --gpu "$GPU" --port "$PORT" --classify emotion --dataset "$DS" \
      --epochs "$EPOCHS" --textf_mode textf0 --lr "$LR" --batch_size "$BS" \
      --hidden_dim "$HID" --win $WIN --heter_n_layers $HETER --drop "$DROP" \
      --shift_win "$SHIFT" --lambd $LAMBD $FLAGS > "$LOG" 2>&1 \
      || { echo "  FAILED: see $LOG"; FAILED+=("$DS/$TAG"); }
  done
done

echo "Done. Failed runs: ${#FAILED[@]}"
for f in "${FAILED[@]:-}"; do [ -n "$f" ] && echo "  - $f"; done
python scripts/summarize.py
