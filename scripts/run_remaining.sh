#!/usr/bin/env bash
# Remaining Table VI ablations, which were not in run_all.sh:
#   No Res, No FC-Res, No Seg          (structure ablations, MERC and MSAC)
#   No L_e, No L_e & L_o               (MSAC block of Table VI)
# 5 variants x 4 datasets = 20 runs, sequential.
# Run from the repository root:  GPU=4 bash scripts/run_remaining.sh
set -uo pipefail

GPU="${GPU:-0}"
L2="${L2:-0.001}"   # paper value (Section IV-C)
LOG_DIR=results/logs
mkdir -p "$LOG_DIR" results

# Same dataset settings as run_all.sh (README / Table III).
DATASETS=(
  "IEMOCAP|120|1e-04|16|512|17 17|7 7 7|0.2|19|1.0 1.0 0.7"
  "IEMOCAP4|120|3e-04|16|256|5 5|4 4 4|0.2|10|1.0 0.6 0.6"
  "MELD|50|7e-05|16|384|3 3|5 5 5|0.2|3|1.0 0.5 0.2"
  "CMUMOSEI7|60|8e-05|32|256|5 5|2 2 2|0.4|2|1.0 0.8 1.0"
)

# tag | extra flags
VARIANTS=(
  "no_Res|--no_res --loss_type emo_sen_sft"
  "no_FCRes|--no_fcres --loss_type emo_sen_sft"
  "no_Seg|--no_seg --loss_type emo_sen_sft"
  "no_Le|--modals tva --loss_type sen_sft"
  "no_Le_Lo|--modals tva --loss_type sen"
)

FAILED=()
PORT=15500

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
      --shift_win "$SHIFT" --lambd $LAMBD --l2 "$L2" $FLAGS > "$LOG" 2>&1 \
      || { echo "  FAILED: see $LOG"; FAILED+=("$DS/$TAG"); }
  done
done

echo "Done. Failed runs: ${#FAILED[@]}"
for f in "${FAILED[@]:-}"; do [ -n "$f" ] && echo "  - $f"; done
python scripts/compare_to_paper.py results/comparison.md
