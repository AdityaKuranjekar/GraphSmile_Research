#!/usr/bin/env bash
# Collapse check (p3): rerun three Table VI ablation variants with the paper
# configuration and --l2 0.001, over seeds 2024, 1 and 2.
#   IEMOCAP   no_Le      (--modals tva --loss_type sen_sft)
#   IEMOCAP4  no_Le      (--modals tva --loss_type sen_sft)
#   IEMOCAP4  no_Le_Lo   (--modals tva --loss_type sen)
# 3 variants x 3 seeds = 9 runs, sequential. Same flags as run_remaining.sh.
# Run from the repository root:  GPU=4 bash scripts/run_collapse_check.sh
set -uo pipefail

GPU="${GPU:-0}"
L2="${L2:-0.001}"   # paper value (Section IV-C)
SEEDS=(2024 1 2)
LOG_DIR=results/logs
mkdir -p "$LOG_DIR" results

# dataset | tag | extra flags
RUNS=(
  "IEMOCAP|no_Le|--modals tva --loss_type sen_sft"
  "IEMOCAP4|no_Le|--modals tva --loss_type sen_sft"
  "IEMOCAP4|no_Le_Lo|--modals tva --loss_type sen"
)

TOTAL=$(( ${#RUNS[@]} * ${#SEEDS[@]} ))
echo "Collapse check: $TOTAL runs (${#RUNS[@]} variants x ${#SEEDS[@]} seeds), sequential."

FAILED=()
PORT=17000

for run_spec in "${RUNS[@]}"; do
  IFS='|' read -r DS TAG FLAGS <<< "$run_spec"
  # Same dataset settings as run_all.sh and run_remaining.sh (README / Table III).
  case "$DS" in
    IEMOCAP)
      EPOCHS=120; LR=1e-04; BS=16; HID=512; WIN="17 17"
      HETER="7 7 7"; DROP=0.2; SHIFT=19; LAMBD="1.0 1.0 0.7" ;;
    IEMOCAP4)
      EPOCHS=120; LR=3e-04; BS=16; HID=256; WIN="5 5"
      HETER="4 4 4"; DROP=0.2; SHIFT=10; LAMBD="1.0 0.6 0.6" ;;
    *) echo "unknown dataset: $DS"; exit 1 ;;
  esac
  for SEED in "${SEEDS[@]}"; do
    LOG="$LOG_DIR/p3_${DS}_${TAG}_seed${SEED}.log"
    PORT=$((PORT + 1))
    echo "[$(date +%T)] $DS / $TAG / seed $SEED (port $PORT) -> $LOG"
    # shellcheck disable=SC2086
    python -u run.py --gpu "$GPU" --port "$PORT" --classify emotion --dataset "$DS" \
      --epochs "$EPOCHS" --textf_mode textf0 --lr "$LR" --batch_size "$BS" \
      --hidden_dim "$HID" --win $WIN --heter_n_layers $HETER --drop "$DROP" \
      --shift_win "$SHIFT" --lambd $LAMBD --l2 "$L2" --seed "$SEED" $FLAGS > "$LOG" 2>&1 \
      || { echo "  FAILED: see $LOG"; FAILED+=("$DS/$TAG/seed$SEED"); }
  done
done

echo "Done. Failed runs: ${#FAILED[@]} of $TOTAL"
for f in "${FAILED[@]:-}"; do [ -n "$f" ] && echo "  - $f"; done
