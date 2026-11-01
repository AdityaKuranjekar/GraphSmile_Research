#!/usr/bin/env bash
# Clean-protocol baseline on the FIXED code (per-epoch shuffle + edge_weight in optimizer).
# Five seeds for one dataset, run one after another on one GPU.
#
# Usage (from the repo root, with dataset paths already set in run.py):
#   bash scripts/run_phase0_fixed.sh <gpu> <dataset>
#     dataset: iemocap6 | iemocap4 | meld | mosei
#
# Output: results/logs/fix-p0_<dataset>_seed<seed>.log
#         results/experiments/fix-p0_<dataset>_seed<seed>.json
# Failures are printed and do not stop the remaining seeds.
set -o pipefail

GPU=${1:?gpu id}
DS=${2:?dataset: iemocap6|iemocap4|meld|mosei}
cd "$(dirname "$0")/.."
mkdir -p results/logs results/experiments

case "$DS" in
  iemocap6)
    ARGS="--dataset IEMOCAP --epochs 120 --lr 1e-04 --batch_size 16 --hidden_dim 512 \
          --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7"
    PORT=1640 ;;
  iemocap4)
    ARGS="--dataset IEMOCAP4 --epochs 120 --lr 3e-04 --batch_size 16 --hidden_dim 256 \
          --win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6"
    PORT=1630 ;;
  meld)
    ARGS="--dataset MELD --epochs 50 --lr 7e-05 --batch_size 16 --hidden_dim 384 \
          --win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2"
    PORT=1620 ;;
  mosei)
    ARGS="--dataset CMUMOSEI7 --epochs 60 --lr 8e-05 --batch_size 32 --hidden_dim 256 \
          --win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0"
    PORT=1610 ;;
  *) echo "unknown dataset: $DS"; exit 2 ;;
esac

i=0
for SEED in 2024 1 2 3 4; do
  i=$((i + 1))
  TAG="fix-p0_${DS}_seed${SEED}"
  echo "[$(date +%T)] START $TAG gpu=$GPU"
  python -u run.py --gpu "$GPU" --port $((PORT + i)) --classify emotion \
    --textf_mode textf0 --loss_type emo_sen_sft --l2 1e-3 \
    $ARGS --seed "$SEED" --select_on valid --clean_valid \
    --out_json "results/experiments/${TAG}.json" \
    > "results/logs/${TAG}.log" 2>&1
  RC=$?
  echo "[$(date +%T)] END   $TAG exit=$RC"
done
echo "[$(date +%T)] DONE $DS"
