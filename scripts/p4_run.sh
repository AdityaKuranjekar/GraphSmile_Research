#!/usr/bin/env bash
# p4 single clean-protocol run on the fixed code.
# Usage: bash scripts/p4_run.sh <gpu> <port> <tag> <iemocap6|iemocap4|meld|mosei> <seed> [extra run.py flags...]
set -o pipefail
GPU=$1; PORT=$2; TAG=$3; DS=$4; SEED=$5; shift 5
cd "$(dirname "$0")/.."; mkdir -p results/logs results/experiments
case "$DS" in
  iemocap6) ARGS="--dataset IEMOCAP --epochs 120 --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7";;
  iemocap4) ARGS="--dataset IEMOCAP4 --epochs 120 --lr 3e-04 --batch_size 16 --hidden_dim 256 --win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6";;
  meld)     ARGS="--dataset MELD --epochs 50 --lr 7e-05 --batch_size 16 --hidden_dim 384 --win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2";;
  mosei)    ARGS="--dataset CMUMOSEI7 --epochs 60 --lr 8e-05 --batch_size 32 --hidden_dim 256 --win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0";;
esac
echo "[$(date +%T)] START $TAG gpu=$GPU"
"${PY:-/DATA/rashmi_2521cs34/miniconda3/envs/graphsmile/bin/python}" -u run.py --gpu "$GPU" --port "$PORT" --classify emotion --textf_mode textf0 --loss_type emo_sen_sft --l2 1e-3 \
  $ARGS --seed "$SEED" --select_on valid --clean_valid --out_json "results/experiments/${TAG}.json" "$@" \
  > "results/logs/${TAG}.log" 2>&1
echo "[$(date +%T)] END $TAG exit=$?"
