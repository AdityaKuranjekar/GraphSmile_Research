#!/usr/bin/env bash
# Ensemble members: baseline MOSEI setting, one seed each, saving selected-epoch probabilities and keys.
# usage: run_ensemble.sh <gpu> <seed> [<seed> ...]
GPU=$1; shift
cd "$(dirname "$0")/.."
MOSEI="--dataset CMUMOSEI7 --epochs 60 --lr 8e-05 --batch_size 32 --hidden_dim 256 --win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0 --classify emotion --textf_mode textf0 --loss_type emo_sen_sft --l2 1e-3 --select_on valid --clean_valid"
mkdir -p results/probs results/experiments results/logs
for s in "$@"; do
  python -u run.py --gpu "$GPU" --port $((18000 + RANDOM % 1000)) --seed "$s" $MOSEI \
    --out_json "results/experiments/ens_seed${s}.json" --save_probs "results/probs/ens_seed${s}.npz" \
    > "results/logs/ens_seed${s}.log" 2>&1
  echo "[$(date +%T)] ens seed=$s done"
done
