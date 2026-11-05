#!/usr/bin/env bash
# MOSEI architecture sweep (gap plan, step 2). One seed, clean protocol, selection by validation F1.
# Saves per-utterance emotion probabilities of the selected epoch for the ensemble step.
#
# usage: run_arch_sweep.sh <gpu> <hidden:layers:win> [<hidden:layers:win> ...]
#   e.g. run_arch_sweep.sh 0 384:2:5 256:3:5 256:2:3
# Baseline is 256:2:5 with lr 8e-5 and dropout 0.4 (already run on the fixed code).
GPU=$1; shift
cd "$(dirname "$0")/.."
mkdir -p results/logs results/experiments results/probs
for cfg in "$@"; do
  IFS=: read -r H L W <<< "$cfg"
  tag="arch_h${H}_l${L}_w${W}"
  python -u run.py --gpu "$GPU" --port $((18000 + RANDOM % 1000)) --seed 2024 \
    --dataset CMUMOSEI7 --epochs 60 --lr 8e-05 --batch_size 32 \
    --hidden_dim "$H" --win "$W" "$W" --heter_n_layers "$L" "$L" "$L" \
    --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0 \
    --classify emotion --textf_mode textf0 --loss_type emo_sen_sft --l2 1e-3 \
    --select_on valid --clean_valid \
    --out_json "results/experiments/${tag}.json" --save_probs "results/probs/${tag}_seed2024.npz" \
    > "results/logs/${tag}.log" 2>&1
  echo "[$(date +%T)] $tag done"
done
