#!/usr/bin/env bash
# Extra IEMOCAP-6 runs that are not in the paper's tables:
#   - L2 check at the paper's stated value (1e-3; the repo default is 1e-4)
#   - network depth sweep (paper Fig. 4, GraphSmile curve only)
#   - window size sweep (paper Fig. 5)
# Run from the repository root:  GPU=4 bash scripts/run_sweeps.sh
set -uo pipefail

GPU="${GPU:-0}"
PORT=15400
mkdir -p results/logs

run() {
  local tag=$1; shift
  PORT=$((PORT + 1))
  echo "[$(date +%T)] $tag"
  python -u run.py --gpu "$GPU" --port "$PORT" --classify emotion --dataset IEMOCAP \
    --modals tva --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft \
    --lr 1e-04 --batch_size 16 --hidden_dim 512 --drop 0.2 --shift_win 19 \
    --lambd 1.0 1.0 0.7 "$@" > "results/logs/IEMOCAP_${tag}.log" 2>&1 \
    || echo "  FAILED: $tag"
}

# Paper setting for L2 is 1e-3 (Section IV-C). Compare with IEMOCAP_full (1e-4).
run l2_1e-3 --l2 0.001 --win 17 17 --heter_n_layers 7 7 7

for L in 1 3 5 10 13 16 20; do
  run "depth_${L}" --win 17 17 --heter_n_layers "$L" "$L" "$L"
done

for W in 1 5 9 13 21 25 33 40; do
  run "win_${W}" --win "$W" "$W" --heter_n_layers 7 7 7
done

python scripts/compare_to_paper.py > results/comparison.md
echo "Done. Summary: results/comparison.md"
