#!/usr/bin/env bash
# Pass 2: IEMOCAP-6 sensitivity to window size, shift window, depth and loss weights.
#
# Every setting is run under both protocols (rules in results/experiments/p0-protocol.md):
#   paper = default split, --select_on test         (paper-comparable number)
#   clean = --clean_valid --select_on valid         (honest estimate)
#
# Stage 1 (default): clean protocol only, seeds 2024 1 2, 8 settings = 24 runs.
#   The baseline clean runs are not repeated: results/experiments/p0c_seed<seed>.json
#   already holds them (same flags), so the baseline is skipped under the clean protocol.
# Stage 2, winners only, run separately:
#   SETTINGS="<winners>" PROTOCOLS="paper" SEEDS="2024 1 2" GPU=0 bash scripts/run_pass2.sh
#   SETTINGS="<winners>" PROTOCOLS="paper clean" SEEDS="3 4" GPU=0 bash scripts/run_pass2.sh
#   The keep rule also needs the baseline under the paper protocol for 5 seeds, so add
#   "base" to SETTINGS for the paper runs. Its clean runs are skipped via p0c.
#
# Run from the repository root:  GPU=0 bash scripts/run_pass2.sh
# Outputs: results/logs/p2_<setting>_<protocol>_seed<seed>.log and
#          results/experiments/p2_<setting>_<protocol>_seed<seed>.json
set -uo pipefail

GPU="${GPU:-0}"
PROTOCOLS="${PROTOCOLS:-clean}"
SEEDS="${SEEDS:-2024 1 2}"
SETTINGS="${SETTINGS:-win_5_5 win_10_10 win_10_2 shift_5 shift_10 heter_4 lambd_0.5 lambd_0.3}"
PORT=16400

# Flags that differ from the baseline. The baseline is the p0c configuration.
flags_for() {
  case "$1" in
    base)      echo "--win 17 17 --shift_win 19 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7" ;;
    win_5_5)   echo "--win 5 5 --shift_win 19 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7" ;;
    win_10_10) echo "--win 10 10 --shift_win 19 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7" ;;
    win_10_2)  echo "--win 10 2 --shift_win 19 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7" ;;
    shift_5)   echo "--win 17 17 --shift_win 5 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7" ;;
    shift_10)  echo "--win 17 17 --shift_win 10 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.7" ;;
    heter_4)   echo "--win 17 17 --shift_win 19 --heter_n_layers 4 4 4 --lambd 1.0 1.0 0.7" ;;
    lambd_0.5) echo "--win 17 17 --shift_win 19 --heter_n_layers 7 7 7 --lambd 1.0 0.5 0.7" ;;
    lambd_0.3) echo "--win 17 17 --shift_win 19 --heter_n_layers 7 7 7 --lambd 1.0 1.0 0.3" ;;
    *) return 1 ;;
  esac
}

run() {  # run SETTING PROTOCOL SEED FLAGS...
  local tag=$1 proto=$2 seed=$3; shift 3
  local name="p2_${tag}_${proto}_seed${seed}"
  local proto_flags
  if [ "$proto" = paper ]; then
    proto_flags=(--select_on test)
  else
    proto_flags=(--clean_valid --select_on valid)
  fi
  PORT=$((PORT + 1))
  echo "[$(date +%T)] $name"
  python -u run.py --gpu "$GPU" --port "$PORT" --classify emotion --dataset IEMOCAP \
    --modals tva --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft \
    --lr 1e-04 --batch_size 16 --hidden_dim 512 --drop 0.2 --l2 1e-3 \
    --seed "$seed" "${proto_flags[@]}" "$@" \
    --out_json "results/experiments/${name}.json" > "results/logs/${name}.log" 2>&1 \
    || echo "  FAILED: $name"
}

# A run is skipped when it duplicates a p0c result (baseline under the clean protocol).
skip() {  # skip SETTING PROTOCOL SEED
  [ "$1" = base ] && [ "$2" = clean ] && [ -f "results/experiments/p0c_seed$3.json" ]
}

mkdir -p results/logs results/experiments

# Count first, so the number is printed before anything starts.
total=0
for tag in $SETTINGS; do
  flags_for "$tag" > /dev/null || { echo "unknown setting: $tag" >&2; exit 1; }
  for proto in $PROTOCOLS; do
    for seed in $SEEDS; do
      skip "$tag" "$proto" "$seed" || total=$((total + 1))
    done
  done
done
echo "Planned runs: $total (settings: $SETTINGS | protocols: $PROTOCOLS | seeds: $SEEDS)"

for tag in $SETTINGS; do
  read -r -a FLAGS <<< "$(flags_for "$tag")"
  for proto in $PROTOCOLS; do
    for seed in $SEEDS; do
      skip "$tag" "$proto" "$seed" && { echo "skip $tag $proto seed$seed (p0c)"; continue; }
      run "$tag" "$proto" "$seed" "${FLAGS[@]}"
    done
  done
done

echo "Done. Logs: results/logs/p2_*.log"
