#!/usr/bin/env bash
# Gap-plan job queue. Jobs run one after another on one GPU.
#
# Usage (from the repo root):  bash scripts/gap_queue.sh <gpu> <job> [<job> ...]
#
# Jobs:
#   p0:<ds>                    Phase 0 (fixed code), 5 seeds. Calls run_phase0_fixed.sh.
#   nodtext:<ds>               Phase 0c, no-text baseline, clean protocol, 3 seeds (--modals va).
#   mod:<t|ta|tv>              Phase 1.3, MOSEI with only the listed modalities, 3 seeds.
#   win:<n>                    Phase 1.4, MOSEI with a symmetric window of n, 3 seeds.
#   sent:<ds>:<noLe|noLeLo>    Phase 4.1, sentiment ablation, 1 seed.
# ds: iemocap6 | iemocap4 | meld | mosei
#
# Outputs: results/logs/<tag>_seed<seed>.log and results/experiments/<tag>_seed<seed>.json
set -o pipefail

GPU=${1:?usage: gap_queue.sh <gpu> <job> [job ...]}
shift
cd "$(dirname "$0")/.."
mkdir -p results/logs results/experiments

# Fail fast if the wrong Python is active (this happened once: waiters started under 'base').
if ! python -c "import torch" 2>/dev/null; then
  echo "ERROR: 'python' cannot import torch. Run 'conda activate graphsmile' first." >&2
  exit 1
fi

COMMON="--classify emotion --textf_mode textf0 --loss_type emo_sen_sft --l2 1e-3"

ds_args() {  # same hyperparameters as run_phase0_fixed.sh and REPRODUCTION.md
  case "$1" in
    iemocap6) echo "--dataset IEMOCAP --epochs 120 --lr 1e-04 --batch_size 16 --hidden_dim 512 --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7" ;;
    iemocap4) echo "--dataset IEMOCAP4 --epochs 120 --lr 3e-04 --batch_size 16 --hidden_dim 256 --win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6" ;;
    meld)     echo "--dataset MELD --epochs 50 --lr 7e-05 --batch_size 16 --hidden_dim 384 --win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2" ;;
    mosei)    echo "--dataset CMUMOSEI7 --epochs 60 --lr 8e-05 --batch_size 32 --hidden_dim 256 --win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

run_one() {  # run_one <tag> <seed> <run.py args...>
  local tag=$1 seed=$2; shift 2
  if [ -s "results/experiments/${tag}_seed${seed}.json" ]; then
    echo "[$(date +%T)] SKIP  $tag seed=$seed (result already exists)"
    return 0
  fi
  local port=$(( 17000 + RANDOM % 2000 ))
  # --select_on valid always picks the epoch by emotion F1 (run.py), so the sentiment
  # ablations (gap4-1) use the repo default instead, which selects by sentiment F1.
  local sel="--select_on valid --clean_valid"
  case "$tag" in gap4-1_*) sel="" ;; esac
  echo "[$(date +%T)] START $tag seed=$seed gpu=$GPU"
  python -u run.py --gpu "$GPU" --port "$port" --seed "$seed" $sel "$@" \
    --out_json "results/experiments/${tag}_seed${seed}.json" \
    > "results/logs/${tag}_seed${seed}.log" 2>&1
  local rc=$?
  echo "[$(date +%T)] END   $tag seed=$seed exit=$rc"
}

for JOB in "$@"; do
  case "$JOB" in
    p0:*)
      bash scripts/run_phase0_fixed.sh "$GPU" "${JOB#p0:}" ;;
    nodtext:*)
      DS=${JOB#nodtext:}
      for s in 2024 1 2; do
        run_one "gap0c_nodtext_${DS}" $s $COMMON $(ds_args "$DS") --modals va
      done ;;
    mod:*)
      M=${JOB#mod:}
      for s in 2024 1 2; do
        run_one "gap1-3_mosei_mod_${M}" $s $COMMON $(ds_args mosei) --modals "$M"
      done ;;
    win:*)
      W=${JOB#win:}
      ARGS=$(ds_args mosei | sed "s/--win 5 5/--win $W $W/")
      for s in 2024 1 2; do
        run_one "gap1-4_mosei_win${W}" $s $COMMON $ARGS --modals tva
      done ;;
    sent:*)
      IFS=: read -r _ DS VAR <<< "$JOB"
      case "$VAR" in
        noLe)   LOSS=sen_sft ;;
        noLeLo) LOSS=sen ;;
        *) echo "unknown ablation: $VAR" >&2; continue ;;
      esac
      run_one "gap4-1_${DS}_${VAR}" 2024 --classify sentiment --textf_mode textf0 --l2 1e-3 \
        --loss_type "$LOSS" $(ds_args "$DS") --modals tva ;;
    *)
      echo "unknown job: $JOB" >&2 ;;
  esac
done
echo "[$(date +%T)] QUEUE DONE gpu=$GPU: $*"
