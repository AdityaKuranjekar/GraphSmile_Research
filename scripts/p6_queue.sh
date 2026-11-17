#!/bin/bash
# p6/p7 queue: IEMOCAP-6 full model (tva), paper config (L=7, shift_win 19, lambd 1 1 0.7, win 17 17).
# Usage: p6_queue.sh JOBFILE NWORKERS. Lines: name|flags. A job naming --teacher_path waits for that file.
source /DATA/rashmi_2521cs34/miniconda3/etc/profile.d/conda.sh; conda activate graphsmile
cd /DATA/rashmi_2521cs34/GS_p6
JOBS=$1; NW=${2:-3}; LOCK=$JOBS.lock
BASE="--classify emotion --dataset IEMOCAP --modals tva --epochs 120 --textf_mode textf0 --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 --drop 0.2 --l2 1e-3 --heter_n_layers 7 7 7 --shift_win 19 --lambd 1.0 1.0 0.7 --win 17 17"
next() { ( flock 9; l=$(head -1 "$JOBS"); sed -i 1d "$JOBS"; echo "$l" ) 9>"$LOCK"; }
worker() { local w=$1
 while :; do j=$(next); [ -z "$j" ] && break; n=${j%%|*}; fl=${j#*|}
  t=$(sed -n 's/.*--teacher_path \([^ ]*\).*/\1/p' <<< "$fl")
  while [ -n "$t" ] && [ ! -f "$t" ]; do sleep 60; done
  for try in 1 2 3 4 5 6; do
   g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
   python -u run.py --gpu $g --port $((18000+w*100+try*7+RANDOM%5)) $BASE $fl --out_json results/experiments/$n.json > results/logs/$n.log 2>&1
   [ -f results/experiments/$n.json ] && break; echo "$(date +%T) retry $n ($try) gpu$g"; sleep 120
  done; done; }
for w in $(seq $NW); do worker $w & sleep 60; done; wait; echo ALLDONE > "$JOBS.done"
