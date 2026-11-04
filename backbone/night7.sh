#!/bin/bash
# v6 with learned objectives (style, events, speed, verb + object pointer) on
# credited objective-rrt edges, then objective-rrt 500M seeds 4-6: start100
# alone vs start100 + 4 learned objectives per scored state.
cd "$(dirname "$0")"
PY=".venv/bin/python"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
wait_pid() { while [ -e /proc/$1 ]; do sleep 30; done; }
wait_pid $(cat run/night5.pid)
$PY train.py --v4 --class-min --edges --init checkpoints/v5/last.pt --steps 20000 --phase1 0 --warmup 500 --lr 1.5e-4 --workers 6 --run_name v6-learned-obj > logs/train_v6.log 2>&1
mkdir -p checkpoints/v6 && cp checkpoints/last.pt checkpoints/v6/last.pt
$PY eval_star.py checkpoints/v6/last.pt > logs/eval_star_v6.txt 2>&1
for i in 0 1 2 3; do
  setsid bash -c "echo \$\$ > run/serve6.$i.pid; exec $PY serve_model.py --ckpt checkpoints/v6/last.pt --star-goals --socket run/model6.sock.$i" > logs/serve_model6.$i.log 2>&1 &
done
for i in $(seq 1 120); do [ $(grep -l serving logs/serve_model6.*.log 2>/dev/null | wc -l) -eq 4 ] && break; sleep 5; done
cd ../objective-rrt
export THREADS=16
P='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10'
BB=/workspaces/sm64-combined/sm64-AI-binds/backbone
M="OBRRT_MODEL=$BB/run/model6.sock OBRRT_MODEL_SERVERS=4 OBRRT_MODEL_ZONES=$BB/data/tree14:$BB/data/tas120 OBRRT_MODEL_START_T=100"
tools/arm_eval.sh builds/model2/obrrt 500000000 mdl6 learned4="$P $M OBRRT_MODEL_LEARNED=4" start100="$P $M" -- 4 5 6 >> logs/mdl6_eval1.txt 2>&1
cd ../backbone
for i in 0 1 2 3; do kill $(cat run/serve6.$i.pid); done
echo "night7 done $(date)" >> logs/night.log
