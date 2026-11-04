#!/bin/bash
# v6: credited objective-rrt edges (learned objectives: end position, zone,
# budget, kind) on top of v5; then held-out eval and A* aimed at the model's
# own subgoals with only the star chosen (by the model, among open courses).
cd "$(dirname "$0")"
PY=".venv/bin/python"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
wait_pid() { while [ -e /proc/$1 ]; do sleep 30; done; }
wait_pid $(cat run/night5.pid)
$PY train.py --v4 --class-min --edges --init checkpoints/v5/last.pt --steps 20000 --phase1 0 --warmup 500 --lr 1.5e-4 --workers 6 --run_name v6-edges > logs/train_v6.log 2>&1
mkdir -p checkpoints/v6 && cp checkpoints/last.pt checkpoints/v6/last.pt
$PY eval_star.py checkpoints/v6/last.pt > logs/eval_star_v6.txt 2>&1
$PY astar.py --ckpt checkpoints/v6/last.pt --moves ga --guide learned --targets auto --stars 5 --parents 8 --attempts 4 --sub-secs 600 --name astar-learned-v6 > logs/astar-learned-v6.log 2>&1
echo "night6 done $(date)" >> logs/night.log
