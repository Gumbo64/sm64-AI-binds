#!/bin/bash
# After v5 (no subgoal head): held-out time-to-star eval and the
# model-picks-the-star A* with v5.
cd "$(dirname "$0")"
PY=".venv/bin/python"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
wait_pid() { while [ -e /proc/$1 ]; do sleep 30; done; }
wait_pid $(cat run/train_v5.pid)
mkdir -p checkpoints/v5 && cp checkpoints/last.pt checkpoints/v5/last.pt
$PY eval_star.py checkpoints/v5/last.pt > logs/eval_star_v5.txt 2>&1
$PY astar.py --ckpt checkpoints/v5/last.pt --moves ga --guide star --targets auto --stars 5 --parents 8 --attempts 4 --sub-secs 600 --name astar-auto-v5 > logs/astar-auto-v5.log 2>&1
echo "night5 done $(date)" >> logs/night.log
