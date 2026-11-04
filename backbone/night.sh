#!/bin/bash
# Overnight queue (2026-10-31, third try): one GPU job at a time.
cd "$(dirname "$0")"
PY=".venv/bin/python"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
wait_pid() { while [ -e /proc/$1 ]; do sleep 30; done; }
wait_pid $(cat run/train_v4.pid)
cp checkpoints/last.pt checkpoints/v4/last.pt
$PY eval_star.py checkpoints/v4/last.pt > logs/eval_star_v4.txt 2>&1
$PY eval_star.py checkpoints/v4/step18k.pt > logs/eval_star_v4_18k.txt 2>&1
$PY astar.py --ckpt checkpoints/v4/last.pt --moves ga --guide star --stars 5 --parents 8 --attempts 4 --sub-secs 600 --name astar-star-v4 > logs/astar-star-v4.log 2>&1
$PY rl_objective.py --steps 20000000 --envs 128 --per-kind 150 --run_name rl-objectives-v1 > logs/rl_v1.log 2>&1
echo "night queue done $(date)" >> logs/night.log
