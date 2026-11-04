#!/bin/bash
# After night2: star-only A* where the model also picks the next star.
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
wait_pid() { while [ -e /proc/$1 ]; do sleep 30; done; }
wait_pid $(cat run/night2.pid)
.venv/bin/python astar.py --ckpt checkpoints/v4/last.pt --moves ga --guide star --targets auto --stars 5 --parents 8 --attempts 4 --sub-secs 600 --name astar-auto-v4 > logs/astar-auto-v4.log 2>&1
echo "night3 done $(date)" >> logs/night.log
