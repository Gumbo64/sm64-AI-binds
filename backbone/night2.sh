#!/bin/bash
# After night.sh: objective-rrt with the v4 backbone (start100, seeds 4-6).
cd "$(dirname "$0")"
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
wait_pid() { while [ -e /proc/$1 ]; do sleep 30; done; }
wait_pid $(cat run/night.pid)
pids=()
for i in 0 1 2 3; do
  .venv/bin/python serve_model.py --ckpt checkpoints/v4/last.pt --star-goals --socket run/model4.sock.$i > logs/serve_model4.$i.log 2>&1 &
  pids+=($!)
done
for i in $(seq 1 120); do [ $(grep -l serving logs/serve_model4.*.log 2>/dev/null | wc -l) -eq 4 ] && break; sleep 5; done
bash ../objective-rrt/tools/queues/lane_model_v4.sh
kill "${pids[@]}"
echo "night2 done $(date)" >> logs/night.log
