#!/bin/bash
# rlv2/launch.sh <name> <log> <command...>: detached, nice 5; run/rlv2/<name>.pid holds the real PID.
cd /workspaces/sm64-combined/sm64-AI-binds/backbone
name=$1; log=$2; shift 2
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup bash -c 'echo $$ > run/rlv2/'"$name"'.pid; exec nice -n 5 "$@"' _ "$@" > "$log" 2>&1 < /dev/null &
