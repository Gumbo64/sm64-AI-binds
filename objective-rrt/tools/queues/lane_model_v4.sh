#!/bin/bash
# The v4 backbone (direct frames-to-star goals) in objective-rrt, start100,
# seeds 4-6, against lane_model.sh's off / v3 arms. Needs the servers started
# by backbone/night2.sh (--star-goals, checkpoints/v4/last.pt).
cd "$(dirname "$0")/../.."
export THREADS=16
P='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10'
BB=/workspaces/sm64-combined/sm64-AI-binds/backbone
M="OBRRT_MODEL=$BB/run/model4.sock OBRRT_MODEL_SERVERS=4 OBRRT_MODEL_ZONES=$BB/data/tree14:$BB/data/tas120"
tools/arm_eval.sh builds/model1/obrrt 500000000 mdl4 start100="$P $M OBRRT_MODEL_START_T=100" -- 4 5 6 >> logs/mdl4_eval1.txt 2>&1
