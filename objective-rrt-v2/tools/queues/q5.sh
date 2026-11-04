#!/bin/bash
# Queue 5 (2026-11-01): rr + q3's speed winners on v2e (context-keyed bests, BNB_SLACK). 100 M, seeds 1-2. Waits for q4.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q4.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
RR="OBRRT_REROUTE=1 OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1"
tools/screen.sh builds/v2e/obrrt2 100000000 q5 \
  rrfo="$P $RR OBRRT_POLISH=0.1 OBRRT_FASTER_ONLY=1" \
  rrbnb="$P $RR OBRRT_BNB=1" \
  rrfobnb="$P $RR OBRRT_POLISH=0.1 OBRRT_FASTER_ONLY=1 OBRRT_BNB=1" \
  -- 1 2
echo "q5 done"
