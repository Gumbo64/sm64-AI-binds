#!/bin/bash
# Queue 3 (2026-11-01): v2d = v2c (v1 start-ranking/fight parity) + speed
# switches (off = v2c's tree). 100 M, seeds 1-2. Waits for q2.
# Parity first (priority: v1 features), then the speed switches.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q2.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
tools/screen.sh builds/v2d/obrrt2 100000000 q3 \
  par="$P" \
  fo="$P OBRRT_POLISH=0.1 OBRRT_FASTER_ONLY=1" \
  beat="$P OBRRT_BEAT=1" \
  bnb="$P OBRRT_BNB=1" \
  speed="$P OBRRT_POLISH=0.1 OBRRT_FASTER_ONLY=1 OBRRT_BEAT=1 OBRRT_BNB=1" \
  -- 1 2
echo "q3 done"
