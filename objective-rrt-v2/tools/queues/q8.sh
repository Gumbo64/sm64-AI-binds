#!/bin/bash
# Queue 8 (2026-11-02): both directions on v2f, P base, 500 M x seeds 1-3,
# 3 seeds in parallel at 5 threads each. Exploration: gamma 0.995, death 200.
# Speed: gamma 0.98, POLISH + FASTER_ONLY without rr. Waits for q7.
# Compare with q6-p (P alone: 5, 5, 4) and v1 pf-mgel-lg-pf (6, 7, 5).
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q7.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2f/obrrt2 500000000 q8 \
  g995="$P OBRRT_GA_GAMMA=0.995" \
  d200="$P OBRRT_RH_DEATH=200" \
  g98="$P OBRRT_GA_GAMMA=0.98" \
  pfo="$P OBRRT_POLISH=0.1 OBRRT_FASTER_ONLY=1" \
  -- 1 2 3
echo "q8 done"
