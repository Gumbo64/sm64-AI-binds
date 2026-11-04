#!/bin/bash
# Queue 10 (2026-11-02): PosTime near b (v2h default) on P, 500 M x seeds 1-3,
# 3 in parallel at 5 threads. Compare with q6-p (P, PosTime self: 5, 5, 4). Waits for q9.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q9.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2h/obrrt2 500000000 q10 \
  ptnear="$P" \
  -- 1 2 3
echo "q10 done"
