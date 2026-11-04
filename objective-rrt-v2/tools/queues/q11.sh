#!/bin/bash
# Queue 11 (2026-11-02): BljAny's existing riser + wall modes (radius 1500
# around b, mover geometry) on P, v2h, 500 M x seeds 1-3, 3 in parallel at 5
# threads. Compare with q10 (same binary and P). Waits for q10.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q10.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2h/obrrt2 500000000 q11 \
  rw="$P OBRRT_BLJ_RISERS=1 OBRRT_BLJ_WALLS=1" \
  -- 1 2 3
echo "q11 done"
