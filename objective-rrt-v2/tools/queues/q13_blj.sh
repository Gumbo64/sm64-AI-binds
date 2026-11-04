#!/bin/bash
# Queue 13 (2026-11-02): BLJ scan + fast setups + motion cells on P (v2i),
# 500 M x seeds 1-3 (3 in parallel, 5 threads). Compare with q10 (v2h P) and
# q11 (v2h P + risers/walls). Waits for q12.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q12.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2i/obrrt2 500000000 q13 \
  bljsfm="$P OBRRT_BLJ_SCAN=1 OBRRT_BLJ_FAST=1 OBRRT_CELL_MOTION=1" \
  -- 1 2 3
echo "q13 done"
