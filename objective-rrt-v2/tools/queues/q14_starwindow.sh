#!/bin/bash
# Queue 14 (2026-11-02): sliding star window on P, v2j, 500 M x seeds 1-3 (3 in
# parallel, 5 threads). Compare with q10 (v2h P, PosTime near: 5, 5, 5) and v1 (6, 7, 5).
cd "$(dirname "$0")/../.."
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2j/obrrt2 500000000 q14 \
  sw2="$P OBRRT_STAR_WINDOW=2" \
  sw3="$P OBRRT_STAR_WINDOW=3" \
  -- 1 2 3
echo "q14 done"
