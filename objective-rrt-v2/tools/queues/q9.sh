#!/bin/bash
# Queue 9 (2026-11-02): learned ETA calibration on v2g, P base, 500 M x seeds 1-3
# (3 in parallel, 5 threads each). Waits for q8.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q8.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2g/obrrt2 500000000 q9 \
  etal="$P OBRRT_ETA_LEARN=1" \
  etal5="$P OBRRT_ETA_LEARN=1 OBRRT_ETA_LEARN_N0=5" \
  -- 1 2 3
echo "q9 done"
