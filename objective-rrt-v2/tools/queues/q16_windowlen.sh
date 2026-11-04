#!/bin/bash
# Queue 16 (2026-11-03): how long the parent window's copy should be, side by side on one build (v2l),
# 500 M x seeds 1-3. off = plain P; pw1 = one parent (q15's best arm); h1, h2 = at least 1 or 2 x the
# objective's GA horizon, scaled down below population 48; f600 = at least 600 frames (20 s) for every objective,
# any population. The new node is a child of the selected node when the copy survives exactly, else a sibling.
cd "$(dirname "$0")/../.."
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
M="OBRRT_PARENT_WINDOW=1 OBRRT_PARENT_WINDOW_MAX=1500"
PAR=3 THREADS=5 tools/screen.sh builds/v2l/obrrt2 500000000 q16 \
  off="$P" \
  pw1="$P OBRRT_PARENT_WINDOW=1" \
  h1="$P $M OBRRT_PARENT_WINDOW_HORIZONS=1" \
  h2="$P $M OBRRT_PARENT_WINDOW_HORIZONS=2" \
  f600="$P $M OBRRT_PARENT_WINDOW_FRAMES=600 OBRRT_PARENT_WINDOW_POP=1" \
  -- 1 2 3
echo "q16 done"
