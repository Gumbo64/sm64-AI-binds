#!/bin/bash
# Lane B5 (8 threads), 2026-10-29: after lane_b4 (pid 39012, the ts eval).
# 1) Route-credit screen (OBRRT_RATE_ROUTE, builds/route1), 100 M × seeds 1-2 on mgel-pf.
# 2) The hierarchical rate sampler on mgel-pf at 500 M × 3 (hier s1 on cd: 6 stars vs 3).
cd "$(dirname "$0")/../.."
while kill -0 39012 2>/dev/null; do sleep 120; done
export THREADS=8
P='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1'
tools/arm_eval.sh builds/route1/obrrt 100000000 rr \
  base="$P" \
  rr1="$P OBRRT_RATE_ROUTE=1" \
  rr3="$P OBRRT_RATE_ROUTE=3" \
  rr3-hier="$P OBRRT_RATE_ROUTE=3 OBRRT_RATE_MODE=hier" \
  rr3-n05="$P OBRRT_RATE_ROUTE=3 OBRRT_RATE_NOVEL=0.5" \
  -- 1 2 >> logs/rr_screen1.txt 2>&1
tools/arm_eval.sh builds/combo2/obrrt 500000000 ph \
  pf-hier="$P OBRRT_RATE_MODE=hier" \
  -- 1 2 3 >> logs/ph_eval1.txt 2>&1
echo "lane_b5 done"
