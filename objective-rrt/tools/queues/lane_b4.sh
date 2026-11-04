#!/bin/bash
# Lane B4 (8 threads), queued 2026-10-29 behind lane_b3 (pk5 then kn5).
# Two stars per visit (OBRRT_TWOSTAR, new in builds/combo2) on the mgel+painting
# config, with and without the C-button genome bits that won the 50 M smoke.
cd "$(dirname "$0")/../.."
while kill -0 7801 2>/dev/null; do sleep 120; done
export THREADS=8
P='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1'
tools/arm_eval.sh builds/combo2/obrrt 500000000 ts \
  pf-ts="$P OBRRT_TWOSTAR=1" \
  pf-ts-cb="$P OBRRT_TWOSTAR=1 OBRRT_GA_CBUTTONS=0.03" \
  -- 1 2 3 >> logs/ts_eval1.txt 2>&1
echo "lane_b4 done"
