#!/bin/bash
# Queue 15 (2026-11-02): parent window on P, v2k, 500 M x seeds 1-3. Starts when q14 ends.
# An attempt starts from the selected node's parent; every gen-0 genome begins with the recorded
# parent-to-node inputs. Compare with q10 (v2h P: 5, 5, 5) and v1 (6, 7, 5).
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q14.pid)" 2>/dev/null; do timeout 60 tail -f /dev/null; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
PAR=3 THREADS=5 tools/screen.sh builds/v2k/obrrt2 500000000 q15 \
  pw1="$P OBRRT_PARENT_WINDOW=1" \
  pw25="$P OBRRT_PARENT_WINDOW=0.25" \
  -- 1 2 3
echo "q15 done"
