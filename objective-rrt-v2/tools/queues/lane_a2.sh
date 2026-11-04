#!/bin/bash
# Lane A2 (16 threads), queued 2026-10-29 after lane A: everything-movement (mgel) + last-generation dense states (lg-k10, the most stars of any arm).
cd "$(dirname "$0")/../.."
while kill -0 1994 2>/dev/null; do sleep 60; done
G='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1'
tools/arm_eval.sh builds/movega1/obrrt 500000000 mx mgel-lg="$G OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" mg-lg="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" -- 1 2 3 >> logs/mx_eval1.txt 2>&1
