#!/bin/bash
# Lane A3 (16 threads), queued 2026-10-29 after lane A2: painting re-entry fix (OBRRT_PAINTING_FAST) on top of everything-movement, with and without last-gen dense states. Binary builds/combo1 = paint1 (all features to date).
cd "$(dirname "$0")/../.."
while kill -0 3523150 2>/dev/null; do sleep 60; done
G='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1'
tools/arm_eval.sh builds/combo1/obrrt 500000000 pf mgel-pf="$G" mgel-lg-pf="$G OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" -- 1 2 3 >> logs/pf_eval1.txt 2>&1
