#!/bin/bash
# Lane A4 (16 threads), restarted 2026-10-29 after the "update claude" container reset:
# the lane_a3 painting-fast eval (never started) and the rest of lane_a2 (mx dense arms).
cd "$(dirname "$0")/../.."
P='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1'
# Painting re-entry on top of everything-movement, +/- last-gen dense.
tools/arm_eval.sh builds/combo1/obrrt 500000000 pf mgel-pf="$P" mgel-lg-pf="$P OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" -- 1 2 3 >> logs/pf_eval1.txt 2>&1
G='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1'
# mx remaining: mgel-lg s2..3 (s1 done) and mg-lg s1..3 (s1 was cut at 172 M).
tools/arm_eval.sh builds/movega1/obrrt 500000000 mx mgel-lg="$G OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" mg-lg="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" -- 2 3 >> logs/mx_eval1.txt 2>&1
tools/arm_eval.sh builds/movega1/obrrt 500000000 mx mg-lg="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10" -- 1 >> logs/mx_eval1.txt 2>&1
echo "lane_a4 done"
