#!/bin/bash
# Route polish eval (2026-10-30): mgel-lg-pf + OBRRT_POLISH=0.1, builds/polish1, 8 threads, seeds 1 3.
# Compare with logs/pf_eval1.txt (mgel-lg-pf: 6, 7, 5 stars).
cd "$(dirname "$0")/../.."
export THREADS=8
tools/arm_eval.sh builds/polish1/obrrt 500000000 pol lgpf-pol="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10 OBRRT_POLISH=0.1" -- 1 3 >> logs/pol_eval1.txt 2>&1
echo "lane_pol_a done"
