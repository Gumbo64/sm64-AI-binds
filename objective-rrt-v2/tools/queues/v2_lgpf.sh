#!/bin/bash
# v2 (v1 GA + v2 ETA objectives + gamma 0.99) on the best v1 config, mgel-lg-pf,
# seeds 1-3 x 500 M, 16 threads, sequential. Compare with v1's
# ../objective-rrt/runs/pf-mgel-lg-pf-s{1,2,3} (logs/pf_eval1.txt: 6, 7, 5 stars).
cd "$(dirname "$0")/../.."
cp target/release/obrrt2 builds/v2a/obrrt2
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
THREADS=16 tools/arm_eval.sh builds/v2a/obrrt2 500000000 v2 lgpf="$P" -- 1 2 3 >> logs/v2_lgpf_eval1.txt 2>&1
echo "v2_lgpf exited $?" >> logs/v2_lgpf_eval1.txt
