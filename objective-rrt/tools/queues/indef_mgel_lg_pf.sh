#!/bin/bash
# Indefinite run of the best 500 M config, mgel-lg-pf (18 stars over 3 seeds; seed 2 had 7),
# started 2026-10-29. Same env as tools/arm_eval.sh, but no OBRRT_MAX_FRAMES and a wall
# limit of ~1 year; outputs are rewritten periodically, so it can be stopped any time.
cd "$(dirname "$0")/../.."
OUT=runs/indef-mgel-lg-pf-s2
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
env $C $BASE $CD $P RAYON_NUM_THREADS=16 nice -n 5 builds/combo1/obrrt ../baserom.us.z64 $OUT 2 525600 > logs/indef-mgel-lg-pf-s2.log 2>&1
echo "indef exited $?"
