#!/bin/bash
# 50M smoke for OBRRT_GROUNDS_FIX / OBRRT_MOATWARP: tools/grounds_smoke.sh <bin> <seed> [arms...]
# Arms: off, fix (OBRRT_GROUNDS_FIX=1), fixmw (+ OBRRT_MOATWARP=1). Then
# tools/analysis/grounds_opening.py runs/gs-*.
BIN=${1:?bin}; SEED=${2:-1}; shift 2; ARMS=${@:-off fix fixmw}; ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1 OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1 OBRRT_LAKITU_ARC=1"
for arm in $ARMS; do
  case $arm in off) E="";; fix) E="OBRRT_GROUNDS_FIX=1";; fixmw) E="OBRRT_GROUNDS_FIX=1 OBRRT_MOATWARP=1";; esac
  env $BASE $E OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=50000000 RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM runs/gs-$arm-s$SEED $SEED 180 > logs/gs-$arm-s$SEED.log 2>&1
  echo "$arm done"
done
