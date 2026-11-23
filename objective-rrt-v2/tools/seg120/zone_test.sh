#!/bin/bash
# One zone-locked tree from a start movie, with the campaign's settings plus extras.
#   tools/seg120/zone_test.sh NAME START.m64 FRAMES THREADS [VAR=value ...]
# Env: BIN (default builds/v2q/obrrt2), SEED (default 1), GRACE (frames after the first good exit, default 20e6).
# Output runs/NAME, log logs/seg120/NAME.log, PID logs/seg120/NAME.pid.
cd "$(dirname "$0")/../.."
name=$1; start=$2; frames=$3; threads=$4; shift 4
BIN=${BIN:-builds/v2q/obrrt2}
rm -rf runs/$name
echo $$ > logs/seg120/$name.pid
exec env OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=6000 OBRRT_ROUND=64 OBRRT_BOOT_NOWAIT=1 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 \
  OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_CHAIN=route OBRRT_SAMPLER=rate \
  OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen \
  OBRRT_KINSHIP_MULT=10 OBRRT_ZONE_LOCK=1 RAYON_NUM_THREADS=$threads OBRRT_MAX_FRAMES=$frames OBRRT_STOP_EXIT_GRACE=${GRACE:-20000000} "$@" \
  nice -n 5 $BIN ../baserom.us.z64 runs/$name ${SEED:-1} 600 $start > logs/seg120/$name.log 2>&1
