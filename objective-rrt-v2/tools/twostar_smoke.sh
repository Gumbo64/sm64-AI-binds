#!/bin/bash
# OBRRT_TWOSTAR smoke from power-on: base env + route chains + rate sampler +
# moves, switch off vs on (run side by side), then stars per visit.
#   tools/twostar_smoke.sh <bin> <frames> <prefix> [seed=1]   (THREADS 4, MEM 1500)
set -u
cd "$(dirname "$0")/.."
BIN=${1:?binary}; FRAMES=${2:?frames}; PREFIX=${3:?prefix}; SEED=${4:-1}
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=${MEM:-1500} OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1"
for arm in off on; do
  e=""; [ $arm = on ] && e="OBRRT_TWOSTAR=1"
  rm -rf runs/$PREFIX-$arm-s$SEED
  env $C $BASE $CD $e RAYON_NUM_THREADS=${THREADS:-4} nice -n 10 "$BIN" ../baserom.us.z64 runs/$PREFIX-$arm-s$SEED "$SEED" 900 > logs/$PREFIX-$arm-s$SEED.log 2>&1 &
done
wait
for arm in off on; do
  echo "$arm: $(grep 'best times' logs/$PREFIX-$arm-s$SEED.log | tail -1)"
  python3 tools/analysis/twostar_visits.py runs/$PREFIX-$arm-s$SEED
done
