#!/bin/bash
# Generic arm runner on the v47 "mw" settings + route chains + rate sampler,
# one run at a time, 16 threads.
#
#   tools/arm_eval.sh <bin> <frames> <prefix> <name>=<"ENV ..."> ... [-- seeds...]
#       e.g. tools/arm_eval.sh builds/x/obrrt 500000000 hy cds="OBRRT_SWIM_TARGETS=1" -- 1 2 3
#       Outputs runs/<prefix>-<name>-s<seed>, logs/<prefix>-<name>-s<seed>.log.
#       Set NOCD=1 to leave out the route chains + rate sampler.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BIN=${1:?binary}; FRAMES=${2:?frames}; PREFIX=${3:?prefix}; shift 3
names=(); envs=(); seeds=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; seeds=("$@"); break; fi
  names+=("${1%%=*}"); envs+=("${1#*=}"); shift
done
[ ${#seeds[@]} -eq 0 ] && seeds=(1)
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"; [ "${NOCD:-0}" = 1 ] && CD=""
for seed in "${seeds[@]}"; do
  for i in "${!names[@]}"; do
    out=runs/$PREFIX-${names[$i]}-s$seed
    rm -rf "$out"
    env $C $BASE $CD ${envs[$i]} RAYON_NUM_THREADS=${THREADS:-16} nice -n 5 "$BIN" $ROM "$out" "$seed" 180 > logs/$PREFIX-${names[$i]}-s$seed.log 2>&1
    echo "${names[$i]} s$seed: $(grep 'best times' logs/$PREFIX-${names[$i]}-s$seed.log | tail -1)"
  done
done
