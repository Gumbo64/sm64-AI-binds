#!/bin/bash
# OBRRT_TWOSTAR lab: the search from a course-entry state (an .m64 from
# power-on, e.g. ref/starts/twostar-wf-entry-hy-cdmz-s2-n2124.m64), switch
# off vs on, one after the other, then stars per visit (tools/analysis/twostar_visits.py).
#
#   tools/twostar_lab.sh <bin> <start.m64> <frames> <prefix> [seed=1] [<name>=<"ENV ..."> ...]
#       default arms: off="" on="OBRRT_TWOSTAR=1"
#       Outputs runs/<prefix>-<name>-s<seed>, logs/<prefix>-<name>-s<seed>.log.
#       THREADS (4), MEM (1500).
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BIN=${1:?binary}; START=${2:?start.m64}; FRAMES=${3:?frames}; PREFIX=${4:?prefix}; SEED=${5:-1}; shift 5 2>/dev/null || shift $#
names=(); envs=()
while [ $# -gt 0 ]; do names+=("${1%%=*}"); envs+=("${1#*=}"); shift; done
[ ${#names[@]} -eq 0 ] && { names=(off on); envs=("" "OBRRT_TWOSTAR=1"); }
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=${MEM:-1500} OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1"
for i in "${!names[@]}"; do
  out=runs/$PREFIX-${names[$i]}-s$SEED
  rm -rf "$out"
  env $C $BASE $CD ${envs[$i]} RAYON_NUM_THREADS=${THREADS:-4} nice -n 10 "$BIN" $ROM "$out" "$SEED" 600 "$START" > logs/$PREFIX-${names[$i]}-s$SEED.log 2>&1
done
for i in "${!names[@]}"; do
  out=runs/$PREFIX-${names[$i]}-s$SEED
  echo "${names[$i]}: $(grep 'best times' logs/$PREFIX-${names[$i]}-s$SEED.log | tail -1)"
  python3 tools/analysis/twostar_visits.py "$out"
done
