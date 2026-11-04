#!/bin/bash
# OBRRT_PAINTING_FAST re-entry smokes: the base env (as
# tools/painting_fast_checks.sh) from post-star castle starts in ref/starts
# (painting_lab export), switch off and on side by side at 2 threads each,
# then tools/analysis/route_timeline.py and painting_reentry.py on both.
#
#   tools/painting_fast_reentry.sh <bin> [frames=40M] [seed=1] [start ...]
# Outputs runs/pf-re-<start>-{off,on}; deletes nothing.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BIN=$1; FRAMES=${2:-40000000}; SEED=${3:-1}
shift 3 2>/dev/null || shift $#
STARTS=${*:-post-star-bob1-hy-cdmz-s2-n976 post-star-wf3-mg-mgel-s1-n5752}
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1 OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1"
for s in $STARTS; do
  for fast in 0 1; do
    side=off; [ $fast = 1 ] && side=on
    out=runs/pf-re-${s#post-star-}-$side
    rm -rf $out
    env $BASE OBRRT_PAINTING_FAST=$fast OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$FRAMES RAYON_NUM_THREADS=2 nice -n 10 $BIN $ROM $out $SEED 600 ref/starts/$s.m64 > $out.log 2>&1 &
  done
  wait
  for side in off on; do
    out=runs/pf-re-${s#post-star-}-$side
    echo "=== $out"
    python3 tools/analysis/route_timeline.py $out | sed -n '/^# /p;/## phases/,/## dwells/p'
  done
done
