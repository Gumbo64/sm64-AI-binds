#!/bin/bash
# Screen arms on v2: tools/screen.sh BIN FRAMES PREFIX name="ENV ..." ... [-- seeds]
# PAR seeds at a time in parallel (default 2) at THREADS threads each (default 8), arm after arm. Every arm
# runs on the v1 best config's base (soft dedup, BASE, CD); arms add to it.
# Per run: runs/<PREFIX>-<name>-s<seed>, logs/<PREFIX>-<name>-s<seed>.log;
# one "best times" line per run and a compare.py milestone table per arm
# go to stdout (the queue appends it to logs/<PREFIX>.txt).
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BIN=${1:?binary}; FRAMES=${2:?frames}; PREFIX=${3:?prefix}; shift 3
names=(); envs=(); seeds=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; seeds=("$@"); break; fi
  names+=("${1%%=*}"); envs+=("${1#*=}"); shift
done
[ ${#seeds[@]} -eq 0 ] && seeds=(1 2)
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
for i in "${!names[@]}"; do
  runs=()
  PAR=${PAR:-2}
  for ((j = 0; j < ${#seeds[@]}; j += PAR)); do
    for seed in "${seeds[@]:j:PAR}"; do
      out=runs/$PREFIX-${names[$i]}-s$seed
      rm -rf "$out"
      env $C $BASE $CD ${envs[$i]} RAYON_NUM_THREADS=${THREADS:-8} nice -n 5 "$BIN" $ROM "$out" "$seed" 600 > logs/$PREFIX-${names[$i]}-s$seed.log 2>&1 &
      runs+=("$PREFIX-${names[$i]}-s$seed")
    done
    wait
  done
  echo "== ${names[$i]}: ${envs[$i]}"
  for r in "${runs[@]}"; do
    echo "$r: $(grep 'best times' logs/$r.log | tail -1)$(grep -q panicked logs/$r.log && echo ' PANICKED')"
  done
  python3 tools/v2sum.py "${runs[@]}"
  python3 tools/compare.py "${runs[@]}" 2>&1 | grep -E "^(run|castle|bob |summit|king hit)"
done
