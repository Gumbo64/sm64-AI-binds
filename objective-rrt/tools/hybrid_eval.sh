#!/bin/bash
# The hybrid's 500 M comparison: v47 "mw" settings × arms × seeds, one run at
# a time, 16 threads (the machine OOM-crashed under parallel load).
#
#   tools/hybrid_eval.sh <bin> [frames] [arms...] [-- seeds...]
#       arms: base cd (OBRRT_CHAIN=route OBRRT_SAMPLER=rate) cdm (cd +
#       OBRRT_MOVES=1) cdmr (cdm + OBRRT_REROUTE with the reroute schema).
#       Outputs runs/hy-<arm>-s<seed>, logs/hy-<arm>-s<seed>.log.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BIN=${1:?binary}; shift
FRAMES=${1:-500000000}; [ $# -gt 0 ] && shift
arms=(); seeds=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; seeds=("$@"); break; fi
  arms+=("$1"); shift
done
[ ${#arms[@]} -eq 0 ] && arms=(base cd cdm cdmr)
[ ${#seeds[@]} -eq 0 ] && seeds=(1 2 3)
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
RR="OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json"
arm_env() {
  case $1 in
    base) echo "";;
    cd) echo "$CD";;
    cdm) echo "$CD OBRRT_MOVES=1";;
    cdmr) echo "$CD OBRRT_MOVES=1 $RR";;
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}
for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    X=$(arm_env "$arm")
    out=runs/hy-$arm-s$seed
    rm -rf "$out"
    env $C $BASE $X RAYON_NUM_THREADS=16 nice -n 5 "$BIN" $ROM "$out" "$seed" 180 > logs/hy-$arm-s$seed.log 2>&1
    echo "$arm s$seed: $(grep '^best times' logs/hy-$arm-s$seed.log | tail -1)"
  done
done
