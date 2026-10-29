#!/bin/bash
# Route chains (OBRRT_CHAIN=route) and the rate sampler (OBRRT_SAMPLER=rate):
# identity check and quick evaluation arms.
#
#   tools/chain_decay_arms.sh identity <new-bin> [<baseline-bin>]
#       switches off, power-on, seed 1, 2 threads, OBRRT_MEM_MB=1000, 10 M
#       frames: nodes.csv, events.jsonl and tree.bin must be byte-identical.
#       Also once with the C+BASE settings below (10 M).
#   tools/chain_decay_arms.sh eval <bin> [arms...] [-- seed]
#       100 M frames, 16 threads, v47 "mw" settings, ONE ARM AT A TIME (the
#       machine OOM-crashed under parallel load). Arms: base route rate both.
#       Outputs runs/cd-<arm>-s<seed>, logs/cd-<arm>-s<seed>.log.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 OBRRT_MAX_FRAMES=100000000"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
arm_env() {
  case $1 in
    base) echo "";;
    route) echo "OBRRT_CHAIN=route";;
    rate) echo "OBRRT_SAMPLER=rate";;
    both) echo "OBRRT_CHAIN=route OBRRT_SAMPLER=rate";;
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}
mode=${1:?identity or eval}; shift
case $mode in
  identity)
    NEW=${1:?new binary}; OLD=${2:-builds/target-reroute/release/obrrt}
    ok=0
    for set in plain mw; do
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/cd-id-$set-$side
        rm -rf $out
        if [ $set = plain ]; then E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
        else E="$C $BASE OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"; fi
        env $E RAYON_NUM_THREADS=2 nice -n 5 $bin $ROM $out 1 60 > logs/cd-id-$set-$side.log 2>&1 &
      done
      wait
      for f in nodes.csv events.jsonl tree.bin; do
        if cmp runs/cd-id-$set-old/$f runs/cd-id-$set-new/$f; then echo "$set $f identical"; else echo "$set $f DIFFERS"; ok=1; fi
      done
      echo "$set nodes: $(($(wc -l < runs/cd-id-$set-new/nodes.csv) - 1))"
    done
    exit $ok;;
  eval)
    BIN=${1:?binary}; shift
    seed=1; arms=()
    while [ $# -gt 0 ]; do
      if [ "$1" = "--" ]; then seed=$2; shift 2; continue; fi
      arms+=("$1"); shift
    done
    [ ${#arms[@]} -eq 0 ] && arms=(base route rate both)
    for arm in "${arms[@]}"; do
      X=$(arm_env $arm)
      env $C $BASE $X RAYON_NUM_THREADS=16 nice -n 5 $BIN $ROM runs/cd-$arm-s$seed $seed 180 > logs/cd-$arm-s$seed.log 2>&1
      echo "$arm done: $(grep '^best times' logs/cd-$arm-s$seed.log | tail -1)"
    done;;
esac
