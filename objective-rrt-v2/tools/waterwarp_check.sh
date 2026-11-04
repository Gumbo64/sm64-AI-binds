#!/bin/bash
# Tech::WaterWarp checks (objectives/src/extra/tech/waterwarp.rs): identity with
# OBRRT_WATERWARP unset vs a reference binary, and a smoke run on/off.
#   tools/waterwarp_check.sh id <new bin> <ref bin>     (seed 1, 2 threads, 1 GB, 10 M; plain + base env)
#   tools/waterwarp_check.sh smoke <bin> [frames=50000000] [threads=4]
#   tools/waterwarp_check.sh starts <bin> <frames> <start.m64>...   (on and off side by side, 4 threads each;
#       runs/ww-st-<start>-{on,off}; prints WaterWarp tries/successes and the first new star after the start)
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1 OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1"
case "$1" in
id)
  for env in plain base; do
    E=""; [ $env = base ] && E="$BASE"
    for which in new ref; do
      bin=$2; [ $which = ref ] && bin=$3
      out=runs/ww-id-$env-$which; rm -rf $out
      env $E OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000 RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 180 > logs/ww-id-$env-$which.log 2>&1
    done
    for f in nodes.csv events.jsonl tree.bin; do
      cmp -s runs/ww-id-$env-new/$f runs/ww-id-$env-ref/$f && echo "$env $f identical" || echo "$env $f DIFFERS"
    done
  done ;;
smoke)
  F=${3:-50000000}; T=${4:-4}
  for arm in off on; do
    W=""; [ $arm = on ] && W="OBRRT_WATERWARP=1"
    out=runs/ww-smoke-$arm; rm -rf $out
    env $BASE $W OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=$T nice -n 10 $2 $ROM $out 1 180 > logs/ww-smoke-$arm.log 2>&1
    echo "$arm: $(grep 'best times' logs/ww-smoke-$arm.log | tail -1)"
  done ;;
starts)
  BIN=$2; F=$3; shift 3
  for st in "$@"; do
    n=$(basename $st .m64)
    for arm in off on; do
      W=""; [ $arm = on ] && W="OBRRT_WATERWARP=1"
      out=runs/ww-st-$n-$arm; rm -rf $out
      env $BASE $W OBRRT_BOOT_NOWAIT=1 OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM $out 1 600 $st > logs/ww-st-$n-$arm.log 2>&1 &
    done
    wait
    for arm in off on; do
      out=runs/ww-st-$n-$arm
      star=$(awk -F, 'NR==2{s0=$9; f0=$12} NR>2 && $9>s0 && (b==""||$12<b){b=$12} END{print (b==""?"none":b-f0)}' $out/nodes.csv)
      ww=$(grep -a '^Tech::WaterWarp' logs/ww-st-$n-$arm.log | tail -1 | awk '{print "tries "$5" success "$6" added "$8}')
      hit=$(grep -ac '"kind":"Tech::WaterWarp".*"success":true' $out/events.jsonl)
      echo "$n $arm: nodes $(($(wc -l < $out/nodes.csv)-1)) | first new star +$star frames | WaterWarp ${ww:-none} | success events $hit | $(grep -a 'best times' logs/ww-st-$n-$arm.log | tail -1)"
    done
  done ;;
esac
