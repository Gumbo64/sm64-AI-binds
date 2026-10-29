#!/bin/bash
# OBRRT_PAINTING_FAST checks (objectives/src/painting.rs).
#
#   tools/painting_fast_checks.sh identity <new-bin> <baseline-bin> [frames=10M]
#       switch unset, seed 1, 2 threads, OBRRT_MEM_MB=1000, plain and base
#       env: nodes.csv / events.jsonl / tree.bin byte-identical
#   tools/painting_fast_checks.sh lab <painting_lab-bin> [seeds=6]
#       painting_lab ga from the post-star castle states of the hy/dn/mg runs
#       (and the lobby after GoToCastle), switch off vs on
#   tools/painting_fast_checks.sh smoke <bin> <out> [frames=50M] [seed=1]
#       base env + OBRRT_PAINTING_FAST=1, 4 threads
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1 OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1"
case ${1:-} in
identity)
  NEW=$2; OLD=$3; FRAMES=${4:-10000000}; ok=0
  for m in plain base; do
    E=""; [ $m = base ] && E="$BASE"
    for side in old new; do
      bin=$OLD; [ $side = new ] && bin=$NEW
      rm -rf runs/pf-id-$m-$side
      env $E OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=$FRAMES RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM runs/pf-id-$m-$side 1 600 > logs/pf-id-$m-$side.log 2>&1 &
    done
    wait
    same=1
    for f in nodes.csv events.jsonl tree.bin; do
      if cmp -s runs/pf-id-$m-old/$f runs/pf-id-$m-new/$f; then echo "$m $f identical"; else echo "$m $f DIFFERS"; same=0; ok=1; fi
    done
    echo "$m nodes: $(($(wc -l < runs/pf-id-$m-new/nodes.csv) - 1))"
    [ $same = 1 ] && rm -rf runs/pf-id-$m-old runs/pf-id-$m-new
  done
  exit $ok;;
lab)
  LAB=$2; SEEDS=${3:-6}
  # run:node:painting (the painting the route entered next); star nodes
  # are answered through the save menu by the lab.
  for spec in ${CASES:-hy-cd-s3:7081:wf mg-mgel-s1:6865:wf hy-cdmz-s2:3339:wf dn-lg-k10-s1:55190:wf dn-lg-k10-s1:16493:bob hy-cdmz-s2:897:bob mg-mgel-s1:831:ccm hy-cd-s3:10472:wf mg-mgel-s1:2711:wf hy-cdmz-s2:8439:bob hy-cdmz-s2:9233:ccm hy-cdmz-s2:1370:wf mg-mgel-s1:1322:bob mg-mgel-s1:7449:bob}; do
    IFS=: read -r run node paint <<< "$spec"
    for fast in 0 1; do
      echo "== $spec FAST=$fast"
      env $BASE OBRRT_PAINTING_FAST=$fast nice -n 10 $LAB $ROM ga tree:runs/$run:$node seeds=$SEEDS paint=$paint ${HORIZON:+horizon=$HORIZON} 2>&1 | grep -E '^(start|summary|seed)'
    done
  done;;
smoke)
  BIN=$2; OUT=$3; FRAMES=${4:-50000000}; SEED=${5:-1}
  rm -rf $OUT
  env $BASE OBRRT_PAINTING_FAST=1 OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$FRAMES RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM $OUT $SEED 600 > $OUT.log 2>&1;;
*) sed -n 2,13p "$0"; exit 2;;
esac
