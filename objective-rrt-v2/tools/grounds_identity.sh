#!/bin/bash
# Identity check for the OBRRT_GROUNDS_FIX / OBRRT_MOATWARP / OBRRT_PAUSE_BUF
# build with the switches unset: tools/grounds_identity.sh <new-bin> <old-bin>
# Runs seed 1, 2 threads, 10M frames, plain and base env, and compares nodes.csv.
NEW=${1:?new}; OLD=${2:?old}; ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
for arm in plain base; do
  E=""; [ $arm = base ] && E="$BASE"
  for v in new old; do
    B=$NEW; [ $v = old ] && B=$OLD
    env $E OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000 RAYON_NUM_THREADS=2 nice -n 10 $B $ROM runs/gid-$arm-$v 1 180 > logs/gid-$arm-$v.log 2>&1 &
  done
  wait
  if cmp -s runs/gid-$arm-new/nodes.csv runs/gid-$arm-old/nodes.csv; then echo "$arm: IDENTICAL ($(wc -l < runs/gid-$arm-new/nodes.csv) nodes)"; else echo "$arm: DIFFER"; fi
done
