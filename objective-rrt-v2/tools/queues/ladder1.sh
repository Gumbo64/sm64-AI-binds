#!/bin/bash
# Feature ladder 1 (2026-11-01): v1's switches re-enabled on v2 one group at a
# time (cumulative), to find which help v2 (v1 GA + ETA objectives + gamma).
# 100 M frames, seeds 1 and 2 in parallel at 8 threads each, rung after rung.
# builds/v2a/obrrt2 is the snapshot under test; v1 references use
# ../objective-rrt/builds/combo1/obrrt (the binary of v1's pf eval).
# Results: logs/ladder1.txt; runs/l1-<rung>-s<seed>.
cd "$(dirname "$0")/../.."
ROM=../baserom.us.z64
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 OBRRT_MAX_FRAMES=100000000"
R1="OBRRT_PICK_BY_B=0 OBRRT_NEW_ELIGIBLE=1 OBRRT_KEEP_WIP=1 OBRRT_STARSET_KEY=1"
R2="$R1 OBRRT_TECH=all OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
R3="$R2 OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
R4="$R3 OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1"
R5="$R4 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"

rung() {  # name binary env...
  local name=$1 bin=$2; shift 2
  for seed in 1 2; do
    out=runs/l1-$name-s$seed
    rm -rf "$out"
    env $C "$@" RAYON_NUM_THREADS=8 nice -n 5 "$bin" $ROM "$out" $seed 180 > logs/l1-$name-s$seed.log 2>&1 &
  done
  wait
  for seed in 1 2; do
    echo "$name s$seed: $(grep 'best times' logs/l1-$name-s$seed.log | tail -1)"
  done
}

V2=builds/v2a/obrrt2
V1=../objective-rrt/builds/combo1/obrrt
rung r0 $V2
rung r1 $V2 $R1
rung r2 $V2 $R2
rung r3 $V2 $R3
rung r4 $V2 $R4
rung r5 $V2 $R5
rung v1r0 $V1
rung v1r5 $V1 $R5
echo "ladder1 done"
