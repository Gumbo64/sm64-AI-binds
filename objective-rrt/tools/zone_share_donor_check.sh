#!/bin/bash
# OBRRT_ZONE_SHARE_DONOR=sample (search/src/share.rs) checks.
#
#   tools/zone_share_donor_check.sh identity <new-bin> <baseline-bin>
#       unset vs baseline, seed 1, 2 threads, OBRRT_MEM_MB=1000: plain and
#       base env at 10 M, base + OBRRT_ZONE_SHARE=1 at 30 M; nodes.csv,
#       events.jsonl, tree.bin byte-identical.
#   tools/zone_share_donor_check.sh det <bin>
#       base + ZONE_SHARE=1 + DONOR=sample, 30 M, 2 vs 4 threads.
#   tools/zone_share_donor_check.sh smoke <bin> <name> [ENV...]
#       100 M, 4 threads, base + route chains, rate sampler, moves, zone
#       share, replay budget 0.1 / max 150, OBRRT_ZONE_SHARE_STATS=1, plus
#       ENV. Outputs runs/zsd-<name>, logs/zsd-<name>.log.
# nice 10, at most 4 threads.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
same() {
  local ok=0
  for f in nodes.csv events.jsonl tree.bin; do
    if cmp -s $1/$f $2/$f; then echo "$3 $f identical"; else echo "$3 $f DIFFERS"; ok=1; fi
  done
  echo "$3 nodes: $(($(wc -l < $2/nodes.csv) - 1))"
  return $ok
}
mkdir -p logs runs
mode=${1:?identity, det or smoke}; shift
case $mode in
  identity)
    NEW=${1:?new}; OLD=${2:?baseline}; ok=0
    for set in plain base zs; do
      case $set in
        plain) E="OBRRT_MAX_FRAMES=10000000";;
        base) E="$BASE OBRRT_MAX_FRAMES=10000000";;
        zs) E="$BASE OBRRT_ZONE_SHARE=1 OBRRT_MAX_FRAMES=30000000";;
      esac
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        rm -rf runs/zsd-id-$set-$side
        env $E OBRRT_MEM_MB=1000 RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM runs/zsd-id-$set-$side 1 120 > logs/zsd-id-$set-$side.log 2>&1 &
      done
      wait
      if same runs/zsd-id-$set-old runs/zsd-id-$set-new $set; then rm -rf runs/zsd-id-$set-*; else ok=1; fi
    done
    exit $ok;;
  det)
    BIN=${1:?binary}
    for t in 2 4; do
      rm -rf runs/zsd-det-t$t
      env $BASE OBRRT_ZONE_SHARE=1 OBRRT_ZONE_SHARE_DONOR=sample OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=30000000 RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM runs/zsd-det-t$t 1 120 > logs/zsd-det-t$t.log 2>&1 &
    done
    wait
    grep "zone share:" logs/zsd-det-t4.log | tail -1
    same runs/zsd-det-t2 runs/zsd-det-t4 det && rm -rf runs/zsd-det-t*;;
  smoke)
    BIN=${1:?binary}; NAME=${2:?name}; shift 2
    rm -rf runs/zsd-$NAME
    env $BASE OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1 OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1 OBRRT_REPLAY_MAX=150 \
      OBRRT_ZONE_SHARE_STATS=1 "$@" OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=100000000 \
      RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM runs/zsd-$NAME 1 900 > logs/zsd-$NAME.log 2>&1
    grep "zone share:" logs/zsd-$NAME.log | tail -1
    grep "^best times" logs/zsd-$NAME.log | tail -1;;
esac
