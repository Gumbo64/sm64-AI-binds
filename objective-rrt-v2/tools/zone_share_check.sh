#!/bin/bash
# Zone-entry input sharing (OBRRT_ZONE_SHARE=1, search/src/share.rs): checks.
#
#   tools/zone_share_check.sh identity <new-bin> [<baseline-bin>]
#       switch off vs the baseline (default builds/hybrid1/obrrt): power-on,
#       seed 1, 2 threads, OBRRT_MEM_MB=1000, 10 M frames, plain and with the
#       "mw" settings: nodes.csv, events.jsonl, tree.bin byte-identical.
#   tools/zone_share_check.sh det <bin>
#       switch on (mw settings + OBRRT_ZONE_SHARE=1), 20 M frames, 2 vs 4
#       threads: the same three files byte-identical.
#   tools/zone_share_check.sh smoke <bin> [threads]
#       20 M frames, mw + route chains, rate sampler, moves, zone share on;
#       prints the zone share line. Outputs runs/zs-smoke, logs/zs-smoke.log.
#       Check shared nodes with tools/zone_share_verify.py runs/zs-smoke.
# Runs at nice 10 and at most 4 threads so a running eval keeps the CPU.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
MW="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
same() { # dir-a dir-b label
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
    NEW=${1:?new binary}; OLD=${2:-builds/hybrid1/obrrt}
    ok=0
    for set in plain mw; do
      E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
      [ $set = mw ] && E="$MW $E"
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        rm -rf runs/zs-id-$set-$side
        env $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM runs/zs-id-$set-$side 1 60 > logs/zs-id-$set-$side.log 2>&1 &
      done
      wait
      same runs/zs-id-$set-old runs/zs-id-$set-new $set || ok=1
    done
    exit $ok;;
  det)
    BIN=${1:?binary}
    for t in 2 4; do
      rm -rf runs/zs-det-t$t
      env $MW OBRRT_ZONE_SHARE=1 OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=20000000 RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM runs/zs-det-t$t 1 120 > logs/zs-det-t$t.log 2>&1 &
    done
    wait
    grep "zone share:" logs/zs-det-t4.log | tail -1
    same runs/zs-det-t2 runs/zs-det-t4 det;;
  smoke)
    BIN=${1:?binary}; T=${2:-4}
    rm -rf runs/zs-smoke
    env $MW OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1 OBRRT_ZONE_SHARE=1 OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=20000000 \
      RAYON_NUM_THREADS=$T nice -n 10 $BIN $ROM runs/zs-smoke 1 120 > logs/zs-smoke.log 2>&1
    grep "zone share:" logs/zs-smoke.log | tail -1
    grep "^best times" logs/zs-smoke.log | tail -1;;
esac
