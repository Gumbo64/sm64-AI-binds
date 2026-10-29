#!/bin/bash
# Checks for OBRRT_REPLAY_BUDGET (search/src/replay.rs).
#
#   tools/replay_budget_checks.sh identity <new-bin> <baseline-bin>
#       switch unset, seed 1, 2 threads, OBRRT_MEM_MB=1000, 10 M frames,
#       plain and BASE env: nodes.csv, events.jsonl, tree.bin byte-identical.
#   tools/replay_budget_checks.sh det <bin> [fraction]
#       switch on (default 0.1) + OBRRT_REROUTE=1 OBRRT_ZONE_SHARE=1
#       OBRRT_REPLAY_MAX=150, 20 M frames, 2 vs 4 threads: identical trees.
#   tools/replay_budget_checks.sh smoke <bin> <arm> [fraction]
#       50 M frames, 4 threads, BASE + chain/rate/moves/reroute(schema)/zone
#       share; arm "off" (no budget) or "on" (OBRRT_REPLAY_BUDGET=fraction).
# Outputs runs/rb-*, logs/rb-*.log.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
cmp3() {
  local ok=0
  for f in nodes.csv events.jsonl tree.bin; do
    if cmp -s $1/$f $2/$f; then echo "$3 $f identical"; else echo "$3 $f DIFFERS"; ok=1; fi
  done
  echo "$3 nodes: $(($(wc -l < $2/nodes.csv) - 1))"
  return $ok
}
mode=${1:?identity, det or smoke}; shift
case $mode in
  identity)
    NEW=${1:?new binary}; OLD=${2:?baseline binary}
    ok=0
    for set in plain base; do
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/rb-id-$set-$side
        rm -rf $out
        E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
        [ $set = base ] && E="$BASE $E"
        env -u OBRRT_REPLAY_BUDGET $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 60 > logs/rb-id-$set-$side.log 2>&1 &
      done
      wait
      cmp3 runs/rb-id-$set-old runs/rb-id-$set-new $set || ok=1
    done
    exit $ok;;
  det)
    BIN=${1:?binary}; F=${2:-0.1}
    for t in 2 4; do
      out=runs/rb-det-t$t
      rm -rf $out
      env $BASE OBRRT_REROUTE=1 OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_MAX=150 OBRRT_REPLAY_BUDGET=$F OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=20000000 \
        RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM $out 1 120 > logs/rb-det-t$t.log 2>&1 &
    done
    wait
    cmp3 runs/rb-det-t2 runs/rb-det-t4 det;;
  smoke)
    BIN=${1:?binary}; ARM=${2:?off or on}; F=${3:-0.1}
    E="$BASE OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1 OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json OBRRT_ZONE_SHARE=1"
    [ $ARM = on ] && E="$E OBRRT_REPLAY_BUDGET=$F"
    out=runs/rb-smoke-$ARM
    rm -rf $out
    env $E OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=50000000 RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM $out 1 600 > logs/rb-smoke-$ARM.log 2>&1;;
  *) echo "unknown mode $mode" >&2; exit 1;;
esac
