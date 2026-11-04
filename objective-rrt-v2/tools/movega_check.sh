#!/usr/bin/env bash
# Checks for OBRRT_MOVE_GA (search/src/movegenome.rs): power-on, seed 1.
#
#   tools/movega_check.sh identity <new bin> [old bin]   # MOVE_GA unset: plain, base, base+MOVES; 2 thr, 10 M
#       (base+MOVES differs on purpose once objectives give arrivals)
#   tools/movega_check.sh det <bin> [frames]             # MOVE_GA=1 (+ MOVES): 2 vs 4 threads, 20 M
#   tools/movega_check.sh smoke <bin> [frames]           # route chain + rate sampler + MOVES + MOVE_GA, 4 thr, 20 M
# X="VAR=1 ..." adds switches to det and smoke (e.g. OBRRT_MOVE_GA_EXTRA=1
# OBRRT_CALM=1); TAG names their outputs (runs/mg-det-$TAG-t2, mg-smoke-$TAG).
#
# Outputs runs/mg-*, logs/mg-*.log.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
same() { # a b label
  local ok=0
  for f in nodes.csv events.jsonl tree.bin; do
    if cmp -s "$1/$f" "$2/$f"; then echo "$3 $f identical"; else echo "$3 $f DIFFERS"; ok=1; fi
  done
  echo "$3 nodes: $(($(wc -l < "$2/nodes.csv") - 1))"
  return $ok
}
mode=${1:?identity, det or smoke}; shift
case $mode in
  identity)
    NEW=${1:?new binary}; OLD=${2:-builds/target-dense/obrrt-v4}
    ok=0
    for set in plain base moves; do
      case $set in
        plain) E="";;
        base) E="$BASE";;
        moves) E="$BASE OBRRT_MOVES=1";;
      esac
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/mg-id-$set-$side
        rm -rf $out
        env $E OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000 RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 60 > logs/mg-id-$set-$side.log 2>&1 &
      done
      wait
      same runs/mg-id-$set-old runs/mg-id-$set-new $set || ok=1
    done
    exit $ok;;
  det)
    BIN=${1:?binary}; F=${2:-20000000}; T=${TAG:-base}
    for t in 2 4; do
      rm -rf runs/mg-det-$T-t$t
      env $BASE OBRRT_MOVES=1 OBRRT_MOVE_GA=1 ${X:-} OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM runs/mg-det-$T-t$t 1 120 > logs/mg-det-$T-t$t.log 2>&1
    done
    same runs/mg-det-$T-t2 runs/mg-det-$T-t4 det-$T;;
  smoke)
    BIN=${1:?binary}; F=${2:-20000000}; T=${TAG:-base}
    rm -rf runs/mg-smoke-$T
    s=$(date +%s)
    env $BASE OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1 OBRRT_MOVE_GA=1 ${X:-} OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=4 nice -n 10 $BIN $ROM runs/mg-smoke-$T 1 120 > logs/mg-smoke-$T.log 2>&1
    echo "smoke-$T exit $? in $(($(date +%s) - s)) s, nodes $(($(wc -l < runs/mg-smoke-$T/nodes.csv) - 1))"; grep -E "move-ga|^best times" logs/mg-smoke-$T.log | tail -2;;
  *) echo "unknown mode $mode" >&2; exit 1;;
esac
