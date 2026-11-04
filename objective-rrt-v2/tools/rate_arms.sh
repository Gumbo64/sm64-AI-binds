#!/bin/bash
# Checks and screening arms for the rate sampler's hierarchical mode and
# no-start skip (search/src/pool.rs rate_refresh_hier, engine.rs
# nostart_recheck): OBRRT_RATE_MODE=hier, OBRRT_RATE_NOSTART_SKIP=1.
#
#   tools/rate_arms.sh identity <new-bin> <baseline-bin>
#       new switches unset, seed 1, 2 threads, OBRRT_MEM_MB=1000, 10 M
#       frames; plain, BASE, and BASE + CHAIN=route SAMPLER=rate: nodes.csv,
#       events.jsonl, tree.bin byte-identical.
#   tools/rate_arms.sh det <bin>
#       BASE + route/rate + hier + skip, 20 M frames, 2 vs 4 threads.
#   tools/rate_arms.sh screen <bin> [frames] [seed] [arm...]
#       4 threads, BASE + CHAIN=route SAMPLER=rate, default 100 M frames,
#       seed 1; arms (default all): rate, hier2e4, hier2e5, hier2e4skip,
#       hier2e5skip. Arms run one after another (4 threads each).
# Outputs runs/rt-*, logs/rt-*.log; tools/analysis/rate_diag.py reads them.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
RATE="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
UNSET="-u OBRRT_RATE_MODE -u OBRRT_RATE_NOSTART_SKIP -u OBRRT_RATE_KIND_SCALE -u OBRRT_RATE_SKIP_BUDGET"
cmp3() {
  local ok=0
  for f in nodes.csv events.jsonl tree.bin; do
    if cmp -s $1/$f $2/$f; then echo "$3 $f identical"; else echo "$3 $f DIFFERS"; ok=1; fi
  done
  echo "$3 nodes: $(($(wc -l < $2/nodes.csv) - 1))"
  return $ok
}
arm_env() {
  case $1 in
    rate) echo "";;
    hier2e4) echo "OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=20000";;
    hier2e5) echo "OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=200000";;
    hier2e4skip) echo "OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=20000 OBRRT_RATE_NOSTART_SKIP=1";;
    hier2e5skip) echo "OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=200000 OBRRT_RATE_NOSTART_SKIP=1";;
    rateskip) echo "OBRRT_RATE_NOSTART_SKIP=1";;
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}
mode=${1:?identity, det or screen}; shift
case $mode in
  identity)
    NEW=${1:?new binary}; OLD=${2:?baseline binary}
    ok=0
    for set in plain base baserate; do
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/rt-id-$set-$side
        rm -rf $out
        E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
        [ $set = base ] && E="$BASE $E"
        [ $set = baserate ] && E="$BASE $RATE $E"
        env $UNSET $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 60 > logs/rt-id-$set-$side.log 2>&1 &
      done
      wait
      cmp3 runs/rt-id-$set-old runs/rt-id-$set-new $set || ok=1
    done
    exit $ok;;
  det)
    BIN=${1:?binary}
    for t in 2 4; do
      out=runs/rt-det-t$t
      rm -rf $out
      env $BASE $RATE $(arm_env hier2e4skip) OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=20000000 \
        RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM $out 1 120 > logs/rt-det-t$t.log 2>&1 &
    done
    wait
    cmp3 runs/rt-det-t2 runs/rt-det-t4 det;;
  screen)
    BIN=${1:?binary}; F=${2:-100000000}; SEED=${3:-1}; shift 3 2>/dev/null || shift $#
    ARMS=${*:-rate hier2e4 hier2e5 hier2e4skip hier2e5skip}
    for a in $ARMS; do
      out=runs/rt-$a-s$SEED
      rm -rf $out
      env $UNSET $BASE $RATE $(arm_env $a) OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=4 \
        nice -n 10 $BIN $ROM $out $SEED 600 > logs/rt-$a-s$SEED.log 2>&1
      echo "$a: $(grep -E '^best times|^first arrivals' logs/rt-$a-s$SEED.log | tail -2 | tr '\n' ' ')"
    done;;
  *) echo "unknown mode $mode" >&2; exit 1;;
esac
