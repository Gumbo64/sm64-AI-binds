#!/bin/bash
# Checks and screening arms for kinship counted in attempts
# (OBRRT_KIN_UNIT=attempt, search/src/tree.rs KIN_ATTEMPT / gen_step):
# dense intermediates and zone-share chain nodes cost no generation, so
# OBRRT_KINSHIP_MULT should no longer need to grow with rollout length.
#
#   tools/kin_arms.sh identity <new-bin> <baseline-bin>
#       OBRRT_KIN_UNIT unset, seed 1, 2 threads, OBRRT_MEM_MB=1000: plain,
#       BASE, BASE + route/rate at 10 M frames, and BASE + route/rate +
#       walg-k10 dense at 20 M: nodes.csv, events.jsonl, tree.bin identical.
#   tools/kin_arms.sh det <bin>
#       BASE + route/rate + walg + KIN_UNIT=attempt, 20 M, 2 vs 4 threads.
#   tools/kin_arms.sh screen <bin> [frames] [seed] [arm...]
#       4 threads, BASE + route/rate, default 100 M frames, seed 1; arms
#       (default all): walg-k10, walg-attempt-k1, walg-attempt-k2,
#       lg-attempt-k1, wa-attempt-k1, walg-attempt-every15/30,
#       walg-attempt-every30-noair, walg-attempt-obj (every60 on request;
#       OBRRT_DENSE_EVERY=<n>|obj, OBRRT_DENSE_AIR). Arms run one after another.
#       KIN_PAR=n runs n arms at once (default 1).
#       Grid arms <add>-<keep>-<kin> (see arm_env), e.g. walg-obj-soft8.
# Outputs runs/kn-*, logs/kn-*.log; tools/analysis/kin_screen.py sums them.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
RATE="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
WALG="OBRRT_ADD=winner_all,last_gen"
UNSET="-u OBRRT_KIN_SOFT -u OBRRT_KIN_FRAMES -u OBRRT_KIN_UNIT -u OBRRT_KIN_CAP -u OBRRT_KINSHIP_MULT -u OBRRT_ADD -u OBRRT_DENSE_EVERY -u OBRRT_DENSE_AIR"
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
    walg-k10) echo "$WALG OBRRT_KINSHIP_MULT=10";;
    walg-attempt-k1) echo "$WALG OBRRT_KIN_UNIT=attempt";;
    walg-attempt-k2) echo "$WALG OBRRT_KIN_UNIT=attempt OBRRT_KINSHIP_MULT=2";;
    lg-attempt-k1) echo "OBRRT_ADD=last_gen OBRRT_KIN_UNIT=attempt";;
    wa-attempt-k1) echo "OBRRT_ADD=winner_all OBRRT_KIN_UNIT=attempt";;
    walg-attempt-every15) echo "$WALG OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=15";;
    walg-attempt-every30) echo "$WALG OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=30";;
    walg-attempt-every60) echo "$WALG OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=60";;
    walg-attempt-obj) echo "$WALG OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=obj";;
    walg-attempt-every30-noair) echo "$WALG OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=30 OBRRT_DENSE_AIR=0";;
    # Grid arms <add>-<keep>-<kin>: add lg|walg; keep all|every30|obj;
    # kin n10 (nodes x10) | att (attempt) | frm (frames, 150/generation) |
    # soft5 / soft8 (attempt unit, OBRRT_KIN_SOFT 0.5 / 0.8).
    *-*-*)
      local add keep kin e
      IFS=- read add keep kin <<< "$1"
      case $add in lg) e="OBRRT_ADD=last_gen";; walg) e="$WALG";; *) echo "unknown add $add" >&2; exit 1;; esac
      case $keep in all) ;; every30) e="$e OBRRT_DENSE_EVERY=30";; obj) e="$e OBRRT_DENSE_EVERY=obj";; *) echo "unknown keep $keep" >&2; exit 1;; esac
      case $kin in
        n10) e="$e OBRRT_KINSHIP_MULT=10";;
        att) e="$e OBRRT_KIN_UNIT=attempt";;
        frm) e="$e OBRRT_KIN_UNIT=frames";;
        soft5) e="$e OBRRT_KIN_UNIT=attempt OBRRT_KIN_SOFT=0.5";;
        soft8) e="$e OBRRT_KIN_UNIT=attempt OBRRT_KIN_SOFT=0.8";;
        *) echo "unknown kin $kin" >&2; exit 1;;
      esac
      echo "$e";;
    *) echo "unknown arm $1" >&2; exit 1;;
  esac
}
mode=${1:?identity, det or screen}; shift
case $mode in
  identity)
    NEW=${1:?new binary}; OLD=${2:?baseline binary}
    ok=0
    for set in plain base baserate dense; do
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/kn-id-$set-$side
        rm -rf $out
        E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
        [ $set = base ] && E="$BASE $E"
        [ $set = baserate ] && E="$BASE $RATE $E"
        [ $set = dense ] && E="$BASE $RATE $WALG OBRRT_KINSHIP_MULT=10 OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=20000000"
        env $UNSET $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 120 > logs/kn-id-$set-$side.log 2>&1 &
      done
      wait
      cmp3 runs/kn-id-$set-old runs/kn-id-$set-new $set || ok=1
    done
    exit $ok;;
  det)
    BIN=${1:?binary}
    for t in 2 4; do
      out=runs/kn-det-t$t
      rm -rf $out
      env $UNSET $BASE $RATE $WALG OBRRT_KIN_UNIT=attempt OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=20000000 \
        RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM $out 1 120 > logs/kn-det-t$t.log 2>&1 &
    done
    wait
    cmp3 runs/kn-det-t2 runs/kn-det-t4 det;;
  screen)
    BIN=${1:?binary}; F=${2:-100000000}; SEED=${3:-1}; shift 3 2>/dev/null || shift $#
    ARMS=${*:-walg-k10 walg-attempt-k1 walg-attempt-k2 lg-attempt-k1 wa-attempt-k1 walg-attempt-every15 walg-attempt-every30 walg-attempt-every30-noair walg-attempt-obj}
    PAR=${KIN_PAR:-1}
    run_arm() {
      local a=$1 out=runs/kn-$1-s$SEED
      rm -rf $out
      env $UNSET $BASE $RATE $(arm_env $a) OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=4 \
        nice -n 10 $BIN $ROM $out $SEED 600 > logs/kn-$a-s$SEED.log 2>&1
      echo "$a s$SEED: $(grep -E '^best times|^first arrivals|kin sets' logs/kn-$a-s$SEED.log | tail -3 | tr '\n' ' ')"
    }
    n=0
    for a in $ARMS; do
      run_arm $a &
      n=$((n + 1))
      if [ $n -ge $PAR ]; then wait -n; n=$((n - 1)); fi
    done
    wait;;
  *) echo "unknown mode $mode" >&2; exit 1;;
esac
