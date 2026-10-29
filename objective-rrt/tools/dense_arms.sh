#!/bin/bash
# Dense adds (OBRRT_ADD), sparse saves (OBRRT_REPLAY_MAX), kinship scaling
# (OBRRT_KINSHIP_MULT) and nearby seeds (OBRRT_NEAR_SEED): checks and arms.
#
#   tools/dense_arms.sh identity <new-bin> [<baseline-bin>]
#       all new switches unset vs the baseline (builds/share1/obrrt): power-on,
#       seed 1, 2 threads, OBRRT_MEM_MB=1000, 10 M frames, plain and with the
#       BASE settings; nodes.csv, events.jsonl, tree.bin must be identical.
#   tools/dense_arms.sh det <bin> <arm> [frames]
#       the arm at 2 vs 4 threads (20 M default): the three files identical.
#   tools/dense_arms.sh smoke <bin> <arm>... [-- seed frames threads]
#       short runs (20 M, 4 threads, OBRRT_MEM_MB=2000, OBRRT_SPARSE_CHECK=200).
#   tools/dense_arms.sh eval <bin> <arm>... [-- seed frames threads]
#       the evaluation (default 500 M, 16 threads), ONE ARM AT A TIME.
#   tools/dense_arms.sh list
#       the default arm matrix.
#
# Arm names: tokens joined by "-":
#   base        the current best (BASE + OBRRT_CHAIN=route OBRRT_SAMPLER=rate)
#   w|wa|lg|all|walg   OBRRT_ADD = winner | winner_all | last_gen | all | winner_all,last_gen
#   r<N>|roff   OBRRT_REPLAY_MAX = N | off (dense adds default to 150)
#   k<M>        OBRRT_KINSHIP_MULT = M
#   ns          OBRRT_NEAR_SEED=1     (nr<R> OBRRT_NEAR_R, np<P> OBRRT_NEAR_P)
#   g<N>        OBRRT_DENSE_GRID = N  (cap<N> OBRRT_DENSE_CAP = N)
#   zs          OBRRT_ZONE_SHARE=1
#   none0       OBRRT_DENSE_NE=0
# e.g. wa-k10-ns, all-g128-k30, w-r150 (sparse saves alone), base.
# Outputs runs/dn-<arm>-s<seed>, logs/dn-<arm>-s<seed>.log.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
C="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
BEST="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
arm_env() {
  local out="" t
  IFS=- read -ra toks <<< "$1"
  for t in "${toks[@]}"; do
    case $t in
      base) ;;
      w) out+=" OBRRT_ADD=winner";;
      wa) out+=" OBRRT_ADD=winner_all";;
      lg) out+=" OBRRT_ADD=last_gen";;
      all) out+=" OBRRT_ADD=all";;
      walg) out+=" OBRRT_ADD=winner_all,last_gen";;
      roff) out+=" OBRRT_REPLAY_MAX=off";;
      r[0-9]*) out+=" OBRRT_REPLAY_MAX=${t#r}";;
      k[0-9]*) out+=" OBRRT_KINSHIP_MULT=${t#k}";;
      ns) out+=" OBRRT_NEAR_SEED=1";;
      nr[0-9]*) out+=" OBRRT_NEAR_R=${t#nr}";;
      np[0-9]*) out+=" OBRRT_NEAR_P=0.${t#np}";;
      g[0-9]*) out+=" OBRRT_DENSE_GRID=${t#g}";;
      cap[0-9]*) out+=" OBRRT_DENSE_CAP=${t#cap}";;
      zs) out+=" OBRRT_ZONE_SHARE=1";;
      none0) out+=" OBRRT_DENSE_NE=0";;
      *) echo "unknown arm token $t" >&2; exit 1;;
    esac
  done
  echo "$out"
}
opts() { # [-- seed frames threads] after the arms
  seed=1; frames=$1; threads=$2; arms=()
  shift 2
  while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then seed=${2:-1}; frames=${3:-$frames}; threads=${4:-$threads}; break; fi
    arms+=("$1"); shift
  done
}
mode=${1:?identity, det, smoke, eval or list}; shift
case $mode in
  list)
    # Sparse saves alone; each ADD mode with kinship 1/3/10/30, with and
    # without nearby seeds; grid 128 for the two widest modes; zone share.
    echo base w-r150 ns
    for a in wa lg walg all; do
      for k in 1 3 10 30; do echo $a-k$k $a-k$k-ns; done
    done
    echo all-g128-k10 all-g128-k30 walg-g128-k10
    echo wa-k10-zs walg-k10-ns-zs
    ;;
  identity)
    NEW=${1:?new binary}; OLD=${2:-builds/share1/obrrt}
    ok=0
    for set in plain mw; do
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/dn-id-$set-$side
        rm -rf $out
        if [ $set = plain ]; then E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
        else E="$C $BASE OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"; fi
        env $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 60 > logs/dn-id-$set-$side.log 2>&1 &
      done
      wait
      for f in nodes.csv events.jsonl tree.bin; do
        if cmp -s runs/dn-id-$set-old/$f runs/dn-id-$set-new/$f; then echo "$set $f identical"; else echo "$set $f DIFFERS"; ok=1; fi
      done
      echo "$set nodes: $(($(wc -l < runs/dn-id-$set-new/nodes.csv) - 1))"
    done
    exit $ok;;
  det)
    BIN=${1:?binary}; ARM=${2:?arm}; F=${3:-20000000}
    X=$(arm_env $ARM) || exit 1
    for t in 2 4; do
      rm -rf runs/dn-det-$ARM-t$t
      env $C $BASE $BEST $X OBRRT_MEM_MB=2000 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM runs/dn-det-$ARM-t$t 1 120 > logs/dn-det-$ARM-t$t.log 2>&1
    done
    ok=0
    for f in nodes.csv events.jsonl tree.bin; do
      if cmp -s runs/dn-det-$ARM-t2/$f runs/dn-det-$ARM-t4/$f; then echo "$ARM $f identical"; else echo "$ARM $f DIFFERS"; ok=1; fi
    done
    exit $ok;;
  smoke|eval)
    BIN=${1:?binary}; shift
    if [ $mode = smoke ]; then opts 20000000 4 "$@"; EXTRA="OBRRT_MEM_MB=2000 OBRRT_SPARSE_CHECK=200"; MIN=120
    else opts 500000000 16 "$@"; EXTRA="OBRRT_MEM_MB=3000"; MIN=600; fi
    [ ${#arms[@]} -eq 0 ] && { echo "no arms" >&2; exit 1; }
    for arm in "${arms[@]}"; do
      X=$(arm_env $arm) || exit 1
      out=runs/dn-$arm-s$seed
      rm -rf $out
      env $C $BASE $BEST $X $EXTRA OBRRT_MAX_FRAMES=$frames RAYON_NUM_THREADS=$threads nice -n 10 $BIN $ROM $out $seed $MIN > logs/dn-$arm-s$seed.log 2>&1 &
      pid=$!; peak=0
      # Peak RSS: the process's own high-water mark, sampled until it exits.
      while kill -0 $pid 2>/dev/null; do
        h=$(awk '/VmHWM/ {print $2}' /proc/$pid/status 2>/dev/null); [ -n "$h" ] && [ "$h" -gt "$peak" ] && peak=$h
        sleep 2
      done
      wait $pid
      echo "peak_rss_kb $peak" >> logs/dn-$arm-s$seed.log
      echo "$arm: nodes $(($(wc -l < $out/nodes.csv) - 1)), peak $((peak / 1024)) MB; $(grep '^best times' logs/dn-$arm-s$seed.log | tail -1)"
    done;;
esac
