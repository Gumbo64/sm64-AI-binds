#!/bin/bash
# Start selection by schema cells (OBRRT_PICK=schema|schema_kin, schemapick.rs):
# checks and arms. Same layout as tools/dense_arms.sh, outputs prefixed pk-.
#
#   tools/pick_arms.sh identity <new-bin> [<baseline-bin>]
#       OBRRT_PICK unset vs the baseline (builds/dense1/obrrt): power-on, seed
#       1, 2 threads, OBRRT_MEM_MB=1000, 10 M frames, plain and with BASE;
#       nodes.csv, events.jsonl, tree.bin must be identical.
#   tools/pick_arms.sh det <bin> <arm> [frames]
#       the arm at 2 vs 4 threads (20 M default): the three files identical.
#   tools/pick_arms.sh smoke <bin> <arm>... [-- seed frames threads]
#       short runs (20 M, 4 threads, OBRRT_MEM_MB=1500).
#   tools/pick_arms.sh eval <bin> <arm>... [-- seed frames threads]
#       the evaluation (default 500 M, 16 threads), ONE ARM AT A TIME.
#   tools/pick_arms.sh list
#
# Arm names: tokens joined by "-" (BASE + OBRRT_CHAIN=route OBRRT_SAMPLER=rate
# always):
#   base        kinship softmax (OBRRT_PICK unset)
#   ps          OBRRT_PICK=schema       (no kinship)
#   psk         OBRRT_PICK=schema_kin   (schema cells x kinship)
#   sc<name>    OBRRT_PICK_SCHEMA=../objective-scattershot/schemas/<name>.json
#   pr<P>       OBRRT_PICK_RAND=0.<P>   (random member share, default 0.1)
#   pkw<W>      OBRRT_PICK_KIN_W=<W>    (schema_kin kin-cell multiplier, 4)
#   w|wa|lg|all|walg|r<N>|roff|k<M>|ns|g<N>  as in tools/dense_arms.sh
# e.g. ps, psk, ps-wa, psk-walg, psk-walg-k10, ps-scstar120.
# Outputs runs/pk-<arm>-s<seed>, logs/pk-<arm>-s<seed>.log.
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
      ps) out+=" OBRRT_PICK=schema";;
      psk) out+=" OBRRT_PICK=schema_kin";;
      sc*) out+=" OBRRT_PICK_SCHEMA=../objective-scattershot/schemas/${t#sc}.json";;
      pr[0-9]*) out+=" OBRRT_PICK_RAND=0.${t#pr}";;
      pkw[0-9]*) out+=" OBRRT_PICK_KIN_W=${t#pkw}";;
      # As in tools/dense_arms.sh:
      w) out+=" OBRRT_ADD=winner";;
      wa) out+=" OBRRT_ADD=winner_all";;
      lg) out+=" OBRRT_ADD=last_gen";;
      all) out+=" OBRRT_ADD=all";;
      walg) out+=" OBRRT_ADD=winner_all,last_gen";;
      roff) out+=" OBRRT_REPLAY_MAX=off";;
      r[0-9]*) out+=" OBRRT_REPLAY_MAX=${t#r}";;
      k[0-9]*) out+=" OBRRT_KINSHIP_MULT=${t#k}";;
      ns) out+=" OBRRT_NEAR_SEED=1";;
      g[0-9]*) out+=" OBRRT_DENSE_GRID=${t#g}";;
      *) echo "unknown arm token $t" >&2; return 1;;
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
    echo base ps psk ps-wa psk-wa ps-walg psk-walg psk-walg-k10 ps-scstar120
    ;;
  identity)
    NEW=${1:?new binary}; OLD=${2:-builds/dense1/obrrt}
    ok=0
    for set in plain mw; do
      for side in old new; do
        bin=$OLD; [ $side = new ] && bin=$NEW
        out=runs/pk-id-$set-$side
        rm -rf $out
        if [ $set = plain ]; then E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"
        else E="$C $BASE OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000"; fi
        env $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 60 > logs/pk-id-$set-$side.log 2>&1 &
      done
      wait
      for f in nodes.csv events.jsonl tree.bin; do
        if cmp -s runs/pk-id-$set-old/$f runs/pk-id-$set-new/$f; then echo "$set $f identical"; else echo "$set $f DIFFERS"; ok=1; fi
      done
      echo "$set nodes: $(($(wc -l < runs/pk-id-$set-new/nodes.csv) - 1))"
    done
    exit $ok;;
  det)
    BIN=${1:?binary}; ARM=${2:?arm}; F=${3:-20000000}
    X=$(arm_env $ARM) || exit 1
    for t in 2 4; do
      rm -rf runs/pk-det-$ARM-t$t
      env $C $BASE $BEST $X OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$F RAYON_NUM_THREADS=$t nice -n 10 $BIN $ROM runs/pk-det-$ARM-t$t 1 120 > logs/pk-det-$ARM-t$t.log 2>&1
    done
    ok=0
    for f in nodes.csv events.jsonl tree.bin; do
      if cmp -s runs/pk-det-$ARM-t2/$f runs/pk-det-$ARM-t4/$f; then echo "$ARM $f identical"; else echo "$ARM $f DIFFERS"; ok=1; fi
    done
    exit $ok;;
  smoke|eval)
    BIN=${1:?binary}; shift
    if [ $mode = smoke ]; then opts 20000000 4 "$@"; EXTRA="OBRRT_MEM_MB=1500"; MIN=120
    else opts 500000000 16 "$@"; EXTRA="OBRRT_MEM_MB=3000"; MIN=600; fi
    [ ${#arms[@]} -eq 0 ] && { echo "no arms" >&2; exit 1; }
    for arm in "${arms[@]}"; do
      X=$(arm_env $arm) || exit 1
      out=runs/pk-$arm-s$seed
      rm -rf $out
      env $C $BASE $BEST $X $EXTRA OBRRT_MAX_FRAMES=$frames RAYON_NUM_THREADS=$threads nice -n 10 $BIN $ROM $out $seed $MIN > logs/pk-$arm-s$seed.log 2>&1 &
      pid=$!; peak=0
      while kill -0 $pid 2>/dev/null; do
        h=$(awk '/VmHWM/ {print $2}' /proc/$pid/status 2>/dev/null); [ -n "$h" ] && [ "$h" -gt "$peak" ] && peak=$h
        sleep 2
      done
      wait $pid
      echo "peak_rss_kb $peak" >> logs/pk-$arm-s$seed.log
      echo "$arm: nodes $(($(wc -l < $out/nodes.csv) - 1)), peak $((peak / 1024)) MB; $(grep '^best times' logs/pk-$arm-s$seed.log | tail -1)"
    done;;
esac
