#!/bin/bash
# Identity check for sample_entry's per-kind tables (pool.rs SampleIdx):
# a new binary against a baseline, seed 1, 2 threads each, nodes.csv,
# events.jsonl and tree.bin byte-identical.
#
#   tools/sample_identity.sh <new-bin> <baseline-bin> [frames] [mode...]
#       modes (default all): plain base rate hier dense
#       (base = the BASE env below, rate = base + CHAIN=route SAMPLER=rate,
#       hier = rate + RATE_MODE=hier RATE_PRIOR=20000 RATE_NOSTART_SKIP=1,
#       dense = rate + ADD=winner_all,last_gen KINSHIP_MULT=10).
#       Default 30 M frames. Pairs run one after another (4 threads).
#       Set CHECK=1 to also run the new side with OBRRT_SAMPLE_CHECK=1
#       (asserts each pick against the full scan).
# Outputs runs/sx-id-<mode>-{old,new}, logs/sx-id-*.log; deletes the runs
# when identical unless KEEP=1.
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
NEW=${1:?new binary}; OLD=${2:?baseline binary}; FRAMES=${3:-30000000}
shift 3 2>/dev/null || shift $#
MODES=${*:-plain base rate hier dense}
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
RATE="$BASE OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
ok=0
for m in $MODES; do
  case $m in
    plain) E="";;
    base) E="$BASE";;
    rate) E="$RATE";;
    hier) E="$RATE OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=20000 OBRRT_RATE_NOSTART_SKIP=1";;
    dense) E="$RATE OBRRT_ADD=winner_all,last_gen OBRRT_KINSHIP_MULT=10";;
    *) echo "unknown mode $m" >&2; exit 2;;
  esac
  for side in old new; do
    bin=$OLD; X=""
    [ $side = new ] && bin=$NEW && [ "${CHECK:-0}" = 1 ] && X="OBRRT_SAMPLE_CHECK=1"
    out=runs/sx-id${TAG:-}-$m-$side
    rm -rf $out
    env $E $X OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=$FRAMES RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 600 > logs/sx-id${TAG:-}-$m-$side.log 2>&1 &
  done
  wait
  same=1
  for f in nodes.csv events.jsonl tree.bin; do
    if cmp -s runs/sx-id${TAG:-}-$m-old/$f runs/sx-id${TAG:-}-$m-new/$f; then echo "$m $f identical"; else echo "$m $f DIFFERS"; same=0; ok=1; fi
  done
  echo "$m nodes: $(($(wc -l < runs/sx-id${TAG:-}-$m-new/nodes.csv) - 1)); $(grep -o 'picks [0-9.]*s started [0-9.]*s no start (sampling [0-9.]*s)' logs/sx-id${TAG:-}-$m-old.log | tail -1) -> $(grep -o 'picks [0-9.]*s started [0-9.]*s no start (sampling [0-9.]*s)' logs/sx-id${TAG:-}-$m-new.log | tail -1)"
  [ $same = 1 ] && [ "${KEEP:-0}" != 1 ] && rm -rf runs/sx-id${TAG:-}-$m-old runs/sx-id${TAG:-}-$m-new
done
exit $ok
