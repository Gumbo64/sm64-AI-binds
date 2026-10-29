#!/bin/bash
# Identity for the mode / technique policies (objectives/src/moves:
# cannon.rs, water_air.rs, slide.rs, pole.rs, tech.rs) with
# OBRRT_MOVES_MODES unset: power-on, seed 1, 2 threads, OBRRT_MEM_MB=1000,
# 10 M frames, plain and the base env; nodes.csv / events.jsonl / tree.bin
# of <new> vs <old> must match. EXTRA="VAR=1 ..." adds to both (e.g. the
# library and the GA: OBRRT_MOVES=1 OBRRT_MOVE_GA=1).
#   tools/modes_identity.sh <new-bin> <old-bin> [tag]
set -u
cd "$(dirname "$0")/.."
ROM=../baserom.us.z64
NEW=${1:?new}; OLD=${2:?old}; TAG=${3:-id}
MW="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
ok=0
for set in plain mw; do
  for side in old new; do
    bin=$OLD; [ $side = new ] && bin=$NEW
    out=runs/md-$TAG-$set-$side; rm -rf $out
    E="OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000 ${EXTRA:-}"; [ $set = mw ] && E="$MW $E"
    env -u OBRRT_MOVES_MODES $E RAYON_NUM_THREADS=2 nice -n 10 $bin $ROM $out 1 60 > logs/md-$TAG-$set-$side.log 2>&1 &
  done
  wait
  for f in nodes.csv events.jsonl tree.bin; do
    if cmp -s runs/md-$TAG-$set-old/$f runs/md-$TAG-$set-new/$f; then echo "$set $f identical"; else echo "$set $f DIFFERS"; ok=1; fi
  done
  echo "$set nodes: $(($(wc -l < runs/md-$TAG-$set-new/nodes.csv) - 1))"
done
exit $ok
