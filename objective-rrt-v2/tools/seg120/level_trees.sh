#!/bin/bash
# Per-zone tree searches from level-entry funnels (120-star plan): tools/seg120/level_trees.sh PREFIX FRAMES PAR THREADS n...
# n = visit numbers of logs/seg120/visits.csv. Output runs/PREFIX-<n>-<level>-act<a>, logs/seg120/PREFIX-*.log.
# Config P on builds/v2l. The start movie only reaches the entry funnel; everything after it is the search's own.
cd "$(dirname "$0")/../.."
PREFIX=$1; FRAMES=$2; PAR=$3; THREADS=$4; shift 4
BIN=${BIN:-builds/v2l/obrrt2}
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=3000 OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES OBRRT_BOOT_NOWAIT=1"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75"
CD="OBRRT_CHAIN=route OBRRT_SAMPLER=rate"
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
run() {
  start=$(ls logs/seg120/starts/$(printf %03d "$1")-*.m64); name=$(basename "$start" .m64); out=runs/$PREFIX-$name
  # Resumable: a run whose summary already reaches the budget is kept.
  if [ -f "$out/summary.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$out/summary.json'))['frames']>=$FRAMES else 1)"; then echo "$PREFIX-$name: kept"; return; fi
  rm -rf "$out"
  env $C $BASE $CD $P $EXTRA RAYON_NUM_THREADS=$THREADS nice -n 5 "$BIN" ../baserom.us.z64 "$out" 1 600 "$start" > "logs/seg120/$PREFIX-$name.log" 2>&1
  echo "$PREFIX-$name: exit $? $(grep -c panicked logs/seg120/$PREFIX-$name.log) panics"
}
export -f run; export PREFIX BIN C BASE CD P EXTRA THREADS FRAMES
printf '%s\n' "$@" | xargs -P "$PAR" -I{} bash -c 'run {}'
echo "$PREFIX done"
