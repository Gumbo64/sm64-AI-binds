#!/bin/bash
# Graft screen: one smoke per arm, same seed and frames, then a summary.
#   tools/graft_screen.sh <bin> <frames> <prefix> <name>="ENV ..." ... [-- seed]
# Common env: the base env + OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1
# + reroute schema + OBRRT_REPLAY_BUDGET=0.1 OBRRT_REPLAY_MAX=150.
# THREADS (4), PAR (arms run at once, 3), OBRRT_MEM_MB (1500).
cd "$(dirname "$0")/.."
BIN=$1; FRAMES=$2; PREFIX=$3; shift 3
names=(); envs=(); seed=1
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then seed=$2; break; fi
  names+=("${1%%=*}"); envs+=("${1#*=}"); shift
done
C="OBRRT_SOFT_DEDUP=1 OBRRT_MEM_MB=${OBRRT_MEM_MB:-1500} OBRRT_ROUND=64 OBRRT_MAX_FRAMES=$FRAMES"
BASE="OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
X="OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1 OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json OBRRT_REPLAY_BUDGET=0.1 OBRRT_REPLAY_MAX=150"
run() {
  local out=runs/$PREFIX-$1-s$seed
  rm -rf "$out"
  env $C $BASE $X $2 RAYON_NUM_THREADS=${THREADS:-4} nice -n 10 "$BIN" ../baserom.us.z64 "$out" "$seed" 180 > logs/$PREFIX-$1-s$seed.log 2>&1
  echo "$1 done ($?)"
}
for i in "${!names[@]}"; do
  run "${names[$i]}" "${envs[$i]}" &
  while [ "$(jobs -rp | wc -l)" -ge "${PAR:-3}" ]; do wait -n; done
done
wait
python3 tools/analysis/graft_report.py $(for n in "${names[@]}"; do echo runs/$PREFIX-$n-s$seed; done)
