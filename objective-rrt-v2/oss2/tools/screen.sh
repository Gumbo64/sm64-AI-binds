#!/bin/bash
# Screen oss2 arms: tools/screen.sh BIN FRAMES SCHEMA PREFIX name="ENV ..." ... [-- seeds]
# Two seeds at a time (8 threads each), arm after arm; runs in runs/<PREFIX>-<name>-s<seed>,
# logs next to them; tools/summ.py's summary per arm on stdout.
set -u
cd "$(dirname "$0")/.."
BIN=${1:?binary}; FRAMES=${2:?frames}; SCHEMA=${3:?schema}; PREFIX=${4:?prefix}; shift 4
names=(); envs=(); seeds=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; seeds=("$@"); break; fi
  names+=("${1%%=*}"); envs+=("${1#*=}"); shift
done
[ ${#seeds[@]} -eq 0 ] && seeds=(1 2)
# The v2 GA's settings, as the obrrt2 screens' best config has them.
GA="OBRRT_TECH=all OBRRT_GP_CATCH=1 OBRRT_SWIM_HOLD=1 OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1"
for i in "${!names[@]}"; do
  runs=()
  for ((j = 0; j < ${#seeds[@]}; j += 2)); do
    for seed in "${seeds[@]:j:2}"; do
      r=$PREFIX-${names[$i]}-s$seed
      rm -rf runs/$r
      env $GA ${envs[$i]} RAYON_NUM_THREADS=8 nice -n 5 "$BIN" "$SCHEMA" 600 "$seed" --frames "$FRAMES" --out runs/$r > runs/$r.log 2>&1 &
      runs+=("$r")
    done
    wait
  done
  echo "== ${names[$i]}: ${envs[$i]}"
  python3 tools/summ.py . "${runs[@]}"
done
