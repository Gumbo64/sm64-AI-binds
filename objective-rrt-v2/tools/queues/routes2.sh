#!/bin/bash
# routes2 (2026-11-02, after the machine restart): window comparison on builds/legs2.
# tools/queues/routes2.sh LANE: A = merged walked route, B = snapped Chomp route.
# 8 threads per lane, QD + 4 gamma, REGRET 60, KING_LANDING, seed 1.
cd "$(dirname "$0")/../.."
E="OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=8"
run() { O=oss2/runs/$1; rm -rf "$O"; mkdir -p "$O"; env $E nice -n 5 builds/legs2/legs tools/spec/routes/$2 --out "$O" --beam 4 --seed 1 "${@:3}" > "$O/legs.log" 2>&1 < /dev/null; echo "$1 exit $?: $(grep '^total' "$O/legs.log" | cut -c1-40)"; }
case "$1" in
A)
  run legs-fast-king-walk-merged-r2-w1-s1 fast-king-walk-merged.json --window 1
  run legs-fast-king-walk-merged-r2-w3-s1 fast-king-walk-merged.json --window 3
  run legs-fast-king-walk-merged-r2-wf600-s1 fast-king-walk-merged.json --window-frames 600
  ;;
B)
  run legs-bob-chomp-r2-w1-s1 bob-chomp.snapped.json --window 1
  run legs-bob-chomp-r2-w3-s1 bob-chomp.snapped.json --window 3
  run legs-bob-chomp-r2-wf600-s1 bob-chomp.snapped.json --window-frames 600
  ;;
esac
echo "routes2 $1 done"
