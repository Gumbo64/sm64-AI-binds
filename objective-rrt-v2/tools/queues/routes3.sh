#!/bin/bash
# routes3 (2026-11-02): lane B = snapped Chomp with ChompPost's probe off, windows 1/3/wf600;
# lane C = BLJ chain (unsnapped Ramp, probe off) seeds 2, 3. builds/legs2.
cd "$(dirname "$0")/../.."
T=${T:-4}
E="OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=$T"
run() { O=oss2/runs/$1; rm -rf "$O"; mkdir -p "$O"; env $E nice -n 5 builds/legs2/legs tools/spec/routes/$2 --out "$O" --beam 4 --seed "$3" "${@:4}" > "$O/legs.log" 2>&1 < /dev/null; echo "$1 exit $?: $(grep -E '^leg [0-9]+ .*FAILED' "$O/legs.log" | cut -c1-40) $(grep '^total' "$O/legs.log" | cut -c1-30)"; }
case "$1" in
B) run legs-bob-chomp-r3-w1-s1 bob-chomp.snapped2.json 1 --window 1
   run legs-bob-chomp-r3-w3-s1 bob-chomp.snapped2.json 1 --window 3
   run legs-bob-chomp-r3-wf600-s1 bob-chomp.snapped2.json 1 --window-frames 600 ;;
C) run legs-fast-king-blj-tas-r3-s2 fast-king-blj-tas.ramp0.json 2 --window 1
   run legs-fast-king-blj-tas-r3-s3 fast-king-blj-tas.ramp0.json 3 --window 1 ;;
esac
echo "routes3 $1 done"
