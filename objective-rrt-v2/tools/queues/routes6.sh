#!/bin/bash
# routes6 (2026-11-02): superjump routes on legs4. tools/queues/routes6.sh ROUTE SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-$1-s$2; rm -rf "$O"; mkdir -p "$O"
env OBRRT_LAKITU_THROUGH_DOOR=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=4 nice -n 5 builds/legs4/legs tools/spec/routes/$1.json --out "$O" --beam 4 --seed "$2" > "$O/legs.log" 2>&1 < /dev/null
echo "$1 s$2 exit $?: $(grep -E '^leg [0-9]+ .*FAILED' "$O/legs.log" | cut -c1-50) $(grep '^total' "$O/legs.log" | cut -c1-30)"
