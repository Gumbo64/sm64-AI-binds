#!/bin/bash
# routes4 (2026-11-02): full route, Hill skip through the door + the King fight
# split per hit (dialogs on hit legs are not Done). legs4 (new movement). tools/queues/routes4.sh SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-fast-king-hits-m-s$1; rm -rf "$O"; mkdir -p "$O"
env OBRRT_LAKITU_THROUGH_DOOR=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=4 nice -n 5 builds/legs4/legs tools/spec/routes/fast-king-hits.json --out "$O" --beam 4 --seed "$1" > "$O/legs.log" 2>&1 < /dev/null
echo "seed $1 exit $?: $(grep -E '^leg [0-9]+ .*FAILED' "$O/legs.log" | cut -c1-40) $(grep '^total' "$O/legs.log" | cut -c1-30)"
