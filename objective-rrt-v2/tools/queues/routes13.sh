#!/bin/bash
# routes13 (2026-11-03): fast-king-no-climb, one-off alongside the king loop. tools/queues/routes13.sh SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-fast-king-no-climb-s$1; rm -rf "$O"; mkdir -p "$O"
env OBRRT_LAKITU_THROUGH_DOOR=1 OBRRT_GA_KEEP_SEED=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=3 nice -n 5 builds/legs11/legs tools/spec/routes/fast-king-no-climb.json --out "$O" --beam 4 --seed "$1" > "$O/legs.log" 2>&1 < /dev/null
echo "seed $1 exit $?"
