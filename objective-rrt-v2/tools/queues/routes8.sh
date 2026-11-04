#!/bin/bash
# routes8 (2026-11-02): only the last segment of the 2026 TAS's first star (frames 2390-2569),
# reference kept in the population. builds/legs5. tools/queues/routes8.sh SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-tas-last-s$1; rm -rf "$O"; mkdir -p "$O"
env OBRRT_GA_KEEP_SEED=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 RAYON_NUM_THREADS=1 nice -n 5 builds/legs5/legs tools/spec/routes/tas-first-star-last.json --out "$O" --beam 8 --seed "$1" > "$O/legs.log" 2>&1
echo "s$1 exit $?: $(grep -E '^leg [0-9]+ ' "$O/legs.log" | cut -c1-110)"
