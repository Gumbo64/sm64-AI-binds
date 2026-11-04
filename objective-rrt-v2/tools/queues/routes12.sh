#!/bin/bash
# routes12 (2026-11-02): fast-king-superjump-c: the TAS's speed chain, a superjump above BoB's slippery
# east slope, then the walked upper path and the King. builds/legs8. tools/queues/routes12.sh SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-fast-king-superjump-c-s$1; rm -rf "$O"; mkdir -p "$O"
env OBRRT_LAKITU_THROUGH_DOOR=1 OBRRT_GA_KEEP_SEED=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=4 nice -n 5 builds/legs8/legs tools/spec/routes/fast-king-superjump-c.json --out "$O" --beam 4 --seed "$1" > "$O/legs.log" 2>&1 < /dev/null
echo "seed $1 exit $?: $(grep -E '^leg [0-9]+ .*FAILED' "$O/legs.log" | cut -c1-40) $(grep '^total' "$O/legs.log" | cut -c1-30)"
