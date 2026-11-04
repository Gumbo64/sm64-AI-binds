#!/bin/bash
# routes14 (2026-11-03): fast-king-speed-blj from scratch on the new chain (closing-speed ETA, GA's stick), one-off beside the king loop.
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-speed-blj-fresh-s$1; rm -rf "$O"; mkdir -p "$O"
env OBRRT_LAKITU_THROUGH_DOOR=1 OBRRT_GA_KEEP_SEED=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 OBRRT_KING_LANDING=1 RAYON_NUM_THREADS=4 nice -n 5 builds/legs13/legs tools/spec/routes/fast-king-speed-blj.json --out "$O" --beam 4 --seed "$1" > "$O/legs.log" 2>&1 < /dev/null
