#!/bin/bash
# routes9 (2026-11-02): VCutM star AND vanish cap switch, from before the 8th red coin, so the GA can
# press the switch first. builds/legs7. tools/queues/routes9.sh ROUTE SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-$1-s$2; rm -rf "$O"; mkdir -p "$O"
env OBRRT_GA_KEEP_SEED=1 OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 RAYON_NUM_THREADS=1 nice -n 5 builds/legs7/legs tools/spec/routes/$1.json --out "$O" --beam 8 --seed "$2" > "$O/legs.log" 2>&1
echo "$1 s$2 exit $?: $(grep -E '^leg [0-9]+ ' "$O/legs.log" | cut -c1-110)"
