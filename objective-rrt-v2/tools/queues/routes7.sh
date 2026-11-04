#!/bin/bash
# routes7 (2026-11-02): beat the 2026 TAS's first star (VCutM, frame 2569) from its own
# frame 1646. Huge budget: pop 256, gens 150, beam 8. legs4. tools/queues/routes7.sh SEED
cd "$(dirname "$0")/../.."
O=oss2/runs/legs-tas-first-star-s$1; rm -rf "$O"; mkdir -p "$O"
env OBRRT_GA_QD=1 OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999 OBRRT_GA_REGRET=60 RAYON_NUM_THREADS=8 nice -n 5 builds/legs4/legs tools/spec/routes/tas-first-star.json --out "$O" --beam 8 --seed "$1" > "$O/legs.log" 2>&1 < /dev/null
echo "s$1 exit $?: $(grep -E '^leg [0-9]+ .*FAILED' "$O/legs.log" | cut -c1-50) $(grep '^total' "$O/legs.log" | cut -c1-30)"
