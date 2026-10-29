#!/bin/bash
# Lane B (8 threads), restarted 2026-10-29: schema pick, kitchen sink, replay budget, hierarchical rate, at 500 M x 3 seeds.
cd "$(dirname "$0")/../.."
export THREADS=8
KS="OBRRT_PICK=schema OBRRT_ADD=winner_all,last_gen OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1"
for s in 1 2 3; do
  tools/arm_eval.sh builds/pick2/obrrt 500000000 pk5 ps-walg="OBRRT_PICK=schema OBRRT_ADD=winner_all,last_gen" ps="OBRRT_PICK=schema" -- $s >> logs/pick_eval1.txt 2>&1
  tools/arm_eval.sh builds/ks1/obrrt 500000000 ks ks="$KS" ksr="$KS OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json" -- $s >> logs/ks_eval1.txt 2>&1
  tools/arm_eval.sh builds/replay2/obrrt 500000000 rb rbz="OBRRT_MOVES=1 OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1 OBRRT_REPLAY_MAX=150" rbz5="OBRRT_MOVES=1 OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.05 OBRRT_REPLAY_MAX=150" -- $s >> logs/rb_eval1.txt 2>&1
  tools/arm_eval.sh builds/rate3/obrrt 500000000 rt hier="OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=20000" hiers="OBRRT_RATE_MODE=hier OBRRT_RATE_PRIOR=20000 OBRRT_RATE_NOSTART_SKIP=1" -- $s >> logs/rate_eval1.txt 2>&1
done
