#!/bin/bash
# Lane A5 (16 threads), queued 2026-10-29 behind lane_a4 (which runs pf then mx).
# Combined arms from ref/next-configs.md, on builds/combo1 (all features, known good).
cd "$(dirname "$0")/../.."
while kill -0 7018 2>/dev/null; do sleep 120; done
P='OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1'
tools/arm_eval.sh builds/combo1/obrrt 500000000 cc \
  combo-schema-dense="$P OBRRT_PICK=schema OBRRT_ADD=winner_all,last_gen" \
  combo-lg-att="$P OBRRT_ADD=last_gen OBRRT_KIN_UNIT=attempt OBRRT_KINSHIP_MULT=2" \
  combo-replay="$P OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1 OBRRT_REPLAY_MAX=150" \
  -- 1 2 3 >> logs/cc_eval1.txt 2>&1
echo "lane_a5 done"
