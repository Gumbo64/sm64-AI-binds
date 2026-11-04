#!/bin/bash
# Queue 2 (2026-11-01): v2b (all objectives ported, heart fix) on the v1 best
# config (mgel-lg-pf), then each v1 feature that config leaves off, added one
# at a time. 100 M, seeds 1-2. Waits for ladder1 (its PID file).
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/ladder1.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
tools/screen.sh builds/v2b/obrrt2 100000000 q2 \
  lgpf="$P" \
  pol="$P OBRRT_POLISH=0.1" \
  rr="$P OBRRT_REROUTE=1 OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1" \
  hier="$P OBRRT_RATE_MODE=hier" \
  route="$P OBRRT_RATE_ROUTE=3" \
  star="$P OBRRT_STAR=1" \
  arr="$P OBRRT_ARRIVAL=1" \
  tail="$P OBRRT_TAIL=1" \
  twostar="$P OBRRT_TWOSTAR=1" \
  -- 1 2
echo "q2 done"
