#!/bin/bash
# Queue 4 (2026-11-01): combine q2's winners on v2d (parity fixes), base P + rr.
# 100 M, seeds 1-2. Waits for q3.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q3.pid)" 2>/dev/null; do sleep 30; done
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
RR="OBRRT_REROUTE=1 OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1"
tools/screen.sh builds/v2d/obrrt2 100000000 q4 \
  rr="$P $RR" \
  rrpol="$P $RR OBRRT_POLISH=0.1" \
  rrpoltail="$P $RR OBRRT_POLISH=0.1 OBRRT_TAIL=1" \
  rrpol2s="$P $RR OBRRT_POLISH=0.1 OBRRT_TWOSTAR=1" \
  rrpolhier="$P $RR OBRRT_POLISH=0.1 OBRRT_RATE_MODE=hier" \
  -- 1 2
echo "q4 done"
