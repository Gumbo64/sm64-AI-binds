#!/bin/bash
# Queue 6 (2026-11-01 night): 500 M x 3 seeds confirmations on v2e against v1's
# pf-mgel-lg-pf (6, 7, 5 stars). Arms: v1 features only (P + rr + polish +
# tail), + speed switches (FASTER_ONLY, BNB), and v2 P alone as the baseline.
cd "$(dirname "$0")/../.."
P="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_PAINTING_FAST=1 OBRRT_ADD=last_gen OBRRT_KINSHIP_MULT=10"
RR="OBRRT_REROUTE=1 OBRRT_ZONE_SHARE=1 OBRRT_REPLAY_BUDGET=0.1"
tools/screen.sh builds/v2e/obrrt2 500000000 q6 \
  rrpoltail="$P $RR OBRRT_POLISH=0.1 OBRRT_TAIL=1" \
  rrfobnb="$P $RR OBRRT_POLISH=0.1 OBRRT_FASTER_ONLY=1 OBRRT_BNB=1" \
  p="$P" \
  -- 1 2 3
echo "q6 done"
