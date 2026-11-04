#!/bin/bash
# Queue 7 (2026-11-01 night): scattershot v2 (oss2) screens, 100 M, zero_star,
# seeds 1-2. GA share (by expansions or frames), harvest, bin coarseness.
# Waits for q6.
cd "$(dirname "$0")/../.."
while kill -0 "$(cat logs/q6.pid)" 2>/dev/null; do sleep 30; done
oss2/tools/screen.sh ../builds/oss2a/oss2 100000000 schemas/zero_star.json q7 \
  ga0="OSS_GA=0" \
  gaf25="OSS_GA=0.25 OSS_GA_BY=frames" \
  gaf50="OSS_GA=0.5 OSS_GA_BY=frames" \
  ga1="OSS_GA=1" \
  ga1ends="OSS_GA=1 OSS_GA_HARVEST=ends" \
  ga1g1="OSS_GA=1 OSS_GRID_ADD=1" \
  ga1g2="OSS_GA=1 OSS_GRID_ADD=2" \
  gaf50g2="OSS_GA=0.5 OSS_GA_BY=frames OSS_GRID_ADD=2" \
  -- 1 2
echo "q7 done"
