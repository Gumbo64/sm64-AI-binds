#!/bin/bash
# A small fixed benchmark of zone trees from the campaign's own states: how many search frames to a new star.
#   tools/seg120/bench.sh ARM THREADS CAP "VAR=value ..." [seeds, default "1 2"]
# Zones: BoB with only star 6 (route-0015), WF with stars 1, 3, 6 (route-0013), JRB with star 3 (route-0021).
# Each run stops at its first exit with a new star (or at CAP frames). Results: logs/seg120/bench-ARM.txt.
cd "$(dirname "$0")/../.."
arm=$1; threads=$2; cap=$3; extra=$4; seeds=${5:-1 2}
echo $$ > logs/seg120/bench-$arm.pid
for seed in $seeds; do for z in bob:0015 wf:0013 jrb:0021; do
  name=bench-$arm-${z%%:*}-s$seed
  if [ ! -s runs/$name/summary.json ]; then
    SEED=$seed GRACE=1 BIN=${BIN:-builds/v2u/obrrt2} tools/seg120/zone_test.sh $name logs/campaign/t1/route-${z##*:}.m64 $cap $threads $extra
    python3 tools/seg120/zone_report.py runs/$name 0 | sed -n '1p;2p;4p' | tr '\n' ' ' >> logs/seg120/bench-$arm.txt; echo >> logs/seg120/bench-$arm.txt
  fi
done; done
