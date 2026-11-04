#!/bin/bash
# Identity check for OBRRT_FASTEXIT / OBRRT_GA_CBUTTONS (unset): new vs baseline binary,
# seed 1, 2 threads, 10 M frames, plain + base env; nodes.csv/events.jsonl/tree.bin compared.
#   tools/fastexit_identity.sh [new-bin] [old-bin]
cd /workspaces/sm64-combined/sm64-AI-binds/objective-rrt
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
for m in plain base; do
  E=""; [ $m = base ] && E="$BASE"
  for side in old new; do
    bin=${2:-builds/rate2/obrrt}; [ $side = new ] && bin=${1:-builds/fastexit1/obrrt}
    out=runs/fx-id-$m-$side; rm -rf $out
    env $E OBRRT_MEM_MB=1000 OBRRT_MAX_FRAMES=10000000 RAYON_NUM_THREADS=2 nice -n 10 $bin ../baserom.us.z64 $out 1 600 > logs/fx-id-$m-$side.log 2>&1 &
  done
  wait
  for f in nodes.csv events.jsonl tree.bin; do cmp -s runs/fx-id-$m-old/$f runs/fx-id-$m-new/$f && echo "$m $f identical" || echo "$m $f DIFFER"; done
done
