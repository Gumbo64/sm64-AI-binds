#!/bin/bash
# Determinism check with OBRRT_FASTEXIT=1 OBRRT_GA_CBUTTONS=0.03 on the base env:
# 2 vs 4 threads, seed 1, 20 M frames, run one after the other.
#   tools/fastexit_det.sh [bin]
cd "$(dirname "$0")/.."
BIN=${1:-builds/fastexit3/obrrt}
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1"
for t in 2 4; do
  rm -rf runs/fx-det-t$t
  env $BASE OBRRT_FASTEXIT=1 OBRRT_GA_CBUTTONS=0.03 OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=20000000 RAYON_NUM_THREADS=$t nice -n 10 $BIN ../baserom.us.z64 runs/fx-det-t$t 1 600 > logs/fx-det-t$t.log 2>&1
done
for f in nodes.csv events.jsonl tree.bin; do cmp -s runs/fx-det-t2/$f runs/fx-det-t4/$f && echo "det $f identical" || echo "det $f DIFFER"; done
