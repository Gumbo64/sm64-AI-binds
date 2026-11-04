#!/bin/bash
# 50 M smokes, 4 threads, seed 1, base env + route chains + rate sampler + moves:
# ctl (switches unset), fx (OBRRT_FASTEXIT=1), cb (OBRRT_GA_CBUTTONS=0.03). One after another.
#   tools/fastexit_smoke.sh [bin] [arms...]
cd "$(dirname "$0")/.."
BIN=${1:-builds/fastexit3/obrrt}; shift
ARMS=${*:-ctl fx cb}
BASE="OBRRT_SOFT_DEDUP=1 OBRRT_ROUND=64 OBRRT_TECH=all OBRRT_NEW_ELIGIBLE=1 OBRRT_STARSET_KEY=1 OBRRT_PICK_BY_B=0 OBRRT_SWIM_HOLD=1 OBRRT_GP_CATCH=1 OBRRT_KEEP_WIP=1 OBRRT_MOVE_WEIGHT=0.75 OBRRT_TAS_BOOT=1 OBRRT_CHAIN=route OBRRT_SAMPLER=rate OBRRT_MOVES=1"
for a in $ARMS; do
  case $a in ctl) X="";; fx) X="OBRRT_FASTEXIT=1";; cb) X="OBRRT_GA_CBUTTONS=0.03";; *) echo "arm $a?"; exit 2;; esac
  out=runs/fx-smoke-$a; rm -rf $out
  env $BASE $X OBRRT_MEM_MB=1500 OBRRT_MAX_FRAMES=50000000 RAYON_NUM_THREADS=4 nice -n 10 $BIN ../baserom.us.z64 $out 1 600 > logs/fx-smoke-$a.log 2>&1
  echo "$a: $(grep 'best times' logs/fx-smoke-$a.log | tail -1)"
done
