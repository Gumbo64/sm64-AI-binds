#!/bin/bash
# Lane A (16 threads), restarted 2026-10-29 after the machine rebooted: the rest of the move-GA 500 M eval.
cd "$(dirname "$0")/../.."
M='OBRRT_MOVES=1'; G='OBRRT_MOVES=1 OBRRT_MOVE_GA=1'
for s in 2 3; do
  tools/arm_eval.sh builds/movega1/obrrt 500000000 mg m="$M" mg="$G" mge="$G OBRRT_MOVE_GA_EXTRA=1" mgc="$G OBRRT_CALM=1" mgel="$G OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1" -- $s >> logs/movega_eval1.txt 2>&1
done
tools/arm_eval.sh builds/movega1/obrrt 500000000 mg mgelr="$G OBRRT_MOVE_GA_EXTRA=1 OBRRT_CALM=1 OBRRT_LAKITU_ARC=1 OBRRT_REROUTE=1 OBRRT_REROUTE_SCHEMA=../objective-scattershot/schemas/reroute.json" -- 1 2 3 >> logs/movega_eval2.txt 2>&1
