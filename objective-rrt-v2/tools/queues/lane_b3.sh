#!/bin/bash
# Lane B3 (8 threads), restarted 2026-10-29 after the "update claude" container reset:
# finish the schema-pick eval (s2 was cut at 370 M, s3 never ran), then the
# attempt-counted kinship / objective-decided dense keep eval (lane_b2).
# s1 results (ps-walg 1:53.1, ps 1:59.6) are already in logs/pick_eval1.txt.
cd "$(dirname "$0")/../.."
export THREADS=8
for s in 2 3; do
  tools/arm_eval.sh builds/pick2/obrrt 500000000 pk5 ps-walg="OBRRT_PICK=schema OBRRT_ADD=winner_all,last_gen" ps="OBRRT_PICK=schema" -- $s >> logs/pick_eval1.txt 2>&1
done
W='OBRRT_ADD=winner_all,last_gen'
tools/arm_eval.sh builds/kin2/obrrt 500000000 kn5 att-k2="$W OBRRT_KIN_UNIT=attempt OBRRT_KINSHIP_MULT=2" att-obj="$W OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=obj" -- 1 2 3 >> logs/kn5_eval1.txt 2>&1
echo "lane_b3 done"
