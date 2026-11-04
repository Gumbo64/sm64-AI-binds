#!/bin/bash
# Lane B2 (8 threads), queued 2026-10-29 after lane B: attempt-counted kinship and objective-decided dense keep at 500 M x 3 seeds.
cd "$(dirname "$0")/../.."
while kill -0 1995 2>/dev/null; do sleep 60; done
export THREADS=8
W='OBRRT_ADD=winner_all,last_gen'
tools/arm_eval.sh builds/kin2/obrrt 500000000 kn5 att-k2="$W OBRRT_KIN_UNIT=attempt OBRRT_KINSHIP_MULT=2" obj-soft8="$W OBRRT_DENSE_EVERY=obj OBRRT_KIN_UNIT=attempt OBRRT_KIN_SOFT=0.8" att-obj="$W OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=obj" att-e30="$W OBRRT_KIN_UNIT=attempt OBRRT_DENSE_EVERY=30" -- 1 2 3 >> logs/kn5_eval1.txt 2>&1
