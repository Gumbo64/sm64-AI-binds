#!/bin/bash
# Step 1: validation split and varied training starts (speed2.txt), same 21 held-out cases.
# pop 800 x 40: "small". V: speed2 with validation; C: control, the old training set with the same budget.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue4.pid
G=builds/gp4/gp; R=../baserom.us.z64
A="--pop 800 --gens 40 --threads 4 --noworld"
for s in 1 2 3; do nice -n 5 $G $R speed2 evolve runs/gp/v2-V$s --seed $s $A > runs/gp/v2-V$s.log 2>&1; done
for s in 1 2 3; do nice -n 5 $G $R speed evolve runs/gp/v2-C$s --seed $s $A > runs/gp/v2-C$s.log 2>&1; done
for s in 1 2; do nice -n 5 $G $R speed2 evolve runs/gp/v2-S$s --seed $s --pop 800 --gens 40 --threads 4 --init gp/seeds/speedchain.prog > runs/gp/v2-S$s.log 2>&1; done
echo finished
