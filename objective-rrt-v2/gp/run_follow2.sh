#!/bin/bash
# Exp. 3b: followers with the reference trace in the language (its inputs, action, position at the aligned
# frame), as the RL arm with reference inputs (fol-r3) has. Baselines: pure reference inputs, seed.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue8.pid
until grep -q finished runs/gp/queue7.out; do [ -d /proc/$(cat runs/gp/queue7.pid) ] || break; python3 -c "import time; time.sleep(10)"; done
G=builds/gp7/gp; R=../baserom.us.z64
nice -n 5 $G $R follow eval gp/seeds/refonly.prog --threads 4 > runs/gp/fol-refonly.eval 2>&1
nice -n 5 $G $R follow eval gp/seeds/followref.prog --threads 4 > runs/gp/fol-followref.eval 2>&1
A="--pop 600 --gens 30 --threads 4 --skip-test --ref"
for s in 1 2 3; do nice -n 5 $G $R follow evolve runs/gp/fol-R$s --seed $s $A > runs/gp/fol-R$s.log 2>&1; done
for s in 1 2; do nice -n 5 $G $R follow evolve runs/gp/fol-T$s --seed $s $A --init gp/seeds/followref.prog > runs/gp/fol-T$s.log 2>&1; done
for a in R1 R2 R3 T1 T2; do nice -n 5 $G $R follow eval runs/gp/fol-$a/best.prog --threads 4 > runs/gp/fol-$a.eval 2>&1; done
echo finished
