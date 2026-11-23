#!/bin/bash
# Exp. 3: the follower on the RL eval's held-out grid. Steer baseline at --starts 64 (cross-check with the
# RL log's steer arm) and 128; evolution on training movies, validation on another run.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue6.pid
until grep -q finished runs/gp/queue5.out; do [ -d /proc/$(cat runs/gp/queue5.pid) ] || break; python3 -c "import time; time.sleep(10)"; done
G=builds/gp5/gp; R=../baserom.us.z64
GP_CASES=gp/cases/follow64.txt nice -n 5 $G $R follow eval gp/seeds/trivial.prog --threads 4 > runs/gp/fol-steer64.log 2>&1
nice -n 5 $G $R follow eval gp/seeds/trivial.prog --threads 4 > runs/gp/fol-steer.log 2>&1
A="--pop 600 --gens 30 --threads 4 --skip-test"
for s in 1 2 3; do nice -n 5 $G $R follow evolve runs/gp/fol-U$s --seed $s $A > runs/gp/fol-U$s.log 2>&1; done
for s in 1 2; do nice -n 5 $G $R follow evolve runs/gp/fol-S$s --seed $s $A --init gp/seeds/trivial.prog > runs/gp/fol-S$s.log 2>&1; done
for a in U1 U2 U3 S1 S2; do nice -n 5 $G $R follow eval runs/gp/fol-$a/best.prog --threads 4 > runs/gp/fol-$a.eval 2>&1; done
echo finished
