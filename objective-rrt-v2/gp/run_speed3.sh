#!/bin/bash
# Exp. 1: no absolute world yaws (sticks relative to the target, facing or motion only).
# Starts after queue 2. Parsimony 0.02 and 0.1.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue3.pid
until grep -q finished runs/gp/queue2.out; do [ -d /proc/$(cat runs/gp/queue2.pid) ] || break; python3 -c "import time; time.sleep(10)"; done
G=builds/gp3/gp; R=../baserom.us.z64
A="--pop 1500 --gens 60 --threads 4 --noworld"
for s in 1 2; do nice -n 5 $G $R speed evolve runs/gp/speed-W$s --seed $s $A > runs/gp/speed-W$s.log 2>&1; done
for s in 1 2; do nice -n 5 $G $R speed evolve runs/gp/speed-P$s --seed $s $A --parsimony 0.1 > runs/gp/speed-P$s.log 2>&1; done
echo finished
