#!/bin/bash
# Exp. 2 again: the seed's depth (about 20) was over max_depth 14, so every seeded child was rejected
# (run lak-S1..3: 3.6 M frames, no search). Depth 40, size 500. Unseeded with more budget too.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue7.pid
until grep -q finished runs/gp/queue6.out; do [ -d /proc/$(cat runs/gp/queue6.pid) ] || break; python3 -c "import time; time.sleep(10)"; done
G=builds/gp6/gp; R=../baserom.us.z64
A="--threads 4 --noworld --max-depth 40 --max-size 500"
for s in 1 2 3; do nice -n 5 $G $R lakitu_chain evolve runs/gp/lak-D$s --seed $s --pop 800 --gens 40 $A --init gp/seeds/hill_noworld.prog > runs/gp/lak-D$s.log 2>&1; done
for s in 4 5 6; do nice -n 5 $G $R lakitu_chain evolve runs/gp/lak-B$s --seed $s --pop 2000 --gens 60 --threads 4 --noworld > runs/gp/lak-B$s.log 2>&1; done
echo finished
