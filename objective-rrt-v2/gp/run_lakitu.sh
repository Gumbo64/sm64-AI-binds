#!/bin/bash
# Exp. 2: the Lakitu chain from chain states only. Unseeded and seeded, no world yaws in new material.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue5.pid
until grep -q finished runs/gp/queue4.out; do [ -d /proc/$(cat runs/gp/queue4.pid) ] || break; python3 -c "import time; time.sleep(10)"; done
G=builds/gp5/gp; R=../baserom.us.z64
nice -n 5 $G $R lakitu_chain baseline --threads 4 > runs/gp/lak-base.log 2>&1
nice -n 5 builds/gp5/gabase $R lakitu_chain 4 1 > runs/gp/lak-ga.log 2>&1
A="--pop 800 --gens 40 --threads 4 --noworld"
for s in 1 2 3; do nice -n 5 $G $R lakitu_chain evolve runs/gp/lak-U$s --seed $s $A > runs/gp/lak-U$s.log 2>&1; done
for s in 1 2 3; do nice -n 5 $G $R lakitu_chain evolve runs/gp/lak-S$s --seed $s $A --init gp/seeds/hill_noworld.prog > runs/gp/lak-S$s.log 2>&1; done
echo finished
