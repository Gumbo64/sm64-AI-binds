#!/bin/bash
# Exp. 1, more budget: the GA baseline, unseeded, seeded, ADFs.
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue2.pid
G=builds/gp2/gp; R=../baserom.us.z64
nice -n 5 builds/gp2/gabase $R speed 4 1 > runs/gp/speed-ga.log 2>&1
A="--pop 1500 --gens 60 --threads 4"
for s in 1 2; do nice -n 5 $G $R speed evolve runs/gp/speed-U$s --seed $s $A > runs/gp/speed-U$s.log 2>&1; done
for s in 1 2; do nice -n 5 $G $R speed evolve runs/gp/speed-S$s --seed $s $A --init gp/seeds/speedchain.prog > runs/gp/speed-S$s.log 2>&1; done
nice -n 5 $G $R speed evolve runs/gp/speed-A1 --seed 1 $A --adf 2 > runs/gp/speed-A1.log 2>&1
echo finished
