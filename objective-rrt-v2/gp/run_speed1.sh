#!/bin/bash
# Exp. 1, first look: unseeded (2 seeds) and seeded.
cd "$(dirname "$0")/.."
G=builds/gp1/gp; R=../baserom.us.z64
for s in 1 2; do nice -n 5 $G $R speed evolve runs/gp/speed-u$s --seed $s --pop 500 --gens 40 --threads 4 > runs/gp/speed-u$s.log 2>&1; done
nice -n 5 $G $R speed evolve runs/gp/speed-s1 --seed 1 --pop 500 --gens 40 --threads 4 --init gp/seeds/speedchain.prog > runs/gp/speed-s1.log 2>&1
echo finished
