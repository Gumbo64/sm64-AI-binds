#!/bin/bash
# Exp. 3b: followers trained on perturbed starts too (follow_train2.txt). R: with the reference in the language;
# T: seeded with reference-replay-with-fallback; N: without the reference. Every program then on the grid and
# on eval_follow.py's rewire starts (follow_rw.txt).
cd "$(dirname "$0")/.."
echo $$ > runs/gp/queue9.pid
G=builds/gp8/gp; R=../baserom.us.z64
evalall() { # name prog
  nice -n 5 $G $R follow eval $2 --threads 4 > runs/gp/fol-$1.eval 2>&1
  GP_CASES=gp/cases/follow_rw.txt nice -n 5 $G $R follow eval $2 --threads 4 > runs/gp/fol-$1.rw 2>&1
}
evalall refonly gp/seeds/refonly.prog
evalall followref gp/seeds/followref.prog
for a in U1 U2 U3 S1 S2; do evalall $a runs/gp/fol-$a/best.prog; done
A="--pop 500 --gens 25 --threads 4"
export GP_CASES=gp/cases/follow_train2.txt
for s in 1 2; do nice -n 5 $G $R follow evolve runs/gp/fol-R$s --seed $s $A --ref > runs/gp/fol-R$s.log 2>&1; done
nice -n 5 $G $R follow evolve runs/gp/fol-T1 --seed 1 $A --ref --init gp/seeds/followref.prog > runs/gp/fol-T1.log 2>&1
nice -n 5 $G $R follow evolve runs/gp/fol-N1 --seed 1 $A > runs/gp/fol-N1.log 2>&1
unset GP_CASES
for a in R1 R2 T1 N1; do evalall $a runs/gp/fol-$a/best.prog; done
echo finished
