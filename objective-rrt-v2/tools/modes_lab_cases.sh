#!/usr/bin/env bash
# The mode policies' GA lab (objectives/src/moves: cannon.rs, water_air.rs,
# slide.rs, pole.rs, tech.rs): run/examples/movega_lab.rs from cannon,
# water, flight, slide and pole states of earlier runs and the 2026
# 120-star TAS, before (OBRRT_MOVES_MODES unset) and after
# (OBRRT_MOVES_MODES=1), both with OBRRT_MOVES=1 OBRRT_MOVE_GA=1 and the
# TAS swim cadence; prints each arm's summary line and the move-ga stats.
#
# tools/modes_lab_cases.sh [lab binary] [seeds] [rom]
#   CASES=regex  only cases whose name matches
set -u
cd "$(dirname "$0")/.."
L=${1:-builds/target-reroute/release/examples/movega_lab}
S=${2:-6}
R=${3:-../baserom.us.z64}
CASES=${CASES:-.}
BASE="OBRRT_MOVES=1 OBRRT_MOVE_GA=1 OBRRT_SWIM_HOLD=1"
ARMS=${ARMS:-"before: after:OBRRT_MOVES_MODES=1"}
IND=runs/indef-v47-combo
TAS=ref/tas/full-120star-2026-U.m64

case_() { # name start target frames [options...]
    local name=$1 start=$2 target=$3 n=$4
    shift 4
    [[ $name =~ $CASES ]] || return 0
    for arm in $ARMS; do
        local an=${arm%%:*} vars=${arm#*:}
        local out
        out=$(env $BASE ${vars//,/ } nice -n 10 "$L" "$R" "$start" "$target" "$n" seeds=$S "$@" 2>&1)
        echo "$name $an: $(echo "$out" | grep '^summary' | sed 's/^summary: //')"
        echo "$out" | grep '^move-ga' | sed "s/^/    /"
    done
}

# Cannons (in the cannon, before it opens): BoB's by the start along its
# range (to pass a far point, to stop at a mid one), WF's up to the fortress.
case_ cannon-bob-far tree:$IND:159352 2540,768,5663 400
case_ cannon-bob-mid tree:$IND:159352 -623,347,5639 400 arrival=stop
case_ cannon-wf tree:$IND:11360 -1585,2181,1915 400
# Swimming: JRB (to the bay floor, and out onto the shore).
case_ swim-jrb-deep tree:$IND:4591 -1949,-1023,3492 400
case_ swim-jrb-exit tree:$IND:4591 -16,1359,2692 400
# Flying (wing cap): TotWC, BoB.
case_ fly-totwc tree:$IND:160230 -1824,2707,500 300
case_ fly-bob tree:$IND:203311 3145,719,1726 300
# Slides: PSS round the first bend, CCM's slide round its first bend
# (targets on the TAS's line).
case_ slide-pss tree:$IND:3775 -5244,3279,1340 300
case_ slide-ccm tree:runs/hy-base-s1:2623 5936,4395,-2967 300
case_ slide-pss-tas tas:$TAS:11600 -5737,3922,-2542 200
# Poles: WF's pole (holding) to the ledge by it, a castle-grounds tree.
# (Targets are states other nodes of the same runs reached.)
case_ pole-wf tree:$IND:19992 2731,1024,4090 300
case_ pole-tree tree:$IND:55952 -6969,797,-2748 300
