#!/usr/bin/env bash
# GA lab (run/examples/movega_lab.rs) on the moves lab cases
# (tools/moves_lab_cases.sh): per case and arm one GA attempt per seed;
# prints each arm's summary line (frames to target, attempt frames,
# frames/s, winner jump starts per 1000 frames, door overshoots).
#
# tools/movega_lab_cases.sh [lab binary] [seeds] [rom]
#   ARMS="name:VAR=1,VAR=1 ..."  (default: off = OBRRT_MOVES=1, on = + OBRRT_MOVE_GA=1)
#   CASES=regex                  only cases whose name matches
set -u
cd "$(dirname "$0")/.."
L=${1:-builds/target-movega/release/examples/movega_lab}
S=${2:-6}
R=${3:-../baserom.us.z64}
ARMS=${ARMS:-"off:OBRRT_MOVES=1 on:OBRRT_MOVES=1,OBRRT_MOVE_GA=1"}
CASES=${CASES:-.}
BOB=tree:runs/tech-kicks-off:334
LOBBY=tree:runs/v47-mw-s1:153
SUMMIT=tree:runs/indef-v47-combo:215743

case_() { # name start target frames [options...]
    local name=$1 start=$2 target=$3 n=$4
    shift 4
    [[ $name =~ $CASES ]] || return 0
    for arm in $ARMS; do
        local an=${arm%%:*} vars=${arm#*:}
        local out
        out=$(env ${vars//,/ } nice -n 10 "$L" "$R" "$start" "$target" "$n" seeds=$S "$@" 2>&1)
        echo "$name $an: $(echo "$out" | grep '^summary' | sed 's/^summary: //')"
        echo "$out" | grep '^move-ga' | sed "s/^/    /"
    done
}

case_ hill-49deg boot -100,1061,5950 900
case_ hill-top boot 50,1204,6100 900
case_ slope-36deg boot 300,832,3200 900
case_ bob-slope-1 $BOB -4778,640,7851 900
case_ flat-2470 boot -3800,260,4354 600
case_ flat-1170 boot -2500,260,4354 600
case_ castle-path boot -1328,369,3000 600
case_ bob-flat $BOB -2500,171,6000 600
case_ bob-gap $BOB -5725,128,6100 300 pre=-5725,128,4700,300
case_ facade-reversal boot -1244,713,1500 200 prestick=-1328,-3000,151
case_ lobby-reversal $LOBBY -1609,0,300 200 prestick=-946,1000,28 prestick=-1727,500,30 prestick=-1727,9000,38
case_ hill-down boot -1328,260,4354 600 pre=300,832,3200,300
case_ summit-drop-full $SUMMIT 3773,2948,-2600 600
case_ summit-pound $SUMMIT 3773,2977,-2900 600
case_ lobby-door-left $LOBBY -271,0,-824 400 door
case_ lobby-door-right $LOBBY -1775,0,-824 400 door
# Lobby (indoors, ceiling ~700 up): across it, towards the stairs, and
# the doors to the west wing and past the right door.
case_ lobby-across $LOBBY -1000,0,-500 300
case_ lobby-stairs $LOBBY -1000,0,-1600 400
case_ lobby-door-west $LOBBY -3122,205,-793 500 door
case_ lobby-door-east $LOBBY 256,0,-1074 500 door
# Slope / high-speed techniques (ref/tas/slope_techniques.md) from the
# TAS's saved states (visit 2, no Lakitu): to the TAS's hill-bottom spot
# from the dive slide start (TAS: 14 frames) and from the hill-climb dive
# (TAS: 20), and from the hill bottom at 104
# up the ramp and over the bridge to the castle door.
ST=raw:ref/starts/full-120star-2026-U
case_ tas-hill-down $ST-f3015.m64 -38,260,3882 80
case_ tas-hill-climb $ST-f3009.m64 -38,260,3882 80
case_ tas-ramp-top $ST-f3030.m64 -24,665,3291 120
case_ tas-bridge $ST-f3030.m64 -40,803,-2900 200
case_ tas-door $ST-f3030.m64 -50,803,-3070 200 door
