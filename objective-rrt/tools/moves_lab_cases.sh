#!/usr/bin/env bash
# Movement library lab cases (run/examples/moves_lab.rs): each case steers
# from a fixed start to a fixed target with the generic controller ("off")
# and with the library ("on"), printing frames to get within 100.
#
# tools/moves_lab_cases.sh [lab binary] [rom]
# Needs runs/tech-kicks-off (BoB start, node 334), runs/v47-mw-s1 (castle
# lobby corridor, node 153) and runs/indef-v47-combo (BoB summit, node
# 215743) for the tree starts.
set -u
cd "$(dirname "$0")/.."
L=${1:-builds/target-moves/release/examples/moves_lab}
R=${2:-../baserom.us.z64}
BOB=tree:runs/tech-kicks-off:334
LOBBY=tree:runs/v47-mw-s1:153
SUMMIT=tree:runs/indef-v47-combo:215743

case_() { # name start target frames [options...]
    local name=$1 start=$2 target=$3 n=$4
    shift 4
    echo "== $name: $start -> $target $*"
    "$L" "$R" "$start" steer "$target" "$n" both "$@" | grep -E "^(set up|off|on|      moves)"
}

# Slopes walking slides down on (castle grounds, 49° very slippery hill).
case_ hill-49deg boot -100,1061,5950 900
case_ hill-top boot 50,1204,6100 900
case_ hill-west boot -1100,1169,6200 900
case_ slope-36deg boot 300,832,3200 900
# BoB mountain-side slopes.
case_ bob-slope-1 $BOB -4778,640,7851 900
case_ bob-slope-2 $BOB -4200,661,8000 900
# Flat runs (long-jump chains).
case_ flat-2470 boot -3800,260,4354 600
case_ flat-2170 boot -3500,260,4354 600
case_ flat-1170 boot -2500,260,4354 600
case_ castle-path boot -1328,369,3000 600
case_ bob-flat $BOB -2500,171,6000 600
# Gap: BoB spawn-area hole (about 300 across), approached at speed.
case_ bob-gap $BOB -5725,128,6100 300 pre=-5725,128,4700,300
# Wall-kick reversal: running at the castle facade (z -121, faces +z) at
# full speed, target behind.
case_ facade-reversal boot -1244,713,1500 200 prestick=-1328,-3000,151
# Lobby: running at the +z wall, target behind (a Toad stands near it).
case_ lobby-reversal $LOBBY -1609,0,300 200 prestick=-946,1000,28 prestick=-1727,500,30 prestick=-1727,9000,38
# Slide kick / belly slide downhill: from the castle hill top down to the flat.
case_ hill-down boot -1328,260,4354 600 pre=300,832,3200,300
# Summit cliff (BoB): the ledge at y ~4110 drops ~1150 to y ~2950 towards
# +z. A target far out (fall damage at low health: fall-safe) and one just
# under the edge (ground pound onto it).
case_ summit-drop-full $SUMMIT 3773,2948,-2600 600
case_ summit-drop-2wedges $SUMMIT 3773,2948,-2600 600 health=2
case_ summit-pound $SUMMIT 3773,2977,-2900 600
# Door approach (arcs::plan_approach_with): lobby to the doors on its far
# side; frames to the door action ("door at frame").
case_ lobby-door-left $LOBBY -271,0,-824 400 door
case_ lobby-door-right $LOBBY -1775,0,-824 400 door
