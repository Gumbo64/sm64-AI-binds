#!/usr/bin/env bash
# Reproduces the slope_lab measurements in ref/tas/slope_techniques.md
# (bridge double jump / Lakitu skip from the 2026 120-star TAS's states).
#   tools/slope_lab_sweeps.sh [lab binary]
# Build: CARGO_TARGET_DIR=builds/target-tas cargo build --release --example slope_lab
set -u
cd "$(dirname "$0")/.."
L=${1:-builds/target-tas/release/examples/slope_lab}
R=../baserom.us.z64
M=ref/tas/full-120star-2026-U.m64
# Event-driven chain from the hill bottom (TAS f3029, DIVE_SLIDE at 104 u/f).
E="ptilt:0.38 land:a:aB ffl:a:aB ptilt:0 ffl:Z hold:1:ZB land:A ptilt:0.38 land:a:aB ptilt:0 run:1 land:A hold:45:a end:60"
P="yaw:0x8000 hold:2:a tilt:0.38 hold:1:aB tilt:1 run:10"  # f3061 -> FREEFALL_LAND at z 117 (f3074)
sum() { sed 's/script "[^"]*"//' | grep -E "DOUBLE_JUMP from|JUMP from|SUMMARY" | cut -c1-260; }

echo "== TAS as played from f3062 (rollout -> slide kick -> rollout -> jump kick)"
$L $R $M 3062 "tas:50" --quiet | sum
echo "== same TAS-like chain grafted onto the first visit (intro Lakitu present)"
$L $R $M 3063 "yaw:0x8000 hold:1:Z hold:1:ZB land:A end:60" --graft 1660 --quiet | sum
echo "== V1: TAS to f3063, jump kick, double jump (no Lakitu / grafted)"
$L $R $M 3061 "$P hold:1:A hold:45:a end:80" --quiet | sum
$L $R $M 3061 "$P hold:1:A hold:45:a end:80" --graft 1660 --quiet | sum
echo "== E: event-driven chain from the hill bottom f3029 (grafted)"
$L $R $M 3029 "yaw:0x8000 flat:A $E" --graft 1648 --quiet | sum
echo "== DJ takeoff z sweep (fv 94)"
for z in 1100 900 700 500 300 117 -100 -176; do echo -n "z=$z "; $L $R $M 3061 "$P setz:$z hold:1:A hold:45:a end:120" --quiet | sum | tail -1; done
echo "== DJ takeoff speed sweep at z=-150 and z=117"
for z in -150 117; do for v in 55 60 65 70 80 94 110; do echo -n "z=$z fv=$v "; $L $R $M 3061 "$P setz:$z setfv:$v hold:1:A hold:45:a end:120" --quiet | sum | tail -1; done; done
echo "== A delay on FREEFALL_LAND, lateral x, facing"
for k in 1 2 3 4; do echo -n "delay=$k "; $L $R $M 3061 "$P run:$k hold:1:A hold:45:a end:120" --quiet | sum | tail -1; done
for x in -500 -400 400 500; do echo -n "x=$x "; $L $R $M 3061 "$P setx:$x hold:1:A hold:45:a end:120" --quiet | sum | tail -1; done
for y in 0x7000 0x7800 0x8800 0x9000; do echo -n "yaw=$y "; $L $R $M 3061 "$P setyaw:$y hold:1:A hold:45:a end:120" --quiet | sum | tail -1; done
echo "== single jump from the rollout landing (FREEFALL_LAND_STOP, fv 93)"
for z in 1119 600 300 0; do echo -n "z=$z "; $L $R $M 3063 "yaw:0x8000 setz:$z hold:1:A hold:45:a end:120" --quiet | sum | tail -1; done
echo "== hill sensitivity: x offset at the dive-slide start f3015, TAS sticks, then E"
for d in -10 -2 2 10 25; do x=$(awk "BEGIN{print 310.1+$d}"); echo -n "dx=$d "; $L $R $M 3015 "setx:$x tas:14 yaw:0x8000 flat:A $E" --quiet | sum | tail -1; done
