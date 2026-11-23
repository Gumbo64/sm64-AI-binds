# Skill library (policy/src/skills.rs)

31 reactive skills. A skill reads an `Obs` (Mario's own state: action, speed,
flags/caps, health, held object, floor class/normal, frames in action, last pad;
plus `Sense`: floor probes, wall rays, edge/ledge, predicted landing, objects) and
returns a `Choice` (button set + stick region + the continuous part).
`library()` returns them in a stable order; `Skill::param_ranges()` and
`defaults()` expose tunable parameters for the GA.

## Core: the speed-chain option table

`options(&F) -> Vec<Opt>` is a port of `scratchpad/speedchain_overlay.py`: for the
current action, every input that keeps speed (`Kind::Keep`), stops (`Stop`),
slows (`Slow`), exits to a named action (`Exit`) or breaks the chain by accident
(`Unsafe`), each with button requirements (`Req::{Up,Fresh,Held}`), a stick
class, the action it leads to, and the next-speed ratio `mult`. `Fresh`
resolves against the previous pad (a held button is released first). Fixes found
while running it on the real game (all in the code, with comments):

* A kick (A held + B fresh at a landing) is only offered if A **was already
  held**; otherwise the A press is a fresh jump press. So `air_hold` and
  `land_wait` hold A at speed (>= 40 u/f).
* `*_LAND` actions take their cancel on the frame the timer runs out:
  `action_timer + 1 >= numFrames` (4; long jump 6), not `action_timer >= 4`.
  One frame late and the landing expires into WALKING (speed capped at 48).
* Kick and crouch entries are priced at 0.5 below 24 u/f (they are useless
  slow); hold-speed options win there.
* Stick for a kick is steered to the target (the direction on the kick frame
  decides the heading of the whole hop).

## Skills (params in [lo, hi], defaults = midpoint unless noted)

| # | skill | what it does | params |
|---|---|---|---|
| 1 | keep_best | best keep option whose predicted landing is safe (zone/floor), stick steered to target | zone margin, steer gain, max deviation |
| 2 | hold_speed | only options that stay in the action (argmax kept speed) | gain, maxdev |
| 3 | steer_target | aim at target; buttons from the best keep option; short stick when close | close-frames, full-stick flag |
| 4 | slide_optimise | stick along the slide, bounded deviation towards target; rollout cancel | gain, maxdev, rollout min speed |
| 5 | rollout_cancel | dive/slide-kick/stomach slide -> rollout (A fresh) | min speed (default 0) |
| 6 | kick_chain | landing refresh: A held through the air, B fresh at the expiry frame, landing point checked against zones; falls back to double jump / crouch | min speed (30), zone margin (50) |
| 7 | slide_kick_chain | crouch slide -> slide kick -> slide kick slide -> rollout -> landing -> crouch slide | min speed (30), steer gain |
| 8 | edge_dive_hop | leave a ledge, dive at <= 33, rollout | max speed, edge distance |
| 9 | wall_kick_chain | aim into a wall (rays), A on the one real frame, steer along the kick | reach |
| 10 | butt_slide_chain | slope butt slide, jump at timer 5, land back in the slide | min speed, gain |
| 11 | jump_chain | single -> double -> triple by landing timing; A held for height | triple min speed, hold A |
| 12 | long_jump_builder | crouch slide + A (x1.5, cap 48); BLJ mode holds the stick back | blj flag |
| 13 | ground_pound | Z in the air above `min_height`, near the target | min height, near-target dist |
| 14 | brake_stop | fastest zero speed (neutral on ground, back on slides, Z in air) | - |
| 15 | slow_bleed | option whose ratio is nearest the wanted speed | target speed |
| 16 | turn_around | held-back stick on the ground, side flip out | angle threshold |
| 17 | exit_jump | leave the chain with jump/double/long/backflip | kind |
| 18 | exit_dive | B on ground (speed kick, fv >= 29, stick > 48) or in air | min speed |
| 19 | ledge_grab | steer onto a ledge lip (Sense.ledge_ahead), climb | reach |
| 20 | pole | approach pole (object category), climb or jump off | category, jump-off flag |
| 21 | swim_cadence | A held or pulsed, stick at target (in water) | pulse |
| 22 | avoid_wall | slide along the wall tangent nearer the target | frames ahead, side bias |
| 23 | avoid_edge | steer along a drop | min distance |
| 24 | approach_object | aim at nearest object of a category, jump if above | category, jump range |
| 25 | avoid_object | steer away from nearest hazard, biased to target | category, radius |
| 26 | wait | stick neutral, no buttons | - |
| 27 | zone_safe_hop | among the jumps now available, the safe one landing furthest towards the target | zone margin |
| 28 | exit_to | leave the chain: 0 stop, 1 slow, 2 jump, 3 dive, 4 ground pound | exit kind, speed |
| 29 | cap_aware | wing cap: A on landings so triple jumps fly; metal cap in water | zone margin |
| 30 | health_guard | low health: away from enemies, else keep_best | health threshold |
| 31 | carry | holding an object: throw (B) near the target, else carry on | throw range |

`wall_deflect()` is applied in every steered skill: if the forward wall ray is
shorter than 8 frames of travel, the stick goes along the wall tangent nearer
the wanted heading. `set_zones()` registers areas Mario must not land in
(`lakitu_zone()` is the box); `ballistic()` predicts the landing of any jump
option from the floor probes.

## What the check shows (bin/skill_check.rs; runs in policy/runs/skills/)

Problem `speed` (SpeedChain from 36 route/TAS start states at 66-106 u/f, 150
frames, the real emulator, Sense computed locally from memory):

* **Port fidelity.** `match` replays 7 movies and compares keep_best's pad with
  the movie's on 1,800 fast frames (>= 55 u/f): button match by action: dive
  slide 79%, double jump 73%, FREEFALL_LAND 65%, rollout 58%, jump kick 47%,
  LAND_STOP 75%; stick class (neutral/short/full) 81-100%. Low rows are mostly
  "TAS holds a button the speed does not need" (slide kick B+Z).
* **It reproduces the real chain.** From the King BLJ state f2435, keep_best
  alone plays rollout -> landing stop -> jump kick -> freefall land -> crouch
  slide -> slide kick -> slide kick slide -> rollout -> ... with the TAS's own
  pads on the same frames (Z at the landing expiry frame, B+Z next, A next) and
  holds 80 u/f over 60 frames (89 -> 80).
* **A hand state->skill table** (dive slide -> rollout_cancel, slide-kick states
  -> slide_kick_chain, rollouts/landings -> kick_chain, else keep_best) finishes
  the task on 16/36 starts; the movie's own inputs finish 29/36 (150 frames).
  Skills alone: keep_best 16/36, slide_kick_chain 18/36, long_jump_builder 21/36
  (fast Done by coasting, not by keeping speed); skills that only steer forward
  also reach 13-19/36, so Done is mostly "coasting is enough" on the easy half.
* **Failure mode (all the other starts):** speed falls to 0 / negative within the
  first hops: Mario meets something the Sense does not model (BoB's moving
  obstacles / object walls at (-4100, 6930) in the King state f2441; the movie
  dodges with a 10-degree different kick stick). Static-wall rays (known zone)
  report nothing there. Needs object-collision rays (Sense.objs for solid
  objects) and the planner/MPC layer to pick the kick angle.

## Known limits / not done

* Ledge, pole, water, cap, held-object skills are implemented but only exercised
  by speed-chain starts (none of which swim, grab poles or carry): treat them as
  untested.
* `Sense.land`, `ledge_ahead`, object categories come from whoever fills Sense
  (skill_check computes them locally; sense.rs belongs to another agent).
* The menu's `realize` is not used here; skill_check has its own (`Aim` class
  +-0x400, Left/Right 0x1400, Side 0x4000, Back 0x8000, Short tilt 0.30-0.38,
  Full 1.0). Change both together.
* Memory rule (user): never delete a skill; retire with a status flag and keep
  the source. The library order in `library()` is the id.
