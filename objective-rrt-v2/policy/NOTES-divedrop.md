# Dive-drop for the Lakitu hill overlay (opt-in, `OBRRT_HILL_DIVE=1`)

Code: `hill` in `objective-rrt-v2/objectives/src/extra/tech/lakituskip.rs` (and the v1 copy in the scratchpad that the fuzz harness builds). Default unchanged: with the flag unset the fuzz outcomes are identical to before on all 3002 starts. Builds: `CARGO_TARGET_DIR=/tmp/claude-1000/target-divedrop`; outputs, patch scripts and the python model in `/tmp/claude-1000/divedrop/` (`off.tsv`, `dive.tsv`, `cmp.py`, `model.py`).

## What the decomp allows (verified in sm64-binds/sm64-port/src/game)
- The premise "slide kick / jump kick into a dive" does not exist. `check_kick_or_dive_in_air` (B press -> `ACT_DIVE` if fv > 28, else `ACT_JUMP_KICK`) is called only from `act_jump` and `act_double_jump`. `act_freefall`, `act_side_flip`, `act_wall_kick_air`, `act_steep_jump`, the triple jumps go to `ACT_DIVE` on B. `act_slide_kick` (gravity 2), `act_jump_kick`, `act_long_jump`, `act_forward_rollout` have no B test at all; the slide kick only becomes freefall after 30 frames and > 500 above the floor, which never happens here. So the dive cannot be entered from the slide kick's lower-gravity arc, and a dive reached from a double jump keeps gravity 4 (same as the double jump), so the "falls twice as fast" argument does not apply to any reachable transition.
- What is reachable: the **ground speed kick**. `check_ground_dive_or_punch` (mario_actions_moving.c:488; called from walking and the moving punch/turn actions, not from `common_landing_cancels`, which has no B test): B pressed, fv >= 29 and stick magnitude > 48 -> `ACT_DIVE`, arg 1, vy = 20. The overlay's `kick()` already issues exactly A+B at the landing but with tilt 0.38 (stickMag < 48) so it becomes a jump kick; the dive-drop is the same presses with tilt 1.0.
- Dive speed: `set_mario_action_airborne` ACT_DIVE: fv = min(fv + 15, 48). From fv 65-90 this **cuts the speed to 48**; that is what shortens the hop (vy 20 as for the kick, gravity 4: 10 frames, ~525 units against ~880 for the kick at fv 81).
- Landing: `ACT_DIVE_SLIDE` (moving group, so a ground frame in the box is caught). `act_dive_slide`: A or B (not above-slide) -> `ACT_FORWARD_ROLLOUT` (airborne, vy 30). Otherwise `update_sliding(8)`: on the default floor class accel 7, loss factor 0.92 + 0.02*intendedMag/32*cos(dyaw) per frame, stop below fv 8 (about 640 units of slide from fv 48).
- `common_landing_cancels` (landing actions): steep floor -> push off, sliding, first person, `numFrames`, A press -> jump, off floor; no B. Irrelevant to the dive except that it confirms the dive can only come from the walking-level B check after the landing frame.

## Model
`Hop::Dive` in `hop_land`: `fly(pos, vy 20, min(fv+15,48), gravity 4, first stick full)`. Check (`/tmp/claude-1000/divedrop/model.py`, same air step as the Rust): dive from z = 300 / 450 / 600 / 800 at fv 81 lands at z = -225 / -88 / +62 / +262 (fv 49.7 after). Double jump from the box edge (z = -147) reaches z = -1232 (fv 40), -1477 (48), -1882 (60), -2263 (70). **So after any dive no double jump can clear the box (needs fv >= ~64 at the box edge): a dive is never a way across, only a way to land short of the box and stop.** Speed cannot be regained on the flat bridge (+0.15 u/f in the air).

## Overlay (flag on)
1. `ACT_FREEFALL_LAND_STOP | PUNCHING | MOVE_PUNCHING` (the kick site): if the sim says the jump kick lands in the box (not `clear_of_box`) and the dive lands at z > -177 + 30, use the dive presses instead of the kick presses.
2. `ACT_FREEFALL_LAND`, n >= 4, z > 300, after the slide kick / kick / double jump alternatives were all declined: dive if it lands clear, instead of the Z gamble.
3. `ACT_DIVE_SLIDE` with fv <= 52 on the bridge approach: A at once (rollout). (A variant that pressed A on every dive slide below z 2500 changed nothing: 0 of 3002 starts differ.)
After the dive Mario is at ~48 u/f, outside the fast-chain conditions; the objective's own steering (`target`, speed < 50) takes him back to the run-up line and the whole hill run starts again. So a dive is an "abort and retry", not a continuation.

## Results (hill_fuzz, seed 11, 1500 + 1500 starts, `OBRRT_LAKITU_ARC=1`)
| | flag off | dive on | gained | lost |
|---|---|---|---|---|
| A spawn | 1340 (89.3%) | 1344 (89.6%) | 5 | 1 |
| B hill bottom (1501) | 825 (55.0%) | 933 (62.2%) | 112 | 4 |
Flag off equals the previous run (`new3.tsv`) outcome for outcome. Dives happened in 178 events (162 B starts, 15 A starts), at z 327-545, fv 48.1-48.2; 123 of the 162 B starts that dived succeeded, 39 did not (17 grounded in the box again, 6 timeouts, 16 falls at z -217..-366 that were the failures of the retry run or the dive landing off the bridge).

Which failure classes (B, off-run last hop of the failed starts):
- Jump kick that lands on the bridge but then falls off or has no way on ("KICK -> FF_LAND", 117 fails): 82 fixed. These are kicks from the landing stop at z ~ 500-600 whose x has drifted to -700..-860; the kick lands at z ~ -135 and the next hop falls. The dive lands short (z ~ 0..100) and the retry works.
- Jump kick that lands in the box ("KICK -> grounded", ~290, the z 200-450 / fv ~81 class): only 24 fixed. The dive's 525-unit hop needs a stop at z >= ~570 to land clear of the box; at z 200-450 it lands in or at the edge of it, so the gate (rightly) stays off.
- Slide-kick Z fallback at z 400-800 (~140): **0 fixed** (1 of the SK class improved). Same reason, and the Z gamble is still taken when the dive does not clear.
- Rollout landing in the box (~60): 0 fixed (5 of the 85 "RO -> stop" ones). The rollout is already in the air; a dive cannot be entered from it.
- Wall/ledge, stalls, timeouts: unchanged.
Remaining B failures with the flag: 568 (465 grounded in the box, 21 wall, 12 stalled, 10 timeouts).

Broken: 4 B starts (415, 716, 744, 1325) and 1 A start (141) that succeeded before. In all five the real kick lands at z ~ -65 (clear) while the model predicts -170..-265 at the kick's z 480-575, fv 65-73, so the gate dives where the kick was fine (the model's kick error is large for these slower, further-out kicks; the dive then costs the retry and 3 of them end in the box again).

Cost (frames until success, B): the gained starts take 461 on average (median 461) against 136 for the baseline successes: a dive-rescued start pays the run-up again, about +325 frames (A: 577 against 220). The dive itself loses 17-40 u/f (fv 65-90 -> 48) and lands ~1000+ units short of where the chain would have been.

## Verdict
Not what was asked, but a real gain: the dive-drop is not a way to fix the gap states (it never can cross: fv 48), it is a safe abort for the part of the kick failures that are not gap states. +108 net on B (+7.2 points), +4 -1 on A, at +325 frames per rescued start. It does not address the three biggest classes (kick at z 200-450 ~266 unfixed, slide-kick Z ~140, rollout ~60). Not a strict gain (5 losses from model error in the kick prediction), so the **defaults are unchanged**; use `OBRRT_HILL_DIVE=1` when success rate matters more than frames. Next steps if wanted: (1) tighten the gate (a margin on the kick's predicted landing, or compare with the kick-then-Z rescue) to remove the 5 losses; (2) in the gap states the only remaining ideas are earlier (ramp-top speed), since after any dive the double jump is out of reach; (3) the retry could skip the full run-up (a dive-slide to rest at z > -147 then a shorter re-approach is not possible: the run-up needs the hill).
