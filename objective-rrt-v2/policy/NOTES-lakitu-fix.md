# Lakitu hill overlay: box-landing fixes (hill::overlay)

Files: `objective-rrt/objectives/src/extra/tech/lakituskip.rs` and the v2 copy `objective-rrt-v2/objectives/src/extra/tech/lakituskip.rs` (same edits to `pub mod hill`; the two copies differ elsewhere). Originals: `/tmp/claude-1000/lakfix/lakituskip.v{1,2}.orig`.

## Rule (decomp, camera_lakitu.inc.c + mario_actions_cutscene.c)
`camera_lakitu_intro_act_trigger_cutscene` tests Mario's object position (x -544..545, y > 800, z -2000..-177) every frame; `set_mario_npc_dialog` succeeds only if `mario_ready_to_speak`: action group STATIONARY or MOVING (or WAITING_FOR_DIALOG). Airborne-group actions (jumps, kicks, DIVE, ROLLOUT, SLIDE_KICK incl. its bounce) are safe; any landing action or slide whose end-of-frame position is in the box is caught.

## What was wrong
1. `dj_ok` used `37.4 fv - 500` (error -80..+30 over fv 60-105, +10 on average) and a 30 margin.
2. `kick_range = 10.5 fv` is the no-stick range; the real kick (stick north) is ~11.1 fv, and kick_plan then also did not check the double jump from the kick's landing with the right fv.
3. FREEFALL_LAND bridge branch: slide-kick Z taken for any z > 300 (the hop is ~790-1000 long: lands in the box for z below ~700-1000); jump kick unconditional at the landing stop.
4. The release-A-first path (last_a) spent a ground frame without checking it.

## Fix (all in `hill`)
- `fly()`: the decomp air step (update_air_without_turn: approach 0.35, +1.5*intendedMag*cos, drag -1 above 32, sideways 10*sin; four quarter steps, landing when next y <= floor(next z) with no vy test; gravity after the steps; slide-kick bounce -vy/2 on the first landing). `floor_at(z)`: floor table of the approach. Hop starts: double jump vy 52+fv/4, fv*0.8, steering to the door; jump kick vy 20, first-frame stick = kick()'s 0.38 tilt via `intended_mag`; slide kick vy 12, fv >= 32, **gravity 2** (apply_gravity, mario_step.c:549).
- Sim vs emulator (fuzz hops, `/tmp/claude-1000/lakfix/sim2.py`): double jump 1946 landings, mean +0.3, sd 1.8, 18 of them +17..19 (one quarter step); jump kick 0 to -24 on the bridge; slide kick -20..0 on the bridge; rollout (from SK_SLIDE) +-20.
- `dj_ok` = sim landing z < -2000 - 20 (a quarter step); `dj_ok_by(s, err)` generalises it.
- `kick_plan`: kick from the rollout's landing by the sim, then the double jump from the kick's landing (HOP_ERR = 30 pessimistic); `kick_range` removed.
- JUMP_KICK arm: landing predicted by `fly` from the current air state (was a 40-step loop on a flat 808).
- FREEFALL_LAND bridge branch: DJ press (release-first only if the state after the release frame still has a clear DJ); at n >= 4, z > 300: Z only if the slide kick (after the crouch frame, `after_crouch_frame`) lands clear of the box (+-30), else the jump kick if A is held and clear, else the double jump if it lands past the box at all (margin 2), else Z (as before).
- Unchanged: the unconditional `kick()` arm at the landing stop (no alternative exists there: a rollout landing gives a single jump only, doubleJumpTimer is set only by landing actions, moving.c:1804), the ramp branch, everything with z > 2500, the run-up.

## Results (hill_fuzz, seed 11, 1500+1500 starts, Lakitu arc on)
| | before | after |
|---|---|---|
| A: spawn, random pos/yaw | 1272 (84.8%) | 1340 (89.3%) |
| B: hill bottom (the `lfuzz` family B, 1501 starts) | 815 (54.3%) | 825 (55.0%) |
B detail: 84 starts gained, 6 lost over A+B (4 A, 2 B timeouts; the A ones were double jumps that landed 0.6 units past -2000 and are now declined by the margin, cf. A396: sim -2000.37, real -2000.59).
Replay (v2): hybrid `lakitu_chain wplain --seeds 1 --scale 0.027` (the "Worker GA" replay): train 21/21, val 8/8, test 10/13, 76.0 / 75.5 / 70.8 frames, identical before and after. `gp lakitu_chain baseline` overlay line: train 18/21, val 6/8 -> 7/8, test 10/13 (before = builds/gp8/gp, after = rebuilt).

## Remaining failure classes (B after: 676 fail)
- Jump kick at the landing stop after a rollout, z 200-450, fv ~81 (about 290): kick lands -460, the double jump would be clear but is not available there (prev action is the rollout). Cause is upstream: the chain arrives at the LAND_STOP at the wrong z for this fv.
- Slide kick Z at z 400-800 with neither hop (nor the double jump) clear (about 140).
- Rollout whose landing is in the box (about 60).
- ~25 wall/ledge (0x800034b), ~35 falls, 12 stalled, 7 timeouts. Family A: 161 fail, mostly the same slide kick / jump kick gap.
These are gap states: hop lengths (~790 slide kick, ~880 kick, ~1100 rollout) against a double jump window of z -147..(37.4 fv - 2530) (about 500 at fv 81, 88 at fv 70). Needs a lookahead over the whole hop sequence (the sims here support it: `fly` reproduces rollout, kick, slide kick, double jump) or a ramp-top phase that arrives faster; not done here.
Not modelled: the ramp floors for z > 1700 (sims are for the bridge approach), x-dependence of the floor (the +-18 outliers), walls.

## Harness
`scratchpad/objective-rrt/run/examples/hill_fuzz.rs` now logs every hop (kind, pre-state, landing) in column 17 (`;`-separated) and `HF_TRACE=1` prints slide-kick frames. Run script `/tmp/claude-1000/lakfix/fuzz.sh out.tsv`, summary `sum.py`, fits `sim.py`/`sim2.py`/`an.py`. Build with `CARGO_TARGET_DIR=/tmp/claude-1000/target-lakfix cargo build --release --example hill_fuzz -p obrrt-run` in the scratchpad copy (its lakituskip.rs = the v1 file).
