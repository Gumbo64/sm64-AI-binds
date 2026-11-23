# Dive grind and edge (ledge) grind: what the decomp says, what the emulator measured

Policies are in `policy/runs/inventory/staging/` (NOT in the active inventory / incoming; the integrator promotes them).
Source references are `sm64-binds/sm64-port/src/game/` (US build). Everything below marked **C** was read in the C source;
**M** was measured with the real stack (`policy ... trace/eval/play`, emulator) and replayed with tasplay (118 / 300 / 300 / 62 / 68 frames
of five plays compared with the policy trace: no frame differs by more than 1.0 in x+z+fv).

## 1. Dive: the rules (C)

* **Forward speed of a dive.** `mario.c:859` (`set_mario_action_airborne`, `case ACT_DIVE`): `fv = min(48, fv + 15)`. The cap is on the *result*:
  a dive at fv 79.5 gives 47.8 (M: 2012 TAS 22563, rollout 79.5 -> `freefall_land_stop` -> B -> dive 47.8). A dive can never raise speed above 48 and cuts anything faster.
* **Ways into a dive.**
  * Ground "speed kick": `check_ground_dive_or_punch` (`mario_actions_moving.c:488`), from `act_walking` only: B pressed, `forwardVel >= 29` **and** `stickMag > 48`; also sets `vel[1] = 20`, `ACT_DIVE` with arg 1.
    A `*_land_stop` action with a stick input first hands over to walking (`check_common_landing_cancels` -> `check_common_action_exits` -> `ACT_WALKING`); `execute_mario_action` loops while an action returns TRUE (`mario.c:1726`), so walking's B check runs **in the same frame**: B at a `freefall_land_stop` frame already dives (M: one frame).
    The moving landings (`jump_land`, `freefall_land`, `long_jump_land`, 4 frames; long jump 6) do not look at B at all.
  * Air: `check_kick_or_dive_in_air` (`mario_actions_airborne.c:109`): B -> `ACT_DIVE` only if `fv > 28`, else jump kick. Used by jump, double jump, (triple jump dives at any speed, `:490`).
    **`act_freefall` (`:531`) turns B into `ACT_DIVE` at ANY speed** (no test). `act_long_jump` has no dive/kick check at all (read `:643`).
* **Dive landing.** `act_dive` `AIR_STEP_LANDED` (`:769`): get stuck (only snow/sand terrain, fall > 1000, normal.y >= 0.866) / hard-fall damage (> 1150 fall) else `ACT_DIVE_SLIDE` (`ACT_DIVE_PICKING_UP` if holding). A dive cannot be followed by a dive.
* **Dive slide.** `act_dive_slide` (`mario_actions_moving.c:1583`): the FIRST thing it does is `if (!(input & INPUT_ABOVE_SLIDE) && (input & (A_PRESSED | B_PRESSED)))` -> `ACT_FORWARD_ROLLOUT` (fv > 0). That test runs before `update_sliding`, so a press on the first slide frame costs **no** friction (M: 47.4 -> 47.2 on the rollout's first frame).
  **B works as well as A** (relevant to the A-button challenge). Without a press the slide loses `fv *= 0.92 + 0.02*intendedMag/32*cos(stick - slideYaw)` per frame (`update_sliding`, `:222`, default floor; 0.94-0.98 slippery) and stops under 8 u/f.
* **Rollout.** `act_forward_rollout` (`:1385`): vy = 30, fv unchanged at entry (no case in `set_mario_action_airborne`), air physics, no B/A/dive check; lands in `ACT_FREEFALL_LAND_STOP`. A wall hit zeroes fv (`:1413`).
* **Air physics** `update_air_without_turn` (`:217`): per frame `fv -= 0.35 toward 0`, `+1.5*cos(stick - facing)*intendedMag/32`, `-1` above 32 (48 for long jump), sideways `10*sin(...)*mag/32`. A stick on the facing gives -0.35+1.5-1 = +0.15 at > 32 (fv stays), a stick 90 degrees off gives -1.35/frame (M: dive 45.3 -> 31.7 in 10 frames with the stick sideways). Mario never turns in the air / in a dive slide / in a landing, so **the heading is fixed for the whole grind cycle**.
* **Walking** `update_walking_speed` (`:437`): target `min(intendedMag, 32)`, **24 on `SURFACE_SLOW`** (grass: both BoB's spawn field and the castle grounds measured here: walking tops out at 24-25); `+= 1.1 - fv/43` below target, **-1 per frame above it only on floors with normal.y >= 0.95**. `hold_walking` (`:882`) scales the stick by 0.4 (target 12.8, 9.6 on slow floors; M: 9.5-10 u/f).

### The dive grind cycle (M, lakitu start and BoB)
`B` at a ground action (fv >= 29, facing the target) -> `dive` (~3-10 frames) -> `dive_slide` (1 frame, A or B pressed) -> `forward_rollout` (~8-16) -> `freefall_land_stop` + B -> `dive` ... 11-25 frames per cycle, no ground friction frame at all.
Steady state: **fv 45.8-47.8, horizontal 45.8 u/f** (lakitu start, grass floor, frames 56-116). ABC variant (B only, no A anywhere): 44-47 u/f (BoB king start, fv 45.9, frames 5-60: 43.8 u/f).

## 2. Claims vs. what holds

| claim | verdict |
|---|---|
| dive grinds (dive -> land -> dive) get Mario to 48 quickly | **True but trivial and capped.** From fv >= 33 a single dive gives 48; from 29 two. From a standing start on SLOW floor (cannot reach 29 by walking) the fastest entry is crouch slide + A (long jump x1.5, then air +0.7/frame to 48) at ~frame 47 (M), a dive grind adds nothing to that. The "dive -> landing -> dive" is not literal: the landing is `dive_slide`, which cannot dive; the loop needs a rollout in between. |
| better than other chains | **No.** Long jump chain (crouch, A, land, crouch...) on the same start: 46.2 u/f vs the dive grind's 45.8 (M, same window). It is only ahead of walking (21.8 mean over 100 frames from rest, max 25) and it is the only 48-speed chain that never needs A (ABC). |
| useful on the `speed` problem (66-106 u/f starts) | **No, it destroys speed there**: every dive cuts to 48. Standalone: success 1/60 fresh, 4/45 field, 0/55 noise vs speed-chain-3 2, 3, 1 and X00006 1, 4, 3 (`eval`, 300 frames). The cap-guarded portfolio `dive-grind-gated` (grind only <= 48.5, speed-chain-3 above) ties speed-chain-3 on fresh/field (2/60, 3/45), +1 on noise (2/55 vs 1/55), mean fitness 21.8 / 19.4 / 21.5 vs 20.1 / 19.4 / 18.7. The standalone grind is straight (heading cannot change) and ends at walls (22552 slope start: stalls at (-3868,5721)). One field start (2026@28661) is solved only by the standalone dive grind (165 frames); the one fresh start it solves faster than speed-chain-3 (Bobomb-GateClip@4215, 149 vs 209) is a bonk recovery that the walk -> crouch -> long jump entry fixes, not the grind. |
| on a slope | Spawn slope top (2012 TAS 22552, dive_slide 34 u/f heading downhill): the dive grind goes straight at 43.7 u/f net displacement over 60 frames; speed-chain-3 / X00006 path 47-48 u/f but only 26.5 net (they zig-zag and go backwards, fv -45), plain slide 26. Slope does not change the grind (the dive slide is skipped; floor normal only enters landings). Not an objective win (stall at the wall). |
| edge grinding while holding an object is much faster than walking with it | **Mechanism real, size unverified, and the stick is the wrong way round**: see section 3. Hop chain with the object (no edge): **25.9 u/f vs 9.7 u/f** hold-walking (frames 30-60, 2.7x, M). Held edge grind itself: no held-object grind position exists in the sampled TAS states (none found, section 3), UNTESTED. |
| ABC uses edge grinding "for speed" | **Unverified, partly plausible.** Not found in the source as anything special. What the source does give an ABC run: B is legal, `act_freefall` dives at ANY fv (+15, so a ledge walk-off at fv >= 14.5 gives the 29 the ground speed kick needs), dive_slide and the rollout cancel on B. So an A-less run can ignite a dive grind from a ledge fall with no run-up, which is the "benefit" the data supports (implemented as the `ignite-freefall` rule in the dive grinds). The edge grind itself does not give speed (section 3). |

## 3. Edge (ledge) grind (C + M)

Mechanism (C, `mario_step.c:260-350`, `mario.c:1360`, `mario_actions_moving.c:1757-1827`): the ground step reports `GROUND_STEP_LEFT_GROUND` when the next floor is more than 100 below
(`nextPos[1] > floorHeight + 100`), `INPUT_OFF_FLOOR` is `pos.y > floorHeight + 100`. Mario is then in `ACT_FREEFALL`; on a lip the air-step drift (the sideways term above) moves him back over the platform
and the air step LANDS the same frame (`pos.y <= floor`): `ACT_FREEFALL_LAND`; the landing's own ground step steps off again (LEFT_GROUND -> `ACT_FREEFALL`), and so on every frame, y flat. Held: `hold_freefall` <-> `hold_freefall_land` (same code through `common_air_action_step` / `common_landing_cancels`). The other idle <-> freefall oscillation is the same thing from a stationary state (`stationary_ground_step` then `INPUT_OFF_FLOOR`).

Found with the real emulator (`src/bin/grind_edge.rs`, 12 stick directions x 40 frames from every n-th controllable ground state of ref/tas 120-star 2012 / 2026 and 70-star + BoB movies): hundreds of (state, direction) hits (about 40 true freefall/freefall_land alternations, the rest are butt/belly slides flipping the air flag) with 20-68 air flips in 40 frames and zero height change, actions `0x100088c` (freefall) / `0x4000471` (freefall_land); a hit needs the stick roughly perpendicular to the facing (side-drifting onto the lip). Mean speed in the grid-search hold: **fv 2-26, horizontal displacement over 80 frames 34-589 (0.4-7.4 u/f)**, versus 300-2700 (4-34 u/f) when the same states are simply walked off facing the stick (`face` rows of `NOTES-grind` table below). **The grind keeps Mario on the lip but is 4-10x slower than running along the edge.** Air physics cap a facing-aligned stick at ~32 (above 32: +0.15/air frame vs x0.98 per landing frame), so at best it matches walking on normal floor; on a 24 floor it would be 1.3x, nothing like "much faster".
Held-object claim: hold-walking is capped at 0.4 x 32 = 12.8 (9.6 slow floor), the grind's air frames are not, so a grind with an object can beat hold-walking (grind rates above are 0.4-7.4 u/f unheld, comparable to or below the 9.6-12.8 of hold-walking, so even this is not shown). The hop chain gets 26 u/f with no geometry at all. The held edge grind policy is therefore parse-checked/traced only.

Table (fv-mean / disp80, 80 frames from each hit state; `const` = the discovering side stick that holds the grind, `face` = stick on the facing (walk off), `+dive` = freefall B dive at fv >= 32, then A on the first slide frame):

| state | const (grind) | face (walk) | face+dive32 |
|---|---|---|---|
| 2012@116821 (L11) | 25.6 / 355 | 32.0 / 2525 | 40.4 / 3183 |
| 2026@88201 (L36) | 5.1 / 84 | 36.0 / 2730 | 43.8 / 3452 |
| 2026@115801 (L15) | 10.0 / 589 | 30.1 / 2416 | 34.0 / 2722 |
| 2026@19201 (L24) | 5.4 / 204 | 18.2 / 1459 | 29.2 / 1649 |
| 2012@71401 (L18) | 6.6 / 410 | 17.8 / 2240 | (falls: 2240) |
| 70star@48201 (L11) | 2.4 / 194 | 28.7 / 2291 | 28.7 / 2291 |

(full 15-state output: `runs/inventory/staging/edge-grind-ctrl.tsv`, the (movie, frame, direction) list `edge-grind-hits.tsv`; many states fall off the level or turn into slides with `face`, those rows are not apples to apples.) The B dive from the freefall frame adds +13% to +26% displacement in 4 of 6 states, 0 in the others.

## 4. Policies (staging)

All parse and run in the real stack. Lambdas: `p0` lean/aim, `p4` facing-error gate, `p5` ignition speed, `p3` crouch speed; see each file's note.

| file | what | tested |
|---|---|---|
| `dive-grind-chain.pol` | A rollout version with walk -> Z -> long-jump entry from slow, dive grind above 29, cap guard, freefall ignition | lakitu start (45.8 u/f), BoB slope, speed sets, tasplay |
| `dive-grind-abc.pol` | no A anywhere; B does the rollout; freefall ignition | BoB king start 45.9 -> 43.8 u/f, speed sets, tasplay |
| `dive-grind-gated.pol` | portfolio: grind <= p0 (48.5), `(switch speed-chain-3)` above; needs `dive-grind-chain` and `speed-chain-3` in the inventory | speed sets (ties/+1 vs speed-chain-3), slope |
| `edge-grind-hold.pol` | keeps a freefall <-> freefall_land grind (stick on facing + lean), optional B dive from freefall at fv >= p1 then A on first slide frame | parse/trace only; grind states cannot be entered in the speed/lakitu start sets |
| `edge-grind-hold-obj.pol` | same for held objects (recognised by holding/hgt/vy: the DSL has no hold_* action names) | parse/trace only, no held grind geometry found |
| `hold-hop-chain.pol` | carrying: A hops (facing within p0 of the target) instead of hold_walking | BoB 2012@26609 holding a bob-omb: 25.9 u/f (frames 30-60) vs 9.7 hold-walking; tasplay |

## 5. Method notes / traps for the next person

* `--dir` of `policy` must be a directory you may create `incoming/` and `archive/` in (`Inventory::open` creates them), and `switch` targets are read from `index.json` entries, not from `*.pol` files lying in the dir: a missing target silently does nothing (an early "gated" run looked identical to plain sliding because of it).
* `act-in` accepts only names in `src/actions.rs` (`hold_*` are missing: traces print them as hex and a `.pol` cannot name them).
* Walking / landing frames are the only turning chances; the grinds are straight lines. Wrap with `tag mpc:...` or choose `p0`/`p4` to bend them.
* New tools (nothing else in the repo changed): `policy/src/bin/grind_scan.rs` (scan movies for held / slow states), `policy/src/bin/grind_edge.rs` (edge-grind search, `REPLAY`, `CTRL`, `LEVELS`, `ONLY_HOLD` env). Built in `/tmp/claude-1000/target-grind`.
