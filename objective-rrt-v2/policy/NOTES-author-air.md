# Air / geometry author notes (LLM policies)

Binary/trace used: a private build from a snapshot of the policy crate (policy2.bin / trace.bin in /tmp/claude-1000/air; private inventory `runs/inv-air`, copied to `runs/inventory/incoming` and `archive/air-author` at the end). Evaluation: `inv eval` (1-2 threads, nice). `Sense` was still a stub throughout, so every geometry quantity below is built from the St fields, `(field "x"|"y"|"z")`, `hgt`, `vy`, `fv`, `inact`, `ny`, `(held A)`. Sense probes/raycasts/land-frames read 0 until sense.rs lands.

## Results (train / val / test successes; lower frames better)

| policy | problem | tr | va | te | note |
|---|---|---|---|---|---|
| lakitu-hill (DSL.md example 2, hand) | lakitu_chain | 9/21 | 2/8 | 5/13 | no slide_kick_slide / crouch_slide rules, so stalls on early starts |
| lak-1 (retired) | lakitu_chain | 15/21 | 5/8 | 9/13 | full chain; pressed a fresh A in FREEFALL_LAND even when the DJ would land short |
| **lak-2** (active) | lakitu_chain | **17/21** | 6/8 | 9/13 | DJ pressed only if `-60 < z < 37.4 fv - (2520+p0)` (p0=40); A released on the kick's last frame only when the DJ from `z-fv` passes the same test |
| lak-3 (retired) | lakitu_chain | 14/21 | 4/8 | 8/13 | pessimistic landing model: blocks releases that worked |
| lak-4 (template) | lakitu_chain | 17/21 | 6/8 | 9/13 | wider margin sampling: 60 draws, none better than default (231.0 -> 231.2) |
| air-chain-1 | speed | 10/15 | - | 8/21 | port of the SpeedChain hand overlay |
| speed-chain-3 -> X00003 | speed | 12/15 | - | 12/21 | aim offsets (3 bands), sampled |
| speed-chain-4 -> X00005 | speed | 12/15 | - | 8/21 | 6 x-bands, 400 draws: fits train, does not generalise |

Of the 13 lakitu_chain test starts, 3 (full-120star-2026-U @1715/1728/1743) are the moat-bank part of the TAS (x 1000..4000, y 600..20): not the bridge chain, unsolvable by it. Effective test 9/10.
Baselines (policy agent): pink-noise GA at 0.56 M emulator frames per start solved speed 15/15 train and 21/21 test; the policies are single rollouts (~70-100 emulator frames per start), far cheaper, but not yet as reliable on `speed`.

## What the traces showed (decomp facts used)

- A fresh A in any landing frame (FREEFALL_LAND etc.) is a double jump (common_landing_cancels aPressed, mario_actions_moving.c:1794): never press a fresh A there unless the DJ is wanted. Holding A (no edge) is harmless; a jump kick refresh needs A held + B fresh on the 4th landing frame.
- To get a fresh A at landing, release A on the jump kick's last air frame (`hgt+vy<3 and vy<0`), else the release costs a ground frame (~90 units).
- The landing step covers only part of fv (observed 21 of 85 units): "next z = z - fv" overestimates progress; real landing z is between z-0.2fv and z-fv.
- DJ range (30 traced doubles, press z to landing z): range = 41.3 fv - 841 where fv is the speed on the press frame. DJ needs landing z < -2000 (box end): press z < 41.3 fv - 2841. At fv 70 that window is nearly empty (z in -60..-20), at 85 it is -60..670.
- Gap states: after a kick hop from the rollout landing (z0 ~1300-1500 at fv ~82) the landing z ~600 is too far south for a DJ (bound ~500) and too far north for a safe second kick (lands inside the box). All remaining lakitu_chain failures I traced (cases 3, 12, 33, 1) are this gap, plus two starts that are already slow (fv<30 / turning_around). The policy cannot create the missing 100 units without losing fv (the DJ range is 41 units per u/f), so these starts need an earlier decision (different chain) or a different mechanic.
- act_forward_rollout zeroes forward speed on any AIR_STEP_HIT_WALL (mario_actions_airborne.c:1413): the `speed` failures at x~-4129 (and ~x 500-700, z ~6100) are rollouts/kicks clipping an obstacle; the heading cannot be changed in the air (update_air_without_turn), so the heading must be fixed on the ground earlier. Aim-offset sampling per x band fits the training starts but does not generalise (test 8/21 vs X00003's 12/21).

## Missing features (for the DSL / sensing)

1. Sense.land (frames, floor, zone) and wall raycasts along the velocity: replaces the hand landing arithmetic above and lets the policy see the obstacle before the hit.
2. A `heading-error-to-waypoint` observation for ground states and a waypoint list (pts[1..]) in the DSL: the chain needs to steer on the ground to a gate, not at the final target.
3. sqrt / atan2 in float expressions (landing-time solving, yaw to a point); float `if` (select) so a parameterised offset by band can be written without duplicating the rule list.
4. A per-start result API for lambdas: `inv sample` scores on the whole train set; for a position-conditioned map a coordinate/CMA search over p0..p5 would beat uniform draws.
