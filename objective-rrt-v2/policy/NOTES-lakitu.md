# Lakitu skip on the real stack (re-wire of the authored policies)

## Why the authored policies fell from 17/21 to 1/21 (root causes)
1. `(field "z")` / `(field "x")` were position reads in the authors' private reader (`src/author_env.rs`); the real Reader has no field `z`, so it reads 0 and every window test (`z > -60`, `z < 37.4 fv - 2560`) is wrong. Port: `pos-z`, `pos-x`.
2. **A `full` cell without an explicit fine_mag is stick magnitude 56.5, not 64** (`menu::mag_range`: 56.5 + 7.5*fine_mag). Air speed per frame is `-0.35 + 1.5*intendedMag/32 - 1` with `intendedMag = (mag/64)^2 * 32`: mag 56.5 gives -0.18 u/f per frame, mag 64 gives +0.15. Over the ~80 air frames of the chain that is ~26 u/f, i.e. the double jump falls ~1000 units short. The authors' reader realized `full` as 64. `(cell A full aim target 0 1 0)` (fine_mag 1) restores it; `POLICY_FULLMAG=1` makes a `full` cell without explicit fine_mag default to 1 (off by default; `speed` problem is unaffected, see below). Recommend turning it on.

## What was added (shared tree)
- `Sense.hyp[4]` / `hyp_zone[4]` / `land_zone` (types.rs, sense.rs): on the ground, where each jump would land if pressed now (double jump, jump kick, single jump, long jump) through the same ballistic predictor as `land`, and whether that landing is in a zone. `sense::flat_range(kind, fv)`: flat-ground range of a jump.
- DSL observations (dsl.rs VARS, eval.rs): `dj-land-{x,y,z,fv,frames,zone}`, `kick-land-*`, `jump-land-*`, `lj-land-*`, `land-{x,y,z,fv}`, `dj-range`, `kick-range`, `jump-range`, `lj-range`; `land-zone` is now the predicted landing in a zone (before it equalled `in-zone`, and the zone list was always empty).
- Zones (eval.rs): on the castle grounds the Lakitu trigger box (x -544..545, y > 800, z -2000..-177) is the forbidden region.
- `POLICY_FULLMAG=1` (dsl.rs).
- Trace rows show the double-jump and kick landing predictions (`hyp dj z/y[Z] kick z/y[Z]`).
- `bin/lfuzz.rs`: the closed-loop fuzz of scratchpad/overlay_fuzz.md for any policy, same start families and rng as `hill_fuzz` (`--skip-a 1500` makes family B start i identical to hill_fuzz's B row i). Success = reached the castle / door circle with Mario never grounded in the box, Lakitu never talking, never in the moat, checked by an independent replay of the played pads (not the objective's own flag).

## Results
### lakitu_chain (21 train / 8 val / 13 test starts), one rollout per start
| policy | train | val | test | emu frames/start |
|---|---|---|---|---|
| authors' numbers, private reader (lak-2) | 17/21 | 6/8 | 9/13 | ~80 |
| lak-2 unchanged, real stack | 1/21 | 0/8 | 2/13 | ~55 |
| lak-2 ported (pos-z) + full stick (lak-2-r-fm) | 15/21 | 6/8 | 8/13 | ~90 |
| lak-6 (landing-prediction hops, steered air stick) | 14/21 | 6/8 | 8/13 | ~90 |
| **lak-6 + `mpc:8:13:70`** | **17/21** | **7/8** | **10/13** | ~8 000 |
| hand overlay replay (Worker "GA") | 21/21 | | 10/13 | ~5 000 (0.005 M) |

### hill-bottom fuzz (family B, 1501 starts: TAS f3029 state with the live Lakitu, position +-60, speed x0.90-1.04, yaw +-0x500; seed 11, same starts as the hand-overlay fuzz)
| policy | success | box-ground failures |
|---|---|---|
| hand overlay (hill_fuzz) | 815 (54.3%) | |
| lak-2 ported, stock (first 101 starts) | 1/101 | 94 |
| lak-2-r-fm | 747 (49.8%) | 575 |
| lak-5-fm (retired) | 673 (44.8%) | |
| lak-6 | 736 (49.0%) | 628 |
| **lak-6 + mpc:8:13:70** | **780 (52.0%)** | |
Per start, lak-6 vs hand: both 624, policy-only 112, hand-only 191. Family A (the spawn run-up, 84.8% for the hand overlay) is not covered: these policies start at the hill.
Failure cause (lak-6): 501 of 629 box-ground failures end in a slide-kick hop taken from z 0..400 at 50-80 u/f. That is the "gap" state: the first kick hop lands too far south for a double jump, and the fallback (Z) hops into the box. Starts that reach the bridge at < ~75 u/f cannot clear the box with any hop (double jump range ~ 0.8*fv*T), so they are lost upstream (the ramp-top speed), not by a wrong button.

### hybrid (repolicy / market / seeded) on lakitu_chain, ported library, 3 seeds (63 train / 24 val / 39 test)
| method | emu/start | train | val | test |
|---|---|---|---|---|
| policy only, 5 candidates | 0.001 M | 15/21 | 7/8 | 8/13 (1 seed) |
| seeded GA | 0.02 - 0.15 M | 46/63 | 21/24 | 25/39 |
| seeded GA | 0.56 M | 50/63 | 21/24 | 25/39 |
| repolicy | 0.02 M / 0.56 M | 49 -> 53/63 | 21/24 | 27/39 |
| market | 0.02 M / 0.56 M | 49 -> 53/63 | 21/24 | 27/39 |
| hand overlay replay | 0.005 M | 63/63 | | 30/39 |
The hybrid improves on the earlier stock-reader numbers (repolicy/market 47-48/63, 27-28/39 with the broken lakitu policies; seeded 33/63, 23/39) only in train; it does not beat the hand overlay replay, which stays ahead on this problem.

### tasplay
126/126 plays agree (lak-2-r, lak-6, lak-5-fm, 42 starts each; `gp/verify.py ... door`; includes the failures, claimed -1 = replay -1). Success in `lfuzz` additionally requires an independent in-process replay without a ground frame in the box or a Lakitu dialog.

## Verdict
- The authors' numbers are reproduced once the two bugs above are fixed (15/21, 6/8, 8/13 plain; 17/21, 7/8, 10/13 with a short-horizon stick search), and they still do not beat the hand overlay on `lakitu_chain` (21/21 train) or in the hill-bottom fuzz (52.0% vs 54.3%).
- What the policies do add: they are the only version that checks each hop's landing against the box (and do not hop into it when a safe option exists); `dj-land-*`/`kick-land-*` give any later policy that check.
- Remaining failures are gap states (speed < ~75 u/f at the bridge). They need either a better ramp-top phase or search (the GA operators barely move them: seeded/repolicy at 0.56 M reach 53/63).

## Inventory
Active (copies in `runs/inventory/incoming/lakitu-*.pol`): lak-6, lak-6-mpc-8_13_70, lak-6-mpc-8_13_40, lak-2-r-fm, X00001 (lak-6 with sampled margin). Retired, kept on disk with reasons in `runs/inv-lakitu/inv`: lak-2-r, lak-5, lak-5-fm, lak-7a, lak-7b, lak-8. Sources in `runs/inv-lakitu/src`, fuzz tables `f*.tsv`, comparison script `cmp.py`, hybrid logs `res/`, plays for tasplay `plays/`.
