# Deploying the policy inventory into the production GA (opt-in, off by default)

## What was built
- `search/src/ga.rs`: `trait SeedSource` (`seeds(base, start, save, stage, rng) -> Vec<(pads, frames_spent)>`, best first), `set_seed_source`, `set_policy_seed(Option<bool>)` / env `OBRRT_POLICY_SEED=1`, `OBRRT_POLICY_SEED_EXTRA=n`.
  `attempt_full` asks the source when on: with no caller seed the best plan is THE seed (so `attempt_open` plays it as is and breeds variants: frames dropped / chunk dropped / cut and steered);
  with a caller seed (oss2's route inputs) that seed stays and the source's plans enter as extra gen-0 individuals (the route-input seed is thus always the fallback). `n` extra plans in either case.
  The source's frames are added to `Attempt::frames`. Unset: nothing is read or drawn; checked byte-identical: `policy ga --pink 0 --scale 0.027` on 6 starts, frozen `clean/bin/policy` vs the new build, identical output (success, frames-to-Done, emu frames per start).
- `policy/src/seed.rs` (feature `ga`): `PolicySeeds` over `runs/inventory` (top-level `*.pol` whose `index.json` status is `active` = 95, plus every `incoming/` and `incoming/done/` policy: 131 in all; retired policies stay on disk, nothing deleted). Each is rolled out once (`eval::play_policy`, `Env::full()`, an `ObjView` task = the objective's ETA, point 0 = `steer_target` else the problem's fixed point), `OBRRT_POLICY_DRAWS=k` adds k `draw_instance` lambda samples each. Order: finished by frames, then closest ETA. `done_only` option. Rollouts are deterministic without draws, so they are cached per (start, objective) for wall time only; the frames are still charged to every attempt (about 13 k frames, 131 rollouts x ~100 frames).
- `policy ga` (bench) installs the source when `OBRRT_POLICY_SEED=1`; env `OBRRT_POLICY_INV` picks the inventory dir.
- oss2: cargo feature `policy-seed`; `OSS_POLICY_SEED=1` (speed.rs `SpeedConfig::from_env`) installs the source with `done_only` (only policies that finish ReachRoute enter, next to the route-input seed). Compiles (`cargo build -p oss2 --features policy-seed`); NOT run end to end on a speed run, and ReachRoute rollouts by policies written for the bob-ramp task were not evaluated. A build without the feature and `OSS_POLICY_SEED=1` is an error.

## Measurement
Worker overlay GA (`policy ga --pink 0 --scale s --seeds 3`), the clean start sets (`runs/clean/cases`: fresh 59, field 45, noise 55 starts), 3 seeds, 2 threads.
Baseline = `runs/clean/res/*/wk_overlay_*` (same code path, same seeds). "+policy seed" = `OBRRT_POLICY_SEED=1` (emu frames include the ~13 k rollout frames). CIs: bootstrap over starts (1000 draws); paired difference = same starts and seeds.
Raw: `runs/deploy/res/<set>/seed_overlay_<s>.log`, `seed4_overlay_<s>.log` (+4 extra gen-0 plans; run for 0.027/0.067/0.2 only), `runs/deploy/summarize.py`, `queue.sh`, `queue2.sh`.
(`res_invalid_seed4_noop/`: a first extras run in which the extras were silently not wired into `attempt_open`; discarded.)


### start set `fresh`

| scale | variant | emu frames/start | success (95% CI) | frames-to-Done on successes (95% CI) | paired success diff vs overlay (95% CI) |
|---|---|---|---|---|---|
| 0.027 | overlay | 66,239 | 37.9% (25-50) n=59 | 140.3 (126-151) |  |
| 0.027 | +policy seed | 62,031 | 37.9% (27-50) n=59 | 164.4 (145-178) | +0.0 pts (-3.4 to +3.4) |
| 0.027 | +seed +4 extra | 62,422 | 37.9% (27-50) n=59 | 164.3 (147-178) | +0.0 pts (-3.4 to +3.4) |
| 0.067 | overlay | 117,837 | 38.4% (27-51) n=59 | 133.9 (121-144) |  |
| 0.067 | +policy seed | 114,932 | 39.0% (27-51) n=59 | 158.9 (141-172) | +0.6 pts (+0.0 to +1.7) |
| 0.067 | +seed +4 extra | 115,327 | 39.0% (27-51) n=59 | 158.8 (140-172) | +0.6 pts (+0.0 to +1.7) |
| 0.2 | overlay | 314,784 | 42.9% (31-55) n=59 | 125.4 (114-139) |  |
| 0.2 | +policy seed | 318,922 | 41.2% (29-54) n=59 | 138.4 (126-150) | -1.7 pts (-5.1 to +1.7) |
| 0.2 | +seed +4 extra | 319,212 | 41.2% (29-53) n=59 | 138.2 (126-150) | -1.7 pts (-5.1 to +1.7) |
| 0.75 | overlay | 1,085,093 | 54.2% (44-65) n=59 | 128.4 (114-143) |  |
| 0.75 | +policy seed | 1,105,893 | 55.9% (45-66) n=59 | 137.9 (124-152) | +1.7 pts (-6.2 to +9.6) |

### start set `field`

| scale | variant | emu frames/start | success (95% CI) | frames-to-Done on successes (95% CI) | paired success diff vs overlay (95% CI) |
|---|---|---|---|---|---|
| 0.027 | overlay | 67,303 | 16.3% (7-27) n=45 | 93.8 (50-146) |  |
| 0.027 | +policy seed | 64,658 | 15.6% (7-27) n=45 | 71.5 (36-106) | -0.7 pts (-2.2 to +0.0) |
| 0.027 | +seed +4 extra | 65,103 | 15.6% (7-29) n=45 | 71.5 (35-107) | -0.7 pts (-2.2 to +0.0) |
| 0.067 | overlay | 122,673 | 19.3% (9-30) n=45 | 121.1 (68-191) |  |
| 0.067 | +policy seed | 119,533 | 15.6% (7-27) n=45 | 70.3 (36-102) | -3.7 pts (-8.1 to +0.0) |
| 0.067 | +seed +4 extra | 119,970 | 15.6% (7-27) n=45 | 70.4 (36-105) | -3.7 pts (-8.9 to +0.0) |
| 0.2 | overlay | 341,790 | 20.7% (10-33) n=45 | 119.9 (66-184) |  |
| 0.2 | +policy seed | 339,062 | 17.0% (7-29) n=45 | 85.0 (50-133) | -3.7 pts (-8.9 to +0.7) |
| 0.2 | +seed +4 extra | 339,474 | 17.0% (7-28) n=45 | 85.2 (52-133) | -3.7 pts (-9.6 to +0.7) |
| 0.75 | overlay | 1,236,080 | 23.0% (11-35) n=45 | 120.9 (76-175) |  |
| 0.75 | +policy seed | 1,239,037 | 20.7% (11-32) n=45 | 101.0 (67-144) | -2.2 pts (-8.1 to +3.0) |

### start set `noise`

| scale | variant | emu frames/start | success (95% CI) | frames-to-Done on successes (95% CI) | paired success diff vs overlay (95% CI) |
|---|---|---|---|---|---|
| 0.027 | overlay | 66,172 | 42.4% (30-55) n=55 | 154.4 (141-168) |  |
| 0.027 | +policy seed | 61,051 | 53.3% (39-67) n=55 | 181.8 (166-196) | +10.9 pts (+3.0 to +20.0) |
| 0.027 | +seed +4 extra | 61,440 | 53.3% (40-66) n=55 | 181.7 (164-197) | +10.9 pts (+3.6 to +20.0) |
| 0.067 | overlay | 117,736 | 44.8% (33-57) n=55 | 152.2 (138-168) |  |
| 0.067 | +policy seed | 112,038 | 53.9% (41-66) n=55 | 171.3 (158-185) | +9.1 pts (+2.4 to +17.0) |
| 0.067 | +seed +4 extra | 112,429 | 53.9% (41-67) n=55 | 171.3 (156-184) | +9.1 pts (+1.8 to +17.0) |
| 0.2 | overlay | 312,044 | 52.7% (41-65) n=55 | 143.0 (127-161) |  |
| 0.2 | +policy seed | 305,665 | 56.4% (44-69) n=55 | 149.4 (137-163) | +3.6 pts (-4.2 to +12.1) |
| 0.2 | +seed +4 extra | 305,997 | 55.8% (43-68) n=55 | 148.6 (134-161) | +3.0 pts (-4.8 to +10.9) |
| 0.75 | overlay | 1,051,455 | 67.3% (58-77) n=55 | 140.3 (125-156) |  |
| 0.75 | +policy seed | 1,045,361 | 64.8% (53-76) n=55 | 136.7 (124-149) | -2.4 pts (-12.7 to +6.1) |

## Reading (honest)
- It does NOT help on the whole. Paired success differences are inside the CIs everywhere except `noise` at the two smallest budgets: +10.9 points at 0.027 (CI +3 to +20) and +9.1 at 0.067 (+2 to +17), i.e. 53% vs 42-45% at about 62-112 k frames. By 0.2 it is +3.6 (n.s.) and at 0.75 it is -2.4 (n.s.).
- `fresh`: no change (+0.0, +0.6, -1.7, +1.7 points). `field`: slightly negative at every budget (-0.7 to -3.7 points, CIs touch 0): the seed finishes some starts quickly but the policy-descended population then lacks the overlay's diversity and the early stop on a finished seed ends the search before the overlay would have solved the rest. At 0.75 M: fresh 55.9 vs 54.2, field 20.7 vs 23.0, noise 64.8 vs 67.3: nothing. The Worker with its overlays keeps scaling; the seed neither speeds up nor limits that, within the noise of 3 seeds.
- Frames-to-Done on successes is worse with the seed on fresh/noise at small budgets (164 vs 140; 182 vs 154 at 0.027): the seed's policy route finishes (the attempt stops early) at about 160-180 frames where the overlay GA, run to its budget, finds shorter ones. The extra successes are slow routes. On field the seed's successes are faster (72 vs 94) because the set of successes differs (the policies solve the short ones).
- Emulator frames are essentially unchanged (-8 k to +20 k per start): the 13 k rollout frames are repaid by early stops.
- Extra gen-0 plans (`+4 extra`, top 5 distinct rollouts) change nothing measurable (identical to within 1 start in every cell): the best policy plan already carries what the others add.
- Against the standalone policy hybrid (RESULTS_new.md, 60 k frames/start): fresh 37-39% = overlay Worker alone 37.9% at 66 k; field 15.6% vs 16.3%; noise 40-42% vs overlay 42.4% and overlay + policy seed 53.3%. So on fresh/field the overlay Worker already matches the hybrid at the same frames; only on noise does the combination exceed both.
- Not done: a `repolicy` operator inside the Worker (cut a member at a random frame and let a policy finish it). It needs a mutation hook inside `breed`/`child` with the policy source available per generation (rollouts of ~100 frames each from mid-trajectory states); it is the component that gave 63/63 in the standalone hybrid, so it is the next thing to try, but it is not a cheap change in `attempt_open` and was not attempted. Lambda draws (`OBRRT_POLICY_DRAWS`) were not measured either.
- Recommendation: leave it off for the speed problem by default. If anything is enabled it is `OBRRT_POLICY_SEED=1` for small budgets (<= 0.07 scale) on starts perturbed by noise; do not enable at 0.2+ or on field starts. oss2 wiring is untested for benefit.
