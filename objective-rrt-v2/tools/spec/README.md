# Spec objectives, the validator, the packer, landmark routes

A separate experiment for a Eureka-style loop: an LLM writes objective specs (JSON) per route segment, a validator checks them against a known route, short search runs score them, and the LLM revises. Plus `legs`, which solves a route written as an ordered list of specs ("landmarks") leg by leg with v2's open GA.

| Part | Where |
|---|---|
| Spec interpreter (one generic `Objective` from JSON) | `spec/src/{lib,model,eval,json,actions}.rs` (crate `obrrt-spec`) |
| The format, every field | [FORMAT.md](FORMAT.md) |
| Tips for landmarks (the lint rules cite them) | [LANDMARK_TIPS.md](LANDMARK_TIPS.md) |
| Validator `spec-validate` | `spec/src/bin/validate.rs` |
| Landmark-route solver `legs` | `spec/src/bin/legs.rs` |
| Context packer | `tools/spec/pack.py` |
| One loop round (validate + screen + feedback; LLM stub) | `tools/spec/loop.py` |
| Example specs (BoB) | `tools/spec/examples/bob/*.json` |
| Example landmark routes | `tools/spec/routes/bob-chomp.json`, `tools/spec/routes/bob-king.json` |
| GA breadth: QD archive, several γ | `search/src/qd.rs` (`OBRRT_GA_QD`, `OBRRT_GA_GAMMAS`) |

Build: `nice -n 10 cargo build --release -p obrrt-spec` (binaries `spec-validate`, `legs`); obrrt2 and oss2 as before. `spec` is a workspace member but not a default member.

## Hooks (all off by default; off = byte-identical)

- `OBRRT_SPEC=<file.json or dir>`: obrrt2 (`run/src/main.rs`: `obrrt_spec::wrap(Arc::new(Registry))`) adds the specs to what `Registry` generates; oss2's GA eligible list (`oss2/src/ga.rs`) gets them too (`obrrt_spec::extra`), loaded and checked at oss2's start. A spec is generated from every b in a listed zone, outside dialogs and menus, whose gate holds (one pool entry per spec and variant). `OBRRT_SPEC_RUN=<run dir>` resolves `{"node": id}` targets for obrrt2.
- `OBRRT_GA_QD=1`, `OBRRT_GA_GAMMAS=...`, `OBRRT_GA_QD_ADD=1`: see "GA breadth" below.
- Checked: nodes.csv and tree.bin byte-identical to builds/v2h/obrrt2 with all of them unset (2 threads, 2 M frames, tools/screen.sh's env with the P arm, OBRRT_MEM_MB=1500, seed 1).

## The validator

```
spec-validate <spec.json|dir> <run_dir> [--node ID] [--schema SCHEMA.json] [--from F] [--to F] [--jump X] [--trace N] [--json OUT]
```

Replays the run's best route (the earliest node with the most stars, as oss2's `--pipe`; or the path to `--node`) from power-on with `read_tree_bin` and nodepath's walk, reading objects every frame. Per spec (variant):

- **gate**: the first controllable, non-modal route state where it holds; that state is the start the ETA folds from (a fresh `Mem`), as obrrt2 would start it.
- **done**: where it reaches Done, and `expect`'s first frame (the milestone the route actually reached) with Done − expect. More than 30 frames apart is a problem.
- **phases**: when each was entered, and the ETA just before and at each transition (a rise is a problem).
- **calibration**: actual frames left ÷ ETA every 5th frame from the start to Done (to `expect` if never Done): median, p10, p90, and the start ETA vs the actual frames. A median outside 0.5-2 is a problem.
- **ETA jumps**: a rise of more than `--jump` (30) frames from one frame to the next.
- **Lost on the route**: a bug when the route did it (it re-arms at the next gate-holding state and says so).
- **schema** (`--schema`, oss2 bins JSON): distinct keys over the segment from the start to Done, keys per 100 frames, and per phase transition whether the frames before and after it share a key (states the spec separates that one cell would merge).
- `--trace N`: every N-th (frame, phase, ETA, frames left, position, node).

It prints a short report and writes the JSON one (`--json`, default `<run>-validate.json`): per spec `gate`, `done`, `expect`, `done_minus_expect`, `phases`, `calibration`, `jumps`, `lost`, `schema`, `problems`, `ok`.

### The example specs on two routes

`spec-validate tools/spec/examples/bob runs/q2-star-s2 --node 12430 --schema oss2/schemas/star120.json` (q2-star-s2 pounds the post; its first star at 3316) and the same on `runs/q6-p-s1 --node 15157` (which clips the gate and never pounds):

| Spec | q2-star-s2 | q6-p-s1 |
|---|---|---|
| EnterBob (lobby → BoB painting; 2 phases, the 70-frame warp) | gate 2064, Done 2356 = expect (+0), calibration median 1.27 (p10 1.08, p90 1.38), start ETA 228 vs 292; OK | gate 2115, Done 2409 = expect, median 1.28; OK |
| BobRamp (spawn → the ramp's foot) | Done 2709 = expect, median 0.84 (0.76-1.03); OK | Done 2895 = expect, median 2.66: the route wanders 369 frames for a 268-frame line |
| ChompGate (post ×3 by `home_dy`, gate gone, star) | Done 3316 = expect (+0), phases posts 2473-2942, gate 2942-3123, star 3123-3316, median 1.46 (1.23-2.10), start ETA 618 vs 843; schema: the wait for the gate shares a key (the gate isn't in star120's key) | never Done (stuck in phase 0: the route never pounds), expect (star) at 3449: the validator shows the route took another way; Lost at 3567 (BoB left) |
| KingHit | gate never holds (neither route goes to the summit) | same |

Iterations that got there: EnterBob's single phase had median 2.25 (the warp after jumping into the painting takes ~70 frames the distance doesn't show), so it got a second, fixed-frames phase; ChompGate's gate wait was 60 frames, the route shows ~180 (the Chomp's lunge), median 1.93 → 1.46.

## The context packer

```
tools/spec/pack.py RUN_DIR --from MILESTONE --to MILESTONE [--node ID] [--out brief.md] [--validate report.json] [--no-docs]
```

MILESTONE: `f<frame>`, `n<id>`, `stars=<n>`, `zone=<level[:area]>` (names work), `flag=<course>:<hex>`, `start`, `end`. The brief has: the segment's start and end states (zone, position, stars, action by name, star flags); the route's nodes in the segment (frame, Δ, zone, position, action, the `via` objective; `~d` = dense); the objects in each zone from the decomp level scripts (`OBJECT`/`OBJECT_WITH_ACTS` in script.c with their acts, `MACRO_OBJECT` via macro_presets.h, `SPECIAL_OBJECT` in collision.inc.c); optionally a validator report; FORMAT.md, this README, oss2's key-coarseness section and objective-scattershot's PORTING.md; and the ref/ lines that mention the level (by short name and decomp dir). E.g. `pack.py runs/q2-star-s2 --node 12430 --from zone=bob:1 --to stars=1` (~35 kB without docs).

## The Eureka loop

`tools/spec/loop.py` is one round without the LLM:

```
tools/spec/loop.py RUN_DIR --from M --to M CAND.json ... [--node ID] [--schema S] [--screen --frames 5000000] [--out DIR]
```

1. **Candidates**: k spec files per segment. `propose(brief, feedback, k)` in loop.py is where an LLM goes: given pack.py's brief and the last round's feedback.md, return k JSON texts (not implemented).
2. **Validate** each against the route (above). A spec that doesn't load is rejected with the parser's message (JSON path, "did you mean").
3. **Screen** (`--screen`): obrrt2 from the segment's start node (its inputs as the start .m64, `OBRRT_BOOT_NOWAIT=1`, 2 threads, OBRRT_MEM_MB=1500, the P arm) with `OBRRT_SPEC=<candidate>`, scored by the frames from the start to the first node that meets the `--to` milestone, the node count and the spec's attempts/successes (summary.json `spec.attempts`).
4. **Feedback**: feedback.md (per candidate: problems, calibration, Done vs expect, the screen; a reflection prompt) and feedback.json, ranked (fewest problems, then the fastest segment end, then calibration error); best.json is the best candidate.
5. **Keep the best**, feed feedback.md back to step 1, repeat.

Checked on three hand-made candidates (`loop1`): the example, the same with the gate wait at 20 frames (flagged: median 2.23) and one with a typo (rejected: `$.phases[0].repeat.count: unknown key "feild" (did you mean "field"?)`).

## Landmark routes and `legs`

A landmark route is a start and an ordered list of legs; each leg is a spec (FORMAT.md) whose Done is the next landmark. `legs` solves them in order with v2's open GA (`Worker::attempt_full`: fixed horizon, γ-discounted ETA fitness, steering / move-policy gen-0 seeding, no rolling horizon), from the previous leg's end states.

```
legs <route.json> [--out DIR] [--seed S] [--beam K] [--pop N] [--gens N] [--horizon-mult X] [--retry-mult X]
```

`legs` does obrrt2's own domain set-up (`obrrt_objectives::setup::from_env`, shared with run/src/main.rs: the steering controller, the movement library, `OBRRT_TECH`), and defaults unset switches to the P arm's movement set: `OBRRT_MOVE_GA_EXTRA=1 OBRRT_MOVES_MODES=1 OBRRT_MOVES_EXTRA=1 OBRRT_MOVES_SEEDED=1 OBRRT_TECH=all` (set one to 0 to turn it off; the movement library, the move GA, calm jumping and the other production settings are always on). `OBRRT_MOVES_SEEDED=1`: legs seeded from a reference play the move policies too (the GA turns them off under a seed otherwise, so on `seed_ref` routes none ever played), and each leg's first generation has `OBRRT_MOVE_GA_N` = population / 2 rollouts unless that is set: every plausible policy alone, then combinations. `move_ga.json` and the log's `move-ga:` line say what played and won. `MODES=0 EXTRA=0 SEEDED=0` is the old behaviour (lakitu-through reproduces its 620 frames byte for byte). `RAYON_NUM_THREADS` sets the beam's parallelism. (Before this, legs installed only the steering controller: no movement library or move policies, whatever the env said; the runs in the first results tables are from then.)

`route-stats <route.m64> [regions.json] [--json OUT]` replays a route from power-on and counts the frames in long jumps, dives, dive slides, rollouts, wall kicks, triple jumps, side flips, backflips, jump kicks (A+B), slide kicks, butt/stomach/crouch slides and ground pounds, per region and in total, plus King Bob-omb's throws and returns (act 5 entries).

### Route JSON

```json
{
  "name": "bob-chomp",
  "start": {"boot": "tas"},
  "reference": [{"run": "runs/q2-star-s2", "node": 12430}, {"m64": "ref/tas/full-120star-2026-U.m64", "frames": 12000}],
  "budget": {"population": 64, "generations": 30},
  "seed_ref": false,
  "legs": [
    {"name": "CastleDoor", "zone": "grounds", "dialogs": "stop",
     "phases": [{"target": {"point": [0, 803, -3155]}, "done": {"zone": "castle"}, "frames": 30, "arrival": "door"}],
     "budget": {"population": 64, "generations": 30, "horizon": 600}, "expect_frames": 170}
  ]
}
```

| key | meaning |
|---|---|
| `start` | `{"boot": "tas"}` (obrrt2's `OBRRT_TAS_BOOT` root: the 2026 TAS's first 1646 frames, castle grounds), `{"run": dir, "node": id}` (the inputs from power-on to that node), or `{"m64": file, "frames": n}` (raw or Mupen header, from power-on, cut at n). Must be controllable |
| `reference` | one or a list, same forms: replayed from power-on, each leg's spec folded in order from the start frame (a leg starts at the first controllable state after the previous one's Done) to get the reference's Done frame per leg. The first is the one `delta` and `seed_ref` use |
| `budget` | the route's default per-attempt GA budget: population (64), generations (30), horizon (default: the leg's start ETA × `--horizon-mult` (2) + 120, 60-5000) |
| `seed_ref` | seed each leg's generation 0 with the first reference's inputs between its Done frames (as `attempt_seeded`: the seed, then variants with frames dropped, a chunk dropped, or cut and steered) |
| `legs[]` | a spec (FORMAT.md; its `gate` is ignored, no `variants`) plus `budget`, `expect_frames` (a reference time, else the first reference's), `dialogs` |
| `reference` (on a leg) | a per-leg reference, same forms as the top-level one: a **redirect**, each leg compared with and seeded from whichever tree has that segment. Consecutive legs with the same reference fold as a run: the first from the zone entry when its predecessor's zones differ (else its start is unknown and only its Done frame is reported), each next one from the previous one's Done. route.json's totals add a "stitched" reference: each leg's own reference's frames summed |
| `native`, `key_contains` (a leg instead of phases) | a native objective: `{"name": "LakituHill", "zone": "grounds", "native": "Tech::LakituSkip", "key_contains": "hill"}`. Generated by `Registry::generate` from each beam state (and the reference folds); of that kind, with the substring in its key (a list: tried in order; "" = any), the one with the lowest start ETA. One generated anywhere is reused from states that may start it but don't generate it (the Registry makes objects' objectives only within 5000 of b; the tree starts them from b's kin the same way). Its own overlays, budget hints and Done apply; the leg's budget overrides the GA budget. Takes name, zone(s), native, key_contains, budget, reference, expect_frames, dialogs |
| `done_requires_next` | default true (the last leg: false): the leg is Done only where the next leg can start: for a spec next leg, Mario controllable, its start ETA finite and not airborne more than 400 above the floor (over a drop); for a native next leg, its objective (one seen so far, else generated there) may start. Until then the leg's ETA is 30 frames, so the GA aims for a usable end (a BLJ ending flying at the tree Tech::PoleRedirect takes). `false` for a leg that must end mid-air. Before it was the default, landmarks reached mid-jump off the path (doomed states: Ledge1 → Ledge2) and BLJ ends nothing could take on were kept |
| `require_next` | the older filter: candidates whose end the next leg can't start from are dropped after the attempt (kept 1 of ~35 BLJ ends) |
| `moves` | `"on"` (default: the env's movement set), `"off"` (no library steering, no policy genomes), `"kicks"` (generic steering, only the uphill-kick, speed-kick, jump-kick and steep-climb policies), `"policies"` (generic steering, every policy) for this leg |
| `dialogs` | `"avoid"` (default): only a dialog within 30 frames of the leg's start (a level's intro) ends a segment, and the GA avoids others (a stalled rollout scores nothing); `"stop"`: any dialog that opens ends the segment (Lakitu's on the castle approach, the King's intro and after-hit dialogs). Such a segment is pressed through and the same leg goes on from there (up to 8 times) |

Writing landmarks: put one wherever the path isn't a straight line the GA can find in one horizon: a bridge, an elevator top, a hill's top, a door. The open GA with a fixed horizon finds a 3000-unit walk easily, a detour around a wall rarely. Take them from the reference tree's route (pack.py's node table) or the TAS.

### Stitched routes (fast-king)

`tools/spec/routes/fast-king-blj.json` and `fast-king-walk.json` stitch three trees: the TAS boot → Hill Lakitu skip (native `Tech::LakituSkip` "hill") → castle door → BoB painting from runs/q8-g995-s1 (castle entry 66.8 s); then either the elevator BLJ (native `Tech::BljRun` "bob-elevator"), a spec leg steering the hand-off backwards into the TAS's tree, the tree redirect (native `Tech::PoleRedirect`) and a hyperspeed climb (native `Tech::HyperClimb`, "bob-summit" else any), from runs/q6-rrfobnb-s3 (which reaches the summit that way at 3552), and the walked landmarks left; or bob-king-landing's walked path; then the fight with the throw fix and the star from runs/q8-g995-s3. Every leg has its own `reference` and `seed_ref` seeds gen 0 with it. Budget 128 / 60, QD + 4 γ, `OBRRT_GA_REGRET=60`.

### How a leg is solved

A leg already Done at a beam state (e.g. a door leg after a skip that ended inside) passes that state on at 0 frames.

Per beam state (in parallel, merged in order): one attempt; the winner if it met the leg and, with `OBRRT_GA_QD=1`, the archive's met picks (distinct bins, at least the best met one per γ) are candidates. Each is replayed from its parent while the leg's ETA is folded (Done must hold on the replay), then through the cutscene tail and any dialog or menu (A on alternate frames: act select picks the first act, the save menu "Save and continue") to the next controllable state. Candidates are bucketed by (zone, 128-unit cell, facing octant, action), the earliest Done per bucket kept, and the K (`--beam`, 4) earliest go on. A parent with no met candidate retries once at `--retry-mult` (2) × generations; a leg no parent meets stops the route, reported with the best ETA left (and its state on stderr). Deterministic for a seed and thread count: every attempt's rng comes from (seed, leg, parent, try, round).

### Live outputs

The run dir is written as the run goes (atomically: tmp, then rename): route.json, paths.json, regions.json, summary.json and route.m64 after every leg, and every 30 s within a leg route.json (the solved legs plus the running one: `"status": "running"`, `ga_frames` so far, `best_eta_left`, `best_done_frame`; also under `running`), paths.json (the running leg's `legs[i].best_so_far: {points, parent}`) and route.m64 (power-on to the best state so far). Top level `"live": true` and `"updated": <unix s>`; `"live": false` at the end. Output only: the search never reads it. Finished legs have `"status": "done"` or `"failed"`.

### Outputs (DIR, default `oss2/runs/legs-<name>-s<seed>/`)

| file | content |
|---|---|
| `route.m64` | the best full path, raw from power-on (nodepath's format) |
| `route.json` | `start`, `start_frame`, `qd`, `landmarks[]` (name, zone(s), target point of the last phase), `legs[]` (name, from, to, frames, ref_frames per reference, expect_frames, delta, ga_frames, attempts, retries, distinct_done_bins, done_bins_by_gamma, kept_by_gamma, failed, budget, `beam[]`: per candidate done_frame, end_frame, leg_frames, parent index into the previous leg's beam, origin (winner/qd), γ, end state zone/pos/yaw/speed/action), `totals` (legs solved, frames, seconds, references, GA frames) |
| `paths.json` | `fields` (frame, level, area, x, y, z), `every` (3), `legs[]` (per beam candidate: parent and its points over that leg), `best` (the best path's candidate index per leg), `references[]` (points along each reference, and its Done frames) |
| `regions.json` | the legs of the best path as timeline regions: `[{"from", "to", "label", "note", "color"}]`, sm64-viewer's `normalizeRegions` format; the viewer's `regionsFor` reads a `regions.json` next to any .m64 that `public/replays/index.json` doesn't annotate |
| `summary.json` | `totals` |

Viewer: oss2's `tools/serve.py` (port 8091, the viewer's `/oss` proxy) serves `oss2/runs/<run>/`, so `http://localhost:5173/sm64-viewer/?m64=/oss/runs/legs-<name>-s<seed>/route.m64` plays the route. The viewer's Routes page, `http://localhost:5173/sm64-viewer/?routes=` (the runs, from serve.py's `/api/routes`) and `?routes=<run>` (the leg table, the map of every beam candidate's path, the references and the landmarks, a frame slider, Play route), shows the runs.

## GA breadth (search/src/qd.rs, search/src/gammaset.rs)

The open GA keeps one mode: γ decides which path is better, but nothing keeps the alternatives alive.

- **`OBRRT_GA_QD=1`**: a MAP-Elites archive inside `attempt_open_inner`. Each rollout's descriptor at its cut (the γ's argmax frame, or the Done frame when met): the 128-unit cell (`OBRRT_GA_QD_GRID`), facing octant, |forward speed| / 16 (≤ 7), action group (ground, air, swim, slide, other), the met flag; fields from `OBRRT_GA_QD_FIELDS` (default `pos,yaw,speed,action,done`; `side`: which side of the start → target line the midpoint passed). The archive keeps the best per bin by fitness, then the earlier end. Parents: with probability `OBRRT_GA_QD_UNIFORM` (0.5) a uniform archive pick (a γ, then a bin), else the fitness-ranked elite as before; the elite, population size, operators and the winner (best under the primary γ: the Attempt's segments) are unchanged. After the attempt, the K (`OBRRT_GA_QD_K`, 4) best distinct-bin rollouts, met ones first and at least the best met one per γ, are left for the caller (`qd::take_picks`, `legs`' beam) and, with `OBRRT_GA_QD_ADD=1`, offered to the dense collector (obrrt2 with `OBRRT_ADD`). summary.json `ga_qd`: bins and picks per attempt, met bins per γ, parents uniform/elite, dense offered.
- **`OBRRT_GA_GAMMAS=0.98,0.99,0.995,0.999`** (with QD): each rollout folds one discounted sum per γ over the same rewards (one fitness and argmax each); the archive bins by (γ, descriptor), ranked by that γ's fitness; γ 0 is `OBRRT_GA_GAMMA`.

## Throws and regret (`OBRRT_KING_LANDING`, `OBRRT_GA_REGRET`)

The open GA's fitness is the best prefix of the discounted ETA decrease, so whatever comes after the peak costs nothing: a King throw off his hill scores as a throw (the ETA drops to "landing") and his way back, his dialog and the regrab come after the peak; the segment is even cut at the doomed throw (the "peak then die" blind spot, RUNS.md q8 d200).

- **`OBRRT_KING_LANDING=1`** (objectives/src/boss.rs `Boss::Hit`; core/src/king.rs): while the King is thrown (act 4) his landing is predicted from his position this frame and last (kept in `Mem`) with gravity 4 and terminal speed 78 against the summit's floor triangles (BoB collision, y 4190-4294); the decomp's rule is that he loses health only landing less than 100 below home. Predicted off, already below, or jumping back (act 5), the ETA is the rest of the fight + his way back (105 frames; 52 in act 5) + DIALOG_128 (90) + the run back to his home + grab, throw and landing: no better than not throwing. Holding him, steering aims at the summit's middle. The spec side is the `thrown_off` predicate with a phase `penalty` (FORMAT.md; tools/spec/routes/bob-king-landing.json, examples/bob/04-king-hit.json). Bowser's throws are not covered.
- **`OBRRT_GA_REGRET=N`** (search/src/ga.rs, frames; off by default): after the fold, each prefix's value is its discounted sum minus the worst discounted rise of its stage's ETA within the next N frames, γ^u·(eta_u − eta_t), a death or Lost within N counting as a rise of `OBRRT_RH_DEATH`; the argmax (and the cut) is taken on these values, for the primary γ and each `OBRRT_GA_GAMMAS` γ. One pass of N per frame over the recorded rewards and ETAs (recorded only when on). When no prefix keeps a positive value after regret (every gain undone within N: dives sliding back down BoB's east slope), the raw argmax stays, so the attempt still keeps its best partial progress; before that fallback such attempts committed nothing, and legs reported them as "ETA left inf".

## Results (2026-11-02; 2 threads, scratchpad spec/)

**Identity.** With `OBRRT_SPEC`, `OBRRT_GA_QD*`, and `OBRRT_GA_GAMMAS` unset, obrrt2 built from this source gives nodes.csv and tree.bin byte-identical to builds/v2h/obrrt2 (2 M frames, seed 1, P arm, OBRRT_MEM_MB=1500; checked before and after the GA changes).

**`OBRRT_SPEC` smoke** (obrrt2, 5 M frames from q6-p-s1's BoB spawn node 2333 as the start .m64, `OBRRT_BOOT_NOWAIT=1`, the four example specs): the specs are generated and attempted: BobRamp 1 success / 2 attempts, ChompGate 0 / 9 (9 states added, mean ETA gain 185 frames per failed attempt), EnterBob 0 / 3 (after returning to the lobby); 144 of 898 nodes were made by spec objectives. No star in 5 M frames with or without specs (without: 1378 nodes). `tools/spec/loop.py --screen` runs the same per candidate.

**QD in obrrt2** (5 M frames from the grounds root, P arm): off 1286 nodes, `OBRRT_GA_QD=1 OBRRT_GA_QD_ADD=1` 2048 nodes. `ga_qd`: 50.7 bins per attempt, 3.7 picks (0.51 met) per attempt, 874 picks offered to the dense collector, parents half uniform half elite. Technique nodes (via, 128-unit cells): LakituSkip 266 nodes in 177 cells → 409 in 259; WallKick::Climb 47 / 37 → 78 / 65; JumpKick 36 / 20 → 83 / 64; SlideKick 27 / 22 → 58 / 44 (BljAny 21 / 15 → none in this run). Neither run left the grounds/lobby in 5 M frames.

**`OBRRT_GAMMA_SET`** (removed by the condense refactor: a per-kind γ draw that did not pay; the five-γ archive of `OBRRT_GA_GAMMAS` stays).

**legs on bob-chomp** (tools/spec/routes/bob-chomp.json: 8 legs from the TAS-boot root to BoB star 6 by the posts; budget 64 / 30 / ETA×2+120, beam 4, seed 1; reference runs/q2-star-s2, the obrrt2 route that pounds the post, star at 3316):

| leg | ref (q2-star-s2) | QD off | QD | QD, 4 γ | QD, beam 1 |
|---|---|---|---|---|---|
| Bridge | 164 | 103 | 110 | 110 | 110 |
| CastleDoor (Lakitu's dialog pressed through; the ref skips it) | 168 | 352 | 354 | 326 | 354 |
| BobPainting (+ act select, intro dialog) | 378 | 458 | 416 | 415 | 575 |
| Ramp | 353 | 313 | 336 | 337 | 335 |
| Elevator | 79 | 89 | 57 | 21 | 52 |
| ChompHill | 40 | 64 | 51 | 72 | 89 |
| ChompPost (3 pounds) | 114 | 326 | 160 | 202 | 111 |
| GateStar (the Chomp cutscene, gate, star) | 374 | 336 | 341 | 366 | 350 |
| **total** (frames from 1646 to the star) | **1670** | 2041 | **1825** | 1849 | 1976 |
| GA frames | | 2.9 M | 14.3 M | 16.0 M | 5.3 M |
| distinct Done bins per leg | | - | 137, 45, 17, 228, 252, 391, 24, 191 | 140, 38, 25, 203, 247, 148, 14, 191 | 137, 8, 3, 244, 78, 89, 12, 47 |

- Every configuration solves all 8 legs; QD off has one candidate per leg (the winner), so its beam is 1 wide. QD's beam is what helps: 1825 vs 1976 at beam 1 (QD) and 2041 (off). With the 4 γ's, γ 0.98-0.999 kept a different best than γ 0.99 in a few Done bins per leg (e.g. Ramp 12 / 11 / 15 of 203) and two of the kept beam states came from γ 0.98 / 0.995 / 0.999, but the total (1849) was not better than one γ (1825) on this seed.
- The run with QD and 4 γ reproduced exactly with the earlier binary (1849), and two runs of one seed gave identical route.m64 and route.json (determinism check at 16 / 4, beam 2).
- route.m64 (QD) replays from power-on to the star at frame 3471 (ref 3316: +155), back in the lobby at 3833. Most of the loss is the Lakitu dialog (+186: the ref uses the Lakitu skip, which no spec expresses) and ChompPost.
- Viewer: `http://localhost:5173/sm64-viewer/?m64=/oss/runs/legs-bob-chomp-qd-s1/route.m64` (copies in oss2/runs/legs-bob-chomp-qd-s1 and -off-s1).
- These five runs used the earlier `"dialogs": "stop"` rule (a dialog after 30 frames of ETA progress). With the current rule (any dialog; changed for the King), QD seed 1 failed at ChompPost (two of three pounds, ETA 27 left after 8 attempts and 4 retries) and seed 2 solved the route in **1880** frames (Bridge 114, CastleDoor 326, BobPainting 429, Ramp 341, Elevator 46, ChompHill 41, ChompPost 248, GateStar 335): `?m64=/oss/runs/legs-bob-chomp-qd-s2/route.m64`. ChompPost is the fragile leg at this budget.
- The 2026 120-star TAS doesn't do BoB this way (none of the legs' Dones fold on its first 12 000 frames), and the BoB IL TASes (ref/tas/il-bob-chomp-*, il-bob-king-3110) start from savestates, so they can't be replayed from power-on; their times are 9.97 s (Chomp gate, a clip) and 31.10 s (King).
**legs on bob-king** (tools/spec/routes/bob-king.json: 10 legs from runs/q8-g995-s3's BoB spawn node 2426 (frame 2504): 8 path landmarks taken from that run's route, the fight (grab ×3 / hit ×3 read from the King's health, his dialogs pressed through), the star; QD with the 4 γ, default budget, beam 4, seed 1):

| leg | ref (q8-g995-s3) | legs |
|---|---|---|
| Hill1 | 471 | 364 |
| ChompSide | 130 | 67 |
| PoleBase | 663 | 134 |
| Ledge1 | 143 | 127 |
| Ledge2 | 122 | 204 |
| BackPath | 114 | 90 |
| UpperPath | 141 | 83 |
| Summit | 256 | 123 |
| KingThrows | 1682 | 2004 |
| KingStar | 267 | 256 |
| **total** (frames from 2504 to the star) | **3989** | **3452** (−537, −17.9 s) |

The ascent is 1192 frames vs the reference's 2040; the fight is 322 slower. 52 M GA frames in all, 32.6 M of them on the fight (32 attempts, 11 retries, 5 dialog rounds). route.m64 replays from power-on to the King's star at frame 5956 (ref 6493); viewer `http://localhost:5173/sm64-viewer/?m64=/oss/runs/legs-bob-king-qd4-s1/route.m64`. A first try without path landmarks (spawn → summit in one leg) failed at ETA 154 left; one with the King's intro dialog under the old "stop after 30 frames of progress" rule stalled in the dialog (the rule is now: any dialog, with `"dialogs": "stop"`). BackPath's sign re-opens its dialog at each restart (8 rounds): the avoid rule's "dialog right after the start" case loops on a sign Mario stands at.

**Full move set** (legs now installs the movement library and move policies, objectives::setup; before, only the steering controller). route-stats, frames per move along the final route:

| run | total | long jump | dive | rollout | triple jump | wall kick | jump kick (A+B) | side flip | backflip | ground pound | crouch slide | legs with a long jump |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bob-chomp QD seed 2, before | 1880 | 62 | 84 | 78 | 35 | 2 | 0 | 0 | 0 | 53 | 5 | 2/8 |
| bob-chomp QD seed 2, full moves | 2110 | 89 | 76 | 56 | 31 | 12 | 9 | 0 | 0 | 95 | 37 | 3/8 |
| bob-chomp QD seed 2, full moves + REGRET 60 | 1922 | 28 | 105 | 100 | 18 | 19 | 0 | 0 | 0 | 56 | 5 | 1/8 |
| bob-king QD 4 γ seed 1, before (king3) | 3452 | 100 | 163 | 156 | 0 | 0 | 79 | 31 | 21 | 22 | 21 | 1/10 |

Per leg (chomp, before → full moves → + regret): Bridge 114 → 116 → 114, CastleDoor 326 → 317 → 362, BobPainting 429 → 519 → 522, Ramp 341 → 328 → 328, Elevator 46 → 78 → 85, ChompHill 41 → 81 → 35, ChompPost 248 → 322 → 140, GateStar 335 → 349 → 336. The library makes more long jumps and wall kicks and the first A+B jump kicks, but on this seed the routes are not faster (one seed each; ChompPost swings ±100 between runs). No slope technique (superjump) appears in these routes. With the full move set, bob-king fails on the ledges at 64 / 30 in all four configurations (base, throw fix, REGRET, both: Ledge2 / Ledge1b / Ledge1c, the rollouts die off the narrow path), and fast-king-walk fails at Ledge1c at 128 / 60, where the controller-only runs walked it. The King's old route (king3, controller only) had 5 throws and 4 returns (route-stats), the off-hill throws the throw fix is for.

**REGRET obrrt2 smoke** (5 M frames, grounds root, P arm): off 1286 nodes (the castle entered), `OBRRT_GA_REGRET=60` 1138 nodes (not entered in 5 M); `OBRRT_KING_LANDING=1` gives the off run's tree exactly (the King isn't reached in 5 M, the switch reads nothing else).

**fast-king** (TAS boot → Hill Lakitu skip → castle → BoB painting → … → King's star; stitched references; 128 / 60, QD + 4 γ, REGRET 60, beam 4, 2-4 threads; runs in oss2/runs/legs-fast-king-*, live):

| leg | reference | best so far | run |
|---|---|---|---|
| LakituHill (native Tech::LakituSkip hill) | 263 (q8-g995-s1) | **177** (−86) | every run |
| CastleDoor | 45 | 44 | |
| BobPainting (+ act select, intro) | 397 | 444 | |
| Ramp | 326 (q6-rrfobnb-s3) | 326 (212 in s4) | |
| Blj (native Tech::BljRun bob-elevator) | 103 (q6-rrfobnb-s3) | 87-91 (210 when its Done must face the tree) | s2-s6 |
| Redirect (native Tech::PoleRedirect, tree (4208, 927, 3772)) | 8 (TAS) | 7-33 | s2-s6 |
| Climb (HyperClimb to the second tree (6799, 2008, -5587)) | 30 (TAS f22610-22640) | 72 (TAS-seeded) | s5, s6 |
| Redirect2 (second tree) | - | 2 | s6 |
| Climb2 (to the summit) | TAS: double jump, wall kick, double jump | failed, 66 ETA frames short (rollouts top out at y ~3100 of 4293) | s6 |

What it took, in order: (1) the BLJ's own Done (the chain at the hand-off speed) doesn't aim at the tree: filtering ends kept 1 of ~35, a spec leg steering into the tree after it never met PoleRedirect's start (rollouts touched it in a jump kick or walking, which interact_pole doesn't take); `done_requires_next` (the BLJ is Done only flying at the tree) solved it. (2) The hyperspeed jump to the second tree fell 2400 units short from random / steered seeds; seeding it from the 2026 TAS's own inputs (a per-leg TAS reference) met it in 72 frames. (3) The second redirect failed on a replay bug (the candidate replay didn't fold the start state, so PoleRedirect's touch latch on it was missed): fixed. The summit climb is open; s7 climbs the generic +1000 instead and walks UpperPath → Summit → the fight.

The walked variants (fast-king-walk-s1b … s5) fail on BoB's east slope (Ledge1 → Ledge2) in every configuration tried: moves on, off, kicks-only, policies-only, extra landmarks, ground-only Done, seeded from the one run that walked it. Causes found: Ledge1's Done fired mid-jump off the path (fixed by done_requires_next's over-a-drop check and ground Done); the straight line to Ledge2 crosses the steep slope, where every rollout dives and slides back (ETA 48 short); with REGRET those attempts committed nothing (fixed by the fallback above). The only walked success remains legs-bob-king-qd4-s1 (3452 frames from BoB spawn, controller-only), which went up that slope on a retry.

- Budgets (both routes): these are the defaults (64 / 30, horizon ETA×2+120, beam 4). Full budgets I'd use without the benchmarks: population 128, 60 generations, beam 8, QD with the 4 γ, `seed_ref` from the best tree route, 8 threads.
