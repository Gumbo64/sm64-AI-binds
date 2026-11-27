# objective-rrt-v2: how objectives are written

v2 is a copy of objective-rrt. The engine (tree, pool, rate sampler, dense adds, model hooks, switches) is unchanged. Two things changed:

1. **An objective is a per-state ETA in frames.** `Objective::eta(s, start, mem) -> Eta`, in `core/src/lib.rs`, replaces v1's `base_heuristic`, `arrival_scale`, `success`, `success_heuristic`, `progress` and `budget` (the `Ga` part).
2. **The GA's fitness is the discounted decrease in ETA** (`search/src/ga.rs`, `attempt_open`). Everything else is v1's open-loop GA: seeding, operators, elite and budgets (`ga_budget()`).
   - `r_t = eta(s_{t-1}) − eta(s_t)`, plus a success bonus on `Done` and a penalty on death or `Lost`;
   - fitness = `max over t of Σ γ^i r_i`, with γ = `OBRRT_GA_GAMMA` (0.99), from the attempt start and across chained stages;
   - the segment is cut at the winner's argmax frame.

   A rolling-horizon controller is also in the code, off by default (`OBRRT_RH=1`, see the end of this file).

Why: v1's `progress()` was the peak over a whole trajectory. That ignores when progress happens (a peak at frame 230 equals one at frame 20), what happens after the peak, and death. The old curiosity GA (`rust/src/population.rs`) controlled Mario far better with a rolling horizon and a 0.99/frame discount. A per-frame discount needs a per-state value, hence the ETA. The same function also ranks starts (`start_eta`), so start ranking and fitness can't disagree.

## The trait (v2)

| Method | Status |
|---|---|
| `kind`, `key`, `obj_weight`, `base_criteria`, `exists_in`, `modal`, `kinship`, `success_value` | unchanged from v1 |
| `eta(&self, s, start, mem) -> Eta` | **new**: the one value function |
| `fixed() -> Option<Budget>` | replaces `budget()`; only `Budget::Macro` / `Budget::Policy` (menus, macros) |
| `ga_budget() -> Option<GaBudget>` | hint: v1's `Budget::Ga` numbers, kept per objective (population, committed-length cap, simulated-frame cap); None = the controller's defaults |
| `precision`, `temperature`, `steer_target`, `steer_button`, `arrival`, `pad_overlay`, `needs_objects` | hints, unchanged in meaning; `temperature` is now in ETA frames (default `START_TEMPERATURE` = 60) |
| `base_heuristic`, `arrival_scale`, `success`, `success_heuristic`, `progress`, `budget` | **gone** |

Helpers in core: `Eta::{Done, Frames(f64), Lost}`, `Eta::frames()` (Done = 0, Lost = ∞), `PACE` (30 u/frame), `frames_for(dist)`, `Mem { f: [f64; 4], flags: u32 }`, `fold(obj, traj)`, `met_at(obj, traj)`, `start_eta(obj, s)`.

## Rules for `eta`

1. **Honest frames, one unit for everything.** Distances become frames with `frames_for(d)` (d / 30). Never use bare distances, 1e5-style phase constants or other scores: one γ, one success bonus and one start temperature must mean the same thing for every objective.
2. **`Done` exactly where v1's `success` held.** Keep v1's predicate, evaluated on `s` (and `start` where v1 used `traj[0]`). `Done` only has to hold on the frame it's met: the controller ends the stage at the first `Done`.
3. **Phases add up.** For a staged objective (approach → grab → turn → throw; pound ×3; ...): `eta = eta_of_current_phase + Σ estimated frames of the remaining phases`. Pick the remaining-phase estimates so ETA **goes down at every phase transition**. A v1 lexicographic score like `holding*1000 - dist` becomes `if holding { THROW_FRAMES + ... } else { frames_for(dist) + GRAB_FRAMES + THROW_FRAMES }`. Write each phase constant as a named const with a one-line comment on where the number comes from. Rough is fine: they get calibrated from runs later (`tools/analysis/eta_calib.py`).
4. **Relative goals compare with `start`.** "More coins than at the start", "boss health below the start's", "this star not yet had at the start".
5. **Order-dependent goals keep state in `mem`.** `eta` is called once per frame, in order, from a fresh `Mem::default()` at the start, and on no other path. Use it for latches (`mem.flags` bits: "has grabbed", "kicked"), accumulators (`mem.f[i]`: swept angle, hits counted) and "frame something first happened". Never assume you see the future. Deadlines: once `s.frame` is past the deadline, return `Lost`.
6. **`Lost` only when it truly can't be met from this state:** the objective's zone was left without success, its object is gone, a deadline passed. The controller treats `Lost` like death. If leaving the zone **is** the objective (doors, paintings, warps), the ETA in the source zone is the approach, and the zone change is `Done`.
7. **Start ranking** is `start_eta(obj, s)` = `eta(s, s, fresh)`. With `start == s`, a relative goal must still give a sensible approach ETA there (not `Done`).
8. **`needs_objects`:** v2 reads objects on every frame for such objectives and folds `eta` over every frame. v1's shortcut of checking only `[first, last]` is gone, so a per-frame success check now sees every frame.
9. **Keep everything else as it was.** Same `kind`, `key`, weights, `base_criteria`, `exists_in` and hint methods. Delete the v1 `budget()` (unless it returned Macro/Policy, which becomes `fixed()`). Remove now-unused helpers only if nothing else uses them.

## Worked example: `coin.rs`

```rust
fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
    if s.zone() != self.zone {
        return Eta::Lost;                       // rule 6
    }
    if s.coins > start.coins && s.dist(self.pos) < TOUCH {
        return Eta::Done;                       // v1's success predicate (rules 2, 4)
    }
    Eta::Frames(frames_for(s.dist(self.pos)))   // rule 1
}
```

v1 had `base_heuristic = dist / 400`, `arrival_scale = 400`, `progress = -min dist over traj` and `budget = Ga 16/5/150`. All of that is gone.

## Scope: castle + BoB only

v2 ports natively only the objectives the castle grounds, castle inside/courtyard and Bob-omb Battlefield can generate. The other modules are left out of the build and listed below, so they can be ported later.

### Not ported (left out of the build)

None of these can be generated in the castle grounds, castle inside, the courtyard or BoB (their behaviours or level checks don't occur there). Their `mod` lines and generator calls are removed; the v1 code stays on disk (and in `../objective-rrt`). Types that shared a file with ported ones were deleted from that file (v1 copy in `../objective-rrt`, same path).

| Kind | File |
|---|---|

## Optimiser (search side)

Default optimiser: v1's open-loop GA (`Worker::attempt_open`, search/src/ga.rs), unchanged (budgets from `ga_budget()` combined over stages as v1 did, gen-0 seeding, operators, elite, early stop, `OBRRT_PINK`, the move GA's policy-seeded plans and the calm input shaping, which are always on) except it ranks rollouts by the discounted ETA decrease above.

`OBRRT_RH=1`: the rolling-horizon controller instead (`OBRRT_RH_POP/_H/_K/_G/_G_PRECISE/_GAMMA/_STALL/_HOT/_MAXLEN`, `Worker::attempt_rh`, parameters in `RhConfig`); it ignores `ga_budget()`. Its gold-standard settings (pop 128, 600 frames, commit 1, pink noise) are the compute floor for a serious run.

Units in ETA frames: per-kind start cutoffs (learned as p90 + 25% + 30 frames), start improvement (`IMPROVE_FRAMES` = 7 in engine.rs) and `time_cost` (default 0.4 frames of ETA per frame of game time).

Bests are per (objective key, context), the context being the attempt start's keys, stars and collected set: meets with less progress don't bound meets with more. `events.jsonl` has `best_improved` lines (`"milestone": "progress"` from `note_best`'s (keys, stars) keys, `"milestone": "objective"` when an objective key is met sooner than before).

The speed-up and calibration experiments of earlier builds (faster-only successes, Beat objectives, branch and bound, the learned ETA calibration, the dense-add modes other than `last_gen`, free setups) were removed by the condense refactor: they lost, or were never on. RUNS.md keeps their results; the tag `archive/pre-condense` the code.

## Mario::PosTime near b

PosTime no longer targets b itself. From each b, 2 candidate points are drawn like PosDist's (r 300–2500, random angle, dy −300..900); a candidate is kept only if some tree node in b's zone and progress context (keys, stars, collected set with OBRRT_STARSET_KEY) has been within 150 of it, and its deadline is that record (the earliest such frame, minus `OBRRT_POSTIME_MARGIN`). Done = within 150 before the deadline; Lost past it, or when even 60 u/frame can't make it; ETA `frames_for(dist)`. Starts only from b's context. Key: zone, context, the point's 400-unit cell and the deadline's `OBRRT_POSTIME_BUCKET` (10 frames) bucket, so a tighter record is a new pool entry. `OBRRT_PACE` still applies its ladder deadline on top (rung hashed from b and the point).

- **Records** (core/src/records.rs): `obrrt_core::earliest_near(zone, ctx, p, radius)`, read-only, on a per-(zone, context) grid of 256-unit cells holding (frame, position) sorted by frame; a query checks exact distances over the overlapped cells. The search builds it only when the generator asks (`Generator::wants_records`), adds every node in `Tree::add` (merge order, so generation in the plan phase sees a deterministic tree) and installs it on the planning thread. Counted in `mem_bytes` and `OBRRT_MEM_DIAG`, not in the save budget.
- **Default change:** this alters v2's default trees (extra RNG draws per b, different PosTime objectives). Runs comparable with builds/v2g and earlier need `OBRRT_POSTIME=self` (the old PosTime at b; byte-identical to v2g). Drivers that keep no records (labs, oss2) fall back to the old PosTime.

## Free setups

Tried (GOALS.md of builds/v2i, RUNS.md rows "free setups"): scripted technique setups replaced by GA-driven ones with phase ETAs and variants. The switch and its code (`tech/freesetup.rs` and the per-objective branches) are gone; the setups are the scripted ones. The Dedupe note that went with it: the duplicate cell (engine.rs `cell`: zone, stars, keys, coins, 64-unit position, modal, fingerprint) and the dense key (the same less the object part) ignore speed, facing and action, so two hand-offs at one 64-unit cell with a different speed or facing merge.
