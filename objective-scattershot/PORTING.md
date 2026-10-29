# Porting objective-rrt's objectives to objective-scattershot

Goal: every objective kind in `../objective-rrt/objectives` (see its `OBJECTIVES.md`) has a counterpart that a scattershot schema (`schemas/*.json`) can use.

## They are different things
**objective-rrt** objectives are *goals*: `base_criteria` (when it applies), `base_heuristic` (distance to go), `success(traj)`, `progress`, `budget`, `steer_target`. The search tries a goal, scores trajectories, grows a tree.

**scattershot** objectives are *key writers*. The search keeps one state ("cell") per distinct key and expands cells by random inputs. An objective's whole job is to decide *which states count as different*: in `eval(st, cx)` it writes bucketed numbers (`cx.put(field, value)`); the key is the list of `(objective id, field, value)` written. Two states with equal keys are the same cell and only the faster is kept. There is no goal, heuristic, success test or budget. See `src/objective.rs` and the finished examples: `src/objectives/lens.rs` (regions, buckets, action classes, `while_object`, `caught`), `boss.rs` (nearest/held object by behaviour), `busy.rs` (door/dialog states), `progress.rs` (Root, Stage).

So porting means: **keep the state distinctions the rrt objective cares about, as bucketed key fields, gated to where it applies.**

## Mapping rules
1. **Gate = `base_criteria`.** Write nothing (return early) when the rrt objective would not apply (wrong zone, star already collected, object missing). Inactive objectives cost no cells.
2. **Per-object objectives become per-behaviour kinds.** rrt makes one objective per object instance; a schema is static. Take a `bhv` name (or list) like `boss.rs` does, use `needs_objects() -> true`, and key on the nearest matching object (or the held one). Use `obrrt_core::bhv_addr`, `Obj.bhv/pos/home/action/health/held_state/...`.
3. **Distance/heuristic becomes a bucket.** `bucket(straight_line_distance, step)` with `step` a parameter (rrt's `arrival_scale` or radii are good defaults). **Straight-line distance only** (no walking/geodesic metric). A couple of fields at most.
4. **What success/progress read becomes fields.** If success is "action is X" or "object health drops" or "holding", write that, bucketed: action *classes* (a few) not raw action numbers; hit points as is; flags as 0/1. Add the target's position on a coarse grid (`pack_pos`, `grid_log2` param, `pos_grid()` returning it) only if where it is matters.
5. **Techniques (`Tech::*`)**: region(s) (sphere `center/radius` or box `min/max`, as `Lens`) where the trick happens, plus exactly the quantities it needs (speed buckets only if the trick needs them, yaw relative to a direction with `yaw_bucket`, action classes, vertical speed). Reuse `Lens` itself where its params already express it, and say so instead of duplicating.
6. **Keep the number of distinct keys small.** Every extra field multiplies cells. Prefer 3-10 values per field. Defaults must be sane; schemas can tighten.
7. **Fields named `*pos`** are packed positions (`util::pack_pos`) and are decoded by the viewer with the objective's `grid_log2` param; return the grid from `pos_grid()`.
8. **Ids and `cx.all_children()`:** write your fields, then call `cx.all_children()` if the objective can have children (techniques are usually leaves; stage-like ones have children). Use `cx.absorb()` only for "all such states are one cell" cases (see `Lens` `caught`).
9. **Not portable?** Some rrt objectives are pure search machinery (macro budgets, policies, steering, kinship, `Mario::Settle` as an *action*). Do not fake them. For those: either find the state distinction they imply (e.g. "stopped" as a field) or leave them out and say why in your module doc.

## Project rules (from the project's standing instructions)
- SM64 knowledge lives only in `src/objectives/`; the engine (`objective.rs`, `archive.rs`, `search.rs`) stays generic. Do not edit the engine; if you truly need an engine hook, say so in your report instead.
- Straight-line distances only. No tree traversal / optimistic selection over states.
- Do not depend on the `objective-rrt` crates other than `obrrt_core` (already a dependency): copy the logic you need, with its constants and a pointer to the source file.
- Kind names: the rrt name without `::`, e.g. `Door::Enter` -> `DoorEnter`, `Tech::Blj` -> `TechBlj`. Params are a `#[derive(Deserialize)]` struct with `#[serde(default = ...)]` defaults.

## Where to work
Each family has one file `src/objectives/<family>.rs`, already registered in `src/objectives/mod.rs`. Add your kinds to that file's `build(kind, params)` (return `None` for kinds you don't know). **Edit only your own file.** Do not touch `util.rs`, `lens.rs`, `mod.rs`, `tests.rs`, schemas or another family's file; define small helpers locally. Put unit tests in a `#[cfg(test)] mod tests` at the bottom of your file, using `crate::testutil` (`st`, `obj`, `with_objects`, `key_of`) for states and objects. Test: gating (silent when inactive), buckets (boundaries), per-object choice, and that two states that must be different cells get different keys and two that must be one cell get the same.

Document at the top of your file (module doc): for each rrt kind in your sources -> the scattershot kind that covers it (fields and params), or "not ported: why". Also put a small example schema fragment in the doc of each kind.

## Building and testing
All agents share one cargo target dir. Keep your file compiling at every step (another agent's build will fail otherwise). Run only `cargo test --lib <your_family>` in `sm64-AI-binds/objective-scattershot`. If the build fails in a file that isn't yours, another agent is mid-edit: wait 30 s and retry, up to 5 times. Do not run searches (`cargo run`/examples) — they use all the CPU. Do not use git.

## Status
All ten families are ported (`src/objectives/{generic,stars,interact,enemies,bosses,npcs,warps,tech_blj,tech_kick,tech_misc}.rs`, plus the earlier `lens`, `boss`, `busy`, `progress`): 128 kinds. Each file's module doc maps every objective-rrt kind to its scattershot kind, or says "not ported: why", and has example schema fragments. `cargo test` (130 tests) includes `every_kind_builds_and_evaluates`, which loads every registered kind with default params and evaluates it on plain states and on states with objects.

Known limits: the kinds are tested on constructed states only, not in real searches; a schema that uses many kinds multiplies cells, so add them one stage at a time; search machinery (budgets, steering, weights, macros) is deliberately not ported — see the hybrid-operator discussion in the project notes.
