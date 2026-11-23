# Policy ecology notes (dsl.rs, eval.rs, evo.rs, inv.rs, bench.rs, bin/policy.rs, DSL.md)

## What exists
- **DSL** (`DSL.md`): typed s-expression trees. Koza GP trees and rule lists are the same thing (`rules` is sugar for nested `if`).
  Outputs: `(cell BTN MAG ANG REF ...)` menu cells, `(skill NAME params...)` (built-ins + the 31 skills.rs skills + `menu` = k-th cell of menu.rs's reduced menu),
  `(switch NAME)` (portfolios), `(tag ...)`. Terminals: Mario state, Sense (wall rays, edge/ledge, landing prediction, objects), waypoints, base-state reads (`start-*`),
  any Reader path via `(field "...")` / `(obj CAT K "field")` (resolved once per play), `p0..p7` bound args, deterministic `rnd`, float `sqrt/sin/cos/atan2/fif`.
  `(ahead N CELL "field")` (what-if) parses but evaluates to 0: not implemented, by the coordinator's priority.
- **Operators** (`evo.rs`): ramped half-and-half, subtree crossover, mutation by type (subtree, constants, conditions, leaves), rule insert/delete/reorder/swap,
  rule-list crossover, rule-fragment graft between policies, constant tuning of elites, portfolio gates.
- **Ecology** (`inv.rs`, `runs/inv*/`): append-only `changelog.jsonl`, `index.json` (lineage, origin, niche, per-start fitness, best_on, usage, credit, status),
  `<id>.pol` per policy. Niche = (air share, slide/landing share, mean speed) bins = 64 cells; admit if new niche, better than the niche elite, or best on at least
  one training start. Market share = origin prior x credit x niche rarity x specialist bonus + exploration bonus for rarely used, floor of half a uniform share each,
  LLM-authored policies guaranteed >= 25% in total. Operator weights are adapted by admissions per try and renormalised. Retirement is a status
  (`inv retire/revive`); auto-retire only touches non-authored, dominated, never-credited entries. Inbox: `incoming/*.pol` (dependencies on `switch` targets wait),
  bind/sample/lambda promotion (`inv sample <id>`), `inv eval file.pol -v` for authors.
- Integrated: menu.rs (`types::realize` -> `menu::realize`), sense.rs (+Reader), skills.rs. MPC (`mpc.rs`) is **not** wired into policy rollouts
  (too costly per frame; per its notes it belongs at a few decision points or as a final polish): not done.

## Baselines: per-start GA, one search per start (problem `speed`: 15 train / 21 held-out starts, 3 seeds each)
`policy <rom> speed ga --pink P --scale S` (runs/baseline/*.json). pink=1 is the pure pink-noise GA, pink=0 the overlay GA.

| GA | emu frames / start | train success | train mean frames-to-Done | test success | test mean frames |
|---|---|---|---|---|---|
| pink-noise, scale 0.75 | 0.56 M train, 0.72 M test | 44/45 | 71.1 | 61/63 | 96.4 |
| pink-noise, scale 3 | 2.0 M / 2.5 M | 44/45 | 68.0 | 61/63 | 86.9 |
| overlay GA, scale 0.75 | 0.43 M / 0.57 M | 45/45 | 69.2 | 62/63 | 85.6 |
| overlay GA, scale 3 | 1.6 M / 2.1 M | 45/45 | 67.0 | 63/63 | 85.5 |

(scale 7.5 and `speed2` baselines were still queued, runs/baseline/queue.sh.)

## Policies (trained once, evaluated with ONE rollout per start)
Best of each run after 14 minutes of continuous generation, 4 threads, with menu + sense + skills + the authors' 18 `.pol` files in the inbox
(final5-*.txt). Training cost = emulator frames of the whole ecology run.

| problem (train starts) | policy | train success | train frames | test success | test frames | emu/start at test | training emu frames |
|---|---|---|---|---|---|---|---|
| speed (15) | R01067 (rule GA) | 14/15 | 69.4 | 9/21 | 67.3 | 81 | 66.6 M |
| speed (15) | K01076 (Koza) | 15/15 | 91.2 | 11/21 | 90.8 | 83 | |
| speed (15) | P01050 (portfolio) | 14/15 | 85.0 | 10/21 | 91.9 | 83 | |
| speed (15) | lak-2 (LLM) | 12/15 | 79.2 | 11/21 | 112.4 | 92 | |
| speed2 (44, +10 val) | R00578 (rule GA) | 31/44 | 79.9 | 11/21 (val 5/10) | 89.9 | 97 | 71.8 M |
| speed2 | f06-refresh-frozen (LLM) | 27/44 | 84.1 | **14/21** (val 5/10) | 107.8 | 104 | |
| speed2 | P00587 (portfolio) | 30/44 | 76.3 | 10/21 | 78.6 | 76 | |

Earlier runs (builtin skills only, 12-25 min): speed best 14/15 train, 9-10/21 test; speed2 28/44 train, 8-12/21 test.

## Verdict
- **The policies do not beat the pink-noise GA on success.** Held-out success is 9-14/21 for one rollout versus 61/63 (97%) for a GA run on each start.
  On the training starts the best policy matches the GA's frames (R01067: 69.4 vs 68-71) but not its success (14/15).
- **They are far cheaper per start**: ~80 emulator frames per held-out start against 0.5-2.5 M for the GA, and the policy is reactive (it can be replayed from any state).
  Equal-total-frames comparison: 67-72 M training frames equal 30-130 GA searches at 0.56-2.5 M, i.e. the policy training budget is about what the GA spends
  to solve all 36 `speed` starts once (36 x 2 M = 72 M); the GA then has a solution for each of those starts, the policy a 25-50% solution for unseen ones.
- Failures are the held-out starts around (-2000..-4100, 6000..6900): obstacle/solid-object states where one more hop needs a precise kick angle.
  Authors' notes agree (sense had low wall precision; planner/MPC for the kick angle was not available inside rollouts).
- The ecology works as designed: 36 niches covered, 12-18 policies hold a best-on-a-start slot (specialists survive next to generalists), operator weights
  moved (gate-mut / graft up, random down), LLM share held at 25%; the portfolio is not clearly better than its best member (P01050 10/21 vs R01067 9/21 on
  test; on speed2 portfolios 10/21 vs rule GA 11/21), so the meta layer has not yet earned its keep.
- Origin of the best by train fitness: speed rule GA > Koza > portfolio > LLM; by held-out success on speed2 the LLM-authored `f06-refresh-frozen` (14/21) was the best.

## What would move the number
1. MPC / planned kick angle at decision points (policy picks the state-dependent branch, MPC the continuous stick), as the skills and MPC notes both conclude.
2. Training on more diverse starts (speed2 helped test only for the LLM policies; GA-evolved ones overfit the 15-44 starts).
3. Using `(ahead ...)` what-if and object-collision-aware landing to choose between kick and double jump.

## Commands (run from objective-rrt-v2; binary runs/policy4.bin)
`policy <rom> <problem> gen --dir D --minutes M --threads N` | `eval <id|file.pol> --dir D -v` | `inv list|report|retire <id> --reason|revive <id>|eval file.pol [--admit]|sample <id> --draws N`
| `ga --pink P --scale S --seeds N --out file.json`.  Scripts: runs/baseline/queue.sh, runs/final.sh.

## Not done / caveats
- tasplay verification of policy plays into m64 movies (`play` command) was not written. Plays are deterministic from the start state, but nothing here is independently replayed.
- `(ahead ...)`, MPC-in-rollout and the CEM/elite-weighted lambda sampler (uniform/normal/choice only) are not implemented.
- Niche descriptors are coarse (3 features); `maintain()` retired few entries in these short runs.
- The scratch build workspace is /tmp/claude-1000/ws-policy; copies of sources in policy/src are what the shared tree has.
