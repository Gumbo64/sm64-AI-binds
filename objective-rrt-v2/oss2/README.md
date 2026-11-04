# oss2: scattershot v2

objective-scattershot (`../../objective-scattershot`) with its random-input rollouts ("pellets") replaced or mixed with objective-rrt-v2's GA, whose fitness is the discounted decrease of a generated objective's per-state ETA.

## What changed vs objective-scattershot

The two improvements over the original Scattershot (TylerKehne/sm64-tas-scripting) were meant to be (a) schema keys as the bins and (b) a GA instead of pellets. objective-scattershot has (a). oss2 adds (b) with v2's GA:

- **New expansion operator `ga`** (`src/ga.rs`). With `OSS_GA=p`, a share p of expansions works like this:
  1. Restore the cell's state and read it with objects.
  2. Generate v2 objectives (`obrrt_objectives::Registry`), keeping those with `base_criteria`, `exists_in`, a finite `start_eta` and not modal.
  3. Pick one by `obj_weight` × its kind's success rate in this run (`OSS_GA_LEARN`).
  4. Run `obrrt_search::ga::Worker::attempt_full` with obrrt2's default `GaParams`: scale 3, new ops, steps fitness, budgets from `ga_budget()` or 16/5/150, so up to 36 000 simulated frames per attempt.
  5. Merge what it reached into the archive exactly as a pellet's states (key, first-or-faster, sparse saves).
- **Pellets unchanged.** Noise, Fitted and PressThrough stay as they were. A busy cell (door, dialog, menu) always gets PressThrough. A modal cell, or one where no objective can start, gets a normal pellet instead, counted as a GA `fallback`.
- **Batches are all-GA or all-pellet.** A GA expansion plays about 200x a pellet's frames, so mixed batches would leave threads idle. `OSS_GA=p` is the chance that a batch is a GA batch. Use `OSS_GA_BY=frames` to make p a share of frames instead.
- **Archive.** `merge` is split into `merge_part` and `credit`. A GA expansion's rollouts (winner, plus ends) are each merged, and the cell is credited once. `Merged.improved` counts the cells whose best was replaced.
- **Stats.**
  - Per operator: `rollouts`, `succ`, `added`, `improved`, `frames`.
  - In stats.json:
    - `ga`: expansions, fallbacks, eligible objectives, met, succ, added, improved, frames.
    - `ga.kinds`: the same per objective kind, plus the learned rate.
    - `replaced` and `improved_per_mframe`.
  - The 5 s log line shows `replaced N (X/Mframe)` and an `operators` line.
- **Coarseness switches** (`OSS_GRID_ADD`, `OSS_PARAMS`, see below). They rewrite the schema at load. The run's `schema.json` is the schema as used.
- **Run options.**
  - `--frames N` stops at N emulated frames, for equal-frame comparisons.
  - `--rom FILE` or `OSS_ROM` sets the ROM. The default is `sm64-AI-binds/baserom.us.z64`, found from the crate dir, so the working directory doesn't matter.
- **Code layout.**
  - Not duplicated: `objective`, `objectives` and `util` (the schema tree and the 128 key-writing kinds) are re-exported from the vendored `../scattershot` crate, which obrrt-search already uses.
  - Copied from objective-scattershot: `archive`, `store`, `inputs`, `search`, `share`, `run`, the tests, examples, schemas and tools.
  - Left unchanged: obrrt-search and obrrt2. oss2 calls ga.rs's public `Worker::attempt_full`.

With `OSS_GA=0` (the default), oss2 makes the same random draws as objective-scattershot. tree.bin, cells.jsonl and the other outputs are the same as an oss run with that seed and thread count. stats.json only gains fields.

## Env switches

| Switch | Default | Meaning |
|---|---|---|
| `OSS_GA=<p>` | 0 | Share of expansions (batches) made by the GA. |
| `OSS_GA_BY=expansions\|frames` | expansions | With `frames`, a GA batch runs whenever GA frames < p × all frames. |
| `OSS_GA_HARVEST=winner\|ends\|all` | winner | Which states go to the archive (see below). |
| `OSS_GA_LEARN=0\|1` | 1 | Pick weight = `obj_weight` × (succ+1)/(tried+2) for the objective's kind. Succ means the expansion added or improved another cell. The table is read as of the batch start, so runs stay deterministic. |
| `OBRRT_GA_SCALE`, `OBRRT_GA_NEW`, `OBRRT_PINK`, `OBRRT_GA_BUDGET`, `OBRRT_GA_POP/_GENS/_HORIZON`, `OBRRT_GA_GAMMA`, `OBRRT_TECH` | obrrt2's | Passed to v2's GA and objectives as obrrt2 reads them. |
| `OSS_REPLAY_MAX`, `OSS_FIT`, `OSS_K`, `OSS_SELECT=flat`, `OSS_KEEP_STALE` | as oss | Unchanged. |
| `OSS_GRID_ADD=<k>` | 0 | Adds k to `grid_log2` of every objective with a position grid, using its default when the schema leaves it out. Root defaults to 10, Lens to 6, Boss to 8, Stage only when set. Positive k is coarser. |
| `OSS_PARAMS=<json>` | – | Sets params on every node of a kind, e.g. `{"Lens": {"speed_step": 0, "yaw_sectors": 4}}`. Applied after `OSS_GRID_ADD`. |

What each harvest mode merges:

| Mode | States merged |
|---|---|
| `winner` | Every state along the winner's committed inputs: the attempt's segments, up to its fitness maximum, plus the automatic tail through cutscenes. obrrt2 keeps the same states as a node. |
| `ends` | The winner's states, plus the end state of each rollout of the GA's last generation, including the winner's uncut end. These come through ga.rs's dense collector in `last_gen` mode. |
| `all` | The winner's states, plus every state along each last-generation rollout. |

All harvested inputs are replayed from the cell and keyed by the schema. The replay frames count in `frames`.

## Speed mode (make a piped obrrt2 route faster)

Scattershot's rule ("one state per key, replaced only by one that reaches it sooner") is DP over the schema's state abstraction. Speed mode feeds it obrrt2's best route and spends its effort beating the route's records. Everything is behind the switches below. With none of them, the run is the default one: tree.bin and cells.jsonl were checked byte-identical to builds/oss2a at 400k frames (`OSS_GA=0`) and at 600k (`OSS_GA=0.5 OSS_GA_BY=frames`). The code is in `src/speed.rs`.

| Switch | Meaning |
|---|---|
| `--pipe <obrrt run dir> [--pipe-node <id>]` | The route: the inputs from power-on to that node of the run's tree.bin. The default node is the earliest (lowest frame) of those with the most stars. Read with obrrt-search's `route::read_tree_bin` and nodepath's walk. Record 0 is obrrt2's boot (fixed or `OBRRT_TAS_BOOT`), so the inputs replay from power-on as they are. The replay is checked against nodes.csv (frame = input count, final stars). Every state is merged from the power-on cell, as `--seed` does, so each route key becomes a cell at its route frame. This replaces the usual menu-skip seed. Milestones are recorded: each star and key count's first frame, and each zone entry with (stars, keys) the route hasn't been in before. |
| `OSS_CORRIDOR=<units>` | A state is kept only within that distance of a route state of the same phase: zone, stars, keys and act. The check uses a grid of the route's positions per phase, bucket = width, 27 neighbours. The GA itself searches freely; only what is merged is filtered. |
| `OSS_SELECT=route` | Base cells. Half the time, the route state with the largest *gap* of 4 drawn uniformly. Gap = lead(t) − lead(t+150), where lead is the route frame minus the best frames for that state's key. This is uniform while nothing is saved, then biased to where a saving stops carrying downstream. A quarter of the time, one of the 64 most recently improved cells. Otherwise, any cell (with a corridor, every cell is a corridor cell). Draws only from the run's rng. |
| `OSS_GA_TARGET=route` | GA expansions (`OSS_GA`) beat a record. The cell is anchored on the route: its key's first route state, else the nearest route state with its zone, stars and keys. A target Δ ∈ [30, 300] frames ahead is drawn, up to 4 draws for one the cell can still beat. The objective is `ReachRoute`. Its ETA is `frames_for(distance)` in the target's phase. In an earlier phase on the way (another zone, before a star), it is the distance to where the route left that phase plus the route's frames from there. It is done on an equal schema key, or within 150 units, in the target's phase. It is Lost once `frame ≥ record` or `frame + 0.7·ETA ≥ record`. The record is min(route frame, best frames of the target's key). v2's GA runs with budget 16 / 5 / Δ+60 at scale 1 (5 generations), seeded with the route's own inputs from anchor to target (`OSS_GA_SEED=0` turns that off). Every state along the winner is merged first-or-faster. A cell with no anchor, at the end, behind every record drawn, or modal gets a pellet (a GA `fallback`). |
| `OSS_PROP=<share>` | Propagation. When a cell is replaced by a faster state, each child node of its old node is queued. Its inputs are replayed from the cell's current state, and every state is merged. A child key reached sooner is replaced and queues its own children, so the saving cascades (as reroute.rs grafts). It runs after each batch, in queue order, while its frames < share × all frames. The queue is capped at 65 536. The dedupe set holds only queued pairs: a pair is forgotten when taken or dropped, so it can't outgrow the queue (the reroute.rs leak). |

Outputs:
- **`stats.json` → `speed`.**
  - `milestones`: name, `route_frame`, `best_frame`, `saved_frames`. Best = the fewest frames of a cell with ≥ n stars/keys, or in the zone with those stars and keys.
  - `saved_s`: the final milestone's saving.
  - `saved_s_sum`: the sum over milestones.
  - `route_keys_faster` / `route_keys`.
  - `improvements_per_mframe`.
  - `prop`: queued, dropped, tried, frames, improved, added, queue, seen.
- **`route.m64` and `fastest.m64`.** The route, and the fastest path to the final milestone. Both are raw (no header) from power-on, like obrrt2's writer and tree.bin.
- **`cells.jsonl`.** Lines gain `route` (the key's first route frame), `saved` (route frame − frames) and `imp` (times improved) where they apply.

`tools/speed_eval.sh RUN_DIR FRAMES SEED` runs `OSS_GA=1 OSS_GA_TARGET=route OSS_SELECT=route OSS_CORRIDOR=600 OSS_PROP=0.1` (env THREADS, GA, CORRIDOR, PROP, NODE, SCHEMA, OSS2_BIN) into `runs/speed-<run>-s<seed>/` and prints the milestone table.

The schema is star120.json. It has a Stage with a 256-unit grid in every level and area, mission counters, and Root's level/keys/stars. zero_star.json has no stage in the courses, so BoB, WF and CCM would be keyed on Root's 1024-unit cubes only: a faster state with a different future would replace the route's state there.

In the viewer (`?oss=speed-<run>-s<seed>`):
- Queries `route >= 0` (piped route keys), `saved > 0` (beaten) and `imp > 0` (improved cells). sm64-viewer's `query.js` META gained `route`, `saved` and `imp`.
- Any cell plays from its row.
- The two paths play with `?m64=/oss/runs/<run>/fastest.m64` and `?m64=/oss/runs/<run>/route.m64` (add `&start=<frame>`).

### Smoke (q6-p-s1, 5 stars, 5M frames, seed 1, 2 threads, 2026-11-02)

See the table in `runs/speed-q6-p-s1-s1/` (`tools/speed_eval.sh ../runs/q6-p-s1 5000000 1`).
- 2 070 of 3 226 route keys were reached sooner.
- Many milestones were beaten by 1–8 s, and the second lobby → WF leg by 52 s.
- Savings mostly don't carry to the next milestone: the final milestone gained 0.37 s. In WF (4★) the 52 s lead holds for ~2 000 route frames, then is lost: propagated replays diverge (object phases, RNG). The GA has to re-bridge such breaks, which is what the gap selection targets.

## Key coarseness (which schema params set the bins)

Too fine a key means a faster path to "the same" state lands in a new cell instead of replacing the slower one. Watch `replaced`/Mframe and the per-operator `improved`. In `schemas/zero_star.json` the bins come from these params:

| Kind | Params | Effect |
|---|---|---|
| `Root` | `grid_log2` (10) | Position cube of 2^k units. Level, keys and stars are always keyed. |
| `Stage` | `grid_log2` (none unless set; zero_star sets 8) | Finer position inside the stage. |
| `Lens` | `grid_log2` (default 6 = 64 units) | Position inside the lens region. |
| `Lens` | `speed_step` (4; 0 = off) | Forward speed buckets. |
| `Lens` | `vspeed_step` (0 = off) | Vertical speed buckets. |
| `Lens` | `yaw_sectors` (16), or `yaw_toward` with `yaw_fine_sectors` (7), `yaw_fine_deg` (45) and `yaw_coarse_sectors` (3) | Facing buckets. |
| `Lens` | `actions` (class list) | Action classes. |
| `Lens` | `center`/`radius` or `min`/`max` | Region size: where the fine key applies at all. |
| `Lens` | `payoff_speed` | Outside the region, keys a speed bucket while faster than this. |
| `Side` | `radius` | Where the side bit is written. |
| `Boss` | `grid_log2` (8), `step` (250), `sectors` (16), `spin_step` (256), `dy_step` (100) | Positions, distance, facing, spin and height buckets. |

The Lens defaults are the fine part: 64-unit cubes × 4 u/f speed buckets × 16 yaws × action classes inside every lens.

Examples:
- Coarser everywhere: `OSS_GRID_ADD=1`.
- No speed or yaw in lenses: `OSS_PARAMS='{"Lens":{"speed_step":0,"yaw_sectors":1,"yaw_fine_sectors":1,"yaw_coarse_sectors":1}}'`.

## Running

```sh
cd sm64-AI-binds/objective-rrt-v2
nice -n 10 cargo build --release -p oss2          # target/release/oss2
cd oss2
# 3M frames, seed 1, 2 threads, half the batches GA:
RAYON_NUM_THREADS=2 OSS_GA=0.5 ../target/release/oss2 schemas/zero_star.json 0 1 --frames 3000000 --out runs/zs-ga05
nice -n 10 cargo test -p oss2
```

- Arguments: `<schema.json> [minutes] [seed] [--seed file.m64]... [--noise NAME] [--iterations N] [--frames N] [--out DIR] [--rom FILE]`. With neither `--iterations` nor `--frames` the run stops after `minutes`.
- Determinism: one seed and thread count give one run, as in oss. The batch size is the thread count, so 2 and 4 threads are different runs.
- Examples (`cargo run --release -p oss2 --example ...`):
  - `verify -- <run dir> [n]`: replays cells and checks their keys.
  - `inspect`.
  - `oss_share`.
  - `gen_star120`.

## Screening a run

- `tools/serve.py [port] [root]` (default 8091) serves `<root>/runs/*`. `root` defaults to this crate's dir, so runs written to `oss2/runs/` show up with no argument. For runs elsewhere, pass the directory that contains their `runs/` folder (or set `OSS_ROOT`).
- The sm64-viewer dev server proxies `/oss` to port 8091: open `http://localhost:5173/sm64-viewer/?oss=`.
- The outputs are the same as oss's: tree.bin, cells.jsonl, cells.meta, cells.grid, summary.json, objectives.json and stats.json. stats.json gains the fields listed above.
- For a quick read without the viewer: `stats.json` → `cells`, `replaced`, `improved_per_mframe`, `operators`, `ga`, `ga.kinds`.

## First smoke (zero_star.json, seed 1, 2 threads, 3M frames, 2026-11-01)

| run | cells | furthest zone (first frame) | replaced | replaced/Mframe | frames/s | GA exp / met |
|---|---|---|---|---|---|---|
| `OSS_GA=0` | 210 175 | grounds | 62 415 | 20 805 | 37 k | – |
| `OSS_GA=0.5` | 4 064 | grounds | 208 | 68 | 107 k | 108 / 17 |
| `OSS_GA=1` | 2 498 | lobby (2107) | 117 | 38 | 129 k | 98 / 18 |
| `OSS_GA=1 OSS_GA_HARVEST=ends` | 1 788 | lobby (2045, 1150 cells), BoB (2439) | 111 | 36 | 128 k | 60 / 13 |
| `OSS_GA=0.5 OSS_GA_BY=frames` | 118 103 | grounds | 36 793 | 12 119 | 51 k | 52 / 7 |

- No run got a key or a star.
- With the per-expansion share, `OSS_GA=0.5` spends 99.6% of its frames in the GA.
- Pellets read objects on every frame for zero_star's `while_object` lens, so they emulate slower. The GA reads them only for objectives that need them.
- Checks:
  - A second `OSS_GA=0.5` run gave identical tree.bin and cells.jsonl.
  - `OSS_GA=0` over 4000 iterations matched objective-scattershot's `oss` byte for byte.
  - `verify` replays GA-harvested cells to their keys.
