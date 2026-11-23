# 120-star run: plan, rules and state

Working notes for the 120-star goal (set 2026-11-09). Keep this current: it is what a new session reads first.

## Goal and rules (the user's)

- Construct a 120-star run with an algorithm loosely like objective RRT or scattershot that routes itself.
- Objectives must be in tree or scattershot form (code from the routes project may be reused; the run may be re-generated each time).
- No inputs from the TAS anywhere in the result. Explore routing and trajectories ourselves within each zone. Imposing prerequisites (wing cap, switches) is fine.
- Improve the tree search: macro objectives (100-coin stars and the like), RRT algorithm changes, micro improvements (input overlays).
- Ideal: RRT* for this game, where a section can be shortened and the saving propagates. Informed RRT* is an option. The game is not holonomic, so these don't apply directly.
- The objective-scattershot schema (oss2/schemas/star120.json) has per-star preconditions: use it to cut distractions, at some cost in novelty.

## Architecture: RRT* over funnels

Measured facts (RUNS.md, rows of 2026-11-09; logs/funnel/):

1. **Funnels.** At a zone entry (first controllable frame after a level or area change) Mario's state is identical across routes. What differs: camera yaw, RNG, objects in a level, and the previous frame's buttons.
2. **Transplants.** Route A's prefix + route B's suffix from the same funnel reaches B's next event on the exact frame in 90% of pairs (castle entries 151/152), when the cut is one frame before control.
3. **Composition.** Shortest path over segments between funnels, from 325 movies, gave new best routes for 1-6 stars with no search (logs/funnel/compose.py).
4. **Phase.** Time before a level loads is free; inside BoB the King route is locked to a 212-frame cycle (logs/funnel/delay.py).

So:

- **Global graph:** nodes = funnels, edges = segments (inputs between two funnels, with their effects: stars, keys, flags). Rewiring = shortest path; a shorter segment shortens every route through it. This is the RRT* part.
- **Local search:** a tree (obrrt2) per zone, started at the zone's entry funnel, produces segments. It explores by itself; no TAS path inside a zone.
- **Stitching convention:** a segment starts at a funnel with nothing held (the last input before control is neutral) and is cut one frame before the next funnel.
- **Menus** (star select, save prompt) are replayed by policy, not by recorded inputs, since the cursor depends on the stars collected.
- **TAS use:** the order levels are visited in, and reaching each zone's entry state so zones can be searched in parallel (tools/seg120/make_starts.py). The final run is replayed from power-on with only our inputs.

## Tools

| Tool | What it does |
|---|---|
| `builds/tools/funnel census` (run/src/funnel.rs) | One row per zone entry and per star of each movie |
| `builds/tools/funnel transplant` | A's prefix then B's suffix; does B's next event arrive |
| logs/funnel/compose.py | Segment library, shortest path to k stars, verified by replay |
| logs/funnel/delay.py, delay2.py | Delay probe (cycle locks) |
| tools/seg120/make_starts.py | Level-entry start states and logs/seg120/visits.csv (106 visits, 120 stars) |
| tools/seg120/level_trees.sh | Per-zone tree searches from those states (resumable) |
| tools/seg120/coverage.py | Which stars each per-zone tree found |
| spec/src/bin/mend.rs | Mender: makes a finished route faster |

## State (2026-11-10 03:35)

- **Campaign t1** (2026-11-10 07:45): 34 stars, **Bowser keys 1 and 2** (key 2 at route frame 98226, step 187) and the **wing cap switch**, in the castle at route frame 98548 (step 188). Since 31 stars: DDD 1, 2, 4. Earlier at 31 stars: Stars: BoB 1, 3, 6; WF 1, 2, 3, 5, 6; JRB 1, 2, 3, 5, 6; CCM 1, 2; BBH 3; HMC 1, 5, 6; LLL 1, 2, 3, 4, 6; SSL 1, 2; PSS 1; TotWC; SA; MIPS 1; Toad 1. Runs on builds/v3s with `OBRRT_GOAL_FOCUS=0.5 OBRRT_GOAL_WIDE=1 OBRRT_GROUNDS_FIX=1 OBRRT_STATIC_ID=1 OBRRT_STALL_ETA=0.03 OBRRT_GOAL_EXPLORE=1 OBRRT_ROAD=logs/road/t1` and the two priors. Skip list: BoB, WF, JRB, CCM, BBH, PSS, BitDW, LLL (cleared when the castle finds nothing else). Next: DDD star 1, BitFS and key 2.
- **Cap switches** (the user's rule: a cap course's switch is pressed on the visit). Since v3p a cap course counts as done only with its switch (0x2 wing, 0x4 metal, 0x8 vanish), and the campaign sends the castle back to a visited cap course whose switch is unpressed (`want_cap`), for the metal cap by way of HMC (`VIA`, untested in a run). **Wing cap: pressed at route frame 82630** (steps 177-178; replay-checked). Metal and vanish: courses not visited yet.
- **Castle between areas** (v3o+): `OBRRT_ZONE_PATH` hops taken from a replay of the route's own inputs (`Zone::Go`), and castle steps run with `OBRRT_GEO=1` (walkable-mesh paths: the roadmap's cost is never more than the straight line, which runs through the basement's walls).
- **Castle stars** (v3r, campaign `castle_star`): a castle step that takes Toad's and MIPS's stars when their gates are met, stopping on the star (`OBRRT_STOP_PROGRESS=1`). Toad 1 in the main route at step 176 (5.3 M frames); the stop needs a state in control (v3s).
- **Throughput (v3u, 2026-11-10 06:40)**: large course trees spent most of their wall clock on the main thread ranking goals while the workers waited (BitFS pre 6130 s of 7796 s): roadmap fields rebuilt per state for targets that move. Capped at 32 builds per round (`OBRRT_ROAD_BUILDS`): pre 165 s -> 3.1 s over the same rounds. t1 and sc1 switch to v3u at their next step.
- **Scout sc1** (from t1's route-0178, 31 stars; only WF, JRB, CCM, BBH searched; side route): **WF red coins (2, 4)** at step 4 after three checkpoints on `RedCoin::Next` (467 M frames); JRB under way.
- **100-coin stars: coin legs** (campaign.py `--coins <frames>`, test campaign coin1 from sc1's WF entry): short trees with `OBRRT_TWOSTAR=1`, each continued from the grounded state with the most coins, until the star (it doesn't end the visit); then a course tree with `OBRRT_ANY_EXIT=1` leaves, and its exit counts the star (exits judged against the campaign's flags). Fewer than `--coin-gain` coins in a leg ends the legs and the course goes on from its entry.
- **Side campaign sslcp** (from t1's route-0152): **SSL star 3 (pyramid)** found with `OBRRT_GOAL_EXPLORE=1` (23 stars at frame 64643); not in t1's route. SSL red coins: not found in 345 M.
- **Checkpoints** (campaign.py; builds/v3i, reworked for v3j): a course tree with no exit is continued by a fresh tree from a state further along one of its goals (summary.json `goals`). v3j keeps a list per goal (per amount of work in progress, the grounded state with the least ETA); the campaign takes, in order: a state in another level (a Bowser arena), then alternately the state with the best ETA progress and the one with the most work in progress. BitDW by hand first: from the entry three trees were at y -1100...190 after 50 M frames; from the first tree's furthest state the arena and the hit on Bowser took 21 M. The key needed `Bowser::Defeated`, 4.2 M frames. The first automatic checkpoint (SSL, step 153) crashed the campaign on a log line; fixed.
- **Experience roadmap** (`OBRRT_ROAD`, search/src/road.rs, builds/v3g+): cells and edges from every tree, kept per level in logs/road/t1; goal ranking by the cost of the known way; `Road::Leg` and `Road::Climb` attempts. Alone it did not reach the arena from the BitDW entry. **JRB red-coin star** with a roadmap seeded from the earlier JRB trees: 125.1 M frames (side campaign jrbcp, 20 stars at frame 52783; not in t1's route), where two trees without it gave none in 110 M and 174 M. One run.
- **SSL pyramid** (star 3): no exit in 123.3 M; 88% of the pyramid's nodes on the bottom floor, max y 1244 of 5050. The star was 11 separate goals (ids by pool slot): `OBRRT_STATIC_ID=1` in v3j. Under test on the side campaign sslcp (builds/v3j, 6 threads, from t1's route-0152).
- **Upstairs at 19 stars**: every course gave nothing at x1 budgets, JRB (174 M) and WF (124 M) nothing at x2. Missing there: red-coin stars (BoB, WF; JRB found on the side), 100 coins everywhere (WF with `OBRRT_TWOSTAR=1`: 81 coins at 300 M, no star; JRB without a coin goal: max 25), BoB 2 and 5, CCM 3-6, BBH 1, 2, 4-6, PSS under 21 s, BitDW red coins.
- **Audit** of every star's precondition chain: STAR120-AUDIT.md (item 3, JRB 2, is solved: the route has it; item 4, act select with six stars, is handled in v3j, untested).
- **Viewer**: `tools/seg120/viewer_publish.py` puts the campaign's route on sm64-viewer.

## Next

1. Red-coin stars: if the wip focus helps (w2 vs w1), turn it on in the campaign; otherwise a red-coin macro (order the 8 coins, ETA to the nearest uncollected one as a goal).
2. JRB ship: after the four chests, climb to the box (states after chest 4 need to be built on: wip focus, or a `Mission` spot gated on the drain).
3. Campaign hub: with every nearby course skipped the castle tree needs to find BitDW (8-star door) and the aquarium; now goals under `OBRRT_GOAL_WIDE`.
4. 100-coin stars (`OBRRT_TWOSTAR`, coins macro), Koopa races, cap stars, SelectAct quirk (audit item 4).
5. Retrain the attempt model on the campaign's own runs (same levels, current budgets) and re-benchmark; the value head (`OBRRT_ATTEMPT_SCORE=value`).
6. Segment library over logs/campaign/t1/exits; mend per segment; propagate.
