# LLM authoring log: gating / portfolio policies (author: Claude, fork)

Tools I added (new files only, nothing in the shared lib was edited):
- `src/author_env.rs`, `src/bin/author_eval.rs`, `src/bin/author_trace.rs`: an Env whose named-field reader answers object queries
  `(obj CAT K "field")` until sense.rs lands (cat: enemy|breakable|coin|star|pole|any; fields: dist dist3 dy relyaw x y z act; absent object: dist = 99999, others 0)
  and a per-frame trace (`author_trace <rom> <problem> <pol> <case|all> [-o frame]`, `-o` lists nearby objects).
  Build: `CARGO_TARGET_DIR=/tmp/claude-1000/target-author cargo build --release -p obrrt-policy --bin author_eval --bin author_trace`.
- Private inventory: `policy/runs/inv-author/` (all .pol files kept, superseded ones retired with reasons, none deleted).
- Deliveries to the shared inventory inbox: `runs/inventory/incoming/llm-*.pol` (gate-speed-8, lakitu-bridge-1, walk-target-1, swim-target-1, pole-climb-1, ledge-grab-1).

Problem `speed` (36 starts, one rollout per start, ~65-80 emulator frames per start): baselines on the same starts:
pink-noise GA at 0.56M emu/start 15/15 train (72.8 frames), 21/21 test (104.0); hand-steer 5/15, 5/21; hand-slide-kick 8/15, 9/21; hand-kick-chain 5/15, 4/21.

## Iterations (train x/15, test x/21, mean frames on solved)

| id | change and why | train | test |
|---|---|---|---|
| gate-speed-1 | regime rules: air at speed holds A, landing presses A first, then A+B (short stick) | 5/15, 61 | 4/21, 58 |
| gate-speed-2 | v1 pressed A+B on every landing frame so B was held, not fresh, at the 4-frame expiry where LAND_STOP cancels into the kick (three runs ended in walking at the 48 cap). Now A alone while landing, A+B only at expiry / first LAND_STOP frame and only if B was up the frame before | 4/15, 65 | 9/21, 88 |
| gate-speed-3 | dodge objects with the 22-degree Left/Right classes (BoB crates at x=-3979, z 6627/6827 stop a rollout dead; a goomba near (-2883,5960)) | 5/15 | 9/21 |
| gate-speed-4 | rollout cannot turn in the air (act_forward_rollout: update_air_without_turn); only the sideways speed of a stick at 90 degrees moves it. Dodge with side-left/side-right, away from the object | 6/15, 59 | 9/21, 88 |
| gate-speed-5a..f | dodge windows 700-1500 units, 14-60 degrees (5b: 10/15, 10/21; 5a: 9/15, 11/21; wider than 1200/45 does not help) | 9-10/15 | 10-11/21 |
| gate-speed-6a/b | slide-kick chain for crouch_slide (Z+B fresh) / slide_kick / slide_kick_slide (fresh A) / dive_slide on a flat floor (fresh A): four starts begin in those states and used to stop crouching; A held only low in the air (up for a wall kick); the dodge tests the path (perpendicular offset = relyaw*dist*0.01745 < 140/160) instead of the bearing | 8/15 (fit 115) | 11/21 (fit 96) |
| gate-speed-7-X | the chain is airborne at 90 u/f when it crosses the 400-unit goal circle, so Done (grounded, controllable) never fires, and the kick bonks the elevator-ramp wall (fv -40). Hand-off: inside X units of the target stop all chain buttons and steer. X=500/700/900/1200 | 10/15, 56 (fit 147) | 11/21, 86.5 |
| gate-speed-8 | v7-700 plus `(> dist 1)` guards: with the stock Env an unknown object reads 0 = "box at distance 0, bearing 0", which dodges forever. Same numbers where objects are readable; with the stock Env (no object reads) 8/15, 9/21 | 10/15 (obj) / 8/15 (stock) | 11/21 / 9/21 |

Problem `lakitu_chain` (21 train / 8 val / 13 test; needs `OBRRT_GROUNDS_FIX=1` (the Lakitu arc is always on)): hand-steer 0/21; gate-speed-8 0/21 (fit 73); hand-kick-chain 3/21.
- lakitu-bridge-1 = gate-speed-8 + the bridge decision of the hand overlay from the decomp: on the bridge approach (z > -170, -400 < x < 500, fv < 106), a FRESH A on a freefall landing gives the double jump when `z - (37.4 fv - 500) < -2030`; A held for the whole double jump; no arrival hand-off. Uses only `(field "x")`/`(field "z")`, which the stock Env has. **train 14/21 (65 frames), val 5/8, test 8/13 (64 frames)**; the hand overlay's reported held-out figure is 10/13 (RUNS.md gp exp. 2), the seeded GP 10/13, the per-start GA 10/10 of its cases.

Problem `follow` (follow64.txt, 384 held-out starts): hand-steer, walk-target-1 and gate-speed-8 all 87/384 (22.7%), mean 93.7 frames. RL follower 70-75%, GP with reference .52-.57. The DSL has no reference variables (ref buttons/action/speed/distance), so a follower cannot be written yet.

- lakitu-bridge-2 (file lakitu-bridge-2-2010.pol; also -2000/-2020, identical on train/test): clearance margin 10 instead of 30 units: val 6/8 (was 5/8), train 14/21, test 8/13 unchanged. Retired lakitu-bridge-1 in `runs/inv-author-lakitu` (kept on disk).
- Why the other 7/21 train starts fail (14 of 21 solved): 11 of the 13 non-solved runs end caught by Lakitu (action 0x20001306) at z about -200..-520, after four or five kick hops. Example (case 0): the landing before the bridge is at z=598 (fv 81); the last landing frame is at z about 365 where the double jump lands at -1940 (inside the box, needs < -2000) and the kick lands at -467 (inside). No single decision is wrong: the hop phase puts the takeoff outside the window z in [-170, 462 at fv 80]. This needs planning one or two hops ahead (shift phase with a slide-kick detour, or stop a hop short), which is the GA's job per the user, or a lookahead observation.

## Lessons
1. Almost all failures of the chain are NOT button-window failures but objects and walls: crates, goombas, the elevator-ramp wall, and static cliffs on the route. The policy needs `wall-f` and routing waypoints (sense.rs) for the rest.
2. A rollout is a dead state: no exit but landing, no turn. Dodge before the rollout or with sideways stick drift (about 10 u/f).
3. Fresh vs held matters at every step: B must be up the frame before the kick; a wall kick needs a fresh A inside a one-frame window, so do not hold A through the air.
4. Handing off before the goal (stop the chain 500-1200 units out) turns "fast but airborne at the goal" into Done: +2 to +4 solved starts.
5. Guard every object read with an existence test; the generic reader returns 0 for unknown things.
6. A policy trained or written for one start family transfers: gate-speed-8 gets 9/21 held-out with no search, one rollout (~80 frames) per start, against 21/21 for the pink-noise GA at 0.56M frames per start. The gap is geometry (walls) and object (enemy) knowledge, not control.

## Missing features (ordered)
1. Wall raycasts and edge/ledge sense (sense.rs) in the shared Env. Chain-ending bonks on static walls are the largest remaining class.
2. Routing: waypoints beyond `pts[0]` (the ramp is not on a straight line from the spawn); `dist1`/`pts[1..]` exist but the speed problem gives one point.
3. Reference variables for follower tasks (ref buttons, action, speed, distance, pace is not enough).
4. Landing-point prediction (`land-frames`, `land-zone`) is zero in the stock Env; the Lakitu rule uses a closed-form range instead.
5. A way to override only the stick of a rule (compose a button rule with a steering rule) instead of duplicating cells.
6. Object categories: the reader needs the same table as author_env (breakable box, enemy list); and enemy "can this kick kill it" (goomba act 2 hit a kick at 84 u/f and knocked Mario back).

## Universal first versions (not scored, no matching problem in the repo)
walk-target-1, swim-target-1, pole-climb-1, ledge-grab-1 are in the inbox with decomp-based notes. walk-target-1 matches hand-steer on `follow` (86 vs 87 of 384). Swim, pole and ledge need water/pole/ledge start states that none of the problems contain.
