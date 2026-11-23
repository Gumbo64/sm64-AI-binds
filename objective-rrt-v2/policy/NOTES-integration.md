# Integration round: real stack + short-horizon stick search (MPC)

State at the start: the policy agent had already wired the real modules into `eval.rs` (`Env::full()`: menu cells, `sense::SenseCtx`, the Reader, `skills::library()`,
DSL terminals for wall rays / edge / ledge / landing / zone, `sqrt`, `atan2`, float `fif`, base-state reads, `switch` keeping bound arguments, `bind` parse errors,
dependency-resolving inbox). What was NOT wired: `mpc.rs`, derived object fields (`dist`, `relyaw`, `dy`: the authors' private reader had them, the Reader only has struct fields, so
`(obj ...)` policies lost their object reads), `play` + tasplay verification, per-frame tracing.

## What I added
- `(tag mpc:H:N:RANGE BODY)`: a rollout search over the stick yaw (H frames, N offsets over +-RANGE degrees from the choice's reference, buttons and magnitude class fixed, so the 48 limit is never crossed;
  scored with the task's own reward; offset 0 wins ties). `(tag mpcm:...)` additionally tries the other cells of the reduced menu. `eval.rs::mpc_pick`. Cost is reported as `+mpc` emulator frames per start.
- Derived object fields for `(obj CAT K "dist"|"dist3"|"dy"|"relyaw"|"radius")` from Sense (they no longer go to the Reader, which made `FieldSet::parse` fail and drop all named fields).
- `policy <rom> <problem> trace <policy> <case substring>` (+ `POLICY_TRACE=1`): one row per frame. `policy ... play <policy> --dump DIR`: raw inputs from power-on + index for `gp/verify.py`.
- `runs/integ/wrap_all.py` / `wrap_mpc.py`: wrap a policy file's whole body (or only its air-steering rules) in an mpc tag.

Not done: a stick-only override form for a rule (`mpc` covers the need), the elite-weighted lambda sampler, `mpc` as a separate polish pass over a finished policy, `(ahead ...)` what-if.

## Result on `speed` (15 train, 21 held-out starts; ONE rollout per start, no per-start search)
Real stack throughout. "+MPC" = whole body wrapped in `mpc:8:13:70` (about 8 k extra emulator frames per start; the pink-noise GA below uses 0.56 M per start).

| policy | train | held-out | held-out frames-to-Done | emu frames / start |
|---|---|---|---|---|
| f06-refresh-frozen = X00006 (LLM, ground author) | 12/15 | 14/21 | 107.8 | 104 |
| **+ MPC** | 13/15 | **19/21** | 104.5 | 102 + 10 271 |
| X00008 | 12/15 | 14/21 | 104.3 | 102 |
| + MPC | 14/15 | 18/21 | 103.8 | 99 + 10 022 |
| g09-refresh | 11/15 | 11/21 | 95.3 | 107 |
| + MPC | 14/15 | **19/21** | | |
| g11-waypoint | 11/15 | 11/21 | 95.3 | 107 |
| + MPC | 13/15 | **19/21** | | |
| m02-gate (portfolio, dependencies now resolved) | 13/15 | 16/21 | 109.7 | 102 |
| + MPC | 13/15 | 18/21 | 106.2 | 102 + 10 335 |
| gate-speed-8 (objects via real Sense) | 9/15 | 7/21 | 71.0 | 76 |
| + MPC | 10/15 | 15/21 | 109.9 | 99 + 10 096 |
| air-speed-chain-3 | 10/15 | 10/21 | 97.9 | 92 |
| + MPC | 13/15 | 14/21 | | |
| hand overlay g01 | 10/15 | 9/21 | | |
| + MPC | 12/15 | 11/21 | | |
| **pink-noise GA, per start (3 seeds)** | 44/45 | 61/63 | 71.1 / 96.4 | 0.56 M |
| pink-noise GA, scale 3 | 44/45 | 61/63 | 68.0 / 86.9 | 2.0 M |

Every one of the 10 base policies tried (g08, air-X00001 and the others in `runs/integ/scan.txt` too) improves with the wrapper (+2 to +8 held-out). The best are at 19/21 = 90% held-out (pink GA: 97%) for about 1/50 of the GA's emulator frames per start; train is 13-14/15 (GA 98%).
Frames-to-Done on solved held-out starts is 104 against 96 (GA at 0.56 M frames) and 87 (2 M): the policy is not faster, it is far cheaper and reactive.

MPC settings are not delicate (speed2 train / validation / held-out, X00006 and g09; `runs/integ/tune.txt`): H 6-10, N 9-13, RANGE 50-70 gave 17-19/21 held-out; H 5 or N 19 or RANGE 90 slightly worse. `mpcm` (also searching the menu's button cells)
was clearly worse: 13/21 held-out (14/15 train) at 6x the cost: a greedy 8-frame search overrides the policy's longer-horizon button logic. Selection note: the first settings were compared on the held-out set; the later scan on speed2 train/validation/held-out shows the same band, so the headline is not a lucky single setting, but treat 19/21 as +-2.

## What fixed what
- The dominant failure was a solid object (the breakable box at King f2441, found by the sense agent) in the path of a rollout/kick that cannot turn in the air: a stick offset found by 8-frame rollouts drifts past the corner.
  `partial-fast-king-speed-blj.old@2441` now passes the box (trace in the session log); the same wrapper fixes other starts in that region and in the 2012 TAS starts.
- Evolution (`gen`, 30 min, 4 threads, seeded with the wrapped LLM policies): children of the wrapped air-speed-chain reached train fitness 160.5 (R00436: 13/15 train, 74.8 frames) but only 14/21 held-out
  (the wrapped LLM parents are better held-out); the GA overfits the 15 training starts as before. Inventory after the run: 471 entries, 99 active (the rest retired by `maintain()`, nothing deleted), 36 niches.

## Still failing (X00006 + mpc:8:13:70)
`partial-fast-king-speed-blj@2435` (train) and `...old@2487`, `full-120star-2012-U@22580`, `@23866`: a bonk after the first kick landing near (-4150, 6700) leaves Mario walking at low speed and the recovery chain (walk, Z, long jumps) stalls; 2487 stalls as a crouch slide at (264, 5773).
An 8-frame horizon does not see that the kick's landing leads into the box; a longer horizon (24) did not help (18/21).

## Independent replay
`policy ... play <policy> --dump runs/integ/plays` then `python3 gp/verify.py builds/tools/tasplay ../baserom.us.z64 <index> ramp`: **tasplay agrees on 36/36 plays for each of X00006+mpc, X00008+mpc, g09+mpc, g11+mpc** (144 plays, including the failures: claimed -1 and replay -1).

## Files
`src/eval.rs` (mpc_pick, trace rows, derived object fields), `src/dsl.rs`, `src/bin/policy.rs` (`trace`, `play`, `+mpc` in reports), `runs/integ/` (wrapped policies `w-*.pol`, `t-*.pol`, `i0*.pol`, scan.txt, tune.txt, plays/, gen-speed.log),
admitted into the inventory as `inventory/incoming/done/mpc-w-*.pol` (origin llm).
