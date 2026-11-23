# Driving `ar` (for Claude Code and other agents)

You plan the route; the GA executes one short objective at a time. You are the search: choose objectives, read states, and back out of dead ends. The rolling-horizon GA is good at control over the next ~100 frames toward an ETA, and bad at detours it can't see. Its fitness discounts by 0.99 per frame (half-life ~69 frames), so anything more than a few seconds away needs an intermediate objective.

## The loop

1. `ar brief`: daemon status, queue, recent events, Mario, progress, nearby objects, the objectives offered here.
2. Keep 2-3 objectives queued (`ar push ...`). Prefer **native** objectives (`ar objectives` shows what is offered and its ETA). Write **lambda**, spec or goto objectives on the fly when no native one says what you want.
3. `ar wait` (run it in the background or under Monitor; it returns on the important events), then act:
   - `met` / `window_low`: push the next objectives.
   - `stalled`: read `ar brief` and `ar geo`. Look at where it is stuck and why: a wall, a slope, water or a drop between Mario and the target. Then do one of these:
     - insert a nearer landmark in front (`ar push ... --at 0`);
     - rewind to before the mistake (`ar rewind cNNNN` or `-N`) and insert the landmark there;
     - raise the budget (`ar edit ID --rh '{"g": 4}'`);
     - try `ar fork 3 SEL`;
     - scrap the objective.
   - `unresolved`: the objective isn't offered here (too far, the object isn't loaded, its gate doesn't hold). Get closer first, or use a goto, lambda or spec.
   - `stuck` / `lost` / `failed`: the daemon paused. Inspect, fix (rewind, edit), then `ar resume`.
4. Write down why: `--note` on push, scrap and rewind. Those notes are the route's history for whoever reads it next.

Pause (`ar pause`) before a series of edits, so the daemon doesn't commit frames between them. `ar resume` afterwards.

## What the window does for you

- **Forward:** the current objective plus the next ones (`window`, 3) shape each plan. Queue the next objectives early: plans then end in states that suit them.
- **Backward:** after every met objective the daemon re-solves at least the last `back_frames` (600) frames, from the start of an objective, and keeps the result if it's sooner (`back_improved` / `back_rejected`). You don't need to ask for it. It costs about as much again as the forward solve over those frames.
  - `ar set back_frames 0` turns it off.
  - A larger value reaches further back: it costs more, and more can change.

## Writing objectives

- Native kinds: `objectives/OBJECTIVES.md` (stars, coins, red coins, doors, paintings, poles, bosses, techniques such as `Tech::LakituSkip`, `Tech::BljRun`, `Tech::PoleRedirect`, `WallKick::Climb`, dialogs and menus). Narrow them with `key` (a substring of `ar objectives`' key column) and `near`.
- Lambda objectives are functions of the base state b. Use them for relative goals ("600 ahead", "200 above the current floor") and reusable patterns. Store good ones with `ar lib add NAME file.json --note "..."`. Keep a revision under a new name with `--parent`; retire, never delete.
  ```json
  {"lambda": {"phases": [{"target": {"point": ["=x+600*sin(yaw)", "$y", "=z+600*cos(yaw)"]},
                          "done": {"in_sphere": {"center": ["=x+600*sin(yaw)", "$y", "=z+600*cos(yaw)"], "r": 150}, "action_class": ["stationary", "moving"]}}]},
   "args": {"y": "=floor_y"}}
  ```
- Spec objectives: `../tools/spec/FORMAT.md` (phases, object targets, predicates, `penalty`, `latch`). Read `../tools/spec/LANDMARK_TIPS.md` first:
  - end on stable ground;
  - a Done is a small grounded region, not a big sphere;
  - don't join landmarks with a straight line across a moat, a slope or a wall;
  - never let a side effect count as finishing.
- After a cutscene or a level change, check `ar state` before pushing position objectives: Mario's position and zone changed.

## Rules for the 120-star work

From the user, for runs meant to count:
- No TAS inputs in the result. The `boot: tas` start is the file-select menus only.
- Cap courses: the switch must be pressed on the visit.
- Explore routes and trajectories yourself.

## Costs

At the default 128 × 600 the daemon commits ~2-3 frames per second on 4-8 threads (a 10-second leg takes ~2-3 minutes). Raising `g` multiplies the time by `g`. The population journal is ~300 KB per game second; `ar pop stats` shows the size. `fork` and `explore` share the CPU with the daemon: pause it first if they should have the CPU to themselves.
