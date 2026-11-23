# Policy DSL

A policy is one s-expression file (`.pol`). Drop files into `policy/runs/inventory/incoming/`; the generation loop
(`policy <rom> <problem> gen`) or `policy <rom> <problem> inv eval <file.pol> [--admit]` evaluates them.
Text is stable: `parse(print(p)) == p`. Comments start with `;`.

```
(policy NAME
  (meta key "value")...            ; origin, parent (ids), note (what changed and why), bind, sample
  BODY)                            ; a choice expression
```

## Choices (type C): what the policy outputs each frame

| form | meaning |
|---|---|
| `(cell BTN MAG ANG REF [fine_ang fine_mag yaw])` | a menu cell: buttons down this frame (`-`, or letters `A` `B` `Z`, e.g. `AB`), stick magnitude class `neutral` `short` (<=48, never a speed-kick dive) `full`; angle class `aim` `left` `right` `side-left` `side-right` `back`; reference direction `target` `face` `vel` `uphill` `downhill` `cam` `world`. `fine_ang` (-1..1, about +-0x800 yaw) and `fine_mag` (0..1) move inside the class; `yaw` is the world yaw for ref `world`. |
| `(skill NAME p...)` | a library skill; params are float expressions (missing ones are 0). Skills available now: `steer` `steer-a` `kick` `slide` `slide-kick` `hold-vel` `neutral` `jump` (more land with skills.rs). |
| `(if BOOL C C)` | |
| `(rules (when BOOL C)... (else C))` | rule list = nested ifs, first match wins |
| `(switch NAME)` | run another inventory policy this frame (portfolios gate between policies) |
| `(tag LABEL C)` | label that shows up in the "rules fired" report |

## Booleans (B)

`(> F F)` `(< F F)` `(and B...)` `(or B...)` `(not B)` `(act-in ACTION...)` `(held BTN)` (the previous pad had these buttons down).
Action names are the decomp's without `ACT_`, lower case: `dive_slide`, `freefall_land`, `forward_rollout`, ... (see `src/actions.rs`).

## Floats (F)

`(+ a b)` `(- a b)` `(* a b)` `(min a b)` `(max a b)` `(abs a)` `(sqrt a)` `(sin deg)` `(cos deg)` `(atan2 y x)` (degrees) `(fif BOOL a b)` (float conditional), numbers, and these observations:

| name | meaning |
|---|---|
| `fv` `vy` `hsp` | forward speed, vertical speed, horizontal speed |
| `ny` `hgt` `down` | floor normal.y, height above floor, slope along Mario's heading (positive = downhill) |
| `inact` `timer` | frames in this action (restarted by a B press, like the hand overlays), `actionTimer` |
| `dist` `dist1` `yawerr` `pace` | distance to the task's target 0 / 1, signed degrees from heading to target 0, the reference's pace |
| `wall-f/l/r/b` `ceil` `edge` `ledge` | raycast distances (forward, left, right, back), headroom, distance to a drop / ledge lip |
| `land-frames` `land-ny` `land-class` `land-zone` `in-zone` | ballistic landing prediction (frames, floor normal, slipperiness class) and the task's forbidden zone flags |
| `ahead-ny` `ahead-class` `ahead-hgt` | floor probe 96 units ahead |
| `wp0-dx` `wp0-dz` `wp1-dx` `wp1-dz` `wp2-dx` `wp2-dz` `pos-x` `pos-z` `vel-x` `vel-z` | world vectors to the task's waypoints 0..2 (steering), Mario's position and velocity |
| `holding` `health` `stick-x` `stick-y` `frame` | Mario state |
| `p0`..`p7` `rnd` | the policy's bound arguments and a deterministic random value in -1..1 (function of the instance seed and frame, so replays are exact) |
| `(field "name")` | any named game field (generic reader; unknown names read 0) |
| `(obj CATEGORY K "field")` | field of the K-th nearest object of a category |
| `(ahead N CELL "field")` | what-if: parsed, but evaluates to 0 for now (optional, low priority) |

## Templates, binding, sampling (lambdas)

A policy that uses `p0`..`p7` is a template. Bind arguments with metadata; sample them with declared distributions:

```
(policy go-holding (meta origin "llm")
  (meta bind "p0=0.2")                      ; default instance
  (meta sample "p0 uniform -1 1")           ; also: normal mu sd, choice a b c
  (skill steer p0))
```

`policy <rom> <problem> inv sample <id> --draws 50` draws instances, scores them on the training starts against the
default args, and promotes the best into a bound copy (`origin=lambda`, `parent=<template>`, `bind` = the winning draw and seed).

## Examples

1. Hand speed chain: refresh a jump kick on every landing, keep the dive slide / rollout steering at the target.

```
(policy speed-chain (meta origin "hand") (meta note "kick refresh on landings")
  (rules (when (act-in freefall_land freefall_land_stop jump_land long_jump_land) (skill kick))
         (when (act-in dive_slide forward_rollout) (skill steer 0))
         (else (skill steer-a 0))))
```

2. Lakitu skip, from the hill bottom (bridge double jump): the window rules in raw cells.

```
(policy lakitu-hill (meta origin "hand")
  (rules
    (when (act-in dive_slide) (cell - full aim target))
    (when (act-in forward_rollout) (cell A full aim target))
    (when (and (act-in freefall_land) (> land-frames 1) (< inact 5)) (cell A full aim target))      ; fresh A within the 4-frame window -> double jump
    (when (and (act-in freefall_land freefall_land_stop) (> ny 0.9)) (cell AB short aim target))    ; jump-kick refresh, short stick (<48)
    (when (act-in double_jump) (cell A full aim target))                                            ; hold A until past the box
    (else (cell - full aim target))))
```

3. Wait for a condition, then act (uses `inact`; any `(field ...)` works the same way):

```
(policy wait-then-dive (meta origin "llm")
  (if (and (act-in jump double_jump) (< vy -10) (> fv 28)) (cell B full aim target) (skill steer 0)))
```

4. Portfolio: gate between inventory policies per frame.

```
(policy mix (meta origin "portfolio")
  (if (act-in dive_slide forward_rollout freefall_land) (switch speed-chain) (switch hand-steer)))
```

Authoring loop: write the file, run `policy <rom> <problem> inv eval file.pol` (per-start results, which rules fired,
where it failed, the niche it would occupy); revise as a new file with `(meta parent "OLDID")` and `(meta note "what changed and why")`;
`inv retire <id> --reason ...` / `inv revive <id>` change status (never a deletion).

## Notes and traps

- `fine_ang` is clamped to -1..1 (about +-11 degrees). For a wider cone use a world-yaw cell: ref `world` with the yaw in 16-bit units,
  `(cell A full aim world 0 0 (* 182.04 (+ (atan2 wp0-dx wp0-dz) p0)))` steers p0 degrees off the direction to waypoint 0.
  Skills `aim-off` (p0 degrees from the target direction) and `aim-lead` (p0 gain on the heading error plus p1 degrees) take wide offsets directly.
- `(switch NAME)` runs the other policy with its own bound arguments (`bind`), so a bound lambda keeps its binding inside a portfolio.
- `(meta bind "...")` must be `pK=value` / `seed=N` separated by spaces; anything else is a parse error.
- Base-state reads: `start-x` `start-y` `start-z` `start-fv` `start-dx` `start-dz` (offset from the start) `since-start` (frames since the start), so gates can use the start position without flipping mid-run.
- Inbox: a file that switches to a policy that is not admitted yet waits (logged `inbox-wait`) until its dependencies are in.

## Short-horizon search (MPC) tags

Wrap any choice in a tag whose label starts with `mpc:` and the stick yaw is searched by rollout each frame the rule fires:

```
(tag mpc:8:13:70 BODY)      ; H=8 frames, N=13 yaw offsets over +-70 degrees, from the choice's reference direction
(tag mpcm:8:13:70 BODY)     ; same, but also tries every other cell of the reduced menu (worse in tests, see NOTES-integration.md)
```

- The buttons and the magnitude class are the choice's (the 48 limit is never crossed); the best offset's yaw is held for H frames in a copy of the game,
  scored with the task's own reward (eta decrease, gamma, bonus, death), the choice itself (offset 0) wins ties. The result is played as a `world` cell.
- Cost: N x H emulator frames per firing frame, reported as `+mpc` per start by `inv eval`. Wrap the whole body (`policy/runs/integ/wrap_all.py file.pol name H:N:RANGE`)
  for a universal dodge/steer correction, or only the air-steering rules (`wrap_mpc.py`).
- Object quantities that are not struct fields are derived from Sense: `(obj breakable 0 "dist")` (xz), `"dist3"`, `"dy"`, `"relyaw"` (degrees from the heading), `"radius"`; other fields go to the Reader.
- `policy <rom> <problem> trace <id|file.pol> <case substring>` with `POLICY_TRACE=1` prints one row per frame (action, speed, position, pad, wall ray, landing, rules fired).
- `policy <rom> <problem> play <id|file.pol> --dump DIR` writes raw inputs from power-on plus an index file for `gp/verify.py` (tasplay replay).
