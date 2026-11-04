# Spec objective format (v1)

A spec is one JSON object that the `obrrt-spec` crate (spec/) interprets as an obrrt2 `Objective`. A file holds one spec, an array of specs, or `{"specs": [...]}`. `OBRRT_SPEC=<file or dir>` loads one file or every `*.json` in a directory (sorted). Unknown keys are errors (with a "did you mean"), and every error names its JSON path, e.g. `$.phases[1].done.dist: expected a number, [lo, hi] or {"lt"|...}`. `doc`, `comment` and `notes` are allowed anywhere and ignored. For where to put landmarks and how to end them, see [LANDMARK_TIPS.md](LANDMARK_TIPS.md).

Units: frames (30/s), SM64 units (y up), yaw 0..65535 with 0 facing +z and 0x4000 facing +x. Numbers may be hex strings (`"0x4000"`).

## Top level

| key | type | default | meaning |
|---|---|---|---|
| `name` | string (letters, digits, `_ - .`) | required | kind = `Spec::<name>`, pool key `Spec <name> [<variant>]` |
| `zone` / `zones` | zone or list of zones | one required | where it is generated and stays alive; outside them the ETA is `Lost` unless the step that left was a phase's `done` (rule 6: leaving the zone may be the goal) |
| `gate` | predicate | always | must hold at the start (`base_criteria`, `exists_in`) and at the base state b it is generated from; must not use the phase target |
| `phases` | 1-16 phases | required | done in order; the spec is `Done` when the last one is |
| `lost` | predicate | none | `Lost` whenever it holds (no phase target; give `dist` a `to`) |
| `hints` | object | see below | controller hints |
| `params` | object | `{}` | default parameter values for `"$name"` substitution |
| `variants` | 1-8 `{name, params}` | none | each variant is a separate pool entry (`Spec <name> <variant>`) with its `params` merged over the defaults; every string `"$p"` anywhere in the spec is replaced by `params.p` |
| `expect` | predicate | none | validator only: the milestone this spec should reach (Done is compared with its first frame) |

**Zone:** `[9, 1]`, `[9]` or `9` (any area), `"bob"`, `"bob:1"`, `"9:1"`. Names: castle (= inside, 6), grounds (16), courtyard (26), bob, wf, jrb, ccm, bbh, hmc, lll, ssl, ddd, sl, wdw, ttm, thi, ttc, rr, pss, sa, wmotr, totwc, cotmc, vcutm, bitdw, bitfs, bits, bowser1-3.

## Phases

| key | type | default | meaning |
|---|---|---|---|
| `name` | string | `phase<i>` | for reports |
| `target` | target | none | where the phase goes: steering, `dist`-type predicates, the ETA's distance |
| `done` | predicate | arriving: inside a box/sphere, else `dist < 150` | when the phase is complete |
| `pace` | u/frame | 30 (core `PACE`) | distance → frames |
| `frames` | frames | 0 | fixed cost added to the travel; a phase without a target counts it down from when it began (a wait, a cutscene) |
| `latch` | bool | true | true: once done, stays done. false: a condition that must keep holding; while a later phase is current and this one's `done` fails, the spec falls back to it (e.g. "holding the King") |
| `repeat` | `n` or `{n, count?, frames?}` | none | the phase must be done n times. Without `count`: rising edges of `done` are counted. With `count: {field, per = 1, obj = "target"}`: the count is `floor(field / per)` of the object (the target's by default) and the phase is done once count ≥ n (and `done`, if given explicitly). `frames`: cost per further repetition (default the phase's `frames`, else 30) |
| `steer` | `true` / `false` / target | the phase target | what the GA's controller-seeded genomes steer at while this phase is current |
| `steer_button` | `"B"` / `"A+B"` / `{button, within = 300}` | the spec's | pressed on alternate frames when within `within` of the target |
| `arrival` | `pass` / `stop` / `door` | the spec's | how the target must be reached (mover hint) |
| `penalty` | `{when, frames}` or a list | none | `frames` added to the ETA while this phase is current and the predicate `when` holds (e.g. `{"when": {"thrown_off": true}, "frames": 230}`: the King's way back, his dialog and the regrab) |

**ETA** (GOALS.md's rules): current phase = `travel(Mario → its target) / pace + frames (counted down without a target) + (repeats left − 1) × repeat frames`, plus for each later phase `travel(previous target → its target) / pace + frames + (n − 1) × repeat frames`. A later phase without a target measures from the last target before it. The ETA falls at each phase transition by the finished phase's frames (keep `frames` honest: the validator flags rises). `Mem`: f[0] phase, f[1] repeat count, f[2] the phase's first frame; flags bit 0 initialised, bit 1 "done last frame" (repeat edges). Several phases can complete on one frame.

## Targets

| form | meaning |
|---|---|
| `[x, y, z]` or `{"point": [x, y, z]}` | a point |
| `{"box": {"min": [..], "max": [..]}}` | distance to the nearest point of the box (0 inside); steering at its centre |
| `{"sphere": {"center": [..], "r": r}}` | distance to the surface (0 inside) |
| `{"object": <selector>}` | an object, followed as it moves (its position + `offset`) |
| `{"node": id}` | the position of node `id` in nodes.csv of a run (the validator's run, or `OBRRT_SPEC_RUN=<run dir>` for obrrt2) |

Add `"metric": "h"` to any target for horizontal distance only.

**Object selector:** `"bhvWoodenPost"` (the one nearest Mario), or `{"bhv": "bhvWoodenPost", "near": [x, y, z], "within": r, "offset": [dx, dy, dz], "required": false}`: with `near`, the object whose home (its position if it has none) is nearest that point, within `within`. `required: true` makes the phase `Lost` when objects are read and none matches (otherwise the `near` point stands in). Names are those of data/behavior_data.c (the `bhv` prefix may be left out). Specs that read objects make the GA read the object list every frame (`needs_objects`).

## Predicates

A predicate is an object; all of its keys must hold (AND). A list of predicate objects is also an AND. Anything that can't be read (no objects, no target) is false.

| key | value | holds when |
|---|---|---|
| `all` / `any` | list of predicates | all / any hold |
| `not` | predicate | it doesn't |
| `stars`, `keys`, `coins`, `red_coins`, `wedges`, `lives` | range | the counter is in range |
| `stars_gained`, `keys_gained`, `coins_gained`, `red_coins_gained` | range | counter − its value at the start |
| `speed` (forward), `hspeed` (horizontal), `vy` | range | Mario's speed |
| `x`, `y`, `z`, `height` (above the floor) | range | Mario's position |
| `frame`, `elapsed` (since the start), `phase_elapsed` | range | frames |
| `action_timer` | range | `actionTimer` |
| `dist`, `hdist`, `dy` (Mario y − target y) | range, may add `"to": target` | to the phase target (or `to`) |
| `has_star` / `lacks_star` | `[course, star]` | star 1-7 of course 1-25 (1-based; 7 = 100 coins) is / isn't collected |
| `star_got` | `[course, star]` | collected now and not at the start |
| `action` | name or list | Mario's action is one of them (`"GROUND_POUND_LAND"`, `"ACT_DIVE"`, hex) |
| `action_class` | name or list | any of: stationary, moving, airborne, submerged, cutscene, automatic, object (action groups); air, intangible, swimming, riding_shell, invulnerable, sliding, diving, on_pole, hanging, idle, attacking (action flags) |
| `holding` | bool | Mario holds something |
| `held` | `"target"` or selector | Mario holds that object |
| `standing_on` | `"target"` or selector | Mario stands on that object (not in the air) |
| `facing` | `{yaw: number \| "target" \| "away" \| "object", within: deg = 45}` | Mario's yaw within `within` degrees of the yaw, of the direction to the target, away from it, or of the target object's own facing ("object": e.g. behind the King) |
| `zone` | zone or list | Mario is in one |
| `zone_changed` | bool | the zone differs from the start's |
| `in_box` / `in_sphere` | `{min, max}` / `{center, r}` | Mario is inside |
| `present` / `absent` | selector | such an object is / isn't loaded (absent needs objects read) |
| `field` | `{name, obj = "target", lt/le/gt/ge/eq}` | an object field in range |
| `alive`, `controllable`, `modal`, `in_water` | bool | |
| `save_flag` | number or hex | any of these bits is set in the save file's flags word (0x2 wing, 0x4 metal, 0x8 vanish cap switch) |
| `thrown_off` | bool (the phase target object), or `{obj, is = true}` | King Bob-omb (or the object) is off his hill: jumping back (act 5), thrown (act 4) and already more than 100 below home, or thrown and predicted to land off the summit from this frame's and last frame's position (core/src/king.rs: gravity 4, terminal 78, the summit's floor triangles from BoB's collision). The spec keeps the object's last position in `Mem` (f[3] and flags bits 8-25) |

**Range:** `5` (exactly), `[lo, hi]` (inclusive; `null` for open), or `{"lt": x, "le": x, "gt": x, "ge": x, "eq": x}`.

**Object fields:** x, y, z, home_x, home_y, home_z, home_dy (home.y − y: how far it sank, e.g. a pounded post), home_dist, mario_dist, action, health, value, held_state, interact_status, intangible_timer, yaw, beh_params, loot_coins, raw0-raw3 (the s16 words at 0x1AC, behaviour-specific).

## Hints

| key | default | meaning |
|---|---|---|
| `obj_weight` | 5 | pool weight (the search's learned multipliers apply on top) |
| `precision` | `explore` | `precise` gets more GA generations per commit |
| `ga_budget` | the controller's (16 / 5 / 150) | `[population, generations, horizon]` or the object form |
| `steer_button`, `arrival` | none, `pass` | defaults for phases that don't set them |
| `success_value` | 0 | counted in the kind's yield when a success adds a state |
| `kinship` | 6 | start kinship limit |
| `temperature` | 60 | start softmax temperature (ETA frames) |
| `done_frames` | 0 | frames the goal still costs after Done (a star's fall) |

Objective hints not exposed: `fixed` (macros), `pad_overlay`, `modal`, `dense_keep`, `start_bias`.

## Generation and keys

With `OBRRT_SPEC` set, every base state b in a listed zone, outside dialogs and menus, whose gate holds generates each spec (one entry per variant); the pool dedupes by key, so each spec/variant is one pool entry. `diag_tag` is `spec/<name>[/<variant>]`, so summary.json's `spec.attempts` has attempts, successes and added states per spec, next to `kinds."Spec::<name>"`.

## Example (tools/spec/examples/bob/03-chomp-gate.json)

```json
{
  "name": "ChompGate",
  "zone": "bob",
  "gate": {"lacks_star": [1, 6], "present": "bhvChainChompGate", "controllable": true},
  "phases": [
    {"name": "posts",
     "target": {"object": {"bhv": "bhvWoodenPost", "near": [260, 735, 1920], "within": 400, "offset": [0, 80, 0]}},
     "repeat": {"n": 3, "count": {"field": "home_dy", "per": 60}, "frames": 35},
     "frames": 25, "steer_button": {"button": "Z", "within": 120}, "arrival": "stop"},
    {"name": "gate", "done": {"absent": "bhvChainChompGate"}, "frames": 180, "steer": {"point": [1550, 1200, 300]}},
    {"name": "star", "target": {"point": [1550, 1200, 300]}, "done": {"star_got": [1, 6]}}
  ],
  "hints": {"obj_weight": 20, "precision": "precise", "success_value": 5, "ga_budget": [24, 6, 300]},
  "expect": {"star_got": [1, 6]}
}
```
