# sense: what a policy sees

Files: `src/sense.rs` (fixed hot-path `Sense`), `src/sense/reader.rs` (any field by name),
`src/sense/behaviours.rs` (object categories and key fields), `src/sense/fields.tsv` (generated),
`FIELDS.md` and `BEHAVIOURS.md` (generated listings), `tools/gen_fields.py` (generator),
`src/bin/sense_check.rs` (validation), `runs/sense/` (logs of the numbers below).

## API

```rust
let mut ctx = sense::SenseCtx::new();            // one per thread; indexes each zone's geometry once
let s: types::Sense = game.with_memory(|m|
    ctx.read(m, &st, &SenseOpts::default(), &regions, &AirModel::from_pad(&st, next_pad)));
ctx.blocks[k]        // what blocks in direction k (static wall, or Object(slot), loaded or predicted)
ctx.floor_height(&st, x, y, z) / ctx.ceil_height(..)   // static queries, game semantics
sense::MarioFlags::of(&st)                       // caps, holding, air/moving/stationary, water depth, quicksand, wedges
sense::landing_in(&s, &regions) / nearest(&s, CAT_ENEMY) / hint(cat)
```

`st.objects` must be present (`read_state(.., true)`) for object probes and object walls.
`SenseOpts` switches each part off (floors, walls, edges, landing, objs, dyn_floors, dyn_walls).

What is filled (`types::Sense`):
- `ahead[6]` floor probes along the facing at 24, 48, 96, 192, 384, 768 and `vel_ahead[2]` along the velocity at
  96, 384: height, normal.y, surface type, slippery class (0 very slippery .. 3 not). Game `find_floor` semantics
  (highest floor at most 78 above his y, intangible floors skipped like the game), static floors plus object floors
  read from the dynamic partition.
- `wall[4]`: distance Mario's centre can travel forward / +0x4000 / -0x4000 / back before a wall collides (the
  game's test: within radius 50 of the plane, inside the projected triangle), minimum over heights 30, 60, 150.
  It merges static walls and object collision walls, **including objects whose collision model is not loaded yet**
  (the game loads it only within `oCollisionDistance`): their `collisionData` is parsed and posed from `oPos` and the
  face angles. `wall_normal_yaw` is the forward hit's. `ctx.blocks` says which thing it is and whether it is loaded.
- `ceil_clear` headroom (static ceilings), `edge_ahead` (first floor drop > 100 or no floor within 768),
  `ledge_ahead` (check_ledge_grab geometry: lower wall hit, no upper wall, ledge floor > 100 above; along the
  velocity in the air, the facing on the ground).
- `land`: ballistic landing prediction while airborne: frames, position, fv, floor probe. Uses `update_air_without_turn`
  (drag, stick term, cap at 32/48), the game's gravity rules (long jump/slide kick 2, cannon 1, wing glide, A-release
  jump cut), four quarter steps per frame against the floor probes. Ignores walls, ceilings and the facing turning.
  `AirModel::from_pad` turns a pad into the stick terms (game `intendedMag`/`intendedYaw` from camera yaw).
- `objs`: up to 3 nearest per category (enemy, coin, pole, star, platform, bomb, breakable, other; numbering as
  `skills::cat`), with distance, yaw relative to the facing, dy, and the object's `hitboxRadius` read from memory.
  `sense::hint(cat)` says what touching it tends to do.
- `in_zone`: Mario inside one of the caller's `Region` boxes.

## Reader: any field by name

```rust
let fs = reader::FieldSet::parse(&["gGlobalTimer", "gRandomSeed16", "gMarioState.health", "gCamera->mode",
    "gLakituState.curPos[0]", "nearest(bhv=bhvBobomb).oBobombFuseTimer", "nearest(cat=enemy,k=1).oAction",
    "obj[3].oPosY", "mario_obj.oAction", "held.oAction", "floor_obj.oPosY", "gMarioState.floor->type",
    "gMarioState.controller->rawStickX", "gHudDisplay.coins"])?;
let mut v = vec![0f32; fs.len()];  fs.fetch(m, &st, &mut v);        // NaN = missing (null pointer, no such object)
fs.fetch_values(m, &st, &mut out);                                     // typed: I(i64) / F(f32) / Missing
reader::list_fields("")                                                // every name, for a DSL / GP to enumerate
```

Path grammar and the field list are in `FIELDS.md` (689 globals, 12 structs, 736 Object fields incl. every `o*`
raw field). Names come from the decomp headers (struct offsets from the `/*0x..*/` comments, checked against
`core/src/layout.rs`: 106 offsets and addresses, 0 differences) and the build's linker map (every global's address).
Regenerate with `python3 policy/tools/gen_fields.py` after rebuilding the game. `BEHAVIOURS.md` lists the ~400
behaviours with their category and key fields (bob-omb fuse, chain chomp state, platform velocities, Bowser phase,
...; `behaviours::KEYS`, `COMMON`; a test in `sense_check fields` checks that every name exists).

## Measured (2026 120-star TAS replayed from power-on; logs in runs/sense/)

| check | result |
|---|---|
| floor probes vs `geo::floor_at` (6 ahead + 2 velocity probes, 26.5k probes over 4 zones) | 0 to 0.07% differ (6 probes in all; not investigated) |
| own-position floor vs the game's `floorHeight` (static floors) | 3102 frames, 1 off by > 1 |
| headroom vs the game's `ceilHeight` | game computes it at the start-of-frame position: compared there, 12 to 29 of 300 to 500 frames differ (sloped ceilings, object ceilings skipped) |
| wall: hit next frame (`st.wall_normal`) vs predicted `wall[0] <= speed+2` | recall 0.86 to 1.00, precision 0.25 to 0.43 (3-frame window: recall 0.75 to 0.91, precision 0.24 to 0.42). Hit counts are small (2 to 13 per range); false positives are mostly sliding along or grazing |
| edge (grounded, v > 15): `edge_ahead <= 12 frames of travel` vs airborne within 14 frames | precision 0.22 to 0.85, recall 0.41 to 0.78 (jumps also end grounded runs, so recall is bounded) |
| ledge: before the TAS's 4 real grabs | `ledge_ahead` finite on 3 of 4 on the frame before the grab (and 2 to 3 of 4 on frames 2 to 4 before); 26 other states in the four range checks were flagged with no grab following (the TAS grabs only 4 ledges in 128808 frames, so there is little truth either way) |
| landing, actual pad: frames to land | exact 60 to 83%, within 1 frame 64 to 90%; median xz error 7 to 19 units, p90 356 to 2456 (flights whose action changes are dropped; the tail is walls, bonks, water, wind, A-release cuts) |
| landing, constant full-forward pad | exact 57 to 78%: knowing the pad matters |
| `Reader` vs `St` (18 fields, 10.9k comparisons) | 7 differences, all `mario_obj.oPosX` during fast moves (the object's position lags the state; cause not investigated) |
| time per call | floors 0.5 to 0.8 us, walls 3 to 10, edges 1.4 to 2.9, landing 2 to 8, objects 2 to 4; all together 11 to 21 us (the 21 is the BBH range, wall-dense), 8 to 17 without object floors/walls; `Reader` 43 fields 1.4 to 2.0 us |

## The King route state (f2441, `logs/kingloop/partial-fast-king-speed-blj.old.m64`)

Mario at (-4790, 126, 7038), 85 u/f toward +x. The static walls show nothing within 1500 units, which is why the skills
saw nothing. What stops him is **`bhvBreakableBox` slot 18** at (-3979, 0, 6827), an object collision model
(surface type 0x76): the game's `m->wall` points at its surface (owner slot 18) on the frame he touches it. Its
collision is only loaded while Mario is within `oCollisionDistance`, so the partition is empty at 838 units and `dyngeo`
reports 0 object surfaces at f2441; the new wall rays predict it from `collisionData` and the object pose:
`wall[0] = 717.7` at f2441, falling 85.5 per frame to 32 at +8 (`runs/sense/block_2441.txt`).

Replaying from f2441 with the stick rotated and the movie's buttons: from -30 to +5 degrees Mario grazes the box's
corner at (-3940, 20, 6977) after 10 frames (the game sets `m->wall` to the box surface for that frame) and slides past,
ending hundreds of units on; at +10 degrees and beyond he hits it in the air after 8 frames at (-4129, 30, 6925)
and ends the replay at about (-4440, 149, 6830), stopped. So the TAS's route passes the corner by about 150 units
(box centre z 6827, route z 6977); a 10 degree shift in the kick stick turns a graze into a head-on hit.

## Gaps

- Object walls are modelled for models with rigid pose (position and face angles; no scale, no animated vertices).
- Floors/ceilings of unloaded object models are not predicted (only walls); landing prediction ignores walls.
- Hitbox contact (enemies, bully pushes) is not blocking geometry here; `objs` gives distance and radius.
- The ledge test has only 4 real grabs in the 120-star TAS; the prediction is checked on those only.
- Wall and edge precision are low because the TAS rarely hits walls; the numbers are small-sample.
- `Reader` gives `f32`; integers above 2^24 lose low bits in `fetch` (use `fetch_values`).
- Zones are indexed when Mario is controllable on a static floor; a first read in a cutscene returns no geometry.
