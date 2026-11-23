# Menu: the reduced input space (`policy/src/menu.rs`)

`menu::menu(&Obs) -> Vec<Cell>` returns, for the current state, the (button set, stick region) cells that can
change what happens next. `menu::realize(&Choice, &Obs) -> GamePad` turns a cell plus its continuous part
(`fine_ang`, `fine_mag` in -1..1) into a pad. Everything else is removed.

Files: `src/menu.rs`, `src/menu_table.rs` (generated), `src/bin/menu_check.rs` (brute force + checks).
Regenerate: `menu_check collect` -> samples jsonl -> `scratchpad/menu_derive.py samples.jsonl,more.jsonl out.json 0.995 5`
-> `out_table.rs` copied to `src/menu_table.rs`. Raw samples and eval logs: `runs/menu/`.

## Method

1. **Brute force** (`menu_check collect`): replay a movie from power-on, and at sampled states (up to 40 per action,
   at least 25 frames apart, `controllable` and alive) try all 8 button sets (A, B, Z levels) x 121 stick cells
   (neutral + 5 magnitudes x 24 angles at 15 deg steps from Mario's facing) = 968 pads. Each is run for 1 frame and
   for 3 frames with the same pad from a savestate. Outcome = (action, speed class, vy class, in air). Speed class
   compares the new speed with the old (zero, <0.7, <0.9, <0.97, kept, gained, reversed); vy class is
   down / flat / up / jump boost.
2. **Derive** (`menu_derive.py`): per action, greedy set cover over (button set, stick class) candidates until 99.5% of
   the (state, outcome) pairs are reached. Stick classes group the grid cells (below). Left/Right mirror pairs are
   kept together. Buttons that the guard table (`graph.json`) reads but the data never needed get two cells (neutral,
   full aim).
3. **Check** (`menu_check eval`): replay again, build the real `Obs`, call `menu::menu`, realize each cell at 3 angle x
   3 magnitude positions inside its class (so `realize` is tested too), run 1 and 3 frames, and compare with the
   full 968-pad outcome set of the same state.

Data: 2026, 2012, rikku 120-star TASs and the 70-star TAS (7,711 states), plus GA/search routes (`logs/kingloop`,
`logs/mend`, `gp/movies` speedchain and lakitu; 2,780 states). 109 actions with >= 5 states in the shipped table.

## Results

| check | states | mean cells | action-level cover | states fully covered |
|---|---|---|---|---|
| 968-pad grid | | 968 | 1 | 1 |
| shipped table, in-sample (all data) | 10,491 | 16.0 | 0.9972 | 0.987 |
| held out: rikku TAS (table from the other 3 TASs) | 2,161 | 12.8 | 0.9908 | 0.978 |
| held out: GA routes (table from TASs only) | 2,780 | 16.0 | 0.9945 | 0.980 |

"Action-level" = the set of (action, vy class, air, horizon) reached; speed class is left out because the speed
class of a hop depends continuously on the angle and magnitude (that is the MPC's job, not a discrete choice).
With the speed class included the cover is 0.984 to 0.990 and 0.89 to 0.93 of the states are fully covered.

Pruning: 14 to 16 cells against 968 grid pads (about 65x) and against about 8 x 65,000 raw stick values (about 30,000x).
The 16.0 includes the airborne landing-prep cells (see below): `Sense::default()` has no landing estimate, so every
airborne state gets them. With a landing far away they are not added.

Selected actions (shipped table, in-sample, with prep cells):

| action | states | cells | outcome cover | perfect states |
|---|---|---|---|---|
| idle | 358 | 18.6 | 0.998 | 0.986 |
| walking | 501 | 32.1 | 0.9995 | 0.992 |
| turning_around | 200 | 36.0 | 0.997 | 0.955 |
| braking | 127 | 30.1 | 1.000 | 1.000 |
| crouch_slide | 228 | 21.1 | 0.9945 | 0.978 |
| jump | 349 | 24.4 | 0.999 | 0.991 |
| double_jump | 307 | 25.6 | 0.999 | 0.990 |
| triple_jump | 177 | 19.0 | 0.998 | 0.989 |
| long_jump | 441 | 10.2 | 0.998 | 0.995 |
| forward_rollout | 249 | 9.0 | 0.996 | 0.992 |
| dive | 288 | 8.0 | 1.000 | 1.000 |
| dive_slide | 237 | 14.5 | 0.999 | 0.996 |
| slide_kick | 135 | 9.0 | 0.993 | 0.985 |
| slide_kick_slide | 117 | 16.1 | 0.998 | 0.991 |
| jump_kick | 262 | 10.4 | 0.996 | 0.992 |
| butt_slide | 188 | 14.6 | 0.999 | 0.995 |
| freefall | 195 | 28.1 | 0.990 | 0.949 |
| wall_kick_air | 182 | 25.3 | 0.990 | 0.956 |
| air_hit_wall | 174 | 21.1 | 0.993 | 0.954 |
| jump_land | 249 | 22.8 | 0.999 | 0.992 |
| freefall_land | 238 | 14.6 | 0.999 | 0.992 |
| freefall_land_stop | 214 | 30.3 | 0.999 | 0.981 |
| long_jump_land | 240 | 11.4 | 0.998 | 0.992 |
| ground_pound | 189 | 3.0 | 1.000 | 1.000 |

Files: `runs/menu/eval_ship2_insample.txt` (per action, everything), `eval_heldout_rikku.txt`, `eval_heldout_bot.txt`.

### Intentionally dropped or not covered

- **Speed-class-only differences** between cells that reach the same action (continuous; refine with `mpc`).
- **Outcomes from contact the pad cannot choose**: `grabbed`, `ledge_grab`, `teleport_fade_out`, `soft_bonk` into a
  wall, `ground_bonk`, `heavy_throw`, object pickups. They show up as the remaining 0.3 to 1% misses in the held-out GA
  routes (sorted worst first in `eval_heldout_bot.txt`), none in a speed-chain action.
- **Cutscene, object, and unsampled non-steered groups**: a single no-op cell. Unsampled stationary / moving /
  airborne actions borrow `idle` / `walking` / `freefall`; unsampled submerged ones borrow the first sampled swim
  action. `water_idle`, `flying` and the cannon states were thin (< 8 states in the TAS-only table) and are the weakest
  held-out rows.
- Outcomes later than 3 frames, except landing prep (below). Time-gated windows (`stomach_slide` A / B after
  `actionTimer` 5, `move_punching` A held) are not visible at sampled early frames; the theory union adds their buttons.

## Stick-region boundary table

Angle is the stick's world yaw minus the reference's, positive = increasing yaw ("Left"). Reference is `Face` for the
table; `Target` and `Vel` cells are added at run time (Aim at target for each button set the table aims with, aim along
and against the velocity when it differs from the facing by more than 22 deg).

| Ang | offset | what the data shows at this edge |
|---|---|---|
| Aim | within 22.5 deg | outcome boundary almost never inside it |
| Left / Right | 22.5 to 67.5 | weak edges (0.05 to 0.2 of states per 15 deg cell edge in air, 0.01 to 0.03 on the ground): the turn without losing the forward push |
| SideLeft / SideRight | 67.5 to 100 | the 90 deg cos(dyaw) = 0 line (air speed gain turns into loss) and the walking turn rate |
| Back | 100 to 260 | **`analog_stick_held_back`: |dyaw| > 0x471C = 100 deg.** Boundary frequency at the 90-105 deg cell edge: turning_around 0.97, walking 0.47, freefall_land_stop 0.45, forward_rollout 0.40, dive 0.53, lava_boost_land 0.83 |

| Mag | stickMag | what the data shows at this edge |
|---|---|---|
| Neutral | 0 | raw |x| < 8 reads 0: INPUT_NONZERO_ANALOG off. Neutral vs 16 differs in 97% of idle states (walking starts), 95% turning_around, 99% punching |
| Short | 8 to 46 | intendedMag = mag^2/128; the walking target speed and ground speed bucket move inside it (idle 16 -> 32 differs 87%) |
| Full | 49 to 64 | **stickMag > 48** (ground B = speed-kick dive, `moving.c:494`): 46 -> 54 differs in 51% of freefall_land_stop states; air steering saturates at 64 |

Raw stick: `raw = stick +/- 6` outside the deadzone, `stickMag` capped at 64 by scaling (`game_init.c:457-490`), so a
tilt above 0.53 of the 121-unit pad range is already full magnitude. `realize` uses `moves::arcs::stick`; a unit test
(`menu::tests::realize_matches_class_bounds`) re-reads the raw pad the way the game does and checks every class stays
inside its magnitude range, never below the deadzone, and that Short never exceeds 48 and Full always does.

## What the brute force found that the guard-table theory missed

Buttons the graph's guards read vs buttons the data says matter (57 of 101 actions differ; selected):

- **A level in the air is a jump-height control, not an edge.** `jump`, `double_jump`, `wall_kick_air`, `steep_jump`:
  theory B/Z only; data A too (releasing A while rising cuts vy; `ACT_FLAG_CONTROL_JUMP_HEIGHT`).
- **Landing actions read B and Z through their expiry cancels.** `jump_land`, `freefall_land`, `double_jump_land`,
  `side_flip_land`, `long_jump_land`: guards list A (jump); data A, B, Z (the LAND_STOP -> idle cancels: B punch / jump kick,
  Z crouch, in the same frame).
- **`air_hit_wall`, `soft_bonk`, `backward_air_kb`**: theory A only; data A, B, Z (the run-twice-on-entry quirk, and
  dive / kick / pound out of the knockback).
- **`butt_slide`, `braking`, `butt_slide_stop`, `turning_around`, `flutter_kick`**: data reads all three where the
  guards list one or two.
- Theory has buttons the data cannot see in 3 frames (time gates): `stomach_slide` A / B, `move_punching` A held,
  `long_jump_land_stop` B. They are added from the graph.
- **Where the angle matters differs by action.** Ground actions have one sharp boundary at the held-back line (100 deg,
  0.45 to 0.97 of states). Air actions have it at the 90 to 105 deg cell edge too (dive 0.53, rollout 0.40, jump kick 0.33,
  long jump 0.28, double jump 0.14 to 0.18) with a spread of weaker boundaries from 45 to 135 deg (0.05 to 0.2). The
  aerial classes therefore keep Left / Right / Side as separate cells; a single forward / back split would lose them.

### Landing prep (`menu_check carry`)

One or three frames of outcomes cannot see that the button levels held *before* touchdown decide the cancels on the ground
(A held + B fresh = jump kick refresh, A fresh = double jump). `carry` holds a button set through the air until touchdown
and then tries every b2 x {neutral, full ahead} for 1 and 3 frames:

- Holding a button set through the air changes the touchdown outcomes in 75 to 98% of airborne states
  (forward_rollout 0.98 for every set, jump 0.90 to 0.96, double_jump 0.94 to 0.99, long_jump A 0.76).
- Share of the touchdown outcomes reached by holding only {none} / {none, A} / {none, A, Z, A+Z} / all eight:
  - rollout .90 / 1.00 / 1.00 / 1.00; long_jump, jump_kick, dive, slide_kick, ground_pound 1.00 with {none} alone;
  - jump .61 / .62 / .78 / 1.00, double_jump .60 / .60 / .77, side_flip .59 / .59 / .75, wall_kick .59 / .60 / .81,
    triple_jump .29 / .29 / .56, freefall .65 / .65 / .80, air_hit_wall .48 / .65 / .85. For these B and Z also start
    dives, kicks and ground pounds in the air, so all eight are different trajectories.
- Rule in `menu()`: airborne and the landing unknown or within `AIR_PREP_FRAMES` (3): add the eight button sets at
  full aim for the jump-like families, and only {none, A} for the fixed-air-action ones (rollouts, dive, jump kick,
  long jump, slide kick, butt slide air, ground pound, lava boost, riding shell). `Sense.land` (sense.rs) turns it off
  when the landing is far.

## API notes for the other layers

- `menu(&Obs) -> Vec<Cell>`: Face-referenced table cells + Target / Vel aim cells + landing prep; deduped and sorted.
- `realize(&Choice, &Obs) -> GamePad`, `default_choice(Cell)`, `ref_yaw(Ref, &Obs, world)`, `ang_range`, `mag_range`,
  `raw_stick`, `buttons_of`, `btn_of`, `edges(last_pad, Btn) -> [A, B, Z]` (0 up / 1 fresh / 2 held / 3 released),
  `relevant_buttons(action)` (mask A=1 B=2 Z=4 of buttons whose level changed outcomes in >= 3% of that action's states).
- `types::realize` is still `unimplemented!`; it should be `crate::menu::realize(c, o)`. `eval.rs` has its own local
  realize (comment says it is a stand-in).
- The mapping to the fine parameters: `fine_ang` in -1..1 moves the stick across the class (Aim +-21 deg, Left/Right
  45 +- 22, Side 84 +- 15, Back 180 +- 78); `fine_mag` in -1..1 is stickMag 27 +- 19 (Short, 8 to 46) or 56.5 +- 7.5 (Full,
  49 to 64; `default_choice` uses 1.0 = 64).
- Left/Right are mirrored in every generated row, so a policy never sees only one turn direction.
