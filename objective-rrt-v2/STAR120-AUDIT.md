# Star reachability audit (2026-11-09)

From a read-only pass over the objectives and the game source (sm64-binds/sm64-port). "Unconfirmed" = geometry or strategy not checked in the emulator. Status column is kept current as items are fixed.

## Facts behind the verdicts

- A star spot in another area pulls nothing toward the area change: `ReachSpot::base_criteria` needs the spot's zone (objectives/src/extra/missions.rs:152-160).
- An accidental area change is thrown away: `PosDist`, `PosTime`, `reach_eta` return `Lost` on a zone change. Only `WarpEnter`, `Door::Enter`, `WarpFloorEnter`, `BbhEnter` count one as done.
- Per-object objectives exist only within 5000 units of b (`OBJECT_RADIUS`, objectives/src/lib.rs:41).
- No objective makes Mario use a cap. Cap-aware: `SecretTrigger::Touch` (BoB sky secrets), `IslandHop`, the flying mode in moves/modes.rs, `Cap::Collect`.
- Castle stars (Toads, MIPS): not in `has_star` (courses 1..=25 only) but counted in `St::stars`, `progress()`, `collected()[25]`; `Star::Collect` succeeds on `progress()` rising.

## Blocked or risky, ranked

| # | Item | Stars | Data | Status |
|---|---|---|---|---|
| 1 | DDD area 1 -> 2, instant warp | DDD 1, 3, 4, 6, 100 | `INSTANT_WARP(3, area 2)` levels/ddd/script.c:92; floor SURFACE_INSTANT_WARP_1E, x 4096..4301, z -511..512, y -511 | objective added (`WarpFloor::Enter`, builds/v2x), untested |
| 2 | WDW area 1 -> 2, instant warp | WDW 5, 6, 100? | `INSTANT_WARP(1, area 2)` levels/wdw/script.c:106; SURFACE_INSTANT_WARP_1C, centre about (352, -1023, -3327); water level may matter | objective added (builds/v2x), untested |
| 3 | JRB eel tail | JRB 2 | `bhvUnagi` 0x01010000 at (8270, -3130, 1846), act 2; touch the tail (`bhvUnagiSubobject`, about 1200 behind the head) within 150; star at (6833, -3654, 2230); unagi.inc.c:14-27, 142-203 | |
| 4 | `Menu::SelectAct` with 5 act stars + the 100-coin star: `act_init` = 7, the code loads act 1 | last act of any course after its 100-coin star (JRB 6 blocked) | objectives/src/menu.rs:298-322; when `act_init > 6` the target is the missing act - 1 (all 7 slots show) | |
| 5 | SSL area 2 -> 3 (Eyerok), instant warp | SSL 4 | `INSTANT_WARP(3, area 3)` levels/ssl/script.c:128; floor x -191..192, y 188..286, z -1421..-1222 | objective added (builds/v2x), untested |
| 6 | TTM slide, painting-warp floors | TTM 4, 100 | SURFACE_PAINTING_WARP_D3/D4/D5, levels/ttm/areas/1/collision.inc.c:2285-2292; painting at (3072, 921.6, -819.2); slide exit at area 1 (-2447, -2457, 3952) | objectives added for the painting floor and the two slide warps (builds/v2x), untested |
| 7 | Goals only offered within 5000 units | LLL 1, 2 (bullies), LLL 5, 6 (volcano warp at (0, 105, 0)), SL 2, SSL 1 (Klepto), THI 1 (piranhas), WF 1, BBH 2 | extend `OBRRT_GOAL_WIDE` beyond King Bob-omb | `OBRRT_GOAL_WIDE=1` now keeps every far object's objectives of weight >= 20, and area warps to an area with a missing star spot weigh 25 (builds/v2x), untested beyond BoB |
| 8 | `exclamation_weight`: 8..=13 => 30, but 9 is a running 1-up and 14 is the star box of index 5 | RR 6 | objectives/src/breakable.rs:16-22; should be `8 | 10..=14` | fixed (builds/v2x) |
| 9 | No aimed cannon target | WF 3 (-2500, 1500, -750), TTM 6 (7300, -3100, 1300), BoB island red coin (4100, 3672, 1629) once star (1,2) is had | `CANNON_TARGETS`, objectives/src/extra/warps/table.rs | |
| 10 | Caps: nothing uses an active cap | wing: BoB 5, SSL 5, TotWC, WMotR; metal: HMC 3, JRB 6, DDD 6; vanish: DDD 6, BBH 6?, SL 6?, VCutM? | switches are in other courses | |
| 11 | 100 coins | DDD (item 1), WDW, TTM (other areas) | | |
| 12 | Minor: star Toads keep weight 15 after their star; buddies keep weight 8 after the cannon opens; no MIPS release step; no TTC clock objective | | | |

DDD 3 (pole red coins) also needs key 2 (poles exist only after BitFS); DDD 1's submarine exists only before key 2.

## Per course (OK unless listed)

- BoB: 1 King needs `OBRRT_GOAL_WIDE`; 4 reds: one coin on the floating island (cannon or wings); 5 wing cap.
- WF: 1 radius; 3 no aimed shot.
- JRB: 1 fixed (WarpFloorEnter); 2 blocked (eel); 5 far shot only or the pillar techniques; 6 metal cap usual, and item 4.
- CCM: 1 needs the slide door within 5000 or `Ccm::SlideSkip`.
- BBH: 2 radius; 6 vanish cap (unconfirmed).
- HMC: 3 metal cap.
- LLL: 1, 2 radius; 5, 6 volcano warp radius.
- SSL: 1 radius; 4 blocked; 5 wing cap.
- DDD: 1, 3, 4, 6, 100 blocked; 2, 5 OK.
- SL: 2 radius; 6 vanish cap (unconfirmed).
- WDW: 2 water level not in the state; 5, 6 blocked; 100 risky.
- TTM: 4 blocked or risky; 6 far shot only; 100 risky.
- THI: 1 radius; 6 long chain.
- TTC: clock speed comes from the entry; RR: 6 box weight (item 8).
- Castle: TotWC, WMotR wing; VCutM vanish (unconfirmed); MIPS release; the rest OK. Bowser fights, keys and doors OK.
