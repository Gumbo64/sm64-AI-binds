# Objective kinds: reference

This is every objective kind in `objectives/src`, the Objective-RRT SM64 bot's domain crate, taken from the code as of 2026-10-25. Each kind's "Def" is the file:line of its `kind()` string, relative to `objectives/src/`. When the code changes, this list should change with it.

There are **142 kind names**. Some kinds also have variants:
- different keys or modes, such as `Tech::LakituSkip air`, the Climb modes, and the Dash, KickRun and Boost variants;
- two sources for one kind, such as `SecretEntrance::Enter`.

## 1. What an objective is, and how they are generated

An objective implements `trait Objective` (core/src/lib.rs:459). Its parts:

- **kind / key.** `kind()` is the type name used in logs and for per-kind weights (e.g. `"Coin::Collect"`). `key()` is the objective's identity in the pool: two objectives with the same key are the same objective.
- **weight.** `obj_weight(b)` is the objective's importance. The pool samples by it, multiplied by the search's learned per-kind multipliers and its failure decay. `success_value()` is how much a success counts in the kind's yield.
- **base.**
  - `base_criteria(s)` says whether state `s` may start an attempt.
  - `base_heuristic(s)` ranks the eligible starts (lower is better). The engine takes a softmax at `temperature()` over this ranking.
  - `kinship()` limits how far up the tree the start may be related to `b` (default 6, often 10).
- **success.** `success(traj)` returns the first index where the objective is met. `success_heuristic` ranks successful trajectories, by default by their frame count.
- **progress.** `progress(traj)` scores an unsuccessful trajectory and says which state to keep (`Progress { score, at }`). It drives the GA, and its kept state can become a tree node.
- **budget.**
  - `Budget::Ga { population, generations, horizon }`: a GA over raw inputs.
  - `Budget::Macro(pads)`: a fixed input sequence.
  - `Budget::Policy { limit, policy }`: a closed-loop pad rule.
- **precision.** `Explore` (the default: a first success is about as good as a refined one) or `Precise` (a narrow window, so evolution past the first success pays). This decides the GA's budget when `OBRRT_GA_BUDGET=1`.
- **hints.**
  - `steer_target(s)`: the point the controller-seeded genomes steer to.
  - `steer_button(s)`: a button pressed on alternate frames.
  - `pad_overlay()`: a per-frame rule that fixes some inputs (exact button timing) while the GA evolves the rest.
- **modal.** `modal()` marks dialog and menu options. These are tried only from modal states, and in a modal state they are the only objectives.

**Generation** (`Registry::generate`, lib.rs:63-145) runs from a base state `b`:
1. **Modal b.** If `b.modal()`, the only objectives are the options of the open dialog, save menu or act select (menu.rs). Generation stops there.
2. **Per state.** Mario's self objectives, from every `b`:
   - 2 x `Mario::PosDist`;
   - `Mario::PosTime`;
   - `Mario::Settle`;
   - `Mario::Wait` (with p = 0.3);
   - `Mario::GoToCastle` (outside the castle).
3. **Per object.** This covers every object within `OBJECT_RADIUS = 5000` of Mario (by the object's position or a star's rest position).
   - The hand-written kinds are matched on the object's `interact_type`: coin, star or key, door, grabbable (King Bob-omb is a boss), NPC (`npc_weight`), stomp/hit enemy (minus `NOT_DEFEATED_BY_VANISHING`), and breakable. Every `bhvExclamationBox` also gets an `Object::Break`.
   - Then `extra::generate(b, o)` runs for every object. It goes through each extra/ category file (bosses, collectibles, enemies, missions, npcs, rides, switches, tech, warps), which matches on behaviour names.
   - Then comes `Boss::Hit` for `BOSSES` whose star is not owned.
4. **Per level.**
   - `extra::generate_level(b)`, once per `b`: red coins, races, missions, coin milestones, the sun spot, the gate clip, and almost every technique (tech/mod.rs:61-89).
   - Then, in the castle, one `Painting::Enter` per painting in b's area (painting_table.rs).

extra/mod.rs says these kinds must not duplicate the hand-written ones in lib.rs's own modules.

## 2. Summary of all kinds

The groups are:
- **gen**: movement/generic
- **modal**: dialog/menu
- **star**: stars
- **coll**: collectibles
- **sw**: switches/interactables. This group is extra; the brief's list has none for switches.
- **enemy**: enemies
- **boss**: bosses
- **npc**: NPCs
- **warp**: warps/doors/paintings
- **ride**: rides
- **mission**: missions
- **tech**: techniques, with their tag

Switch "on" means always on.

| Kind | Group | Purpose | Switch |
|---|---|---|---|
| `Mario::PosDist` | gen | Reach a random point 300-2500 from b | on |
| `Mario::PosTime` | gen | Reach a point near b sooner than the tree's record there (`OBRRT_POSTIME=self`: b's position sooner than b did) | on (`OBRRT_POSTIME`, `_N`, `_BUCKET`, `_MARGIN`) |
| `Mario::Settle` | gen | Come to a stop | on |
| `Mario::Wait` | gen | Stand still 15-90 frames | on |
| `Mario::GoToCastle` | gen | Pause, Exit Course | on (`OBRRT_FAST_MODAL`) |
| `Dialog::Read` | modal | Read a dialog to its end | on (`OBRRT_FAST_MODAL`) |
| `Dialog::Yes` | modal | Answer yes | on (`OBRRT_FAST_MODAL`) |
| `Dialog::No` | modal | Answer no | on (`OBRRT_FAST_MODAL`) |
| `Menu::SaveAndContinue` | modal | Save menu, option 1 | on (`OBRRT_FAST_MODAL`) |
| `Menu::ContinueNoSave` | modal | Save menu, option 3 | on (`OBRRT_FAST_MODAL`) |
| `Menu::SelectAct` | modal | Pick an act | on (`OBRRT_FAST_MODAL`) |
| `Star::Collect` | star | Touch a star or key | on (ground-pound catch; `gateclip` softens the BoB-6 block) |
| `Coin::Collect` | coll | Touch a coin | on |
| `Object::Break` | coll | Break a box or ! box | on |
| `HiddenBlueCoin::Collect` | coll | Collect a switch's blue coin | on |
| `BlueCoinSwitch::Press` | coll | Pound the blue coin switch (collectibles version) | on |
| `Cap::Collect` | coll | Touch a wing, metal, vanish or normal cap | on |
| `RedCoin::Next` | coll | Collect the nearest red coin | on |
| `CoinFormation::Sweep` | coll | Collect a whole coin formation | on |
| `CrazyBox::Bounce` | coll | Bounce a jumping box until it breaks | on |
| `Heart::Spin` | coll | Heal at a recovery heart | on |
| `OneUp::Collect` | coll | Touch a 1-up | on |
| `OneUpTrigger::Touch` | coll | Touch a hidden-1-up trigger | on |
| `SecretTrigger::Touch` | coll | Touch a hidden-star secret | on |
| `PurpleSwitch::Press` | sw | Press a purple floor switch | on |
| `CapSwitch::Press` | sw | Press a cap switch | on |
| `BlueCoinSwitch::Pound` | sw | Pound the blue coin switch (switches version) | on |
| `Pillar::Pound` | sw | Pound the castle pillars (drains the moat) | on |
| `Post::Pound` | sw | Pound the Chain Chomp's post | on |
| `Post::Circle` | sw | Circle a wooden post for coins | on |
| `IslandTop::Pound` | sw | Pound THI's tiny island top | on |
| `WaterDiamond::Touch` | sw | Touch a WDW water diamond | on |
| `PyramidPillar::Touch` | sw | Touch an SSL pyramid pillar | on |
| `Book::Punch` | sw | Punch the BBH books in order | on |
| `Block::Push` | sw | Push a metal box 200 | on |
| `Pole::Climb` | sw | Climb to the top of a pole or tree | on |
| `Enemy::Defeat` | enemy | Defeat a stomp/hit enemy | on |
| `Object::Grab` | enemy | Pick up a grabbable | on |
| `Boo::HitFromBehind` | enemy | Hit a Boo from behind | on |
| `Bully::PushOff` | enemy | Push a small bully into lava | on |
| `MrI::Circle` | enemy | Circle Mr. I | on |
| `MrBlizzard::Circle` | enemy | Circle Mr. Blizzard | on |
| `Klepto::Hit` | enemy | Hit Klepto for the star or cap | on |
| `Moneybag::Reveal` | enemy | Reveal a hidden moneybag | on |
| `Moneybag::Stomp` | enemy | Kill a moneybag | on |
| `PiranhaPlant::Sneak` | enemy | Kill a sleeping piranha plant | on |
| `FirePiranha::Hit` | enemy | Hit a THI fire piranha | on |
| `Chuckya::Throw` | enemy | Grab and throw a Chuckya | on |
| `Bobomb::Throw` | enemy | Grab and throw a bob-omb | on |
| `Ukiki::ToCage` | enemy | Send the cage Ukiki to its cage | on |
| `Ukiki::RetrieveCap` | enemy | Get the cap back from Ukiki | on |
| `Whomp::GroundPound` | enemy | Pound a small Whomp's back | on |
| `Boss::Hit` | boss | King Bob-omb: hit, then the star | on |
| `WhompKing::Pound` | boss | Pound the Whomp King | on |
| `BigBoo::Hit` | boss | Hit a Big Boo | on |
| `Wiggler::Stomp` | boss | Stomp Wiggler | on |
| `Eyerok::PunchEye` | boss | Punch an Eyerok eye | on |
| `Eyerok::Wake` | boss | Wake Eyerok | on |
| `BigBully::PushOff` | boss | Push a big bully off | on |
| `Bowser::GrabTail` | boss | Grab Bowser's tail | on |
| `Bowser::ThrowAtBomb` | boss | Spin and throw Bowser into a bomb | on |
| `Npc::Talk` | npc | Talk to a Buddy, Toad or Koopa the Quick | on |
| `Sign::Read` | npc | Read a sign | on |
| `Yoshi::Talk` | npc | Talk to Yoshi | on |
| `Lakitu::Intro` | npc | Trigger the intro Lakitu | off while `lakituskip` is on |
| `Mips::Catch` | npc | Catch MIPS | on |
| `Ukiki::Catch` | npc | Grab the cage Ukiki | on |
| `Ukiki::FollowToCage` | npc | Follow Ukiki to the cage | on |
| `Hoot::Wake` | npc | Wake Hoot | on |
| `Hoot::Ride` | npc | Ride Hoot | on |
| `KoopaRace::Claim` | npc | Claim the Koopa race star | on |
| `PenguinRace::Talk` | npc | Start the penguin race | on |
| `PenguinRace::Claim` | npc | Claim the penguin race star | on |
| `Snowman::WakeBody` | npc | Start the snowman's body rolling | on |
| `Snowman::Finish` | npc | Talk to the snowman's head | on |
| `KoopaRace::ReachFlag` | npc | Beat Koopa the Quick to the flag | on |
| `PenguinRace::Win` | npc | Beat the penguin down the slide | on |
| `Tuxie::ReturnBaby` | npc | Return the baby penguin | on |
| `Snowman::LeadBody` | npc | Lead the snowman's body to its head | on |
| `Door::Enter` | warp | Go through a door | on |
| `Painting::Enter` | warp | Enter a castle painting | on |
| `SecretEntrance::Enter` | warp | Secret warp (PSS, SA, WMotR, VCutM, CotMC) and the TotWC sun look-up | on (bitdw: `lblj`) |
| `BowserStage::Enter` | warp | Warp into a Bowser course or fight | on (`lblj` waives the bitdw stars) |
| `Teleporter::Enter` | warp | Use a fading-warp teleporter | on |
| `Pipe::Enter` | warp | Enter a warp pipe | on |
| `WarpHole::Enter` | warp | Enter a `bhvWarp` hole | on |
| `Warp::Enter` | warp | Enter any other warp | on |
| `SunSpot::Stand` | warp | Stand on the lobby sun spot (TotWC setup) | on |
| `BbhEntrance::Enter` | warp | Enter BBH via the boo cage | on |
| `Tornado::Enter` | warp | Get picked up by a Tweester | on |
| `StarDoor::Enter` | warp | Go through a star door | on |
| `KeyDoor::Enter` | warp | Go through a key door | on |
| `Cannon::Enter` | warp | Get into a cannon | on |
| `Cannon::Fire` | warp | Fire a cannon (far, or aimed at a star target) | on |
| `Moat::Drain` | warp | (dead code, never generated) | - |
| `Chest::Open` | ride | Open the next treasure chest | on (`chestfacing` mode) |
| `Platform::Board` | ride | Get onto a moving platform | on |
| `Seesaw::Board` | ride | Get onto a seesaw | on |
| `Platform::Ride` | ride | Ride a platform a set distance | on |
| `WaterRing::SwimThrough` | ride | Swim through a DDD ring | on |
| `Shell::Ride` | ride | Get on a Koopa shell | on |
| `Mission::ReachStar` | mission | Reach a star spot from the table | on |
| `Mission::ReachBox` | mission | Reach a star's ! box | on |
| `Mission::ReachRedCoinStar` | mission | Reach the red-coin star spot | on |
| `Mission::ReachHiddenStar` | mission | Reach the hidden-star spot | on |
| `Mission::ReachLoadedStar` | mission | Reach a far loaded star | on |
| `RedCoins::CollectNext` | mission | One more red coin | **never generated** (`OFFER_COUNTERS = false`) |
| `Secrets::FindNext` | mission | One more secret | **never generated** |
| `Coins::Milestone` | mission | Next 10 coins on a course | on |
| `Coins::Reach100` | mission | Reach 100 coins | on |
| `Mission::PssUnder21` | mission | PSS in under 21 s | on |
| `WallKick::Climb` | tech (movement) | Climb 350-3000 with wall kicks | `wallkick` |
| `TripleJumpWallKick::Climb` | tech (movement) | Triple jump + wall kick, +650-1150 | `tjwk` |
| `WallKick::SideFlip` | tech (movement) | Side flip + wall kick, +350-1150 | `wksideflip` |
| `WallKick::Redirect` | tech (movement) | Wall kick as a speed-keeping 180-degree turn | `wkredirect` |
| `WallKick::Chain` | tech (movement) | 3+ kicks between two walls | `wkchain` |
| `Tech::DiveRollout` | tech (movement) | Travel fast with dive rollouts (Plain, Hop, High, Boost) | `diverollout` (`OBRRT_KICKS_OFF`) |
| `Tech::LongJumpChain` | tech (movement) | Travel fast with 2+ long jumps | `ljchain` |
| `Tech::SlideKick` | tech (movement) | Travel fast with a slide kick (or Boost) | `slidekick` (`OBRRT_KICKS_OFF`) |
| `Tech::JumpKick` | tech (movement) | Travel fast with a jump kick | `jumpkick` (`OBRRT_KICKS_OFF`) |
| `Tech::FlipHeight` | tech (movement) | +250 with a side flip or backflip | `flipheight` |
| `Tech::Blj` | tech (movement) | BLJ hint GA at wf-stairs | `blj` (`OBRRT_BLJ_ALIGN_SUCCESS`) |
| `Tech::BljRun` | tech (movement/location) | BLJ from a spot's rough start region (600) to its destination: generic overlay (auto crouch slide + long jump when running away from the destination, Z held / stick back in the air, A+Z on landings from the chain's floor; the GA's stick on the first jump and the presses), progress = chain metric (setup score < 1.1 before a chain: facing, long jump, backwards landings, step-ups), success = chain at the spot's speed (lobby-steps 300, tree 300, free exit 250, else 150) | `blj` / `bljspots`; default: the per-spot pose rules (setup to the TAS pose, riser aim), the only one that succeeds so far; `OBRRT_BLJ_GENERIC=1`: the generic one above, with a static-floor riser probe (core geo `static_floors`) as its first-landing aim; `OBRRT_BLJ_ALIGN`, `OBRRT_BLJ_ALIGN_SUCCESS` (pose mode) |
| `Tech::BljAny` | tech (movement) | BLJ anywhere `blj_ok` holds | `bljany` (+ `bljkinds`) |
| `Tech::HyperWalk` | tech (movement) | Spend backwards hyperspeed on a goal | `hyperwalk` |
| `Tech::HyperClimb` | tech (movement) | Spend +fv hyperspeed on height | `hyperclimb` |
| `Tech::PoleRedirect` | tech (movement) | Redirect hyperspeed through a tree or pole | `poleredirect` |
| `Tech::BombClip` | tech (movement) | Bloated bob-omb push through a thin wall | `bombclip` |
| `Tech::Lblj` | tech (location) | Lobby BLJ past the 8-star door | `lblj` (`OBRRT_BLJ_ALIGN_SUCCESS`) |
| `Tech::LakituSkip` (+ `air` key) | tech (location) | Reach the castle door without the Lakitu dialog | `lakituskip` |
| `Jrb::CaveClip` | tech (location, JRB) | Swim-clip into the JRB cave | `jrbcaveclip` (+ `jrbcavewide`) |
| `Opener::Wf` | tech (location, WF) | WF landing to the high ground in 6/8 s | `wfopener` |
| `Opener::Bob` | tech (location, BoB) | BoB landing to the Chomp plateau in 6/8 s | `bobopener` |
| `Bobomb::GateClip` | tech (star, BoB 6) | Bob-omb push through the Chomp gate | `gateclip` (`OBRRT_DEATHFADE`) |
| `Boss::GrabRelease` | tech (star, BoB 1) | Punch-grab and release King Bob-omb | `kinghits` |
| `Boss::WhompFast` | tech (star, WF 1) | Whomp King in the minimum 114-frame cycle, through to the star | `whomphits` |
| `IslandHop::Jump` | tech (star, BoB 3) | Jump onto the floating island | `islandhop` |
| `IslandHop::Star` | tech (star, BoB 3) | From the island, the star | `islandhop` |
| `Owlless::Cage` | tech (star, WF 5) | Into the cage without Hoot | `owlless` |
| `Wf::TopKick` | tech (star, WF 2) | Wall-kick up the tower top | `wftop` |
| `Ccm::SlideSkip` | tech (star, CCM 1) | Kick out of the slide tube | `ccmslideskip` |
| `Wf::Cless` | tech (star, WF 6) | Cannonless ledge grab | `wfcless` |
| `Jrb::PillarCannonless` | tech (star, JRB 5) | Cannonless via the spire and a wall kick | `jrbcless` |
| `Jrb::PillarHyper` | tech (star, JRB 5) | TAS dock-BLJ route | `jrbclesshyper` |

**Counts per group:**

| Group | Kinds |
|---|---|
| movement/generic | 5 |
| dialog/menu | 6 |
| stars | 1 |
| collectibles | 12 |
| switches | 12 |
| enemies | 16 |
| bosses | 9 |
| NPCs | 18 |
| warps/doors/paintings | 16 (one dead) |
| rides | 6 |
| missions | 10 (two never generated) |
| techniques | 32 (17 movement, 5 location, 10 star incl. GateClip; BljRun is counted under movement) |
| **Total** | **143** |

`TtcClock::EnterAt(speed)` (rides.rs:29) appears only in a `NEEDS:` comment and is not a kind.

## 3. Per-group details

Notation used below:
- `GA p/g/h` means `Budget::Ga { population p, generations g, horizon h }`.
- `w` is `obj_weight`; `sv` is `success_value`.
- "Precise" means `precision()` returns `Precision::Precise`. Anything not marked uses the default, Explore.
- "dist" is the straight-line distance. Unless a row says otherwise, the base heuristic is `dist / 400`.
- "Zone" means `(level, area)`. Unless a row says otherwise, every base criterion includes "same zone as the objective" and "object present".

### 3.1 Movement / generic (mario/)

| Kind | Def | Base criteria | Success | Progress | w / budget | Switch |
|---|---|---|---|---|---|---|
| `Mario::PosDist` | mario/pos.rs:21 | Generated twice per b at a random point: r 300-2500, dy -300..900. Start must be more than 150 from it. Heuristic dist/500 | Within 150 in the zone | Min distance in the zone | w 1; GA 12/4/90; steer to the point | always on |
| `Mario::PosTime` | mario/time.rs | Default (`OBRRT_POSTIME=near`): `OBRRT_POSTIME_N` (2) points drawn like PosDist's, kept where a node of b's zone and context has been within 150, deadline = that earliest frame − margin; starts only from b's context. `OBRRT_POSTIME=self` (or no records): once per b, at `b.pos` with deadline `b.frame - margin`. Start must be more than 150 away and have `frame + dist/60 < deadline` | Within 150 before the deadline. Ranked by last frame | Min distance before the deadline | w 1, sv 1; GA 16/5/150 | always on; `OBRRT_POSTIME_MARGIN` (default 1 frame) |
| `Mario::Settle` | mario/idle.rs:26 | Every b. Start must not be still | First still frame: stationary flag, \|fv\| and \|vy\| < 0.1 | Min \|fv\|+\|vy\| | w 0.3; Macro (60 neutral frames); kinship 2 | always on |
| `Mario::Wait` | mario/idle.rs:74 | With p = 0.3 per b, for n in {15, 30, 60, 90}. Start must be still | Trajectory longer than n | Trajectory length | w 0.15; Macro (n neutral frames); kinship 2 | always on |
| `Mario::GoToCastle` | mario/exit.rs:33 | Per b, outside the castle, grounds and courtyard. Start must be pause-exit-able (`ACT_FLAG_PAUSE_EXIT`). Heuristic -(10*stars + 0.1*coins) | Any castle level reached | 0 | w 2 (0.5 with 0 stars); Policy 300 (pause, Exit Course) when fast, else Macro; temperature 0.5 | always on; `OBRRT_FAST_MODAL` picks Policy or Macro |

### 3.2 Dialog / menu (modal) (menu.rs)

These are generated only when `b.modal()`, and then they are the only objectives offered (lib.rs:66-75). `modal()` returns true for all of them. Each has w 1, Explore, and heuristic `frame/30`.

| Kind | Def | Base criteria | Success | Progress | Budget | Switch |
|---|---|---|---|---|---|---|
| `Dialog::Read` | menu.rs:121 | In the same dialog id, which has no choice, and not yet released | The dialog closes | 0 | Fast: Policy 900, A/B alternating while the text box is open. Old: Macro, A every 4th frame | `OBRRT_FAST_MODAL` (default on) |
| `Dialog::Yes` | menu.rs:122 | Same dialog, which has a choice | It closes with menu line 1 | 0 | As above, stick x -127 | same |
| `Dialog::No` | menu.rs:123 | Same dialog, which has a choice | It closes with menu line 2 | 0 | As above, stick x +127 | same |
| `Menu::SaveAndContinue` | menu.rs:205 | In the save menu | It closes on line 1 | 0 | Fast: Policy 900, which waits for the menu, moves the cursor, then A/Start alternating after menu_timer > 110. Old: Macro | same |
| `Menu::ContinueNoSave` | menu.rs:205 | In the save menu | It closes on line 3 | 0 | as above | same |
| `Menu::SelectAct` | menu.rs:307 | On the act select screen of the same course; when fast, the act must still be an option. Acts offered: act_init when its star is missing; acts 1-6 once 0-5 are owned and the 100-coin star is not; otherwise act_init clamped to 1-6 | Act select ends with the cursor on the target | 0 | Fast: Policy 900, which moves the stick, then A/B once menu_timer >= 11. Old: Macro | same |

Save and Quit is never offered.

### 3.3 Stars (star.rs)

| Kind | Def | Base criteria | Success | Progress | w / budget | Switch |
|---|---|---|---|---|---|---|
| `Star::Collect` | star.rs:125 | Generated per object with `INTERACT_STAR_OR_KEY`, or a spawned star from its spawn on (lib.rs:104, 108). The star must not be owned and not `star_blocked`. Its position is `rest_pos` (a spawned star's home) | `has_star(c, i)` on the first frame; for a key, any rise in progress | Min distance in the zone. Ranked by frames, plus fall frames when GP catch is on | w 40, sv 50; GA 24/6/240 (horizon 360 while spawning); Precise; kinship 10; steer to the star | always on. `OBRRT_GP_CATCH=1` (default off) adds a Z hint just below the star and ranks by the landing frame. `gateclip` softens `star_blocked` |

`star_blocked` (star.rs:103) applies to BoB star 6 while the Chain Chomp gate exists. When `gateclip` is on and `clip::clip_setup(s)` holds, the block is lifted.

The other star-reaching kinds (`Mission::Reach*`) are listed under Missions.

### 3.4 Collectibles

| Kind | Def | Generated from | Base criteria | Success | Progress | w / budget |
|---|---|---|---|---|---|---|
| `Coin::Collect` | coin.rs:28 | Per object with `INTERACT_COIN`, except `bhvHiddenBlueCoin` | present | Coins rise within 250 | min dist | w red 6, blue 4, else 2; GA 16/5/150 |
| `Object::Break` | breakable.rs:36 | Per object with `INTERACT_BREAKABLE` (w 1), plus every `bhvExclamationBox` | Present; a ! box must be in action 2 | Gone, or action >= 3 | min dist | w: ! box with a star 30, with a cap 3, other ! box 2, breakable 1; GA 16/5/150 |
| `HiddenBlueCoin::Collect` | extra/collectibles/blue.rs:126 | Per object, `bhvHiddenBlueCoin` in action 2 (out) | still out | Coins rise and the coin is gone | min dist | w 4, sv 2; GA 16/5/120 |
| `BlueCoinSwitch::Press` | extra/collectibles/blue.rs:48 | Per object, `bhvBlueCoinSwitch` idle | idle | Switch gone or action > 0 | min dist; 500-hdist while pounding within 135 | w 5, sv 3; GA 16/5/150; A/Z hints |
| `Cap::Collect` | extra/collectibles/caps.rs:45 | Per object: wing, metal, vanish or normal cap | Mario doesn't have that cap flag | The flag is set | min dist | w wing 7, normal 6, metal/vanish 4; GA 16/5/150 |
| `RedCoin::Next` | extra/collectibles/coins.rs:146 | Per level, one per zone with red coins whose red-coin star is missing | a red coin in the zone | red_coins rises | dist to the nearest red coin | w 5, sv 3; GA 20/5/240 |
| `CoinFormation::Sweep` | extra/collectibles/coins.rs:62 | Per object, `bhvCoinFormation` | Not spawned yet, or some of its coins left | Coins rose and none of its coins are left | 1000 per coin minus dist | w 2.5; GA 16/5/200 |
| `CrazyBox::Bounce` | extra/collectibles/coins.rs:202 | Per object, `bhvJumpingBox` | Box free or held by Mario | Box gone with a stomach slide or coins | 1000+\|vy\| while holding, else -dist | w 1.5; GA 16/5/200; A/B hints |
| `Heart::Spin` | extra/collectibles/heart.rs:27 | Per object, `bhvRecoveryHeart` when wedges < 8 | alive, hurt | Wedges rise within 600 | -dist + 5\|fv\| near it | w 0.5 per missing wedge; GA 12/4/150 |
| `OneUp::Collect` | extra/collectibles/oneup.rs:41 | Per object: 7 1-up behaviours | Present; hidden ones must be revealed | Lives rise | min dist | w 1; GA 16/5/150 |
| `OneUpTrigger::Touch` | extra/collectibles/triggers.rs:64 | Per object, hidden 1-up triggers | present | Trigger gone | min dist | w 0.5; GA 16/5/200 |
| `SecretTrigger::Touch` | extra/collectibles/triggers.rs:64 | Per object, `bhvHiddenStarTrigger` on courses 1-15, while its star is missing | In BoB, needs the wing cap | Trigger gone | min dist | w 8, sv 6; GA 16/5/200; kinship 10 |

All collectibles are always on and use Explore precision.

### 3.5 Switches and interactables (extra/switches.rs)

The kinds below come from the shared `Switch` struct (switches.rs:112-389). Each is generated per object by behaviour, when the object is `ready` and its star (if any) is not owned. The base criteria are: same zone, star missing, object found and ready. Heuristic dist/400. Progress is -dist unless stated otherwise.

The pound kinds share a hint: Z in the air within 70 of the target (300 for IslandTop), A on the ground near it. All are always on.

| Kind | Def | Object / ready | Success | w / budget |
|---|---|---|---|---|
| `PurpleSwitch::Press` | switches.rs:234 | Floor switches, in action 0 | action 1, 2 or 4 | w 2-4; GA 16/5/150 |
| `CapSwitch::Press` | switches.rs:235 | `bhvCapSwitch`, in action 1 | action >= 2 | w 10, sv 10; GA 20/5/200 |
| `BlueCoinSwitch::Pound` | switches.rs:236 | `bhvBlueCoinSwitch`, in action 0 | gone, or action >= 1 | w 3; GA 20/6/200; Precise |
| `Pillar::Pound` | switches.rs:237 | Castle `bhvWaterLevelPillar`, not pounded (this drains the moat) | action >= 1 | w 7, sv 8; GA 20/6/200; Precise |
| `Post::Pound` | switches.rs:238 | The BoB Chain Chomp's wooden post, not sunk | Sunk to home-189, or 50 since the start. Progress: -10 x remaining depth - hdist | w 20, sv 15; GA 24/6/300; Precise |
| `Post::Circle` | switches.rs:239 | Other wooden posts that still have coins | loot_coins == 0. Progress: angle swept within 380 x 1000 | w 2; GA 20/5/240; steers along a 220 circle |
| `IslandTop::Pound` | switches.rs:240 | THI tiny-island top, in action 0 | action >= 1 | w 8, sv 8; GA 20/6/200; Precise |
| `WaterDiamond::Touch` | switches.rs:241 | `bhvWaterLevelDiamond`, in action 1 | action >= 2 | w 4; GA 20/5/200 |
| `PyramidPillar::Touch` | switches.rs:242 | SSL pyramid pillar detector | detector gone | w 8, sv 6; GA 20/5/240 |
| `Book::Punch` | switches.rs:243 | BBH book switch next in order | action 2 | w 5; GA 16/5/150; B hint |
| `Block::Push` | switches.rs:244 | `bhvPushableMetalBox` | Box moved 200. Progress: 10 x displacement - dist | w 1.5; GA 20/5/200 |
| `Pole::Climb` | switches.rs:245 | Any `INTERACT_POLE`, when not already at its top | At the top of this pole. Progress: 10000 + height while on it | w: giant pole 3, tree 0.4, flag 0.2, else 0.6; GA 20/5/240 |

### 3.6 Enemies

| Kind | Def | Generated from | Success | Progress | w / budget / hints |
|---|---|---|---|---|---|
| `Enemy::Defeat` | enemy.rs:27 | Per object: bounce-top, hit-from-below and koopa interact types. Excludes the `NOT_DEFEATED_BY_VANISHING` behaviours and NPCs | Object gone | min dist | w = sv = 1 with coins, else 0.2; GA 16/5/150; steer 120 above |
| `Object::Grab` | grab.rs:30 | Per object with `INTERACT_GRABBABLE`; a boss is skipped once its star is owned | Holding it. Base: hands free | min dist | w 1, or 20 for King Bob-omb (from behind); GA 16/5/150; Precise; B hint |
| `Boo::HitFromBehind` | extra/enemies/boo.rs:55 | Per object: Boo, ghost-hunt, merry-go-round and caged Boos | Gone, or action 3/4 | -dist - 0.5 x dist to the point 130 behind it | w: caged 15, BBH 2 (6 while its star is missing), else 1; GA 16/5/180; B within 220 |
| `Bully::PushOff` | extra/enemies/bully.rs:83 | Per object, small bullies (`INTERACT_BULLY`, not the big ones) | Gone, or lava/death-plane action | Push from home + 2 x drop - 0.25 x dist | w 1.5 (LLL minions 8); GA 20/5/200; Precise |
| `MrI::Circle` | extra/enemies/circle.rs:41 | Per object, `bhvMrI`, while alive and its star is missing | Gone, or action 3 | Angle swept within 800 x 200 - 0.1 x hdist | w 20 with a star, else 2; GA 16/5/240; circle steer r 300 |
| `MrBlizzard::Circle` | extra/enemies/circle.rs:57 | Per object, `bhvMrBlizzard` | Gone, or action 6 | As Mr. I, within 1000 | w 4 when capless, else 1.5; GA 16/5/240 |
| `Klepto::Hit` | extra/enemies/klepto.rs:53 | Per object, `bhvKlepto` carrying the SSL star or Mario's cap | action 6 (struck) | -dist | w 20 star / 4 cap; GA 20/6/300; Precise; B hint |
| `Moneybag::Reveal` | extra/enemies/moneybag.rs:60 | Per object, hidden moneybag | It turns (action 1) or is gone | -dist | w 0.5; GA 12/4/120 |
| `Moneybag::Stomp` | extra/enemies/moneybag.rs:61 | Per object, `bhvMoneybag` with action < 4 | action 5 (killed; fading away doesn't count) | -dist | w 2; GA 16/5/180 |
| `PiranhaPlant::Sneak` | extra/enemies/piranha.rs:57 | Per object, `bhvPiranhaPlant` | action 5-7 | -dist | w 2; GA 16/5/150; B (dive) within 450 |
| `FirePiranha::Hit` | extra/enemies/piranha.rs:58 | Per object, THI's giant fire piranhas | Health drops, or gone | -dist | w 6 while THI star 1 is missing, else 1; GA 16/5/150 |
| `Chuckya::Throw` | extra/enemies/throw.rs:33 | Per object, `bhvChuckya` | Gone | 1000 while holding, else -dist | w 2; GA 16/5/180; Precise; grab from behind |
| `Bobomb::Throw` | extra/enemies/throw.rs:38 | Per object, `bhvBobomb` | Gone, or exploding | As Chuckya | w 1 with its coin, else 0.2; GA 16/5/180; Precise |
| `Ukiki::ToCage` | extra/enemies/ukiki.rs:63 | Per object, TTM cage Ukiki (bp2 = 0) while the star is missing | Gone, or action 4 | 1000 while holding (+100 in the dialog) | w 20; GA 16/5/240; Precise |
| `Ukiki::RetrieveCap` | extra/enemies/ukiki.rs:64 | Per object, TTM cap Ukiki (bp2 = 1) when capless | Cap back on the head | as above | w 4; GA 16/5/240; Precise |
| `Whomp::GroundPound` | extra/enemies/whomp.rs:73 | Per object, `bhvSmallWhomp` | Gone, or action 8 | While it lies flat: 1000 (+300 on top) - dist to its back; else -dist to the point in front | w 2; GA 16/5/240; Precise; dodge, A/Z hints |

All are always on.

### 3.7 Bosses

The extra/bosses.rs kinds share these properties:
- Base criteria: `Fight::base`, meaning same zone, alive, the boss present and its star not owned.
- Success: `hurt_or`, meaning boss health below its start value (when that is 1-4), or a kind-specific act.
- Precise, `needs_objects`, kinship 10.
- None of them is generated once the boss's star is owned.

| Kind | Def | Generated from | Success (extra) | Progress | w / sv / budget / hints |
|---|---|---|---|---|---|
| `Boss::Hit` | boss.rs:110 | Per object, King Bob-omb (`BOSSES`) while star (1,0) is missing. Stages: Alive, Beaten (act 7), Done (act 8 with the star loaded) | Alive: hp drops or Beaten. Beaten: dialog or Done. Done: star got | Done: -dist to the star. Beaten: 0. Holding: 1000. Alive: -dist | 30 / 10; GA 24/6/200; steer behind him, B |
| `WhompKing::Pound` | extra/bosses.rs:274 | Per object, `bhvWhompKingBoss` hp 1-3, before it dies | action >= 8 | Lying: 1000 - hdist to his back (+500 on top, +500 pounding). Falling: 500. Else: -dist to the point in front | 30 / 10; GA 24/6/300; dodge then back, A/Z |
| `BigBoo::Hit` | extra/bosses.rs:379 | Per object, the 3 Big Boos | Enters HURT or DEAD | 300 if tangible, minus dist | 30 / 10; GA 24/6/240; side approach, B |
| `Wiggler::Stomp` | extra/bosses.rs:442 | Per object, `bhvWigglerHead` while fighting (hp 2-4) | Enters JUMPED_ON | +200 when 50 above, minus hdist | 30 / 10; GA 20/6/200; steer 200 above |
| `Eyerok::PunchEye` | extra/bosses.rs:498 | Per object, `bhvEyerokHand` while fighting; skipped once star (8,3) is owned | Enters DIE | +200 in front, +300 with the eye open, minus dist | 30 / 10; GA 24/6/300; B when the eye is open |
| `Eyerok::Wake` | extra/bosses.rs:560 | Per object, `bhvEyerokBoss` asleep | Boss action != 0 | -dist | 5 / 2; GA 16/5/200 |
| `BigBully::PushOff` | extra/bosses.rs:627 | Per object, the 3 big bullies, while active | Star owned, or lava/death-plane action | 2 x push gained (+150 in knockback) - 0.2 x dist | 30 / 10; GA 24/6/300; circles behind, B |
| `Bowser::GrabTail` | extra/bosses.rs:763 | Per object, `bhvBowser` while the fight is open (keys) | Picking up or holding Bowser | +400 behind him, minus dist to the tail | 20 / 3; GA 20/6/240; swing round, B |
| `Bowser::ThrowAtBomb` | extra/bosses.rs:828 | Per object, `bhvBowser`, as above | hp drops, or DEAD | Holding: 1000 + spin. Thrown: 2500 + (4000 - min distance to a bomb). Else: the grab score | 30 / 15; GA 24/6/240; spin stick, B facing a bomb |

All are always on, except that `Tech::GrabRelease` (kinghits) adds a second King Bob-omb objective, and with `whomphits` on `WhompKing::Pound` keeps only the approach (Whomp King in act 0) and `Boss::WhompFast` takes the fight.

### 3.8 NPCs

Most of these share the `Approach` struct (npcs.rs:157-267):
- Base criteria: same zone, a star minimum, an optional hands-free requirement, the star missing, and the object in its start action.
- Success: the spec's `done` list, usually "dialog within N".
- Progress: -dist to the aim point.
- Hints: steer to the aim point, plus the spec's button within its radius.

| Kind | Def | Generated from | Success | w / budget | Switch |
|---|---|---|---|---|---|
| `Npc::Talk` | npc.rs:28 | Per object, `npc_weight` behaviours: cannon Bob-omb Buddy 8, other Buddy 0.5, star Toads 15, other Toads 1, Koopa the Quick 10 | Waiting for or reading NPC dialog within 400 | w = npc_weight; GA 12/4/150; B within 200 | always on |
| `Sign::Read` | extra/npcs.rs:274 | Per object, signs | Reading the sign within 400 | w 0.15; GA 12/4/120; B within 220 | always on |
| `Yoshi::Talk` | extra/npcs.rs:291 | Per object, `bhvYoshi` idle or walking | Dialog within 400 | w 1; GA 12/4/150 | always on |
| `Lakitu::Intro` | extra/npcs.rs:309 | Per object, the intro camera Lakitu, untriggered | Lakitu's action changes (aim at (0, 850, -1100)) | w 0.3; GA 12/4/150 | not offered while `lakituskip` is on (npcs.rs:821) |
| `Mips::Catch` | extra/npcs.rs:329 | Per object, `bhvMips` with 15+ stars, free | Holding MIPS | w 15, sv 20; GA 24/6/240; Precise; B | always on |
| `Ukiki::Catch` | extra/npcs.rs:348 | Per object, TTM cage Ukiki before it runs | Holding it | w 10; GA 20/5/240; Precise | always on |
| `Ukiki::FollowToCage` | extra/npcs.rs:369 | Per object, the cage Ukiki going to the cage | Dialog within 400 | w 12; GA 16/5/300 | always on |
| `Hoot::Wake` | extra/npcs.rs:387 | Per object, Hoot asleep at his spawn | Within 50, or dialog | w 3; GA 16/5/240 | always on |
| `Hoot::Ride` | extra/npcs.rs:405 | Per object, Hoot awake | `ACT_RIDING_HOOT` | w 4; GA 20/5/240 | always on |
| `KoopaRace::Claim` | extra/npcs.rs:430 | Per object, Koopa the Quick after the race | Dialog within 600 | w 12; GA 16/5/240; Precise | always on |
| `PenguinRace::Talk` | extra/npcs.rs:448 | Per object, the racing penguin waiting | Dialog, or his action changes | w 8; GA 12/4/150 | always on |
| `PenguinRace::Claim` | extra/npcs.rs:480 | Per object, the penguin after a fair win | Dialog within 600 | w 12; GA 16/5/240; Precise | always on |
| `Snowman::WakeBody` | extra/npcs.rs:499 | Per object, the snowman body at rest, star missing | Body action changes, or dialog | w 8; GA 12/4/150 | always on |
| `Snowman::Finish` | extra/npcs.rs:518 | Per object, the snowman head with the body under it | Head action changes, or dialog | w 15; GA 16/5/200 | always on |
| `KoopaRace::ReachFlag` | extra/npcs.rs:594 | Per level, Koopa racing plus the nearest race endpoint | Mario within 390 of the flag while Koopa is 400+ from it. Progress: -dist to the goal. Heuristic (dm + max(dm - dr, 0))/400 | w 20, sv 20; GA 16/5/600; Precise; kinship 10 | always on |
| `PenguinRace::Win` | extra/npcs.rs:595 | Per level, the penguin racing plus the finish line | Within 990 of the finish, past it in z, no cheat | as above | always on |
| `Tuxie::ReturnBaby` | extra/npcs.rs:689 | Per level, one per matching baby penguin for Tuxie's mother | Mother's action becomes 1. Progress: -(remaining carry distance) | w 15, sv 20; GA 20/5/300; Precise; B at the baby | always on |
| `Snowman::LeadBody` | extra/npcs.rs:766 | Per level, the snowman body rolling | Body action 3/4. Progress: -(body to site + 0.25 x Mario to site), +3000 while the checkpoint exists | w 15; GA 16/5/400; Precise | always on |

### 3.9 Warps, doors and paintings

The kinds in the first row share the `WarpEnter` struct (extra/warps.rs:146-263). They are generated per object with `INTERACT_WARP` and classified through `warps/table.rs`. The table's star requirements gate the base criteria.

**Success:** reaching the table's destination zone, or any other zone for warps not in the table. Mario disappearing or fading out at the pad also counts.

**Progress:** closest approach to the pad.

**Weight:** multiplied by `dest_worth`, which is 0.15 when the destination course or fight is done.

| Kind | Def | Generated from | Success / progress | w / budget | Switch |
|---|---|---|---|---|---|
| `SecretEntrance::Enter` / `BowserStage::Enter` / `Teleporter::Enter` / `Pipe::Enter` / `WarpHole::Enter` / `Warp::Enter` | extra/warps.rs:164-170 | Per object `INTERACT_WARP`. The kind is chosen by table class (Secret, Bowser, Teleport), then by behaviour (`bhvWarpPipe`, `bhvFadingWarp`, `bhvWarp`, other) | see above | w: Bowser 15, Secret 12 (vcutm 6), area pipe 5, other area 6, teleport 2, untabled fading 2, other 4. GA 20/6/300 (Secret, Bowser) or 16/5/200 | always on. The `bitdw` star requirement is waived when `lblj` is on and Mario is behind the 8-star door (warps.rs:221) |
| `SecretEntrance::Enter` (key `SecretEntrance totwc`, `SunLookUp`) | extra/warps.rs:923 | Per level, castle lobby with 10+ stars. Base: idle on the sun spot facing -z | `LEVEL_TOTWC` reached. Progress 0 | w 12 x dest_worth; Macro (C-up, look up); temperature 0.5 | always on |
| `SunSpot::Stand` | extra/warps.rs:867 | Per level, castle lobby with 10+ stars, when not already set up | Idle on the sun spot facing within 0x0E00 of -z. Progress: dist + yaw error/64 (+100 if not idle) | w 8 x dest_worth; GA 16/5/200 | always on |
| `BbhEntrance::Enter` | extra/warps.rs:284 | Per object `INTERACT_BBH_ENTRANCE`, the boo cage on the ground | In BBH, or the BBH-enter actions | w 15 x dest_worth; GA 16/5/200 | always on |
| `Tornado::Enter` | extra/warps.rs:345 | Per object `INTERACT_TORNADO` | `ACT_TORNADO_TWIRLING`. Progress: min hdist to the moving tornado | w 3; GA 12/4/150 | always on |
| `StarDoor::Enter` | extra/warps.rs:416 | Per object, castle `INTERACT_DOOR` that needs stars. Base: enough stars | Zone change, or a door action within 300 | w 12 (8+ stars) / 10; 0 without the stars; GA 16/5/180 | always on |
| `KeyDoor::Enter` | extra/warps.rs:477 | Per object, castle key warp doors. Base: enough keys | as StarDoor | w 12 with the keys, else 0; GA 16/5/180 | always on |
| `Door::Enter` | door.rs:52 | Per object, `INTERACT_WARP_DOOR` / `INTERACT_DOOR`. Base: stars and keys enough | Zone change, or a door action within 300 | w 8, sv 2; GA 16/5/180 | always on |
| `Painting::Enter` | painting.rs:61 | Per level in the castle: one per painting in `painting_table.rs` in b's area. Base: course not done, stars >= needed | `s.level == dest`. Progress: min dist to the centre | w 15, sv 5; GA 20/6/300; kinship 10 | always on |
| `Cannon::Enter` | extra/warps.rs:535 | Per object, `bhvCannon` (opened) | `ACT_IN_CANNON` within 300 | w 8 (1 when the course is done); GA 16/5/180; Precise | always on |
| `Cannon::Fire` (far) | extra/warps.rs:712 | Per object, every cannon. Base: in the cannon. Heuristic frame/1e6 | On the ground 1500+ away. Progress: max hdist in flight | w 3 (0.5 when the course is done); GA 16/5/300; Precise | always on |
| `Cannon::Fire` (aimed) | extra/warps.rs:712 | Per object, per `CANNON_TARGETS` entry (BoB island, WF wall, RR box), when the star is missing and a pitch solves the shot | Within 500 of the target, or landed near it | w 10; Macro (solved yaw and pitch, A); Precise | always on |
| `Moat::Drain` (`PillarPound`) | extra/warps.rs:791 | **Never generated** (dead code; `Pillar::Pound` replaced it) | - | (w 6; GA 16/5/200) | - |

### 3.10 Rides (extra/rides/)

| Kind | Def | Generated from | Success | Progress | w / budget | Switch |
|---|---|---|---|---|---|---|
| `Chest::Open` | extra/rides/chest.rs:176 | Per object, a treasure chest in the JRB cave (star 3,2, w 12), JRB ship (3,0, w 6) or DDD (9,1, w 12) set. Base: exactly the earlier chests open | This chest opens | -dist to the point 120 in front; cut off at a buzz | GA 16/5/200 | `chestfacing` (default on) adds a facing penalty and detours around wrong chests |
| `Platform::Board` | extra/rides/platform.rs:168 | Per object, about 20 moving-platform behaviours. Base: Mario not on it | Grounded in its collision box | -dist to its top | w 0.6 x spec (spec 1-5); GA 16/5/150 | always on |
| `Seesaw::Board` | extra/rides/platform.rs:168 | Per object, `bhvSeesawPlatform` | as above, +250 height tolerance | as above | w 0.5; GA 16/5/150 | always on |
| `Platform::Ride` | extra/rides/platform.rs:228 | Per object, the same platforms except seesaws | On it while it moves the spec's travel (200-1000) | 1000 + distance moved while on it, else -dist | w = spec; GA 12/4/300 | always on |
| `WaterRing::SwimThrough` | extra/rides/ring.rs:66 | Per object, the DDD manta and jet-stream ring spawners while their star is missing | An open ring passed (order not checked) | -dist to the nearest open ring | w 8; GA 16/5/200 | always on |
| `Shell::Ride` | extra/rides/shell.rs:44 | Per object, `bhvKoopaShell` idle | Riding-shell action flag | -dist | w 4 in LLL/SSL/BitFS, else 2; GA 16/5/150 | always on |

`TtcClock::EnterAt(speed)` appears only in a `NEEDS:` comment (rides.rs:29). It is not implemented.

### 3.11 Missions (extra/missions.rs)

All missions are generated per level, on courses that have a course number. The per-object `generate` is empty.

| Kind | Def | Generated from | Base criteria | Success | Progress | w / budget | Switch |
|---|---|---|---|---|---|---|---|
| `Mission::ReachStar` | missions.rs:129 | `table::SPOTS` (71 spots, courses 1-24), star missing, act allowed | Same zone, more than 160 away, not `star_blocked` | Within 160, or the star collected | Closest approach | w 30, sv 5; GA 20/5/240; Precise; kinship 10 | always on |
| `Mission::ReachBox` | missions.rs:130 | as above, spot is a ! box | radius 250 | as above | as above | w 25 | always on |
| `Mission::ReachRedCoinStar` | missions.rs:131 | as above | Also needs 8 red coins | as above | as above | w 35 | always on |
| `Mission::ReachHiddenStar` | missions.rs:132 | as above | Also needs no secret triggers left | as above | as above | w 35 | always on |
| `Mission::ReachLoadedStar` | missions.rs:208 | Loaded star objects more than 4500 away and not in SPOTS | star missing, object present | Within 160, or collected | Closest approach | w 30; GA 20/5/240; Precise | always on |
| `RedCoins::CollectNext` | missions.rs:289 | **Never generated**: `OFFER_COUNTERS = false` (missions.rs:507) | - | red_coins rises | closest approach | (w 6; GA 16/5/180) | constant off |
| `Secrets::FindNext` | missions.rs:290 | **Never generated**, as above | - | A trigger is gone | closest approach | (w 6) | constant off |
| `Coins::Milestone` | missions.rs:374 | Courses 1-15 while the 100-coin star is missing. Target: the next multiple of 10 | target-15 <= coins < target | coins >= target | 1000 x coins gained - nearest-coin distance | w 2 + target/20, sv 1; GA 16/5/240 | always on |
| `Coins::Reach100` | missions.rs:374 | As above, with target 100 (also added at 85+ coins) | as above | coins >= 100 | as above | w 10, sv 10; GA 16/5/240 | always on |
| `Mission::PssUnder21` | missions.rs:454 | PSS while star (19,1) is missing | On the entry ledge before the timer (x > 2922, y > 6000) | In the end quad within 620 frames | Closest approach to the end within 620 frames | w 25, sv 10; GA 16/6/630; Precise | always on |
### 3.12 Techniques (extra/tech/, plus extra/enemies/clip.rs)

Each technique is gated by one `OBRRT_TECH` name from `TECHNIQUES` (extra/tech/mod.rs:117-285). Every one of them is on by default, because `OBRRT_TECH` defaults to `all`.

| `OBRRT_TECH` name | Tags | Course |
|---|---|---|
| `wftop` | star | 2 |
| `ccmslideskip` | star | 4 |
| `wfcless` | star, location | 2 |
| `gateclip` | star | 1 |
| `wallkick` | movement | - |
| `tjwk` | movement | - |
| `wksideflip` | movement | - |
| `wkredirect` | movement | - |
| `wkchain` | movement | - |
| `diverollout` | movement | - |
| `ljchain` | movement | - |
| `flipheight` | movement | - |
| `islandhop` | star | 1 |
| `owlless` | star, location | 2 |
| `jrbcless` | star | 3 |
| `jrbcaveclip` | location | 3 |
| `lblj` | location | - |
| `blj` | movement | - |
| `bljany` | movement | - |
| `bljkinds` | movement | - |
| `hyperwalk` | movement | - |
| `bombclip` | movement | - |
| `kinghits` | star | 1 |
| `whomphits` | star | 2 |
| `lakituskip` | location | - |
| `slidekick` | movement | - |
| `jumpkick` | movement | - |
| `wfopener` | location | 2 |
| `bobopener` | location | 1 |
| `poleredirect` | movement | - |
| `hyperclimb` | movement | - |
| `bljspots` | location | - |
| `jrbcavewide` | location | 3 |
| `jrbclesshyper` | star | 3 |
| `chestfacing` | location | - |

That is 33 names. Six of them are modifiers that add no kind of their own:
- `bljkinds` adds extra BljAny kinds.
- `bljspots` adds extra BljRun spots.
- `jrbcavewide` widens Jrb::CaveClip.
- `chestfacing` changes Chest::Open.
- `whomphits` limits WhompKing::Pound to the approach (act 0).
- `hyperwalk` and `poleredirect` also change the BLJ hand-off success.

Almost all technique objectives are generated per level, from `tech::generate_level`. Three come per object instead: `Boss::GrabRelease` from King Bob-omb, `Boss::WhompFast` from the Whomp King and `Tech::LakituSkip` from the intro Lakitu. `Bobomb::GateClip` is generated per level in extra/enemies.rs:106.

"Gated" in the Progress column means progress is 0 (or the start's own score) until the technique's signature action appears in the rollout. See section 5.

#### Movement techniques (tag `movement`)

| Kind | Def | Switch | Base criteria | Success | Progress | w / budget / hints |
|---|---|---|---|---|---|---|
| `WallKick::Climb` | tech/wallkick.rs:385 | `wallkick` | Standing b. Targets: the 2 nearest objects 350-3000 above (marked), else +650 400 ahead. Start within 700 of the anchor | 1+ kick, standing at goal height | **Gated on a kick**: min(y, goal+100) - 0.25 x hdist + 150 per kick + 50 standing | w 1 marked, 0.4 unmarked; GA 16/5/(90+35 per 480 gain, at most 300); Precise; A in the kick window |
| `TripleJumpWallKick::Climb` | tech/wallkick.rs:386 | `tjwk` | Climb with a gain of 650-1150 (fallback +800, 900 ahead) | A kick after a triple jump | **Gated** on a post-triple-jump kick; +150 for the triple jump | w 0.7 / 0.28; GA 16/5/180; Precise; A on jump landings too |
| `WallKick::SideFlip` | tech/wallkick.rs:387 | `wksideflip` | Climb with a gain of 350-1150 and \|fv\| <= 40; adds a wall target at +650 | 1+ kick (a side flip is not required) | **Gated on a kick**; +150 for a side flip | w 0.8 / 0.32; GA 16/5/120; Precise; the `side_flip_overlay` forces A + stick on the turn and in the kick window |
| `WallKick::Redirect` | tech/wallkick.rs:564 | `wkredirect` | b at a wall, fv >= 32, facing it, not kicking | The kick keeps 0.9x the pre-contact fv (>= 32) and closes on a goal 1000 out along the normal | Before the kick: 5fv - 0.25 x dist to the wall. After: 1000 + 10 x kick fv - 0.25 x dist to the goal. Success needs the kick | w 0.4; GA 16/4/90; Precise; `kick_overlay` |
| `WallKick::Chain` | tech/wallkick.rs:710 | `wkchain` | b at a wall; 2 objectives, +700 and +1500 | 3+ kicks, two wall normals 60+ degrees apart, at anchor+gain | **Gated on a kick**: min(y, cap) + 120 per kick + 200 for two walls | w 0.5 / 0.35; GA 16/5/(up to 300); Precise; `kick_overlay` |
| `Tech::DiveRollout` | tech/travel.rs:95 | `diverollout` | `Dash::applies`. Plain: a point 900-2000 away, level. Variants **Hop** (G05: grounded at fv < 30, a point 400-1000 away, level to 300 below), **High** (G17: fv >= 29, 120-350 above) and **Boost** (G07, see SlideKick), each picked by a hash half the time | Within 150 by d/30+6 frames, with a rollout (Hop: a freefall then a dive; High: a dive from a jump at vy > 20) | **Gated**: 0 until the move, then 200 + (d0 - d) | w 0.5; GA 16/5/(limit+10); B/A hints; overlays for Hop, High and Boost |
| `Tech::LongJumpChain` | tech/travel.rs:96 | `ljchain` | As Plain DiveRollout | Within 150 by d/33+6, with 2+ long jumps | **Gated**: 200 x min(n, 2) + (d0 - d) | w 0.5; GA 16/5; Z/A hints |
| `Tech::SlideKick` | tech/kicks.rs:304 | `slidekick` | `KickRun::applies`. A point 900-2400 away, either 250-800 below (speed 40) or level (speed 27). **Boost** sub-mode (G07, half the time when eligible on a downhill floor): 1500-3000 down the fall line | Within 150 by d/speed+8, with a slide kick. Boost: an air action at hs >= 75, then 1500+ from the start, within 60 frames | **Gated**: 200 + (d0 - d). Boost: displacement counts only after air hs > 48 | w 0.5; GA 16/5 (Boost 16/5/70); Z/B/A hints; Boost uses `landing_kick_overlay` |
| `Tech::JumpKick` | tech/kicks.rs:305 | `jumpkick` | A level point 700-1600 away (speed 28) | Within 150 by d/28+8, with a jump kick | **Gated**: 200 + (d0 - d) | w 0.5; GA 16/5; `landing_kick_overlay` (fix `jk`) |
| `Tech::FlipHeight` | tech/flip.rs:76 | `flipheight` | Grounded b; steering point 300-480 up | Standing (or ledge grab) 250+ up within 60 frames, after a side flip or backflip | **Gated**: 0 without a flip, then 100 + gain from the flip | w 0.5; GA 12/4/70; A/Z hints |
| `Tech::Blj` | tech/blj.rs:798 | `blj` | The `wf-stairs` spot (no rule): in its area, star (2,2) missing, within 700 of the start | Chain <= -150 in a long jump. Or, once chain >= 100: the Wild Blue ledge or the star | **Gated** BLJ chain metric + alignment (`progress_toward`). **Builds** speed | w 8, sv 3; GA 24/6/240; Precise; stick-back + A/Z |
| `Tech::BljRun` | tech/blj.rs:798 | `blj` (bob-elevator, lobby-steps) or `bljspots` (lobby-centre, wf-room-steps, lobby-left-pss, castle-seam, bob-slope, ccm-east, ccm-plateau) | Spot open, controllable, fv > -100, within 1500 (`blj`) or 400 (`bljspots`) | Chain <= -target from a slower base. Target: 150; lobby-steps 300; bob-elevator 300 with `poleredirect`; free exits 250 with `hyperwalk`. Or, once chain >= 100: the exit goal, star or tree | **Gated** chain metric + alignment to the goal (the BoB tree with poles). **Builds** speed | w 8, sv 3; GA 48/7/(horizon+150); Precise; `staged_overlay` (setup pose, then the TAS rule; the genome's stick on press frames, and it steers free exits) |
| `Tech::BljAny` | tech/bljany.rs:241 | `bljany` (+ `bljkinds`) | `blj_ok` kinds: Elevator, Ceiling, Slope, Fast (fv <= -53). `bljkinds` adds RisingFloor, WaterSlope, BlockedStep, StepRiser. Up to 2 per b, one per 400 cell and kind | `hyperwalk` off: LJ landing <= -100 800+ away, or a point 2500 behind within 41 frames. `hyperwalk` on: hand-off speed only (LJ at fv <= -150) | **Gated** chain metric, no alignment term. **Builds** speed | w 4, sv 2; GA 32/5/150; Precise; `blj::overlay` |
| `Tech::HyperWalk` | tech/hyperwalk.rs:265 | `hyperwalk` | Hyper b (\|fv\| >= 100 in a walk, landing, LJ or similar) or a `blj_ok` b. Goals: stars, entries, holdables, far places, `BEHIND_STAR_DOORS`. At least 800 from the goal. Heuristic dist/\|fv\| | Within the deadline with \|fv\| > 60 throughout. Any of: within 300 of the goal; 3000+ away and 1500 closer; a star (Star goals); a zone change (Entry goals) | (d0 - dmin)/d0 + 0.2 x speed kept, only over the \|fv\| > 60 prefix. **Spends** speed | w 25 (`SPEND_WEIGHT`), sv 2; GA 32/6/(60-360); Precise; hyperwalk rule overlay |
| `Tech::HyperClimb` | tech/hyperclimb.rs:199 | `hyperclimb` | fv >= 150 on the ground. Targets: generic +1000; the 3 nearest stars, ! boxes or King; the 2 nearest poles; the BoB summit and island | Within 90 frames of a launch: standing on the target, or 1000+ up (generic) | **Gated on a launch**: 0 without one | w 25, sv 2; GA 24/5/90; Precise; keyed overlay (A launch, A held, A on wall, B dive) |
| `Tech::PoleRedirect` | tech/poleredirect.rs:183 | `poleredirect` | \|vel_xz\| >= 150 heading into a pole or tree; plus aimed variants (poles 300+ higher, hyperclimb places) | Within 10 frames of the touch, a carry action with fv >= 0.8 x the stored speed (aimed: travel within 30 degrees of the aim) | Before the touch: closeness of the miss. After: 1 + fv/stored. **Spends/redirects** speed | w 25, sv 2; GA 24/5/40; Precise; keyed overlay (stick + Z + C-up on IDLE, A+B in BRAKING) |
| `Tech::BombClip` | tech/bombclip.rs | `bombclip` | At a near-vertical wall (`wall_normal`), holding a bob-omb or one within 600 (hands free), a floor *seen* (learned map of floors that generated b states stood on, 100-unit cells) 150-300 behind the wall plane within 150 of Mario's height; base states within 300 of the wall point, in front. One per state, key 400 cell + wall octant | Past the plane by 60+, lateral < 150, within 300 of the far floor's height, not holding, a bloated bob-omb (explode / just-launched) or explosion within 500, standing alive (or a star gained) | **Gated**: 0 until a push frame (moved 20+ beyond his speed with a bloated bob-omb within 160); then 1000 + depth gained on push frames only (clamped -900..1500), +500 once a push frame ends past the plane | w 3, sv 3; GA 20/6/240; Precise; keyed rule overlay (fetch, hold-jump air throw towards the wall, regrab as it bloats, carry facing away, release window at the wall); objects every frame |

`OBRRT_KICKS_OFF` (default empty, so every fix is on) turns off the catalogue fixes in kicks.rs and travel.rs. It takes a comma list:
- `jk`: the landing-frame jump-kick overlay and the A|B hint.
- `hop`: the DiveRollout Hop variant and the Plain FREEFALL-B hint.
- `high`: the DiveRollout High variant.
- `boost`: the Boost variant of DiveRollout and SlideKick.
- `all`: all four.

An unknown word panics.

#### Location techniques (tag `location`)

| Kind | Def | Switch | Base criteria | Success | Progress | w / budget / hints |
|---|---|---|---|---|---|---|
| `Tech::Lblj` | tech/lblj.rs:90 | `lblj` | Lobby, fewer than 8 stars, 0 keys, in front of the 8-star door, within 2600 | Chain <= -150 in the lobby. Or, once chain >= 100: BitDW or behind the door | **Gated** chain metric + alignment to the trap. **Builds** speed | w 12, sv 6; GA 24/6/240; Precise; stick-back + A/Z. Also opens the `bitdw` warp from behind the door (warps.rs:221) |
| `Tech::LakituSkip` (edge) | tech/lakituskip.rs:194 (key :198) | `lakituskip` | Per object, intro Lakitu still waiting. Mario in the castle grounds outside the trigger box, y > 700, near the bridge | The castle, or within 350 of the door, with no dialog 34 or moat fall before it | Depth through the box over **safe states only**; 0 before the box | w 12, sv 1; GA 16/5/200; Precise; steer onto and along the ~10-wide strip; `lj_rule` overlay |
| `Tech::LakituSkip air` | tech/lakituskip.rs:199 | `lakituskip` | as edge | as edge | as edge | As edge, plus a crouch-slide LJ onto the strip at z -1450, Z+A chains and a late dive |
| `Jrb::CaveClip` | tech/jrbcaveclip.rs:145 | `jrbcaveclip` (+ `jrbcavewide`) | Swimming in the JRB bay within 3000 of the entry. Wide: any submerged action at hs > 26 | In the cave (z < -907, y < -918) | -dist to the clip point; **only states at hs > 26 count** (speed-gated); stops at the normal mouth | w 3, sv 5; GA 20/6/200; Precise; aim band y -850..-150 (wide: up to 950 at hs > 60) |
| `Opener::Wf` | tech/openers.rs:75 | `wfopener` | WF landing (within 250 of the spawn, \|fv\| < 20). Two deadlines: entry+180 and entry+240 frames | At the high ground (3300, 1600, 2000) by the deadline | -(min distance before the deadline); **not gated** (a pace-lag gate was tried and reverted) | w 6 x 180/limit (6 or 4.5), sv 2; GA 16/6/(limit+10); Precise; Z/A/B hints, wall-kick steer |
| `Opener::Bob` | tech/openers.rs:76 | `bobopener` | BoB landing, as above | At the Chain Chomp plateau (1600, 768, 4150) by the deadline | as above | as above |

#### Star techniques (tag `star`)

| Kind | Def | Switch | Base criteria | Success | Progress | w / budget / hints |
|---|---|---|---|---|---|---|
| `Bobomb::GateClip` | extra/enemies/clip.rs | `gateclip` | BoB, star 6 missing, gate and a bob-omb loaded. Mario in front of the gate within 2500, holding a bob-omb or with one within 1200 (rough start region) | Star (1,5), or behind the bars (endpoint) | An instance of `Tech::BombClip` (the gate as the wall, normal away from the star): its **gated** push-frame progress (0 until a bloated-bob-omb push frame) | w 4, sv 10; GA 20/6/240; Precise; bombclip's keyed rule overlay (always, regardless of the `bombclip` switch). `OBRRT_DEATHFADE=1` (default off): succeed on star 6 only. Also softens `star_blocked` |
| `Boss::GrabRelease` | tech/kinghits.rs:83 | `kinghits` | Per object, King Bob-omb with hp > 0 and star (1,0) missing; Mario holding him or the King in act 1/2/6 | His health drops, act 7, or the star | -dist to behind him; holding 1000 - 2 x frames held; thrown 2000 - 2 x held | w 30, sv 10; GA 24/6/240; Precise; B |
| `Boss::WhompFast` | tech/whomphits.rs:249 | `whomphits` | Per object, Whomp King while star (2,0) missing; controllable; act 0 within 1500, acts 2-8 within 3000, act 9 with the spawned star loaded | Staged: act 0: dialog 114 opens or he chases; alive: his hp below the start's (or act 8/9); act 8: dialog 115 opens or act 9; act 9: the star | 1e5 x real hp drops - (predicted frame of the next hit - start frame): his next possible attack (act-2 entry + 32, seen in the rollout, else optimistic), Mario's straight-line time to his cone and to above his back at 24 u/f, the fixed 37-frame fall, 101-frame flat window; absolute, so waiting never scores; squished ends it | w 30, sv 10; GA 24/6/300; Precise; steer to the cone (act 2), above his back (falling/flat), Z above it; success heuristic = frames |
| `IslandHop::Jump` | tech/islandhop.rs:133 | `islandhop` | BoB, star 3 missing, no wing cap, not in the cannon, near the ledge or summit launch | The star, or on the floating island | -(min distance to the landing), counted until Mario drops below the launch; not gated | w 15, sv 10; GA 24/6/240; Precise; Z/A/B at the edge |
| `IslandHop::Star` | tech/islandhop.rs:267 | `islandhop` | On the island, or the spawned star loaded | Has the star | -min distance to the box or star | w 30, sv 10; GA 24/6/300; Precise; A near the box |
| `Owlless::Cage` | tech/owlless.rs:89 | `owlless` | WF, star 5 missing, not riding Hoot, y >= 3300, within 1600 of the cage | The star, or in the cage (y >= 4086) | -dist to the aim point; +200 in a wall kick near it; not gated | w 15, sv 10; GA 24/6/180; Precise; A on wall hit |
| `Wf::TopKick` | tech/wftop.rs:98 | `wftop` | WF, star 2 missing, y 3500-5326 near the tower axis | Has star 2 | -dist(ledge), +400 after a wall kick above 4400; on top: 1000 - dist(star); not gated | w 6, sv 10; GA 24/6/240; Precise; A in the air near the tower |
| `Ccm::SlideSkip` | tech/ccmslideskip.rs:80 | `ccmslideskip` | CCM slide (area 2), star 1 missing, y > 6300 near the start | The star, or the exit room, or CCM area 1 | -dist(exit door), +500 after a wall kick at the top; not gated | w 6, sv 10; GA 24/6/330; Precise; A near the kick spot |
| `Wf::Cless` | tech/wfcless.rs:91 | `wfcless` | WF, star 6 missing, y > 2500 within 2000 of the drop | Has star 6 | On top: -2000 - hdist(ledge). Below: -dist(star), +500 in a ledge action near it; states below y 1800 are excluded | w 8, sv 50; GA 20/6/200; Precise; no buttons |
| `Jrb::PillarCannonless` | tech/jrbcless.rs:166 | `jrbcless` | JRB, star 5 missing, ! box loaded, within 4000 of the ledge, not in the cannon | The star, or on the platform (never via the cannon) | -dist(ledge), +250 after a wall kick above the water; not gated | w 4, sv 10; GA 24/6/240; Precise; A near the ledge |
| `Jrb::PillarHyper` | tech/jrbcless.rs:336 | `jrbclesshyper` | Dock and slow, or fast (fv < -150) between the dock and the spire, or on the spire at fv < -75 | as PillarCannonless | 100 x BLJ metric, then fast states 2000 + nearness, then 3000 + height on the spire. Spire credit **needs hyper first** (gated). Builds, then spends, inside one objective | w 4, sv 10; GA 24/6/240; Precise; `dock_rule` overlay, then B/A forced in the dive and rollout windows |

`Chest::Open` facing mode (`chestfacing`, tag location) is listed under Rides.

## 4. Switches that change objective behaviour

Every setting is a field of `SearchConfig` (`env/src/config.rs`, the source of truth: each field's doc names its `OBRRT_*` variable and its default; `obrrt-env` fills it from the environment or a TOML file). Production settings are the defaults: the movement library, the move GA, calm jumping, the Lakitu arc, fast painting entries, the swim hold cadence, the ground-pound catch, soft dedup, route chains and the rate sampler are always on and have no switch.

| Variable | Effect on objectives |
|---|---|
| `OBRRT_TECH` | Which technique objectives (section 3.12) exist: a comma list over names and tags (`movement`, `location`, `star`), read left to right; `all` by default |
| `OBRRT_MOVES_EXTRA`, `OBRRT_MOVES_MODES` | The GA also offers the extra movement policies (extra.rs) and the control modes' and speed techniques' policies (cannon, water_air, slide, pole, tech) |
| `OBRRT_MOVE_WEIGHT` | The 0.5-weight techniques weigh this much (0.75) |
| `OBRRT_SWIM_TARGETS` | One more `PosDist` per `b` in water, in the water column |
| `OBRRT_LAKITU_THROUGH_DOOR`, `OBRRT_GROUNDS_FIX`, `OBRRT_KING_LANDING`, `OBRRT_TWOSTAR`, `OBRRT_STATIC_ID`, `OBRRT_GEO` | Lakitu skip through the door; the castle-grounds fixes; the king bob-omb landing; the two-star missions; objects keyed by static id; walkable distances over the level's floors |
| `OBRRT_BLJ_SPOTS`, `OBRRT_BLJ_SPOTS_CATALOG`, `OBRRT_BLJ_DETECTOR` | The BLJ spot registry (tech/bljany/spots.rs): record the spots a run meets and where chains start; the catalogue to seed it from; the clause thresholds file |
| `OBRRT_SPEC`, `OBRRT_SPEC_RUN` | Declarative objectives (spec crate) added to the pool; the run their node targets resolve in |
| `OBRRT_GOAL_FOCUS`, `OBRRT_GOAL_WIDE`, `OBRRT_GOAL_EXPLORE`, `OBRRT_STALL_ETA` | Share of plans drawn toward a goal; far objects still give goals; stalled goals explore; the stall stop |
| `OBRRT_POSGOAL_WEIGHT`, `OBRRT_POSGOAL_SHARE` | The web miner's position goal (`Mario::UserGoal`): its pool weight and the share of plans from inside its bubble |

Constants that act as switches:
- `OFFER_COUNTERS = false` (missions.rs) keeps RedCoins::CollectNext and Secrets::FindNext off.
- `OBJECT_RADIUS = 5000` (lib.rs) is the object radius for generation.

The other `OBRRT_*` variables control the search, the GA, memory, stopping and resuming; see `env/src/config.rs`. Switches that earlier logs (RUNS.md, GOALS.md) name and that are gone were deleted by the condense refactor; the tag `archive/pre-condense` has them.

## 5. Conventions the code follows

- **Straight-line heuristics.** Base heuristics, progress and steering use straight-line distance (`dist`, `hdist`), usually divided by 400, or by speed for the hyperspeed spenders (`dist/|fv|`). No objective uses a walking or geodesic metric. Height and yaw terms are added explicitly where they matter (e.g. SunSpot, BljRun pose).
- **Domain knowledge only in objectives/.** Coordinates, behaviour names, actions, star indices and TAS routes all live in this crate (objectives depends only on obrrt-core). The search engine is generic: it sees only the `Objective` trait (weights, budgets, precision, steer hints, overlays). Route optimisation is optional and off by default (`OBRRT_ROUTE=0`).
- **Technique progress must be signature-gated.** The RUNS.md "audit" row found that walks were being credited as techniques. For example, WallKick::Climb succeeded without a kick, the BLJ kinds' nodes were cut before the BLJ, and `flipheight` had no flip. The "auditfix" and "bljtree" rows fixed this. The rule is that a technique objective's progress (and the node it keeps) counts only once the technique's own action is in the rollout:
  - a kick for the WallKick family;
  - the move count for Dash and KickRun;
  - a flip for FlipHeight;
  - hs > 26 for CaveClip;
  - a bloated-bob-omb push frame for BombClip and GateClip;
  - a growth landing for the BLJ metric (with the node only from 2 growths);
  - a launch for HyperClimb;
  - the |fv| > 60 prefix for HyperWalk.

  Otherwise the walk that sets up a technique gets credited as the technique, and fills the tree with non-technique nodes. The exceptions are the star-specific routes (IslandHop, Owlless, Wf::TopKick, Ccm::SlideSkip, Wf::Cless, Jrb::PillarCannonless) and the openers. They use plain distance progress, because their goal is a place or star and not a move. The audit tried a pace-lag gate for the openers and reverted it.
- **Build-speed vs spend-speed (BLJ family).** The BLJ kinds only **build** speed. Tech::Blj, BljRun, Lblj and BljAny score the graded chain metric (blj.rs `metric_at`, `progress`). They succeed on a chain speed reached in the trajectory: -150, or the spot's hand-off speed (300 at lobby-steps and at bob-elevator with `poleredirect`, 250 at free exits with `hyperwalk`). Places, stars and the tree count only after the chain reached -100. Separate **spend** kinds turn that speed into travel, all at `SPEND_WEIGHT` 25:
  - Tech::HyperWalk takes backwards |fv| >= 100 to a goal;
  - Tech::PoleRedirect turns |v| >= 150 into +fv at a tree;
  - Tech::HyperClimb turns +fv >= 150 into height.

  With `hyperwalk` on, BljAny drops its place successes and succeeds on hand-off speed only. Jrb::PillarHyper is the one kind that builds and spends inside a single objective.
- **blj.rs is changing.** Another agent is editing extra/tech/blj.rs right now, for a BLJ alignment reward. This file describes the code as of 2026-10-25:
  - `progress_toward` adds 1.5 x max(0, cos) to the goal bearing once a chain has grown;
  - `OBRRT_BLJ_ALIGN_SUCCESS` gates speed success on 30-degree alignment.

  Line numbers and constants in blj.rs may shift.
