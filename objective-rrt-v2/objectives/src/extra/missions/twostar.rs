//! Two stars per course visit (`OBRRT_TWOSTAR=1`, default off): the
//! 100-coin star plus an act star, as the 120-star TAS does in all 15 main
//! courses (ref/tas/star_strategies.md, §2 item 4).
//!
//! Why it works in the game: the 100-coin star doesn't end the visit. When
//! the coin count passes 100 in a main course, `interact_coin` (interaction.c)
//! calls `bhv_spawn_star_no_level_exit(6)` (sparkle_spawn_star.inc.c), which
//! sets INT_SUBTYPE_NO_EXIT on the star; `interact_star_or_key` then picks
//! ACT_STAR_DANCE_NO_EXIT and Mario plays on in the course. Any other star
//! ends it, and the coin count restarts at 0 on the next entry. So the
//! coins must come before the act star, in one visit.
//!
//! Why the search didn't: once any branch grabs the act star, every state at
//! the old star count is behind the frontier and retired, the coin-collecting
//! ones too (the most coins in any course in the hy/dn runs: 39). The
//! generic `OBRRT_KEEP_WIP` keeps states with work under way the frontier
//! lacks (`St::wip`); with this switch the coin count toward a missing
//! 100-coin star is such work (`wip_coins`, installed as the core's wip
//! extra). The 100-coin star's own grab then moves the frontier while Mario
//! is still in the course, so its act stars stay open from there.
//!
//! Also, with the switch:
//! - `Coins::Milestone` / `Coins::Reach100` weigh more (`milestone_weight`)
//!   and steer by `coin_target` instead of to the nearest coin.
//! - `TwoStar::CoinRun`: 10 more coins in one attempt, steering to the
//!   densest nearby coins (coin-dropping enemies count, by their loot),
//!   biased toward an act star still missing (the coins along the way).

use std::sync::OnceLock;

use obrrt_core::layout::{INTERACT_COIN, LEVEL_CCM, LEVEL_DDD, LEVEL_LLL, LEVEL_SL, LEVEL_SSL, LEVEL_THI, LEVEL_TTM, LEVEL_WDW};
use obrrt_core::{Eta, Mem, Objective, St, course_of_level, dist, frames_for};

/// `OBRRT_TWOSTAR=1`.
pub fn on() -> bool {
    obrrt_core::config::current().objectives.twostar
}

/// Installs `wip_coins` as the core's wip extra (with the switch on; once).
pub fn install() {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        if on() {
            obrrt_core::set_wip_extra(wip_coins);
            obrrt_core::geo::want_static_floors();
        }
    });
}

/// From this many coins a state's coins count as work under way.
const WIP_FROM: i16 = 20;

/// The main course `s` is in, when its 100-coin star is still missing.
fn hundred_course(s: &St) -> Option<i16> {
    let course = course_of_level(s.level as u32);
    ((1..=15).contains(&course) && !s.has_star(course, 6)).then_some(course)
}

/// Whether `s` is in a main course whose 100-coin star is missing (for the other objectives' gating).
pub fn goal_on(s: &St) -> bool {
    on() && hundred_course(s).is_some()
}

/// A state that can no longer reach 100: its coins plus every coin source it sees (`coin_sources`) fall short
/// by more than `HOPELESS_SLACK` (sources not seen: enemies that are not spawned yet). The courses have almost no
/// coins to spare (JRB's entry: 101 in all; its leg 5 took 20 of a blue coin switch's 30, the switch is gone
/// after its 240 frames, and the legs went on 94 -> 94 -> 94 for 8 legs, 7 M-frame trees each, from a state
/// that could not get the star; SSL, DDD likewise, t1 steps 406-419, 556-567). Needs the objects.
pub fn hopeless(s: &St) -> bool {
    s.objects.is_some() && s.coins + coin_sources(s).iter().map(|x| x.1).sum::<f32>() as i16 + HOPELESS_SLACK < 100
}

/// Coins a state may seem short and still count (see [`hopeless`]).
const HOPELESS_SLACK: i16 = 3;

/// Work toward the 100-coin star: the coin count, in a main course whose
/// 100-coin star is missing, from `WIP_FROM` coins up to 99; none when [`hopeless`].
pub fn wip_coins(s: &St) -> i64 {
    match hundred_course(s) {
        Some(_) if (WIP_FROM..100).contains(&s.coins) && !hopeless_in_area(s) => s.coins as i64,
        _ => 0,
    }
}

/// [`hopeless`], only for a course with a single area (whose coins are all in view): in the others (BoB, WF, JRB, BBH,
/// HMC, TTC, RR; not SSL's pyramid, DDD's sub, SL, CCM, WDW's downtown, LLL, THI, TTM) the coins of the other areas aren't in the object pool.
fn hopeless_in_area(s: &St) -> bool {
    matches!(course_of_level(s.level as u32), 1 | 2 | 3 | 5 | 6 | 14 | 15) && hopeless(s)
}

/// `Coins` objective weight with the switch on (None: the default).
pub fn milestone_weight(target: i16) -> Option<f64> {
    on().then(|| if target >= 100 { 30.0 } else { 6.0 + target as f64 / 5.0 })
}

/// Coins gained for a `CoinRun` success.
const RUN_COINS: i16 = 10;

/// Whether a coin formation's coins float (they don't drop to the floor:
/// spawn_coin_in_formation's `sp38`): a vertical line or ring (types 1, 3)
/// or the 0x10 flag. Up in the air, like BoB's five sky rings at y 3400-5317.
fn floats(o: &obrrt_core::Obj) -> bool {
    let t = (o.beh_params >> 16) & 0xFF;
    t & 0x10 != 0 || matches!(t & 7, 1 | 3)
}

/// Whether a floating formation ([`floats`]) is out of reach without the wing cap: its lowest coin (the
/// formation's own y) more than `JUMP_REACH` over the floor under it, and not under water (swimming gets
/// there). Every floating formation used to count as air: JRB's coin legs stopped at 94 coins with four
/// 5-8 coin lines/rings at y -4420 .. 1160 (the sunken ship, the cliffs: types 0x11-0x13) never counted as
/// coins to get (t1 steps 535-547); SSL and RR have such lines and rings too. Without the zone's static
/// floors (`OBRRT_TWOSTAR` asks for them; they come once a state is seen in control) the old rule holds.
/// Cached per formation.
fn needs_wings(s: &St, o: &obrrt_core::Obj) -> bool {
    if !floats(o) {
        return false;
    }
    type Key = ((u16, u8), [i32; 3]);
    static CACHE: OnceLock<std::sync::RwLock<std::collections::HashMap<Key, bool>>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key: Key = (s.zone(), [o.pos[0] as i32, o.pos[1] as i32, o.pos[2] as i32]);
    if let Some(&v) = cache.read().unwrap().get(&key) {
        return v;
    }
    let Some(floors) = obrrt_core::geo::static_floors(s.zone()) else { return true };
    let [x, y, z] = o.pos;
    let wet = obrrt_core::geo::static_water(s.zone()).is_some_and(|w| w.iter().any(|b| b.is_water() && b.contains(x, z) && b.y > y));
    let v = !wet && obrrt_core::geo::floor_below(&floors, x, z, y + 300.0).is_none_or(|f| y - f > JUMP_REACH);
    cache.write().unwrap().insert(key, v);
    v
}

/// How high over the floor a formation's lowest coin may be and still be had by jumping (a triple jump peaks
/// near 300-400 over the take-off; Mario's reach adds ~160 above his feet).
const JUMP_REACH: f32 = 450.0;

/// Wing cap on (`MARIO_WING_CAP` in m->flags).
fn winged(s: &St) -> bool {
    s.mario_flags & 0x08 != 0
}

/// Coin sources in `s`: (position, coins): loaded coins (`oDamageOrCoinValue`,
/// 5 for a blue coin) and objects that drop coins when beaten
/// (`oNumLootCoins`: goombas, piranha plants' blue coin, ...). Also coin
/// formations not spawned yet (their `value`, read.rs). A formation up in the
/// air counts only with the wing cap on; without it, with the wing cap switch
/// pressed, the wing cap's ! box (or the cap out of it) counts for the air
/// coins. BoB's coin legs stopped at 39-47 coins with five sky rings (40
/// coins) and three wing cap boxes left (the user: "pretty sure you need wing
/// cap for the last coin rings", 2026-11-10).
fn coin_sources(s: &St) -> Vec<([f32; 3], f32)> {
    let objs = s.objects();
    let air: f32 = objs.iter().filter(|o| o.bhv_name() == "bhvCoinFormation" && o.action == 0 && needs_wings(s, o)).map(|o| o.value.max(0) as f32).sum();
    let want_cap = air > 0.0 && !winged(s) && s.save_flags & 0x2 != 0;
    let mut out: Vec<([f32; 3], f32)> = Vec::new();
    // Blue coins not out yet (action < 2, invisible and intangible) are had by the blue coin switch: ground pound
    // it (blue_coin.inc.c; they last 240 frames after). Each waits on the nearest switch, and is counted there
    // with its 5 coins (DDD's area 2 had six, 30 coins, 5000 units from where t1's coin legs ended at 63).
    let switches: Vec<&obrrt_core::Obj> = objs.iter().filter(|o| o.bhv_name() == "bhvBlueCoinSwitch" && o.action == 0).collect();
    let mut pending = vec![0.0f32; switches.len()];
    for o in objs.iter() {
        let t = (o.beh_params >> 16) & 0xFF;
        match o.bhv_name() {
            "bhvCoinFormation" if o.action == 0 && o.value > 0 => {
                if !needs_wings(s, o) || winged(s) {
                    out.push((o.pos, o.value as f32));
                }
            }
            "bhvCoinFormation" => {}
            "bhvHiddenBlueCoin" if o.action < 2 => {
                if let Some(i) = (0..switches.len()).min_by(|&i, &j| dist(switches[i].pos, o.pos).total_cmp(&dist(switches[j].pos, o.pos))) {
                    pending[i] += 5.0;
                }
            }
            // A box's coins (exclamation_box.inc.c sExclamationBoxContents: 4 one, 5 three, 6 ten coins; the box
            // goes when broken).
            "bhvExclamationBox" if matches!(t, 4..=6) && o.action <= 2 => out.push((o.pos, [1.0, 3.0, 10.0][t as usize - 4])),
            "bhvExclamationBox" if want_cap && o.action <= 2 && t == 0 => out.push((o.pos, air)),
            "bhvWingCap" if want_cap => out.push((o.pos, air)),
            // With the cap on: an open cannon, for a shot at the rings (zt-ring3, BoB at 39 coins:
            // with the cap in a cannon, TwoStar::WingRing 20 of 26 and 65 coins; the other seed
            // never got from the cap to a cannon).
            "bhvCannon" if air > 0.0 && winged(s) && s.action != obrrt_core::layout::ACT_IN_CANNON => out.push((o.pos, air)),
            _ if o.interact_type == INTERACT_COIN => out.push((o.pos, o.value.max(1) as f32)),
            _ if o.loot_coins > 0 && o.health >= 0 => out.push((o.pos, o.loot_coins as f32)),
            _ => {}
        }
    }
    for (sw, n) in switches.iter().zip(pending) {
        if n > 0.0 {
            out.push((sw.pos, n));
        }
    }
    // The coins of the course's other areas: at the way there (a warp or a warp floor), so the coin plan walks to it
    // when this area is used up ([`other_areas`]).
    let seen: f32 = out.iter().map(|x| x.1).sum();
    // Only once this area is nearly used up (fewer than 10 coins seen in it): SSL's pyramid has no way back, and
    // its 59 at the entrance would draw Mario in with the outside's coins still left (lost once he is in).
    for (_, left, pos) in if seen < 10.0 { other_areas(s, seen) } else { Vec::new() } {
        out.push((pos, left));
    }
    out
}

/// Coins each area of a course with several areas holds (fixed coins plus what enemies drop), from the level files:
/// tools/seg120/coin_inventory.py (coin_inventory.json). Only the courses whose 100-coin star needs them. SSL's
/// outside is 72 in all (t1's coin legs stopped at 72 twice) and the pyramid (area 2) 59; DDD's sub (area 2) 64.
const AREA_COINS: &[(u32, u8, i16)] = &[
    (LEVEL_CCM, 1, 83), (LEVEL_CCM, 2, 77), (LEVEL_LLL, 1, 94), (LEVEL_LLL, 2, 24), (LEVEL_SSL, 1, 72), (LEVEL_SSL, 2, 59),
    (LEVEL_DDD, 1, 42), (LEVEL_DDD, 2, 64), (LEVEL_SL, 1, 79), (LEVEL_SL, 2, 38), (LEVEL_WDW, 1, 110), (LEVEL_WDW, 2, 66),
    (LEVEL_TTM, 1, 61), (LEVEL_TTM, 2, 12), (LEVEL_TTM, 3, 37), (LEVEL_TTM, 4, 13), (LEVEL_THI, 1, 66), (LEVEL_THI, 2, 29),
    (LEVEL_THI, 3, 36),
];

/// Coins the course's other areas still hold, (area, coins, the way there from `s`'s area): the coin counter is per
/// visit of the level, not per area (level_update.c: numCoins is reset only on entering the level), and a coin taken
/// stays taken when the area is left and entered again (coin.inc.c sets the respawn bits of its spawn info), so
/// what an area still holds is its inventory less what was taken there. The state doesn't say where its coins came
/// from, so: the coins taken in `s`'s own area are its inventory less the sources still loaded in it (`seen`), and
/// the rest of `s.coins` came from the other areas (all of it from the one, in a course of two areas). Without this
/// every coin of another area was invisible to `coin_sources`, `plan` aimed at nothing once the area was used up
/// (SSL, t1 steps 774-787: 72 of the outside's 72, 59 coins in the pyramid never counted) and the campaign's
/// `area_has_coins` judged a move there by the loaded objects only.
fn other_areas(s: &St, seen: f32) -> Vec<(u8, f32, [f32; 3])> {
    let level = s.level as u32;
    let here = AREA_COINS.iter().find(|a| a.0 == level && a.1 == s.area as u8).map_or(0, |a| a.2) as f32;
    let elsewhere = (s.coins as f32 - (here - seen).max(0.0)).max(0.0);
    let mut out = Vec::new();
    for &(l, area, total) in AREA_COINS.iter().filter(|a| a.0 == level && a.1 != s.area as u8) {
        let left = total as f32 - elsewhere;
        if left < 1.0 {
            continue;
        }
        if let Some(pos) = super::super::warps::area_exits(l, s.area as u8).into_iter().find(|e| e.0 == area).map(|e| e.1) {
            out.push((area, left, pos));
        }
    }
    out
}

/// Frames per coin left that no source covers (a coin not in view, or behind a mechanism: a long way).
const UNSEEN_COIN_FRAMES: f64 = 150.0;
/// Most stops a [`plan`] counts one by one; the rest is `FRAMES_PER_COIN` each.
const PLAN_STOPS: usize = 14;

/// The coin route from `s` for `need` more coins: greedily the source with the best coins over (walk + 300
/// units), until the coins are had; the frames of the walk (the first leg by walking distance when the mesh is
/// on, `OBRRT_GEO=1`), and the first stop. The ETA it replaces, the nearest coin's distance plus 40 frames a
/// coin, said nothing of where the rest are: states at the same count looked equal whether the remaining 15
/// coins lay 500 or 8000 units off, and a leg's best state (most coins) was a dead end as often as not (JRB,
/// RR, SSL: legs 0 -> 0 -> 0 at 94, 72, 64 coins). `star` biases the choice away from the missing act star's
/// side, as [`coin_target`] always did.
fn plan(s: &St, need: i16, star: Option<[f32; 3]>) -> (f64, Option<[f32; 3]>) {
    let mut src = coin_sources(s);
    let mut at = s.pos;
    let mut need = need as f32;
    let mut frames = 0.0;
    let mut first = None;
    for stop in 0..PLAN_STOPS {
        if need <= 0.0 {
            return (frames, first);
        }
        let along = |c: [f32; 3]| star.map_or(0.0, |p| 0.3 * (dist(c, p) - dist(at, p)).max(0.0));
        let Some((i, _)) = src
            .iter()
            .enumerate()
            .map(|(i, &(c, v))| (i, v.min(need) / (dist(at, c) + along(c) + 300.0)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
        else {
            break;
        };
        let (c, v) = src.swap_remove(i);
        let d = if stop == 0 { s.path_dist(c) } else { dist(at, c) };
        frames += frames_for(d);
        if stop == 0 {
            first = Some(c);
        }
        need -= v;
        at = c;
    }
    if need > 0.0 {
        frames += need as f64 * if src.is_empty() { UNSEEN_COIN_FRAMES } else { super::FRAMES_PER_COIN };
    }
    (frames, first)
}

/// ETA for the coin count to reach `goal`: [`plan`]'s walk (`Done` at the goal).
fn plan_eta(s: &St, goal: i16, star: Option<[f32; 3]>) -> Eta {
    if s.coins >= goal {
        return Eta::Done;
    }
    Eta::Frames(plan(s, goal - s.coins, star).0.max(1.0))
}

/// [`plan_eta`] for the `Coins` milestones (without the act star's bias).
pub(crate) fn coins_plan_eta(s: &St, goal: i16) -> Eta {
    plan_eta(s, goal, None)
}

/// Where to steer for coins from `s`: the first stop of the coin route ([`plan`]) to 100, a little away from
/// `star` (the missing act star).
pub fn coin_target(s: &St, star: Option<[f32; 3]>) -> Option<[f32; 3]> {
    plan(s, (100 - s.coins).max(1), star).1
}


/// `TwoStar::CoinRun`: `RUN_COINS` more coins (or 100) in one attempt, in a
/// main course whose 100-coin star is missing.
pub struct CoinRun {
    level: u16,
    course: i16,
    /// A missing act star's spot in this zone and act, if any (for the bias).
    star: Option<[f32; 3]>,
}

impl CoinRun {
    pub fn new(b: &St) -> Option<CoinRun> {
        if !on() || b.coins >= 100 {
            return None;
        }
        let course = hundred_course(b)?;
        let star = super::table::SPOTS
            .iter()
            .filter(|p| p.level == b.level as u32 && p.area == b.area && !b.has_star(p.course, p.index) && p.act_ok(b))
            .map(|p| p.pos)
            .min_by(|a, c| b.dist(*a).total_cmp(&b.dist(*c)));
        Some(CoinRun { level: b.level, course, star })
    }

    fn target(&self, s: &St) -> Option<[f32; 3]> {
        coin_target(s, self.star)
    }
}

impl Objective for CoinRun {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "TwoStar::CoinRun"
    }
    fn key(&self) -> String {
        format!("TwoStar coins {}", self.course)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        8.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.level == self.level && s.coins < 100 && !s.has_star(self.course, 6)
    }
    /// [`super::coins_eta`] to the start's coins + `RUN_COINS` (at most
    /// 100), steering by `coin_target`; `Done` there, alive, in the level
    /// (v1's success); the level left or Mario dead: `Lost`.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 360 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.level != self.level || !s.alive() {
            return Eta::Lost;
        }
        let goal = (start.coins + RUN_COINS).min(100);
        plan_eta(s, goal, self.star)
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.target(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == self.level && !s.has_star(self.course, 6)
    }
}

/// `TwoStar::WingRing`: with the wing cap on, a coin ring up in the air
/// ([`floats`]): 5 more coins (or 100) in one attempt, steering straight to
/// the nearest ring with coins left (in flight the straight line is the
/// way). The flight policies (moves/water_air.rs `fly-plan`, `fly-high`)
/// steer by the target. With only `CoinRun` and the `Coins` milestones the
/// wing cap was had (Cap::Collect 95%) and Mario flew (200 flying states in
/// 50 M frames from BoB at 39 coins) but no ring was collected: 32 CoinRun
/// attempts in all, 0 successes.
pub struct WingRing {
    level: u16,
    course: i16,
}

/// Coins a ring attempt is for.
const RING_COINS: i16 = 5;
/// How far under its ring a `WingRing` start may be.
const RING_BELOW: f32 = 800.0;

impl WingRing {
    pub fn new(b: &St) -> Option<WingRing> {
        if !on() || b.coins >= 100 || !winged(b) {
            return None;
        }
        let course = hundred_course(b)?;
        ring_target(b)?;
        Some(WingRing { level: b.level, course })
    }
}

/// Where to fly for the air rings: the nearest coin of a ring once its coins are out (bhvCoinFormationSpawn,
/// within 2000 of Mario), else the nearest ring's centre (a vertical ring, type 3, is a 200-radius circle about
/// a point 200 above the formation; coin.inc.c). A coin's hitbox is a 100-radius cylinder, so a flight through
/// the centre misses every coin; steering at the formation's own position (its lowest point) got WingRing 0-4%
/// from free flight (zt-fly-s6/s7, zt-fly2-s6/s7).
fn ring_target(s: &St) -> Option<[f32; 3]> {
    // The nearest ring's formation position (what cannon shots aimed at: 65 and 60 coins from 39).
    s.objects()
        .iter()
        .filter(|o| o.bhv_name() == "bhvCoinFormation" && o.value > 0 && needs_wings(s, o))
        .map(|o| o.pos)
        .min_by(|a, b| s.dist(*a).total_cmp(&s.dist(*b)))
}

impl Objective for WingRing {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: true }
    }
    fn kind(&self) -> &'static str {
        "TwoStar::WingRing"
    }
    fn key(&self) -> String {
        format!("TwoStar ring {}", self.course)
    }
    /// Over OBRRT_GOAL_MIN (20): a goal, so plans are drawn toward it.
    fn obj_weight(&self, _b: &St) -> f64 {
        25.0
    }
    /// From high enough: flight loses speed climbing, and from a take-off on
    /// the ground it topped out at y 2400-2460 with 2-12 units/frame left
    /// (BoB, rings at 3400-5317: the nearest flying state 1037 below one,
    /// 48 attempts, none). So it starts within `RING_BELOW` under the ring
    /// or above it, to dive onto it, or in a cannon (the cannon policy aims
    /// by the target: BoB's sky rings are the cannon's, act 5). High enough
    /// alone starved it: no state with the cap on came within 800 of a
    /// ring's height in 2 x 50 M frames.
    fn base_criteria(&self, s: &St) -> bool {
        s.level == self.level && s.coins < 100 && !s.has_star(self.course, 6) && winged(s)
            && (s.action == obrrt_core::layout::ACT_IN_CANNON || ring_target(s).is_some_and(|t| s.pos[1] >= t[1] - RING_BELOW))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 600 })
    }
    /// The cap lost, the level left or Mario dead: `Lost`.
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.level != self.level || !s.alive() || (!winged(s) && s.coins < start.coins + RING_COINS) {
            return Eta::Lost;
        }
        let goal = (start.coins + RING_COINS).min(100);
        super::coins_eta(s.coins, goal, ring_target(s).map(|c| s.dist(c)))
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        ring_target(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == self.level && !s.has_star(self.course, 6)
    }
}

/// A text dump of `s`'s coin sources and the coin route to 100 (the probe tool, with `OBRRT_TWOSTAR=1`).
pub fn debug_plan(s: &St) -> String {
    let (frames, first) = plan(s, (100 - s.coins).max(1), None);
    let mut src = coin_sources(s);
    src.sort_by(|a, b| s.dist(a.0).total_cmp(&s.dist(b.0)));
    let total: f32 = src.iter().map(|x| x.1).sum();
    let mut t = format!("coins {} sources {} worth {total}; route to 100: {frames:.0} frames, first {first:?}\n", s.coins, src.len());
    t += &format!("   static floors known: {}\n", obrrt_core::geo::static_floors(s.zone()).is_some());
    for o in s.objects().iter().filter(|o| o.bhv_name() == "bhvCoinFormation") {
        t += &format!("   formation {:?} params {:#x} value {} act {} needs_wings {}\n", o.pos, o.beh_params, o.value, o.action, needs_wings(s, o));
    }
    for (p, v) in src.iter().take(40) {
        t += &format!("   {v:>4} at {:>6.0} {:>6.0} {:>6.0}  d {:.0}\n", p[0], p[1], p[2], s.dist(*p));
    }
    t
}
