//! Coin objectives beyond a single `Coin::Collect`:
//!
//! - `CoinFormation::Sweep(id)`: every coin of a bhvCoinFormation (lines,
//!   rings, arrows; coin.inc.c). The formation spawns its uncollected coins
//!   when Mario is within 2000 (action 0 -> 1) and despawns them past 2100
//!   (action 2 -> 0). The spawned coins switch their behaviour to
//!   bhvYellowCoin (bhv_coin_formation_spawn_loop), so they're found as
//!   yellow coins within the formation's extent: lines span ±320 along it,
//!   rings 300 around, vertical rings 400 up, ground ones snap to the floor.
//! - `RedCoin::Next`: the course's next red coin, steering to the nearest
//!   one left, from states with more red coins first. The 8th makes the red
//!   coin star appear (bhvHiddenRedCoinStar / bhvBowserCourseRedCoinStar,
//!   hidden_star.inc.c), which `Star::Collect` then takes.
//! - `RedCoin::Ride`: HMC's red coins on the checkered track platform's route
//!   (the "elevate" of "Elevate for 8 Red Coins"); see [`RedRide`].
//! - `CrazyBox::Bounce(id)`: a bhvJumpingBox (jumping_box.inc.c) is grabbed;
//!   pressing A while holding it starts ACT_CRAZY_BOX_BOUNCE
//!   (mario_actions_stationary.c / _moving.c), and the third bounce's landing
//!   breaks it into 5 coins and puts Mario in ACT_STOMACH_SLIDE
//!   (mario_actions_airborne.c, act_crazy_box_bounce). Dropping it instead
//!   just deletes it on landing, with no coins.

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::INTERACT_COIN;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::{approach, find, gone};

/// How far from a formation's origin its coins can be.
const FORMATION_RADIUS: f32 = 450.0;
const FORMATION_HEIGHT: f32 = 800.0;

/// sm64.h.
const ACT_STOMACH_SLIDE: u32 = 0x008C0453;
const ACT_CRAZY_BOX_BOUNCE: u32 = 0x000008AE;

/// Close to the box to holding it: lining up and the B grab (~1/3 s).
const BOX_GRAB_FRAMES: f64 = 10.0;
/// Holding it to the first bounce: the A press.
const BOX_START_FRAMES: f64 = 2.0;
/// Air time of each bounce: 2 vy / 4 (gravity) for act_crazy_box_bounce's
/// vy 45, 60 and 100.
const BOUNCE_AIR: [f64; 3] = [22.5, 30.0, 50.0];
/// Mario's gravity in the air, per frame (apply_gravity).
const GRAVITY: f64 = 4.0;
/// `mem` of CrazyBox::Bounce: bit 0 = in a bounce last frame; f[0] = bounces
/// started (1..3), f[1] = last frame's vy.
const BOUNCING: u32 = 1;

/// Frames per coin still to take after the nearest: the spacing of a
/// formation's coins (lines 160, rings ~240) at PACE.
const COIN_FRAMES: f64 = 8.0;
/// Coins assumed in a formation not spawned yet (lines and rings hold 5 or
/// 8; whether any are left is unknown until it spawns).
const UNSPAWNED_COINS: f64 = 5.0;

pub struct Sweep {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl Sweep {
    pub fn new(b: &St, o: &Obj) -> Sweep {
        Sweep { id: o.id, zone: b.zone(), pos: o.pos }
    }

    fn coins<'a>(&self, s: &'a St) -> impl Iterator<Item = &'a Obj> + 'a {
        let c = self.pos;
        s.objects().iter().filter(move |o| {
            o.interact_type == INTERACT_COIN
                && o.bhv_name() == "bhvYellowCoin"
                && ((o.pos[0] - c[0]).powi(2) + (o.pos[2] - c[2]).powi(2)).sqrt() < FORMATION_RADIUS
                && (o.pos[1] - c[1]).abs() < FORMATION_HEIGHT
        })
    }

    fn nearest(&self, s: &St) -> Option<[f32; 3]> {
        self.coins(s).map(|o| o.pos).min_by(|a, b| s.dist(*a).total_cmp(&s.dist(*b)))
    }
}

impl Objective for Sweep {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "CoinFormation::Sweep"
    }
    fn key(&self) -> String {
        format!("CoinFormation {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        2.5
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Not yet spawned (whether any are left is unknown then), or coins out.
        find(s, self.zone, self.id).is_some_and(|f| f.action == 0 || self.coins(s).next().is_some())
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 200 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let Some(f) = find(s, self.zone, self.id) else { return Eta::Lost };
        let left = self.coins(s).count();
        // Spawned, none of its coins left, and some taken in this attempt.
        if f.action == 1 && left == 0 && s.coins > start.coins && s.alive() {
            return Eta::Done;
        }
        if f.action != 1 || left == 0 {
            // Not spawned (or despawning, or its coins not out yet on the
            // spawn frame): go there, then the coins.
            return Eta::Frames(frames_for(s.dist(self.pos)) + COIN_FRAMES * UNSPAWNED_COINS);
        }
        // Each coin taken drops ETA by COIN_FRAMES minus the walk to the next.
        Eta::Frames(approach(s, self.nearest(s)).frames() + COIN_FRAMES * (left - 1) as f64)
    }
    fn success_value(&self) -> f64 {
        1.5
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.nearest(s).unwrap_or(self.pos))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub struct RedNext {
    zone: (u16, u8),
    /// The red coin star's (course, index), when known.
    star: Option<(i16, u8)>,
}

impl RedNext {
    /// One per zone, when `b` has red coins left.
    pub fn new(b: &St) -> Option<RedNext> {
        let objs = b.objects();
        if !objs.iter().any(|o| o.bhv_name() == "bhvRedCoin") {
            return None;
        }
        let index = objs.iter().find_map(|o| match o.bhv_name() {
            "bhvHiddenRedCoinStar" => Some((o.beh_params >> 24) as u8),
            // Bowser courses have a single star.
            "bhvBowserCourseRedCoinStar" => Some(0),
            _ => None,
        });
        let star = index.filter(|&i| i <= 6 && (1..=24).contains(&b.course)).map(|i| (b.course, i));
        if star.is_some_and(|(c, i)| b.has_star(c, i)) {
            return None;
        }
        Some(RedNext { zone: b.zone(), star })
    }

    fn nearest(&self, s: &St) -> Option<[f32; 3]> {
        if s.zone() != self.zone {
            return None;
        }
        // OBRRT_RED_V2=1 with OBRRT_GEO=1: the nearest by walking distance where the mesh knows it (a coin
        // behind a wall or up a cliff is far), else straight.
        let d = |p: [f32; 3]| if red_v2() { obrrt_core::geo::path_dist(s.zone(), s.pos, p).unwrap_or_else(|| s.dist(p)) } else { s.dist(p) };
        s.objects().iter().filter(|o| o.bhv_name() == "bhvRedCoin").map(|o| o.pos).min_by(|a, b| d(*a).total_cmp(&d(*b)))
    }
}

/// OBRRT_RED_V2=1: red coins by walking distance and with longer attempts (600 frames, not 240). The red-coin
/// stars were the search's weakest: HMC `RedCoin::Next` 1 success of 768 attempts; t1 spent 3.1 B frames on
/// HMC's and ended with 7 of 8 coins.
fn red_v2() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_RED_V2").is_ok_and(|v| v == "1"))
}

/// RedCoin::Next start bias per red coin already had: v1's 2 heuristic
/// units a coin left, in frames.
const RED_HAD_BIAS_FRAMES: f64 = 2.0 * obrrt_core::V1_UNIT_FRAMES;

impl Objective for RedNext {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "RedCoin::Next"
    }
    fn key(&self) -> String {
        format!("RedNext {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // OBRRT_GOAL_WIDE=1: a goal (weight 25), so the goal focus builds on
        // the states nearest the next coin, ranked with their red coins
        // (engine `index_frontier`, OBRRT_GOAL_WIP_W). No red-coin star had
        // been found by any tree before 2026-11-09: states with 2 of 8 were a
        // few hundred among tens of thousands and almost never a start.
        static W: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *W.get_or_init(|| obrrt_core::sys::env::var("OBRRT_GOAL_WIDE").is_ok_and(|v| v == "1")) { 25.0 } else { 5.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.nearest(s).is_some() && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: if red_v2() { 600 } else { 240 } })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.alive() && s.red_coins > start.red_coins {
            return Eta::Done;
        }
        // WDW's downtown: the coins' boxes break only on dry ground (rides/water.rs `town_drain_frames`).
        match approach(s, self.nearest(s)) {
            Eta::Frames(f) => Eta::Frames(f + crate::extra::rides::water::town_drain_frames(s)),
            e => e,
        }
    }
    fn start_bias(&self, s: &St) -> f64 {
        // v1 ranked by red coins left (2 heuristic units each), then the
        // nearest: more already had first.
        -(s.red_coins.clamp(0, 8) as f64) * RED_HAD_BIAS_FRAMES
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.nearest(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

/// HMC's checkered platform (`bhvPlatformOnTrack`, behParam 0x09A40000 in hmc/script.c, path 4 of
/// sPlatformOnTrackPaths = hmc_seg7_trajectory_0702B86C): it waits for Mario to stand on it for 20 frames, then
/// follows the waypoints below at 10 units a frame (platform_on_track.inc.c, `act_move_along_track`, ~1400
/// frames to the end); if Mario is off it for 150 frames it blinks and resets to waypoint 0.
pub(crate) const HMC_TRACK: [[f32; 3]; 14] = [
    [1272.0, 2000.0, 5120.0],
    [1272.0, 2000.0, 3500.0],
    [1800.0, 2100.0, 3000.0],
    [4100.0, 2100.0, 3000.0],
    [4640.0, 2160.0, 3980.0],
    [5416.0, 2100.0, 3642.0],
    [6720.0, 1950.0, 4440.0],
    [6700.0, 1880.0, 5040.0],
    [6060.0, 1880.0, 5640.0],
    [5440.0, 1850.0, 6200.0],
    [3820.0, 1990.0, 7130.0],
    [1270.0, 1990.0, 7130.0],
    [1272.0, 2000.0, 6000.0],
    [1272.0, 2000.0, 5400.0],
];
const TRACK_SPEED: f64 = 10.0;
/// Standing on it before it starts (`act_wait_for_mario`: oTimer > 20).
const TRACK_WAIT: f64 = 20.0;
/// Blinking out and resetting after Mario got off (150 + 40 blinks).
const TRACK_RESET: f64 = 190.0;
/// A red coin counts as on the track when within this horizontally and 0..TRACK_LIFT above it (a jump from the
/// platform takes it: coin 7 (4640, 2250, 3980) is 90 above waypoint 4, coin 8 (6400, 2090, 5320) is 210 above
/// the leg from waypoint 7 to 8).
const TRACK_NEAR: f32 = 250.0;
const TRACK_LIFT: f32 = 400.0;
/// Done when the platform is this close (along the track) before the coin's nearest point, with Mario on it.
const TRACK_LEAD: f32 = 120.0;

pub(crate) fn track_len() -> Vec<f32> {
    let mut l = vec![0.0];
    for w in HMC_TRACK.windows(2) {
        let n = l.len();
        let h = ((w[1][0] - w[0][0]).powi(2) + (w[1][2] - w[0][2]).powi(2)).sqrt();
        l.push(l[n - 1] + h);
    }
    l
}

/// Position along the track (arc length) of the point nearest `p` horizontally, and that distance and the
/// track's height there.
pub(crate) fn track_project(p: [f32; 3]) -> (f32, f32, f32) {
    let len = track_len();
    let mut best = (0.0, f32::MAX, 0.0);
    for (i, w) in HMC_TRACK.windows(2).enumerate() {
        let (dx, dz) = (w[1][0] - w[0][0], w[1][2] - w[0][2]);
        let l2 = dx * dx + dz * dz;
        let t = (((p[0] - w[0][0]) * dx + (p[2] - w[0][2]) * dz) / l2).clamp(0.0, 1.0);
        let (qx, qz) = (w[0][0] + t * dx, w[0][2] + t * dz);
        let d = ((p[0] - qx).powi(2) + (p[2] - qz).powi(2)).sqrt();
        if d < best.1 {
            best = (len[i] + t * l2.sqrt(), d, w[0][1] + t * (w[1][1] - w[0][1]));
        }
    }
    best
}

/// `RedCoin::Ride` (HMC star 2, "Elevate for 8 Red Coins"): ride the checkered track platform to where it
/// passes the red coins still left. The red coin room's coins 7 and 8 hang over its route and nowhere else is
/// within reach of them: coin 8 at (6400, 2090, 5320) sits on an isolated beam 160 above the platform's leg
/// (6700, 1880, 5040) to (6060, 1880, 5640), reached by standing on the platform for ~900 frames (its route
/// from waypoint 0 is 8.7 k units at 10 a frame; `RedCoin::Next` heads for the nearest coin in a straight line
/// over 240 or 600 frames, i.e. away from the platform, and its states with 6 or 7 coins never boarded it:
/// t1 steps 299-307 and 681-685 ended at 6 to 7 of 8). The ETA is the board (walk to it, 20 frames wait) plus
/// the track distance to the nearest ahead coin; Done: standing on it within `TRACK_LEAD` before that coin
/// (the jump is then `RedCoin::Next`'s, a few frames), or a red coin taken.
pub struct RedRide {
    zone: (u16, u8),
}

impl RedRide {
    pub fn new(b: &St) -> Option<RedRide> {
        if b.level as u32 != obrrt_core::layout::LEVEL_HMC || b.has_star(6, 1) || b.red_coins >= 8 {
            return None;
        }
        let r = RedRide { zone: b.zone() };
        (r.plat(b).is_some() && r.coins(b).next().is_some()).then_some(r)
    }

    fn plat<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone {
            return None;
        }
        s.objects().iter().find(|o| o.bhv_name() == "bhvPlatformOnTrack")
    }

    /// Arc positions of the red coins left that the track passes.
    fn coins<'a>(&self, s: &'a St) -> impl Iterator<Item = f32> + 'a {
        s.objects().iter().filter(|o| o.bhv_name() == "bhvRedCoin").filter_map(|o| {
            let (a, d, y) = track_project(o.pos);
            (d < TRACK_NEAR && (0.0..TRACK_LIFT).contains(&(o.pos[1] - y))).then_some(a)
        })
    }

    fn on(&self, s: &St, p: &Obj) -> bool {
        s.floor_obj == Some(p.slot) && ((s.pos[0] - p.pos[0]).powi(2) + (s.pos[2] - p.pos[2]).powi(2)).sqrt() < 400.0
    }

    /// Frames until the platform is `lead` before the nearest coin ahead of it (or, past the last, after it has
    /// reset and come round); with the flag: whether it is already there.
    fn ride(&self, s: &St, p: &Obj) -> (f64, bool) {
        let len = *track_len().last().unwrap();
        let (a, d, _) = track_project(p.pos);
        // A platform off its track (falling at the end) restarts.
        let a = if d < 100.0 { a } else { 0.0 };
        let waiting = if p.action == 1 { TRACK_WAIT } else { 0.0 };
        let mut best: Option<f32> = None;
        for c in self.coins(s) {
            let ahead = c - TRACK_LEAD - a;
            if ahead >= -TRACK_LEAD * 2.0 {
                best = Some(best.map_or(ahead, |b: f32| b.min(ahead)));
            }
        }
        match best {
            Some(ahead) => (waiting + ahead.max(0.0) as f64 / TRACK_SPEED, ahead <= 0.0),
            None => {
                let first = self.coins(s).fold(f32::MAX, f32::min);
                ((len - a) as f64 / TRACK_SPEED + TRACK_RESET + TRACK_WAIT + (first - TRACK_LEAD).max(0.0) as f64 / TRACK_SPEED, false)
            }
        }
    }
}

impl Objective for RedRide {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "RedCoin::Ride"
    }
    fn key(&self) -> String {
        format!("RedRide {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // A goal (over OBRRT_GOAL_MIN 20) like RedCoin::Next: the goal focus then builds on the states nearest
        // the platform. At 8 (a stepping stone) the first test (h2, 15 M frames) made 10 attempts in all.
        25.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        !s.has_star(6, 1)
            && self.coins(s).next().is_some()
            && self.plat(s).is_some_and(|p| p.action != 4 && !(self.on(s, p) && self.ride(s, p).1))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 400 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.alive() && s.red_coins > start.red_coins {
            return Eta::Done;
        }
        let Some(p) = self.plat(s) else { return Eta::Lost };
        let (ride, there) = self.ride(s, p);
        if self.on(s, p) {
            return if there && s.alive() { Eta::Done } else { Eta::Frames(ride) };
        }
        // Not on it: the straight line to the platform (walking distance is wrong there: the mesh has no floor under it, only the
        // pit's far below, so `path_dist` is off by thousands), then the ride.
        Eta::Frames(frames_for(s.dist(p.pos)) + 10.0 + ride)
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.plat(s).map(|p| p.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub struct CrazyBox {
    id: ObjId,
    zone: (u16, u8),
}

impl CrazyBox {
    pub fn new(b: &St, o: &Obj) -> CrazyBox {
        CrazyBox { id: o.id, zone: b.zone() }
    }

    fn holding(&self, s: &St) -> bool {
        s.held().is_some_and(|h| h.id == self.id)
    }
}

impl Objective for CrazyBox {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "CrazyBox::Bounce"
    }
    fn key(&self) -> String {
        format!("CrazyBox {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.5
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Free (not mid-throw or dropped), or already in Mario's hands.
        find(s, self.zone, self.id).is_some_and(|o| o.held_state == 0 || self.holding(s)) && (!s.holding || self.holding(s))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 200 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // Broken by the bounces: gone, Mario stomach-sliding or coins up.
        if gone(s, self.zone, self.id) {
            return if s.action == ACT_STOMACH_SLIDE || s.coins > start.coins { Eta::Done } else { Eta::Lost };
        }
        let Some(o) = find(s, self.zone, self.id) else { return Eta::Lost };
        let vy = s.vel[1] as f64;
        if s.action != ACT_CRAZY_BOX_BOUNCE {
            mem.flags &= !BOUNCING;
            let bounces: f64 = BOUNCE_AIR.iter().sum();
            let grab = if self.holding(s) { 0.0 } else { frames_for(dist(s.pos, o.pos)) + BOX_GRAB_FRAMES };
            return Eta::Frames(grab + BOX_START_FRAMES + bounces);
        }
        // A new bounce: entering the action, or vy kicked up on a landing.
        if mem.flags & BOUNCING == 0 || vy > mem.f[1] + 10.0 {
            mem.f[0] = (mem.f[0] + 1.0).min(3.0);
        }
        mem.flags |= BOUNCING;
        mem.f[1] = vy;
        // This bounce's landing (falling to the floor from vy), then the rest.
        let land = (vy + (vy * vy + 2.0 * GRAVITY * (s.pos[1] - s.floor_y).max(0.0) as f64).sqrt()) / GRAVITY;
        let rest: f64 = BOUNCE_AIR[mem.f[0] as usize..].iter().sum();
        Eta::Frames(land + rest)
    }
    fn success_value(&self) -> f64 {
        1.5
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        find(s, self.zone, self.id).map(|o| o.pos)
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Grab it when close, then bounce on it.
        if self.holding(s) {
            return A_BUTTON;
        }
        match find(s, self.zone, self.id) {
            Some(o) if dist(s.pos, o.pos) < 200.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        find(s, self.zone, self.id).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{obj, st};
    use obrrt_core::fold;

    /// Approach, holding, three bounces (counted in `mem`), broken with the
    /// coins: the ETA falls at each phase change and the end is `Done`.
    #[test]
    fn crazy_box_bounces() {
        let bx = || obj("bhvJumpingBox", 6, [500.0, 0.0, 0.0], 0, 0);
        let mut far = st(9, [0.0; 3], vec![bx()]);
        far.coins = 3;
        let mut held = far.clone();
        held.pos = [500.0, 0.0, 0.0];
        held.held_slot = Some(6);
        held.holding = true;
        let bounce = |vy: f32, y: f32| {
            let mut s = held.clone();
            s.action = ACT_CRAZY_BOX_BOUNCE;
            s.vel[1] = vy;
            s.pos[1] = y;
            s
        };
        let mut broken = st(9, [600.0, 0.0, 0.0], vec![]);
        broken.coins = 8;
        let traj = vec![
            far.clone(),
            held.clone(),
            bounce(41.0, 45.0),
            bounce(-40.0, 5.0),
            bounce(56.0, 60.0),
            bounce(-55.0, 5.0),
            bounce(96.0, 100.0),
            bounce(-95.0, 10.0),
            broken,
        ];
        let c = CrazyBox::new(&far, &far.objects()[0]);
        let (met, etas) = fold(&c, &traj);
        assert_eq!(met, Some(8));
        let f: Vec<f64> = etas[..8].iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA must fall: {f:?}");
        // Dropped and deleted without coins: lost.
        let gone = st(9, [600.0, 0.0, 0.0], vec![]);
        assert_eq!(fold(&c, &[far, gone]).1[1], Eta::Lost);
    }

    /// HMC's track platform: walk to it, ride it (ETA falls 1 a frame, the wait counted), Done standing on it
    /// 120 before red coin 8's nearest point (6400, 2090, 5320), which sits 210 above the leg 7 -> 8.
    #[test]
    fn red_ride_follows_the_track() {
        use obrrt_core::layout::LEVEL_HMC;
        let coin = |p| obj("bhvRedCoin", 5, p, 0, 0);
        let plat = |p: [f32; 3], act| obj("bhvPlatformOnTrack", 9, p, act, 0);
        let objs = |pl: [f32; 3], act| vec![coin([6400.0, 2090.0, 5320.0]), plat(pl, act)];
        let base = st(LEVEL_HMC, [800.0, 2048.0, 5120.0], objs([1272.0, 2000.0, 5120.0], 1));
        let r = RedRide::new(&base).expect("coin 8 is on the track");
        let at = |pl: [f32; 3], on: bool| {
            let mut s = st(LEVEL_HMC, pl, objs(pl, 2));
            s.pos[1] += 60.0;
            s.floor_obj = on.then_some(9);
            s
        };
        let wait = {
            let mut s = at([1272.0, 2000.0, 5120.0], true);
            s.objects = base.objects.clone();
            s
        };
        let mid = at([4640.0, 2160.0, 3980.0], true);
        let late = at([6700.0, 1880.0, 5040.0], true);
        let there = at([6476.0, 1880.0, 5250.0], true);
        let (a, b, c, d, e) = (
            r.eta(&base, &base, &mut Mem::default()).frames(),
            r.eta(&wait, &base, &mut Mem::default()).frames(),
            r.eta(&mid, &base, &mut Mem::default()).frames(),
            r.eta(&late, &base, &mut Mem::default()).frames(),
            r.eta(&there, &base, &mut Mem::default()),
        );
        assert!(a > b && b > c && c > d, "{a} {b} {c} {d}");
        assert_eq!(e, Eta::Done, "late {d}");
        // Off the platform at the same place is further than on it.
        let off = at([4640.0, 2160.0, 3980.0], false);
        assert!(r.eta(&off, &base, &mut Mem::default()).frames() >= c);
    }
}
