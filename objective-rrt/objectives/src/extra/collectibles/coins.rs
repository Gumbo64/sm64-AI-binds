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
//! - `CrazyBox::Bounce(id)`: a bhvJumpingBox (jumping_box.inc.c) is grabbed;
//!   pressing A while holding it starts ACT_CRAZY_BOX_BOUNCE
//!   (mario_actions_stationary.c / _moving.c), and the third bounce's landing
//!   breaks it into 5 coins and puts Mario in ACT_STOMACH_SLIDE
//!   (mario_actions_airborne.c, act_crazy_box_bounce). Dropping it instead
//!   just deletes it on landing, with no coins.

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::INTERACT_COIN;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use super::common::{best, closest, find, gone};

/// How far from a formation's origin its coins can be.
const FORMATION_RADIUS: f32 = 450.0;
const FORMATION_HEIGHT: f32 = 800.0;

/// sm64.h.
const ACT_STOMACH_SLIDE: u32 = 0x008C0453;

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
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Spawned, none of its coins left, and some taken in this attempt.
        let c0 = traj[0].coins;
        traj.iter().position(|s| {
            s.coins > c0 && s.alive() && find(s, self.zone, self.id).is_some_and(|f| f.action == 1) && self.coins(s).next().is_none()
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let c0 = traj[0].coins;
        best(traj, |s| {
            (s.zone() == self.zone).then(|| {
                let d = s.dist(self.nearest(s).unwrap_or(self.pos)) as f64;
                (s.coins - c0) as f64 * 1000.0 - d
            })
        })
    }
    fn success_value(&self) -> f64 {
        1.5
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
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
        s.objects().iter().filter(|o| o.bhv_name() == "bhvRedCoin").map(|o| o.pos).min_by(|a, b| s.dist(*a).total_cmp(&s.dist(*b)))
    }
}

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
        5.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.nearest(s).is_some() && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(p) = self.nearest(s) else { return f64::MAX };
        (8 - s.red_coins.clamp(0, 8)) as f64 * 2.0 + s.dist(p) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let r0 = traj[0].red_coins;
        traj.iter().position(|s| s.zone() == self.zone && s.alive() && s.red_coins > r0)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, |s| self.nearest(s))
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 5, horizon: 240 }
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
    fn base_heuristic(&self, s: &St) -> f64 {
        if self.holding(s) {
            return 0.0;
        }
        find(s, self.zone, self.id).map_or(f64::MAX, |o| 3.0 + s.dist(o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Broken by the bounces: gone, Mario stomach-sliding or coins up.
        let c0 = traj[0].coins;
        traj.iter().position(|s| gone(s, self.zone, self.id) && (s.action == ACT_STOMACH_SLIDE || s.coins > c0))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |s| {
            let o = find(s, self.zone, self.id)?;
            Some(if self.holding(s) { 1000.0 + s.vel[1].abs() as f64 } else { -(dist(s.pos, o.pos) as f64) })
        })
    }
    fn success_value(&self) -> f64 {
        1.5
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
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
