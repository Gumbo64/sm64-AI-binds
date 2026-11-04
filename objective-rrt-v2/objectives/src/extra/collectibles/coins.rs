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
        s.objects().iter().filter(|o| o.bhv_name() == "bhvRedCoin").map(|o| o.pos).min_by(|a, b| s.dist(*a).total_cmp(&s.dist(*b)))
    }
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
        5.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.nearest(s).is_some() && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.alive() && s.red_coins > start.red_coins {
            return Eta::Done;
        }
        approach(s, self.nearest(s))
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
}
