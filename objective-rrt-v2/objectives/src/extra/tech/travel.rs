//! Fast-travel technique objectives: `Tech::DiveRollout` (`diverollout`) and
//! `Tech::LongJumpChain` (`ljchain`).
//!
//! The duplicate cell ignores speed and action, so "did a dive rollout" is
//! no success by itself. Success is instead about time: get within
//! [`RADIUS`] of a point `p` (about 900–2000 units from `b`, straight line,
//! roughly level) within `dist(start, p) / speed` frames of the start, with
//! the technique in the segment. The speeds are above what running gives
//! (Mario's top walking speed on flat ground is 32, winning travel segments
//! end at about 21–23 u/frame) and below what the chains hold:
//!
//! - **Dive rollout** (`speed` 30). The dive adds 15 forward speed, capped at
//!   48 (set_mario_action_airborne, mario.c); B on the ground needs
//!   `forwardVel >= 29` and the stick past 48 (check_ground_dive_or_punch,
//!   mario_actions_moving.c), in the air `forwardVel > 28`, else it's a jump
//!   kick (mario_actions_airborne.c). Landing goes to ACT_DIVE_SLIDE, where A
//!   or B rolls out (act_dive_slide): ACT_FORWARD_ROLLOUT, 30 up, forward
//!   speed kept (air drag only above 32). It lands in
//!   ACT_FREEFALL_LAND_STOP, which keeps the speed only if Mario walks on
//!   (stick held) at once; walking at 29+ dives again. Chains hold about
//!   32–48.
//! - **Long-jump chain** (`speed` 33, above running's 32). A in ACT_CROUCH_SLIDE (Z while
//!   running) with `forwardVel > 10` long-jumps: forward speed ×1.5, capped
//!   at 48, 30 up, gravity 2 (mario.c, mario_step.c); air drag only above 48.
//!   On ACT_LONG_JUMP_LAND, A with Z held long-jumps again
//!   (act_long_jump_land, sLongJumpLandAction). Chains hold 48.
//!
//! Steering hints only make the buttons available on alternate frames
//! (`steer_button`); the GA does the timing.
//!
//! Dive-rollout variants (TAS catalogue G05, G17, G07; one is picked per b
//! from those b qualifies for, half the time, else the plain one; same kind,
//! so they share its pick weight):
//!
//! - **Hop** (`hop`, G05 edge dive-hop). act_freefall turns B into ACT_DIVE
//!   at any speed (mario_actions_airborne.c:535), while jumps dive only above
//!   28. The dive adds 15 (capped at 48) and lands next frame as DIVE_SLIDE,
//!   which steps off an edge, seam or post top into FREEFALL again: 0 → 48 in
//!   about 6 frames. Start grounded below 30, a target 400–1000 away and up
//!   to 300 below; success = there by `dist / 24 + 6` with a FREEFALL → DIVE
//!   (or DIVE_SLIDE) in the segment. A rule overlay presses B on every
//!   FREEFALL frame, nothing on DIVE_SLIDE until 45 (below that the
//!   genome's), then A. Under `hop` the plain variant's
//!   hint also gets `ACT_FREEFALL => B` (its air arm needed fv > 28).
//! - **High** (`high`, G17 high dive). A on a landing frame jumps (vy 42 +
//!   fv/4, ×0.8 speed; a double jump 52), B the next frame dives with that vy
//!   kept and fv back to 48: about +300 height at full dive speed. Start
//!   grounded at 29+, a target 120–350 above and 400–900 away; success =
//!   within 150 by `dist / 30 + 6` with a dive entered from a jump while
//!   rising (vy > 20). The overlay presses A on landing frames and B on a
//!   jump's first frame.
//! - **Boost** (`boost`, G07 slope dive-slide): kicks.rs [`Boost`].

use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_MOVING, ACT_FLAG_ON_POLE, ACT_FLAG_SWIMMING, ACT_FLAG_HANGING};
use obrrt_core::{Eta, GamePad, Mem, Objective, PadOverlay, St, buttons, frames_for};

use super::kicks::{Boost, move_weight};

/// Actions (include/sm64.h).
pub(super) const ACT_DIVE: u32 = 0x0188088A;
pub(super) const ACT_DIVE_SLIDE: u32 = 0x00880456;
pub(super) const ACT_FORWARD_ROLLOUT: u32 = 0x010008A6;
pub(super) const ACT_LONG_JUMP: u32 = 0x03000888;
pub(super) const ACT_LONG_JUMP_LAND: u32 = 0x00000479;
pub(super) const ACT_CROUCH_SLIDE: u32 = 0x04808459;
pub(super) const ACT_FREEFALL: u32 = 0x0100088C;
pub(super) const ACT_JUMP: u32 = 0x03000880;
pub(super) const ACT_DOUBLE_JUMP: u32 = 0x03000881;
pub(super) const ACT_JUMP_LAND: u32 = 0x04000470;
pub(super) const ACT_FREEFALL_LAND: u32 = 0x04000471;
pub(super) const ACT_DOUBLE_JUMP_LAND: u32 = 0x04000472;
pub(super) const ACT_JUMP_LAND_STOP: u32 = 0x0C000230;
pub(super) const ACT_DOUBLE_JUMP_LAND_STOP: u32 = 0x0C000231;
pub(super) const ACT_FREEFALL_LAND_STOP: u32 = 0x0C000232;

/// Arrival radius, as PosDist's.
const RADIUS: f32 = 150.0;
/// Start bias per u/f of forward speed (a running start helps): v1's
/// `dist / 500 - forward_vel / 32` worth 500 / 32 units a u/f, ~0.52 frames
/// at PACE.
pub(super) const RUNNING_START_FRAMES: f64 = 500.0 / 32.0 / obrrt_core::PACE;
/// Start distances to the target an attempt may begin at.
const MIN_START: f32 = 600.0;
const MAX_START: f32 = 2600.0;
/// Frames of slack on top of `dist / speed` (the first frames of a chain
/// are slow: the run-up to a dive, the crouch slide before a long jump).
const SLACK: f32 = 6.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    DiveRollout,
    LongJumpChain,
}

impl Move {
    fn kind(self) -> &'static str {
        match self {
            Move::DiveRollout => "Tech::DiveRollout",
            Move::LongJumpChain => "Tech::LongJumpChain",
        }
    }
    /// Mean straight-line speed (u/frame) a success needs.
    fn speed(self) -> f32 {
        match self {
            Move::DiveRollout => 30.0,
            Move::LongJumpChain => 33.0,
        }
    }
    /// How many times the technique must start in the segment.
    fn needed(self) -> usize {
        match self {
            Move::DiveRollout => 1,
            Move::LongJumpChain => 2,
        }
    }
    /// Whether the frame from action `prev` into `s` enters the technique's
    /// key action (a dive rollout counts at the rollout, a long jump at its
    /// take-off).
    fn started(self, prev: u32, s: &St) -> bool {
        let key = match self {
            Move::DiveRollout => ACT_FORWARD_ROLLOUT,
            Move::LongJumpChain => ACT_LONG_JUMP,
        };
        s.action == key && prev != key
    }
    fn salt(self) -> u64 {
        match self {
            Move::DiveRollout => 0xD1,
            Move::LongJumpChain => 0x17,
        }
    }
}

/// A deterministic pseudo-random number in [0, 1) from `b` (generate_level
/// gets no rng; this keeps runs reproducible and consumes nothing).
pub(super) fn unit(b: &St, salt: u64, i: u64) -> f32 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (b.frame, b.pos.map(f32::to_bits), b.level, b.area, salt, i).hash(&mut h);
    (h.finish() >> 40) as f32 / (1u64 << 24) as f32
}

/// Whether Mario is in a mode these techniques start from: in control, on
/// the ground or in the air (not swimming, on a pole or hanging).
pub(super) fn can_start(s: &St) -> bool {
    s.controllable() && s.action & (ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE | ACT_FLAG_HANGING) == 0
}

/// Frames one more start of the technique costs on top of the approach (the
/// run-up to a dive, the crouch slide before a long jump, a kick's wind-up):
/// a rough figure from the TAS catalogue (chains start within ~5–10 frames).
pub(super) const START_FRAMES: f64 = 8.0;

/// [`Mem`] layout of a timed run ([`timed_eta_with`]): `f[0]` frames since the
/// start (the index in the rollout), `f[1]` the previous frame's action,
/// `f[2]` technique starts so far.

/// A timed run's goal: get to `p` of `zone` within `limit` frames of the
/// start with `needed` starts of the technique (`started(prev_action, s)`
/// says whether the frame into `s` is one) by then. Arrival is within
/// [`RADIUS`]. As v1: the first arrival decides: with enough starts `Done`,
/// else `Lost`; past `limit`, or out of the zone, `Lost`. Before that
/// `frames_for(dist) + START_FRAMES × starts still needed`.
#[allow(clippy::too_many_arguments)]
pub(super) fn timed_eta_with(
    s: &St,
    mem: &mut Mem,
    zone: (u16, u8),
    p: [f32; 3],
    limit: usize,
    needed: usize,
    started: impl Fn(u32, &St) -> bool,
) -> Eta {
    let i = mem.f[0] as usize;
    if i > 0 && started(mem.f[1] as u32, s) {
        mem.f[2] += 1.0;
    }
    mem.f[0] += 1.0;
    mem.f[1] = s.action as f64;
    if i > limit {
        return Eta::Lost;
    }
    let n = mem.f[2] as usize;
    let arrived = s.zone() == zone && s.dist(p) <= RADIUS;
    if arrived {
        return if n >= needed { Eta::Done } else { Eta::Lost };
    }
    if s.zone() != zone {
        return Eta::Lost;
    }
    let left = needed.saturating_sub(n);
    Eta::Frames(frames_for(s.dist(p)) + START_FRAMES * left as f64)
}

/// Which dive-rollout objective a [`Dash`] is (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Variant {
    Plain,
    Hop,
    High,
    Boost,
}

pub struct Dash {
    mv: Move,
    p: [f32; 3],
    zone: (u16, u8),
    var: Variant,
    /// The slope-boost objective, for `Variant::Boost`.
    boost: Option<Boost>,
}

/// `pad` with `button` pressed this frame: set if `prev` didn't hold it,
/// else released (so the next frame can press it).
pub(super) fn press(pad: GamePad, prev: Option<&GamePad>, button: u16) -> GamePad {
    let held = prev.is_some_and(|p| p.button & button != 0);
    let b = if held { pad.button & !button } else { pad.button | button };
    GamePad::new(b, pad.stick_x, pad.stick_y)
}

/// A dive from FREEFALL (the edge dive-hop) on the frame from `prev` into
/// `s`: FREEFALL then DIVE, or DIVE_SLIDE when the dive lands in the frame
/// it starts (the usual case at a seam or an edge: TAS WF f19807–19813, RR
/// f113148–113153).
fn hop(prev: u32, s: &St) -> bool {
    prev == ACT_FREEFALL && matches!(s.action, ACT_DIVE | ACT_DIVE_SLIDE)
}

/// An entry into ACT_DIVE from a rising jump (the high dive).
fn high_dive(prev: u32, s: &St) -> bool {
    s.action == ACT_DIVE && matches!(prev, ACT_JUMP | ACT_DOUBLE_JUMP) && s.vel[1] > 20.0
}

impl Dash {
    /// From `b`: a target 900–2000 units away, level within ±150, half the
    /// time within 45° of where Mario faces (chains go straight), else in
    /// any direction.
    pub fn new(b: &St, mv: Move) -> Dash {
        let s = mv.salt();
        let r = 900.0 + 1100.0 * unit(b, s, 0);
        let face = b.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let a = if unit(b, s, 1) < 0.5 {
            face + (unit(b, s, 2) - 0.5) * std::f32::consts::FRAC_PI_2
        } else {
            unit(b, s, 2) * std::f32::consts::TAU
        };
        let dy = (unit(b, s, 3) - 0.5) * 300.0;
        let var = if mv == Move::DiveRollout { Self::variant(b) } else { Variant::Plain };
        let (r, dy) = match var {
            Variant::Hop => (400.0 + 600.0 * unit(b, s, 0), 100.0 - 400.0 * unit(b, s, 3)),
            Variant::High => (400.0 + 500.0 * unit(b, s, 0), 120.0 + 230.0 * unit(b, s, 3)),
            _ => (r, dy),
        };
        let p = [b.pos[0] + r * a.sin(), b.pos[1] + dy, b.pos[2] + r * a.cos()];
        let boost = (var == Variant::Boost).then(|| Boost::new(b, s));
        Dash { mv, p, zone: b.zone(), var, boost }
    }

    /// The variant from `b`: half the time the plain one, else one of those
    /// `b` qualifies for (none qualify with every fix off).
    fn variant(b: &St) -> Variant {
        let air = b.action & ACT_FLAG_AIR != 0;
        let mut can = Vec::new();
        if !air && b.forward_vel < 30.0 {
            can.push(Variant::Hop);
        }
        if !air && b.forward_vel >= 29.0 {
            can.push(Variant::High);
        }
        if Boost::eligible(b) {
            can.push(Variant::Boost);
        }
        let u = unit(b, Move::DiveRollout.salt(), 5);
        if can.is_empty() || u < 0.5 {
            return Variant::Plain;
        }
        can[(((u - 0.5) * 2.0 * can.len() as f32) as usize).min(can.len() - 1)]
    }

    /// Mean straight-line speed (u/frame) a success needs.
    fn speed(&self) -> f32 {
        match self.var {
            Variant::Hop => 24.0,
            _ => self.mv.speed(),
        }
    }

    /// Start distances to the target an attempt may begin at.
    fn starts(&self) -> (f32, f32) {
        match self.var {
            Variant::Hop => (300.0, 1400.0),
            Variant::High => (350.0, 1000.0),
            _ => (MIN_START, MAX_START),
        }
    }

    /// Frames a start at distance `d` has to arrive in.
    fn limit(&self, d: f32) -> usize {
        (d / self.speed() + SLACK).ceil() as usize
    }

    /// Whether the frame from `prev` into `s` starts the technique.
    fn started(&self, prev: u32, s: &St) -> bool {
        match self.var {
            Variant::Hop => hop(prev, s),
            Variant::High => high_dive(prev, s),
            _ => self.mv.started(prev, s),
        }
    }

    /// How many starts a success needs.
    fn needed(&self) -> usize {
        match self.var {
            Variant::Hop | Variant::High => 1,
            _ => self.mv.needed(),
        }
    }

    pub fn applies(b: &St) -> bool {
        can_start(b)
    }
}

impl Objective for Dash {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        self.mv.kind()
    }
    fn key(&self) -> String {
        if let Some(bo) = &self.boost {
            return bo.key(self.mv.kind(), self.zone);
        }
        let c = |v: f32| (v / 400.0).round() as i32;
        let tag = match self.var {
            Variant::Plain | Variant::Boost => "",
            Variant::Hop => " hop",
            Variant::High => " high",
        };
        format!("{}{} {:?} {},{},{}", self.mv.kind(), tag, self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]))
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        move_weight(0.5)
    }
    fn base_criteria(&self, s: &St) -> bool {
        if let Some(bo) = &self.boost {
            return s.zone() == self.zone && bo.base(s);
        }
        let d = s.dist(self.p);
        let (lo, hi) = self.starts();
        let air = s.action & ACT_FLAG_AIR != 0;
        let mode = match self.var {
            Variant::Hop => !air && s.forward_vel < 30.0,
            Variant::High => !air && s.forward_vel >= 29.0,
            _ => true,
        };
        s.zone() == self.zone && can_start(s) && (lo..=hi).contains(&d) && mode
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        if let Some(bo) = &self.boost {
            return bo.ga_budget();
        }
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: self.limit(self.starts().1) + 10 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if let Some(bo) = &self.boost {
            return bo.eta(s, start, mem, self.zone);
        }
        let limit = self.limit(start.dist(self.p));
        timed_eta_with(s, mem, self.zone, self.p, limit, self.needed(), |prev, s| self.started(prev, s))
    }
    fn start_bias(&self, s: &St) -> f64 {
        // v1: a running start ranks ahead (not for boosts, ranked by their own).
        if self.boost.is_some() { 0.0 } else { -(s.forward_vel.max(0.0) as f64) * RUNNING_START_FRAMES }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.boost.as_ref().map_or(self.p, |bo| bo.p))
    }
    fn steer_button(&self, s: &St) -> u16 {
        use buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
        let air = s.action & ACT_FLAG_AIR != 0;
        let moving = s.action & ACT_FLAG_MOVING != 0;
        match (self.mv, self.var) {
            (Move::DiveRollout, Variant::Boost) => Boost::button(s, false),
            (Move::DiveRollout, Variant::Hop) => match s.action {
                // B on every FREEFALL frame (the overlay's rule; here for
                // the controller-seeded genomes), A in DIVE_SLIDE once fast.
                ACT_FREEFALL => B_BUTTON,
                ACT_DIVE_SLIDE if s.forward_vel >= 45.0 => A_BUTTON,
                ACT_DIVE_SLIDE | ACT_DIVE | ACT_FORWARD_ROLLOUT => 0,
                _ if moving && s.forward_vel >= 29.0 => B_BUTTON,
                _ => 0,
            },
            (Move::DiveRollout, Variant::High) => match s.action {
                ACT_JUMP | ACT_DOUBLE_JUMP if s.forward_vel > 28.0 => B_BUTTON,
                ACT_DIVE_SLIDE => A_BUTTON,
                ACT_DIVE | ACT_FORWARD_ROLLOUT => 0,
                _ if !air && s.forward_vel >= 29.0 => A_BUTTON,
                _ => 0,
            },
            (Move::DiveRollout, Variant::Plain) => match s.action {
                ACT_DIVE_SLIDE | ACT_FORWARD_ROLLOUT => A_BUTTON,
                ACT_DIVE => 0,
                // act_freefall dives on B at any speed (G05).
                ACT_FREEFALL => B_BUTTON,
                _ if air && s.forward_vel > 28.0 => B_BUTTON,
                // The rollout lands in ACT_FREEFALL_LAND_STOP, which zeroes
                // forward speed unless the stick (held towards p) walks
                // on at once (check_common_action_exits keeps the speed):
                // then a ground dive.
                _ if moving && s.forward_vel >= 29.0 => B_BUTTON,
                _ => 0,
            },
            (Move::LongJumpChain, _) => match s.action {
                ACT_CROUCH_SLIDE | ACT_LONG_JUMP | ACT_LONG_JUMP_LAND => Z_TRIG | A_BUTTON,
                _ if moving && !air && s.forward_vel >= 16.0 => Z_TRIG,
                _ => 0,
            },
        }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        use buttons::{A_BUTTON, B_BUTTON};
        match self.var {
            Variant::Plain => None,
            Variant::Boost => Some(Boost::overlay()),
            // G05: B on every FREEFALL frame (pressed, so released first if
            // held), B released in the dive, A once DIVE_SLIDE is at 45
            // (below that the genome's: on flat ground a ground dive's 44
            // never gets there).
            Variant::Hop => Some(PadOverlay::keyed(|traj, played, genome| {
                let s = traj.last().unwrap();
                let prev = played.last();
                let clear = |b: u16| GamePad::new(genome.button & !b, genome.stick_x, genome.stick_y);
                match s.action {
                    ACT_FREEFALL => (press(genome, prev, B_BUTTON), true),
                    ACT_DIVE => (clear(B_BUTTON), false),
                    ACT_DIVE_SLIDE if s.forward_vel >= 45.0 => (press(genome, prev, A_BUTTON), false),
                    _ => (genome, false),
                }
            })),
            // G17: A on a landing frame at 29+, B on the jump's first frame.
            Variant::High => Some(PadOverlay::keyed(|traj, played, genome| {
                let s = traj.last().unwrap();
                let prev = played.last();
                let first = traj.len() < 2 || traj[traj.len() - 2].action != s.action;
                match s.action {
                    ACT_JUMP | ACT_DOUBLE_JUMP if first && s.forward_vel > 28.0 => (press(genome, prev, B_BUTTON), true),
                    ACT_FREEFALL_LAND_STOP | ACT_JUMP_LAND_STOP | ACT_DOUBLE_JUMP_LAND_STOP | ACT_JUMP_LAND
                    | ACT_FREEFALL_LAND | ACT_DOUBLE_JUMP_LAND
                        if s.forward_vel >= 29.0 =>
                    {
                        (press(genome, prev, A_BUTTON), true)
                    }
                    _ => (genome, false),
                }
            })),
        }
    }
}
