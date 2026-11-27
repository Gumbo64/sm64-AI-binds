//! Level openers (techniques `wfopener` and `bobopener`): the first seconds
//! of a level as the TAS plays them, as a race against the clock from the
//! first controllable frame.
//!
//! - **WF** (`Opener::Wf`, course 2). Every synced WF TAS star opens the same
//!   way (ref/tas/dumps/full-120star-2026-U.csv, first WF entry): from the
//!   painting landing at (2680, 256, 5078) a jump kick, a long jump up the
//!   first ramp (3060, 386, 4590 at +24 frames), a wall kick at
//!   (3276, 740, 3890) at +44, then dives and rollouts at ~50 u/frame up to
//!   (3355, 1535, 2185) at +84: the high ground near (3300, 1600, 2000) in
//!   under 3 s. v35's trees take 2,000–4,300 frames from entering WF to
//!   stand there.
//! - **BoB** (`Opener::Bob`, course 1). The 120-star TAS goes from the
//!   landing at (-6518, 153, 6480) to the Chain Chomp plateau
//!   (1614, 768, 4158) in 154 frames (5.1 s) with slide-kick / dive-rollout
//!   chains at 86–96 u/frame over the slopes; v35's trees take 350–440
//!   frames from the level's first state (fade-in included, ~100 of them).
//!
//! When the level was entered isn't in `St`, so the objective is made from
//! a state `b` near the level's spawn point in control (the painting
//! landing): `b.frame` stands for the first controllable frame and the
//! deadline is 6 s (or, as a step, 8 s) later ([`DEADLINES`]). It's in the key, like PosTime's,
//! so each entry gets its own race. Starts are any state of that level
//! that can still make the deadline at a speed nothing beats
//! (`frame + dist / max_speed < deadline`) and isn't from before the entry.
//! Ranked by straight-line distance to the target.
//!
//! Success: at the target (alive) by the deadline (v2: past it, or where
//! even `max_speed` can't make it, `Lost`). That's a place the
//! slower arrivals already have, but the soft dedupe keeps the sooner state
//! at full weight and halves the later ones' (crowd), so a success makes
//! the fast arrival the one starts come from.
//!
//! Steering hints only (the GA times everything): the stick towards the
//! wall kick and then the target (WF) or the target (BoB); A in the wall
//! kick window; Z / Z+A for long jumps before the WF wall; B to dive at
//! speed, A to roll out of dives and slide kicks, B to slide-kick out of a
//! crouch slide.

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_MOVING, LEVEL_BOB, LEVEL_WF};
use obrrt_core::{Eta, Mem, Objective, Precision, St, frames_for};

use super::travel::{ACT_CROUCH_SLIDE, ACT_DIVE, ACT_DIVE_SLIDE, ACT_FORWARD_ROLLOUT, ACT_LONG_JUMP, ACT_LONG_JUMP_LAND, can_start};

/// Frames after the entry the target must be reached by: 6 s, and an
/// 8 s step on the way (in fresh 50 M-frame tests the GA's best 6 s
/// attempts from the BoB landing still ended ~4,600 from the plateau).
const DEADLINES: [u32; 2] = [180, 240];
/// An entry state: this close (horizontally) to the spawn point, about at
/// its height, and not yet running.
const ENTRY_RADIUS: f32 = 250.0;
const ENTRY_HEIGHT: f32 = 200.0;
const ENTRY_SPEED: f32 = 20.0;

/// Wall-kick window actions (include/sm64.h; as in wallkick.rs).
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;
const ACT_SOFT_BONK: u32 = 0x0102_08B6;
const ACT_BACKWARD_AIR_KB: u32 = 0x0102_08B0;
const ACT_FORWARD_AIR_KB: u32 = 0x0102_08B1;
const ACT_WALL_KICK_AIR: u32 = 0x0300_0886;
const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080_045A;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Which {
    Wf,
    Bob,
}

impl Which {
    fn name(self) -> &'static str {
        match self {
            Which::Wf => "Opener::Wf",
            Which::Bob => "Opener::Bob",
        }
    }
    fn zone(self) -> (u16, u8) {
        match self {
            Which::Wf => (LEVEL_WF as u16, 1),
            Which::Bob => (LEVEL_BOB as u16, 1),
        }
    }
    /// MARIO_POS of the level's area 1 (levels/*/script.c).
    fn spawn(self) -> [f32; 3] {
        match self {
            Which::Wf => [2600.0, 256.0, 5120.0],
            Which::Bob => [-6558.0, 0.0, 6464.0],
        }
    }
    fn target(self) -> [f32; 3] {
        match self {
            Which::Wf => [3300.0, 1600.0, 2000.0],
            Which::Bob => [1600.0, 768.0, 4150.0],
        }
    }
    /// At the target: WF on the high ground (y >= 1400) within 400 of it
    /// horizontally; BoB on the plateau's level within 400 of it.
    fn at(self, s: &St) -> bool {
        let t = self.target();
        s.zone() == self.zone()
            && s.alive()
            && s.hdist(t) <= 400.0
            && match self {
                Which::Wf => s.pos[1] >= 1400.0,
                Which::Bob => (s.pos[1] - t[1]).abs() <= 200.0,
            }
    }
    /// Straight-line speed (u/frame) no route beats, for which starts can
    /// still make the deadline (the TAS peaks at ~51 in WF, ~96 in BoB).
    fn max_speed(self) -> f32 {
        match self {
            Which::Wf => 60.0,
            Which::Bob => 100.0,
        }
    }
}

/// Mario's state lets him kick off the wall he just hit with A.
fn kick_window(s: &St) -> bool {
    matches!(s.action, ACT_AIR_HIT_WALL | ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB | ACT_FORWARD_AIR_KB)
}

pub struct Opener {
    which: Which,
    /// The entry's frame (the first controllable frame, as near as `b`
    /// gives it).
    entry: u32,
    /// Frames from the entry to the deadline.
    limit: u32,
}

/// One opener per entry state `b` of its level.
pub fn generate_level(b: &St, which: Which, out: &mut Vec<Arc<dyn Objective>>) {
    let sp = which.spawn();
    if b.zone() == which.zone()
        && b.controllable()
        && b.alive()
        && b.hdist(sp) < ENTRY_RADIUS
        && (b.pos[1] - sp[1]).abs() < ENTRY_HEIGHT
        && b.forward_vel.abs() < ENTRY_SPEED
    {
        for limit in DEADLINES {
            out.push(Arc::new(Opener { which, entry: b.frame, limit }));
        }
    }
}

impl Opener {
    fn deadline(&self) -> u32 {
        self.entry + self.limit
    }
}

impl Objective for Opener {
    fn kind(&self) -> &'static str {
        self.which.name()
    }
    fn key(&self) -> String {
        format!("{} from {} in {}", self.which.name(), self.entry, self.limit)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // The 6 s race is worth more than the step towards it.
        if b.zone() == self.which.zone() { 6.0 * 180.0 / self.limit as f64 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        let t = self.which.target();
        s.zone() == self.which.zone()
            && can_start(s)
            && s.frame >= self.entry
            && !self.which.at(s)
            && (s.frame as f32 + s.dist(t) / self.which.max_speed()) < self.deadline() as f32
    }
    /// `Done` at the target by the deadline; `Lost` past it, out of the
    /// level, or where even [`Which::max_speed`] can't make it any more;
    /// else the straight-line distance in frames.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 6, horizon: self.limit as usize + 10 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.which.zone() || s.frame > self.deadline() {
            return Eta::Lost;
        }
        if self.which.at(s) {
            return Eta::Done;
        }
        let d = s.dist(self.which.target());
        if s.frame as f32 + d / self.which.max_speed() >= self.deadline() as f32 {
            return Eta::Lost;
        }
        Eta::Frames(frames_for(d))
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.which.zone() && s.frame <= self.deadline()
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        match self.which {
            // The wall kick's wall until Mario is above the first ramp.
            Which::Wf if s.pos[1] < 700.0 && s.pos[2] > 3700.0 => Some([3276.0, 640.0, 3890.0]),
            _ => Some(self.which.target()),
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        if kick_window(s) {
            return A_BUTTON;
        }
        let air = s.action & ACT_FLAG_AIR != 0;
        let moving = s.action & ACT_FLAG_MOVING != 0;
        let before_wall = self.which == Which::Wf && s.pos[1] < 700.0 && s.pos[2] > 3700.0;
        match s.action {
            // The wall kick's own controller (no button) steers it.
            ACT_WALL_KICK_AIR | ACT_DIVE | ACT_SLIDE_KICK => 0,
            ACT_DIVE_SLIDE | ACT_FORWARD_ROLLOUT | ACT_SLIDE_KICK_SLIDE => A_BUTTON,
            // WF: long jumps up the first ramp; A (on alternate frames)
            // also kicks off the wall when one ends there.
            ACT_CROUCH_SLIDE | ACT_LONG_JUMP | ACT_LONG_JUMP_LAND if before_wall => Z_TRIG | A_BUTTON,
            _ if before_wall && moving && !air && s.forward_vel >= 16.0 => Z_TRIG,
            // Elsewhere: slide kicks out of crouch slides, dives at speed.
            ACT_CROUCH_SLIDE => B_BUTTON,
            _ if air && s.forward_vel > 28.0 => B_BUTTON,
            _ if moving && s.forward_vel >= 29.0 => B_BUTTON,
            _ => 0,
        }
    }
}
