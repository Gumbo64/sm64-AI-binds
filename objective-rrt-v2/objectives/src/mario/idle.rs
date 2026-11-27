//! `Mario::Settle` and `Mario::Wait(n)`: standing still, for clean start
//! states and for timing.

use obrrt_core::layout::ACT_FLAG_STATIONARY;
use obrrt_core::{Budget, Eta, GamePad, Mem, Objective, St};

/// Forward speed lost per frame with the stick let go
/// (update_decelerating_speed, mario_actions_moving.c: approach 0 by 1).
const DECEL: f64 = 1.0;
/// Vertical speed lost per frame in the air (gravity, apply_gravity in
/// mario_step.c: 4 a frame).
const GRAVITY: f64 = 4.0;
/// Frames from moving to a stationary action once the speed is gone (the
/// stopping animations: walking -> idle, landing stops). Rough.
const STOP_FRAMES: f64 = 5.0;

fn still(s: &St) -> bool {
    s.action_has(ACT_FLAG_STATIONARY) && s.forward_vel.abs() < 0.1 && s.vel[1].abs() < 0.1
}

/// Come to a stop on the ground.
pub struct Settle {
    zone: (u16, u8),
    cell: (i32, i32, i32),
}

impl Settle {
    pub fn new(b: &St) -> Settle {
        let c = |v: f32| (v / 800.0).round() as i32;
        Settle { zone: b.zone(), cell: (c(b.pos[0]), c(b.pos[1]), c(b.pos[2])) }
    }
}

impl Objective for Settle {
    fn kind(&self) -> &'static str {
        "Mario::Settle"
    }
    fn key(&self) -> String {
        format!("Settle {:?} {:?}", self.zone, self.cell)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.3
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !still(s)
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // Still on a frame after the start (v1: traj index >= 1).
        if s.frame > start.frame && still(s) {
            return Eta::Done;
        }
        let moving = if s.action_has(ACT_FLAG_STATIONARY) { 0.0 } else { STOP_FRAMES };
        Eta::Frames(s.forward_vel.abs() as f64 / DECEL + s.vel[1].abs() as f64 / GRAVITY + moving)
    }
    fn fixed(&self) -> Option<Budget> {
        Some(Budget::Macro(vec![GamePad::new(0, 0, 0); 60]))
    }
    fn kinship(&self) -> u32 {
        2
    }
}

/// Stand still for `n` frames: lets platforms, enemies and cycles line up.
pub struct Wait {
    zone: (u16, u8),
    n: usize,
}

impl Wait {
    pub fn new(b: &St, n: usize) -> Wait {
        Wait { zone: b.zone(), n }
    }
}

impl Objective for Wait {
    fn kind(&self) -> &'static str {
        "Mario::Wait"
    }
    fn key(&self) -> String {
        format!("Wait {:?} {}", self.zone, self.n)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.15
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && still(s)
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // v1: met at trajectory index n.
        let left = self.n as f64 - s.frame.saturating_sub(start.frame) as f64;
        if left <= 0.0 { Eta::Done } else { Eta::Frames(left) }
    }
    fn fixed(&self) -> Option<Budget> {
        Some(Budget::Macro(vec![GamePad::new(0, 0, 0); self.n]))
    }
    fn kinship(&self) -> u32 {
        2
    }
}
