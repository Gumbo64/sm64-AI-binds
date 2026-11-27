//! `Tech::FlipHeight` (`flipheight`): height with no run-up, from a side
//! flip or a backflip.
//!
//! - Side flip: A in ACT_TURNING_AROUND / ACT_FINISH_TURNING_AROUND (stick
//!   reversed while running at 16+; mario_actions_moving.c), 62 up, 8 forward
//!   towards the stick (mario.c). It can ledge-grab.
//! - Backflip: A in ACT_START_CROUCHING / ACT_CROUCHING / ACT_STOP_CROUCHING
//!   (Z while standing; mario_actions_stationary.c), 62 up, 16 backwards.
//!
//! Both rise about 480 units; a single jump about 220–260, a double about
//! 340, and both need a landing first. Success is about time and a new
//! place: within [`LIMIT`] frames of the start, with a flip in the segment,
//! Mario stands (or hangs from a ledge) at least [`RISE`] units above where
//! he started. The target `p` is only a steering point: up and a little to
//! one side (behind Mario when he's running, so steering towards it turns
//! him around into a side flip).

use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_STATIONARY, ACT_FLAG_SWIMMING};
use obrrt_core::{Eta, Mem, Objective, St, buttons, frames_for};

use super::travel::{can_start, unit};

const ACT_SIDE_FLIP: u32 = 0x01000887;
const ACT_BACKFLIP: u32 = 0x01000883;
const ACT_LEDGE_GRAB: u32 = 0x0800034B;
const ACT_TURNING_AROUND: u32 = 0x00000443;
const ACT_FINISH_TURNING_AROUND: u32 = 0x00000444;
const ACT_START_CROUCHING: u32 = 0x0C008221;
const ACT_CROUCHING: u32 = 0x0C008220;
const ACT_STOP_CROUCHING: u32 = 0x0C008222;

/// Frames from the start a success must land in.
const LIMIT: usize = 60;
/// Height above the start a success must stand at.
const RISE: f32 = 250.0;
const SALT: u64 = 0xF1;

pub struct FlipHeight {
    /// The start's cell (200 units) and the steering point.
    from: [i32; 3],
    p: [f32; 3],
    zone: (u16, u8),
}

/// Frames into a flip (the turn-around or crouch before the A): a side flip
/// needs ACT_TURNING_AROUND at 16+, a backflip one crouch frame; ~4.
const FLIP_SETUP_FRAMES: f64 = 4.0;
/// Frames from a flip's take-off to standing 250+ higher: 62 up at gravity
/// 4 rises ~15 frames, then ~9 falling to a ledge 250-480 up.
const FLIP_AIR_FRAMES: f64 = 24.0;
/// `Mem::flags` bit: a side flip or backflip happened (frame in `f[1]`).
const FLIPPED: u32 = 1;

fn flipping(s: &St) -> bool {
    matches!(s.action, ACT_SIDE_FLIP | ACT_BACKFLIP)
}

/// Standing (not in the air or water) or hanging from a ledge.
fn landed(s: &St) -> bool {
    s.action == ACT_LEDGE_GRAB || s.action & (ACT_FLAG_AIR | ACT_FLAG_SWIMMING) == 0
}

impl FlipHeight {
    pub fn new(b: &St) -> FlipHeight {
        let face = b.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let a = if b.forward_vel >= 16.0 {
            face + std::f32::consts::PI + (unit(b, SALT, 0) - 0.5) * 1.0
        } else {
            unit(b, SALT, 0) * std::f32::consts::TAU
        };
        let r = 100.0 + 250.0 * unit(b, SALT, 1);
        let up = 300.0 + 180.0 * unit(b, SALT, 2);
        let p = [b.pos[0] + r * a.sin(), b.pos[1] + up, b.pos[2] + r * a.cos()];
        let c = |v: f32| (v / 200.0).round() as i32;
        FlipHeight { from: [c(b.pos[0]), c(b.pos[1]), c(b.pos[2])], p, zone: b.zone() }
    }

    pub fn applies(b: &St) -> bool {
        can_start(b) && b.action & ACT_FLAG_AIR == 0
    }
}

impl Objective for FlipHeight {
    fn kind(&self) -> &'static str {
        "Tech::FlipHeight"
    }
    fn key(&self) -> String {
        // One per start cell and side (octant of the steering point).
        let o = (self.p[0] - self.from[0] as f32 * 200.0).atan2(self.p[2] - self.from[2] as f32 * 200.0);
        let oct = ((o / std::f32::consts::FRAC_PI_4).round() as i32).rem_euclid(8);
        format!("Tech::FlipHeight {:?} {},{},{} {}", self.zone, self.from[0], self.from[1], self.from[2], oct)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        super::kicks::move_weight(0.5)
    }
    fn base_criteria(&self, s: &St) -> bool {
        // On the ground, near the start cell and below the steering point.
        s.zone() == self.zone
            && can_start(s)
            && s.action & ACT_FLAG_AIR == 0
            && s.hdist(self.p) < 600.0
            && s.pos[1] < self.p[1] - RISE
    }
    /// As v1: the first frame (within [`LIMIT`]) standing [`RISE`]+ above
    /// the start decides: `Done` with a flip by then, else `Lost`. Before
    /// the flip: setup + flight + the walk to the steering point; after it,
    /// what is left of the flight.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 12, generations: 4, horizon: LIMIT + 10 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        let i = mem.f[0];
        mem.f[0] += 1.0;
        if flipping(s) && mem.flags & FLIPPED == 0 {
            mem.flags |= FLIPPED;
            mem.f[1] = i;
        }
        if i as usize > LIMIT || s.zone() != self.zone {
            return Eta::Lost;
        }
        let flipped = mem.flags & FLIPPED != 0;
        if landed(s) && s.pos[1] >= start.pos[1] + RISE {
            return if flipped { Eta::Done } else { Eta::Lost };
        }
        let walk = frames_for(s.hdist(self.p));
        if flipped {
            Eta::Frames(walk + (FLIP_AIR_FRAMES - (i - mem.f[1])).max(1.0))
        } else {
            Eta::Frames(walk + FLIP_SETUP_FRAMES + FLIP_AIR_FRAMES)
        }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.p)
    }
    fn steer_button(&self, s: &St) -> u16 {
        use buttons::{A_BUTTON, Z_TRIG};
        match s.action {
            ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND => A_BUTTON,
            ACT_START_CROUCHING | ACT_CROUCHING | ACT_STOP_CROUCHING => A_BUTTON,
            // Standing: crouch, for a backflip.
            a if a & ACT_FLAG_STATIONARY != 0 => Z_TRIG,
            // Running, in the air: the usual controller.
            _ => 0,
        }
    }
}
