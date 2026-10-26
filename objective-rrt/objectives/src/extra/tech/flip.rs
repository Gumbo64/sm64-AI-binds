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
use obrrt_core::{Budget, Objective, Progress, St, buttons};

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

fn flipped(traj: &[St]) -> bool {
    traj.iter().any(|s| matches!(s.action, ACT_SIDE_FLIP | ACT_BACKFLIP))
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
    fn base_heuristic(&self, s: &St) -> f64 {
        s.hdist(self.p) as f64 / 300.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let y0 = traj.first()?.pos[1];
        let i = traj
            .iter()
            .take(LIMIT + 1)
            .position(|s| s.zone() == self.zone && landed(s) && s.pos[1] >= y0 + RISE)?;
        flipped(&traj[..=i]).then_some(i)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Signature-gated: no flip in the time allowed scores 0 (what the
        // start alone scores), so a climb by walking or jumping earns
        // nothing; height counts only from the flip on.
        let Some(first) = traj.first() else { return Progress { score: f64::MIN, at: 0 } };
        let y0 = first.pos[1];
        let Some(f) = traj.iter().take(LIMIT + 1).position(|s| matches!(s.action, ACT_SIDE_FLIP | ACT_BACKFLIP)) else {
            return Progress { score: 0.0, at: 0 };
        };
        let (at, gain) = traj
            .iter()
            .take(LIMIT + 1)
            .enumerate()
            .skip(f)
            .filter(|(_, s)| s.zone() == self.zone)
            .map(|(i, s)| {
                let g = (s.pos[1] - y0).max(0.0);
                (i, if landed(s) { g } else { 0.4 * g })
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((f, 0.0));
        Progress { score: 100.0 + gain as f64, at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 12, generations: 4, horizon: LIMIT + 10 }
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
