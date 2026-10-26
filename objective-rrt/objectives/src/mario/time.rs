//! `Mario::PosTime(p)`: get within a radius of `p` sooner than the search
//! already has. Generated with `p` = b's position and the frame b got there,
//! so it looks for faster routes to places already reached.
//!
//! `OBRRT_PACE=1` (off by default): the deadline also comes from a target
//! pace. From a start `s` at distance `d0` the arrival must be within
//! `d0 / pace` frames of `s` as well as before b's frame. The pace is a rung
//! of a ladder picked per b (by a hash of b, so it's deterministic and
//! stateless): `OBRRT_PACE_SPEED` (35 u/frame, TAS-like; the TAS averages
//! 44 in levels, the tree 16-24), `OBRRT_PACE_SPEED2` (25), or none (the old
//! rule, the fallback). The rung is in the key. A rung that keeps failing
//! decays like any objective, so the pool's weight drifts to the rungs that
//! succeed. With the switch on, PosTime and PosDist also get a dive-rollout
//! steering hint ([`pace_button`], `OBRRT_PACE_HINT=0` turns it off).

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_MOVING, ACT_FLAG_SWIMMING};
use obrrt_core::{Budget, Objective, Progress, St};

const RADIUS: f32 = 150.0;
/// Units per frame no route beats (Mario's top speeds are around 50), so
/// `frame + dist / MAX_SPEED` is an optimistic arrival time.
const MAX_SPEED: f32 = 60.0;
/// A typical speed, for the arrival estimate starts are ranked by.
const SPEED: f32 = 30.0;
/// How much sooner than `b` counts: any frame, the same "strictly sooner"
/// the duplicate check uses (PosDist has no margin either). It was 30, a
/// whole second, which made every smaller real gain count for nothing; each
/// success makes a new, harder PosTime (the deadline is in the key), so
/// small steps chain.
const MARGIN: u32 = 1;

/// The margin, overridable with OBRRT_POSTIME_MARGIN (frames) for experiments.
fn margin() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| std::env::var("OBRRT_POSTIME_MARGIN").ok().and_then(|v| v.parse().ok()).unwrap_or(MARGIN))
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `OBRRT_PACE=1`: travel objectives push for speed (see the module doc).
pub(super) fn pace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_PACE").is_ok_and(|v| v == "1"))
}

/// The pace ladder (u/frame): `OBRRT_PACE_SPEED`, `OBRRT_PACE_SPEED2`.
fn ladder() -> [f32; 2] {
    static L: std::sync::OnceLock<[f32; 2]> = std::sync::OnceLock::new();
    *L.get_or_init(|| [env_f32("OBRRT_PACE_SPEED", 35.0), env_f32("OBRRT_PACE_SPEED2", 25.0)])
}

/// Steering hint for travel under the pace switch: dive at running speed
/// (B on the ground needs `forwardVel >= 29` and the stick past 48, in the
/// air `> 28`), roll out of the dive slide with A. It only replaces the
/// controller's default "A while the target is higher" when it presses
/// something, so jumps up still happen; the GA's genomes are untouched
/// (its own B/Z/A mutations find long jumps and jump kicks).
pub(super) fn pace_button(s: &St, target: [f32; 3]) -> u16 {
    static HINT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !pace_on() || !*HINT.get_or_init(|| std::env::var("OBRRT_PACE_HINT").map_or(true, |v| v != "0")) {
        return 0;
    }
    const ACT_DIVE: u32 = 0x0188088A;
    const ACT_DIVE_SLIDE: u32 = 0x00880456;
    const ACT_FORWARD_ROLLOUT: u32 = 0x010008A6;
    if s.action & ACT_FLAG_SWIMMING != 0 || target[1] > s.pos[1] + 50.0 || s.hdist(target) < 300.0 {
        return 0;
    }
    let air = s.action & ACT_FLAG_AIR != 0;
    let moving = s.action & ACT_FLAG_MOVING != 0;
    match s.action {
        ACT_DIVE_SLIDE | ACT_FORWARD_ROLLOUT => A_BUTTON,
        ACT_DIVE => 0,
        _ if air && s.forward_vel > 28.0 => B_BUTTON,
        _ if moving && !air && s.forward_vel >= 29.0 => B_BUTTON,
        _ => 0,
    }
}

pub struct PosTime {
    p: [f32; 3],
    zone: (u16, u8),
    /// Arrive before this frame.
    deadline: u32,
    /// Pace rung (u/frame) under `OBRRT_PACE=1`: also arrive within
    /// `d0 / pace` frames of the start. 0 = none.
    pace: f32,
}

impl PosTime {
    pub fn new(b: &St) -> PosTime {
        let pace = if pace_on() {
            // A stateless, deterministic rung per b: 35, 25 or none.
            let h = (b.frame as u64)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(b.pos[0].to_bits() as u64 ^ ((b.pos[2].to_bits() as u64) << 32))
                .wrapping_mul(0xBF58_476D_1CE4_E5B9);
            match (h >> 40) % 3 {
                0 => ladder()[0],
                1 => ladder()[1],
                _ => 0.0,
            }
        } else {
            0.0
        };
        PosTime { p: b.pos, zone: b.zone(), deadline: b.frame.saturating_sub(margin()), pace }
    }

    /// Arrive before this frame, for an attempt from `start`: the earlier of
    /// b's deadline and the pace one.
    fn deadline_from(&self, start: Option<&St>) -> u32 {
        match start {
            Some(s0) if self.pace > 0.0 => {
                let by = s0.frame + (s0.dist(self.p) / self.pace).ceil() as u32;
                self.deadline.min(by)
            }
            _ => self.deadline,
        }
    }
}

impl Objective for PosTime {
    fn kind(&self) -> &'static str {
        "Mario::PosTime"
    }
    fn key(&self) -> String {
        let c = |v: f32| (v / 400.0).round() as i32;
        // The deadline (in frames) is part of it: reaching a place sooner
        // makes a new, harder PosTime for it instead of the stale one.
        let k = format!("PosTime {:?} {},{},{} by {}", self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]), self.deadline);
        if self.pace > 0.0 { format!("{k} pace {}", self.pace) } else { k }
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        let d = s.dist(self.p);
        s.zone() == self.zone && d > RADIUS && (s.frame as f32 + d / MAX_SPEED) < self.deadline as f32
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Estimated arrival, in seconds.
        (s.frame as f64 + (s.dist(self.p) / SPEED) as f64) / 30.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let deadline = self.deadline_from(traj.first());
        traj.iter().position(|s| s.zone() == self.zone && s.dist(self.p) <= RADIUS && s.frame < deadline)
    }
    fn success_heuristic(&self, traj: &[St]) -> f64 {
        traj.last().map_or(f64::MAX, |s| s.frame as f64)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let deadline = self.deadline_from(traj.first());
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == self.zone && s.frame < deadline)
            .map(|(i, s)| (i, s.dist(self.p)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 150 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.p)
    }
    fn steer_button(&self, s: &St) -> u16 {
        pace_button(s, self.p)
    }
}
