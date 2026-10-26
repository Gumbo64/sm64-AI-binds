//! `Mario::PosDist(p)`: get within a radius of a point near `b`, as fast as
//! possible. Plain RRT is this objective alone, with `p` uniform.
//!
//! `OBRRT_PACE=1` (time.rs): success also needs a mean pace, arrival within
//! `d0 / OBRRT_PACE_DIST` (20 u/frame) + [`PACE_GRACE`] frames of the start
//! (`d0` its distance to `p`), so a slow walk no longer counts as a fast run.

use obrrt_core::{Budget, Objective, Progress, St};

const RADIUS: f32 = 150.0;
/// Frames on top of the pace allowance, for getting up to speed.
const PACE_GRACE: f32 = 10.0;

/// The latest trajectory index that meets the pace, under `OBRRT_PACE=1`.
fn pace_limit(traj: &[St], p: [f32; 3]) -> usize {
    static SPEED: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    if !super::time::pace_on() {
        return usize::MAX;
    }
    let speed = *SPEED.get_or_init(|| std::env::var("OBRRT_PACE_DIST").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0));
    traj.first().map_or(usize::MAX, |s0| (s0.dist(p) / speed + PACE_GRACE) as usize)
}

pub struct PosDist {
    p: [f32; 3],
    zone: (u16, u8),
}

impl PosDist {
    pub fn new(b: &St, p: [f32; 3]) -> PosDist {
        PosDist { p, zone: b.zone() }
    }
}

impl Objective for PosDist {
    fn kind(&self) -> &'static str {
        "Mario::PosDist"
    }
    fn key(&self) -> String {
        // Points in the same 400-unit cell are the same objective.
        let c = |v: f32| (v / 400.0).round() as i32;
        format!("PosDist {:?} {},{},{}", self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]))
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.dist(self.p) > RADIUS
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.p) as f64 / 500.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(500.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let limit = pace_limit(traj, self.p);
        traj.iter().position(|s| s.zone() == self.zone && s.dist(self.p) <= RADIUS).filter(|&i| i <= limit)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Under the pace switch, how close it got by the pace limit.
        let limit = pace_limit(traj, self.p);
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(i, s)| *i <= limit && s.zone() == self.zone)
            .map(|(i, s)| (i, s.dist(self.p)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 12, generations: 4, horizon: 90 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.p)
    }
    fn steer_button(&self, s: &St) -> u16 {
        super::time::pace_button(s, self.p)
    }
}
