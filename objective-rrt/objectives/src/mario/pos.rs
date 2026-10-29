//! `Mario::PosDist(p)`: get within a radius of a point near `b`, as fast as
//! possible. Plain RRT is this objective alone, with `p` uniform.
//!
//! `OBRRT_PACE=1` (time.rs): success also needs a mean pace, arrival within
//! `d0 / OBRRT_PACE_DIST` (20 u/frame) + [`PACE_GRACE`] frames of the start
//! (`d0` its distance to `p`), so a slow walk no longer counts as a fast run.

use obrrt_core::{Budget, Objective, Progress, St};
use rand::Rng;
use rand::rngs::SmallRng;

const RADIUS: f32 = 150.0;

/// `OBRRT_SWIM_TARGETS=1`: from a `b` in water, one more `PosDist` with its
/// point in the water column ([`underwater_point`]). The two plain ones sit
/// -300..+900 around `b`: from the surface (`waterLevel - 80`) at most 380
/// deep, and two thirds of them out of the water, so exploration in deep
/// water (JRB, DDD) only ever pulled Mario along the top or out.
pub fn swim_targets() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("OBRRT_SWIM_TARGETS").is_ok_and(|v| v == "1");
        if on {
            // The floor under a point bounds its depth (geo::floor_below).
            obrrt_core::geo::want_static_floors();
        }
        on
    })
}

/// A point 300-1200 away horizontally from Mario in water, between 150
/// below the surface and 80 above the static floor there, at most 1000
/// below Mario; `None` out of water, or where the
/// highest floor is within 230 of the surface or above it (shallows, land,
/// pillars).
pub fn underwater_point(b: &St, rng: &mut SmallRng) -> Option<[f32; 3]> {
    let wl = b.water_level as f32;
    // No water box: find_water_level gives FLOOR_LOWER_LIMIT (-11000).
    if wl < -10000.0 || b.pos[1] >= wl {
        return None;
    }
    let r = rng.random_range(300.0..1200.0f32);
    let a = rng.random_range(0.0..std::f32::consts::TAU);
    let (x, z) = (b.pos[0] + r * a.sin(), b.pos[2] + r * a.cos());
    let top = wl - 150.0;
    let deepest = b.pos[1].min(top) - 1000.0;
    // The column's highest floor: one above `top` is land, or a pillar or
    // rock whose inside the floor under it would wrongly count as water.
    let floor = obrrt_core::geo::static_floors(b.zone()).and_then(|t| obrrt_core::geo::floor_below(&t, x, z, f32::MAX));
    let bottom = floor.map_or(deepest, |f| (f + 80.0).max(deepest));
    if bottom >= top {
        return None;
    }
    Some([x, rng.random_range(bottom..top), z])
}

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
    /// From [`underwater_point`]: a longer horizon (turning underwater is
    /// slow; the lab's median to such a point is about 100 frames).
    swim: bool,
}

impl PosDist {
    pub fn new(b: &St, p: [f32; 3]) -> PosDist {
        PosDist { p, zone: b.zone(), swim: false }
    }

    /// An `OBRRT_SWIM_TARGETS` point ([`underwater_point`]).
    pub fn new_swim(b: &St, p: [f32; 3]) -> PosDist {
        PosDist { p, zone: b.zone(), swim: true }
    }
}

impl Objective for PosDist {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): exploration: a state every 30 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 30, air: false }
    }
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
        Budget::Ga { population: 12, generations: 4, horizon: if self.swim { 180 } else { 90 } }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.p)
    }
    fn steer_button(&self, s: &St) -> u16 {
        super::time::pace_button(s, self.p)
    }
}
