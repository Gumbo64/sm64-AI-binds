//! `Mario::PosDist(p)`: get within a radius of a point near `b`, as fast as
//! possible. Plain RRT is this objective alone, with `p` uniform.
//!

use obrrt_core::{Eta, Mem, Objective, St, frames_for};
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
        let on = obrrt_core::config::current().objectives.swim_targets;
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

pub struct PosDist {
    p: [f32; 3],
    zone: (u16, u8),
    /// From [`underwater_point`]: a longer horizon (turning underwater is
    /// slow; the lab's median to such a point is about 100 frames).
    swim: bool,
    /// A roadmap leg (`OBRRT_ROAD`, search/src/road.rs): its frame budget.
    /// Its own kind (`Road::Leg`), so its record is apart from exploration's.
    leg: Option<u16>,
    /// `Road::Climb` (road.rs): toward a far goal under the roadmap's direct
    /// cost (frames per unit of height above a jump, the jump height), for a
    /// goal no known way leads to: height first, then distance.
    climb: Option<(f64, f32)>,
}

impl PosDist {
    pub fn new(b: &St, p: [f32; 3]) -> PosDist {
        PosDist { p, zone: b.zone(), swim: false, leg: None, climb: None }
    }

    /// Progress toward the far goal `p` under the roadmap's direct cost.
    pub fn climb(b: &St, p: [f32; 3], per_unit: f64, jump: f32) -> PosDist {
        PosDist { p, zone: b.zone(), swim: false, leg: None, climb: Some((per_unit, jump)) }
    }

    /// A roadmap leg to `p`, an edge the tree made in `frames` frames.
    pub fn leg(b: &St, p: [f32; 3], frames: u32) -> PosDist {
        PosDist { p, zone: b.zone(), swim: false, leg: Some((frames * 3 / 2 + 45).min(600) as u16), climb: None }
    }

    /// An `OBRRT_SWIM_TARGETS` point ([`underwater_point`]).
    pub fn new_swim(b: &St, p: [f32; 3]) -> PosDist {
        PosDist { p, zone: b.zone(), swim: true, leg: None, climb: None }
    }
}

impl Objective for PosDist {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): exploration: a state every 30 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 30, air: false }
    }
    fn kind(&self) -> &'static str {
        if self.leg.is_some() {
            "Road::Leg"
        } else if self.climb.is_some() {
            "Road::Climb"
        } else {
            "Mario::PosDist"
        }
    }
    fn key(&self) -> String {
        // Points in the same 400-unit cell are the same objective.
        let c = |v: f32| (v / 400.0).round() as i32;
        if let Some(h) = self.leg {
            return format!("Leg {:?} {},{},{} {}", self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]), h / 60);
        }
        if self.climb.is_some() {
            return format!("Climb {:?} {},{},{}", self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]));
        }
        format!("PosDist {:?} {},{},{}", self.zone, c(self.p[0]), c(self.p[1]), c(self.p[2]))
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.dist(self.p) > RADIUS
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let d = s.dist(self.p);
        if d <= RADIUS {
            return Eta::Done;
        }
        if let Some((per, jump)) = self.climb {
            // road.rs direct_s: water up to its surface is free.
            // The height of the footing: an airborne state has the floor's.
            let (wl, y) = (s.water_level as f32, s.footing()[1]);
            let from = if y < wl { wl.min(self.p[1]).max(y) } else { y };
            let up = (self.p[1] - from - jump).max(0.0);
            let down = (y - self.p[1]).max(0.0);
            return Eta::Frames(frames_for(s.hdist(self.p).max(down * 0.5)) + per * up as f64);
        }
        Eta::Frames(frames_for(d))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        if let Some(h) = self.leg {
            return Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: h as usize });
        }
        if self.climb.is_some() {
            return Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 });
        }
        Some(obrrt_core::GaBudget { population: 12, generations: 4, horizon: if self.swim { 180 } else { 90 } })
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.p)
    }
}
