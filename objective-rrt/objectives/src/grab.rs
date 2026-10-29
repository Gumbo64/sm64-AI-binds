//! `Object::Grab(id)`: pick up a grabbable object (`INTERACT_GRABBABLE`:
//! bob-ombs, small boxes, King Bob-omb from behind). Holding something is the
//! start of throwing it.

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

pub struct Grab {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
    from_behind: bool,
}

impl Grab {
    pub fn new(b: &St, o: &Obj, weight: f64, from_behind: bool) -> Grab {
        Grab { id: o.id, zone: b.zone(), weight, from_behind }
    }

    fn pos_in(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| o.pos)
    }
}

impl Objective for Grab {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Object::Grab"
    }
    fn key(&self) -> String {
        format!("Grab {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.holding && self.pos_in(s).is_some()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.pos_in(s).map_or(f64::MAX, |p| dist(s.pos, p) as f64 / 400.0)
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.held().is_some_and(|h| h.id == self.id))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter_map(|(i, s)| self.pos_in(s).map(|p| (i, dist(s.pos, p))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 150 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Bosses are grabbed from behind: aim for a point behind it first.
        let o = s.find(self.id)?;
        if self.from_behind && dist(s.pos, o.behind(150.0)) > 120.0 && dist(s.pos, o.pos) < dist(o.behind(150.0), o.pos) * 4.0 {
            Some(o.behind(150.0))
        } else {
            Some(o.pos)
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match self.pos_in(s) {
            Some(p) if dist(s.pos, p) < 180.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.pos_in(s).is_some()
    }
}
