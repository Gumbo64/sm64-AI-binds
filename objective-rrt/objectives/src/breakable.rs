//! `Object::Break(id)`: break a box. Breakable boxes (`INTERACT_BREAKABLE`)
//! break to a punch, kick or ground pound; `!` boxes (`bhvExclamationBox`) to
//! a hit from below or a ground pound, and hold coins, caps or stars.

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

pub struct Break {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
    exclamation: bool,
}

/// What a `!` box holds, from its second behaviour-parameter byte
/// (exclamation_box.inc.c): 0-2 caps (wing, metal, vanish), 8-13 stars.
pub fn exclamation_weight(o: &Obj) -> f64 {
    match (o.beh_params >> 16) & 0xFF {
        8..=13 => 30.0,
        0..=2 => 3.0,
        _ => 2.0,
    }
}

impl Break {
    pub fn new(b: &St, o: &Obj, weight: f64) -> Break {
        Break { id: o.id, zone: b.zone(), weight, exclamation: o.bhv_name() == "bhvExclamationBox" }
    }

    fn obj<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        s.find(self.id)
    }
}

impl Objective for Break {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Object::Break"
    }
    fn key(&self) -> String {
        format!("Break {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        // A ! box is only breakable while solid (action 2); action 1 is the
        // outline before its cap switch is pressed, 5 hidden until it respawns.
        s.zone() == self.zone && self.obj(s).is_some_and(|o| !self.exclamation || o.action == 2)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.obj(s).map_or(f64::MAX, |o| dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Gone, or (for ! boxes, which stay as an empty shell) its action
        // moved past its first two.
        let a0 = self.obj(&traj[0])?.action;
        traj.iter().position(|s| s.zone() == self.zone && self.obj(s).is_none_or(|o| o.action != a0 && o.action >= 3))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter_map(|(i, s)| self.obj(s).map(|o| (i, dist(s.pos, o.pos))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn success_value(&self) -> f64 {
        self.weight
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 150 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.obj(s).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
