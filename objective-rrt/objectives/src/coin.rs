//! `Coin::Collect(id)`: touch a coin (`INTERACT_COIN`: yellow, red, blue and
//! spawned coins).

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

/// A coin counts as this one's if Mario is this close when the count rises.
const TOUCH: f32 = 250.0;

pub struct Collect {
    id: ObjId,
    bhv: &'static str,
    pos: [f32; 3],
    zone: (u16, u8),
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Collect {
        Collect { id: o.id, bhv: o.bhv_name(), pos: o.pos, zone: b.zone() }
    }

    fn here(&self, s: &St) -> bool {
        s.find(self.id).is_some()
    }
}

impl Objective for Collect {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Coin::Collect"
    }
    fn key(&self) -> String {
        format!("Coin {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.bhv {
            "bhvRedCoin" => 6.0,
            "bhvBlueCoin" | "bhvMovingBlueCoin" => 4.0,
            _ => 2.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && self.here(s)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let c0 = traj[0].coins;
        traj.iter().position(|s| s.coins > c0 && s.dist(self.pos) < TOUCH)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .map(|(i, s)| (i, dist(s.pos, self.pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        Progress { score: -(d as f64), at }
    }
    fn success_value(&self) -> f64 {
        if self.bhv == "bhvRedCoin" { 3.0 } else { 1.0 }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 150 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.here(s)
    }
}
