//! `Enemy::Defeat(id)`: defeat an enemy that dies to a stomp or a hit
//! (`INTERACT_BOUNCE_TOP`, `BOUNCE_TOP2`, `HIT_FROM_BELOW`, `KOOPA`): goombas,
//! koopas, piranha plants and the like. Most drop a coin.

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

pub struct Defeat {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
}

impl Defeat {
    pub fn new(b: &St, o: &Obj) -> Defeat {
        // One that drops coins is worth defeating; one that doesn't is only
        // worth it to clear the way, so it keeps a little weight.
        Defeat { id: o.id, zone: b.zone(), weight: if o.loot_coins > 0 { 1.0 } else { 0.2 } }
    }

    fn pos_in(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| o.pos)
    }
}

impl Objective for Defeat {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Enemy::Defeat"
    }
    fn key(&self) -> String {
        format!("Defeat {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && self.pos_in(s).is_some()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.pos_in(s).map_or(f64::MAX, |p| dist(s.pos, p) as f64 / 400.0)
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Gone while Mario stayed in the zone (not unloaded by a warp).
        traj.iter().position(|s| s.zone() == self.zone && self.pos_in(s).is_none())
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
        // Aim a little above it, so the controller jumps onto it.
        self.pos_in(s).map(|p| [p[0], p[1] + 120.0, p[2]])
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
