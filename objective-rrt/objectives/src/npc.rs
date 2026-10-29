//! `Npc::Talk(id)`: start a conversation (Bob-omb Buddies open cannons,
//! Koopa the Quick starts his race, Toads give stars). Talking needs Mario
//! touching the NPC, facing it within 90 degrees, and pressing A or B
//! (interaction.c, check_npc_talk).

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::layout::*;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

pub struct Talk {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
}

impl Talk {
    pub fn new(b: &St, o: &Obj, weight: f64) -> Talk {
        Talk { id: o.id, zone: b.zone(), weight }
    }

    fn pos_in(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| o.pos)
    }
}

impl Objective for Talk {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): talking: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Npc::Talk"
    }
    fn key(&self) -> String {
        format!("Talk {:?} {}", self.zone, self.id)
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
        traj.iter().position(|s| {
            matches!(s.action, ACT_WAITING_FOR_DIALOG | ACT_READING_NPC_DIALOG)
                && self.pos_in(s).is_some_and(|p| dist(s.pos, p) < 400.0)
        })
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
        Budget::Ga { population: 12, generations: 4, horizon: 150 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.pos_in(s)
    }
    fn steer_button(&self, s: &St) -> u16 {
        match self.pos_in(s) {
            Some(p) if dist(s.pos, p) < 200.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
