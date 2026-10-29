//! `Door::Enter(id)`: go through a door. Warp doors (`INTERACT_WARP_DOOR`,
//! e.g. the castle's front doors) change level or area; ordinary doors
//! (`INTERACT_DOOR`) put Mario in a door action.

use obrrt_core::layout::*;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

fn in_door(s: &St) -> bool {
    matches!(
        s.action,
        ACT_PULLING_DOOR | ACT_PUSHING_DOOR | ACT_ENTERING_STAR_DOOR | ACT_UNLOCKING_STAR_DOOR | ACT_UNLOCKING_KEY_DOOR
    )
}

pub struct Enter {
    id: ObjId,
    pos: [f32; 3],
    zone: (u16, u8),
    /// Stars and keys the door needs.
    needs: i16,
    keys: u8,
}

impl Enter {
    pub fn new(b: &St, o: &Obj) -> Enter {
        // Castle doors (bhvDoor, bhvStarDoor) need the stars in their first
        // behaviour byte (interaction.c interact_door / interact_star_door).
        let needs = if b.level as u32 == obrrt_core::layout::LEVEL_CASTLE && matches!(o.bhv_name(), "bhvDoor" | "bhvStarDoor") {
            (o.beh_params >> 24) as i16
        } else {
            0
        };
        // Castle key doors (INTERACT_WARP_DOOR, interaction.c
        // interact_warp_door): door 2 (basement) needs key 1, door 1
        // (upstairs) needs key 2. Key 1 always comes first (Bowser in the
        // Dark World before the Fire Sea), so they need 1 and 2 keys.
        let keys = if b.level as u32 == obrrt_core::layout::LEVEL_CASTLE && o.interact_type == obrrt_core::layout::INTERACT_WARP_DOOR {
            match o.beh_params >> 24 {
                2 => 1,
                1 => 2,
                _ => 0,
            }
        } else {
            0
        };
        Enter { id: o.id, pos: o.pos, zone: b.zone(), needs, keys }
    }
}

impl Objective for Enter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): doors: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Door::Enter"
    }
    fn key(&self) -> String {
        format!("Door {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        8.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.stars >= self.needs && s.keys >= self.keys
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() != self.zone || (in_door(s) && s.dist(self.pos) < 300.0))
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
        2.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 180 }
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Door
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
}
