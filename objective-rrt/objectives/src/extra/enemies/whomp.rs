//! `Whomp::GroundPound(id)`: beat a small Whomp by ground pounding its back.
//!
//! A small Whomp (bhvSmallWhomp) patrols; when Mario is within 300 in front
//! of it (and within 0x2000 of its facing) it jumps and falls flat on its
//! face (actions 3-5), then lies there (action 6) for 100 frames. Standing
//! on its back gives a coin; ground pounding it there kills it (action 8)
//! with 5 coins (whomp_patrol, whomp_on_ground, whomp_die; whomp.inc.c).
//! The Whomp King shares the code but is a boss.

use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use super::common::{best, find, front, hdist, in_air};

const WHOMP_ACT_ON_GROUND: i32 = 6;
const WHOMP_ACT_DIE: i32 = 8;

pub struct GroundPound {
    id: ObjId,
    zone: (u16, u8),
}

impl GroundPound {
    pub fn new(b: &St, o: &Obj) -> GroundPound {
        GroundPound { id: o.id, zone: b.zone() }
    }

    fn dead(o: Option<&Obj>) -> bool {
        o.is_none_or(|o| o.action == WHOMP_ACT_DIE)
    }

    /// The middle of its back when it lies flat: it falls forward about its
    /// feet, so its body lies along its facing.
    fn back(o: &Obj) -> [f32; 3] {
        let p = front(o, 200.0);
        [p[0], o.pos[1] + 100.0, p[2]]
    }

    fn plan(&self, s: &St) -> Option<([f32; 3], u16)> {
        let o = find(s, self.id, false)?;
        Some(match o.action {
            WHOMP_ACT_ON_GROUND => {
                let back = Self::back(o);
                let button = if hdist(s.pos, back) > 220.0 {
                    0
                } else if in_air(s) && s.pos[1] > o.pos[1] + 60.0 {
                    Z_TRIG
                } else {
                    A_BUTTON
                };
                (back, button)
            }
            // Falling: get out from under it, sideways.
            3..=5 => (side(o, 450.0), 0),
            // Patrolling: stand in front of it to make it fall.
            _ => (front(o, 280.0), 0),
        })
    }
}

/// A point `d` to the object's side (its right, perpendicular to its
/// facing), level with it.
fn side(o: &Obj, d: f32) -> [f32; 3] {
    let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
    [o.pos[0] - d * a.cos(), o.pos[1], o.pos[2] + d * a.sin()]
}

impl Objective for GroundPound {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Whomp::GroundPound"
    }
    fn key(&self) -> String {
        format!("WhompPound {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        2.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !Self::dead(find(s, self.id, false))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = find(s, self.id, false) else { return f64::MAX };
        // Lying flat already is the best start.
        let lying = if o.action == WHOMP_ACT_ON_GROUND { 0.0 } else { 2.0 };
        lying + dist(s.pos, o.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && Self::dead(find(s, self.id, false)))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |s| {
            let o = find(s, self.id, false)?;
            Some(if o.action == WHOMP_ACT_ON_GROUND {
                let on_top = if s.pos[1] > o.pos[1] + 60.0 { 300.0 } else { 0.0 };
                1000.0 + on_top - dist(s.pos, Self::back(o)) as f64
            } else {
                -(dist(s.pos, front(o, 280.0)) as f64)
            })
        })
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 240 }
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.plan(s).map(|p| p.0)
    }
    fn steer_button(&self, s: &St) -> u16 {
        self.plan(s).map_or(0, |p| p.1)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
