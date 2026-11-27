//! `Whomp::GroundPound(id)`: beat a small Whomp by ground pounding its back.
//!
//! A small Whomp (bhvSmallWhomp) patrols; when Mario is within 300 in front
//! of it (and within 0x2000 of its facing) it jumps and falls flat on its
//! face (actions 3-5), then lies there (action 6) for 100 frames. Standing
//! on its back gives a coin; ground pounding it there kills it (action 8)
//! with 5 coins (whomp_patrol, whomp_on_ground, whomp_die; whomp.inc.c).
//! The Whomp King shares the code but is a boss.

use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::{find, front, hdist, in_air};

const WHOMP_ACT_ON_GROUND: i32 = 6;
const WHOMP_ACT_DIE: i32 = 8;
/// whomp_prepare_jump, whomp_jump, whomp_land: falling flat.
const WHOMP_ACT_PREPARE_JUMP: i32 = 3;
const WHOMP_ACT_LAND: i32 = 5;

// ETA phases, in frames (as the Whomp King's, extra/bosses.rs).
/// Standing in front of it to its jump windup starting (whomp_patrol turns
/// to face Mario first, ~1/3 s).
const LURE_FRAMES: f64 = 10.0;
/// Windup to lying flat: whomp_prepare_jump's animation (~20), whomp_jump's
/// 8 frames and its pitch to 0x4000 at +0x100/frame² (~11), the landing.
const FALL_FRAMES: f64 = 45.0;
/// Mid-fall (actions 3-5): about half of FALL_FRAMES left.
const FALLING_LEFT_FRAMES: f64 = 20.0;
/// Lying flat: from beside its back up onto it (a jump of ~100, 1/3 s).
const CLIMB_FRAMES: f64 = 10.0;
/// On its back to the pound landing: a jump (~10), the pound's 10-frame
/// spin and the drop (act_ground_pound).
const POUND_FRAMES: f64 = 25.0;
/// Mid-pound over its back to the landing.
const POUND_FALL_FRAMES: f64 = 8.0;
/// sm64.h.
const ACT_GROUND_POUND: u32 = 0x0080_08A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;

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

    /// Frames to the pound on its back: lure it from in front, its fall
    /// (dodged to the side), onto its back, the pound. Its back is where it
    /// will be once it lies flat.
    fn pound_eta(s: &St, o: &Obj) -> f64 {
        let back = Self::back(o);
        let from = |p: [f32; 3]| frames_for(dist(p, back)) + CLIMB_FRAMES + POUND_FRAMES;
        match o.action {
            WHOMP_ACT_ON_GROUND => {
                // On top: above it (v1's +300 bonus) and over its back.
                let on_top = s.pos[1] > o.pos[1] + 60.0 && hdist(s.pos, back) < 220.0;
                if on_top && matches!(s.action, ACT_GROUND_POUND | ACT_GROUND_POUND_LAND) {
                    POUND_FALL_FRAMES
                } else if on_top {
                    POUND_FRAMES
                } else {
                    from(s.pos)
                }
            }
            WHOMP_ACT_PREPARE_JUMP..=WHOMP_ACT_LAND => {
                let dodge = side(o, 450.0);
                FALLING_LEFT_FRAMES + frames_for(dist(s.pos, dodge)) + from(dodge)
            }
            _ => {
                let f = front(o, 280.0);
                frames_for(dist(s.pos, f)) + LURE_FRAMES + FALL_FRAMES + from(f)
            }
        }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        // Gone or dying; a state whose objects weren't read can't tell.
        if s.objects.is_some() && Self::dead(o) {
            return Eta::Done;
        }
        o.map_or(Eta::Lost, |o| Eta::Frames(Self::pound_eta(s, o)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{falls, obj, st};
    use obrrt_core::fold;

    /// Patrolling far, in front, falling (dodged aside), flat, on its back,
    /// pounding, dead: the ETA falls at every phase change.
    #[test]
    fn small_whomp_pound_phases() {
        let w = |action| obj("bhvSmallWhomp", 4, [0.0; 3], action, 0);
        let at = |pos: [f32; 3], action, mario: u32| {
            let mut s = st(24, pos, vec![w(action)]);
            if mario != 0 {
                s.action = mario;
            }
            s
        };
        let o = w(1);
        let back = GroundPound::back(&o);
        let on_back = [back[0], back[1] + 10.0, back[2]];
        let traj = vec![
            at([0.0, 0.0, 1500.0], 1, 0),
            at(front(&o, 280.0), 1, 0),
            at(front(&o, 280.0), WHOMP_ACT_PREPARE_JUMP, 0),
            at(side(&o, 450.0), 4, 0),
            at(side(&o, 450.0), WHOMP_ACT_ON_GROUND, 0),
            at(on_back, WHOMP_ACT_ON_GROUND, 0),
            at(on_back, WHOMP_ACT_ON_GROUND, ACT_GROUND_POUND),
            at(on_back, WHOMP_ACT_DIE, 0),
        ];
        let g = GroundPound::new(&traj[0], &o);
        let (met, etas) = fold(&g, &traj);
        assert_eq!(met, Some(7));
        let f: Vec<f64> = etas[..7].iter().map(|e| e.frames()).collect();
        assert!(falls(&f), "ETA must fall: {f:?}");
    }
}
