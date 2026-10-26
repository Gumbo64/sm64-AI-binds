//! `MrI::Circle(id)` / `MrBlizzard::Circle(id)`: run circles around an enemy
//! that turns to face Mario until it gets dizzy and dies. Neither can be
//! attacked directly (both are pure damage to touch).
//!
//! - Mr. I (`INTERACT_DAMAGE`) turns to follow Mario while he's within 800;
//!   once it has turned a full 0x10000 in one direction without a pause it
//!   dies (mr_i_act_2 -> action 3, mr_i_act_3). The big one (2nd param
//!   byte 1, BBH's secret room) leaves the star named by its 1st param byte;
//!   small ones leave a blue coin (mr_i.inc.c).
//! - Mr. Blizzard (`INTERACT_MR_BLIZZARD`) turns to face Mario on the
//!   ground, building dizziness; past a 0x3000 roll it falls over
//!   (`MR_BLIZZARD_ACT_DEATH` = 6), drops 3 coins, and gives back Mario's cap
//!   if it was wearing it (mr_blizzard_act_rotate / _death,
//!   mr_blizzard.inc.c).

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, course_of_level, dist};

use super::common::{bp1, bp2, capless, find, hdist};

const MR_I_ACT_DYING: i32 = 3;
const MR_BLIZZARD_ACT_DEATH: i32 = 6;

pub struct Circle {
    id: ObjId,
    zone: (u16, u8),
    kind: &'static str,
    weight: f64,
    star: Option<(i16, u8)>,
    /// Mario must stay this close for the turning to count.
    range: f32,
    radius: f32,
    death_action: i32,
}

impl Circle {
    pub fn mr_i(b: &St, o: &Obj) -> Circle {
        let star = (bp2(o) != 0).then(|| (course_of_level(b.level as u32), bp1(o)));
        Circle {
            id: o.id,
            zone: b.zone(),
            kind: "MrI::Circle",
            weight: if star.is_some() { 20.0 } else { 2.0 },
            star,
            range: 800.0,
            radius: 300.0,
            death_action: MR_I_ACT_DYING,
        }
    }

    pub fn mr_blizzard(b: &St, o: &Obj) -> Circle {
        // Without its cap Mario takes double damage; Mr. Blizzard may be
        // wearing it (SAVE_FLAG_CAP_ON_MR_BLIZZARD, not readable here).
        let capless = capless(b);
        Circle {
            id: o.id,
            zone: b.zone(),
            kind: "MrBlizzard::Circle",
            weight: if capless { 4.0 } else { 1.5 },
            star: None,
            range: 1000.0,
            radius: 250.0,
            death_action: MR_BLIZZARD_ACT_DEATH,
        }
    }

    fn dead(&self, o: Option<&Obj>) -> bool {
        o.is_none_or(|o| o.action == self.death_action)
    }
}

impl Objective for Circle {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind, self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && !self.dead(find(s, self.id, false))
            && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        find(s, self.id, false).map_or(f64::MAX, |o| dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && self.dead(find(s, self.id, false)))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // How far Mario has gone round it in one direction while in range.
        let mut acc = 0.0f64;
        let mut prev: Option<f64> = None;
        let mut out = Progress { score: f64::MIN, at: 0 };
        for (i, s) in traj.iter().enumerate() {
            let Some(o) = find(s, self.id, false) else {
                prev = None;
                continue;
            };
            let d = hdist(s.pos, o.pos);
            if d > self.range {
                acc = 0.0;
                prev = None;
            } else {
                let a = ((s.pos[0] - o.pos[0]) as f64).atan2((s.pos[2] - o.pos[2]) as f64);
                if let Some(p) = prev {
                    let mut da = a - p;
                    if da > std::f64::consts::PI {
                        da -= std::f64::consts::TAU;
                    } else if da < -std::f64::consts::PI {
                        da += std::f64::consts::TAU;
                    }
                    acc += da;
                }
                prev = Some(a);
            }
            let score = acc.abs() * 200.0 - d as f64 * 0.1;
            if score > out.score {
                out = Progress { score, at: i };
            }
        }
        out
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 240 }
    }
    fn success_value(&self) -> f64 {
        if self.star.is_some() { 10.0 } else { 1.5 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = find(s, self.id, false)?;
        if hdist(s.pos, o.pos) > self.range * 0.8 {
            Some([o.pos[0], s.pos[1], o.pos[2]])
        } else {
            Some(super::common::circle_point(s, o.pos, self.radius))
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
