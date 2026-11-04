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

use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, PACE, St, course_of_level, frames_for};

use super::common::{bp1, bp2, capless, find, hdist};

const MR_I_ACT_DYING: i32 = 3;
const MR_BLIZZARD_ACT_DEATH: i32 = 6;

/// Angle to sweep round it in one direction: Mr. I dies past a 0x10000 turn
/// (mr_i_act_2); Mr. Blizzard's dizziness (+8/frame² while Mario is off its
/// facing) passes the 0x3000 roll after ~55 frames of steady circling,
/// about one lap at its radius and PACE (mr_blizzard_act_rotate).
const LAP_ANGLE: f64 = std::f64::consts::TAU;
/// `mem` (v1's progress accumulator, folded per frame): bit 0 = in range
/// last frame; f[0] = signed angle swept since entering range, f[1] = last
/// frame's angle.
const IN_RANGE: u32 = 1;

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

    /// Frames of circling left at this radius for the angle still to sweep.
    fn laps_left(&self, swept: f64) -> f64 {
        (LAP_ANGLE - swept.abs()).max(0.0) * self.radius as f64 / PACE
    }
}

impl Objective for Circle {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        // Gone or dying; a state whose objects weren't read can't tell.
        if s.objects.is_some() && self.dead(o) {
            return Eta::Done;
        }
        let Some(o) = o else {
            mem.flags &= !IN_RANGE;
            return Eta::Lost;
        };
        let d = hdist(s.pos, o.pos);
        if d > self.range {
            // Out of range: the turning resets (v1's accumulator did too).
            mem.flags &= !IN_RANGE;
            mem.f[0] = 0.0;
            return Eta::Frames(frames_for(d - self.radius) + self.laps_left(0.0));
        }
        let a = ((s.pos[0] - o.pos[0]) as f64).atan2((s.pos[2] - o.pos[2]) as f64);
        if mem.flags & IN_RANGE != 0 {
            let mut da = a - mem.f[1];
            if da > std::f64::consts::PI {
                da -= std::f64::consts::TAU;
            } else if da < -std::f64::consts::PI {
                da += std::f64::consts::TAU;
            }
            mem.f[0] += da;
        }
        mem.flags |= IN_RANGE;
        mem.f[1] = a;
        // Getting to the circle's radius, then the rest of the lap.
        Eta::Frames(frames_for((d - self.radius).abs()) + self.laps_left(mem.f[0]))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{obj, st};
    use obrrt_core::fold;

    /// Circling Mr. I: the swept angle (in `mem`) lowers the ETA each step,
    /// leaving its range resets it, and its dying act is `Done`.
    #[test]
    fn mr_i_circle_accumulates() {
        let mr_i = |action| obj("bhvMrI", 3, [0.0; 3], action, 0);
        let round = |a: f32, action| st(4, [300.0 * a.sin(), 0.0, 300.0 * a.cos()], vec![mr_i(action)]);
        let mut traj: Vec<St> = (0..10).map(|i| round(i as f32 * 0.6, 0)).collect();
        let c = Circle::mr_i(&traj[0], &traj[0].objects()[0]);
        let f: Vec<f64> = fold(&c, &traj).1.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA must fall: {f:?}");
        // Out of range: back to a whole lap (and the way in).
        traj.push(st(4, [0.0, 0.0, 1200.0], vec![mr_i(0)]));
        let e = fold(&c, &traj).1;
        assert!(e[10].frames() > f[0]);
        traj.push(round(0.0, MR_I_ACT_DYING));
        assert_eq!(fold(&c, &traj).0, Some(11));
    }
}
