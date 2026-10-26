//! `Bully::PushOff(id)`: knock a small bully (or small chill bully) off its
//! platform into lava, icy water or the void.
//!
//! Bullies (`INTERACT_BULLY`) can't be stomped to death: any attack (punch,
//! kick, dive, ground pound, stomp) sends the bully flying along Mario's
//! facing yaw at 3392 / hitboxRadius (interact_bully, interaction.c), and it
//! dies only when it lands on lava or a death plane (bully_step,
//! obj_check_floor_death: `BULLY_ACT_LAVA_DEATH` = 100,
//! `BULLY_ACT_DEATH_PLANE_DEATH` = 101) or, for chill bullies, drops below
//! y 1030 (SL's water). A small one then spits out a coin
//! (bully_act_level_death). The three minions on LLL's big platform count
//! towards the Big Bully that appears once they're gone (star 2 of LLL,
//! bhv_big_bully_with_minions_*; bully.inc.c).
//!
//! The big bullies are bosses (the `bosses` category).

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::layout::LEVEL_LLL;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use super::common::{best, find, hdist};

const BULLY_ACT_LAVA_DEATH: i32 = 100;
const BULLY_ACT_DEATH_PLANE_DEATH: i32 = 101;

/// Where big_bully_spawn_minion puts LLL's three minions (bully.inc.c).
const MINIONS: [[f32; 3]; 3] = [[4454.0, 307.0, -5426.0], [3840.0, 307.0, -6041.0], [3226.0, 307.0, -5426.0]];

pub struct PushOff {
    id: ObjId,
    zone: (u16, u8),
    /// The LLL minion star (course 7, index 1) this counts towards.
    minion_of: Option<(i16, u8)>,
}

impl PushOff {
    pub fn new(b: &St, o: &Obj) -> PushOff {
        let minion = b.level as u32 == LEVEL_LLL && MINIONS.iter().any(|m| hdist(*m, o.home) < 150.0);
        PushOff { id: o.id, zone: b.zone(), minion_of: minion.then_some((7, 1)) }
    }

    fn dying(o: &Obj) -> bool {
        o.action == BULLY_ACT_LAVA_DEATH || o.action == BULLY_ACT_DEATH_PLANE_DEATH
    }

    /// Where to go and what to press: get onto the platform side of the
    /// bully (its home is on the platform), then run through it outward,
    /// attacking.
    fn plan(&self, s: &St) -> Option<([f32; 3], u16)> {
        let o = find(s, self.id, false)?;
        let (mut dx, mut dz) = (o.pos[0] - o.home[0], o.pos[2] - o.home[2]);
        let mut n = (dx * dx + dz * dz).sqrt();
        if n < 50.0 {
            // At home: push it away from Mario, whichever way that is.
            dx = o.pos[0] - s.pos[0];
            dz = o.pos[2] - s.pos[2];
            n = (dx * dx + dz * dz).sqrt();
            if n < 1.0 {
                return Some((o.pos, B_BUTTON));
            }
        }
        let (ux, uz) = (dx / n, dz / n);
        let (rx, rz) = (s.pos[0] - o.pos[0], s.pos[2] - o.pos[2]);
        let along = rx * ux + rz * uz;
        if along < -60.0 {
            // On the inside: charge outward through it.
            let button = if hdist(s.pos, o.pos) < 250.0 { B_BUTTON } else { 0 };
            Some(([o.pos[0] + ux * 300.0, s.pos[1], o.pos[2] + uz * 300.0], button))
        } else {
            // Outside or level with it: go round, beside it, to the inside.
            let side = if rx * -uz + rz * ux >= 0.0 { 1.0 } else { -1.0 };
            let p = [o.pos[0] - ux * 200.0 - side * uz * 180.0, s.pos[1], o.pos[2] - uz * 200.0 + side * ux * 180.0];
            Some((p, 0))
        }
    }
}

impl Objective for PushOff {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Bully::PushOff"
    }
    fn key(&self) -> String {
        format!("BullyPushOff {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        match self.minion_of {
            Some((c, i)) if !b.has_star(c, i) => 8.0,
            _ => 1.5,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && find(s, self.id, false).is_some_and(|o| !Self::dying(o))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        find(s, self.id, false).map_or(f64::MAX, |o| dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && find(s, self.id, false).is_none_or(Self::dying))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Pushed far from home and down from where it stood, Mario near it.
        let y0 = find(&traj[0], self.id, false).map_or(0.0, |o| o.pos[1]);
        best(traj, |s| {
            let o = find(s, self.id, false)?;
            Some(hdist(o.pos, o.home) as f64 + 2.0 * (y0 - o.pos[1]).max(0.0) as f64 - 0.25 * dist(s.pos, o.pos) as f64)
        })
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 5, horizon: 200 }
    }
    fn success_value(&self) -> f64 {
        if self.minion_of.is_some() { 3.0 } else { 1.0 }
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
