//! `Klepto::Hit`: hit the condor in SSL so it drops what it carries.
//!
//! Klepto (`INTERACT_HIT_FROM_BELOW`) is struck by jumping into it from
//! below or by a punch or kick; it doesn't die, it goes to
//! `KLEPTO_ACT_STRUCK_BY_MARIO` (6) and drops the star (act 1, 2nd param
//! byte 1: "In the Talons of the Big Bird", SSL star 1) or Mario's cap
//! (bhv_klepto_update, klepto.inc.c). In the other acts it starts waiting
//! for Mario (action 2) and dives to steal his cap; while it carries the cap
//! it circles its targets (actions 0 and 1).
//!
//! Klepto steers by moving its own `oHome` between targets
//! (klepto_change_target), so it is found by behaviour, not by `ObjId`;
//! there's one per level.

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use super::common::{best, bp2, find};

const KLEPTO_ACT_STRUCK_BY_MARIO: i32 = 6;
/// SSL's Klepto star: course 8, index 0.
const STAR: (i16, u8) = (8, 0);

pub struct Hit {
    id: ObjId,
    zone: (u16, u8),
    star: bool,
}

impl Hit {
    /// Only a Klepto carrying the star or the cap is worth hitting.
    pub fn new(b: &St, o: &Obj) -> Option<Hit> {
        let star = bp2(o) != 0;
        if star && b.has_star(STAR.0, STAR.1) {
            return None;
        }
        // 2nd byte 0 starts in action 2 (waiting) unless it has the cap
        // (bhv_klepto_init); actions 0/1 mean it's carrying it.
        let cap = !star && (o.action == 0 || o.action == 1);
        (star || cap).then_some(Hit { id: o.id, zone: b.zone(), star })
    }

    fn klepto<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        find(s, self.id, true)
    }
}

impl Objective for Hit {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Klepto::Hit"
    }
    fn key(&self) -> String {
        format!("KleptoHit {:?} {}", self.zone, self.star)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        if self.star { 20.0 } else { 4.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && self.klepto(s).is_some_and(|o| o.action != KLEPTO_ACT_STRUCK_BY_MARIO && (self.star || o.action <= 1))
            && !(self.star && s.has_star(STAR.0, STAR.1))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.klepto(s).map_or(f64::MAX, |o| dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && self.klepto(s).is_some_and(|o| o.action == KLEPTO_ACT_STRUCK_BY_MARIO))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |s| self.klepto(s).map(|o| -(dist(s.pos, o.pos) as f64)))
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 6, horizon: 300 }
    }
    fn success_value(&self) -> f64 {
        if self.star { 10.0 } else { 2.0 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.klepto(s).map(|o| o.pos)
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Jump kick / punch when close; otherwise the usual jumping.
        match self.klepto(s) {
            Some(o) if dist(s.pos, o.pos) < 220.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.klepto(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}
