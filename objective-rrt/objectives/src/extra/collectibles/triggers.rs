//! Invisible triggers that deactivate when Mario touches them
//! (obj_check_if_collided_with_object, 100x100 hitbox in behavior_data.c):
//!
//! - `SecretTrigger::Touch` (bhvHiddenStarTrigger, hidden_star.inc.c): each
//!   one counts up the course's bhvHiddenStar; at 5 its star appears. They
//!   are the "secrets" of BoB (in the sky coin rings: Mario Wings to the
//!   Sky), WDW (Secrets in the Shallows & Sky), SSL (Pyramid Puzzle, inside
//!   the pyramid) and THI (Five Itty Bitty Secrets).
//! - `OneUpTrigger::Touch` (bhvHidden1upTrigger, bhvHidden1upInPoleTrigger,
//!   mushroom_1up.inc.c): each counts up the nearest hidden 1-up, which comes
//!   out once its count reaches its 2nd behaviour byte. The pole ones are
//!   spawned 0 and 200 below a bhvHidden1upInPoleSpawner (usually atop a
//!   pole or tree) when Mario comes within 700 of it.

use obrrt_core::layout::*;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St};

use super::common::{closest, find, gone};

/// sm64.h.
const MARIO_WING_CAP: u32 = 0x08;

/// The star a course's bhvHiddenStar gives (its behaviour byte 0, from
/// levels/{bob,wdw,ssl,thi}/script.c).
fn secret_star(level: u32) -> Option<u8> {
    match level {
        LEVEL_BOB => Some(4),
        LEVEL_WDW => Some(2),
        LEVEL_SSL => Some(5),
        LEVEL_THI => Some(3),
        _ => None,
    }
}

pub struct Touch {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
    secret: bool,
    star: Option<(i16, u8)>,
}

impl Touch {
    pub fn secret(b: &St, o: &Obj) -> Touch {
        // Prefer the hidden star's own parameter when it's loaded.
        let index = b
            .objects()
            .iter()
            .find(|h| h.bhv_name() == "bhvHiddenStar")
            .map(|h| (h.beh_params >> 24) as u8)
            .filter(|&i| i <= 5)
            .or_else(|| secret_star(b.level as u32));
        let star = index.filter(|_| (1..=15).contains(&b.course)).map(|i| (b.course, i));
        Touch { id: o.id, zone: b.zone(), pos: o.pos, secret: true, star }
    }

    pub fn one_up(b: &St, o: &Obj) -> Touch {
        Touch { id: o.id, zone: b.zone(), pos: o.pos, secret: false, star: None }
    }
}

impl Objective for Touch {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        if self.secret { "SecretTrigger::Touch" } else { "OneUpTrigger::Touch" }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", if self.secret { "Secret" } else { "OneUpTrigger" }, self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        if self.secret { 8.0 } else { 0.5 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        // BoB's five secrets float 3600-5500 up over the field
        // (levels/bob/areas/1/macro.inc.c): only reachable flying, so only
        // from states with the wing cap. The other courses' secrets sit on
        // the level (WDW, SSL's pyramid, THI) and need no cap.
        let needs_wings = self.secret && self.zone.0 as u32 == obrrt_core::layout::LEVEL_BOB;
        find(s, self.zone, self.id).is_some()
            && self.star.is_none_or(|(c, i)| !s.has_star(c, i))
            && (!needs_wings || s.mario_flags & MARIO_WING_CAP != 0)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let mut h = s.dist(self.pos) as f64 / 400.0;
        // A secret high above Mario (BoB's sky rings) wants the wing cap.
        if self.secret && self.pos[1] > s.pos[1] + 800.0 && s.mario_flags & MARIO_WING_CAP == 0 {
            h += 5.0;
        }
        h
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| gone(s, self.zone, self.id))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, |s| Some(self.pos).filter(|_| s.zone() == self.zone))
    }
    fn success_value(&self) -> f64 {
        if self.secret { 6.0 } else { 0.3 }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    fn kinship(&self) -> u32 {
        if self.secret { 10 } else { 6 }
    }
}
