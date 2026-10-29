//! Moneybags (moneybag.inc.c).
//!
//! - `Moneybag::Reveal(id)`: a hidden one (bhvMoneybagHidden) looks like a
//!   coin but hurts (`INTERACT_DAMAGE`); coming within 400 of it turns it
//!   into a Moneybag (`FAKE_MONEYBAG_COIN_ACT_TRANSFORM` = 1).
//! - `Moneybag::Stomp(id)`: the Moneybag (`INTERACT_BOUNCE_TOP`) hops
//!   around; stomped or hit it dies (`MONEYBAG_ACT_DEATH` = 5) with 5 coins.
//!   If Mario goes far away it fades out instead (`MONEYBAG_ACT_DISAPPEAR` =
//!   4), which `Enemy::Defeat` (success = gone) can't tell from a kill, so
//!   this one watches its action.

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use super::common::{best, find};

const FAKE_MONEYBAG_COIN_ACT_TRANSFORM: i32 = 1;
const MONEYBAG_ACT_DISAPPEAR: i32 = 4;
const MONEYBAG_ACT_DEATH: i32 = 5;

#[derive(Clone, Copy, PartialEq)]
enum Goal {
    Reveal,
    Stomp,
}

pub struct Moneybag {
    id: ObjId,
    zone: (u16, u8),
    goal: Goal,
}

impl Moneybag {
    pub fn new(b: &St, o: &Obj) -> Option<Moneybag> {
        let goal = match o.bhv_name() {
            "bhvMoneybagHidden" if o.action != FAKE_MONEYBAG_COIN_ACT_TRANSFORM => Goal::Reveal,
            "bhvMoneybag" if o.action < MONEYBAG_ACT_DISAPPEAR => Goal::Stomp,
            _ => return None,
        };
        Some(Moneybag { id: o.id, zone: b.zone(), goal })
    }

    fn done(&self, o: Option<&Obj>) -> bool {
        match self.goal {
            Goal::Reveal => o.is_none_or(|o| o.action == FAKE_MONEYBAG_COIN_ACT_TRANSFORM),
            Goal::Stomp => o.is_some_and(|o| o.action == MONEYBAG_ACT_DEATH),
        }
    }

    fn live(&self, o: &Obj) -> bool {
        match self.goal {
            Goal::Reveal => o.action != FAKE_MONEYBAG_COIN_ACT_TRANSFORM,
            Goal::Stomp => o.action < MONEYBAG_ACT_DISAPPEAR,
        }
    }
}

impl Objective for Moneybag {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        match self.goal {
            Goal::Reveal => "Moneybag::Reveal",
            Goal::Stomp => "Moneybag::Stomp",
        }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.goal {
            Goal::Reveal => 0.5,
            Goal::Stomp => 2.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && find(s, self.id, false).is_some_and(|o| self.live(o))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        find(s, self.id, false).map_or(f64::MAX, |o| dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && self.done(find(s, self.id, false)))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |s| find(s, self.id, false).map(|o| -(dist(s.pos, o.pos) as f64)))
    }
    fn budget(&self) -> Budget {
        match self.goal {
            Goal::Reveal => Budget::Ga { population: 12, generations: 4, horizon: 120 },
            Goal::Stomp => Budget::Ga { population: 16, generations: 5, horizon: 180 },
        }
    }
    fn success_value(&self) -> f64 {
        match self.goal {
            Goal::Reveal => 0.3,
            Goal::Stomp => 2.0,
        }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = find(s, self.id, false)?;
        match self.goal {
            // Close enough to transform it, without touching it.
            Goal::Reveal => Some([o.pos[0], s.pos[1], o.pos[2]]),
            // Aim above it, to land on it.
            Goal::Stomp => Some([o.pos[0], o.pos[1] + 120.0, o.pos[2]]),
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
