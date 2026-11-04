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

use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::find;

const FAKE_MONEYBAG_COIN_ACT_TRANSFORM: i32 = 1;
const MONEYBAG_ACT_DISAPPEAR: i32 = 4;
const MONEYBAG_ACT_DEATH: i32 = 5;
/// A hidden one transforms with Mario within 400 (bhv_moneybag_hidden_loop).
const REVEAL_RADIUS: f32 = 400.0;
/// In range to its next update seeing him.
const REVEAL_FRAMES: f64 = 1.0;
/// Close to it to landing on it: a jump onto a hopping target (~2/3 s).
const STOMP_FRAMES: f64 = 20.0;

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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(match self.goal {
            Goal::Reveal => GaBudget { population: 12, generations: 4, horizon: 120 },
            Goal::Stomp => GaBudget { population: 16, generations: 5, horizon: 180 },
        })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        // Reveal counts it gone as done; a state whose objects weren't read
        // can't tell.
        if (o.is_some() || s.objects.is_some()) && self.done(o) {
            return Eta::Done;
        }
        // Gone, or a Moneybag fading out (Mario went too far): no kill.
        let Some(o) = o.filter(|o| self.live(o)) else { return Eta::Lost };
        let d = dist(s.pos, o.pos);
        Eta::Frames(match self.goal {
            Goal::Reveal => frames_for((d - REVEAL_RADIUS).max(0.0)) + REVEAL_FRAMES,
            Goal::Stomp => frames_for(d) + STOMP_FRAMES,
        })
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
