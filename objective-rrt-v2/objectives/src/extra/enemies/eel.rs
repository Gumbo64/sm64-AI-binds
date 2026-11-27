//! `Eel::TouchTail`: JRB star 2, "Can the Eel Come Out to Play?".
//!
//! The act-2 eel (`bhvUnagi`, 2nd param byte 1, home (8270, -3130, 1846))
//! carries the star on its tail (oAnimState 1). Within 3000 units of Mario
//! it spawns eight `bhvUnagiSubobject`s along its body (unagi.inc.c
//! bhv_unagi_loop); the one with 2nd byte -4, 1200 units behind the head, is
//! the tail tip: Mario within 150 of it spawns the star at (6833, -3654,
//! 2230) (bhv_unagi_subobject_loop). The eel waits in its hole (action 3)
//! and comes out along its path once Mario is within 1000 of its head side
//! for a while, so the tail is only in reach while it swims.
//!
//! Nothing covered this (its interaction type, INTERACT_CLAM_OR_BUBBA, has
//! no objective): audit of 2026-11-09, STAR120-AUDIT.md item 3.

use obrrt_core::{Eta, GaBudget, Mem, Obj, Objective, St, dist, frames_for};

use super::common::bp2;

/// JRB's eel star: course 3, index 1.
const STAR: (i16, u8) = (3, 1);
const TOUCH: f32 = 150.0;

pub struct TouchTail {
    zone: (u16, u8),
}

/// The star-carrying eel.
fn eel(s: &St) -> Option<&Obj> {
    s.objects().iter().find(|o| o.bhv_name() == "bhvUnagi" && bp2(o) == 1)
}

/// Its tail tip, once the body parts exist.
fn tail(s: &St) -> Option<&Obj> {
    s.objects().iter().find(|o| o.bhv_name() == "bhvUnagiSubobject" && bp2(o) == 0xFC)
}

impl TouchTail {
    pub fn new(b: &St, o: &Obj) -> Option<TouchTail> {
        (o.bhv_name() == "bhvUnagi" && bp2(o) == 1 && !b.has_star(STAR.0, STAR.1)).then_some(TouchTail { zone: b.zone() })
    }

    /// Where to go: the tail tip, or the eel itself until its parts spawn.
    fn target(s: &St) -> Option<[f32; 3]> {
        tail(s).or_else(|| eel(s)).map(|o| o.pos)
    }
}

impl Objective for TouchTail {
    fn kind(&self) -> &'static str {
        "Eel::TouchTail"
    }
    fn key(&self) -> String {
        "EelTouchTail".into()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.alive() && !s.has_star(STAR.0, STAR.1) && eel(s).is_some()
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 20, generations: 6, horizon: 300 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // The star is out (spawned at the tail, it flies to its place).
        if s.has_star(STAR.0, STAR.1) || s.objects().iter().any(crate::star::spawned_star) {
            return Eta::Done;
        }
        let Some(t) = Self::target(s) else { return Eta::Lost };
        Eta::Frames(frames_for((dist(s.pos, t) - TOUCH).max(0.0)))
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Self::target(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && eel(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}
