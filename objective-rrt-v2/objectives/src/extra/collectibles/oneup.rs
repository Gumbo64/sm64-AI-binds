//! `OneUp::Collect(id)`: touch a 1-up mushroom (mushroom_1up.inc.c). 1-ups
//! have no interact type: they check collision with Mario themselves and add
//! a life, so success is `lives` going up. Most move: walking ones drift,
//! running-away ones flee Mario (action 1), sliding ones roll downhill once he
//! is within 1000, pole ones home in on him; all are steered to as they move.
//!
//! The hidden ones (bhvHidden1up, bhvHidden1upInPole) stay invisible in
//! action 0 until their triggers are touched (`OneUpTrigger::Touch`), so they
//! are only attempted once revealed.

use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St};

use super::common::{approach, find};

pub fn is_one_up(bhv: &str) -> bool {
    matches!(
        bhv,
        "bhv1Up" | "bhv1upWalking" | "bhv1upRunningAway" | "bhv1upSliding" | "bhv1upJumpOnApproach" | "bhvHidden1up" | "bhvHidden1upInPole"
    )
}

pub struct Collect {
    id: ObjId,
    zone: (u16, u8),
    hidden: bool,
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Collect {
        Collect { id: o.id, zone: b.zone(), hidden: matches!(o.bhv_name(), "bhvHidden1up" | "bhvHidden1upInPole") }
    }

    fn obj<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        // A hidden 1-up in action 0 is not out yet.
        find(s, self.zone, self.id).filter(|o| !self.hidden || o.action != 0)
    }
}

impl Objective for Collect {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "OneUp::Collect"
    }
    fn key(&self) -> String {
        format!("OneUp {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.obj(s).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.lives > start.lives {
            return Eta::Done;
        }
        approach(s, self.obj(s).map(|o| o.pos))
    }
    fn success_value(&self) -> f64 {
        0.5
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.obj(s).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        // Kept while it exists at all: a hidden one may be revealed later.
        find(s, self.zone, self.id).is_some()
    }
}
