//! `Object::Break(id)`: break a box. Breakable boxes (`INTERACT_BREAKABLE`)
//! break to a punch, kick or ground pound; `!` boxes (`bhvExclamationBox`) to
//! a hit from below or a ground pound, and hold coins, caps or stars.

use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

pub struct Break {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
    exclamation: bool,
}

/// What a `!` box holds, from its second behaviour-parameter byte
/// (exclamation_box.inc.c sExclamationBoxContents): 0-2 caps (wing, metal,
/// vanish), 8 and 10-14 stars (indices 0-5); 9 is a 1-up that runs away
/// (it weighed 30 and the star box of index 5, Rainbow Ride's, weighed 2,
/// until 2026-11-09).
pub fn exclamation_weight(o: &Obj) -> f64 {
    match (o.beh_params >> 16) & 0xFF {
        8 | 10..=14 => 30.0,
        0..=2 => 3.0,
        _ => 2.0,
    }
}

impl Break {
    pub fn new(b: &St, o: &Obj, weight: f64) -> Break {
        Break { id: o.id, zone: b.zone(), weight, exclamation: o.bhv_name() == "bhvExclamationBox" }
    }

    fn obj<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        s.find(self.id)
    }
}

impl Objective for Break {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Object::Break"
    }
    fn key(&self) -> String {
        format!("Break {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        // A ! box is only breakable while solid (action 2); action 1 is the
        // outline before its cap switch is pressed, 5 hidden until it respawns.
        s.zone() == self.zone && self.obj(s).is_some_and(|o| !self.exclamation || o.action == 2)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // v1 judged the break against the box's action at the start.
        let Some(a0) = self.obj(start).map(|o| o.action) else { return Eta::Lost };
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // Gone, or (for ! boxes, which stay as an empty shell) its action
        // moved past its first two.
        match self.obj(s) {
            None => Eta::Done,
            Some(o) if o.action != a0 && o.action >= 3 => Eta::Done,
            Some(o) => Eta::Frames(frames_for(dist(s.pos, o.pos))),
        }
    }
    fn success_value(&self) -> f64 {
        self.weight
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.obj(s).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
