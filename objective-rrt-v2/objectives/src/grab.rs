//! `Object::Grab(id)`: pick up a grabbable object (`INTERACT_GRABBABLE`:
//! bob-ombs, small boxes, King Bob-omb from behind). Holding something is the
//! start of throwing it.

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

pub struct Grab {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
    from_behind: bool,
}

impl Grab {
    pub fn new(b: &St, o: &Obj, weight: f64, from_behind: bool) -> Grab {
        Grab { id: o.id, zone: b.zone(), weight, from_behind }
    }

    fn pos_in(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| o.pos)
    }
}

impl Objective for Grab {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Object::Grab"
    }
    fn key(&self) -> String {
        format!("Grab {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.holding && self.pos_in(s).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.held().is_some_and(|h| h.id == self.id) {
            return Eta::Done;
        }
        // Gone without being held (exploded, broken, unloaded): can't be grabbed.
        match self.pos_in(s) {
            Some(p) => Eta::Frames(frames_for(dist(s.pos, p))),
            None => Eta::Lost,
        }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Bosses are grabbed from behind: aim for a point behind it first.
        let o = s.find(self.id)?;
        if self.from_behind && dist(s.pos, o.behind(150.0)) > 120.0 && dist(s.pos, o.pos) < dist(o.behind(150.0), o.pos) * 4.0 {
            Some(o.behind(150.0))
        } else {
            Some(o.pos)
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match self.pos_in(s) {
            Some(p) if dist(s.pos, p) < 180.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.pos_in(s).is_some()
    }
}
