//! `Coin::Collect(id)`: touch a coin (`INTERACT_COIN`: yellow, red, blue and
//! spawned coins).

use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, frames_for};

/// A coin counts as this one's if Mario is this close when the count rises.
const TOUCH: f32 = 250.0;

pub struct Collect {
    id: ObjId,
    bhv: &'static str,
    pos: [f32; 3],
    zone: (u16, u8),
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Collect {
        Collect { id: o.id, bhv: o.bhv_name(), pos: o.pos, zone: b.zone() }
    }

    fn here(&self, s: &St) -> bool {
        s.find(self.id).is_some()
    }
}

impl Objective for Collect {
    fn kind(&self) -> &'static str {
        "Coin::Collect"
    }
    fn key(&self) -> String {
        format!("Coin {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.bhv {
            "bhvRedCoin" => 6.0,
            "bhvBlueCoin" | "bhvMovingBlueCoin" => 4.0,
            _ => 2.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && self.here(s)
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.coins > start.coins && s.dist(self.pos) < TOUCH {
            return Eta::Done;
        }
        Eta::Frames(frames_for(s.dist(self.pos)))
    }
    fn success_value(&self) -> f64 {
        if self.bhv == "bhvRedCoin" { 3.0 } else { 1.0 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.here(s)
    }
}
