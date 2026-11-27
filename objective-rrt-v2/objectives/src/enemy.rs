//! `Enemy::Defeat(id)`: defeat an enemy that dies to a stomp or a hit
//! (`INTERACT_BOUNCE_TOP`, `BOUNCE_TOP2`, `HIT_FROM_BELOW`, `KOOPA`): goombas,
//! koopas, piranha plants and the like. Most drop a coin.

use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

pub struct Defeat {
    id: ObjId,
    zone: (u16, u8),
    weight: f64,
}

impl Defeat {
    pub fn new(b: &St, o: &Obj) -> Defeat {
        // One that drops coins is worth defeating; one that doesn't is only
        // worth it to clear the way, so it keeps a little weight.
        Defeat { id: o.id, zone: b.zone(), weight: if o.loot_coins > 0 { 1.0 } else { 0.2 } }
    }

    fn pos_in(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| o.pos)
    }
}

impl Objective for Defeat {
    fn kind(&self) -> &'static str {
        "Enemy::Defeat"
    }
    fn key(&self) -> String {
        format!("Defeat {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && self.pos_in(s).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // Gone while Mario stayed in the zone (not unloaded by a warp).
        match self.pos_in(s) {
            None => Eta::Done,
            Some(p) => Eta::Frames(frames_for(dist(s.pos, p))),
        }
    }
    fn success_value(&self) -> f64 {
        self.weight
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Aim a little above it, so the controller jumps onto it.
        self.pos_in(s).map(|p| [p[0], p[1] + 120.0, p[2]])
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}
