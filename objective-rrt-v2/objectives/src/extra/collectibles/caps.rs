//! `Cap::Collect(id)`: touch a cap (`INTERACT_CAP`, cap.inc.c): the wing,
//! metal and vanish caps that pop out of ! boxes, and Mario's own cap
//! (bhvNormalCap) after wind, Klepto or Ukiki took it. Touching one sets its
//! flag in `m->flags` (interaction.c, interact_cap / get_mario_cap_flag), so
//! success is read from Mario, not from coins or the object.
//!
//! Wing and vanish caps are intangible for their first 20 frames, and every
//! cap flickers out 300 frames after it spawns (cap_despawn).

use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St};

use super::common::{approach, find};

/// sm64.h.
const MARIO_NORMAL_CAP: u32 = 0x01;
const MARIO_VANISH_CAP: u32 = 0x02;
const MARIO_METAL_CAP: u32 = 0x04;
const MARIO_WING_CAP: u32 = 0x08;

pub struct Collect {
    id: ObjId,
    zone: (u16, u8),
    flag: u32,
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Option<Collect> {
        let flag = match o.bhv_name() {
            "bhvWingCap" => MARIO_WING_CAP,
            "bhvMetalCap" => MARIO_METAL_CAP,
            "bhvVanishCap" => MARIO_VANISH_CAP,
            "bhvNormalCap" => MARIO_NORMAL_CAP,
            _ => return None,
        };
        Some(Collect { id: o.id, zone: b.zone(), flag })
    }

    fn has(&self, s: &St) -> bool {
        s.mario_flags & self.flag != 0
    }
}

impl Objective for Collect {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Cap::Collect"
    }
    fn key(&self) -> String {
        format!("Cap {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.flag {
            // Wings reach the sky secrets, red coins and cap-stage stars.
            MARIO_WING_CAP => 7.0,
            // Without his cap Mario takes 1.5x damage (interaction.c, take_damage_from_interact_object).
            MARIO_NORMAL_CAP => 6.0,
            _ => 4.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Already wearing it: nothing to gain.
        !self.has(s) && find(s, self.zone, self.id).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // Already wearing it at the start: nothing to gain.
        if self.has(start) || s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.alive() && self.has(s) {
            return Eta::Done;
        }
        approach(s, find(s, self.zone, self.id).map(|o| o.pos))
    }
    fn success_value(&self) -> f64 {
        match self.flag {
            MARIO_WING_CAP => 5.0,
            MARIO_NORMAL_CAP => 4.0,
            _ => 3.0,
        }
    }
    fn needs_objects(&self) -> bool {
        // The cap moves (it pops out of its box and slides), and is steered to.
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        find(s, self.zone, self.id).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        find(s, self.zone, self.id).is_some()
    }
}
