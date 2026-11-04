//! The blue coin switch and the blue coins it reveals (blue_coin.inc.c).
//!
//! `BlueCoinSwitch::Press`: the switch (bhvBlueCoinSwitch) only reacts to
//! Mario standing on it in ACT_GROUND_POUND_LAND; it then recedes
//! (action 1) and ticks (action 2) for 240 frames, during which its
//! bhvHiddenBlueCoins are tangible (action 2), each worth 5 coins.
//!
//! `HiddenBlueCoin::Collect`: one of those coins, attempted only from states
//! where it is out (`Coin::Collect` would also chase it while it is still
//! invisible and intangible, action 1).

use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::layout::ACT_FLAG_AIR;
use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, St, frames_for};

use super::common::{find, gone};

/// object_constants.h.
pub const BLUE_COIN_SWITCH_ACT_IDLE: i32 = 0;
pub const HIDDEN_BLUE_COIN_ACT_ACTIVE: i32 = 2;

/// sm64.h.
const ACT_GROUND_POUND: u32 = 0x008008A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080023C;

/// Horizontal distance within which Mario counts as over the switch (its
/// collision is a ~100-unit-wide square once scaled 3x).
const OVER: f32 = 90.0;

/// Next to the switch to the pound landing on it: a jump (~10), the
/// pound's 10-frame spin and the drop (act_ground_pound).
const POUND_FRAMES: f64 = 25.0;
/// Mid-pound over it to the landing (the rest of the spin, the drop).
const POUND_FALL_FRAMES: f64 = 8.0;

pub struct Press {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl Press {
    pub fn new(b: &St, o: &Obj) -> Press {
        Press { id: o.id, zone: b.zone(), pos: o.pos }
    }

    fn idle<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        find(s, self.zone, self.id).filter(|o| o.action == BLUE_COIN_SWITCH_ACT_IDLE)
    }
}

impl Objective for Press {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "BlueCoinSwitch::Press"
    }
    fn key(&self) -> String {
        format!("BlueSwitch {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        5.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.idle(s).is_some()
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // Receding or ticking, or already deleted after its timer.
        if s.alive() && (gone(s, self.zone, self.id) || find(s, self.zone, self.id).is_some_and(|o| o.action > BLUE_COIN_SWITCH_ACT_IDLE)) {
            return Eta::Done;
        }
        // Pounding while over it (v1's progress bonus), else the approach and
        // a whole pound.
        if matches!(s.action, ACT_GROUND_POUND | ACT_GROUND_POUND_LAND) && s.hdist(self.pos) < OVER * 1.5 {
            return Eta::Frames(POUND_FALL_FRAMES);
        }
        Eta::Frames(frames_for(s.dist(self.pos)) + POUND_FRAMES)
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.idle(s).map_or(self.pos, |o| o.pos))
    }
    fn steer_button(&self, s: &St) -> u16 {
        let Some(o) = self.idle(s) else { return 0 };
        if s.hdist(o.pos) > OVER {
            return 0;
        }
        // Over it: jump if standing, pound if in the air above it.
        if s.action_has(ACT_FLAG_AIR) {
            if s.pos[1] > o.pos[1] { Z_TRIG } else { 0 }
        } else {
            A_BUTTON
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub struct Collect {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Collect {
        Collect { id: o.id, zone: b.zone(), pos: o.pos }
    }
}

impl Objective for Collect {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "HiddenBlueCoin::Collect"
    }
    fn key(&self) -> String {
        format!("HiddenBlue {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        4.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        find(s, self.zone, self.id).is_some_and(|o| o.action == HIDDEN_BLUE_COIN_ACT_ACTIVE)
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        // The coins blink out 240 frames after the press.
        Some(GaBudget { population: 16, generations: 5, horizon: 120 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // Gone with the coin count up; it also vanishes when the switch's
        // timer runs out, without the coins.
        if gone(s, self.zone, self.id) {
            return if s.coins > start.coins { Eta::Done } else { Eta::Lost };
        }
        Eta::Frames(frames_for(s.dist(self.pos)))
    }
    fn success_value(&self) -> f64 {
        2.0
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
}
