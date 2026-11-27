//! `Chuckya::Throw(id)` / `Bobomb::Throw(id)`: pick up a grabbable enemy and
//! throw it, which kills it. `Object::Grab` covers the pick-up alone; this
//! carries on to the throw, and prefers starts already holding it.
//!
//! - Chuckya (`INTERACT_GRABBABLE` with `INT_SUBTYPE_GRABS_MARIO`) grabs
//!   Mario if he comes at it from the front, so it's taken from behind; once
//!   thrown it dies on hitting a wall, the floor or water and drops 5 coins
//!   (chuckya_act_2, chuckya.inc.c).
//! - A black Bob-omb explodes when thrown and lands (bobomb_act_launched),
//!   when its fuse runs out, or when it touches something; it drops a coin
//!   the first time unless bit 8 of its params says it already has
//!   (bobomb_spawn_coin), then respawns (`BOBOMB_ACT_EXPLODE` = 3;
//!   bobomb.inc.c).

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::{ahead_of_mario, find, holding};

const BOBOMB_ACT_LAUNCHED: i32 = 1;
const BOBOMB_ACT_EXPLODE: i32 = 3;
/// Chuckya's thrown act (chuckya_act_2): dies on hitting something.
const CHUCKYA_ACT_THROWN: i32 = 2;

/// Close to holding it: lining up and the B grab (~1/3 s; Chuckya's from
/// behind takes longer but shares it).
const GRAB_FRAMES: f64 = 10.0;
/// Holding to letting go: the throw animation (~1/6 s).
const THROW_FRAMES: f64 = 5.0;
/// Thrown to its death: the flight until it lands or hits a wall (~1/2 s).
const LAND_FRAMES: f64 = 15.0;

pub struct Throw {
    id: ObjId,
    zone: (u16, u8),
    kind: &'static str,
    weight: f64,
    from_behind: bool,
    bobomb: bool,
}

impl Throw {
    pub fn chuckya(b: &St, o: &Obj) -> Throw {
        Throw { id: o.id, zone: b.zone(), kind: "Chuckya::Throw", weight: 2.0, from_behind: true, bobomb: false }
    }

    pub fn bobomb(b: &St, o: &Obj) -> Throw {
        let coin = (o.beh_params >> 8) & 1 == 0;
        Throw { id: o.id, zone: b.zone(), kind: "Bobomb::Throw", weight: if coin { 1.0 } else { 0.2 }, from_behind: false, bobomb: true }
    }

    /// Flying from Mario's throw.
    fn thrown(&self, o: &Obj) -> bool {
        o.action == if self.bobomb { BOBOMB_ACT_LAUNCHED } else { CHUCKYA_ACT_THROWN } && o.held_state != 1
    }

    fn dead(&self, o: Option<&Obj>) -> bool {
        match o {
            None => true,
            Some(o) => self.bobomb && o.action == BOBOMB_ACT_EXPLODE,
        }
    }
}

impl Objective for Throw {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind, self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && !self.dead(find(s, self.id, false))
            && (!s.holding || holding(s, self.id, false))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 180 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        if self.dead(o) {
            return Eta::Done;
        }
        let o = o.expect("not dead, so present");
        Eta::Frames(if holding(s, self.id, false) {
            THROW_FRAMES + LAND_FRAMES
        } else if self.thrown(o) {
            LAND_FRAMES
        } else {
            frames_for(dist(s.pos, o.pos)) + GRAB_FRAMES + THROW_FRAMES + LAND_FRAMES
        })
    }
    fn success_value(&self) -> f64 {
        self.weight
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if holding(s, self.id, false) {
            return Some(ahead_of_mario(s, 400.0));
        }
        let o = find(s, self.id, false)?;
        let back = o.behind(150.0);
        if self.from_behind && dist(s.pos, back) > 120.0 && dist(s.pos, o.pos) < 600.0 {
            Some(back)
        } else {
            Some(o.pos)
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Grab when close (from behind, for Chuckya); throw once holding.
        if holding(s, self.id, false) {
            return B_BUTTON;
        }
        match find(s, self.id, false) {
            Some(o) if dist(s.pos, o.pos) < 180.0 && (!self.from_behind || dist(s.pos, o.behind(150.0)) < 150.0) => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !self.dead(find(s, self.id, false))
    }
}
