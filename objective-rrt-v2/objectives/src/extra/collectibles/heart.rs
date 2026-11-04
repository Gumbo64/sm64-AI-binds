//! `Heart::Spin(id)`: heal at a spinning heart (bhvRecoveryHeart,
//! recovery_heart.inc.c). It has no interact type: while Mario touches it it
//! spins at 200 * forwardVel + 1000 per frame, and every full turn adds 4 to
//! his heal counter (one wedge). Running through it fast heals fastest.

use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, frames_for};

use super::common::find;

/// A wedge gained this close to the heart counts as its (v1).
const HEAL_RANGE: f32 = 600.0;
/// Mario counts as touching it within this distance (v1's "close", 120).
const TOUCH: f32 = 120.0;
/// One full turn of the heart (a 16-bit angle) heals a wedge.
const TURN: f64 = 65536.0;
/// Spin per frame while touched at the reference pace: 200 * 30 + 1000, so
/// a turn is ~9 frames running through it at 30.
const SPIN_AT_PACE: f64 = 200.0 * 30.0 + 1000.0;

/// Full health is 8 wedges (0x880).
const FULL: i32 = 8;

pub struct Spin {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl Spin {
    pub fn new(b: &St, o: &Obj) -> Spin {
        Spin { id: o.id, zone: b.zone(), pos: o.pos }
    }
}

impl Objective for Spin {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Heart::Spin"
    }
    fn key(&self) -> String {
        format!("Heart {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // Worth more the more hurt Mario is, but kept small: healing is
        // rarely what a run needs (0.5 per wedge in v1 swamped the pool).
        0.1 * (FULL - b.wedges()).max(0) as f64
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.alive() && s.wedges() < FULL && find(s, self.zone, self.id).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 12, generations: 4, horizon: 150 })
    }
    /// Approach, then the rest of a turn: `mem.f[0]` adds up the heart's
    /// spin (200 * |fv| + 1000 per frame touched), so running through it
    /// fast lowers the ETA the way it heals.
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if s.wedges() > start.wedges() && s.dist(self.pos) < HEAL_RANGE {
            return Eta::Done;
        }
        let d = s.dist(self.pos);
        if d < TOUCH {
            mem.f[0] += 200.0 * s.forward_vel.abs() as f64 + 1000.0;
        }
        // Clamped, not wrapped: health rises a few frames after the turn.
        let turn_left = (TURN - mem.f[0]).max(0.0) / SPIN_AT_PACE;
        Eta::Frames(frames_for((d - TOUCH).max(0.0)) + turn_left)
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.objects.as_ref().is_none_or(|_| s.find(self.id).is_some())
    }
}
