//! Helpers shared by the collectible objectives.

use obrrt_core::{Eta, Obj, ObjId, St, dist, frames_for};

/// The object in `s`, if `s` is in `zone` and has it loaded.
pub fn find(s: &St, zone: (u16, u8), id: ObjId) -> Option<&Obj> {
    if s.zone() == zone { s.find(id) } else { None }
}

/// Whether the object was collected (or otherwise removed) in `s`: Mario is
/// alive in the same zone and the object is no longer in the pool.
pub fn gone(s: &St, zone: (u16, u8), id: ObjId) -> bool {
    s.zone() == zone && s.alive() && s.objects.is_some() && s.find(id).is_none()
}

/// Frames to the point `p` the objective tracks in `s`; `Lost` without one
/// (its object gone or unloaded without success).
pub fn approach(s: &St, p: Option<[f32; 3]>) -> Eta {
    p.map_or(Eta::Lost, |p| Eta::Frames(frames_for(dist(s.pos, p))))
}
