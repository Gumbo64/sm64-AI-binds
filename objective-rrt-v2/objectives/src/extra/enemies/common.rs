//! Helpers shared by the enemy objectives.

use obrrt_core::layout::ACT_FLAG_AIR;
use obrrt_core::{Obj, ObjId, St};

/// The object `id` in `s`. `by_bhv` finds the first object with its
/// behaviour instead, for enemies that rewrite their own `oHome` (Klepto
/// steers by moving its home between targets, klepto.inc.c), so their
/// [`ObjId`] isn't stable.
pub fn find(s: &St, id: ObjId, by_bhv: bool) -> Option<&Obj> {
    if by_bhv { s.objects().iter().find(|o| o.bhv == id.bhv) } else { s.find(id) }
}

/// Whether Mario holds the object `id`.
pub fn holding(s: &St, id: ObjId, by_bhv: bool) -> bool {
    s.held().is_some_and(|h| if by_bhv { h.bhv == id.bhv } else { h.id == id })
}

pub fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// A point `d` in front of the object (along its facing yaw).
pub fn front(o: &Obj, d: f32) -> [f32; 3] {
    o.behind(-d)
}

/// A point `d` in front of Mario.
pub fn ahead_of_mario(s: &St, d: f32) -> [f32; 3] {
    let a = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
    [s.pos[0] + d * a.sin(), s.pos[1], s.pos[2] + d * a.cos()]
}

pub fn in_air(s: &St) -> bool {
    s.action & ACT_FLAG_AIR != 0
}

/// The 2nd behaviour parameter byte (`oBehParams2ndByte`).
pub fn bp2(o: &Obj) -> u32 {
    (o.beh_params >> 16) & 0xFF
}

/// The 1st behaviour parameter byte: the star index for star-givers
/// (spawn_star copies the spawner's params, spawn_star.inc.c).
pub fn bp1(o: &Obj) -> u8 {
    (o.beh_params >> 24) as u8
}

/// The next point on a circle of radius `r` around `c`, a little ahead of
/// Mario's current angle (counter-clockwise), at Mario's height (so the
/// controller doesn't jump).
pub fn circle_point(s: &St, c: [f32; 3], r: f32) -> [f32; 3] {
    let a = (s.pos[0] - c[0]).atan2(s.pos[2] - c[2]) + 0.6;
    [c[0] + r * a.sin(), s.pos[1], c[2] + r * a.cos()]
}

/// Mario's flags (include/sm64.h).
pub const MARIO_CAP_ON_HEAD: u32 = 0x10;
pub const MARIO_CAP_IN_HAND: u32 = 0x20;

/// Whether Mario's cap is neither on his head nor in his hand (Klepto,
/// Ukiki, Mr. Blizzard or the wind has it).
pub fn capless(s: &St) -> bool {
    s.mario_flags & (MARIO_CAP_ON_HEAD | MARIO_CAP_IN_HAND) == 0
}
