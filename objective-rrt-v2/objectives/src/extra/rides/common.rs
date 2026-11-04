//! Shared helpers: identifying objects whose `ObjId` is degenerate, Mario
//! being "on" an object's collision box, distances.

use obrrt_core::layout::{ACT_FLAG_AIR, ACT_GROUP_MASK};
use obrrt_core::{Obj, ObjId, St, bhv_name};

/// sm64.h: ACT_GROUP_STATIONARY (0 << 6) and ACT_GROUP_MOVING (1 << 6).
const ACT_GROUP_STATIONARY: u32 = 0x000;
const ACT_GROUP_MOVING: u32 = 0x040;

/// How an objective finds its object again in other states.
///
/// `oHome` is only set by a behaviour's `SET_HOME()` (behavior_script.c);
/// `allocate_object` (spawn_object.c) zeroes every other field, so objects
/// without it all have home 0,0,0 and share one `ObjId` (the 4 chest bottoms,
/// the rings, Koopa shells, the WF tower platforms, the ferris wheel
/// platforms...). Those are found by position if they never move, by pool
/// slot (stable while loaded) if they do.
#[derive(Debug, Clone, Copy)]
pub enum Ident {
    Id(ObjId),
    At { bhv: u32, pos: [f32; 3] },
    Slot { bhv: u32, slot: u16 },
}

impl Ident {
    pub fn of(o: &Obj, moving: bool) -> Ident {
        if o.id.home != [0, 0, 0] {
            Ident::Id(o.id)
        } else if moving {
            Ident::Slot { bhv: o.bhv, slot: o.slot }
        } else {
            Ident::At { bhv: o.bhv, pos: o.pos }
        }
    }

    pub fn find<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        match *self {
            Ident::Id(id) => s.find(id),
            Ident::At { bhv, pos } => {
                s.objects().iter().find(|o| o.bhv == bhv && hdist(o.pos, pos) < 40.0 && (o.pos[1] - pos[1]).abs() < 200.0)
            }
            Ident::Slot { bhv, slot } => s.objects().iter().find(|o| o.slot == slot && o.bhv == bhv),
        }
    }

    pub fn key(&self) -> String {
        match *self {
            Ident::Id(id) => id.to_string(),
            Ident::At { bhv, pos } => format!(
                "{}@{},{},{}",
                bhv_name(bhv).unwrap_or("?"),
                pos[0].round() as i32,
                pos[1].round() as i32,
                pos[2].round() as i32
            ),
            Ident::Slot { bhv, slot } => format!("{}#{}", bhv_name(bhv).unwrap_or("?"), slot),
        }
    }
}

pub fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Mario is standing or walking on something (not airborne, swimming,
/// hanging, on a pole or in a cutscene).
pub fn grounded(s: &St) -> bool {
    let g = s.action & ACT_GROUP_MASK;
    (g == ACT_GROUP_STATIONARY || g == ACT_GROUP_MOVING) && !s.action_has(ACT_FLAG_AIR)
}

/// `oFaceAngleYaw` in radians.
fn yaw_rad(o: &Obj) -> f32 {
    o.yaw as f32 * std::f32::consts::TAU / 65536.0
}

/// A world point's (x, z) in the object's frame. SM64 yaw maps local +z to
/// world (sin, cos) and local +x to (cos, -sin); this is the inverse.
pub fn to_local(o: &Obj, p: [f32; 3]) -> (f32, f32) {
    let (sn, cs) = yaw_rad(o).sin_cos();
    let (dx, dz) = (p[0] - o.pos[0], p[2] - o.pos[2]);
    (dx * cs - dz * sn, dx * sn + dz * cs)
}

pub fn to_world(o: &Obj, lx: f32, ly: f32, lz: f32) -> [f32; 3] {
    let (sn, cs) = yaw_rad(o).sin_cos();
    [o.pos[0] + lx * cs + lz * sn, o.pos[1] + ly, o.pos[2] - lx * sn + lz * cs]
}

/// An object's collision box in its own frame (min, max per axis), from its
/// collision.inc.c vertices (before yaw; pitch and roll ignored).
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub x: [f32; 2],
    pub y: [f32; 2],
    pub z: [f32; 2],
}

impl Shape {
    pub const fn new(x: [f32; 2], y: [f32; 2], z: [f32; 2]) -> Shape {
        Shape { x, y, z }
    }

    /// Whether `p` is within the box's footprint (grown by `mxz`) and its
    /// height range (grown by `my`).
    pub fn contains(&self, o: &Obj, p: [f32; 3], mxz: f32, my: f32) -> bool {
        let (lx, lz) = to_local(o, p);
        let dy = p[1] - o.pos[1];
        lx >= self.x[0] - mxz
            && lx <= self.x[1] + mxz
            && lz >= self.z[0] - mxz
            && lz <= self.z[1] + mxz
            && dy >= self.y[0] - my
            && dy <= self.y[1] + my
    }

    /// The middle of the box's top, in the world.
    pub fn top(&self, o: &Obj) -> [f32; 3] {
        to_world(o, (self.x[0] + self.x[1]) / 2.0, self.y[1], (self.z[0] + self.z[1]) / 2.0)
    }
}
