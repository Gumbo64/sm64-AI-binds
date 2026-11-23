//! What a policy sees of the world around Mario: floor probes, wall rays, edges,
//! ledges, headroom, a ballistic landing prediction, nearby objects and Mario's
//! own flags. Fills [`crate::types::Sense`] once per frame from the emulator's
//! memory and the level's static triangles (indexed in a 2D grid, so a query
//! touches a handful of triangles instead of walking the game's linked lists).
//! For anything not in that hot struct there is [`reader`]: any game field by
//! name (`gMarioState.health`, `obj[3].oAction`, `nearest(bhv=bhvBobomb,0).oPosX`).
//!
//! Conventions (game units, game yaw: 0 = +z, 0x4000 = +x, increasing yaw turns
//! left seen from above):
//! - `Sense::ahead[i]`: floor under the point `d` ahead on Mario's heading, found
//!   like the game's `find_floor` called with Mario's current y (highest static or
//!   object floor at most 78 above it).
//! - `Sense::wall[0..4]`: distance Mario's centre can travel along heading,
//!   heading+0x4000 (left), heading-0x4000 (right) and heading+0x8000 (back)
//!   before a wall collides (the game's test: centre within radius 50 of the wall's
//!   plane and inside its projected triangle), minimum over heights 30, 60, 150
//!   above his feet. `f32::INFINITY` = none within [`WALL_RANGE`].
//! - Static geometry only for walls and ceilings (object walls/ceilings are not
//!   seen); the wall rays also see object collision walls when [`SenseOpts::dyn_walls`] and
//!   report the nearest of both (`SenseCtx::blocks` says which: static, or an object slot).
//!   Floors include object (dynamic) floors when [`SenseOpts::dyn_floors`].
//! - Landing prediction ignores walls and ceilings and holds the facing; stick and
//!   A are the [`AirModel`]'s.

pub mod behaviours;
pub mod reader;

use std::collections::HashMap;

use obrrt_core::geo::{self, Tri};
use obrrt_core::layout::*;
use obrrt_core::St;

use crate::types::{FloorProbe, Landing, ObjProbe, Sense};

pub const WALL_RANGE: f32 = 1500.0;
/// Mario's wall collision radius in the air and on the ground (upper probe).
pub const WALL_R: f32 = 50.0;
/// Heights above his feet at which walls are probed.
pub const WALL_H: [f32; 3] = [30.0, 60.0, 150.0];
pub const AHEAD_D: [f32; 6] = [24.0, 48.0, 96.0, 192.0, 384.0, 768.0];
pub const VEL_D: [f32; 2] = [96.0, 384.0];
/// Edge scan: step and range.
const EDGE_STEP: f32 = 16.0;
const EDGE_RANGE: f32 = 768.0;

// Object categories (ObjProbe::cat).
pub const CAT_NONE: u8 = 0;
pub const CAT_ENEMY: u8 = 1;
pub const CAT_COIN: u8 = 2;
pub const CAT_POLE: u8 = 3;
pub const CAT_STAR: u8 = 4;
pub const CAT_PLATFORM: u8 = 5;
pub const CAT_BOMB: u8 = 6;
pub const CAT_BREAKABLE: u8 = 7;
pub const CAT_OTHER: u8 = 8;
pub const N_CAT: usize = 9;

pub const CAT_NAMES: [&str; N_CAT] = ["none", "enemy", "coin", "pole", "star", "platform", "bomb", "breakable", "other"];

/// What touching an object of a category tends to do (a hint for policies).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hint {
    Nothing,
    /// Collect / grab (coin, star, pole).
    Grab,
    /// Hurts unless attacked (enemy, bomb).
    Hurt,
    /// Stand on it.
    Ride,
    /// Break it by attacking.
    Break,
}

pub fn hint(cat: u8) -> Hint {
    match cat {
        CAT_COIN | CAT_STAR | CAT_POLE => Hint::Grab,
        CAT_ENEMY | CAT_BOMB => Hint::Hurt,
        CAT_PLATFORM => Hint::Ride,
        CAT_BREAKABLE => Hint::Break,
        _ => Hint::Nothing,
    }
}

/// An axis-aligned box (a forbidden or target zone, e.g. Lakitu's trigger box).
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub lo: [f32; 3],
    pub hi: [f32; 3],
}

impl Region {
    pub fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|i| p[i] > self.lo[i] && p[i] < self.hi[i])
    }
}

/// What to compute (each part costs time).
#[derive(Clone, Copy, Debug)]
pub struct SenseOpts {
    pub floors: bool,
    pub walls: bool,
    pub edges: bool,
    pub landing: bool,
    pub objs: bool,
    /// Include object (moving platform) floors in floor probes.
    pub dyn_floors: bool,
    /// Include object collision walls (doors, gates, blocks, platforms' sides) in the wall rays.
    pub dyn_walls: bool,
}

impl Default for SenseOpts {
    fn default() -> Self {
        SenseOpts { floors: true, walls: true, edges: true, landing: true, objs: true, dyn_floors: true, dyn_walls: true }
    }
}

/// What the stick and A do during a predicted fall.
#[derive(Clone, Copy, Debug)]
pub struct AirModel {
    /// Forward component of `intendedMag/32 * cos(dyaw)`: 1 = full stick along the
    /// facing, 0 = neutral, -1 = full back.
    pub fwd: f32,
    /// Sideways: `intendedMag/32 * sin(dyaw)`.
    pub side: f32,
    /// A held (jump height control and the wing cap's glide).
    pub a_down: bool,
}

impl Default for AirModel {
    fn default() -> Self {
        AirModel { fwd: 1.0, side: 0.0, a_down: true }
    }
}

impl AirModel {
    /// The model of a pad: the game's `intendedMag`/`intendedYaw` from the stick
    /// and the camera yaw, relative to Mario's facing.
    pub fn from_pad(st: &St, pad: obrrt_core::GamePad) -> AirModel {
        let (sx, sy) = (dead(pad.stick_x as f32), dead(pad.stick_y as f32));
        let mag = (sx * sx + sy * sy).sqrt().min(64.0);
        let a_down = pad.button & obrrt_core::buttons::A_BUTTON != 0;
        if mag <= 0.0 {
            return AirModel { fwd: 0.0, side: 0.0, a_down };
        }
        let m = ((mag / 64.0) * (mag / 64.0)) * 64.0 / 2.0;
        // intendedYaw = atan2s(-stickY, stickX) + camera yaw, atan2s(a, b) being the angle from the a axis towards b.
        let iyaw = sx.atan2(-sy) * 65536.0 / std::f32::consts::TAU + st.cam_yaw as f32;
        let d = (iyaw - st.yaw as f32) * std::f32::consts::TAU / 65536.0;
        AirModel { fwd: d.cos() * m / 32.0, side: d.sin() * m / 32.0, a_down }
    }
}

/// The controller's dead zone (|raw| < 8 -> 0, then 6 less in magnitude).
fn dead(v: f32) -> f32 {
    if v > -8.0 && v < 8.0 {
        0.0
    } else if v > 0.0 {
        v - 6.0
    } else {
        v + 6.0
    }
}

/// Decoded Mario state flags (everything is in [`St`] or its action bits).
#[derive(Clone, Copy, Debug, Default)]
pub struct MarioFlags {
    pub normal_cap: bool,
    pub vanish_cap: bool,
    pub metal_cap: bool,
    pub wing_cap: bool,
    pub holding: bool,
    pub kicking: bool,
    pub punching: bool,
    pub airborne: bool,
    pub moving: bool,
    pub stationary: bool,
    pub swimming: bool,
    pub invulnerable: bool,
    pub intangible: bool,
    /// Water above his feet (0 = out of water).
    pub water_depth: f32,
    pub quicksand: bool,
    pub wedges: i32,
    pub health: i32,
    /// Height of his feet over the floor.
    pub height_above_floor: f32,
}

impl MarioFlags {
    pub fn of(st: &St) -> MarioFlags {
        let (a, f) = (st.action, st.mario_flags);
        MarioFlags {
            normal_cap: f & 0x1 != 0,
            vanish_cap: f & 0x2 != 0,
            metal_cap: f & 0x4 != 0,
            wing_cap: f & 0x8 != 0,
            holding: st.holding,
            kicking: f & 0x0020_0000 != 0,
            punching: f & 0x0010_0000 != 0,
            airborne: a & ACT_FLAG_AIR != 0,
            moving: a & ACT_FLAG_MOVING != 0,
            stationary: a & ACT_FLAG_STATIONARY != 0,
            swimming: a & ACT_FLAG_SWIMMING != 0,
            invulnerable: a & ACT_FLAG_INVULNERABLE != 0,
            intangible: a & ACT_FLAG_INTANGIBLE != 0,
            water_depth: if st.water_level > -10000 { (st.water_level as f32 - st.pos[1]).max(0.0) } else { 0.0 },
            quicksand: matches!(st.floor_type, 0x21 | 0x22 | 0x23 | 0x24 | 0x27 | 0x28) || a == 0x0002_020D,
            wedges: st.wedges(),
            health: st.health as i32,
            height_above_floor: st.pos[1] - st.floor_y,
        }
    }
}

const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_SLIDE_KICK: u32 = 0x0380_08AA;
const ACT_SHOT_FROM_CANNON: u32 = 0x1080_08A9;
const ACT_FLAG_CONTROL_JUMP_HEIGHT: u32 = 1 << 25;

/// Slippery class of a surface type (0 very slippery, 1 slippery, 2 default, 3 not).
pub fn floor_class(ty: i16) -> u8 {
    match ty {
        0x15 | 0x37 | 0x7A => 3,
        0x14 | 0x2A | 0x35 | 0x79 => 1,
        0x13 | 0x2E | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => 0,
        _ => 2,
    }
}

// ---------------------------------------------------------------------------
// Triangle grid.

const CELL: f32 = 256.0;
const HALF: i32 = 32; // 64 x 64 cells over +-8192
const SIDE: i32 = 2 * HALF;

struct Tr {
    v: [[i32; 3]; 3],
    n: [f32; 3],
    oo: f32,
    kind: i16,
    ymin: f32,
    ymax: f32,
    xproj: bool,
}

struct Grid {
    tris: Vec<Tr>,
    cells: Vec<Vec<u32>>,
    stamp: Vec<u32>,
    epoch: u32,
}

fn cell_ix(x: f32, z: f32) -> Option<usize> {
    let (cx, cz) = ((x / CELL).floor() as i32 + HALF, (z / CELL).floor() as i32 + HALF);
    ((0..SIDE).contains(&cx) && (0..SIDE).contains(&cz)).then(|| (cz * SIDE + cx) as usize)
}

impl Grid {
    fn new(src: &[Tri]) -> Grid {
        let mut tris = Vec::with_capacity(src.len());
        let mut cells = vec![Vec::new(); (SIDE * SIDE) as usize];
        for t in src {
            let v = t.v.map(|p| [p[0] as i32, p[1] as i32, p[2] as i32]);
            let oo = -(t.n[0] * v[0][0] as f32 + t.n[1] * v[0][1] as f32 + t.n[2] * v[0][2] as f32);
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for p in &v {
                lo = lo.min(p[1]);
                hi = hi.max(p[1]);
            }
            let id = tris.len() as u32;
            let (x0, x1) = (v.iter().map(|p| p[0]).min().unwrap() as f32, v.iter().map(|p| p[0]).max().unwrap() as f32);
            let (z0, z1) = (v.iter().map(|p| p[2]).min().unwrap() as f32, v.iter().map(|p| p[2]).max().unwrap() as f32);
            tris.push(Tr { v, n: t.n, oo, kind: t.kind, ymin: lo as f32 - 5.0, ymax: hi as f32 + 5.0, xproj: t.n[0].abs() > 0.707 });
            let (c0x, c1x) = (((x0 / CELL).floor() as i32 + HALF).max(0), ((x1 / CELL).floor() as i32 + HALF).min(SIDE - 1));
            let (c0z, c1z) = (((z0 / CELL).floor() as i32 + HALF).max(0), ((z1 / CELL).floor() as i32 + HALF).min(SIDE - 1));
            for cz in c0z..=c1z {
                for cx in c0x..=c1x {
                    cells[(cz * SIDE + cx) as usize].push(id);
                }
            }
        }
        let n = tris.len();
        Grid { tris, cells, stamp: vec![0; n], epoch: 0 }
    }
}

/// The game's lateral test for a floor (`floor`) or ceiling triangle at integer (x, z).
fn inside_xz(t: &Tr, x: i32, z: i32, floor: bool) -> bool {
    let [a, b, c] = t.v;
    let side = |p: [i32; 3], q: [i32; 3]| (p[2] - z).wrapping_mul(q[0] - p[0]).wrapping_sub((p[0] - x).wrapping_mul(q[2] - p[2]));
    let out = |v: i32| if floor { v < 0 } else { v > 0 };
    !(out(side(a, b)) || out(side(b, c)) || out(side(c, a)))
}

fn s16(v: f32) -> i32 {
    (v as i32) as i16 as i32
}

#[derive(Clone, Copy)]
struct SurfHit {
    y: f32,
    kind: i16,
    ny: f32,
}

impl Grid {
    /// find_floor on the static partition: the highest floor at most y + 78 over (x, z).
    fn floor(&self, x: f32, y: f32, z: f32) -> Option<SurfHit> {
        let (xi, zi) = (s16(x), s16(z));
        let cell = cell_ix(xi as f32, zi as f32)?;
        let mut yref = s16(y) as f32;
        for pass in 0..2 {
            let mut best: Option<SurfHit> = None;
            for &id in &self.cells[cell] {
                let t = &self.tris[id as usize];
                if t.kind == 0x72 || t.n[1] == 0.0 {
                    continue;
                }
                if !inside_xz(t, xi, zi, true) {
                    continue;
                }
                let h = -(xi as f32 * t.n[0] + t.n[2] * zi as f32 + t.oo) / t.n[1];
                if h > yref + 78.0 {
                    continue;
                }
                if best.is_none_or(|b| h > b.y) {
                    best = Some(SurfHit { y: h, kind: t.kind, ny: t.n[1] });
                }
            }
            match best {
                // find_floor searches again under an intangible floor and takes what it finds.
                Some(b) if b.kind == 0x12 && pass == 0 => yref = b.y - 200.0,
                other => return other,
            }
        }
        None
    }

    /// find_ceil: the lowest ceiling at least y - 78 over (x, z).
    fn ceil(&self, x: f32, y: f32, z: f32) -> Option<f32> {
        let (xi, zi) = (s16(x), s16(z));
        let cell = cell_ix(xi as f32, zi as f32)?;
        let yref = s16(y) as f32;
        let mut best: Option<f32> = None;
        for &id in &self.cells[cell] {
            let t = &self.tris[id as usize];
            if t.n[1] == 0.0 || !inside_xz(t, xi, zi, false) {
                continue;
            }
            let h = -(xi as f32 * t.n[0] + t.n[2] * zi as f32 + t.oo) / t.n[1];
            if h < yref - 78.0 {
                continue;
            }
            if best.is_none_or(|b| h < b) {
                best = Some(h);
            }
        }
        best
    }

    /// Nearest contact of a horizontal ray with the wall triangles (see module doc).
    /// `ids` are the candidate triangles of the ray.
    fn ray(&self, ids: &[u32], o: [f32; 3], d: [f32; 2], len: f32, r: f32) -> Option<(f32, usize)> {
        let mut best: Option<(f32, usize)> = None;
        for &id in ids {
            if let Some(ta) = ray_tri(&self.tris[id as usize], o, d, len, r)
                && best.is_none_or(|b| ta < b.0)
            {
                best = Some((ta, id as usize));
            }
        }
        best
    }

    /// Unique triangle ids along a horizontal segment.
    fn along(&mut self, o: [f32; 2], d: [f32; 2], len: f32, out: &mut Vec<u32>) {
        out.clear();
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
        let steps = (len / (CELL * 0.5)).ceil() as i32 + 1;
        let mut last = usize::MAX;
        for i in 0..=steps {
            let t = (i as f32 * CELL * 0.5).min(len);
            let Some(c) = cell_ix(o[0] + d[0] * t, o[1] + d[1] * t) else { continue };
            if c == last {
                continue;
            }
            last = c;
            for &id in &self.cells[c] {
                if self.stamp[id as usize] != self.epoch {
                    self.stamp[id as usize] = self.epoch;
                    out.push(id);
                }
            }
        }
    }
}

/// Distance a horizontal ray from `o` along `d` (unit, xz) can travel before the game's wall
/// test fires for `t`: centre within `r` of the wall's plane and inside its projected triangle.
fn ray_tri(t: &Tr, o: [f32; 3], d: [f32; 2], len: f32, r: f32) -> Option<f32> {
    if o[1] < t.ymin || o[1] > t.ymax {
        return None;
    }
    let nd = t.n[0] * d[0] + t.n[2] * d[1];
    let o0 = t.n[0] * o[0] + t.n[1] * o[1] + t.n[2] * o[2] + t.oo;
    let (mut ta, mut tb) = (0.0f32, len);
    if nd.abs() < 1e-6 {
        if o0.abs() > r {
            return None;
        }
    } else {
        let (t1, t2) = ((-r - o0) / nd, (r - o0) / nd);
        ta = ta.max(t1.min(t2));
        tb = tb.min(t1.max(t2));
        if ta > tb {
            return None;
        }
    }
    // The triangle's slice at height y, in the projection's u axis.
    let (u0, du) = if t.xproj { (o[2], d[1]) } else { (o[0], d[0]) };
    let uv = |p: [i32; 3]| if t.xproj { p[2] as f32 } else { p[0] as f32 };
    let (mut ua, mut ub) = (f32::INFINITY, f32::NEG_INFINITY);
    for k in 0..3 {
        let (p, q) = (t.v[k], t.v[(k + 1) % 3]);
        let (py, qy) = (p[1] as f32, q[1] as f32);
        let y = o[1];
        if (py - y) * (qy - y) <= 0.0 {
            if py == qy {
                ua = ua.min(uv(p)).min(uv(q));
                ub = ub.max(uv(p)).max(uv(q));
            } else {
                let u = uv(p) + (y - py) * (uv(q) - uv(p)) / (qy - py);
                ua = ua.min(u);
                ub = ub.max(u);
            }
        }
    }
    if ua > ub {
        return None;
    }
    if du.abs() < 1e-6 {
        if u0 < ua || u0 > ub {
            return None;
        }
    } else {
        let (t1, t2) = ((ua - u0) / du, (ub - u0) / du);
        ta = ta.max(t1.min(t2));
        tb = tb.min(t1.max(t2));
        if ta > tb {
            return None;
        }
    }
    Some(ta)
}

struct ZoneGeo {
    floors: Grid,
    ceils: Grid,
    walls: Grid,
    max_floor: f32,
}

// ---------------------------------------------------------------------------
// Context (per thread; holds the per-zone grids and caches).

/// What blocks Mario in a direction (the nearest of the static walls and the objects' collision walls).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockSrc {
    None,
    Static,
    /// An object's collision model; its pool slot.
    Object(u16),
}

#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub dist: f32,
    pub src: BlockSrc,
    pub normal_yaw: i16,
    /// For an object: its collision model is already in the game's surface partition (Mario is within
    /// its `oCollisionDistance`); otherwise the block is predicted from the model data and the object's pose.
    pub loaded: bool,
}

impl Default for Block {
    fn default() -> Self {
        Block { dist: f32::INFINITY, src: BlockSrc::None, normal_yaw: 0, loaded: true }
    }
}

#[derive(Default)]
pub struct SenseCtx {
    zones: HashMap<(u16, u8), ZoneGeo>,
    cat: HashMap<u32, u8>,
    scratch: Vec<u32>,
    /// What the last [`SenseCtx::read`] found blocking in the four wall directions
    /// (forward, +0x4000, -0x4000, back); `Sense::wall` holds the distances.
    pub blocks: [Block; 4],
    models: HashMap<usize, Option<std::rc::Rc<Model>>>,
    /// Wall triangles of nearby objects' collision models in world space this frame (slot, loaded).
    pending: Vec<(Tr, u16, bool)>,
}

/// An object collision model (`o->collisionData`): vertices and triangles in object space.
struct Model {
    verts: Vec<[f32; 3]>,
    tris: Vec<[u16; 3]>,
    radius: f32,
}

const COL_TRI_STOP: i16 = 0x41;

fn surface_has_force(t: i16) -> bool {
    matches!(t as u32, 0x0004 | 0x000E | 0x0024 | 0x0025 | 0x0027 | 0x002C | 0x002D)
}

/// Parses a collision model in the game's format (`COL_INIT`, vertex block, triangle lists up to
/// `COL_TRI_STOP`; surface_load.c `load_object_collision_model`).
fn parse_model(m: &[u8], ptr: usize) -> Option<Model> {
    let mut a = ptr;
    let rd = |a: usize| rd_i16(m, a);
    if rd(a)? != 0x40 {
        return None;
    }
    a += 2;
    let nv = rd(a)? as usize;
    a += 2;
    if nv == 0 || nv > 600 {
        return None;
    }
    let mut verts = Vec::with_capacity(nv);
    let mut radius = 0.0f32;
    for _ in 0..nv {
        let v = [rd(a)? as f32, rd(a + 2)? as f32, rd(a + 4)? as f32];
        radius = radius.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
        verts.push(v);
        a += 6;
    }
    let mut tris = Vec::new();
    loop {
        let ty = rd(a)?;
        if ty == COL_TRI_STOP {
            break;
        }
        a += 2;
        let n = rd(a)? as usize;
        a += 2;
        if n > 1200 {
            return None;
        }
        let w = if surface_has_force(ty) { 4 } else { 3 };
        for _ in 0..n {
            let t = [rd(a)? as u16, rd(a + 2)? as u16, rd(a + 4)? as u16];
            if t.iter().any(|&i| i as usize >= nv) {
                return None;
            }
            tris.push(t);
            a += 2 * w;
        }
        if tris.len() > 3000 {
            return None;
        }
    }
    Some(Model { verts, tris, radius })
}

/// mtxf_rotate_zxy_and_translate (math_util.c).
fn zxy_matrix(pos: [f32; 3], rot: [i16; 3]) -> [[f32; 3]; 4] {
    let f = |a: i16| a as f32 * std::f32::consts::TAU / 65536.0;
    let (sx, cx, sy, cy, sz, cz) = (f(rot[0]).sin(), f(rot[0]).cos(), f(rot[1]).sin(), f(rot[1]).cos(), f(rot[2]).sin(), f(rot[2]).cos());
    [
        [cy * cz + sx * sy * sz, cx * sz, -sy * cz + sx * cy * sz],
        [-cy * sz + sx * sy * cz, cx * cz, sy * sz + sx * cy * cz],
        [cx * sy, -sx, cx * cy],
        pos,
    ]
}

/// The wall triangles of `model` posed at `pos` with face angles `rot` (the game's classification:
/// normal.y within +-0.01 is a wall).
fn model_walls(model: &Model, pos: [f32; 3], rot: [i16; 3], out: &mut Vec<(Tr, u16, bool)>, slot: u16, loaded: bool) {
    let mx = zxy_matrix(pos, rot);
    let w: Vec<[i32; 3]> = model
        .verts
        .iter()
        .map(|v| {
            [
                s16(v[0] * mx[0][0] + v[1] * mx[1][0] + v[2] * mx[2][0] + mx[3][0]),
                s16(v[0] * mx[0][1] + v[1] * mx[1][1] + v[2] * mx[2][1] + mx[3][1]),
                s16(v[0] * mx[0][2] + v[1] * mx[1][2] + v[2] * mx[2][2] + mx[3][2]),
            ]
        })
        .collect();
    for t in &model.tris {
        let (a, b, c) = (w[t[0] as usize], w[t[1] as usize], w[t[2] as usize]);
        let (x1, y1, z1, x2, y2, z2, x3, y3, z3) = (a[0] as f32, a[1] as f32, a[2] as f32, b[0] as f32, b[1] as f32, b[2] as f32, c[0] as f32, c[1] as f32, c[2] as f32);
        let mut n = [(y2 - y1) * (z3 - z2) - (z2 - z1) * (y3 - y2), (z2 - z1) * (x3 - x2) - (x2 - x1) * (z3 - z2), (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)];
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if mag < 1e-4 {
            continue;
        }
        n = [n[0] / mag, n[1] / mag, n[2] / mag];
        if n[1] > 0.01 || n[1] < -0.01 {
            continue;
        }
        let oo = -(n[0] * x1 + n[1] * y1 + n[2] * z1);
        let (lo, hi) = (a[1].min(b[1]).min(c[1]) as f32 - 5.0, a[1].max(b[1]).max(c[1]) as f32 + 5.0);
        out.push((Tr { v: [a, b, c], n, oo, kind: 0, ymin: lo, ymax: hi, xproj: n[0].abs() > 0.707 }, slot, loaded));
    }
}

/// Parses a `struct Surface` at `s` into a wall triangle, with its owner object's address.
fn read_tr(m: &[u8], s: usize) -> Option<(Tr, usize)> {
    let v = |k: usize| -> Option<[i32; 3]> { Some([rd_i16(m, s + 0x0A + 6 * k)? as i32, rd_i16(m, s + 0x0A + 6 * k + 2)? as i32, rd_i16(m, s + 0x0A + 6 * k + 4)? as i32]) };
    let n = [rd_f32(m, s + SURFACE_normal)?, rd_f32(m, s + SURFACE_normal_y)?, rd_f32(m, s + SURFACE_normal + 8)?];
    let oo = rd_f32(m, s + SURFACE_originOffset)?;
    let kind = rd_i16(m, s + SURFACE_type)?;
    let (lo, hi) = (rd_i16(m, s + 6)? as f32, rd_i16(m, s + 8)? as f32);
    let obj = rd_u32(m, s + SURFACE_object)? as usize;
    Some((Tr { v: [v(0)?, v(1)?, v(2)?], n, oo, kind, ymin: lo, ymax: hi, xproj: n[0].abs() > 0.707 }, obj))
}

/// The object (dynamic) walls hit by a horizontal ray at the heights `hs`: nearest distance, owner slot, normal.
fn dyn_wall_ray(m: &[u8], o: [f32; 3], hs: &[f32], d: [f32; 2], len: f32, r: f32) -> Option<(f32, u16, [f32; 3])> {
    let (b, c) = (LEVEL_BOUNDARY_MAX as i32, CELL_SIZE as i32);
    let mut seen: [usize; 24] = [0; 24];
    let mut nseen = 0usize;
    let mut best: Option<(f32, u16, [f32; 3])> = None;
    let steps = (len / (c as f32 * 0.25)).ceil() as i32;
    let mut last = usize::MAX;
    for i in 0..=steps {
        let t = (i as f32 * c as f32 * 0.25).min(len);
        let (x, z) = (s16(o[0] + d[0] * t), s16(o[2] + d[1] * t));
        if x <= -b || x >= b || z <= -b || z >= b {
            continue;
        }
        let cell = ((((z + b) / c) & (NUM_CELLS as i32 - 1)) * NUM_CELLS as i32 + (((x + b) / c) & (NUM_CELLS as i32 - 1))) as usize;
        if cell == last {
            continue;
        }
        last = cell;
        let head = ADDR_gDynamicSurfacePartition as usize + cell * SPATIAL_PARTITION_CELL_SIZE + SPATIAL_PARTITION_WALLS as usize * SURFACE_NODE_SIZE;
        let mut node = rd_u32(m, head).unwrap_or(0);
        let mut guard = 0;
        while node != 0 && guard < 400 {
            guard += 1;
            let Some(surf) = rd_u32(m, node as usize + 4) else { break };
            node = rd_u32(m, node as usize).unwrap_or(0);
            let surf = surf as usize;
            if surf == 0 || seen[..nseen].contains(&surf) {
                continue;
            }
            if nseen < seen.len() {
                seen[nseen] = surf;
                nseen += 1;
            }
            let Some((tr, obj)) = read_tr(m, surf) else { continue };
            for &h in hs {
                if let Some(t) = ray_tri(&tr, [o[0], o[1] + h, o[2]], d, len, r)
                    && best.is_none_or(|b| t < b.0)
                {
                    let slot = if obj >= ADDR_gObjectPool as usize { ((obj - ADDR_gObjectPool as usize) / OBJECT_SIZE) as u16 } else { u16::MAX };
                    best = Some((t, slot, tr.n));
                }
            }
        }
    }
    best
}

fn rd_u32(m: &[u8], a: usize) -> Option<u32> {
    m.get(a..a + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}
fn rd_i16(m: &[u8], a: usize) -> Option<i16> {
    m.get(a..a + 2).map(|b| i16::from_le_bytes(b.try_into().unwrap()))
}
fn rd_f32(m: &[u8], a: usize) -> Option<f32> {
    rd_u32(m, a).map(f32::from_bits)
}

fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
}

fn dir_of(yaw: i32) -> [f32; 2] {
    let r = yaw as f32 * std::f32::consts::TAU / 65536.0;
    [r.sin(), r.cos()]
}

pub use behaviours::categorize;

impl SenseCtx {
    pub fn new() -> SenseCtx {
        SenseCtx::default()
    }

    /// Indexes the zone's static geometry from memory `m` (once per zone; call
    /// when Mario is in control on static ground).
    fn ensure(&mut self, m: &[u8], st: &St) -> bool {
        let z = st.zone();
        if self.zones.contains_key(&z) {
            return true;
        }
        if !st.controllable() {
            return false;
        }
        let mario = ADDR_gMarioStates as usize;
        let Some(floor) = rd_u32(m, mario + MARIO_floor).filter(|&f| f != 0) else { return false };
        let (floors, seen) = geo::read_floors(m);
        if !seen.contains(&floor) {
            return false;
        }
        let max_floor = floors.iter().flat_map(|t| t.v.iter().map(|p| p[1])).fold(-11000.0f32, f32::max);
        let walls = geo::read_walls(m);
        let ceils = geo::read_ceils(m);
        self.zones.insert(z, ZoneGeo { floors: Grid::new(&floors), ceils: Grid::new(&ceils), walls: Grid::new(&walls), max_floor });
        true
    }

    pub fn has_zone(&self, st: &St) -> bool {
        self.zones.contains_key(&st.zone())
    }

    /// Static-geometry floor under (x, z) as `find_floor` called from height y: (height, surface type, normal.y).
    pub fn floor_height(&self, st: &St, x: f32, y: f32, z: f32) -> Option<(f32, i16, f32)> {
        self.zones.get(&st.zone())?.floors.floor(x, y, z).map(|h| (h.y, h.kind, h.ny))
    }

    /// Static-geometry ceiling over (x, z) as `find_ceil` called from height y.
    pub fn ceil_height(&self, st: &St, x: f32, y: f32, z: f32) -> Option<f32> {
        self.zones.get(&st.zone())?.ceils.ceil(x, y, z)
    }

    /// Object floors (moving platforms): the highest dynamic floor at most y + 78 over (x, z).
    fn dyn_floor(m: &[u8], x: f32, y: f32, z: f32) -> Option<SurfHit> {
        let (xi, zi) = (s16(x), s16(z));
        let (b, c) = (LEVEL_BOUNDARY_MAX as i32, CELL_SIZE as i32);
        if xi <= -b || xi >= b || zi <= -b || zi >= b {
            return None;
        }
        let (cx, cz) = (((xi + b) / c) & (NUM_CELLS as i32 - 1), ((zi + b) / c) & (NUM_CELLS as i32 - 1));
        let cell = (cz * NUM_CELLS as i32 + cx) as usize;
        let head = ADDR_gDynamicSurfacePartition as usize + cell * SPATIAL_PARTITION_CELL_SIZE + SPATIAL_PARTITION_FLOORS as usize * SURFACE_NODE_SIZE;
        let mut node = rd_u32(m, head)?;
        let mut best: Option<SurfHit> = None;
        let yref = s16(y) as f32;
        let mut guard = 0;
        while node != 0 && guard < 2000 {
            guard += 1;
            let surf = rd_u32(m, node as usize + 4)?;
            node = rd_u32(m, node as usize)?;
            let s = surf as usize;
            let vx = |k: usize| rd_i16(m, s + 0x0A + 6 * k).map(|v| v as i32);
            let vz = |k: usize| rd_i16(m, s + 0x0A + 6 * k + 4).map(|v| v as i32);
            let (x1, z1, x2, z2, x3, z3) = (vx(0)?, vz(0)?, vx(1)?, vz(1)?, vx(2)?, vz(2)?);
            let side = |za: i32, xa: i32, zb: i32, xb: i32| (za - zi).wrapping_mul(xb - xa).wrapping_sub((xa - xi).wrapping_mul(zb - za));
            if side(z1, x1, z2, x2) < 0 || side(z2, x2, z3, x3) < 0 || side(z3, x3, z1, x1) < 0 {
                continue;
            }
            let kind = rd_i16(m, s + SURFACE_type)?;
            if kind as u32 == SURFACE_CAMERA_BOUNDARY {
                continue;
            }
            let n = [rd_f32(m, s + SURFACE_normal)?, rd_f32(m, s + SURFACE_normal_y)?, rd_f32(m, s + SURFACE_normal + 8)?];
            let oo = rd_f32(m, s + SURFACE_originOffset)?;
            if n[1] == 0.0 {
                continue;
            }
            let h = -(xi as f32 * n[0] + n[2] * zi as f32 + oo) / n[1];
            if h > yref + 78.0 {
                continue;
            }
            if best.is_none_or(|b| h > b.y) {
                best = Some(SurfHit { y: h, kind, ny: n[1] });
            }
        }
        best
    }

    /// The floor at (x, z) seen from height y: static grid, then object floors.
    fn floor_at(zg: &ZoneGeo, m: &[u8], dynf: bool, x: f32, y: f32, zz: f32) -> Option<SurfHit> {
        let s = zg.floors.floor(x, y, zz);
        if dynf && let Some(d) = Self::dyn_floor(m, x, y, zz) {
            return match s {
                Some(s) if s.y >= d.y => Some(s),
                _ => Some(d),
            };
        }
        s
    }

    /// Reads the geometry, edges, landing and nearby objects. Other fields: [`reader`].
    pub fn read(&mut self, m: &[u8], st: &St, o: &SenseOpts, regions: &[Region], air: &AirModel) -> Sense {
        let mut s = Sense { edge_ahead: f32::INFINITY, ledge_ahead: f32::INFINITY, wall: [f32::INFINITY; 4], ceil_clear: f32::INFINITY, ..Default::default() };
        s.in_zone = regions.iter().any(|r| r.contains(st.pos));
        // (land / hyp zone flags are set at the end)
        let have = self.ensure(m, st);
        if have {
            if o.walls && o.dyn_walls {
                self.gather_pending(m, st);
            }
            let mut scratch = std::mem::take(&mut self.scratch);
            if let Some(zg) = self.zones.get_mut(&st.zone()) {
                let probe = |zg: &ZoneGeo, x: f32, z: f32, y: f32, d: f32| -> FloorProbe {
                    match Self::floor_at(zg, m, o.dyn_floors, x, y, z) {
                        Some(h) => FloorProbe { dist: d, y: h.y, ny: h.ny, ty: h.kind as u16, class: floor_class(h.kind), has_floor: true },
                        None => FloorProbe { dist: d, has_floor: false, y: -11000.0, ..Default::default() },
                    }
                };
                let me = st.pos;
                let hd = dir_of(st.yaw as i32);
                if o.floors {
                    for (i, &d) in AHEAD_D.iter().enumerate() {
                        s.ahead[i] = probe(zg, me[0] + hd[0] * d, me[2] + hd[1] * d, me[1], d);
                    }
                    let (vx, vz) = (st.vel[0], st.vel[2]);
                    let vl = vx.hypot(vz);
                    let vd = if vl > 1e-3 { [vx / vl, vz / vl] } else { hd };
                    for (i, &d) in VEL_D.iter().enumerate() {
                        s.vel_ahead[i] = probe(zg, me[0] + vd[0] * d, me[2] + vd[1] * d, me[1], d);
                    }
                    // Headroom (Mario's query, vec3f_find_ceil: from max(floor, y) + 80).
                    let yq = if st.floor_y > -10000.0 { st.floor_y.max(me[1]) + 80.0 } else { me[1] + 80.0 };
                    s.ceil_clear = zg.ceils.ceil(me[0], yq, me[2]).map_or(f32::INFINITY, |c| c - me[1]);
                }
                if o.walls {
                    let dirs = [0, 0x4000, -0x4000, 0x8000];
                    for (k, &dy) in dirs.iter().enumerate() {
                        let d = dir_of(st.yaw as i32 + dy);
                        let mut blk = Block::default();
                        zg.walls.along([me[0], me[2]], d, WALL_RANGE, &mut scratch);
                        if !scratch.is_empty() {
                            for &h in &WALL_H {
                                if let Some((t, id)) = zg.walls.ray(&scratch, [me[0], me[1] + h, me[2]], d, WALL_RANGE, WALL_R)
                                    && t < blk.dist
                                {
                                    let n = zg.walls.tris[id].n;
                                    blk = Block { dist: t, src: BlockSrc::Static, normal_yaw: yaw_of(n[0], n[2]), loaded: true };
                                }
                            }
                        }
                        if o.dyn_walls
                            && let Some((t, slot, n)) = dyn_wall_ray(m, me, &WALL_H, d, WALL_RANGE.min(blk.dist), WALL_R)
                            && t < blk.dist
                        {
                            blk = Block { dist: t, src: BlockSrc::Object(slot), normal_yaw: yaw_of(n[0], n[2]), loaded: true };
                        }
                        // Objects whose collision isn't loaded yet (predicted from their model and pose).
                        if o.dyn_walls {
                            for (tr, slot, loaded) in &self.pending {
                                for &h in &WALL_H {
                                    if let Some(t) = ray_tri(tr, [me[0], me[1] + h, me[2]], d, WALL_RANGE.min(blk.dist), WALL_R)
                                        && t < blk.dist
                                    {
                                        blk = Block { dist: t, src: BlockSrc::Object(*slot), normal_yaw: yaw_of(tr.n[0], tr.n[2]), loaded: *loaded };
                                    }
                                }
                            }
                        }
                        s.wall[k] = blk.dist;
                        if k == 0 {
                            s.wall_normal_yaw = blk.normal_yaw;
                        }
                        self.blocks[k] = blk;
                    }
                }
                if o.edges {
                    let ref_y = if st.floor_y > -10000.0 { st.floor_y } else { me[1] };
                    let mut d = EDGE_STEP;
                    while d <= EDGE_RANGE {
                        let (x, z) = (me[0] + hd[0] * d, me[2] + hd[1] * d);
                        match Self::floor_at(zg, m, o.dyn_floors, x, me[1], z) {
                            Some(h) if h.y >= ref_y - 100.0 => {}
                            _ => {
                                s.edge_ahead = d;
                                break;
                            }
                        }
                        d += EDGE_STEP;
                    }
                    // Ledge: a lower wall within reach with no upper wall, and a floor more than 100
                    // above him 60 units into the wall (check_ledge_grab). In the air the wall is
                    // found along his velocity, on the ground along his facing.
                    if o.walls {
                        let hv = st.vel[0].hypot(st.vel[2]);
                        let ld = if st.action & (1 << 11) != 0 && hv > 3.0 { [st.vel[0] / hv, st.vel[2] / hv] } else { hd };
                        zg.walls.along([me[0], me[2]], ld, 400.0, &mut scratch);
                        let low = zg.walls.ray(&scratch, [me[0], me[1] + 30.0, me[2]], ld, 400.0, WALL_R);
                        let up = zg.walls.ray(&scratch, [me[0], me[1] + 150.0, me[2]], ld, 400.0, WALL_R);
                        if let Some((tl, id)) = low
                            && up.is_none_or(|u| u.0 > tl + 1.0)
                        {
                            let n = zg.walls.tris[id].n;
                            let cx = me[0] + ld[0] * tl - n[0] * 60.0;
                            let cz = me[2] + ld[1] * tl - n[2] * 60.0;
                            if let Some(f) = Self::floor_at(zg, m, o.dyn_floors, cx, me[1] + 160.0, cz)
                                && f.y - me[1] > 100.0
                            {
                                s.ledge_ahead = tl;
                            }
                        }
                    }
                }
                if o.landing && st.action & (1 << 11) != 0 {
                    s.land = Self::land(zg, m, st, o, air);
                } else if o.landing && st.floor_y > -10000.0 && (st.pos[1] - st.floor_y) < 100.0 {
                    // grounded: where would each jump land if pressed now?
                    let full = AirModel { fwd: 1.0, side: 0.0, a_down: true };
                    let fv = st.forward_vel;
                    let mk = |action: u32, vy: f32, fvn: f32| {
                        let mut h = st.clone();
                        h.action = action;
                        h.vel[1] = vy;
                        h.forward_vel = fvn;
                        h
                    };
                    let kinds = [
                        mk(0x0300_0881, 52.0 + fv / 4.0, fv * 0.8),
                        mk(0x0180_08AC, 20.0, fv),
                        mk(0x0300_0880, 42.0 + fv / 4.0, fv * 0.8),
                        mk(ACT_LONG_JUMP, 30.0, (fv * 1.5).min(48.0).max(if fv < 0.0 { fv * 1.5 } else { 0.0 })),
                    ];
                    for (i, h) in kinds.iter().enumerate() {
                        s.hyp[i] = Self::land(zg, m, h, o, &full);
                    }
                }
            }
            self.scratch = scratch;
        }
        if o.objs {
            self.objects(m, st, &mut s);
        }
        s.land_zone = s.land.is_some_and(|l| regions.iter().any(|r| r.contains(l.pos)));
        for i in 0..4 {
            s.hyp_zone[i] = s.hyp[i].is_some_and(|l| regions.iter().any(|r| r.contains(l.pos)));
        }
        s
    }

    /// Ballistic fall under `air` until a floor is reached.
    fn land(zg: &ZoneGeo, m: &[u8], st: &St, o: &SenseOpts, air: &AirModel) -> Option<Landing> {
        let mut pos = st.pos;
        let mut vy = st.vel[1];
        let mut fv = st.forward_vel;
        let a = st.action;
        let thr = if a == ACT_LONG_JUMP { 48.0 } else { 32.0 };
        let yr = st.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let (sy, cy) = (yr.sin(), yr.cos());
        let wing = st.mario_flags & 0x8 != 0;
        for f in 1..=150u32 {
            // update_air_without_turn
            fv = if fv > 0.0 { (fv - 0.35).max(0.0) } else { (fv + 0.35).min(0.0) };
            fv += 1.5 * air.fwd;
            if fv > thr {
                fv -= 1.0;
            }
            if fv < -16.0 {
                fv += 2.0;
            }
            let side = air.side * 10.0;
            // sideways speed is along heading + 0x4000 (sin/cos of yaw+0x4000 = cos, -sin)
            let vx = fv * sy + side * cy;
            let vz = fv * cy - side * sy;
            let mut landed = None;
            for _ in 0..4 {
                let np = [pos[0] + vx / 4.0, pos[1] + vy / 4.0, pos[2] + vz / 4.0];
                if np[1] <= zg.max_floor + 1.0
                    && let Some(h) = Self::floor_at(zg, m, o.dyn_floors, np[0], np[1], np[2])
                    && np[1] <= h.y
                {
                    landed = Some((np, h));
                    break;
                }
                pos = np;
            }
            if let Some((np, h)) = landed {
                return Some(Landing {
                    frames: f,
                    pos: [np[0], h.y, np[2]],
                    fv,
                    probe: FloorProbe { dist: 0.0, y: h.y, ny: h.ny, ty: h.kind as u16, class: floor_class(h.kind), has_floor: true },
                });
            }
            // apply_gravity
            if a == ACT_SHOT_FROM_CANNON {
                vy = (vy - 1.0).max(-75.0);
            } else if a == ACT_LONG_JUMP || a == ACT_SLIDE_KICK {
                vy = (vy - 2.0).max(-75.0);
            } else if !air.a_down && vy > 20.0 && a & ACT_FLAG_CONTROL_JUMP_HEIGHT != 0 {
                vy /= 4.0;
            } else if wing && vy < 0.0 && air.a_down {
                vy -= 2.0;
                if vy < -37.5 {
                    vy += 4.0;
                    if vy > -37.5 {
                        vy = -37.5;
                    }
                }
            } else {
                vy = (vy - 4.0).max(-75.0);
            }
            if pos[1] < -11000.0 {
                return None;
            }
        }
        None
    }

    /// Walls of nearby objects with a collision model that is not loaded yet (the game loads it
    /// while Mario is within the object's `oCollisionDistance`): from `o->collisionData` and the pose.
    fn gather_pending(&mut self, m: &[u8], st: &St) {
        self.pending.clear();
        let Some(objs) = st.objects.as_ref() else { return };
        let me = st.pos;
        for ob in objs.iter() {
            let base = ADDR_gObjectPool as usize + ob.slot as usize * OBJECT_SIZE;
            let Some(cd) = rd_u32(m, base + 536).filter(|&p| p != 0) else { continue };
            let (dx, dy, dz) = (ob.pos[0] - me[0], ob.pos[1] - me[1], ob.pos[2] - me[2]);
            let dist3 = (dx * dx + dy * dy + dz * dz).sqrt();
            // Loaded objects are in the partition too, but the partition lags a frame behind the pose
            // (and an object that just came in range isn't there yet), so every model in range is used.
            let loaded = rd_f32(m, base + 404).is_some_and(|cdist| dist3 < cdist);
            let model = self.models.entry(cd as usize).or_insert_with(|| parse_model(m, cd as usize).map(std::rc::Rc::new)).clone();
            let Some(model) = model else { continue };
            let dh = dx.hypot(dz);
            if dh > WALL_RANGE + model.radius + 100.0 {
                continue;
            }
            let rot = [rd_u32(m, base + 208).unwrap_or(0) as i16, rd_u32(m, base + 212).unwrap_or(0) as i16, rd_u32(m, base + 216).unwrap_or(0) as i16];
            model_walls(&model, ob.pos, rot, &mut self.pending, ob.slot, loaded);
        }
    }

    fn objects(&mut self, m: &[u8], st: &St, s: &mut Sense) {
        let Some(objs) = st.objects.as_ref() else { return };
        let mut per: [Vec<ObjProbe>; N_CAT] = Default::default();
        for ob in objs.iter() {
            let c = behaviours::cat_of(ob.bhv);
            if c == CAT_NONE {
                continue;
            }
            let (dx, dy, dz) = (ob.pos[0] - st.pos[0], ob.pos[1] - st.pos[1], ob.pos[2] - st.pos[2]);
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > 4000.0 {
                continue;
            }
            let hr = rd_f32(m, ADDR_gObjectPool as usize + ob.slot as usize * OBJECT_SIZE + 504).filter(|r| r.is_finite() && *r > 0.0 && *r < 3000.0);
            let p = ObjProbe { cat: c, dist, rel_yaw: yaw_of(dx, dz).wrapping_sub(st.yaw), dy, radius: hr.unwrap_or_else(|| behaviours::default_radius(c)) };
            let v = &mut per[c as usize];
            if v.len() < 3 {
                v.push(p);
                v.sort_by(|a, b| a.dist.total_cmp(&b.dist));
            } else if dist < v[2].dist {
                v[2] = p;
                v.sort_by(|a, b| a.dist.total_cmp(&b.dist));
            }
        }
        s.objs = per.into_iter().flatten().collect();
        s.objs.sort_by(|a, b| a.dist.total_cmp(&b.dist));
    }
}

/// Horizontal distance a jump covers on flat ground at forward speed `fv` (the speed before the jump's own
/// scaling), full stick along the facing and A held: kind 0 double jump, 1 jump kick, 2 jump, 3 long jump.
/// Pure physics (update_air_without_turn, gravity, the take-off scaling), no geometry: a quick estimate for
/// "does this hop clear X units".
pub fn flat_range(kind: usize, fv: f32) -> f32 {
    let (mut v, mut vy, g, thr) = match kind {
        0 => (fv * 0.8, 52.0 + fv / 4.0, 4.0, 32.0),
        1 => (fv, 20.0, 4.0, 32.0),
        2 => (fv * 0.8, 42.0 + fv / 4.0, 4.0, 32.0),
        _ => ((fv * 1.5).min(48.0), 30.0, 2.0, 48.0),
    };
    let (mut y, mut d) = (0.0f32, 0.0f32);
    for _ in 0..200 {
        v = if v > 0.0 { (v - 0.35).max(0.0) } else { (v + 0.35).min(0.0) };
        v += 1.5;
        if v > thr {
            v -= 1.0;
        }
        d += v;
        y += vy;
        if y <= 0.0 && vy < 0.0 {
            break;
        }
        vy = (vy - g).max(-75.0);
    }
    d
}

/// Whether the predicted landing lies in any of the regions.
pub fn landing_in(s: &Sense, regions: &[Region]) -> bool {
    s.land.is_some_and(|l| regions.iter().any(|r| r.contains(l.pos)))
}

/// The nearest object of a category.
pub fn nearest(s: &Sense, cat: u8) -> Option<&ObjProbe> {
    s.objs.iter().find(|o| o.cat == cat)
}
