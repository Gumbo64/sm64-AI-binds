//! A library of reactive movement skills. A skill is a small state machine over
//! the observation: every frame it reads Mario's own state (action, speed, flags,
//! caps, held object, floor), the surroundings ([`Sense`]: floor probes, wall
//! rays, edges, predicted landing, zones, objects) and the last pad, and returns a
//! [`Choice`] (button set + stick region + the continuous part).
//!
//! The speed-chain tables (what keeps speed, what stops or slows, the exits) are a
//! port of scratchpad/speedchain_overlay.py (decomp-derived; `file:line` refs are
//! in that file). Everything geometry-aware goes through [`F`], which degrades
//! gracefully when the [`Sense`] is empty (see [`F::sense_ok`]).
//!
//! Params: every skill exposes `param_ranges()`; a GA tunes them. `defaults()` is
//! the midpoint unless a skill says otherwise.
//!
//! Zones (areas Mario must not be on the ground in, e.g. Lakitu's trigger box):
//! [`set_zones`]. A landing whose predicted point is inside a zone is "unsafe".

use crate::types::*;
use obrrt_core::St;
use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use std::f32::consts::TAU;
use std::sync::RwLock;

// ------------------------------------------------------------------ zones

/// An axis-aligned box in the world (x, z ranges; y above `y_min`).
#[derive(Clone, Copy, Debug)]
pub struct Zone {
    pub x: (f32, f32),
    pub z: (f32, f32),
    pub y_min: f32,
}

static ZONES: RwLock<Vec<Zone>> = RwLock::new(Vec::new());

/// Set the forbidden-landing zones (global; the task sets them at start).
pub fn set_zones(z: Vec<Zone>) {
    *ZONES.write().unwrap() = z;
}

/// Lakitu's intro trigger box on the castle grounds (camera_lakitu.inc.c).
pub fn lakitu_zone() -> Zone {
    Zone { x: (-544.0, 545.0), z: (-2000.0, -177.0), y_min: 800.0 }
}

// ----------------------------------------------------------------- trait

pub trait Skill: Send + Sync {
    fn name(&self) -> &'static str;
    fn n_params(&self) -> usize {
        self.param_ranges().len()
    }
    fn param_ranges(&self) -> &[(f32, f32)];
    fn defaults(&self) -> Vec<f32> {
        self.param_ranges().iter().map(|r| 0.5 * (r.0 + r.1)).collect()
    }
    fn choose(&self, o: &Obs, p: &[f32]) -> Choice;
}

struct Fs {
    name: &'static str,
    ranges: &'static [(f32, f32)],
    /// Default params (empty = midpoints).
    defs: &'static [f32],
    f: fn(&F, &[f32]) -> Choice,
}

impl Skill for Fs {
    fn name(&self) -> &'static str {
        self.name
    }
    fn param_ranges(&self) -> &[(f32, f32)] {
        self.ranges
    }
    fn defaults(&self) -> Vec<f32> {
        if self.defs.is_empty() { self.ranges.iter().map(|r| 0.5 * (r.0 + r.1)).collect() } else { self.defs.to_vec() }
    }
    fn choose(&self, o: &Obs, p: &[f32]) -> Choice {
        let f = F::new(o);
        let mut q: Vec<f32> = p.to_vec();
        for (i, r) in self.ranges.iter().enumerate() {
            if i >= q.len() {
                q.push(0.5 * (r.0 + r.1));
            } else {
                q[i] = q[i].clamp(r.0, r.1);
            }
        }
        (self.f)(&f, &q)
    }
}

// ----------------------------------------------------------- feature view

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cls {
    VerySlippery,
    Slippery,
    Default,
    NotSlippery,
}

pub struct F<'a> {
    pub o: &'a Obs<'a>,
    pub s: &'a St,
    pub sn: &'a Sense,
    pub a: u32,
    pub fv: f32,
    /// Horizontal speed along the velocity (what actually moves Mario).
    pub hv: f32,
    pub vy: f32,
    pub cls: Cls,
    pub ny: f32,
    pub down: bool,
    pub slope: bool,
    pub abv: bool,
    /// Frames in the action, restarted by a B press (gp's in_action).
    pub n: u32,
    pub last: u16,
    pub tgt: [f32; 3],
    pub has_tgt: bool,
    pub tgt_yaw: i16,
    pub vel_yaw: i16,
    pub face: i16,
    pub sense_ok: bool,
}

pub fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / TAU).round() as i64 as i32) as i16
}

fn dyaw(a: i16, b: i16) -> i32 {
    (a as i32 - b as i32) as i16 as i32
}

impl<'a> F<'a> {
    pub fn new(o: &'a Obs<'a>) -> F<'a> {
        let s = o.ctx.s;
        let cls = match s.floor_type {
            0x13 => Cls::VerySlippery,
            0x14 | 0x2A | 0x35 | 0x79 => Cls::Slippery,
            0x15 | 0x37 | 0x7A => Cls::NotSlippery,
            _ => Cls::Default,
        };
        let ny = s.floor_normal[1];
        let sy = (s.yaw as f32) * TAU / 65536.0;
        // The normal's horizontal part points downhill.
        let down = s.floor_normal[0] * sy.sin() + s.floor_normal[2] * sy.cos() > 0.0;
        let slope_ny = match cls {
            Cls::VerySlippery => 0.9961947,
            Cls::Slippery => 0.9848077,
            Cls::Default => 0.9659258,
            Cls::NotSlippery => 0.9396926,
        };
        let abv_ny = match cls {
            Cls::VerySlippery => 0.9848077,
            Cls::Slippery => 0.9396926,
            Cls::Default => 0.7880108,
            Cls::NotSlippery => -1.0,
        };
        let t = o.ctx.pts[0];
        let has_tgt = t != [0.0; 3];
        let sn = o.sense;
        // A default Sense (walls at 0, no headroom) was never filled.
        let sense_ok = !(sn.ceil_clear == 0.0 && sn.wall == [0.0; 4]);
        let hv = s.vel[0].hypot(s.vel[2]);
        F {
            o,
            s,
            sn,
            a: s.action,
            fv: s.forward_vel,
            hv,
            vy: s.vel[1],
            cls,
            ny,
            down,
            slope: ny <= slope_ny,
            abv: ny <= abv_ny,
            n: o.ctx.in_act,
            last: o.ctx.last.button,
            tgt: t,
            has_tgt,
            tgt_yaw: yaw_of(t[0] - s.pos[0], t[2] - s.pos[2]),
            vel_yaw: if hv > 0.5 { yaw_of(s.vel[0], s.vel[2]) } else { s.yaw },
            face: s.yaw,
            sense_ok,
        }
    }

    fn was(&self, b: u16) -> bool {
        self.last & b != 0
    }
    fn tgt_dist(&self) -> f32 {
        ((self.tgt[0] - self.s.pos[0]).powi(2) + (self.tgt[2] - self.s.pos[2]).powi(2)).sqrt()
    }
    pub fn airborne(&self) -> bool {
        self.a & 0x800 != 0
    }
    fn grounded_moving(&self) -> bool {
        self.a & 0x400 != 0 && !self.airborne()
    }
    /// Floor probe ahead at (about) distance `d` along the heading.
    fn probe(&self, d: f32) -> Option<FloorProbe> {
        if !self.sense_ok {
            return None;
        }
        const D: [f32; 6] = [24.0, 48.0, 96.0, 192.0, 384.0, 768.0];
        let i = D.iter().position(|x| *x >= d).unwrap_or(5);
        Some(self.sn.ahead[i])
    }
    fn height(&self) -> f32 {
        self.s.pos[1] - self.s.floor_y
    }
    fn nearest(&self, cat: u8) -> Option<ObjProbe> {
        self.sn.objs.iter().filter(|p| p.cat == cat).min_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap()).copied()
    }
    fn wall_fwd(&self) -> f32 {
        if self.sense_ok { self.sn.wall[0] } else { f32::INFINITY }
    }
    /// (wing, metal, vanish)
    fn caps(&self) -> (bool, bool, bool) {
        let m = self.s.mario_flags;
        (m & 0x8 != 0, m & 0x4 != 0, m & 0x2 != 0)
    }
    fn in_water(&self) -> bool {
        self.a & 0x2000 != 0 || self.s.pos[1] < self.s.water_level as f32 - 100.0
    }
}

// ----------------------------------------------------------- choices

fn mk(btn: u8, mag: Mag, ang: Ang, r: Ref) -> Choice {
    Choice { cell: Cell { btn: Btn(btn), stick: StickCls { mag, ang, r } }, yaw: 0, fine_ang: 0.0, fine_mag: 0.0 }
}

fn world(btn: u8, mag: Mag, yaw: i16) -> Choice {
    Choice { cell: Cell { btn: Btn(btn), stick: StickCls { mag, ang: Ang::Aim, r: Ref::World } }, yaw, fine_ang: 0.0, fine_mag: 0.0 }
}

fn neutral(btn: u8) -> Choice {
    mk(btn, Mag::Neutral, Ang::Aim, Ref::Face)
}

/// A button requirement of an option.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Req {
    Any,
    Up,
    Fresh,
    Held,
}

/// The A/B/Z levels for this frame given what was down on the last one: a
/// button that must be fresh but is held is released first (costs a frame).
fn lv(f: &F, a: Req, b: Req, z: Req) -> u8 {
    let one = |r: Req, bit: u16, flag: u8| -> u8 {
        match r {
            Req::Any | Req::Up => 0,
            Req::Held => flag,
            Req::Fresh => {
                if f.was(bit) {
                    0
                } else {
                    flag
                }
            }
        }
    };
    one(a, A_BUTTON, Btn::A) | one(b, B_BUTTON, Btn::B) | one(z, Z_TRIG, Btn::Z)
}

/// `base` nudged towards `want` by `gain` x the error, at most `maxdev`.
fn blend(base: i16, want: i16, gain: f32, maxdev: f32) -> i16 {
    let err = dyaw(want, base) as f32;
    (base as i32 + (err * gain).clamp(-maxdev, maxdev) as i32) as i16
}

// ------------------------------------------------- hidden speed internals
// update_air_without_turn airborne.c:217-246; moving.c:222-312, 369-466.

fn air_v(v: f32, k: f32, cosd: f32, thr: f32) -> f32 {
    let mut v = if v > 0.0 { (v - 0.35).max(0.0) } else { (v + 0.35).min(0.0) };
    v += 1.5 * k * cosd;
    if v > thr {
        v -= 1.0;
    }
    if v < -16.0 {
        v += 2.0;
    }
    v
}

fn cls_i(c: Cls) -> usize {
    match c {
        Cls::VerySlippery => 0,
        Cls::Slippery => 1,
        Cls::Default => 2,
        Cls::NotSlippery => 3,
    }
}
const SLOPE_ACCEL: [f32; 4] = [5.3, 2.7, 1.7, 0.0];
const LAND_DECEL: [f32; 4] = [0.4, 1.4, 4.0, 6.0];
const SLIDE_BASE: [f32; 4] = [0.98, 0.96, 0.92, 0.92];
const SLIDE_ACCEL: [f32; 4] = [10.0, 8.0, 7.0, 5.0];

fn sin_slope(ny: f32) -> f32 {
    (1.0 - ny * ny).max(0.0).sqrt()
}

fn land_v(f: &F, stick: bool) -> f32 {
    let c = cls_i(f.cls);
    let mut v = f.fv;
    if stick {
        if f.slope {
            v += (if f.down { 1.0 } else { -1.0 }) * SLOPE_ACCEL[c] * sin_slope(f.ny);
        } else {
            v *= 0.98;
        }
    } else if v >= 16.0 {
        v = (v - LAND_DECEL[c]).max(0.0);
        if f.slope {
            v += (if f.down { 1.0 } else { -1.0 }) * SLOPE_ACCEL[c] * sin_slope(f.ny);
        }
    }
    v
}

fn slide_v(f: &F, k: f32, fwd: f32) -> f32 {
    let c = cls_i(f.cls);
    let l = k * fwd * 0.02 + SLIDE_BASE[c];
    let mut v = f.fv;
    if f.slope {
        v += (if f.down { 1.0 } else { -1.0 }) * SLIDE_ACCEL[c] * sin_slope(f.ny);
    }
    (v * l).min(100.0 * l)
}

fn walk_v(f: &F) -> f32 {
    let mut v = f.fv;
    if v <= 32.0 {
        v += 1.1 - v / 43.0;
    } else if f.ny >= 0.95 {
        v -= 1.0;
    }
    v = v.min(48.0);
    if f.slope {
        v += (if f.down { 1.0 } else { -1.0 }) * SLOPE_ACCEL[cls_i(f.cls)] * sin_slope(f.ny);
    }
    v
}

// ------------------------------------------------------- ballistic landing

#[derive(Clone, Copy, Debug)]
pub struct Pred {
    pub frames: u32,
    pub pos: [f32; 3],
    pub fv: f32,
    /// The floor under the landing point exists (false = a drop / void).
    pub floor: bool,
}

/// Where a jump with initial vertical speed `vy0`, forward speed `fv0`, gravity
/// `g` along `yaw` comes down (full stick along the heading: the speed-keeping
/// stick). The floor height comes from the forward probes.
pub fn ballistic(f: &F, vy0: f32, fv0: f32, g: f32, yaw: i16, thr: f32) -> Pred {
    let (mut y, mut vy, mut v) = (f.s.pos[1], vy0, fv0);
    let r = yaw as f32 * TAU / 65536.0;
    let (sy, cy) = (r.sin(), r.cos());
    let (mut x, mut z) = (f.s.pos[0], f.s.pos[2]);
    let mut dist = 0.0f32;
    let mut floor_y = f.s.floor_y;
    let mut has = true;
    for t in 1..=120u32 {
        v = air_v(v, 1.0, 1.0, thr);
        x += sy * v;
        z += cy * v;
        dist += v.abs();
        y += vy;
        vy = (vy - g).max(-75.0);
        if let Some(p) = f.probe(dist) {
            if p.has_floor {
                floor_y = p.y;
                has = true;
            } else {
                has = false;
                floor_y = f.s.floor_y - 2000.0;
            }
        }
        if y <= floor_y && vy < 0.0 {
            return Pred { frames: t, pos: [x, floor_y, z], fv: v, floor: has };
        }
    }
    Pred { frames: 120, pos: [x, y, z], fv: v, floor: has }
}

/// Safe = lands on a floor, and not inside (or within `margin` of) a zone.
fn pred_safe(p: &Pred, margin: f32) -> bool {
    if !p.floor {
        return false;
    }
    let zs = ZONES.read().unwrap();
    !zs.iter().any(|z| {
        p.pos[0] > z.x.0 - margin && p.pos[0] < z.x.1 + margin && p.pos[2] > z.z.0 - margin && p.pos[2] < z.z.1 + margin && p.pos[1] > z.y_min - 300.0
    })
}

// ----------------------------------------------------------- the options

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Keep,
    Stop,
    Slow,
    Exit,
    Unsafe,
}

#[derive(Clone, Copy, Debug)]
pub enum Sp {
    Neutral,
    /// 0 < mag <= 48 along `Ref`.
    Short(Ref),
    /// Full stick along `Ref`.
    Full(Ref),
    Back,
}

#[derive(Clone, Copy, Debug)]
pub struct Opt {
    pub id: &'static str,
    pub kind: Kind,
    pub a: Req,
    pub b: Req,
    pub z: Req,
    pub stick: Sp,
    pub leads: u32,
    /// Next-speed ratio (v_next / v) at this state; NaN = unknown.
    pub mult: f32,
}

#[allow(clippy::too_many_arguments)]
fn o(id: &'static str, kind: Kind, a: Req, b: Req, z: Req, stick: Sp, leads: u32, mult: f32) -> Opt {
    Opt { id, kind, a, b, z, stick, leads, mult }
}

fn ratio(x: f32, v: f32) -> f32 {
    if v.abs() < 1e-3 { f32::NAN } else { x / v }
}

use act::*;

fn is_air(a: u32) -> bool {
    matches!(
        a,
        JUMP | DOUBLE_JUMP | TRIPLE_JUMP | LONG_JUMP | FREEFALL | WALL_KICK_AIR | STEEP_JUMP | JUMP_KICK | FORWARD_ROLLOUT | SLIDE_KICK | DIVE | BUTT_SLIDE_AIR | SIDE_FLIP | BACKFLIP
    )
}

/// sLandingAction numFrames (moving.c:26-72).
fn land_len(a: u32) -> Option<u32> {
    match a {
        JUMP_LAND | FREEFALL_LAND | DOUBLE_JUMP_LAND | SIDE_FLIP_LAND | TRIPLE_JUMP_LAND | BACKFLIP_LAND => Some(4),
        LONG_JUMP_LAND => Some(6),
        _ => None,
    }
}

fn land_stop(a: u32) -> bool {
    matches!(a, JUMP_LAND_STOP | FREEFALL_LAND_STOP | DOUBLE_JUMP_LAND_STOP | LONG_JUMP_LAND_STOP | SIDE_FLIP_LAND_STOP | TRIPLE_JUMP_LAND_STOP | BACKFLIP_LAND_STOP)
}

fn next_jump(a: u32) -> u32 {
    match a {
        JUMP_LAND | FREEFALL_LAND | SIDE_FLIP_LAND | JUMP_LAND_STOP | FREEFALL_LAND_STOP | SIDE_FLIP_LAND_STOP => DOUBLE_JUMP,
        DOUBLE_JUMP_LAND | DOUBLE_JUMP_LAND_STOP => TRIPLE_JUMP,
        LONG_JUMP_LAND | LONG_JUMP_LAND_STOP => LONG_JUMP,
        BACKFLIP_LAND | BACKFLIP_LAND_STOP => BACKFLIP,
        _ => JUMP,
    }
}

/// Every option (keep, break) for the state. Port of speedchain_overlay.options().
pub fn options(f: &F) -> Vec<Opt> {
    use Kind::*;
    use Req::*;
    let (a, v) = (f.a, f.fv);
    let mut out: Vec<Opt> = Vec::new();
    if is_air(a) {
        let thr = if a == LONG_JUMP { 48.0 } else { 32.0 };
        out.push(o("air_hold", Keep, if v >= 40.0 { Held } else { Any }, Any, Any, Sp::Full(Ref::Face), a, ratio(air_v(v, 1.0, 1.0, thr), v)));
        if f.sense_ok && f.sn.land.is_some_and(|l| l.frames <= 1) {
            out.push(o("air_prepare_landing", Keep, Held, Any, Any, Sp::Short(Ref::Face), FREEFALL_LAND, f32::NAN));
        }
        let has_b = matches!(a, JUMP | DOUBLE_JUMP | TRIPLE_JUMP | FREEFALL | WALL_KICK_AIR | STEEP_JUMP | SIDE_FLIP);
        if has_b && v <= 28.0 {
            out.push(o("air_jump_kick", Keep, Any, Fresh, Any, Sp::Full(Ref::Face), JUMP_KICK, 1.0));
        }
        if f.wall_fwd() < 120.0 && v > 16.0 && matches!(a, JUMP | DOUBLE_JUMP | TRIPLE_JUMP | LONG_JUMP | FREEFALL | WALL_KICK_AIR | DIVE) {
            out.push(o("air_to_wall", Keep, Any, Any, Any, Sp::Full(Ref::Face), AIR_HIT_WALL, 1.0));
        }
        let pound = matches!(a, JUMP | DOUBLE_JUMP | TRIPLE_JUMP | FREEFALL | WALL_KICK_AIR | SIDE_FLIP | BACKFLIP);
        if pound {
            out.push(o("pound", Stop, Any, Any, Fresh, Sp::Neutral, GROUND_POUND, 0.0));
        } else {
            out.push(o("land_idle", Stop, Any, Any, Any, Sp::Neutral, FREEFALL_LAND_STOP, 0.0));
        }
        out.push(o("air_neutral", Slow, Any, Any, Any, Sp::Neutral, a, ratio(air_v(v, 0.0, 1.0, thr), v)));
        out.push(o("air_back", Slow, Any, Any, Any, Sp::Back, a, ratio(air_v(v, 1.0, -1.0, thr), v)));
        if has_b && v > 28.0 {
            out.push(o("air_dive", Exit, Any, Fresh, Any, Sp::Full(Ref::Face), DIVE, ratio((v + 15.0).min(48.0), v)));
        }
        if a == FORWARD_ROLLOUT && f.sn.land.is_some_and(|l| l.frames <= 1) {
            out.push(o("rollout_stop", Unsafe, Up, Up, Up, Sp::Neutral, FREEFALL_LAND_STOP, 0.0));
        }
    } else if a == AIR_HIT_WALL {
        out.push(o("wall_kick", Keep, Fresh, Any, Any, Sp::Full(Ref::Face), WALL_KICK_AIR, 1.0));
        out.push(o("wall_bonk", Unsafe, Up, Any, Any, Sp::Neutral, if v >= 38.0 { BACKWARD_AIR_KB } else { SOFT_BONK }, f32::NAN));
    } else if let Some(n) = land_len(a) {
        let wait_ok = f.s.action_timer as u32 + 1 < n;
        let m = ratio(land_v(f, true), v);
        if wait_ok {
            out.push(o("land_wait", Keep, if v >= 40.0 { Held } else { Any }, Any, Any, Sp::Full(Ref::Face), a, m));
        } else {
            if f.was(A_BUTTON) {
                out.push(o("land_kick", Keep, Held, Fresh, Any, Sp::Short(Ref::Face), JUMP_KICK, if v >= 24.0 { m } else { 0.5 }));
            }
            out.push(o("land_crouch", Keep, Any, Any, Fresh, Sp::Full(Ref::Face), CROUCH_SLIDE, ratio(slide_v(f, 1.0, 1.0), v)));
        }
        if f.abv && f.down {
            out.push(o("land_auto_slide", Keep, Any, Any, Any, Sp::Full(Ref::Face), BUTT_SLIDE, 1.0));
        }
        let j = next_jump(a);
        let jm = if j == LONG_JUMP { ratio((1.5 * v).min(48.0), v) } else { 0.8 };
        out.push(o("land_jump", Exit, Fresh, Any, if j == LONG_JUMP { Held } else { Any }, Sp::Full(Ref::Face), j, jm));
        out.push(o("land_dive", Exit, Any, Fresh, Any, Sp::Full(Ref::Face), DIVE, ratio((v + 15.0).min(48.0), v)));
        if !wait_ok {
            out.push(o("land_expire_walk", Slow, Any, Any, Any, Sp::Full(Ref::Face), WALKING, ratio(walk_v(f), v)));
        }
        out.push(o("land_idle", Stop, Any, Any, Any, Sp::Neutral, FREEFALL_LAND_STOP, 0.0));
        out.push(o("land_neutral", Slow, Any, Any, Any, Sp::Neutral, a, ratio(land_v(f, false), v)));
    } else if land_stop(a) {
        if f.n <= 1 {
            if f.was(A_BUTTON) {
                out.push(o("stop_kick", Keep, Held, Fresh, Any, Sp::Short(Ref::Face), JUMP_KICK, if v >= 24.0 { 0.99 } else { 0.5 }));
            }
            out.push(o("stop_crouch", Keep, Any, Any, Fresh, Sp::Full(Ref::Face), CROUCH_SLIDE, if v >= 24.0 { 0.94 } else { 0.5 }));
            out.push(o("stop_jump", Exit, Fresh, Any, Any, Sp::Full(Ref::Face), next_jump(a), 0.8));
            out.push(o("stop_dive", Exit, Any, Fresh, Any, Sp::Full(Ref::Face), DIVE, ratio((v + 15.0).min(48.0), v)));
            out.push(o("stop_walk", Slow, Any, Any, Any, Sp::Full(Ref::Face), WALKING, ratio(walk_v(f), v)));
            out.push(o("stop_turn", Slow, Any, Any, Any, Sp::Back, TURNING_AROUND, 0.91));
        }
        out.push(o("stop_idle", Stop, Any, Any, Any, Sp::Neutral, a, 0.0));
    } else if matches!(a, DIVE_SLIDE | BUTT_SLIDE | STOMACH_SLIDE | SLIDE_KICK_SLIDE | CROUCH_SLIDE) {
        let up = ratio(slide_v(f, 1.0, 1.0), v);
        out.push(o("slide_hold", Keep, Any, Any, Any, Sp::Full(Ref::Vel), a, up));
        if a == DIVE_SLIDE && !f.abv {
            out.push(o("slide_rollout", Keep, Fresh, Any, Any, Sp::Full(Ref::Vel), FORWARD_ROLLOUT, 1.0));
        }
        if a == STOMACH_SLIDE && !f.abv && f.n >= 5 {
            out.push(o("slide_rollout", Keep, Fresh, Any, Any, Sp::Full(Ref::Vel), FORWARD_ROLLOUT, 1.0));
        }
        if a == SLIDE_KICK_SLIDE && f.ny > 0.8660254 {
            out.push(o("slide_rollout", Keep, Fresh, Any, Any, Sp::Full(Ref::Vel), FORWARD_ROLLOUT, 1.0));
        }
        if a == CROUCH_SLIDE && v >= 10.0 {
            out.push(o("crouch_kick", Keep, Any, Fresh, Held, Sp::Full(Ref::Vel), SLIDE_KICK, ratio(v.max(32.0), v)));
        }
        if a == CROUCH_SLIDE && v > 10.0 && f.n <= 30 {
            out.push(o("crouch_lj", Exit, Fresh, Any, Held, Sp::Full(Ref::Vel), LONG_JUMP, ratio((1.5 * v).min(48.0), v)));
        }
        if a == BUTT_SLIDE && f.n >= 5 {
            out.push(o("butt_jump", Exit, Fresh, Any, Any, Sp::Full(Ref::Vel), JUMP, 0.8));
        }
        out.push(o("slide_neutral", Slow, Any, Any, Any, Sp::Neutral, a, ratio(slide_v(f, 0.0, 0.0), v)));
        out.push(o("slide_back", Slow, Any, Any, Any, Sp::Back, a, ratio(slide_v(f, 1.0, -1.0), v)));
        out.push(o("slide_stop", Stop, Any, Any, Any, Sp::Back, a, 0.0));
    } else if matches!(a, TURNING_AROUND | BRAKING | DECELERATING) {
        out.push(o("wind_down", Slow, Any, Any, Any, Sp::Neutral, a, ratio((v - if a == BRAKING { LAND_DECEL[cls_i(f.cls)] } else { 1.0 }).max(0.0), v)));
        out.push(o("slow_jump", Exit, Fresh, Any, Any, Sp::Full(Ref::Face), JUMP, 0.8));
        out.push(o("resume", Slow, Any, Any, Any, Sp::Full(Ref::Face), WALKING, ratio(walk_v(f), v)));
    } else if a == WALKING {
        if f.was(A_BUTTON) {
            out.push(o("walk_kick", Keep, Held, Fresh, Any, Sp::Short(Ref::Face), JUMP_KICK, if v >= 24.0 { 0.99 } else { 0.5 }));
        }
        out.push(o("walk_crouch", Keep, Any, Any, Fresh, Sp::Full(Ref::Face), CROUCH_SLIDE, if v >= 24.0 { 0.94 } else { 0.5 }));
        out.push(o("walk_hold", Keep, Any, Any, Any, Sp::Full(Ref::Face), WALKING, ratio(walk_v(f), v)));
        out.push(o("walk_jump", Exit, Fresh, Any, Any, Sp::Full(Ref::Face), JUMP, 0.8));
        out.push(o("walk_dive", Exit, Any, Fresh, Any, Sp::Full(Ref::Face), DIVE, ratio((v + 15.0).min(48.0), v)));
        out.push(o("walk_back", Slow, Any, Any, Any, Sp::Back, TURNING_AROUND, 0.91));
        out.push(o("walk_neutral", Slow, Any, Any, Any, Sp::Neutral, DECELERATING, ratio((v - 1.0).max(0.0), v)));
        out.push(o("walk_brake", Stop, Any, Any, Any, Sp::Neutral, BRAKING, ratio((v - LAND_DECEL[cls_i(f.cls)]).max(0.0), v)));
    }
    // Slow/exit options that lose < 3% are keep options.
    for x in out.iter_mut() {
        if matches!(x.kind, Slow | Exit) && x.mult.is_finite() && x.mult >= 0.97 {
            x.kind = Keep;
        }
    }
    out
}

fn stick_choice(sp: Sp, btn: u8) -> Choice {
    let (mag, ang, r) = match sp {
        Sp::Neutral => (Mag::Neutral, Ang::Aim, Ref::Face),
        Sp::Short(r) => (Mag::Short, Ang::Aim, r),
        Sp::Full(r) => (Mag::Full, Ang::Aim, r),
        Sp::Back => (Mag::Full, Ang::Back, Ref::Face),
    };
    mk(btn, mag, ang, r)
}

pub fn opt_choice(f: &F, x: &Opt) -> Choice {
    stick_choice(x.stick, lv(f, x.a, x.b, x.z))
}

/// Where the jump an option starts would land (None = not a jump option).
fn opt_pred(f: &F, x: &Opt) -> Option<Pred> {
    let (v, yaw) = (f.fv, f.face);
    match x.leads {
        JUMP_KICK => Some(ballistic(f, 20.0, v, 4.0, yaw, 32.0)),
        DOUBLE_JUMP => Some(ballistic(f, 52.0 + v / 4.0, v * 0.8, 4.0, yaw, 32.0)),
        TRIPLE_JUMP => Some(ballistic(f, 69.0, v * 0.8, 4.0, yaw, 32.0)),
        JUMP => Some(ballistic(f, 42.0 + v / 4.0, v * 0.8, 4.0, yaw, 32.0)),
        LONG_JUMP => Some(ballistic(f, 30.0, (1.5 * v).min(48.0), 2.0, yaw, 48.0)),
        BACKFLIP => Some(ballistic(f, 62.0, -16.0, 4.0, yaw.wrapping_add(i16::MIN), 32.0)),
        _ => None,
    }
}

fn keep_opts(f: &F) -> Vec<Opt> {
    options(f).into_iter().filter(|x| x.kind == Kind::Keep).collect()
}

fn best_by_mult(v: &[Opt]) -> Option<Opt> {
    v.iter().copied().max_by(|a, b| {
        let (x, y) = (if a.mult.is_nan() { 0.5 } else { a.mult }, if b.mult.is_nan() { 0.5 } else { b.mult });
        x.partial_cmp(&y).unwrap()
    })
}

/// If a wall is in the way of `yaw` (the forward ray), slide along it: the
/// tangent nearer to `yaw`. Needs a filled Sense; otherwise `yaw` unchanged.
pub fn wall_deflect(f: &F, yaw: i16) -> i16 {
    if !f.sense_ok || f.hv < 8.0 {
        return yaw;
    }
    let w = f.sn.wall[0];
    if !(w.is_finite() && w < 8.0 * f.hv && dyaw(yaw, f.face).abs() < 0x2800) {
        return yaw;
    }
    let n = f.sn.wall_normal_yaw;
    let (t1, t2) = (n.wrapping_add(0x4000), n.wrapping_sub(0x4000));
    if dyaw(t1, yaw).abs() < dyaw(t2, yaw).abs() { t1 } else { t2 }
}

/// A choice with its stick reference steered towards the target by `gain`
/// (the speed cost is the cosine of the deviation).
fn steered(f: &F, c: Choice, gain: f32, maxdev: f32) -> Choice {
    if !f.has_tgt || c.cell.stick.mag == Mag::Neutral || c.cell.stick.ang != Ang::Aim {
        return c;
    }
    let base = match c.cell.stick.r {
        Ref::Face => f.face,
        Ref::Vel => f.vel_yaw,
        Ref::World => c.yaw,
        _ => return c,
    };
    let yaw = wall_deflect(f, blend(base, f.tgt_yaw, gain, maxdev));
    Choice { cell: Cell { stick: StickCls { r: Ref::World, ..c.cell.stick }, ..c.cell }, yaw, ..c }
}

fn fwd(f: &F, btn: u8) -> Choice {
    steered(f, mk(btn, Mag::Full, Ang::Aim, Ref::Face), 1.0, 0x1800 as f32)
}

// -------------------------------------------------------------- the skills

/// The best speed-keeping option whose landing is safe (zone, floor).
fn keep_best(f: &F, p: &[f32]) -> Choice {
    let (margin, gain, maxdev) = (p[0], p[1], p[2]);
    let ks = keep_opts(f);
    let ok: Vec<Opt> = ks.iter().copied().filter(|x| opt_pred(f, x).is_none_or(|pr| pred_safe(&pr, margin))).collect();
    match best_by_mult(&ok).or_else(|| best_by_mult(&ks)) {
        Some(x) => steered(f, opt_choice(f, &x), gain, maxdev),
        None => steered(f, mk(0, Mag::Full, Ang::Aim, Ref::Face), gain, maxdev),
    }
}

/// Only the option that stays in this action (argmax of the kept speed).
fn hold_speed(f: &F, p: &[f32]) -> Choice {
    let ks = keep_opts(f);
    let same: Vec<Opt> = ks.iter().copied().filter(|x| x.leads == f.a).collect();
    match best_by_mult(&same).or_else(|| best_by_mult(&ks)) {
        Some(x) => steered(f, opt_choice(f, &x), p[0], p[1]),
        None => mk(0, Mag::Full, Ang::Aim, Ref::Face),
    }
}

/// Aim at the target; the buttons come from the best keep option.
fn steer_target(f: &F, p: &[f32]) -> Choice {
    let yaw = if f.has_tgt { f.tgt_yaw } else { f.face };
    let btn = best_by_mult(&keep_opts(f)).map_or(0, |x| lv(f, x.a, x.b, x.z));
    // p[0]: tilt used when close (frames' worth of speed); p[1]: full stick if > .5
    let close = f.has_tgt && f.tgt_dist() < p[0] * f.hv.max(8.0);
    let mag = if p[1] > 0.5 && !close { Mag::Full } else { Mag::Short };
    world(btn, mag, yaw)
}

/// Stick along the slide, steering within a speed-loss budget (lossFactor has
/// k*cos(dyaw)*0.02: a modest deviation is nearly free).
fn slide_optimise(f: &F, p: &[f32]) -> Choice {
    let (gain, maxdev, rollout_below) = (p[0], p[1], p[2]);
    if matches!(f.a, DIVE_SLIDE | BUTT_SLIDE | STOMACH_SLIDE | SLIDE_KICK_SLIDE | CROUCH_SLIDE) {
        let yaw = if f.has_tgt { blend(f.vel_yaw, f.tgt_yaw, gain, maxdev) } else { f.vel_yaw };
        let btn = if f.a == DIVE_SLIDE && !f.abv && f.fv > rollout_below { lv(f, Req::Fresh, Req::Any, Req::Any) } else { 0 };
        return world(btn, Mag::Full, yaw);
    }
    steer_target(f, &[0.0, 1.0])
}

/// Slides into rollouts, the cancel that keeps all speed.
fn rollout_cancel(f: &F, p: &[f32]) -> Choice {
    let min_fv = p[0];
    let c = match f.a {
        DIVE_SLIDE if !f.abv && f.fv >= min_fv => lv(f, Req::Fresh, Req::Any, Req::Any),
        SLIDE_KICK_SLIDE if f.ny > 0.8660254 && f.fv >= min_fv => lv(f, Req::Fresh, Req::Any, Req::Any),
        STOMACH_SLIDE if !f.abv && f.n >= 5 && f.fv >= min_fv => lv(f, Req::Fresh, Req::Any, Req::Any),
        _ => 0,
    };
    steered(f, mk(c, Mag::Full, Ang::Aim, Ref::Vel), 1.0, 0x1000 as f32)
}

/// Landing refresh with jump kicks (A held + B fresh at a short stick), the
/// landing point checked against the zones (the hand overlay's main failure).
fn kick_chain(f: &F, p: &[f32]) -> Choice {
    let (min_fv, margin) = (p[0], p[1]);
    let dflt = fwd(f, 0);
    if f.fv < min_fv {
        return dflt;
    }
    match f.a {
        FORWARD_ROLLOUT | JUMP_KICK | JUMP | FREEFALL | DOUBLE_JUMP | WALL_KICK_AIR | DIVE => steered(f, mk(Btn::A, Mag::Full, Ang::Aim, Ref::Face), 0.8, 0x0C00 as f32),
        FREEFALL_LAND | JUMP_LAND | DOUBLE_JUMP_LAND | FREEFALL_LAND_STOP | JUMP_LAND_STOP | PUNCHING | MOVE_PUNCHING => {
            // Landing actions (not the *_LAND_STOP ones) only take a cancel on the frame the
            // timer runs out (++actionTimer >= 4 in common_landing_cancels); before that
            // hold A and the stick so the kick on that frame is A-held + B-fresh.
            let expiring = !matches!(f.a, FREEFALL_LAND | JUMP_LAND | DOUBLE_JUMP_LAND) || f.s.action_timer as u32 + 1 >= 4;
            if !expiring {
                return steered(f, mk(Btn::A, Mag::Full, Ang::Aim, Ref::Face), 1.0, 0x1800 as f32);
            }
            let kick = ballistic(f, 20.0, f.fv, 4.0, f.face, 32.0);
            if f.was(A_BUTTON) && pred_safe(&kick, margin) {
                steered(f, mk(lv(f, Req::Held, Req::Fresh, Req::Any), Mag::Short, Ang::Aim, Ref::Face), 1.0, 0x1800 as f32)
            } else {
                let dj = ballistic(f, 52.0 + f.fv / 4.0, f.fv * 0.8, 4.0, f.face, 32.0);
                if pred_safe(&dj, margin) && !f.was(A_BUTTON) {
                    steered(f, mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Aim, Ref::Face), 1.0, 0x1800 as f32)
                } else if matches!(f.a, FREEFALL_LAND | JUMP_LAND | DOUBLE_JUMP_LAND) {
                    steered(f, mk(lv(f, Req::Any, Req::Any, Req::Fresh), Mag::Full, Ang::Aim, Ref::Face), 1.0, 0x1800 as f32)
                } else {
                    dflt
                }
            }
        }
        _ => dflt,
    }
}

/// crouch slide -> slide kick -> slide kick slide -> rollout -> landing -> crouch slide.
fn slide_kick_chain(f: &F, p: &[f32]) -> Choice {
    let min_fv = p[0];
    let st = |btn: u8| steered(f, mk(btn, Mag::Full, Ang::Aim, Ref::Vel), p[1], 0x1000 as f32);
    match f.a {
        CROUCH_SLIDE if f.fv >= 10.0 => st(lv(f, Req::Any, Req::Fresh, Req::Held)),
        SLIDE_KICK => st(0),
        SLIDE_KICK_SLIDE => st(lv(f, Req::Fresh, Req::Any, Req::Any)),
        FORWARD_ROLLOUT => steered(f, mk(Btn::A, Mag::Full, Ang::Aim, Ref::Face), p[1], 0x1000 as f32),
        FREEFALL_LAND | FREEFALL_LAND_STOP | JUMP_LAND | JUMP_LAND_STOP | DOUBLE_JUMP_LAND | WALKING if f.fv >= min_fv => {
            steered(f, mk(lv(f, Req::Any, Req::Any, Req::Fresh), Mag::Full, Ang::Aim, Ref::Face), p[1], 0x1000 as f32)
        }
        _ => steered(f, mk(0, Mag::Full, Ang::Aim, Ref::Face), p[1], 0x1000 as f32),
    }
}

/// Leave a ledge, dive at <= 33 (+15, cap 48), rollout, repeat.
fn edge_dive_hop(f: &F, p: &[f32]) -> Choice {
    let max_fv = p[0];
    match f.a {
        FREEFALL | JUMP | DOUBLE_JUMP | WALL_KICK_AIR if f.fv >= 12.0 && f.fv <= max_fv && f.height() > 40.0 => fwd(f, lv(f, Req::Any, Req::Fresh, Req::Any)),
        DIVE => fwd(f, 0),
        DIVE_SLIDE if !f.abv => steered(f, mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Aim, Ref::Vel), 1.0, 0x1000 as f32),
        WALKING if f.fv >= 12.0 && f.sense_ok && f.sn.edge_ahead < p[1] => fwd(f, lv(f, Req::Fresh, Req::Any, Req::Any)),
        _ => fwd(f, 0),
    }
}

/// Aim into a wall, A on the one real frame, then along the kick.
fn wall_kick_chain(f: &F, p: &[f32]) -> Choice {
    let reach = p[0];
    match f.a {
        AIR_HIT_WALL => mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Aim, Ref::Face),
        WALL_KICK_AIR => {
            if f.has_tgt {
                world(0, Mag::Full, blend(f.face, f.tgt_yaw, 1.0, 0x2000 as f32))
            } else {
                mk(0, Mag::Full, Ang::Aim, Ref::Face)
            }
        }
        _ if f.airborne() && f.wall_fwd() < reach && f.fv > 16.0 => mk(0, Mag::Full, Ang::Aim, Ref::Face),
        _ if f.airborne() && f.sense_ok && f.sn.wall[0].is_finite() && f.sn.wall[0] < 600.0 => world(0, Mag::Full, f.sn.wall_normal_yaw.wrapping_add(i16::MIN)),
        _ => fwd(f, 0),
    }
}

/// Slope butt slide: jump at actionTimer 5, land back in the slide (x1.015/lap).
fn butt_slide_chain(f: &F, p: &[f32]) -> Choice {
    let (min_fv, gain) = (p[0], p[1]);
    let along = |btn: u8| steered(f, mk(btn, Mag::Full, Ang::Aim, Ref::Vel), gain, 0x1000 as f32);
    match f.a {
        BUTT_SLIDE if f.n >= 5 && f.fv >= min_fv => along(lv(f, Req::Fresh, Req::Any, Req::Any)),
        BUTT_SLIDE | STOMACH_SLIDE | JUMP | BUTT_SLIDE_AIR | JUMP_LAND | FREEFALL_LAND | DOUBLE_JUMP_LAND | LONG_JUMP_LAND => along(0),
        _ => steered(f, mk(0, Mag::Full, Ang::Aim, Ref::Face), gain, 0x1000 as f32),
    }
}

/// single -> double -> triple by landing timing (doubleJumpTimer), A held for height.
fn jump_chain(f: &F, p: &[f32]) -> Choice {
    let (min_fv_triple, hold_a) = (p[0], p[1]);
    match f.a {
        JUMP | DOUBLE_JUMP | TRIPLE_JUMP | LONG_JUMP | WALL_KICK_AIR | STEEP_JUMP => fwd(f, if hold_a > 0.5 { Btn::A } else { 0 }),
        JUMP_LAND | FREEFALL_LAND | SIDE_FLIP_LAND | WALKING | IDLE | JUMP_LAND_STOP | FREEFALL_LAND_STOP => fwd(f, lv(f, Req::Fresh, Req::Any, Req::Any)),
        DOUBLE_JUMP_LAND => {
            let c = if f.fv > min_fv_triple { Req::Fresh } else { Req::Up };
            fwd(f, lv(f, c, Req::Any, Req::Any))
        }
        _ => fwd(f, 0),
    }
}

/// Crouch slide + A -> long jump (x1.5, cap 48); with `blj` the stick is held
/// back so the negative speed grows (BLJ).
fn long_jump_builder(f: &F, p: &[f32]) -> Choice {
    let blj = p[0] > 0.5;
    let ang = if blj { Ang::Back } else { Ang::Aim };
    match f.a {
        WALKING if f.fv > 10.0 => mk(lv(f, Req::Any, Req::Any, Req::Fresh), Mag::Full, Ang::Aim, Ref::Face),
        CROUCH_SLIDE if f.fv > 10.0 && f.n <= 30 => mk(lv(f, Req::Fresh, Req::Any, Req::Held), Mag::Full, ang, Ref::Face),
        LONG_JUMP => mk(lv(f, Req::Any, Req::Any, Req::Held), Mag::Full, ang, Ref::Face),
        LONG_JUMP_LAND | LONG_JUMP_LAND_STOP => mk(lv(f, Req::Fresh, Req::Any, Req::Held), Mag::Full, ang, Ref::Face),
        _ => fwd(f, 0),
    }
}

/// Z in the air above the floor by `min_height` (zeroes the forward speed).
fn ground_pound(f: &F, p: &[f32]) -> Choice {
    let (min_h, near_tgt) = (p[0], p[1]);
    let close = !f.has_tgt || f.tgt_dist() < near_tgt;
    if f.airborne() && f.height() > min_h && close && f.vy < 0.0 && matches!(f.a, JUMP | DOUBLE_JUMP | TRIPLE_JUMP | FREEFALL | WALL_KICK_AIR | SIDE_FLIP | BACKFLIP) {
        return mk(lv(f, Req::Any, Req::Any, Req::Fresh), Mag::Neutral, Ang::Aim, Ref::Face);
    }
    fwd(f, 0)
}

/// The fastest way to zero speed.
fn brake_stop(f: &F, _p: &[f32]) -> Choice {
    match f.a {
        WALKING | DECELERATING | TURNING_AROUND => neutral(0),
        DIVE_SLIDE | BUTT_SLIDE | STOMACH_SLIDE | SLIDE_KICK_SLIDE | CROUCH_SLIDE => mk(0, Mag::Full, Ang::Back, Ref::Face),
        a if is_air(a) => {
            let pound = matches!(a, JUMP | DOUBLE_JUMP | TRIPLE_JUMP | FREEFALL | WALL_KICK_AIR | SIDE_FLIP | BACKFLIP);
            if pound && f.height() > 150.0 {
                neutral(lv(f, Req::Any, Req::Any, Req::Fresh))
            } else {
                neutral(0)
            }
        }
        _ => neutral(0), // landing: no cancel, *_LAND_STOP zeroes it
    }
}

/// Bleed to a target speed without stopping (the option whose ratio is nearest the need).
fn slow_bleed(f: &F, p: &[f32]) -> Choice {
    let target = p[0];
    if f.fv <= target {
        return hold_speed(f, &[1.0, 0x1000 as f32]);
    }
    let want = (target / f.fv.max(1.0)).clamp(0.0, 1.0);
    let opts: Vec<Opt> = options(f).into_iter().filter(|x| matches!(x.kind, Kind::Slow | Kind::Keep) && x.mult.is_finite()).collect();
    match opts.iter().min_by(|a, b| (a.mult - want).abs().partial_cmp(&(b.mult - want).abs()).unwrap()) {
        Some(x) => opt_choice(f, x),
        None => neutral(0),
    }
}

/// Reverse the heading: held-back stick on the ground (TURNING_AROUND, then a side flip), steer round in the air.
fn turn_around(f: &F, p: &[f32]) -> Choice {
    let thr = p[0];
    let err = dyaw(f.tgt_yaw, f.face).abs() as f32;
    if f.has_tgt && err > thr {
        if f.a == WALKING && f.fv >= 16.0 {
            return mk(0, Mag::Full, Ang::Back, Ref::Face);
        }
        if f.a == TURNING_AROUND {
            return mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Aim, Ref::Target);
        }
    }
    world(0, Mag::Full, if f.has_tgt { f.tgt_yaw } else { f.face })
}

/// Leave the chain with a jump (0 jump, 1 double, 2 long, 3 backflip).
fn exit_jump(f: &F, p: &[f32]) -> Choice {
    let kind = p[0];
    match f.a {
        WALKING | IDLE | BRAKING | DECELERATING | TURNING_AROUND | JUMP_LAND | FREEFALL_LAND | DOUBLE_JUMP_LAND | LONG_JUMP_LAND | JUMP_LAND_STOP | FREEFALL_LAND_STOP | BUTT_SLIDE => {
            if kind < 1.5 {
                mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Aim, Ref::Face)
            } else if kind < 2.5 {
                mk(lv(f, Req::Fresh, Req::Any, Req::Held), Mag::Full, Ang::Aim, Ref::Face)
            } else {
                mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Back, Ref::Face)
            }
        }
        CROUCH_SLIDE => mk(lv(f, Req::Fresh, Req::Any, Req::Held), Mag::Full, Ang::Aim, Ref::Face),
        _ => fwd(f, 0),
    }
}

/// B at the right moment: on the ground (speed kick: fv >= 29 and stick > 48) or in the air.
fn exit_dive(f: &F, p: &[f32]) -> Choice {
    let min_fv = p[0];
    match f.a {
        WALKING | JUMP_LAND | FREEFALL_LAND | DOUBLE_JUMP_LAND | JUMP_LAND_STOP | FREEFALL_LAND_STOP if f.fv >= min_fv.max(29.0) => fwd(f, lv(f, Req::Any, Req::Fresh, Req::Any)),
        JUMP | DOUBLE_JUMP | TRIPLE_JUMP | FREEFALL | WALL_KICK_AIR | SIDE_FLIP if f.fv > 28.0 => fwd(f, lv(f, Req::Any, Req::Fresh, Req::Any)),
        _ => fwd(f, 0),
    }
}

/// Steer onto a ledge lip and grab; stick forward climbs.
fn ledge_grab(f: &F, p: &[f32]) -> Choice {
    let reach = p[0];
    match f.a {
        LEDGE_GRAB | LEDGE_CLIMB_FAST => mk(0, Mag::Full, Ang::Aim, Ref::Face),
        a if is_air(a) && f.sense_ok && f.sn.ledge_ahead < reach => mk(0, Mag::Full, Ang::Aim, Ref::Face),
        _ => fwd(f, 0),
    }
}

/// Approach a pole, grab, climb or jump off.
fn pole(f: &F, p: &[f32]) -> Choice {
    let (cat, jump_off) = (p[0] as u8, p[1] > 0.5);
    match f.a {
        GRAB_POLE_SLOW | GRAB_POLE_FAST | HOLDING_POLE | CLIMBING_POLE | TOP_OF_POLE => {
            if jump_off && f.n >= 2 {
                mk(lv(f, Req::Fresh, Req::Any, Req::Any), Mag::Full, Ang::Aim, Ref::Face)
            } else {
                mk(0, Mag::Full, Ang::Aim, Ref::Face)
            }
        }
        _ => match f.nearest(cat) {
            Some(ob) => world(0, Mag::Full, f.face.wrapping_add(ob.rel_yaw)),
            None => fwd(f, 0),
        },
    }
}

/// Swimming: A held (or pulsed every k frames), the stick at the target.
fn swim_cadence(f: &F, p: &[f32]) -> Choice {
    let pulse = p[0] as u32;
    let btn = if pulse == 0 || f.s.action_timer as u32 % pulse == 0 { Btn::A } else { 0 };
    if f.in_water() {
        world(btn, Mag::Full, if f.has_tgt { f.tgt_yaw } else { f.face })
    } else {
        fwd(f, 0)
    }
}

/// Slide along a wall instead of bonking into it.
fn avoid_wall(f: &F, p: &[f32]) -> Choice {
    let (frames_ahead, side_bias) = (p[0], p[1]);
    if f.sense_ok && f.wall_fwd() < frames_ahead * f.hv.max(8.0) {
        let n = f.sn.wall_normal_yaw;
        let (t1, t2) = (n.wrapping_add(0x4000), n.wrapping_sub(0x4000));
        let want = if f.has_tgt { f.tgt_yaw } else { f.face };
        let pick = if (dyaw(t1, want).abs() + (side_bias * 1000.0) as i32) < dyaw(t2, want).abs() { t1 } else { t2 };
        return world(0, Mag::Full, pick);
    }
    fwd(f, 0)
}

/// Don't run off a drop.
fn avoid_edge(f: &F, p: &[f32]) -> Choice {
    if f.sense_ok && f.sn.edge_ahead < p[0] + 2.0 * f.hv.max(1.0) && f.grounded_moving() {
        return world(0, Mag::Full, f.face.wrapping_add(0x4000));
    }
    fwd(f, 0)
}

/// Aim at the nearest object of a category (coin/star/enemy/pole...); jump when it is above.
fn approach_object(f: &F, p: &[f32]) -> Choice {
    let (cat, jump_range) = (p[0] as u8, p[1]);
    match f.nearest(cat) {
        Some(ob) => {
            let btn = if ob.dist < jump_range && ob.dy > 40.0 && !f.airborne() { lv(f, Req::Fresh, Req::Any, Req::Any) } else { 0 };
            world(btn, Mag::Full, f.face.wrapping_add(ob.rel_yaw))
        }
        None => fwd(f, 0),
    }
}

/// Steer away from the nearest hazard of a category.
fn avoid_object(f: &F, p: &[f32]) -> Choice {
    let (cat, radius) = (p[0] as u8, p[1]);
    match f.nearest(cat) {
        Some(ob) if ob.dist < radius + ob.radius => {
            let away = f.face.wrapping_add(ob.rel_yaw).wrapping_add(i16::MIN);
            world(0, Mag::Full, if f.has_tgt { blend(away, f.tgt_yaw, 0.5, 0x3000 as f32) } else { away })
        }
        _ => fwd(f, 0),
    }
}

fn wait(_f: &F, _p: &[f32]) -> Choice {
    neutral(0)
}

/// Among the jumps available now, the one whose landing is safe and furthest towards the target.
fn zone_safe_hop(f: &F, p: &[f32]) -> Choice {
    let margin = p[0];
    let mut best: Option<(f32, Opt)> = None;
    for x in options(f).into_iter().filter(|x| matches!(x.kind, Kind::Keep | Kind::Exit)) {
        let Some(pr) = opt_pred(f, &x) else { continue };
        if !pred_safe(&pr, margin) {
            continue;
        }
        let score = if f.has_tgt { -((f.tgt[0] - pr.pos[0]).hypot(f.tgt[2] - pr.pos[2])) + 5.0 * pr.fv } else { pr.fv };
        if best.is_none_or(|b| score > b.0) {
            best = Some((score, x));
        }
    }
    match best {
        Some((_, x)) => steered(f, opt_choice(f, &x), 1.0, 0x1000 as f32),
        None => keep_best(f, &[margin, 1.0, 0x1000 as f32]),
    }
}

/// Leave the chain on purpose: 0 stop, 1 slow to p[1], 2 jump, 3 dive, 4 ground pound.
fn exit_to(f: &F, p: &[f32]) -> Choice {
    match p[0] as u32 {
        0 => brake_stop(f, &[]),
        1 => slow_bleed(f, &[p[1]]),
        2 => exit_jump(f, &[1.0]),
        3 => exit_dive(f, &[29.0]),
        _ => ground_pound(f, &[100.0, 100000.0]),
    }
}

/// Wing cap: keep A on landings so triple jumps fly; metal cap in water walks to the target.
fn cap_aware(f: &F, p: &[f32]) -> Choice {
    let (wing, metal, _vanish) = f.caps();
    if wing && f.airborne() && f.fv > 20.0 {
        return fwd(f, lv(f, Req::Fresh, Req::Any, Req::Any));
    }
    if metal && f.in_water() {
        return world(0, Mag::Full, if f.has_tgt { f.tgt_yaw } else { f.face });
    }
    keep_best(f, &[p[0], 1.0, 0x1000 as f32])
}

/// Low health: steer clear of enemies; otherwise keep the speed.
fn health_guard(f: &F, p: &[f32]) -> Choice {
    if f.s.health < p[0] as i16 {
        if let Some(ob) = f.nearest(cat::ENEMY) {
            if ob.dist < 400.0 {
                return world(0, Mag::Full, f.face.wrapping_add(ob.rel_yaw).wrapping_add(i16::MIN));
            }
        }
    }
    keep_best(f, &[100.0, 1.0, 0x1000 as f32])
}

/// Holding an object: throw (B) within range of the target, else carry on.
fn carry(f: &F, p: &[f32]) -> Choice {
    if f.s.holding {
        if f.has_tgt && f.tgt_dist() < p[0] {
            return mk(lv(f, Req::Any, Req::Fresh, Req::Any), Mag::Neutral, Ang::Aim, Ref::Face);
        }
        return world(0, Mag::Full, if f.has_tgt { f.tgt_yaw } else { f.face });
    }
    keep_best(f, &[100.0, 1.0, 0x1000 as f32])
}

macro_rules! sk {
    ($name:expr, $ranges:expr, $f:ident) => {
        Box::new(Fs { name: $name, ranges: &$ranges, defs: &[], f: $f }) as Box<dyn Skill>
    };
    ($name:expr, $ranges:expr, $defs:expr, $f:ident) => {
        Box::new(Fs { name: $name, ranges: &$ranges, defs: &$defs, f: $f }) as Box<dyn Skill>
    };
}

/// All skills. The order is stable (ids for genomes).
pub fn library() -> Vec<Box<dyn Skill>> {
    vec![
        sk!("keep_best", [(0.0, 300.0), (0.0, 2.0), (256.0, 8192.0)], keep_best),
        sk!("hold_speed", [(0.0, 2.0), (256.0, 8192.0)], hold_speed),
        sk!("steer_target", [(0.0, 30.0), (0.0, 1.0)], steer_target),
        sk!("slide_optimise", [(0.0, 3.0), (256.0, 8192.0), (30.0, 100.0)], slide_optimise),
        sk!("rollout_cancel", [(0.0, 100.0)], [0.0], rollout_cancel),
        sk!("kick_chain", [(0.0, 80.0), (0.0, 300.0)], [30.0, 50.0], kick_chain),
        sk!("slide_kick_chain", [(0.0, 80.0), (0.0, 2.0)], [30.0, 1.0], slide_kick_chain),
        sk!("edge_dive_hop", [(20.0, 36.0), (40.0, 400.0)], edge_dive_hop),
        sk!("wall_kick_chain", [(60.0, 400.0)], wall_kick_chain),
        sk!("butt_slide_chain", [(20.0, 80.0), (0.0, 2.0)], butt_slide_chain),
        sk!("jump_chain", [(18.0, 40.0), (0.0, 1.0)], jump_chain),
        sk!("long_jump_builder", [(0.0, 1.0)], long_jump_builder),
        sk!("ground_pound", [(60.0, 600.0), (200.0, 5000.0)], ground_pound),
        sk!("brake_stop", [], brake_stop),
        sk!("slow_bleed", [(0.0, 48.0)], slow_bleed),
        sk!("turn_around", [(12000.0, 30000.0)], turn_around),
        sk!("exit_jump", [(0.0, 3.0)], exit_jump),
        sk!("exit_dive", [(29.0, 60.0)], exit_dive),
        sk!("ledge_grab", [(60.0, 400.0)], ledge_grab),
        sk!("pole", [(0.0, 6.0), (0.0, 1.0)], pole),
        sk!("swim_cadence", [(0.0, 6.0)], swim_cadence),
        sk!("avoid_wall", [(2.0, 12.0), (0.0, 1.0)], avoid_wall),
        sk!("avoid_edge", [(30.0, 300.0)], avoid_edge),
        sk!("approach_object", [(0.0, 6.0), (100.0, 500.0)], approach_object),
        sk!("avoid_object", [(0.0, 6.0), (50.0, 500.0)], avoid_object),
        sk!("wait", [], wait),
        sk!("zone_safe_hop", [(0.0, 300.0)], zone_safe_hop),
        sk!("exit_to", [(0.0, 4.0), (0.0, 48.0)], exit_to),
        sk!("cap_aware", [(0.0, 300.0)], cap_aware),
        sk!("health_guard", [(512.0, 2176.0)], health_guard),
        sk!("carry", [(100.0, 1500.0)], carry),
    ]
}

/// Object categories used by [`Sense::objs`] (sense.rs fills the same numbers).
pub mod cat {
    pub const ENEMY: u8 = 1;
    pub const COIN: u8 = 2;
    pub const POLE: u8 = 3;
    pub const STAR: u8 = 4;
    pub const PLATFORM: u8 = 5;
    pub const BOMB: u8 = 6;
}

/// Action ids (include/sm64.h), generated.
#[allow(dead_code)]
pub mod act {
    pub const IDLE: u32 = 0x0C400201;
    pub const WALKING: u32 = 0x04000440;
    pub const DECELERATING: u32 = 0x0400044A;
    pub const BRAKING: u32 = 0x04000445;
    pub const BRAKING_STOP: u32 = 0x0C00023D;
    pub const TURNING_AROUND: u32 = 0x00000443;
    pub const FINISH_TURNING_AROUND: u32 = 0x00000444;
    pub const CROUCHING: u32 = 0x0C008220;
    pub const START_CROUCHING: u32 = 0x0C008221;
    pub const STOP_CROUCHING: u32 = 0x0C008222;
    pub const CROUCH_SLIDE: u32 = 0x04808459;
    pub const SLIDE_KICK: u32 = 0x018008AA;
    pub const SLIDE_KICK_SLIDE: u32 = 0x0080045A;
    pub const SLIDE_KICK_SLIDE_STOP: u32 = 0x08000225;
    pub const BEGIN_SLIDING: u32 = 0x00000050;
    pub const BUTT_SLIDE: u32 = 0x00840452;
    pub const STOMACH_SLIDE: u32 = 0x008C0453;
    pub const DIVE_SLIDE: u32 = 0x00880456;
    pub const JUMP_LAND: u32 = 0x04000470;
    pub const FREEFALL_LAND: u32 = 0x04000471;
    pub const DOUBLE_JUMP_LAND: u32 = 0x04000472;
    pub const LONG_JUMP_LAND: u32 = 0x00000479;
    pub const TRIPLE_JUMP_LAND: u32 = 0x04000478;
    pub const SIDE_FLIP_LAND: u32 = 0x04000473;
    pub const BACKFLIP_LAND: u32 = 0x0400047A;
    pub const JUMP_LAND_STOP: u32 = 0x0C000230;
    pub const FREEFALL_LAND_STOP: u32 = 0x0C000232;
    pub const DOUBLE_JUMP_LAND_STOP: u32 = 0x0C000231;
    pub const LONG_JUMP_LAND_STOP: u32 = 0x0800023B;
    pub const TRIPLE_JUMP_LAND_STOP: u32 = 0x0800023A;
    pub const SIDE_FLIP_LAND_STOP: u32 = 0x0C000233;
    pub const BACKFLIP_LAND_STOP: u32 = 0x0800022F;
    pub const JUMP: u32 = 0x03000880;
    pub const DOUBLE_JUMP: u32 = 0x03000881;
    pub const TRIPLE_JUMP: u32 = 0x01000882;
    pub const LONG_JUMP: u32 = 0x03000888;
    pub const BACKFLIP: u32 = 0x01000883;
    pub const SIDE_FLIP: u32 = 0x01000887;
    pub const STEEP_JUMP: u32 = 0x03000885;
    pub const WALL_KICK_AIR: u32 = 0x03000886;
    pub const DIVE: u32 = 0x0188088A;
    pub const JUMP_KICK: u32 = 0x018008AC;
    pub const FREEFALL: u32 = 0x0100088C;
    pub const FORWARD_ROLLOUT: u32 = 0x010008A6;
    pub const GROUND_POUND: u32 = 0x008008A9;
    pub const GROUND_POUND_LAND: u32 = 0x0080023C;
    pub const AIR_HIT_WALL: u32 = 0x000008A7;
    pub const SOFT_BONK: u32 = 0x010208B6;
    pub const BACKWARD_AIR_KB: u32 = 0x010208B0;
    pub const FORWARD_AIR_KB: u32 = 0x010208B1;
    pub const BUTT_SLIDE_AIR: u32 = 0x0300088E;
    pub const PUNCHING: u32 = 0x00800380;
    pub const MOVE_PUNCHING: u32 = 0x00800457;
    pub const LEDGE_GRAB: u32 = 0x0800034B;
    pub const HANGING: u32 = 0x00200349;
    pub const START_HANGING: u32 = 0x08200348;
    pub const GRAB_POLE_SLOW: u32 = 0x00100341;
    pub const GRAB_POLE_FAST: u32 = 0x00100342;
    pub const HOLDING_POLE: u32 = 0x08100340;
    pub const CLIMBING_POLE: u32 = 0x00100343;
    pub const TOP_OF_POLE: u32 = 0x00100345;
    pub const TOP_OF_POLE_JUMP: u32 = 0x0300088D;
    pub const WATER_PLUNGE: u32 = 0x300022E2;
    pub const BREASTSTROKE: u32 = 0x300024D0;
    pub const FLUTTER_KICK: u32 = 0x300024D2;
    pub const SWIMMING_END: u32 = 0x300024D1;
    pub const WATER_IDLE: u32 = 0x380022C0;
    pub const WATER_JUMP: u32 = 0x01000889;
    pub const FIRST_PERSON: u32 = 0x0C000227;
    pub const STANDING_AGAINST_WALL: u32 = 0x0C400209;
    pub const HOLD_IDLE: u32 = 0x08000207;
    pub const HOLD_WALKING: u32 = 0x00000442;
    pub const HOLD_JUMP: u32 = 0x030008A0;
    pub const HOLD_FREEFALL: u32 = 0x010008A1;
    pub const LAVA_BOOST: u32 = 0x010208B7;
    pub const BACKWARD_GROUND_KB: u32 = 0x00020462;
    pub const FORWARD_GROUND_KB: u32 = 0x00020463;
    pub const TWIRLING: u32 = 0x108008A4;
    pub const FLYING_TRIPLE_JUMP: u32 = 0x03000894;
    pub const SPECIAL_TRIPLE_JUMP: u32 = 0x030008AF;
    pub const LEDGE_CLIMB_FAST: u32 = 0x0000054F;}
