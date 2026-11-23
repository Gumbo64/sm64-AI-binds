//! The language: typed trees returning a pad.
//!
//! Types: `Pad` (buttons + stick), `Btn`, `Stk`, `Bool`. `if` exists at the
//! first three; the slot's type is what makes an `if` a pad, button or stick
//! choice, so crossover only swaps subtrees of the same type.
//!
//! Text form (s-expressions, `;` comments; the parser takes the slot's type
//! from context):
//!
//! ```text
//! (if (act FREEFALL_LAND)
//!     (if (gt InAct 3.5) (pad Z (stick T0 0 1)) (pad A (stick T0 0 1)))
//!     (pad _ (stick T0 0 1)))
//! ```
//!
//! Buttons: letters of A B Z; upper case = held, lower case = "fresh": pressed
//! only if it wasn't held on the last frame (`if last_a {0} else {A}` in the
//! hand-written overlays). `_` none. Stick: `(stick REF off tilt)`, REF is
//! `T<i>` (towards point i: 0 the target, 1 the next waypoint, 2.. the
//! problem's landmarks), `face`, `vel` (motion direction), `world` (off is the
//! yaw), `turn<i>` (facing + the angle to point i, clamped to ±off).

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};
use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::{GamePad, St};
use rand::Rng;
use rand::rngs::SmallRng;

/// Useful actions (include/sm64.h), the ones movement goes through.
pub const ACTIONS: &[(&str, u32)] = &[
    ("IDLE", 0x0C40_0201),
    ("WALKING", 0x0400_0440),
    ("DECELERATING", 0x0400_044A),
    ("BRAKING", 0x0400_0445),
    ("BRAKING_STOP", 0x0C00_023D),
    ("TURNING_AROUND", 0x0000_0443),
    ("FINISH_TURNING_AROUND", 0x0000_0444),
    ("CROUCHING", 0x0C00_8220),
    ("START_CROUCHING", 0x0C00_8221),
    ("STOP_CROUCHING", 0x0C00_8222),
    ("CROUCH_SLIDE", 0x0480_8459),
    ("BUTT_SLIDE", 0x0084_0452),
    ("STOMACH_SLIDE", 0x008C_0453),
    ("DIVE_SLIDE", 0x0088_0456),
    ("SLIDE_KICK_SLIDE", 0x0080_045A),
    ("SLIDE_KICK", 0x0180_08AA),
    ("JUMP", 0x0300_0880),
    ("DOUBLE_JUMP", 0x0300_0881),
    ("TRIPLE_JUMP", 0x0100_0882),
    ("BACKFLIP", 0x0100_0883),
    ("STEEP_JUMP", 0x0300_0885),
    ("WALL_KICK_AIR", 0x0300_0886),
    ("SIDE_FLIP", 0x0100_0887),
    ("LONG_JUMP", 0x0300_0888),
    ("DIVE", 0x0188_088A),
    ("FREEFALL", 0x0100_088C),
    ("BUTT_SLIDE_AIR", 0x0300_088E),
    ("FORWARD_ROLLOUT", 0x0100_08A6),
    ("BACKWARD_ROLLOUT", 0x0100_08AD),
    ("AIR_HIT_WALL", 0x0000_08A7),
    ("GROUND_POUND", 0x0080_08A9),
    ("JUMP_KICK", 0x0180_08AC),
    ("SOFT_BONK", 0x0102_08B6),
    ("PUNCHING", 0x0080_0380),
    ("MOVE_PUNCHING", 0x0080_0457),
    ("JUMP_LAND", 0x0400_0470),
    ("FREEFALL_LAND", 0x0400_0471),
    ("DOUBLE_JUMP_LAND", 0x0400_0472),
    ("SIDE_FLIP_LAND", 0x0400_0473),
    ("TRIPLE_JUMP_LAND", 0x0400_0478),
    ("LONG_JUMP_LAND", 0x0000_0479),
    ("BACKFLIP_LAND", 0x0400_047A),
    ("JUMP_LAND_STOP", 0x0C00_0230),
    ("DOUBLE_JUMP_LAND_STOP", 0x0C00_0231),
    ("FREEFALL_LAND_STOP", 0x0C00_0232),
    ("SIDE_FLIP_LAND_STOP", 0x0C00_0233),
    ("TRIPLE_JUMP_LAND_STOP", 0x0800_023A),
    ("LONG_JUMP_LAND_STOP", 0x0800_023B),
    ("BACKFLIP_LAND_STOP", 0x0800_022F),
    ("GROUND_POUND_LAND", 0x0080_023C),
    ("LEDGE_GRAB", 0x0800_034B),
];

/// Action flags (mask tests).
pub const FLAGS: &[(&str, u32)] = &[
    ("STATIONARY", 0x200),
    ("MOVING", 0x400),
    ("AIR", 0x800),
    ("BUTT_OR_STOMACH", 0x0004_0000),
    ("DIVING", 0x0008_0000),
    ("ATTACKING", 0x0080_0000),
    ("AIRSTEP_WIND", 0x0100_0000),
    ("CONTROL_JUMP_HEIGHT", 0x0200_0000),
];

/// `ACT_GROUP` values (bits 6-8).
pub const GROUPS: &[(&str, u32)] = &[("GSTATIONARY", 0x000), ("GMOVING", 0x040), ("GAIRBORNE", 0x080), ("GCUTSCENE", 0x100), ("GAUTOMATIC", 0x140)];

/// Floor types worth a test (surface_terrains.h).
pub const FLOOR_TYPES: &[i16] = &[0x00, 0x13, 0x14, 0x15, 0x29, 0x2A];

/// How many points a state test or stick can refer to.
pub const MAX_POINTS: usize = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Ty {
    Pad,
    Btn,
    Stk,
    Bool,
}

/// State terminals. The ones with a point take its index (0 the target).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Var {
    /// Frames in the current action, restarted by a B press (the overlays').
    InAct,
    /// The game's actionTimer.
    Timer,
    Fv,
    Hsp,
    Vy,
    /// Height above the floor.
    Hgt,
    /// Floor normal y.
    Ny,
    /// The floor's downhill component along the facing (+ = downhill ahead).
    Down,
    Dist(u8),
    /// |angle| from the facing to the point, degrees.
    Ang(u8),
    /// Signed angle from the facing to the point, degrees.
    SAng(u8),
    /// |angle| from the motion direction to the point, degrees.
    VAng(u8),
    /// Speed towards the point.
    Close(u8),
    /// Distance from the ballistic landing point (at the current floor height)
    /// to the point: where a jump in progress comes down.
    Land(u8),
    /// Distance from where a double jump from here (from the landing, if in
    /// the air) comes down to the point: range 37.4 fv − 500 along the motion
    /// (lakituskip.rs `dj_land_z`).
    Dj(u8),
    /// The point's height minus Mario's.
    Dy(u8),
    /// The reference's speed into the current waypoint (followers).
    Pace,
    /// Distance to the reference at the aligned frame, 3D.
    RefDist,
    /// The reference's forward speed at the aligned frame.
    RefFv,
    X,
    Y,
    Z,
}

/// Stick reference.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ref {
    To(u8),
    Face,
    Vel,
    World,
    Turn(u8),
    /// The reference's stick direction (off added); tilt scales its tilt.
    RefSt,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Op {
    If(Ty),
    /// `(pad BTN STK)`
    Make,
    Btn { hold: u8, fresh: u8 },
    /// The reference's A, B, Z at the aligned frame.
    RefBtn,
    RefHeld(u8),
    RefAct(u8),
    Stick { r: Ref, off: i16, tilt: f32 },
    And,
    Or,
    Not,
    Act(u8),
    Flag(u8),
    Group(u8),
    Floor(i16),
    Last(u8),
    Lt(Var, f32),
    Gt(Var, f32),
    /// Automatically defined functions (no arguments).
    CallP(u8),
    CallB(u8),
}

impl Op {
    pub fn ty(&self) -> Ty {
        match self {
            Op::If(t) => *t,
            Op::Make | Op::CallP(_) => Ty::Pad,
            Op::Btn { .. } | Op::RefBtn => Ty::Btn,
            Op::Stick { .. } => Ty::Stk,
            _ => Ty::Bool,
        }
    }

    fn has_constant(&self) -> bool {
        !matches!(self, Op::If(_) | Op::Make | Op::And | Op::Or | Op::Not)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Node {
    pub op: Op,
    pub kids: Vec<Node>,
}

impl Node {
    pub fn size(&self) -> usize {
        1 + self.kids.iter().map(Node::size).sum::<usize>()
    }

    pub fn depth(&self) -> usize {
        1 + self.kids.iter().map(Node::depth).max().unwrap_or(0)
    }

    pub fn ty(&self) -> Ty {
        self.op.ty()
    }

    /// Every node's path.
    pub fn points(&self, path: &mut Vec<u16>, out: &mut Vec<(Vec<u16>, Ty, bool)>) {
        out.push((path.clone(), self.ty(), !self.kids.is_empty()));
        for (i, k) in self.kids.iter().enumerate() {
            path.push(i as u16);
            k.points(path, out);
            path.pop();
        }
    }

    pub fn at(&self, path: &[u16]) -> &Node {
        path.iter().fold(self, |n, &i| &n.kids[i as usize])
    }

    pub fn at_mut(&mut self, path: &[u16]) -> &mut Node {
        path.iter().fold(self, |n, &i| &mut n.kids[i as usize])
    }

    pub fn has_call(&self) -> bool {
        matches!(self.op, Op::CallP(_) | Op::CallB(_)) || self.kids.iter().any(Node::has_call)
    }
}

/// A program: the main (pad) branch and its defined functions.
#[derive(Clone, PartialEq, Debug)]
pub struct Prog {
    pub main: Node,
    pub adf_p: Vec<Node>,
    pub adf_b: Vec<Node>,
}

/// What a program sees on a frame.
pub struct Ctx<'a> {
    pub s: &'a St,
    pub last: GamePad,
    pub in_act: u32,
    pub pts: [[f32; 3]; MAX_POINTS],
    /// The reference's pace into the current target (followers).
    pub pace: f32,
    /// The reference at its frame nearest Mario (followers).
    pub rf: RefNow,
}

/// The reference trace at the aligned frame: what it pressed and where it was.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefNow {
    pub btn: u16,
    /// World yaw of its stick.
    pub yaw: i16,
    pub tilt: f32,
    pub act: u32,
    pub fv: f32,
    pub dist: f32,
}

fn rad(yaw: i32) -> f32 {
    yaw as f32 * std::f32::consts::TAU / 65536.0
}

fn yaw_to_deg(y: i16) -> f32 {
    y as f32 * 360.0 / 65536.0
}

fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
}

/// The stick for a world yaw and tilt (moves::arcs::stick's math).
pub fn stick(cam_yaw: i16, yaw: i16, tilt: f32) -> (i8, i8) {
    obrrt_objectives::moves::arcs::stick(cam_yaw, yaw, tilt)
}

impl Ctx<'_> {
    fn delta(&self, p: u8) -> (f32, f32) {
        let t = self.pts[p as usize];
        (t[0] - self.s.pos[0], t[2] - self.s.pos[2])
    }

    fn yaw_to(&self, p: u8) -> i16 {
        let (dx, dz) = self.delta(p);
        yaw_of(dx, dz)
    }

    fn var(&self, v: Var) -> f32 {
        let s = self.s;
        match v {
            Var::InAct => self.in_act as f32,
            Var::Timer => s.action_timer as f32,
            Var::Fv => s.forward_vel,
            Var::Hsp => s.vel[0].hypot(s.vel[2]),
            Var::Vy => s.vel[1],
            Var::Hgt => s.pos[1] - s.floor_y,
            Var::Ny => s.floor_normal[1],
            Var::Down => {
                let y = rad(s.yaw as i32);
                s.floor_normal[0] * y.sin() + s.floor_normal[2] * y.cos()
            }
            Var::Dist(p) => {
                let (dx, dz) = self.delta(p);
                dx.hypot(dz)
            }
            Var::Ang(p) => yaw_to_deg((self.yaw_to(p) as i32 - s.yaw as i32) as i16).abs(),
            Var::SAng(p) => yaw_to_deg((self.yaw_to(p) as i32 - s.yaw as i32) as i16),
            Var::VAng(p) => yaw_to_deg((self.yaw_to(p) as i32 - yaw_of(s.vel[0], s.vel[2]) as i32) as i16).abs(),
            Var::Close(p) => {
                let (dx, dz) = self.delta(p);
                (s.vel[0] * dx + s.vel[2] * dz) / dx.hypot(dz).max(1.0)
            }
            Var::Land(p) | Var::Dj(p) => {
                // Ballistic with gravity 4 down to the floor below (ground: here).
                let (mut y, mut vy, mut x, mut z) = (s.pos[1], s.vel[1], s.pos[0], s.pos[2]);
                if s.action & 0x800 != 0 {
                    for _ in 0..60 {
                        y += vy;
                        x += s.vel[0];
                        z += s.vel[2];
                        if y <= s.floor_y && vy < 0.0 {
                            break;
                        }
                        vy = (vy - 4.0).max(-75.0);
                    }
                }
                if let Var::Dj(_) = v {
                    let h = s.vel[0].hypot(s.vel[2]).max(1.0);
                    let range = (37.4 * s.forward_vel - 500.0).max(0.0);
                    x += range * s.vel[0] / h;
                    z += range * s.vel[2] / h;
                }
                let t = self.pts[p as usize];
                (t[0] - x).hypot(t[2] - z)
            }
            Var::Dy(p) => self.pts[p as usize][1] - s.pos[1],
            Var::Pace => self.pace,
            Var::RefDist => self.rf.dist,
            Var::RefFv => self.rf.fv,
            Var::X => s.pos[0],
            Var::Y => s.pos[1],
            Var::Z => s.pos[2],
        }
    }
}

/// Branch counts per `if` (keys: node addresses), for pruning.
pub type Trace = RefCell<HashMap<*const Node, (u32, u32)>>;

fn btn_mask(bits: u8) -> u16 {
    let mut m = 0;
    if bits & 1 != 0 {
        m |= A_BUTTON;
    }
    if bits & 2 != 0 {
        m |= B_BUTTON;
    }
    if bits & 4 != 0 {
        m |= Z_TRIG;
    }
    m
}

impl Prog {
    pub fn size(&self) -> usize {
        self.main.size() + self.adf_p.iter().map(Node::size).sum::<usize>() + self.adf_b.iter().map(Node::size).sum::<usize>()
    }

    pub fn depth(&self) -> usize {
        self.adf_p.iter().chain(&self.adf_b).map(Node::depth).max().unwrap_or(0).max(self.main.depth())
    }

    /// The frame's pad.
    pub fn pad(&self, c: &Ctx, tr: Option<&Trace>) -> GamePad {
        let (b, st) = self.ev_pad(&self.main, c, tr);
        GamePad::new(b, st.0, st.1)
    }

    fn cond(&self, n: &Node, c: &Ctx, tr: Option<&Trace>) -> bool {
        let v = self.ev_bool(&n.kids[0], c, tr);
        if let Some(t) = tr {
            let mut m = t.borrow_mut();
            let e = m.entry(n as *const Node).or_insert((0, 0));
            if v {
                e.0 += 1;
            } else {
                e.1 += 1;
            }
        }
        v
    }

    fn ev_pad(&self, n: &Node, c: &Ctx, tr: Option<&Trace>) -> (u16, (i8, i8)) {
        match &n.op {
            Op::If(_) => {
                if self.cond(n, c, tr) {
                    self.ev_pad(&n.kids[1], c, tr)
                } else {
                    self.ev_pad(&n.kids[2], c, tr)
                }
            }
            Op::Make => (self.ev_btn(&n.kids[0], c, tr), self.ev_stk(&n.kids[1], c, tr)),
            Op::CallP(i) => self.ev_pad(&self.adf_p[*i as usize], c, tr),
            o => panic!("not a pad: {o:?}"),
        }
    }

    fn ev_btn(&self, n: &Node, c: &Ctx, tr: Option<&Trace>) -> u16 {
        match &n.op {
            Op::If(_) => {
                if self.cond(n, c, tr) {
                    self.ev_btn(&n.kids[1], c, tr)
                } else {
                    self.ev_btn(&n.kids[2], c, tr)
                }
            }
            Op::Btn { hold, fresh } => {
                let f = btn_mask(*fresh);
                btn_mask(*hold) | (f & !c.last.button)
            }
            Op::RefBtn => c.rf.btn & (A_BUTTON | B_BUTTON | Z_TRIG),
            o => panic!("not buttons: {o:?}"),
        }
    }

    fn ev_stk(&self, n: &Node, c: &Ctx, tr: Option<&Trace>) -> (i8, i8) {
        match &n.op {
            Op::If(_) => {
                if self.cond(n, c, tr) {
                    self.ev_stk(&n.kids[1], c, tr)
                } else {
                    self.ev_stk(&n.kids[2], c, tr)
                }
            }
            Op::Stick { r, off, tilt } => {
                let s = c.s;
                if let Ref::RefSt = r {
                    return stick(s.cam_yaw, c.rf.yaw.wrapping_add(*off), c.rf.tilt * *tilt);
                }
                let yaw = match r {
                    Ref::RefSt => unreachable!(),
                    Ref::To(p) => c.yaw_to(*p).wrapping_add(*off),
                    Ref::Face => s.yaw.wrapping_add(*off),
                    Ref::Vel => yaw_of(s.vel[0], s.vel[2]).wrapping_add(*off),
                    Ref::World => *off,
                    Ref::Turn(p) => {
                        let m = off.unsigned_abs().min(0x7FFF) as i16;
                        let d = c.yaw_to(*p).wrapping_sub(s.yaw).clamp(-m, m);
                        s.yaw.wrapping_add(d)
                    }
                };
                stick(s.cam_yaw, yaw, *tilt)
            }
            o => panic!("not a stick: {o:?}"),
        }
    }

    fn ev_bool(&self, n: &Node, c: &Ctx, tr: Option<&Trace>) -> bool {
        let s = c.s;
        match &n.op {
            Op::And => self.ev_bool(&n.kids[0], c, tr) && self.ev_bool(&n.kids[1], c, tr),
            Op::Or => self.ev_bool(&n.kids[0], c, tr) || self.ev_bool(&n.kids[1], c, tr),
            Op::Not => !self.ev_bool(&n.kids[0], c, tr),
            Op::Act(i) => s.action == ACTIONS[*i as usize].1,
            Op::Flag(i) => s.action & FLAGS[*i as usize].1 != 0,
            Op::Group(i) => s.action & 0x1C0 == GROUPS[*i as usize].1,
            Op::Floor(t) => s.floor_type == *t,
            Op::Last(b) => {
                let m = btn_mask(*b);
                c.last.button & m == m
            }
            Op::RefHeld(b) => {
                let m = btn_mask(*b);
                c.rf.btn & m == m
            }
            Op::RefAct(i) => c.rf.act == ACTIONS[*i as usize].1,
            Op::Lt(v, x) => c.var(*v) < *x,
            Op::Gt(v, x) => c.var(*v) > *x,
            Op::CallB(i) => self.ev_bool(&self.adf_b[*i as usize], c, tr),
            o => panic!("not a test: {o:?}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Random trees.

/// What the random generator may use.
#[derive(Clone, Copy, Debug)]
pub struct GenCfg {
    pub points: u8,
    /// X, Y, Z tests (absolute positions).
    pub use_pos: bool,
    /// Sticks at absolute world yaws (a program can then memorise a route's line).
    pub use_world: bool,
    /// Follower tests: Dy, Pace.
    pub follow: bool,
    /// Follower terminals on the reference trace (its inputs, action, position).
    pub use_ref: bool,
    pub adf_p: u8,
    pub adf_b: u8,
}

fn pick_var(g: &GenCfg, r: &mut SmallRng) -> Var {
    let p = if g.points <= 1 || r.random_bool(0.6) { 0 } else { r.random_range(0..g.points) };
    let n = if g.use_pos { 16 } else { 13 };
    if g.use_ref && r.random_bool(0.15) {
        return if r.random_bool(0.6) { Var::RefDist } else { Var::RefFv };
    }
    if g.follow && r.random_bool(0.2) {
        return if r.random_bool(0.6) { Var::Dy(p) } else { Var::Pace };
    }
    match r.random_range(0..n) {
        0 | 1 => Var::InAct,
        2 => Var::Timer,
        3 => Var::Fv,
        4 => Var::Hsp,
        5 => Var::Vy,
        6 => Var::Hgt,
        7 => Var::Ny,
        8 => Var::Down,
        9 => Var::Dist(p),
        10 => Var::Ang(p),
        11 => Var::Close(p),
        12 => {
            match r.random_range(0..4) {
                0 => Var::Land(p),
                1 => Var::SAng(p),
                2 => Var::VAng(p),
                _ => Var::Dj(p),
            }
        }
        13 => Var::X,
        14 => Var::Y,
        _ => Var::Z,
    }
}

fn threshold(v: Var, r: &mut SmallRng) -> f32 {
    let x: f32 = match v {
        Var::InAct => r.random_range(1..12) as f32 + 0.5,
        Var::Timer => r.random_range(0..30) as f32 + 0.5,
        Var::Fv | Var::Hsp => r.random_range(0.0..120.0),
        Var::Vy => r.random_range(-75.0..60.0),
        Var::Hgt => {
            let e: f32 = r.random_range(0.0..6.0);
            e.exp() * 1.5
        }
        Var::Ny => r.random_range(0.5..1.0),
        Var::Down => r.random_range(-0.7..0.7),
        Var::Dist(_) | Var::Land(_) | Var::Dj(_) => {
            let e: f32 = r.random_range(3.0..9.0);
            e.exp()
        }
        Var::Ang(_) | Var::VAng(_) => r.random_range(0.0..180.0),
        Var::SAng(_) => r.random_range(-180.0..180.0),
        Var::Close(_) => r.random_range(-30.0..120.0),
        Var::Dy(_) => r.random_range(-500.0..500.0),
        Var::Pace | Var::RefFv => r.random_range(5.0..100.0),
        Var::RefDist => {
            let e: f32 = r.random_range(2.5..7.0);
            e.exp()
        }
        Var::X | Var::Z => r.random_range(-8000.0..8000.0),
        Var::Y => r.random_range(-500.0..5000.0),
    };
    (x * 100.0).round() / 100.0
}

fn perturb_threshold(v: Var, x: f32, r: &mut SmallRng) -> f32 {
    let scale = match v {
        Var::InAct | Var::Timer => {
            return (x + r.random_range(-2i32..=2) as f32).max(0.5);
        }
        Var::Fv | Var::Hsp | Var::Close(_) | Var::Vy | Var::Pace | Var::RefFv => 6.0,
        Var::RefDist => 0.2 * x.abs().max(20.0),
        Var::Dy(_) => 40.0,
        Var::Hgt => 15.0,
        Var::Ny | Var::Down => 0.05,
        Var::Dist(_) | Var::Land(_) | Var::Dj(_) => 0.15 * x.abs().max(50.0),
        Var::Ang(_) | Var::VAng(_) | Var::SAng(_) => 12.0,
        Var::X | Var::Y | Var::Z => 200.0,
    };
    let n = r.random_range(-1.0f32..1.0) + r.random_range(-1.0f32..1.0);
    ((x + n * scale) * 100.0).round() / 100.0
}

fn pick_point(g: &GenCfg, r: &mut SmallRng) -> u8 {
    if g.points <= 1 || r.random_bool(0.6) { 0 } else { r.random_range(0..g.points) }
}

fn rand_offset(r: &mut SmallRng) -> i16 {
    const O: [i16; 9] = [0, 0, 0, 0x400, -0x400, 0x1000, -0x1000, 0x4000, -0x4000];
    let b = O[r.random_range(0..O.len())];
    if r.random_bool(0.2) { b.wrapping_add(r.random_range(-0x800..0x800)) } else { b }
}

fn rand_tilt(r: &mut SmallRng) -> f32 {
    const T: [f32; 6] = [1.0, 1.0, 1.0, 0.38, 0.6, 0.0];
    T[r.random_range(0..T.len())]
}

fn rand_stick(g: &GenCfg, r: &mut SmallRng) -> Op {
    if g.use_ref && r.random_bool(0.3) {
        return Op::Stick { r: Ref::RefSt, off: 0, tilt: 1.0 };
    }
    if r.random_bool(0.45) {
        return Op::Stick { r: Ref::To(pick_point(g, r)), off: 0, tilt: 1.0 };
    }
    let rf = match r.random_range(0..6) {
        0 | 1 => Ref::To(pick_point(g, r)),
        2 => Ref::Face,
        3 => Ref::Vel,
        4 if g.use_world => Ref::World,
        4 => Ref::Face,
        _ => Ref::Turn(pick_point(g, r)),
    };
    let off = match rf {
        Ref::World => r.random_range(-0x8000i32..0x8000) as i16,
        Ref::Turn(_) => [0x200i16, 0x400, 0x800, 0x1000][r.random_range(0..4)],
        _ => rand_offset(r),
    };
    Op::Stick { r: rf, off, tilt: rand_tilt(r) }
}

fn rand_btn(r: &mut SmallRng) -> Op {
    let (mut hold, mut fresh) = (0u8, 0u8);
    for b in [1u8, 2, 4] {
        match r.random_range(0..3) {
            0 => {}
            1 => hold |= b,
            _ => fresh |= b,
        }
    }
    Op::Btn { hold, fresh }
}

fn rand_test(g: &GenCfg, r: &mut SmallRng) -> Op {
    if g.use_ref && r.random_bool(0.2) {
        return if r.random_bool(0.6) { Op::RefHeld(1 << r.random_range(0..3)) } else { Op::RefAct(r.random_range(0..ACTIONS.len()) as u8) };
    }
    match r.random_range(0..100) {
        0..45 => Op::Act(r.random_range(0..ACTIONS.len()) as u8),
        45..52 => Op::Flag(r.random_range(0..FLAGS.len()) as u8),
        52..55 => Op::Group(r.random_range(0..GROUPS.len()) as u8),
        55..60 => Op::Floor(FLOOR_TYPES[r.random_range(0..FLOOR_TYPES.len())]),
        60..72 => Op::Last(1 << r.random_range(0..3)),
        n => {
            if g.adf_b > 0 && n < 76 {
                return Op::CallB(r.random_range(0..g.adf_b));
            }
            let v = pick_var(g, r);
            let x = threshold(v, r);
            if r.random_bool(0.5) { Op::Lt(v, x) } else { Op::Gt(v, x) }
        }
    }
}

/// A random tree of type `ty`, depth ≤ `d` (full: functions down to the limit).
pub fn random(ty: Ty, d: usize, full: bool, g: &GenCfg, calls: bool, r: &mut SmallRng) -> Node {
    let gc = GenCfg { adf_b: if calls { g.adf_b } else { 0 }, adf_p: if calls { g.adf_p } else { 0 }, ..*g };
    match ty {
        Ty::Bool => {
            let func = if d <= 1 { false } else if full { true } else { r.random_bool(0.2) };
            if func {
                match r.random_range(0..5) {
                    0 | 1 => Node { op: Op::And, kids: vec![random(Ty::Bool, d - 1, full, g, calls, r), random(Ty::Bool, d - 1, full, g, calls, r)] },
                    2 | 3 => Node { op: Op::Or, kids: vec![random(Ty::Bool, d - 1, full, g, calls, r), random(Ty::Bool, d - 1, full, g, calls, r)] },
                    _ => Node { op: Op::Not, kids: vec![random(Ty::Bool, d - 1, full, g, calls, r)] },
                }
            } else {
                Node { op: rand_test(&gc, r), kids: vec![] }
            }
        }
        Ty::Btn | Ty::Stk => {
            let func = d > 2 && if full { r.random_bool(0.5) } else { r.random_bool(0.2) };
            if func {
                let c = random(Ty::Bool, (d - 1).min(3), full, g, calls, r);
                let a = random(ty, d - 1, full, g, calls, r);
                let b = random(ty, d - 1, full, g, calls, r);
                Node { op: Op::If(ty), kids: vec![c, a, b] }
            } else if ty == Ty::Btn {
                if g.use_ref && r.random_bool(0.3) {
                    return Node { op: Op::RefBtn, kids: vec![] };
                }
                Node { op: rand_btn(r), kids: vec![] }
            } else {
                Node { op: rand_stick(g, r), kids: vec![] }
            }
        }
        Ty::Pad => {
            if d > 2 && gc.adf_p > 0 && r.random_bool(0.1) {
                return Node { op: Op::CallP(r.random_range(0..gc.adf_p)), kids: vec![] };
            }
            let func = d > 3 && (full || r.random_bool(0.75));
            if func {
                let c = random(Ty::Bool, (d - 1).min(3), full && r.random_bool(0.3), g, calls, r);
                let a = random(Ty::Pad, d - 1, full, g, calls, r);
                let b = random(Ty::Pad, d - 1, full, g, calls, r);
                Node { op: Op::If(Ty::Pad), kids: vec![c, a, b] }
            } else {
                let b = random(Ty::Btn, (d - 1).min(3), false, g, calls, r);
                let s = random(Ty::Stk, (d - 1).min(3), false, g, calls, r);
                Node { op: Op::Make, kids: vec![b, s] }
            }
        }
    }
}

/// Changes one node's constants.
pub fn perturb(op: &mut Op, g: &GenCfg, r: &mut SmallRng) {
    match op {
        Op::Btn { .. } => {
            let (mut h, mut f) = if let Op::Btn { hold, fresh } = op { (*hold, *fresh) } else { unreachable!() };
            let b = 1u8 << r.random_range(0..3);
            h &= !b;
            f &= !b;
            match r.random_range(0..3) {
                0 => {}
                1 => h |= b,
                _ => f |= b,
            }
            *op = Op::Btn { hold: h, fresh: f };
        }
        Op::Stick { r: rf, off, tilt } => match r.random_range(0..4) {
            0 => *off = off.wrapping_add(r.random_range(-0x400..0x400)),
            1 => *off = 0,
            2 => *tilt = rand_tilt(r),
            _ => {
                if let Op::Stick { r: nr, .. } = rand_stick(g, r) {
                    *rf = nr;
                }
            }
        },
        Op::Act(i) => *i = r.random_range(0..ACTIONS.len()) as u8,
        Op::Flag(i) => *i = r.random_range(0..FLAGS.len()) as u8,
        Op::Group(i) => *i = r.random_range(0..GROUPS.len()) as u8,
        Op::Floor(t) => *t = FLOOR_TYPES[r.random_range(0..FLOOR_TYPES.len())],
        Op::Last(b) | Op::RefHeld(b) => *b = 1 << r.random_range(0..3),
        Op::RefAct(i) => *i = r.random_range(0..ACTIONS.len()) as u8,
        Op::Lt(v, x) | Op::Gt(v, x) => *x = perturb_threshold(*v, *x, r),
        _ => {}
    }
}

/// A node with constants, if any, at random.
pub fn constant_points(n: &Node, path: &mut Vec<u16>, out: &mut Vec<Vec<u16>>) {
    if n.op.has_constant() {
        out.push(path.clone());
    }
    for (i, k) in n.kids.iter().enumerate() {
        path.push(i as u16);
        constant_points(k, path, out);
        path.pop();
    }
}

// ---------------------------------------------------------------------------
// Text.

fn var_name(v: Var) -> String {
    match v {
        Var::InAct => "InAct".into(),
        Var::Timer => "Timer".into(),
        Var::Fv => "Fv".into(),
        Var::Hsp => "Hsp".into(),
        Var::Vy => "Vy".into(),
        Var::Hgt => "Hgt".into(),
        Var::Ny => "Ny".into(),
        Var::Down => "Down".into(),
        Var::Dist(p) => format!("Dist{p}"),
        Var::Ang(p) => format!("Ang{p}"),
        Var::SAng(p) => format!("SAng{p}"),
        Var::VAng(p) => format!("VAng{p}"),
        Var::Close(p) => format!("Close{p}"),
        Var::Land(p) => format!("Land{p}"),
        Var::Dj(p) => format!("Dj{p}"),
        Var::Dy(p) => format!("Dy{p}"),
        Var::Pace => "Pace".into(),
        Var::RefDist => "RefDist".into(),
        Var::RefFv => "RefFv".into(),
        Var::X => "X".into(),
        Var::Y => "Y".into(),
        Var::Z => "Z".into(),
    }
}

fn parse_var(s: &str) -> Result<Var> {
    Ok(match s {
        "InAct" => Var::InAct,
        "Timer" => Var::Timer,
        "Fv" => Var::Fv,
        "Hsp" => Var::Hsp,
        "Vy" => Var::Vy,
        "Hgt" => Var::Hgt,
        "Ny" => Var::Ny,
        "Down" => Var::Down,
        "Pace" => Var::Pace,
        "RefDist" => Var::RefDist,
        "RefFv" => Var::RefFv,
        "X" => Var::X,
        "Y" => Var::Y,
        "Z" => Var::Z,
        _ => {
            let i = s.find(|c: char| c.is_ascii_digit()).ok_or_else(|| anyhow!("var {s}"))?;
            let p: u8 = s[i..].parse()?;
            match &s[..i] {
                "Dist" => Var::Dist(p),
                "Ang" => Var::Ang(p),
                "SAng" => Var::SAng(p),
                "VAng" => Var::VAng(p),
                "Close" => Var::Close(p),
                "Land" => Var::Land(p),
                "Dj" => Var::Dj(p),
                "Dy" => Var::Dy(p),
                _ => bail!("var {s}"),
            }
        }
    })
}

fn btn_text(hold: u8, fresh: u8) -> String {
    let mut t = String::new();
    for (b, u, l) in [(1u8, 'A', 'a'), (2, 'B', 'b'), (4, 'Z', 'z')] {
        if hold & b != 0 {
            t.push(u);
        } else if fresh & b != 0 {
            t.push(l);
        }
    }
    if t.is_empty() { "_".into() } else { t }
}

fn ref_text(r: Ref) -> String {
    match r {
        Ref::To(p) => format!("T{p}"),
        Ref::Face => "face".into(),
        Ref::Vel => "vel".into(),
        Ref::World => "world".into(),
        Ref::Turn(p) => format!("turn{p}"),
        Ref::RefSt => "refst".into(),
    }
}

pub fn sexp(n: &Node) -> String {
    let mut s = String::new();
    write_sexp(n, &mut s);
    s
}

fn write_sexp(n: &Node, o: &mut String) {
    let kids = |o: &mut String, name: &str| {
        o.push('(');
        o.push_str(name);
        for k in &n.kids {
            o.push(' ');
            write_sexp(k, o);
        }
        o.push(')');
    };
    match &n.op {
        Op::If(_) => kids(o, "if"),
        Op::Make => kids(o, "pad"),
        Op::And => kids(o, "and"),
        Op::Or => kids(o, "or"),
        Op::Not => kids(o, "not"),
        Op::Btn { hold, fresh } => o.push_str(&btn_text(*hold, *fresh)),
        Op::RefBtn => o.push_str("ref"),
        Op::RefHeld(b) => o.push_str(&format!("(refheld {})", btn_text(*b, 0))),
        Op::RefAct(i) => o.push_str(&format!("(refact {})", ACTIONS[*i as usize].0)),
        Op::Stick { r, off, tilt } => o.push_str(&format!("(stick {} {} {})", ref_text(*r), off, tilt)),
        Op::Act(i) => o.push_str(&format!("(act {})", ACTIONS[*i as usize].0)),
        Op::Flag(i) => o.push_str(&format!("(flag {})", FLAGS[*i as usize].0)),
        Op::Group(i) => o.push_str(&format!("(flag {})", GROUPS[*i as usize].0)),
        Op::Floor(t) => o.push_str(&format!("(floor {t})")),
        Op::Last(b) => o.push_str(&format!("(last {})", btn_text(*b, 0))),
        Op::Lt(v, x) => o.push_str(&format!("(lt {} {})", var_name(*v), x)),
        Op::Gt(v, x) => o.push_str(&format!("(gt {} {})", var_name(*v), x)),
        Op::CallP(i) => o.push_str(&format!("(callp {i})")),
        Op::CallB(i) => o.push_str(&format!("(callb {i})")),
    }
}

impl Prog {
    /// One line per branch; `from_text` reads it back.
    pub fn text(&self) -> String {
        let mut o = String::new();
        for (i, a) in self.adf_p.iter().enumerate() {
            o.push_str(&format!("(defp {i} {})\n", sexp(a)));
        }
        for (i, a) in self.adf_b.iter().enumerate() {
            o.push_str(&format!("(defb {i} {})\n", sexp(a)));
        }
        o.push_str(&sexp(&self.main));
        o.push('\n');
        o
    }

    pub fn from_text(t: &str) -> Result<Prog> {
        let mut toks = tokens(t);
        toks.reverse();
        let mut p = Prog { main: Node { op: Op::Make, kids: vec![] }, adf_p: vec![], adf_b: vec![] };
        let mut main = None;
        // Defined macros are expanded by the text: `(let NAME tree)` at top
        // level for seeds: NAME then stands for the tree wherever it is used.
        let mut lets: HashMap<String, Vec<String>> = HashMap::new();
        while let Some(tk) = toks.last().cloned() {
            if tk != "(" {
                bail!("expected ( at top level, got {tk}");
            }
            let head = toks.get(toks.len().wrapping_sub(2)).cloned().unwrap_or_default();
            match head.as_str() {
                "let" => {
                    toks.pop();
                    toks.pop();
                    let name = toks.pop().ok_or_else(|| anyhow!("let name"))?;
                    let body = take_group(&mut toks)?;
                    expect(&mut toks, ")")?;
                    let body = expand(&body, &lets);
                    lets.insert(name, body);
                }
                "defp" | "defb" => {
                    toks.pop();
                    toks.pop();
                    toks.pop();
                    let body = take_group(&mut toks)?;
                    expect(&mut toks, ")")?;
                    let mut b: Vec<String> = expand(&body, &lets);
                    b.reverse();
                    if head == "defp" {
                        p.adf_p.push(parse(&mut b, Ty::Pad)?);
                    } else {
                        p.adf_b.push(parse(&mut b, Ty::Bool)?);
                    }
                }
                _ => {
                    let body = take_group(&mut toks)?;
                    let mut b: Vec<String> = expand(&body, &lets);
                    b.reverse();
                    main = Some(parse(&mut b, Ty::Pad)?);
                }
            }
        }
        p.main = main.ok_or_else(|| anyhow!("no main branch"))?;
        Ok(p)
    }

    /// Readable form, for the report.
    pub fn pretty(&self) -> String {
        let mut o = String::new();
        for (i, a) in self.adf_b.iter().enumerate() {
            o.push_str(&format!("test{i} := {}\n", pb(a)));
        }
        for (i, a) in self.adf_p.iter().enumerate() {
            o.push_str(&format!("fn pad{i}:\n"));
            pretty_pad(a, 1, &mut o);
        }
        pretty_pad(&self.main, 0, &mut o);
        o
    }
}

fn pb(n: &Node) -> String {
    match &n.op {
        Op::And => format!("({} && {})", pb(&n.kids[0]), pb(&n.kids[1])),
        Op::Or => format!("({} || {})", pb(&n.kids[0]), pb(&n.kids[1])),
        Op::Not => format!("!{}", pb(&n.kids[0])),
        Op::Act(i) => ACTIONS[*i as usize].0.to_string(),
        Op::Flag(i) => format!("flag:{}", FLAGS[*i as usize].0),
        Op::Group(i) => format!("group:{}", GROUPS[*i as usize].0),
        Op::Floor(t) => format!("floor=={t:#x}"),
        Op::Last(b) => format!("last:{}", btn_text(*b, 0)),
        Op::RefHeld(b) => format!("ref:{}", btn_text(*b, 0)),
        Op::RefAct(i) => format!("ref={}", ACTIONS[*i as usize].0),
        Op::Lt(v, x) => format!("{}<{}", var_name(*v), x),
        Op::Gt(v, x) => format!("{}>{}", var_name(*v), x),
        Op::CallB(i) => format!("test{i}"),
        o => format!("{o:?}"),
    }
}

fn pinline(n: &Node) -> String {
    match &n.op {
        Op::If(_) => format!("({} ? {} : {})", pb(&n.kids[0]), pinline(&n.kids[1]), pinline(&n.kids[2])),
        Op::Btn { hold, fresh } => btn_text(*hold, *fresh),
        Op::RefBtn => "ref".into(),
        Op::Stick { r, off, tilt } => {
            let o = if *off == 0 { String::new() } else { format!("{off:+#x}", off = *off as i32).replace("+-", "-") };
            let t = if *tilt == 1.0 { String::new() } else { format!(" @{tilt}") };
            format!("{}{o}{t}", ref_text(*r))
        }
        o => format!("{o:?}"),
    }
}

fn pretty_pad(n: &Node, ind: usize, o: &mut String) {
    let pad = "  ".repeat(ind);
    match &n.op {
        Op::If(_) => {
            o.push_str(&format!("{pad}if {}:\n", pb(&n.kids[0])));
            pretty_pad(&n.kids[1], ind + 1, o);
            // else-if chains stay flat.
            let mut e = &n.kids[2];
            while let Op::If(_) = e.op {
                o.push_str(&format!("{pad}elif {}:\n", pb(&e.kids[0])));
                pretty_pad(&e.kids[1], ind + 1, o);
                e = &e.kids[2];
            }
            o.push_str(&format!("{pad}else:\n"));
            pretty_pad(e, ind + 1, o);
        }
        Op::Make => o.push_str(&format!("{pad}{} {}\n", pinline(&n.kids[0]), pinline(&n.kids[1]))),
        Op::CallP(i) => o.push_str(&format!("{pad}pad{i}()\n")),
        x => o.push_str(&format!("{pad}{x:?}\n")),
    }
}

fn tokens(t: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in t.lines() {
        let line = line.split(';').next().unwrap_or("");
        let mut cur = String::new();
        for c in line.chars() {
            if c == '(' || c == ')' || c.is_whitespace() {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                if !c.is_whitespace() {
                    out.push(c.to_string());
                }
            } else {
                cur.push(c);
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    out
}

fn expect(t: &mut Vec<String>, x: &str) -> Result<()> {
    match t.pop() {
        Some(k) if k == x => Ok(()),
        k => bail!("expected {x}, got {k:?}"),
    }
}

/// One whole group (or atom) off the reversed token stack, in reading order.
fn take_group(t: &mut Vec<String>) -> Result<Vec<String>> {
    let first = t.pop().ok_or_else(|| anyhow!("unexpected end"))?;
    let mut out = vec![first.clone()];
    if first != "(" {
        return Ok(out);
    }
    let mut depth = 1;
    while depth > 0 {
        let k = t.pop().ok_or_else(|| anyhow!("unbalanced"))?;
        if k == "(" {
            depth += 1;
        } else if k == ")" {
            depth -= 1;
        }
        out.push(k);
    }
    Ok(out)
}

fn expand(body: &[String], lets: &HashMap<String, Vec<String>>) -> Vec<String> {
    let mut out = Vec::with_capacity(body.len());
    for k in body {
        match lets.get(k) {
            Some(v) => out.extend(v.iter().cloned()),
            None => out.push(k.clone()),
        }
    }
    out
}

fn num<T: std::str::FromStr>(t: &mut Vec<String>) -> Result<T> {
    let k = t.pop().ok_or_else(|| anyhow!("number expected"))?;
    k.parse().map_err(|_| anyhow!("bad number {k}"))
}

fn lookup(tbl: &[(&str, u32)], k: &str) -> Option<u8> {
    tbl.iter().position(|(n, _)| *n == k).map(|i| i as u8)
}

fn parse_btn(k: &str) -> Result<Op> {
    let (mut hold, mut fresh) = (0, 0);
    if k != "_" {
        for c in k.chars() {
            match c {
                'A' => hold |= 1,
                'B' => hold |= 2,
                'Z' => hold |= 4,
                'a' => fresh |= 1,
                'b' => fresh |= 2,
                'z' => fresh |= 4,
                _ => bail!("buttons {k}"),
            }
        }
    }
    Ok(Op::Btn { hold, fresh })
}

/// A tree of type `ty` off the reversed token stack.
fn parse(t: &mut Vec<String>, ty: Ty) -> Result<Node> {
    let k = t.pop().ok_or_else(|| anyhow!("unexpected end"))?;
    if k != "(" {
        if ty == Ty::Btn && k == "ref" {
            return Ok(Node { op: Op::RefBtn, kids: vec![] });
        }
        if ty == Ty::Btn {
            return Ok(Node { op: parse_btn(&k)?, kids: vec![] });
        }
        bail!("unexpected atom {k} for {ty:?}");
    }
    let head = t.pop().ok_or_else(|| anyhow!("unexpected end"))?;
    let node = match head.as_str() {
        "if" => {
            let c = parse(t, Ty::Bool)?;
            let a = parse(t, ty)?;
            let b = parse(t, ty)?;
            Node { op: Op::If(ty), kids: vec![c, a, b] }
        }
        "pad" => {
            let b = parse(t, Ty::Btn)?;
            let s = parse(t, Ty::Stk)?;
            Node { op: Op::Make, kids: vec![b, s] }
        }
        "stick" => {
            let r = t.pop().ok_or_else(|| anyhow!("stick ref"))?;
            let rf = match r.as_str() {
                "face" => Ref::Face,
                "vel" => Ref::Vel,
                "world" => Ref::World,
                "refst" => Ref::RefSt,
                x if x.starts_with("turn") => Ref::Turn(x[4..].parse()?),
                x if x.starts_with('T') => Ref::To(x[1..].parse()?),
                x => bail!("stick ref {x}"),
            };
            let off: i32 = {
                let k = t.pop().ok_or_else(|| anyhow!("stick off"))?;
                if let Some(h) = k.strip_prefix("0x") { i32::from_str_radix(h, 16)? } else { k.parse()? }
            };
            let tilt = num(t)?;
            Node { op: Op::Stick { r: rf, off: off as i16, tilt }, kids: vec![] }
        }
        "and" | "or" => {
            // n-ary in the text, binary in the tree.
            let mut items = vec![parse(t, Ty::Bool)?];
            while t.last().is_some_and(|k| k != ")") {
                items.push(parse(t, Ty::Bool)?);
            }
            let op = if head == "and" { Op::And } else { Op::Or };
            let mut n = items.pop().unwrap();
            while let Some(i) = items.pop() {
                n = Node { op: op.clone(), kids: vec![i, n] };
            }
            n
        }
        "not" => Node { op: Op::Not, kids: vec![parse(t, Ty::Bool)?] },
        "act" => {
            let k = t.pop().unwrap_or_default();
            Node { op: Op::Act(lookup(ACTIONS, &k).ok_or_else(|| anyhow!("action {k}"))?), kids: vec![] }
        }
        "flag" => {
            let k = t.pop().unwrap_or_default();
            let op = if let Some(i) = lookup(FLAGS, &k) {
                Op::Flag(i)
            } else {
                Op::Group(lookup(GROUPS, &k).ok_or_else(|| anyhow!("flag {k}"))?)
            };
            Node { op, kids: vec![] }
        }
        "refheld" => {
            let k = t.pop().unwrap_or_default();
            let Op::Btn { hold, .. } = parse_btn(&k)? else { unreachable!() };
            Node { op: Op::RefHeld(hold), kids: vec![] }
        }
        "refact" => {
            let k = t.pop().unwrap_or_default();
            Node { op: Op::RefAct(lookup(ACTIONS, &k).ok_or_else(|| anyhow!("action {k}"))?), kids: vec![] }
        }
        "floor" => Node { op: Op::Floor(num(t)?), kids: vec![] },
        "last" => {
            let k = t.pop().unwrap_or_default();
            let Op::Btn { hold, .. } = parse_btn(&k)? else { unreachable!() };
            Node { op: Op::Last(hold), kids: vec![] }
        }
        "lt" | "gt" => {
            let v = parse_var(&t.pop().unwrap_or_default())?;
            let x = num(t)?;
            Node { op: if head == "lt" { Op::Lt(v, x) } else { Op::Gt(v, x) }, kids: vec![] }
        }
        "callp" => Node { op: Op::CallP(num(t)?), kids: vec![] },
        "callb" => Node { op: Op::CallB(num(t)?), kids: vec![] },
        h => bail!("unknown form {h}"),
    };
    expect(t, ")")?;
    if node.ty() != ty {
        bail!("{head} is {:?} in a {ty:?} slot", node.ty());
    }
    Ok(node)
}

// ---------------------------------------------------------------------------
// Pruning by coverage.

/// `node` without the `if`s that always went one way in `tr`.
pub fn prune(n: &Node, tr: &HashMap<*const Node, (u32, u32)>) -> Node {
    if let Op::If(_) = n.op
        && let Some(&(t, f)) = tr.get(&(n as *const Node))
    {
        if f == 0 {
            return prune(&n.kids[1], tr);
        }
        if t == 0 {
            return prune(&n.kids[2], tr);
        }
    }
    let kids: Vec<Node> = n.kids.iter().map(|k| prune(k, tr)).collect();
    if let Op::If(_) = n.op
        && kids[1] == kids[2]
    {
        return kids[1].clone();
    }
    Node { op: n.op.clone(), kids }
}

impl Prog {
    pub fn pruned(&self, tr: &HashMap<*const Node, (u32, u32)>) -> Prog {
        Prog {
            main: prune(&self.main, tr),
            adf_p: self.adf_p.iter().map(|a| prune(a, tr)).collect(),
            adf_b: self.adf_b.clone(),
        }
    }
}
