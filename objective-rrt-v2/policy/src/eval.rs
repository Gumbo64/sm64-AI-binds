//! Interpreter for [`crate::dsl`] policies and the play loop (a copy of
//! `obrrt_gp::rollout::play` that also hands the game to the controller, so
//! senses and named fields can read memory, and records what fired).

use std::collections::HashMap;

use anyhow::Result;
use obrrt_core::{Eta, GamePad, Mem, St, buttons, read_state};
use obrrt_gp::lang::{Ctx, MAX_POINTS, RefNow};
use obrrt_gp::rollout::{Case, End, Outcome, Roll, Task};
use sm64_binds::SM64Game;

use crate::dsl::{self, B, C, F, VARS};
use crate::types::{Ang, Btn, Cell as TCell, Choice, Mag, Obs, Ref, Sense, StickCls};

pub type SkillFn = Box<dyn Fn(&Obs, &[f32]) -> Choice + Send + Sync>;

pub struct SkillDef {
    pub name: String,
    pub nparams: usize,
    pub range: Vec<(f32, f32)>,
    pub defaults: Vec<f32>,
    pub f: SkillFn,
}

/// Everything a policy can call: skills, the sense provider, the named-field reader.
pub struct Env {
    /// Compute the real Sense (geometry rays, landing prediction, objects) each frame.
    pub use_sense: bool,
    pub skills: Vec<SkillDef>,
    pub sense: Box<dyn Fn(&SM64Game, &St) -> Sense + Send + Sync>,
    pub field: Box<dyn Fn(&SM64Game, &St, &str) -> Option<f32> + Send + Sync>,
}

pub fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
}

fn cell(btn: u8, mag: Mag, ang: Ang, r: Ref) -> Choice {
    Choice { cell: TCell { btn: Btn(btn), stick: StickCls { mag, ang, r } }, yaw: 0, fine_ang: 0.0, fine_mag: 0.0 }
}

impl Env {
    /// Minimal built-in skills and an empty sense (until skills.rs / sense.rs land).
    pub fn builtin() -> Env {
        let mk = |name: &str, np: usize, f: SkillFn| SkillDef { name: name.into(), nparams: np, range: vec![(-1.0, 1.0); np], defaults: vec![0.0; np], f };
        Env {
            use_sense: false,
            skills: vec![
                mk("steer", 1, Box::new(|_, p| Choice { fine_ang: p.first().copied().unwrap_or(0.0), ..cell(0, Mag::Full, Ang::Aim, Ref::Target) })),
                mk("steer-a", 1, Box::new(|_, p| Choice { fine_ang: p.first().copied().unwrap_or(0.0), ..cell(Btn::A, Mag::Full, Ang::Aim, Ref::Target) })),
                mk("kick", 0, Box::new(|_, _| cell(Btn::A | Btn::B, Mag::Short, Ang::Aim, Ref::Target))),
                mk("slide", 0, Box::new(|_, _| cell(Btn::Z, Mag::Full, Ang::Aim, Ref::Target))),
                mk("slide-kick", 0, Box::new(|_, _| cell(Btn::Z | Btn::B, Mag::Full, Ang::Aim, Ref::Target))),
                mk("hold-vel", 0, Box::new(|_, _| cell(0, Mag::Full, Ang::Aim, Ref::Vel))),
                mk("neutral", 0, Box::new(|_, _| cell(0, Mag::Neutral, Ang::Aim, Ref::Face))),
                SkillDef {
                    name: "menu".into(),
                    nparams: 1,
                    range: vec![(0.0, 16.0)],
                    defaults: vec![0.0],
                    f: Box::new(|o, p| {
                        // k-th cell of the reduced per-state menu (buttons x stick regions that change what happens next)
                        let m = crate::menu::menu(o);
                        let k = (p.first().copied().unwrap_or(0.0).max(0.0) as usize) % m.len().max(1);
                        m.get(k).map_or_else(|| cell(0, Mag::Full, Ang::Aim, Ref::Target), |c| crate::menu::default_choice(*c))
                    }),
                },
                mk("aim-off", 1, Box::new(|o, p| {
                    let t = yaw_of(o.ctx.pts[0][0] - o.ctx.s.pos[0], o.ctx.pts[0][2] - o.ctx.s.pos[2]);
                    let d = p.first().copied().unwrap_or(0.0).clamp(-180.0, 180.0);
                    Choice { yaw: t.wrapping_add((d * 65536.0 / 360.0) as i32 as i16), ..cell(0, Mag::Full, Ang::Aim, Ref::World) }
                })),
                mk("aim-lead", 2, Box::new(|o, p| {
                    let s = o.ctx.s;
                    let t = yaw_of(o.ctx.pts[0][0] - s.pos[0], o.ctx.pts[0][2] - s.pos[2]);
                    let err = t.wrapping_sub(s.yaw) as f32 * 360.0 / 65536.0;
                    let d = (p.first().copied().unwrap_or(0.0) * err + p.get(1).copied().unwrap_or(0.0)).clamp(-180.0, 180.0);
                    Choice { yaw: t.wrapping_add((d * 65536.0 / 360.0) as i32 as i16), ..cell(0, Mag::Full, Ang::Aim, Ref::World) }
                })),
                mk("wallslide", 0, Box::new(|o, _| {
                    let y = wall_tangent(o.ctx.s, o.ctx.pts[0]).unwrap_or(o.ctx.s.yaw);
                    Choice { yaw: y, ..cell(0, Mag::Full, Ang::Aim, Ref::World) }
                })),
                mk("face", 0, Box::new(|_, _| cell(0, Mag::Full, Ang::Aim, Ref::Face))),
                mk("left", 1, Box::new(|_, p| Choice { fine_ang: p.first().copied().unwrap_or(0.0), ..cell(0, Mag::Full, Ang::Left, Ref::Face) })),
                mk("right", 1, Box::new(|_, p| Choice { fine_ang: p.first().copied().unwrap_or(0.0), ..cell(0, Mag::Full, Ang::Right, Ref::Face) })),
                mk("jump", 0, Box::new(|_, _| cell(Btn::A, Mag::Full, Ang::Aim, Ref::Target))),
            ],
            sense: Box::new(|_, _| Sense::default()),
            field: Box::new(|_, s, n| match n {
                "x" => Some(s.pos[0]),
                "y" => Some(s.pos[1]),
                "z" => Some(s.pos[2]),
                _ => None,
            }),
        }
    }

    /// Built-ins plus the skills.rs library (31 skills, parameter ranges and defaults).
    pub fn full() -> Env {
        let mut e = Env::builtin();
        e.use_sense = true;
        for sk in crate::skills::library() {
            let mut name = sk.name().to_string();
            if e.skills.iter().any(|d| d.name == name) {
                name = format!("lib-{name}");
            }
            let (range, defaults, np) = (sk.param_ranges().to_vec(), sk.defaults(), sk.n_params());
            e.skills.push(SkillDef { name, nparams: np, range, defaults, f: Box::new(move |o, p| sk.choose(o, p)) });
        }
        e
    }

    pub fn skill_ix(&self, name: &str) -> Option<usize> {
        self.skills.iter().position(|s| s.name == name)
    }
}

/// The inputs that initialise a template: bound arguments p0..p7 and an RNG
/// seed (`rnd` is a deterministic function of seed and frame, in -1..1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Instance {
    pub args: [f32; 8],
    pub seed: u64,
}

fn hash01(seed: u64, frame: u32) -> f32 {
    let mut x = seed ^ (frame as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51AFD7ED558CCD);
    x ^= x >> 33;
    ((x >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
}

/// One frame's evaluation context.
pub struct Frame<'a> {
    pub inst: &'a Instance,
    pub env: &'a Env,
    pub game: &'a SM64Game,
    pub ctx: &'a Ctx<'a>,
    pub sense: &'a Sense,
    pub pace: f32,
    /// The state the play started from (base-state reads: start-x, start-fv, ...).
    pub start: &'a St,
    /// This frame's named-field values (the Reader), by path.
    pub vals: &'a HashMap<String, f32>,
    /// A `(tag mpc:H:N:RANGE ...)` node asked for a short-horizon stick search this frame.
    pub req: &'a std::cell::Cell<Option<(u8, u8, f32, bool)>>,
}

fn var(f: &Frame, i: u8) -> f32 {
    let (c, s) = (f.ctx, f.ctx.s);
    let d = |p: [f32; 3]| (p[0] - s.pos[0]).hypot(p[2] - s.pos[2]);
    let sn = f.sense;
    match VARS[i as usize] {
        "fv" => s.forward_vel,
        "vy" => s.vel[1],
        "hsp" => s.vel[0].hypot(s.vel[2]),
        "ny" => s.floor_normal[1],
        "hgt" => s.pos[1] - s.floor_y,
        "inact" => c.in_act as f32,
        "timer" => s.action_timer as f32,
        "down" => {
            let y = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
            s.floor_normal[0] * y.sin() + s.floor_normal[2] * y.cos()
        }
        "health" => s.health as f32,
        "dist" => d(c.pts[0]),
        "dist1" => d(c.pts[1]),
        "yawerr" => {
            let t = yaw_of(c.pts[0][0] - s.pos[0], c.pts[0][2] - s.pos[2]);
            (t.wrapping_sub(s.yaw)) as f32 * 360.0 / 65536.0
        }
        "pace" => f.pace,
        "wall-f" => sn.wall[0],
        "wall-l" => sn.wall[1],
        "wall-r" => sn.wall[2],
        "wall-b" => sn.wall[3],
        "ceil" => sn.ceil_clear,
        "edge" => sn.edge_ahead,
        "ledge" => sn.ledge_ahead,
        "land-frames" => sn.land.map_or(999.0, |l| l.frames as f32),
        "land-ny" => sn.land.map_or(1.0, |l| l.probe.ny),
        "land-class" => sn.land.map_or(0.0, |l| l.probe.class as f32),
        "land-zone" => sn.land_zone as u8 as f32,
        "in-zone" => sn.in_zone as u8 as f32,
        "land-x" => sn.land.map_or(0.0, |l| l.pos[0]),
        "land-z" => sn.land.map_or(0.0, |l| l.pos[2]),
        "land-y" => sn.land.map_or(0.0, |l| l.pos[1]),
        "land-fv" => sn.land.map_or(0.0, |l| l.fv),
        // flat-ground range of a jump from the speed it would start at (the landing speed after a landing frame's x0.98)
        "dj-range" | "kick-range" | "jump-range" | "lj-range" => {
            let base = if s.action & 0x800 != 0 { sn.land.map_or(s.forward_vel, |l| l.fv) * 0.98 } else { s.forward_vel };
            let k = match VARS[i as usize] { "dj-range" => 0, "kick-range" => 1, "jump-range" => 2, _ => 3 };
            crate::sense::flat_range(k, base)
        }
        n if n.contains("-land-") && !n.starts_with("land-") => {
            // dj-land-*, kick-land-*, jump-land-*, lj-land-*: where that jump would land if pressed now (grounded only)
            let (k, rest) = n.split_once("-land-").unwrap();
            let ix = match k { "dj" => 0, "kick" => 1, "jump" => 2, _ => 3 };
            match (sn.hyp[ix], rest) {
                (Some(l), "x") => l.pos[0],
                (Some(l), "z") => l.pos[2],
                (Some(l), "y") => l.pos[1],
                (Some(l), "fv") => l.fv,
                (Some(l), "frames") => l.frames as f32,
                (Some(_), "zone") => sn.hyp_zone[ix] as u8 as f32,
                // no landing found (no floor / too far): far away, never in a zone
                (None, "z") | (None, "x") | (None, "y") => -99999.0,
                (None, "frames") => 999.0,
                _ => 0.0,
            }
        }
        "ahead-ny" => sn.ahead[2].ny,
        "ahead-class" => sn.ahead[2].class as f32,
        "ahead-hgt" => sn.ahead[2].y - s.pos[1],
        "holding" => s.holding as u8 as f32,
        "stick-x" => c.last.stick_x as f32,
        "stick-y" => c.last.stick_y as f32,
        "frame" => s.frame as f32,
        "slope-class" => 0.0,
        "obj-dist" => sn.objs.first().map_or(9999.0, |o| o.dist),
        "rnd" => hash01(f.inst.seed, s.frame),
        "start-x" => f.start.pos[0],
        "start-y" => f.start.pos[1],
        "start-z" => f.start.pos[2],
        "start-fv" => f.start.forward_vel,
        "start-dx" => s.pos[0] - f.start.pos[0],
        "start-dz" => s.pos[2] - f.start.pos[2],
        "since-start" => (s.frame.saturating_sub(f.start.frame)) as f32,
        "wp0-dx" => c.pts[0][0] - s.pos[0],
        "wp0-dz" => c.pts[0][2] - s.pos[2],
        "wp1-dx" => c.pts[1][0] - s.pos[0],
        "wp1-dz" => c.pts[1][2] - s.pos[2],
        "wp2-dx" => c.pts[2][0] - s.pos[0],
        "wp2-dz" => c.pts[2][2] - s.pos[2],
        "pos-x" => s.pos[0],
        "pos-z" => s.pos[2],
        "vel-x" => s.vel[0],
        "vel-z" => s.vel[2],
        "wallhit" => s.wall_normal.is_some() as u8 as f32,
        "walltan" => wall_tangent(s, c.pts[0]).map_or(0.0, |y| y.wrapping_sub(s.yaw) as f32 * 360.0 / 65536.0),
        n if n.len() == 2 && n.starts_with('p') => f.inst.args[(n.as_bytes()[1] - b'0') as usize],
        _ => 0.0,
    }
}

/// World yaw of the wall tangent (in the floor plane) that points closer to the target.
pub fn wall_tangent(s: &St, target: [f32; 3]) -> Option<i16> {
    let n = s.wall_normal?;
    let (dx, dz) = (target[0] - s.pos[0], target[2] - s.pos[2]);
    let (t1, t2) = ((-n[2], n[0]), (n[2], -n[0]));
    let t = if t1.0 * dx + t1.1 * dz >= t2.0 * dx + t2.1 * dz { t1 } else { t2 };
    Some(yaw_of(t.0, t.1))
}

pub fn ef(e: &F, f: &Frame) -> f32 {
    match e {
        F::Const(x) => *x,
        F::Var(i) => var(f, *i),
        F::Field(n) => f.vals.get(n.as_str()).copied().filter(|v| v.is_finite()).unwrap_or(0.0),
        F::Obj(cat, k, n) if dsl::is_derived_obj_field(n) => {
            let ci = crate::sense::CAT_NAMES.iter().position(|c| c == cat).unwrap_or(0) as u8;
            let mut v: Vec<&crate::types::ObjProbe> = f.sense.objs.iter().filter(|o| o.cat == ci).collect();
            v.sort_by(|a, b| a.dist.total_cmp(&b.dist));
            match v.get(*k as usize) {
                None => match n.as_str() {
                    "dist" | "dist3" => 99999.0,
                    _ => 0.0,
                },
                Some(o) => match n.as_str() {
                    "dist3" => o.dist,
                    "dist" => (o.dist * o.dist - o.dy * o.dy).max(0.0).sqrt(),
                    "dy" => o.dy,
                    "relyaw" => o.rel_yaw as f32 * 360.0 / 65536.0,
                    _ => o.radius,
                },
            }
        }
        F::Obj(cat, k, n) => f.vals.get(&dsl::obj_path(cat, *k, n)).copied().filter(|v| v.is_finite()).unwrap_or(0.0),
        // What-if observations are off until the core works (docs: optional, low priority).
        F::Ahead(..) => 0.0,
        F::Add(a, b) => ef(a, f) + ef(b, f),
        F::Sub(a, b) => ef(a, f) - ef(b, f),
        F::Mul(a, b) => ef(a, f) * ef(b, f),
        F::Min(a, b) => ef(a, f).min(ef(b, f)),
        F::Max(a, b) => ef(a, f).max(ef(b, f)),
        F::Abs(a) => ef(a, f).abs(),
        F::Sqrt(a) => ef(a, f).max(0.0).sqrt(),
        F::Sin(a) => ef(a, f).to_radians().sin(),
        F::Cos(a) => ef(a, f).to_radians().cos(),
        F::Atan2(a, b) => ef(a, f).atan2(ef(b, f)).to_degrees(),
        F::If(c, a, b) => {
            if eb(c, f) {
                ef(a, f)
            } else {
                ef(b, f)
            }
        }
    }
}

pub fn eb(e: &B, f: &Frame) -> bool {
    match e {
        B::Gt(a, b) => ef(a, f) > ef(b, f),
        B::Lt(a, b) => ef(a, f) < ef(b, f),
        B::And(v) => v.iter().all(|x| eb(x, f)),
        B::Or(v) => v.iter().any(|x| eb(x, f)),
        B::Not(a) => !eb(a, f),
        B::ActIn(v) => v.contains(&f.ctx.s.action),
        B::PrevActIn(_) => false,
        B::Held(m) => {
            let l = f.ctx.last.button;
            (m & Btn::A == 0 || l & buttons::A_BUTTON != 0) && (m & Btn::B == 0 || l & buttons::B_BUTTON != 0) && (m & Btn::Z == 0 || l & buttons::Z_TRIG != 0)
        }
    }
}

/// Resolves `(switch NAME)` to a policy body (the inventory provides it).
pub trait Resolver {
    /// The body and the bound instance of another policy: a switch runs it with ITS arguments.
    fn body(&self, name: &str) -> Option<(&C, Instance)>;
}
pub struct NoSwitch;
impl Resolver for NoSwitch {
    fn body(&self, _: &str) -> Option<(&C, Instance)> {
        None
    }
}

pub fn cell_of(c: &dsl::Cell, f: &Frame) -> Choice {
    use Ang::*;
    let angs = [Aim, Left, Right, SideLeft, SideRight, Back];
    let mags = [Mag::Neutral, Mag::Short, Mag::Full];
    let refs = [Ref::Target, Ref::Face, Ref::Vel, Ref::Uphill, Ref::Downhill, Ref::Cam, Ref::World];
    Choice {
        cell: TCell { btn: Btn(c.btn), stick: StickCls { mag: mags[c.mag as usize], ang: angs[c.ang as usize], r: refs[c.r as usize] } },
        yaw: ef(&c.yaw, f) as i32 as i16,
        fine_ang: ef(&c.fine_ang, f).clamp(-1.0, 1.0),
        fine_mag: ef(&c.fine_mag, f).clamp(0.0, 1.0),
    }
}

/// Evaluate a choice tree; `fired` collects the tags and skills used this frame.
pub fn ec(e: &C, f: &Frame, res: &dyn Resolver, fired: &mut Vec<String>, depth: u8) -> Choice {
    match e {
        C::If(b, x, y) => {
            if eb(b, f) {
                ec(x, f, res, fired, depth)
            } else {
                ec(y, f, res, fired, depth)
            }
        }
        C::Tag(t, c) => {
            fired.push(t.clone());
            // mpc:H:N:RANGE searches the stick yaw; mpcm:H:N:RANGE also the menu's button/stick cells.
            if let Some((rest, cells)) = t.strip_prefix("mpc:").map(|r| (r, false)).or_else(|| t.strip_prefix("mpcm:").map(|r| (r, true))) {
                let v: Vec<f32> = rest.split(':').filter_map(|x| x.parse().ok()).collect();
                if v.len() >= 3 {
                    f.req.set(Some((v[0].clamp(2.0, 60.0) as u8, v[1].clamp(3.0, 41.0) as u8, v[2].clamp(5.0, 120.0), cells)));
                }
            }
            ec(c, f, res, fired, depth)
        }
        C::Cell(c) => cell_of(c, f),
        C::Skill(n, ps) => {
            fired.push(format!("skill:{n}"));
            let Some(i) = f.env.skill_ix(n) else { return cell(0, Mag::Full, Ang::Aim, Ref::Target) };
            let sk = &f.env.skills[i];
            let p: Vec<f32> = (0..sk.nparams)
                .map(|j| match ps.get(j) {
                    Some(e) => {
                        let v = ef(e, f);
                        sk.range.get(j).map_or(v, |r| v.clamp(r.0.min(r.1), r.0.max(r.1)))
                    }
                    None => sk.defaults.get(j).copied().unwrap_or(0.0),
                })
                .collect();
            let obs = Obs { ctx: f.ctx, sense: f.sense };
            (sk.f)(&obs, &p)
        }
        C::Switch(n) => {
            fired.push(format!("switch:{n}"));
            match (depth < 4).then(|| res.body(n)).flatten() {
                Some((b, inst)) => {
                    let g = Frame { inst: &inst, env: f.env, game: f.game, ctx: f.ctx, sense: f.sense, pace: f.pace, start: f.start, vals: f.vals, req: f.req };
                    ec(b, &g, res, fired, depth + 1)
                }
                None => cell(0, Mag::Full, Ang::Aim, Ref::Target),
            }
        }
    }
}

/// Turn a choice into a pad. (menu::realize replaces this once it lands.)
pub fn realize(c: &Choice, ctx: &Ctx) -> GamePad {
    let s = ctx.s;
    let st = c.cell.stick;
    let base: i16 = match st.r {
        Ref::Target => yaw_of(ctx.pts[0][0] - s.pos[0], ctx.pts[0][2] - s.pos[2]),
        Ref::Face => s.yaw,
        Ref::Vel => {
            if s.vel[0].hypot(s.vel[2]) > 1.0 { yaw_of(s.vel[0], s.vel[2]) } else { s.yaw }
        }
        Ref::Downhill => yaw_of(s.floor_normal[0], s.floor_normal[2]),
        Ref::Uphill => yaw_of(s.floor_normal[0], s.floor_normal[2]).wrapping_add(0x8000u16 as i16),
        Ref::Cam => s.cam_yaw,
        Ref::World => c.yaw,
    };
    let off: i32 = match st.ang {
        Ang::Aim => 0,
        Ang::Left => 0x1000,
        Ang::Right => -0x1000,
        Ang::SideLeft => 0x4000,
        Ang::SideRight => -0x4000,
        Ang::Back => 0x8000,
    } + (c.fine_ang * 0x800 as f32) as i32;
    let tilt = match st.mag {
        Mag::Neutral => 0.0,
        Mag::Short => 0.30 + 0.06 * c.fine_mag,
        Mag::Full => 1.0,
    };
    let (x, y) = if tilt == 0.0 { (0, 0) } else { obrrt_gp::lang::stick(s.cam_yaw, base.wrapping_add(off as i16), tilt) };
    let mut b = 0u16;
    if c.cell.btn.a() {
        b |= buttons::A_BUTTON;
    }
    if c.cell.btn.b() {
        b |= buttons::B_BUTTON;
    }
    if c.cell.btn.z() {
        b |= buttons::Z_TRIG;
    }
    GamePad::new(b, x, y)
}

/// The named fields a body reads, including those of the policies it switches to.
fn collect_fields(body: &C, res: &dyn Resolver, out: &mut Vec<String>, depth: u8) {
    body.fields(out);
    if depth >= 3 {
        return;
    }
    let mut sw = vec![];
    body.switches(&mut sw);
    for n in sw {
        if let Some((b, _)) = res.body(&n) {
            collect_fields(b, res, out, depth + 1);
        }
    }
}

/// What a play looked like (for niche descriptors and failure analysis).
#[derive(Clone, Debug, Default)]
pub struct Desc {
    pub frames: u32,
    pub air: u32,
    pub slide: u32,
    pub land: u32,
    pub ground: u32,
    pub fv_sum: f64,
    pub end_pos: [f32; 3],
    pub end_action: u32,
    pub end_frame: u32,
    pub fired: HashMap<String, u32>,
    /// Per-frame rows, only when POLICY_TRACE is set.
    pub rows: Vec<String>,
    pub mpc_frames: u64,
    pub mpc_calls: u32,
}

pub struct Played {
    pub out: Outcome,
    pub desc: Desc,
}

fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("POLICY_TRACE").is_ok())
}

fn class_of(a: u32) -> u8 {
    // ACT_FLAG_AIR 0x800; sliding actions have ACT_FLAG_MOVING 0x400 and a name in the slide family.
    let n = dsl::action_name(a);
    if a & 0x800 != 0 {
        0
    } else if n.contains("slide") || n.contains("rollout") {
        1
    } else if n.contains("land") {
        2
    } else {
        3
    }
}

/// Short-horizon search for a rule flagged `(tag mpc:H:N:RANGE ...)` or `(tag mpcm:H:N:RANGE ...)`: the world yaw of the
/// stick is held for H frames at N offsets over +-RANGE degrees from the choice's reference; `mpcm` also tries the
/// other cells of the reduced menu (button sets x stick classes). Magnitude classes are never crossed (the 48 limit).
/// Each candidate is rolled out with the task's own reward; the base choice wins ties. Returns the choice and the
/// emulator frames spent.
#[allow(clippy::too_many_arguments)]
fn mpc_pick(game: &mut SM64Game, task: &dyn Task, base: &Choice, ctx: &Ctx, sense: &Sense, start: &St, mem: &Mem, frame0: u32, h: usize, n: usize, range: f32, cells: bool, r: &Roll) -> Result<(Choice, u64)> {
    let obs = Obs { ctx, sense };
    let cur = ctx.s;
    let world = |btn: Btn, mag: Mag, yaw: i16, fine_mag: f32| Choice { cell: TCell { btn, stick: StickCls { mag, ang: Ang::Aim, r: Ref::World } }, yaw, fine_ang: 0.0, fine_mag };
    let mut offs: Vec<f32> = (0..n).map(|i| (i as f32 / (n - 1) as f32 * 2.0 - 1.0) * range).collect();
    offs.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
    let mut cands: Vec<Choice> = Vec::new();
    let (centre, _) = crate::menu::ang_range(base.cell.stick.ang);
    let r0 = crate::menu::ref_yaw(base.cell.stick.r, &obs, base.yaw) as i32 + centre as i32;
    for &off in &offs {
        cands.push(world(base.cell.btn, base.cell.stick.mag, (r0 + (off * 65536.0 / 360.0) as i32) as i16, base.fine_mag));
    }
    if cells {
        for c in crate::menu::menu(&obs) {
            let d = crate::menu::default_choice(c);
            if c.btn == base.cell.btn && c.stick.mag == base.cell.stick.mag {
                continue;
            }
            let (ctr, _) = crate::menu::ang_range(d.cell.stick.ang);
            let rr = crate::menu::ref_yaw(d.cell.stick.r, &obs, d.yaw) as i32 + ctr as i32;
            if d.cell.stick.mag == Mag::Neutral {
                cands.push(world(d.cell.btn, Mag::Neutral, rr as i16, 0.0));
                continue;
            }
            for &off in &offs {
                cands.push(world(d.cell.btn, d.cell.stick.mag, (rr + (off * 65536.0 / 360.0) as i32) as i16, d.fine_mag));
            }
        }
    }
    let snap = game.save_state();
    let mut cost = 0u64;
    let mut best = (f64::NEG_INFINITY, 0usize);
    for (ci, cand) in cands.iter().enumerate() {
        game.load_state(&snap)?;
        let pad = crate::types::realize(cand, &obs);
        let mut m2 = *mem;
        let e0 = task.eta(cur, start, &mut m2).frames();
        let (mut sum, mut disc, mut prev) = (0.0, 1.0, e0);
        for k in 0..h {
            game.step_game(pad)?;
            cost += 1;
            let st = read_state(game, frame0 + k as u32 + 1, false);
            let eta = if !st.alive() || st.lives < start.lives { Eta::Lost } else { task.eta(&st, start, &mut m2) };
            let lost = match eta {
                Eta::Lost => true,
                Eta::Frames(x) => !x.is_finite(),
                Eta::Done => false,
            };
            if lost {
                sum -= disc * r.death;
                break;
            }
            sum += disc * (prev - eta.frames());
            prev = eta.frames();
            if eta.done() {
                sum += disc * r.bonus;
                break;
            }
            disc *= r.gamma;
        }
        if sum > best.0 + 1e-9 {
            best = (sum, ci);
        }
    }
    game.load_state(&snap)?;
    Ok((cands[best.1], cost))
}


static NO_INST: Instance = Instance { args: [0.0; 8], seed: 0 };

/// What drives the pad during one segment of a policy list.
pub enum Drv<'a> {
    /// A reactive policy with its bound arguments / seed.
    Pol { body: &'a C, inst: Instance },
    /// A raw-input kind (pink noise from a seed, or a recorded slice): pad `i` of the segment is `pads[i]`
    /// (neutral past the end). It is one more policy kind, not a flattening of the others.
    Pads(Vec<GamePad>),
}

/// A button forced on / suppressed for `len` frames starting `at` frames into the segment.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BtnSpan {
    pub at: u16,
    pub len: u16,
    pub mask: u16,
    pub on: bool,
}

/// Per-segment closed-loop fine genes: they ride on whatever the segment's policy outputs, frame by frame.
/// `dang` / `dmag` are added to the policy's `Choice.fine_ang` / `fine_mag` (menu fine space: 1.0 of fine_ang = 0x800
/// = 11.25 degrees; fine_mag only acts on Short-magnitude cells), `btn` overrides A / B for a span inside the hold,
/// `mpc` = (H, N, range degrees) wraps the segment in the stick-only MPC of `mpc_pick` (its emulator frames are
/// counted in `Desc::mpc_frames`; when it fires it chooses the stick yaw, so `dang` is then moot).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Fine {
    pub dang: f32,
    pub dmag: f32,
    pub btn: Option<BtnSpan>,
    pub mpc: Option<(u8, u8, f32)>,
}

/// One segment of a genome: a driver held for `hold` frames (the last segment is held to the horizon).
pub struct Seg<'a> {
    pub drv: Drv<'a>,
    pub hold: usize,
    pub fine: Fine,
}

/// `rollout::play` with the policy as the controller.
pub fn play_policy(
    game: &mut SM64Game,
    case: &Case,
    task: &dyn Task,
    body: &C,
    res: &dyn Resolver,
    env: &Env,
    r: &Roll,
    keep: bool,
    inst: &Instance,
) -> Result<Played> {
    play_segs(game, case, task, &[Seg { drv: Drv::Pol { body, inst: *inst }, hold: usize::MAX, fine: Fine::default() }], res, env, r, keep)
}

/// Closed-loop rollout of a list of segments: segment k drives for `hold` frames from whatever state the earlier
/// segments left (policies read the live game, so they re-react to it), the last one to the end.
#[allow(clippy::too_many_arguments)]
pub fn play_segs(
    game: &mut SM64Game,
    case: &Case,
    task: &dyn Task,
    segs: &[Seg],
    res: &dyn Resolver,
    env: &Env,
    r: &Roll,
    keep: bool,
) -> Result<Played> {
    anyhow::ensure!(!segs.is_empty(), "empty genome");
    game.load_state(&case.save)?;
    for p in &case.noise {
        game.step_game(*p)?;
    }
    let mut traj = case.pre_states.clone();
    let mut played = case.pre_pads.clone();
    let mut mem = Mem::default();
    let start = &case.start;
    let e0 = task.eta(start, start, &mut mem).frames();
    let mut out = Outcome { fitness: f64::NEG_INFINITY, done: None, end: End::Horizon, frames: 0, best_eta: e0, start_eta: e0, pads: Vec::new(), report: String::new() };
    let mut desc = Desc::default();
    let (mut sum, mut disc, mut prev) = (0.0, 1.0, e0);
    let mut best_t = 0;
    let base = case.frame as u32;
    let horizon = task.horizon().unwrap_or(r.horizon);
    let mut fired: Vec<String> = Vec::new();
    // Real senses (geometry, rays, landing, objects) and the Reader's named fields, resolved once per play.
    let mut sc = crate::sense::SenseCtx::new();
    let mut names: Vec<String> = Vec::new();
    for sg in segs {
        if let Drv::Pol { body, .. } = &sg.drv {
            collect_fields(body, res, &mut names, 0);
        }
    }
    names.sort();
    names.dedup();
    let fs = if names.is_empty() { None } else { crate::sense::reader::FieldSet::parse(&names.iter().map(|s| s.as_str()).collect::<Vec<_>>()).ok() };
    let mut vals: HashMap<String, f32> = HashMap::new();
    let mut buf: Vec<f32> = Vec::new();
    // Zones the task forbids: Lakitu's trigger box on the castle grounds (any ground frame in it is a catch).
    let regions: Vec<crate::sense::Region> = if start.level as u32 == obrrt_core::layout::LEVEL_CASTLE_GROUNDS && start.area == 1 {
        vec![crate::sense::Region { lo: [-544.0, 800.0, -2000.0], hi: [545.0, 100000.0, -177.0] }]
    } else {
        Vec::new()
    };
    let (mut si, mut seg_t) = (0usize, 0usize);
    for t in 0..horizon {
        if seg_t >= segs[si].hold && si + 1 < segs.len() {
            si += 1;
            seg_t = 0;
        }
        let (body, inst) = match &segs[si].drv {
            Drv::Pol { body, inst } => (Some(*body), inst),
            Drv::Pads(_) => (None, &NO_INST),
        };
        let cur = traj.last().unwrap();
        let pts = task.points(cur, &mem);
        let pace = task.pace(&mem);
        let rf: RefNow = task.reference(cur, &mem);
        let ctx = Ctx { s: cur, last: played.last().copied().unwrap_or(GamePad::new(0, 0, 0)), in_act: obrrt_gp::rollout::in_action(&traj, &played), pts, pace, rf };
        let sense = if body.is_some() && env.use_sense {
            let air = crate::sense::AirModel::from_pad(cur, ctx.last);
            game.with_memory(|m| sc.read(m, cur, &crate::sense::SenseOpts::default(), &regions, &air))
        } else {
            Sense::default()
        };
        if let (Some(fs), Some(_)) = (&fs, body) {
            buf.clear();
            buf.resize(fs.len(), 0.0);
            game.with_memory(|m| fs.fetch(m, cur, &mut buf));
            vals.clear();
            for (n, v) in names.iter().zip(&buf) {
                vals.insert(n.clone(), *v);
            }
        }
        let reqc = std::cell::Cell::new(None);
        fired.clear();
        let mut ch = match body {
            Some(body) => {
                let fr = Frame { inst, env, game, ctx: &ctx, sense: &sense, pace, start, vals: &vals, req: &reqc };
                Some(ec(body, &fr, res, &mut fired, 0))
            }
            None => None,
        };
        let fine = segs[si].fine;
        if let Some(c) = ch.as_mut() {
            c.fine_ang += fine.dang;
            c.fine_mag = (c.fine_mag + fine.dmag).clamp(-1.0, 1.0);
        }
        let rq = reqc.take().or(if body.is_some() { fine.mpc.map(|(h, n, rg)| (h, n.max(2), rg, false)) } else { None });
        if let (Some((h, n, range, cells)), Some(c0)) = (rq, ch) {
            let (c2, cost) = mpc_pick(game, task, &c0, &ctx, &sense, start, &mem, base + t as u32, h as usize, n as usize, range, cells, r)?;
            ch = Some(c2);
            desc.mpc_frames += cost;
            desc.mpc_calls += 1;
        }
        let mut pad = match &segs[si].drv {
            Drv::Pads(p) => p.get(seg_t).copied().unwrap_or(GamePad::new(0, 0, 0)),
            Drv::Pol { .. } => crate::types::realize(&ch.unwrap(), &Obs { ctx: &ctx, sense: &sense }),
        };
        if let Some(b) = fine.btn {
            if seg_t >= b.at as usize && seg_t < b.at as usize + b.len as usize {
                if b.on {
                    pad.button |= b.mask
                } else {
                    pad.button &= !b.mask
                }
            }
        }
        seg_t += 1;
        if trace_on() {
            let tags = fired.clone();
            desc.rows.push(format!("f{:>4} {:<22} fv {:>6.1} vy {:>6.1} pos ({:>6.0},{:>5.0},{:>6.0}) yaw {:>6} pad {:04x} ({:>4},{:>4}) wallf {:>6.0} blk {} land {:?} hyp dj {} kick {} | {}", t, dsl::action_name(cur.action), cur.forward_vel, cur.vel[1], cur.pos[0], cur.pos[1], cur.pos[2], cur.yaw as u16, pad.button, pad.stick_x, pad.stick_y, sense.wall[0], sense.objs.len(), sense.land.map(|l| l.frames), sense.hyp[0].map_or("-".to_string(), |l| format!("{:.0}/{:.0}{}", l.pos[2], l.pos[1], if sense.hyp_zone[0] { "Z" } else { "" })), sense.hyp[1].map_or("-".to_string(), |l| format!("{:.0}/{:.0}{}", l.pos[2], l.pos[1], if sense.hyp_zone[1] { "Z" } else { "" })), tags.join(",")));
        }
        for k in fired.drain(..) {
            *desc.fired.entry(k).or_insert(0) += 1;
        }
        let cls = class_of(cur.action);
        match cls {
            0 => desc.air += 1,
            1 => desc.slide += 1,
            2 => desc.land += 1,
            _ => desc.ground += 1,
        }
        desc.fv_sum += cur.forward_vel as f64;
        game.step_game(pad)?;
        out.frames += 1;
        desc.frames += 1;
        if keep {
            out.pads.push(pad);
        }
        played.push(pad);
        let st = read_state(game, base + t as u32 + 1, false);
        let mut rw;
        let eta = if !st.alive() || st.lives < start.lives { Eta::Lost } else { task.eta(&st, start, &mut mem) };
        let lost = match eta {
            Eta::Lost => true,
            Eta::Frames(x) => !x.is_finite(),
            Eta::Done => false,
        };
        if lost {
            rw = -r.death;
            out.end = if st.alive() { End::Lost } else { End::Died };
        } else {
            rw = prev - eta.frames();
            prev = eta.frames();
            if eta.done() {
                rw += r.bonus;
                out.done = Some(t + 1);
                out.end = End::Done;
            }
        }
        out.best_eta = out.best_eta.min(prev);
        sum += disc * rw;
        disc *= r.gamma;
        if sum > out.fitness {
            out.fitness = sum;
            best_t = t;
        }
        desc.end_pos = st.pos;
        desc.end_action = st.action;
        desc.end_frame = t as u32 + 1;
        if lost || eta.done() {
            break;
        }
        traj.push(st);
        if t.saturating_sub(best_t) > r.stall {
            out.end = End::Stall;
            break;
        }
    }
    out.report = task.report(&mem, out.frames as usize, out.end);
    Ok(Played { out, desc })
}
