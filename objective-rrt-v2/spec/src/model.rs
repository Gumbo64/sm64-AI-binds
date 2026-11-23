//! The parsed spec: predicates, targets, phases and hints, and their JSON
//! readers (tools/spec/FORMAT.md documents every field).

use anyhow::{Result, anyhow};
use obrrt_core::layout::*;
use obrrt_core::{Arrival, GaBudget, Precision};
use serde_json::Value;

use crate::json::{J, check_keys, err, idx, key};

/// A level (and area; `None` = any area).
pub type Zone = (u16, Option<u8>);

pub fn zone_has(zones: &[Zone], z: (u16, u8)) -> bool {
    zones.iter().any(|&(l, a)| l == z.0 && a.is_none_or(|a| a == z.1))
}

const LEVEL_NAMES: &[(&str, u32)] = &[
    ("castle", LEVEL_CASTLE), ("inside", LEVEL_CASTLE), ("lobby", LEVEL_CASTLE), ("grounds", LEVEL_CASTLE_GROUNDS), ("courtyard", LEVEL_CASTLE_COURTYARD),
    ("bob", LEVEL_BOB), ("wf", LEVEL_WF), ("jrb", LEVEL_JRB), ("ccm", LEVEL_CCM), ("bbh", LEVEL_BBH), ("hmc", LEVEL_HMC),
    ("lll", LEVEL_LLL), ("ssl", LEVEL_SSL), ("ddd", LEVEL_DDD), ("sl", LEVEL_SL), ("wdw", LEVEL_WDW), ("ttm", LEVEL_TTM),
    ("thi", LEVEL_THI), ("ttc", LEVEL_TTC), ("rr", LEVEL_RR), ("pss", LEVEL_PSS), ("sa", LEVEL_SA), ("wmotr", LEVEL_WMOTR),
    ("totwc", LEVEL_TOTWC), ("cotmc", LEVEL_COTMC), ("vcutm", LEVEL_VCUTM), ("bitdw", LEVEL_BITDW), ("bitfs", LEVEL_BITFS),
    ("bits", LEVEL_BITS), ("bowser1", LEVEL_BOWSER_1), ("bowser2", LEVEL_BOWSER_2), ("bowser3", LEVEL_BOWSER_3),
];

/// `[9, 1]`, `[9]`, `9`, `"bob"`, `"bob:1"`, `"9:1"`.
pub fn parse_zone(j: J) -> Result<Zone> {
    let level = |s: &str| -> Result<u16> {
        if let Ok(n) = s.parse::<u16>() {
            return Ok(n);
        }
        LEVEL_NAMES.iter().find(|(n, _)| *n == s.to_ascii_lowercase()).map(|(_, l)| *l as u16).ok_or_else(|| {
            anyhow!("{}: unknown level \"{s}\"; use a number or one of {}", j.path, LEVEL_NAMES.iter().map(|x| x.0).collect::<Vec<_>>().join(", "))
        })
    };
    match j.v {
        Value::String(s) => {
            let mut it = s.splitn(2, ':');
            let l = level(it.next().unwrap())?;
            let a = match it.next() {
                Some(a) => Some(a.parse::<u8>().map_err(|_| anyhow!("{}: bad area in \"{s}\"", j.path))?),
                None => None,
            };
            Ok((l, a))
        }
        Value::Number(_) => Ok((j.u32()? as u16, None)),
        Value::Array(a) if (1..=2).contains(&a.len()) => {
            let p0 = idx(j.path, 0);
            let l = match &a[0] {
                Value::String(s) => level(s)?,
                v => J::new(v, &p0).u32()? as u16,
            };
            let a1 = match a.get(1) {
                Some(v) => {
                    let p1 = idx(j.path, 1);
                    Some(J::new(v, &p1).u32()? as u8)
                }
                None => None,
            };
            Ok((l, a1))
        }
        _ => err(j.path, "expected a zone: [level, area], level, \"bob\", \"bob:1\""),
    }
}

/// One zone or a list of zones.
pub fn parse_zones(j: J) -> Result<Vec<Zone>> {
    if let Value::Array(a) = j.v {
        if a.first().is_some_and(|x| x.is_array() || x.is_string()) {
            return a
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let p = idx(j.path, i);
                    parse_zone(J::new(v, &p))
                })
                .collect();
        }
    }
    Ok(vec![parse_zone(j)?])
}

/// An inclusive numeric range.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub lo: f64,
    pub hi: f64,
    /// Strict bounds (`lt` / `gt`).
    pub lo_open: bool,
    pub hi_open: bool,
}

impl Range {
    pub fn has(&self, x: f64) -> bool {
        (if self.lo_open { x > self.lo } else { x >= self.lo }) && (if self.hi_open { x < self.hi } else { x <= self.hi })
    }
}

/// `5` (exactly), `[lo, hi]` (inclusive), `{"lt": x, "ge": y, ...}`.
pub fn parse_range(j: J) -> Result<Range> {
    let mut r = Range { lo: f64::NEG_INFINITY, hi: f64::INFINITY, lo_open: false, hi_open: false };
    match j.v {
        Value::Number(_) | Value::String(_) => {
            let x = j.f64()?;
            r.lo = x;
            r.hi = x;
        }
        Value::Array(a) if a.len() == 2 => {
            let (p0, p1) = (idx(j.path, 0), idx(j.path, 1));
            r.lo = if a[0].is_null() { f64::NEG_INFINITY } else { J::new(&a[0], &p0).f64()? };
            r.hi = if a[1].is_null() { f64::INFINITY } else { J::new(&a[1], &p1).f64()? };
        }
        Value::Object(m) => {
            check_keys(m, j.path, &["lt", "le", "gt", "ge", "eq", "to", "obj", "name", "of"])?;
            for (k, v) in m {
                let p = key(j.path, k);
                let x = || J::new(v, &p).f64();
                match k.as_str() {
                    "lt" => (r.hi, r.hi_open) = (x()?, true),
                    "le" => r.hi = x()?,
                    "gt" => (r.lo, r.lo_open) = (x()?, true),
                    "ge" => r.lo = x()?,
                    "eq" => {
                        r.lo = x()?;
                        r.hi = r.lo;
                    }
                    _ => {}
                }
            }
            if !m.keys().any(|k| matches!(k.as_str(), "lt" | "le" | "gt" | "ge" | "eq")) {
                return err(j.path, "a range needs at least one of lt, le, gt, ge, eq");
            }
        }
        _ => return err(j.path, "expected a number, [lo, hi] or {\"lt\"|\"le\"|\"gt\"|\"ge\"|\"eq\": x}"),
    }
    Ok(r)
}

/// How an object is picked from a state's object list.
#[derive(Debug, Clone)]
pub struct ObjSel {
    pub bhv_name: String,
    pub bhv: u32,
    /// Pick the one whose home (its position when it has no home) is
    /// nearest this point; without it, the one nearest Mario.
    pub near: Option<[f32; 3]>,
    /// Only objects within this of `near` (of Mario without `near`).
    pub within: f32,
    /// Added to the object's position for distances and steering.
    pub offset: [f32; 3],
    /// Lost (as a phase target) while objects are read and none matches.
    pub required: bool,
}

pub fn parse_objsel(j: J) -> Result<ObjSel> {
    let mut sel = ObjSel { bhv_name: String::new(), bhv: 0, near: None, within: f32::INFINITY, offset: [0.0; 3], required: false };
    let name = match j.v {
        Value::String(s) => s.as_str(),
        Value::Object(m) => {
            check_keys(m, j.path, &["bhv", "near", "within", "offset", "required"])?;
            for (k, v) in m {
                let p = key(j.path, k);
                let jj = J::new(v, &p);
                match k.as_str() {
                    "near" => sel.near = Some(jj.vec3()?),
                    "within" => sel.within = jj.f32()?,
                    "offset" => sel.offset = jj.vec3()?,
                    "required" => sel.required = jj.bool()?,
                    _ => {}
                }
            }
            m.get("bhv").and_then(|v| v.as_str()).ok_or_else(|| anyhow!("{}: an object selector needs \"bhv\" (a behaviour name)", j.path))?
        }
        _ => return err(j.path, "expected a behaviour name (\"bhvWoodenPost\") or {\"bhv\": ..., \"near\": [x,y,z], \"within\": r}"),
    };
    let name = if name.starts_with("bhv") { name.to_string() } else { format!("bhv{name}") };
    sel.bhv = obrrt_core::bhv_addr(&name).ok_or_else(|| anyhow!("{}: unknown behaviour \"{name}\" (names as in data/behavior_data.c, e.g. bhvWoodenPost)", j.path))?;
    sel.bhv_name = name;
    Ok(sel)
}

/// Where a phase goes.
#[derive(Debug, Clone)]
pub enum Shape {
    Point([f32; 3]),
    Box { min: [f32; 3], max: [f32; 3] },
    Sphere { center: [f32; 3], r: f32 },
    Object(ObjSel),
}

#[derive(Debug, Clone)]
pub struct Target {
    pub shape: Shape,
    /// Horizontal distance only (`"metric": "h"`).
    pub horizontal: bool,
}

/// Node positions for `{"node": id}` targets (nodes.csv of a run).
pub type NodePos = dyn Fn(u32) -> Option<[f32; 3]>;

pub fn parse_target(j: J, nodes: Option<&NodePos>) -> Result<Target> {
    if j.v.is_array() {
        return Ok(Target { shape: Shape::Point(j.vec3()?), horizontal: false });
    }
    let m = j.obj()?;
    check_keys(m, j.path, &["point", "box", "sphere", "object", "node", "metric"])?;
    let shapes: Vec<&String> = m.keys().filter(|k| matches!(k.as_str(), "point" | "box" | "sphere" | "object" | "node")).collect();
    if shapes.len() != 1 {
        return err(j.path, "a target needs exactly one of point, box, sphere, object, node");
    }
    let k = shapes[0].as_str();
    let p = key(j.path, k);
    let jj = J::new(&m[k], &p);
    let shape = match k {
        "point" => Shape::Point(jj.vec3()?),
        "box" => {
            let b = jj.obj()?;
            check_keys(b, &p, &["min", "max"])?;
            let (pm, px) = (key(&p, "min"), key(&p, "max"));
            let min = J::new(b.get("min").ok_or_else(|| anyhow!("{p}: needs min"))?, &pm).vec3()?;
            let max = J::new(b.get("max").ok_or_else(|| anyhow!("{p}: needs max"))?, &px).vec3()?;
            let (lo, hi) = (std::array::from_fn(|i| min[i].min(max[i])), std::array::from_fn(|i| min[i].max(max[i])));
            Shape::Box { min: lo, max: hi }
        }
        "sphere" => {
            let b = jj.obj()?;
            check_keys(b, &p, &["center", "r"])?;
            let (pc, pr) = (key(&p, "center"), key(&p, "r"));
            let center = J::new(b.get("center").ok_or_else(|| anyhow!("{p}: needs center"))?, &pc).vec3()?;
            let r = J::new(b.get("r").ok_or_else(|| anyhow!("{p}: needs r"))?, &pr).f32()?;
            Shape::Sphere { center, r }
        }
        "object" => Shape::Object(parse_objsel(jj)?),
        _ => {
            let id = jj.u32()?;
            let f = nodes.ok_or_else(|| anyhow!("{p}: node targets need a run (the validator's run dir, or OBRRT_SPEC_RUN=<run dir> for obrrt2)"))?;
            Shape::Point(f(id).ok_or_else(|| anyhow!("{p}: node {id} not in the run's nodes.csv"))?)
        }
    };
    let horizontal = match m.get("metric") {
        None => false,
        Some(v) => {
            let p = key(j.path, "metric");
            match J::new(v, &p).str()? {
                "h" | "horizontal" | "xz" => true,
                "3d" | "xyz" => false,
                s => return err(&p, format!("\"{s}\": metric is \"3d\" (default) or \"h\"")),
            }
        }
    };
    Ok(Target { shape, horizontal })
}

/// Quantities numeric predicates compare.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Qty {
    Stars,
    Keys,
    Coins,
    RedCoins,
    Wedges,
    Lives,
    Speed,
    HSpeed,
    Vy,
    X,
    Y,
    Z,
    Frame,
    Elapsed,
    PhaseElapsed,
    ActionTimer,
    Height,
    Dist,
    HDist,
    Dy,
}

pub const QTYS: &[(&str, Qty)] = &[
    ("stars", Qty::Stars), ("keys", Qty::Keys), ("coins", Qty::Coins), ("red_coins", Qty::RedCoins), ("wedges", Qty::Wedges),
    ("lives", Qty::Lives), ("speed", Qty::Speed), ("hspeed", Qty::HSpeed), ("vy", Qty::Vy), ("x", Qty::X), ("y", Qty::Y),
    ("z", Qty::Z), ("frame", Qty::Frame), ("elapsed", Qty::Elapsed), ("phase_elapsed", Qty::PhaseElapsed),
    ("action_timer", Qty::ActionTimer), ("height", Qty::Height), ("dist", Qty::Dist), ("hdist", Qty::HDist), ("dy", Qty::Dy),
];

/// Counters that "*_gained" compares with the start.
pub const GAINED: &[(&str, Qty)] =
    &[("stars_gained", Qty::Stars), ("keys_gained", Qty::Keys), ("coins_gained", Qty::Coins), ("red_coins_gained", Qty::RedCoins)];

/// Object fields a `field` predicate or a repeat count reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Field {
    X,
    Y,
    Z,
    HomeX,
    HomeY,
    HomeZ,
    /// home.y - pos.y: how far it has sunk (a pounded post).
    HomeDy,
    /// |pos - home|.
    HomeDist,
    MarioDist,
    Action,
    Health,
    Value,
    HeldState,
    InteractStatus,
    IntangibleTimer,
    Yaw,
    BehParams,
    LootCoins,
    Raw(usize),
}

pub const FIELDS: &[(&str, Field)] = &[
    ("x", Field::X), ("y", Field::Y), ("z", Field::Z), ("home_x", Field::HomeX), ("home_y", Field::HomeY), ("home_z", Field::HomeZ),
    ("home_dy", Field::HomeDy), ("home_dist", Field::HomeDist), ("mario_dist", Field::MarioDist), ("action", Field::Action),
    ("health", Field::Health), ("value", Field::Value), ("held_state", Field::HeldState), ("interact_status", Field::InteractStatus),
    ("intangible_timer", Field::IntangibleTimer), ("yaw", Field::Yaw), ("beh_params", Field::BehParams), ("loot_coins", Field::LootCoins),
    ("raw0", Field::Raw(0)), ("raw1", Field::Raw(1)), ("raw2", Field::Raw(2)), ("raw3", Field::Raw(3)),
];

pub fn parse_field(j: J) -> Result<Field> {
    let s = j.str()?;
    FIELDS.iter().find(|(n, _)| *n == s).map(|(_, f)| *f).ok_or_else(|| {
        anyhow!("{}: unknown object field \"{s}\"; one of {}", j.path, FIELDS.iter().map(|x| x.0).collect::<Vec<_>>().join(", "))
    })
}

/// An object reference in a predicate: the phase target's object, or one
/// picked by its own selector.
#[derive(Debug, Clone)]
pub enum ObjRef {
    Target,
    Sel(ObjSel),
}

fn parse_objref(j: J) -> Result<ObjRef> {
    if j.v.as_str() == Some("target") {
        return Ok(ObjRef::Target);
    }
    Ok(ObjRef::Sel(parse_objsel(j)?))
}

#[derive(Debug, Clone, Copy)]
pub enum YawRef {
    Fixed(i32),
    /// Facing the phase target.
    Target,
    /// Facing away from it.
    Away,
    /// The target object's own facing (e.g. behind a boss: "same as it").
    ObjFacing,
}

/// Classes of Mario's action: an action group or an action flag.
#[derive(Debug, Clone, Copy)]
pub enum Class {
    Group(u32),
    Flag(u32),
}

pub const CLASSES: &[(&str, Class)] = &[
    ("stationary", Class::Group(0x000)), ("moving", Class::Group(0x040)), ("airborne", Class::Group(0x080)),
    ("submerged", Class::Group(0x0C0)), ("cutscene", Class::Group(0x100)), ("automatic", Class::Group(0x140)),
    ("object", Class::Group(0x180)), ("air", Class::Flag(1 << 11)), ("intangible", Class::Flag(1 << 12)),
    ("swimming", Class::Flag(1 << 13)), ("riding_shell", Class::Flag(1 << 16)), ("invulnerable", Class::Flag(1 << 17)),
    ("sliding", Class::Flag(1 << 18)), ("diving", Class::Flag(1 << 19)), ("on_pole", Class::Flag(1 << 20)),
    ("hanging", Class::Flag(1 << 21)), ("idle", Class::Flag(1 << 22)), ("attacking", Class::Flag(1 << 23)),
];

#[derive(Debug, Clone)]
pub enum Pred {
    All(Vec<Pred>),
    Any(Vec<Pred>),
    Not(Box<Pred>),
    Num(Qty, Range, Option<Box<Target>>),
    Gained(Qty, Range),
    HasStar(i16, u8, bool),
    StarGot(i16, u8),
    Action(Vec<u32>),
    ActionClass(Vec<Class>),
    Holding(bool),
    Held(ObjRef),
    Facing(YawRef, f64),
    InZone(Vec<Zone>),
    ZoneChanged(bool),
    InBox([f32; 3], [f32; 3]),
    InSphere([f32; 3], f32),
    Present(ObjSel, bool),
    Field(ObjRef, Field, Range),
    StandingOn(ObjRef),
    Alive(bool),
    Controllable(bool),
    Modal(bool),
    InWater(bool),
    /// Any of these bits set in the save file's flags word (caps, keys, ...).
    SaveFlag(u32),
    /// King Bob-omb (or the object) off his hill: thrown and predicted to
    /// land off it, already below it, or jumping back (core's king.rs).
    ThrownOff(ObjRef, bool),
}

impl Pred {
    /// Whether evaluating it reads the object list.
    pub fn needs_objects(&self) -> bool {
        match self {
            Pred::All(v) | Pred::Any(v) => v.iter().any(Pred::needs_objects),
            Pred::Not(p) => p.needs_objects(),
            Pred::Num(_, _, Some(t)) => matches!(t.shape, Shape::Object(_)),
            Pred::Held(_) | Pred::Present(..) | Pred::Field(..) | Pred::StandingOn(_) | Pred::ThrownOff(..) => true,
            _ => false,
        }
    }

    /// Whether it reads the phase target (dist, facing "target", ...).
    pub fn uses_target(&self) -> bool {
        match self {
            Pred::All(v) | Pred::Any(v) => v.iter().any(Pred::uses_target),
            Pred::Not(p) => p.uses_target(),
            Pred::Num(q, _, t) => t.is_none() && matches!(q, Qty::Dist | Qty::HDist | Qty::Dy),
            Pred::Facing(y, _) => !matches!(y, YawRef::Fixed(_)),
            Pred::Held(ObjRef::Target) | Pred::Field(ObjRef::Target, ..) | Pred::StandingOn(ObjRef::Target) | Pred::ThrownOff(ObjRef::Target, _) => true,
            _ => false,
        }
    }

    /// The first `thrown_off` object reference in it.
    pub fn thrown_ref(&self) -> Option<&ObjRef> {
        match self {
            Pred::All(v) | Pred::Any(v) => v.iter().find_map(Pred::thrown_ref),
            Pred::Not(p) => p.thrown_ref(),
            Pred::ThrownOff(r, _) => Some(r),
            _ => None,
        }
    }
}

/// Every key a predicate object may have.
pub fn pred_keys() -> Vec<&'static str> {
    let mut k = vec![
        "all", "any", "not", "has_star", "lacks_star", "star_got", "action", "action_class", "holding", "held", "facing", "zone",
        "zone_changed", "in_box", "in_sphere", "present", "absent", "field", "standing_on", "alive", "controllable", "modal",
        "in_water", "thrown_off", "save_flag",
    ];
    k.extend(QTYS.iter().map(|x| x.0));
    k.extend(GAINED.iter().map(|x| x.0));
    k
}

fn parse_star(j: J) -> Result<(i16, u8)> {
    let a = j.arr()?;
    if a.len() != 2 {
        return err(j.path, "expected [course, star] (star 1-7 as in the game's star select, 7 = 100 coins)");
    }
    let (p0, p1) = (idx(j.path, 0), idx(j.path, 1));
    let c = J::new(&a[0], &p0).u32()?;
    let n = J::new(&a[1], &p1).u32()?;
    if !(1..=25).contains(&c) || !(1..=7).contains(&n) {
        return err(j.path, "course is 1-25, star 1-7 (1-based: star 6 of BoB is \"Behind Chain Chomp's Gate\")");
    }
    Ok((c as i16, (n - 1) as u8))
}

fn parse_action(j: J) -> Result<u32> {
    if let Some(s) = j.v.as_str() {
        if s.starts_with("0x") || s.starts_with("0X") {
            return Ok(j.f64()? as u32);
        }
        let name = s.strip_prefix("ACT_").unwrap_or(s).to_ascii_uppercase();
        return crate::actions::ACTIONS.iter().find(|(n, _)| *n == name).map(|(_, a)| *a).ok_or_else(|| {
            anyhow!("{}: unknown action \"{s}\" (names from include/sm64.h without ACT_, e.g. GROUND_POUND_LAND, or hex)", j.path)
        });
    }
    j.u32()
}

fn list<'a>(j: J<'a>) -> Vec<(&'a Value, String)> {
    match j.v {
        Value::Array(a) => a.iter().enumerate().map(|(i, v)| (v, idx(j.path, i))).collect(),
        v => vec![(v, j.path.to_string())],
    }
}

/// A predicate: an object whose keys all must hold, or a list of such.
pub fn parse_pred(j: J, nodes: Option<&NodePos>) -> Result<Pred> {
    if let Value::Array(a) = j.v {
        let mut out = Vec::new();
        for (i, v) in a.iter().enumerate() {
            let p = idx(j.path, i);
            out.push(parse_pred(J::new(v, &p), nodes)?);
        }
        return Ok(Pred::All(out));
    }
    let m = j.obj()?;
    check_keys(m, j.path, &pred_keys())?;
    let mut out = Vec::new();
    for (k, v) in m {
        if matches!(k.as_str(), "doc" | "comment" | "notes") {
            continue;
        }
        let p = key(j.path, k);
        let jj = J::new(v, &p);
        let pred = match k.as_str() {
            "all" | "any" => {
                let mut sub = Vec::new();
                for (i, v) in jj.arr()?.iter().enumerate() {
                    let pi = idx(&p, i);
                    sub.push(parse_pred(J::new(v, &pi), nodes)?);
                }
                if k == "all" { Pred::All(sub) } else { Pred::Any(sub) }
            }
            "not" => Pred::Not(Box::new(parse_pred(jj, nodes)?)),
            "has_star" | "lacks_star" => {
                let (c, n) = parse_star(jj)?;
                Pred::HasStar(c, n, k == "has_star")
            }
            "star_got" => {
                let (c, n) = parse_star(jj)?;
                Pred::StarGot(c, n)
            }
            "action" => Pred::Action(list(jj).iter().map(|(v, p)| parse_action(J::new(v, p))).collect::<Result<_>>()?),
            "action_class" => Pred::ActionClass(
                list(jj)
                    .iter()
                    .map(|(v, p)| {
                        let s = J::new(v, p).str()?;
                        CLASSES.iter().find(|(n, _)| *n == s).map(|(_, c)| *c).ok_or_else(|| {
                            anyhow!("{p}: unknown action class \"{s}\"; one of {}", CLASSES.iter().map(|x| x.0).collect::<Vec<_>>().join(", "))
                        })
                    })
                    .collect::<Result<_>>()?,
            ),
            "holding" => Pred::Holding(jj.bool()?),
            "held" => Pred::Held(parse_objref(jj)?),
            "facing" => {
                let f = jj.obj()?;
                check_keys(f, &p, &["yaw", "within"])?;
                let py = key(&p, "yaw");
                let yv = f.get("yaw").ok_or_else(|| anyhow!("{p}: needs yaw (a number, \"target\", \"away\" or \"object\")"))?;
                let yaw = match yv.as_str() {
                    Some("target") => YawRef::Target,
                    Some("away") => YawRef::Away,
                    Some("object") => YawRef::ObjFacing,
                    _ => YawRef::Fixed(J::new(yv, &py).f64()? as i32),
                };
                let pw = key(&p, "within");
                let within = match f.get("within") {
                    Some(w) => J::new(w, &pw).f64()?,
                    None => 45.0,
                };
                Pred::Facing(yaw, within)
            }
            "zone" => Pred::InZone(parse_zones(jj)?),
            "zone_changed" => Pred::ZoneChanged(jj.bool()?),
            "in_box" => {
                let b = jj.obj()?;
                check_keys(b, &p, &["min", "max"])?;
                let (pm, px) = (key(&p, "min"), key(&p, "max"));
                let a = J::new(b.get("min").ok_or_else(|| anyhow!("{p}: needs min"))?, &pm).vec3()?;
                let c = J::new(b.get("max").ok_or_else(|| anyhow!("{p}: needs max"))?, &px).vec3()?;
                Pred::InBox(std::array::from_fn(|i| a[i].min(c[i])), std::array::from_fn(|i| a[i].max(c[i])))
            }
            "in_sphere" => {
                let b = jj.obj()?;
                check_keys(b, &p, &["center", "r"])?;
                let (pc, pr) = (key(&p, "center"), key(&p, "r"));
                Pred::InSphere(
                    J::new(b.get("center").ok_or_else(|| anyhow!("{p}: needs center"))?, &pc).vec3()?,
                    J::new(b.get("r").ok_or_else(|| anyhow!("{p}: needs r"))?, &pr).f32()?,
                )
            }
            "present" | "absent" => Pred::Present(parse_objsel(jj)?, k == "present"),
            "field" => {
                let f = jj.obj()?;
                check_keys(f, &p, &["obj", "name", "lt", "le", "gt", "ge", "eq"])?;
                let pn = key(&p, "name");
                let name = parse_field(J::new(f.get("name").ok_or_else(|| anyhow!("{p}: needs name (the object field)"))?, &pn))?;
                let po = key(&p, "obj");
                let obj = match f.get("obj") {
                    Some(o) => parse_objref(J::new(o, &po))?,
                    None => ObjRef::Target,
                };
                Pred::Field(obj, name, parse_range(jj)?)
            }
            "standing_on" => Pred::StandingOn(parse_objref(jj)?),
            "alive" => Pred::Alive(jj.bool()?),
            "controllable" => Pred::Controllable(jj.bool()?),
            "modal" => Pred::Modal(jj.bool()?),
            "in_water" => Pred::InWater(jj.bool()?),
            "save_flag" => Pred::SaveFlag(jj.u32()?),
            "thrown_off" => {
                // true / false (the phase target), or {"obj": ..., "is": bool}.
                match v {
                    Value::Bool(b) => Pred::ThrownOff(ObjRef::Target, *b),
                    _ => {
                        let m = jj.obj()?;
                        check_keys(m, &p, &["obj", "is"])?;
                        let r = match m.get("obj") {
                            Some(o) => parse_objref(J::new(o, &key(&p, "obj")))?,
                            None => ObjRef::Target,
                        };
                        let is = match m.get("is") {
                            Some(b) => J::new(b, &key(&p, "is")).bool()?,
                            None => true,
                        };
                        Pred::ThrownOff(r, is)
                    }
                }
            }
            k => {
                if let Some((_, q)) = GAINED.iter().find(|(n, _)| *n == k) {
                    Pred::Gained(*q, parse_range(jj)?)
                } else {
                    let q = QTYS.iter().find(|(n, _)| *n == k).map(|x| x.1).unwrap();
                    // dist/hdist/dy may name their own target: {"lt": 150, "to": {...}}.
                    let to = match v.get("to") {
                        Some(t) if matches!(q, Qty::Dist | Qty::HDist | Qty::Dy) => {
                            let pt = key(&p, "to");
                            Some(Box::new(parse_target(J::new(t, &pt), nodes)?))
                        }
                        Some(_) => return err(&p, "\"to\" only applies to dist, hdist and dy"),
                        None => None,
                    };
                    Pred::Num(q, parse_range(jj)?, to)
                }
            }
        };
        out.push(pred);
    }
    Ok(if out.len() == 1 { out.pop().unwrap() } else { Pred::All(out) })
}

/// How often a phase must be done.
#[derive(Debug, Clone)]
pub struct Repeat {
    pub n: u32,
    /// Count from an object field (`floor(field / per)`), else rising edges
    /// of `done`.
    pub count: Option<(ObjRef, Field, f64)>,
    /// Frames per further repetition (default: the phase's `frames`, or 30).
    pub frames: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Phase {
    pub name: String,
    pub target: Option<Target>,
    pub done: Pred,
    /// `done` was given (not the default "arrive").
    pub done_explicit: bool,
    pub pace: f64,
    pub frames: f64,
    /// Once done, stays done (default). `false`: a condition that must keep
    /// holding; when it stops, the spec falls back to this phase.
    pub latch: bool,
    pub repeat: Option<Repeat>,
    /// Steering: the phase target by default; `Some(None)` = none.
    pub steer: Option<Option<Target>>,
    pub button: Option<(u16, f32)>,
    pub arrival: Option<Arrival>,
    /// Frames added to the ETA while the phase is current and `when` holds
    /// (e.g. the King thrown off his hill: his way back and the dialog).
    pub penalty: Vec<(Pred, f64)>,
}

pub fn parse_button(j: J) -> Result<u16> {
    use obrrt_core::buttons::*;
    let mut out = 0;
    for c in j.str()?.split('+') {
        out |= match c.trim().to_ascii_uppercase().as_str() {
            "A" => A_BUTTON,
            "B" => B_BUTTON,
            "Z" => Z_TRIG,
            "" | "NONE" => 0,
            s => return err(j.path, format!("unknown button \"{s}\": A, B, Z or a mix like \"A+B\"")),
        };
    }
    Ok(out)
}

pub fn parse_arrival(j: J) -> Result<Arrival> {
    match j.str()? {
        "pass" => Ok(Arrival::Pass),
        "stop" => Ok(Arrival::Stop),
        "door" => Ok(Arrival::Door),
        s => err(j.path, format!("\"{s}\": arrival is pass, stop or door")),
    }
}

/// `"B"` or `{"button": "B", "within": 300}`.
fn parse_steer_button(j: J) -> Result<(u16, f32)> {
    if j.v.is_string() {
        return Ok((parse_button(j)?, 300.0));
    }
    let m = j.obj()?;
    check_keys(m, j.path, &["button", "within"])?;
    let pb = key(j.path, "button");
    let b = parse_button(J::new(m.get("button").ok_or_else(|| anyhow!("{}: needs button", j.path))?, &pb))?;
    let pw = key(j.path, "within");
    let w = match m.get("within") {
        Some(v) => J::new(v, &pw).f32()?,
        None => 300.0,
    };
    Ok((b, w))
}

fn reads_target_obj(p: &Pred) -> bool {
    match p {
        Pred::All(v) | Pred::Any(v) => v.iter().any(reads_target_obj),
        Pred::Not(p) => reads_target_obj(p),
        Pred::Held(ObjRef::Target) | Pred::Field(ObjRef::Target, ..) | Pred::StandingOn(ObjRef::Target) => true,
        Pred::Facing(YawRef::ObjFacing, _) | Pred::ThrownOff(ObjRef::Target, _) => true,
        _ => false,
    }
}

pub const PHASE_KEYS: &[&str] = &["name", "target", "done", "pace", "frames", "latch", "repeat", "steer", "steer_button", "arrival", "penalty"];

pub fn parse_phase(j: J, i: usize, nodes: Option<&NodePos>) -> Result<Phase> {
    let m = j.obj()?;
    check_keys(m, j.path, PHASE_KEYS)?;
    let get = |k: &str| m.get(k);
    let pk = |k: &str| key(j.path, k);
    let name = match get("name") {
        Some(v) => J::new(v, &pk("name")).str()?.to_string(),
        None => format!("phase{i}"),
    };
    let target = match get("target") {
        Some(v) => Some(parse_target(J::new(v, &pk("target")), nodes)?),
        None => None,
    };
    let done = match get("done") {
        Some(v) => parse_pred(J::new(v, &pk("done")), nodes)?,
        None => match &target {
            // Default: arrive (inside a box or sphere, within 150 otherwise).
            Some(t) => match t.shape {
                Shape::Box { .. } | Shape::Sphere { .. } => Pred::Num(Qty::Dist, Range { lo: f64::NEG_INFINITY, hi: 0.0, lo_open: false, hi_open: false }, None),
                _ => Pred::Num(Qty::Dist, Range { lo: f64::NEG_INFINITY, hi: 150.0, lo_open: false, hi_open: true }, None),
            },
            None => return err(j.path, "a phase needs \"done\" (or a \"target\", whose default done is arriving)"),
        },
    };
    if target.is_none() && done.uses_target() {
        return err(&pk("done"), "uses the phase target (dist/hdist/dy/facing target/\"target\" object) but the phase has no \"target\"");
    }
    let pace = match get("pace") {
        Some(v) => J::new(v, &pk("pace")).f64()?,
        None => obrrt_core::PACE,
    };
    if !(pace > 0.0) {
        return err(&pk("pace"), "pace must be > 0 (units per frame)");
    }
    let frames = match get("frames") {
        Some(v) => J::new(v, &pk("frames")).f64()?,
        None => 0.0,
    };
    if frames < 0.0 {
        return err(&pk("frames"), "frames must be ≥ 0");
    }
    let latch = match get("latch") {
        Some(v) => J::new(v, &pk("latch")).bool()?,
        None => true,
    };
    let repeat = match get("repeat") {
        None => None,
        Some(v) => {
            let p = pk("repeat");
            let rj = J::new(v, &p);
            if v.is_number() {
                Some(Repeat { n: rj.u32()?, count: None, frames: None })
            } else {
                let r = rj.obj()?;
                check_keys(r, &p, &["n", "count", "frames"])?;
                let pn = key(&p, "n");
                let n = J::new(r.get("n").ok_or_else(|| anyhow!("{p}: needs n"))?, &pn).u32()?;
                let frames = match r.get("frames") {
                    Some(f) => Some(J::new(f, &key(&p, "frames")).f64()?),
                    None => None,
                };
                let count = match r.get("count") {
                    None => None,
                    Some(c) => {
                        let pc = key(&p, "count");
                        let cm = J::new(c, &pc).obj()?;
                        check_keys(cm, &pc, &["obj", "field", "per"])?;
                        let pf = key(&pc, "field");
                        let field = parse_field(J::new(cm.get("field").ok_or_else(|| anyhow!("{pc}: needs field"))?, &pf))?;
                        let per = match cm.get("per") {
                            Some(x) => J::new(x, &key(&pc, "per")).f64()?,
                            None => 1.0,
                        };
                        if !(per > 0.0) {
                            return err(&key(&pc, "per"), "per must be > 0");
                        }
                        let obj = match cm.get("obj") {
                            Some(o) => parse_objref(J::new(o, &key(&pc, "obj")))?,
                            None => ObjRef::Target,
                        };
                        if matches!(obj, ObjRef::Target) && !matches!(target.as_ref().map(|t| &t.shape), Some(Shape::Object(_))) {
                            return err(&pc, "counts the target object's field, but the phase target isn't an object (give \"obj\")");
                        }
                        Some((obj, field, per))
                    }
                };
                Some(Repeat { n, count, frames })
            }
        }
    };
    if repeat.as_ref().is_some_and(|r| r.n == 0) {
        return err(&pk("repeat"), "n must be ≥ 1");
    }
    let steer = match get("steer") {
        None => None,
        Some(Value::Bool(true)) => None,
        Some(Value::Bool(false)) | Some(Value::Null) => Some(None),
        Some(v) => Some(Some(parse_target(J::new(v, &pk("steer")), nodes)?)),
    };
    let button = match get("steer_button") {
        Some(v) => Some(parse_steer_button(J::new(v, &pk("steer_button")))?),
        None => None,
    };
    let arrival = match get("arrival") {
        Some(v) => Some(parse_arrival(J::new(v, &pk("arrival")))?),
        None => None,
    };
    if !matches!(target.as_ref().map(|t| &t.shape), Some(Shape::Object(_))) && reads_target_obj(&done) {
        return err(&pk("done"), "reads the target object (\"target\" in held/field/standing_on/facing object) but the target isn't an object");
    }
    let done_explicit = get("done").is_some();
    let mut penalty = Vec::new();
    if let Some(v) = get("penalty") {
        let pp = pk("penalty");
        let list: Vec<(&Value, String)> = match v {
            Value::Array(a) => a.iter().enumerate().map(|(i, x)| (x, idx(&pp, i))).collect(),
            x => vec![(x, pp.clone())],
        };
        for (x, pi) in list {
            let m = J::new(x, &pi).obj()?;
            check_keys(m, &pi, &["when", "frames"])?;
            let when = parse_pred(J::new(m.get("when").ok_or_else(|| anyhow!("{pi}: needs when"))?, &key(&pi, "when")), nodes)?;
            let f = J::new(m.get("frames").ok_or_else(|| anyhow!("{pi}: needs frames"))?, &key(&pi, "frames")).f64()?;
            if reads_target_obj(&when) && !matches!(target.as_ref().map(|t| &t.shape), Some(Shape::Object(_))) {
                return err(&pi, "reads the target object but the target isn't an object");
            }
            penalty.push((when, f));
        }
    }
    Ok(Phase { name, target, done, done_explicit, pace, frames, latch, repeat, steer, button, arrival, penalty })
}

#[derive(Debug, Clone)]
pub struct Hints {
    pub steer_button: Option<(u16, f32)>,
    pub arrival: Arrival,
    pub precision: Precision,
    pub ga_budget: Option<GaBudget>,
    pub obj_weight: f64,
    pub success_value: f64,
    pub kinship: u32,
    pub temperature: Option<f64>,
    pub done_frames: f64,
}

pub const HINT_KEYS: &[&str] =
    &["steer_button", "arrival", "precision", "ga_budget", "obj_weight", "success_value", "kinship", "temperature", "done_frames"];

pub fn parse_hints(m: &serde_json::Map<String, Value>, path: &str) -> Result<Hints> {
    let mut h = Hints {
        steer_button: None,
        arrival: Arrival::Pass,
        precision: Precision::Explore,
        ga_budget: None,
        obj_weight: 5.0,
        success_value: 0.0,
        kinship: 6,
        temperature: None,
        done_frames: 0.0,
    };
    for k in HINT_KEYS {
        let Some(v) = m.get(*k) else { continue };
        let p = key(path, k);
        let j = J::new(v, &p);
        match *k {
            "steer_button" => h.steer_button = Some(parse_steer_button(j)?),
            "arrival" => h.arrival = parse_arrival(j)?,
            "precision" => {
                h.precision = match j.str()? {
                    "explore" => Precision::Explore,
                    "precise" => Precision::Precise,
                    s => return err(&p, format!("\"{s}\": precision is explore or precise")),
                }
            }
            "ga_budget" => {
                let (pop, gens, hor) = if let Value::Array(a) = v {
                    if a.len() != 3 {
                        return err(&p, "expected [population, generations, horizon]");
                    }
                    let q = |i: usize| J::new(&a[i], &idx(&p, i)).u32();
                    (q(0)?, q(1)?, q(2)?)
                } else {
                    let b = j.obj()?;
                    check_keys(b, &p, &["population", "generations", "horizon"])?;
                    let q = |k: &str| -> Result<u32> {
                        J::new(b.get(k).ok_or_else(|| anyhow!("{p}: needs {k}"))?, &key(&p, k)).u32()
                    };
                    (q("population")?, q("generations")?, q("horizon")?)
                };
                if pop < 2 || gens < 1 || hor < 1 || pop > 256 || gens > 64 || hor > 3000 {
                    return err(&p, "population 2-256, generations 1-64, horizon 1-3000");
                }
                h.ga_budget = Some(GaBudget { population: pop as usize, generations: gens as usize, horizon: hor as usize });
            }
            "obj_weight" => {
                h.obj_weight = j.f64()?;
                if !(h.obj_weight >= 0.0) {
                    return err(&p, "must be ≥ 0");
                }
            }
            "success_value" => h.success_value = j.f64()?,
            "kinship" => h.kinship = j.u32()?,
            "temperature" => h.temperature = Some(j.f64()?),
            "done_frames" => h.done_frames = j.f64()?,
            _ => unreachable!(),
        }
    }
    Ok(h)
}
