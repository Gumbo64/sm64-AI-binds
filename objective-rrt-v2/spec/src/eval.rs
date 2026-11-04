//! Evaluating predicates and targets on a state.

use obrrt_core::{Obj, St, dist};

use crate::model::*;

/// Finds the object a selector means in `s` (None: none loaded or matching,
/// or no objects read).
pub fn find<'a>(sel: &ObjSel, s: &'a St) -> Option<&'a Obj> {
    let anchor = |o: &Obj| if o.home == [0.0; 3] { o.pos } else { o.home };
    let mut best: Option<(&Obj, f32)> = None;
    for o in s.objects().iter().filter(|o| o.bhv == sel.bhv) {
        let d = match sel.near {
            Some(p) => dist(anchor(o), p),
            None => dist(o.pos, s.pos),
        };
        if d <= sel.within && best.is_none_or(|(_, b)| d < b) {
            best = Some((o, d));
        }
    }
    best.map(|b| b.0)
}

/// A target's distance from `from`, and the point it is measured to (its
/// centre for steering). None: an object target that isn't loaded and has
/// no `near` point.
pub fn target_dist(t: &Target, s: &St, from: [f32; 3]) -> Option<(f32, [f32; 3])> {
    let d = |a: [f32; 3], b: [f32; 3]| {
        if t.horizontal { ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() } else { dist(a, b) }
    };
    match &t.shape {
        Shape::Point(p) => Some((d(from, *p), *p)),
        Shape::Box { min, max } => {
            let q: [f32; 3] = std::array::from_fn(|i| from[i].clamp(min[i], max[i]));
            let c = std::array::from_fn(|i| (min[i] + max[i]) / 2.0);
            Some((d(from, q), c))
        }
        Shape::Sphere { center, r } => Some(((d(from, *center) - r).max(0.0), *center)),
        Shape::Object(sel) => {
            let p = match find(sel, s) {
                Some(o) => std::array::from_fn(|i| o.pos[i] + sel.offset[i]),
                None => {
                    let n = sel.near?;
                    std::array::from_fn(|i| n[i] + sel.offset[i])
                }
            };
            Some((d(from, p), p))
        }
    }
}

/// What a predicate sees besides the state.
pub struct Cx<'a> {
    pub s: &'a St,
    pub start: &'a St,
    pub target: Option<&'a Target>,
    pub phase_start: u32,
}

impl Cx<'_> {
    fn target_obj(&self) -> Option<&Obj> {
        match &self.target?.shape {
            Shape::Object(sel) => find(sel, self.s),
            _ => None,
        }
    }

    fn obj(&self, r: &ObjRef) -> Option<&Obj> {
        match r {
            ObjRef::Target => self.target_obj(),
            ObjRef::Sel(sel) => find(sel, self.s),
        }
    }
}

pub fn field(o: &Obj, f: Field, s: &St) -> f64 {
    (match f {
        Field::X => o.pos[0],
        Field::Y => o.pos[1],
        Field::Z => o.pos[2],
        Field::HomeX => o.home[0],
        Field::HomeY => o.home[1],
        Field::HomeZ => o.home[2],
        Field::HomeDy => o.home[1] - o.pos[1],
        Field::HomeDist => dist(o.home, o.pos),
        Field::MarioDist => dist(o.pos, s.pos),
        Field::Action => return o.action as f64,
        Field::Health => return o.health as f64,
        Field::Value => return o.value as f64,
        Field::HeldState => return o.held_state as f64,
        Field::InteractStatus => return o.interact_status as f64,
        Field::IntangibleTimer => return o.intangible_timer as f64,
        Field::Yaw => return o.yaw as f64,
        Field::BehParams => return o.beh_params as f64,
        Field::LootCoins => return o.loot_coins as f64,
        Field::Raw(i) => return o.raw_1ac[i] as f64,
    }) as f64
}

/// The object field a repeat counts, in the current state.
pub fn count_of(cx: &Cx, r: &ObjRef, f: Field, per: f64) -> Option<u32> {
    let o = cx.obj(r)?;
    Some((field(o, f, cx.s) / per).floor().max(0.0) as u32)
}

/// Yaw (0 = +z, 0x4000 = +x) from `a` towards `b`.
pub fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    let ang = ((b[0] - a[0]) as f64).atan2((b[2] - a[2]) as f64);
    (ang * 65536.0 / std::f64::consts::TAU).round() as i32
}

fn qty(cx: &Cx, q: Qty, to: Option<&Target>) -> Option<f64> {
    let s = cx.s;
    Some(match q {
        Qty::Stars => s.stars as f64,
        Qty::Keys => s.keys as f64,
        Qty::Coins => s.coins as f64,
        Qty::RedCoins => s.red_coins as f64,
        Qty::Wedges => s.wedges() as f64,
        Qty::Lives => s.lives as f64,
        Qty::Speed => s.forward_vel as f64,
        Qty::HSpeed => (s.vel[0].hypot(s.vel[2])) as f64,
        Qty::Vy => s.vel[1] as f64,
        Qty::X => s.pos[0] as f64,
        Qty::Y => s.pos[1] as f64,
        Qty::Z => s.pos[2] as f64,
        Qty::Frame => s.frame as f64,
        Qty::Elapsed => s.frame as f64 - cx.start.frame as f64,
        Qty::PhaseElapsed => s.frame as f64 - cx.phase_start as f64,
        Qty::ActionTimer => s.action_timer as f64,
        Qty::Height => (s.pos[1] - s.floor_y) as f64,
        Qty::Dist | Qty::HDist | Qty::Dy => {
            let t = to.or(cx.target)?;
            match q {
                Qty::Dist => target_dist(t, s, s.pos)?.0 as f64,
                Qty::HDist => {
                    let h = Target { shape: t.shape.clone(), horizontal: true };
                    target_dist(&h, s, s.pos)?.0 as f64
                }
                _ => (s.pos[1] - target_dist(t, s, s.pos)?.1[1]) as f64,
            }
        }
    })
}

fn counter(s: &St, q: Qty) -> f64 {
    match q {
        Qty::Stars => s.stars as f64,
        Qty::Keys => s.keys as f64,
        Qty::Coins => s.coins as f64,
        _ => s.red_coins as f64,
    }
}

/// Whether `p` holds. Anything it can't read (no objects, no target) is
/// false.
pub fn holds(p: &Pred, cx: &Cx) -> bool {
    let s = cx.s;
    match p {
        Pred::All(v) => v.iter().all(|p| holds(p, cx)),
        Pred::Any(v) => v.iter().any(|p| holds(p, cx)),
        Pred::Not(p) => !holds(p, cx),
        Pred::Num(q, r, to) => qty(cx, *q, to.as_deref()).is_some_and(|x| r.has(x)),
        Pred::Gained(q, r) => r.has(counter(s, *q) - counter(cx.start, *q)),
        Pred::HasStar(c, n, want) => s.has_star(*c, *n) == *want,
        Pred::StarGot(c, n) => s.has_star(*c, *n) && !cx.start.has_star(*c, *n),
        Pred::Action(v) => v.contains(&s.action),
        Pred::ActionClass(v) => v.iter().any(|c| match c {
            Class::Group(g) => s.action & 0x1C0 == *g,
            Class::Flag(f) => s.action & f != 0,
        }),
        Pred::Holding(b) => s.holding == *b,
        Pred::Held(r) => s.holding && cx.obj(r).is_some_and(|o| Some(o.slot) == s.held_slot),
        Pred::Facing(y, within) => {
            let want = match y {
                YawRef::Fixed(y) => *y,
                YawRef::Target | YawRef::Away => {
                    let Some(t) = cx.target else { return false };
                    let Some((_, c)) = target_dist(t, s, s.pos) else { return false };
                    yaw_to(s.pos, c) + if matches!(y, YawRef::Away) { 0x8000 } else { 0 }
                }
                YawRef::ObjFacing => match cx.target_obj() {
                    Some(o) => o.yaw,
                    None => return false,
                },
            };
            let d = (s.yaw as i32 - want) as i16 as i32;
            (d.abs() as f64) * 360.0 / 65536.0 <= *within
        }
        Pred::InZone(z) => zone_has(z, s.zone()),
        Pred::ZoneChanged(b) => (s.zone() != cx.start.zone()) == *b,
        Pred::InBox(lo, hi) => (0..3).all(|i| s.pos[i] >= lo[i] && s.pos[i] <= hi[i]),
        Pred::InSphere(c, r) => dist(s.pos, *c) <= *r,
        // Absent only when objects were read (else it can't be told).
        Pred::Present(sel, want) => s.objects.is_some() && find(sel, s).is_some() == *want,
        Pred::Field(r, f, range) => cx.obj(r).is_some_and(|o| range.has(field(o, *f, s))),
        Pred::StandingOn(r) => s.action & 0x800 == 0 && cx.obj(r).is_some_and(|o| s.floor_obj == Some(o.slot)),
        Pred::Alive(b) => s.alive() == *b,
        Pred::Controllable(b) => s.controllable() == *b,
        Pred::Modal(b) => s.modal() == *b,
        Pred::InWater(b) => (s.pos[1] < s.water_level as f32) == *b,
        Pred::SaveFlag(m) => s.save_flags & *m != 0,
        Pred::ThrownOff(r, want) => cx.obj(r).is_some_and(|o| thrown_off(o) == *want),
    }
}

thread_local! {
    /// The tracked (`thrown_off`) object's position on the previous frame,
    /// set by `Spec::eta_phase` from its Mem around each evaluation.
    pub static PREV_OBJ: std::cell::Cell<Option<[f32; 3]>> = const { std::cell::Cell::new(None) };
}

/// Off the hill: already (core's king::is_off), or thrown (act 4) and
/// predicted to land off it from this frame's and last frame's position.
fn thrown_off(o: &Obj) -> bool {
    use obrrt_core::king;
    king::is_off(o.action, o.pos[1], o.home[1])
        || (o.action == 4 && PREV_OBJ.with(|p| p.get()).is_some_and(|p| !king::lands_on_stage(p, o.pos, o.home[1])))
}

/// The object a `thrown_off` reference means in `s` for a phase target.
pub fn resolve<'a>(r: &ObjRef, target: Option<&Target>, s: &'a St) -> Option<&'a Obj> {
    match r {
        ObjRef::Sel(sel) => find(sel, s),
        ObjRef::Target => match &target?.shape {
            Shape::Object(sel) => find(sel, s),
            _ => None,
        },
    }
}
