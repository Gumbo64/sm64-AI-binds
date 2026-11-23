//! BLJ spots on object collision (`OBRRT_BLJ_SCAN_OBJECTS=1` with
//! `OBRRT_BLJ_SCAN=1`; default off).
//!
//! Generic, no behaviour list: every object with surfaces in the dynamic
//! partition near b ([`obrrt_core::dyngeo`], read with b's objects) gets the
//! static scan's clauses run on its current triangles:
//! - **Slope:** its floor triangles ([`super::scan::slope_ok`], at least
//!   [`OBJ_SLOPE_AREA`]);
//! - **Ceiling:** its floors under its own ceilings or the zone's static
//!   ones, and static floors under its ceilings (headroom 0..50);
//! - **Riser:** its wall triangles that are a short step's riser over its
//!   floors and the static floors around it ([`super::riser::find`]);
//! - **Wall:** those risers ending against a tall wall, its or static
//!   ([`super::riser::corners`]).
//!
//! A candidate is stored in the object's local frame (relative to its
//! position and facing yaw, [`ObjRef`]) with the triangle that defines it,
//! so the spot follows the object: [`holds`] re-derives the world start
//! and facing from the object's current position and yaw, and the gate is
//! the clause itself: the defining triangle must still be among the
//! object's current triangles at the same place in its local frame (a pitch
//! or roll, a different pose, the object gone: not), and a slope still a
//! slope, a ceiling's headroom still 0..50. No per-kind rules were needed
//! (a Whomp's top only matches while it lies flat, the pose it was found in).
//!
//! Moving objects (any |velocity| > [`MOVING_V`] or angle velocity >
//! [`MOVING_ANG`] in the state's object fields) are weighted lower.

use obrrt_core::dyngeo::{self, DynGeo};
use obrrt_core::geo::Tri;
use obrrt_core::{Obj, St};

use super::riser::{self, Corner, Riser};
use super::scan::{self, Cand};
use super::{Kind, RMode};

/// Smallest object slope triangle (projected).
pub const OBJ_SLOPE_AREA: f32 = 5000.0;
/// Vertex tolerance of the defining triangle's match (units, local frame).
pub const MATCH: f32 = 6.0;
pub const MOVING_V: f32 = 1.0;
pub const MOVING_ANG: i32 = 16;

/// The object and the spot in its frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjRef {
    pub kind: Kind,
    pub slot: u16,
    pub bhv: u32,
    pub id_home: [i32; 3],
    pub local_start: [f32; 3],
    pub local_yaw: i32,
    /// The defining triangle (local vertices): the slope / ceiling's floor
    /// or ceiling; for risers the riser itself in `riser`.
    pub key: [[f32; 3]; 3],
    /// Which list `key` is in (0 floor, 1 ceiling).
    pub key_list: u8,
    pub riser: Option<(Riser, RMode)>,
    pub pin: Option<Corner>,
    pub moving: bool,
}

/// The object's frame: position and facing yaw.
#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub pos: [f32; 3],
    pub yaw: i32,
}

impl Frame {
    pub fn of(o: &Obj) -> Frame {
        Frame { pos: o.pos, yaw: o.yaw }
    }
    fn cs(&self) -> (f32, f32) {
        let a = self.yaw as f32 * std::f32::consts::TAU / 65536.0;
        (a.cos(), a.sin())
    }
    /// World direction (x, z) of a local one.
    pub fn dir_w(&self, d: [f32; 2]) -> [f32; 2] {
        let (c, s) = self.cs();
        [d[0] * c + d[1] * s, -d[0] * s + d[1] * c]
    }
    pub fn dir_l(&self, d: [f32; 2]) -> [f32; 2] {
        let (c, s) = self.cs();
        [d[0] * c - d[1] * s, d[0] * s + d[1] * c]
    }
    pub fn to_w(&self, p: [f32; 3]) -> [f32; 3] {
        let d = self.dir_w([p[0], p[2]]);
        [self.pos[0] + d[0], self.pos[1] + p[1], self.pos[2] + d[1]]
    }
    pub fn to_l(&self, p: [f32; 3]) -> [f32; 3] {
        let d = self.dir_l([p[0] - self.pos[0], p[2] - self.pos[2]]);
        [d[0], p[1] - self.pos[1], d[1]]
    }
    pub fn riser(&self, r: &Riser, to_world: bool) -> Riser {
        let p = |q: [f32; 2], y: f32| {
            let v = if to_world { self.to_w([q[0], y, q[1]]) } else { self.to_l([q[0], y, q[1]]) };
            [v[0], v[2]]
        };
        let dy = if to_world { self.pos[1] } else { -self.pos[1] };
        let n = if to_world { self.dir_w(r.n) } else { self.dir_l(r.n) };
        Riser { a: p(r.a, r.ylo), b: p(r.b, r.ylo), n, ylo: r.ylo + dy, yhi: r.yhi + dy, corner: r.corner }
    }
    pub fn corner(&self, c: &Corner, to_world: bool) -> Corner {
        let pin = if to_world { self.to_w(c.pin) } else { self.to_l(c.pin) };
        let wall_n = if to_world { self.dir_w(c.wall_n) } else { self.dir_l(c.wall_n) };
        Corner { riser: self.riser(&c.riser, to_world), wall_n, pin, push_off: c.push_off }
    }
}

fn moving(g: &DynGeo, slot: u16) -> bool {
    g.motion_of(slot).is_some_and(|m| m.vel.iter().any(|v| v.abs() > MOVING_V) || m.ang_vel.iter().any(|a| a.abs() > MOVING_ANG))
}

fn bbox(ts: &[Tri]) -> [f32; 4] {
    let mut b = [f32::MAX, f32::MIN, f32::MAX, f32::MIN];
    for t in ts {
        for p in &t.v {
            b[0] = b[0].min(p[0]);
            b[1] = b[1].max(p[0]);
            b[2] = b[2].min(p[2]);
            b[3] = b[3].max(p[2]);
        }
    }
    b
}

fn near_box(ts: &[Tri], b: [f32; 4], m: f32) -> Vec<Tri> {
    ts.iter()
        .filter(|t| t.v.iter().any(|p| p[0] > b[0] - m && p[0] < b[1] + m && p[2] > b[2] - m && p[2] < b[3] + m))
        .map(|t| Tri { v: t.v, n: t.n, kind: t.kind, room: t.room })
        .collect()
}

fn lists(g: &DynGeo, slot: u16) -> [Vec<Tri>; 3] {
    let mut out = [Vec::new(), Vec::new(), Vec::new()];
    for s in g.of_slot(slot) {
        out[s.list as usize].push(s.tri());
    }
    out
}

/// The zone's published static floors, ceilings and walls (empty until
/// published, as the static scan).
fn statics(zone: (u16, u8)) -> (Vec<Tri>, Vec<Tri>, Vec<Tri>) {
    if !obrrt_core::mover::published(zone) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let cp = |v: Option<std::sync::Arc<[Tri]>>| v.map_or(Vec::new(), |a| a.iter().map(|t| Tri { v: t.v, n: t.n, kind: t.kind, room: t.room }).collect());
    (cp(obrrt_core::geo::static_floors(zone)), cp(obrrt_core::geo::static_ceils(zone)), cp(obrrt_core::geo::static_walls(zone)))
}

/// Every object candidate near b (world start and facing for now, the
/// local spot in `obj`). Empty without b's object collision.
pub fn cands(b: &St) -> Vec<Cand> {
    let Some(g) = dyngeo::of(b) else { return Vec::new() };
    let zone = b.zone();
    let (sf, sc, sw) = statics(zone);
    let mut out = Vec::new();
    let objs = b.objects();
    let mut d = scan::diag().lock().unwrap();
    for slot in g.slots() {
        let Some(o) = objs.iter().find(|o| o.slot == slot) else { continue };
        let [fl, ce, wa] = lists(&g, slot);
        if fl.is_empty() && ce.is_empty() && wa.is_empty() {
            continue;
        }
        let f = Frame::of(o);
        let mv = moving(&g, slot);
        d.obj_scanned.entry(zone).or_default().insert(o.id.to_string());
        let mk = |kind: Kind, start: [f32; 3], yaw: Option<i32>, key: [[f32; 3]; 3], key_list: u8, riser: Option<(Riser, RMode)>, pin: Option<Corner>| {
            // Ceilings: the fan's facing, chosen at generation (stored 0).
            let r = ObjRef {
                kind,
                slot,
                bhv: o.bhv,
                id_home: o.id.home,
                local_start: f.to_l(start),
                local_yaw: yaw.map_or(0, |y| y - f.yaw),
                key: key.map(|p| f.to_l(p)),
                key_list,
                riser: riser.map(|(r, m)| (f.riser(&r, false), m)),
                pin: pin.map(|c| f.corner(&c, false)),
                moving: mv,
            };
            Cand { kind, start, yaw, riser, pin, slot: None, anchor: start, obj: Some(r) }
        };
        let mut local: Vec<Cand> = Vec::new();
        for (c, i) in scan::slopes_ix(&fl, super::params::get().obj_slope_area) {
            local.push(mk(Kind::Slope, c.start, c.yaw, fl[i].v, 0, None, None));
        }
        let bb = bbox(&fl.iter().chain(&ce).chain(&wa).map(|t| Tri { v: t.v, n: t.n, kind: t.kind, room: t.room }).collect::<Vec<_>>());
        let sf_near = near_box(&sf, bb, 200.0);
        let sc_near = near_box(&sc, bb, 200.0);
        let sw_near = near_box(&sw, bb, 200.0);
        // Its floors under its ceilings and the static ones.
        let all_ce: Vec<Tri> = ce.iter().chain(&sc_near).map(|t| Tri { v: t.v, n: t.n, kind: t.kind, room: t.room }).collect();
        for (c, fi, _) in scan::ceilings_ix(&fl, &all_ce) {
            local.push(mk(Kind::Ceiling, c.start, None, fl[fi].v, 0, None, None));
        }
        // Static floors under its ceilings.
        if !ce.is_empty() {
            for (c, _, ci) in scan::ceilings_ix(&sf_near, &ce) {
                local.push(mk(Kind::Ceiling, c.start, None, ce[ci].v, 1, None, None));
            }
        }
        if !wa.is_empty() {
            let floors: Vec<Tri> = fl.iter().chain(&sf_near).map(|t| Tri { v: t.v, n: t.n, kind: t.kind, room: t.room }).collect();
            let rs = riser::find(&floors, &wa, super::params::get().riser_h_max);
            for c in scan::riser_cands(&rs) {
                local.push(mk(Kind::Riser, c.start, c.yaw, [[0.0; 3]; 3], 2, c.riser, None));
            }
            let walls: Vec<Tri> = wa.iter().chain(&sw_near).map(|t| Tri { v: t.v, n: t.n, kind: t.kind, room: t.room }).collect();
            for c in scan::wall_cands(&riser::corners(&walls, &rs)) {
                local.push(mk(Kind::Wall, c.start, c.yaw, [[0.0; 3]; 3], 2, c.riser, c.pin));
            }
        }
        for c in &local {
            let r = c.obj.unwrap();
            if super::spots::on() {
                super::spots::note(zone, &super::spots::owner_of(o), &format!("{:?}", c.kind), r.local_start, r.local_yaw, c.start, "dynamic", b.frame, None, false);
            }
            let lc = [(r.local_start[0] / 100.0).round() as i32, (r.local_start[1] / 100.0).round() as i32, (r.local_start[2] / 100.0).round() as i32];
            d.obj_cands.entry(zone).or_default().insert((o.id.to_string(), format!("{:?}", c.kind), lc), mv);
        }
        out.extend(local);
    }
    out
}

fn tri_match(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> bool {
    let close = |p: &[f32; 3], q: &[f32; 3]| (p[0] - q[0]).abs() < MATCH && (p[1] - q[1]).abs() < MATCH && (p[2] - q[2]).abs() < MATCH;
    a.iter().all(|p| b.iter().any(|q| close(p, q)))
}

fn riser_match(r: &Riser, w: &Riser) -> bool {
    let close = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < MATCH && (p[1] - q[1]).abs() < MATCH;
    (r.ylo - w.ylo).abs() < MATCH && (r.yhi - w.yhi).abs() < MATCH && ((close(r.a, w.a) && close(r.b, w.b)) || (close(r.a, w.b) && close(r.b, w.a)))
}

/// A wall triangle as a riser outline (its lexicographic end points and
/// height range; the normal from the candidate).
fn wall_outline(t: &Tri, n: [f32; 2]) -> Riser {
    let mut pts: Vec<[f32; 2]> = t.v.iter().map(|p| [p[0], p[2]]).collect();
    pts.sort_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
    let ys = t.v.map(|p| p[1]);
    Riser { a: pts[0], b: pts[2], n, ylo: ys[0].min(ys[1]).min(ys[2]), yhi: ys[0].max(ys[1]).max(ys[2]), corner: false }
}

/// Whether the spot still holds in `s` (module doc), and its world start,
/// facing, riser and pin now.
pub fn holds(s: &St, r: &ObjRef) -> Option<([f32; 3], i32, Option<(Riser, RMode)>, Option<Corner>)> {
    let o = s.objects().iter().find(|o| o.slot == r.slot && o.bhv == r.bhv)?;
    let g = dyngeo::of(s)?;
    let f = Frame::of(o);
    let start = f.to_w(r.local_start);
    let yaw = r.local_yaw + f.yaw;
    let [fl, ce, wa] = lists(&g, r.slot);
    let ok = match (r.riser, r.key_list) {
        (Some((rl, _)), _) => {
            // The riser wall still there in the object's frame.
            wa.iter().any(|t| {
                let lt = Tri { v: t.v.map(|p| f.to_l(p)), n: t.n, kind: t.kind, room: t.room };
                riser_match(&rl, &wall_outline(&lt, rl.n))
            })
        }
        (None, 0) => fl.iter().any(|t| {
            tri_match(&t.v.map(|p| f.to_l(p)), &r.key)
                && if r.kind == Kind::Slope {
                    // Still steep and not slippery.
                    scan::slope_ok(t)
                } else {
                    // The headroom over this floor at the start still 0..50.
                    let fy = scan::floor_y_at(t, start[0], start[2]);
                    let sc = if obrrt_core::mover::published(s.zone()) { obrrt_core::geo::static_ceils(s.zone()) } else { None };
                    fy.is_some_and(|fy| {
                        let c1 = scan::ceil_over(&ce, start[0], start[2], fy + 2.0);
                        let c2 = sc.as_ref().and_then(|c| scan::ceil_over(c, start[0], start[2], fy + 2.0));
                        let cy = match (c1, c2) {
                            (Some(a), Some(b)) => Some(a.min(b)),
                            (a, b) => a.or(b),
                        };
                        let pd = super::params::get();
                        cy.is_some_and(|cy| (pd.ceil_head_lo..pd.ceil_head_hi).contains(&(cy - fy - 160.0)))
                    })
                }
        }),
        (None, _) => ce.iter().any(|t| tri_match(&t.v.map(|p| f.to_l(p)), &r.key)),
    };
    ok.then(|| (start, yaw, r.riser.map(|(rl, m)| (f.riser(&rl, true), m)), r.pin.map(|c| f.corner(&c, true))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trip() {
        let f = Frame { pos: [100.0, 50.0, -200.0], yaw: 0x3000 };
        let p = [321.0, 77.0, -45.0];
        let q = f.to_w(f.to_l(p));
        assert!((p[0] - q[0]).abs() < 1e-3 && (p[1] - q[1]).abs() < 1e-3 && (p[2] - q[2]).abs() < 1e-3);
        // Local +z is the object's facing.
        let w = f.dir_w([0.0, 1.0]);
        let a = 0x3000 as f32 * std::f32::consts::TAU / 65536.0;
        assert!((w[0] - a.sin()).abs() < 1e-5 && (w[1] - a.cos()).abs() < 1e-5);
    }

    #[test]
    fn matches_follow_the_object_not_a_tilt() {
        let key = [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 120.0, 100.0]];
        let mut perm = [key[2], key[0], key[1]];
        assert!(tri_match(&perm, &key));
        perm[0][1] += 20.0;
        assert!(!tri_match(&perm, &key));
    }
}
