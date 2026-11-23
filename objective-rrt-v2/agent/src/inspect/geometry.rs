//! `geo`: collision around a point.

use std::collections::BTreeMap;

use obrrt_core::dyngeo;
use obrrt_core::geo::{self, Tri};
use serde_json::{Value, json};

use super::{View, atlas, r2, surface_name, v3};

type P = [f32; 3];
const LISTS: [&str; 3] = ["floor", "ceil", "wall"];

fn sub(a: P, b: P) -> P {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: P, b: P) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add_s(a: P, b: P, s: f32) -> P {
    [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
}

/// Distance from `p` to triangle `t` (closest point, Ericson's Real-Time Collision Detection 5.1.5).
fn tri_dist(p: P, t: &[P; 3]) -> f32 {
    let [a, b, c] = *t;
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let q = if d1 <= 0.0 && d2 <= 0.0 {
        a
    } else {
        let bp = sub(p, b);
        let (d3, d4) = (dot(ab, bp), dot(ac, bp));
        if d3 >= 0.0 && d4 <= d3 {
            b
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                add_s(a, ab, d1 / (d1 - d3))
            } else {
                let cp = sub(p, c);
                let (d5, d6) = (dot(ab, cp), dot(ac, cp));
                if d6 >= 0.0 && d5 <= d6 {
                    c
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        add_s(a, ac, d2 / (d2 - d6))
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            add_s(b, sub(c, b), w)
                        } else {
                            let denom = 1.0 / (va + vb + vc);
                            add_s(add_s(a, ab, vb * denom), ac, vc * denom)
                        }
                    }
                }
            }
        }
    };
    dot(sub(p, q), sub(p, q)).sqrt()
}

struct Near<'a> {
    d: f32,
    t: &'a Tri,
}

fn tri_json(n: &Near) -> Value {
    json!({"d": r2(n.d), "type": surface_name(n.t.kind), "v": [v3(n.t.v[0]), v3(n.t.v[1]), v3(n.t.v[2])], "n": [r2(n.t.n[0]), r2(n.t.n[1]), r2(n.t.n[2])]})
}

/// Triangles of `tris` within `radius` of `p`: counts by surface type and the `keep` nearest.
fn summarize(tris: &[Tri], p: P, radius: f32, keep: usize) -> Value {
    let mut near: Vec<Near> = tris.iter().filter_map(|t| {
        // Cheap reject on the bounding box before the exact distance.
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for v in &t.v {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        if (0..3).any(|k| p[k] < lo[k] - radius || p[k] > hi[k] + radius) {
            return None;
        }
        let d = tri_dist(p, &t.v);
        (d <= radius).then_some(Near { d, t })
    }).collect();
    near.sort_by(|a, b| a.d.total_cmp(&b.d));
    let mut by: BTreeMap<&str, u32> = BTreeMap::new();
    for n in &near {
        *by.entry(surface_name(n.t.kind)).or_default() += 1;
    }
    json!({"count": near.len(), "by_type": by, "nearest": near.iter().take(keep).map(tri_json).collect::<Vec<_>>()})
}

const DIRS: [(&str, f32, f32); 8] = [("+x", 1.0, 0.0), ("+x+z", 0.7071, 0.7071), ("+z", 0.0, 1.0), ("-x+z", -0.7071, 0.7071), ("-x", -1.0, 0.0), ("-x-z", -0.7071, -0.7071), ("-z", 0.0, -1.0), ("+x-z", 0.7071, -0.7071)];

pub fn geo(v: &View, radius: f32, point: Option<P>) -> Value {
    let st = v.st;
    let p = point.unwrap_or(st.pos);
    let pick = atlas::pick(st);
    let out = v.game.with_memory(|m| {
        let floors = geo::read_floors(m).0;
        let walls = geo::read_walls(m);
        let ceils = geo::read_ceils(m);
        let probe = |x: f32, z: f32| {
            let f = geo::floor_at(m, x, p[1], z);
            let c = f.and_then(|f| geo::ceil_at(m, x, f.y + 80.0, z));
            let tr = pick.as_ref().ok().and_then(|pk| atlas::trav_probe(pk, x, z, f.map_or(p[1], |f| f.y)));
            json!({
                "floor_y": f.map(|f| r2(f.y)), "type": f.map(|f| surface_name(f.kind)), "dy": f.map(|f| r2(f.y - p[1])),
                "normal_y": f.map(|f| r2(f.normal[1])), "ceil_y": c.map(|c| r2(c.y)), "atlas": tr,
            })
        };
        let mut probes = vec![];
        for d in [100.0f32, 300.0] {
            for (name, dx, dz) in DIRS {
                let mut e = probe(p[0] + dx * d, p[2] + dz * d);
                e["dir"] = json!(name);
                e["dist"] = json!(d);
                probes.push(e);
            }
        }
        let water: Vec<Value> = geo::read_water(m)
            .iter()
            .filter(|b| {
                let (dx, dz) = ((b.lo[0] - p[0]).max(p[0] - b.hi[0]).max(0.0), (b.lo[1] - p[2]).max(p[2] - b.hi[1]).max(0.0));
                dx.hypot(dz) <= radius
            })
            .map(|b| json!({"id": b.id, "kind": if b.is_water() { "water" } else { "gas" }, "x": [b.lo[0], b.hi[0]], "z": [b.lo[1], b.hi[1]], "y": b.y, "contains_point": b.contains(p[0], p[2])}))
            .collect();
        let dg = dyngeo::read(m, p);
        let mut slots: BTreeMap<u16, (u32, f32)> = BTreeMap::new();
        let mut dyn_tris: Vec<(f32, &dyngeo::DynSurf, Tri)> = vec![];
        for s in &dg.surfs {
            let t = s.tri();
            let d = tri_dist(p, &t.v);
            if d <= radius {
                let e = slots.entry(s.slot).or_insert((0, f32::MAX));
                e.0 += 1;
                e.1 = e.1.min(d);
                dyn_tris.push((d, s, t));
            }
        }
        let owners: Vec<Value> = slots
            .iter()
            .map(|(slot, (n, d))| {
                let mo = dg.motion_of(*slot);
                json!({"slot": slot, "bhv": st.objects().iter().find(|o| o.slot == *slot).map_or("?", |o| o.bhv_name()), "tris": n, "nearest": r2(*d),
                    "vel": mo.map(|m| v3(m.vel)), "ang_vel": mo.map(|m| m.ang_vel)})
            })
            .collect();
        dyn_tris.sort_by(|a, b| a.0.total_cmp(&b.0));
        let dyn_near: Vec<Value> = dyn_tris
            .iter()
            .take(6)
            .map(|(d, s, t)| json!({"d": r2(*d), "slot": s.slot, "list": LISTS[s.list as usize % 3], "type": surface_name(s.kind), "v": [v3(t.v[0]), v3(t.v[1]), v3(t.v[2])], "n": [r2(s.n[0]), r2(s.n[1]), r2(s.n[2])]}))
            .collect();
        json!({
            "floors": summarize(&floors, p, radius, 8), "walls": summarize(&walls, p, radius, 8), "ceilings": summarize(&ceils, p, radius, 4),
            "total_static": {"floors": floors.len(), "walls": walls.len(), "ceilings": ceils.len()},
            "at_point": probe(p[0], p[2]), "probes": probes, "water": water,
            "dynamic": {"owners": owners, "nearest": dyn_near},
        })
    });
    let mut res = json!({"frame": st.frame, "center": v3(p), "radius": radius, "mario_water_level": (st.water_level > -11000).then_some(st.water_level)});
    res["atlas"] = match &pick {
        Ok(pk) => json!({"map": pk.map, "variant": pk.variant, "note": pk.note}),
        Err(why) => json!({"available": false, "reason": why}),
    };
    if let (Some(r), Value::Object(o)) = (res.as_object_mut(), out) {
        r.extend(o);
    }
    res
}
