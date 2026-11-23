//! Author tool: an Env whose named-field reader also answers object queries
//! `obj:<cat>:<k>:<field>` (cat: enemy|breakable|coin|star|pole|any; k: k-th nearest; field: dist dist3 dy relyaw x y z act)
//! until sense.rs provides them. Cached per frame.
use obrrt_core::{St, read_state};
use obrrt_policy::eval::Env;
use sm64_binds::SM64Game;
use std::cell::RefCell;

struct O { cat: &'static str, d: f32, d3: f32, dy: f32, rel: f32, p: [f32; 3], act: i32 }

thread_local! { static CACHE: RefCell<Option<(u32, [f32; 3], Vec<O>)>> = const { RefCell::new(None) }; }

fn cat_of(n: &str) -> &'static str {
    const ENEMY: [&str; 18] = ["Goomba", "Bobomb", "BobOmb", "Chomp", "Koopa", "Bully", "Spiny", "Thwomp", "Whomp", "Piranha", "Boo", "Chuckya", "Scuttlebug", "KingBobomb", "Mr", "Fly", "Snufit", "Amp"];
    if n.contains("BreakableBox") { "breakable" }
    else if n.contains("Coin") { "coin" }
    else if n.contains("Star") && !n.contains("Spawn") { "star" }
    else if n.contains("Pole") { "pole" }
    else if ENEMY.iter().any(|e| n.contains(e)) { "enemy" }
    else { "other" }
}

pub fn env() -> Env {
    let mut env = Env::builtin();
    env.field = Box::new(|g: &SM64Game, s: &St, n: &str| -> Option<f32> {
        match n {
            "x" => return Some(s.pos[0]),
            "y" => return Some(s.pos[1]),
            "z" => return Some(s.pos[2]),
            _ => {}
        }
        let rest = n.strip_prefix("obj:")?;
        let mut it = rest.split(':');
        let (cat, k, name) = (it.next()?, it.next()?.parse::<usize>().ok()?, it.next()?);
        CACHE.with(|c| {
            let mut c = c.borrow_mut();
            if c.as_ref().is_none_or(|(f, p, _)| *f != s.frame || *p != s.pos) {
                let so = read_state(g, s.frame, true);
                let mut v: Vec<O> = so.objects().iter().filter_map(|o| {
                    let name = o.bhv_name();
                    let ct = cat_of(name);
                    if ct == "other" { return None; }
                    let (dx, dz) = (o.pos[0] - s.pos[0], o.pos[2] - s.pos[2]);
                    let yaw = (dx.atan2(dz) * 65536.0 / std::f32::consts::TAU) as i64 as i32 as i16;
                    let rel = (yaw.wrapping_sub(s.yaw)) as f32 * 360.0 / 65536.0;
                    Some(O { cat: ct, d: dx.hypot(dz), d3: (dx * dx + dz * dz + (o.pos[1] - s.pos[1]).powi(2)).sqrt(), dy: o.pos[1] - s.pos[1], rel, p: o.pos, act: o.action })
                }).collect();
                v.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap());
                *c = Some((s.frame, s.pos, v));
            }
            let v = &c.as_ref().unwrap().2;
            let o = v.iter().filter(|o| cat == "any" || o.cat == cat).nth(k);
            Some(match (o, name) {
                (None, "dist" | "dist3") => 99999.0,
                (None, _) => 0.0,
                (Some(o), "dist") => o.d,
                (Some(o), "dist3") => o.d3,
                (Some(o), "dy") => o.dy,
                (Some(o), "relyaw") => o.rel,
                (Some(o), "x") => o.p[0],
                (Some(o), "y") => o.p[1],
                (Some(o), "z") => o.p[2],
                (Some(o), "act") => o.act as f32,
                _ => 0.0,
            })
        })
    });
    env
}
