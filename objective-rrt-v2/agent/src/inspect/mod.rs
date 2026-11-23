//! State analysis for `ar`: what a game state looks like, as compact JSON
//! (`state`, `progress`, `objects`, `geo`, `behavior`, `level`, `objectives`,
//! `diff`) and as short text ([`render`]). Reads the game's memory through
//! `obrrt_core`'s layout; static level data comes from sm64-atlas.

#![allow(non_upper_case_globals)]

mod atlas;
mod behavior_src;
mod geometry;
mod names;
mod offers;
mod render;

use obrrt_core::layout::*;
use obrrt_core::{Obj, St, bhv_name};
use serde_json::{Value, json};
use sm64_binds::SM64Game;

pub use names::{action_name, course_name, level_name, surface_name};
pub use render::render;

/// A state with the game it was read from (`st` read with objects).
pub struct View<'a> {
    pub game: &'a SM64Game,
    pub st: &'a St,
}

// Object field offsets the layout doesn't have, from include/object_fields.h
// (the hex in its comments is the byte offset in `struct Object`; they agree
// with layout.rs: oAction 0x14C = 332, oTimer 0x154 = 340).
const OBJ_oFlags: usize = 0x8C;
const OBJ_oPrevAction: usize = 0x18C;

/// Little-endian reads that give 0 past the end of memory.
pub(crate) struct Mem<'a>(pub &'a [u8]);

impl Mem<'_> {
    pub fn u32(&self, a: usize) -> u32 {
        self.0.get(a..a + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }
    pub fn i32(&self, a: usize) -> i32 {
        self.u32(a) as i32
    }
    pub fn i16(&self, a: usize) -> i16 {
        self.0.get(a..a + 2).map_or(0, |b| i16::from_le_bytes(b.try_into().unwrap()))
    }
    pub fn u16(&self, a: usize) -> u16 {
        self.i16(a) as u16
    }
    pub fn i8(&self, a: usize) -> i8 {
        self.0.get(a).map_or(0, |&b| b as i8)
    }
    pub fn f32(&self, a: usize) -> f32 {
        f32::from_bits(self.u32(a))
    }
    pub fn vec3(&self, a: usize) -> [f32; 3] {
        [self.f32(a), self.f32(a + 4), self.f32(a + 8)]
    }
}

const MARIO: usize = ADDR_gMarioStates as usize;
const POOL: usize = ADDR_gObjectPool as usize;

// ---- small JSON helpers ----

pub(crate) fn r2(x: f32) -> Value {
    let v = (x as f64 * 100.0).round() / 100.0;
    if v.is_finite() { json!(v) } else { Value::Null }
}

pub(crate) fn v3(p: [f32; 3]) -> Value {
    json!([r2(p[0]), r2(p[1]), r2(p[2])])
}

pub(crate) fn hex(x: u32) -> String {
    format!("{x:#x}")
}

fn yaw_deg(y: i32) -> f32 {
    (y as u16) as f32 * 360.0 / 65536.0
}

pub(crate) fn bhv_of(addr: u32) -> String {
    bhv_name(addr).map_or_else(|| if addr == 0 { "(none)".to_string() } else { format!("bhv@{addr:#x}") }, str::to_string)
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    obrrt_core::dist(a, b)
}

// ---- state ----

/// Mario's floor class and slope flags (mario_get_floor_class, mario_floor_is_slope,
/// mario_floor_is_steep in mario.c). Not modelled: the slide terrain type
/// (every floor of a slide area counts as very slippery in the game) and
/// `steep`'s "not facing downhill" exemption.
fn floor_info(st: &St) -> Value {
    if st.floor_type < 0 {
        return json!({"y": r2(st.floor_y), "none": true});
    }
    let class = match st.floor_type {
        0x15 | 0x37 | 0x7A => "not_slippery",
        0x14 | 0x2A | 0x35 | 0x79 => "slippery",
        0x13 | 0x2E | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => "very_slippery",
        _ => "default",
    };
    let ny = st.floor_normal[1];
    let slope = ny <= match class { "very_slippery" => 0.9961947, "slippery" => 0.9848077, "not_slippery" => 0.9396926, _ => 0.9659258 };
    let steep = ny <= match class { "very_slippery" => 0.9659258, "slippery" => 0.9396926, _ => 0.8660254 };
    let obj = st.floor_obj.map(|s| json!({"slot": s, "bhv": slot_bhv(st, s)}));
    json!({
        "y": r2(st.floor_y), "type": surface_name(st.floor_type), "type_id": hex(st.floor_type as u32), "normal": v3(st.floor_normal),
        "class": class, "slope": slope, "steep": steep, "object": obj,
    })
}

fn slot_bhv(st: &St, slot: u16) -> String {
    st.objects().iter().find(|o| o.slot == slot).map_or_else(|| "?".to_string(), |o| o.bhv_name().to_string())
}

pub fn state(v: &View) -> Value {
    let st = v.st;
    let (cap_timer, invinc, hurt, heal, squish) = v.game.with_memory(|m| {
        let m = Mem(m);
        (m.u16(MARIO + MARIO_capTimer), m.i16(MARIO + MARIO_invincTimer), m.i8(MARIO + MARIO_hurtCounter), m.i8(MARIO + MARIO_healCounter), m.u16(MARIO + MARIO_squishTimer))
    });
    let held = st.held_slot.map(|s| json!({"slot": s, "bhv": slot_bhv(st, s)}));
    let modal = if st.in_dialog() {
        json!({"kind": "dialog", "dialog_id": st.dialog, "box_state": st.dialog_state, "last_page": st.dialog_last_page, "choice": st.dialog_choice, "cursor": st.menu_line})
    } else if st.in_save_menu() {
        json!({"kind": "save_menu", "cursor": st.menu_line, "timer": st.menu_timer})
    } else if st.act_select {
        json!({"kind": "act_select", "cursor": st.act_cursor, "first_missing_act": st.act_init, "timer": st.menu_timer})
    } else if st.dialog >= 0 {
        json!({"kind": "dialog_closing", "dialog_id": st.dialog, "box_state": st.dialog_state})
    } else {
        Value::Null
    };
    let water = (st.water_level > -11000).then(|| json!({"level": st.water_level, "depth": r2(st.water_level as f32 - st.pos[1])}));
    let cap_flags: Vec<String> = names::bits(names::MARIO_FLAGS, st.mario_flags & 0x3F).iter().map(|s| s.trim_start_matches("MARIO_").to_string()).collect();
    json!({
        "frame": st.frame,
        "zone": {"level": st.level, "level_name": level_name(st.level), "area": st.area, "act": st.act, "course": st.course, "course_name": course_name(st.course)},
        "mario": {
            "pos": v3(st.pos), "vel": v3(st.vel), "forward_vel": r2(st.forward_vel),
            "yaw": st.yaw as u16, "yaw_deg": r2(yaw_deg(st.yaw as i32)), "pitch": st.pitch,
            "action": {"value": hex(st.action), "name": action_name(st.action),
                "flags": names::bits(names::ACT_FLAGS, st.action & !0x1FF & !0x1C0).iter().map(|s| s.trim_start_matches("ACT_FLAG_").to_string()).collect::<Vec<_>>()},
            "action_timer": st.action_timer,
            "health": {"raw": st.health, "wedges": st.wedges()},
            "coins": st.coins, "lives": st.lives, "stars": st.stars, "keys": st.keys,
            "cap": {"flags": cap_flags, "timer": cap_timer},
            "invincibility": invinc, "hurt_counter": hurt, "heal_counter": heal, "squish_timer": squish,
            "held": held,
            "mario_flags": names::bits(names::MARIO_FLAGS, st.mario_flags),
            "floor": floor_info(st),
            "ceil_y": (st.ceil_y < 19999.0).then(|| r2(st.ceil_y)),
            "wall_normal": st.wall_normal.map(v3),
            "water": water,
        },
        "controllable": st.controllable(),
        "play_mode": st.play_mode, "transition": st.transition,
        "modal": modal,
        "camera_yaw": st.cam_yaw as u16, "camera_yaw_deg": r2(yaw_deg(st.cam_yaw as i32)),
    })
}

// ---- progress ----

pub fn progress(st: &St) -> Value {
    let mut courses = serde_json::Map::new();
    for (i, &b) in st.course_stars.iter().enumerate() {
        let course = i as i16 + 1;
        if b & 0x7F == 0 && b & 0x80 == 0 {
            continue;
        }
        let stars: Vec<u32> = (0..7).filter(|k| b & (1 << k) != 0).map(|k| k + 1).collect();
        let mut e = json!({"stars": stars});
        if b & 0x80 != 0 {
            e["cannon_open"] = json!(true);
        }
        e["course"] = json!(course);
        courses.insert(course_name(course).to_string(), e);
    }
    let course_total: u32 = st.course_stars.iter().map(|b| (b & 0x7F).count_ones()).sum();
    let castle = ((st.save_flags >> 24) & 0x7F).count_ones();
    let flags = names::bits(names::SAVE_FLAGS, st.save_flags);
    let all: Vec<&str> = names::SAVE_FLAGS.iter().filter(|(b, _)| st.save_flags & b == 0).map(|(_, n)| &n["SAVE_FLAG_".len()..]).collect();
    let loaded_red = st.objects().iter().filter(|o| o.bhv_name() == "bhvRedCoin").count();
    json!({
        "stars": st.stars, "stars_from_flags": course_total + castle, "course_stars": course_total, "castle_stars": castle, "keys": st.keys,
        "courses": courses,
        "save_flags": {"raw": hex(st.save_flags), "set": flags.iter().map(|s| s.trim_start_matches("SAVE_FLAG_")).collect::<Vec<_>>(), "clear": all},
        "red_coins": {"collected": st.red_coins, "loaded_now": loaded_red, "in": course_name(st.course)},
    })
}

// ---- objects ----

fn obj_json(m: &Mem, slot: usize, mario: [f32; 3]) -> Option<Value> {
    let o = POOL + slot * OBJECT_SIZE;
    let active = m.i16(o + OBJ_activeFlags) as u16 as u32;
    if active & ACTIVE_FLAG_ACTIVE == 0 {
        return None;
    }
    if o == m.u32(MARIO + MARIO_marioObj) as usize {
        return None; // Mario himself (St.objects leaves him out too)
    }
    let bhv = m.u32(o + OBJ_behavior);
    let pos = m.vec3(o + OBJ_oPosX);
    let it = m.u32(o + OBJ_oInteractType);
    let status = m.u32(o + OBJ_oInteractStatus);
    let yaw = m.i32(o + OBJ_oFaceAngleYaw);
    Some(json!({
        "slot": slot, "bhv": bhv_of(bhv), "pos": v3(pos), "home": v3(m.vec3(o + OBJ_oHomeX)), "dist": r2(dist3(pos, mario)),
        "yaw": yaw as u16, "yaw_deg": r2(yaw_deg(yaw)),
        "vel": v3([m.f32(o + OBJ_oVelX), m.f32(o + OBJ_oVelY), m.f32(o + OBJ_oVelZ)]), "forward_vel": r2(m.f32(o + OBJ_oForwardVel)),
        "action": m.i32(o + OBJ_oAction), "prev_action": m.i32(o + OBJ_oPrevAction), "sub_action": m.i32(o + OBJ_oSubAction), "timer": m.i32(o + OBJ_oTimer),
        "oflags": hex(m.u32(o + OBJ_oFlags)), "health": m.i32(o + OBJ_oHealth),
        "interact": {"type": names::bits(names::INTERACTS, it).iter().map(|s| s.trim_start_matches("INTERACT_")).collect::<Vec<_>>(),
            "status": names::bits(names::INT_STATUS, status).iter().map(|s| s.trim_start_matches("INT_STATUS_")).collect::<Vec<_>>()},
        "held_state": m.u32(o + OBJ_oHeldState), "params": hex(m.u32(o + OBJ_oBehParams)), "intangible": m.i32(o + OBJ_oIntangibleTimer),
        "hitbox": [r2(m.f32(o + OBJ_hitboxRadius)), r2(m.f32(o + OBJ_hitboxHeight))],
        "active": names::bits(names::ACTIVE_FLAGS, active & !ACTIVE_FLAG_ACTIVE).iter().map(|s| s.trim_start_matches("ACTIVE_FLAG_")).collect::<Vec<_>>(),
    }))
}

/// Strips a leading `bhv` and lowercases, for matching names.
fn bhv_key(s: &str) -> String {
    let l = s.to_lowercase();
    l.strip_prefix("bhv").unwrap_or(&l).to_string()
}

pub fn objects(v: &View, near: Option<f32>, bhv: Option<&str>) -> Value {
    let pat = bhv.map(bhv_key);
    let (mut list, total) = v.game.with_memory(|m| {
        let m = Mem(m);
        let mut total = 0;
        let list: Vec<Value> = (0..OBJECT_POOL_CAPACITY as usize)
            .filter_map(|i| obj_json(&m, i, v.st.pos))
            .inspect(|_| total += 1)
            .filter(|o| near.is_none_or(|n| o["dist"].as_f64().unwrap_or(f64::MAX) <= n as f64))
            .filter(|o| pat.as_ref().is_none_or(|p| bhv_key(o["bhv"].as_str().unwrap_or("")).contains(p.as_str())))
            .collect();
        (list, total)
    });
    list.sort_by(|a, b| a["dist"].as_f64().partial_cmp(&b["dist"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    json!({"frame": v.st.frame, "mario": v3(v.st.pos), "active_total": total, "shown": list.len(), "near": near, "bhv": bhv, "objects": list})
}

// ---- diff ----

fn trimmed(t: &'static [(u32, &'static str)], v: u32, prefix: &str) -> Vec<String> {
    names::bits(t, v).iter().map(|s| s.trim_start_matches(prefix).to_string()).collect()
}

fn pair<T: serde::Serialize + PartialEq>(a: T, b: T) -> Value {
    if a == b { Value::Null } else { json!({"from": a, "to": b}) }
}

fn obj_brief(o: &Obj) -> Value {
    json!({"slot": o.slot, "bhv": o.bhv_name(), "pos": v3(o.pos), "action": o.action})
}

pub fn diff(a: &View, b: &View) -> Value {
    let (sa, sb) = (a.st, b.st);
    let mut out = serde_json::Map::new();
    let mut put = |k: &str, v: Value| {
        if !v.is_null() {
            out.insert(k.to_string(), v);
        }
    };
    put("frames", json!({"from": sa.frame, "to": sb.frame, "delta": sb.frame as i64 - sa.frame as i64}));
    put("zone", if sa.zone() == sb.zone() && sa.act == sb.act {
        Value::Null
    } else {
        json!({"from": format!("{} a{} act{}", level_name(sa.level), sa.area, sa.act), "to": format!("{} a{} act{}", level_name(sb.level), sb.area, sb.act)})
    });
    let d = [sb.pos[0] - sa.pos[0], sb.pos[1] - sa.pos[1], sb.pos[2] - sa.pos[2]];
    put("pos", json!({"from": v3(sa.pos), "to": v3(sb.pos), "delta": v3(d), "dist": r2(dist3(sa.pos, sb.pos)), "hdist": r2(d[0].hypot(d[2]))}));
    put("action", pair(action_name(sa.action), action_name(sb.action)));
    put("forward_vel", pair(sa.forward_vel.round() as i32, sb.forward_vel.round() as i32));
    put("health", pair(sa.wedges(), sb.wedges()));
    put("coins", pair(sa.coins, sb.coins));
    put("stars", pair(sa.stars, sb.stars));
    put("keys", pair(sa.keys, sb.keys));
    put("lives", pair(sa.lives, sb.lives));
    put("red_coins", pair(sa.red_coins, sb.red_coins));
    put("held", pair(sa.held().map(|o| o.bhv_name()), sb.held().map(|o| o.bhv_name())));
    if sa.mario_flags & 0x3F != sb.mario_flags & 0x3F {
        put("cap", json!({"from": trimmed(names::MARIO_FLAGS, sa.mario_flags & 0x3F, "MARIO_"), "to": trimmed(names::MARIO_FLAGS, sb.mario_flags & 0x3F, "MARIO_")}));
    }
    let set = sb.save_flags & !sa.save_flags;
    let cleared = sa.save_flags & !sb.save_flags;
    if set | cleared != 0 {
        put("save_flags", json!({"set": trimmed(names::SAVE_FLAGS, set, "SAVE_FLAG_"), "cleared": trimmed(names::SAVE_FLAGS, cleared, "SAVE_FLAG_")}));
    }
    let gained: Vec<Value> = (0..25)
        .filter_map(|i| {
            let (x, y) = (sa.course_stars[i], sb.course_stars[i]);
            let new = y & !x;
            (new != 0).then(|| json!({"course": course_name(i as i16 + 1), "stars": (0..7).filter(|k| new & (1 << k) != 0).map(|k| k + 1).collect::<Vec<_>>(), "cannon": new & 0x80 != 0}))
        })
        .collect();
    put("course_stars_gained", if gained.is_empty() { Value::Null } else { json!(gained) });
    // Objects by pool slot.
    let find = |s: &St, slot: u16| s.objects().iter().find(|o| o.slot == slot).cloned();
    let (mut spawned, mut despawned, mut changed) = (vec![], vec![], vec![]);
    for ob in sb.objects() {
        match find(sa, ob.slot) {
            Some(oa) if oa.bhv == ob.bhv => {
                let moved = dist3(oa.pos, ob.pos);
                if oa.action != ob.action || oa.health != ob.health || moved >= 30.0 || oa.held_state != ob.held_state {
                    let mut e = json!({"slot": ob.slot, "bhv": ob.bhv_name()});
                    if oa.action != ob.action {
                        e["action"] = json!([oa.action, ob.action]);
                    }
                    if oa.health != ob.health {
                        e["health"] = json!([oa.health, ob.health]);
                    }
                    if oa.held_state != ob.held_state {
                        e["held_state"] = json!([oa.held_state, ob.held_state]);
                    }
                    if moved >= 30.0 {
                        e["moved"] = json!({"to": v3(ob.pos), "dist": r2(moved)});
                    }
                    changed.push(e);
                }
            }
            Some(oa) => {
                despawned.push(obj_brief(&oa));
                spawned.push(obj_brief(ob));
            }
            None => spawned.push(obj_brief(ob)),
        }
    }
    for oa in sa.objects() {
        if find(sb, oa.slot).is_none() {
            despawned.push(obj_brief(oa));
        }
    }
    if sa.objects.is_some() && sb.objects.is_some() {
        put("objects", json!({"spawned": spawned, "despawned": despawned, "changed": changed, "count": [sa.objects().len(), sb.objects().len()]}));
    }
    Value::Object(out)
}

// ---- other entry points ----

pub fn geo(v: &View, radius: f32, point: Option<[f32; 3]>) -> Value {
    geometry::geo(v, radius, point)
}

pub fn behavior(name: &str) -> anyhow::Result<Value> {
    behavior_src::behavior(name)
}

pub fn level(st: &St, all: bool) -> anyhow::Result<Value> {
    atlas::level(st, all)
}

pub fn objectives(st: &St, kind: Option<&str>) -> Value {
    offers::objectives(st, kind)
}

pub fn summary_line(st: &St) -> String {
    format!(
        "f{} {} a{}{} ({:.0},{:.0},{:.0}) {} {}* {}c {}hp",
        st.frame,
        level_name(st.level),
        st.area,
        if st.act > 0 { format!(" act{}", st.act) } else { String::new() },
        st.pos[0],
        st.pos[1],
        st.pos[2],
        action_name(st.action),
        st.stars,
        st.coins,
        st.wedges()
    )
}
