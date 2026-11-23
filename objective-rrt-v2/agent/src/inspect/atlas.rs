//! sm64-atlas data (static level info, traversability) for a state's map.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use obrrt_core::St;
use serde_json::{Value, json};

use super::{course_name, level_name, v3};

pub fn root() -> PathBuf {
    PathBuf::from(std::env::var("SM64_ATLAS").unwrap_or_else(|_| "/workspaces/sm64-combined/sm64-atlas".into()))
}

/// The atlas map and variant that best match a state.
pub struct Pick {
    pub map: String,
    pub variant: String,
    pub label: String,
    pub dir: PathBuf,
    pub note: String,
}

fn read_json(p: &PathBuf) -> Result<Arc<Value>> {
    static C: OnceLock<Mutex<HashMap<PathBuf, Arc<Value>>>> = OnceLock::new();
    let c = C.get_or_init(Default::default);
    if let Some(v) = c.lock().unwrap().get(p) {
        return Ok(v.clone());
    }
    let v: Value = serde_json::from_slice(&std::fs::read(p).with_context(|| format!("reading {}", p.display()))?).with_context(|| format!("parsing {}", p.display()))?;
    let v = Arc::new(v);
    c.lock().unwrap().insert(p.clone(), v.clone());
    Ok(v)
}

/// Acts a variant slug like `act1-2-3-4-6` covers.
fn slug_acts(slug: &str) -> Vec<u8> {
    slug.strip_prefix("act").map_or(vec![], |r| r.split('-').filter_map(|p| p.parse().ok()).collect())
}

/// Picks the map for `st`'s (level, area) and the variant that matches its act,
/// save flags, star count and cap switches best. `Err(reason)` when the atlas
/// has no such map.
pub fn pick(st: &St) -> Result<Pick, String> {
    let index = read_json(&root().join("out/index.json")).map_err(|e| format!("atlas index unavailable: {e:#}"))?;
    let maps = index["maps"].as_array().ok_or("atlas index has no maps")?;
    let map = maps
        .iter()
        .find(|m| m["level"].as_u64() == Some(st.level as u64) && m["area"].as_u64() == Some(st.area as u64))
        .ok_or_else(|| format!("atlas has no map for level {} ({}) area {}", st.level, level_name(st.level), st.area))?;
    let variants = map["variants"].as_array().ok_or("map has no variants")?;
    // The act group the state is in (variants with an act and an `actN-..` slug).
    let act_ok = |v: &Value| {
        let slug = v["variant"].as_str().unwrap_or("");
        match v["act"].as_u64() {
            None => true,
            Some(a) if slug.starts_with("act") => slug_acts(slug).contains(&st.act) || (st.act == 0 && a == 1),
            Some(_) => true,
        }
    };
    let base_act = variants.iter().find(|v| v["act"].as_u64().is_some() && v["variant"].as_str().unwrap_or("").starts_with("act") && act_ok(v)).and_then(|v| v["act"].as_u64());
    let cap_bit = |s: &str| match s {
        "cap_wing" => Some(0x2u32),
        "cap_metal" => Some(0x4),
        "cap_vanish" => Some(0x8),
        _ => None,
    };
    let mut best: Option<(i64, &Value, String)> = None;
    for v in variants {
        let t = &v["toggles"];
        let flags = t["save_flags"].as_u64().unwrap_or(0) as u32;
        let stars = t["stars"].as_i64().unwrap_or(0);
        let switches: Vec<&str> = t["switches"].as_array().map_or(vec![], |a| a.iter().filter_map(|x| x.as_str()).collect());
        let slug = v["variant"].as_str().unwrap_or("");
        if !act_ok(v) || flags & !st.save_flags != 0 || stars > st.stars as i64 {
            continue;
        }
        if let (Some(a), Some(b)) = (v["act"].as_u64(), base_act) {
            if !slug.starts_with("act") && a != b {
                continue;
            }
        }
        if switches.iter().any(|s| cap_bit(s).is_none_or(|b| st.save_flags & b == 0)) {
            continue;
        }
        let score = flags.count_ones() as i64 * 1000 + stars * 10 + switches.len() as i64 * 100 + slug.starts_with("act") as i64;
        if best.as_ref().is_none_or(|b| score > b.0) {
            best = Some((score, v, slug.to_string()));
        }
    }
    let (_, v, slug) = best.or_else(|| variants.first().map(|v| (0, v, v["variant"].as_str().unwrap_or("").to_string()))).ok_or("map has no usable variant")?;
    let m = map["map"].as_str().unwrap_or("?").to_string();
    let n = variants.len();
    Ok(Pick {
        dir: root().join("out").join(&m).join(&slug),
        label: v["label"].as_str().unwrap_or("").to_string(),
        note: format!("variant chosen of {n} by act/save flags/stars/cap switches; cannon, floor-switch and THI-drain toggles are not visible in the save flags, so those variants are never chosen"),
        variant: slug,
        map: m,
    })
}

pub fn scene(p: &Pick) -> Result<Arc<Value>> {
    read_json(&p.dir.join("scene.json"))
}

/// Flag bit names of trav.json `flag_bits`, in bit order.
const FLAG_NAMES: [(&str, u32); 16] = [
    ("standable", 0), ("steep", 1), ("slippery_type", 2), ("death", 3), ("lava", 4), ("quicksand", 5), ("instant_warp", 6), ("warp", 7),
    ("underwater", 8), ("low_ceiling", 9), ("near_wall", 10), ("ledge", 11), ("object_floor", 12), ("flowing", 13), ("hangable_ceiling", 14), ("reachable", 15),
];

/// Traversability at (x, z): the atlas point (pole) nearest in the grid whose floor is
/// closest to `y_ref`, if within 150 units of it. Reads a few bytes of trav.bin.
pub fn trav_probe(p: &Pick, x: f32, z: f32, y_ref: f32) -> Option<Value> {
    let hdr = read_json(&p.dir.join("trav.json")).ok()?;
    let (x0, z0, s) = (hdr["x0"].as_f64()?, hdr["z0"].as_f64()?, hdr["s"].as_f64()?);
    let (nx, nz) = (hdr["nx"].as_i64()?, hdr["nz"].as_i64()?);
    let (ix, iz) = (((x as f64 - x0) / s).round() as i64, ((z as f64 - z0) / s).round() as i64);
    if ix < 0 || iz < 0 || ix >= nx || iz >= nz {
        return None;
    }
    let idx = (iz * nx + ix) as u64;
    let off = |k: &str| hdr["layers"][k]["offset"].as_u64();
    let mut f = std::fs::File::open(p.dir.join("trav.bin")).ok()?;
    let mut buf4 = [0u8; 4];
    let mut at = |o: u64| -> Option<[u8; 4]> {
        f.seek(SeekFrom::Start(o)).ok()?;
        f.read_exact(&mut buf4).ok()?;
        Some(buf4)
    };
    let a = u32::from_le_bytes(at(off("pole_start")? + 4 * idx)?) as u64;
    let b = u32::from_le_bytes(at(off("pole_start")? + 4 * idx + 4)?) as u64;
    let mut best: Option<(f32, u64)> = None;
    for k in a..b.min(a + 64) {
        let y = f32::from_le_bytes(at(off("y")? + 4 * k)?);
        if best.is_none_or(|(by, _)| (y - y_ref).abs() < (by - y_ref).abs()) {
            best = Some((y, k));
        }
    }
    let (y, k) = best.filter(|(y, _)| (y - y_ref).abs() <= 150.0)?;
    let flags = u32::from_le_bytes(at(off("flags")? + 4 * k)?);
    let ceil = f32::from_le_bytes(at(off("ceil")? + 4 * k)?);
    let push = f32::from_le_bytes(at(off("wall_push")? + 4 * k)?);
    let region = i32::from_le_bytes(at(off("region")? + 4 * k)?);
    let dist = f32::from_le_bytes(at(off("dist")? + 4 * k)?);
    let water = f32::from_le_bytes(at(off("water")? + 4 * k)?);
    let names: Vec<&str> = FLAG_NAMES.iter().filter(|(_, b)| flags & (1 << b) != 0).map(|(n, _)| *n).collect();
    Some(json!({
        "y": super::r2(y), "flags": names, "ceil": (ceil < 19999.0).then(|| super::r2(ceil)), "wall_push": super::r2(push), "region": region,
        "walk_frames": dist.is_finite().then(|| super::r2(dist)), "water": (water > -10000.0).then(|| super::r2(water)),
    }))
}

fn acts_list(mask: u64) -> Vec<u8> {
    (1..=6).filter(|a| mask & (1 << (a - 1)) != 0).collect()
}

/// Categories listed one by one in the compact view; the rest are counted.
const COMPACT: &[&str] = &["star", "boss", "door", "warp", "cannon", "switch", "red_coin", "hazard", "npc"];

pub fn level(st: &St, all: bool) -> Result<Value> {
    let zone = json!({"level": st.level, "level_name": level_name(st.level), "area": st.area, "act": st.act, "course": course_name(st.course)});
    let p = match pick(st) {
        Ok(p) => p,
        Err(why) => return Ok(json!({"available": false, "reason": why, "zone": zone})),
    };
    let scene = scene(&p)?;
    let act_bit = if st.act >= 1 { 1u64 << (st.act - 1) } else { u64::MAX };
    let mut counts: std::collections::BTreeMap<String, (u32, u32)> = Default::default();
    let mut objs = vec![];
    let mut grouped: std::collections::BTreeMap<String, std::collections::BTreeMap<String, u32>> = Default::default();
    for o in scene["objects"].as_array().map_or(&[][..], |v| v.as_slice()) {
        let cat = o["category"].as_str().unwrap_or("?");
        let acts = o["acts"].as_u64().unwrap_or(31);
        let live = acts & act_bit != 0;
        let c = counts.entry(cat.to_string()).or_default();
        c.0 += 1;
        c.1 += live as u32;
        if !live && !all {
            continue;
        }
        if !all && !COMPACT.contains(&cat) {
            *grouped.entry(cat.to_string()).or_default().entry(o["bhv"].as_str().unwrap_or("?").to_string()).or_insert(0) += 1;
            continue;
        }
        let pos = &o["pos"];
        let mut e = json!({"i": o["i"], "bhv": o["bhv"], "pos": pos, "cat": cat, "toggle": o["toggle"]});
        if let Some(p) = pos.as_array().filter(|p| p.len() == 3) {
            let f = |k: usize| p[k].as_f64().unwrap_or(0.0) as f32;
            e["dist"] = json!(super::r2(obrrt_core::dist([f(0), f(1), f(2)], st.pos)));
        }
        if acts != 31 && acts != 63 {
            e["acts"] = json!(acts_list(acts));
        }
        if o["note"].as_str().is_some_and(|n| !n.is_empty()) {
            e["note"] = o["note"].clone();
        }
        if all {
            e["bparam"] = json!(super::hex(o["bparam"].as_u64().unwrap_or(0) as u32));
            e["model"] = o["model"].clone();
            e["origin"] = o["origin"].clone();
            e["angle"] = o["angle"].clone();
            e["collision"] = o["collision"].clone();
        }
        objs.push(e);
    }
    if !all {
        objs.sort_by(|a, b| a["dist"].as_f64().partial_cmp(&b["dist"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    }
    Ok(json!({
        "available": true, "zone": zone,
        "atlas": {"map": p.map, "variant": p.variant, "label": p.label, "note": p.note},
        "bounds": scene["bounds"], "all": all,
        "category_counts": counts.iter().map(|(k, v)| (k.clone(), json!({"total": v.0, "this_act": v.1}))).collect::<serde_json::Map<_, _>>(),
        "objects": objs, "others": grouped, "warps": scene["warps"], "instant_warps": scene["instant_warps"],
        "water": scene["water"], "entries": scene["entries"],
        "mario": v3(st.pos),
    }))
}
