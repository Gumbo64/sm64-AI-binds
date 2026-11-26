//! The BLJ spot registry (`OBRRT_BLJ_SPOTS=1`, default off: nothing is
//! kept, written or generated).
//!
//! Every BLJ spot the run learns of, deduplicated:
//! - **key:** zone; owner (`static`, or the object's behaviour and home
//!   position, never its slot); clause; the start on a 64-unit grid (in the
//!   object's frame for object spots); the facing octant. Near-duplicates
//!   (same zone, owner and clause, start within [`MERGE_DIST`], facing
//!   within [`MERGE_YAW`]) merge into the first spot (counted in `merges`).
//! - **sources:** `static` (the scan's zone candidates), `dynamic` (object
//!   candidates found in-game), `observed` (a long-jump chain in the tree,
//!   any objective, that reached fv <= -100: its first chain node, fv <=
//!   -16, is the spot), `preset` (blj.rs SPOTS), `catalog`
//!   (`OBRRT_BLJ_SPOTS_CATALOG=<file>`, tools/analysis/blj_spots.py's merge
//!   of earlier runs).
//! - **per spot:** first seen (frame, node), seen as a candidate, attempted,
//!   succeeded (BljAny attempts at it), observed chains, best |fv|, mean
//!   segment frames of its attempts, the setup variants tried.
//! - **misses:** an observed spot with no static / dynamic / preset /
//!   catalog spot within [`MERGE_DIST`] (nor a static zone candidate) is a
//!   miss: its context (floor, ceiling headroom, wall, owner, facing and
//!   speed) and the triangles within [`DUMP_R`] of the start and of the
//!   point [`DUMP_R`] along the chain go to blj_misses.jsonl, for
//!   tools/analysis/blj_tune.py.
//! - **in-run use:** observed and catalogue spots on static geometry are
//!   BljAny scan candidates (clause `Observed` unless they are a slope or
//!   ceiling).
//!
//! Determinism: updated only on the planning thread (generation) and at
//! merge (the tree's node hook), both in a fixed order.
//!
//! Output (each report, full snapshots): blj_spots.jsonl (one line a
//! spot), blj_misses.jsonl, blj_zones.json (the static floors, walls and
//! ceilings of every zone with a miss, for the tuner), summary.json
//! `blj_spots`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

use obrrt_core::St;
use serde_json::{Value, json};

use super::objscan::Frame;

pub fn on() -> bool {
    obrrt_core::config::current().objectives.blj_spots
}

pub const MERGE_DIST: f32 = 100.0;
pub const MERGE_YAW: i32 = 0x1000;
pub const DUMP_R: f32 = 300.0;
/// A chain this fast makes an observed spot.
pub const OBSERVED_FV: f32 = 100.0;
/// A chain's first tree node slower than this is its start (a miss can be
/// judged); faster, the tree joined it mid-chain.
pub const START_FV: f32 = 50.0;

#[derive(Debug, Clone)]
pub struct Spot {
    pub zone: (u16, u8),
    pub owner: String,
    pub clause: String,
    /// Start (object frame for object spots) and facing (likewise).
    pub local: [f32; 3],
    pub yaw: i32,
    /// World start when first seen.
    pub world: [f32; 3],
    pub sources: BTreeSet<String>,
    pub first_frame: u32,
    pub first_node: Option<u32>,
    pub seen: u64,
    pub attempted: u64,
    pub succeeded: u64,
    pub observed: u64,
    pub best_fv: f32,
    pub seg_sum: f64,
    pub variants: BTreeSet<u8>,
    pub runs: Vec<String>,
}

type Key = ((u16, u8), String, String, [i32; 3], i32);

struct ChainStart {
    node: u32,
    st: St,
    best: f32,
    registered: Option<usize>,
}

#[derive(Default)]
struct Reg {
    spots: Vec<Spot>,
    index: HashMap<Key, usize>,
    merges: u64,
    misses: Vec<Value>,
    starts: Vec<ChainStart>,
    chains: HashMap<u32, usize>,
    miss_zones: BTreeSet<(u16, u8)>,
    /// Observed chains whose first tree node was already past START_FV.
    mid_chain: u64,
}

fn reg() -> &'static Mutex<Reg> {
    static R: OnceLock<Mutex<Reg>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Reg::default()))
}

fn grid(p: [f32; 3]) -> [i32; 3] {
    p.map(|v| (v / 64.0).floor() as i32)
}

fn octant(yaw: i32) -> i32 {
    ((yaw + 0x1000).rem_euclid(0x10000)) / 0x2000
}

fn yaw_diff(a: i32, b: i32) -> i32 {
    ((a - b) as i16 as i32).abs()
}

fn d3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// An object owner's name: behaviour and home (rounded).
pub fn owner_of(o: &obrrt_core::Obj) -> String {
    format!("{}@{},{},{}", o.bhv_name(), o.home[0].round() as i32, o.home[1].round() as i32, o.home[2].round() as i32)
}

/// Sets up the registry once: the node hook, the presets, the catalogue.
pub fn install() {
    static DONE: OnceLock<()> = OnceLock::new();
    if !on() {
        return;
    }
    DONE.get_or_init(|| {
        obrrt_core::hooks::set_node_hook(node_hook);
        for (name, zone, start, yaw, riser) in super::super::blj::preset_spots() {
            let i = note(zone, "static", if riser { "Riser" } else { "Preset" }, start, yaw.unwrap_or(0), start, "preset", 0, None, false);
            reg().lock().unwrap().spots[i].runs.push(format!("preset:{name}"));
        }
        if let Some(path) = &obrrt_core::config::current().objectives.blj_spots_catalog {
            load_catalog(path);
        }
    });
}

fn load_catalog(path: &str) {
    let Ok(text) = obrrt_core::sys::fs::read_to_string(path) else {
        eprintln!("blj spots: catalog {path} unreadable");
        return;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        eprintln!("blj spots: catalog {path} is not JSON");
        return;
    };
    let mut n = 0;
    for s in v["spots"].as_array().into_iter().flatten() {
        let z = s["zone"].as_array().map(|a| (a[0].as_u64().unwrap_or(0) as u16, a[1].as_u64().unwrap_or(0) as u8));
        let p = |k: &str| s[k].as_array().map(|a| [0, 1, 2].map(|i| a.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32));
        let (Some(zone), Some(local)) = (z, p("local")) else { continue };
        let world = p("world").unwrap_or(local);
        let owner = s["owner"].as_str().unwrap_or("static");
        let clause = s["clause"].as_str().unwrap_or("Observed");
        let i = note(zone, owner, clause, local, s["yaw"].as_i64().unwrap_or(0) as i32, world, "catalog", 0, None, false);
        let runs: Vec<String> = s["runs"].as_array().into_iter().flatten().filter_map(|r| r.as_str().map(str::to_string)).collect();
        reg().lock().unwrap().spots[i].runs.extend(runs);
        n += 1;
    }
    eprintln!("blj spots: catalog {path}: {n} spots");
}

/// Notes a spot from `source`; returns its index. `seen`: counts as seen
/// as a candidate.
#[allow(clippy::too_many_arguments)]
pub fn note(zone: (u16, u8), owner: &str, clause: &str, local: [f32; 3], yaw: i32, world: [f32; 3], source: &str, frame: u32, node: Option<u32>, seen: bool) -> usize {
    let mut r = reg().lock().unwrap();
    let key: Key = (zone, owner.to_string(), clause.to_string(), grid(local), octant(yaw));
    let i = match r.index.get(&key) {
        Some(&i) => i,
        None => {
            let near = r.spots.iter().position(|s| s.zone == zone && s.owner == owner && s.clause == clause && d3(s.local, local) <= MERGE_DIST && yaw_diff(s.yaw, yaw) <= MERGE_YAW);
            let i = match near {
                Some(j) => {
                    r.merges += 1;
                    j
                }
                None => {
                    r.spots.push(Spot {
                        zone,
                        owner: owner.to_string(),
                        clause: clause.to_string(),
                        local,
                        yaw,
                        world,
                        sources: BTreeSet::new(),
                        first_frame: frame,
                        first_node: node,
                        seen: 0,
                        attempted: 0,
                        succeeded: 0,
                        observed: 0,
                        best_fv: 0.0,
                        seg_sum: 0.0,
                        variants: BTreeSet::new(),
                        runs: Vec::new(),
                    });
                    r.spots.len() - 1
                }
            };
            r.index.insert(key, i);
            i
        }
    };
    let s = &mut r.spots[i];
    s.sources.insert(source.to_string());
    s.seen += seen as u64;
    i
}

/// An attempt's outcome at spot `i` (from BljAny::note_result).
pub fn note_result(i: usize, var: u8, success: bool, frames: usize, end_fv: Option<f32>) {
    let mut r = reg().lock().unwrap();
    let Some(s) = r.spots.get_mut(i) else { return };
    s.attempted += 1;
    s.succeeded += success as u64;
    s.seg_sum += frames as f64;
    s.variants.insert(var);
    if success {
        s.best_fv = s.best_fv.max(end_fv.map_or(0.0, |v| -v));
    }
}

fn long_jumping(s: &St) -> bool {
    matches!(s.action, 0x0300_0888 | 0x0000_0479)
}

/// The clause a chain start looks like (for observed spots).
fn classify(st: &St) -> &'static str {
    let pd = super::params::get();
    if st.floor_obj_vy.is_some_and(|v| v > 2.0) {
        "Elevator"
    } else if (pd.ceil_head_lo..pd.ceil_head_hi).contains(&(st.ceil_y - st.floor_y - 160.0)) {
        "Ceiling"
    } else if st.floor_normal[1] > pd.slope_ny_min && st.floor_normal[1] <= pd.slope_ny_max {
        "Slope"
    } else {
        "Observed"
    }
}

/// The tree's node hook: chains (fv <= -16 in a long jump) are followed
/// from their first node; one reaching -[`OBSERVED_FV`] registers it.
fn node_hook(id: u32, parent: u32, st: &St) {
    if !long_jumping(st) || st.forward_vel > -16.0 {
        return;
    }
    let mut r = reg().lock().unwrap();
    let k = match r.chains.get(&parent) {
        Some(&k) => k,
        None => {
            r.starts.push(ChainStart { node: id, st: st.clone(), best: 0.0, registered: None });
            r.starts.len() - 1
        }
    };
    r.chains.insert(id, k);
    let fv = -st.forward_vel;
    r.starts[k].best = r.starts[k].best.max(fv);
    if let Some(i) = r.starts[k].registered {
        r.spots[i].best_fv = r.spots[i].best_fv.max(fv);
        return;
    }
    if fv < OBSERVED_FV {
        return;
    }
    let cs = r.starts[k].st.clone();
    let node = r.starts[k].node;
    drop(r);
    // Owner: the object under the chain start, if any.
    let (owner, local, yaw) = match cs.floor_obj.and_then(|sl| cs.objects().iter().find(|o| o.slot == sl).cloned()) {
        Some(o) => {
            let f = Frame::of(&o);
            (owner_of(&o), f.to_l(cs.pos), cs.yaw as i32 - o.yaw)
        }
        None => ("static".to_string(), cs.pos, cs.yaw as i32),
    };
    let clause = classify(&cs);
    let zone = cs.zone();
    // A miss: nothing the detectors or presets had within the tolerance.
    let known = {
        let r = reg().lock().unwrap();
        r.spots.iter().any(|s| s.zone == zone && s.sources.iter().any(|x| x != "observed") && d3(s.world, cs.pos) <= MERGE_DIST)
    } || super::scan::zone_cands(zone).is_some_and(|v| v.iter().any(|c| d3(c.start, cs.pos) <= MERGE_DIST));
    let i = note(zone, &owner, clause, local, yaw, cs.pos, "observed", cs.frame, Some(node), false);
    // A miss once per spot, and only when the tree saw the chain start
    // itself (its first chain node still slow: the first tree node of many
    // chains is already the hand-off, which says nothing of the spot).
    let fresh = reg().lock().unwrap().spots[i].observed == 0;
    let ctx = (!known && fresh && cs.forward_vel > -START_FV).then(|| miss_record(&cs, node, &owner, clause));
    let mut r = reg().lock().unwrap();
    if cs.forward_vel <= -START_FV {
        r.mid_chain += 1;
    }
    r.starts[k].registered = Some(i);
    let s = &mut r.spots[i];
    s.observed += 1;
    s.best_fv = s.best_fv.max(fv);
    if let Some(m) = ctx {
        r.misses.push(m);
        r.miss_zones.insert(zone);
    }
}

fn tri_json(list: u8, kind: i16, v: [[f32; 3]; 3], n: [f32; 3], owner: &str) -> Value {
    json!({"l": list, "k": kind, "v": v, "n": n, "o": owner})
}

/// A miss's context and triangle dump (module doc).
fn miss_record(cs: &St, node: u32, owner: &str, clause: &str) -> Value {
    let zone = cs.zone();
    let r = (cs.yaw as f32) * std::f32::consts::TAU / 65536.0;
    // The BLJ moves Mario against his facing.
    let along = [cs.pos[0] - DUMP_R * r.sin(), cs.pos[1], cs.pos[2] - DUMP_R * r.cos()];
    // A vertex within DUMP_R of either point, or either point over it.
    let near = |v: &[[f32; 3]; 3]| {
        let t = obrrt_core::geo::Tri { v: *v, n: [0.0, 1.0, 0.0], kind: 0, room: 0 };
        v.iter().any(|p| [cs.pos, along].iter().any(|c| (p[0] - c[0]).hypot(p[2] - c[2]) <= DUMP_R && (p[1] - c[1]).abs() <= 2.0 * DUMP_R))
            || [cs.pos, along].iter().any(|c| super::scan::inside_xz_pub(&t, c[0], c[2]) && v.iter().any(|p| (p[1] - c[1]).abs() <= 2.0 * DUMP_R))
    };
    let mut tris = Vec::new();
    if obrrt_core::mover::published(zone) {
        for (list, ts) in [(0u8, obrrt_core::geo::static_floors(zone)), (1, obrrt_core::geo::static_ceils(zone)), (2, obrrt_core::geo::static_walls(zone))] {
            for t in ts.iter().flat_map(|a| a.iter()) {
                if near(&t.v) {
                    tris.push(tri_json(list, t.kind, t.v, t.n, "static"));
                }
            }
        }
    }
    if let Some(g) = obrrt_core::dyngeo::of(cs) {
        let objs = cs.objects();
        for s in &g.surfs {
            let t = s.tri();
            if near(&t.v) {
                let o = objs.iter().find(|o| o.slot == s.slot).map_or("?".to_string(), owner_of);
                tris.push(tri_json(s.list, s.kind, t.v, t.n, &o));
            }
        }
    }
    json!({
        "zone": [zone.0, zone.1], "node": node, "frame": cs.frame, "owner": owner, "clause": clause,
        "pos": cs.pos, "yaw": cs.yaw, "fv": cs.forward_vel, "action": format!("{:#x}", cs.action),
        "floor_y": cs.floor_y, "floor_normal": cs.floor_normal, "floor_type": cs.floor_type,
        "headroom": cs.ceil_y - cs.floor_y - 160.0, "wall_normal": cs.wall_normal,
        "floor_obj": cs.floor_obj, "tris": tris,
    })
}

/// Static-owner observed and catalogue spots within `r` of `b` (BljAny
/// scan candidates): (index, clause, world start, facing).
pub fn extra_cands(b: &St, r: f32) -> Vec<(usize, String, [f32; 3], i32)> {
    if !on() {
        return Vec::new();
    }
    let reg = reg().lock().unwrap();
    reg.spots
        .iter()
        .enumerate()
        .filter(|(_, s)| s.zone == b.zone() && s.owner == "static" && (s.sources.contains("observed") || s.sources.contains("catalog")) && b.dist(s.world) <= r)
        .map(|(i, s)| (i, s.clause.clone(), s.world, s.yaw))
        .collect()
}

fn spot_json(s: &Spot) -> Value {
    json!({
        "zone": [s.zone.0, s.zone.1], "owner": s.owner, "clause": s.clause, "local": s.local, "yaw": s.yaw, "world": s.world,
        "sources": s.sources, "first_frame": s.first_frame, "first_node": s.first_node, "seen": s.seen, "attempted": s.attempted,
        "succeeded": s.succeeded, "observed": s.observed, "best_fv": s.best_fv,
        "mean_seg_frames": if s.attempted > 0 { Some(s.seg_sum / s.attempted as f64) } else { None }, "variants": s.variants, "runs": s.runs,
    })
}

/// summary.json `blj_spots` (None when off).
pub fn summary_block() -> Option<(String, String)> {
    if !on() {
        return None;
    }
    let r = reg().lock().unwrap();
    let mut by: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    for s in &r.spots {
        for src in &s.sources {
            *by.entry(src.clone()).or_default().entry(s.clause.clone()).or_default() += 1;
        }
    }
    let only_observed = r.spots.iter().filter(|s| s.sources.len() == 1 && s.sources.contains("observed")).count();
    let v = json!({"spots": r.spots.len(), "merges": r.merges, "misses": r.misses.len(), "observed_mid_chain": r.mid_chain, "only_observed": only_observed, "by_source_clause": by, "chains_followed": r.starts.len()});
    Some(("blj_spots".to_string(), v.to_string()))
}

/// Writes blj_spots.jsonl, blj_misses.jsonl and blj_zones.json (module doc).
pub fn write(dir: &str) {
    if !on() {
        return;
    }
    let r = reg().lock().unwrap();
    let mut out = String::new();
    for s in &r.spots {
        out.push_str(&spot_json(s).to_string());
        out.push('\n');
    }
    let _ = obrrt_core::sys::fs::write(format!("{dir}/blj_spots.jsonl"), out);
    let mut m = String::new();
    for x in &r.misses {
        m.push_str(&x.to_string());
        m.push('\n');
    }
    let _ = obrrt_core::sys::fs::write(format!("{dir}/blj_misses.jsonl"), m);
    let mut zones = serde_json::Map::new();
    for z in &r.miss_zones {
        if !obrrt_core::mover::published(*z) {
            continue;
        }
        let dump = |ts: Option<std::sync::Arc<[obrrt_core::geo::Tri]>>| -> Vec<Value> { ts.iter().flat_map(|a| a.iter()).map(|t| json!({"k": t.kind, "v": t.v, "n": t.n})).collect() };
        zones.insert(
            format!("{}:{}", z.0, z.1),
            json!({"floors": dump(obrrt_core::geo::static_floors(*z)), "ceils": dump(obrrt_core::geo::static_ceils(*z)), "walls": dump(obrrt_core::geo::static_walls(*z))}),
        );
    }
    let _ = obrrt_core::sys::fs::write(format!("{dir}/blj_zones.json"), Value::Object(zones).to_string());
}

/// Test / lab access: a miss record for `st` as if its chain started
/// there.
pub fn miss_for(st: &St) -> Value {
    let owner = st.floor_obj.and_then(|sl| st.objects().iter().find(|o| o.slot == sl).map(owner_of)).unwrap_or_else(|| "static".to_string());
    miss_record(st, 0, &owner, classify(st))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_merges_near_duplicates() {
        // A private zone keeps this test apart from others.
        let z = (999, 9);
        let a = note(z, "static", "Slope", [100.0, 0.0, 100.0], 0x100, [100.0, 0.0, 100.0], "static", 1, None, true);
        // Same grid cell and octant: the same spot.
        let b = note(z, "static", "Slope", [110.0, 0.0, 120.0], 0x200, [110.0, 0.0, 120.0], "dynamic", 2, None, true);
        assert_eq!(a, b);
        // Another cell, within 100 and 0x1000: merged.
        let c = note(z, "static", "Slope", [180.0, 0.0, 150.0], 0x800, [180.0, 0.0, 150.0], "observed", 3, None, false);
        assert_eq!(a, c);
        // Another clause, or far: new spots.
        let d = note(z, "static", "Ceiling", [100.0, 0.0, 100.0], 0x100, [100.0, 0.0, 100.0], "static", 4, None, true);
        let e = note(z, "static", "Slope", [400.0, 0.0, 100.0], 0x100, [400.0, 0.0, 100.0], "static", 5, None, true);
        assert!(d != a && e != a && d != e);
        let r = reg().lock().unwrap();
        assert_eq!(r.spots[a].sources.len(), 3);
        assert_eq!(r.spots[a].seen, 2);
    }
}
