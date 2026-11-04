//! BljAny geometry-scan lab (OBRRT_BLJ_SCAN, objectives/src/extra/tech/bljany/scan.rs):
//! replays to a state, snapshots its zone's static collision as the search
//! does (geo::observe_floors' test), and lists the scan's candidates.
//!
//! bljscan_lab <rom> tree:<run_dir>:<node> [at=x,y,z ...] [r=2500] [n=3]
//! bljscan_lab <rom> movie:<m64>:<frame> [at=x,y,z ...]
//!
//! Prints the zone's candidates per clause, the selection the generator
//! makes at that state (b), and for every `at` point the nearest candidate
//! of each clause (distance, start, facing).

use anyhow::{Result, bail};
use obrrt_core::geo::Tri;
use obrrt_core::{GamePad, St, read_state};
use obrrt_objectives::extra::tech::bljany::scan::{self, Cand};
use sm64_binds::{SM64Game, SM64GameGenerator};

fn movie(path: &str) -> Result<Vec<GamePad>> {
    let bytes = std::fs::read(path)?;
    let body = if bytes.len() >= 0x400 && &bytes[..4] == b"M64\x1a" {
        let n = u32::from_le_bytes(bytes[0x18..0x1C].try_into().unwrap()) as usize;
        let d = &bytes[0x400..];
        d[..(n * 4).min(d.len() / 4 * 4)].to_vec()
    } else {
        bytes[..bytes.len() / 4 * 4].to_vec()
    };
    Ok(body.chunks_exact(4).map(|c| GamePad::new(u16::from_be_bytes([c[0], c[1]]), c[2] as i8, c[3] as i8)).collect())
}

fn rd_u32(m: &[u8], a: usize) -> u32 {
    u32::from_le_bytes(m[a..a + 4].try_into().unwrap())
}

type Geo = (Vec<Tri>, Vec<Tri>, Vec<Tri>);

/// Floors, walls, ceilings if the partition is surely this area's.
fn snapshot(g: &SM64Game) -> Option<Geo> {
    g.with_memory(|m| {
        let mario = obrrt_core::layout::ADDR_gMarioStates as usize;
        let floor = rd_u32(m, mario + 0x68);
        if floor == 0 || rd_u32(m, floor as usize + 0x2C) != 0 {
            return None;
        }
        let (tris, seen) = obrrt_core::geo::read_floors(m);
        seen.contains(&floor).then(|| (tris, obrrt_core::geo::read_walls(m), obrrt_core::geo::read_ceils(m)))
    })
}

fn fmt(c: &Cand) -> String {
    let extra = match (c.riser, c.slot) {
        (Some((r, m)), _) => format!(" {m:?} riser h {:.0} len {:.0} foot {:.0}", r.h(), r.len(), r.ylo),
        (_, Some(s)) => format!(" slot {s}"),
        _ => String::new(),
    };
    format!("{:?} start ({:.0}, {:.0}, {:.0}) yaw {}{extra}", c.kind, c.start[0], c.start[1], c.start[2], c.yaw.map_or("fan".to_string(), |y| format!("{:#06x}", y.rem_euclid(0x10000))))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        bail!("usage: bljscan_lab <rom> tree:<dir>:<node>|movie:<m64>:<frame> [at=x,y,z ...]");
    }
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    let pads: Vec<GamePad> = if let Some(rest) = args[2].strip_prefix("tree:") {
        let (dir, node) = rest.rsplit_once(':').unwrap();
        let node: usize = node.parse()?;
        let records = obrrt_search::route::read_tree_bin(&format!("{dir}/tree.bin"))?;
        let mut chain = vec![node + 1];
        while let Some(&r) = chain.last() {
            if r == 0 {
                break;
            }
            let parent = records[r].0;
            chain.push(if parent >= 0 { parent as usize + 1 } else { 0 });
        }
        chain.iter().rev().flat_map(|r| records[*r].1.clone()).collect()
    } else if let Some(rest) = args[2].strip_prefix("movie:") {
        let (m, f) = rest.rsplit_once(':').unwrap();
        let f: usize = f.parse()?;
        movie(m)?.into_iter().take(f).collect()
    } else {
        bail!("start: tree:<dir>:<node> or movie:<m64>:<frame>");
    };
    let ats: Vec<[f32; 3]> = args
        .iter()
        .filter_map(|a| a.strip_prefix("at="))
        .map(|v| {
            let p: Vec<f32> = v.split(',').map(|x| x.parse().unwrap()).collect();
            [p[0], p[1], p[2]]
        })
        .collect();
    // The object scan's static context (published geometry) and collision.
    obrrt_objectives::moves::install_controller();
    obrrt_core::geo::want_static_floors();
    let mut g = gen_.create_game()?;
    let mut frame = 0u32;
    let mut geo: Option<((u16, u8), Geo)> = None;
    for p in &pads {
        g.step_game(*p)?;
        frame += 1;
        let s = read_state(&g, frame, false);
        // The latest snapshot of the final zone (as the search, which
        // stores the first; static geometry doesn't change in a zone).
        if s.controllable() && geo.as_ref().is_none_or(|(z, _)| *z != s.zone()) {
            if let Some(x) = snapshot(&g) {
                geo = Some((s.zone(), x));
            }
        }
    }
    let _ = read_state(&g, frame, true);
    obrrt_core::mover::publish();
    let b: St = read_state(&g, frame, true);
    println!("b: frame {frame} zone {:?} pos ({:.0}, {:.0}, {:.0}) action {:#x} fv {:.1} floor ny {:.3} type {:#x}", b.zone(), b.pos[0], b.pos[1], b.pos[2], b.action, b.forward_vel, b.floor_normal[1], b.floor_type);
    let Some((z, (fl, wa, ce))) = geo.filter(|(z, _)| *z == b.zone()) else { bail!("no geometry snapshot in b's zone") };
    println!("zone {z:?}: {} floors, {} walls, {} ceilings", fl.len(), wa.len(), ce.len());
    let t0 = std::time::Instant::now();
    let all = scan::find_static(&fl, &wa, &ce);
    println!("candidates in {:.1} ms:", t0.elapsed().as_secs_f64() * 1e3);
    let mut by: std::collections::BTreeMap<String, usize> = Default::default();
    for c in &all {
        *by.entry(format!("{:?}", c.kind)).or_default() += 1;
    }
    let el = scan::elevators(&b, scan::scan_r());
    println!("  {by:?}; elevator objects within R of b: {}", el.len());
    if obrrt_core::dyngeo::on() {
        use obrrt_objectives::extra::tech::bljany::objscan;
        let dg = obrrt_core::dyngeo::of(&b);
        let n_obj = dg.as_ref().map_or(0, |d| d.slots().len());
        let oc = objscan::cands(&b);
        let mov = oc.iter().filter(|c| c.obj.is_some_and(|r| r.moving)).count();
        println!("object collision near b: {n_obj} objects, {} surfaces; {} object candidates ({mov} on moving objects):", dg.as_ref().map_or(0, |d| d.surfs.len()), oc.len());
        let objs = b.objects();
        for slot in dg.as_ref().map_or(vec![], |d| d.slots()) {
            let o = objs.iter().find(|o| o.slot == slot);
            let name = o.map_or("?".to_string(), |o| o.id.to_string());
            let here: Vec<&Cand> = oc.iter().filter(|c| c.obj.is_some_and(|r| r.slot == slot)).collect();
            let ns = dg.as_ref().unwrap().of_slot(slot).count();
            println!("  {name} (slot {slot}, {ns} surfaces, {:.0} away): {} candidates", o.map_or(0.0, |o| b.dist(o.pos)), here.len());
            if std::env::var("DUMP").is_ok() {
                for t in dg.as_ref().unwrap().of_slot(slot) {
                    println!("    surf list {} kind {:#x} n ({:.2}, {:.2}, {:.2}) v {:?}", t.list, t.kind, t.n[0], t.n[1], t.n[2], t.v);
                }
            }
            for c in here.iter().take(6) {
                let r = c.obj.unwrap();
                println!("    {:?} local ({:.0}, {:.0}, {:.0}) yaw {:#06x} world {} | {:.0} from b{}", c.kind, r.local_start[0], r.local_start[1], r.local_start[2], r.local_yaw.rem_euclid(0x10000), fmt(c), b.dist(c.start), if objscan::holds(&b, &r).is_some() { "" } else { " (holds: NO)" });
            }
        }
    }
    // miss=<dir>: append b as a test miss (blj_misses.jsonl, as the registry
    // writes them) and its zone's static geometry (blj_zones.json) there.
    if let Some(dir) = args.iter().find_map(|a| a.strip_prefix("miss=")) {
        use std::io::Write;
        std::fs::create_dir_all(dir)?;
        let m = obrrt_objectives::extra::tech::bljany::spots::miss_for(&b);
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/blj_misses.jsonl"))?;
        writeln!(f, "{m}")?;
        let zp = format!("{dir}/blj_zones.json");
        let mut zones: serde_json::Value = std::fs::read_to_string(&zp).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(serde_json::json!({}));
        let dump = |ts: &[Tri]| -> Vec<serde_json::Value> { ts.iter().map(|t| serde_json::json!({"k": t.kind, "v": t.v, "n": t.n})).collect() };
        zones[format!("{}:{}", z.0, z.1)] = serde_json::json!({"floors": dump(&fl), "ceils": dump(&ce), "walls": dump(&wa)});
        std::fs::write(&zp, zones.to_string())?;
        println!("miss written to {dir} ({} triangles dumped)", m["tris"].as_array().map_or(0, |a| a.len()));
    }
    println!("selection at b (r {}, n {}):", scan::scan_r(), scan::scan_n());
    for c in scan::near(&b, &all) {
        println!("  {:.0} away: {}", b.dist(c.start), fmt(&c));
    }
    for p in ats {
        println!("at ({:.0}, {:.0}, {:.0}):", p[0], p[1], p[2]);
        let mut kinds: Vec<String> = all.iter().chain(el.iter()).map(|c| format!("{:?}", c.kind)).collect();
        kinds.sort();
        kinds.dedup();
        for k in kinds {
            let d = |c: &Cand| ((c.start[0] - p[0]).powi(2) + (c.start[1] - p[1]).powi(2) + (c.start[2] - p[2]).powi(2)).sqrt();
            if let Some(c) = all.iter().chain(el.iter()).filter(|c| format!("{:?}", c.kind) == k).min_by(|a, b| d(a).total_cmp(&d(b))) {
                println!("  nearest {k}: {:.0} away{}: {}", d(c), if d(c) <= 300.0 { " (<= 300)" } else { "" }, fmt(c));
            }
        }
    }
    Ok(())
}
