//! BLJ setup timing (OBRRT_BLJ_FAST's measurement): for every successful
//! BLJ attempt of a run that added a node (events.jsonl, kinds
//! Tech::BljAny / Tech::BljRun / Tech::Blj), replays tree.bin from the
//! attempt's start node to the added node and reports, per objective key:
//! the frames from the attempt start to the first chain jump (the first
//! ACT_LONG_JUMP at fv <= -16), the frames spent standing on an object
//! floor before it (riding a platform), and the peak backwards speed.
//!
//! bljsetup_lab <rom> <run_dir> [max=40]
//! bljsetup_lab <rom> ga start=<run_dir>:<node> ... [seeds=2] [only=substr]
//!     one GA attempt per seed (as the tree runs it: 32/48 x gens, scale 3)
//!     of every BljAny / BljRun objective generated at each start node (set
//!     OBRRT_BLJ_SCAN / OBRRT_BLJ_FAST as for a run); the same timing on the
//!     kept segment. Run with and without OBRRT_BLJ_FAST for before/after.

use std::collections::BTreeMap;

use anyhow::Result;
use obrrt_core::{GamePad, read_state};
use sm64_binds::SM64GameGenerator;

const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_FLAG_AIR: u32 = 0x800;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gen_ = SM64GameGenerator::from_file(&args[1])?;
    if args[2] == "ga" {
        return ga_mode(&gen_, &args);
    }
    let dir = &args[2];
    let max: usize = args.iter().find_map(|a| a.strip_prefix("max=")).map_or(40, |v| v.parse().unwrap());
    let records = obrrt_search::route::read_tree_bin(&format!("{dir}/tree.bin"))?;
    let mut evs: Vec<(String, usize, usize)> = Vec::new();
    for l in std::fs::read_to_string(format!("{dir}/events.jsonl"))?.lines() {
        if !l.contains("\"kind\":\"Tech::Blj") {
            continue;
        }
        let e: serde_json::Value = serde_json::from_str(l)?;
        if e["success"] != true {
            continue;
        }
        let (Some(added), Some(start)) = (e["added"].as_u64(), e["start"].as_u64()) else { continue };
        evs.push((e["key"].as_str().unwrap_or("?").to_string(), start as usize, added as usize));
    }
    let mut per: BTreeMap<String, Vec<(usize, usize, usize, f32)>> = BTreeMap::new();
    for (key, start, added) in evs.into_iter().take(max) {
        // Path root -> added (record index = node + 1; record 0 is the root's boot).
        let mut chain = vec![added + 1];
        while let Some(&r) = chain.last() {
            if r == 0 {
                break;
            }
            let p = records[r].0;
            chain.push(if p >= 0 { p as usize + 1 } else { 0 });
        }
        chain.reverse();
        let Some(si) = chain.iter().position(|&r| r == start + 1) else {
            eprintln!("{key}: start {start} not an ancestor of {added}");
            continue;
        };
        let pre: usize = chain[..=si].iter().map(|r| records[*r].1.len()).sum();
        let pads: Vec<GamePad> = chain.iter().flat_map(|r| records[*r].1.clone()).collect();
        let mut g = gen_.create_game()?;
        let (mut first, mut ride, mut peak) = (None, 0usize, 0f32);
        for (i, p) in pads.iter().enumerate() {
            g.step_game(*p)?;
            if i + 1 <= pre {
                continue;
            }
            let s = read_state(&g, (i + 1) as u32, false);
            let t = i + 1 - pre;
            if first.is_none() {
                if matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) && s.forward_vel <= -16.0 {
                    first = Some(t);
                } else if s.floor_obj.is_some() && s.action & ACT_FLAG_AIR == 0 {
                    ride += 1;
                }
            }
            peak = peak.max(-s.forward_vel);
        }
        let seg = pads.len() - pre;
        println!("{key}: start {start} -> {added}: first chain jump at {:?} of {seg} frames, riding {ride}, peak {peak:.1}", first);
        per.entry(key).or_default().push((first.unwrap_or(seg), seg, ride, peak));
    }
    println!("\nper key: successes, mean frames to the first chain jump, mean riding, mean segment");
    for (k, v) in &per {
        let n = v.len() as f64;
        let m = |f: &dyn Fn(&(usize, usize, usize, f32)) -> f64| v.iter().map(f).sum::<f64>() / n;
        println!("  {k}: {} | setup {:.1} | ride {:.1} | seg {:.1}", v.len(), m(&|x| x.0 as f64), m(&|x| x.2 as f64), m(&|x| x.1 as f64));
    }
    Ok(())
}

fn path(records: &[(i32, Vec<GamePad>)], node: usize) -> Vec<GamePad> {
    let mut chain = vec![node + 1];
    while let Some(&r) = chain.last() {
        if r == 0 {
            break;
        }
        let p = records[r].0;
        chain.push(if p >= 0 { p as usize + 1 } else { 0 });
    }
    chain.iter().rev().flat_map(|r| records[*r].1.clone()).collect()
}

/// (frames to the first chain jump, riding frames before it, peak).
fn timing(traj: &[obrrt_core::St]) -> (Option<usize>, usize, f32) {
    let (mut first, mut ride, mut peak) = (None, 0usize, 0f32);
    for (t, s) in traj.iter().enumerate().skip(1) {
        if first.is_none() {
            if matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND) && s.forward_vel <= -16.0 {
                first = Some(t);
            } else if s.floor_obj.is_some() && s.action & ACT_FLAG_AIR == 0 {
                ride += 1;
            }
        }
        peak = peak.max(-s.forward_vel);
    }
    (first, ride, peak)
}

fn ga_mode(gen_: &SM64GameGenerator, args: &[String]) -> Result<()> {
    use obrrt_core::Objective;
    use obrrt_search::ga::{Fitness, GaParams, Worker};
    use rand::SeedableRng;
    obrrt_objectives::moves::install();
    obrrt_core::geo::want_static_floors();
    let seeds: u64 = args.iter().find_map(|a| a.strip_prefix("seeds=")).map_or(2, |v| v.parse().unwrap());
    let only = args.iter().find_map(|a| a.strip_prefix("only="));
    let params = GaParams { scale: 3.0, fitness: Fitness::Steps, new_ops: true, pink: 0.0, per_objective_budget: false };
    let mut w = Worker::new(gen_)?;
    let mut g = gen_.create_game()?;
    // per key: (attempts, successes, sum setup frames, setups seen, sum ride, best peak)
    let mut tally: BTreeMap<String, (u32, u32, f64, u32, f64, f32)> = BTreeMap::new();
    for spec in args.iter().filter_map(|a| a.strip_prefix("start=")) {
        let (dir, node) = spec.rsplit_once(':').unwrap();
        let node: usize = node.parse()?;
        let records = obrrt_search::route::read_tree_bin(&format!("{dir}/tree.bin"))?;
        let pads = path(&records, node);
        let mut g0 = gen_.create_game()?;
        for p in &pads {
            g0.step_game(*p)?;
        }
        let f0 = pads.len() as u32;
        // Read twice with objects: the first stores the zone's geometry
        // (observe_floors), publish makes it visible, the second is b.
        let _ = read_state(&g0, f0, true);
        obrrt_core::mover::publish();
        let st = read_state(&g0, f0, true);
        let save = g0.save_state();
        let mut objs: Vec<std::sync::Arc<dyn Objective>> = Vec::new();
        obrrt_objectives::extra::tech::bljany::BljAny::generate(&st, &mut objs);
        obrrt_objectives::extra::tech::blj::Blj::generate(&st, &mut objs);
        let mut seen = std::collections::BTreeSet::new();
        for o in objs.iter().filter(|o| o.base_criteria(&st) && only.is_none_or(|x| o.key().contains(x))) {
            if !seen.insert(o.key()) {
                continue;
            }
            for seed in 0..seeds {
                let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
                let at = w.attempt(&[o.as_ref()], &st, &save, &mut rng, params)?;
                let Some(seg) = at.segments.first() else {
                    println!("{spec} | {} | seed {seed} | no segment | attempt frames {}", o.key(), at.frames);
                    continue;
                };
                g.load_state(&save)?;
                let mut traj = vec![st.clone()];
                for (k, p) in seg.inputs.iter().enumerate() {
                    g.step_game(*p)?;
                    traj.push(read_state(&g, f0 + k as u32 + 1, false));
                }
                if std::env::var("TRACE").is_ok() {
                    for (k, t) in traj.iter().enumerate() {
                        eprintln!("  {k} act {:#x} fv {:.1} pos ({:.0}, {:.0}, {:.0}) yaw {:#06x} floor_obj {:?} vy {:?} pad {:?}", t.action, t.forward_vel, t.pos[0], t.pos[1], t.pos[2], t.yaw as u16, t.floor_obj, t.floor_obj_vy, seg.inputs.get(k).map(|p| (p.button, p.stick_x, p.stick_y)));
                    }
                }
                let (first, ride, peak) = timing(&traj);
                println!("{spec} | {} | seed {seed} | success {} | first chain jump {:?} of {} | ride {ride} | peak {peak:.1} | attempt frames {}", o.key(), seg.success, first, seg.inputs.len(), at.frames);
                let t = tally.entry(o.key()).or_default();
                t.0 += 1;
                t.1 += seg.success as u32;
                if let Some(f) = first {
                    t.2 += f as f64;
                    t.3 += 1;
                }
                t.4 += ride as f64;
                t.5 = t.5.max(peak);
            }
        }
    }
    println!("\nper key: attempts, successes, mean frames to the first chain jump (of those with one), mean riding, best peak");
    for (k, t) in &tally {
        println!("  {k}: {} | {} | {} | {:.1} | {:.1}", t.0, t.1, if t.3 > 0 { format!("{:.1} ({} with a chain)", t.2 / t.3 as f64, t.3) } else { "-".into() }, t.4 / t.0 as f64, t.5);
    }
    Ok(())
}
