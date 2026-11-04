//! Replays an obrrt2 run's route and checks spec objectives (and a schema)
//! against it: where each spec's gate first holds, whether and where it
//! reaches Done (vs its `expect` milestone), ETA calibration from gate to
//! Done, ETA jumps, Lost firing on a successful route, and (with --schema)
//! how the schema's keys bin the segment.
//!
//! spec-validate <spec.json|dir> <run_dir> [--node ID] [--schema FILE]
//!     [--from FRAME] [--to FRAME] [--jump FRAMES] [--json OUT] [--rom FILE]
//!
//! The route is power-on -> the node (default: the earliest node with the
//! most stars, as oss2's --pipe), read with obrrt-search's read_tree_bin and
//! walked parent by parent like nodepath. Prints a short report; writes the
//! JSON one to --json (default <run_dir name>-validate.json in the cwd).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use anyhow::{Context, Result, bail};
use obrrt_core::{Eta, Mem, St, read_state};
use serde_json::{Value, json};
use sm64_binds::SM64GameGenerator;

struct Args {
    spec: String,
    run: String,
    node: Option<u32>,
    schema: Option<String>,
    from: u32,
    to: u32,
    jump: f64,
    json: Option<String>,
    rom: String,
    /// Print every n-th recorded (frame, phase, ETA, frames left, pos).
    trace: usize,
}

fn parse_args() -> Result<Args> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 {
        bail!("usage: spec-validate <spec.json|dir> <run_dir> [--node ID] [--schema FILE] [--from F] [--to F] [--jump X] [--json OUT] [--rom FILE] [--trace N]");
    }
    let mut out = Args {
        spec: a[1].clone(),
        run: a[2].clone(),
        node: None,
        schema: None,
        from: 0,
        to: u32::MAX,
        jump: 30.0,
        json: None,
        trace: 0,
        rom: format!("{}/../../baserom.us.z64", env!("CARGO_MANIFEST_DIR")),
    };
    let mut i = 3;
    while i < a.len() {
        let v = a.get(i + 1).cloned().unwrap_or_default();
        match a[i].as_str() {
            "--node" => out.node = Some(v.parse()?),
            "--schema" => out.schema = Some(v),
            "--from" => out.from = v.parse()?,
            "--to" => out.to = v.parse()?,
            "--jump" => out.jump = v.parse()?,
            "--json" => out.json = Some(v),
            "--rom" => out.rom = v,
            "--trace" => out.trace = v.parse()?,
            s => bail!("unknown argument {s}"),
        }
        i += 2;
    }
    Ok(out)
}

/// One nodes.csv row, as far as the report needs.
struct Row {
    frame: u32,
    stars: i16,
    via: String,
}

fn read_nodes(run: &str) -> Result<HashMap<u32, Row>> {
    let csv = std::fs::read_to_string(format!("{run}/nodes.csv")).with_context(|| format!("{run}/nodes.csv"))?;
    let mut out = HashMap::new();
    for l in csv.lines().skip(1) {
        let c: Vec<&str> = l.split(',').collect();
        out.insert(c[0].parse()?, Row { frame: c[11].parse()?, stars: c[8].parse()?, via: c[12].to_string() });
    }
    Ok(out)
}

/// Per spec (variant): what happened along the route.
#[derive(Default)]
struct Track {
    start: Option<St>,
    mem: Mem,
    gate: Option<(u32, [f32; 3], (u16, u8))>,
    done: Option<u32>,
    expect: Option<u32>,
    /// (frame, eta frames, phase) from the gate on, to Done.
    etas: Vec<(u32, f64, usize)>,
    /// Mario's position per recorded ETA (for --trace).
    pos: Vec<[f32; 3]>,
    lost: Vec<(u32, usize, &'static str)>,
    /// Frame each phase was entered.
    phase_in: Vec<Option<u32>>,
    last_phase: usize,
    episodes: u32,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((v.len() - 1) as f64 * q).round() as usize;
    v[i]
}

fn key_hash(k: &objective_scattershot::objective::Key) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    k.hash(&mut h);
    h.finish()
}

fn r1(x: f64) -> Value {
    if x.is_finite() { json!((x * 10.0).round() / 10.0) } else { Value::Null }
}

fn r3(x: f64) -> Value {
    if x.is_finite() { json!((x * 1000.0).round() / 1000.0) } else { Value::Null }
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let specs = obrrt_spec::load(&args.spec, Some(&args.run))?;
    let records = obrrt_search::route::read_tree_bin(&format!("{}/tree.bin", args.run))?;
    let rows = read_nodes(&args.run)?;
    // Default node: the earliest of those with the most stars.
    let node = match args.node {
        Some(n) => n,
        None => {
            let most = rows.values().map(|r| r.stars).max().unwrap_or(0);
            *rows.iter().filter(|(_, r)| r.stars == most).min_by_key(|(id, r)| (r.frame, **id)).unwrap().0
        }
    };
    // nodepath's walk: node -> root, records[n + 1] is node n, record 0 the boot.
    let mut chain = vec![node];
    while let Some(&n) = chain.last() {
        let p = records[n as usize + 1].0;
        if p < 0 {
            break;
        }
        chain.push(p as u32);
    }
    chain.reverse();
    let schema = match &args.schema {
        Some(f) => Some(objective_scattershot::objective::Tree::load(&std::fs::read_to_string(f)?)?),
        None => None,
    };

    let game_gen = SM64GameGenerator::from_file(&args.rom)?;
    let mut game = game_gen.create_game()?;
    let mut frame = 0u32;
    for p in &records[0].1 {
        game.step_game(*p)?;
        frame += 1;
    }
    let mut tracks: Vec<Track> = specs.iter().map(|s| Track { phase_in: vec![None; s.phases.len() + 1], ..Default::default() }).collect();
    // Per frame: the node it belongs to, a schema key hash.
    let mut frame_node: Vec<(u32, u32)> = Vec::new(); // (first frame, node)
    let mut keys: HashMap<u32, u64> = HashMap::new();
    // Route milestones: star counts and zone entries.
    let mut milestones: Vec<(u32, String)> = Vec::new();
    let mut prev: Option<St>;
    let step = |s: &St, tracks: &mut Vec<Track>, keys: &mut HashMap<u32, u64>| {
        if let Some(t) = &schema {
            keys.insert(s.frame, key_hash(&t.key(s)));
        }
        for (spec, t) in specs.iter().zip(tracks.iter_mut()) {
            if t.done.is_some() {
                continue;
            }
            if t.start.is_none() {
                if s.controllable() && spec.gate_holds(s) {
                    t.start = Some(s.clone());
                    t.mem = Mem::default();
                    t.episodes += 1;
                    if t.gate.is_none() {
                        t.gate = Some((s.frame, s.pos, s.zone()));
                    }
                } else {
                    continue;
                }
            }
            let start = t.start.clone().unwrap();
            if t.expect.is_none() {
                if let Some(e) = &spec.expect {
                    if obrrt_spec::holds(e, &obrrt_spec_cx(s, &start)) {
                        t.expect = Some(s.frame);
                    }
                }
            }
            let (eta, phase) = spec.eta_phase(s, &start, &mut t.mem);
            if phase != t.last_phase || t.phase_in[0].is_none() {
                if t.phase_in[phase.min(spec.phases.len())].is_none() {
                    t.phase_in[phase.min(spec.phases.len())] = Some(s.frame);
                }
                t.last_phase = phase;
            }
            match eta {
                Eta::Done => {
                    t.done = Some(s.frame);
                    t.etas.push((s.frame, 0.0, phase));
                    t.pos.push(s.pos);
                }
                Eta::Lost => {
                    let why = if !s.alive() { "dead" } else if !obrrt_spec::model::zone_has(&spec.zones, s.zone()) { "zone left" } else { "lost predicate / object gone" };
                    t.lost.push((s.frame, phase, why));
                    // Re-arm: the next gate-holding state starts again.
                    t.start = None;
                    t.etas.clear();
                    t.pos.clear();
                    t.phase_in = vec![None; spec.phases.len() + 1];
                    t.last_phase = 0;
                }
                Eta::Frames(f) => {
                    t.etas.push((s.frame, f, phase));
                    t.pos.push(s.pos);
                }
            }
        }
    };
    let root = read_state(&game, frame, true);
    if frame >= args.from && frame <= args.to {
        step(&root, &mut tracks, &mut keys);
    }
    prev = Some(root);
    for &n in &chain[1..] {
        frame_node.push((frame + 1, n));
        for p in &records[n as usize + 1].1 {
            game.step_game(*p)?;
            frame += 1;
            if frame < args.from || frame > args.to {
                if frame > args.to {
                    break;
                }
                continue;
            }
            let s = read_state(&game, frame, true);
            if let Some(pv) = &prev {
                if s.stars > pv.stars {
                    milestones.push((frame, format!("stars {}", s.stars)));
                }
                if s.zone() != pv.zone() {
                    milestones.push((frame, format!("zone {}:{}", s.level, s.area)));
                }
            }
            step(&s, &mut tracks, &mut keys);
            prev = Some(s);
        }
        if frame > args.to {
            break;
        }
    }
    let end_frame = frame;
    let node_at = |f: u32| frame_node.iter().rev().find(|(ff, _)| *ff <= f).map_or(0, |x| x.1);
    let via_at = |f: u32| rows.get(&node_at(f)).map_or("?".to_string(), |r| r.via.clone());

    // ---- report ----
    let mut out_specs = Vec::new();
    println!("route: {} -> node {node} ({} nodes, frames {}..{end_frame})", args.run, chain.len(), records[0].1.len());
    println!("milestones: {}", milestones.iter().map(|(f, m)| format!("{m}@{f}")).collect::<Vec<_>>().join(", "));
    for (spec, t) in specs.iter().zip(&tracks) {
        println!("\n== {}  ({} phases, kind {})", spec.label(), spec.phases.len(), obrrt_core::Objective::kind(spec));
        let mut problems: Vec<String> = Vec::new();
        let Some((gf, gpos, gz)) = t.gate else {
            println!("  gate never holds on the route");
            problems.push("gate never holds".into());
            out_specs.push(json!({"name": spec.name, "variant": spec.variant, "gate": null, "problems": problems}));
            continue;
        };
        println!("  gate: frame {gf} node {} ({}) zone {}:{} pos {:.0},{:.0},{:.0}", node_at(gf), via_at(gf), gz.0, gz.1, gpos[0], gpos[1], gpos[2]);
        let start_frame = t.etas.first().map_or(gf, |e| e.0);
        if t.episodes > 1 {
            println!("  re-armed {} times after Lost; the last episode starts at frame {start_frame}", t.episodes - 1);
        }
        match t.done {
            Some(d) => println!("  done: frame {d} node {} ({}), {} frames after its start", node_at(d), via_at(d), d - start_frame),
            None => {
                println!("  done: never (last phase {} '{}')", t.last_phase, spec.phases.get(t.last_phase).map_or("-", |p| p.name.as_str()));
                problems.push(format!("never Done (stuck in phase {} '{}')", t.last_phase, spec.phases.get(t.last_phase).map_or("-", |p| p.name.as_str())));
            }
        }
        let expect_delta = match (t.done, t.expect) {
            (Some(d), Some(e)) => Some(d as i64 - e as i64),
            _ => None,
        };
        if spec.expect.is_some() {
            match t.expect {
                Some(e) => println!(
                    "  expect: frame {e} node {} ({}); done - expect = {}",
                    node_at(e),
                    via_at(e),
                    expect_delta.map_or("n/a".into(), |d| format!("{d:+}"))
                ),
                None => println!("  expect: never holds on the route"),
            }
            if let Some(d) = expect_delta {
                if d.abs() > 30 {
                    problems.push(format!("Done {d:+} frames from the expected milestone"));
                }
            }
        }
        let mut phases_out = Vec::new();
        let mut line = String::from("  phases:");
        for (i, p) in spec.phases.iter().enumerate() {
            let f = t.phase_in[i];
            // ETA just before and at the transition into phase i+1.
            let next = t.phase_in.get(i + 1).copied().flatten();
            let (before, after) = match next {
                Some(nf) => {
                    let b = t.etas.iter().rev().find(|e| e.0 < nf).map(|e| e.1);
                    let a = t.etas.iter().find(|e| e.0 == nf).map(|e| e.1);
                    (b, a)
                }
                None => (None, None),
            };
            line += &format!(" [{i} {} in@{}{}]", p.name, f.map_or("-".into(), |f| f.to_string()), match (before, after) {
                (Some(b), Some(a)) => format!(" eta {b:.0}->{a:.0}"),
                _ => String::new(),
            });
            if let (Some(b), Some(a)) = (before, after) {
                if a > b {
                    problems.push(format!("ETA rises at the transition {i}->{} ({b:.0} -> {a:.0})", i + 1));
                }
            }
            phases_out.push(json!({"name": p.name, "entered": f, "eta_before_exit": before.map(r1), "eta_after_exit": after.map(r1)}));
        }
        println!("{line}");
        // Calibration: actual frames left / ETA, from gate to Done (or to
        // the expected milestone when Done never came).
        let end = t.done.or(t.expect);
        let mut ratios = Vec::new();
        let mut start_ratio = f64::NAN;
        let mut start_eta = f64::NAN;
        if let Some(end) = end {
            for (k, &(f, e, _)) in t.etas.iter().enumerate() {
                if f >= end {
                    break;
                }
                let actual = (end - f) as f64;
                if k == 0 {
                    start_eta = e;
                    start_ratio = actual / e.max(1.0);
                }
                if k % 5 == 0 && e >= 1.0 {
                    ratios.push(actual / e);
                }
            }
        }
        let n = ratios.len();
        let (p10, med, p90) = (pct(&mut ratios, 0.1), pct(&mut ratios, 0.5), pct(&mut ratios, 0.9));
        if n > 0 {
            println!(
                "  calibration (actual/ETA, {n} points{}): median {med:.2}, p10 {p10:.2}, p90 {p90:.2}; start ETA {start_eta:.0} vs {:.0} actual ({start_ratio:.2})",
                if t.done.is_none() { ", to expect" } else { "" },
                start_ratio * start_eta.max(1.0)
            );
            if !(0.5..=2.0).contains(&med) {
                problems.push(format!("ETA off by {med:.2}x (median actual/ETA)"));
            }
        }
        if args.trace > 0 {
            let end = t.done.or(t.expect);
            for (k, (e, p)) in t.etas.iter().zip(&t.pos).enumerate() {
                if k % args.trace == 0 || k + 1 == t.etas.len() {
                    let left = end.map_or("-".into(), |d| (d as i64 - e.0 as i64).to_string());
                    println!("    trace @{} ph{} eta {:.0} left {left} pos {:.0},{:.0},{:.0} node {}", e.0, e.2, e.1, p[0], p[1], p[2], node_at(e.0));
                }
            }
        }
        // ETA jumps: a rise of more than --jump frames from one frame to the next.
        let mut jumps = Vec::new();
        for w in t.etas.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b.1 - a.1 > args.jump && b.0 == a.0 + 1 {
                jumps.push(json!({"frame": b.0, "from": r1(a.1), "to": r1(b.1), "phase": b.2, "node": node_at(b.0)}));
            }
        }
        if !jumps.is_empty() {
            println!(
                "  ETA jumps > {}: {} (first: {})",
                args.jump,
                jumps.len(),
                jumps.iter().take(4).map(|j| format!("@{} {}->{} ph{}", j["frame"], j["from"], j["to"], j["phase"])).collect::<Vec<_>>().join(", ")
            );
            problems.push(format!("{} ETA jumps > {} frames", jumps.len(), args.jump));
        }
        if !t.lost.is_empty() {
            println!(
                "  LOST on the route (a bug if the route did this): {}",
                t.lost.iter().take(5).map(|(f, p, w)| format!("@{f} ph{p} {w}")).collect::<Vec<_>>().join(", ")
            );
            problems.push(format!("Lost {} times on the route", t.lost.len()));
        }
        // Schema: keys along the segment, and whether phase transitions separate.
        let mut schema_out = Value::Null;
        if schema.is_some() {
            if let Some(end) = end.or(Some(end_frame)) {
                let seg: Vec<(u32, u64)> = (start_frame..=end).filter_map(|f| keys.get(&f).map(|k| (f, *k))).collect();
                let distinct: HashSet<u64> = seg.iter().map(|x| x.1).collect();
                let len = seg.len().max(1) as f64;
                let mut trans = Vec::new();
                // Between phases (entering Done is not a transition to bin).
                for i in 1..spec.phases.len() {
                    let Some(tf) = t.phase_in[i] else { continue };
                    let (Some(kb), Some(ka)) = (keys.get(&(tf - 1)), keys.get(&tf)) else { continue };
                    let lo = t.phase_in[i - 1].unwrap_or(start_frame);
                    let hi = t.phase_in.get(i + 1).copied().flatten().unwrap_or(end);
                    let before: HashSet<u64> = (lo..tf).filter_map(|f| keys.get(&f).copied()).collect();
                    let after: HashSet<u64> = (tf..hi.max(tf + 1)).filter_map(|f| keys.get(&f).copied()).collect();
                    let shared = before.intersection(&after).count();
                    trans.push(json!({"into_phase": i, "frame": tf, "same_key_across": kb == ka, "keys_before": before.len(), "keys_after": after.len(), "shared": shared}));
                    if kb == ka || shared > 0 {
                        problems.push(format!("schema: phase {}->{i} not separated ({} shared keys{})", i - 1, shared, if kb == ka { ", same key across the transition" } else { "" }));
                    }
                }
                let per100 = distinct.len() as f64 * 100.0 / len;
                println!(
                    "  schema: {} distinct keys over {} frames ({per100:.1} per 100 frames); transitions: {}",
                    distinct.len(),
                    seg.len(),
                    trans.iter().map(|t| format!("->{} {}{}", t["into_phase"], if t["same_key_across"] == true { "SAME" } else { "split" }, if t["shared"].as_u64().unwrap_or(0) > 0 { format!(" ({} shared)", t["shared"]) } else { String::new() })).collect::<Vec<_>>().join(", ")
                );
                schema_out = json!({"frames": seg.len(), "distinct_keys": distinct.len(), "keys_per_100_frames": r1(per100), "transitions": trans});
            }
        }
        if problems.is_empty() {
            println!("  OK");
        } else {
            println!("  problems: {}", problems.join("; "));
        }
        out_specs.push(json!({
            "name": spec.name, "variant": spec.variant, "kind": obrrt_core::Objective::kind(spec),
            "gate": {"frame": gf, "node": node_at(gf), "via": via_at(gf), "zone": [gz.0, gz.1], "pos": [r1(gpos[0] as f64), r1(gpos[1] as f64), r1(gpos[2] as f64)]},
            "start_frame": start_frame,
            "episodes": t.episodes,
            "done": t.done.map(|d| json!({"frame": d, "node": node_at(d), "via": via_at(d), "frames_from_start": d - start_frame})),
            "expect": t.expect.map(|e| json!({"frame": e, "node": node_at(e), "via": via_at(e)})),
            "done_minus_expect": expect_delta,
            "phases": phases_out,
            "calibration": {"points": n, "median": r3(med), "p10": r3(p10), "p90": r3(p90), "start_eta": r1(start_eta), "start_ratio": r3(start_ratio), "to": if t.done.is_some() { "done" } else if t.expect.is_some() { "expect" } else { "none" }},
            "jumps": jumps,
            "lost": t.lost.iter().map(|(f, p, w)| json!({"frame": f, "phase": p, "why": w, "node": node_at(*f)})).collect::<Vec<_>>(),
            "schema": schema_out,
            "problems": problems,
            "ok": problems.is_empty(),
        }));
    }
    let report = json!({
        "run": args.run, "node": node, "spec": args.spec, "schema": args.schema,
        "frames": [records[0].1.len(), end_frame],
        "milestones": milestones.iter().map(|(f, m)| json!({"frame": f, "what": m, "node": node_at(*f)})).collect::<Vec<_>>(),
        "specs": out_specs,
    });
    let path = args.json.clone().unwrap_or_else(|| {
        let stem = std::path::Path::new(&args.run).file_name().map_or("run".into(), |s| s.to_string_lossy().into_owned());
        format!("{stem}-validate.json")
    });
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    println!("\njson: {path}");
    Ok(())
}

/// A predicate context for `expect` (no phase target).
fn obrrt_spec_cx<'a>(s: &'a St, start: &'a St) -> obrrt_spec::PredCx<'a> {
    obrrt_spec::PredCx { s, start, target: None, phase_start: start.frame }
}
