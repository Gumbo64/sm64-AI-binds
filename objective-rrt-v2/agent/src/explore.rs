//! Optional wider search, called by the agent when it wants it:
//! `ar fork` (several independent controllers on one objective, kept as side
//! branches) and `ar explore` (a bounded obrrt2 tree from the path's end,
//! summarised, with `explore import` to adopt a node's path as a branch).

use anyhow::{Result, anyhow, bail};
use obrrt_core::{GamePad, Objective};
use obrrt_search::ga::rhstep::{RhConfig, RhController};
use serde_json::{Value, json};
use std::collections::HashMap;

use crate::ckpt;
use crate::objective;
use crate::session::{Commit, Session, summary};
use crate::util::*;

fn flag<'a>(f: &'a HashMap<String, String>, k: &str) -> Option<&'a str> {
    f.get(k).map(|s| s.as_str())
}

/// Saves `pads` (played from the path's end) as a side branch off HEAD:
/// the work in progress, then these. HEAD doesn't move (`ar rewind ID`
/// adopts it).
fn side_branch(s: &Session, prefix: &str, kind: &str, objective: Option<&str>, pads: &[GamePad], st: &obrrt_core::St, note: &str) -> Result<Commit> {
    let head = s.head()?;
    let (w, wp) = s.wip_pads()?;
    let mut all = if w.parent == head { wp } else { Vec::new() };
    let from = if w.parent == head { w.from } else { s.commit_map()?[&head].to };
    all.extend_from_slice(pads);
    let id = format!("{prefix}{:.0}", now() * 1000.0);
    std::fs::write(s.dir.join(format!("inputs/{id}.pads")), pads_to_bytes(&all))?;
    let c = Commit { id, parent: Some(head), kind: kind.into(), objective: objective.map(Into::into), from, to: from + all.len() as u32, prefix_of: None, summary: summary(st), note: note.into(), time: now() };
    append_jsonl(s.dir.join("commits.jsonl"), &serde_json::to_value(&c)?)?;
    Ok(c)
}

pub fn fork(s: &Session, pos: &[String], f: &HashMap<String, String>) -> Result<()> {
    let n: usize = pos.first().ok_or_else(|| anyhow!("fork N SEL"))?.parse()?;
    let sel: Value = serde_json::from_str(pos.get(1).ok_or_else(|| anyhow!("fork N SEL (a selector JSON)"))?)?;
    objective::validate(&sel)?;
    let limit: u32 = flag(f, "frames").map_or(Ok(900), |x| x.parse())?;
    let mut rh = s.cfg.rh.clone();
    if let Some(over) = flag(f, "rh") {
        let mut v = serde_json::to_value(&rh)?;
        for (k, x) in serde_json::from_str::<Value>(over)?.as_object().ok_or_else(|| anyhow!("--rh: an object"))? {
            v[k] = x.clone();
        }
        rh = serde_json::from_value::<RhConfig>(v)?;
        if let Some(why) = crate::session::below_floor(&rh) {
            bail!("--rh: {why}");
        }
    }
    if s.serve_pid().is_some() {
        eprintln!("note: the daemon is running; forks share the CPU with it (`ar pause` first to give them all of it)");
    }
    let gg = sm64_binds::SM64GameGenerator::from_file(&s.cfg.rom)?;
    let mut game = gg.create_game()?;
    let mut ck = ckpt::Store::open(&s.dir)?;
    let st0 = s.load_at(&mut ck, &mut game, &s.head()?, None)?;
    let save = game.save_state();
    let o = objective::resolve(&sel, "fork", &st0, &[])?.ok_or_else(|| anyhow!("{sel} is not offered at the path's end (frame {})", st0.frame))?;
    let threads = (s.cfg.threads / n.max(1)).max(1);
    println!("fork: {n} controllers × {threads} threads on {} from frame {} (limit {limit} frames)", o.key(), st0.frame);
    let results: Vec<Result<(usize, Option<u32>, Vec<GamePad>, obrrt_core::St, f64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n)
            .map(|i| {
                let (gg, save, st0, o, mut rh) = (&gg, &save, st0.clone(), o.clone(), rh.clone());
                sc.spawn(move || -> Result<_> {
                    rh.salt = rh.salt.wrapping_add(0x9e37_79b9 * (i as u64 + 1));
                    let mut ctl = RhController::new(gg, threads, save, st0.clone(), rh)?;
                    let stages: Vec<&dyn Objective> = vec![o.as_ref()];
                    let mut pads = Vec::new();
                    while (pads.len() as u32) < limit {
                        let st = ctl.state();
                        if !st.controllable() || st.modal() {
                            let p = GamePad::new(if pads.len() % 2 == 0 { 0x8000 } else { 0 }, 0, 0);
                            ctl.play_fixed(&[p], true)?;
                            pads.push(p);
                            continue;
                        }
                        let out = ctl.step(&stages, None)?;
                        if out.stuck {
                            break;
                        }
                        pads.extend(out.pads);
                        if ctl.met() > 0 {
                            return Ok((i, Some(pads.len() as u32), pads, ctl.state().clone(), 0.0));
                        }
                    }
                    let eta = ctl.peak();
                    Ok((i, None, pads, ctl.state().clone(), eta))
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut rows = Vec::new();
    for r in results {
        let (i, done, pads, st, eta) = r?;
        let c = side_branch(s, &format!("k{i}-"), "fork", None, &pads, &st, &format!("fork {i} on {}: {}", o.key(), done.map_or(format!("not met in {limit} (best ETA {eta:.0})"), |d| format!("met in {d}"))))?;
        rows.push(json!({"fork": i, "met_frames": done, "best_eta": eta, "commit": c.id, "end": summary(&st)}));
        println!("  fork {i}: {:<22} commit {} (end frame {})", done.map_or(format!("not met (ETA {eta:.0})"), |d| format!("met in {d} frames")), c.id, c.to);
    }
    s.event("fork", json!({"objective": o.key(), "results": rows}))?;
    println!("adopt one with `ar rewind <commit>` (the work in progress is kept as an abandoned commit)");
    Ok(())
}

const OBRRT2_ENV: &[(&str, &str)] = &[
    ("OBRRT_MEM_MB", "2500"),
    ("OBRRT_ROUND", "64"),
    ("OBRRT_TECH", "all"),
    ("OBRRT_NEW_ELIGIBLE", "1"),
    ("OBRRT_MOVE_WEIGHT", "0.75"),
    ("OBRRT_BOOT_NOWAIT", "1"),
];

/// The newest obrrt2: a snapshot `builds/*/obrrt2` (screen.sh's) or cargo's
/// `target/release/obrrt2`.
fn latest_obrrt2() -> Result<String> {
    let mut v: Vec<_> = std::fs::read_dir(format!("{ROOT}/builds"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("obrrt2"))
        .chain([std::path::PathBuf::from(format!("{ROOT}/target/release/obrrt2"))])
        .filter_map(|p| Some((p.metadata().ok()?.modified().ok()?, p)))
        .collect();
    v.sort();
    v.last().map(|x| x.1.display().to_string()).ok_or_else(|| {
        anyhow!("no obrrt2 binary (builds/*/obrrt2 or target/release/obrrt2): build it with `cargo build --release -p obrrt-run --bin obrrt2`, or pass --bin PATH")
    })
}

pub fn explore(s: &mut Session, pos: &[String], f: &HashMap<String, String>) -> Result<()> {
    if pos.first().map(|x| x.as_str()) == Some("import") {
        return import(s, pos.get(1).ok_or_else(|| anyhow!("explore import RUN NODE"))?, pos.get(2).ok_or_else(|| anyhow!("explore import RUN NODE"))?.parse()?);
    }
    let minutes = flag(f, "minutes").unwrap_or("10");
    let frames = flag(f, "frames").unwrap_or("50000000");
    let threads = flag(f, "threads").unwrap_or("4");
    let bin = flag(f, "bin").map(Into::into).map_or_else(latest_obrrt2, Ok)?;
    let head = s.head()?;
    let pads = s.path_pads(&head, true)?;
    let dir = s.dir.join(format!("explore/{:.0}", now()));
    std::fs::create_dir_all(&dir)?;
    let start = dir.join("start.m64");
    write_m64(&start.display().to_string(), &pads)?;
    let out = dir.join("run");
    println!("explore: {bin}, {minutes} min / {frames} frames, {threads} threads, from frame {} → {}", pads.len(), out.display());
    let mut cmd = std::process::Command::new("nice");
    cmd.args(["-n", "10", &bin, &s.cfg.rom, &out.display().to_string(), "1", minutes, &start.display().to_string()]);
    for (k, v) in OBRRT2_ENV {
        if std::env::var(k).is_err() {
            cmd.env(k, v);
        }
    }
    cmd.env("OBRRT_MAX_FRAMES", frames).env("RAYON_NUM_THREADS", threads);
    let log = std::fs::File::create(dir.join("obrrt2.log"))?;
    let st = cmd.stdout(log.try_clone()?).stderr(log).status()?;
    if !st.success() {
        bail!("obrrt2 failed ({st}); see {}/obrrt2.log", dir.display());
    }
    let report = report(&out, pads.len() as u32)?;
    write_json(dir.join("report.json"), &report)?;
    s.event("explore", json!({"run": out.display().to_string(), "report": report}))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!("adopt a node with `ar explore import {} NODE`", out.display());
    Ok(())
}

fn report(run: &std::path::Path, start_frame: u32) -> Result<Value> {
    let text = std::fs::read_to_string(run.join("nodes.csv"))?;
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap_or("").split(',').collect();
    let col = |n: &str| head.iter().position(|h| *h == n);
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
    fn cell<'a>(head: &[&str], r: &[&'a str], n: &str) -> &'a str {
        head.iter().position(|h| *h == n).and_then(|i| r.get(i)).copied().unwrap_or("")
    }
    let get = |r: &Vec<&'_ str>, n: &str| -> String { cell(&head, r, n).to_string() };
    let _ = col;
    let num = |r: &Vec<&str>, n: &str| get(r, n).parse::<f64>().unwrap_or(0.0);
    let s0 = rows.first().map_or(0.0, |r| num(r, "stars"));
    let mut zones: HashMap<String, (usize, f64)> = HashMap::new();
    let mut vias: HashMap<String, usize> = HashMap::new();
    for r in &rows {
        let z = format!("{}:{}", get(r, "level"), get(r, "area"));
        let e = zones.entry(z).or_insert((0, f64::INFINITY));
        e.0 += 1;
        e.1 = e.1.min(num(r, "frame"));
        *vias.entry(get(r, "via")).or_default() += 1;
    }
    let mut star_nodes: Vec<&Vec<&str>> = rows.iter().filter(|r| num(r, "stars") > s0).collect();
    star_nodes.sort_by(|a, b| num(a, "frame").total_cmp(&num(b, "frame")));
    let node = |r: &Vec<&str>| json!({"id": get(r, "id"), "frame": num(r, "frame"), "rel": num(r, "frame") - start_frame as f64, "level": get(r, "level"), "area": get(r, "area"), "pos": [get(r, "x"), get(r, "y"), get(r, "z")], "stars": num(r, "stars"), "coins": num(r, "coins"), "via": get(r, "via"), "star_flags": get(r, "star_flags")});
    let mut zl: Vec<_> = zones.into_iter().collect();
    zl.sort_by(|a, b| a.1.1.total_cmp(&b.1.1));
    let mut vl: Vec<_> = vias.into_iter().collect();
    vl.sort_by(|a, b| b.1.cmp(&a.1));
    let highest = rows.iter().max_by(|a, b| num(a, "y").total_cmp(&num(b, "y")));
    Ok(json!({
        "nodes": rows.len(),
        "zones_reached": zl.iter().map(|(z, (n, f))| json!({"zone": z, "nodes": n, "first_frame": f})).collect::<Vec<_>>(),
        "first_star_nodes": star_nodes.iter().take(8).map(|r| node(r)).collect::<Vec<_>>(),
        "highest": highest.map(node),
        "via_counts": vl.iter().take(15).map(|(v, n)| json!([v, n])).collect::<Vec<_>>(),
    }))
}

fn import(s: &Session, run: &str, node: u32) -> Result<()> {
    let pads = node_inputs(&resolve(run), node)?;
    let head = s.head()?;
    let ours = s.path_pads(&head, true)?;
    if pads.len() < ours.len() || pads[..ours.len()] != ours[..] {
        bail!("node {node}'s inputs don't extend this session's path (they must start from its end, as `ar explore` runs do)");
    }
    let gg = sm64_binds::SM64GameGenerator::from_file(&s.cfg.rom)?;
    let mut game = gg.create_game()?;
    let mut ck = ckpt::Store::open(&s.dir)?;
    s.load_at(&mut ck, &mut game, &head, None)?;
    for p in &pads[ours.len()..] {
        game.step_game(*p)?;
    }
    let st = obrrt_core::read_state(&game, pads.len() as u32, true);
    let c = side_branch(s, "i", "import", None, &pads[ours.len()..], &st, &format!("explore import {run} node {node}"))?;
    println!("imported node {node} as {} (f{}..{}); `ar rewind {}` adopts it", c.id, c.from, c.to, c.id);
    Ok(())
}
