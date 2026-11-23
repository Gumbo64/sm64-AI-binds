//! policy <rom> <problem> ga [--pink 0|1] [--scale x] [--seeds n] [--threads n] [--out file.json]
//! policy <rom> <problem> gen  [--dir D] [--rounds n] [--per n] [--minutes m] [--seed s] [--threads n]
//! policy <rom> <problem> eval <id|file.pol> [--dir D]            one rollout per start (train/val/test)
//! policy <rom> <problem> inv list [--status active|retired] | report | eval <file.pol> [--admit]
//!                            | retire <id> --reason "..." | revive <id>
//! Run from objective-rrt-v2. Inventory dir default policy/runs/inventory.

use anyhow::{Result, bail};
use obrrt_gp::problems::load;
use obrrt_gp::rollout::Case;
use obrrt_policy::dsl;
use obrrt_policy::eval::{Env, Instance, Played};
use obrrt_policy::inv::{Eval, Inventory, niche_of};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SM64GameGenerator};

fn games(g: &SM64GameGenerator, n: usize) -> Result<Vec<SM64Game>> {
    (0..n).map(|_| g.create_game()).collect()
}

fn split(c: &Case) -> &'static str {
    obrrt_policy::bench::split(c)
}

/// Per-start table + summary for one body.
fn report(ev: &Eval, label: &str, ps: &[(usize, Played)], verbose: bool) {
    for sp in ["tr", "VA", "TE"] {
        let rows: Vec<&(usize, Played)> = ps.iter().filter(|(i, _)| split(&ev.pb.cases[*i]) == sp).collect();
        if rows.is_empty() {
            continue;
        }
        let succ = rows.iter().filter(|(_, p)| p.out.done.is_some()).count();
        let fr = rows.iter().filter_map(|(_, p)| p.out.done).sum::<usize>() as f64 / succ.max(1) as f64;
        let fit = rows.iter().map(|(_, p)| p.out.fitness).sum::<f64>() / rows.len() as f64;
        let emu = rows.iter().map(|(_, p)| p.out.frames).sum::<u64>() as f64 / rows.len() as f64;
        let mpc = rows.iter().map(|(_, p)| p.desc.mpc_frames).sum::<u64>() as f64 / rows.len() as f64;
        println!("{label}: {sp} success {succ}/{} mean frames-to-Done {fr:.1} mean fitness {fit:.1} emu/start {emu:.0} +mpc {mpc:.0}", rows.len());
    }
    if verbose {
        for (i, p) in ps {
            let c = &ev.pb.cases[*i];
            let mut top: Vec<_> = p.desc.fired.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1));
            println!(
                "  {:<3} {:<40} done {:>5} {:?} fit {:>7.0} end act {} pos ({:.0},{:.0},{:.0}) fired {}",
                split(c),
                c.name,
                p.out.done.map_or("-".to_string(), |d| d.to_string()),
                p.out.end,
                p.out.fitness,
                dsl::action_name(p.desc.end_action),
                p.desc.end_pos[0],
                p.desc.end_pos[1],
                p.desc.end_pos[2],
                top.iter().take(4).map(|(k, v)| format!("{k}x{v}")).collect::<Vec<_>>().join(",")
            );
        }
    }
}

fn eval_all(ev: &Eval, gs: &mut [SM64Game], inv: &Inventory, body: &dsl::C, inst: &Instance) -> Result<Vec<(usize, Played)>> {
    let idx: Vec<usize> = (0..ev.pb.cases.len()).collect();
    let ps = ev.run(gs, &inv.bodies, body, &idx, inst, false)?;
    Ok(idx.into_iter().zip(ps).collect())
}

fn opt(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("OBRRT_LAKITU_ARC", "1");
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("policy <rom> <problem> <command> ...");
    }
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let threads: usize = opt(&args, "--threads").map_or(Ok(4), |s| s.parse())?;
    match args[3].as_str() {
        #[cfg(feature = "ga")]
        "ga" => {
            let pb = load(&args[2], ".", &gg, false)?;
            let pink: f64 = opt(&args, "--pink").map_or(Ok(1.0), |s| s.parse())?;
            let scale: f64 = opt(&args, "--scale").map_or(Ok(3.0), |s| s.parse())?;
            let seeds: u64 = opt(&args, "--seeds").map_or(Ok(1), |s| s.parse())?;
            // OBRRT_POLICY_SEED=1: the inventory seeds the Worker GA (policy/src/seed.rs); unset: the plain baseline
            if obrrt_search::ga::policy_seed_on() {
                let mut cfg = obrrt_policy::seed::SeedCfg::from_env();
                let c0 = &pb.cases[0];
                cfg.fallback_pts = obrrt_gp::rollout::Task::points(pb.tasks[c0.task].as_ref(), &c0.start, &obrrt_core::Mem::default()).to_vec();
                println!("policy seeds: {} active policies, draws {}, keep {}", obrrt_policy::seed::install(cfg.clone()), cfg.draws, cfg.keep);
            }
            // OBRRT_POLICY_REPOLICY=p: the inventory as a breeding operator (policy/src/repolicy.rs)
            if obrrt_search::ga::policy_repolicy() > 0.0 {
                let mut cfg = obrrt_policy::seed::SeedCfg::from_env();
                let c0 = &pb.cases[0];
                cfg.fallback_pts = obrrt_gp::rollout::Task::points(pb.tasks[c0.task].as_ref(), &c0.start, &obrrt_core::Mem::default()).to_vec();
                println!("repolicy operator: p {} over {} policies", obrrt_search::ga::policy_repolicy(), obrrt_policy::repolicy::install(cfg));
            }
            let t0 = std::time::Instant::now();
            let runs = obrrt_policy::bench::ga::baseline(&gg, &pb, threads, seeds, scale, pink)?;
            let mut js = vec![];
            for (s, r) in runs.iter().enumerate() {
                for (c, x) in pb.cases.iter().zip(r) {
                    println!("seed{} {:<3} {:<48} {:>6} {:>10}", s + 1, obrrt_policy::bench::split(c), c.name, x.done.map_or("-".to_string(), |d| d.to_string()), x.emu);
                }
                js.push(obrrt_policy::bench::summarize(&pb, &format!("GA pink={pink} scale={scale} seed{}", s + 1), r));
            }
            if obrrt_search::ga::policy_repolicy() > 0.0 {
                let st = &obrrt_search::ga::REPOLICY_STATS;
                let l = |i: usize| st[i].load(std::sync::atomic::Ordering::Relaxed);
                println!("repolicy children {} attempts won by one {} of {} hook frames {}", l(0), l(1), l(3), l(2));
            }
            println!("wall {:.0}s", t0.elapsed().as_secs_f64());
            if let Some(o) = opt(&args, "--out") {
                std::fs::write(o, serde_json::to_string_pretty(&serde_json::json!({"pink": pink, "scale": scale, "seeds": js}))?)?;
            }
        }
        "gen" => {
            let pb = load(&args[2], ".", &gg, false)?;
            let env = Env::full();
            let ev = Eval::new(&pb, &env);
            let dir = opt(&args, "--dir").unwrap_or("policy/runs/inventory".into());
            let rounds: u64 = opt(&args, "--rounds").map_or(Ok(1_000_000), |s| s.parse())?;
            let per: usize = opt(&args, "--per").map_or(Ok(40), |s| s.parse())?;
            let minutes: f64 = opt(&args, "--minutes").map_or(Ok(1e9), |s| s.parse())?;
            let seed: u64 = opt(&args, "--seed").map_or(Ok(1), |s| s.parse())?;
            let mut gs = games(&gg, threads)?;
            let mut inv = Inventory::open(std::path::Path::new(&dir))?;
            let mut r = SmallRng::seed_from_u64(seed * 7919 + inv.round);
            if inv.entries.is_empty() {
                inv.seed_hand(&ev, &mut gs)?;
            }
            let t0 = std::time::Instant::now();
            for _ in 0..rounds {
                if t0.elapsed().as_secs_f64() > minutes * 60.0 {
                    break;
                }
                let n = inv.inbox(&ev, &mut gs)?;
                let (tried, kept) = inv.round(&ev, &mut gs, &mut r, per)?;
                inv.save()?;
                let act = inv.entries.iter().filter(|e| e.status == "active").count();
                let niches: std::collections::BTreeSet<_> = inv.entries.iter().filter(|e| e.status == "active").map(|e| e.niche.clone()).collect();
                let best = inv.entries.iter().max_by(|a, b| a.fit.partial_cmp(&b.fit).unwrap());
                println!(
                    "round {} tried {tried} kept {kept} inbox {n} | entries {} active {act} niches {} | best {} fit {:.1} succ {} | emu {:.1}M | {}",
                    inv.round,
                    inv.entries.len(),
                    niches.len(),
                    best.map_or("-", |b| b.id.as_str()),
                    best.map_or(0.0, |b| b.fit),
                    best.map_or(0, |b| b.succ),
                    ev.emu.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
                    inv.share_line()
                );
            }
        }
        "eval" => {
            let pb = load(&args[2], ".", &gg, false)?;
            let env = Env::full();
            let ev = Eval::new(&pb, &env);
            let dir = opt(&args, "--dir").unwrap_or("policy/runs/inventory".into());
            let inv = Inventory::open(std::path::Path::new(&dir))?;
            let mut gs = games(&gg, threads)?;
            let what = args.get(4).cloned().unwrap_or_default();
            let pol = if what.ends_with(".pol") { dsl::parse(&std::fs::read_to_string(&what)?)? } else { inv.bodies.get(&what).cloned().ok_or_else(|| anyhow::anyhow!("no policy {what}"))? };
            let ps = eval_all(&ev, &mut gs, &inv, &pol.body, &obrrt_policy::inv::instance_of(&pol))?;
            report(&ev, &pol.name, &ps, args.iter().any(|a| a == "-v"));
        }
        "play" => {
            // policy <rom> <problem> play <id|file.pol> --dump DIR [--dir D]: plays the policy once per start and writes
            // the full inputs from power-on (raw dumps) plus an index file for gp/verify.py (tasplay replay).
            let pb = load(&args[2], ".", &gg, false)?;
            let env = Env::full();
            let ev = Eval::new(&pb, &env);
            let dir = opt(&args, "--dir").unwrap_or("policy/runs/inventory".into());
            let inv = Inventory::open(std::path::Path::new(&dir))?;
            let what = args.get(4).cloned().unwrap_or_default();
            let pol = if what.ends_with(".pol") { dsl::parse(&std::fs::read_to_string(&what)?)? } else { inv.bodies.get(&what).cloned().ok_or_else(|| anyhow::anyhow!("no policy {what}"))? };
            let out = opt(&args, "--dump").unwrap_or("policy/runs/integ/plays".into());
            std::fs::create_dir_all(&out)?;
            let idx: Vec<usize> = (0..pb.cases.len()).collect();
            let mut gs = games(&gg, threads)?;
            let ps = ev.run(&mut gs, &inv.bodies, &pol.body, &idx, &obrrt_policy::inv::instance_of(&pol), true)?;
            let mut index = String::new();
            for (i, p) in idx.iter().zip(&ps) {
                let c = &pb.cases[*i];
                let mut pads = pb.movies[c.movie].1[..c.frame - c.noise.len()].to_vec();
                pads.extend_from_slice(&c.noise);
                pads.extend_from_slice(&p.out.pads);
                let file = format!("{out}/{}-{}.raw", pol.name, c.name.replace(['/', ','], "_"));
                obrrt_gp::problems::write_raw(&file, &pads)?;
                index.push_str(&format!("{file} {} {} {}\n", c.frame, p.out.done.map_or(-1, |d| d as i64), p.out.pads.len()));
            }
            let ixf = format!("{out}/{}.index", pol.name);
            std::fs::write(&ixf, index)?;
            println!("wrote {} plays, index {ixf}", idx.len());
        }
        "trace" => {
            // policy <rom> <problem> trace <id|file.pol> <case-substring> [--dir D]   (set POLICY_TRACE=1)
            let pb = load(&args[2], ".", &gg, false)?;
            let env = Env::full();
            let ev = Eval::new(&pb, &env);
            let dir = opt(&args, "--dir").unwrap_or("policy/runs/inventory".into());
            let inv = Inventory::open(std::path::Path::new(&dir))?;
            let what = args.get(4).cloned().unwrap_or_default();
            let pol = if what.ends_with(".pol") { dsl::parse(&std::fs::read_to_string(&what)?)? } else { inv.bodies.get(&what).cloned().ok_or_else(|| anyhow::anyhow!("no policy {what}"))? };
            let pat = args.get(5).cloned().unwrap_or_default();
            let idx: Vec<usize> = (0..pb.cases.len()).filter(|&i| pb.cases[i].name.contains(&pat)).collect();
            let mut gs = games(&gg, 1)?;
            let ps = ev.run(&mut gs, &inv.bodies, &pol.body, &idx, &obrrt_policy::inv::instance_of(&pol), false)?;
            for (i, p) in idx.iter().zip(&ps) {
                println!("== {} done {:?} {:?}", pb.cases[*i].name, p.out.done, p.out.end);
                for r in &p.desc.rows {
                    println!("{r}");
                }
            }
        }
        "inv" => {
            let dir = opt(&args, "--dir").unwrap_or("policy/runs/inventory".into());
            let mut inv = Inventory::open(std::path::Path::new(&dir))?;
            match args.get(4).map(|s| s.as_str()) {
                Some("list") => {
                    let st = opt(&args, "--status");
                    println!("{:<8} {:<9} {:<8} {:<8} {:>8} {:>5} {:>6} {:>6} parents / note", "id", "origin", "status", "niche", "fit", "succ", "usage", "credit");
                    for e in inv.entries.iter().filter(|e| st.as_ref().is_none_or(|s| &e.status == s)) {
                        println!("{:<8} {:<9} {:<8} {:<8} {:>8.1} {:>5} {:>6} {:>6.2} {:?} {}", e.id, e.origin, e.status, e.niche, e.fit, e.succ, e.usage, e.credit, e.parents, e.note);
                    }
                }
                Some("report") => {
                    let mut by: std::collections::BTreeMap<String, (usize, usize, f64)> = Default::default();
                    for e in &inv.entries {
                        let x = by.entry(e.origin.clone()).or_default();
                        x.0 += 1;
                        x.1 += (e.status == "active") as usize;
                        x.2 = x.2.max(e.fit);
                    }
                    for (k, v) in &by {
                        println!("origin {k:<10} total {:>4} active {:>4} best fit {:.1}", v.0, v.1, v.2);
                    }
                    let niches: std::collections::BTreeSet<_> = inv.entries.iter().filter(|e| e.status == "active").map(|e| e.niche.clone()).collect();
                    println!("niches covered {} of 64; best-on-a-start holders: {:?}", niches.len(), inv.best_start.iter().map(|b| b.1.clone()).collect::<std::collections::BTreeSet<_>>());
                    println!("operator weights {:?}", inv.op_w);
                    println!("{}", inv.share_line());
                }
                Some("retire") => {
                    inv.set_status(&args[5], "retired", &opt(&args, "--reason").unwrap_or_default())?;
                    println!("retired {}", args[5]);
                }
                Some("sample") => {
                    let pb = load(&args[2], ".", &gg, false)?;
                    let env = Env::full();
                    let ev = Eval::new(&pb, &env);
                    let mut gs = games(&gg, threads)?;
                    let mut r = SmallRng::seed_from_u64(opt(&args, "--seed").map_or(1, |s| s.parse().unwrap_or(1)));
                    let n: usize = opt(&args, "--draws").map_or(Ok(50), |s| s.parse())?;
                    println!("{}", inv.sample_template(&ev, &mut gs, &args[5], n, &mut r)?);
                    inv.save()?;
                }
                Some("revive") => {
                    inv.set_status(&args[5], "active", "revived")?;
                    println!("revived {}", args[5]);
                }
                Some("eval") => {
                    let pb = load(&args[2], ".", &gg, false)?;
                    let env = Env::full();
                    let ev = Eval::new(&pb, &env);
                    let mut gs = games(&gg, threads)?;
                    let pol = dsl::parse(&std::fs::read_to_string(&args[5])?)?;
                    let ps = eval_all(&ev, &mut gs, &inv, &pol.body, &obrrt_policy::inv::instance_of(&pol))?;
                    report(&ev, &pol.name, &ps, true);
                    let tr: Vec<Played> = ps.iter().filter(|(i, _)| pb.cases[*i].train).map(|(_, p)| Played { out: p.out.clone(), desc: p.desc.clone() }).collect();
                    println!("niche {}", niche_of(&tr));
                    if args.iter().any(|a| a == "--admit") {
                        let (ok, id, _) = inv.consider(&ev, &mut gs, pol, "llm", "cli", &[], &opt(&args, "--note").unwrap_or_default(), true)?;
                        inv.save()?;
                        println!("admitted {ok} as {id}");
                    }
                }
                _ => bail!("inv list|report|eval <file>|retire <id>|revive <id>"),
            }
        }
        c => bail!("unknown command {c}"),
    }
    Ok(())
}
