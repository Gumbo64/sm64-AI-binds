//! hybrid <rom> <problem> <variant> [--seeds n] [--threads n] [--budgets 20000,50000,150000,560000]
//!        [--cands a,b,c] [--draws k] [--pop n] [--out file.json] [--lib dir,dir]
//! variants: pink | seeded | repolicy | market | polish | policy   (V0..V4, P)
//! Run from objective-rrt-v2. Needs the `ga` feature.
#[cfg(not(feature = "ga"))]
fn main() {
    eprintln!("build with --features ga");
}

#[cfg(feature = "ga")]
fn main() -> anyhow::Result<()> {
    use obrrt_gp::problems::load;
    use obrrt_policy::eval::Env;
    use obrrt_policy::hybrid::{Library, Params, Variant, run, summarize};
    use sm64_binds::SM64GameGenerator;
    use std::path::Path;
    unsafe {
        std::env::set_var("OBRRT_LAKITU_ARC", "1");
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    anyhow::ensure!(args.len() >= 4, "hybrid <rom> <problem> <variant> ...");
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let pb = load(&args[2], ".", &gg, false)?;
    let wseed = args[3] == "wseed" || args[3] == "wplain";
    let variant = if wseed { Variant::Seeded } else { Variant::parse(&args[3]).ok_or_else(|| anyhow::anyhow!("variant {}", args[3]))? };
    let seeds: u64 = opt("--seeds").map_or(Ok(3), |s| s.parse())?;
    let threads: usize = opt("--threads").map_or(Ok(4), |s| s.parse())?;
    let budgets: Vec<u64> = opt("--budgets").unwrap_or("20000,50000,150000,560000".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let dirs: Vec<String> = opt("--lib").unwrap_or("policy/runs/hybrid/lib".into()).split(',').map(String::from).collect();
    let dpaths: Vec<&Path> = dirs.iter().map(|d| Path::new(d.as_str())).collect();
    let lib = Library::load(&dpaths);
    let cands: Vec<String> = match opt("--cands") {
        Some(c) => c.split(',').map(String::from).filter(|n| lib.bodies.contains_key(n)).collect(),
        None => default_cands(&args[2]).into_iter().filter(|n| lib.bodies.contains_key(*n)).map(String::from).collect(),
    };
    println!("library {} policies; candidates {:?}", lib.bodies.len(), cands);
    if wseed {
        // the v2 Worker GA seeded with the best policy rollout (wseed) or not (wplain, = the baseline with the same code path)
        let scale: f64 = opt("--scale").map_or(Ok(0.2), |s| s.parse())?;
        let pink: f64 = opt("--pink").map_or(Ok(0.0), |s| s.parse())?;
        let env = Env::full();
        let t0 = std::time::Instant::now();
        let rs = obrrt_policy::hybrid::worker_seeded(&gg, &pb, &lib, &env, &cands, threads, seeds, scale, pink, args[3] == "wseed")?;
        let mut js = vec![];
        for (s, r) in rs.iter().enumerate() {
            js.push(obrrt_policy::bench::summarize(&pb, &format!("{} pink={pink} scale={scale} seed{}", args[3], s + 1), r));
        }
        println!("wall {:.0}s", t0.elapsed().as_secs_f64());
        if let Some(path) = opt("--rows") {
            let mut t = String::new();
            for (si, r) in rs.iter().enumerate() {
                for (c, x) in pb.cases.iter().zip(r) {
                    t += &format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", args[3], si + 1, obrrt_policy::bench::split(c), c.name, scale, x.done.map_or(-1i64, |d| d as i64), x.emu);
                }
            }
            std::fs::write(path, t)?;
        }
        if let Some(o) = opt("--out") {
            std::fs::write(o, serde_json::to_string_pretty(&serde_json::json!({"pink": pink, "scale": scale, "seeds": js}))?)?;
        }
        return Ok(());
    }
    let p = Params { variant, budgets: budgets.clone(), pop: opt("--pop").map_or(Ok(16), |s| s.parse())?, cands, draws: opt("--draws").map_or(Ok(1), |s| s.parse())? };
    let env = Env::full();
    let mut games = (0..threads).map(|_| gg.create_game()).collect::<anyhow::Result<Vec<_>>>()?;
    let t0 = std::time::Instant::now();
    let rows = run(&mut games, &pb, &lib, &env, &p, seeds)?;
    let label = format!("{} {:?}", args[2], variant);
    let js = summarize(&pb, &label, &rows, &budgets);
    // per-start rows (clean validation): variant seed split case budget done(-1 none) emu
    if let Some(path) = opt("--rows") {
        let mut t = String::new();
        for (si, seedrows) in rows.iter().enumerate() {
            for (c, snaps) in pb.cases.iter().zip(seedrows) {
                for sn in snaps {
                    t += &format!("{}\t{}\t{}\t{}\t{}\t{}\t{}\n", args[3], si + 1, obrrt_policy::bench::split(c), c.name, sn.budget, sn.done.map_or(-1i64, |d| d as i64), sn.emu);
                }
            }
        }
        std::fs::write(path, t)?;
    }
    println!("wall {:.0}s", t0.elapsed().as_secs_f64());
    if let Some(o) = opt("--out") {
        std::fs::write(o, serde_json::to_string_pretty(&serde_json::json!({"problem": args[2], "variant": args[3], "budgets": budgets, "summary": js}))?)?;
    }
    Ok(())
}

#[cfg(feature = "ga")]
fn default_cands(problem: &str) -> Vec<&'static str> {
    if problem.starts_with("lakitu") {
        vec!["air-lak-2", "air-lak-4", "llm-lakitu-bridge-2", "ground-m02-gate", "ground-f06-refresh-frozen", "llm-gate-speed-8"]
    } else {
        vec!["ground-m02-gate", "ground-f06-refresh-frozen", "ground-f07-slidekick-frozen", "llm-gate-speed-8", "air-X00003", "ground-X00005", "ground-X00006", "air-speed-chain-3", "ground-g08-aligned", "ground-g09-refresh", "ground-g10-slidekick", "ground-g11-waypoint"]
    }
}
