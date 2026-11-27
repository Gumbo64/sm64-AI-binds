//! plga <rom> <problem> run   [--seeds n] [--threads n] [--budgets a,b,c,d] [--lib dir] [--pool a,b,c|all] [--pop n]
//!                            [--nomarket] [--no-fine] [--mpc] [--no-seed0] [--no-eps] [--no-tune] [--no-repol] [--lib2 dir] [--rows file.tsv] [--genomes file] [--out file.json]
//! plga <rom> <problem> adapt [--lib dir] [--n starts]     closed-loop property check (later segment re-reacts)
//! Run from objective-rrt-v2. Needs the `ga` feature. Genome = list of (policy + args/seed | pink seed, hold frames).
#[cfg(not(feature = "ga"))]
fn main() {
    eprintln!("build with --features ga");
}

#[cfg(feature = "ga")]
fn main() -> anyhow::Result<()> {
    use obrrt_gp::problems::load;
    use obrrt_policy::eval::Env;
    use obrrt_policy::hybrid::{Library, summarize};
    use obrrt_policy::plga::*;
    use sm64_binds::SM64GameGenerator;
    use std::path::Path;
    unsafe {
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned());
    anyhow::ensure!(args.len() >= 4, "plga <rom> <problem> run|adapt ...");
    let gg = SM64GameGenerator::from_file(&args[1])?;
    let pb = load(&args[2], ".", &gg, false)?;
    let flag = |k: &str| args.iter().any(|a| a == k);
    let seeds: u64 = opt("--seeds").map_or(Ok(3), |s| s.parse())?;
    let threads: usize = opt("--threads").map_or(Ok(2), |s| s.parse())?;
    let budgets: Vec<u64> = opt("--budgets").unwrap_or("20000,60000,200000,560000".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let dir = opt("--lib").unwrap_or("policy/runs/clean/lib".into());
    let extra = opt("--lib2");
    let mut dirs: Vec<&Path> = vec![Path::new(&dir)];
    if let Some(e) = extra.as_deref() {
        dirs.push(Path::new(e));
    }
    let lib = Library::load(&dirs);
    let pool: Vec<String> = match opt("--pool").as_deref() {
        Some("all") => {
            let mut v: Vec<String> = lib.bodies.keys().cloned().collect();
            v.sort();
            v
        }
        Some(c) => c.split(',').map(String::from).filter(|n| lib.bodies.contains_key(n)).collect(),
        None => ["ground-m02-gate", "ground-f06-refresh-frozen", "ground-f07-slidekick-frozen", "llm-gate-speed-8", "air-X00003", "ground-X00005", "ground-X00006", "air-speed-chain-3", "ground-g08-aligned", "ground-g09-refresh", "ground-g10-slidekick", "ground-g11-waypoint"]
            .iter()
            .filter(|n| lib.bodies.contains_key(**n))
            .map(|s| s.to_string())
            .collect(),
    };
    println!("library {} policies; pool {} {:?}", lib.bodies.len(), pool.len(), pool);
    let env = Env::full();
    let mut games = (0..threads).map(|_| gg.create_game()).collect::<anyhow::Result<Vec<_>>>()?;
    if args[3] == "adapt" {
        return adapt(&mut games[0], &pb, &lib, &env, &pool, opt("--n").map_or(Ok(6), |s| s.parse())?);
    }
    let p = Params { budgets: budgets.clone(), pop: opt("--pop").map_or(Ok(16), |s| s.parse())?, pool, market: !args.iter().any(|a| a == "--nomarket"), fine_ops: !flag("--no-fine"), mpc_wrap: flag("--mpc"), seed0: !flag("--no-seed0"), eps: !flag("--no-eps"), finetune: !flag("--no-tune"), repol: !flag("--no-repol") };
    let t0 = std::time::Instant::now();
    let rows = run(&mut games, &pb, &lib, &env, &p, seeds, opt("--genomes").as_deref())?;
    let js = summarize(&pb, &format!("{} plga", args[2]), &rows, &budgets);
    if let Some(path) = opt("--rows") {
        let mut t = String::new();
        for (si, seedrows) in rows.iter().enumerate() {
            for (c, snaps) in pb.cases.iter().zip(seedrows) {
                for sn in snaps {
                    t += &format!("plga\t{}\t{}\t{}\t{}\t{}\t{}\n", si + 1, obrrt_policy::bench::split(c), c.name, sn.budget, sn.done.map_or(-1i64, |d| d as i64), sn.emu);
                }
            }
        }
        std::fs::write(path, t)?;
    }
    println!("wall {:.0}s", t0.elapsed().as_secs_f64());
    if let Some(o) = opt("--out") {
        std::fs::write(o, serde_json::to_string_pretty(&serde_json::json!({"problem": args[2], "budgets": budgets, "summary": js}))?)?;
    }
    Ok(())
}

/// Same later segment B, different earlier segment A: B must emit different pads (it re-reacts); replaying B's
/// recorded pads instead (the flattened version) loses that and ends elsewhere.
#[cfg(feature = "ga")]
fn adapt(g: &mut sm64_binds::SM64Game, pb: &obrrt_gp::problems::Problem, lib: &obrrt_policy::hybrid::Library, env: &obrrt_policy::eval::Env, pool: &[String], n: usize) -> anyhow::Result<()> {
    use obrrt_policy::eval::{Drv, Seg, play_segs};
    use obrrt_policy::inv::instance_of;
    use obrrt_policy::plga::*;
    anyhow::ensure!(pool.len() >= 3, "pool too small");
    let (la, lb) = (25usize, 40usize);
    let mut diff_tot = 0;
    let mut cmp_tot = 0;
    let mut det_ok = true;
    let mut ol_diff = 0;
    let mut ol_cmp = 0;
    for (ci, c) in pb.cases.iter().enumerate().take(n) {
        let cx = Ctx { case: c, task: pb.tasks[c.task].as_ref(), roll: pb.roll, lib, env };
        let b = pool[0].clone();
        let gene = |name: &str, hold| Gene { kind: Kind::Pol { name: name.to_string(), inst: instance_of(&lib.bodies[name]) }, hold, fine: Default::default() };
        // prefixes: other policies and pink noise
        let mut prefixes: Vec<Gene> = pool[1..4].iter().map(|n| gene(n, la)).collect();
        prefixes.push(Gene { kind: Kind::Pink { seed: 7 }, hold: la, fine: Default::default() });
        prefixes.push(Gene { kind: Kind::Pink { seed: 8 }, hold: la, fine: Default::default() });
        let mut runs: Vec<(Vec<obrrt_core::GamePad>, Genome)> = Vec::new();
        for a in &prefixes {
            let gn = vec![a.clone(), gene(&b, lb)];
            let o = rollout(g, &cx, &gn, true)?;
            let o2 = rollout(g, &cx, &gn, true)?;
            det_ok &= o.pads == o2.pads;
            runs.push((o.pads, gn));
        }
        // window of B in each run (if the run lasted that long)
        let win: Vec<&[obrrt_core::GamePad]> = runs.iter().map(|r| r.0.get(la..(la + lb).min(r.0.len())).unwrap_or(&[])).collect();
        let mut cd = 0;
        let mut cc = 0;
        for i in 0..win.len() {
            for j in i + 1..win.len() {
                let m = win[i].len().min(win[j].len());
                if m == 0 {
                    continue;
                }
                cc += 1;
                if win[i][..m] != win[j][..m] {
                    cd += 1;
                }
            }
        }
        diff_tot += cd;
        cmp_tot += cc;
        // open-loop control: run 0's B window as recorded pads after the other prefixes
        let slice: Vec<obrrt_core::GamePad> = win[0].to_vec();
        for k in 1..prefixes.len() {
            let a_drv = match &prefixes[k].kind {
                Kind::Pol { name, inst } => Drv::Pol { body: &lib.bodies[name].body, inst: *inst },
                Kind::Pink { seed } => {
                    let mut r = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(*seed);
                    Drv::Pads(obrrt_search::noise::pink_inputs(&mut r, 300))
                }
            };
            let segs = [Seg { drv: a_drv, hold: la, fine: Default::default() }, Seg { drv: Drv::Pads(slice.clone()), hold: lb, fine: Default::default() }];
            let pl = play_segs(g, c, pb.tasks[c.task].as_ref(), &segs, lib, env, &pb.roll, true)?;
            let wp = pl.out.pads.get(la..(la + lb).min(pl.out.pads.len())).unwrap_or(&[]);
            let m = wp.len().min(win[k].len());
            let (e_closed, e_open) = (runs[k].0.len(), pl.out.pads.len());
            ol_cmp += 1;
            // closed-loop B vs the recorded replay of run 0's B: they differ whenever B re-reacted
            if m > 0 && wp[..m] != win[k][..m] {
                ol_diff += 1;
            }
            let _ = (e_closed, e_open);
        }
        let hamming: Vec<String> = (1..win.len()).map(|j| {
            let m = win[0].len().min(win[j].len());
            format!("{}/{}", (0..m).filter(|&t| win[0][t] != win[j][t]).count(), m)
        }).collect();
        println!("start {ci} {}: same B ({b} x{lb}) after 5 different A's; frames in B window differing from A#0 run: {}", c.name, hamming.join(" "));
    }
    println!("pairs of runs whose B-window pads differ: {diff_tot}/{cmp_tot}");
    println!("closed-loop B differs from the recorded replay of another run's B: {ol_diff}/{ol_cmp}");
    println!("determinism (same genome twice -> identical pads): {det_ok}");
    Ok(())
}
