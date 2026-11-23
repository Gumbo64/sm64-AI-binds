//! gp <rom> <problem> info
//! gp <rom> <problem> baseline [--dump dir]
//! gp <rom> <problem> eval <program file> [--dump dir]
//! gp <rom> <problem> evolve <out dir> [--seed n] [--pop n] [--gens n] [--threads n]
//!     [--init file[,file]] [--adf n] [--parsimony x] [--tournament n] [--budget M]
//!
//! Run from objective-rrt-v2. Program files are in lang.rs's text form.

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::{Result, bail};
use obrrt_core::GamePad;
use obrrt_gp::evolve::{Params, eval_cases, evolve};
use obrrt_gp::lang::Prog;
use obrrt_gp::problems::{Problem, load, write_raw};
use obrrt_gp::rollout::{Case, End, OverlayCtrl, Outcome, PadsCtrl, ProgCtrl, play};
use sm64_binds::{SM64Game, SM64GameGenerator};

fn opt(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn split(c: &Case) -> &'static str {
    if c.train {
        "tr"
    } else if c.val {
        "VA"
    } else {
        "TE"
    }
}

fn row(c: &Case, o: &Outcome) -> String {
    format!(
        "{:<3} {:<44} {:>5} {:>7.0} {:>8} {:>7} {:?} eta {:.0}->{:.0} {}",
        split(c),
        c.name,
        c.start.forward_vel as i32,
        o.fitness,
        o.done.map_or("-".to_string(), |d| d.to_string()),
        o.frames,
        o.end,
        o.start_eta,
        o.best_eta,
        o.report
    )
}

fn summary(pb: &Problem, label: &str, os: &[Outcome]) {
    for (sp, tag) in [("tr", "train"), ("VA", "val  "), ("TE", "test ")] {
        let rows: Vec<&Outcome> = pb.cases.iter().zip(os).filter(|(c, _)| split(c) == sp).map(|(_, o)| o).collect();
        if rows.is_empty() {
            continue;
        }
        let succ = rows.iter().filter(|o| o.done.is_some()).count();
        let fr = rows.iter().filter_map(|o| o.done).sum::<usize>() as f64 / succ.max(1) as f64;
        let fit = rows.iter().map(|o| o.fitness).sum::<f64>() / rows.len() as f64;
        println!("{label}: {tag} success {succ}/{} mean frames {fr:.1} mean fitness {fit:.1} emu {}", rows.len(), rows.iter().map(|o| o.frames).sum::<u64>());
    }
}

fn dump(pb: &Problem, dir: &str, tag: &str, os: &[Outcome]) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut index = String::new();
    for (c, o) in pb.cases.iter().zip(os) {
        let mut pads: Vec<GamePad> = pb.movies[c.movie].1[..c.frame - c.noise.len()].to_vec();
        pads.extend_from_slice(&c.noise);
        pads.extend_from_slice(&o.pads);
        let file = format!("{dir}/{tag}-{}.raw", c.name.replace(['/', ','], "_"));
        write_raw(&file, &pads)?;
        let way = pb.tasks[c.task].way_text();
        if !way.is_empty() {
            std::fs::write(format!("{file}.wp"), way)?;
        }
        index.push_str(&format!("{file} {} {} {}\n", c.frame, o.done.map_or(-1, |d| d as i64), o.pads.len()));
    }
    std::fs::write(format!("{dir}/{tag}.index"), index)?;
    Ok(())
}

fn games(g: &SM64GameGenerator, n: usize) -> Result<Vec<SM64Game>> {
    (0..n).map(|_| g.create_game()).collect()
}

fn run_ctrl(pb: &Problem, gs: &mut [SM64Game], f: &(dyn Fn(usize) -> Box<dyn obrrt_gp::rollout::Ctrl> + Sync), keep: bool) -> Result<Vec<Outcome>> {
    obrrt_gp::rollout::par(gs, pb.cases.len(), |g, i| {
        let c = &pb.cases[i];
        let mut ctrl = f(i);
        play(g, c, pb.tasks[c.task].as_ref(), ctrl.as_mut(), &pb.roll, keep)
    })
}

/// Over all the cases on one thread with coverage, so the program can be shown
/// without what never ran.
fn pruned(pb: &Problem, game: &mut SM64Game, prog: &Prog) -> Result<Prog> {
    let trace: obrrt_gp::lang::Trace = RefCell::new(HashMap::new());
    for c in &pb.cases {
        play(game, c, pb.tasks[c.task].as_ref(), &mut ProgCtrl { prog, trace: Some(&trace) }, &pb.roll, false)?;
    }
    Ok(prog.pruned(&trace.borrow()))
}

fn main() -> Result<()> {
    // The Lakitu skip's Hill mode exists only with this on.
    unsafe {
        std::env::set_var("OBRRT_LAKITU_ARC", "1");
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!("usage: gp <rom> <problem> <info|baseline|eval|evolve> ...");
    }
    let gen_gen = SM64GameGenerator::from_file(&args[1])?;
    let skip_test = args.iter().any(|a| a == "--skip-test");
    let mut pb = load(&args[2], ".", &gen_gen, skip_test)?;
    // Reports play to the horizon; only the evolution stops stalled plays.
    let stall_evolve = pb.roll.stall;
    pb.roll.stall = usize::MAX / 2;
    let threads: usize = opt(&args, "--threads").map_or(Ok(4), |s| s.parse())?;
    let dump_dir = opt(&args, "--dump");
    let mut gs = games(&gen_gen, threads)?;
    let ntrain = pb.cases.iter().filter(|c| c.train).count();
    let nval = pb.cases.iter().filter(|c| c.val).count();
    println!("{}: {} cases ({} train, {} val, {} held out)", pb.name, pb.cases.len(), ntrain, nval, pb.cases.len() - ntrain - nval);
    match args[3].as_str() {
        "info" => {
            for c in &pb.cases {
                let s = &c.start;
                let e = pb.tasks[c.task].eta(s, s, &mut obrrt_core::Mem::default());
                println!("{:<3} {:<44} pos {:7.0} {:6.0} {:7.0} fv {:5.1} act {:#010x} eta {:?}", split(c), c.name, s.pos[0], s.pos[1], s.pos[2], s.forward_vel, s.action, e);
            }
        }
        "baseline" => {
            println!("--- the movie's own inputs");
            let t0 = std::time::Instant::now();
            let os = run_ctrl(&pb, &mut gs, &|i| Box::new(PadsCtrl { pads: Box::leak(pb.cases[i].ref_pads.clone().into_boxed_slice()), at: 0 }), dump_dir.is_some())?;
            for (c, o) in pb.cases.iter().zip(&os) {
                println!("{}", row(c, o));
            }
            summary(&pb, "movie", &os);
            if let Some(d) = &dump_dir {
                dump(&pb, d, "movie", &os)?;
            }
            println!("--- the hand overlay on a steer-at-target stick");
            // Followers: steering straight at the waypoint is a program.
            let f = |i: usize| -> Box<dyn obrrt_gp::rollout::Ctrl> {
                let t = pb.cases[i].task;
                let task: &'static dyn obrrt_gp::rollout::Task = unsafe { std::mem::transmute(pb.tasks[t].as_ref()) };
                Box::new(OverlayCtrl { overlay: task.overlay().unwrap_or(obrrt_core::PadOverlay::new(|_, _, g| g)), steer: Box::new(move |s| task.target_for_baseline(s)) })
            };
            let os = run_ctrl(&pb, &mut gs, &f, dump_dir.is_some())?;
            for (c, o) in pb.cases.iter().zip(&os) {
                println!("{}", row(c, o));
            }
            summary(&pb, "overlay", &os);
            if let Some(d) = &dump_dir {
                dump(&pb, d, "overlay", &os)?;
            }
            println!("baselines {:.1}s", t0.elapsed().as_secs_f64());
        }
        "eval" => {
            let prog = Prog::from_text(&std::fs::read_to_string(&args[4])?)?;
            let idx: Vec<usize> = (0..pb.cases.len()).collect();
            let os = eval_cases(&mut gs, &pb, &prog, &idx, true)?;
            for (c, o) in pb.cases.iter().zip(&os) {
                println!("{}", row(c, o));
            }
            summary(&pb, "program", &os);
            if let Some(d) = &dump_dir {
                dump(&pb, d, "prog", &os)?;
            }
            let p = pruned(&pb, &mut gs[0], &prog)?;
            let os2 = eval_cases(&mut gs, &pb, &p, &idx, true)?;
            let same = os.iter().zip(&os2).all(|(a, b)| a.pads == b.pads);
            println!("size {} pruned {} (same inputs on all cases: {same})", prog.size(), p.size());
            println!("{}", p.pretty());
        }
        "evolve" => {
            let out = &args[4];
            std::fs::create_dir_all(out)?;
            let mut params = Params::default();
            if let Some(s) = opt(&args, "--seed") {
                params.seed = s.parse()?;
            }
            if let Some(s) = opt(&args, "--pop") {
                params.pop = s.parse()?;
            }
            if let Some(s) = opt(&args, "--gens") {
                params.gens = s.parse()?;
            }
            if let Some(s) = opt(&args, "--parsimony") {
                params.parsimony = s.parse()?;
            }
            if let Some(s) = opt(&args, "--tournament") {
                params.tournament = s.parse()?;
            }
            if let Some(s) = opt(&args, "--max-depth") {
                params.max_depth = s.parse()?;
            }
            if let Some(s) = opt(&args, "--max-size") {
                params.max_size = s.parse()?;
            }
            if let Some(s) = opt(&args, "--budget") {
                params.budget = (s.parse::<f64>()? * 1e6) as u64;
            }
            pb.roll.stall = opt(&args, "--stall").map_or(Ok(stall_evolve), |s| s.parse())?;
            if let Some(s) = opt(&args, "--adf") {
                let n: u8 = s.parse()?;
                pb.gen_cfg.adf_p = n;
                pb.gen_cfg.adf_b = n;
            }
            if args.iter().any(|a| a == "--ref") {
                pb.gen_cfg.use_ref = true;
            }
            if args.iter().any(|a| a == "--noworld") {
                pb.gen_cfg.use_world = false;
            }
            let seeds: Vec<Prog> = match opt(&args, "--init") {
                Some(f) => f.split(',').map(|f| Prog::from_text(&std::fs::read_to_string(f)?)).collect::<Result<_>>()?,
                None => vec![],
            };
            println!("{params:?}");
            let mut logf = std::fs::File::create(format!("{out}/gens.txt"))?;
            let t0 = std::time::Instant::now();
            let run = evolve(&mut gs, &pb, &params, &seeds, &mut logf)?;
            println!("{}", run.log.last().cloned().unwrap_or_default());
            pb.roll.stall = usize::MAX / 2;
            std::fs::write(format!("{out}/best.prog"), run.best.text())?;
            std::fs::write(format!("{out}/best_train.prog"), run.best_train.text())?;
            let idx: Vec<usize> = (0..pb.cases.len()).collect();
            let os = eval_cases(&mut gs, &pb, &run.best, &idx, true)?;
            for (c, o) in pb.cases.iter().zip(&os) {
                println!("{}", row(c, o));
            }
            summary(&pb, "best", &os);
            if run.best_train != run.best {
                let ot = eval_cases(&mut gs, &pb, &run.best_train, &idx, false)?;
                summary(&pb, "best_train", &ot);
            }
            let p = pruned(&pb, &mut gs[0], &run.best)?;
            let os2 = eval_cases(&mut gs, &pb, &p, &idx, true)?;
            let same = os.iter().zip(&os2).all(|(a, b)| a.pads == b.pads);
            println!("emu frames {:.1} M in {:.0}s", run.emu_total as f64 / 1e6, t0.elapsed().as_secs_f64());
            println!("size {} pruned {} (same inputs on all cases: {same})", run.best.size(), p.size());
            let pretty = p.pretty();
            println!("{pretty}");
            std::fs::write(format!("{out}/best.pruned.prog"), p.text())?;
            std::fs::write(format!("{out}/best.txt"), pretty)?;
            let _ = End::Done;
            if let Some(d) = &dump_dir {
                dump(&pb, d, "best", &os)?;
            }
        }
        c => bail!("unknown command {c}"),
    }
    Ok(())
}
