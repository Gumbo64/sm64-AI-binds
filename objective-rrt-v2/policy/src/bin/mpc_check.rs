//! Comparison of the stick refiners (policy/src/mpc.rs) on real start states.
//!
//!   mpc_check <rom> info   <problem>
//!   mpc_check <rom> curve  <problem> [threads=4] [knots=3] [horizon=24] [evals=256] [seeds=3]
//!   mpc_check <rom> smooth <problem> [threads=4]
//!   mpc_check <rom> loop   <problem> [threads=4] [frames=48]
//!
//! Instances: for each case of the problem, start states are taken along the
//! case's reference movie (`ref_pads`), with the movie's own buttons held; the
//! optimiser moves only the stick (angle offset from an aim, magnitude).

use std::fmt::Write as _;
use std::time::Instant;

use anyhow::Result;
use obrrt_core::{GamePad, St, read_state};
use obrrt_gp::problems::{Problem, load};
use obrrt_gp::rollout::par;
use obrrt_policy::mpc::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SM64GameGenerator, SaveState};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Steer towards a point 35-70 degrees off the velocity direction.
    Steer,
    /// Keep horizontal speed through a ground/air transition.
    Hold,
    /// Land at a spot beside the natural landing, not on it.
    Land,
    /// The problem's own task fitness (discounted ETA).
    Task,
}

struct Inst {
    kind: Kind,
    case: usize,
    off: usize,
    name: String,
    /// Steer: offset of the target from the velocity direction (radians).
    theta: f32,
    /// Land: natural landing of the reference (and the goal/forbidden derived from it).
    land0: Option<[f32; 3]>,
    side: f32,
}

const AIR: u32 = 0x800;

fn air(s: &St) -> bool {
    s.action & AIR != 0
}

fn hspeed(s: &St) -> f32 {
    s.vel[0].hypot(s.vel[2])
}

/// Case start, then `ref_pads` for up to `n` frames: the states after each.
struct RefSim {
    pads: Vec<GamePad>,
    traj: Vec<St>,
}

fn ref_sim(game: &mut SM64Game, pb: &Problem, ci: usize, n: usize) -> Result<RefSim> {
    let c = &pb.cases[ci];
    game.load_state(&c.save)?;
    for p in &c.noise {
        game.step_game(*p)?;
    }
    let mut traj = vec![c.start.clone()];
    let pads: Vec<GamePad> = c.ref_pads.iter().take(n).copied().collect();
    for (t, p) in pads.iter().enumerate() {
        game.step_game(*p)?;
        traj.push(read_state(game, c.start.frame + t as u32 + 1, false));
    }
    Ok(RefSim { pads, traj })
}

/// The savestate `off` reference frames after the case start.
fn base_at(game: &mut SM64Game, pb: &Problem, ci: usize, rs: &RefSim, off: usize) -> Result<SaveState> {
    let c = &pb.cases[ci];
    game.load_state(&c.save)?;
    for p in &c.noise {
        game.step_game(*p)?;
    }
    for p in &rs.pads[..off] {
        game.step_game(*p)?;
    }
    Ok(game.save_state())
}

fn pick_insts(game: &mut SM64Game, pb: &Problem, kind: Kind, h: usize, per_case: usize, max_cases: usize) -> Result<Vec<Inst>> {
    let mut out = vec![];
    let mut seen_movie = std::collections::HashMap::<usize, usize>::new();
    for ci in 0..pb.cases.len() {
        let c = &pb.cases[ci];
        if *seen_movie.get(&c.movie).unwrap_or(&0) >= 2 {
            continue;
        }
        let rs = ref_sim(game, pb, ci, 160)?;
        let mut cand: Vec<usize> = vec![];
        for off in 0..rs.pads.len().saturating_sub(h + 2) {
            let s = &rs.traj[off];
            let ok = match kind {
                Kind::Steer => hspeed(s) >= 35.0 && off % 12 == 0,
                Kind::Hold => {
                    // a ground/air change within the horizon
                    hspeed(s) >= 35.0 && rs.traj[off..off + h].iter().any(|x| air(x) != air(s)) && off % 6 == 0
                }
                Kind::Land => {
                    if !air(s) || hspeed(s) < 25.0 || off % 4 != 0 {
                        false
                    } else {
                        let rem = rs.traj[off..off + h].iter().position(|x| !air(x));
                        rem.is_some_and(|r| (8..=h - 2).contains(&r))
                    }
                }
                Kind::Task => off % 15 == 0 && hspeed(s) >= 20.0,
            };
            if ok {
                cand.push(off);
            }
        }
        if cand.is_empty() {
            continue;
        }
        *seen_movie.entry(c.movie).or_default() += 1;
        // spread the picks
        for k in 0..per_case.min(cand.len()) {
            let off = cand[k * cand.len() / per_case.min(cand.len())];
            let s = &rs.traj[off];
            let theta = [0.6f32, -0.6, 1.2, -1.2][(ci + k) % 4];
            let land0 = if kind == Kind::Land { rs.traj[off..off + h].iter().find(|x| !air(x)).map(|x| x.pos) } else { None };
            out.push(Inst { kind, case: ci, off, name: format!("{}@{}+{} fv{:.0} {}", c.name.split('@').next().unwrap_or(""), c.frame, off, s.forward_vel, ""), theta, land0, side: if (ci + k) % 2 == 0 { 1.0 } else { -1.0 } });
        }
        if out.len() >= max_cases {
            break;
        }
    }
    Ok(out)
}

/// Everything needed to build a Prob for an instance (owns the scorer).
struct Built<'a> {
    scorer: Box<dyn Scorer + 'a>,
    aim: Aim,
    ang_max: f32,
    mag_lo: f32,
    mag_hi: f32,
}

fn build<'a>(pb: &'a Problem, inst: &Inst, start: &St, h: usize) -> Built<'a> {
    let v = hspeed(start);
    let vyaw = if v > 5.0 { yaw_of(start.vel[0], start.vel[2]) } else { start.yaw };
    let rad = |y: i16| y as f32 * std::f32::consts::TAU / 65536.0;
    match inst.kind {
        Kind::Steer => {
            let r = (0.6 * v * h as f32).max(300.0);
            let a = rad(vyaw) + inst.theta;
            let target = [start.pos[0] + r * a.sin(), start.pos[1], start.pos[2] + r * a.cos()];
            Built { scorer: Box::new(Reach::new(target)), aim: Aim::Target(target), ang_max: 1.4, mag_lo: 0.15, mag_hi: 1.0 }
        }
        Kind::Hold => Built { scorer: Box::new(HoldSpeed::new()), aim: Aim::Vel, ang_max: 1.0, mag_lo: 0.15, mag_hi: 1.0 },
        Kind::Land => {
            let l0 = inst.land0.unwrap_or(start.pos);
            let a = rad(vyaw) + std::f32::consts::FRAC_PI_2 * inst.side;
            let d = 140.0;
            let goal = [l0[0] + d * a.sin(), l0[1], l0[2] + d * a.cos()];
            Built { scorer: Box::new(LandAt::new(goal, Some((l0, 70.0)))), aim: Aim::Vel, ang_max: 1.2, mag_lo: 0.15, mag_hi: 1.0 }
        }
        Kind::Task => {
            let task = pb.tasks[pb.cases[inst.case].task].as_ref();
            Built { scorer: Box::new(TaskFit::new(task, pb.roll)), aim: Aim::Vel, ang_max: 1.0, mag_lo: 0.15, mag_hi: 1.0 }
        }
    }
}

/// Replay the movie's own pads (stick included) for the horizon.
fn replay_ref(game: &mut SM64Game, base: &SaveState, start: &St, pads: &[GamePad], sc: &mut dyn Scorer, h: usize) -> Result<f64> {
    game.load_state(base)?;
    sc.start(start);
    for t in 0..h {
        game.step_game(pads.get(t).copied().unwrap_or(GamePad::new(0, 0, 0)))?;
        let st = read_state(game, start.frame + t as u32 + 1, false);
        if !sc.step(t, &st) {
            break;
        }
    }
    Ok(sc.finish())
}

fn algos() -> Vec<Algo> {
    vec![
        Algo::Random,
        Algo::Cem { pop: 12, elite: 0.25, smooth: 0.3, sigma0: 0.5 },
        Algo::Es { lambda: 6, sigma0: 0.4 },
        Algo::Cma { pop: 8, sigma0: 0.4 },
        Algo::Spsa { a: 0.5, c: 0.15 },
        Algo::Fd { step: 0.3, eps: 0.08 },
        Algo::Grid { n_ang: 9, n_mag: 5 },
    ]
}

fn arg<T: std::str::FromStr>(a: &[String], i: usize, d: T) -> T {
    a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn kinds_for(problem: &str) -> Vec<Kind> {
    match problem {
        "speed" | "speed2" => vec![Kind::Steer, Kind::Hold, Kind::Land, Kind::Task],
        _ => vec![Kind::Steer, Kind::Hold, Kind::Land, Kind::Task],
    }
}

fn main() -> Result<()> {
    unsafe {
        std::env::set_var("OBRRT_LAKITU_ARC", "1");
        std::env::set_var("OBRRT_GROUNDS_FIX", "1");
    }
    obrrt_objectives::moves::install_controller();
    let a: Vec<String> = std::env::args().collect();
    let gg = SM64GameGenerator::from_file(&a[1])?;
    let cmd = a[2].as_str();
    let pname = a.get(3).map_or("speed", |s| s.as_str());
    let pb = load(pname, ".", &gg, false)?;
    let threads: usize = arg(&a, 4, 4);
    let mut games: Vec<SM64Game> = (0..threads).map(|_| gg.create_game()).collect::<Result<_>>()?;
    match cmd {
        "info" => info(&mut games[0], &pb),
        "curve" => curve(&mut games, &pb, pname, arg(&a, 5, 3), arg(&a, 6, 24), arg(&a, 7, 256), arg(&a, 8, 3)),
        "smooth" => smooth(&mut games, &pb, pname),
        "loop" => closed_loop(&mut games, &pb, pname, arg(&a, 5, 48)),
        _ => anyhow::bail!("info|curve|smooth|loop"),
    }
}

// --------------------------------------------------------------------- info

fn info(game: &mut SM64Game, pb: &Problem) -> Result<()> {
    println!("problem {} cases {} horizon {}", pb.name, pb.cases.len(), pb.roll.horizon);
    // timing
    let c = &pb.cases[0];
    game.load_state(&c.save)?;
    let t0 = Instant::now();
    let n = 20000;
    for i in 0..n {
        game.step_game(GamePad::new(0, 0, 0))?;
        if i % 500 == 499 {
            game.load_state(&c.save)?;
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    println!("step_game: {:.0} frames/s ({:.1} us)", n as f64 / dt, dt / n as f64 * 1e6);
    let t0 = Instant::now();
    for _ in 0..2000 {
        let _ = read_state(game, 1, false);
    }
    println!("read_state: {:.1} us", t0.elapsed().as_secs_f64() / 2000.0 * 1e6);
    let t0 = Instant::now();
    let mut sv = game.save_state();
    for _ in 0..200 {
        sv = game.save_state();
    }
    println!("save_state: {:.0} us", t0.elapsed().as_secs_f64() / 200.0 * 1e6);
    let t0 = Instant::now();
    for _ in 0..200 {
        game.load_state(&sv)?;
    }
    println!("load_state: {:.0} us", t0.elapsed().as_secs_f64() / 200.0 * 1e6);
    for (ci, c) in pb.cases.iter().enumerate().take(12) {
        let rs = ref_sim(game, pb, ci, 90)?;
        let mut runs: Vec<(u32, usize, f32)> = vec![];
        for s in &rs.traj {
            match runs.last_mut() {
                Some(r) if r.0 == s.action => r.1 += 1,
                _ => runs.push((s.action, 1, s.forward_vel)),
            }
        }
        let txt: Vec<String> = runs.iter().take(8).map(|r| format!("{:#x}x{}@{:.0}", r.0 & 0xfff, r.1, r.2)).collect();
        println!("{ci:>3} {} fv{:.0} {}", c.name, c.start.forward_vel, txt.join(" "));
    }
    Ok(())
}

// -------------------------------------------------------------------- curve

#[derive(Default, Clone)]
struct Agg {
    n: usize,
    /// improvement over x0 (zero plan) at each checkpoint
    gain: Vec<f64>,
    /// improvement over the movie's own stick
    vs_ref: Vec<f64>,
    frames: f64,
    secs: f64,
}

const CKPT: [u64; 5] = [16, 32, 64, 128, 256];

fn at(trace: &[(u64, f64)], n: u64) -> f64 {
    trace.iter().take_while(|t| t.0 <= n).last().map_or(f64::NEG_INFINITY, |t| t.1)
}

fn curve(games: &mut [SM64Game], pb: &Problem, pname: &str, knots: usize, h: usize, evals: usize, seeds: u64) -> Result<()> {
    let mut report = String::new();
    for kind in kinds_for(pname) {
        let insts = pick_insts(&mut games[0], pb, kind, h, 2, 8)?;
        if insts.is_empty() {
            writeln!(report, "== {kind:?}: no instances")?;
            continue;
        }
        let al = algos();
        // job = (inst, algo, seed)
        let jobs: Vec<(usize, usize, u64)> = (0..insts.len()).flat_map(|i| (0..al.len()).flat_map(move |a| (0..seeds).map(move |s| (i, a, s)))).collect();
        let t_all = Instant::now();
        let res = par(games, jobs.len(), |game, j| {
            let (ii, ai, seed) = jobs[j];
            let inst = &insts[ii];
            let rs = ref_sim(game, pb, inst.case, inst.off + h + 2)?;
            let base = base_at(game, pb, inst.case, &rs, inst.off)?;
            let start = rs.traj[inst.off].clone();
            let mut bt = build(pb, inst, &start, h);
            let refv = replay_ref(game, &base, &start, &rs.pads[inst.off..], bt.scorer.as_mut(), h)?;
            let pads_ref: Vec<GamePad> = rs.pads[inst.off..].to_vec();
            let btn = move |t: usize, _: &St, _: GamePad| pads_ref.get(t).map_or(0, |p| p.button);
            let prob = Prob { aim: bt.aim.clone(), horizon: h, knots, ang_max: bt.ang_max, mag_lo: bt.mag_lo, mag_hi: bt.mag_hi, fix_mag: None, buttons: &btn, last_pad: GamePad::new(0, 0, 0), scorer: bt.scorer.as_mut() };
            let mut ev = Evaluator::new(game, &base, start.clone(), prob);
            let mut rng = SmallRng::seed_from_u64(seed * 1009 + ii as u64);
            let t0 = Instant::now();
            let x0 = vec![0.0; ev.prob.dim()];
            let r = optimize(&al[ai], &mut ev, &x0, Budget { evals, frames: 0 }, &mut rng)?;
            Ok((ii, ai, r.x0_value, refv, r.trace, r.cost.frames, t0.elapsed().as_secs_f64()))
        })?;
        let mut agg = vec![Agg::default(); al.len()];
        for (_, ai, x0v, refv, trace, frames, secs) in &res {
            let g = &mut agg[*ai];
            g.n += 1;
            if g.gain.is_empty() {
                g.gain = vec![0.0; CKPT.len()];
                g.vs_ref = vec![0.0; CKPT.len()];
            }
            for (k, c) in CKPT.iter().enumerate() {
                let v = at(trace, *c);
                g.gain[k] += (v - x0v).max(0.0);
                g.vs_ref[k] += v - refv;
            }
            g.frames += *frames as f64;
            g.secs += secs;
        }
        let ref_mean: f64 = res.iter().map(|r| r.3).sum::<f64>() / res.len() as f64;
        let x0_mean: f64 = res.iter().map(|r| r.2).sum::<f64>() / res.len() as f64;
        writeln!(report, "\n== {kind:?} {pname} knots {knots} horizon {h}: {} starts x {} seeds ({:.0}s); mean value of x0 (aim, mid tilt) {:.1}, of the movie's own stick {:.1}", insts.len(), seeds, t_all.elapsed().as_secs_f64(), x0_mean, ref_mean)?;
        writeln!(report, "{:<7} | mean gain over x0 at evals {:?} | mean (value - movie) at same | frames/run  s/run", "algo", CKPT)?;
        for (i, g) in agg.iter().enumerate() {
            let n = g.n.max(1) as f64;
            let gs: Vec<String> = g.gain.iter().map(|v| format!("{:>7.1}", v / n)).collect();
            let rs: Vec<String> = g.vs_ref.iter().map(|v| format!("{:>7.1}", v / n)).collect();
            writeln!(report, "{:<7} | {} | {} | {:>9.0} {:>5.2}", al[i].name(), gs.join(" "), rs.join(" "), g.frames / n, g.secs / n)?;
        }
        print!("{report}");
        report.clear();
    }
    Ok(())
}

// ------------------------------------------------------------------- smooth

fn smooth(games: &mut [SM64Game], pb: &Problem, pname: &str) -> Result<()> {
    let h = 24;
    for kind in kinds_for(pname) {
        let insts = pick_insts(&mut games[0], pb, kind, h, 2, 8)?;
        let res = par(games, insts.len(), |game, ii| {
            let inst = &insts[ii];
            let rs = ref_sim(game, pb, inst.case, inst.off + h + 2)?;
            let base = base_at(game, pb, inst.case, &rs, inst.off)?;
            let start = rs.traj[inst.off].clone();
            let mut bt = build(pb, inst, &start, h);
            let pads_ref: Vec<GamePad> = rs.pads[inst.off..].to_vec();
            let btn = move |t: usize, _: &St, _: GamePad| pads_ref.get(t).map_or(0, |p| p.button);
            let prob = Prob { aim: bt.aim.clone(), horizon: h, knots: 1, ang_max: bt.ang_max, mag_lo: bt.mag_lo, mag_hi: bt.mag_hi, fix_mag: None, buttons: &btn, last_pad: GamePad::new(0, 0, 0), scorer: bt.scorer.as_mut() };
            let mag_lo = bt.mag_lo;
            let mag_hi = bt.mag_hi;
            let mut ev = Evaluator::new(game, &base, start, prob);
            let mut rng = SmallRng::seed_from_u64(7 + ii as u64);
            let x0 = vec![0.0f32, 0.0];
            let f0 = ev.eval(&x0)?;
            let scale = f0.abs().max(10.0);
            let mut line = String::new();
            for eps in [0.03f32, 0.1] {
                let s = smoothness(&mut ev, &x0, eps, 40, scale * 1e-4, scale * 0.05, &mut rng)?;
                write!(line, " eps{eps}: flat {:.2} smooth {:.2} cliff {:.2} med|d| {:.2};", s.frac_flat, s.frac_smooth, s.frac_cliff, s.median_abs_delta)?;
            }
            // axis scans: angle (dim 0) and magnitude (dim 1)
            let sa = scan_axis(&mut ev, &x0, 0, 61)?;
            let sm = scan_axis(&mut ev, &x0, 1, 61)?;
            let ca = cliffs(&sa, scale * 0.05);
            let cm = cliffs(&sm, scale * 0.05);
            let tilt = |u: f32| mag_lo + (u * 0.5 + 0.5) * (mag_hi - mag_lo);
            let fmt_c = |c: &[(f32, f64)], f: &dyn Fn(f32) -> f32| c.iter().map(|(u, d)| format!("{:.3}({:+.0})", f(*u), d)).collect::<Vec<_>>().join(" ");
            let range = |s: &[(f32, f64)]| {
                let (lo, hi) = s.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.1), a.1.max(p.1)));
                (lo, hi)
            };
            let (alo, ahi) = range(&sa);
            let (mlo, mhi) = range(&sm);
            Ok(format!(
                "{}\n   f0 {:.1};{}\n   angle scan [{:.1}..{:.1}] cliffs(angle offset rad) {}\n   tilt scan [{:.1}..{:.1}] cliffs(tilt) {}",
                insts[ii].name,
                f0,
                line,
                alo,
                ahi,
                fmt_c(&ca, &|u| u * 1.4),
                mlo,
                mhi,
                fmt_c(&cm, &tilt)
            ))
        })?;
        println!("\n== smoothness {kind:?} {pname}: random-direction steps (flat = |d| <= 1e-4*scale, cliff = |d| > 5%*scale), axis scans of 61 points");
        for r in res {
            println!("{r}");
        }
    }
    Ok(())
}

// --------------------------------------------------------------------- loop

struct LoopCfg {
    name: &'static str,
    algo: Algo,
    horizon: usize,
    knots: usize,
    evals: usize,
    replan: usize,
}

fn closed_loop(games: &mut [SM64Game], pb: &Problem, pname: &str, frames: usize) -> Result<()> {
    let cfgs: Vec<LoopCfg> = vec![
        LoopCfg { name: "cma h12 r1", algo: Algo::Cma { pop: 8, sigma0: 0.4 }, horizon: 12, knots: 2, evals: 32, replan: 1 },
        LoopCfg { name: "cma h12 r4", algo: Algo::Cma { pop: 8, sigma0: 0.4 }, horizon: 12, knots: 2, evals: 32, replan: 4 },
        LoopCfg { name: "cma h24 r1", algo: Algo::Cma { pop: 8, sigma0: 0.4 }, horizon: 24, knots: 3, evals: 32, replan: 1 },
        LoopCfg { name: "cma h24 r4", algo: Algo::Cma { pop: 8, sigma0: 0.4 }, horizon: 24, knots: 3, evals: 32, replan: 4 },
        LoopCfg { name: "cma h24 r8", algo: Algo::Cma { pop: 8, sigma0: 0.4 }, horizon: 24, knots: 3, evals: 48, replan: 8 },
        LoopCfg { name: "cma h48 r8", algo: Algo::Cma { pop: 8, sigma0: 0.4 }, horizon: 48, knots: 4, evals: 48, replan: 8 },
        LoopCfg { name: "cem h24 r4", algo: Algo::Cem { pop: 12, elite: 0.25, smooth: 0.3, sigma0: 0.5 }, horizon: 24, knots: 3, evals: 36, replan: 4 },
        LoopCfg { name: "es  h24 r4", algo: Algo::Es { lambda: 6, sigma0: 0.4 }, horizon: 24, knots: 3, evals: 36, replan: 4 },
        LoopCfg { name: "rnd h24 r4", algo: Algo::Random, horizon: 24, knots: 3, evals: 36, replan: 4 },
        LoopCfg { name: "grid h24 r4", algo: Algo::Grid { n_ang: 7, n_mag: 3 }, horizon: 24, knots: 1, evals: 36, replan: 4 },
        LoopCfg { name: "fd  h24 r4", algo: Algo::Fd { step: 0.3, eps: 0.08 }, horizon: 24, knots: 3, evals: 36, replan: 4 },
        LoopCfg { name: "spsa h24 r4", algo: Algo::Spsa { a: 0.5, c: 0.15 }, horizon: 24, knots: 3, evals: 36, replan: 4 },
    ];
    for kind in kinds_for(pname) {
        if kind == Kind::Land {
            continue;
        }
        let insts = pick_insts(&mut games[0], pb, kind, 24, 1, 6)?;
        // job = (inst, cfg index or usize::MAX for the movie replay, usize::MAX-1 for fixed aim)
        let jobs: Vec<(usize, usize)> = (0..insts.len()).flat_map(|i| (0..cfgs.len() + 2).map(move |c| (i, c))).collect();
        let res = par(games, jobs.len(), |game, j| {
            let (ii, ci) = jobs[j];
            let inst = &insts[ii];
            let rs = ref_sim(game, pb, inst.case, inst.off + frames + 2)?;
            let base0 = base_at(game, pb, inst.case, &rs, inst.off)?;
            let start0 = rs.traj[inst.off].clone();
            let pads_ref: Vec<GamePad> = rs.pads[inst.off..].to_vec();
            let h0 = 24;
            let mut bt = build(pb, inst, &start0, h0.max(frames));
            let t0 = Instant::now();
            game.load_state(&base0)?;
            bt.scorer.start(&start0);
            let mut st = start0.clone();
            let mut last = GamePad::new(0, 0, 0);
            let mut spent = 0u64;
            let mut rng = SmallRng::seed_from_u64(31 + ii as u64);
            let mut refiner = None;
            if ci < cfgs.len() {
                let c = &cfgs[ci];
                refiner = Some(Refiner::new(c.algo.clone(), Budget { evals: c.evals, frames: 0 }, c.replan));
            }
            let mut gain_sum = 0.0;
            let mut nrep = 0;
            for t in 0..frames {
                let pad = if ci == cfgs.len() {
                    // the movie's own stick
                    pads_ref.get(t).copied().unwrap_or(GamePad::new(0, 0, 0))
                } else {
                    let btn_bits = pads_ref.get(t).map_or(0, |p| p.button);
                    if ci == cfgs.len() + 1 {
                        // fixed aim at mid tilt (the x0 plan)
                        let spec = Spec { aim: bt.aim.clone(), horizon: 24, knots: 1, ang_max: bt.ang_max, mag_lo: bt.mag_lo, mag_hi: bt.mag_hi, fix_mag: None };
                        let (x, y) = spec.stick_for(&st, 0.0, 0.5);
                        GamePad::new(btn_bits, x, y)
                    } else {
                        let c = &cfgs[ci];
                        let rf = refiner.as_mut().unwrap();
                        let spec = Spec { aim: bt.aim.clone(), horizon: c.horizon, knots: c.knots, ang_max: bt.ang_max, mag_lo: bt.mag_lo, mag_hi: bt.mag_hi, fix_mag: None };
                        let dec = if rf.will_replan() {
                            let cur = game.save_state();
                            // the plan's buttons: the movie's from here on
                            let pr = pads_ref.clone();
                            let tt = t;
                            let btn = move |k: usize, _: &St, _: GamePad| pr.get(tt + k).map_or(0, |p| p.button);
                            // a fresh scorer of the same kind for the plan's rollouts
                            let mut sc = build(pb, inst, &st, c.horizon.max(24));
                            let prob = Prob { aim: bt.aim.clone(), horizon: c.horizon, knots: c.knots, ang_max: bt.ang_max, mag_lo: bt.mag_lo, mag_hi: bt.mag_hi, fix_mag: None, buttons: &btn, last_pad: last, scorer: sc.scorer.as_mut() };
                            let mut ev = Evaluator::new(game, &cur, st.clone(), prob);
                            let d = rf.decide(&mut ev, &mut rng)?;
                            spent += d.cost.frames;
                            gain_sum += d.gain;
                            nrep += 1;
                            // restore the real game
                            game.load_state(&cur)?;
                            d
                        } else {
                            rf.replay(&spec)
                        };
                        let (x, y) = spec.stick_for(&st, dec.fine_ang, dec.fine_mag);
                        GamePad::new(btn_bits, x, y)
                    }
                };
                game.step_game(pad)?;
                last = pad;
                st = read_state(game, start0.frame + t as u32 + 1, false);
                if !bt.scorer.step(t, &st) {
                    break;
                }
            }
            let v = bt.scorer.finish();
            Ok((ii, ci, v, spent, t0.elapsed().as_secs_f64(), if nrep > 0 { gain_sum / nrep as f64 } else { 0.0 }))
        })?;
        println!("\n== closed loop {kind:?} {pname}: {} starts, {frames} frames each; mean value over the starts (higher is better)", insts.len());
        println!("{:<13} {:>10} {:>14} {:>7} {:>9}", "controller", "mean value", "emu frames/run", "s/run", "plan gain");
        for ci in 0..cfgs.len() + 2 {
            let rows: Vec<_> = res.iter().filter(|r| r.1 == ci).collect();
            let n = rows.len().max(1) as f64;
            let name = if ci < cfgs.len() { cfgs[ci].name } else if ci == cfgs.len() { "movie stick" } else { "aim+mid tilt" };
            println!("{:<13} {:>10.1} {:>14.0} {:>7.2} {:>9.2}", name, rows.iter().map(|r| r.2).sum::<f64>() / n, rows.iter().map(|r| r.3 as f64).sum::<f64>() / n, rows.iter().map(|r| r.4).sum::<f64>() / n, rows.iter().map(|r| r.5).sum::<f64>() / n);
        }
    }
    Ok(())
}
