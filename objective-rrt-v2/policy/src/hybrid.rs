//! Hybrid per-start search: a GA over open-loop pad sequences that is seeded and
//! guided by the policy inventory.
//!
//! One search per (start, seed), exact emulator-frame accounting, anytime: the
//! best result is snapshotted at budget checkpoints, so one 0.56 M run gives
//! the 0.02 / 0.05 / 0.15 / 0.56 M points.
//!
//! Variants (see [`Variant`]):
//! - `Pink`    : the same GA with a pink-noise population (the equal-machinery baseline)
//! - `Seeded`  : (a) generation 0 holds the realised pads of rolling out the inventory policies (+ lambda draws)
//! - `Repolicy`: (b) + an operator that keeps a parent's pads up to a random frame and then lets a policy drive reactively
//! - `Market`  : (c) + operator probabilities adapted online by credit (score gained per frame spent)
//! - `Polish`  : (d) + a trim operator on the best individual (delete one frame, carry the rest: the mend idea)
//! - `PolicyOnly`: every candidate policy once per start, no GA (the cost of "try the inventory")

use anyhow::Result;
use obrrt_core::{GamePad, St, read_state};
use obrrt_gp::problems::Problem;
use obrrt_gp::rollout::{Case, PREFIX, PadsCtrl, Roll, Task, par, play};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sm64_binds::SM64Game;
use std::collections::HashMap;
use std::path::Path;

use crate::dsl::{self, C, Pol};
use crate::eval::{self, Env, Instance, Resolver};
use crate::inv::{draw_instance, instance_of};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Pink,
    Seeded,
    Repolicy,
    Market,
    Polish,
    PolicyOnly,
}

impl Variant {
    pub fn parse(s: &str) -> Option<Variant> {
        Some(match s {
            "pink" | "V0" => Variant::Pink,
            "seeded" | "V1" => Variant::Seeded,
            "repolicy" | "V2" => Variant::Repolicy,
            "market" | "V3" => Variant::Market,
            "polish" | "V4" => Variant::Polish,
            "policy" | "P" => Variant::PolicyOnly,
            _ => return None,
        })
    }
    fn seeded(self) -> bool {
        !matches!(self, Variant::Pink)
    }
    fn repolicy(self) -> bool {
        matches!(self, Variant::Repolicy | Variant::Market | Variant::Polish)
    }
    fn market(self) -> bool {
        matches!(self, Variant::Market | Variant::Polish)
    }
    fn trim(self) -> bool {
        matches!(self, Variant::Polish)
    }
}

/// The policies available to the search, by name.
pub struct Library {
    pub bodies: HashMap<String, Pol>,
}

impl Resolver for Library {
    fn body(&self, name: &str) -> Option<(&C, Instance)> {
        self.bodies.get(name).map(|p| (&p.body, instance_of(p)))
    }
}

impl Library {
    /// Every `*.pol` in the given directories, by policy name.
    pub fn load(dirs: &[&Path]) -> Library {
        let mut bodies = HashMap::new();
        for d in dirs {
            let Ok(rd) = std::fs::read_dir(d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "pol") {
                    if let Ok(t) = std::fs::read_to_string(&p) {
                        if let Ok(pol) = dsl::parse(&t) {
                            // by file stem too: names collide across authors (two X00005s)
                            let stem = p.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                            bodies.insert(stem, pol.clone());
                            bodies.entry(pol.name.clone()).or_insert(pol);
                        }
                    }
                }
            }
        }
        Library { bodies }
    }
}

/// One candidate of the search: open-loop pads, optionally followed by a policy driving reactively.
#[derive(Clone)]
struct Ind {
    pads: Vec<GamePad>,
    tail: Option<(String, Instance)>,
    score: f64,
    done: Option<usize>,
    origin: u8,
}

const OPS: [&str; 9] = ["pink", "jitter", "button", "dropins", "resplice", "cross", "repolicy", "trim", "policy-draw"];

fn score(done: Option<usize>, best_eta: f64) -> f64 {
    match done {
        Some(d) => 1e6 - d as f64,
        None => -best_eta,
    }
}

/// The case with `pads` already played (the policy tail then starts from there).
fn with_prefix(g: &mut SM64Game, case: &Case, pads: &[GamePad]) -> Result<Case> {
    g.load_state(&case.save)?;
    for p in &case.noise {
        g.step_game(*p)?;
    }
    let mut states: Vec<St> = case.pre_states.clone();
    let mut all_pads: Vec<GamePad> = case.pre_pads.clone();
    for (i, p) in pads.iter().enumerate() {
        g.step_game(*p)?;
        states.push(read_state(g, case.frame as u32 + i as u32 + 1, false));
        all_pads.push(*p);
    }
    let skip_s = states.len().saturating_sub(PREFIX + 1);
    let skip_p = all_pads.len().saturating_sub(PREFIX);
    let mut noise = case.noise.clone();
    noise.extend_from_slice(pads);
    Ok(Case {
        name: case.name.clone(),
        train: case.train,
        val: case.val,
        movie: case.movie,
        frame: case.frame + pads.len(),
        noise,
        save: case.save.clone(),
        pre_states: states[skip_s..].to_vec(),
        pre_pads: all_pads[skip_p..].to_vec(),
        start: case.start.clone(),
        task: case.task,
        ref_pads: case.ref_pads.clone(),
    })
}

struct Ctx<'a> {
    case: &'a Case,
    task: &'a dyn Task,
    roll: Roll,
    lib: &'a Library,
    env: &'a Env,
}

/// Plays an individual; returns (realised pads, done frames in total, best eta, emulator frames spent).
fn run_ind(g: &mut SM64Game, cx: &Ctx, pads: &[GamePad], tail: &Option<(String, Instance)>) -> Result<(Vec<GamePad>, Option<usize>, f64, u64)> {
    match tail {
        None => {
            let mut c = PadsCtrl { pads, at: 0 };
            let o = play(g, cx.case, cx.task, &mut c, &cx.roll, true)?;
            Ok((o.pads, o.done, o.best_eta, o.frames))
        }
        Some((name, inst)) => {
            let Some(pol) = cx.lib.bodies.get(name) else {
                let mut c = PadsCtrl { pads, at: 0 };
                let o = play(g, cx.case, cx.task, &mut c, &cx.roll, true)?;
                return Ok((o.pads, o.done, o.best_eta, o.frames));
            };
            let mut roll = cx.roll;
            let horizon = cx.task.horizon().unwrap_or(roll.horizon);
            roll.horizon = horizon.saturating_sub(pads.len()).max(10);
            let dc = if pads.is_empty() { None } else { Some(with_prefix(g, cx.case, pads)?) };
            let c = dc.as_ref().unwrap_or(cx.case);
            let pl = eval::play_policy(g, c, cx.task, &pol.body, cx.lib, cx.env, &roll, true, inst)?;
            let mut all = pads.to_vec();
            all.extend_from_slice(&pl.out.pads);
            Ok((all, pl.out.done.map(|d| d + pads.len()), pl.out.best_eta, pads.len() as u64 + pl.out.frames))
        }
    }
}

fn stick_noise(r: &mut SmallRng, sigma: f32) -> i8 {
    let g = (r.random::<f32>() + r.random::<f32>() + r.random::<f32>() - 1.5) * 2.0 * sigma;
    g.round().clamp(-80.0, 80.0) as i8
}

fn clamp8(x: i32) -> i8 {
    x.clamp(-80, 80) as i8
}

fn pink(r: &mut SmallRng, n: usize) -> Vec<GamePad> {
    obrrt_search::noise::pink_inputs(r, n)
}

/// Pads of a parent, made `len` long (extended with pink noise).
fn fit(mut p: Vec<GamePad>, len: usize, r: &mut SmallRng) -> Vec<GamePad> {
    if p.len() < len {
        let extra = pink(r, len - p.len());
        p.extend(extra);
    }
    p.truncate(len);
    p
}

const BTNS: [u16; 3] = [obrrt_core::buttons::A_BUTTON, obrrt_core::buttons::B_BUTTON, obrrt_core::buttons::Z_TRIG];

fn mutate(op: usize, pads: &[GamePad], other: &[GamePad], h: usize, r: &mut SmallRng) -> Vec<GamePad> {
    let mut p = fit(pads.to_vec(), h.max(pads.len().min(h)), r);
    let n = p.len();
    match op {
        0 => pink(r, h),
        1 => {
            // stick jitter in a window
            let w = r.random_range(3..=20).min(n);
            let a = r.random_range(0..=n - w);
            let (sx, sy) = (stick_noise(r, 25.0) as i32, stick_noise(r, 25.0) as i32);
            for q in &mut p[a..a + w] {
                q.stick_x = clamp8(q.stick_x as i32 + sx);
                q.stick_y = clamp8(q.stick_y as i32 + sy);
            }
            p
        }
        2 => {
            // set or clear a button over a span
            let w = r.random_range(1..=12).min(n);
            let a = r.random_range(0..=n - w);
            let b = BTNS[r.random_range(0..3)];
            let on = r.random_bool(0.5);
            for q in &mut p[a..a + w] {
                if on {
                    q.button |= b
                } else {
                    q.button &= !b
                }
            }
            p
        }
        3 => {
            // drop or duplicate a few frames (time shift)
            let k = r.random_range(1..=4).min(n);
            let a = r.random_range(0..=n - k);
            if r.random_bool(0.5) {
                p.drain(a..a + k);
            } else {
                let seg: Vec<GamePad> = p[a..a + k].to_vec();
                for (i, q) in seg.into_iter().enumerate() {
                    p.insert(a + i, q);
                }
            }
            fit(p, h, r)
        }
        4 => {
            // keep a prefix, refill with pink noise
            let a = r.random_range(0..n);
            p.truncate(a);
            let extra = pink(r, h - a.min(h));
            p.extend(extra);
            p
        }
        5 => {
            // crossover: splice with another parent
            let o = fit(other.to_vec(), h, r);
            let a = r.random_range(0..n.min(o.len()));
            let mut c = p[..a].to_vec();
            c.extend_from_slice(&o[a..]);
            c
        }
        _ => p,
    }
}

/// One search's result at the budget checkpoints.
#[derive(Clone, Debug, Default)]
pub struct Snap {
    pub budget: u64,
    pub done: Option<usize>,
    pub emu: u64,
}

pub struct Params {
    pub variant: Variant,
    pub budgets: Vec<u64>,
    pub pop: usize,
    pub cands: Vec<String>,
    pub draws: usize,
}

/// A policy-seeded search from one start. Returns one snapshot per budget checkpoint.
fn search(g: &mut SM64Game, cx: &Ctx, p: &Params, seed: u64) -> Result<(Vec<Snap>, Vec<(String, f64)>, Option<Vec<GamePad>>)> {
    let mut r = SmallRng::seed_from_u64(seed);
    let total = *p.budgets.last().unwrap();
    let h = cx.task.horizon().unwrap_or(cx.roll.horizon).min(300);
    let mut spent: u64 = 0;
    let mut snaps: Vec<Snap> = Vec::new();
    let mut next_snap = 0usize;
    let mut best_done: Option<usize> = None;
    let mut best_pads: Option<Vec<GamePad>> = None; // clean validation: the best finished route
    let mut pop: Vec<Ind> = Vec::new();
    let mut policy_scores: Vec<(String, f64)> = Vec::new();

    macro_rules! eval_push {
        ($pads:expr, $tail:expr, $origin:expr) => {{
            let (realised, done, beta, fr) = run_ind(g, cx, $pads, $tail)?;
            spent += fr;
            if let Some(d) = done {
                if best_done.is_none_or(|b| d < b) {
                    best_pads = Some(realised.clone());
                }
                best_done = Some(best_done.map_or(d, |b| b.min(d)));
            }
            let ind = Ind { pads: realised, tail: None, score: score(done, beta), done, origin: $origin };
            while next_snap < p.budgets.len() && spent >= p.budgets[next_snap] {
                snaps.push(Snap { budget: p.budgets[next_snap], done: best_done, emu: spent });
                next_snap += 1;
            }
            ind
        }};
    }

    // generation 0
    if p.variant.seeded() {
        for name in &p.cands {
            let Some(pol) = cx.lib.bodies.get(name) else { continue };
            let mut best = f64::NEG_INFINITY;
            for k in 0..=p.draws {
                let inst = if k == 0 { instance_of(pol) } else { draw_instance(pol, &mut r) };
                let ind = eval_push!(&[], &Some((name.clone(), inst)), 1);
                best = best.max(ind.score);
                pop.push(ind);
            }
            policy_scores.push((name.clone(), best));
        }
    }
    if p.variant == Variant::PolicyOnly {
        while next_snap < p.budgets.len() {
            snaps.push(Snap { budget: p.budgets[next_snap], done: best_done, emu: spent });
            next_snap += 1;
        }
        return Ok((snaps, policy_scores, best_pads));
    }
    // top up with pink noise (and pink mutants of the seeds)
    while pop.len() < p.pop {
        if !pop.is_empty() && p.variant.seeded() && r.random_bool(0.5) {
            let a = pop[r.random_range(0..pop.len())].pads.clone();
            let op = r.random_range(1..5);
            let c = mutate(op, &a, &a, h, &mut r);
            let ind = eval_push!(&c, &None, 2);
            pop.push(ind);
        } else {
            let c = pink(&mut r, h);
            let ind = eval_push!(&c, &None, 0);
            pop.push(ind);
        }
        if spent >= total {
            break;
        }
    }
    // operator weights
    let nops = OPS.len();
    let mut active = vec![true; nops];
    active[6] = p.variant.repolicy() && !p.cands.is_empty();
    active[7] = p.variant.trim();
    active[8] = false;
    let mut credit = vec![1.0f64; nops];
    let pick_op = |credit: &[f64], active: &[bool], market: bool, r: &mut SmallRng| -> usize {
        let idx: Vec<usize> = (0..nops).filter(|&i| active[i] && i != 0).collect();
        let pmin = 0.04;
        let w: Vec<f64> = if market {
            let s: f64 = idx.iter().map(|&i| credit[i]).sum::<f64>().max(1e-9);
            idx.iter().map(|&i| pmin + (1.0 - pmin * idx.len() as f64).max(0.0) * credit[i] / s).collect()
        } else {
            vec![1.0; idx.len()]
        };
        let tot: f64 = w.iter().sum();
        let mut x = r.random::<f64>() * tot;
        for (k, &i) in idx.iter().enumerate() {
            x -= w[k];
            if x <= 0.0 {
                return i;
            }
        }
        *idx.last().unwrap()
    };
    let tournament = |pop: &[Ind], r: &mut SmallRng| -> usize {
        let mut b = r.random_range(0..pop.len());
        for _ in 0..2 {
            let c = r.random_range(0..pop.len());
            if pop[c].score > pop[b].score {
                b = c;
            }
        }
        b
    };
    let mut op_count = vec![0u64; nops];
    while spent < total {
        let op = pick_op(&credit, &active, p.variant.market(), &mut r);
        op_count[op] += 1;
        let pa = tournament(&pop, &mut r);
        let before = spent;
        let best_i = (0..pop.len()).max_by(|&a, &b| pop[a].score.partial_cmp(&pop[b].score).unwrap()).unwrap();
        let (parent_score, child) = if op == 6 {
            // keep a parent's pads up to a random frame, then a policy drives reactively
            let par_ind = &pop[pa];
            let cut = if par_ind.pads.is_empty() { 0 } else { r.random_range(0..par_ind.pads.len()) };
            let prefix = par_ind.pads[..cut].to_vec();
            let name = if !policy_scores.is_empty() && r.random_bool(0.6) {
                // favour policies that did well from this start
                let mut v = policy_scores.clone();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                v[r.random_range(0..v.len().min(4))].0.clone()
            } else {
                p.cands[r.random_range(0..p.cands.len())].clone()
            };
            let pol = &cx.lib.bodies[&name];
            let inst = if r.random_bool(0.5) { draw_instance(pol, &mut r) } else { instance_of(pol) };
            (par_ind.score, eval_push!(&prefix, &Some((name, inst)), 3))
        } else if op == 7 {
            // trim: delete one frame of the best individual's realised pads, carry the rest
            let b = &pop[best_i];
            let mut c = b.pads.clone();
            if c.is_empty() {
                continue;
            }
            let i = r.random_range(0..c.len());
            c.remove(i);
            (b.score, eval_push!(&c, &None, 4))
        } else {
            let pb = tournament(&pop, &mut r);
            let c = mutate(op, &pop[pa].pads, &pop[pb].pads, h, &mut r);
            (pop[pa].score, eval_push!(&c, &None, 5))
        };
        let cost = (spent - before).max(1) as f64;
        let gain = (child.score - parent_score).max(0.0).min(300.0);
        credit[op] = 0.9 * credit[op] + 0.1 * gain / cost * 100.0 + 1e-6;
        // steady state: replace the worst if the child is at least as good
        let worst = (0..pop.len()).min_by(|&a, &b| pop[a].score.partial_cmp(&pop[b].score).unwrap()).unwrap();
        if child.score >= pop[worst].score {
            pop[worst] = child;
        }
    }
    while next_snap < p.budgets.len() {
        snaps.push(Snap { budget: p.budgets[next_snap], done: best_done, emu: spent });
        next_snap += 1;
    }
    Ok((snaps, policy_scores, best_pads))
}

/// Runs the variant on every (case, seed) of the problem.
pub fn run(games: &mut [SM64Game], pb: &Problem, lib: &Library, env: &Env, p: &Params, seeds: u64) -> Result<Vec<Vec<Vec<Snap>>>> {
    let n = pb.cases.len() * seeds as usize;
    let res = par(games, n, |g, i| {
        let (ci, seed) = (i % pb.cases.len(), (i / pb.cases.len()) as u64 + 1);
        let c = &pb.cases[ci];
        let cx = Ctx { case: c, task: pb.tasks[c.task].as_ref(), roll: pb.roll, lib, env };
        let (s, _, bp) = search(g, &cx, p, seed * 1000 + ci as u64)?;
        // clean validation: CLEAN_DUMP=dir writes the best finished route as raw inputs from power-on (+ an .idx line for gp/verify.py)
        if let (Ok(dir), Some(bp)) = (std::env::var("CLEAN_DUMP"), bp) {
            let done = s.last().and_then(|x| x.done);
            if let Some(d) = done {
                std::fs::create_dir_all(&dir)?;
                let mut pads = pb.movies[c.movie].1[..c.frame - c.noise.len()].to_vec();
                pads.extend_from_slice(&c.noise);
                pads.extend_from_slice(&bp);
                let file = format!("{dir}/{:?}-s{seed}-{}.raw", p.variant, c.name.replace(['/', ','], "_"));
                obrrt_gp::problems::write_raw(&file, &pads)?;
                std::fs::write(format!("{file}.idx"), format!("{file} {} {} {}\n", c.frame, d, bp.len()))?;
            }
        }
        Ok((ci, seed, s))
    })?;
    let mut out: Vec<Vec<Vec<Snap>>> = vec![vec![]; seeds as usize];
    for (ci, seed, s) in res {
        let v = &mut out[seed as usize - 1];
        if v.len() <= ci {
            v.resize(ci + 1, vec![]);
        }
        v[ci] = s;
    }
    Ok(out)
}

/// Summary per budget checkpoint and split: (success, n, mean frames-to-Done over successes, mean emu).
pub fn summarize(pb: &Problem, label: &str, rows: &[Vec<Vec<Snap>>], budgets: &[u64]) -> serde_json::Value {
    let mut js = serde_json::Map::new();
    for (bi, b) in budgets.iter().enumerate() {
        for (sp, tag) in [("tr", "train"), ("VA", "val"), ("TE", "test")] {
            let mut succ = 0;
            let mut n = 0;
            let mut fr = 0.0;
            let mut emu = 0.0;
            for seedrows in rows {
                for (c, s) in pb.cases.iter().zip(seedrows) {
                    if crate::bench::split(c) != sp {
                        continue;
                    }
                    n += 1;
                    if let Some(x) = s.get(bi) {
                        emu += x.emu as f64;
                        if let Some(d) = x.done {
                            succ += 1;
                            fr += d as f64;
                        }
                    }
                }
            }
            if n == 0 {
                continue;
            }
            let mf = fr / (succ.max(1) as f64);
            println!("{label} @{:>7}: {tag:<5} success {succ}/{n} mean frames-to-Done {mf:.1} mean emu/start {:.3} M", b, emu / n as f64 / 1e6);
            js.insert(format!("{b}-{tag}"), serde_json::json!({"n": n, "success": succ, "mean_frames": mf, "emu_per_start": emu / n as f64}));
        }
    }
    serde_json::Value::Object(js)
}

/// The v2 Worker GA (obrrt_search) seeded with the best policy rollout of the inventory: every candidate policy
/// is rolled out once from the start (counted), the best one's realised pads are the Worker's `seed`
/// (`attempt_seeded`: played as is, then variants with frames dropped or cut and steered).
#[cfg(feature = "ga")]
pub fn worker_seeded(
    gg: &sm64_binds::SM64GameGenerator,
    pb: &Problem,
    lib: &Library,
    env: &Env,
    cands: &[String],
    threads: usize,
    seeds: u64,
    scale: f64,
    pink: f64,
    seed_it: bool,
) -> Result<Vec<Vec<crate::bench::Res>>> {
    use obrrt_search::ga::{Fitness, GaParams, Worker};
    let params = GaParams { scale, fitness: Fitness::Steps, new_ops: true, pink, per_objective_budget: false };
    let n = pb.cases.len() * seeds as usize;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<crate::bench::Res>>> = (0..n).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                let mut w = Worker::new(gg).expect("worker");
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let (ci, seed) = (i % pb.cases.len(), (i / pb.cases.len()) as u64 + 1);
                    let c = &pb.cases[ci];
                    let task = pb.tasks[c.task].as_ref();
                    let obj = task.objective().expect("objective");
                    let cx = Ctx { case: c, task, roll: pb.roll, lib, env };
                    let mut r = SmallRng::seed_from_u64(seed * 1000 + ci as u64);
                    // the inventory's rollouts
                    let mut spent = 0u64;
                    let mut best: Option<(f64, Vec<GamePad>, Option<usize>)> = None;
                    if seed_it {
                        for name in cands {
                            let Some(pol) = lib.bodies.get(name) else { continue };
                            for k in 0..=1 {
                                let inst = if k == 0 { instance_of(pol) } else { draw_instance(pol, &mut r) };
                                let (pads, done, beta, fr) = run_ind(&mut w.base, &cx, &[], &Some((name.clone(), inst))).expect("rollout");
                                spent += fr;
                                let sc_ = score(done, beta);
                                if best.as_ref().is_none_or(|b| sc_ > b.0) {
                                    best = Some((sc_, pads, done));
                                }
                            }
                        }
                    }
                    w.base.load_state(&c.save).expect("load");
                    let st = read_state(&w.base, c.frame as u32, true);
                    let seed_pads = best.as_ref().map(|b| b.1.as_slice());
                    let att = w.attempt_seeded(&[obj.as_ref()], &st, &c.save, &mut r, params, seed_pads).expect("attempt");
                    let done = att.segments.first().filter(|s| s.success).map(|s| s.played);
                    // a policy rollout that is already Done counts even if the Worker's replay missed it
                    let done = match (done, best.as_ref().and_then(|b| b.2)) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (None, b) => b,
                        (a, None) => a,
                    };
                    *out[i].lock().unwrap() = Some(crate::bench::Res { done, emu: att.frames + spent });
                }
            });
        }
    });
    let flat: Vec<crate::bench::Res> = out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
    Ok(flat.chunks(pb.cases.len()).map(|c| c.to_vec()).collect())
}
