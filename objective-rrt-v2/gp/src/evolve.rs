//! Koza-style evolution: ramped half-and-half, tournaments, subtree crossover
//! and mutation, constant mutation, elitism, a parsimony term and hard size
//! limits. Deterministic from the seed: fitness comes back in index order
//! whatever the thread count, and a program's text memoises its result.

use std::collections::HashMap;
use std::io::Write;

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sm64_binds::SM64Game;

use crate::lang::{GenCfg, Node, Op, Prog, Ty, constant_points, perturb, random};
use crate::problems::Problem;
use crate::rollout::{Outcome, ProgCtrl, par, play};

#[derive(Clone, Debug)]
pub struct Params {
    pub pop: usize,
    pub gens: usize,
    pub tournament: usize,
    pub crossover: f64,
    pub mutation: f64,
    pub const_mut: f64,
    pub elite: usize,
    pub max_depth: usize,
    pub max_size: usize,
    pub init_depth: (usize, usize),
    /// Fitness lost per node.
    pub parsimony: f64,
    /// Share of the first generation that is mutants of the seed programs.
    pub seed_share: f64,
    pub seed: u64,
    /// Stops after this many emulator frames.
    pub budget: u64,
    /// Programs per generation scored on the validation cases.
    pub val_top: usize,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            pop: 500,
            gens: 40,
            tournament: 5,
            crossover: 0.8,
            mutation: 0.08,
            const_mut: 0.07,
            elite: 5,
            max_depth: 14,
            max_size: 300,
            init_depth: (3, 7),
            parsimony: 0.02,
            seed_share: 0.2,
            seed: 1,
            budget: u64::MAX,
            val_top: 8,
        }
    }
}

/// A program's score over the training cases.
#[derive(Clone, Debug, Default)]
pub struct Score {
    pub raw: f64,
    pub fit: f64,
    pub succ: usize,
    pub frames_done: f64,
    pub emu: u64,
}

fn new_prog(g: &GenCfg, depth: usize, full: bool, r: &mut SmallRng) -> Prog {
    Prog {
        main: random(Ty::Pad, depth, full, g, true, r),
        adf_p: (0..g.adf_p).map(|_| random(Ty::Pad, 4, false, g, false, r)).collect(),
        adf_b: (0..g.adf_b).map(|_| random(Ty::Bool, 3, false, g, false, r)).collect(),
    }
}

/// Picks a branch weighted by size: 0 main, then adf_p, then adf_b.
fn branch_mut(p: &mut Prog, i: usize) -> &mut Node {
    if i == 0 {
        &mut p.main
    } else if i <= p.adf_p.len() {
        &mut p.adf_p[i - 1]
    } else {
        let k = i - 1 - p.adf_p.len();
        &mut p.adf_b[k]
    }
}

fn branch(p: &Prog, i: usize) -> &Node {
    if i == 0 {
        &p.main
    } else if i <= p.adf_p.len() {
        &p.adf_p[i - 1]
    } else {
        &p.adf_b[i - 1 - p.adf_p.len()]
    }
}

fn pick_branch(p: &Prog, r: &mut SmallRng) -> usize {
    let n = 1 + p.adf_p.len() + p.adf_b.len();
    let total: usize = (0..n).map(|i| branch(p, i).size()).sum();
    let mut x = r.random_range(0..total);
    for i in 0..n {
        let s = branch(p, i).size();
        if x < s {
            return i;
        }
        x -= s;
    }
    0
}

fn pick_point(points: &[(Vec<u16>, Ty, bool)], r: &mut SmallRng) -> usize {
    // Koza: 90% internal points, when there are any.
    let internal: Vec<usize> = (0..points.len()).filter(|&i| points[i].2).collect();
    if !internal.is_empty() && r.random_bool(0.9) { internal[r.random_range(0..internal.len())] } else { r.random_range(0..points.len()) }
}

/// Subtree crossover: a child of `a` with a same-typed subtree of `b` (same branch).
fn crossover(a: &Prog, b: &Prog, r: &mut SmallRng) -> Option<Prog> {
    let br = pick_branch(a, r);
    let mut pa = Vec::new();
    branch(a, br).points(&mut Vec::new(), &mut pa);
    let ia = pick_point(&pa, r);
    let ty = pa[ia].1;
    let mut pb = Vec::new();
    branch(b, br).points(&mut Vec::new(), &mut pb);
    pb.retain(|x| x.1 == ty);
    if pb.is_empty() {
        return None;
    }
    let ib = pick_point(&pb, r);
    let mut c = a.clone();
    let donor = branch(b, br).at(&pb[ib].0).clone();
    *branch_mut(&mut c, br).at_mut(&pa[ia].0) = donor;
    Some(c)
}

fn mutate(a: &Prog, g: &GenCfg, r: &mut SmallRng) -> Prog {
    let br = pick_branch(a, r);
    let mut pa = Vec::new();
    branch(a, br).points(&mut Vec::new(), &mut pa);
    let ia = r.random_range(0..pa.len());
    let mut c = a.clone();
    let d = r.random_range(2..=5);
    *branch_mut(&mut c, br).at_mut(&pa[ia].0) = random(pa[ia].1, d, false, g, br == 0, r);
    c
}

fn mutate_const(a: &Prog, g: &GenCfg, r: &mut SmallRng) -> Prog {
    let mut c = a.clone();
    let n = 1 + r.random_range(0..3);
    for _ in 0..n {
        let br = pick_branch(&c, r);
        let mut pts = Vec::new();
        constant_points(branch(&c, br), &mut Vec::new(), &mut pts);
        if pts.is_empty() {
            continue;
        }
        let path = pts[r.random_range(0..pts.len())].clone();
        let node = branch_mut(&mut c, br).at_mut(&path);
        if !matches!(node.op, Op::CallP(_) | Op::CallB(_)) {
            perturb(&mut node.op, g, r);
        }
    }
    c
}

pub struct Run {
    /// Best-of-run: best on the validation cases among each generation's
    /// top programs by training fitness (without validation cases: on training).
    pub best: Prog,
    pub best_score: Score,
    /// Best on training fitness alone, for comparison.
    pub best_train: Prog,
    pub emu_total: u64,
    pub log: Vec<String>,
}

/// One program over some cases.
pub fn eval_cases(games: &mut [SM64Game], pb: &Problem, prog: &Prog, idx: &[usize], keep: bool) -> Result<Vec<Outcome>> {
    par(games, idx.len(), |g, i| {
        let c = &pb.cases[idx[i]];
        play(g, c, pb.tasks[c.task].as_ref(), &mut ProgCtrl { prog, trace: None }, &pb.roll, keep)
    })
}

/// Evaluates new programs (one task per (program, case)).
fn eval_pop(games: &mut [SM64Game], pb: &Problem, train: &[usize], progs: &[&Prog], params: &Params) -> Result<Vec<Score>> {
    let nc = train.len();
    let results = par(games, progs.len() * nc, |g, i| {
        let (p, c) = (progs[i / nc], &pb.cases[train[i % nc]]);
        play(g, c, pb.tasks[c.task].as_ref(), &mut ProgCtrl { prog: p, trace: None }, &pb.roll, false)
    })?;
    Ok(progs
        .iter()
        .enumerate()
        .map(|(pi, p)| {
            let rs = &results[pi * nc..(pi + 1) * nc];
            let raw = rs.iter().map(|o| o.fitness).sum::<f64>() / nc as f64;
            let succ = rs.iter().filter(|o| o.done.is_some()).count();
            let frames_done = if succ > 0 { rs.iter().filter_map(|o| o.done).sum::<usize>() as f64 / succ as f64 } else { 0.0 };
            Score { raw, fit: raw - params.parsimony * p.size() as f64, succ, frames_done, emu: rs.iter().map(|o| o.frames).sum() }
        })
        .collect())
}

pub fn evolve(games: &mut [SM64Game], pb: &Problem, params: &Params, seeds: &[Prog], log: &mut dyn Write) -> Result<Run> {
    let mut r = SmallRng::seed_from_u64(params.seed);
    let train: Vec<usize> = (0..pb.cases.len()).filter(|&i| pb.cases[i].train).collect();
    let val: Vec<usize> = (0..pb.cases.len()).filter(|&i| pb.cases[i].val).collect();
    let mut val_memo: HashMap<String, f64> = HashMap::new();
    let mut best_val: Option<(Prog, f64)> = None;
    let g = pb.gen_cfg;
    // Ramped half-and-half; the seed programs and their mutants first.
    let mut pop: Vec<Prog> = Vec::with_capacity(params.pop);
    pop.extend(seeds.iter().cloned());
    if !seeds.is_empty() {
        let n = (params.seed_share * params.pop as f64) as usize;
        for i in 0..n {
            let s = &seeds[i % seeds.len()];
            pop.push(if r.random_bool(0.5) { mutate_const(s, &g, &mut r) } else { mutate(s, &g, &mut r) });
        }
    }
    let (d0, d1) = params.init_depth;
    let mut i = 0;
    while pop.len() < params.pop {
        let depth = d0 + i % (d1 - d0 + 1);
        pop.push(new_prog(&g, depth, i % 2 == 0, &mut r));
        i += 1;
    }

    let mut memo: HashMap<String, Score> = HashMap::new();
    let mut emu_total = 0u64;
    let mut best: Option<(Prog, Score)> = None;
    let mut lines = Vec::new();
    let t0 = std::time::Instant::now();
    for gen_no in 0..=params.gens {
        // Fitness of the unseen programs, in order.
        let keys: Vec<String> = pop.iter().map(Prog::text).collect();
        let mut seen = std::collections::HashSet::new();
        let new: Vec<usize> = (0..pop.len()).filter(|&i| !memo.contains_key(&keys[i]) && seen.insert(keys[i].clone())).collect();
        let progs: Vec<&Prog> = new.iter().map(|&i| &pop[i]).collect();
        let scores = eval_pop(games, pb, &train, &progs, params)?;
        for (&i, s) in new.iter().zip(scores) {
            emu_total += s.emu;
            memo.insert(keys[i].clone(), s);
        }
        let fits: Vec<Score> = keys.iter().map(|k| memo[k].clone()).collect();
        let bi = (0..pop.len()).max_by(|&a, &b| fits[a].fit.total_cmp(&fits[b].fit)).unwrap();
        if best.as_ref().is_none_or(|(_, s)| fits[bi].fit > s.fit) {
            best = Some((pop[bi].clone(), fits[bi].clone()));
        }
        let mean = fits.iter().map(|s| s.raw).sum::<f64>() / fits.len() as f64;
        let msize = pop.iter().map(Prog::size).sum::<usize>() as f64 / pop.len() as f64;
        // Validation: the generation's top programs by training fitness.
        let mut vbest = f64::NAN;
        if !val.is_empty() {
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&a, &b| fits[b].fit.total_cmp(&fits[a].fit).then(a.cmp(&b)));
            order.dedup_by(|a, b| keys[*a] == keys[*b]);
            let top: Vec<usize> = order.into_iter().take(params.val_top).collect();
            let fresh: Vec<usize> = top.iter().copied().filter(|&i| !val_memo.contains_key(&keys[i])).collect();
            let progs: Vec<&Prog> = fresh.iter().map(|&i| &pop[i]).collect();
            for (&i, s) in fresh.iter().zip(eval_pop(games, pb, &val, &progs, params)?) {
                emu_total += s.emu;
                val_memo.insert(keys[i].clone(), s.fit);
            }
            for &i in &top {
                let v = val_memo[&keys[i]];
                vbest = if vbest.is_nan() { v } else { vbest.max(v) };
                if best_val.as_ref().is_none_or(|(_, b)| v > *b) {
                    best_val = Some((pop[i].clone(), v));
                }
            }
        }
        let line = format!(
            "gen {gen_no:3} val {vbest:8.1} best {:8.1} raw {:8.1} succ {}/{} frames {:6.1} size {:3} mean {:8.1} msize {:5.1} new {:4} emu {:.1} M {:.0}s",
            fits[bi].fit,
            fits[bi].raw,
            fits[bi].succ,
            train.len(),
            fits[bi].frames_done,
            pop[bi].size(),
            mean,
            msize,
            new.len(),
            emu_total as f64 / 1e6,
            t0.elapsed().as_secs_f64()
        );
        writeln!(log, "{line}")?;
        log.flush()?;
        lines.push(line);
        if gen_no == params.gens || emu_total >= params.budget {
            break;
        }
        // The next generation.
        let mut order: Vec<usize> = (0..pop.len()).collect();
        order.sort_by(|&a, &b| fits[b].fit.total_cmp(&fits[a].fit).then(a.cmp(&b)));
        let mut next: Vec<Prog> = order.iter().take(params.elite).map(|&i| pop[i].clone()).collect();
        let tour = |r: &mut SmallRng| {
            let mut w = r.random_range(0..pop.len());
            for _ in 1..params.tournament {
                let c = r.random_range(0..pop.len());
                if fits[c].fit > fits[w].fit {
                    w = c;
                }
            }
            w
        };
        while next.len() < params.pop {
            let a = tour(&mut r);
            let x: f64 = r.random();
            let child = if x < params.crossover {
                let b = tour(&mut r);
                crossover(&pop[a], &pop[b], &mut r).unwrap_or_else(|| pop[a].clone())
            } else if x < params.crossover + params.mutation {
                mutate(&pop[a], &g, &mut r)
            } else if x < params.crossover + params.mutation + params.const_mut {
                mutate_const(&pop[a], &g, &mut r)
            } else {
                pop[a].clone()
            };
            // Over the limits: the parent instead (Koza).
            next.push(if child.depth() > params.max_depth || child.size() > params.max_size { pop[a].clone() } else { child });
        }
        pop = next;
        // The memo only needs what can come back.
        if memo.len() > 200_000 {
            memo.clear();
        }
    }
    let (b, s) = best.unwrap();
    let chosen = best_val.map_or_else(|| b.clone(), |x| x.0);
    Ok(Run { best: chosen, best_score: s, best_train: b, emu_total, log: lines })
}
