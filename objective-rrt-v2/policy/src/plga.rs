//! Policy-list GA (`plga`): the genome is ALWAYS an ordered list of segments
//! `[(driver, hold frames), ...]` and is never flattened to recorded inputs while searching.
//!
//! A driver is a reactive policy with its lambda args / RNG seed (`Kind::Pol`) or pink noise from a seed
//! (`Kind::Pink`, one more policy kind). Fitness is a closed-loop rollout from the start state
//! (`eval::play_segs`): every policy reads the live game, so changing an early segment changes the state and
//! every later segment re-reacts to it. The switch rule is the hold duration (the last segment holds to the
//! horizon). Operators work on the list; `flatten` (rollout with `keep`) is only used to export / verify.
//! Score, horizon, Roll, Task, Library and the start sets are those of hybrid.rs.

use anyhow::Result;
use obrrt_core::GamePad;
use obrrt_gp::problems::Problem;
use obrrt_gp::rollout::{Case, Roll, Task, par};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sm64_binds::SM64Game;

use crate::eval::{self, BtnSpan, Drv, Env, Fine, Instance, Seg};
use crate::hybrid::{Library, Snap};
use crate::inv::{draw_instance, instance_of};

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Pol { name: String, inst: Instance },
    Pink { seed: u64 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gene {
    pub kind: Kind,
    pub hold: usize,
    /// closed-loop fine genes of this segment (stick / magnitude offsets, button span, MPC wrap)
    pub fine: Fine,
}

fn fine_text(f: &Fine) -> String {
    if *f == Fine::default() {
        return String::new();
    }
    let mut v = Vec::new();
    if f.dang != 0.0 || f.dmag != 0.0 {
        v.push(format!("off {:+.2}/{:+.2}", f.dang, f.dmag));
    }
    if let Some(b) = f.btn {
        v.push(format!("btn {}{} @{}+{}", if b.on { "+" } else { "-" }, if b.mask == obrrt_core::buttons::A_BUTTON { "A" } else { "B" }, b.at, b.len));
    }
    if let Some((h, n, r)) = f.mpc {
        v.push(format!("mpc {h}:{n}:{r}"));
    }
    format!("{{{}}}", v.join(","))
}

pub type Genome = Vec<Gene>;

pub fn show(g: &Genome) -> String {
    g.iter()
        .map(|s| match &s.kind {
            Kind::Pol { name, inst } => format!("{name}[{}]x{}{}", crate::inv::bind_text(inst), s.hold, fine_text(&s.fine)),
            Kind::Pink { seed } => format!("pink[{seed}]x{}{}", s.hold, fine_text(&s.fine)),
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn pink_pads(seed: u64, n: usize) -> Vec<GamePad> {
    let mut r = SmallRng::seed_from_u64(seed);
    obrrt_search::noise::pink_inputs(&mut r, n)
}

pub struct Ctx<'a> {
    pub case: &'a Case,
    pub task: &'a dyn Task,
    pub roll: Roll,
    pub lib: &'a Library,
    pub env: &'a Env,
}

pub struct Outc {
    pub pads: Vec<GamePad>,
    pub done: Option<usize>,
    pub best_eta: f64,
    pub frames: u64,
}

pub fn score(done: Option<usize>, best_eta: f64) -> f64 {
    match done {
        Some(d) => 1e6 - d as f64,
        None => -best_eta,
    }
}

/// Closed-loop rollout of the policy list. `keep` records the realised pads (flatten, for export only).
pub fn rollout(g: &mut SM64Game, cx: &Ctx, gn: &Genome, keep: bool) -> Result<Outc> {
    let horizon = cx.task.horizon().unwrap_or(cx.roll.horizon);
    let mut segs: Vec<Seg> = Vec::with_capacity(gn.len());
    for s in gn {
        let drv = match &s.kind {
            Kind::Pol { name, inst } => match cx.lib.bodies.get(name) {
                Some(p) => Drv::Pol { body: &p.body, inst: *inst },
                None => Drv::Pads(Vec::new()),
            },
            Kind::Pink { seed } => Drv::Pads(pink_pads(*seed, horizon)),
        };
        segs.push(Seg { drv, hold: s.hold, fine: s.fine });
    }
    let pl = eval::play_segs(g, cx.case, cx.task, &segs, cx.lib, cx.env, &cx.roll, keep)?;
    Ok(Outc { pads: pl.out.pads, done: pl.out.done, best_eta: pl.out.best_eta, frames: pl.out.frames + pl.desc.mpc_frames })
}

#[derive(Clone)]
struct Ind {
    gn: Genome,
    len: usize,
    score: f64,
    origin: usize,
}

pub const OPS: [&str; 15] = ["swap", "resample", "perturb", "shift", "insert", "delete", "dup", "cross", "reseed", "handoff", "nudge", "btnspan", "mpcwrap", "hold1", "repol"];
/// operators 10.. are the policy-level fine operators
const FINE_OPS: [usize; 4] = [10, 11, 12, 13];

pub struct Params {
    pub budgets: Vec<u64>,
    pub pop: usize,
    pub pool: Vec<String>,
    pub market: bool,
    /// per-segment stick / magnitude offsets, button span, hold +-1..3 (ops nudge, btnspan, hold1)
    pub fine_ops: bool,
    /// per-segment MPC wrap gene (op mpcwrap)
    pub mpc_wrap: bool,
    /// generation 0 from the portfolio: pool policies alone (+ lambda draws), best few chained into 2-segment lists
    pub seed0: bool,
    /// epsilon-greedy operator choice + credit re-warm at checkpoints (credit never starves an operator)
    pub eps: bool,
    /// last 30 % of every checkpoint window: fine operators only, on the elite
    pub finetune: bool,
    /// hybrid-style repolicy: keep the list up to a random time, a probe-ranked policy takes over (never pink)
    pub repol: bool,
}

struct Pool<'a> {
    lib: &'a Library,
    names: Vec<String>,
    /// per-policy credit (EMA); the start's probe scores seed it
    w: Vec<f64>,
}

impl Pool<'_> {
    fn pick(&self, r: &mut SmallRng) -> usize {
        let tot: f64 = self.w.iter().sum();
        let mut x = r.random::<f64>() * tot;
        for (i, w) in self.w.iter().enumerate() {
            x -= w;
            if x <= 0.0 {
                return i;
            }
        }
        self.w.len() - 1
    }
    fn gene(&self, r: &mut SmallRng, hold: usize) -> (Gene, Option<usize>) {
        // 15 %: a pink-noise segment (the raw-input kind)
        if r.random_bool(0.3) {
            let hold = if hold >= 300 { hold } else { r.random_range(2..=30) };
            return (Gene { kind: Kind::Pink { seed: r.random() }, hold, fine: Fine::default() }, None);
        }
        let i = self.pick(r);
        let pol = &self.lib.bodies[&self.names[i]];
        let inst = if r.random_bool(0.5) { instance_of(pol) } else { draw_instance(pol, r) };
        (Gene { kind: Kind::Pol { name: self.names[i].clone(), inst }, hold, fine: Fine::default() }, Some(i))
    }
}

fn rand_hold(r: &mut SmallRng) -> usize {
    r.random_range(6..=60)
}

/// Applies operator `op`; returns the child and the pool policy it introduced (for credit), or None when n/a.
fn apply(op: usize, a: &Genome, b: &Genome, len: usize, pool: &Pool, r: &mut SmallRng) -> Option<(Genome, Option<usize>)> {
    let mut c = a.clone();
    let n = c.len();
    let k = r.random_range(0..n);
    match op {
        0 => {
            let (g, p) = pool.gene(r, c[k].hold);
            c[k] = g;
            Some((c, p))
        }
        1 => match &c[k].kind {
            Kind::Pol { name, .. } => {
                let inst = draw_instance(&pool.lib.bodies[name], r);
                c[k].kind = Kind::Pol { name: name.clone(), inst };
                Some((c, None))
            }
            Kind::Pink { .. } => None,
        },
        2 => match &c[k].kind {
            Kind::Pol { name, inst } => {
                let mut i2 = *inst;
                let m = r.random_range(1..=3);
                for _ in 0..m {
                    let j = r.random_range(0..8);
                    let g = (r.random::<f32>() + r.random::<f32>() + r.random::<f32>() - 1.5) * 2.0;
                    i2.args[j] += g * (0.15 * i2.args[j].abs() + 0.1);
                }
                if r.random_bool(0.3) {
                    i2.seed = r.random();
                }
                c[k].kind = Kind::Pol { name: name.clone(), inst: i2 };
                Some((c, None))
            }
            Kind::Pink { .. } => None,
        },
        3 => {
            // move a boundary: lengthen / shorten the hold (later segments start later / earlier)
            let d = r.random_range(1..=12) as i64 * if r.random_bool(0.5) { 1 } else { -1 };
            c[k].hold = (c[k].hold as i64 + d).clamp(1, 300) as usize;
            Some((c, None))
        }
        4 => {
            let h = rand_hold(r);
            let (g, p) = pool.gene(r, h);
            c.insert(r.random_range(0..=n), g);
            Some((c, p))
        }
        5 => {
            if n < 2 {
                return None;
            }
            c.remove(k);
            Some((c, None))
        }
        6 => {
            let mut g = c[k].clone();
            if r.random_bool(0.5) {
                match &mut g.kind {
                    Kind::Pink { seed } => *seed = r.random(),
                    Kind::Pol { name, inst } => *inst = draw_instance(&pool.lib.bodies[name], r),
                }
            }
            c.insert(k + 1, g);
            Some((c, None))
        }
        7 => {
            // splice at segment boundaries; the cut in b is the boundary nearest in time to the cut in a
            let start = |g: &Genome, i: usize| g[..i].iter().map(|s| s.hold).sum::<usize>();
            let i = r.random_range(0..=n);
            let t = start(a, i);
            let j = (0..=b.len()).min_by_key(|&j| (start(b, j) as i64 - t as i64).abs()).unwrap();
            let mut child: Genome = a[..i].to_vec();
            child.extend_from_slice(&b[j..]);
            if child.is_empty() {
                return None;
            }
            Some((child, None))
        }
        8 => {
            // resample the seed of a pink-noise segment (or of a policy's rnd seed)
            let ks: Vec<usize> = (0..n).filter(|&i| matches!(c[i].kind, Kind::Pink { .. })).collect();
            if ks.is_empty() {
                return None;
            }
            let k = ks[r.random_range(0..ks.len())];
            c[k].kind = Kind::Pink { seed: r.random() };
            Some((c, None))
        }
        10 => {
            // nudge: add a small bounded stick-angle / magnitude offset to what this segment's policy outputs
            if !matches!(c[k].kind, Kind::Pol { .. }) {
                return None;
            }
            let gs = |r: &mut SmallRng| (r.random::<f32>() + r.random::<f32>() + r.random::<f32>() - 1.5) * 2.0;
            let f = &mut c[k].fine;
            let u = r.random::<f32>();
            if u < 0.12 {
                f.dang = 0.0;
                f.dmag = 0.0;
            } else {
                if u < 0.62 || u > 0.9 {
                    f.dang = (f.dang + gs(r) * 0.35).clamp(-1.5, 1.5);
                }
                if u >= 0.5 {
                    f.dmag = (f.dmag + gs(r) * 0.35).clamp(-1.0, 1.0);
                }
            }
            Some((c, None))
        }
        11 => {
            // button-override span: force / suppress A or B for a few frames inside the hold (or tweak / clear it)
            let hold = c[k].hold.min(120);
            let f = &mut c[k].fine;
            match f.btn {
                Some(b) if r.random_bool(0.5) => {
                    if r.random_bool(0.25) {
                        f.btn = None;
                    } else {
                        let at = (b.at as i32 + r.random_range(-3..=3)).clamp(0, 300) as u16;
                        let len = (b.len as i32 + r.random_range(-2..=2)).clamp(1, 16) as u16;
                        f.btn = Some(BtnSpan { at, len, ..b });
                    }
                }
                _ => {
                    f.btn = Some(BtnSpan {
                        at: r.random_range(0..hold.max(1)) as u16,
                        len: r.random_range(1..=8),
                        mask: if r.random_bool(0.5) { obrrt_core::buttons::A_BUTTON } else { obrrt_core::buttons::B_BUTTON },
                        on: r.random_bool(0.5),
                    })
                }
            }
            Some((c, None))
        }
        12 => {
            // MPC wrap: toggle / resample the stick-only short-horizon search around this segment's policy
            if !matches!(c[k].kind, Kind::Pol { .. }) {
                return None;
            }
            let f = &mut c[k].fine;
            if f.mpc.is_some() && r.random_bool(0.4) {
                f.mpc = None;
            } else {
                f.mpc = Some(([3u8, 4, 6][r.random_range(0..3)], [3u8, 5][r.random_range(0..2)], [8.0f32, 15.0, 25.0][r.random_range(0..3)]));
            }
            Some((c, None))
        }
        14 => {
            let t = r.random_range(1..=len.max(2));
            let mut out: Genome = Vec::new();
            let mut at = 0usize;
            for s in &c {
                if at >= t {
                    break;
                }
                let mut s = s.clone();
                s.hold = s.hold.min(t - at);
                at += s.hold;
                out.push(s);
            }
            let i = if r.random_bool(0.6) {
                let mut v: Vec<usize> = (0..pool.w.len()).collect();
                v.sort_by(|&a, &b| pool.w[b].partial_cmp(&pool.w[a]).unwrap());
                v[r.random_range(0..v.len().min(4))]
            } else {
                r.random_range(0..pool.names.len())
            };
            let pol = &pool.lib.bodies[&pool.names[i]];
            let inst = if r.random_bool(0.5) { draw_instance(pol, r) } else { instance_of(pol) };
            out.push(Gene { kind: Kind::Pol { name: pool.names[i].clone(), inst }, hold: 300, fine: Fine::default() });
            Some((out, Some(i)))
        }
        13 => {
            let d = r.random_range(1..=3) as i64 * if r.random_bool(0.5) { 1 } else { -1 };
            c[k].hold = (c[k].hold as i64 + d).clamp(1, 300) as usize;
            Some((c, None))
        }
        _ => {
            // handoff: keep the list up to a random time `t` (cutting the segment that spans it), then a pool
            // policy (or pink) takes over from whatever state that prefix produced
            let t = r.random_range(1..=len.max(2));
            let mut out: Genome = Vec::new();
            let mut at = 0usize;
            if r.random_bool(0.5) {
                // a fresh pink-noise prefix of t frames (a different state to react from)
                out.push(Gene { kind: Kind::Pink { seed: r.random() }, hold: t, fine: Fine::default() });
                at = t;
            }
            for s in &c {
                if at >= t {
                    break;
                }
                let mut s = s.clone();
                s.hold = s.hold.min(t - at);
                at += s.hold;
                out.push(s);
            }
            let (g, p) = pool.gene(r, 300);
            out.push(g);
            Some((out, p))
        }
    }
}

pub struct Found {
    pub snaps: Vec<Snap>,
    pub best: Option<(usize, Genome, Vec<GamePad>)>,
    pub opstat: Vec<(u64, f64)>,
}

pub fn search(g: &mut SM64Game, cx: &Ctx, p: &Params, seed: u64) -> Result<Found> {
    let mut r = SmallRng::seed_from_u64(seed);
    let total = *p.budgets.last().unwrap();
    let mut spent = 0u64;
    let mut snaps: Vec<Snap> = Vec::new();
    let mut ns = 0usize;
    let mut best: Option<(usize, Genome, Vec<GamePad>)> = None;
    let mut pool = Pool { lib: cx.lib, names: p.pool.clone(), w: vec![1.0; p.pool.len()] };
    let mut pop: Vec<Ind> = Vec::new();

    macro_rules! eval_push {
        ($gn:expr, $origin:expr) => {{
            let gn: Genome = $gn;
            let o = rollout(g, cx, &gn, true)?;
            spent += o.frames;
            if let Some(d) = o.done {
                if best.as_ref().is_none_or(|b| d < b.0) {
                    best = Some((d, gn.clone(), o.pads.clone()));
                }
            }
            while ns < p.budgets.len() && spent >= p.budgets[ns] {
                snaps.push(Snap { budget: p.budgets[ns], done: best.as_ref().map(|b| b.0), emu: spent });
                ns += 1;
            }
            Ind { gn, len: o.pads.len(), score: score(o.done, o.best_eta), origin: $origin }
        }};
    }

    // generation 0: each pool policy alone (default instance) = the per-start probe
    let mut probe: Vec<f64> = Vec::new();
    for name in &p.pool {
        let pol = &cx.lib.bodies[name];
        let ind = eval_push!(vec![Gene { kind: Kind::Pol { name: name.clone(), inst: instance_of(pol) }, hold: 300, fine: Fine::default() }], 99);
        probe.push(ind.score);
        pop.push(ind);
    }
    // per-start policy weights from the probe: better policies from this start are drawn more, all stay possible
    let mx = probe.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    pool.w = probe.iter().map(|s| 0.2 + (-((mx - s).min(100.0)) / 25.0).exp()).collect();
    if p.seed0 {
        // portfolio seeding (as hybrid's gen 0): one lambda draw of every pool policy alone ...
        let mut best_of = probe.clone();
        for (i, name) in p.pool.iter().enumerate() {
            if spent >= total {
                break;
            }
            let pol = &cx.lib.bodies[name];
            let ind = eval_push!(vec![Gene { kind: Kind::Pol { name: name.clone(), inst: draw_instance(pol, &mut r) }, hold: 300, fine: Fine::default() }], 97);
            best_of[i] = best_of[i].max(ind.score);
            pop.push(ind);
        }
        // ... then the best three policies chained into 2-segment lists (a for h frames, then b)
        let mut order: Vec<usize> = (0..p.pool.len()).collect();
        order.sort_by(|&a, &b| best_of[b].partial_cmp(&best_of[a]).unwrap());
        let top: Vec<usize> = order.into_iter().take(3).collect();
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for &a in &top {
            for &b in &top {
                if a != b {
                    pairs.push((a, b));
                }
            }
        }
        for _ in 0..4.min(pairs.len()) {
            if spent >= total {
                break;
            }
            let (a, b) = pairs.swap_remove(r.random_range(0..pairs.len()));
            let ga = |i: usize, hold: usize, r: &mut SmallRng| Gene { kind: Kind::Pol { name: p.pool[i].clone(), inst: instance_of(&cx.lib.bodies[&p.pool[i]]) }, hold, fine: Fine::default() };
            let h = r.random_range(8..=60);
            let gn = vec![ga(a, h, &mut r), ga(b, 300, &mut r)];
            let ind = eval_push!(gn, 96);
            pop.push(ind);
        }
    }
    while pop.len() < p.pop && spent < total {
        let gn: Genome = if r.random_bool(0.25) {
            vec![Gene { kind: Kind::Pink { seed: r.random() }, hold: 300, fine: Fine::default() }]
        } else {
            let t = r.random_range(1..=80);
            let mut v = vec![Gene { kind: Kind::Pink { seed: r.random() }, hold: t, fine: Fine::default() }];
            v.push(pool.gene(&mut r, 300).0);
            v
        };
        let ind = eval_push!(gn, 98);
        pop.push(ind);
    }
    let nops = OPS.len();
    let mut active = vec![true; nops];
    active[10] = p.fine_ops;
    active[11] = p.fine_ops;
    active[13] = p.fine_ops;
    active[12] = p.mpc_wrap;
    active[14] = p.repol;
    let fine_active: Vec<usize> = FINE_OPS.iter().cloned().filter(|&i| active[i]).collect();
    let aidx: Vec<usize> = (0..nops).filter(|&i| active[i]).collect();
    let mut credit = vec![1.0f64; nops];
    let mut cnt = vec![0u64; nops];
    let tour = |pop: &[Ind], r: &mut SmallRng| -> usize {
        let mut b = r.random_range(0..pop.len());
        for _ in 0..2 {
            let c = r.random_range(0..pop.len());
            if pop[c].score > pop[b].score {
                b = c;
            }
        }
        b
    };
    let mut fails = 0u32;
    while spent < total {
        // fine-tuning phase: last 30 % of every checkpoint window, fine operators only, on the elite
        let (prev_b, next_b) = (if ns == 0 { 0 } else { p.budgets[ns - 1] }, p.budgets[ns.min(p.budgets.len() - 1)]);
        let tune = p.finetune && !fine_active.is_empty() && spent as f64 >= prev_b as f64 + 0.7 * (next_b - prev_b) as f64;
        let op = if tune {
            let idx = &fine_active;
            let s: f64 = idx.iter().map(|&i| credit[i]).sum::<f64>().max(1e-9);
            let w: Vec<f64> = idx.iter().map(|&i| 0.1 + 0.6 * credit[i] / s).collect();
            let tot: f64 = w.iter().sum();
            let mut x = r.random::<f64>() * tot;
            let mut o = *idx.last().unwrap();
            for (k, &i) in idx.iter().enumerate() {
                x -= w[k];
                if x <= 0.0 {
                    o = i;
                    break;
                }
            }
            o
        } else if p.eps && r.random_bool(0.25) {
            // epsilon-greedy: a uniformly random active operator, whatever its credit
            aidx[r.random_range(0..aidx.len())]
        } else {
            // market share of operators (credit = score gain per frame spent, EMA)
            let pmin = if p.eps { 0.03 } else { 0.04 };
            let s: f64 = aidx.iter().map(|&i| credit[i]).sum::<f64>().max(1e-9);
            let w: Vec<f64> = aidx.iter().map(|&i| if p.market { pmin + (1.0 - pmin * aidx.len() as f64).max(0.0) * credit[i] / s } else { 1.0 }).collect();
            let tot: f64 = w.iter().sum();
            let mut x = r.random::<f64>() * tot;
            let mut o = *aidx.last().unwrap();
            for (k, &i) in aidx.iter().enumerate() {
                x -= w[k];
                if x <= 0.0 {
                    o = i;
                    break;
                }
            }
            o
        };
        let (ia, ib) = if tune {
            // elite: one of the best three
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&a, &b| pop[b].score.partial_cmp(&pop[a].score).unwrap());
            let e = order[r.random_range(0..order.len().min(3))];
            (e, e)
        } else {
            (tour(&pop, &mut r), tour(&pop, &mut r))
        };
        let Some((child, pidx)) = apply(op, &pop[ia].gn, &pop[ib].gn, pop[ia].len, &pool, &mut r) else {
            fails += 1;
            if fails > 100 {
                spent += 1; // no applicable child for a long time (e.g. only-pink elite in the fine phase): force progress
            }
            continue;
        };
        if child == pop[ia].gn {
            fails += 1;
            if fails > 100 {
                spent += 1;
            }
            continue;
        }
        fails = 0;
        let before = spent;
        let ns0 = ns;
        let ind = eval_push!(child, op);
        cnt[op] += 1;
        let cost = (spent - before).max(1) as f64;
        let gain = (ind.score - pop[ia].score).clamp(0.0, 300.0);
        credit[op] = 0.9 * credit[op] + 0.1 * gain / cost * 100.0 + 1e-6;
        if p.eps && ns > ns0 {
            // a checkpoint passed: pull every credit halfway back to the mean so late operators are not starved
            let m = aidx.iter().map(|&i| credit[i]).sum::<f64>() / aidx.len() as f64;
            for &i in &aidx {
                credit[i] = 0.5 * credit[i] + 0.5 * m;
            }
        }
        if let Some(pi) = pidx {
            pool.w[pi] = (0.95 * pool.w[pi] + 0.05 * (0.2 + gain / 10.0)).max(0.05);
        }
        if tune {
            // hill-climb the elite (neutral moves accepted: drift along plateaus)
            if ind.score >= pop[ia].score {
                pop[ia] = ind;
            }
            continue;
        }
        let worst = (0..pop.len()).min_by(|&a, &b| pop[a].score.partial_cmp(&pop[b].score).unwrap()).unwrap();
        if ind.score >= pop[worst].score {
            pop[worst] = ind;
        }
    }
    while ns < p.budgets.len() {
        snaps.push(Snap { budget: p.budgets[ns], done: best.as_ref().map(|b| b.0), emu: spent });
        ns += 1;
    }
    Ok(Found { snaps, best, opstat: (0..nops).map(|i| (cnt[i], credit[i])).collect() })
}

/// Every (case, seed) of the problem; returns snapshots[seed][case] and writes the best genomes (policy lists)
/// to `genomes` if given. CLEAN_DUMP=dir flattens the best finished route to raw inputs from power-on for tasplay.
pub fn run(games: &mut [SM64Game], pb: &Problem, lib: &Library, env: &Env, p: &Params, seeds: u64, genomes: Option<&str>) -> Result<Vec<Vec<Vec<Snap>>>> {
    let n = pb.cases.len() * seeds as usize;
    let glog = std::sync::Mutex::new(String::new());
    let ops = std::sync::Mutex::new(vec![(0u64, 0.0f64); OPS.len()]);
    let res = par(games, n, |g, i| {
        let (ci, seed) = (i % pb.cases.len(), (i / pb.cases.len()) as u64 + 1);
        let c = &pb.cases[ci];
        let cx = Ctx { case: c, task: pb.tasks[c.task].as_ref(), roll: pb.roll, lib, env };
        let f = search(g, &cx, p, seed * 1000 + ci as u64)?;
        {
            let mut o = ops.lock().unwrap();
            for (k, (cn, cr)) in f.opstat.iter().enumerate() {
                o[k].0 += cn;
                o[k].1 += cr / n as f64;
            }
        }
        if let Some((d, gn, bp)) = &f.best {
            glog.lock().unwrap().push_str(&format!("{}\tseed{seed}\t{d}\t{}\n", c.name, show(gn)));
            if let Ok(dir) = std::env::var("CLEAN_DUMP") {
                std::fs::create_dir_all(&dir)?;
                let mut pads = pb.movies[c.movie].1[..c.frame - c.noise.len()].to_vec();
                pads.extend_from_slice(&c.noise);
                pads.extend_from_slice(bp);
                let file = format!("{dir}/Plga-s{seed}-{}.raw", c.name.replace(['/', ','], "_"));
                obrrt_gp::problems::write_raw(&file, &pads)?;
                std::fs::write(format!("{file}.idx"), format!("{file} {} {} {}\n", c.frame, d, bp.len()))?;
            }
        }
        Ok((ci, seed, f.snaps))
    })?;
    if let Some(path) = genomes {
        std::fs::write(path, glog.into_inner().unwrap())?;
    }
    let o = ops.into_inner().unwrap();
    println!("operator use (children) and mean final credit: {}", OPS.iter().zip(&o).map(|(n, (c, cr))| format!("{n}={c}/{cr:.2}")).collect::<Vec<_>>().join(" "));
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
