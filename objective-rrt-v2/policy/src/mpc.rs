//! Derivative-free (and finite-difference) refinement of the continuous part
//! of the stick, by short emulator rollouts from a savestate.
//!
//! The decision this module answers: the buttons (and the stick *class*) are
//! fixed by the caller; where inside the class (angle offset from a reference
//! direction, magnitude) should the stick sit over the next `horizon` frames?
//!
//! * [`Prob`]: what to optimise (reference direction, ranges, buttons, scorer).
//! * [`Scorer`]: the value of a rollout, fed one state per frame (may stop early).
//! * [`Evaluator`]: plan -> rollout -> score, with cost accounting.
//! * [`Algo`]: the optimisers (random shooting, CEM, (1+lambda)-ES, separable
//!   CMA, SPSA, coordinate finite differences, grid + pattern search).
//! * [`Refiner`]: receding-horizon use (re-plan every k frames, warm start,
//!   emulator-frame accounting) returning a [`Decision`] the policy turns into
//!   a `types::Choice`.
//! * [`smoothness`], [`scan_axis`]: how smooth the score surface is (cliffs).
//!
//! A plan is a point `x` in `[-1, 1]^D`: `knots` piecewise-linear knots over
//! the horizon, each an angle offset (`x * ang_max` radians from the aim) and,
//! unless the magnitude is fixed, a tilt (`mag_lo..mag_hi`).

use anyhow::Result;
use obrrt_core::{Eta, GamePad, Mem, St, read_state};
use obrrt_gp::rollout::{Roll, Task};
use rand::Rng;
use rand::rngs::SmallRng;
use sm64_binds::{SM64Game, SaveState};

// ------------------------------------------------------------------ problem

/// The reference direction the angle offset is measured from, per frame.
#[derive(Clone, Debug)]
pub enum Aim {
    World(i16),
    /// Direction to a world point (xz).
    Target([f32; 3]),
    Face,
    /// Direction of the horizontal velocity (slides).
    Vel,
}

pub fn yaw_of(dx: f32, dz: f32) -> i16 {
    ((dx.atan2(dz) * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
}

impl Aim {
    pub fn yaw(&self, s: &St) -> i16 {
        match self {
            Aim::World(y) => *y,
            Aim::Target(t) => yaw_of(t[0] - s.pos[0], t[2] - s.pos[2]),
            Aim::Face => s.yaw,
            Aim::Vel => {
                if s.vel[0].hypot(s.vel[2]) < 1e-3 {
                    s.yaw
                } else {
                    yaw_of(s.vel[0], s.vel[2])
                }
            }
        }
    }
}

/// The value of a rollout. Fed the start state, then each frame's state.
pub trait Scorer {
    fn start(&mut self, s0: &St);
    /// `false` ends the rollout (done, dead, landed...).
    fn step(&mut self, t: usize, s: &St) -> bool;
    /// Higher is better.
    fn finish(&mut self) -> f64;
}

/// Buttons per frame: `(t, state before the frame, last pad) -> button bits`.
pub type BtnFn<'a> = &'a dyn Fn(usize, &St, GamePad) -> u16;

pub struct Prob<'a> {
    pub aim: Aim,
    pub horizon: usize,
    pub knots: usize,
    /// Largest angle offset, radians.
    pub ang_max: f32,
    /// Tilt range (0..1, `moves::arcs::stick`'s scale: 0.397 is raw magnitude 48).
    pub mag_lo: f32,
    pub mag_hi: f32,
    /// Fix the tilt (drop the magnitude dimensions).
    pub fix_mag: Option<f32>,
    pub buttons: BtnFn<'a>,
    pub last_pad: GamePad,
    pub scorer: &'a mut dyn Scorer,
}

impl Prob<'_> {
    pub fn dim(&self) -> usize {
        self.knots * if self.fix_mag.is_some() { 1 } else { 2 }
    }
    pub fn spec(&self) -> Spec {
        Spec { aim: self.aim.clone(), horizon: self.horizon, knots: self.knots, ang_max: self.ang_max, mag_lo: self.mag_lo, mag_hi: self.mag_hi, fix_mag: self.fix_mag }
    }
}

/// The plan parameterisation of a [`Prob`] (everything but buttons and scorer).
#[derive(Clone, Debug)]
pub struct Spec {
    pub aim: Aim,
    pub horizon: usize,
    pub knots: usize,
    pub ang_max: f32,
    pub mag_lo: f32,
    pub mag_hi: f32,
    pub fix_mag: Option<f32>,
}

impl Spec {
    /// The pad stick for a decision's fine values on state `st`.
    pub fn stick_for(&self, st: &St, fine_ang: f32, fine_mag: f32) -> (i8, i8) {
        let ang = fine_ang * self.ang_max;
        let tilt = self.fix_mag.unwrap_or(self.mag_lo + fine_mag * (self.mag_hi - self.mag_lo));
        let yaw = self.aim.yaw(st).wrapping_add((ang * 65536.0 / std::f32::consts::TAU).round() as i32 as i16);
        stick(st.cam_yaw, yaw, tilt)
    }
}

/// Plan values at frame `t` of `h`: (angle offset radians, tilt).
fn plan_at(x: &[f32], knots: usize, fixed_mag: Option<f32>, ang_max: f32, mag: (f32, f32), t: usize, h: usize) -> (f32, f32) {
    let stride = if fixed_mag.is_some() { 1 } else { 2 };
    let pos = if knots <= 1 || h <= 1 { 0.0 } else { t as f32 / (h - 1) as f32 * (knots - 1) as f32 };
    let k0 = (pos.floor() as usize).min(knots - 1);
    let k1 = (k0 + 1).min(knots - 1);
    let w = pos - k0 as f32;
    let g = |k: usize, c: usize| x[k * stride + c].clamp(-1.0, 1.0);
    let ang = g(k0, 0) * (1.0 - w) + g(k1, 0) * w;
    let tilt = match fixed_mag {
        Some(m) => m,
        None => {
            let u = g(k0, 1) * (1.0 - w) + g(k1, 1) * w;
            mag.0 + (u * 0.5 + 0.5) * (mag.1 - mag.0)
        }
    };
    (ang * ang_max, tilt)
}

// ---------------------------------------------------------------- evaluator

#[derive(Clone, Copy, Debug, Default)]
pub struct Cost {
    pub evals: u64,
    pub frames: u64,
    pub loads: u64,
}

pub struct Evaluator<'a, 'b> {
    pub game: &'a mut SM64Game,
    pub base: &'a SaveState,
    pub start: St,
    pub prob: Prob<'b>,
    pub cost: Cost,
    pub frame0: u32,
}

impl<'a, 'b> Evaluator<'a, 'b> {
    pub fn new(game: &'a mut SM64Game, base: &'a SaveState, start: St, prob: Prob<'b>) -> Self {
        let frame0 = start.frame;
        Evaluator { game, base, start, prob, cost: Cost::default(), frame0 }
    }

    /// The pad of a plan at frame `t` (also used to replay a decision).
    pub fn pad_for(&self, x: &[f32], t: usize, st: &St, last: GamePad) -> GamePad {
        let p = &self.prob;
        let (ang, tilt) = plan_at(x, p.knots, p.fix_mag, p.ang_max, (p.mag_lo, p.mag_hi), t, p.horizon);
        let yaw = p.aim.yaw(st).wrapping_add((ang * 65536.0 / std::f32::consts::TAU).round() as i32 as i16);
        let (sx, sy) = stick(st.cam_yaw, yaw, tilt);
        GamePad::new((p.buttons)(t, st, last), sx, sy)
    }

    /// Roll `x` out from the base state; the scorer's value.
    pub fn eval(&mut self, x: &[f32]) -> Result<f64> {
        self.game.load_state(self.base)?;
        self.cost.loads += 1;
        self.cost.evals += 1;
        self.prob.scorer.start(&self.start);
        let mut st = self.start.clone();
        let mut last = self.prob.last_pad;
        for t in 0..self.prob.horizon {
            let pad = self.pad_for(x, t, &st, last);
            self.game.step_game(pad)?;
            self.cost.frames += 1;
            last = pad;
            st = read_state(self.game, self.frame0 + t as u32 + 1, false);
            if !self.prob.scorer.step(t, &st) {
                break;
            }
        }
        Ok(self.prob.scorer.finish())
    }
}

/// `moves::arcs::stick` (raw stick for a world yaw and tilt 0..1).
pub fn stick(cam_yaw: i16, yaw: i16, tilt: f32) -> (i8, i8) {
    obrrt_gp::lang::stick(cam_yaw, yaw, tilt)
}

// ------------------------------------------------------------------ scorers

fn d_xz(a: [f32; 3], b: [f32; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() as f64
}

/// Land (first ground contact after being airborne) near `goal`, avoid a
/// forbidden disc; if no landing in the horizon, the last position scores.
pub struct LandAt {
    pub goal: [f32; 3],
    pub forbid: Option<([f32; 3], f32)>,
    pub forbid_pen: f64,
    airborne: bool,
    end: [f32; 3],
    landed: bool,
    dead: bool,
}
impl LandAt {
    pub fn new(goal: [f32; 3], forbid: Option<([f32; 3], f32)>) -> Self {
        LandAt { goal, forbid, forbid_pen: 1000.0, airborne: false, end: goal, landed: false, dead: false }
    }
}
impl Scorer for LandAt {
    fn start(&mut self, s0: &St) {
        self.airborne = s0.action & 0x800 != 0;
        self.end = s0.pos;
        self.landed = false;
        self.dead = false;
    }
    fn step(&mut self, _t: usize, s: &St) -> bool {
        self.end = s.pos;
        if !s.alive() {
            self.dead = true;
            return false;
        }
        let air = s.action & 0x800 != 0;
        if self.airborne && !air {
            self.landed = true;
            return false;
        }
        if air {
            self.airborne = true;
        }
        true
    }
    fn finish(&mut self) -> f64 {
        let mut v = -d_xz(self.end, self.goal);
        if !self.landed {
            v -= 300.0;
        }
        if self.dead {
            v -= 3000.0;
        }
        if let Some((c, r)) = self.forbid {
            if d_xz(self.end, c) < r as f64 {
                v -= self.forbid_pen;
            }
        }
        v
    }
}

/// Steer to a point: minus the closest horizontal approach over the horizon
/// plus a small weight on the end being close.
pub struct Reach {
    pub target: [f32; 3],
    best: f64,
    last: f64,
}
impl Reach {
    pub fn new(target: [f32; 3]) -> Self {
        Reach { target, best: f64::INFINITY, last: 0.0 }
    }
}
impl Scorer for Reach {
    fn start(&mut self, s0: &St) {
        self.best = f64::INFINITY;
        self.last = d_xz(s0.pos, self.target);
    }
    fn step(&mut self, _t: usize, s: &St) -> bool {
        self.last = d_xz(s.pos, self.target);
        self.best = self.best.min(self.last);
        s.alive()
    }
    fn finish(&mut self) -> f64 {
        -(self.best.min(1e9) + 0.25 * self.last)
    }
}

/// Keep speed: mean horizontal speed over the horizon plus half the final
/// speed, minus a heavy penalty for dying.
#[derive(Default)]
pub struct HoldSpeed {
    sum: f64,
    n: usize,
    last: f64,
    dead: bool,
}
impl HoldSpeed {
    pub fn new() -> Self {
        Self::default()
    }
}
impl Scorer for HoldSpeed {
    fn start(&mut self, _s0: &St) {
        *self = HoldSpeed::default();
    }
    fn step(&mut self, _t: usize, s: &St) -> bool {
        let v = s.vel[0].hypot(s.vel[2]) as f64;
        self.sum += v;
        self.n += 1;
        self.last = v;
        if !s.alive() {
            self.dead = true;
        }
        s.alive()
    }
    fn finish(&mut self) -> f64 {
        self.sum / self.n.max(1) as f64 + 0.5 * self.last - if self.dead { 500.0 } else { 0.0 }
    }
}

/// The gp rollout's fitness for a [`Task`] (discounted ETA decrease, bonus on
/// Done, penalty on death/Lost), so refinement optimises what the GA/GP score.
pub struct TaskFit<'t> {
    pub task: &'t dyn Task,
    pub roll: Roll,
    mem: Mem,
    start: Option<St>,
    prev: f64,
    sum: f64,
    disc: f64,
    best: f64,
}
impl<'t> TaskFit<'t> {
    pub fn new(task: &'t dyn Task, roll: Roll) -> Self {
        TaskFit { task, roll, mem: Mem::default(), start: None, prev: 0.0, sum: 0.0, disc: 1.0, best: f64::NEG_INFINITY }
    }
}
impl Scorer for TaskFit<'_> {
    fn start(&mut self, s0: &St) {
        self.mem = Mem::default();
        self.prev = self.task.eta(s0, s0, &mut self.mem).frames();
        self.start = Some(s0.clone());
        self.sum = 0.0;
        self.disc = 1.0;
        self.best = f64::NEG_INFINITY;
    }
    fn step(&mut self, _t: usize, s: &St) -> bool {
        let start = self.start.as_ref().unwrap();
        let eta = if !s.alive() || s.lives < start.lives { Eta::Lost } else { self.task.eta(s, start, &mut self.mem) };
        let lost = match eta {
            Eta::Lost => true,
            Eta::Frames(x) => !x.is_finite(),
            Eta::Done => false,
        };
        let mut rw;
        if lost {
            rw = -self.roll.death;
        } else {
            rw = self.prev - eta.frames();
            self.prev = eta.frames();
            if eta.done() {
                rw += self.roll.bonus;
            }
        }
        self.sum += self.disc * rw;
        self.disc *= self.roll.gamma;
        self.best = self.best.max(self.sum);
        !(lost || eta.done())
    }
    fn finish(&mut self) -> f64 {
        if self.best.is_finite() { self.best } else { 0.0 }
    }
}

// -------------------------------------------------------------- optimisers

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub evals: usize,
    /// Stop when this many emulator frames are spent (0 = no frame limit).
    pub frames: u64,
}

#[derive(Clone, Debug)]
pub enum Algo {
    Random,
    Cem { pop: usize, elite: f32, smooth: f32, sigma0: f32 },
    Es { lambda: usize, sigma0: f32 },
    /// Separable CMA-ES.
    Cma { pop: usize, sigma0: f32 },
    Spsa { a: f32, c: f32 },
    /// Central finite differences per coordinate, ascent with backtracking.
    Fd { step: f32, eps: f32 },
    /// Coarse grid over a constant plan, then pattern search.
    Grid { n_ang: usize, n_mag: usize },
}

impl Algo {
    pub fn name(&self) -> &'static str {
        match self {
            Algo::Random => "random",
            Algo::Cem { .. } => "cem",
            Algo::Es { .. } => "es",
            Algo::Cma { .. } => "cma",
            Algo::Spsa { .. } => "spsa",
            Algo::Fd { .. } => "fd",
            Algo::Grid { .. } => "grid",
        }
    }
}

fn normal(r: &mut SmallRng) -> f32 {
    let u1: f32 = r.random_range(1e-7..1.0);
    let u2: f32 = r.random_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn clamp(x: &mut [f32]) {
    for v in x {
        *v = v.clamp(-1.0, 1.0);
    }
}

struct Run<'e, 'a, 'b> {
    ev: &'e mut Evaluator<'a, 'b>,
    bx: Vec<f32>,
    bv: f64,
    /// (evals so far, best value) at each improvement.
    trace: Vec<(u64, f64)>,
    budget: Budget,
    f0: u64,
    e0: u64,
}

impl Run<'_, '_, '_> {
    fn left(&self) -> bool {
        ((self.ev.cost.evals - self.e0) as usize) < self.budget.evals && (self.budget.frames == 0 || self.ev.cost.frames - self.f0 < self.budget.frames)
    }
    fn f(&mut self, x: &[f32]) -> Result<f64> {
        let v = self.ev.eval(x)?;
        if v > self.bv {
            self.bv = v;
            self.bx = x.to_vec();
            self.trace.push((self.ev.cost.evals - self.e0, v));
        }
        Ok(v)
    }
}

#[derive(Clone, Debug)]
pub struct OptResult {
    pub x: Vec<f32>,
    pub value: f64,
    pub x0_value: f64,
    pub cost: Cost,
    pub trace: Vec<(u64, f64)>,
}

fn sort_desc<T>(c: &mut [(f64, T)]) {
    c.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
}

/// Optimise from `x0` (the baseline plan, always evaluated first so the
/// result is never worse than it) within `budget`.
pub fn optimize(algo: &Algo, ev: &mut Evaluator, x0: &[f32], budget: Budget, rng: &mut SmallRng) -> Result<OptResult> {
    let d = ev.prob.dim();
    let c0 = ev.cost;
    let mut run = Run { bx: x0.to_vec(), bv: f64::NEG_INFINITY, trace: vec![], budget, f0: c0.frames, e0: c0.evals, ev };
    let v0 = run.f(x0)?;
    match algo {
        Algo::Random => {
            while run.left() {
                let x: Vec<f32> = (0..d).map(|_| rng.random_range(-1.0..1.0)).collect();
                run.f(&x)?;
            }
        }
        Algo::Cem { pop, elite, smooth, sigma0 } => {
            let (mut mu, mut sg) = (x0.to_vec(), vec![*sigma0; d]);
            let ne = ((*pop as f32 * elite).ceil() as usize).max(2);
            while run.left() {
                let mut cand: Vec<(f64, Vec<f32>)> = vec![];
                for _ in 0..*pop {
                    if !run.left() {
                        break;
                    }
                    let mut x: Vec<f32> = (0..d).map(|i| mu[i] + sg[i] * normal(rng)).collect();
                    clamp(&mut x);
                    let v = run.f(&x)?;
                    cand.push((v, x));
                }
                if cand.len() < ne {
                    break;
                }
                sort_desc(&mut cand);
                for i in 0..d {
                    let m = cand[..ne].iter().map(|c| c.1[i]).sum::<f32>() / ne as f32;
                    let var = cand[..ne].iter().map(|c| (c.1[i] - m).powi(2)).sum::<f32>() / ne as f32;
                    mu[i] = (1.0 - smooth) * m + smooth * mu[i];
                    sg[i] = ((1.0 - smooth) * var.sqrt() + smooth * sg[i]).max(0.02);
                }
            }
        }
        Algo::Es { lambda, sigma0 } => {
            let mut cur = x0.to_vec();
            let mut cv = v0;
            let mut sg = *sigma0;
            while run.left() {
                let (mut succ, mut n) = (0, 0);
                for _ in 0..*lambda {
                    if !run.left() {
                        break;
                    }
                    let mut x: Vec<f32> = (0..d).map(|i| cur[i] + sg * normal(rng)).collect();
                    clamp(&mut x);
                    let v = run.f(&x)?;
                    n += 1;
                    if v > cv {
                        cv = v;
                        cur = x;
                        succ += 1;
                    }
                }
                sg *= if n > 0 && succ as f32 / n as f32 > 0.2 { 1.4 } else { 0.8 };
                sg = sg.clamp(0.01, 1.0);
            }
        }
        Algo::Cma { pop, sigma0 } => {
            let lam = (*pop).max(4);
            let mu_n = lam / 2;
            let w: Vec<f32> = (0..mu_n).map(|i| ((mu_n as f32 + 0.5).ln() - ((i + 1) as f32).ln()).max(0.0)).collect();
            let ws: f32 = w.iter().sum();
            let w: Vec<f32> = w.iter().map(|x| x / ws).collect();
            let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f32>();
            let n = d as f32;
            let cs = (mueff + 2.0) / (n + mueff + 5.0);
            let ds = 1.0 + cs;
            let c1 = 2.0 / ((n + 1.3).powi(2) + mueff) * (n + 2.0) / 3.0;
            let cmu = (2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0).powi(2) + mueff) * (n + 2.0) / 3.0).min(1.0 - c1);
            let cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n);
            let mut m = x0.to_vec();
            let mut c = vec![1.0f32; d];
            let (mut ps, mut pc) = (vec![0.0f32; d], vec![0.0f32; d]);
            let mut sg = *sigma0;
            let chi = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n));
            while run.left() {
                let mut cand: Vec<(f64, (Vec<f32>, Vec<f32>))> = vec![];
                for _ in 0..lam {
                    if !run.left() {
                        break;
                    }
                    let z: Vec<f32> = (0..d).map(|_| normal(rng)).collect();
                    let mut x: Vec<f32> = (0..d).map(|i| m[i] + sg * c[i].sqrt() * z[i]).collect();
                    clamp(&mut x);
                    let v = run.f(&x)?;
                    let y: Vec<f32> = (0..d).map(|i| (x[i] - m[i]) / sg).collect();
                    cand.push((v, (x, y)));
                }
                if cand.len() < lam {
                    break;
                }
                sort_desc(&mut cand);
                let old = m.clone();
                for i in 0..d {
                    m[i] = (0..mu_n).map(|k| w[k] * cand[k].1.0[i]).sum();
                }
                let yw: Vec<f32> = (0..d).map(|i| (m[i] - old[i]) / sg).collect();
                for i in 0..d {
                    ps[i] = (1.0 - cs) * ps[i] + (cs * (2.0 - cs) * mueff).sqrt() * yw[i] / c[i].sqrt();
                    pc[i] = (1.0 - cc) * pc[i] + (cc * (2.0 - cc) * mueff).sqrt() * yw[i];
                    let rank_mu: f32 = (0..mu_n).map(|k| w[k] * cand[k].1.1[i] * cand[k].1.1[i]).sum();
                    c[i] = ((1.0 - c1 - cmu) * c[i] + c1 * pc[i] * pc[i] + cmu * rank_mu).clamp(0.01, 4.0);
                }
                let pn = ps.iter().map(|v| v * v).sum::<f32>().sqrt();
                sg = (sg * ((cs / ds) * (pn / chi - 1.0)).exp()).clamp(0.01, 1.0);
            }
        }
        Algo::Spsa { a, c } => {
            let mut x = x0.to_vec();
            let mut k = 0usize;
            while run.left() {
                let ak = a / ((k + 1) as f32 + 5.0).powf(0.602);
                let ck = c / ((k + 1) as f32).powf(0.101);
                let delta: Vec<f32> = (0..d).map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 }).collect();
                let mut xp: Vec<f32> = (0..d).map(|i| x[i] + ck * delta[i]).collect();
                let mut xm: Vec<f32> = (0..d).map(|i| x[i] - ck * delta[i]).collect();
                clamp(&mut xp);
                clamp(&mut xm);
                if !run.left() {
                    break;
                }
                let fp = run.f(&xp)?;
                if !run.left() {
                    break;
                }
                let fm = run.f(&xm)?;
                // normalised so `a` is in plan units per unit relative gain
                let norm = (fp.abs() + fm.abs()).max(1.0) * 0.5;
                let slope = (fp - fm) / (2.0 * ck as f64) / norm;
                for i in 0..d {
                    x[i] += (ak as f64 * slope / delta[i] as f64) as f32 * 10.0;
                }
                clamp(&mut x);
                k += 1;
                if k % 4 == 0 && run.left() {
                    run.f(&x)?;
                }
            }
        }
        Algo::Fd { step, eps } => {
            let mut x = x0.to_vec();
            let mut xv = v0;
            let mut st = *step;
            while run.left() && st > 0.005 {
                let mut g = vec![0.0f64; d];
                for i in 0..d {
                    if !run.left() {
                        break;
                    }
                    let (mut xp, mut xm) = (x.clone(), x.clone());
                    xp[i] = (xp[i] + eps).min(1.0);
                    xm[i] = (xm[i] - eps).max(-1.0);
                    let fp = run.f(&xp)?;
                    let fm = run.f(&xm)?;
                    g[i] = (fp - fm) / (xp[i] - xm[i]).max(1e-6) as f64;
                }
                let gn = g.iter().map(|v| v * v).sum::<f64>().sqrt();
                if gn < 1e-9 {
                    st *= 0.5;
                    continue;
                }
                let mut improved = false;
                for _ in 0..4 {
                    if !run.left() {
                        break;
                    }
                    let mut xn: Vec<f32> = (0..d).map(|i| x[i] + st * (g[i] / gn) as f32).collect();
                    clamp(&mut xn);
                    let v = run.f(&xn)?;
                    if v > xv {
                        x = xn;
                        xv = v;
                        improved = true;
                        break;
                    }
                    st *= 0.5;
                }
                if !improved {
                    st *= 0.5;
                }
            }
        }
        Algo::Grid { n_ang, n_mag } => {
            let has_mag = run.ev.prob.fix_mag.is_none();
            let stride = if has_mag { 2 } else { 1 };
            let k = run.ev.prob.knots;
            let mags = if has_mag { *n_mag } else { 1 };
            let (mut bx, mut bv) = (x0.to_vec(), v0);
            for ia in 0..*n_ang {
                let a = -1.0 + 2.0 * (ia as f32 + 0.5) / *n_ang as f32;
                for im in 0..mags {
                    if !run.left() {
                        break;
                    }
                    let m = -1.0 + 2.0 * (im as f32 + 0.5) / mags.max(1) as f32;
                    let mut x = vec![0.0; d];
                    for kk in 0..k {
                        x[kk * stride] = a;
                        if has_mag {
                            x[kk * stride + 1] = m;
                        }
                    }
                    let v = run.f(&x)?;
                    if v > bv {
                        bv = v;
                        bx = x;
                    }
                }
            }
            let mut st = 2.0 / *n_ang as f32;
            while run.left() && st > 0.01 {
                let mut moved = false;
                for i in 0..d {
                    for sgn in [1.0f32, -1.0] {
                        if !run.left() {
                            break;
                        }
                        let mut x = bx.clone();
                        x[i] = (x[i] + sgn * st).clamp(-1.0, 1.0);
                        let v = run.f(&x)?;
                        if v > bv {
                            bv = v;
                            bx = x;
                            moved = true;
                        }
                    }
                }
                if !moved {
                    st *= 0.5;
                }
            }
        }
    }
    let c1 = run.ev.cost;
    Ok(OptResult {
        x: run.bx.clone(),
        value: run.bv,
        x0_value: v0,
        cost: Cost { evals: c1.evals - c0.evals, frames: c1.frames - c0.frames, loads: c1.loads - c0.loads },
        trace: run.trace.clone(),
    })
}

// -------------------------------------------------------------- smoothness

#[derive(Clone, Debug, Default)]
pub struct Smooth {
    pub dirs: usize,
    /// Fraction of random directions where |f(x+e d) - f(x)| <= tol.
    pub frac_flat: f64,
    /// Fraction where the change is above `tol` but below `cliff` (usable gradient).
    pub frac_smooth: f64,
    /// Fraction where the change exceeds `cliff` (a jump).
    pub frac_cliff: f64,
    pub median_abs_delta: f64,
}

/// Local smoothness of the score at `x`: perturb along random unit
/// directions by `eps` (plan units) and classify the change.
pub fn smoothness(ev: &mut Evaluator, x: &[f32], eps: f32, dirs: usize, tol: f64, cliff: f64, rng: &mut SmallRng) -> Result<Smooth> {
    let d = x.len();
    let f0 = ev.eval(x)?;
    let (mut flat, mut smooth, mut cl) = (0, 0, 0);
    let mut deltas = vec![];
    for _ in 0..dirs {
        let mut dir: Vec<f32> = (0..d).map(|_| normal(rng)).collect();
        let n = dir.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for v in &mut dir {
            *v /= n;
        }
        let mut xn: Vec<f32> = (0..d).map(|i| x[i] + eps * dir[i]).collect();
        clamp(&mut xn);
        let df = (ev.eval(&xn)? - f0).abs();
        deltas.push(df);
        if df <= tol {
            flat += 1;
        } else if df <= cliff {
            smooth += 1;
        } else {
            cl += 1;
        }
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(Smooth { dirs, frac_flat: flat as f64 / dirs as f64, frac_smooth: smooth as f64 / dirs as f64, frac_cliff: cl as f64 / dirs as f64, median_abs_delta: deltas[deltas.len() / 2] })
}

/// The score along one axis (all other coordinates at `x`), `n` points over [-1, 1].
pub fn scan_axis(ev: &mut Evaluator, x: &[f32], dim: usize, n: usize) -> Result<Vec<(f32, f64)>> {
    let mut out = vec![];
    for i in 0..n {
        let mut xn = x.to_vec();
        xn[dim] = -1.0 + 2.0 * i as f32 / (n - 1).max(1) as f32;
        out.push((xn[dim], ev.eval(&xn)?));
    }
    Ok(out)
}

/// Midpoints and jump sizes where consecutive scan points differ by more than `thr`.
pub fn cliffs(scan: &[(f32, f64)], thr: f64) -> Vec<(f32, f64)> {
    scan.windows(2).filter(|w| (w[1].1 - w[0].1).abs() > thr).map(|w| ((w[0].0 + w[1].0) * 0.5, w[1].1 - w[0].1)).collect()
}

// ---------------------------------------------------------- receding horizon

/// What a policy takes from a refinement: the fine values of this frame (as
/// `types::Choice`'s `fine_ang` in -1..1 and `fine_mag` in 0..1) and the plan.
#[derive(Clone, Debug)]
pub struct Decision {
    pub fine_ang: f32,
    pub fine_mag: f32,
    /// Per-frame (angle offset radians, tilt) over the horizon.
    pub plan: Vec<(f32, f32)>,
    pub x: Vec<f32>,
    /// Cost of the planning step (zero on replayed frames).
    pub cost: Cost,
    pub value: f64,
    pub gain: f64,
    pub replanned: bool,
}

pub struct Refiner {
    pub algo: Algo,
    pub budget: Budget,
    /// Re-plan every k frames; between re-plans the stored plan is replayed.
    pub replan_every: usize,
    warm: Option<Vec<f32>>,
    age: usize,
    last: Option<Decision>,
    /// Total emulator frames spent over all decisions.
    pub total_frames: u64,
    pub decisions: u64,
}

impl Refiner {
    pub fn new(algo: Algo, budget: Budget, replan_every: usize) -> Self {
        Refiner { algo, budget, replan_every: replan_every.max(1), warm: None, age: 0, last: None, total_frames: 0, decisions: 0 }
    }

    /// Call each frame with the evaluator on the current state (its base
    /// savestate is the current one). Re-plans every `replan_every` frames,
    /// warm-started from the previous plan shifted by the frames elapsed;
    /// in between it replays the stored plan (no emulator use).
    pub fn decide(&mut self, ev: &mut Evaluator, rng: &mut SmallRng) -> Result<Decision> {
        let replan = self.will_replan();
        if replan {
            let d = ev.prob.dim();
            let has_mag = ev.prob.fix_mag.is_none();
            let x0 = match self.warm.take() {
                Some(w) if w.len() == d => shift(&w, ev.prob.knots, has_mag, self.age.max(1), ev.prob.horizon),
                _ => vec![0.0; d],
            };
            let r = optimize(&self.algo, ev, &x0, self.budget, rng)?;
            self.total_frames += r.cost.frames;
            self.decisions += 1;
            self.age = 0;
            self.warm = Some(r.x.clone());
            let p = &ev.prob;
            let plan: Vec<(f32, f32)> = (0..p.horizon).map(|t| plan_at(&r.x, p.knots, p.fix_mag, p.ang_max, (p.mag_lo, p.mag_hi), t, p.horizon)).collect();
            self.last = Some(Decision { fine_ang: 0.0, fine_mag: 0.0, plan, x: r.x, cost: r.cost, value: r.value, gain: r.value - r.x0_value, replanned: true });
        }
        let spec = ev.prob.spec();
        Ok(self.emit(&spec, replan))
    }

    /// Whether the next [`Self::decide`] re-plans (needs an evaluator on the
    /// current state); if not, [`Self::replay`] gives the frame's decision
    /// without any emulator use.
    pub fn will_replan(&self) -> bool {
        self.last.is_none() || self.age >= self.replan_every
    }

    /// The stored plan's next frame (call only when `!will_replan()`).
    pub fn replay(&mut self, spec: &Spec) -> Decision {
        self.emit(spec, false)
    }

    fn emit(&mut self, spec: &Spec, replan: bool) -> Decision {
        let mut dec = self.last.clone().unwrap();
        dec.replanned = replan;
        if !replan {
            dec.cost = Cost::default();
        }
        if let Some(&(a, t)) = dec.plan.get(self.age) {
            dec.fine_ang = (a / spec.ang_max.max(1e-6)).clamp(-1.0, 1.0);
            dec.fine_mag = match spec.fix_mag {
                Some(_) => t,
                None => ((t - spec.mag_lo) / (spec.mag_hi - spec.mag_lo).max(1e-6)).clamp(0.0, 1.0),
            };
        }
        self.age += 1;
        dec
    }

    pub fn reset(&mut self) {
        self.warm = None;
        self.age = 0;
        self.last = None;
    }
}

/// Time-shift a plan by `k` frames of an `h`-frame horizon (resample, hold the end).
pub fn shift(x: &[f32], knots: usize, has_mag: bool, k: usize, h: usize) -> Vec<f32> {
    if knots <= 1 || h <= 1 {
        return x.to_vec();
    }
    let stride = if has_mag { 2 } else { 1 };
    let mut out = x.to_vec();
    for kk in 0..knots {
        let t = ((kk as f32 / (knots - 1) as f32) * (h - 1) as f32 + k as f32).min((h - 1) as f32);
        let pos = t / (h - 1) as f32 * (knots - 1) as f32;
        let k0 = (pos.floor() as usize).min(knots - 1);
        let k1 = (k0 + 1).min(knots - 1);
        let w = pos - k0 as f32;
        for c in 0..stride {
            out[kk * stride + c] = x[k0 * stride + c] * (1.0 - w) + x[k1 * stride + c] * w;
        }
    }
    out
}
