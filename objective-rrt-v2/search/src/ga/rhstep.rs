//! A long-lived, resumable rolling-horizon controller for agent-driven runs
//! (objective-rrt-v2/agent, `ar`). The curiosity GA's control (population
//! 128, 600-frame plans, one frame committed per generation, a persistent
//! shifted population, pink noise) with v2's objectives as the fitness: the
//! per-frame reward is the current stage's ETA decrease, discounted by γ per
//! frame, the fitness the best running sum ([`play`]).
//!
//! Unlike [`Worker::attempt_rh`] the population survives objective
//! boundaries: the caller passes the objective window on every
//! [`RhController::step`] and slides it with [`RhController::pop_front`]
//! when the front objective is met. Plans are played in parallel, one game
//! per thread; children and refills are drawn from one rng per generation
//! seeded from (seed, salt, frame, generation in step), so a run does not
//! depend on the thread count and a rewind to a frame replays the same way.
//!
//! A child module of `ga` so it can use its private machinery (play, Gen,
//! breed, rank, Ctx); ga.rs's own attempts are unchanged.

use super::*;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

/// The controller's settings. Defaults: the user's gold standard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RhConfig {
    pub pop: usize,
    /// Frames a plan looks ahead.
    pub h: usize,
    /// Frames committed per step.
    pub k: usize,
    /// Generations per step.
    pub g: usize,
    pub gamma: f64,
    /// Reward for meeting a stage, in ETA frames.
    pub bonus: f64,
    /// Penalty for dying or `Lost`.
    pub death: f64,
    /// Share of a fresh population that is pink noise (1: all, and no
    /// steering seeds or move policies).
    pub pink: f64,
    /// Shifted plans are refilled with pink noise (else v1's random holds).
    pub refill_pink: bool,
    /// Children steer past their inputs where the stage has a target.
    pub steer: bool,
    pub seed: u64,
    /// Mixed into every generation's rng (a new branch from the same frame
    /// explores differently with another salt).
    pub salt: u64,
    /// Committed history kept for overlays and policies (frames).
    pub keep: usize,
    /// The window's backward part: plans start at least this many frames
    /// behind the frontier, at an objective's start (0: plain rolling horizon).
    pub back: usize,
    /// Frames behind the frontier after which, with no success to round
    /// back to, the base moves to `back` behind anyway.
    pub back_max: usize,
    /// Share of the children bred that keep the best plan's inputs up to its
    /// most recent success.
    pub anchor: f64,
    /// Plans press A through dialogs, cutscenes and menus (unless the stage
    /// is a dialog/menu objective).
    pub press: bool,
}

impl Default for RhConfig {
    fn default() -> Self {
        RhConfig {
            pop: 128,
            h: 600,
            k: 1,
            g: 1,
            gamma: 0.99,
            bonus: 300.0,
            death: 600.0,
            pink: 1.0,
            refill_pink: true,
            steer: false,
            seed: 1,
            salt: 0,
            keep: 900,
            back: 0,
            back_max: 3000,
            anchor: 0.5,
            press: true,
        }
    }
}

/// A plan as recorded (the population journal, rewinds).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanRec {
    pub pads: Vec<GamePad>,
    pub raw: u32,
    pub origin: u8,
    pub generation: u16,
    /// Evaluation (NaN / -1 when not evaluated).
    pub fitness: f64,
    pub at: u32,
    pub first_done: i32,
    /// 0 horizon, 1 done, 2 died, 3 lost, 4 cut, 255 not evaluated.
    pub end: u8,
    pub final_eta: f32,
}

pub fn origin_name(o: u8) -> &'static str {
    ORIGINS.get(o as usize).copied().unwrap_or("?")
}

/// What a step did.
pub struct StepOut {
    /// Committed inputs (k frames, fewer where a stage was met).
    pub pads: Vec<GamePad>,
    /// Stages met by this commit (the window's front ones, in order).
    pub met: usize,
    /// The current stage's ETA after the commit (0 when all are met).
    pub eta: f64,
    /// The best plan's fitness and its end.
    pub fitness: f64,
    pub best_end: u8,
    /// The best plan met every stage of the window.
    pub best_done_all: bool,
    /// No frame could be committed (every plan dies or is lost at once).
    pub stuck: bool,
    /// Frames emulated by this step.
    pub frames: u64,
    /// With journaling: each generation's population after evaluation.
    pub gens: Vec<Vec<PlanRec>>,
    /// Offsets into `pads` at which the frozen stages were met.
    pub met_at: Vec<usize>,
}

fn end_code(e: End) -> u8 {
    match e {
        End::Horizon => 0,
        End::Done => 1,
        End::Died => 2,
        End::Lost => 3,
        End::Cut => 4,
    }
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub struct RhController {
    pub cfg: RhConfig,
    rh: Rh,
    /// One per thread; `cur` holds the committed state.
    workers: Vec<Worker>,
    pool: rayon::ThreadPool,
    pop: Vec<Gen>,
    ok: Vec<usize>,
    repolicies: Vec<usize>,
    ctx: Ctx,
    /// The `start` of [`play`]: frame base and lives.
    start: St,
    /// Keys of the window the context was folded with (current stage first).
    keys: Vec<String>,
    /// Generations run so far.
    pub generation: u64,
    /// Generations in the current step (the rng's counter).
    pub journal: bool,
    pub frames: u64,
    /// A prior plan offered for the next step (see [`RhController::set_prior`]).
    prior: Option<Vec<GamePad>>,
    /// Children anchored so far.
    pub anchored: u64,
    /// Frontier offset from the base, the best plan's inputs from the base,
    /// its successes within the frontier and its ETA there.
    n: usize,
    best_pads: Vec<GamePad>,
    best_dones: Vec<usize>,
    best_eta: f64,
}

impl RhController {
    /// At `st` (the state `save` holds, read with objects).
    #[cfg(not(feature = "sm64coin"))]
    pub fn new(game_gen: &sm64_binds::SM64GameGenerator, threads: usize, save: &SaveState, st: St, cfg: RhConfig) -> Result<RhController> {
        Self::with_workers(threads, save, st, cfg, || Worker::new(game_gen))
    }

    /// With feature `sm64coin` the workers are forks of `proto` (a mining
    /// game, perturbed as the verifier replays).
    #[cfg(feature = "sm64coin")]
    pub fn new(proto: &SM64Game, threads: usize, save: &SaveState, st: St, cfg: RhConfig) -> Result<RhController> {
        Self::with_workers(threads, save, st, cfg, || Worker::forked(proto))
    }

    fn with_workers(threads: usize, save: &SaveState, st: St, cfg: RhConfig, mut mk: impl FnMut() -> Result<Worker>) -> Result<RhController> {
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let mut w = mk()?;
            w.cur.load_state(save)?;
            workers.push(w);
        }
        let rh = Self::rh_of(&cfg);
        Ok(RhController {
            cfg,
            rh,
            workers,
            pool,
            pop: Vec::new(),
            ok: Vec::new(),
            repolicies: Vec::new(),
            ctx: Ctx::new(&[], &st),
            start: st,
            keys: Vec::new(),
            generation: 0,
            journal: false,
            frames: 0,
            prior: None,
            anchored: 0,
            n: 0,
            best_pads: Vec::new(),
            best_dones: Vec::new(),
            best_eta: f64::INFINITY,
        })
    }

    fn rh_of(cfg: &RhConfig) -> Rh {
        Rh {
            pop: cfg.pop.max(2),
            h: cfg.h.max(1),
            k: cfg.k.max(1),
            g: cfg.g.max(1),
            g_precise: cfg.g.max(1),
            gamma: cfg.gamma,
            bonus: cfg.bonus,
            death: cfg.death,
            stall: usize::MAX,
            hot: 1,
            maxlen: usize::MAX,
            through: true,
            press: cfg.press,
            flat: 0,
        }
    }

    /// Changes the settings (population size and horizon apply from the
    /// next breed / refill).
    pub fn set_config(&mut self, cfg: RhConfig) {
        self.rh = Self::rh_of(&cfg);
        self.cfg = cfg;
    }

    /// The committed state (with objects).
    pub fn state(&self) -> &St {
        self.ctx.last()
    }

    pub fn save_state(&self) -> SaveState {
        self.workers[0].cur.save_state()
    }

    pub fn game(&self) -> &SM64Game {
        &self.workers[0].cur
    }

    /// The current stage's ETA and the window's progress.
    pub fn eta(&self) -> f64 {
        self.ctx.eta
    }

    /// The current stage's lowest ETA since it started.
    pub fn peak(&self) -> f64 {
        self.ctx.peak.0
    }

    /// Moves the committed state to (`save`, `st`) (a rewind, a pressed-
    /// through cutscene), dropping the context; the population is kept
    /// unless `fresh` (then a new one is made at the next step).
    pub fn load(&mut self, save: &SaveState, st: St, fresh: bool) -> Result<()> {
        for w in &mut self.workers {
            w.cur.load_state(save)?;
        }
        self.start = st.clone();
        self.ctx = Ctx::new(&[], &st);
        self.keys.clear();
        self.n = 0;
        self.best_pads.clear();
        self.best_dones.clear();
        self.best_eta = f64::INFINITY;
        if fresh {
            self.pop.clear();
        } else {
            for g in &mut self.pop {
                g.eval = None;
            }
        }
        Ok(())
    }

    /// Steps `pads` on the committed state without the GA (a press-through),
    /// returning the states after each. The context is refolded at the end;
    /// the population is shifted past them.
    /// (With `back` > 0 the tentative route is dropped: use `step`'s
    /// `forced` inputs there instead.)
    pub fn play_fixed(&mut self, pads: &[GamePad], objects_last: bool) -> Result<Vec<St>> {
        self.n = 0;
        self.best_pads.clear();
        self.best_dones.clear();
        let mut out = Vec::with_capacity(pads.len());
        let base = self.ctx.last().frame;
        let results: Vec<Result<Vec<St>>> = self.pool.install(|| {
            self.workers
                .par_iter_mut()
                .enumerate()
                .map(|(i, w)| {
                    let mut v = Vec::new();
                    for (f, p) in pads.iter().enumerate() {
                        w.cur.step_game(*p)?;
                        if i == 0 {
                            let last = f + 1 == pads.len();
                            v.push(read_state(&w.cur, base + f as u32 + 1, last && objects_last));
                        }
                    }
                    Ok(v)
                })
                .collect()
        });
        for (i, r) in results.into_iter().enumerate() {
            let v = r?;
            if i == 0 {
                out = v;
            }
        }
        self.frames += (pads.len() * self.workers.len()) as u64;
        if let Some(last) = out.last() {
            let last = if objects_last { last.clone() } else { read_state(&self.workers[0].cur, last.frame, true) };
            self.start = last.clone();
            self.ctx = Ctx::new(&[], &last);
            self.keys.clear();
        }
        let n = pads.len();
        let mut rng = SmallRng::seed_from_u64(mix(self.cfg.seed ^ mix(self.cfg.salt ^ mix(base as u64 ^ 0xf1e1d))));
        let (pink, h) = (self.cfg.refill_pink, self.rh.h);
        for g in &mut self.pop {
            shift(g, n, h, pink, &mut rng);
        }
        Ok(out)
    }


    /// Drops the window's first `n` stages (met; the caller slides its
    /// window by as many).
    pub fn pop_front(&mut self, n: usize) {
        let n = n.min(self.ctx.stage).min(self.keys.len());
        self.ctx.stage -= n;
        self.ctx.ends.drain(..n.min(self.ctx.ends.len()));
        self.keys.drain(..n);
    }

    /// Stages met so far in the current window (not yet popped).
    pub fn met(&self) -> usize {
        self.ctx.stage
    }

    /// The population as recorded.
    pub fn population(&self) -> Vec<PlanRec> {
        self.pop.iter().map(rec).collect()
    }

    /// Replaces the population (a journal generation, another branch's).
    pub fn set_population(&mut self, plans: &[PlanRec]) {
        self.pop = plans
            .iter()
            .map(|p| {
                let mut g = Gen::new(p.pads.clone(), None, p.origin);
                g.raw = p.raw as usize;
                g.generation = p.generation;
                g
            })
            .collect();
    }

    /// Adds plans (e.g. inputs the agent wrote) to the population, played
    /// as recorded (no overlays on them), kept by the next breed only if fit.
    pub fn inject(&mut self, plans: Vec<Vec<GamePad>>) {
        for p in plans {
            let raw = p.len();
            let mut g = Gen::new(p, None, ORIGIN_SEED);
            g.raw = raw;
            self.pop.push(g);
        }
    }

    /// For the next step only: a prior plan (e.g. the incumbent route's
    /// inputs from here), played as recorded and added to the population.
    pub fn set_prior(&mut self, pads: Vec<GamePad>) {
        self.prior = (!pads.is_empty()).then_some(pads);
    }

    /// Folds the context for this window if it changed below the current
    /// stage (a new front, an edit): from the committed state, fresh scratch.
    fn sync_window(&mut self, stages: &[&dyn Objective]) {
        let keys: Vec<String> = stages.iter().map(|o| o.key()).collect();
        let upto = (self.ctx.stage + 1).min(keys.len());
        let same = self.keys.len() >= upto && self.keys[..upto] == keys[..upto];
        if !same {
            let last = self.ctx.last().clone();
            let mut c = Ctx::new(stages, &last);
            // Keep the committed history (overlays read it); only the
            // window's scratch restarts.
            let drop = std::mem::replace(&mut self.ctx, Ctx::new(&[], &last));
            c.pads = drop.pads;
            c.traj = drop.traj;
            let base = c.traj.len() - 1;
            c.ends = c.ends.iter().map(|_| base).collect();
            c.peak = (c.eta, base);
            self.ctx = c;
            for g in &mut self.pop {
                g.eval = None;
            }
        }
        self.keys = keys;
    }

    /// Trims the committed history to `keep` frames (overlays and policies
    /// see that much): the context is cloned into every play that needs it.
    fn trim(&mut self) {
        let keep = self.cfg.keep.max(self.rh.h);
        let n = self.ctx.pads.len();
        if n <= keep + keep / 2 {
            return;
        }
        let d = n - keep;
        self.ctx.pads.drain(..d);
        self.ctx.traj.drain(..d);
        for e in &mut self.ctx.ends {
            *e = e.saturating_sub(d);
        }
        self.ctx.peak.1 = self.ctx.peak.1.saturating_sub(d);
        self.start.frame += d as u32;
    }

    fn rng(&self, j: usize) -> SmallRng {
        let f = self.ctx.last().frame as u64;
        SmallRng::seed_from_u64(mix(self.cfg.seed ^ mix(self.cfg.salt ^ mix(f ^ mix(j as u64 + 1)))))
    }

    /// Plays every plan not yet evaluated for `len` frames from the base.
    fn evaluate(&mut self, stages: &[&dyn Objective], len: usize, rh: &Rh) -> Result<()> {
        let todo: Vec<usize> = (0..self.pop.len()).filter(|&i| self.pop[i].eval.is_none()).collect();
        let (pop, ctx, start) = (&self.pop, &self.ctx, &self.start);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: Vec<Result<Vec<(usize, Eval)>>> = self.pool.install(|| {
            self.workers
                .par_iter_mut()
                .map(|w| {
                    let mut out = Vec::new();
                    loop {
                        let j = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&i) = todo.get(j) else { break };
                        let pl = &pop[i];
                        let l = if pl.full { pl.pads.len().max(len) } else { len };
                        w.tmp.sync_from(&w.cur)?;
                        let src = Src::Plan { pads: &pl.pads, raw: pl.raw, steer: pl.steer, policy: pl.policy.as_ref() };
                        let e = play(&mut w.tmp, &mut None, stages, start, ctx, src, l, false, rh)?;
                        out.push((i, e));
                    }
                    Ok(out)
                })
                .collect()
        });
        for r in results {
            for (i, e) in r? {
                self.frames += e.stepped as u64;
                self.pop[i].take(e);
            }
        }
        Ok(())
    }

    /// Restores a tentative route (a daemon restart): it becomes a plan and
    /// the best plan, and the frontier is set to its end.
    pub fn set_tentative(&mut self, pads: Vec<GamePad>) {
        if pads.is_empty() {
            return;
        }
        let mut g = Gen::new(pads.clone(), None, ORIGIN_SEED);
        g.raw = pads.len();
        self.pop.push(g);
        self.n = pads.len();
        self.best_pads = pads;
    }

    /// The tentative route: the best plan's inputs from the base to the
    /// frontier (`frontier()` frames), as of the last step.
    pub fn tentative(&self) -> &[GamePad] {
        &self.best_pads[..self.n.min(self.best_pads.len())]
    }

    /// `frozen` (the inputs frozen so far) followed by the tentative route.
    pub fn state_frame_pads(&self, frozen: &[GamePad]) -> Vec<GamePad> {
        let mut v = frozen.to_vec();
        v.extend_from_slice(self.tentative());
        v
    }

    /// Frames from the base (the last frozen state) to the frontier.
    pub fn frontier(&self) -> usize {
        self.n
    }

    /// Offsets from the base at which the best plan met stages within the
    /// frontier (the tentative successes), and its ETA at the frontier.
    pub fn tentative_dones(&self) -> &[usize] {
        &self.best_dones
    }

    pub fn frontier_eta(&self) -> f64 {
        self.best_eta
    }

    /// The state at the frontier (replayed on a scratch game, with objects),
    /// and its save state.
    pub fn frontier_state(&mut self) -> Result<(St, SaveState)> {
        let w = &mut self.workers[0];
        w.tmp.sync_from(&w.cur)?;
        let n = self.n.min(self.best_pads.len());
        for p in &self.best_pads[..n] {
            w.tmp.step_game(*p)?;
        }
        let f = self.ctx.last().frame + n as u32;
        Ok((read_state(&w.tmp, f, true), w.tmp.save_state()))
    }

    /// One step. The plans start at the base B (the last frozen state) and
    /// are played to the frontier n plus the horizon h, so every generation
    /// may still change the frames behind the frontier, including those of
    /// objectives the best plan already met there (the window's backward
    /// part). Fitness: [`play`]'s discounted ETA decrease from B, judged at
    /// or after the frontier. `forced`: inputs every plan takes at the
    /// frontier this step (a press-through, a menu program); else the
    /// frontier moves on by one frame.
    ///
    /// Freezing: with `back` 0, the frames up to the frontier (one, or to
    /// where a stage was met) are frozen every step, the plain rolling
    /// horizon. Otherwise B moves to the latest stage success of the best
    /// plan that is at least `back` frames behind the frontier (the start of
    /// the objective after it), so the window keeps at least `back` frames
    /// behind, rounded back to an objective's start; past `back_max` frames
    /// without one, to `back` behind. Frozen frames are committed and the
    /// stages met in them counted in [`RhController::met`].
    ///
    /// Breeding: in a share `anchor` of the children bred, the inputs up to
    /// the best plan's most recent success are its own, so that part of the
    /// population only varies after it.
    pub fn step(&mut self, stages: &[&dyn Objective], forced: Option<&[GamePad]>) -> Result<StepOut> {
        anyhow::ensure!(!stages.is_empty(), "rhstep: empty window");
        let f0 = self.frames;
        self.start.lives = self.ctx.last().lives;
        self.sync_window(stages);
        self.trim();
        let h = self.rh.h;
        // Stages already met where the window was (re)folded (a door passed
        // in a press-through): nothing to play until the caller pops them.
        if self.ctx.stage > 0 {
            return Ok(StepOut { pads: Vec::new(), met: self.ctx.stage, eta: self.ctx.eta, fitness: 0.0, best_end: 1, best_done_all: self.ctx.stage >= stages.len(), stuck: false, frames: 0, gens: Vec::new(), met_at: Vec::new() });
        }
        let target = self.cfg.steer && stages[0].steer_target(self.ctx.last()).is_some();
        if self.pop.is_empty() {
            let mut rng = self.rng(0);
            let last = self.ctx.last().clone();
            let mut rh0 = self.rh;
            rh0.h = h + self.n;
            let (pop, _plans, ok) = self.workers[0].gen0(stages, &last, &mut rng, &rh0, self.cfg.pink, None, None)?;
            self.pop = pop;
            self.ok = ok;
            // Keep the tentative route: the new population starts from it.
            if !self.best_pads.is_empty() {
                let mut g = Gen::new(self.best_pads.clone(), None, ORIGIN_SEED);
                g.raw = self.n.min(self.best_pads.len());
                self.pop.push(g);
            }
        }
        // The frontier moves on.
        let adv = match forced {
            Some(p) if !p.is_empty() => {
                let (a, b) = (self.n, self.n + p.len());
                for g in &mut self.pop {
                    if g.pads.len() < b {
                        g.pads.resize(b, GamePad::default());
                    }
                    g.pads[a..b].copy_from_slice(p);
                    g.raw = g.raw.max(b);
                    g.eval = None;
                }
                p.len()
            }
            _ => 1,
        };
        self.n += adv;
        let n = self.n;
        let len = n + h;
        let mut rng_fill = self.rng(0x5eed);
        for g in &mut self.pop {
            if g.pads.len() < len && g.steer.is_none() && g.policy.is_none() {
                let k = len - g.pads.len();
                g.pads.extend(crate::noise::pink_inputs(&mut rng_fill, k));
                g.eval = None;
            }
        }
        // Judged at or after the frontier.
        let mut rh = self.rh;
        rh.k = n.max(1);
        // Behind the frontier: net progress (undiscounted); past it, γ per frame.
        rh.flat = n.saturating_sub(1);
        let breed_ctl = |s: &Self| Breed { target, h: len, ok: &[], lineage_cap: crate::movegenome::enabled() && s.ok.is_empty() };
        let mut gens = Vec::new();
        for gi in 0..self.rh.g {
            self.evaluate(stages, len, &rh)?;
            self.generation += 1;
            if self.journal {
                gens.push(self.population());
            }
            if gi + 1 < self.rh.g {
                let mut rng = self.rng(gi + 1);
                let b = breed_ctl(self);
                let ok = self.ok.clone();
                let b = Breed { ok: &ok, ..b };
                breed(&mut self.pop, self.rh.pop, &mut rng, &b, self.generation.min(u16::MAX as u64) as u16, &mut self.repolicies);
                self.anchor_best(&mut rng);
                self.evaluate(stages, len, &rh)?;
            }
        }
        if let Some(pads) = self.prior.take() {
            let raw = pads.len();
            let mut g = Gen::new(pads, None, ORIGIN_SEED);
            g.raw = raw;
            g.full = true;
            self.pop.push(g);
            self.evaluate(stages, len, &rh)?;
        }
        let bi = (0..self.pop.len())
            .filter(|&i| self.pop[i].eval.is_some())
            .max_by(|&a, &b| rank(self.pop[a].eval.as_ref().unwrap(), self.pop[b].eval.as_ref().unwrap()))
            .unwrap();
        let best = self.pop[bi].eval.as_ref().unwrap();
        let (fitness, best_end) = (best.fitness, end_code(best.end));
        let best_done_all = best.end == End::Done;
        self.best_pads = best.pads.clone();
        if self.best_pads.len() < self.pop[bi].pads.len() {
            self.best_pads.extend_from_slice(&self.pop[bi].pads[best.pads.len()..]);
        }
        self.best_dones = best.dones.iter().copied().filter(|&d| d <= n).collect();
        self.best_eta = if best.end == End::Done {
            0.0
        } else if best.pads.len() >= n { best.eta.get(n.saturating_sub(1)).copied().unwrap_or(self.ctx.eta) } else { f64::INFINITY };
        let reach = best.pads.len();
        // How far to freeze.
        let back = self.cfg.back;
        let c = if back == 0 {
            let mut c = n.min(reach);
            if let Some(&d) = best.dones.first() {
                c = c.min(d);
            }
            c
        } else {
            let mut c = self.best_dones.iter().copied().filter(|&d| n - d >= back).max().unwrap_or(0);
            if c == 0 && n > self.cfg.back_max.max(back) {
                c = n - back;
            }
            c.min(reach)
        };
        let stuck = reach < n && c == 0 && back == 0;
        let mut out = StepOut { pads: Vec::new(), met: 0, eta: self.best_eta, fitness, best_end, best_done_all, stuck, frames: 0, gens, met_at: Vec::new() };
        // Breed on this generation's ranking (anchored to the best plan's
        // most recent success), then freeze and shift.
        let mut rng = self.rng(self.rh.g + 1);
        let b = breed_ctl(self);
        let ok = self.ok.clone();
        let b = Breed { ok: &ok, ..b };
        breed(&mut self.pop, self.rh.pop, &mut rng, &b, self.generation.min(u16::MAX as u64) as u16, &mut self.repolicies);
        self.anchor_best(&mut rng);
        if c == 0 {
            out.frames = self.frames - f0;
            return Ok(out);
        }
        let pads = self.best_pads[..c].to_vec();
        let met0 = self.ctx.stage;
        let (first, rest) = self.workers.split_at_mut(1);
        let (ctx, start) = (&self.ctx, &self.start);
        let rh_c = self.rh;
        let (e0, others): (Result<Eval>, Vec<Result<()>>) = self.pool.install(|| {
            rayon::join(
                || play(&mut first[0].cur, &mut None, stages, start, ctx, Src::Replay(&pads), pads.len(), true, &rh_c),
                || {
                    rest.par_iter_mut()
                        .map(|w| {
                            for p in &pads {
                                w.cur.step_game(*p)?;
                            }
                            Ok(())
                        })
                        .collect()
                },
            )
        });
        for r in others {
            r?;
        }
        let e0 = e0?;
        self.frames += e0.stepped as u64 * self.workers.len() as u64;
        anyhow::ensure!(e0.pads.len() == pads.len(), "rhstep: the frozen frames did not replay as played ({} of {})", e0.pads.len(), pads.len());
        self.ctx.absorb(e0);
        let f = self.ctx.last().frame;
        *self.ctx.traj.last_mut().unwrap() = read_state(&self.workers[0].cur, f, true);
        out.met = self.ctx.stage - met0;
        out.pads = pads;
        out.met_at = self.best_dones.iter().copied().filter(|&d| d <= c).collect();
        // Everything is now relative to the new base.
        self.n -= c;
        self.best_pads.drain(..c);
        self.best_dones = self.best_dones.iter().filter(|&&d| d > c).map(|&d| d - c).collect();
        for g in &mut self.pop {
            let k = c.min(g.pads.len());
            g.pads.drain(..k);
            g.raw = g.raw.saturating_sub(c);
            g.eval = None;
            g.full = false;
            g.policy = None;
            g.steer = None;
        }
        out.frames = self.frames - f0;
        Ok(out)
    }

    /// The share `anchor` of the children past the elite take the best
    /// plan's inputs up to its most recent success within the frontier.
    fn anchor_best(&mut self, rng: &mut SmallRng) {
        let share = self.cfg.anchor;
        let Some(&keep) = self.best_dones.iter().max() else { return };
        if share <= 0.0 || keep == 0 || self.best_pads.len() < keep {
            return;
        }
        let elite = (self.rh.pop / 4).max(1).min(self.pop.len());
        for g in self.pop[elite..].iter_mut() {
            if !rng.random_bool(share.min(1.0)) {
                continue;
            }
            if g.pads.len() < keep {
                g.pads.resize(keep, GamePad::default());
            }
            g.pads[..keep].copy_from_slice(&self.best_pads[..keep]);
            g.raw = g.raw.max(keep);
            g.eval = None;
            self.anchored += 1;
        }
    }
}

/// Moves a plan past `c` committed frames, refilled to `h` (pink or v1's holds).
fn shift(g: &mut Gen, c: usize, h: usize, pink: bool, rng: &mut SmallRng) {
    if pink {
        g.shift(c, 0, rng);
        if g.pads.len() < h {
            let n = h - g.pads.len();
            g.pads.extend(crate::noise::pink_inputs(rng, n));
        }
    } else {
        g.shift(c, h, rng);
    }
}

fn rec(g: &Gen) -> PlanRec {
    let (fitness, at, first_done, end, final_eta) = match &g.eval {
        Some(e) => (e.fitness, e.at as u32, e.dones.first().map_or(-1, |&d| d as i32), end_code(e.end), e.fin.eta as f32),
        None => (f64::NAN, 0, -1, 255, f32::NAN),
    };
    PlanRec { pads: g.pads.clone(), raw: g.raw as u32, origin: g.origin, generation: g.generation, fitness, at, first_done, end, final_eta }
}
