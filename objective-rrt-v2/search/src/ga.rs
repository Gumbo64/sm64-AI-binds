//! Step 5: attempting objectives from one start state.
//!
//! By default with v1's open-loop GA over raw per-frame inputs
//! ([`Worker::attempt_open`]), whose only change is its fitness: the
//! discounted decrease of the stages' ETAs (GOALS.md); or the objective's
//! fixed macro or policy, played once. With `OBRRT_RH=1`, a rolling-horizon
//! controller instead ([`Worker::attempt_rh`]): a persistent population of
//! plans from the committed state, scored the same way, the best plan's
//! first `OBRRT_RH_K` frames committed, every plan shifted and re-planned
//! (the curiosity GA's loop, rust/src/population.rs).

/// The agent tools' long-lived rolling-horizon controller (objective-rrt-v2/agent).
pub mod rhstep;

use std::cmp::Ordering;
use std::f32::consts::TAU;
use std::sync::OnceLock;

use anyhow::Result;
use obrrt_core::{Budget, Eta, GaBudget, GamePad, Mem, Objective, PadOverlay, PadPolicy, Precision, St, read_state};
use rand::Rng;
use rand::rngs::SmallRng;
use sm64_binds::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use crate::game::{Game as SM64Game, GameSave as SaveState};

/// Most frames the automatic tail spends pressing A through menus, dialog and
/// warps before giving up.
const TAIL_LIMIT: usize = 1500;

/// The rolling-horizon controller's settings (`OBRRT_RH_*`, read once).
#[derive(Debug, Clone, Copy)]
pub struct Rh {
    /// Plans in the population.
    pub pop: usize,
    /// Frames a plan looks ahead.
    pub h: usize,
    /// Frames committed per re-plan.
    pub k: usize,
    /// Generations per commit, and with any `Precise` stage.
    pub g: usize,
    pub g_precise: usize,
    /// Discount per frame.
    pub gamma: f64,
    /// Reward for meeting a stage, in ETA frames.
    pub bonus: f64,
    /// Penalty for dying or losing the objective, in ETA frames.
    pub death: f64,
    /// Commits without a new lowest ETA before giving up.
    pub stall: usize,
    /// Generations before the first commit (the population is new).
    pub hot: usize,
    /// Cap on the committed length (frames).
    pub maxlen: usize,
    /// Plays go on past a stage met where Mario isn't controllable (agent
    /// tools' lagged window, ga/rhstep.rs); off: such a play is cut there.
    pub through: bool,
    /// In dialogs, cutscenes and menus (but for a modal stage) plays press A
    /// on alternate frames instead of their own inputs (agent tools).
    pub press: bool,
    /// The discount starts after this many frames of a play (agent tools:
    /// the frames behind the frontier count undiscounted); 0: from the start.
    pub flat: usize,
}

/// `OBRRT_RH=1`: attempts use the rolling-horizon controller instead of
/// v1's open-loop GA (the default; see [`Worker::attempt_open`]).
pub fn rh_on() -> bool {
    obrrt_core::config::current().research.rh.on
}

/// [`Rh`] from the environment.
pub fn rh() -> &'static Rh {
    obrrt_core::per_config!(Rh, |cfg| {
        let c = &cfg.research.rh;
        Rh {
            pop: c.pop.max(2),
            h: c.h.max(1),
            k: c.k.max(1),
            g: c.g.max(1),
            g_precise: c.g_precise.max(1),
            gamma: c.gamma,
            bonus: c.bonus,
            death: c.death,
            stall: c.stall.max(1),
            hot: c.hot.max(1),
            maxlen: c.maxlen.max(60),
            through: false,
            press: false,
            flat: 0,
        }
    })
}

/// One worker's games: `base` holds the start state, `cur` the committed
/// state of the attempt under way (reset from `base` at its start), and
/// `tmp` is reset from `cur` (cheaply, by dirty pages) for every plan.
pub struct Worker {
    pub base: SM64Game,
    pub cur: SM64Game,
    pub tmp: SM64Game,
}

impl Worker {
    #[cfg(not(feature = "hooked"))]
    pub fn new(game_gen: &sm64_binds::SM64GameGenerator) -> Result<Worker> {
        Ok(Worker { base: game_gen.create_game()?, cur: game_gen.create_game()?, tmp: game_gen.create_game()? })
    }

    /// The state tree and attempts over raw-memory saves are default-only.
    #[cfg(feature = "hooked")]
    pub fn new(_: &sm64_binds::SM64GameGenerator) -> Result<Worker> {
        anyhow::bail!("Worker::new: unavailable with feature hooked (use Worker::forked)")
    }

    /// A worker whose games are forks of `proto` (a hooked game carries its
    /// hook's state, so it can't come from the generator).
    #[cfg(feature = "hooked")]
    pub fn forked(proto: &SM64Game) -> Result<Worker> {
        Ok(Worker { base: proto.fork()?, cur: proto.fork()?, tmp: proto.fork()? })
    }
}


/// One objective's part of an attempt. `S` is how the end's save state is
/// kept: a whole [`SaveState`], or (the tree's attempts, see
/// [`Attempt::map_saves`]) packed as a delta right in the worker.
pub struct Segment<S = SaveState> {
    /// Inputs from the previous segment's end (or the start) to this one's
    /// chosen state; the last segment also has the automatic tail if needed.
    pub inputs: Vec<GamePad>,
    /// The chosen state (success, or the lowest ETA reached), read with
    /// objects. None when the segment is empty or the tail failed.
    pub end: Option<(St, S)>,
    pub success: bool,
    /// Minus the objective's ETA (frames) at the end: 0 for a success.
    pub progress: f64,
    /// The controller's plan beyond the end: the best plan's inputs not yet
    /// committed when the segment ended (seeds the next attempt from the
    /// node made of it; empty for fixed plays and after an automatic tail).
    pub tail: Vec<GamePad>,
    /// How many of `inputs` the controller (or fixed play) committed: the
    /// rest is the automatic tail. For a success, the frames to `Done`.
    pub played: usize,
}

/// How the controller's committed inputs came about (for diagnostics).
#[derive(Debug, Clone, Copy)]
pub struct GaInfo {
    /// Generations evaluated, over all commits.
    pub generations: usize,
    /// v2: commits made.
    pub winner_gen: u16,
    /// What made most of the committed plans: "seeded" (controller),
    /// "random", "tail" (carried), or the operator.
    pub winner_origin: &'static str,
    /// (stages met, fitness) of the first evaluation's best plan; (stages
    /// met, minus the final ETA) of the committed inputs.
    pub gen0: (usize, f64),
    pub best: (usize, f64),
    /// Frames the winner spent in each of [`MOVES`].
    pub moves: [u16; 7],
    /// Seeded attempts ([`Worker::attempt_seeded`]): whether the seed genome
    /// itself met every stage, and at which frame of the attempt.
    pub seed: Option<(bool, u32)>,
}

/// Fast or skilled movement actions counted in winners: long jump, dive,
/// wall kick, triple jump, side flip, backflip, slide kick.
pub const MOVES: [(&str, u32); 7] = [
    ("long_jump", 0x03000888),
    ("dive", 0x0188088A),
    ("wall_kick", 0x03000886),
    ("triple_jump", 0x01000882),
    ("side_flip", 0x01000887),
    ("backflip", 0x01000883),
    ("slide_kick", 0x018008AA),
];

pub const ORIGINS: [&str; 16] = [
    "seeded", "random", "span", "turn", "a-toggle", "resteer", "press", "pink", "key", "backseed", "policy", "repolicy", "model",
    "repair", "crossover", "tail",
];

/// [`ORIGINS`] indices.
const ORIGIN_RANDOM: u8 = 1;
const ORIGIN_PINK: u8 = 7;
const ORIGIN_KEY: u8 = 8;
const ORIGIN_SEED: u8 = 9;
const ORIGIN_POLICY: u8 = 10;
const ORIGIN_REPOLICY: u8 = 11;
const ORIGIN_CROSSOVER: u8 = 14;
const ORIGIN_TAIL: u8 = 15;

thread_local! {
    /// The next attempt's carried plan (the start node's [`Segment::tail`]),
    /// set by the engine.
    static TAIL: std::cell::RefCell<Option<Vec<GamePad>>> = const { std::cell::RefCell::new(None) };
}

/// A rollout that met all its stages, for [`set_met_sink`].
#[derive(Clone)]
pub struct MetRollout {
    pub genome: Vec<GamePad>,
    /// The state at the last stage's Done (no objects).
    pub end: St,
}

thread_local! {
    /// The route mender (spec/src/bin/mend.rs): while set, every rollout on
    /// this thread that meets all its stages is kept here, whichever survive
    /// selection (the capacity keeps the soonest). None otherwise: nothing
    /// is recorded and no draw or decision depends on it.
    static MET_SINK: std::cell::RefCell<Option<(usize, Vec<MetRollout>)>> = const { std::cell::RefCell::new(None) };
}

/// Starts collecting met rollouts on this thread, at most `cap`.
pub fn set_met_sink(cap: usize) {
    MET_SINK.with(|s| *s.borrow_mut() = Some((cap, Vec::new())));
}

/// Stops collecting; the met rollouts, soonest first.
pub fn take_met_sink() -> Vec<MetRollout> {
    MET_SINK.with(|s| s.borrow_mut().take()).map_or(Vec::new(), |(_, v)| v)
}

fn sink_met(genome: &[GamePad], end: &St) {
    MET_SINK.with(|s| {
        let mut g = s.borrow_mut();
        let Some((cap, v)) = g.as_mut() else { return };
        let at = v.partition_point(|m| m.genome.len() <= genome.len());
        if at >= *cap || v.iter().any(|m| m.genome.len() == genome.len() && m.genome == genome) {
            return;
        }
        let mut end = end.clone();
        end.objects = None;
        v.insert(at, MetRollout { genome: genome.to_vec(), end });
        v.truncate(*cap);
    });
}

/// Sets the carried plan for the next attempt on this thread.
pub fn set_tail(t: Option<Vec<GamePad>>) {
    TAIL.with(|r| *r.borrow_mut() = t);
}

/// A source of whole-attempt seeds that does not live in this crate (the
/// reactive policy inventory, `obrrt-policy`, which depends on this crate).
/// Given the attempt's start (`base` is free to be played on and is restored
/// afterwards), it returns candidate pad sequences from the start, best first,
/// with the emulator frames each cost to find.
pub trait SeedSource: Send + Sync {
    fn seeds(
        &self,
        base: &mut SM64Game,
        start: &St,
        save: &SaveState,
        stage: &dyn Objective,
        rng: &mut SmallRng,
    ) -> Vec<(Vec<GamePad>, u64)>;
}

static SEED_SOURCE: std::sync::RwLock<Option<std::sync::Arc<dyn SeedSource>>> = std::sync::RwLock::new(None);

/// Installs (or removes) the process-wide [`SeedSource`]. It is only
/// consulted when `OBRRT_POLICY_SEED=1` (or [`set_policy_seed`] turned it on).
pub fn set_seed_source(s: Option<std::sync::Arc<dyn SeedSource>>) {
    *SEED_SOURCE.write().unwrap() = s;
}

static POLICY_SEED_FORCED: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// API switch for the policy seeds: Some(b) overrides the environment,
/// None restores `OBRRT_POLICY_SEED`.
pub fn set_policy_seed(on: Option<bool>) {
    POLICY_SEED_FORCED.store(on.map_or(-1, |b| b as i8), std::sync::atomic::Ordering::Relaxed);
}

/// `OBRRT_POLICY_SEED=0|1` (0): attempts ask the [`SeedSource`] (when one is
/// installed). Without a seed of their own they take its best plan as the
/// seed; with one (the route's inputs) it stays and one policy plan is added
/// as an extra first-generation individual. `OBRRT_POLICY_SEED_EXTRA` (0)
/// more plans are added in either case. The
/// seed's emulator frames are added to the attempt's.
pub fn policy_seed_on() -> bool {
    match POLICY_SEED_FORCED.load(std::sync::atomic::Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            static ON: OnceLock<bool> = OnceLock::new();
            *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POLICY_SEED").is_ok_and(|v| v == "1"))
        }
    }
}

fn policy_seed_extra() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POLICY_SEED_EXTRA").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// A policy-driven breeding operator that does not live in this crate (the
/// reactive policy inventory): `repolicy` of the standalone hybrid. One
/// [`BreedSession`] per GA attempt (so what it learns about the start stays
/// in the attempt).
pub trait BreedHook: Send + Sync {
    fn session(&self) -> Box<dyn BreedSession>;
}

pub trait BreedSession {
    /// A child of a parent: `prefix` is the parent's pads up to the cut and
    /// `states` its trajectory over them (`states[0]` the attempt's start,
    /// `states.len() == prefix.len() + 1`). `game` is free to use: the hook
    /// replays `prefix` from `save` (the attempt's start) to the cut state,
    /// lets a policy play at most `horizon - prefix.len()` frames from there
    /// and returns the realised pads (prefix included) and the emulator
    /// frames it spent (prefix replay included). `None`: no child.
    #[allow(clippy::too_many_arguments)]
    fn breed(
        &mut self,
        game: &mut SM64Game,
        start: &St,
        save: &SaveState,
        stage: &dyn Objective,
        prefix: &[GamePad],
        states: &[St],
        horizon: usize,
        rng: &mut SmallRng,
    ) -> Option<(Vec<GamePad>, u64)>;
}

static BREED_HOOK: std::sync::RwLock<Option<std::sync::Arc<dyn BreedHook>>> = std::sync::RwLock::new(None);

/// Installs (or removes) the process-wide [`BreedHook`]. It is only used
/// when `OBRRT_POLICY_REPOLICY` is above 0 (or [`set_policy_repolicy`]).
pub fn set_breed_hook(h: Option<std::sync::Arc<dyn BreedHook>>) {
    *BREED_HOOK.write().unwrap() = h;
}

static REPOLICY_FORCED: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(-1);

/// API override of `OBRRT_POLICY_REPOLICY` (None restores the environment).
pub fn set_policy_repolicy(p: Option<f64>) {
    REPOLICY_FORCED.store(p.map_or(-1, |p| (p * 1e6) as i64), std::sync::atomic::Ordering::Relaxed);
}

/// `OBRRT_POLICY_REPOLICY=p` (0, off): per-offspring probability of the
/// policy breeding operator (a parent's pads to a random cut, then an
/// inventory policy plays on; the realised pads are the child; origin
/// `repolicy`). With `OBRRT_POLICY_REPOLICY_MARKET=1` the probability is
/// `2 p` times the operator's share of the credit (score gained per
/// emulator frame, EMA over generations, against all other operators),
/// within [p/4, 4p].
pub fn policy_repolicy() -> f64 {
    let f = REPOLICY_FORCED.load(std::sync::atomic::Ordering::Relaxed);
    if f >= 0 {
        return f as f64 / 1e6;
    }
    static P: OnceLock<f64> = OnceLock::new();
    *P.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POLICY_REPOLICY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}

fn repolicy_market() -> bool {
    static M: OnceLock<bool> = OnceLock::new();
    *M.get_or_init(|| obrrt_core::sys::env::var("OBRRT_POLICY_REPOLICY_MARKET").is_ok_and(|v| v == "1"))
}

/// Process-wide tallies of the policy breeding operator: children made,
/// attempts won by one, emulator frames spent in the hook (replayed prefix +
/// policy), attempts that used it.
pub static REPOLICY_STATS: [std::sync::atomic::AtomicU64; 4] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

thread_local! {
    /// Extra first-generation individuals from the [`SeedSource`], for this attempt.
    static EXTRA_SEEDS: std::cell::RefCell<Vec<Vec<GamePad>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Share of children of a parent with key frames (a keyed overlay's) that
/// change only those frames' sticks, by small Gaussian steps.
const KEY_P: f64 = 0.5;
/// The steps' standard deviation, in raw stick units.
const KEY_SIGMA: f32 = 20.0;

/// The GA's settings.
#[derive(Debug, Clone, Copy)]
pub struct GaParams {
    /// Multiplies generations.
    pub scale: f64,
    /// Share of the starting population made of full-length pink-noise
    /// inputs (the pre-tree GAs' inputs). At 1, the GA is purely the old
    /// kind: no steering controller anywhere, and mutations splice and
    /// refill with pink noise.
    pub pink: f64,
}

/// An attempt's budget class for diagnostics: `"macro"` (a fixed play,
/// [`Objective::fixed`]), `"explore"` (every stage [`Precision::Explore`])
/// or `"precise"` (more generations per commit).
pub fn budget_class(stages: &[&dyn Objective]) -> &'static str {
    if stages.len() == 1 && stages[0].fixed().is_some() {
        "macro"
    } else if stages.iter().all(|o| o.precision() == Precision::Explore) {
        "explore"
    } else {
        "precise"
    }
}

pub struct Attempt<S = SaveState> {
    /// One per objective attempted, in order: each met objective, then (if
    /// one wasn't met) the one that was being worked on.
    pub segments: Vec<Segment<S>>,
    pub frames: u64,
    pub died: bool,
    pub ga: Option<GaInfo>,
    /// Dense adds (`OBRRT_ADD`): more sampled states as candidates.
    pub dense: Option<Box<crate::dense::Out>>,
}

/// What a dense-add attempt collects, and against what (see dense.rs).
pub struct DenseReq<'a> {
    /// The tree's dense-key index as of the round's start: whether a state
    /// with this key at this frame would be new.
    pub novel: &'a (dyn Fn(u64, u32) -> bool + Sync),
    /// Sparse saves: a candidate gets a save when more than this many frames
    /// from the last save on its chain (the start's anchor counts).
    pub replay_max: u32,
    /// Frames from the start's anchor to the start.
    pub start_dist: u32,
    /// Packs a save as the tree stores it (against the start's base).
    pub pack: &'a (dyn Fn(&SaveState) -> crate::store::Delta + Sync),
}

/// One rollout's candidates: (inputs played, state, save, key).
struct DChain {
    genome: Vec<GamePad>,
    cands: Vec<(usize, St, Option<crate::store::Delta>, u64)>,
}

/// Collects an attempt's dense candidates.
pub(crate) struct Collector<'a> {
    req: &'a DenseReq<'a>,
    /// The earliest frame per key among this attempt's candidates.
    best: std::collections::HashMap<u64, u32>,
    chains: Vec<DChain>,
    stats: crate::dense::Stats,
}

impl<'a> Collector<'a> {
    fn new(req: &'a DenseReq<'a>) -> Self {
        Collector { req, best: Default::default(), chains: Vec::new(), stats: Default::default() }
    }

    /// Whether `st` (`t` inputs into its rollout) is a new candidate; if so
    /// it is noted as the attempt's earliest at its key.
    fn wanted(&mut self, st: &St, start: &St) -> Option<u64> {
        self.stats.seen += 1;
        if !crate::dense::eligible(st, start) {
            return None;
        }
        let k = crate::dense::dkey(st, crate::dense::GRID);
        if !(self.req.novel)(k, st.frame) || self.best.get(&k).is_some_and(|&f| f <= st.frame) {
            return None;
        }
        self.stats.novel += 1;
        self.best.insert(k, st.frame);
        Some(k)
    }

    /// Whether a candidate `t` inputs into a rollout whose last save was at
    /// `last` needs a save of its own.
    fn needs_save(&self, t: usize, last: Option<usize>) -> bool {
        let d = match last {
            Some(l) => (t - l) as u64,
            None => self.req.start_dist as u64 + t as u64,
        };
        d > self.req.replay_max as u64
    }

    /// The kept candidates, chained (see dense.rs).
    fn finish(mut self) -> crate::dense::Out {
        let mut taken = std::collections::HashSet::new();
        let mut all: Vec<(u32, usize, usize)> = Vec::new();
        for (ci, ch) in self.chains.iter().enumerate() {
            for (j, c) in ch.cands.iter().enumerate() {
                if self.best.get(&c.3) == Some(&c.1.frame) && taken.insert(c.3) {
                    all.push((ci as u32, j, c.1.frame as usize));
                } else {
                    self.stats.beaten += 1;
                }
            }
        }
        let mut out = crate::dense::Out::default();
        let mut prev: Vec<usize> = vec![0; self.chains.len()];
        let mut chains: Vec<Vec<Option<(usize, St, Option<crate::store::Delta>, u64)>>> =
            self.chains.iter_mut().map(|c| std::mem::take(&mut c.cands).into_iter().map(Some).collect()).collect();
        for (ci, j, _) in all {
            let (t, st, save, key) = chains[ci as usize][j].take().unwrap();
            let g = &self.chains[ci as usize].genome;
            let inputs = g[prev[ci as usize]..t].to_vec();
            prev[ci as usize] = t;
            out.cands.push(crate::dense::Cand { chain: ci, inputs, st, save, key });
        }
        self.stats.returned = out.cands.len() as u64;
        out.stats = self.stats;
        out
    }
}

impl Attempt {
    /// The same attempt with each end's save state turned into `f`'s (the
    /// tree packs them as deltas in the worker: a round holds 64 attempts'
    /// ends until they are merged, 8.6 MB each as whole states).
    pub fn map_saves<T>(self, mut f: impl FnMut(SaveState) -> T) -> Attempt<T> {
        let segments = self
            .segments
            .into_iter()
            .map(|s| Segment {
                inputs: s.inputs,
                end: s.end.map(|(st, save)| (st, f(save))),
                success: s.success,
                progress: s.progress,
                tail: s.tail,
                played: s.played,
            })
            .collect();
        Attempt { segments, frames: self.frames, died: self.died, ga: self.ga, dense: self.dense }
    }
}


/// Diagnostics (diag::door_overshoot): for a door target, whether the
/// committed inputs came within 150 of the door airborne or in a landing
/// before meeting it.
fn door_overshoot(obj: &dyn Objective, start: &St, traj: &[St], ends: &[usize]) {
    if obj.arrival(start) != obrrt_core::Arrival::Door {
        return;
    }
    let (Some(m), Some(t)) = (obrrt_core::mover::mover(), obj.steer_target(start)) else { return };
    let end = ends.first().map_or(traj.len(), |&e| e + 1).min(traj.len());
    let over = traj[..end].iter().any(|s| m.in_flight(s) && s.hdist(t) < 150.0);
    crate::diag::door_overshoot(over, !ends.is_empty());
}

/// ETA frames a commit must gain on the stage's lowest so far to count as
/// progress for `OBRRT_RH_STALL`: an ETA that only jitters around a floor it
/// can't get past (a closed cannon, an unreachable ledge) stops the attempt
/// instead of creeping down by fractions of a frame.
const STALL_GAIN: f64 = 0.5;

/// Where plans start: the committed part of the attempt.
struct Ctx {
    /// Inputs and states from the attempt's start (`traj[0]` the start), the
    /// states without objects but the last.
    pads: Vec<GamePad>,
    traj: Vec<St>,
    /// The stage worked on, the state it started from (with objects; the
    /// `start` of its ETA) and its ETA scratch, folded up to the last state.
    stage: usize,
    seg_start: St,
    mem: Mem,
    /// Its ETA at the last state, in frames.
    eta: f64,
    /// `traj` index at which each met stage was met.
    ends: Vec<usize>,
    /// The stage's lowest ETA on `traj` and where (the earliest on ties).
    peak: (f64, usize),
}

impl Ctx {
    /// At the attempt's start, past the stages already met there.
    fn new(stages: &[&dyn Objective], start: &St) -> Ctx {
        let mut c = Ctx {
            pads: Vec::new(),
            traj: vec![start.clone()],
            stage: 0,
            seg_start: start.clone(),
            mem: Mem::default(),
            eta: 0.0,
            ends: Vec::new(),
            peak: (f64::INFINITY, 0),
        };
        while c.stage < stages.len() {
            let mut mem = Mem::default();
            let e = stages[c.stage].eta(start, start, &mut mem);
            if e.done() {
                c.ends.push(0);
                c.stage += 1;
                continue;
            }
            c.mem = mem;
            c.eta = e.frames();
            break;
        }
        c.peak = (c.eta, 0);
        c
    }

    fn last(&self) -> &St {
        self.traj.last().unwrap()
    }

    /// Appends a replay of committed frames (`e`, played from the last
    /// state with `keep`); whether the stage's ETA reached a new low (by
    /// [`STALL_GAIN`]) or a stage was met.
    fn absorb(&mut self, e: Eval) -> bool {
        let mut improved = false;
        let low = self.peak.0;
        let mut dones = e.dones.iter().peekable();
        let mut stage = self.stage;
        for (i, (p, st)) in e.pads.into_iter().zip(e.states).enumerate() {
            if let Some(last) = self.traj.last_mut() {
                last.objects = None;
            }
            self.pads.push(p);
            self.traj.push(st);
            let at = self.traj.len() - 1;
            while dones.next_if(|&&d| d == i + 1).is_some() {
                self.ends.push(at);
            }
            if e.stage[i] as usize != stage {
                stage = e.stage[i] as usize;
                self.peak = (e.eta[i], at);
                improved = true;
            } else if e.eta[i] < self.peak.0 {
                self.peak = (e.eta[i], at);
            }
        }
        improved |= self.peak.0 < low - STALL_GAIN;
        self.stage = e.fin.stage;
        self.mem = e.fin.mem;
        self.eta = e.fin.eta;
        if let Some(s) = e.fin.seg_start {
            self.seg_start = s;
        }
        improved
    }
}

/// How a plan's play ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum End {
    /// Its frames ran out.
    Horizon,
    /// Every stage met.
    Done,
    Died,
    /// The stage's ETA was `Lost` (or its dialog closed unmet).
    Lost,
    /// A stage was met where Mario isn't controllable (a star dance, a
    /// dialog) with more to go, or the next stage can't be met from there:
    /// the rest isn't attempted.
    Cut,
}

/// Where a play left the stage, scratch and stage start.
struct Fin {
    stage: usize,
    mem: Mem,
    eta: f64,
    /// The stage's start, if a stage was met during the play.
    seg_start: Option<St>,
}

/// One play from the committed state.
struct Eval {
    /// Inputs played (overlays applied; not the dying or losing frame's).
    pads: Vec<GamePad>,
    /// The state after each, objects dropped (kept with `keep`).
    states: Vec<St>,
    /// After each: the stage worked on and its ETA.
    stage: Vec<u8>,
    eta: Vec<f64>,
    /// Frames into the play at which stages were met, in order.
    dones: Vec<usize>,
    end: End,
    /// Frames into the play a keyed overlay marked as key.
    key: Vec<usize>,
    /// max over t ≥ min(k, frames) of Σ_{i<t} γ^i r_i (see [`play`]).
    fitness: f64,
    /// Frames to that maximum.
    at: usize,
    /// Frames emulated (the dying or losing one included).
    stepped: usize,
    fin: Fin,
}

/// Greater is better: fitness, then the earlier first success, then the
/// fewer frames to the fitness's maximum.
fn rank(a: &Eval, b: &Eval) -> Ordering {
    a.fitness
        .total_cmp(&b.fitness)
        .then_with(|| match (a.dones.first(), b.dones.first()) {
            (Some(x), Some(y)) => y.cmp(x),
            (Some(_), None) => Ordering::Greater,
            (None, Some(_)) => Ordering::Less,
            (None, None) => Ordering::Equal,
        })
        .then_with(|| b.at.cmp(&a.at))
}

/// Where a play's inputs come from.
#[derive(Clone, Copy)]
enum Src<'a> {
    /// A plan: its inputs (overlays applied after the first `raw`, which
    /// were recorded as played), then with `Some(every)` the steering
    /// controller towards the stage's target, or the mover's policies of
    /// `policy` (`OBRRT_MOVE_GA`, movegenome.rs) when given.
    Plan { pads: &'a [GamePad], raw: usize, steer: Option<usize>, policy: Option<&'a crate::movegenome::Plan> },
    /// A fixed policy ([`Budget::Policy`]).
    Policy(&'a PadPolicy),
    /// Committed inputs replayed as played.
    Replay(&'a [GamePad]),
}

/// Plays up to `len` frames on `game` (which holds `ctx`'s last state) from
/// `src`, folding the current stage's ETA every frame. The reward of frame
/// t is eta_{t-1} − eta_t; meeting a stage adds `bonus` and the next stage's
/// ETA starts from that state (from fresh scratch, no reward for the jump);
/// death or `Lost` gives −`death` and ends the play. Fitness is the best
/// discounted running sum at or after frame min(k, frames) (the curiosity
/// GA's, rust/src/curiosity_scoring.rs), so a plan is judged at least on
/// what would be committed.
#[allow(clippy::too_many_arguments)]
fn play(
    game: &mut SM64Game,
    stages: &[&dyn Objective],
    start: &St,
    ctx: &Ctx,
    src: Src,
    len: usize,
    keep: bool,
    rh: &Rh,
) -> Result<Eval> {
    let objects = stages.iter().any(|o| o.needs_objects());
    // Objectives' per-frame overlays on the controller's pads (not on
    // policies, which play their own inputs, or replays, already applied).
    let (overlays, raw): (Vec<Option<PadOverlay>>, usize) = match src {
        Src::Plan { raw, .. } => (stages.iter().map(|o| o.pad_overlay()).collect(), raw),
        _ => (Vec::new(), 0),
    };
    // Overlays and policies see the trajectory and inputs from the start.
    let need_ctx = overlays.iter().any(Option::is_some) || matches!(src, Src::Policy(_));
    let mut tctx: Vec<St> = if need_ctx { ctx.traj.clone() } else { Vec::new() };
    let mut gctx: Vec<GamePad> = if need_ctx { ctx.pads.clone() } else { Vec::new() };
    let base = ctx.pads.len();
    let mut stage = ctx.stage;
    let mut mem = ctx.mem;
    let mut seg_start: Option<St> = None;
    let mut prev = ctx.eta;
    let mut last = ctx.last().clone();
    let mut e = Eval {
        pads: Vec::with_capacity(len),
        states: Vec::with_capacity(len),
        stage: Vec::with_capacity(len),
        eta: Vec::with_capacity(len),
        dones: Vec::new(),
        end: End::Horizon,
        key: Vec::new(),
        fitness: 0.0,
        at: 0,
        stepped: 0,
        fin: Fin { stage, mem, eta: prev, seg_start: None },
    };
    let mut sums: Vec<f64> = Vec::with_capacity(len);
    let (mut sum, mut disc) = (0.0, 1.0);
    let mut reward = |r: f64, sums: &mut Vec<f64>| {
        sum += disc * r;
        if sums.len() >= rh.flat {
            disc *= rh.gamma;
        }
        sums.push(sum);
    };
    for f in 0..len {
        if stage >= stages.len() {
            e.end = End::Done;
            break;
        }
        let abs = base + f;
        let obj = stages[stage];
        let pad = match src {
            Src::Plan { pads, steer: every, policy, .. } => match (pads.get(f), every) {
                (Some(p), _) => *p,
                (None, Some(every)) => match policy {
                    Some(pl) => crate::movegenome::pad(pl, f - pads.len(), obj, &last, abs, every),
                    None => steer(obj, &last, abs, every),
                },
                (None, None) => break,
            },
            Src::Policy(p) => p.pad(&tctx, &gctx),
            Src::Replay(p) => match p.get(f) {
                Some(p) => *p,
                None => break,
            },
        };
        let pad = match overlays.get(stage) {
            Some(Some(o)) if f >= raw => {
                let (p, k) = o.pad_keyed(&tctx, &gctx, pad);
                if k {
                    e.key.push(f);
                }
                p
            }
            _ => pad,
        };
        let pad = if rh.press && !obj.modal() && (!last.controllable() || last.modal()) { GamePad::new(if abs % 2 == 0 { A_BUTTON } else { 0 }, 0, 0) } else { pad };
        game.step_game(pad)?;
        e.stepped += 1;
        let st = read_state(game, start.frame + abs as u32 + 1, objects);
        if !st.alive() || st.lives < start.lives {
            reward(-rh.death, &mut sums);
            e.end = End::Died;
            break;
        }
        let eta = obj.eta(&st, seg_start.as_ref().unwrap_or(&ctx.seg_start), &mut mem);
        let lost = match eta {
            Eta::Lost => true,
            Eta::Frames(x) => !x.is_finite(),
            Eta::Done => false,
        } || (obj.modal() && !eta.done() && !obj.exists_in(&st));
        if lost {
            reward(-rh.death, &mut sums);
            e.end = End::Lost;
            break;
        }
        e.pads.push(pad);
        let mut r = prev - eta.frames();
        if eta.done() {
            // What the goal still costs after this state (0 for most).
            let after = obj.done_frames(&st, &mem);
            r += if after == 0.0 { rh.bonus } else { rh.bonus - after };
            e.dones.push(f + 1);
            stage += 1;
            mem = Mem::default();
            seg_start = Some(st.clone());
            // The next stage's ETA from here (its fold's first state).
            while stage < stages.len() {
                let e2 = stages[stage].eta(&st, &st, &mut mem);
                if e2.done() {
                    r += rh.bonus;
                    e.dones.push(f + 1);
                    stage += 1;
                    mem = Mem::default();
                    continue;
                }
                prev = e2.frames();
                if !prev.is_finite() {
                    e.end = End::Cut;
                }
                break;
            }
            if stage < stages.len() && !st.controllable() && !rh.through {
                e.end = End::Cut;
            }
        } else {
            prev = eta.frames();
        }
        reward(r, &mut sums);
        e.stage.push(stage.min(255) as u8);
        e.eta.push(if stage < stages.len() { prev } else { 0.0 });
        if need_ctx {
            tctx.push(st.clone());
            gctx.push(pad);
        }
        let mut kept = st.clone();
        if !keep {
            kept.objects = None;
        }
        e.states.push(kept);
        last = st;
        if e.end == End::Cut {
            break;
        }
        if stage >= stages.len() {
            e.end = End::Done;
            break;
        }
    }
    let min_t = rh.k.min(sums.len()).max(1);
    if let Some((i, &s)) = sums.iter().enumerate().skip(min_t - 1).max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(&a.0))) {
        e.fitness = s;
        e.at = i + 1;
    }
    e.fin = Fin { stage, mem, eta: if stage < stages.len() { prev } else { 0.0 }, seg_start };
    Ok(e)
}

/// One plan of the population.
struct Gen {
    /// Inputs from the committed state on.
    pads: Vec<GamePad>,
    /// The first `raw` were recorded as played (a seed, a carried tail):
    /// overlays aren't applied to them again.
    raw: usize,
    /// Steering past `pads` (period of its A rhythm), or nothing.
    steer: Option<usize>,
    /// `OBRRT_MOVE_GA`: the mover's policies past `pads` (with `steer`),
    /// until first played.
    policy: Option<crate::movegenome::Plan>,
    origin: u8,
    generation: u16,
    /// `OBRRT_MOVE_GA`: the policy-seeded plan it is or descends from (index
    /// into the attempt's plans; `u32::MAX` none), and the policy of the
    /// `repolicy` mutation that made it (`u8::MAX` none).
    lineage: u32,
    rp: u8,
    /// Played to its own length, not the horizon (a seed as is).
    full: bool,
    eval: Option<Eval>,
}

impl Gen {
    fn new(pads: Vec<GamePad>, steer: Option<usize>, origin: u8) -> Gen {
        Gen { pads, raw: 0, steer, policy: None, origin, generation: 0, lineage: u32::MAX, rp: u8::MAX, full: false, eval: None }
    }

    /// Keeps what was played (overlays applied, steering and policies
    /// recorded) as its inputs, then whatever it had beyond.
    fn take(&mut self, e: Eval) {
        let mut pads = e.pads.clone();
        if self.pads.len() > pads.len() {
            pads.extend_from_slice(&self.pads[pads.len()..]);
        }
        self.pads = pads;
        self.policy = None;
        self.full = false;
        self.eval = Some(e);
    }

    /// Moves it past `c` committed frames; its end is refilled to `h`
    /// frames with random inputs (as the curiosity GA did: base policies
    /// are only in the first generation, as in v1).
    fn shift(&mut self, c: usize, h: usize, rng: &mut SmallRng) {
        self.pads.drain(..c.min(self.pads.len()));
        self.raw = self.raw.saturating_sub(c);
        self.eval = None;
        self.full = false;
        self.policy = None;
        self.steer = None;
        if self.pads.len() < h {
            let n = h - self.pads.len();
            self.pads.extend(random_inputs(rng, n));
        }
    }
}

/// What children are made from: the controller's settings for this
/// attempt, and (`OBRRT_MOVE_GA`) the plausible policies for `repolicy`.
struct Breed<'a> {
    target: bool,
    h: usize,
    ok: &'a [usize],
    /// `OBRRT_MOVE_GA`: policy lineages fill at most half the elite.
    lineage_cap: bool,
}

/// A child of the elite `parents` (each evaluated): v1's operators on one
/// (repolicy, key sticks, span, turn, A toggle, re-steer, press; each
/// steering past its inputs where there is a target, as v1's children
/// did), or a crossover of two (population.rs).
fn child(parents: &[Gen], rng: &mut SmallRng, b: &Breed, generation: u16, repolicies: &mut Vec<usize>) -> Gen {
    let (target, h) = (b.target, b.h);
    let steer_after = target.then_some(4);
    let p = &parents[rng.random_range(0..parents.len())];
    let mut g = p.pads.clone();
    let lineage = p.lineage;
    let out = |g: Vec<GamePad>, steer: Option<usize>, origin: u8, raw: usize| {
        let mut c = Gen::new(g, steer, origin);
        c.raw = raw;
        c.generation = generation;
        c.lineage = lineage;
        c
    };
    // OBRRT_MOVE_GA: re-simulate a span under another policy.
    if !b.ok.is_empty() && rng.random_bool(crate::movegenome::repolicy_p()) {
        if let Some((pre, pl)) = crate::movegenome::repolicy(&g, b.ok, h, rng) {
            let raw = p.raw.min(pre.len());
            let mut c = out(pre, Some(4), ORIGIN_REPOLICY, raw);
            c.rp = pl.segs[0].0 as u8;
            repolicies.push(pl.segs[0].0);
            c.policy = Some(pl);
            return c;
        }
    }
    // Crossover (population.rs): runs of ~30 frames from either parent,
    // sometimes one's stick with the other's buttons.
    if parents.len() > 1 && rng.random_bool(0.3) {
        let q = &parents[rng.random_range(0..parents.len())];
        let n = p.pads.len().max(q.pads.len()).min(h);
        let mut r: f64 = rng.random();
        let mut c = Vec::with_capacity(n);
        for j in 0..n {
            if j == 0 || rng.random_bool(1.0 / 30.0) {
                r = rng.random();
            }
            let (a, b) = (p.pads.get(j).or(q.pads.get(j)).copied().unwrap(), q.pads.get(j).or(p.pads.get(j)).copied().unwrap());
            c.push(if r < 0.4 {
                a
            } else if r < 0.8 {
                b
            } else if r < 0.9 {
                GamePad { button: b.button, ..a }
            } else {
                GamePad { button: a.button, ..b }
            });
        }
        return out(c, steer_after, ORIGIN_CROSSOVER, 0);
    }
    // A keyed overlay's key frames: sometimes nudge just their sticks.
    if let Some(key) = p.eval.as_ref().map(|e| &e.key).filter(|k| !k.is_empty()) {
        if rng.random_bool(KEY_P) {
            let pick = rng.random_range(0..key.len());
            for (j, &i) in key.iter().enumerate() {
                // The chosen frame always, the others half the time.
                if i >= g.len() || (j != pick && rng.random_bool(0.5)) {
                    continue;
                }
                let gauss = |rng: &mut SmallRng| {
                    let (u, v): (f32, f32) = (rng.random::<f32>().max(1e-7), rng.random::<f32>());
                    (-2.0 * u.ln()).sqrt() * (TAU * v).cos() * KEY_SIGMA
                };
                let (dx, dy) = (gauss(rng), gauss(rng));
                g[i].stick_x = (g[i].stick_x as f32 + dx).round().clamp(-128.0, 127.0) as i8;
                g[i].stick_y = (g[i].stick_y as f32 + dy).round().clamp(-128.0, 127.0) as i8;
            }
            g.truncate(h);
            return out(g, steer_after, ORIGIN_KEY, p.raw);
        }
    }
    let len = g.len().max(1);
    let a = rng.random_range(0..len);
    let op = rng.random_range(0..5u8);
    match op {
        0 => {
            let b = (a + rng.random_range(3..40)).min(h);
            g.truncate(h);
            let mut seg = Vec::new();
            random_segment(rng, &mut seg, b.saturating_sub(a).max(1));
            g.splice(a.min(g.len())..b.min(g.len()), seg);
            g.truncate(h);
        }
        1 => {
            let b = (a + rng.random_range(5..60)).min(g.len());
            let turn = rng.random_range(-40i32..40);
            for p in &mut g[a.min(b)..b] {
                let ang = (p.stick_y as f32).atan2(p.stick_x as f32) + turn as f32 * TAU / 360.0;
                let mag = ((p.stick_x as f32).powi(2) + (p.stick_y as f32).powi(2)).sqrt().min(127.0);
                p.stick_x = (mag * ang.cos()) as i8;
                p.stick_y = (mag * ang.sin()) as i8;
            }
        }
        2 => {
            let b = (a + rng.random_range(1..12)).min(g.len());
            for p in &mut g[a.min(b)..b] {
                p.button ^= A_BUTTON;
            }
        }
        4 => {
            // Press A, B or Z for a few frames somewhere.
            let mut button = [A_BUTTON, A_BUTTON, Z_TRIG, Z_TRIG, B_BUTTON][rng.random_range(0..5)];
            // Calm: an A press only as often as A looks useful.
            if button == A_BUTTON && !rng.random_bool(a_mult()) {
                button = 0;
            }
            let n = rng.random_range(1..5);
            g.truncate(h);
            while g.len() < (a + n).min(h) {
                g.push(GamePad::new(0, 0, 0));
            }
            for p in &mut g[a.min(h.saturating_sub(1))..(a + n).min(h)] {
                p.button |= button;
            }
        }
        _ => {
            // Cut, then steered (or random) from there.
            g.truncate(a);
            if target {
                let every = [2, 4, 8, 16, 30][rng.random_range(0..5)];
                let raw = p.raw.min(a);
                return out(g, Some(every), 2 + op, raw);
            }
            let rest = random_inputs(rng, h.saturating_sub(a));
            g.extend(rest);
        }
    }
    if !target && g.len() < h {
        let n = h - g.len();
        g.extend(random_inputs(rng, n));
    }
    let raw = p.raw.min(a);
    out(g, steer_after, 2 + op, raw)
}

/// Sorts `pop` best first, keeps the elite (a quarter; with
/// `lineage_cap`, policy lineages at most half of it, the best always) and
/// fills it back up to `n` with their children.
fn breed(pop: &mut Vec<Gen>, n: usize, rng: &mut SmallRng, b: &Breed, generation: u16, repolicies: &mut Vec<usize>) {
    pop.sort_by(|a, b| match (&a.eval, &b.eval) {
        (Some(x), Some(y)) => rank(y, x),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });
    let elite = (n / 4).max(1).min(pop.len());
    if b.lineage_cap && elite > 1 {
        let cap = elite.div_ceil(2);
        let (mut keep, mut rest) = (Vec::with_capacity(elite), Vec::new());
        let mut k = 0;
        for (i, g) in pop.drain(..).enumerate() {
            let lineage = g.lineage != u32::MAX;
            if keep.len() < elite && (i == 0 || !lineage || k < cap) {
                k += lineage as usize;
                keep.push(g);
            } else {
                rest.push(g);
            }
        }
        let short = elite.saturating_sub(keep.len());
        keep.extend(rest.into_iter().take(short));
        *pop = keep;
    }
    pop.truncate(elite);
    let mut made = [0u64; ORIGINS.len()];
    while pop.len() < n {
        let c = child(&pop[..elite], rng, b, generation, repolicies);
        made[c.origin as usize] += 1;
        pop.push(c);
    }
    let mut m = makeup().lock().unwrap();
    m.breeds += 1;
    for (t, x) in m.children.iter_mut().zip(made) {
        *t += x;
    }
}

/// Population makeup (diagnostics only, never read by the search): the
/// first generations' members and the children bred, by origin.
#[derive(Debug, Default)]
pub struct Makeup {
    pub attempts: u64,
    pub gen0: [u64; ORIGINS.len()],
    pub breeds: u64,
    pub children: [u64; ORIGINS.len()],
}

pub fn makeup() -> &'static std::sync::Mutex<Makeup> {
    static M: OnceLock<std::sync::Mutex<Makeup>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// E.g. `gen0/attempt: seeded 16 random 16; children/breed: crossover 7.2 ...`.
pub fn makeup_line() -> String {
    let m = makeup().lock().unwrap();
    let part = |xs: &[u64; ORIGINS.len()], n: u64| {
        ORIGINS
            .iter()
            .zip(xs)
            .filter(|(_, x)| **x > 0)
            .map(|(o, x)| format!("{o} {:.1}", *x as f64 / n.max(1) as f64))
            .collect::<Vec<_>>()
            .join(" ")
    };
    format!("gen0 per attempt: {}; children per breed: {}", part(&m.gen0, m.attempts), part(&m.children, m.breeds))
}

// Calm: random genomes' A presses scaled by how useful jumping looks from
// the start (the mover's `jump_useful`), and fewer jump starts as a tiebreak.

fn a_mult() -> f64 {
    A_MULT.with(|a| a.get())
}

thread_local! {
    /// The attempt's multiplier on random genomes' A presses (`OBRRT_CALM`;
    /// exactly 1 otherwise, so the draws are unchanged).
    static A_MULT: std::cell::Cell<f64> = const { std::cell::Cell::new(1.0) };
    /// While a seeded attempt runs on this thread (open GA): the
    /// objectives' pad overlays are not applied to a rollout's given prefix
    /// frames (a seed is recorded tree inputs, already as played), only to
    /// steered ones.
    static RAW_PREFIX: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Closed-loop steering towards the objective's target, per control mode.
pub(crate) use crate::steer::steer;

/// A held stick with each button pressed for its own span within the
/// segment, so presses can overlap or follow each other (Z then A).
pub(crate) fn random_segment_v2(rng: &mut SmallRng, out: &mut Vec<GamePad>, len: usize) {
    let angle = rng.random_range(0.0..TAU);
    let mag = if rng.random_bool(0.1) { 0.0 } else if rng.random_bool(0.2) { rng.random_range(20.0..127.0) } else { 127.0 };
    let (sx, sy) = ((mag * angle.sin()) as i8, (mag * angle.cos()) as i8);
    let at = out.len();
    out.extend((0..len).map(|_| GamePad::new(0, sx, sy)));
    for (button, p) in [(A_BUTTON, 0.4 * a_mult()), (B_BUTTON, 0.12), (Z_TRIG, 0.15)] {
        if rng.random_bool(p) {
            let from = rng.random_range(0..len);
            let n = if rng.random_bool(0.5) { rng.random_range(1..4) } else { rng.random_range(1..=len - from) };
            for pad in &mut out[at + from..(at + from + n).min(at + len)] {
                pad.button |= button;
            }
        }
    }
}

fn random_segment(rng: &mut SmallRng, out: &mut Vec<GamePad>, len: usize) {
    let angle = rng.random_range(0.0..TAU);
    let mag = if rng.random_bool(0.1) { 0.0 } else if rng.random_bool(0.2) { rng.random_range(20.0..127.0) } else { 127.0 };
    let (sx, sy) = ((mag * angle.sin()) as i8, (mag * angle.cos()) as i8);
    let mut button = 0;
    if rng.random_bool(0.35 * a_mult()) {
        button |= A_BUTTON;
    }
    if rng.random_bool(0.08) {
        button |= B_BUTTON;
    }
    if rng.random_bool(0.05) {
        button |= Z_TRIG;
    }
    let tap = rng.random_bool(0.5);
    for i in 0..len {
        let b = if tap && i > 0 { button & !A_BUTTON & !B_BUTTON } else { button };
        out.push(GamePad::new(b, sx, sy));
    }
}

/// `n` frames of random held inputs (v1's new operators' segments).
fn random_inputs(rng: &mut SmallRng, n: usize) -> Vec<GamePad> {
    let mut out = Vec::with_capacity(n + 32);
    while out.len() < n {
        let len = rng.random_range(3..30);
        random_segment_v2(rng, &mut out, len);
    }
    out.truncate(n);
    out
}


impl Worker {
    /// Attempts `stages` in order from `start` (whose save state is `save`).
    /// A single objective with a fixed play plays it; otherwise the
    /// rolling-horizon controller works through the stages.
    pub fn attempt(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        save: &SaveState,
        rng: &mut SmallRng,
        params: GaParams,
    ) -> Result<Attempt> {
        self.attempt_seeded(stages, start, save, rng, params, None)
    }

    /// [`Worker::attempt`] with a known genome (`seed`, e.g. recorded tree
    /// inputs that met the stages from this start) in the first generation:
    /// the seed as is (played to its length, without overlays), then
    /// variants with single frames or a chunk dropped, or cut and steered.
    /// The committed-length cap is stretched to the seed's length. `None` is
    /// `attempt`.
    pub fn attempt_seeded(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        save: &SaveState,
        rng: &mut SmallRng,
        params: GaParams,
        seed: Option<&[GamePad]>,
    ) -> Result<Attempt> {
        self.attempt_full(stages, start, Some(save), rng, params, seed, None)
    }

    /// [`Worker::attempt_seeded`] with `save` None when `base` already holds
    /// the start (`Tree::load_into`), and dense adds when `dense` is given.
    /// A carried plan comes from [`set_tail`] (used with `OBRRT_TAIL=1`, or
    /// by the rolling-horizon controller).
    #[allow(clippy::too_many_arguments)]
    pub fn attempt_full(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        save: Option<&SaveState>,
        rng: &mut SmallRng,
        params: GaParams,
        seed: Option<&[GamePad]>,
        dense: Option<&DenseReq>,
    ) -> Result<Attempt> {
        let tail = TAIL.with(|t| t.borrow_mut().take()).filter(|t| !t.is_empty());
        // OBRRT_POLICY_SEED: a seed from the installed source when the caller has none.
        let mut policy_frames = 0u64;
        let mut policy_pick: Option<Vec<GamePad>> = None;
        EXTRA_SEEDS.with(|e| e.borrow_mut().clear());
        let caller_seed = seed.is_some();
        if policy_seed_on() {
            let src = SEED_SOURCE.read().unwrap().clone();
            if let Some(src) = src {
                let own;
                let sv = match save {
                    Some(s) => s,
                    None => {
                        own = self.base.save_state();
                        &own
                    }
                };
                let mut got = src.seeds(&mut self.base, start, sv, stages[0], rng);
                self.base.load_state(sv)?;
                policy_frames = got.iter().map(|g| g.1).sum();
                got.retain(|g| !g.0.is_empty());
                if !got.is_empty() {
                    let mut it = got.into_iter().map(|g| g.0);
                    // The caller's own seed (the route's inputs) stays the seed; the
                    // policy plans then come in as extra first-generation individuals.
                    if !caller_seed {
                        policy_pick = it.next();
                    }
                    let extra: Vec<Vec<GamePad>> = it.take(policy_seed_extra() + caller_seed as usize).collect();
                    EXTRA_SEEDS.with(|e| *e.borrow_mut() = extra);
                }
            }
        }
        let seed = if policy_pick.is_some() { policy_pick.as_deref() } else { seed };
        // Calm: random genomes' A presses scaled by how useful
        // jumping looks from the start (the mover's jump_useful).
        {
            let m = match (obrrt_core::mover::mover(), stages[0].steer_target(start)) {
                (Some(m), Some(t)) => (m.jump_useful(start, t, stages[0].arrival(start)) as f64).clamp(0.05, 1.0),
                _ => 1.0,
            };
            A_MULT.with(|a| a.set(m));
        }
        let mut dc = dense.map(Collector::new);
        let out = if rh_on() {
            self.attempt_rh(&mut dc, stages, start, save, rng, params, seed, tail)
        } else {
            self.attempt_open(&mut dc, stages, start, save, rng, params, seed)
        };
        A_MULT.with(|a| a.set(1.0));
        EXTRA_SEEDS.with(|e| e.borrow_mut().clear());
        out.map(|mut a| {
            a.frames += policy_frames;
            a
        })
    }

    /// The first generation, as v1's: per slot a policy-seeded plan
    /// (`OBRRT_MOVE_GA`, in its slots or with `_EXTRA` on top), the seed or
    /// a variant of it, pink noise (`OBRRT_PINK`), a steered rollout after a
    /// noise prefix (even slots, with a target) or random inputs; then the
    /// carried plan (as is and two mutated copies) and the learned model's
    /// (`OBRRT_MODEL`) on top. Also the attempt's policy plans and the
    /// plausible policies (for `repolicy`).
    #[allow(clippy::too_many_arguments)]
    fn gen0(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        rng: &mut SmallRng,
        rh: &Rh,
        pink: f64,
        seed: Option<&[GamePad]>,
        tail: Option<Vec<GamePad>>,
    ) -> Result<(Vec<Gen>, Vec<crate::movegenome::Plan>, Vec<usize>)> {
        let h = rh.h;
        let population = rh.pop;
        let pure = pink >= 1.0;
        let target = !pure && stages[0].steer_target(start).is_some();
        let steer_after = target.then_some(4);
        let mut pop: Vec<Gen> = Vec::with_capacity(population + 16);
        let move_ga = crate::movegenome::enabled() && target && (seed.is_none() || crate::movegenome::with_seed());
        let ok = if move_ga { crate::movegenome::plausible(stages[0], start) } else { Vec::new() };
        let extra = crate::movegenome::extra();
        let pass = stages[0].arrival(start) == obrrt_core::Arrival::Pass;
        let plans = if ok.is_empty() {
            Vec::new()
        } else {
            let n = crate::movegenome::gen0_budget(ok.len());
            crate::movegenome::gen0_plans(&ok, if extra { n } else { n.min(crate::movegenome::slots_in(population, pass)) }, rng)
        };
        for i in 0..population + if extra { plans.len() } else { 0 } {
            let k = if extra { i.checked_sub(population) } else { crate::movegenome::slot(i, pass).filter(|&k| k < plans.len()) };
            if let Some(k) = k {
                let mut g = Gen::new(Vec::new(), Some(4), ORIGIN_POLICY);
                g.policy = Some(plans[k].clone());
                g.lineage = k as u32;
                pop.push(g);
                continue;
            }
            if let Some(s) = seed {
                let mut g = s.to_vec();
                let mut steer_every = None;
                if i > 0 && g.len() > 2 {
                    match i % 3 {
                        // Single frames dropped.
                        1 => {
                            for _ in 0..rng.random_range(1..=8) {
                                if g.len() > 1 {
                                    g.remove(rng.random_range(0..g.len()));
                                }
                            }
                        }
                        // A chunk dropped.
                        2 => {
                            let n = rng.random_range(2..=30).min(g.len() - 1);
                            let a = rng.random_range(0..=g.len() - n);
                            g.drain(a..a + n);
                        }
                        // Cut, then steered (or random) from there.
                        _ => {
                            let a = rng.random_range(g.len() / 3..g.len());
                            g.truncate(a);
                            if target {
                                steer_every = Some([2, 4, 8, 16, 30][rng.random_range(0..5)]);
                            } else {
                                g.extend(random_inputs(rng, h.saturating_sub(a)));
                            }
                        }
                    }
                }
                let raw = g.len();
                let mut pl = Gen::new(g, steer_every, ORIGIN_SEED);
                pl.raw = raw;
                pl.full = i == 0;
                pop.push(pl);
                continue;
            }
            if pink > 0.0 && rng.random_bool(pink.min(1.0)) {
                pop.push(Gen::new(crate::noise::pink_inputs(rng, h), None, ORIGIN_PINK));
                continue;
            }
            pop.push(if target && i % 2 == 0 {
                let every = [2, 4, 8, 16, 30][rng.random_range(0..5)];
                let n = rng.random_range(0..h / 4 + 1);
                Gen::new(random_inputs(rng, n), Some(every), 0)
            } else {
                Gen::new(random_inputs(rng, h), None, ORIGIN_RANDOM)
            });
        }
        for x in EXTRA_SEEDS.with(|e| std::mem::take(&mut *e.borrow_mut())) {
            let raw = x.len();
            let mut pl = Gen::new(x, None, ORIGIN_SEED);
            pl.raw = raw;
            pl.full = true;
            pop.push(pl);
        }
        if let Some(t) = tail {
            let mut g = Gen::new(t, steer_after, ORIGIN_TAIL);
            g.raw = g.pads.len();
            let parent = [Gen { pads: g.pads.clone(), raw: g.raw, steer: g.steer, ..Gen::new(Vec::new(), None, ORIGIN_TAIL) }];
            pop.push(g);
            let b = Breed { target, h, ok: &[], lineage_cap: false };
            for _ in 0..2 {
                let mut c = child(&parent, rng, &b, 0, &mut Vec::new());
                c.origin = ORIGIN_TAIL;
                pop.push(c);
            }
        }
        let mut m = makeup().lock().unwrap();
        m.attempts += 1;
        for g in &pop {
            m.gen0[g.origin as usize] += 1;
        }
        Ok((pop, plans, ok))
    }

    /// The rolling-horizon controller's attempt (`OBRRT_RH=1`).
    #[allow(clippy::too_many_arguments)]
    fn attempt_rh(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        save: Option<&SaveState>,
        rng: &mut SmallRng,
        params: GaParams,
        seed: Option<&[GamePad]>,
        tail: Option<Vec<GamePad>>,
    ) -> Result<Attempt> {
        let rh = *rh();
        if let Some(save) = save {
            self.base.load_state(save)?;
        }
        let n = stages.len();
        let mut frames = 0u64;
        let mut ctx = Ctx::new(stages, start);
        // Per met stage: the best plan's inputs past where it was met.
        let mut seg_tails: Vec<Vec<GamePad>> = vec![Vec::new(); ctx.ends.len()];
        // Dense adds, `last_gen`: the last population's plays (inputs from
        // the start, final state).
        let mut last_gen: Vec<(Vec<GamePad>, St)> = Vec::new();
        let mut ga = None;
        let (mut died, mut cut) = (false, false);
        // The best plan's inputs past the last commit.
        let mut rest: Vec<GamePad> = Vec::new();
        let fixed = if n == 1 { stages[0].fixed() } else { None };
        if let Some(b) = fixed {
            // Played once from the start, as v1 did.
            self.tmp.sync_from(&self.base)?;
            let e = match &b {
                Budget::Macro(pads) => {
                    let raw = if seed.is_some() { pads.len() } else { 0 };
                    play(&mut self.tmp, stages, start, &ctx, Src::Plan { pads, raw, steer: None, policy: None }, pads.len(), true, &rh)?
                }
                Budget::Policy { limit, policy } => play(&mut self.tmp, stages, start, &ctx, Src::Policy(policy), *limit, true, &rh)?,
            };
            frames += e.stepped as u64;
            died = e.end == End::Died;
            cut = e.end == End::Cut;
            ctx.absorb(e);
            seg_tails.resize(ctx.ends.len(), Vec::new());
        } else if ctx.stage < n {
            self.cur.sync_from(&self.base)?;
            let target = |c: &Ctx| c.stage < n && stages[c.stage].steer_target(c.last()).is_some();
            let precise = stages.iter().any(|o| o.precision() == Precision::Precise);
            let gens = if precise { rh.g_precise } else { rh.g };
            // Committed-length cap: twice the start's ETA and 3 s, at least
            // 2 s, at most OBRRT_RH_MAXLEN (and room for a seed).
            let cap = (2.0 * ctx.eta + 90.0).clamp(60.0, rh.maxlen as f64) as usize;
            let cap = cap.max(seed.map_or(0, |s| s.len() + rh.k));
            let (mut pop, _plans, ok) = self.gen0(stages, start, rng, &rh, params.pink, seed, tail)?;
            let mut repolicies = Vec::new();
            let lineage_cap = crate::movegenome::enabled();
            let (mut commits, mut stall, mut total_gens) = (0usize, 0usize, 0usize);
            let mut gen0 = None;
            let mut seed_hit = None;
            let mut origins = [0u32; ORIGINS.len()];
            loop {
                let tg = target(&ctx);
                let n_gens = if commits == 0 { rh.hot } else { gens };
                let plen = rh.h.min(cap.saturating_sub(ctx.pads.len()).max(rh.k));
                for g in 0..n_gens {
                    for pl in pop.iter_mut().filter(|g| g.eval.is_none()) {
                        let len = if pl.full { pl.pads.len().max(plen) } else { plen };
                        self.tmp.sync_from(&self.cur)?;
                        let src = Src::Plan { pads: &pl.pads, raw: pl.raw, steer: pl.steer, policy: pl.policy.as_ref() };
                        let e = play(&mut self.tmp, stages, start, &ctx, src, len, false, &rh)?;
                        frames += e.stepped as u64;
                        if pl.full && pl.origin == ORIGIN_SEED && seed_hit.is_none() {
                            let all = ctx.ends.len() + e.dones.len() >= n;
                            seed_hit = Some((all, e.dones.last().copied().unwrap_or(0) as u32));
                        }
                        pl.take(e);
                    }
                    total_gens += 1;
                    if gen0.is_none() {
                        let b = pop.iter().filter_map(|g| g.eval.as_ref()).max_by(|a, b| rank(a, b)).unwrap();
                        gen0 = Some((ctx.ends.len() + b.dones.len(), b.fitness));
                    }
                    if g + 1 < n_gens {
                        let b = Breed { target: tg, h: rh.h, ok: &ok, lineage_cap };
                        breed(&mut pop, rh.pop, rng, &b, total_gens as u16, &mut repolicies);
                    }
                }
                let bi = (0..pop.len()).max_by(|&a, &b| rank(pop[a].eval.as_ref().unwrap(), pop[b].eval.as_ref().unwrap())).unwrap();
                let best = pop[bi].eval.as_ref().unwrap();
                origins[pop[bi].origin as usize] += 1;
                // Commit its first K frames, or up to where it meets a stage
                // (that segment ends there and the next is planned afresh).
                let mut c = rh.k.min(best.pads.len());
                if let Some(&d) = best.dones.first() {
                    c = c.min(d);
                }
                let terminal = c == best.pads.len() && matches!(best.end, End::Died | End::Lost | End::Cut | End::Done);
                let pre = ctx.pads.len();
                let improved = self.cur_replay(stages, start, &mut ctx, &best.pads[..c], &rh, &mut frames)?;
                rest = best.pads[c..].to_vec();
                while seg_tails.len() < ctx.ends.len() {
                    seg_tails.push(rest.clone());
                }
                commits += 1;
                stall = if improved { 0 } else { stall + 1 };
                let stop = ctx.stage >= n || terminal || c == 0 || ctx.pads.len() >= cap || stall >= rh.stall;
                if stop {
                    if terminal {
                        died = best.end == End::Died;
                        cut = best.end == End::Cut;
                    }
                    if dc.is_some() {
                        last_gen = pop
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != bi)
                            .filter_map(|(_, g)| g.eval.as_ref())
                            .filter(|e| !e.states.is_empty())
                            .map(|e| {
                                let mut g = ctx.pads[..pre].to_vec();
                                g.extend_from_slice(&e.pads);
                                (g, e.states.last().unwrap().clone())
                            })
                            .collect();
                    }
                    break;
                }
                let b = Breed { target: target(&ctx), h: rh.h, ok: &ok, lineage_cap };
                breed(&mut pop, rh.pop, rng, &b, total_gens as u16, &mut repolicies);
                for g in &mut pop {
                    g.shift(c, rh.h, rng);
                }
            }
            door_overshoot(stages[0], start, &ctx.traj, &ctx.ends);
            let top = (0..ORIGINS.len()).max_by_key(|&i| (origins[i], std::cmp::Reverse(i))).unwrap();
            ga = Some(GaInfo {
                generations: total_gens,
                winner_gen: commits.min(u16::MAX as usize) as u16,
                winner_origin: ORIGINS[top],
                gen0: gen0.unwrap_or((0, 0.0)),
                best: (ctx.ends.len(), if ctx.ends.len() == n { 0.0 } else { -ctx.peak.0 }),
                moves: MOVES.map(|(_, act)| ctx.traj.iter().filter(|s| s.action == act).count() as u16),
                seed: seed_hit,
            });
        }

        // Dense adds after the controller: the committed end, the last
        // population's final states.
        let dense = match dc.take() {
            Some(mut c) => {
                let k = ctx.traj.len() - 1;
                let genome = ctx.pads.clone();
                self.collect_posthoc(&mut c, &genome, &[(k, &ctx.traj[k])], start)?;
                for (g, last) in &last_gen {
                    self.collect_posthoc(&mut c, g, &[(g.len(), last)], start)?;
                }
                let out = c.finish();
                frames += out.stats.replay_frames;
                Some(Box::new(out))
            }
            None => None,
        };
        // Segment boundaries: where each met stage was met, then the unmet
        // one's lowest ETA (its tail: the committed inputs past it, then the
        // best plan's).
        let mut bounds: Vec<(usize, bool, f64, Vec<GamePad>)> =
            ctx.ends.iter().zip(seg_tails).map(|(&e, t)| (e, true, 0.0, t)).collect();
        if ctx.ends.len() < n && !cut {
            let (eta, at) = ctx.peak;
            let mut t = ctx.pads[at.min(ctx.pads.len())..].to_vec();
            t.extend_from_slice(&rest);
            bounds.push((at, false, -eta, t));
        }
        let genome = std::mem::take(&mut ctx.pads);
        let mut segments = Vec::with_capacity(bounds.len());
        self.tmp.sync_from(&self.base)?;
        let mut played = 0;
        let nb = bounds.len();
        for (i, (at, success, progress, tail)) in bounds.into_iter().enumerate() {
            if at <= played {
                // Met where the previous one was (or at the start): nothing to add.
                segments.push(Segment { inputs: Vec::new(), end: None, success, progress, tail: Vec::new(), played: 0 });
                continue;
            }
            let mut inputs = genome[played..at].to_vec();
            let committed = inputs.len();
            for p in &inputs {
                self.tmp.step_game(*p)?;
            }
            frames += (at - played) as u64;
            played = at;
            if i != nb - 1 {
                let st = read_state(&self.tmp, start.frame + played as u32, true);
                segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress, tail, played: committed });
                continue;
            }
            // Universal tail: press A through cutscenes and warps until the
            // player is in control again, or a dialog or menu opens (its
            // options are objectives of their own, see Objective::modal).
            let mut st = read_state(&self.tmp, start.frame + played as u32, true);
            let mut extra = 0;
            while !st.controllable() && !st.modal() && !st.ending() && extra < TAIL_LIMIT {
                let pad = GamePad::new(if extra % 4 == 0 { A_BUTTON } else { 0 }, 0, 0);
                self.tmp.step_game(pad)?;
                inputs.push(pad);
                extra += 1;
                st = read_state(&self.tmp, start.frame + (played + extra) as u32, false);
            }
            frames += extra as u64;
            // A life lost after the run's progress (keys, stars) went up is
            // a price paid, not a failure: a star touched during a death's
            // fade-out still counts, and the death warp is just the way out.
            let gained = st.progress() > start.progress();
            if !(st.controllable() || st.modal() || st.ending()) || (!gained && (!st.alive() || st.lives < start.lives)) {
                segments.push(Segment { inputs: Vec::new(), end: None, success: false, progress, tail: Vec::new(), played: 0 });
                return Ok(Attempt { segments, frames, died: true, ga, dense });
            }
            let st = read_state(&self.tmp, start.frame + (played + extra) as u32, true);
            // The plan was for the state before the automatic tail.
            let tail = if extra == 0 { tail } else { Vec::new() };
            segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress, tail, played: committed });
        }
        Ok(Attempt { segments, frames, died, ga, dense })
    }

    /// Commits `pads` (played from `ctx`'s last state on `cur`): replays
    /// them there, reading the states with objects, and appends them.
    /// Whether the stage's ETA reached a new low or a stage was met.
    fn cur_replay(&mut self, stages: &[&dyn Objective], start: &St, ctx: &mut Ctx, pads: &[GamePad], rh: &Rh, frames: &mut u64) -> Result<bool> {
        if pads.is_empty() {
            return Ok(false);
        }
        let e = play(&mut self.cur, stages, start, ctx, Src::Replay(pads), pads.len(), true, rh)?;
        *frames += e.stepped as u64;
        debug_assert_eq!(e.pads.len(), pads.len(), "a commit replays as it was played");
        Ok(ctx.absorb(e))
    }


    /// Dense adds, after the GA: candidates among `traj` (the rollout of
    /// `genome` from the start, objects dropped), each with its inputs
    /// count, replayed once from the start to read their objects and take
    /// the saves they need.
    fn collect_posthoc(&mut self, c: &mut Collector, genome: &[GamePad], states: &[(usize, &St)], start: &St) -> Result<()> {
        let mut want = Vec::new();
        for &(t, st) in states {
            if t == 0 || t > genome.len() {
                continue;
            }
            if let Some(k) = c.wanted(st, start) {
                want.push((t, k));
            }
        }
        if want.is_empty() {
            return Ok(());
        }
        self.tmp.sync_from(&self.base)?;
        let mut cands = Vec::with_capacity(want.len());
        let mut played = 0;
        let mut last = None;
        for (t, k) in want {
            while played < t {
                self.tmp.step_game(genome[played])?;
                played += 1;
            }
            let st = read_state(&self.tmp, start.frame + t as u32, true);
            let save = if c.needs_save(t, last) {
                last = Some(t);
                c.stats.saves += 1;
                Some((c.req.pack)(&self.tmp.save_state()))
            } else {
                None
            };
            cands.push((t, st, save, k));
        }
        c.stats.replay_frames += played as u64;
        c.chains.push(DChain { genome: genome[..played].to_vec(), cands });
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The default optimiser: v1's open-loop GA, with the ETA fitness.

/// v1's random inputs: `new` (the new operators) uses
/// [`random_segment_v2`]'s per-button spans.
fn random_inputs_v1(rng: &mut SmallRng, n: usize) -> Vec<GamePad> {
    let mut out = Vec::with_capacity(n + 32);
    while out.len() < n {
        let len = rng.random_range(3..30);
        random_segment_v2(rng, &mut out, len);
    }
    out.truncate(n);
    out
}

/// `OBRRT_GA_GAMMA` (default 0.99): the open GA's discount per frame.
/// `OBRRT_GA_REGRET=N` (frames; unset or 0: off): a prefix whose ETA rises
/// within the next N frames loses the discounted rise (see [`regret_best`]).
fn ga_regret() -> usize {
    obrrt_core::config::current().ga.regret
}

/// The open fitness with regret: per step k (state k + 1) the discounted sum
/// S_k minus max over j in (k, k + n] of γ^j × (eta_j − eta_k) for a rise in
/// the same stage, or γ^j × `death` for a death or Lost at j; the best value
/// (> 0, the earliest on ties) and its state index, and that step's cut ETA
/// (None: the empty prefix). One pass of n per step after the fold.
fn regret_best(rec: &[(f64, f64, usize, f64)], gamma: f64, n: usize, death: f64) -> (f64, usize, Option<f64>) {
    let mut disc = vec![1.0; rec.len()];
    for k in 1..rec.len() {
        disc[k] = disc[k - 1] * gamma;
    }
    let (mut best, mut at, mut ce) = (0.0, 0usize, None);
    let mut sum = 0.0;
    for k in 0..rec.len() {
        sum += disc[k] * rec[k].0;
        let mut worst = 0.0f64;
        if !rec[k].1.is_nan() {
            for j in k + 1..(k + 1 + n).min(rec.len()) {
                let rise = if rec[j].1.is_nan() { death } else if rec[j].2 == rec[k].2 { rec[j].1 - rec[k].1 } else { 0.0 };
                if rise > 0.0 {
                    worst = worst.max(disc[j] * rise);
                }
            }
        }
        let v = sum - worst;
        if v > best {
            best = v;
            at = k + 1;
            if !rec[k].1.is_nan() {
                ce = Some(rec[k].3);
            }
        }
    }
    (best, at, ce)
}

fn ga_gamma() -> f64 {
    obrrt_core::config::current().ga.gamma
}

/// The GA budget of objectives without [`Objective::ga_budget`]:
/// 16 individuals, 5 generations, horizon 150 (v1's most common budget).
/// OBRRT_REACH_PRIOR=<file>: a JSON object, objective kind -> the start ETA
/// (frames) beyond which attempts of that kind have not succeeded in past
/// runs (backbone/attempts/reach.py, from every attempt logged). An attempt
/// from further away still moves Mario toward the objective and adds a state,
/// but at the full budget it cost about five times a move: Boss::Hit from
/// over 200 frames away was 13,607 attempts, 1 success and 0.96 B frames.
fn reach_prior() -> &'static std::collections::HashMap<String, f64> {
    obrrt_core::per_config!(std::collections::HashMap<String, f64>, |cfg| {
        let Some(path) = cfg.priors.reach_prior.clone() else { return Default::default() };
        let text = obrrt_core::sys::fs::read_to_string(&path).unwrap_or_else(|e| panic!("OBRRT_REACH_PRIOR {path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("OBRRT_REACH_PRIOR {path}: {e}"))
    })
}

/// The budget of an attempt from beyond its kind's reach (12, 3, 120: about
/// what a move costs).
fn reach_budget() -> GaBudget {
    GaBudget { population: 12, generations: 3, horizon: 120 }
}

/// OBRRT_RESCUE_PRIOR=<file>: a JSON object, objective kind -> how often an
/// attempt of that kind still met its objective when no rollout of
/// generation 0 did, in past runs (backbone/attempts/rescue.py). Across the
/// 776,061 single-objective attempts on disk on 2026-11-09, generation 0 met
/// the objective in 25% and 92.5% of those succeeded; of the rest 8.2%
/// succeeded, and they took 83% of all frames. So for a kind whose rescue
/// rate is under 0.05, an attempt whose
/// generation 0 missed stops after 2 more generations unless one has met the objective by then: it still adds its best end
/// state, as an approach. Returns that number of generations for `o`.
fn gen0_gate(o: &dyn Objective) -> Option<usize> {
    let (prior, min, gens) = obrrt_core::per_config!((std::collections::HashMap<String, f64>, f64, usize), |cfg| {
        let Some(path) = cfg.priors.rescue_prior.clone() else { return Default::default() };
        let text = obrrt_core::sys::fs::read_to_string(&path).unwrap_or_else(|e| panic!("OBRRT_RESCUE_PRIOR {path}: {e}"));
        (serde_json::from_str(&text).unwrap_or_else(|e| panic!("OBRRT_RESCUE_PRIOR {path}: {e}")), 0.05, 2usize)
    });
    prior.get(o.kind()).is_some_and(|r| r < min).then_some(*gens)
}

/// Whether `start` is further from `o` than its kind's reach (false without a prior).
pub fn beyond_reach(o: &dyn Objective, start: &St) -> bool {
    reach_prior().get(o.kind()).is_some_and(|&r| obrrt_core::start_eta(o, start) > r)
}

fn default_budget() -> GaBudget {
    GaBudget { population: 16, generations: 5, horizon: 150 }
}

#[derive(Clone)]
struct Rollout {
    genome: Vec<GamePad>,
    /// Without objects (dropped once the rollout is scored: a population's
    /// object lists took most of the heap); the attempt replays the winner's
    /// for the states it needs them in.
    traj: Vec<St>,
    /// Trajectory index at which each met objective was met, in order.
    ends: Vec<usize>,
    /// max over t of Σ_{i<t} γ^i r_i, the discount from the attempt's start
    /// across stages (see [`Worker::rollout_inner`]).
    fitness: f64,
    /// Trajectory index of that maximum (the earliest on ties): the cut.
    at: usize,
    /// The ETA of the stage worked on at `at`, and at the rollout's end.
    cut_eta: f64,
    final_eta: f64,
    died: bool,
    /// Stopped because an objective was met where Mario isn't controllable
    /// (or the next can't be met from there): the next one wasn't attempted.
    cut: bool,
    generation: u16,
    origin: u8,
    /// `OBRRT_MOVE_GA`: the policy-seeded plan this rollout is or descends
    /// from (index into the attempt's plans; `u32::MAX` none), and the
    /// policy of the `repolicy` mutation that made it (`u8::MAX` none).
    plan: u32,
    rp: u8,
    /// Calm: jump starts from the ground in the trajectory (the
    /// mover's [`obrrt_core::mover::Mover::is_jump_start`]; 0 when off).
    jumps: u32,
    /// Genome indices an objective's keyed overlay marked as key frames.
    key: Vec<usize>,
    /// `OBRRT_GA_GAMMAS` (qd.rs): (fitness, argmax) under each extra γ;
    /// empty otherwise.
    qd_fit: Vec<(f64, usize)>,
}

/// OBRRT_GA_QD: files `r` into the archive under each γ (qd.rs).
fn qd_insert(a: &mut crate::qd::Archive, q: &crate::qd::Qd, r: &Rollout, n_stages: usize, start: &St, target: Option<[f32; 3]>) {
    let met = r.ends.len() == n_stages;
    for g in 0..q.gammas.len() {
        let (fit, at) = if g == 0 { (r.fitness, r.at) } else { r.qd_fit.get(g - 1).copied().unwrap_or((r.fitness, r.at)) };
        let end_t = if met { *r.ends.last().unwrap() } else { at }.min(r.traj.len() - 1);
        let bin = crate::qd::descriptor(q, g as u8, &r.traj[end_t], met, start, &r.traj[end_t / 2], target);
        if a.bins.get(&bin).is_some_and(|o| o.fitness > fit || (o.fitness == fit && o.end_t <= end_t)) {
            continue;
        }
        a.insert(bin, crate::qd::Entry { genome: r.genome.clone(), plan: r.plan, key: r.key.clone(), fitness: fit, end_t, met, end: r.traj[end_t].clone() });
    }
}

impl Rollout {
    /// Greater is better: fitness, then fewer frames to its maximum, then
    /// the lower final ETA, then fewer jump starts.
    fn better(&self, other: &Rollout) -> Ordering {
        let o = self
            .fitness
            .total_cmp(&other.fitness)
            .then(other.at.cmp(&self.at))
            .then(other.final_eta.total_cmp(&self.final_eta));
        if o.is_eq() { other.jumps.cmp(&self.jumps) } else { o }
    }
}

impl Worker {
    fn rollout(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        horizon: usize,
    ) -> Result<Rollout> {
        self.rollout_inner(stages, start, prefix, steer_to, None, None, horizon)
    }

    /// Plays `prefix` from the start, then (with `steer_to`) steers
    /// closed-loop towards the current objective (or follows `plan`, the
    /// mover's policies, or `policy`, a [`Budget::Policy`]) until `horizon`
    /// frames, recording every state, folding the current stage's ETA every
    /// frame (one scratch from the start per stage) and moving on to the
    /// next stage as each is met. The reward of frame t is
    /// eta_{t-1} − eta_t, plus `OBRRT_RH_BONUS` on a stage's first Done
    /// (the next stage's ETA starts from that state: no reward for the
    /// jump); death or `Lost` give −`OBRRT_RH_DEATH` and end the rollout.
    /// Fitness is the best discounted running sum (γ = `OBRRT_GA_GAMMA`
    /// per frame from the start, across stages). Also stops once all are
    /// met, or when one is met where Mario isn't controllable.
    #[allow(clippy::too_many_arguments)]
    fn rollout_inner(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        policy: Option<&PadPolicy>,
        plan: Option<&crate::movegenome::Plan>,
        horizon: usize,
    ) -> Result<Rollout> {
        let rh = rh();
        let gamma = ga_gamma();
        self.tmp.sync_from(&self.base)?;
        let objects = stages.iter().any(|o| o.needs_objects());
        let mut genome = Vec::with_capacity(horizon);
        let mut traj = Vec::with_capacity(horizon + 1);
        // Objectives' per-frame overlays on the GA's pads (none for policies,
        // which play their own inputs).
        let overlays: Vec<Option<PadOverlay>> =
            if policy.is_some() { Vec::new() } else { stages.iter().map(|o| o.pad_overlay()).collect() };
        let raw = if RAW_PREFIX.with(|r| r.get()) { prefix.len() } else { 0 };
        traj.push(start.clone());
        let mut ends = Vec::new();
        let mut key = Vec::new();
        let mut seg = 0;
        let (mut died, mut cut) = (false, false);
        let mut mem = Mem::default();
        let mut prev = 0.0;
        // Objectives already met at the start; the first unmet one's ETA there.
        while ends.len() < stages.len() {
            mem = Mem::default();
            let e = stages[ends.len()].eta(start, start, &mut mem);
            if !e.done() {
                prev = e.frames();
                break;
            }
            ends.push(0);
        }
        if ends.len() < stages.len() && !prev.is_finite() {
            cut = true;
        }
        let (mut sum, mut disc) = (0.0, 1.0);
        let (mut best, mut at, mut cut_eta) = (0.0, 0, prev);
        // OBRRT_GA_REGRET=N: per step (reward, the stage's ETA after it, the
        // stage index, the cut ETA), for the regretted argmax after the fold.
        let regret = ga_regret();
        let mut rec: Vec<(f64, f64, usize, f64)> = Vec::new();
        // OBRRT_GA_GAMMAS (qd.rs): the same rewards under the extra γ's.
        let qx = crate::qd::extra_gammas();
        let (mut qx_sum, mut qx_disc, mut qx_best, mut qx_at) = (vec![0.0; qx.len()], vec![1.0; qx.len()], vec![0.0; qx.len()], vec![0usize; qx.len()]);
        'frames: for f in 0..horizon {
            if ends.len() == stages.len() || cut {
                break;
            }
            let k = ends.len();
            let pad = match (prefix.get(f), steer_to) {
                (Some(p), _) => *p,
                (None, _) if policy.is_some() => policy.unwrap().pad(&traj, &genome),
                (None, Some(every)) => match plan {
                    Some(pl) => crate::movegenome::pad(pl, f - prefix.len(), stages[k], traj.last().unwrap(), f, every),
                    None => steer(stages[k], traj.last().unwrap(), f, every),
                },
                (None, None) => break,
            };
            let pad = match overlays.get(k) {
                Some(Some(o)) if f >= raw => {
                    let (p, kf) = o.pad_keyed(&traj, &genome, pad);
                    if kf {
                        key.push(genome.len());
                    }
                    p
                }
                _ => pad,
            };
            self.tmp.step_game(pad)?;
            let st = read_state(&self.tmp, start.frame + f as u32 + 1, objects);
            let mut r = 0.0;
            let mut end = false;
            if !st.alive() || st.lives < start.lives {
                died = true;
                r -= rh.death;
                end = true;
            }
            let eta = if end { Eta::Lost } else { stages[k].eta(&st, &traj[seg], &mut mem) };
            // A dialog or menu option is over once its dialog or menu is,
            // whichever option was taken.
            let lost = !end
                && (match eta {
                    Eta::Lost => true,
                    Eta::Frames(x) => !x.is_finite(),
                    Eta::Done => false,
                } || (stages[k].modal() && !eta.done() && !stages[k].exists_in(&st)));
            if lost {
                r -= rh.death;
                end = true;
            }
            if end {
                if regret > 0 {
                    // A death or Lost: a rise of the death penalty.
                    rec.push((r, f64::NAN, ends.len(), cut_eta));
                }
                sum += disc * r;
                if sum > best {
                    (best, at) = (sum, traj.len());
                }
                for i in 0..qx.len() {
                    qx_sum[i] += qx_disc[i] * r;
                    if qx_sum[i] > qx_best[i] {
                        (qx_best[i], qx_at[i]) = (qx_sum[i], traj.len());
                    }
                }
                break;
            }
            genome.push(pad);
            traj.push(st);
            r += prev - eta.frames();
            if eta.done() {
                // What the goal still costs after this state (0 for most).
                let after = stages[k].done_frames(traj.last().unwrap(), &mem);
                r += if after == 0.0 { rh.bonus } else { rh.bonus - after };
                seg = traj.len() - 1;
                ends.push(seg);
                // The next stage's ETA from here (its fold's first state).
                while ends.len() < stages.len() {
                    mem = Mem::default();
                    let e = stages[ends.len()].eta(&traj[seg], &traj[seg], &mut mem);
                    if e.done() {
                        r += rh.bonus;
                        ends.push(seg);
                        continue;
                    }
                    prev = e.frames();
                    if !prev.is_finite() {
                        cut = true;
                    }
                    break;
                }
                if ends.len() < stages.len() && !traj[seg].controllable() {
                    cut = true;
                }
            } else {
                prev = eta.frames();
            }
            if regret > 0 {
                let ce = if ends.len() < stages.len() { prev } else { 0.0 };
                // The ETA of the stage this step ended in (a met stage: 0).
                let stage_eta = if eta.done() { 0.0 } else { eta.frames() };
                rec.push((r, stage_eta, if eta.done() { ends.len() - 1 } else { ends.len() }, ce));
            }
            sum += disc * r;
            disc *= gamma;
            if sum > best {
                (best, at, cut_eta) = (sum, traj.len() - 1, if ends.len() < stages.len() { prev } else { 0.0 });
            }
            for i in 0..qx.len() {
                qx_sum[i] += qx_disc[i] * r;
                qx_disc[i] *= qx[i];
                if qx_sum[i] > qx_best[i] {
                    (qx_best[i], qx_at[i]) = (qx_sum[i], traj.len() - 1);
                }
            }
            if cut {
                break 'frames;
            }
        }
        // OBRRT_GA_REGRET: the argmax by regretted prefix values (the primary
        // γ and the extra ones alike).
        // When no prefix keeps a positive value after regret (every gain is
        // undone within N, e.g. dives sliding back down a steep slope), the
        // raw argmax stays: the attempt still keeps its best partial progress
        // instead of nothing.
        if regret > 0 && !rec.is_empty() {
            let (b, a, c) = regret_best(&rec, gamma, regret, rh.death);
            if b > 0.0 {
                (best, at) = (b, a);
                if let Some(c) = c {
                    cut_eta = c;
                }
            }
            for i in 0..qx.len() {
                let (b, a, _) = regret_best(&rec, qx[i], regret, rh.death);
                if b > 0.0 {
                    (qx_best[i], qx_at[i]) = (b, a);
                }
            }
        }
        if ends.len() == stages.len() && !died {
            sink_met(&genome[..*ends.last().unwrap()], &traj[*ends.last().unwrap()]);
        }
        let at = at.min(traj.len() - 1);
        let traj_len = traj.len();
        let final_eta = if ends.len() < stages.len() { prev } else { 0.0 };
        let jumps = match obrrt_core::mover::mover() {
            Some(m) => traj.windows(2).filter(|w| m.is_jump_start(&w[0], &w[1])).count() as u32,
            _ => 0,
        };
        if objects {
            for s in &mut traj {
                s.objects = None;
            }
        }
        Ok(Rollout {
            genome,
            traj,
            ends,
            fitness: best,
            at,
            cut_eta,
            final_eta,
            died,
            cut,
            generation: 0,
            origin: 0,
            plan: u32::MAX,
            rp: u8::MAX,
            jumps,
            key,
            qd_fit: qx_best.into_iter().zip(qx_at).map(|(b, a)| (b, a.min(traj_len - 1))).collect(),
        })
    }

    /// v1's attempt (the default): a single objective with a fixed play
    /// plays it; otherwise an open-loop GA over raw inputs whose budget
    /// ([`Objective::ga_budget`], combined over the stages as v1 did) covers
    /// all the stages, ranked by the ETA fitness ([`Worker::rollout_inner`]).
    /// The kept segment of an unmet stage ends at the winner's fitness
    /// maximum. `params.scale` multiplies generations.
    #[allow(clippy::too_many_arguments)]
    fn attempt_open(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        save: Option<&SaveState>,
        rng: &mut SmallRng,
        params: GaParams,
        seed: Option<&[GamePad]>,
    ) -> Result<Attempt> {
        RAW_PREFIX.with(|r| r.set(seed.is_some()));
        let out = self.attempt_open_inner(dc, stages, start, save, rng, params, seed);
        RAW_PREFIX.with(|r| r.set(false));
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn attempt_open_inner(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        save: Option<&SaveState>,
        rng: &mut SmallRng,
        params: GaParams,
        seed: Option<&[GamePad]>,
    ) -> Result<Attempt> {
        let ga_scale = params.scale;
        if let Some(save) = save {
            self.base.load_state(save)?;
        }
        let mut frames = 0u64;
        // Dense adds, `last_gen`: the last generation's other rollouts
        // (inputs, final state); the winner's is taken from `best`.
        let mut last_gen: Vec<(Vec<GamePad>, St)> = Vec::new();
        let mut ga = None;
        // OBRRT_GA_QD: the archive's picks, and (uniform, elite) parents.
        let mut qd_out: Option<crate::qd::Out> = None;
        let mut qd_parents = (0u64, 0u64);
        let fixed = match (stages.len(), stages[0].fixed()) {
            (1, Some(Budget::Macro(pads))) => Some((pads.len(), pads, None)),
            (1, Some(Budget::Policy { limit, policy })) => Some((limit, Vec::new(), Some(policy))),
            _ => None,
        };
        let best = if let Some((limit, pads, policy)) = fixed {
            let r = self.rollout_inner(stages, start, &pads, None, policy.as_ref(), None, limit)?;
            frames += r.genome.len() as u64;
            r
        } else {
            let (mut population, mut generations, mut horizon) = (1, 1, 0);
            for o in stages {
                let mut b = o.ga_budget().unwrap_or_else(default_budget);
                // OBRRT_REACH_PRIOR: from further than the kind has ever
                // succeeded from, the attempt is an approach, at a move's budget.
                if stages.len() == 1 && beyond_reach(*o, start) {
                    let d = reach_budget();
                    b = GaBudget { population: b.population.min(d.population), generations: b.generations.min(d.generations), horizon: b.horizon.min(d.horizon) };
                }
                population = population.max(b.population);
                generations = generations.max(b.generations);
                horizon += b.horizon;
            }
            generations += stages.len() - 1;
            if let Some(s) = seed {
                horizon = horizon.max(s.len());
            }
            let generations = ((generations as f64 * ga_scale).round() as usize).max(1);
            let pure = params.pink >= 1.0;
            let target = if pure { None } else { stages[0].steer_target(start).map(|_| ()) };
            let mut pop: Vec<Rollout> = Vec::with_capacity(population);
            let mut seed_hit = None;
            // OBRRT_MOVE_GA: policy-seeded rollouts in place of some of the
            // first generation (movegenome::slot), from the mover's plausible
            // policies; `ok` also feeds the repolicy mutation.
            let move_ga = crate::movegenome::enabled() && target.is_some() && (seed.is_none() || crate::movegenome::with_seed()) && !pure;
            let ok = if move_ga { crate::movegenome::plausible(stages[0], start) } else { Vec::new() };
            // OBRRT_MOVE_GA_EXTRA: on top of the population instead.
            let extra = crate::movegenome::extra();
            let pass = stages[0].arrival(start) == obrrt_core::Arrival::Pass;
            let plans = if ok.is_empty() {
                Vec::new()
            } else {
                let n = crate::movegenome::gen0_budget(ok.len());
                crate::movegenome::gen0_plans(&ok, if extra { n } else { n.min(crate::movegenome::slots_in(population, pass)) }, rng)
            };
            let mut repolicies: Vec<usize> = Vec::new();
            let mut gen0_origins = [0u64; ORIGINS.len()];
            for i in 0..population + if extra { plans.len() } else { 0 } {
                let k = if extra { i.checked_sub(population) } else { crate::movegenome::slot(i, pass).filter(|&k| k < plans.len()) };
                if let Some(k) = k {
                    let mut r = self.rollout_inner(stages, start, &[], Some(4), None, Some(&plans[k]), horizon)?;
                    r.origin = ORIGIN_POLICY;
                    r.plan = k as u32;
                    frames += r.genome.len() as u64;
                    pop.push(r);
                    continue;
                }
                if let Some(s) = seed {
                    let mut g = s.to_vec();
                    g.truncate(horizon);
                    let mut steer_every = None;
                    if i > 0 && g.len() > 2 {
                        match i % 3 {
                            // Single frames dropped.
                            1 => {
                                for _ in 0..rng.random_range(1..=8) {
                                    if g.len() > 1 {
                                        g.remove(rng.random_range(0..g.len()));
                                    }
                                }
                            }
                            // A chunk dropped.
                            2 => {
                                let n = rng.random_range(2..=30).min(g.len() - 1);
                                let a = rng.random_range(0..=g.len() - n);
                                g.drain(a..a + n);
                            }
                            // Cut, then steered (or random) from there.
                            _ => {
                                let a = rng.random_range(g.len() / 3..g.len());
                                g.truncate(a);
                                if target.is_some() {
                                    steer_every = Some([2, 4, 8, 16, 30][rng.random_range(0..5)]);
                                } else {
                                    g.extend(random_inputs_v1(rng, horizon - a));
                                }
                            }
                        }
                    }
                    let mut r = self.rollout(stages, start, &g, steer_every, horizon)?;
                    if i == 0 {
                        seed_hit = Some((r.ends.len() == stages.len(), r.ends.last().copied().unwrap_or(0) as u32));
                    }
                    r.origin = ORIGIN_SEED;
                    frames += r.genome.len() as u64;
                    pop.push(r);
                    continue;
                }
                if params.pink > 0.0 && rng.random_bool(params.pink.min(1.0)) {
                    let g = crate::noise::pink_inputs(rng, horizon);
                    let mut r = self.rollout(stages, start, &g, None, horizon)?;
                    r.origin = ORIGIN_PINK;
                    frames += r.genome.len() as u64;
                    pop.push(r);
                    continue;
                }
                let mut r = match target {
                    Some(()) if i % 2 == 0 => {
                        let every = [2, 4, 8, 16, 30][rng.random_range(0..5)];
                        let n = rng.random_range(0..horizon / 4 + 1);
                        let noise = random_inputs_v1(rng, n);
                        self.rollout(stages, start, &noise, Some(every), horizon)?
                    }
                    _ => {
                        let mut r = self.rollout(stages, start, &random_inputs_v1(rng, horizon), None, horizon)?;
                        r.origin = ORIGIN_RANDOM;
                        r
                    }
                };
                r.generation = 0;
                frames += r.genome.len() as u64;
                pop.push(r);
            }
            // OBRRT_POLICY_SEED (+_EXTRA): further source plans, played as given.
            for x in EXTRA_SEEDS.with(|e| std::mem::take(&mut *e.borrow_mut())) {
                let mut g = x;
                g.truncate(horizon);
                RAW_PREFIX.with(|r| r.set(true));
                let r = self.rollout(stages, start, &g, None, horizon);
                RAW_PREFIX.with(|r| r.set(seed.is_some()));
                let mut r = r?;
                r.origin = ORIGIN_SEED;
                frames += r.genome.len() as u64;
                pop.push(r);
            }
            for r in &pop {
                gen0_origins[r.origin as usize] += 1;
            }
            // OBRRT_GA_KEEP_SEED: the seed as given (slot 0's rollout) stays
            // in the elite every generation, so the reference can't be bred out.
            let pinned = if seed.is_some() && obrrt_core::config::current().ga.keep_seed {
                pop.iter().find(|r| r.origin == ORIGIN_SEED).cloned()
            } else {
                None
            };
            let gen0_best = pop.iter().max_by(|a, b| a.better(b)).unwrap();
            let gen0 = (gen0_best.ends.len(), gen0_best.fitness);
            let gen0_plan = (gen0_best.origin == ORIGIN_POLICY).then_some(gen0_best.plan as usize);
            let mut made = [0u64; ORIGINS.len()];
            let mut breeds = 0u64;
            // OBRRT_POLICY_REPOLICY: the policy breeding operator (single stage only).
            let rp_p = if stages.len() == 1 { policy_repolicy() } else { 0.0 };
            let mut rp_session = if rp_p > 0.0 { BREED_HOOK.read().unwrap().clone().map(|h| h.session()) } else { None };
            let rp_save = if rp_session.is_some() { Some(self.base.save_state()) } else { None };
            let (mut rp_credit, mut rp_other) = (1.0f64, 1.0f64);
            let mut rp_ref = 0.0f64;
            let (mut rp_n_h, mut rp_fr_h, mut rp_f0, mut rp_made_total) = (0u64, 0u64, 0u64, 0u64);
            if rp_session.is_some() {
                REPOLICY_STATS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // OBRRT_GA_QD (qd.rs): the archive of distinct outcomes.
            let qd = crate::qd::settings();
            let mut arch = qd.map(|_| crate::qd::Archive::default());
            let qd_target = if qd.is_some() { stages[0].steer_target(start) } else { None };
            let (mut q_uni, mut q_eli) = (0u64, 0u64);
            // OBRRT_RESCUE_PRIOR: see gen0_gate.
            let gate = if stages.len() == 1 && gen0.0 < 1 && seed.is_none() { gen0_gate(stages[0]) } else { None };
            for generation in 1..generations {
                pop.sort_by(|a, b| b.better(a));
                if gate.is_some_and(|g| generation > g) && pop[0].ends.len() < stages.len() {
                    break;
                }
                if let (Some(q), Some(a)) = (qd, arch.as_mut()) {
                    for r in &pop {
                        qd_insert(a, q, r, stages.len(), start, qd_target);
                    }
                }
                // OBRRT_POLICY_REPOLICY_MARKET: credit of the last breed's children.
                if rp_session.is_some() && generation > 1 {
                    let (mut gh, mut go) = (0.0, 0.0);
                    for r in pop.iter().filter(|r| r.generation as usize == generation - 1) {
                        let g = (r.fitness - rp_ref).max(0.0);
                        if r.origin == ORIGIN_REPOLICY { gh += g } else { go += g }
                    }
                    let fr_o = frames.saturating_sub(rp_f0).saturating_sub(rp_fr_h).max(1) as f64;
                    if rp_n_h > 0 {
                        rp_credit = 0.7 * rp_credit + 0.3 * gh / (rp_fr_h.max(1) as f64) * 1e3;
                    }
                    if rp_made_total > rp_n_h {
                        rp_other = 0.7 * rp_other + 0.3 * go / fr_o * 1e3;
                    }
                    (rp_n_h, rp_fr_h, rp_made_total) = (0, 0, 0);
                }
                breeds += 1;
                let elite = (population / 4).max(1).min(pop.len());
                if rp_session.is_some() {
                    rp_ref = pop[..elite].iter().map(|r| r.fitness).sum::<f64>() / elite as f64;
                }
                rp_f0 = frames;
                // OBRRT_MOVE_GA: policy lineages fill at most half the elite
                // (the best always stays), so similar policy rollouts don't
                // crowd out the other genomes.
                if move_ga && !plans.is_empty() && elite > 1 {
                    let cap = elite.div_ceil(2);
                    let (mut keep, mut rest) = (Vec::with_capacity(elite), Vec::new());
                    let mut n = 0;
                    for (i, r) in pop.drain(..).enumerate() {
                        let lineage = r.plan != u32::MAX;
                        if keep.len() < elite && (i == 0 || !lineage || n < cap) {
                            n += lineage as usize;
                            keep.push(r);
                        } else {
                            rest.push(r);
                        }
                    }
                    let short = elite.saturating_sub(keep.len());
                    keep.extend(rest.into_iter().take(short));
                    pop = keep;
                }
                pop.truncate(elite);
                if let Some(p) = &pinned {
                    if !pop.iter().any(|r| r.genome == p.genome) {
                        *pop.last_mut().unwrap() = p.clone();
                    }
                }
                while pop.len() < population {
                    // OBRRT_POLICY_REPOLICY: cut a parent, let an inventory policy play on.
                    if let (Some(sess), Some(sv)) = (rp_session.as_mut(), rp_save.as_ref()) {
                        let p_eff = if repolicy_market() {
                            (rp_p * 2.0 * rp_credit / (rp_credit + rp_other)).clamp(rp_p * 0.25, (rp_p * 4.0).min(0.9))
                        } else {
                            rp_p
                        };
                        rp_made_total += 1;
                        if rng.random_bool(p_eff.min(1.0)) {
                            let parent = &pop[rng.random_range(0..elite)];
                            let cut = rng.random_range(0..parent.genome.len().max(1)).min(parent.genome.len());
                            if let Some((pads, spent)) =
                                sess.breed(&mut self.tmp, start, sv, stages[0], &parent.genome[..cut], &parent.traj[..=cut], horizon, rng)
                            {
                                let (lineage, rmode) = (parent.plan, seed.is_some());
                                let mut g = pads;
                                g.truncate(horizon);
                                RAW_PREFIX.with(|r| r.set(true));
                                let child = self.rollout(stages, start, &g, None, horizon);
                                RAW_PREFIX.with(|r| r.set(rmode));
                                let mut child = child?;
                                child.generation = generation as u16;
                                child.origin = ORIGIN_REPOLICY;
                                child.plan = lineage;
                                frames += spent + child.genome.len() as u64;
                                rp_n_h += 1;
                                rp_fr_h += spent + child.genome.len() as u64;
                                REPOLICY_STATS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                REPOLICY_STATS[2].fetch_add(spent, std::sync::atomic::Ordering::Relaxed);
                                made[ORIGIN_REPOLICY as usize] += 1;
                                pop.push(child);
                                continue;
                            }
                        }
                    }
                    // OBRRT_GA_QD: a uniform archive pick (a γ, then a bin)
                    // some of the time; else the elite as before.
                    let qpick = match (qd, arch.as_ref()) {
                        (Some(q), Some(a)) if !a.bins.is_empty() && rng.random_bool(q.uniform.clamp(0.0, 1.0)) => {
                            a.pick(rng, q.gammas.len()).map(|e| (e.genome.clone(), e.plan, e.key.clone()))
                        }
                        _ => None,
                    };
                    if qd.is_some() {
                        if qpick.is_some() { q_uni += 1 } else { q_eli += 1 }
                    }
                    let (mut g, lineage, pkey) = match qpick {
                        Some(x) => x,
                        None => {
                            let parent = &pop[rng.random_range(0..elite)];
                            (parent.genome.clone(), parent.plan, parent.key.clone())
                        }
                    };
                    // OBRRT_MOVE_GA: re-simulate a span under another policy.
                    if !ok.is_empty() && rng.random_bool(crate::movegenome::repolicy_p()) {
                        if let Some((pre, pl)) = crate::movegenome::repolicy(&g, &ok, horizon, rng) {
                            let mut child = self.rollout_inner(stages, start, &pre, Some(4), None, Some(&pl), horizon)?;
                            child.generation = generation as u16;
                            child.origin = ORIGIN_REPOLICY;
                            child.plan = lineage;
                            child.rp = pl.segs[0].0 as u8;
                            repolicies.push(pl.segs[0].0);
                            frames += child.genome.len() as u64;
                            made[ORIGIN_REPOLICY as usize] += 1;
                            pop.push(child);
                            continue;
                        }
                    }
                    // A keyed overlay's key frames: sometimes nudge just
                    // their sticks (no draw at all for other objectives).
                    if !pkey.is_empty() && rng.random_bool(KEY_P) {
                        let key = pkey;
                        let pick = rng.random_range(0..key.len());
                        for (j, &i) in key.iter().enumerate() {
                            // The chosen frame always, the others half the time.
                            if i >= g.len() || (j != pick && rng.random_bool(0.5)) {
                                continue;
                            }
                            let gauss = |rng: &mut SmallRng| {
                                let (u, v): (f32, f32) = (rng.random::<f32>().max(1e-7), rng.random::<f32>());
                                (-2.0 * u.ln()).sqrt() * (TAU * v).cos() * KEY_SIGMA
                            };
                            let (dx, dy) = (gauss(rng), gauss(rng));
                            g[i].stick_x = (g[i].stick_x as f32 + dx).round().clamp(-128.0, 127.0) as i8;
                            g[i].stick_y = (g[i].stick_y as f32 + dy).round().clamp(-128.0, 127.0) as i8;
                        }
                        g.truncate(horizon);
                        let mut child = self.rollout(stages, start, &g, target.map(|_| 4), horizon)?;
                        child.generation = generation as u16;
                        child.origin = ORIGIN_KEY;
                        child.plan = lineage;
                        frames += child.genome.len() as u64;
                        made[ORIGIN_KEY as usize] += 1;
                        pop.push(child);
                        continue;
                    }
                    // Children may continue past where the parent stopped.
                    let len = g.len().max(1);
                    let a = rng.random_range(0..len);
                    // The "press" mutation is new; the pure old GA doesn't have it.
                    let op = rng.random_range(0..if pure { 4 } else { 5 });
                    let mut child = match op {
                        0 => {
                            let b = (a + rng.random_range(3..40)).min(horizon);
                            g.truncate(horizon);
                            let mut seg = Vec::new();
                            if pure {
                                seg = crate::noise::pink_inputs(rng, b.saturating_sub(a).max(1));
                            } else {
                                random_segment(rng, &mut seg, b.saturating_sub(a).max(1));
                            }
                            g.splice(a.min(g.len())..b.min(g.len()), seg);
                            g.truncate(horizon);
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        1 => {
                            let b = (a + rng.random_range(5..60)).min(g.len());
                            let turn = rng.random_range(-40i32..40);
                            for p in &mut g[a.min(b)..b] {
                                let ang = (p.stick_y as f32).atan2(p.stick_x as f32) + turn as f32 * TAU / 360.0;
                                let mag = ((p.stick_x as f32).powi(2) + (p.stick_y as f32).powi(2)).sqrt().min(127.0);
                                p.stick_x = (mag * ang.cos()) as i8;
                                p.stick_y = (mag * ang.sin()) as i8;
                            }
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        2 => {
                            let b = (a + rng.random_range(1..12)).min(g.len());
                            for p in &mut g[a.min(b)..b] {
                                p.button ^= A_BUTTON;
                            }
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        4 => {
                            // Press A, B or Z for a few frames somewhere.
                            let mut button = [A_BUTTON, A_BUTTON, Z_TRIG, Z_TRIG, B_BUTTON][rng.random_range(0..5)];
                            // Calm: an A press only as often as A looks useful.
                            if button == A_BUTTON && !rng.random_bool(a_mult()) {
                                button = 0;
                            }
                            let n = rng.random_range(1..5);
                            g.truncate(horizon);
                            while g.len() < (a + n).min(horizon) {
                                g.push(GamePad::new(0, 0, 0));
                            }
                            for p in &mut g[a.min(horizon.saturating_sub(1))..(a + n).min(horizon)] {
                                p.button |= button;
                            }
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        _ => {
                            g.truncate(a);
                            match target {
                                Some(()) => {
                                    let every = [2, 4, 8, 16, 30][rng.random_range(0..5)];
                                    self.rollout(stages, start, &g, Some(every), horizon)?
                                }
                                None if pure => {
                                    g.extend(crate::noise::pink_inputs(rng, horizon - a));
                                    self.rollout(stages, start, &g, None, horizon)?
                                }
                                None => {
                                    let rest = random_inputs_v1(rng, horizon - a);
                                    g.extend(rest);
                                    self.rollout(stages, start, &g, None, horizon)?
                                }
                            }
                        }
                    };
                    child.generation = generation as u16;
                    child.origin = 2 + op as u8;
                    child.plan = lineage;
                    frames += child.genome.len() as u64;
                    made[child.origin as usize] += 1;
                    pop.push(child);
                }
            }
            {
                let mut m = makeup().lock().unwrap();
                m.attempts += 1;
                m.breeds += breeds;
                for i in 0..ORIGINS.len() {
                    m.gen0[i] += gen0_origins[i];
                    m.children[i] += made[i];
                }
            }
            pop.sort_by(|a, b| b.better(a));
            if let (Some(q), Some(mut a)) = (qd, arch.take()) {
                for r in &pop {
                    qd_insert(&mut a, q, r, stages.len(), start, qd_target);
                }
                let picks = a.picks(q.k);
                // [0]: distinct met descriptors; [g > 0]: those where γ g's
                // best differs from γ 0's (met rollouts end at their Done
                // frame under every γ, so the bins are the same).
                let mut met_bins = vec![0usize; q.gammas.len()];
                for (b, e) in a.bins.iter().filter(|(b, e)| e.met && b.0 == 0) {
                    met_bins[0] += 1;
                    for g in 1..q.gammas.len() {
                        if a.bins.get(&(g as u8, b.1)).is_some_and(|o| o.genome != e.genome) {
                            met_bins[g] += 1;
                        }
                    }
                }
                qd_out = Some(crate::qd::Out { picks, bins: a.bins.len(), met_bins });
                let _ = (q_uni, q_eli);
                qd_parents = (q_uni, q_eli);
            }
            if dc.is_some() {
                last_gen = pop[1..].iter().map(|r| (r.genome.clone(), r.traj.last().unwrap().clone())).collect();
            }
            let best = pop.swap_remove(0);
            door_overshoot(stages[0], start, &best.traj, &best.ends);
            if rp_session.is_some() && best.origin == ORIGIN_REPOLICY {
                REPOLICY_STATS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if move_ga {
                crate::movegenome::record(&crate::movegenome::Outcome {
                    plans: &plans,
                    repolicies: &repolicies,
                    gen0_best: gen0_plan,
                    winner: (best.plan != u32::MAX).then_some((best.plan as usize, best.origin == ORIGIN_POLICY)),
                    winner_repolicy: (best.origin == ORIGIN_REPOLICY).then_some(best.rp as usize),
                });
            }
            ga = Some(GaInfo {
                generations,
                winner_gen: best.generation,
                winner_origin: ORIGINS[best.origin as usize],
                gen0,
                best: (best.ends.len(), best.fitness),
                moves: MOVES.map(|(_, act)| best.traj.iter().filter(|s| s.action == act).count() as u16),
                seed: seed_hit,
            });
            best
        };

        let mut qd_added = 0u64;
        // Dense adds after the GA: the winner's end, the last generation's
        // final states.
        let dense = match dc.take() {
            Some(mut c) => {
                let n = best.traj.len() - 1;
                self.collect_posthoc(&mut c, &best.genome, &[(n, &best.traj[n])], start)?;
                for (g, last) in &last_gen {
                    self.collect_posthoc(&mut c, g, &[(g.len(), last)], start)?;
                }
                // OBRRT_GA_QD_ADD: the archive's distinct picks as candidates.
                if let Some(o) = qd_out.as_ref().filter(|_| crate::qd::settings().is_some_and(|q| q.add)) {
                    for p in &o.picks {
                        if !p.genome.is_empty() {
                            self.collect_posthoc(&mut c, &p.genome, &[(p.genome.len(), &p.end)], start)?;
                            qd_added += 1;
                        }
                    }
                }
                let out = c.finish();
                frames += out.stats.replay_frames;
                Some(Box::new(out))
            }
            None => None,
        };
        if let Some(o) = qd_out {
            crate::qd::note(&o, qd_parents.0, qd_parents.1, qd_added);
            crate::qd::set_last(o);
        }
        // Segment boundaries: where each met objective was met, then the
        // unmet one's cut (the winner's fitness maximum, if past them).
        let mut bounds: Vec<(usize, bool, f64)> = best.ends.iter().map(|&e| (e, true, 0.0)).collect();
        if best.ends.len() < stages.len() && !best.cut && best.at > best.ends.last().copied().unwrap_or(0) {
            bounds.push((best.at, false, -best.cut_eta));
        }
        let mut segments = Vec::with_capacity(bounds.len());
        self.tmp.sync_from(&self.base)?;
        let mut played = 0;
        let nb = bounds.len();
        for (i, (at, success, progress)) in bounds.into_iter().enumerate() {
            if at <= played {
                // Met where the previous one was (or at the start): nothing to add.
                segments.push(Segment { inputs: Vec::new(), end: None, success, progress, tail: Vec::new(), played: 0 });
                continue;
            }
            let mut inputs = best.genome[played..at].to_vec();
            let committed = inputs.len();
            for p in &inputs {
                self.tmp.step_game(*p)?;
            }
            frames += committed as u64;
            played = at;
            let tail: Vec<GamePad> = Vec::new();
            if i != nb - 1 {
                let st = read_state(&self.tmp, start.frame + played as u32, true);
                segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress, tail, played: committed });
                continue;
            }
            // Universal tail: press A through cutscenes and warps until the
            // player is in control again, or a dialog or menu opens (its
            // options are objectives of their own, see Objective::modal).
            let mut st = read_state(&self.tmp, start.frame + played as u32, true);
            let mut extra = 0;
            while !st.controllable() && !st.modal() && !st.ending() && extra < TAIL_LIMIT {
                let pad = GamePad::new(if extra % 4 == 0 { A_BUTTON } else { 0 }, 0, 0);
                self.tmp.step_game(pad)?;
                inputs.push(pad);
                extra += 1;
                st = read_state(&self.tmp, start.frame + (played + extra) as u32, false);
            }
            frames += extra as u64;
            // A life lost after the run's progress (keys, stars) went up is
            // a price paid, not a failure: a star touched during a death's
            // fade-out still counts, and the death warp is just the way out.
            let gained = st.progress() > start.progress();
            if !(st.controllable() || st.modal() || st.ending()) || (!gained && (!st.alive() || st.lives < start.lives)) {
                segments.push(Segment { inputs: Vec::new(), end: None, success: false, progress, tail: Vec::new(), played: 0 });
                return Ok(Attempt { segments, frames, died: true, ga, dense });
            }
            let st = read_state(&self.tmp, start.frame + (played + extra) as u32, true);
            // The plan was for the state before the automatic tail.
            let tail = if extra == 0 { tail } else { Vec::new() };
            segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress, tail, played: committed });
        }
        Ok(Attempt { segments, frames, died: best.died, ga, dense })
    }
}
