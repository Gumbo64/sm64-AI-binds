//! Step 5: attempting one objective from one start state, with a genetic
//! algorithm over raw per-frame inputs (or the objective's fixed macro).

use std::f32::consts::TAU;
use std::sync::OnceLock;

use anyhow::Result;
use obrrt_core::{Budget, GamePad, Objective, PadOverlay, PadPolicy, Precision, St, read_state};
use rand::Rng;
use rand::rngs::SmallRng;
use sm64_binds::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use sm64_binds::{SM64Game, SaveState};

/// Most frames the automatic tail spends pressing A through menus, dialog and
/// warps before giving up.
const TAIL_LIMIT: usize = 1500;

/// `OBRRT_GA_TIME=1`: rank rollouts by how soon they make their progress.
/// An unmet stage's progress is bucketed (width `OBRRT_GA_TIME_EPS`, default
/// 0.02, times the start's |progress| on it); within a bucket the rollout
/// that entered it first ranks higher, the kept segment is cut at that frame
/// (not at a later, marginally better state), and the end state's progress
/// rate over the last frames breaks ties. Successes: success heuristic, then
/// that rate. Off by default (the old ranking and cut, unchanged).
fn time_eps() -> Option<f64> {
    static T: OnceLock<Option<f64>> = OnceLock::new();
    *T.get_or_init(|| {
        if !std::env::var("OBRRT_GA_TIME").is_ok_and(|v| v == "1") {
            return None;
        }
        Some(std::env::var("OBRRT_GA_TIME_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.02))
    })
}

/// Frames the end momentum is measured over.
const MOMENTUM_FRAMES: usize = 4;

/// Scores beyond this are sentinels ("never in range"), not distances.
const SANE: f64 = 1e12;

/// Progress rate (score units per frame) into `traj[end]` over the last few
/// frames, from per-state progress: e.g. speed towards a distance target.
fn momentum(obj: &dyn Objective, traj: &[St], from: usize, end: usize) -> f64 {
    let k = MOMENTUM_FRAMES.min(end - from.min(end));
    if k == 0 {
        return 0.0;
    }
    let one = |i: usize| obj.progress(std::slice::from_ref(&traj[i])).score;
    let (a, b) = (one(end - k), one(end));
    if a.is_finite() && b.is_finite() && a.abs() < SANE && b.abs() < SANE { (b - a) / k as f64 } else { 0.0 }
}

/// One worker's two games: `base` holds the start state, `tmp` is reset from
/// it (cheaply, by dirty pages) for every candidate.
pub struct Worker {
    pub base: SM64Game,
    pub tmp: SM64Game,
}

/// One objective's part of an attempt. `S` is how the end's save state is
/// kept: a whole [`SaveState`], or (the tree's attempts, see
/// [`Attempt::map_saves`]) packed as a delta right in the worker.
pub struct Segment<S = SaveState> {
    /// Inputs from the previous segment's end (or the start) to this one's
    /// chosen state; the last segment also has the automatic tail if needed.
    pub inputs: Vec<GamePad>,
    /// The chosen state (success, or peak progress), read with objects. None
    /// when the segment is empty or the tail failed.
    pub end: Option<(St, S)>,
    pub success: bool,
    /// Best progress score reached (successes included).
    pub progress: f64,
}

/// How the GA's winner came about (for diagnostics): which generation and
/// what made it, and the best of generation 0 against the final best.
#[derive(Debug, Clone, Copy)]
pub struct GaInfo {
    pub generations: usize,
    pub winner_gen: u16,
    /// "seeded" (controller), "random", or the mutation that made it.
    pub winner_origin: &'static str,
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

/// How rollouts are ranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fitness {
    /// Goals met, then progress on the next (or the last's success heuristic).
    Met,
    /// Frames to goal 1 (sooner is better, unmet is worst), then frames to
    /// goal 2, ...; at the first goal neither met, progress on it; all met,
    /// the last's success heuristic.
    Steps,
}

pub const ORIGINS: [&str; 12] =
    ["seeded", "random", "span", "turn", "a-toggle", "resteer", "press", "pink", "key", "backseed", "policy", "repolicy"];

/// [`ORIGINS`] index of a first-generation policy-seeded rollout and of a
/// `repolicy` child (`OBRRT_MOVE_GA`, movegenome.rs).
const ORIGIN_POLICY: u8 = 10;
const ORIGIN_REPOLICY: u8 = 11;

/// Share of children of a parent with key frames (a keyed overlay's) that
/// change only those frames' sticks, by small Gaussian steps.
const KEY_P: f64 = 0.5;
/// The steps' standard deviation, in raw stick units.
const KEY_SIGMA: f32 = 20.0;

/// The GA's settings (experiment knobs included).
#[derive(Debug, Clone, Copy)]
pub struct GaParams {
    /// Multiplies generations.
    pub scale: f64,
    pub fitness: Fitness,
    /// New input operators (per-button spans in random segments, a mutation
    /// that inserts A/B/Z presses) and no stopping after the first success.
    pub new_ops: bool,
    /// Share of the starting population made of full-length pink-noise
    /// inputs (the pre-tree GAs' inputs). At 1, the GA is purely the old
    /// kind: no steering controller anywhere, and mutations splice and
    /// refill with pink noise.
    pub pink: f64,
    /// Per-objective budgets: an attempt whose stages are all
    /// [`Precision::Explore`] runs 1x generations and stops one generation
    /// after a full success (new operators and fitness unchanged); one with
    /// any `Precise` stage gets `scale` and no early stop as usual.
    pub per_objective_budget: bool,
}

/// An attempt's budget class for diagnostics: `"macro"` (plays a fixed
/// input sequence), `"explore"` (every stage [`Precision::Explore`]) or
/// `"precise"`. Classified whether or not per-objective budgets are on.
pub fn budget_class(stages: &[&dyn Objective]) -> &'static str {
    if stages.len() == 1 && matches!(stages[0].budget(), Budget::Macro(_) | Budget::Policy { .. }) {
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
    pub modes: crate::dense::Modes,
    pub grid: f32,
    pub cap: usize,
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
    /// The rollout being played (`all`): its candidates, and the inputs
    /// count at its last save (None: the start's anchor).
    cur: Vec<(usize, St, Option<crate::store::Delta>, u64)>,
    cur_save: Option<usize>,
}

impl<'a> Collector<'a> {
    fn new(req: &'a DenseReq<'a>) -> Self {
        Collector { req, best: Default::default(), chains: Vec::new(), stats: Default::default(), cur: Vec::new(), cur_save: None }
    }

    /// Whether `st` (`t` inputs into its rollout) is a new candidate; if so
    /// it is noted as the attempt's earliest at its key.
    fn wanted(&mut self, st: &St, start: &St) -> Option<u64> {
        self.stats.seen += 1;
        if !crate::dense::eligible(st, start) {
            return None;
        }
        let k = crate::dense::dkey(st, self.req.grid);
        if !(self.req.novel)(k, st.frame) || self.best.get(&k).is_some_and(|&f| f <= st.frame) {
            return None;
        }
        // With a cap, at most 2x it are held (each with its objects, ~10 KB)
        // until the attempt ends: later ones are refused.
        if self.req.cap > 0 && self.stats.novel >= 2 * self.req.cap as u64 {
            self.stats.capped += 1;
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
        if self.req.cap > 0 && all.len() > self.req.cap {
            let mut by_frame: Vec<usize> = (0..all.len()).collect();
            by_frame.sort_by_key(|&i| (all[i].2, i));
            let mut keep = vec![false; all.len()];
            for &i in &by_frame[..self.req.cap] {
                keep[i] = true;
            }
            self.stats.capped += (all.len() - self.req.cap) as u64;
            all = all.into_iter().zip(keep).filter(|x| x.1).map(|x| x.0).collect();
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
            .map(|s| Segment { inputs: s.inputs, end: s.end.map(|(st, save)| (st, f(save))), success: s.success, progress: s.progress })
            .collect();
        Attempt { segments, frames: self.frames, died: self.died, ga: self.ga, dense: self.dense }
    }
}

struct Rollout {
    genome: Vec<GamePad>,
    /// Without objects (dropped once the rollout is scored: a population's
    /// object lists took most of the heap); `attempt` replays the winner's
    /// for the states it needs them in.
    traj: Vec<St>,
    /// Trajectory index at which each met objective was met, in order.
    ends: Vec<usize>,
    /// Objectives met, then progress on the next one (or, all met, the last
    /// one's success heuristic, negated): compared in that order.
    fitness: (usize, f64),
    /// Trajectory index of the last segment's chosen state.
    peak: usize,
    progress: f64,
    died: bool,
    /// Stopped because an objective was met where Mario isn't controllable:
    /// the next one wasn't attempted.
    cut: bool,
    generation: u16,
    origin: u8,
    /// `OBRRT_MOVE_GA`: the policy-seeded plan this rollout is or descends
    /// from (index into the attempt's plans; `u32::MAX` none), and the
    /// policy of the `repolicy` mutation that made it (`u8::MAX` none).
    plan: u32,
    rp: u8,
    /// `OBRRT_CALM`: jump starts from the ground in the trajectory (the
    /// mover's [`obrrt_core::mover::Mover::is_jump_start`]; 0 when off).
    jumps: u32,
    /// Genome indices an objective's keyed overlay marked as key frames.
    key: Vec<usize>,
    /// With `OBRRT_GA_TIME`: the tiebreaks after goals met (greater is
    /// better): success heuristic negated or progress bucket, then minus
    /// the frame it was reached, end momentum, raw progress.
    tkey: [f64; 4],
}

impl Rollout {
    /// Greater is better.
    fn better(&self, other: &Rollout, mode: Fitness) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        if mode == Fitness::Steps {
            for k in 0.. {
                match (self.ends.get(k), other.ends.get(k)) {
                    (Some(a), Some(b)) if a != b => return b.cmp(a),
                    (Some(_), Some(_)) => continue,
                    (Some(_), None) => return Greater,
                    (None, Some(_)) => return Less,
                    (None, None) => break,
                }
            }
            // Same frames for every met goal: the shared tiebreak below.
            if time_eps().is_some() {
                return self.tcmp(other);
            }
            // OBRRT_CALM: at least one goal met at the same frames: fewer
            // jump starts first.
            if calm() && !self.ends.is_empty() && self.jumps != other.jumps {
                return other.jumps.cmp(&self.jumps);
            }
            return self.fitness.1.total_cmp(&other.fitness.1).then(Equal);
        }
        if time_eps().is_some() {
            return self.fitness.0.cmp(&other.fitness.0).then_with(|| self.tcmp(other));
        }
        let o = self.fitness.0.cmp(&other.fitness.0).then(self.fitness.1.total_cmp(&other.fitness.1));
        if o.is_eq() && calm() { other.jumps.cmp(&self.jumps) } else { o }
    }

    fn tcmp(&self, other: &Rollout) -> std::cmp::Ordering {
        let o = |i: usize| self.tkey[i].total_cmp(&other.tkey[i]);
        // OBRRT_CALM: within a bucket reached at the same frame, fewer jump
        // starts before the momentum and raw-progress tiebreaks.
        let jumps = if calm() { other.jumps.cmp(&self.jumps) } else { std::cmp::Ordering::Equal };
        o(0).then(o(1)).then(jumps).then(o(2)).then(o(3))
    }
}

/// Whether `obj` is met at the last state of `traj` (which starts at its
/// segment's start). Objectives with objects are asked about just the first
/// and last states (their success is per state, relative to the first), so
/// checking every frame stays cheap.
fn met(obj: &dyn Objective, traj: &[St]) -> bool {
    if traj.len() > 2 && obj.needs_objects() {
        obj.success(&[traj[0].clone(), traj[traj.len() - 1].clone()]).is_some()
    } else {
        obj.success(traj).is_some()
    }
}

/// Diagnostics (diag::door_overshoot): for a door target, whether the
/// winner came within 150 of the door airborne or in a landing before
/// meeting it.
fn door_overshoot(obj: &dyn Objective, start: &St, best: &Rollout) {
    if obj.arrival(start) != obrrt_core::Arrival::Door {
        return;
    }
    let (Some(m), Some(t)) = (obrrt_core::mover::mover(), obj.steer_target(start)) else { return };
    let end = best.ends.first().map_or(best.traj.len(), |&e| e + 1).min(best.traj.len());
    let over = best.traj[..end].iter().any(|s| m.in_flight(s) && s.hdist(t) < 150.0);
    crate::diag::door_overshoot(over, !best.ends.is_empty());
}

/// `OBRRT_CALM=1`: fewer pointless jumps (random genomes' A scaled by the
/// mover's `jump_useful`, ties broken by fewer jump starts).
fn calm() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_CALM").is_ok_and(|v| v == "1"))
}

fn a_mult() -> f64 {
    A_MULT.with(|a| a.get())
}

thread_local! {
    /// The attempt's multiplier on random genomes' A presses (`OBRRT_CALM`;
    /// exactly 1 otherwise, so the draws are unchanged).
    static A_MULT: std::cell::Cell<f64> = const { std::cell::Cell::new(1.0) };
    /// While a seeded attempt runs on this thread: the objectives' pad
    /// overlays are not applied to a rollout's given prefix frames (a seed
    /// is recorded tree inputs, already as played), only to steered ones.
    static RAW_PREFIX: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Closed-loop steering towards the objective's target, per control mode.
pub(crate) use crate::steer::steer;

/// A random held input for a few frames.
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
    c_turn(rng, out, at, len);
}

/// `OBRRT_GA_CBUTTONS=<p>`: the chance per random segment of a C-left or
/// C-right span (default 0: off, no extra draws). The stick is read
/// relative to the camera's yaw, which a held C button turns (~310 a
/// frame), so a fixed stick then steers along a turning line, and
/// directions between the 8-bit stick's become reachable.
fn c_buttons_p() -> f64 {
    static P: OnceLock<f64> = OnceLock::new();
    *P.get_or_init(|| std::env::var("OBRRT_GA_CBUTTONS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0))
}

fn c_turn(rng: &mut SmallRng, out: &mut [GamePad], at: usize, len: usize) {
    let p = c_buttons_p();
    if p <= 0.0 || len == 0 || !rng.random_bool(p.min(1.0)) {
        return;
    }
    let button = if rng.random_bool(0.5) { sm64_binds::buttons::L_CBUTTONS } else { sm64_binds::buttons::R_CBUTTONS };
    let from = rng.random_range(0..len);
    let n = if rng.random_bool(0.5) { rng.random_range(1..4) } else { rng.random_range(1..=len - from) };
    for pad in &mut out[at + from..(at + from + n).min(at + len)] {
        pad.button |= button;
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
    let at = out.len();
    for i in 0..len {
        let b = if tap && i > 0 { button & !A_BUTTON & !B_BUTTON } else { button };
        out.push(GamePad::new(b, sx, sy));
    }
    c_turn(rng, out, at, len);
}

fn random_inputs(rng: &mut SmallRng, n: usize, new: bool) -> Vec<GamePad> {
    let mut out = Vec::with_capacity(n + 32);
    while out.len() < n {
        let len = rng.random_range(3..30);
        if new { random_segment_v2(rng, &mut out, len) } else { random_segment(rng, &mut out, len) }
    }
    out.truncate(n);
    out
}

impl Worker {
    /// Plays `prefix`, then (with a target) steers closed-loop towards the
    /// current objective until `horizon` frames, recording every state and
    /// moving on to the next objective as each is met. Stops early at death,
    /// once all are met, or when one is met where Mario isn't controllable
    /// (a star dance, dialog): the next needs control.
    fn rollout(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        horizon: usize,
    ) -> Result<Rollout> {
        self.rollout_with(dc, stages, start, prefix, steer_to, None, horizon)
    }

    /// [`Worker::rollout`], with each frame after `prefix` from `policy` if
    /// given (a [`Budget::Policy`]).
    #[allow(clippy::too_many_arguments)]
    fn rollout_with(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        policy: Option<&PadPolicy>,
        horizon: usize,
    ) -> Result<Rollout> {
        self.rollout_inner(dc, stages, start, prefix, steer_to, policy, None, horizon)
    }

    /// [`Worker::rollout_with`], the steered frames after `prefix` from
    /// `plan` (movegenome.rs: mover policies, fixed pads, then the default
    /// steering) when given.
    #[allow(clippy::too_many_arguments)]
    fn rollout_inner(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        policy: Option<&PadPolicy>,
        plan: Option<&crate::movegenome::Plan>,
        horizon: usize,
    ) -> Result<Rollout> {
        self.tmp.sync_from(&self.base)?;
        let objects = stages.iter().any(|o| o.needs_objects());
        let mut genome = Vec::with_capacity(horizon);
        let mut traj = Vec::with_capacity(horizon + 1);
        // Objectives' per-frame overlays on the GA's pads (none for macros
        // and policies, which play their own inputs).
        let overlays: Vec<Option<PadOverlay>> =
            if policy.is_some() { Vec::new() } else { stages.iter().map(|o| o.pad_overlay()).collect() };
        let raw = if RAW_PREFIX.with(|r| r.get()) { prefix.len() } else { 0 };
        traj.push(start.clone());
        let mut ends = Vec::new();
        let mut key = Vec::new();
        let mut seg = 0;
        let mut died = false;
        let mut cut = false;
        // Objectives already met at the start.
        while ends.len() < stages.len() && met(stages[ends.len()], &traj[seg..]) {
            ends.push(0);
        }
        'frames: for f in 0..horizon {
            if ends.len() == stages.len() {
                break;
            }
            let pad = match (prefix.get(f), steer_to) {
                (Some(p), _) => *p,
                (None, _) if policy.is_some() => policy.unwrap().pad(&traj, &genome),
                (None, Some(every)) => match plan {
                    Some(pl) => crate::movegenome::pad(pl, f - prefix.len(), stages[ends.len()], traj.last().unwrap(), f, every),
                    None => steer(stages[ends.len()], traj.last().unwrap(), f, every),
                },
                (None, None) => break,
            };
            let pad = match overlays.get(ends.len()) {
                Some(Some(o)) if f >= raw => {
                    let (p, k) = o.pad_keyed(&traj, &genome, pad);
                    if k {
                        key.push(genome.len());
                    }
                    p
                }
                _ => pad,
            };
            self.tmp.step_game(pad)?;
            let st = read_state(&self.tmp, start.frame + f as u32 + 1, objects);
            if !st.alive() || st.lives < start.lives {
                died = true;
                break;
            }
            genome.push(pad);
            // Dense adds, `all`: candidates as they are played (objects
            // and saves read right here, as objective-scattershot does).
            if let Some(c) = dc.as_mut().filter(|c| c.req.modes.all).filter(|_| {
                // OBRRT_DENSE_EVERY / _AIR (the rollout's end isn't known
                // here: every n-th frame only).
                crate::dense::settings().filter(|d| d.thins()).is_none_or(|d| {
                    let rule = d.rule(stages[ends.len().min(stages.len() - 1)]);
                    crate::dense::thinned(rule, genome.len(), usize::MAX, &st, traj.last())
                })
            }) {
                if let Some(k) = c.wanted(&st, start) {
                    let t = genome.len();
                    let full = if st.objects.is_some() { st.clone() } else { read_state(&self.tmp, st.frame, true) };
                    let save = if c.needs_save(t, c.cur_save) {
                        c.cur_save = Some(t);
                        c.stats.saves += 1;
                        Some((c.req.pack)(&self.tmp.save_state()))
                    } else {
                        None
                    };
                    c.cur.push((t, full, save, k));
                }
            }
            traj.push(st);
            // A dialog or menu option is over once its dialog or menu is,
            // whichever option was taken.
            let k = ends.len();
            if stages[k].modal() && !met(stages[k], &traj[seg..]) && !stages[k].exists_in(traj.last().unwrap()) {
                break;
            }
            while ends.len() < stages.len() && met(stages[ends.len()], &traj[seg..]) {
                seg = traj.len() - 1;
                ends.push(seg);
                if ends.len() < stages.len() && !traj[seg].controllable() {
                    cut = true;
                    break 'frames;
                }
            }
        }
        if let Some(c) = dc.as_mut().filter(|c| c.req.modes.all) {
            c.cur_save = None;
            if !c.cur.is_empty() {
                let cands = std::mem::take(&mut c.cur);
                let n = cands.last().unwrap().0;
                c.chains.push(DChain { genome: genome[..n].to_vec(), cands });
            }
        }
        let done = ends.len();
        let mut tkey = [0.0; 4];
        let (fitness, peak, progress) = if done == stages.len() {
            let from = if done >= 2 { ends[done - 2] } else { 0 };
            let last = &traj[from..=ends[done - 1]];
            let obj = stages[done - 1];
            let sh = -obj.success_heuristic(last);
            if time_eps().is_some() {
                tkey = [sh, 0.0, momentum(obj, &traj, from, ends[done - 1]), 0.0];
            }
            ((done, sh), ends[done - 1], obj.progress(last).score.max(0.0))
        } else {
            let obj = stages[done];
            let p = obj.progress(&traj[seg..]);
            match time_eps() {
                None => ((done, p.score), seg + p.at, p.score),
                Some(eps) => {
                    // Bucket width from the attempt's start (the same for
                    // every rollout of this attempt and stage).
                    let p0 = obj.progress(std::slice::from_ref(start)).score;
                    let w = if p0.is_finite() && p0.abs() < SANE { eps * p0.abs() } else { 0.0 };
                    let score = |i: usize| obj.progress(&traj[seg..=seg + i]).score;
                    let (bucket, thr) = if w > 0.0 && p.score.abs() < SANE {
                        let q = (p.score / w).floor();
                        (q, (q * w).min(p.score))
                    } else {
                        (p.score, p.score)
                    };
                    // The first frame whose best-so-far reaches the bucket
                    // (lower bound; `hi` always qualifies).
                    let (mut lo, mut hi) = (0, p.at);
                    while lo < hi {
                        let mid = (lo + hi) / 2;
                        if score(mid) >= thr { hi = mid } else { lo = mid + 1 }
                    }
                    let at = seg + hi;
                    let reached = if hi == p.at { p.score } else { score(hi) };
                    // The frame it was reached counted from the attempt's
                    // start (`at`), not from this stage's: a chain that got
                    // to this stage sooner and reaches the bucket at the same
                    // game frame ties, one that reaches it later loses.
                    tkey = [bucket, -(at as f64), momentum(obj, &traj, seg, at), p.score];
                    ((done, p.score), at, reached)
                }
            }
        };
        let jumps = match obrrt_core::mover::mover() {
            Some(m) if calm() => traj.windows(2).filter(|w| m.is_jump_start(&w[0], &w[1])).count() as u32,
            _ => 0,
        };
        if objects {
            for s in &mut traj {
                s.objects = None;
            }
        }
        Ok(Rollout { genome, traj, ends, fitness, peak, progress, died, cut, generation: 0, origin: 0, plan: u32::MAX, rp: u8::MAX, jumps, key, tkey })
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

    /// Attempts `stages` in order from `start` (whose save state is `save`).
    /// A single objective with a macro budget plays its macro; otherwise a GA
    /// over raw inputs whose budget covers all the stages.
    /// `ga_scale` multiplies the GA's generations (an experiment knob).
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
    /// inputs that met the stages from this start) in the GA's first
    /// generation: the seed as is (played without overlays), then variants
    /// with single frames or a chunk dropped, or cut and steered. The
    /// horizon is stretched to the seed's length. `None` is `attempt`.
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
        RAW_PREFIX.with(|r| r.set(seed.is_some()));
        // OBRRT_CALM: random genomes' A presses scaled by how useful
        // jumping looks from the start (the mover's jump_useful).
        if calm() {
            let m = match (obrrt_core::mover::mover(), stages[0].steer_target(start)) {
                (Some(m), Some(t)) => (m.jump_useful(start, t, stages[0].arrival(start)) as f64).clamp(0.05, 1.0),
                _ => 1.0,
            };
            A_MULT.with(|a| a.set(m));
        }
        let mut dc = dense.filter(|d| d.modes.any()).map(Collector::new);
        let out = self.attempt_inner(&mut dc, stages, start, save, rng, params, seed);
        RAW_PREFIX.with(|r| r.set(false));
        A_MULT.with(|a| a.set(1.0));
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn attempt_inner(
        &mut self,
        dc: &mut Option<Collector>,
        stages: &[&dyn Objective],
        start: &St,
        save: Option<&SaveState>,
        rng: &mut SmallRng,
        params: GaParams,
        seed: Option<&[GamePad]>,
    ) -> Result<Attempt> {
        // Per-objective budgets: all-Explore attempts get the cheap GA (1x,
        // early stop); everything else is unchanged.
        let explore = params.per_objective_budget && stages.iter().all(|o| o.precision() == Precision::Explore);
        let ga_scale = if explore { 1.0 } else { params.scale };
        let early_stop = !params.new_ops || explore;
        let mode = params.fitness;
        if let Some(save) = save {
            self.base.load_state(save)?;
        }
        let mut frames = 0u64;
        // Dense adds, `last_gen`: the last generation's other rollouts
        // (inputs, final state); the winner's is taken from `best`.
        let mut last_gen: Vec<(Vec<GamePad>, St)> = Vec::new();
        let mut ga = None;
        let fixed = match (stages.len(), stages[0].budget()) {
            (1, Budget::Macro(pads)) => Some((pads.len(), pads, None)),
            (1, Budget::Policy { limit, policy }) => Some((limit, Vec::new(), Some(policy))),
            _ => None,
        };
        let best = if let Some((limit, pads, policy)) = fixed {
            let r = self.rollout_with(dc, stages, start, &pads, None, policy.as_ref(), limit)?;
            frames += r.genome.len() as u64;
            r
        } else {
            let (mut population, mut generations, mut horizon) = (1, 1, 0);
            for o in stages {
                if let Budget::Ga { population: p, generations: g, horizon: h } = o.budget() {
                    population = population.max(p);
                    generations = generations.max(g);
                    horizon += h;
                }
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
            let move_ga = crate::movegenome::enabled() && target.is_some() && seed.is_none() && !pure;
            let ok = if move_ga { crate::movegenome::plausible(stages[0], start) } else { Vec::new() };
            // OBRRT_MOVE_GA_EXTRA: on top of the population instead.
            let extra = crate::movegenome::extra();
            let pass = stages[0].arrival(start) == obrrt_core::Arrival::Pass;
            let plans = if ok.is_empty() {
                Vec::new()
            } else {
                let n = crate::movegenome::budget_n();
                crate::movegenome::gen0_plans(&ok, if extra { n } else { n.min(crate::movegenome::slots_in(population, pass)) }, rng)
            };
            let mut repolicies: Vec<usize> = Vec::new();
            for i in 0..population + if extra { plans.len() } else { 0 } {
                let k = if extra { i.checked_sub(population) } else { crate::movegenome::slot(i, pass).filter(|&k| k < plans.len()) };
                if let Some(k) = k {
                    let mut r = self.rollout_inner(dc, stages, start, &[], Some(4), None, Some(&plans[k]), horizon)?;
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
                                    g.extend(random_inputs(rng, horizon - a, params.new_ops));
                                }
                            }
                        }
                    }
                    let mut r = self.rollout(dc, stages, start, &g, steer_every, horizon)?;
                    if i == 0 {
                        seed_hit = Some((r.ends.len() == stages.len(), r.ends.last().copied().unwrap_or(0) as u32));
                    }
                    r.origin = 9;
                    frames += r.genome.len() as u64;
                    pop.push(r);
                    continue;
                }
                if params.pink > 0.0 && rng.random_bool(params.pink.min(1.0)) {
                    let g = crate::noise::pink_inputs(rng, horizon);
                    let mut r = self.rollout(dc, stages, start, &g, None, horizon)?;
                    r.origin = 7;
                    frames += r.genome.len() as u64;
                    pop.push(r);
                    continue;
                }
                let mut r = match target {
                    Some(()) if i % 2 == 0 => {
                        let every = [2, 4, 8, 16, 30][rng.random_range(0..5)];
                        let n = rng.random_range(0..horizon / 4 + 1);
                        let noise = random_inputs(rng, n, params.new_ops);
                        self.rollout(dc, stages, start, &noise, Some(every), horizon)?
                    }
                    _ => {
                        let mut r = self.rollout(dc, stages, start, &random_inputs(rng, horizon, params.new_ops), None, horizon)?;
                        r.origin = 1;
                        r
                    }
                };
                r.generation = 0;
                frames += r.genome.len() as u64;
                pop.push(r);
            }
            let gen0_best = pop.iter().max_by(|a, b| a.better(b, mode)).unwrap();
            let gen0 = gen0_best.fitness;
            let gen0_plan = (gen0_best.origin == ORIGIN_POLICY).then_some(gen0_best.plan as usize);
            let mut found_at = None;
            for generation in 1..generations {
                pop.sort_by(|a, b| b.better(a, mode));
                if pop[0].ends.len() == stages.len() && early_stop {
                    match found_at {
                        None => found_at = Some(generation),
                        Some(_) => break,
                    }
                }
                let elite = (population / 4).max(1);
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
                while pop.len() < population {
                    let parent = &pop[rng.random_range(0..elite)];
                    let mut g = parent.genome.clone();
                    let lineage = parent.plan;
                    // OBRRT_MOVE_GA: re-simulate a span under another policy.
                    if !ok.is_empty() && rng.random_bool(crate::movegenome::repolicy_p()) {
                        if let Some((pre, pl)) = crate::movegenome::repolicy(&g, &ok, horizon, rng) {
                            let mut child = self.rollout_inner(dc, stages, start, &pre, Some(4), None, Some(&pl), horizon)?;
                            child.generation = generation as u16;
                            child.origin = ORIGIN_REPOLICY;
                            child.plan = lineage;
                            child.rp = pl.segs[0].0 as u8;
                            repolicies.push(pl.segs[0].0);
                            frames += child.genome.len() as u64;
                            pop.push(child);
                            continue;
                        }
                    }
                    // A keyed overlay's key frames: sometimes nudge just
                    // their sticks (no draw at all for other objectives).
                    if !parent.key.is_empty() && rng.random_bool(KEY_P) {
                        let key = parent.key.clone();
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
                        let mut child = self.rollout(dc, stages, start, &g, target.map(|_| 4), horizon)?;
                        child.generation = generation as u16;
                        child.origin = 8;
                        child.plan = lineage;
                        frames += child.genome.len() as u64;
                        pop.push(child);
                        continue;
                    }
                    // Children may continue past where the parent stopped.
                    let len = g.len().max(1);
                    let a = rng.random_range(0..len);
                    // The "press" mutation is new; the pure old GA doesn't have it.
                    let op = rng.random_range(0..if params.new_ops && !pure { 5 } else { 4 });
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
                            self.rollout(dc, stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        1 => {
                            let b = (a + rng.random_range(5..60)).min(g.len());
                            let turn = rng.random_range(-40i32..40);
                            for p in &mut g[a..b] {
                                let ang = (p.stick_y as f32).atan2(p.stick_x as f32) + turn as f32 * TAU / 360.0;
                                let mag = ((p.stick_x as f32).powi(2) + (p.stick_y as f32).powi(2)).sqrt().min(127.0);
                                p.stick_x = (mag * ang.cos()) as i8;
                                p.stick_y = (mag * ang.sin()) as i8;
                            }
                            self.rollout(dc, stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        2 => {
                            let b = (a + rng.random_range(1..12)).min(g.len());
                            for p in &mut g[a..b] {
                                p.button ^= A_BUTTON;
                            }
                            self.rollout(dc, stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        4 => {
                            // Press A, B or Z for a few frames somewhere.
                            let mut button = [A_BUTTON, A_BUTTON, Z_TRIG, Z_TRIG, B_BUTTON][rng.random_range(0..5)];
                            // OBRRT_CALM: an A press only as often as A looks useful.
                            if button == A_BUTTON && calm() && !rng.random_bool(a_mult()) {
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
                            self.rollout(dc, stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        _ => {
                            g.truncate(a);
                            match target {
                                Some(()) => {
                                    let every = [2, 4, 8, 16, 30][rng.random_range(0..5)];
                                    self.rollout(dc, stages, start, &g, Some(every), horizon)?
                                }
                                None if pure => {
                                    g.extend(crate::noise::pink_inputs(rng, horizon - a));
                                    self.rollout(dc, stages, start, &g, None, horizon)?
                                }
                                None => {
                                    let rest = random_inputs(rng, horizon - a, params.new_ops);
                                    g.extend(rest);
                                    self.rollout(dc, stages, start, &g, None, horizon)?
                                }
                            }
                        }
                    };
                    child.generation = generation as u16;
                    child.origin = 2 + op as u8;
                    child.plan = lineage;
                    frames += child.genome.len() as u64;
                    pop.push(child);
                }
            }
            pop.sort_by(|a, b| b.better(a, mode));
            if dc.as_ref().is_some_and(|c| c.req.modes.last_gen && !c.req.modes.all) {
                last_gen = pop[1..].iter().map(|r| (r.genome.clone(), r.traj.last().unwrap().clone())).collect();
            }
            let best = pop.swap_remove(0);
            door_overshoot(stages[0], start, &best);
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
                gen0: (gen0.0, gen0.1),
                best: best.fitness,
                moves: MOVES.map(|(_, act)| best.traj.iter().filter(|s| s.action == act).count() as u16),
                seed: seed_hit,
            });
            best
        };

        // Dense adds after the GA: the winner's states, the last generation's
        // final states (`all` collected them as they were played).
        let dense = match dc.take() {
            Some(mut c) => {
                if !c.req.modes.all {
                    if c.req.modes.winner_all {
                        // OBRRT_DENSE_EVERY / _AIR: thinned by the rule of
                        // the stage each frame worked on (the winner's ends
                        // split its rollout into stages).
                        let ds = crate::dense::settings().filter(|d| d.thins());
                        let len = best.traj.len() - 1;
                        let stage_of = |t: usize| best.ends.iter().filter(|&&e| e < t).count().min(stages.len() - 1);
                        let states: Vec<(usize, &St)> = best
                            .traj
                            .iter()
                            .enumerate()
                            .skip(1)
                            .filter(|(t, st)| {
                                ds.as_ref().is_none_or(|d| crate::dense::thinned(d.rule(stages[stage_of(*t)]), *t, len, st, best.traj.get(t - 1)))
                            })
                            .collect();
                        self.collect_posthoc(&mut c, &best.genome, &states, start)?;
                    }
                    if c.req.modes.last_gen {
                        let n = best.traj.len() - 1;
                        self.collect_posthoc(&mut c, &best.genome, &[(n, &best.traj[n])], start)?;
                        for (g, last) in &last_gen {
                            self.collect_posthoc(&mut c, g, &[(g.len(), last)], start)?;
                        }
                    }
                }
                let out = c.finish();
                frames += out.stats.replay_frames;
                Some(Box::new(out))
            }
            None => None,
        };
        // Segment boundaries: where each met objective was met, then the
        // peak of the unmet one's progress.
        let mut bounds: Vec<(usize, bool)> = best.ends.iter().map(|&e| (e, true)).collect();
        if best.ends.len() < stages.len() && !best.cut {
            bounds.push((best.peak, false));
        }
        // The winner's states with objects (rollouts drop them), up to the
        // last met goal's: the same states again, since the game is
        // deterministic.
        let need = bounds.iter().filter(|b| b.1).map(|b| b.0).max().unwrap_or(0);
        let replayed;
        let traj: &[St] = if need > 0 && stages.iter().any(|o| o.needs_objects()) {
            self.tmp.sync_from(&self.base)?;
            let mut t = Vec::with_capacity(need + 1);
            t.push(start.clone());
            for (f, p) in best.genome[..need].iter().enumerate() {
                self.tmp.step_game(*p)?;
                t.push(read_state(&self.tmp, start.frame + f as u32 + 1, true));
            }
            replayed = t;
            &replayed
        } else {
            &best.traj
        };
        let progress_of = |i: usize, from: usize, to: usize| -> f64 {
            if bounds[i].1 {
                stages[i].progress(&traj[from..=to]).score.max(0.0)
            } else {
                best.progress
            }
        };
        let mut segments = Vec::with_capacity(bounds.len());
        self.tmp.sync_from(&self.base)?;
        let mut played = 0;
        for (i, &(at, success)) in bounds.iter().enumerate() {
            let progress = progress_of(i, played, at.max(played));
            if at <= played {
                // Met where the previous one was (or at the start): nothing to add.
                segments.push(Segment { inputs: Vec::new(), end: None, success, progress });
                continue;
            }
            let mut inputs = best.genome[played..at].to_vec();
            for p in &inputs {
                self.tmp.step_game(*p)?;
            }
            played = at;
            let last = i == bounds.len() - 1;
            if !last {
                let st = read_state(&self.tmp, start.frame + played as u32, true);
                segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress });
                continue;
            }
            // Universal tail: press A through cutscenes and warps until the
            // player is in control again, or a dialog or menu opens (its
            // options are objectives of their own, see Objective::modal).
            let mut st = read_state(&self.tmp, start.frame + played as u32, true);
            let mut tail = 0;
            while !st.controllable() && !st.modal() && tail < TAIL_LIMIT {
                let pad = GamePad::new(if tail % 4 == 0 { A_BUTTON } else { 0 }, 0, 0);
                self.tmp.step_game(pad)?;
                inputs.push(pad);
                tail += 1;
                st = read_state(&self.tmp, start.frame + (played + tail) as u32, false);
            }
            frames += tail as u64;
            // A life lost after the run's progress (keys, stars) went up is
            // a price paid, not a failure: a star touched during a death's
            // fade-out still counts, and the death warp is just the way out.
            let gained = st.progress() > start.progress();
            if !(st.controllable() || st.modal()) || (!gained && (!st.alive() || st.lives < start.lives)) {
                segments.push(Segment { inputs: Vec::new(), end: None, success: false, progress });
                return Ok(Attempt { segments, frames, died: true, ga, dense });
            }
            let st = read_state(&self.tmp, start.frame + (played + tail) as u32, true);
            segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress });
        }
        Ok(Attempt { segments, frames, died: best.died, ga, dense })
    }
}
