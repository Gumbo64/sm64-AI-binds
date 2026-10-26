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

pub const ORIGINS: [&str; 10] = ["seeded", "random", "span", "turn", "a-toggle", "resteer", "press", "pink", "key", "backseed"];

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
        Attempt { segments, frames: self.frames, died: self.died, ga: self.ga }
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
            return self.fitness.1.total_cmp(&other.fitness.1).then(Equal);
        }
        if time_eps().is_some() {
            return self.fitness.0.cmp(&other.fitness.0).then_with(|| self.tcmp(other));
        }
        self.fitness.0.cmp(&other.fitness.0).then(self.fitness.1.total_cmp(&other.fitness.1))
    }

    fn tcmp(&self, other: &Rollout) -> std::cmp::Ordering {
        self.tkey.iter().zip(&other.tkey).map(|(a, b)| a.total_cmp(b)).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal)
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

thread_local! {
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
    for (button, p) in [(A_BUTTON, 0.4), (B_BUTTON, 0.12), (Z_TRIG, 0.15)] {
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
    if rng.random_bool(0.35) {
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
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        horizon: usize,
    ) -> Result<Rollout> {
        self.rollout_with(stages, start, prefix, steer_to, None, horizon)
    }

    /// [`Worker::rollout`], with each frame after `prefix` from `policy` if
    /// given (a [`Budget::Policy`]).
    fn rollout_with(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        prefix: &[GamePad],
        steer_to: Option<usize>,
        policy: Option<&PadPolicy>,
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
                (None, Some(every)) => steer(stages[ends.len()], traj.last().unwrap(), f, every),
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
        if objects {
            for s in &mut traj {
                s.objects = None;
            }
        }
        Ok(Rollout { genome, traj, ends, fitness, peak, progress, died, cut, generation: 0, origin: 0, key, tkey })
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
        RAW_PREFIX.with(|r| r.set(seed.is_some()));
        let out = self.attempt_inner(stages, start, save, rng, params, seed);
        RAW_PREFIX.with(|r| r.set(false));
        out
    }

    fn attempt_inner(
        &mut self,
        stages: &[&dyn Objective],
        start: &St,
        save: &SaveState,
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
        self.base.load_state(save)?;
        let mut frames = 0u64;
        let mut ga = None;
        let fixed = match (stages.len(), stages[0].budget()) {
            (1, Budget::Macro(pads)) => Some((pads.len(), pads, None)),
            (1, Budget::Policy { limit, policy }) => Some((limit, Vec::new(), Some(policy))),
            _ => None,
        };
        let best = if let Some((limit, pads, policy)) = fixed {
            let r = self.rollout_with(stages, start, &pads, None, policy.as_ref(), limit)?;
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
            for i in 0..population {
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
                    let mut r = self.rollout(stages, start, &g, steer_every, horizon)?;
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
                    let mut r = self.rollout(stages, start, &g, None, horizon)?;
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
                        self.rollout(stages, start, &noise, Some(every), horizon)?
                    }
                    _ => {
                        let mut r = self.rollout(stages, start, &random_inputs(rng, horizon, params.new_ops), None, horizon)?;
                        r.origin = 1;
                        r
                    }
                };
                r.generation = 0;
                frames += r.genome.len() as u64;
                pop.push(r);
            }
            let gen0 = pop.iter().max_by(|a, b| a.better(b, mode)).map(|r| r.fitness).unwrap();
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
                pop.truncate(elite);
                while pop.len() < population {
                    let parent = &pop[rng.random_range(0..elite)];
                    let mut g = parent.genome.clone();
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
                        let mut child = self.rollout(stages, start, &g, target.map(|_| 4), horizon)?;
                        child.generation = generation as u16;
                        child.origin = 8;
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
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
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
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        2 => {
                            let b = (a + rng.random_range(1..12)).min(g.len());
                            for p in &mut g[a..b] {
                                p.button ^= A_BUTTON;
                            }
                            self.rollout(stages, start, &g, target.map(|_| 4), horizon)?
                        }
                        4 => {
                            // Press A, B or Z for a few frames somewhere.
                            let button = [A_BUTTON, A_BUTTON, Z_TRIG, Z_TRIG, B_BUTTON][rng.random_range(0..5)];
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
                                    let rest = random_inputs(rng, horizon - a, params.new_ops);
                                    g.extend(rest);
                                    self.rollout(stages, start, &g, None, horizon)?
                                }
                            }
                        }
                    };
                    child.generation = generation as u16;
                    child.origin = 2 + op as u8;
                    frames += child.genome.len() as u64;
                    pop.push(child);
                }
            }
            pop.sort_by(|a, b| b.better(a, mode));
            let best = pop.swap_remove(0);
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
                return Ok(Attempt { segments, frames, died: true, ga });
            }
            let st = read_state(&self.tmp, start.frame + (played + tail) as u32, true);
            segments.push(Segment { inputs, end: Some((st, self.tmp.save_state())), success, progress });
        }
        Ok(Attempt { segments, frames, died: best.died, ga })
    }
}
