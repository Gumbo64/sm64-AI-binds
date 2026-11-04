//! `beam`: successive halving over windows of the route, with a small beam of
//! alternative seeds.
//!
//! - **Arms**: windows of the best route: start at a reached route state, end
//!   `len` reached states later (`len` in 1, 2, 4, 8, … and the whole route,
//!   stride `len / 2`), each either *dense* (every reached state in the window
//!   is a stage) or *sparse* (only the essential ones plus the window's end,
//!   so the GA may take another path between them).
//! - **Brackets / rungs**: a bracket starts with every arm at rung 0 (a cheap
//!   GA). When a rung's queue is empty, the arms are ranked by what they did
//!   in it (improved > faster span > met all > partial) and the top half moves
//!   to the next rung (a bigger GA). When one arm is left or the top rung is
//!   done, a new bracket starts (arms rebuilt from the current best route).
//! - **Hot arms**: an arm that improved the route is re-run straight away at
//!   the same rung (its window just changed and often has more to give).
//! - **Extensions**: an arm that was faster over its window but didn't
//!   improve the route (the rest of the stream doesn't follow on from where
//!   it ends) spawns a longer window, twice as many reached states, so the GA
//!   must also meet the states after it; run straight away.
//! - **Seed beam**: the bench keeps one best route and doesn't hand back
//!   trial streams, so the alternative seeds are built from the best stream:
//!   with probability [`ALT_SEED`] the window's seed is the best span with a
//!   few frames cut at random spots (forced shortenings the GA's elitism would
//!   otherwise rarely keep).

use std::collections::VecDeque;

use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};

/// GA sizes per rung (population, generations).
const RUNGS: [(usize, usize); 4] = [(6, 3), (10, 5), (16, 8), (24, 10)];
/// Probability a trial uses a trimmed seed instead of the best span.
const ALT_SEED: f64 = 0.25;
/// Most frames cut from an alternative seed.
const ALT_MAX_CUT: usize = 8;

/// Most extension trials per bracket (a window that was faster locally but
/// not over the route, re-solved with more states after it).
const MAX_EXTEND: usize = 24;

#[derive(Clone)]
struct Arm {
    /// Route-state indices of the window's start and end.
    from: usize,
    to: usize,
    sparse: bool,
    /// Score in the current rung (None: not run yet in it).
    score: Option<f64>,
}

#[derive(Default)]
pub struct Beam {
    arms: Vec<Arm>,
    rung: usize,
    /// Arms still to run in the current rung.
    queue: VecDeque<usize>,
    /// Arms in the current rung.
    pool: Vec<usize>,
    /// Improved arms to re-run first.
    hot: VecDeque<usize>,
    brackets: u32,
    /// Extension arms run this bracket (see [`MAX_EXTEND`]).
    extended: usize,
    last_goal: u32,
}

impl Beam {
    fn new_bracket(&mut self, ctx: &Ctx, rng: &mut SmallRng) {
        let reached = ctx.reached();
        let n = reached.len() - 1; // windows end at reached[1..]
        let mut lens = Vec::new();
        let mut l = 1;
        while l < n {
            lens.push(l);
            l *= 2;
        }
        lens.push(n);
        self.arms.clear();
        for &len in &lens {
            let stride = (len / 2).max(1);
            let mut s = 0;
            loop {
                let e = (s + len).min(n);
                for sparse in [false, true] {
                    // Sparse only differs when the window has non-essential
                    // states before its end.
                    if sparse && !reached[s + 1..e].iter().any(|&i| !ctx.essential(i)) {
                        continue;
                    }
                    self.arms.push(Arm { from: reached[s], to: reached[e], sparse, score: None });
                }
                if e == n {
                    break;
                }
                s += stride;
            }
        }
        self.arms.dedup_by(|a, b| a.from == b.from && a.to == b.to && a.sparse == b.sparse);
        self.rung = 0;
        self.pool = (0..self.arms.len()).collect();
        self.pool.shuffle(rng);
        self.queue = self.pool.iter().copied().collect();
        self.hot.clear();
        self.extended = 0;
        self.brackets += 1;
    }

    /// Ends the current rung: the top half moves up, or a new bracket starts.
    fn next_rung(&mut self, ctx: &Ctx, rng: &mut SmallRng) {
        let mut pool = std::mem::take(&mut self.pool);
        pool.shuffle(rng);
        pool.sort_by(|&a, &b| {
            let s = |i: usize| self.arms[i].score.unwrap_or(f64::MIN);
            s(b).total_cmp(&s(a))
        });
        let keep = pool.len() / 2;
        if keep == 0 || self.rung + 1 >= RUNGS.len() {
            return self.new_bracket(ctx, rng);
        }
        pool.truncate(keep);
        for &i in &pool {
            self.arms[i].score = None;
        }
        self.rung += 1;
        self.queue = pool.iter().copied().collect();
        self.pool = pool;
    }

    /// The arm's trial against the current best route.
    fn trial(&self, arm: usize, ctx: &Ctx, rng: &mut SmallRng) -> Option<Trial> {
        let a = &self.arms[arm];
        let goal = ctx.goal();
        // Start at the latest reached state at or before the arm's start.
        let from = (0..=a.from.min(goal)).rev().find(|&i| ctx.best.arrival[i].is_some())?;
        if from >= goal {
            return None;
        }
        let to = a.to.clamp(from + 1, goal);
        let mut stages: Vec<usize> =
            (from + 1..=to).filter(|&i| ctx.best.arrival[i].is_some() && (!a.sparse || i == to || ctx.essential(i))).collect();
        if stages.last() != Some(&to) {
            // The end itself was skipped: end at the next reached state.
            let next = (to..=goal).find(|&i| ctx.best.arrival[i].is_some())?;
            stages.retain(|&i| i < next);
            stages.push(next);
        }
        let (population, generations) = RUNGS[self.rung];
        let mut seed = None;
        let mut tag = format!("r{} {}..{}{}", self.rung, from, stages.last().unwrap(), if a.sparse { " sparse" } else { "" });
        if rng.random_bool(ALT_SEED) {
            let (_, p0) = ctx.best.arrival[from].unwrap();
            let last = *stages.last().unwrap();
            let p1 = ctx.best.arrival[last].unwrap().1.max(p0);
            let mut s = ctx.best.stream[p0..p1].to_vec();
            if s.len() > 4 {
                let cuts = rng.random_range(1..=3);
                let mut total = 0;
                for _ in 0..cuts {
                    let n = rng.random_range(1..=ALT_MAX_CUT).min(s.len().saturating_sub(2));
                    if n == 0 {
                        break;
                    }
                    let at = rng.random_range(0..=s.len() - n);
                    s.drain(at..at + n);
                    total += n;
                }
                tag += &format!(" cut{total}");
                seed = Some(s);
            }
        }
        Some(Trial { from, stages, spec: GaSpec { population, generations, horizon: None }, seed, mutation: None, tag, ..Default::default() })
    }
}

impl Method for Beam {
    fn name(&self) -> &'static str {
        "beam"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        if self.last_goal == 0 {
            self.last_goal = ctx.best.goal_frame;
        }
        if self.arms.is_empty() {
            self.new_bracket(ctx, rng);
        }
        for _ in 0..10_000 {
            let arm = match self.hot.pop_front().or_else(|| self.queue.pop_front()) {
                Some(a) => a,
                None => {
                    self.next_rung(ctx, rng);
                    continue;
                }
            };
            if let Some(mut t) = self.trial(arm, ctx, rng) {
                t.tag = format!("{}/{arm}:{}", self.brackets, t.tag);
                return t;
            }
            // The arm no longer fits the route: it drops out of this rung.
            self.arms[arm].score = Some(f64::MIN);
        }
        super::full::Full.propose(ctx, rng)
    }

    fn feedback(&mut self, o: &Outcome, ctx: &Ctx) {
        let saved = self.last_goal as i64 - o.goal_frame as i64;
        self.last_goal = o.goal_frame;
        // Only arms of the current bracket (the tag is "bracket/arm:...").
        let id = o.trial.tag.split(':').next().and_then(|s| s.split_once('/'));
        let Some(arm) = id.filter(|(b, _)| b.parse() == Ok(self.brackets)).and_then(|(_, a)| a.parse::<usize>().ok()) else {
            return;
        };
        let Some(a) = self.arms.get_mut(arm) else { return };
        let stages = o.trial.stages.len().max(1) as f64;
        let score = if o.improved {
            1000.0 + saved as f64
        } else {
            match (o.span_frames, o.best_span_frames) {
                (Some(s), Some(b)) if s < b => 100.0 + (b - s) as f64,
                (Some(_), _) => 10.0,
                _ => o.met as f64 / stages,
            }
        };
        a.score = Some(a.score.map_or(score, |s: f64| s.max(score)));
        let (from, sparse) = (a.from, a.sparse);
        if o.improved {
            // Its window changed: run it again first, unless it's already queued.
            if !self.hot.contains(&arm) {
                self.hot.push_back(arm);
            }
        } else if let (Some(s), Some(b)) = (o.span_frames, o.best_span_frames)
            && s < b
        {
            // Faster over its window but not over the route: the rest of the
            // stream doesn't follow on from where it ends. Re-solve a longer
            // window that also has to meet the next states (twice as many
            // reached states, up to the goal).
            let reached = ctx.reached();
            let last = *o.trial.stages.last().unwrap();
            let i0 = reached.iter().position(|&i| i >= from).unwrap_or(0);
            let i1 = reached.iter().position(|&i| i >= last).unwrap_or(reached.len() - 1);
            let to = reached[(i1 + (i1 - i0).max(1)).min(reached.len() - 1)];
            if to > last {
                let ext = match self.arms.iter().position(|x| x.from == from && x.to == to && x.sparse == sparse) {
                    Some(e) => e,
                    None => {
                        self.arms.push(Arm { from, to, sparse, score: None });
                        self.arms.len() - 1
                    }
                };
                if !self.hot.contains(&ext) && self.extended < MAX_EXTEND {
                    self.extended += 1;
                    self.hot.push_back(ext);
                }
            }
        }
    }
}
