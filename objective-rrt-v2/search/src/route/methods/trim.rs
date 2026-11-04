//! `trim`: input-level trimming. Shortens the route by deleting frames of the
//! best stream rather than searching for new movement.
//!
//! Each trial is one candidate deletion `(at, len)` in the best stream,
//! checked cheaply: a single rollout (`population: 1, generations: 1`) of
//! the best stream with those frames removed, from the last reached state
//! before `at` to a reached state at least [`LOOKAHEAD`] stream frames past
//! the cut (the essential states in between are stages too). The bench
//! splices and verifies anything that arrives sooner.
//!
//! Candidates, best first:
//! - **hot**: around deletions that just worked (delta-debugging: the same
//!   cut again, twice as long, the frames before), and the halves of a
//!   failed "waste" run (binary search of the longest deletable piece);
//! - **waste runs**: maximal runs of wasted frames (neutral stick and no
//!   buttons, or the stick reversed against the last few frames), whole;
//! - **chunks** of 1, 2, 3, 5, 8, 12, 20 and 30 frames tiling the stream,
//!   one queue per size, each ordered by how wasted its frames are and how
//!   slow the route is between its waypoints there;
//! - a small **GA** (8 x 4) with a deleting mutation over a 80-250 frame
//!   span, near recent successes.
//!
//! Which of the waste runs, the chunk sizes and the GA goes next is a bandit:
//! in proportion to improvements per frame spent (counting the bench's verify
//! replay for trials that passed the local check), at least 3% each. Failed
//! cuts are remembered (remapped when the stream changes), so the queues
//! sweep the whole route.

use std::collections::{HashSet, VecDeque};

use obrrt_core::GamePad;
use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Trial};

/// Stream frames checked past the end of a cut.
const LOOKAHEAD: usize = 60;
/// Chunk sizes, one queue each.
const SIZES: [usize; 8] = [1, 2, 3, 5, 8, 12, 20, 30];
const MAX_CUT: usize = 30;

/// A candidate deletion: stream frames `at..at + len`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Cut {
    at: usize,
    len: usize,
    /// From a waste run: split in halves if it fails.
    waste: bool,
}

/// A proposed trial, matched to its outcome in order.
struct Inflight {
    cut: Option<Cut>,
    goal: u32,
    arm: Option<usize>,
}

/// Arms of the bandit choosing what to try next: the waste runs, each chunk
/// size, then the GA.
const ARMS: usize = SIZES.len() + 2;
const GA_ARM: usize = ARMS - 1;

#[derive(Default)]
pub struct Trim {
    /// The stream the queues were built for.
    stream: Vec<GamePad>,
    dirty: bool,
    hot: VecDeque<Cut>,
    waste_runs: VecDeque<Cut>,
    chunks: Vec<VecDeque<Cut>>,
    /// Per arm: improvements and frames spent (rollouts, plus an estimate
    /// of the bench's verify replay when a trial passed the local check).
    arm_wins: [f64; ARMS],
    arm_cost: [f64; ARMS],
    failed: HashSet<(usize, usize)>,
    inflight: VecDeque<Inflight>,
    goal: u32,
    /// Stream positions of recent successes (for the GA mode).
    wins: Vec<usize>,
    built: bool,
}

/// How wasted each stream frame looks: 1 for a neutral stick and no
/// buttons, 0.7 for the stick reversed against the last 8 frames.
fn waste(stream: &[GamePad]) -> Vec<f32> {
    let mag = |p: &GamePad| ((p.stick_x as f32).powi(2) + (p.stick_y as f32).powi(2)).sqrt();
    (0..stream.len())
        .map(|i| {
            let p = &stream[i];
            if mag(p) < 20.0 && p.button == 0 {
                return 1.0;
            }
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for q in &stream[i.saturating_sub(8)..i] {
                sx += q.stick_x as f32;
                sy += q.stick_y as f32;
            }
            let m = (sx * sx + sy * sy).sqrt();
            if mag(p) > 60.0 && m > 60.0 * 4.0 {
                let cos = (sx * p.stick_x as f32 + sy * p.stick_y as f32) / (m * mag(p));
                if cos < -0.35 {
                    return 0.7;
                }
            }
            0.0
        })
        .collect()
}

/// Delete 1-30 frames, often inside a run of neutral-stick frames.
fn delete_mutation(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    let len = g.len();
    if len < 2 {
        return;
    }
    if rng.random_bool(0.5) {
        let neutral: Vec<usize> = (0..len).filter(|&i| g[i].stick_x.unsigned_abs() < 15 && g[i].stick_y.unsigned_abs() < 15 && g[i].button == 0).collect();
        if !neutral.is_empty() {
            let a = neutral[rng.random_range(0..neutral.len())];
            let b = (a + 1..len).find(|&i| !(g[i].stick_x.unsigned_abs() < 15 && g[i].stick_y.unsigned_abs() < 15 && g[i].button == 0)).unwrap_or(len);
            let n = rng.random_range(1..=(b - a).min(MAX_CUT));
            g.drain(a..a + n);
            return;
        }
    }
    let a = rng.random_range(0..len);
    let n = if rng.random_bool(0.6) { rng.random_range(1..4) } else { rng.random_range(1..=MAX_CUT) };
    g.drain(a..(a + n).min(len));
}

impl Trim {
    /// The best stream changed: keep the failed cuts outside the changed
    /// region (shifted past it), drop the rest.
    fn remap(&mut self, new: &[GamePad]) {
        let old = &self.stream;
        let p = old.iter().zip(new).take_while(|(a, b)| a == b).count();
        let q = old.iter().rev().zip(new.iter().rev()).take_while(|(a, b)| a == b).count().min(old.len() - p).min(new.len() - p);
        let (old_end, delta) = (old.len() - q, new.len() as i64 - old.len() as i64);
        let shift = |(a, k): (usize, usize)| -> Option<(usize, usize)> {
            if a + k <= p {
                Some((a, k))
            } else if a >= old_end {
                Some(((a as i64 + delta) as usize, k))
            } else {
                None
            }
        };
        self.failed = self.failed.iter().filter_map(|&c| shift(c)).collect();
        self.hot = self.hot.iter().filter_map(|c| shift((c.at, c.len)).map(|(at, len)| Cut { at, len, waste: c.waste })).collect();
        self.wins = self.wins.iter().filter_map(|&w| shift((w, 0)).map(|c| c.0)).collect();
        self.stream = new.to_vec();
        self.dirty = true;
    }

    /// Rebuild the waste-run and chunk queues for the current stream.
    fn build(&mut self, ctx: &Ctx, rng: &mut SmallRng) {
        let stream = &ctx.best.stream;
        let end = ctx.best.arrival[ctx.goal()].map_or(stream.len(), |a| a.1).min(stream.len());
        let w = waste(stream);
        // Slowness per stream frame: 1 - speed/30 between consecutive
        // reached states (0.5 across zone changes).
        let mut slow = vec![0.5f32; stream.len()];
        let reached = ctx.reached();
        for pair in reached.windows(2) {
            let (i, j) = (pair[0], pair[1]);
            let (Some(a), Some(b)) = (ctx.best.arrival[i], ctx.best.arrival[j]) else { continue };
            let (Some(si), Some(sj)) = (&ctx.best.states[i], &ctx.best.states[j]) else { continue };
            let frames = (b.0 - a.0).max(1) as f32;
            let s = if si.zone() == sj.zone() { 1.0 - (obrrt_core::dist(si.pos, sj.pos) / frames / 30.0).min(1.0) } else { 0.5 };
            for x in &mut slow[a.1.min(stream.len())..b.1.min(stream.len())] {
                *x = s;
            }
        }
        let score = |at: usize, len: usize| -> f32 {
            let m = w[at..at + len].iter().sum::<f32>() / len as f32;
            let s = slow[at..at + len].iter().sum::<f32>() / len as f32;
            m + 0.5 * s
        };
        let ok = |c: &Cut, failed: &HashSet<(usize, usize)>| c.at + c.len <= end && !failed.contains(&(c.at, c.len));
        // Waste runs.
        let mut runs = Vec::new();
        let mut i = 0;
        while i < end {
            if w[i] >= 0.7 {
                let j = (i..end).find(|&j| w[j] < 0.7).unwrap_or(end);
                if j - i >= 2 {
                    runs.push(Cut { at: i, len: (j - i).min(MAX_CUT), waste: true });
                }
                i = j;
            } else {
                i += 1;
            }
        }
        runs.retain(|c| ok(c, &self.failed));
        runs.sort_by(|a, b| b.len.cmp(&a.len));
        self.waste_runs = runs.into();
        // Chunks per size.
        self.chunks = SIZES
            .iter()
            .map(|&size| {
                let step = (size / 2).max(1);
                let mut v: Vec<(f32, Cut)> = (0..end.saturating_sub(size))
                    .step_by(step)
                    .map(|at| Cut { at, len: size, waste: false })
                    .filter(|c| ok(c, &self.failed))
                    .map(|c| (score(c.at, c.len) + rng.random_range(0.0..0.05), c))
                    .collect();
                v.sort_by(|a, b| b.0.total_cmp(&a.0));
                v.into_iter().map(|x| x.1).collect()
            })
            .collect();
        self.dirty = false;
        self.built = true;
    }

    /// The next cut: a hot one, else from an arm picked in proportion to
    /// its improvements per frame spent (each arm gets at least 3%). None:
    /// run the GA.
    fn next_cut(&mut self, end: usize, rng: &mut SmallRng) -> Option<(Cut, Option<usize>)> {
        let valid = |c: &Cut, failed: &HashSet<(usize, usize)>| c.len > 0 && c.at + c.len <= end && !failed.contains(&(c.at, c.len));
        while let Some(c) = self.hot.pop_front() {
            if valid(&c, &self.failed) {
                return Some((c, None));
            }
        }
        loop {
            let open = |a: usize, t: &Trim| match a {
                0 => !t.waste_runs.is_empty(),
                GA_ARM => true,
                _ => !t.chunks[a - 1].is_empty(),
            };
            let rate: Vec<f64> = (0..ARMS).map(|a| if open(a, self) { (self.arm_wins[a] + 0.5) / (self.arm_cost[a] + 1e5) } else { 0.0 }).collect();
            let total: f64 = rate.iter().sum();
            let n_open = rate.iter().filter(|r| **r > 0.0).count() as f64;
            let weight: Vec<f64> = rate.iter().map(|r| if *r > 0.0 { 0.97 * r / total + 0.03 / n_open } else { 0.0 }).collect();
            let mut x = rng.random_range(0.0..weight.iter().sum::<f64>());
            let arm = (0..ARMS).find(|&a| {
                x -= weight[a];
                x < 0.0 && weight[a] > 0.0
            });
            let arm = arm.unwrap_or(GA_ARM);
            if arm == GA_ARM {
                return None;
            }
            let q = if arm == 0 { &mut self.waste_runs } else { &mut self.chunks[arm - 1] };
            while let Some(c) = q.pop_front() {
                if valid(&c, &self.failed) {
                    return Some((c, Some(arm)));
                }
            }
        }
    }

    /// A small GA with the deleting mutation over a span: near a recent
    /// success if any, else anywhere.
    fn ga_trial(&self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        let reached = ctx.reached();
        let pos_of = |k: usize| ctx.best.arrival[k].unwrap().1;
        let k0 = if !self.wins.is_empty() && rng.random_bool(0.7) {
            let w = self.wins[rng.random_range(0..self.wins.len())];
            reached.iter().rposition(|&k| pos_of(k) <= w).unwrap_or(0)
        } else {
            rng.random_range(0..reached.len() - 1)
        };
        let from = reached[k0];
        let span = rng.random_range(80..250);
        let to = reached[k0 + 1..].iter().copied().find(|&k| pos_of(k) >= pos_of(from) + span).unwrap_or(ctx.goal());
        Trial {
            from,
            stages: stages(ctx, from, to),
            spec: GaSpec { population: 8, generations: 4, horizon: None },
            seed: None,
            mutation: Some(delete_mutation),
            tag: "trim ga".into(), ..Default::default() }
    }
}

/// The essential states in `(from, to)`, then `to`.
fn stages(ctx: &Ctx, from: usize, to: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (from + 1..to).filter(|&k| ctx.best.arrival[k].is_some() && ctx.essential(k)).collect();
    v.push(to);
    v
}

impl Method for Trim {
    fn name(&self) -> &'static str {
        "trim"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        if !self.built {
            self.stream = ctx.best.stream.clone();
            self.goal = ctx.best.goal_frame;
        } else if self.stream != ctx.best.stream {
            self.remap(&ctx.best.stream.clone());
        }
        if !self.built || self.dirty {
            self.build(ctx, rng);
        }
        let goal_pos = ctx.best.arrival[ctx.goal()].map_or(ctx.best.stream.len(), |a| a.1).min(ctx.best.stream.len());
        let Some((cut, arm)) = self.next_cut(goal_pos, rng) else {
            self.inflight.push_back(Inflight { cut: None, goal: ctx.best.goal_frame, arm: Some(GA_ARM) });
            return self.ga_trial(ctx, rng);
        };
        let reached = ctx.reached();
        let pos_of = |k: usize| ctx.best.arrival[k].unwrap().1;
        let from = reached.iter().copied().filter(|&k| pos_of(k) <= cut.at).last().unwrap_or(0);
        let to = reached.iter().copied().find(|&k| k > from && pos_of(k) >= cut.at + cut.len + LOOKAHEAD).unwrap_or(ctx.goal());
        let p0 = pos_of(from);
        let p1 = ctx.best.arrival[to].map_or(ctx.best.stream.len(), |a| a.1).max(cut.at + cut.len);
        let mut seed = ctx.best.stream[p0..p1].to_vec();
        seed.drain(cut.at - p0..cut.at - p0 + cut.len);
        self.inflight.push_back(Inflight { cut: Some(cut), goal: ctx.best.goal_frame, arm });
        Trial {
            from,
            stages: stages(ctx, from, to),
            spec: GaSpec { population: 1, generations: 1, horizon: None },
            seed: Some(seed),
            mutation: None,
            tag: format!("trim del {}@{}{}", cut.len, cut.at, if cut.waste { " waste" } else { "" }), ..Default::default() }
    }

    fn feedback(&mut self, outcome: &crate::route::Outcome, ctx: &Ctx) {
        let Some(inf) = self.inflight.pop_front() else { return };
        // Proposed against a route that has since changed: its cut no
        // longer means the same frames.
        let stale = inf.goal != self.goal;
        if let Some(a) = inf.arm {
            let local = matches!((outcome.span_frames, outcome.best_span_frames), (Some(s), Some(b)) if s < b);
            self.arm_cost[a] += outcome.frames_used as f64 + if local { 2.0 * ctx.best.goal_frame as f64 } else { 0.0 };
            self.arm_wins[a] += outcome.improved as u8 as f64;
        }
        if outcome.improved {
            self.goal = outcome.goal_frame;
            self.remap(&ctx.best.stream.clone());
            if let (Some(c), false) = (inf.cut, stale) {
                self.wins.push(c.at);
                // Delta-debugging around the success: what now sits at the
                // cut, a longer cut, the frames before.
                let mut push = |at: usize, len: usize| self.hot.push_back(Cut { at, len: len.min(MAX_CUT), waste: c.waste });
                push(c.at, c.len);
                push(c.at, c.len * 2);
                if c.at >= c.len {
                    push(c.at - c.len, c.len);
                }
                push(c.at, 1);
                if c.at > 0 {
                    push(c.at - 1, 1);
                }
            } else {
                self.wins.push(ctx.best.arrival[outcome.trial.from].map_or(0, |a| a.1));
            }
            if self.wins.len() > 32 {
                self.wins.remove(0);
            }
            return;
        }
        if stale {
            return;
        }
        if let Some(c) = inf.cut {
            self.failed.insert((c.at, c.len));
            // Binary search inside a failed waste run.
            if c.waste && c.len > 1 {
                let h = c.len / 2;
                self.hot.push_back(Cut { at: c.at, len: h, waste: true });
                self.hot.push_back(Cut { at: c.at + h, len: c.len - h, waste: true });
            }
        }
    }
}
