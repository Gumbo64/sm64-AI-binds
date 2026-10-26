//! `window`: sliding windows of a few waypoints, each re-solved from the best
//! route's state at the window start; the bench splices improvements in.
//!
//! - A window is `from` (a reached route state) and the next `size` reached
//!   waypoints after it (default 4).
//! - Windows are chosen three ways:
//!   - a sweep, start to goal and then again;
//!   - sampling weighted by how slow the best route is there (controllable
//!     frames per waypoint compared with the route's average), favouring
//!     starts that improved recently and avoiding starts that keep failing;
//!   - the neighbourhood of a recent improvement (the route changed there, so
//!     the windows next to it are worth another look).
//! - Per start, the window grows by 2 (up to `MAX_SIZE`) after `GROW_AFTER`
//!   trials without an improvement, and shrinks back to the base size when
//!   one improves.
//! - A window ending at a non-essential waypoint can only shorten the route
//!   if the rest still works from where Mario ends up. So a window is
//!   sometimes stretched to end at the next essential waypoint (if one is
//!   within `SNAP_REACH`), and always is (however far) at a start where a
//!   faster window was rejected by the bench's verification; such starts are
//!   also sampled more.
//! - Horizon: the best route's frames over the window × 1.3 + 60, so trials
//!   that fail don't play the default +300 frames.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};

/// Waypoints per window to start with, and at most.
const BASE_SIZE: usize = 4;
const MAX_SIZE: usize = 16;
/// Trials without an improvement at a start before its window grows.
const GROW_AFTER: u32 = 4;
/// How far past the window's end an essential waypoint may be to stretch to it.
const SNAP_REACH: usize = 3;
/// Trials within which an improvement counts as recent.
const RECENT: u64 = 60;
/// Mode probabilities: sweep, near a recent improvement; the rest weighted.
const P_SWEEP: f64 = 0.35;
const P_NEAR: f64 = 0.15;
const P_SNAP: f64 = 0.4;

#[derive(Clone, Default)]
struct Stat {
    fails: u32,
    size: usize,
    last_improved: Option<u64>,
    /// A faster window from here didn't make the whole route faster.
    rejected: u32,
}

#[derive(Default)]
pub struct Window {
    stats: Vec<Stat>,
    /// Sweep position in the reached list.
    sweep: usize,
    /// Route states of recent improvements (newest last).
    recent: Vec<(usize, u64)>,
}

impl Window {
    fn ensure(&mut self, n: usize) {
        if self.stats.len() < n {
            self.stats.resize(n, Stat { size: BASE_SIZE, ..Stat::default() });
        }
    }

    /// Controllable frames (stream positions) from reached[p] to reached[q].
    fn stream_span(ctx: &Ctx, reached: &[usize], p: usize, q: usize) -> f64 {
        let a = ctx.best.arrival[reached[p]].unwrap().1;
        let b = ctx.best.arrival[reached[q]].unwrap().1;
        b.saturating_sub(a) as f64
    }

    fn window_end(&self, ctx: &Ctx, reached: &[usize], p: usize, rng: &mut SmallRng) -> usize {
        let last = reached.len() - 1;
        let st = &self.stats[reached[p]];
        let mut q = (p + st.size).min(last);
        // After a rejection, stretch to the next essential waypoint however far.
        let reach = if st.rejected > 0 { last } else { (q + SNAP_REACH).min(last) };
        if !ctx.essential(reached[q]) && (st.rejected > 0 || rng.random_bool(P_SNAP)) {
            if let Some(e) = (q + 1..=reach).find(|&e| ctx.essential(reached[e])) {
                q = e;
            }
        }
        q
    }
}

impl Method for Window {
    fn name(&self) -> &'static str {
        "window"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        self.ensure(ctx.goal() + 1);
        let reached = ctx.reached();
        // Starts: every reached state but the goal (the goal is always reached).
        let starts = reached.len() - 1;
        let mode;
        let p = if rng.random_bool(P_SWEEP) || starts == 1 {
            mode = "sweep";
            let p = self.sweep % starts;
            self.sweep = p + 1;
            p
        } else if !self.recent.is_empty() && rng.random_bool(P_NEAR / (1.0 - P_SWEEP)) {
            mode = "near";
            let (state, _) = self.recent[rng.random_range(0..self.recent.len())];
            // The reached position at or after the improved start, then a step back or forward.
            let at = reached.iter().position(|&r| r >= state).unwrap_or(starts - 1).min(starts - 1);
            (at as i64 + rng.random_range(-2i64..=2)).clamp(0, starts as i64 - 1) as usize
        } else {
            mode = "weighted";
            let avg = Self::stream_span(ctx, &reached, 0, starts) / starts as f64;
            let weights: Vec<f64> = (0..starts)
                .map(|p| {
                    let st = &self.stats[reached[p]];
                    let q = (p + st.size).min(starts);
                    let per = Self::stream_span(ctx, &reached, p, q) / (q - p) as f64;
                    let slow = (per / avg.max(1.0)).clamp(0.25, 4.0);
                    let recent = st.last_improved.is_some_and(|t| ctx.trials.saturating_sub(t) < RECENT);
                    let promising = if recent { 3.0 } else if st.rejected > 0 { 2.0 } else { 1.0 };
                    slow * slow * promising / (1.0 + 0.15 * st.fails as f64)
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let mut x = rng.random_range(0.0..total);
            weights.iter().position(|w| {
                x -= w;
                x < 0.0
            }).unwrap_or(starts - 1)
        };
        let q = self.window_end(ctx, &reached, p, rng);
        let from = reached[p];
        let frames = ctx.best.arrival[reached[q]].unwrap().0 - ctx.best.arrival[from].unwrap().0;
        Trial {
            from,
            stages: reached[p + 1..=q].to_vec(),
            spec: GaSpec { population: 16, generations: 8, horizon: Some((frames as f64 * 1.3) as usize + 60) },
            seed: None,
            mutation: None,
            tag: format!("window {mode} from {from} n{}", q - p), ..Default::default() }
    }

    fn feedback(&mut self, o: &Outcome, ctx: &Ctx) {
        self.ensure(ctx.goal() + 1);
        let from = o.trial.from;
        let st = &mut self.stats[from];
        if o.improved {
            st.fails = 0;
            st.rejected = 0;
            st.size = BASE_SIZE;
            st.last_improved = Some(ctx.trials);
            self.recent.push((from, ctx.trials));
            self.recent.retain(|&(_, t)| ctx.trials.saturating_sub(t) < RECENT);
            return;
        }
        st.fails += 1;
        let faster = matches!((o.span_frames, o.best_span_frames), (Some(s), Some(b)) if s < b);
        if faster {
            // Faster over the window but the rest of the route broke or lost
            // the time: end the window at an essential waypoint, or reach further.
            st.rejected += 1;
        }
        if st.fails % GROW_AFTER == 0 {
            st.size = (st.size + 2).min(MAX_SIZE);
        }
        self.recent.retain(|&(_, t)| ctx.trials.saturating_sub(t) < RECENT);
    }
}
