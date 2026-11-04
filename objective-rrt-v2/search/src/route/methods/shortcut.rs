//! `shortcut`: random deletion of waypoints to find shortcuts. Each trial
//! keeps every essential waypoint in its span plus each other reached one
//! with probability `p`, as one chain, so the GA is free to cut corners
//! between the kept ones.
//!
//! Two small bandits pick, per trial:
//! - the span ([`Span`]): root to goal, a random reached essential waypoint
//!   to the goal, or a random reached essential waypoint to the next one;
//! - the keep-probability `p` from [`KEEP`].
//!
//! Each arm's score is its verified frames saved per frame emulated, with an
//! optimistic prior so every arm gets tried; 20% of picks are random. (Spans
//! that end before the goal are often faster but then fail verification:
//! the open-loop rest of the route doesn't fit the new arrival state, so
//! unverified span savings earn nothing.)
//! A custom mutation deletes a longer chunk (15-60 frames) than the
//! built-in delete, to try skipping a whole stretch.

use std::collections::VecDeque;

use obrrt_core::GamePad;
use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};

/// Keep-probabilities for non-essential waypoints.
const KEEP: [f64; 4] = [0.3, 0.5, 0.7, 0.85];
#[derive(Clone, Copy)]
enum Span {
    RootToGoal,
    EssentialToGoal,
    EssentialToNext,
}
const SPANS: [Span; 3] = [Span::RootToGoal, Span::EssentialToGoal, Span::EssentialToNext];
const EXPLORE: f64 = 0.2;
/// Prior: as if each arm had saved `PRIOR_SAVED` frames in `PRIOR_FRAMES`.
const PRIOR_SAVED: f64 = 4.0;
const PRIOR_FRAMES: f64 = 2.0e6;

#[derive(Clone, Copy, Default)]
struct Arm {
    saved: f64,
    frames: f64,
}

impl Arm {
    fn rate(&self) -> f64 {
        (self.saved + PRIOR_SAVED) / (self.frames + PRIOR_FRAMES)
    }
}

#[derive(Default)]
pub struct Shortcut {
    keep: [Arm; KEEP.len()],
    span: [Arm; SPANS.len()],
    /// Arms of the trials proposed and not yet heard back, in order.
    pending: VecDeque<(usize, usize)>,
    last_goal: Option<u32>,
}

fn pick(arms: &[Arm], rng: &mut SmallRng) -> usize {
    if rng.random_bool(EXPLORE) {
        return rng.random_range(0..arms.len());
    }
    (0..arms.len()).max_by(|&a, &b| arms[a].rate().total_cmp(&arms[b].rate())).unwrap()
}

/// Deletes a longer chunk than the built-in delete.
fn delete_chunk(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    if g.len() < 20 {
        return;
    }
    let a = rng.random_range(0..g.len());
    let b = (a + rng.random_range(15..60)).min(g.len());
    g.drain(a..b);
}

impl Method for Shortcut {
    fn name(&self) -> &'static str {
        "shortcut"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        if self.last_goal.is_none() {
            self.last_goal = Some(ctx.best.goal_frame);
        }
        let reached = ctx.reached();
        let goal = ctx.goal();
        let ki = pick(&self.keep, rng);
        let si = pick(&self.span, rng);
        let p = KEEP[ki];
        // Essential reached states (the root counts, the goal is last).
        let ess: Vec<usize> = reached.iter().copied().filter(|&i| ctx.essential(i)).collect();
        let k = rng.random_range(0..ess.len() - 1);
        let (from, to) = match SPANS[si] {
            Span::RootToGoal => (0, goal),
            Span::EssentialToGoal => (ess[k], goal),
            Span::EssentialToNext => (ess[k], ess[k + 1]),
        };
        let stages: Vec<usize> = reached
            .iter()
            .copied()
            .filter(|&i| i > from && i <= to && (i == to || ctx.essential(i) || rng.random_bool(p)))
            .collect();
        let spec = if from == 0 && to == goal {
            GaSpec::default()
        } else {
            // Short spans are cheap: a little more search each.
            GaSpec { population: 16, generations: 10, horizon: None }
        };
        self.pending.push_back((ki, si));
        Trial {
            from,
            stages,
            spec,
            seed: None,
            mutation: Some(delete_chunk),
            tag: format!("shortcut {from}->{to} p={p}"), ..Default::default() }
    }

    fn feedback(&mut self, o: &Outcome, _ctx: &Ctx) {
        let Some((ki, si)) = self.pending.pop_front() else { return };
        let before = self.last_goal.unwrap_or(o.goal_frame);
        let saved = before.saturating_sub(o.goal_frame) as f64;
        self.last_goal = Some(o.goal_frame);
        for arm in [&mut self.keep[ki], &mut self.span[si]] {
            arm.saved += saved;
            arm.frames += o.frames_used as f64;
        }
    }
}
