//! `essential`: only the essential waypoints (zone changes, stars/keys,
//! dialogs/menus, the goal) as stages, for maximal freedom between them.
//!
//! Three trial shapes, mixed:
//! - **segment** (most trials): from one essential waypoint to the next one.
//!   Cheap, and the GA has the whole segment to itself. Segments are picked
//!   by an adaptive weight: the controllable frames in the segment (more
//!   frames, more to gain) times a score that doubles on an improvement and
//!   decays on a miss. Segments with no controllable frames (dialog to
//!   dialog) are skipped: nothing there can change.
//! - **window**: two to four consecutive essential segments in one trial, so
//!   a change that costs a little in one segment and gains more in the next
//!   can be found.
//! - **suffix**: from any reached route state (essential or not) to the
//!   next essential waypoint. The GA's mutations land only in the rest of
//!   the segment, so a long segment (a 2,000-frame one in Chain Chomp's
//!   route) is worked on piece by piece, and late pieces are cheap.
//! - **chain**: every essential waypoint from the root, like `full` with
//!   fewer stages.
//!
//! Without intermediate waypoints the steering controller has less guidance
//! once the stream runs out, so the seed matters more: a custom mutation
//! deletes bigger spans (10–60 frames), which the built-in ones rarely do.

use std::collections::HashMap;

use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};
use obrrt_core::GamePad;

/// Probabilities of the trial shapes (the rest are segments).
const P_CHAIN: f64 = 0.05;
const P_WINDOW: f64 = 0.20;
const P_SUFFIX: f64 = 0.40;
/// Score bounds and updates for a segment.
const SCORE_MIN: f64 = 0.15;
const SCORE_MAX: f64 = 8.0;
const SCORE_HIT: f64 = 2.0;
const SCORE_MISS: f64 = 0.9;

#[derive(Default)]
pub struct Essential {
    /// Per segment start (an essential route state): its score.
    score: HashMap<usize, f64>,
}

/// Deletes a bigger span (10–60 frames): a shortcut the steering-free
/// stretches between essential waypoints may allow.
fn big_delete(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    if g.len() < 12 {
        return;
    }
    let a = rng.random_range(0..g.len() - 10);
    let b = (a + rng.random_range(10..60)).min(g.len());
    g.drain(a..b);
}

impl Essential {
    /// The essential route states the best route reached, root first, and
    /// each one's stream position.
    fn chain(ctx: &Ctx) -> Vec<(usize, usize)> {
        ctx.reached()
            .into_iter()
            .filter(|&i| ctx.essential(i))
            .map(|i| (i, ctx.best.arrival[i].unwrap().1))
            .collect()
    }

    fn score(&self, from: usize) -> f64 {
        self.score.get(&from).copied().unwrap_or(1.0)
    }
}

impl Method for Essential {
    fn name(&self) -> &'static str {
        "essential"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        let chain = Self::chain(ctx);
        // Segments k -> k+1 with controllable frames between them.
        let segs: Vec<usize> = (0..chain.len() - 1).filter(|&k| chain[k + 1].1 > chain[k].1).collect();
        let roll: f64 = rng.random();
        if segs.is_empty() || roll < P_CHAIN {
            return Trial {
                from: 0,
                stages: chain.iter().map(|c| c.0).filter(|&i| i > 0).collect(),
                spec: GaSpec { population: 16, generations: 8, horizon: None },
                seed: None,
                mutation: Some(big_delete),
                tag: "chain".into(), ..Default::default() };
        }
        let weights: Vec<f64> = segs
            .iter()
            .map(|&k| (chain[k + 1].1 - chain[k].1) as f64 * self.score(chain[k].0))
            .collect();
        let total: f64 = weights.iter().sum();
        let mut x = rng.random::<f64>() * total;
        let mut k = segs[segs.len() - 1];
        for (s, w) in segs.iter().zip(&weights) {
            if x < *w {
                k = *s;
                break;
            }
            x -= w;
        }
        if roll < P_CHAIN + P_SUFFIX {
            // Any reached state strictly inside segment k (or its start).
            let (a, b) = (chain[k].0, chain[k + 1].0);
            let inside: Vec<usize> = ctx
                .reached()
                .into_iter()
                .filter(|&i| i >= a && i < b && ctx.best.arrival[i].unwrap().1 < chain[k + 1].1)
                .collect();
            let from = inside[rng.random_range(0..inside.len())];
            return Trial {
                from,
                stages: vec![b],
                spec: GaSpec { population: 16, generations: 12, horizon: None },
                seed: None,
                mutation: Some(big_delete),
                tag: format!("suffix {from}->{b}"), ..Default::default() };
        }
        let from = chain[k].0;
        if roll < P_CHAIN + P_SUFFIX + P_WINDOW && k + 2 < chain.len() {
            let n = rng.random_range(2..=4).min(chain.len() - 1 - k);
            return Trial {
                from,
                stages: chain[k + 1..=k + n].iter().map(|c| c.0).collect(),
                spec: GaSpec { population: 16, generations: 10, horizon: None },
                seed: None,
                mutation: Some(big_delete),
                tag: format!("window {from}+{n}"), ..Default::default() };
        }
        // Single segment: cheap, so a longer GA.
        Trial {
            from,
            stages: vec![chain[k + 1].0],
            spec: GaSpec { population: 16, generations: 12, horizon: None },
            seed: None,
            mutation: Some(big_delete),
            tag: format!("segment {from}->{}", chain[k + 1].0), ..Default::default() }
    }

    fn feedback(&mut self, o: &Outcome, _ctx: &Ctx) {
        if o.trial.tag == "chain" || o.trial.tag.starts_with("suffix") {
            return;
        }
        let s = self.score.entry(o.trial.from).or_insert(1.0);
        let faster = matches!((o.span_frames, o.best_span_frames), (Some(a), Some(b)) if a < b);
        *s = if o.improved {
            (*s * SCORE_HIT).min(SCORE_MAX)
        } else if faster {
            // Faster over the span but the full replay disagreed: keep it.
            *s
        } else {
            (*s * SCORE_MISS).max(SCORE_MIN)
        };
    }
}
