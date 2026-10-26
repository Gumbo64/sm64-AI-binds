//! `window_delete`: local shortcut-finding. Each trial takes a window of a few
//! (~6) reached waypoints after a reached `from`, and randomly drops some of
//! the non-essential ones inside it (never the window's last, so the splice
//! point stays defined): the GA can then cut corners between the survivors.
//!
//! Adapted from feedback:
//! - **window size**: grows when trials meet every stage but aren't faster
//!   (room to find a shortcut is too small), shrinks when they miss stages
//!   (too hard), and is pulled towards the size of windows that improved.
//! - **deletion probability**: pulled towards the deleted fraction of
//!   windows that were faster, decays a little when trials with deletions
//!   miss stages.
//! - **placement**: windows start at reached states weighted by how slow the
//!   best route is there (frames per waypoint over the next window), boosted
//!   where recent trials improved or came close, damped where many trials in
//!   a row failed to help.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};

const MIN_WINDOW: f64 = 3.0;
const MAX_WINDOW: f64 = 14.0;
const MIN_DELETE: f64 = 0.05;
const MAX_DELETE: f64 = 0.8;

pub struct WindowDelete {
    /// Mean window size (waypoints after `from`, kept or deleted).
    window: f64,
    /// Probability of deleting each non-essential, non-last waypoint.
    delete: f64,
    /// Per route state as `from`: recent success (decays).
    heat: Vec<f64>,
    /// Per route state as `from`: trials since it last helped.
    stale: Vec<u32>,
}

impl Default for WindowDelete {
    fn default() -> Self {
        WindowDelete { window: 6.0, delete: 0.3, heat: Vec::new(), stale: Vec::new() }
    }
}

impl WindowDelete {
    fn ensure(&mut self, n: usize) {
        if self.heat.len() < n {
            self.heat.resize(n, 0.0);
            self.stale.resize(n, 0);
        }
    }
}

/// The best route's stream position at route state `i` (reached): counts
/// only controllable frames, so cutscenes and dialogs don't look "slow".
fn stream_pos(ctx: &Ctx, i: usize) -> usize {
    ctx.best.arrival[i].map_or(0, |a| a.1)
}

/// Waypoints with object requirements (coins, enemies, boss hits) list only
/// what changed at them, so deleting one drops that requirement for good:
/// rarely worth it (the splice tends to fail later).
fn has_objects(ctx: &Ctx, i: usize) -> bool {
    let w = &ctx.waypoints[i - 1];
    !w.health.is_empty() || !w.gone.is_empty()
}

impl Method for WindowDelete {
    fn name(&self) -> &'static str {
        "window_delete"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        let reached = ctx.reached();
        self.ensure(reached.last().map_or(1, |&l| l + 1));
        // Window size for this trial: jitter around the mean.
        let w = (self.window * rng.random_range(0.6..1.5)).round().clamp(2.0, MAX_WINDOW) as usize;
        // Candidate starts: every reached state with at least one reached after it.
        let starts = reached.len().saturating_sub(1).max(1);
        let weights: Vec<f64> = (0..starts)
            .map(|k| {
                let end = (k + w).min(reached.len() - 1);
                let span = stream_pos(ctx, reached[end]).saturating_sub(stream_pos(ctx, reached[k])) as f64;
                let per_wp = span / (end - k).max(1) as f64;
                let i = reached[k];
                (per_wp + 1.0) * (1.0 + self.heat[i]) / (1.0 + 0.3 * self.stale[i] as f64)
            })
            .collect();
        let total: f64 = weights.iter().sum();
        let mut pick = rng.random_range(0.0..total.max(1e-9));
        let mut k = starts - 1;
        for (j, wt) in weights.iter().enumerate() {
            if pick < *wt {
                k = j;
                break;
            }
            pick -= wt;
        }
        let from = reached[k];
        let window = &reached[k + 1..(k + 1 + w).min(reached.len())];
        // Sometimes explore a much higher deletion rate.
        let p = if rng.random_bool(0.15) { (self.delete * 2.0).min(0.9) } else { self.delete };
        let last = window.len() - 1;
        let stages: Vec<usize> = window
            .iter()
            .enumerate()
            .filter(|&(j, &i)| {
                let q = if has_objects(ctx, i) { p * 0.2 } else { p };
                j == last || ctx.essential(i) || !rng.random_bool(q)
            })
            .map(|(_, &i)| i)
            .collect();
        let deleted = window.len() - stages.len();
        Trial {
            from,
            stages,
            spec: GaSpec::default(),
            seed: None,
            mutation: None,
            tag: format!("window_delete w{} del{} p{:.2}", window.len(), deleted, p), ..Default::default() }
    }

    fn feedback(&mut self, o: &Outcome, ctx: &Ctx) {
        let t = &o.trial;
        let last = *t.stages.last().unwrap();
        self.ensure(ctx.best.arrival.len().max(last + 1));
        // Recover the window: reached states in (from, last] — read against
        // the route the trial was proposed on; if the route changed, close enough.
        let size = ctx.reached().iter().filter(|&&i| i > t.from && i <= last).count().max(t.stages.len());
        let deletable = (t.from + 1..last).filter(|&i| !ctx.essential(i)).count().max(1);
        let deleted = size - t.stages.len();
        let frac = deleted as f64 / deletable as f64;
        for h in &mut self.heat {
            *h *= 0.995;
        }
        let all = o.met == t.stages.len();
        let faster = all && match (o.span_frames, o.best_span_frames) {
            (Some(s), Some(b)) => s < b,
            (Some(_), None) => true,
            _ => false,
        };
        if o.improved || faster {
            let gain = match (o.span_frames, o.best_span_frames) {
                (Some(s), Some(b)) => b.saturating_sub(s) as f64,
                _ => 1.0,
            };
            if o.improved {
                self.heat[t.from] += 2.0 + gain.min(10.0);
                self.stale[t.from] = 0;
            } else {
                // Faster over the span but the splice didn't verify faster:
                // promising, but don't let it pin windows here forever.
                self.heat[t.from] += 0.3;
                self.stale[t.from] += 1;
            }
            let rate = if o.improved { 0.3 } else { 0.1 };
            self.window += rate * (size as f64 - self.window);
            self.delete += rate * (frac - self.delete);
        } else {
            self.stale[t.from] += 1;
            if all {
                // Met everything but no faster: look wider, cut more.
                self.window += 0.05;
                self.delete += 0.005;
            } else {
                self.window -= 0.08;
                if deleted > 0 {
                    self.delete *= 0.99;
                }
            }
        }
        self.window = self.window.clamp(MIN_WINDOW, MAX_WINDOW);
        self.delete = self.delete.clamp(MIN_DELETE, MAX_DELETE);
    }
}
