//! `direct`: large shortcuts. The old route's inputs are thrown away: each
//! trial re-solves a span from scratch with the steering controller heading
//! straight for each goal (plus noisy variants), and keeps only the goals the
//! route can't skip: essential waypoints (zone changes, stars/keys,
//! dialogs/menus, the goal) and waypoints with a real requirement (objects
//! hit or removed, holding something). Every detour in between (coins on the
//! way, wandering) is dropped. Seeding from the old inputs keeps a GA near
//! the old path; starting from steering lets it find the straight one.
//!
//! Spans: from a reached state to the next required goal (one hop) or the
//! next two or three (to line up what follows), weighted by wasted time:
//! frames the best route takes minus the straight-line distance at running
//! speed.

use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};

/// Running speed (units per frame), for the straight-line estimate.
const SPEED: f32 = 30.0;

#[derive(Default)]
pub struct Direct {
    /// Per start state: times it improved / was tried, to favour what works.
    tried: Vec<u32>,
    won: Vec<u32>,
}

impl Direct {
    /// Route states the route can't skip.
    fn required(ctx: &Ctx, i: usize) -> bool {
        let w = &ctx.waypoints[i - 1];
        w.essential || !w.health.is_empty() || !w.gone.is_empty() || w.holding
    }
}

impl Method for Direct {
    fn name(&self) -> &'static str {
        "direct"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        let n = ctx.goal();
        if self.tried.len() < n + 1 {
            self.tried.resize(n + 1, 0);
            self.won.resize(n + 1, 0);
        }
        let reached = ctx.reached();
        // Candidate starts: reached states before the goal, weighted by the
        // waste between them and the next required goal.
        let mut cands: Vec<(usize, usize, f64)> = Vec::new();
        for &from in &reached {
            if from >= n {
                continue;
            }
            // The next 1-3 required goals after `from`.
            let req: Vec<usize> = ((from + 1)..=n).filter(|&k| Self::required(ctx, k)).take(3).collect();
            let Some(&first) = req.first() else { continue };
            let hops = rng.random_range(1..=req.len());
            let last = req[hops - 1];
            let Some((f0, _)) = ctx.best.arrival[from] else { continue };
            let Some((f1, _)) = (last..ctx.best.arrival.len()).find_map(|k| ctx.best.arrival[k]) else { continue };
            // Straight-line frames through the required goals (same zone hops only).
            let mut straight = 0.0f32;
            let mut at = ctx.best.states[from].as_ref().map(|s| s.pos).unwrap_or(ctx.waypoints[first - 1].pos);
            for &k in &req[..hops] {
                let w = &ctx.waypoints[k - 1];
                straight += obrrt_core::dist(at, w.pos) / SPEED;
                at = w.pos;
            }
            let waste = (f1.saturating_sub(f0)) as f64 - straight as f64;
            if waste < 20.0 {
                continue;
            }
            let bonus = 1.0 + 2.0 * self.won[from] as f64 / (1.0 + self.tried[from] as f64);
            cands.push((from, last, waste * bonus));
        }
        let pick = if cands.is_empty() {
            None
        } else {
            let total: f64 = cands.iter().map(|c| c.2).sum();
            let mut x = rng.random_range(0.0..total);
            cands.iter().find(|c| {
                x -= c.2;
                x < 0.0
            })
        };
        let (from, last) = match pick {
            Some(&(f, l, _)) => (f, l),
            None => (0, n),
        };
        self.tried[from] += 1;
        let stages: Vec<usize> = ((from + 1)..=last).filter(|&k| Self::required(ctx, k) || k == last).collect();
        // Horizon: the best route's span (a straight route is shorter) plus slack.
        let span = match (ctx.best.arrival[from], (last..ctx.best.arrival.len()).find_map(|k| ctx.best.arrival[k])) {
            (Some(a), Some(b)) => b.0.saturating_sub(a.0),
            _ => 600,
        };
        Trial {
            from,
            stages,
            spec: GaSpec { population: 24, generations: 10, horizon: Some(span + 120) },
            // No old inputs: steering from the first frame (noise comes from
            // the GA's mutations of the steered genomes).
            seed: Some(Vec::new()),
            tag: format!("direct {from}->{last}"),
            ..Default::default()
        }
    }

    fn feedback(&mut self, o: &Outcome, _ctx: &Ctx) {
        if o.improved && o.trial.from < self.won.len() {
            self.won[o.trial.from] += 1;
        }
    }
}
