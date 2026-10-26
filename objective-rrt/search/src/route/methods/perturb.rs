//! `perturb`: targeted perturbation of the best route where it is slow.
//!
//! Each span between consecutive reached route states gets a slack: its
//! controllable frames minus the frames a straight line between the two
//! places would take at [`SPEED`]. Trials re-solve a short window (1–5 spans)
//! around a span sampled ∝ slack × a learned weight (stages: the window, then
//! the rest of the route through the goal, so the ranking is goal time), with
//! a custom mutation that makes small,
//! correlated changes to the stick (a 1/f-style drift of angle and magnitude
//! over a stretch, or a nudge to full tilt). Feedback raises a span's weight
//! when trials there got faster and decays it when they didn't.

use std::collections::{HashMap, VecDeque};

use obrrt_core::{GamePad, dist};
use rand::Rng;
use rand::rngs::SmallRng;

use crate::route::{Ctx, GaSpec, Method, Outcome, Trial};

/// Reference speed (units per frame) for a span's ideal time: about Mario's
/// top running speed.
const SPEED: f32 = 32.0;
/// Most best-route frames a trial window may cover (more spans are dropped).
const MAX_WINDOW_FRAMES: usize = 700;
const POPULATION: usize = 12;
const GENERATIONS: usize = 8;

#[derive(Default)]
pub struct Perturb {
    /// Learned weight per span, keyed by the route state the span ends at.
    weight: HashMap<usize, f64>,
    /// Spans in each proposed trial's window, in proposal order (feedback
    /// comes in the same order).
    windows: VecDeque<usize>,
}

impl Perturb {
    fn weight(&self, end: usize) -> f64 {
        self.weight.get(&end).copied().unwrap_or(1.0)
    }
}

/// Place and zone of route state `i` on the best route.
fn place(ctx: &Ctx, i: usize) -> ([f32; 3], (u16, u8), bool) {
    if i == 0 {
        let s = ctx.best.states[0].as_ref().unwrap();
        (s.pos, s.zone(), false)
    } else {
        let w = &ctx.waypoints[i - 1];
        (w.pos, w.zone, w.modal.is_some())
    }
}

/// Slack per span (reached[k-1] -> reached[k]), index k-1.
fn slacks(ctx: &Ctx, reached: &[usize]) -> Vec<f64> {
    reached
        .windows(2)
        .map(|p| {
            let (a, b) = (p[0], p[1]);
            // Controllable frames only (stream positions): dialogs, cutscenes
            // and warps inside a span are not slack.
            let frames = ctx.best.arrival[b].unwrap().1.saturating_sub(ctx.best.arrival[a].unwrap().1) as f64;
            let (pa, za, ma) = place(ctx, a);
            let (pb, zb, mb) = place(ctx, b);
            if za != zb || ma || mb {
                // Warps and dialogs: mostly fixed time.
                return 0.0;
            }
            (frames - (dist(pa, pb) / SPEED) as f64).max(0.0)
        })
        .collect()
}

impl Method for Perturb {
    fn name(&self) -> &'static str {
        "perturb"
    }

    fn propose(&mut self, ctx: &Ctx, rng: &mut SmallRng) -> Trial {
        let reached = ctx.reached();
        let slack = slacks(ctx, &reached);
        let scores: Vec<f64> = slack.iter().enumerate().map(|(k, s)| (s + 2.0) * self.weight(reached[k + 1])).collect();
        let total: f64 = scores.iter().sum();
        let mut x = rng.random_range(0.0..total);
        let mut k = scores.len() - 1;
        for (j, s) in scores.iter().enumerate() {
            if x < *s {
                k = j;
                break;
            }
            x -= s;
        }
        // Span k runs reached[k] -> reached[k+1]; widen by up to 2 each way.
        let mut lo = k.saturating_sub(rng.random_range(0..=2));
        let mut hi = (k + 1 + rng.random_range(0..=2)).min(reached.len() - 1);
        let frames = |lo: usize, hi: usize| ctx.best.arrival[reached[hi]].unwrap().0 - ctx.best.arrival[reached[lo]].unwrap().0;
        while hi - lo > 1 && frames(lo, hi) > MAX_WINDOW_FRAMES {
            if hi - (k + 1) >= k - lo { hi -= 1 } else { lo += 1 }
        }
        self.windows.push_back(hi - lo);
        Trial {
            from: reached[lo],
            // The window's stages plus the rest of the route: a splice that
            // only meets the window's last waypoint arrives in a slightly
            // different state, and the best route's remaining inputs then
            // miss the goal (seen in every smoke trial), so the GA re-solves
            // through the goal and ranks by goal time.
            stages: reached[lo + 1..].to_vec(),
            spec: GaSpec { population: POPULATION, generations: GENERATIONS, horizon: None },
            seed: None,
            mutation: Some(stick_tweak),
            tag: format!("perturb span {}->{} slack {:.0} window {}..{}", reached[k], reached[k + 1], slack[k], reached[lo], reached[hi]), ..Default::default() }
    }

    fn feedback(&mut self, o: &Outcome, _ctx: &Ctx) {
        let faster = matches!((o.span_frames, o.best_span_frames), (Some(s), Some(b)) if s < b);
        let factor = if o.improved {
            2.5
        } else if faster {
            1.3
        } else {
            0.8
        };
        // Credit every span the window covered (not the tail).
        let window = self.windows.pop_front().unwrap_or(1);
        for &end in o.trial.stages.iter().take(window) {
            let w = self.weight.entry(end).or_insert(1.0);
            *w = (*w * factor).clamp(0.05, 20.0);
        }
    }
}

/// Smooth correlated noise: three AR(1) processes (time constants 4, 16, 64
/// frames), roughly unit variance.
struct Drift {
    x: [f32; 3],
}

impl Drift {
    const RHO: [f32; 3] = [0.75, 0.9375, 0.984375];

    fn new(rng: &mut SmallRng) -> Drift {
        Drift { x: [gauss(rng), gauss(rng), gauss(rng)] }
    }

    fn next(&mut self, rng: &mut SmallRng) -> f32 {
        let mut sum = 0.0;
        for (x, r) in self.x.iter_mut().zip(Self::RHO) {
            *x = r * *x + (1.0 - r * r).sqrt() * gauss(rng);
            sum += *x;
        }
        sum / 3f32.sqrt()
    }
}

fn gauss(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.random_range(1e-7..1.0);
    let u2: f32 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn polar(p: &GamePad) -> (f32, f32) {
    let (x, y) = (p.stick_x as f32, p.stick_y as f32);
    (y.atan2(x), (x * x + y * y).sqrt())
}

fn set_polar(p: &mut GamePad, ang: f32, mag: f32) {
    let mag = mag.clamp(0.0, 127.0);
    p.stick_x = (mag * ang.cos()).round().clamp(-128.0, 127.0) as i8;
    p.stick_y = (mag * ang.sin()).round().clamp(-128.0, 127.0) as i8;
}

/// The custom mutation: over a stretch of 6–150 frames, either drift the
/// stick angle and magnitude by smooth correlated noise (tapered at both
/// ends), push the stick to full tilt, or apply a small constant turn. Frames
/// with the stick at rest are left alone.
fn stick_tweak(g: &mut Vec<GamePad>, rng: &mut SmallRng) {
    let len = g.len();
    if len < 2 {
        return;
    }
    // Biased toward the genome's start, where the trial's slow window is.
    let a = ((len as f32 * rng.random::<f32>().powi(2)) as usize).min(len - 1);
    let b = (a + rng.random_range(6..150)).min(len);
    let n = (b - a) as f32;
    match rng.random_range(0..10) {
        0..6 => {
            let amp = rng.random_range(2.0f32..20.0).to_radians();
            let mag_amp = if rng.random_bool(0.5) { rng.random_range(0.0f32..25.0) } else { 0.0 };
            let (mut da, mut dm) = (Drift::new(rng), Drift::new(rng));
            for (i, p) in g[a..b].iter_mut().enumerate() {
                let taper = (std::f32::consts::PI * (i as f32 + 0.5) / n).sin();
                let (ang, mag) = polar(p);
                let (ta, tm) = (da.next(rng), dm.next(rng));
                if mag < 8.0 {
                    continue;
                }
                set_polar(p, ang + amp * taper * ta, mag + mag_amp * taper * tm);
            }
        }
        6..8 => {
            for p in &mut g[a..b] {
                let (ang, mag) = polar(p);
                if mag >= 8.0 {
                    set_polar(p, ang, 127.0);
                }
            }
        }
        _ => {
            let turn = rng.random_range(-6.0f32..6.0).to_radians();
            for p in &mut g[a..b] {
                let (ang, mag) = polar(p);
                if mag >= 8.0 {
                    set_polar(p, ang + turn, mag);
                }
            }
        }
    }
}
