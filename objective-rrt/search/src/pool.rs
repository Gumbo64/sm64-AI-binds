//! The objective pool: objectives keyed by identity, each with its weight
//! from creation and its failure decay.

use std::collections::HashMap;
use std::sync::Arc;

use obrrt_core::{Objective, St};
use rand::Rng;

pub struct Entry {
    pub obj: Arc<dyn Objective>,
    /// `obj_weight(b)` when it entered the pool.
    pub base_weight: f64,
    /// ×`decay` per failed attempt; back to 1 on success or when a new state
    /// is added where it is attemptable.
    pub decay: f64,
    pub attempts: u32,
    pub successes: u32,
    /// Rounds since any state could attempt it.
    pub unattemptable: u32,
    /// Best (lowest) base_heuristic of any start it was attempted from: its
    /// decay only resets for a start meaningfully better than this.
    pub best_start: f64,
    /// Arrival ranking: best (soonest) estimated arrival of any start.
    pub best_arrival: f64,
    /// Start wear (`Config::start_wear`): per start node, attempts of this
    /// objective from it that didn't improve on it (no success, no state
    /// added with a better base_heuristic than the start's).
    pub wear: HashMap<u32, u32>,
    /// Route recoveries (`Config::decay_route`) left: decay recoveries from
    /// progress along a start other than the best one.
    pub recoveries: u32,
    /// Entry bandit (`Config::entry_bandit`): discounted improvements and
    /// attempts. Each round its `decay` is redrawn from
    /// Beta(improvements + 1, failures + 1), scaled so a new entry averages 1.
    pub bandit_k: f64,
    pub bandit_n: f64,
}

impl Entry {
    /// Entry bandit: one attempt, improved (a new success, or a kept state
    /// better than its start) or not. Older evidence fades by `gamma`.
    pub fn observe(&mut self, improved: bool, gamma: f64) {
        self.bandit_k = self.bandit_k * gamma + improved as u32 as f64;
        self.bandit_n = self.bandit_n * gamma + 1.0;
    }
}

/// How many times one entry's decay may recover through progress from a
/// start other than its best (`Config::decay_route`).
pub const ROUTE_RECOVERIES: u32 = 8;

#[derive(Default)]
pub struct Pool {
    pub entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

impl Pool {
    /// Adds `obj` unless an objective with its key is already in the pool.
    pub fn insert(&mut self, obj: Arc<dyn Objective>, b: &St) -> bool {
        let key = obj.key();
        if self.index.contains_key(&key) {
            return false;
        }
        let base_weight = obj.obj_weight(b).max(0.0) * kind_weight(obj.kind());
        self.index.insert(key, self.entries.len());
        self.entries.push(Entry { obj, base_weight, decay: 1.0, attempts: 0, successes: 0, unattemptable: 0, best_start: f64::INFINITY, best_arrival: f64::INFINITY, wear: HashMap::new(), recoveries: ROUTE_RECOVERIES, bandit_k: 0.0, bandit_n: 0.0 });
        true
    }

    /// Entry bandit: redraws every entry's `decay` as a Thompson sample of
    /// its improvement rate, ×2 (the prior mean is 1/2, so a new entry
    /// averages the old starting weight of 1).
    pub fn redraw(&mut self, rng: &mut impl Rng) {
        for e in &mut self.entries {
            let a = e.bandit_k + 1.0;
            let b = (e.bandit_n - e.bandit_k).max(0.0) + 1.0;
            e.decay = 2.0 * beta(rng, a, b);
        }
    }

    /// The entry with this key, if it's in the pool.
    pub fn find(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    /// Removes objectives that no state could attempt for `rounds` rounds.
    pub fn expire(&mut self, rounds: u32) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| e.unattemptable < rounds);
        self.index = self.entries.iter().enumerate().map(|(i, e)| (e.obj.key(), i)).collect();
        before - self.entries.len()
    }

    /// Samples an entry by `base_weight × decay × kind_multiplier(kind)`.
    pub fn sample(&self, rng: &mut impl Rng, kind_multiplier: impl Fn(&str) -> f64) -> Option<usize> {
        let weights: Vec<f64> =
            self.entries.iter().map(|e| e.base_weight * e.decay * kind_multiplier(e.obj.kind())).collect();
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let mut x = rng.random_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                return Some(i);
            }
            x -= w;
        }
        Some(weights.len() - 1)
    }
}

/// `OBRRT_KIND_WEIGHTS=<path>`: a JSON object {kind: multiplier} read once;
/// each entry's base weight is × its kind's multiplier (1 for kinds not in
/// it, or when unset). It feeds everything base_weight feeds: sampling, the
/// chain/zone values and the progress-only gate (weight >= 1).
fn kind_weight(kind: &str) -> f64 {
    static T: std::sync::OnceLock<Option<HashMap<String, f64>>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let path = std::env::var("OBRRT_KIND_WEIGHTS").ok()?;
        let txt = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("OBRRT_KIND_WEIGHTS {path}: {e}"));
        let t: HashMap<String, f64> = serde_json::from_str(&txt).unwrap_or_else(|e| panic!("OBRRT_KIND_WEIGHTS {path}: {e}"));
        eprintln!("kind weights: {} kinds from {path}", t.len());
        Some(t)
    })
    .as_ref()
    .map_or(1.0, |t| t.get(kind).copied().unwrap_or(1.0))
}

/// A Beta(a, b) draw for a, b >= 1, from two Gamma draws.
fn beta(rng: &mut impl Rng, a: f64, b: f64) -> f64 {
    let x = gamma(rng, a);
    let y = gamma(rng, b);
    if x + y > 0.0 { x / (x + y) } else { 0.5 }
}

/// A Gamma(shape, 1) draw for shape >= 1 (Marsaglia and Tsang).
fn gamma(rng: &mut impl Rng, shape: f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // A standard normal draw (Box-Muller).
        let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.random_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
        if u.ln() < 0.5 * z * z + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}
