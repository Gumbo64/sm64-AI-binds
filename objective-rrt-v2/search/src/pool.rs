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
    /// Rate sampler (`Config::sampler_rate`): improvement and frames spent,
    /// both faded by exp(-Δ/tau) in emulated frames since `rate_t`.
    pub rate_i: f64,
    pub rate_f: f64,
    pub rate_t: u64,
    /// Rate sampler: the soonest frame any success (or credited arrival) of
    /// it ended at, and the best (h, arrival) any kept end reached.
    pub best_hit: u32,
    pub best_end: (f64, Option<f64>),
    /// OBRRT_RATE_NOSTART_SKIP: nonzero when no live state with id below it
    /// could start this entry (whatever b): it is left out of sampling until
    /// a newer state can (`Engine::nostart_recheck`).
    pub ns_upto: u32,
}

/// Rate sampler settings (`Config::sampler_rate`, OBRRT_SAMPLER=rate).
#[derive(Debug, Clone, Copy)]
pub struct RateParams {
    /// Multiplier bounds around the median entry's rate.
    pub floor: f64,
    pub cap: f64,
    /// Evidence fades by exp(-frames / tau), in emulated frames of the run.
    pub tau: f64,
    /// Prior: this many frames at the pool's overall rate, so an entry with
    /// little evidence sits near the median and forgotten evidence returns
    /// there gradually (no resets).
    pub prior_frames: f64,
    /// Frames charged (with no improvement) for a pick that found no start.
    pub nostart_frames: f64,
    /// OBRRT_RATE_MODE=hier: an entry's prior mean is its kind's rate (kind
    /// evidence pooled over its entries, no-start charges left out), whose
    /// own prior is `kind_scale × prior_frames` frames at the pool's rate.
    /// A no-start charge is `nostart_frames × prior_frames / 2e5`, so a prior
    /// sweep doesn't change how many no-starts it takes to halve a weight.
    pub hier: bool,
    pub kind_scale: f64,
    /// OBRRT_RATE_NOSTART_SKIP=1: entries no live state could start are
    /// left out of sampling (see `Entry::ns_upto`), not charged.
    pub nostart_skip: bool,
    /// Base-criteria checks per round for re-checking skipped entries.
    pub skip_budget: u64,
    /// OBRRT_RATE_ROUTE=c: a new or sooner best time for a milestone
    /// (keys, stars) credits c × (5 if first, else seconds saved, at most 5)
    /// to the entries that made the states on its path since the previous
    /// star, split evenly; 0 = off.
    pub route: f64,
    /// OBRRT_RATE_NOVEL: improvement for a new state (default 1).
    pub novel: f64,
}

/// Rate sampler: the pool's overall rate and the last refresh's spread.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RateStats {
    pub g_i: f64,
    pub g_f: f64,
    #[serde(skip)]
    pub g_t: u64,
    /// Overall improvement per frame, and the median entry rate.
    pub r0: f64,
    pub median: f64,
    /// Multipliers (non-modal entries) at the last refresh.
    pub entries: usize,
    pub min: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub max: f64,
    pub p10: f64,
    pub p90: f64,
    pub at_floor: usize,
    pub at_cap: usize,
    /// OBRRT_RATE_MODE=hier: kinds with a rate, and the spread of their
    /// rates over the median entry's.
    pub kinds: usize,
    pub kind_min: f64,
    pub kind_p50: f64,
    pub kind_max: f64,
    /// Improvement credited so far, by source.
    pub imp_novel: f64,
    pub imp_first: f64,
    pub imp_sooner: f64,
    pub imp_closer: f64,
    pub imp_route: f64,
    pub nostart_charges: u64,
    /// Planning diagnostics (wall time, diagnostics only): seconds in picks
    /// that started / found no start, and in `sample_entry` alone; plans
    /// that ran out of retries.
    pub pick_secs: f64,
    pub nostart_secs: f64,
    pub sample_secs: f64,
    pub plans_none: u64,
    /// OBRRT_RATE_NOSTART_SKIP: no-starts with no live start anywhere (the
    /// entry is then skipped) vs. only none in b's kin set; entries skipped
    /// now, released by a re-check, and the re-checks' seconds.
    pub nostart_global: u64,
    pub nostart_kin: u64,
    pub skipped: usize,
    pub released: u64,
    pub recheck_secs: f64,
}

/// OBRRT_RATE_MODE=hier: one kind's pooled evidence (faded like an
/// entry's) and its rate at the last refresh.
#[derive(Debug, Clone, Default)]
pub struct KindRate {
    pub i: f64,
    pub f: f64,
    pub t: u64,
    pub r: f64,
}

impl Entry {
    /// Rate sampler: this entry's improvement per frame at `now`, with the
    /// prior `r0` over `prior_frames`.
    pub fn rate(&self, now: u64, r0: f64, p: &RateParams) -> f64 {
        let k = (-(now.saturating_sub(self.rate_t) as f64) / p.tau).exp();
        (self.rate_i * k + r0 * p.prior_frames) / (self.rate_f * k + p.prior_frames)
    }
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
    /// Rate sampler state (unused when off).
    pub rate: RateStats,
    /// OBRRT_RATE_MODE=hier: per kind.
    pub kind_rate: std::collections::BTreeMap<&'static str, KindRate>,
    /// `Engine::sample_entry`'s per-kind tables (b-independent eligibility):
    /// built at a round's first pick, dropped when its planning ends.
    pub sidx: Option<SampleIdx>,
}

/// One kind's entries (pool order) for `SampleIdx`: weight
/// `base_weight × decay` (0 when not eligible), and the running sum of
/// those weights, accumulated in pool order exactly as `sample_entry`'s
/// kind sum is, so its last value is bit-identical to that sum.
#[derive(Default)]
pub struct KindTab {
    pub name: &'static str,
    pub members: Vec<u32>,
    pub w: Vec<f64>,
    pub elig: Vec<bool>,
    pub prefix: Vec<f64>,
    /// Eligible members, and the first position whose prefix is stale.
    pub count: usize,
    dirty_from: usize,
}

/// Per-kind sampling tables for `Engine::sample_entry` when eligibility is
/// `!modal && ns_upto == 0` (no b). Within a round's planning an entry's
/// weight or eligibility changes only through a no-start pick (`touch`)
/// and entries are only appended (`sync`); everything else changes between
/// rounds, when the tables are dropped.
#[derive(Default)]
pub struct SampleIdx {
    pub kinds: Vec<KindTab>,
    pub by_name: std::collections::BTreeMap<&'static str, u16>,
    kind_of: Vec<u16>,
    pos: Vec<u32>,
    modal: Vec<bool>,
    /// Eligible entries whose weight is NaN or negative (then the exact
    /// old path is used: its two sums would differ).
    pub irregular: usize,
}

impl SampleIdx {
    fn weight(e: &Entry, modal: bool) -> (f64, bool) {
        let elig = !modal && e.ns_upto == 0;
        (if elig { e.base_weight * e.decay } else { 0.0 }, elig)
    }

    /// Adds the entries appended since the last call.
    pub fn sync(&mut self, entries: &[Entry]) {
        for i in self.kind_of.len()..entries.len() {
            let e = &entries[i];
            let name = e.obj.kind();
            let k = match self.by_name.get(name) {
                Some(&k) => k,
                None => {
                    let k = self.kinds.len() as u16;
                    self.by_name.insert(name, k);
                    self.kinds.push(KindTab { name, ..Default::default() });
                    k
                }
            };
            let modal = e.obj.modal();
            let (w, elig) = Self::weight(e, modal);
            let t = &mut self.kinds[k as usize];
            self.kind_of.push(k);
            self.pos.push(t.members.len() as u32);
            self.modal.push(modal);
            t.members.push(i as u32);
            t.w.push(w);
            t.elig.push(elig);
            t.count += elig as usize;
            self.irregular += (elig && !(w >= 0.0)) as usize;
        }
    }

    /// Every entry's weight and eligibility re-read (a new round), plus the
    /// entries appended since; kinds and modality are kept (an objective's
    /// don't change; `Pool::expire`, which renumbers, drops the tables).
    pub fn reweigh(&mut self, entries: &[Entry]) {
        self.irregular = 0;
        for t in &mut self.kinds {
            t.count = 0;
            for p in 0..t.members.len() {
                let i = t.members[p] as usize;
                let (w, elig) = Self::weight(&entries[i], self.modal[i]);
                t.w[p] = w;
                t.elig[p] = elig;
                t.count += elig as usize;
                self.irregular += (elig && !(w >= 0.0)) as usize;
            }
            t.dirty_from = 0;
        }
        self.sync(entries);
    }

    /// Entry `i`'s weight or eligibility may have changed.
    pub fn touch(&mut self, entries: &[Entry], i: usize) {
        self.sync(entries);
        let (k, p) = (self.kind_of[i] as usize, self.pos[i] as usize);
        let (w, elig) = Self::weight(&entries[i], self.modal[i]);
        let t = &mut self.kinds[k];
        self.irregular -= (t.elig[p] && !(t.w[p] >= 0.0)) as usize;
        self.irregular += (elig && !(w >= 0.0)) as usize;
        t.count = t.count + elig as usize - t.elig[p] as usize;
        t.w[p] = w;
        t.elig[p] = elig;
        t.dirty_from = t.dirty_from.min(p);
    }

    /// Brings every kind's running sums up to date.
    pub fn refresh(&mut self) {
        for t in &mut self.kinds {
            let n = t.w.len();
            if t.dirty_from >= n && t.prefix.len() == n {
                continue;
            }
            let from = t.dirty_from.min(t.prefix.len());
            t.prefix.truncate(from);
            let mut s = if from == 0 { 0.0 } else { t.prefix[from - 1] };
            for p in from..n {
                // Skipping an ineligible entry (as the old loop did) and
                // adding its 0.0 give the same value.
                if t.elig[p] {
                    s += t.w[p];
                }
                t.prefix.push(s);
            }
            t.dirty_from = usize::MAX;
        }
    }
}

impl KindTab {
    /// The kind's eligible weight sum (bit-identical to summing them in
    /// pool order from 0.0).
    pub fn sum(&self) -> f64 {
        self.prefix.last().copied().unwrap_or(0.0)
    }

    /// `weighted`'s pick over this kind's eligible entries for a draw `x`
    /// in [0, sum): the first entry at which `x` minus the weights before
    /// it (subtracted one by one in f64) falls below its weight. Found by
    /// binary search on the running sums when `x` is clear of every
    /// boundary by more than both computations' rounding error can move it
    /// (each is at most len × eps/2 × sum: all terms are non-negative and
    /// every partial result lies in [0, sum]), else by the exact loop.
    pub fn pick(&self, x: f64) -> u32 {
        let m = self.prefix.len();
        let total = self.sum();
        let k = self.prefix.partition_point(|&p| p <= x);
        let delta = 4.0 * (m as f64 + 2.0) * f64::EPSILON * total;
        if k < m && self.prefix[k] - x > delta && (k == 0 || x - self.prefix[k - 1] > delta) {
            return self.members[k];
        }
        self.pick_exact(x)
    }

    /// The old loop (`weighted`) itself.
    pub fn pick_exact(&self, mut x: f64) -> u32 {
        let mut last = None;
        for p in 0..self.members.len() {
            if !self.elig[p] {
                continue;
            }
            let w = self.w[p].max(0.0);
            if x < w {
                return self.members[p];
            }
            x -= w;
            last = Some(self.members[p]);
        }
        last.expect("a kind with positive weight has an eligible entry")
    }
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
        self.entries.push(Entry { obj, base_weight, decay: 1.0, attempts: 0, successes: 0, unattemptable: 0, best_start: f64::INFINITY, best_arrival: f64::INFINITY, wear: HashMap::new(), recoveries: ROUTE_RECOVERIES, bandit_k: 0.0, bandit_n: 0.0, rate_i: 0.0, rate_f: 0.0, rate_t: 0, best_hit: u32::MAX, best_end: (f64::INFINITY, None), ns_upto: 0 });
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

    /// Rate sampler: adds `imp` improvement and `frames` frames to entry `i`
    /// (and the pool's totals) at emulated frame `now`, older evidence faded.
    /// `nostart`: a no-start charge (hier: scaled, and not the kind's).
    pub fn rate_note(&mut self, i: usize, imp: f64, frames: f64, now: u64, p: &RateParams, nostart: bool) {
        let fade = |t: u64| (-(now.saturating_sub(t) as f64) / p.tau).exp();
        let frames = if p.hier && nostart { frames * p.prior_frames / 200_000.0 } else { frames };
        if p.hier && !nostart {
            let kr = self.kind_rate.entry(self.entries[i].obj.kind()).or_default();
            let k = fade(kr.t);
            kr.i = kr.i * k + imp;
            kr.f = kr.f * k + frames;
            kr.t = now;
        }
        let g = &mut self.rate;
        let k = fade(g.g_t);
        g.g_i = g.g_i * k + imp;
        g.g_f = g.g_f * k + frames;
        g.g_t = now;
        let e = &mut self.entries[i];
        let k = fade(e.rate_t);
        e.rate_i = e.rate_i * k + imp;
        e.rate_f = e.rate_f * k + frames;
        e.rate_t = now;
    }

    /// Rate sampler: the pool's overall rate (a small default before any
    /// frames are spent).
    fn r0(&self, now: u64, p: &RateParams) -> f64 {
        let g = &self.rate;
        let k = (-(now.saturating_sub(g.g_t) as f64) / p.tau).exp();
        if g.g_f * k > 0.0 { (g.g_i * k / (g.g_f * k)).max(1e-9) } else { 1e-5 }
    }

    /// Rate sampler: every entry's `decay` becomes its rate over the median
    /// non-modal entry's, clamped to [floor, cap] (modal entries: 1). Called
    /// at the start of each round; deterministic (entries in pool order).
    pub fn rate_refresh(&mut self, now: u64, p: &RateParams) {
        if p.hier {
            return self.rate_refresh_hier(now, p);
        }
        let r0 = self.r0(now, p);
        let mut rates: Vec<f64> = self.entries.iter().filter(|e| !e.obj.modal()).map(|e| e.rate(now, r0, p)).collect();
        rates.sort_by(f64::total_cmp);
        let median = if rates.is_empty() { r0 } else { rates[rates.len() / 2] }.max(1e-12);
        let mut mults = Vec::with_capacity(rates.len());
        for e in &mut self.entries {
            e.decay = if e.obj.modal() { 1.0 } else { (e.rate(now, r0, p) / median).clamp(p.floor, p.cap) };
            if !e.obj.modal() {
                mults.push(e.decay);
            }
        }
        self.rate_spread(mults, r0, median, p);
    }

    /// The last refresh's multiplier quantiles.
    fn rate_spread(&mut self, mut mults: Vec<f64>, r0: f64, median: f64, p: &RateParams) {
        mults.sort_by(f64::total_cmp);
        let q = |f: f64| if mults.is_empty() { 0.0 } else { mults[((mults.len() - 1) as f64 * f).round() as usize] };
        let g = &mut self.rate;
        g.r0 = r0;
        g.median = median;
        g.entries = mults.len();
        (g.min, g.p25, g.p50, g.p75, g.max) = (q(0.0), q(0.25), q(0.5), q(0.75), q(1.0));
        (g.p10, g.p90) = (q(0.1), q(0.9));
        g.at_floor = mults.iter().filter(|&&m| m <= p.floor).count();
        g.at_cap = mults.iter().filter(|&&m| m >= p.cap).count();
    }

    /// OBRRT_RATE_MODE=hier: the pool's rate from its kinds' evidence (no
    /// no-start charges), each kind's rate shrunk toward it, each entry's
    /// toward its kind's; multipliers over the median entry's as before.
    fn rate_refresh_hier(&mut self, now: u64, p: &RateParams) {
        let fade = |t: u64| (-(now.saturating_sub(t) as f64) / p.tau).exp();
        let (si, sf) = self.kind_rate.values().fold((0.0, 0.0), |(a, b), k| (a + k.i * fade(k.t), b + k.f * fade(k.t)));
        let r0 = if sf > 0.0 { (si / sf).max(1e-9) } else { 1e-5 };
        let pk = p.prior_frames * p.kind_scale;
        for k in self.kind_rate.values_mut() {
            let f = fade(k.t);
            k.r = (k.i * f + r0 * pk) / (k.f * f + pk);
        }
        let kind_r = |kr: &std::collections::BTreeMap<&'static str, KindRate>, k: &str| kr.get(k).map_or(r0, |k| k.r);
        let mut rates: Vec<f64> =
            self.entries.iter().filter(|e| !e.obj.modal()).map(|e| e.rate(now, kind_r(&self.kind_rate, e.obj.kind()), p)).collect();
        rates.sort_by(f64::total_cmp);
        let median = if rates.is_empty() { r0 } else { rates[rates.len() / 2] }.max(1e-12);
        let mut mults = Vec::with_capacity(rates.len());
        for e in &mut self.entries {
            e.decay = if e.obj.modal() { 1.0 } else { (e.rate(now, kind_r(&self.kind_rate, e.obj.kind()), p) / median).clamp(p.floor, p.cap) };
            if !e.obj.modal() {
                mults.push(e.decay);
            }
        }
        let mut km: Vec<f64> = self.kind_rate.values().map(|k| k.r / median).collect();
        km.sort_by(f64::total_cmp);
        let g = &mut self.rate;
        g.kinds = km.len();
        (g.kind_min, g.kind_p50, g.kind_max) =
            if km.is_empty() { (0.0, 0.0, 0.0) } else { (km[0], km[km.len() / 2], km[km.len() - 1]) };
        self.rate_spread(mults, r0, median, p);
    }

    /// Rate sampler: recomputes one entry's multiplier against the last
    /// refresh's median (after a no-start charge, so a round's later picks
    /// see it).
    pub fn rate_one(&mut self, i: usize, now: u64, p: &RateParams) {
        let (r0, median) = (self.rate.r0.max(1e-9), self.rate.median.max(1e-12));
        let r0 = if p.hier { self.kind_rate.get(self.entries[i].obj.kind()).filter(|k| k.r > 0.0).map_or(r0, |k| k.r) } else { r0 };
        let e = &mut self.entries[i];
        if !e.obj.modal() {
            e.decay = (e.rate(now, r0, p) / median).clamp(p.floor, p.cap);
        }
    }

    /// The entry with this key, if it's in the pool.
    pub fn find(&self, key: &str) -> Option<usize> {
        self.index.get(key).copied()
    }

    /// Removes objectives that no state could attempt for `rounds` rounds.
    pub fn expire(&mut self, rounds: u32) -> usize {
        let before = self.entries.len();
        self.sidx = None;
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
pub(crate) fn kind_weight(kind: &str) -> f64 {
    static T: std::sync::OnceLock<Option<HashMap<String, f64>>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let path = obrrt_core::sys::env::var("OBRRT_KIND_WEIGHTS").ok()?;
        let txt = obrrt_core::sys::fs::read_to_string(&path).unwrap_or_else(|e| panic!("OBRRT_KIND_WEIGHTS {path}: {e}"));
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
