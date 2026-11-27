//! The objective pool: objectives keyed by identity, each with its weight
//! from creation and its rate multiplier (the rate sampler).

use std::collections::HashMap;
use std::sync::Arc;

use obrrt_core::{Objective, St};

pub struct Entry {
    pub obj: Arc<dyn Objective>,
    /// `obj_weight(b)` when it entered the pool.
    pub base_weight: f64,
    /// The rate sampler's multiplier (the entry's rate over the median entry's, clamped), set each round.
    pub decay: f64,
    pub attempts: u32,
    pub successes: u32,
    /// Rounds since any state could attempt it.
    pub unattemptable: u32,
    /// Best (lowest) start rank of any start it was attempted from.
    pub best_start: f64,
    /// Rate sampler: improvement and frames spent,
    /// both faded by exp(-Δ/tau) in emulated frames since `rate_t`.
    pub rate_i: f64,
    pub rate_f: f64,
    pub rate_t: u64,
    /// Rate sampler: the soonest frame any success (or credited arrival) of
    /// it ended at, and the best start rank any kept end reached.
    pub best_hit: u32,
    pub best_end: f64,
}

/// Rate sampler settings.
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
    /// Improvement for a new state (default 1).
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
    /// Improvement credited so far, by source.
    pub imp_novel: f64,
    pub imp_first: f64,
    pub imp_sooner: f64,
    pub imp_closer: f64,
    pub nostart_charges: u64,
    /// Planning diagnostics (wall time, diagnostics only): seconds in picks
    /// that started / found no start, and in `sample_entry` alone; plans
    /// that ran out of retries.
    pub pick_secs: f64,
    pub nostart_secs: f64,
    pub sample_secs: f64,
    pub plans_none: u64,
}

impl Entry {
    /// Rate sampler: this entry's improvement per frame at `now`, with the
    /// prior `r0` over `prior_frames`.
    pub fn rate(&self, now: u64, r0: f64, p: &RateParams) -> f64 {
        let k = (-(now.saturating_sub(self.rate_t) as f64) / p.tau).exp();
        (self.rate_i * k + r0 * p.prior_frames) / (self.rate_f * k + p.prior_frames)
    }
}

#[derive(Default)]
pub struct Pool {
    pub entries: Vec<Entry>,
    index: HashMap<String, usize>,
    /// Rate sampler state.
    pub rate: RateStats,
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
/// `!modal` (no b). Within a round's planning an entry's
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
        let elig = !modal;
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
        let base_weight = obj.obj_weight(b).max(0.0);
        self.index.insert(key, self.entries.len());
        self.entries.push(Entry { obj, base_weight, decay: 1.0, attempts: 0, successes: 0, unattemptable: 0, best_start: f64::INFINITY, rate_i: 0.0, rate_f: 0.0, rate_t: 0, best_hit: u32::MAX, best_end: f64::INFINITY });
        true
    }

    /// Rate sampler: adds `imp` improvement and `frames` frames to entry `i`
    /// (and the pool's totals) at emulated frame `now`, older evidence faded.
    pub fn rate_note(&mut self, i: usize, imp: f64, frames: f64, now: u64, p: &RateParams) {
        let fade = |t: u64| (-(now.saturating_sub(t) as f64) / p.tau).exp();
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

    /// Rate sampler: recomputes one entry's multiplier against the last
    /// refresh's median (after a no-start charge, so a round's later picks
    /// see it).
    pub fn rate_one(&mut self, i: usize, now: u64, p: &RateParams) {
        let (r0, median) = (self.rate.r0.max(1e-9), self.rate.median.max(1e-12));
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

}

