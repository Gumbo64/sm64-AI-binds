//! In-run ETA calibration (`OBRRT_ETA_LEARN=1`, off by default; off, the
//! search is byte-identical to before).
//!
//! Objectives' ETAs are rough: per kind, successes take 0.6–2.3× the frames
//! their start ETA said (`tools/analysis/eta_calib.py`). An underestimated
//! kind looks closer than it is. This learns, per objective kind, a factor
//! c_kind from the run's own successful segments and scales the start-side
//! estimates by it.
//!
//! * **Sample**: each successful segment with a finite start ETA > 0 gives
//!   `ln(clamp(seg_frames / start_eta, 0.25, 4))` (the same ratio
//!   eta_calib.py reports), noted at merge time in merge order.
//! * **Estimate**: the median of the kind's last `OBRRT_ETA_LEARN_WINDOW`
//!   (256) log ratios, shrunk toward 0 (c = 1) with a prior of
//!   `OBRRT_ETA_LEARN_N0` (20) successes: `ln c = n / (n + n0) × median`,
//!   n = the samples in the window. So c = exp of that, in [0.25, 4].
//! * **Snapshot**: the factors used are the ones at the round's start
//!   ([`EtaLearn::snapshot`]): planning, the merge's improvement checks and
//!   the BnB bounds handed to the workers all see one value per round,
//!   whatever the thread count.
//!
//! Where c applies (start side only): the start rank (`c × start_eta +
//! start_bias`, engine `rank_of` / `keys_of` / `pick_start`, schemapick),
//! the arrival estimate (`frame + c × eta`), and the BnB / Beat optimism
//! (`min(c, 1) × OBRRT_BEAT_OPT`: an overestimating kind's bound tightens to
//! what it really takes; an underestimating one keeps 0.7, never more
//! aggressive). Per-kind cutoffs stay on the raw ETA: they are learned from
//! raw start ETAs of successes and compared with the raw ETA, so they don't
//! shift when c does. Not in GA fitness: within an attempt the kind is
//! constant, so scaling would only rescale the rewards against the success
//! bonus and the death penalty.

use std::collections::{BTreeMap, HashMap, VecDeque};

pub struct EtaLearn {
    pub on: bool,
    n0: f64,
    window: usize,
    kinds: BTreeMap<&'static str, KindCal>,
    /// The round-start factors (kinds with samples only; others 1).
    snap: HashMap<&'static str, f64>,
}

#[derive(Default)]
struct KindCal {
    /// The last `window` log ratios.
    logs: VecDeque<f64>,
    /// All successes noted, and the median log ratio of the window.
    n: u64,
    raw_median: f64,
    dirty: bool,
}

/// Clamp on a single ratio before its log.
const RATIO_MIN: f64 = 0.25;
const RATIO_MAX: f64 = 4.0;

impl EtaLearn {
    pub fn from_env() -> EtaLearn {
        let num = |k: &str, d: f64| obrrt_core::sys::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        EtaLearn {
            on: obrrt_core::sys::env::var("OBRRT_ETA_LEARN").is_ok_and(|v| v == "1"),
            n0: num("OBRRT_ETA_LEARN_N0", 20.0).max(0.0),
            window: (num("OBRRT_ETA_LEARN_WINDOW", 256.0) as usize).max(1),
            kinds: BTreeMap::new(),
            snap: HashMap::new(),
        }
    }

    /// The round-start factor for `kind` (1 when off or unlearned).
    #[inline]
    pub fn c(&self, kind: &str) -> f64 {
        if !self.on {
            return 1.0;
        }
        self.snap.get(kind).copied().unwrap_or(1.0)
    }

    /// The BnB / Beat optimism for `kind`: `min(c, 1) × opt` (`opt` when off).
    pub fn opt(&self, kind: &str, opt: f64) -> f64 {
        if !self.on {
            return opt;
        }
        self.c(kind).min(1.0) * opt
    }

    /// A merged segment of `kind`: noted if it met its objective from a
    /// finite positive start ETA.
    pub fn note(&mut self, kind: &'static str, success: bool, start_eta: f64, frames: usize) {
        if !self.on || !success || !(start_eta.is_finite() && start_eta > 0.0) {
            return;
        }
        let r = (frames as f64 / start_eta).clamp(RATIO_MIN, RATIO_MAX);
        let k = self.kinds.entry(kind).or_default();
        if k.logs.len() == self.window {
            k.logs.pop_front();
        }
        k.logs.push_back(r.ln());
        k.n += 1;
        k.dirty = true;
    }

    fn factor(&self, k: &KindCal) -> f64 {
        let n = k.logs.len() as f64;
        (k.raw_median * n / (n + self.n0)).exp()
    }

    /// At a round's start: the factors this round uses.
    pub fn snapshot(&mut self) {
        if !self.on {
            return;
        }
        for k in self.kinds.values_mut().filter(|k| k.dirty) {
            let mut v: Vec<f64> = k.logs.iter().copied().collect();
            v.sort_by(f64::total_cmp);
            let m = v.len();
            k.raw_median = if m % 2 == 1 { v[m / 2] } else { 0.5 * (v[m / 2 - 1] + v[m / 2]) };
            k.dirty = false;
        }
        let snap: HashMap<&'static str, f64> = self.kinds.iter().map(|(&name, k)| (name, self.factor(k))).collect();
        self.snap = snap;
    }

    /// summary.json's `eta_learn`: per kind, successes noted, the window's
    /// raw median ratio and the factor in use.
    pub fn summary(&self) -> serde_json::Value {
        let kinds: BTreeMap<&str, serde_json::Value> = self
            .kinds
            .iter()
            .map(|(&name, k)| {
                (name, serde_json::json!({"n": k.n, "window": k.logs.len(), "raw_median": k.raw_median.exp(), "c": self.c(name)}))
            })
            .collect();
        serde_json::json!({"on": self.on, "n0": self.n0, "window": self.window, "kinds": kinds})
    }

    /// One log line: the kinds whose factor is furthest from 1 (|ln c|), at
    /// most `top`.
    pub fn report(&self, top: usize) -> String {
        if !self.on {
            return String::new();
        }
        let mut v: Vec<(&str, f64, u64)> = self.kinds.iter().map(|(&name, k)| (name, self.c(name), k.n)).collect();
        v.sort_by(|a, b| b.1.ln().abs().total_cmp(&a.1.ln().abs()).then(a.0.cmp(b.0)));
        let parts: Vec<String> = v.iter().take(top).map(|(k, c, n)| format!("{k} {c:.2} (n {n})")).collect();
        format!("eta learn: {} kinds; top deviations: {}\n", self.kinds.len(), parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on(n0: f64) -> EtaLearn {
        EtaLearn { on: true, n0, window: 8, kinds: BTreeMap::new(), snap: HashMap::new() }
    }

    #[test]
    fn off_is_one() {
        let mut e = on(0.0);
        e.on = false;
        e.note("A", true, 10.0, 40);
        e.snapshot();
        assert_eq!(e.c("A"), 1.0);
        assert_eq!(e.opt("A", 0.7), 0.7);
    }

    #[test]
    fn learns_median_shrunk_and_snapshots() {
        let mut e = on(8.0);
        for _ in 0..8 {
            e.note("A", true, 10.0, 20);
        }
        e.note("A", false, 10.0, 1000); // failures ignored
        e.note("A", true, f64::INFINITY, 5); // no ETA ignored
        assert_eq!(e.c("A"), 1.0); // not before the snapshot
        e.snapshot();
        // median 2, n = 8 = n0: half way in log space.
        assert!((e.c("A") - 2f64.sqrt()).abs() < 1e-12);
        assert!((e.opt("A", 0.7) - 0.7).abs() < 1e-12);
        assert_eq!(e.c("B"), 1.0);
    }

    #[test]
    fn clamps_and_windows() {
        let mut e = on(0.0);
        for _ in 0..8 {
            e.note("A", true, 1.0, 100); // ratio 100 -> 4
        }
        e.snapshot();
        assert!((e.c("A") - 4.0).abs() < 1e-12);
        for _ in 0..8 {
            e.note("A", true, 10.0, 5); // 0.5; the window forgets the 4s
        }
        e.snapshot();
        assert!((e.c("A") - 0.5).abs() < 1e-12);
        assert!((e.opt("A", 0.7) - 0.35).abs() < 1e-12);
    }
}
