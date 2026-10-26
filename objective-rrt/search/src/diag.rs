//! Diagnostics: which objective types and which states use the compute, and
//! which turn it into progress.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Serialize;

use crate::tree::Tree;

/// Frames an attempt costs beyond the frames it plays.
pub const ATTEMPT_OVERHEAD: f64 = 500.0;

/// Per objective kind, over the whole run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct KindStats {
    /// Times picked from the pool.
    pub samples: u64,
    /// Picks that found no eligible start.
    pub no_start: u64,
    pub attempts: u64,
    pub successes: u64,
    pub frames: u64,
    pub added: u64,
    pub frontier: u64,
    /// States that were the first in their (level, area).
    pub new_zones: u64,
    /// Sum of success_value over successes that added a state.
    pub value: f64,
    pub died: u64,
    /// Sum of progress at failure, for the mean.
    pub fail_progress: f64,
    pub fails: u64,
    /// Learned sampling multiplier (automatic tuning).
    pub multiplier: f64,
    /// Why the multiplier last changed.
    pub reason: String,
    /// base_heuristic of the starts of successful attempts.
    #[serde(skip)]
    pub success_starts: Vec<f64>,
    /// Learned eligibility cutoff on base_heuristic (automatic tuning):
    /// starts above it aren't picked. Infinite until learned.
    pub cutoff: f64,
    /// Picks / no-start picks since the cutoff last changed.
    #[serde(skip)]
    pub window_samples: u64,
    #[serde(skip)]
    pub window_no_start: u64,
    pub pool_size: usize,
    /// Successes as the 2nd or later objective of a chain.
    pub chained_successes: u64,
    /// Decay recoveries from progress along another route (decay_route).
    pub route_recoveries: u64,
    /// Kind bandit (`Config::kind_bandit`): discounted improvements and
    /// attempts of this kind (the entry bandit's signal), faded by
    /// `kind_gamma` per attempt of this kind. Reported in summary.json's
    /// `kind_bandit` section, not here.
    #[serde(skip)]
    pub bandit_k: f64,
    #[serde(skip)]
    pub bandit_n: f64,
}

/// Per chain length (objectives per attempt), over the whole run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChainStats {
    pub attempts: u64,
    pub frames: u64,
    /// Attempts by how many of the chain's objectives were met (index = count).
    pub met: Vec<u64>,
    pub added: u64,
    pub frontier: u64,
    pub new_zones: u64,
    pub value: f64,
}

impl ChainStats {
    /// As [`KindStats::yield_per_kframe`], per attempt of this length.
    pub fn yield_per_kframe(&self) -> f64 {
        let cost = self.frames as f64 + ATTEMPT_OVERHEAD * self.attempts as f64;
        (self.added as f64 + 20.0 * self.frontier as f64 + 10.0 * self.new_zones as f64 + self.value) / (cost / 1000.0).max(1.0)
    }
}

impl KindStats {
    /// Kind bandit: one attempt of this kind, improved or not.
    pub fn observe(&mut self, improved: bool, gamma: f64) {
        self.bandit_k = self.bandit_k * gamma + improved as u32 as f64;
        self.bandit_n = self.bandit_n * gamma + 1.0;
    }

    /// Kind bandit: posterior mean of the improvement rate per attempt,
    /// Beta(k + 1, n - k + 1). No Thompson draw: optimism over many arms
    /// is what we avoid.
    pub fn bandit_mean(&self) -> f64 {
        (self.bandit_k + 1.0) / (self.bandit_n + 2.0)
    }

    /// Frames an attempt of this kind costs, overhead included (as the kind
    /// floor in `sample_entry` reckons it; 5000 before any attempt).
    pub fn cost_per_attempt(&self) -> f64 {
        if self.attempts == 0 { 5000.0 } else { self.frames as f64 / self.attempts as f64 + ATTEMPT_OVERHEAD }
    }

    pub fn success_rate(&self) -> f64 {
        self.successes as f64 / self.attempts.max(1) as f64
    }

    /// Wilson 95% interval for the success rate.
    pub fn success_interval(&self) -> (f64, f64) {
        let n = self.attempts as f64;
        if n == 0.0 {
            return (0.0, 1.0);
        }
        let p = self.success_rate();
        let z = 1.96;
        let d = 1.0 + z * z / n;
        let c = (p + z * z / (2.0 * n)) / d;
        let h = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt() / d;
        ((c - h).max(0.0), (c + h).min(1.0))
    }

    /// New states per 1,000 frames of cost; a frontier advance counts 20,
    /// a new zone 10, and successes their objective's success_value. Each attempt also costs `ATTEMPT_OVERHEAD` frames
    /// (planning, restoring its start), so cheap macros aren't overrated.
    pub fn yield_per_kframe(&self) -> f64 {
        let cost = self.frames as f64 + ATTEMPT_OVERHEAD * self.attempts as f64;
        (self.added as f64 + 20.0 * self.frontier as f64 + 10.0 * self.new_zones as f64 + self.value) / (cost / 1000.0).max(1.0)
    }

    pub fn flags(&self, sample_share: f64, frame_share: f64, mean_yield: f64) -> Vec<&'static str> {
        let mut f = Vec::new();
        if (sample_share > 0.25 || frame_share > 0.25) && self.attempts >= 20 && self.yield_per_kframe() < 0.5 * mean_yield {
            f.push("DOMINANT-UNPRODUCTIVE");
        }
        if self.attempts >= 20 && self.success_rate() < 0.05 {
            f.push("LOW-SUCCESS");
        }
        if self.samples >= 20 && self.no_start as f64 > 0.5 * self.samples as f64 {
            f.push("STARVED");
        }
        if self.attempts >= 100 && self.successes == 0 {
            f.push("NEVER-SUCCEEDS");
        }
        f
    }
}

/// RRT* rewiring.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RewireStats {
    /// States a sooner state reached the place of.
    pub dominated: u64,
    pub queued: u64,
    pub attempts: u64,
    /// Rewiring attempts that reached the target's place sooner, and by how
    /// many frames in all.
    pub improved: u64,
    pub frames_saved: u64,
}

#[derive(Default)]
pub struct Diagnostics {
    pub rewire: RewireStats,
    /// Earliest game frame reaching each (keys, stars) and each zone, and
    /// how many times it improved.
    pub best_progress: BTreeMap<(u8, i16), (u32, u32)>,
    pub best_zone: BTreeMap<(u16, u8), (u32, u32)>,
    pub kinds: BTreeMap<&'static str, KindStats>,
    pub chains: BTreeMap<usize, ChainStats>,
    /// Per GA budget class ([`crate::ga::budget_class`]: macro, explore,
    /// precise); `met` is indexed by objectives met, as for chains.
    pub budgets: BTreeMap<&'static str, ChainStats>,
    /// Probability of extending an attempt by one more objective (learned).
    pub chain_prob: f64,
    /// s0 picks per node in the current window, for concentration.
    pub window_starts: Vec<u32>,
}

impl Diagnostics {
    pub fn note_best(&mut self, progress: (u8, i16), zone: (u16, u8), frame: u32) {
        for slot in [self.best_progress.entry(progress).or_insert((u32::MAX, 0)), ] {
            if frame < slot.0 {
                slot.1 += (slot.0 != u32::MAX) as u32;
                slot.0 = frame;
            }
        }
        let z = self.best_zone.entry(zone).or_insert((u32::MAX, 0));
        if frame < z.0 {
            z.1 += (z.0 != u32::MAX) as u32;
            z.0 = frame;
        }
    }

    pub fn kind(&mut self, k: &'static str) -> &mut KindStats {
        self.kinds.entry(k).or_insert_with(|| KindStats { multiplier: 1.0, cutoff: f64::INFINITY, ..Default::default() })
    }

    /// Kind bandit: sets every observed kind's multiplier to its posterior
    /// mean improvements per kframe, normalised to geometric mean 1 over those kinds
    /// (unobserved kinds, modal ones among them, keep 1), then clamped to
    /// [0.1, 5], the old yield target's range: the bandit changes the signal
    /// and the forgetting, not the dynamic range.
    ///
    /// Per kframe, not per attempt: a pick of a kind spends that kind's cost
    /// in frames, and frames are the budget. Per attempt, a GA kind costing
    /// ~10x a macro would get the same picks at the same rate, so ~10x the
    /// compute for the same improvements per pick. Dividing by cost makes
    /// the pick share follow improvements per frame; `ATTEMPT_OVERHEAD`
    /// keeps very cheap kinds from being overrated, and the cost-scaled
    /// floor (and the cap) in `sample_entry` still bound the shares.
    pub fn refresh_kind_bandit(&mut self) {
        let rate = |k: &KindStats| k.bandit_mean() / (k.cost_per_attempt() / 1000.0);
        let seen: Vec<f64> = self.kinds.values().filter(|k| k.bandit_n > 0.0).map(rate).collect();
        if seen.is_empty() {
            return;
        }
        // Geometric mean, so one or two cheap macro kinds with very high
        // rates don't set the scale for every other kind. Rates are > 0
        // ((k+1)/(n+2) > 0, cost > 0), so ln is finite.
        let mean = (seen.iter().map(|r| r.ln()).sum::<f64>() / seen.len() as f64).exp();
        for k in self.kinds.values_mut().filter(|k| k.bandit_n > 0.0) {
            k.multiplier = (rate(k) / mean).clamp(0.1, 5.0);
        }
    }

    pub fn mean_yield(&self) -> f64 {
        let ys: Vec<f64> = self.kinds.values().filter(|k| k.attempts >= 5).map(|k| k.yield_per_kframe()).collect();
        if ys.is_empty() { 1.0 } else { ys.iter().sum::<f64>() / ys.len() as f64 }
    }

    /// Effective number of distinct starts in the window: 1 / Σ p².
    pub fn effective_starts(&self) -> f64 {
        let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
        for &s in &self.window_starts {
            *counts.entry(s).or_default() += 1;
        }
        let n = self.window_starts.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        1.0 / counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
    }

    pub fn report(&self, tree: &Tree, secs: f64, total_frames: u64) -> String {
        let total_samples: u64 = self.kinds.values().map(|k| k.samples).sum::<u64>().max(1);
        let total_kframes: u64 = self.kinds.values().map(|k| k.frames).sum::<u64>().max(1);
        let mean = self.mean_yield();
        let mut s = String::new();
        let (keys, stars) = tree.frontier_progress;
        let _ = writeln!(
            s,
            "\n== {:.0}s  nodes {}  frontier keys {} stars {} ({} states)  {:.0} frames/s  states {:.0} MB  eff. starts {:.1}",
            secs,
            tree.nodes.len(),
            keys,
            stars,
            tree.frontier.len(),
            total_frames as f64 / secs.max(1.0),
            tree.delta_bytes as f64 / 1e6,
            self.effective_starts()
        );
        let _ = writeln!(
            s,
            "{:<22} {:>6} {:>6} {:>6} {:>6} {:>12} {:>6} {:>7} {:>5} {:>6} {:>5} {:>5}  flags",
            "kind", "pool", "samp%", "cpu%", "tries", "success", "added", "yield", "front", "mult", "nostr", "chain"
        );
        let _ = s.len();
        for (name, k) in &self.kinds {
            let ss = k.samples as f64 / total_samples as f64;
            let fs = k.frames as f64 / total_kframes as f64;
            let (lo, hi) = k.success_interval();
            let _ = writeln!(
                s,
                "{:<22} {:>6} {:>5.1}% {:>5.1}% {:>6} {:>4.0}% {:>2.0}-{:<3.0} {:>6} {:>7.2} {:>5} {:>6.2} {:>5} {:>5}  {}",
                name,
                k.pool_size,
                100.0 * ss,
                100.0 * fs,
                k.attempts,
                100.0 * k.success_rate(),
                100.0 * lo,
                100.0 * hi,
                k.added,
                k.yield_per_kframe(),
                k.frontier,
                k.multiplier,
                k.no_start,
                k.chained_successes,
                {
                    let mut f = k.flags(ss, fs, mean).join(" ");
                    if k.cutoff.is_finite() {
                        f += &format!(" cutoff {:.1}", k.cutoff);
                    }
                    f
                }
            );
        }
        // Chains: does planning ahead pay?
        let _ = writeln!(s, "chains (p extend {:.2}):", self.chain_prob);
        for (len, c) in &self.chains {
            let met: Vec<String> = c.met.iter().map(|m| format!("{:.0}%", 100.0 * *m as f64 / c.attempts.max(1) as f64)).collect();
            let _ = writeln!(
                s,
                "  len {len}: {:>6} tries  met 0..{len}: {}  added {}  front {}  yield {:.2}",
                c.attempts,
                met.join(" / "),
                c.added,
                c.frontier,
                c.yield_per_kframe()
            );
        }
        // Budget classes: where the GA's frames go.
        let budget_frames: u64 = self.budgets.values().map(|c| c.frames).sum::<u64>().max(1);
        let _ = writeln!(s, "budgets:");
        for (class, c) in &self.budgets {
            let _ = writeln!(
                s,
                "  {class:<8} {:>6} tries  {:>5.1}% frames  {:>6.0} frames/try  added {}  front {}  yield {:.2}",
                c.attempts,
                100.0 * c.frames as f64 / budget_frames as f64,
                c.frames as f64 / c.attempts.max(1) as f64,
                c.added,
                c.frontier,
                c.yield_per_kframe()
            );
        }
        // Milestones: the earliest game time for each, and how often it improved.
        let clock = |f: u32| format!("{}:{:04.1}", f / 1800, (f % 1800) as f64 / 30.0);
        let _ = writeln!(
            s,
            "best times: {}",
            self.best_progress.iter().map(|((k, st), (f, n))| format!("k{k}s{st} {} ({n}x better)", clock(*f))).collect::<Vec<_>>().join(", ")
        );
        let _ = writeln!(
            s,
            "first arrivals: {}",
            self.best_zone.iter().map(|((l, a), (f, n))| format!("{l}:{a} {} ({n}x)", clock(*f))).collect::<Vec<_>>().join(", ")
        );
        if self.rewire.queued > 0 {
            let r = &self.rewire;
            let _ = writeln!(
                s,
                "rewire: dominated {}  queued {}  tried {}  improved {}  saved {} frames",
                r.dominated, r.queued, r.attempts, r.improved, r.frames_saved
            );
        }
        // States: concentration and flags.
        let starts: u64 = tree.nodes.iter().map(|n| n.stats.as_start as u64).sum::<u64>().max(1);
        let n = tree.nodes.len() as f64;
        let over = tree
            .nodes
            .iter()
            .filter(|nd| nd.stats.as_start as f64 > 20.0 * starts as f64 / n && nd.stats.as_start >= 10 && nd.stats.added * 10 < nd.stats.as_start)
            .count();
        let never = tree.frontier.iter().filter(|&&i| {
            let st = &tree.nodes[i as usize].stats;
            st.as_b == 0 && st.as_start == 0
        }).count();
        let barren = tree.nodes.iter().filter(|nd| nd.stats.barren >= 10).count();
        let _ = writeln!(
            s,
            "states: OVER-SAMPLED {over}  barren(10+) {barren}  frontier NEVER-SAMPLED {never}/{}",
            tree.frontier.len()
        );
        s
    }
}
