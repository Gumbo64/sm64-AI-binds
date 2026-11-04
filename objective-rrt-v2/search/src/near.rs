//! Seeds from nearby starts (`OBRRT_NEAR_SEED=1`, off by default): when
//! objective O is attempted from start S, the GA's first generation is
//! seeded (as backchain seeds it, `Worker::attempt_seeded`) with the recorded
//! inputs of one of O's past attempts that started near S: same zone,
//! straight-line distance under `OBRRT_NEAR_R` (300), the same action group,
//! not S itself. Successes first (the one that took the fewest frames), else
//! the kept end that came closest by O's heuristic; ties by the nearest
//! start. A plan is seeded with probability `OBRRT_NEAR_P` (0.5). Generic: no
//! game knowledge beyond positions and the action group.

use std::collections::HashMap;
use std::sync::OnceLock;

use obrrt_core::{GamePad, St};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::tree::Tree;

#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub radius: f32,
    pub p: f64,
    /// Records kept per objective key.
    pub per_key: usize,
}

static SETTINGS: OnceLock<Option<Settings>> = OnceLock::new();

pub fn configure(s: Option<Settings>) {
    let _ = SETTINGS.set(s);
}

pub fn settings() -> Option<Settings> {
    *SETTINGS.get_or_init(|| None)
}

/// One past attempt of an objective: start node, kept end node (a child of
/// the start), whether it met the objective, and its rank (lower is better:
/// frames taken for successes, the end's heuristic otherwise).
#[derive(Clone, Copy)]
struct Rec {
    start: u32,
    end: u32,
    success: bool,
    rank: f64,
}

#[derive(Default, serde::Serialize, Clone)]
pub struct Stats {
    pub recorded: u64,
    pub asked: u64,
    pub no_record: u64,
    pub skipped_p: u64,
    pub seeded: u64,
    pub seeded_success: u64,
    /// Seeded attempts whose seed itself met every stage, and successes.
    pub seed_met: u64,
    pub successes: u64,
}

pub struct NearSeed {
    pub on: Option<Settings>,
    recs: HashMap<String, Vec<Rec>>,
    rng: SmallRng,
    pub stats: Stats,
}

fn group(st: &St) -> u32 {
    st.action & obrrt_core::layout::ACT_GROUP_MASK
}

impl NearSeed {
    pub fn new(seed: u64) -> NearSeed {
        NearSeed { on: settings(), recs: HashMap::new(), rng: SmallRng::seed_from_u64(seed ^ 0x6e65_6172_7365_6564), stats: Stats::default() }
    }

    /// On merging the first segment of an attempt of `key` from `start`
    /// whose end was kept as `end` (a child of `start`).
    pub fn note(&mut self, key: String, start: u32, end: u32, success: bool, rank: f64) {
        let Some(s) = self.on else { return };
        self.stats.recorded += 1;
        let v = self.recs.entry(key).or_default();
        v.push(Rec { start, end, success, rank });
        if v.len() > s.per_key {
            // Drop the worst: failures before successes, then the highest rank.
            let (i, _) = v
                .iter()
                .enumerate()
                .max_by(|a, b| (!a.1.success, a.1.rank).partial_cmp(&(!b.1.success, b.1.rank)).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            v.remove(i);
        }
    }

    /// A seed for attempting `key` from `start`, if a nearby record exists.
    pub fn pick(&mut self, tree: &Tree, key: &str, start: u32) -> Option<Vec<GamePad>> {
        let s = self.on?;
        self.stats.asked += 1;
        let st = &tree.nodes[start as usize].st;
        let best = self.recs.get(key).and_then(|v| {
            v.iter()
                .filter(|r| r.start != start && tree.nodes[r.end as usize].parent == Some(r.start))
                .filter_map(|r| {
                    let s0 = &tree.nodes[r.start as usize].st;
                    let d = obrrt_core::dist(s0.pos, st.pos);
                    (s0.zone() == st.zone() && d < s.radius && group(s0) == group(st)).then_some((r, d))
                })
                .min_by(|a, b| (!a.0.success, a.0.rank, a.1).partial_cmp(&(!b.0.success, b.0.rank, b.1)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(r, _)| *r)
        });
        let Some(r) = best else {
            self.stats.no_record += 1;
            return None;
        };
        if !self.rng.random_bool(s.p) {
            self.stats.skipped_p += 1;
            return None;
        }
        self.stats.seeded += 1;
        self.stats.seeded_success += r.success as u64;
        Some(tree.nodes[r.end as usize].inputs.clone())
    }
}
