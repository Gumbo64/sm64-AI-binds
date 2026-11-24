//! Route polish (`OBRRT_POLISH=<share>`, off by default): speeding up the
//! best route's earlier legs after later stars have been found.
//!
//! Once the frontier gains a star, every state with fewer stars is retired,
//! so nothing ever attempts an earlier leg again and each star keeps the time
//! it had when it was first the frontier (the 12 G-frame indefinite run of
//! 2026-10-29 kept stars 1-7 at their 500 M times). Polish re-attacks them:
//!
//! * The best route (soonest state of the best progress, back to the root)
//!   is pinned (`Tree::pinned`): retiring it keeps its saves and objects, so
//!   its states stay loadable starts for polish, never for ordinary plans.
//! * Each round, up to `share` of the plans re-attempt a route milestone (a
//!   route node whose objective was met there) from a route state a few
//!   nodes before it, with the GA seeded by the route's own inputs.
//! * A result of the same progress set at least the margin sooner is pinned
//!   and grafted forward: the rest of the route is replayed onto it edge by
//!   edge (`reroute::KIND_ROUTE`, accepted like a leg edge: same movement key,
//!   no longer than the original edge). A finished chain is a new soonest
//!   state of the best progress, so the route moves onto it.
//! * Replays drift in courses (objects and RNG see other timers), so an edge
//!   that diverges becomes a *repair*: a seeded attempt at the next route
//!   milestone from the chain's last kept node; its sooner result continues
//!   the chain the same way. Repairs go before new polish plans.
//!
//! Pins of nodes that are neither on the current route nor recent
//! (`keep_rounds`) are released each round.

use std::collections::{HashMap, VecDeque};

use crate::tree::Tree;

/// A polish or repair job: attempt route node `target`'s objective from
/// `from`, seeded with the route's inputs from `anchor` (the route state
/// `from` stands for) down to `target`; `end` is the route's end then.
#[derive(Clone, Copy, Debug)]
pub struct Job {
    pub from: u32,
    pub anchor: u32,
    pub target: u32,
    pub end: u32,
    pub repair: bool,
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct PolishStats {
    pub plans: u64,
    pub repairs: u64,
    /// Results: same progress set and sooner (frames saved summed), or not.
    pub sooner: u64,
    pub sooner_frames: u64,
    pub not_sooner: u64,
    pub other_set: u64,
    pub failed: u64,
    /// Route-graft chains: started, edges kept, diverged edges (each queues
    /// a repair), chains that reached the route's end (frames saved there).
    pub chains: u64,
    pub edges_kept: u64,
    pub diverged: u64,
    pub done: u64,
    pub done_frames: u64,
    pub repairs_dropped: u64,
    /// Sooner results by how they continued: the target's movement key
    /// (chain), another (repair), or neither (no route child / no milestone
    /// below the target).
    pub same_key: u64,
    pub other_key: u64,
    pub no_child: u64,
    pub no_next: u64,
    pub no_target: u64,
    pub pinned_peak: usize,
}

pub struct Polish {
    pub share: f64,
    pub keep_rounds: u64,
    /// Most route nodes a polish start is above its target.
    pub back: usize,
    pub margin: u32,
    pub repairs: VecDeque<Job>,
    /// Attempts per target since its last improvement.
    pub tries: HashMap<u32, u32>,
    /// Soonest polish or repair result per target so far: a later result
    /// must beat it, not just the route node.
    pub best: HashMap<u32, u32>,
    /// Round each pinned node was last pinned or confirmed on the route.
    pub pin_round: HashMap<u32, u64>,
    /// The best route this round, root first, and its end.
    pub route: Vec<u32>,
    pub end: u32,
    /// The engine's round at the last refresh.
    pub round: u64,
    pub stats: PolishStats,
}

const REPAIR_CAP: usize = 512;

/// A route milestone: a state made by a non-modal objective that it met, or
/// where stars or keys went up (often met in passing, by another objective's
/// attempt, and modal then: the star dance), or the route's end.
fn milestone(tree: &Tree, i: u32, end: u32) -> bool {
    let n = &tree.nodes[i as usize];
    if !n.made_by.as_ref().is_some_and(|m| !m.modal()) {
        return false;
    }
    let up = n.parent.is_some_and(|p| {
        let q = &tree.nodes[p as usize].st;
        n.st.stars > q.stars || n.st.keys > q.keys
    });
    n.met || up || i == end
}

impl Polish {
    pub fn new(share: f64) -> Polish {
        let env = |k: &str, d: u64| obrrt_core::sys::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        Polish {
            share,
            keep_rounds: env("OBRRT_POLISH_KEEP", 60),
            back: env("OBRRT_POLISH_BACK", 8) as usize,
            margin: env("OBRRT_POLISH_MARGIN", 10) as u32,
            repairs: VecDeque::new(),
            tries: HashMap::new(),
            best: HashMap::new(),
            pin_round: HashMap::new(),
            route: Vec::new(),
            end: 0,
            round: 0,
            stats: PolishStats::default(),
        }
    }

    /// The best route (soonest frontier state of the most stars, then keys),
    /// pinned; stale pins released.
    pub fn refresh(&mut self, tree: &mut Tree, round: u64) {
        self.round = round;
        // Pinned by the tree itself (route ancestors at a frontier advance).
        let mut fresh: Vec<u32> = tree.pinned.iter().copied().filter(|i| !self.pin_round.contains_key(i)).collect();
        fresh.sort_unstable();
        for i in fresh {
            self.pin_round.insert(i, round);
        }
        let best = tree
            .frontier
            .iter()
            .copied()
            .min_by_key(|&i| {
                let st = &tree.nodes[i as usize].st;
                (std::cmp::Reverse((st.stars, st.keys)), st.frame, i)
            })
            .unwrap_or(0);
        self.end = best;
        self.route.clear();
        let mut at = Some(best);
        while let Some(i) = at {
            self.route.push(i);
            at = tree.nodes[i as usize].parent;
        }
        self.route.reverse();
        for &i in &self.route {
            tree.pinned.insert(i);
            self.pin_round.insert(i, round);
        }
        let keep = self.keep_rounds;
        let mut stale: Vec<u32> = self.pin_round.iter().filter(|e| *e.1 + keep < round).map(|e| *e.0).collect();
        stale.sort_unstable();
        for i in stale {
            self.pin_round.remove(&i);
            tree.unpin(i);
        }
        self.stats.pinned_peak = self.stats.pinned_peak.max(tree.pinned.len());
    }

    /// Notes a node pinned now (a polish result or chain edge).
    pub fn pinned(&mut self, id: u32) {
        self.pin_round.insert(id, self.round);
    }

    /// The route milestones (route indices whose node met its objective).
    pub fn milestones(&self, tree: &Tree) -> Vec<usize> {
        (1..self.route.len()).filter(|&j| milestone(tree, self.route[j], self.end)).collect()
    }

    /// The route's inputs from node `anchor` (exclusive) down to `target`
    /// (inclusive), walking up from `target`; None if `anchor` isn't above it.
    pub fn inputs(tree: &Tree, anchor: u32, target: u32) -> Option<Vec<obrrt_core::GamePad>> {
        let mut path = Vec::new();
        let mut at = target;
        while at != anchor {
            path.push(at);
            at = tree.nodes[at as usize].parent?;
        }
        let mut out = Vec::new();
        for &i in path.iter().rev() {
            out.extend_from_slice(&tree.nodes[i as usize].inputs);
        }
        Some(out)
    }

    /// The next milestone at or below `from` on the path down to `end`
    /// (a node that met its objective), if any.
    pub fn next_milestone(tree: &Tree, from: u32, end: u32) -> Option<u32> {
        let mut path = Vec::new();
        let mut at = end;
        loop {
            path.push(at);
            if at == from {
                break;
            }
            at = tree.nodes[at as usize].parent?;
        }
        path.into_iter().rev().find(|&i| milestone(tree, i, end))
    }

    pub fn line(&self, tree: &Tree) -> String {
        let s = &self.stats;
        format!(
            "   polish: plans {} (repairs {}, queued {}), sooner {} ({} frames), not sooner {}, other set {}, failed {}; same key {} other key {} no child {} no next {}; chains {} edges kept {} diverged {} done {} ({} frames saved at the end), repairs dropped {}, no target {}; pinned {} (peak {})",
            s.plans, s.repairs, self.repairs.len(), s.sooner, s.sooner_frames, s.not_sooner, s.other_set, s.failed, s.same_key, s.other_key, s.no_child, s.no_next, s.chains, s.edges_kept, s.diverged, s.done, s.done_frames, s.repairs_dropped, s.no_target, tree.pinned.len(), s.pinned_peak
        )
    }

    pub fn push_repair(&mut self, j: Job) {
        if self.repairs.len() >= REPAIR_CAP {
            self.stats.repairs_dropped += 1;
            return;
        }
        self.repairs.push_back(j);
    }
}
