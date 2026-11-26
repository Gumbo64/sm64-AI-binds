//! Replay budget (`OBRRT_REPLAY_BUDGET=<fraction>`, off by default).
//!
//! Off, reroute grafts and zone-share jobs each take a plan slot (a share
//! of `round_plans`), although a job replays a few hundred frames and a GA
//! attempt tens of thousands; and a job whose start was retired before it
//! ran is skipped as stale. On, plans get every slot, and after the round's
//! plans are merged replay jobs run in parallel waves (each merged in queue
//! order, as `Engine::run_replays` does) until their frames reach
//! `fraction` × the frames the round's plans used.
//!
//! Jobs are chosen when a wave is formed, in this order: route (`hot`)
//! grafts, route share jobs, this round's background share jobs (all FIFO),
//! then the cold queues of both: jobs whose start is on or near the route
//! (it or an ancestor within `NEAR` generations is a route node) first, then
//! the rest, newest start first (LIFO: the faster states just found are the
//! ones worth sharing while they are still relevant). A full cold queue
//! drops its oldest job (`lifo` on `Reroute`/`ZoneShare`).
//!
//! A retired start is not stale by itself: with sparse saves
//! (`OBRRT_REPLAY_MAX`) it is rebuilt from its anchor (`Tree::load_into`)
//! while that holds a save. Jobs whose start can't be loaded any more are
//! skipped (and purged from the queues each wave), as are jobs with nothing
//! to replay (an empty edge chain). Everything depends only on merge order
//! and the tree, so a seed gives one run whatever the thread count.

use std::collections::{HashSet, VecDeque};

use crate::reroute::{Graft, Reroute};
use crate::share::{ShareJob, ZoneShare};
use crate::tree::Tree;

/// Generations above a start searched for a route node ("near the route").
const NEAR: u32 = 3;
/// Waves a round at most (cascading grafts queue more after each).
pub const MAX_WAVES: usize = 8;
/// Rounds kept in the per-round record (summary.json), the first ones.
const CURVE_MAX: usize = 4000;

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub struct KindStats {
    pub taken: u64,
    pub taken_route: u64,
    pub taken_near: u64,
    pub taken_cold: u64,
    /// Skipped: the start can't be loaded (retired, no live anchor).
    pub stale: u64,
    /// Taken with a retired start, rebuilt from its anchor.
    pub rebuilt: u64,
    /// Skipped: nothing to replay (the donor's edge chain is empty).
    pub gone: u64,
    /// Estimated frames of the jobs taken (edges + rebuild).
    pub est_frames: u64,
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub struct BudgetStats {
    pub rounds: u64,
    pub waves: u64,
    /// Frames the rounds' plans used (attempts and their start rebuilds).
    pub plan_frames: u64,
    /// Frames replay jobs used (replays and their start rebuilds).
    pub replay_frames: u64,
    pub budget_frames: u64,
    pub last_plan_frames: u64,
    pub last_replay_frames: u64,
    pub last_waves: u64,
    pub max_queue: u64,
    pub grafts: KindStats,
    pub shares: KindStats,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct RoundRec {
    pub round: u64,
    pub plan_frames: u64,
    pub replay_frames: u64,
    pub waves: u32,
    pub grafts: u32,
    pub shares: u32,
    pub queue: u64,
}

pub struct ReplayBudget {
    pub fraction: f64,
    pub stats: BudgetStats,
    pub curve: Vec<RoundRec>,
}

/// Whether `from` can be loaded: Some(retired) when it can (a retired start
/// is rebuilt from its anchor), None when not.
fn loadable(tree: &Tree, from: u32) -> Option<bool> {
    let n = &tree.nodes[from as usize];
    if from == 0 {
        return Some(false);
    }
    if !n.retired || tree.is_pinned(from) {
        return tree.can_anchor(from).then_some(false);
    }
    (tree.sparse.is_some() && tree.can_anchor(from)).then_some(true)
}

fn near(tree: &Tree, hot: &HashSet<u32>, from: u32) -> bool {
    let mut at = from;
    for _ in 0..=NEAR {
        if hot.contains(&at) {
            return true;
        }
        match tree.nodes[at as usize].parent {
            Some(p) => at = p,
            None => return false,
        }
    }
    false
}

fn graft_cost(tree: &Tree, g: &Graft) -> u64 {
    tree.nodes[g.child as usize].inputs.len() as u64
}

fn share_cost(tree: &Tree, zs: &ZoneShare, j: &ShareJob) -> u64 {
    zs.path(tree, j.donor).iter().map(|&e| tree.nodes[e as usize].inputs.len() as u64).sum()
}

impl ReplayBudget {
    pub fn new(fraction: f64) -> ReplayBudget {
        ReplayBudget { fraction, stats: BudgetStats::default(), curve: Vec::new() }
    }

    /// Jobs for one wave, their estimated frames at least `budget` (the
    /// last one may cross it) or every runnable one queued.
    pub fn select(
        &mut self,
        mut rr: Option<&mut Reroute>,
        mut zs: Option<&mut ZoneShare>,
        tree: &Tree,
        hot: &HashSet<u32>,
        budget: u64,
    ) -> (Vec<Graft>, Vec<ShareJob>) {
        let (mut grafts, mut shares) = (Vec::new(), Vec::new());
        let mut spent = 0u64;
        let st = &mut self.stats;
        // Purge what can never run again (retirement and freed saves are final).
        if let Some(rr) = rr.as_deref_mut() {
            for q in [&mut rr.hot, &mut rr.cold] {
                q.retain(|g| {
                    let ok = loadable(tree, g.from).is_some();
                    st.grafts.stale += !ok as u64;
                    ok
                });
            }
        }
        if let Some(zs) = zs.as_deref_mut() {
            for q in [&mut zs.hot, &mut zs.bgq, &mut zs.cold] {
                q.retain(|j| {
                    let ok = loadable(tree, j.from).is_some();
                    st.shares.stale += !ok as u64;
                    ok
                });
            }
        }
        // Route ones, FIFO.
        if let Some(rr) = rr.as_deref_mut() {
            while spent < budget {
                let Some(g) = rr.hot.pop_front() else { break };
                if let Some(c) = take_graft(st, tree, &g, 0) {
                    spent += c;
                    grafts.push(g);
                }
            }
        }
        if let Some(zs) = zs.as_deref_mut() {
            for bg in [false, true] {
                while spent < budget {
                    let q = if bg { &mut zs.bgq } else { &mut zs.hot };
                    let Some(j) = q.pop_front() else { break };
                    if let Some(c) = take_share(st, tree, zs, &j, if bg { 2 } else { 0 }) {
                        spent += c;
                        shares.push(j);
                    }
                }
            }
        }
        if spent >= budget {
            return (grafts, shares);
        }
        // Cold: near the route first, then newest start first; grafts before
        // share jobs on a tie, then newest queued first.
        let mut cands: Vec<(bool, u32, bool, usize)> = Vec::new();
        if let Some(rr) = rr.as_deref() {
            cands.extend(rr.cold.iter().enumerate().map(|(i, g)| (near(tree, hot, g.from), g.from, false, i)));
        }
        if let Some(zs) = zs.as_deref() {
            cands.extend(zs.cold.iter().enumerate().map(|(i, j)| (near(tree, hot, j.from), j.from, true, i)));
        }
        cands.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)).then(a.2.cmp(&b.2)).then(b.3.cmp(&a.3)));
        let st = &mut self.stats;
        let (mut gone_g, mut gone_s) = (Vec::new(), Vec::new());
        for &(is_near, _, is_share, i) in &cands {
            if spent >= budget {
                break;
            }
            let class = if is_near { 1 } else { 2 };
            if is_share {
                let zs = zs.as_deref().unwrap();
                let j = zs.cold[i];
                if let Some(c) = take_share(st, tree, zs, &j, class) {
                    spent += c;
                    shares.push(j);
                }
                gone_s.push(i);
            } else {
                let g = rr.as_deref().unwrap().cold[i];
                if let Some(c) = take_graft(st, tree, &g, class) {
                    spent += c;
                    grafts.push(g);
                }
                gone_g.push(i);
            }
        }
        if let Some(rr) = rr {
            remove_at(&mut rr.cold, gone_g);
        }
        if let Some(zs) = zs {
            remove_at(&mut zs.cold, gone_s);
        }
        (grafts, shares)
    }

    /// Records a finished round.
    pub fn end_round(&mut self, round: u64, plan_frames: u64, replay_frames: u64, waves: u32, grafts: u32, shares: u32, queue: u64) {
        let s = &mut self.stats;
        s.rounds += 1;
        s.plan_frames += plan_frames;
        s.replay_frames += replay_frames;
        s.budget_frames += (plan_frames as f64 * self.fraction) as u64;
        s.last_plan_frames = plan_frames;
        s.last_replay_frames = replay_frames;
        s.last_waves = waves as u64;
        s.waves += waves as u64;
        s.max_queue = s.max_queue.max(queue);
        if self.curve.len() < CURVE_MAX {
            self.curve.push(RoundRec { round, plan_frames, replay_frames, waves, grafts, shares, queue });
        }
    }

    pub fn summary(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self.stats).unwrap();
        v["fraction"] = self.fraction.into();
        v["curve"] = serde_json::to_value(&self.curve).unwrap();
        v
    }

    pub fn line(&self) -> String {
        let s = &self.stats;
        format!(
            "replay budget {}: {} rounds, {} waves; plans {} frames, replays {} frames ({:.1}% of plans; budget {}); last round {} / {} frames, {} waves; max queue {}",
            self.fraction,
            s.rounds,
            s.waves,
            s.plan_frames,
            s.replay_frames,
            100.0 * s.replay_frames as f64 / s.plan_frames.max(1) as f64,
            s.budget_frames,
            s.last_plan_frames,
            s.last_replay_frames,
            s.last_waves,
            s.max_queue
        )
    }

    /// The per-kind suffix for the reroute:/zone share: lines.
    pub fn kind_line(k: &KindStats) -> String {
        format!(
            "budget: {} taken ({} route, {} near, {} cold), {} stale skipped, {} rebuilt from anchor, {} empty, {} est frames",
            k.taken, k.taken_route, k.taken_near, k.taken_cold, k.stale, k.rebuilt, k.gone, k.est_frames
        )
    }
}

fn count(k: &mut KindStats, rebuilt: bool, class: u8, cost: u64) {
    k.taken += 1;
    k.rebuilt += rebuilt as u64;
    k.est_frames += cost;
    match class {
        0 => k.taken_route += 1,
        1 => k.taken_near += 1,
        _ => k.taken_cold += 1,
    }
}

/// `class`: 0 route, 1 near the route, 2 cold (background share jobs count as cold).
fn take_graft(st: &mut BudgetStats, tree: &Tree, g: &Graft, class: u8) -> Option<u64> {
    let Some(rebuilt) = loadable(tree, g.from) else {
        st.grafts.stale += 1;
        return None;
    };
    let n = graft_cost(tree, g);
    if n == 0 {
        st.grafts.gone += 1;
        return None;
    }
    let c = n + tree.dist(g.from) as u64;
    count(&mut st.grafts, rebuilt, class, c);
    Some(c)
}

fn take_share(st: &mut BudgetStats, tree: &Tree, zs: &ZoneShare, j: &ShareJob, class: u8) -> Option<u64> {
    let Some(rebuilt) = loadable(tree, j.from) else {
        st.shares.stale += 1;
        return None;
    };
    let n = share_cost(tree, zs, j);
    if n == 0 {
        st.shares.gone += 1;
        return None;
    }
    let c = n + tree.dist(j.from) as u64;
    count(&mut st.shares, rebuilt, class, c);
    Some(c)
}

fn remove_at<T>(q: &mut VecDeque<T>, mut idx: Vec<usize>) {
    if idx.is_empty() {
        return;
    }
    idx.sort_unstable();
    let mut k = 0;
    let mut i = 0;
    q.retain(|_| {
        let drop = k < idx.len() && idx[k] == i;
        k += drop as usize;
        i += 1;
        !drop
    });
}
