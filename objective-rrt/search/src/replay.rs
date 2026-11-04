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
    /// `OBRRT_GRAFT_BIAS`: the rotation's offset, one step per wave.
    rot: usize,
}

/// `OBRRT_GRAFT_BIAS=<list>` (unset: the old order, see the module doc).
/// Set, the cold jobs (every kind: reroute grafts of each kind, zone-share
/// jobs) are grouped by (kind, destination zone) and taken round-robin over
/// the groups, the starting group rotating each wave, so no group (the
/// castle hub's many legs) starves the rest. Within a group: near the route
/// first, then by the product of the listed weights, then newest start:
///
/// * `none`: no weights (plain rotation);
/// * `len`: 1 + ln(1 + frames) of the job (longer paths cost the search
///   more to find again);
/// * `prec`: ×3 when the edge's objective is `Precision::Precise`;
/// * `branch`: 1 + ln(1 + children) of the donor edge's node (a hub);
/// * `star`: ×4 when the donor lies on a path that reached more stars;
/// * `retry`: 1 + ln(1 + attempts per state added) at the donor's parent
///   (`as_start / (added + 1)`: how hard edges from there were to find);
/// * `useful`: 1 + (added + successes) / (as_start + 1) of the donor (its
///   yield as a start) + 1 when it or a near ancestor is a route node;
/// * `all`: every weight.
///
/// Comma lists combine (`len,prec`).
#[derive(Default, Clone, Copy, Debug)]
pub struct Bias {
    pub len: bool,
    pub prec: bool,
    pub branch: bool,
    pub star: bool,
    pub retry: bool,
    pub useful: bool,
}

pub fn graft_bias() -> Option<&'static Bias> {
    static B: std::sync::OnceLock<Option<Bias>> = std::sync::OnceLock::new();
    B.get_or_init(|| {
        let v = std::env::var("OBRRT_GRAFT_BIAS").ok().filter(|v| !v.is_empty())?;
        let mut b = Bias::default();
        for w in v.split(',') {
            match w.trim() {
                "none" => {}
                "len" => b.len = true,
                "prec" => b.prec = true,
                "branch" => b.branch = true,
                "star" => b.star = true,
                "retry" => b.retry = true,
                "useful" => b.useful = true,
                "all" => b = Bias { len: true, prec: true, branch: true, star: true, retry: true, useful: true },
                x => panic!("OBRRT_GRAFT_BIAS: unknown {x:?} (none, len, prec, branch, star, retry, useful, all)"),
            }
        }
        Some(b)
    })
    .as_ref()
}

/// The bias weight of a job replaying `donor`'s edge(s) (`frames` long).
fn weight(b: &Bias, tree: &Tree, rr: Option<&Reroute>, hot: &HashSet<u32>, donor: u32, frames: u64) -> f64 {
    let n = &tree.nodes[donor as usize];
    let mut w = 1.0;
    if b.len {
        w *= 1.0 + (1.0 + frames as f64).ln();
    }
    if b.prec && n.made_by.as_ref().is_some_and(|o| o.precision() == obrrt_core::Precision::Precise) {
        w *= 3.0;
    }
    if b.branch {
        w *= 1.0 + (1.0 + tree.children[donor as usize].len() as f64).ln();
    }
    if b.star && rr.is_some_and(|r| r.star_path.contains(&donor)) {
        w *= 4.0;
    }
    if b.retry {
        if let Some(p) = n.parent {
            let ps = &tree.nodes[p as usize].stats;
            w *= 1.0 + (1.0 + ps.as_start as f64 / (ps.added as f64 + 1.0)).ln();
        }
    }
    if b.useful {
        let s = &n.stats;
        w *= 1.0 + (s.added + s.successes) as f64 / (s.as_start as f64 + 1.0) + near(tree, hot, donor) as u8 as f64;
    }
    w
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
        ReplayBudget { fraction, stats: BudgetStats::default(), curve: Vec::new(), rot: 0 }
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
        if let Some(b) = graft_bias() {
            cands = self.rotate(b, cands, rr.as_deref(), zs.as_deref(), tree, hot);
        }
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

    /// `OBRRT_GRAFT_BIAS`: `cands` (in the old order) regrouped by (kind,
    /// destination zone), each group by near, weight, the old order; then
    /// round-robin over the groups from a rotating start.
    fn rotate(
        &mut self,
        b: &Bias,
        cands: Vec<(bool, u32, bool, usize)>,
        rr: Option<&Reroute>,
        zs: Option<&ZoneShare>,
        tree: &Tree,
        hot: &HashSet<u32>,
    ) -> Vec<(bool, u32, bool, usize)> {
        let mut groups: std::collections::BTreeMap<(u8, u16, u8), Vec<(bool, f64, usize, (bool, u32, bool, usize))>> = Default::default();
        for (rank, &c) in cands.iter().enumerate() {
            let (is_near, _, is_share, i) = c;
            let (kind, donor, frames) = if is_share {
                let j = zs.unwrap().cold[i];
                (crate::reroute::KINDS as u8, j.donor, share_cost(tree, zs.unwrap(), &j))
            } else {
                let g = rr.unwrap().cold[i];
                (g.kind, g.child, graft_cost(tree, &g))
            };
            let z = tree.nodes[donor as usize].st.zone();
            let w = weight(b, tree, rr, hot, donor, frames);
            groups.entry((kind, z.0, z.1)).or_default().push((is_near, w, rank, c));
        }
        let mut gs: Vec<Vec<(bool, f64, usize, (bool, u32, bool, usize))>> = groups.into_values().collect();
        for g in &mut gs {
            g.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
        }
        let n = gs.len();
        let off = if n > 0 { self.rot % n } else { 0 };
        self.rot = self.rot.wrapping_add(1);
        let mut out = Vec::with_capacity(cands.len());
        let mut r = 0;
        while out.len() < cands.len() {
            for k in 0..n {
                if let Some(e) = gs[(off + k) % n].get(r) {
                    out.push(e.3);
                }
            }
            r += 1;
        }
        out
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
