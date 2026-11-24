//! Zone-entry input sharing (`OBRRT_ZONE_SHARE=1`, off by default), after
//! objective-scattershot's share.rs: the inputs one state took since it
//! entered its zone (level, area) are replayed from ANOTHER state that
//! entered the same zone, and whatever new or sooner states the replay
//! passes through are kept. The replayed states need not match the donor's
//! ("at least attempt the moves").
//!
//! A node's zone entry is itself when its zone differs from its parent's
//! (or it is the root), else its parent's entry; so every node between an
//! entry and a node with that entry is in the zone (at node boundaries). A
//! job replays, from entry `from`, the edges from the donor's entry down to
//! the donor, recording the state at each edge boundary; the engine adds
//! them as a chain under `from` (see `Engine::merge_share`).
//!
//! Jobs come from two places:
//! 1. Zone entries: each node added whose zone differs from its parent's
//!    gets jobs replaying the paths of the zone's route donors (the deepest
//!    best-route (`hot`) nodes of each pass through the zone) and of its top
//!    K donors (the best node of each other entry by progress gained since
//!    the entry, then straight-line distance from the entry's position;
//!    the K best entries).
//! 2. Background (like scattershot): each round a few jobs from a donor
//!    drawn as the furthest into its zone (frames since entry) of 8 random
//!    nodes, onto a random other live entry of its zone.
//!
//! `OBRRT_ZONE_SHARE_DONOR=sample` replaces 1. (background stays): per
//! zone, each milestone reached from an entry of it (a star or save-flag
//! bit gained, e.g. a key; leaving it for a given zone) keeps only its
//! SHORTEST known path (the node reaching it soonest after its entry). A
//! new entry queues `OBRRT_ZONE_SHARE_M` (4) jobs whose donors are cut
//! points along those paths: a random milestone (stars/flags twice as
//! likely as exits), then its path's deepest node within t frames of the
//! entry, t log-uniform in [30, cap] (so most are a few hundred frames and
//! a path shorter than t is taken whole). Hot when the new entry is on the
//! best route, else cold. `OBRRT_ZONE_SHARE_STATS=1` (or sample) adds the
//! queued jobs' frames histogram to the summary and line.
//!
//! Every path is at most `cap` frames: a donor further in is replaced by its
//! deepest ancestor within the cap. (from, donor) pairs are tried once.
//! Nodes are noted in id order at the start of each round, queues are FIFO
//! and the background draws use their own seeded RNG, so a seed gives one
//! run whatever the thread count.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::OnceLock;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::tree::Tree;

/// Replay donor `donor`'s since-entry path from entry `from`.
#[derive(Clone, Copy, Debug)]
pub struct ShareJob {
    pub from: u32,
    pub donor: u32,
}

#[derive(Default, Clone, Copy, Debug, serde::Serialize)]
pub struct ShareStats {
    /// Zone entries noted (nodes whose zone differs from their parent's).
    pub entries: u64,
    pub queued_route: u64,
    pub queued_top: u64,
    pub queued_bg: u64,
    pub dropped: u64,
    /// Jobs taken whose start had been retired by the time they ran.
    pub stale: u64,
    pub run: u64,
    pub frames: u64,
    /// Edge boundaries replayed (states recorded).
    pub boundaries: u64,
    pub added: u64,
    /// Added where the duplicate cell already had (only later) states.
    pub sooner: u64,
    /// Added in the same cell as the donor's own state for that edge, sooner.
    pub matched_sooner: u64,
    pub matched_frames_saved: u64,
    /// Boundaries not added (an earlier state at their place); their inputs
    /// carry on to the next added state.
    pub duplicate: u64,
    pub died: u64,
    pub left_zone: u64,
    pub traps: u64,
    pub new_zones: u64,
    /// Jobs that added at least one state.
    pub jobs_ok: u64,
    pub max_chain: u64,
}

pub struct ZoneShare {
    pub k: usize,
    pub cap: u32,
    pub bg: usize,
    /// Per node: its zone entry, by node id; nodes below `entry.len()` are noted.
    pub entry: Vec<u32>,
    /// Per zone: its entry nodes, ascending.
    entries: BTreeMap<(u16, u8), Vec<u32>>,
    /// Per zone, per entry: its best donor's (score, node).
    best: BTreeMap<(u16, u8), BTreeMap<u32, ((i32, u32), u32)>>,
    pub(crate) hot: VecDeque<ShareJob>,
    pub(crate) bgq: VecDeque<ShareJob>,
    pub(crate) cold: VecDeque<ShareJob>,
    tried: HashSet<(u32, u32)>,
    rng: SmallRng,
    pub stats: ShareStats,
    /// The last few added nodes (for checking them by replay).
    pub recent: VecDeque<u32>,
    /// Replay budget (`crate::replay`): a full cold queue drops its oldest
    /// job for a new one instead of the new one.
    pub lifo: bool,
    /// OBRRT_ZONE_SHARE_DONOR=sample: jobs per new entry.
    pub sample_m: Option<usize>,
    /// Per zone, per milestone: its shortest path's (frames since entry, node).
    milestones: BTreeMap<(u16, u8), BTreeMap<u32, (u32, u32)>>,
    pub extra: SampleStats,
}

/// `OBRRT_ZONE_SHARE_DONOR=sample`: Some(M) (OBRRT_ZONE_SHARE_M, 4).
fn donor_sample() -> Option<usize> {
    static M: OnceLock<Option<usize>> = OnceLock::new();
    *M.get_or_init(|| {
        (obrrt_core::sys::env::var("OBRRT_ZONE_SHARE_DONOR").is_ok_and(|v| v == "sample"))
            .then(|| obrrt_core::sys::env::var("OBRRT_ZONE_SHARE_M").ok().and_then(|v| v.parse().ok()).unwrap_or(4))
    })
}

fn extra_stats() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| donor_sample().is_some() || obrrt_core::sys::env::var("OBRRT_ZONE_SHARE_STATS").is_ok_and(|v| v == "1"))
}

/// Upper edges of the queued-path frames histogram (last bucket: above).
const HIST: [u32; 7] = [50, 100, 200, 400, 800, 1600, 3000];

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct SampleStats {
    pub queued_sample: u64,
    /// Sample draws that found no new (entry, donor) pair.
    pub sample_miss: u64,
    /// Milestones (zone, what) with a shortest path.
    pub milestones: u64,
    /// Frames since entry of queued jobs' donors, by source, over HIST.
    pub hist_route: [u64; 8],
    pub hist_top: [u64; 8],
    pub hist_sample: [u64; 8],
    pub hist_bg: [u64; 8],
    pub frames_route: u64,
    pub frames_top: u64,
    pub frames_sample: u64,
    pub frames_bg: u64,
}

fn bucket(f: u32) -> usize {
    HIST.iter().position(|&h| f <= h).unwrap_or(HIST.len())
}

const COLD_CAP: usize = 50_000;
const RECENT: usize = 64;

impl ZoneShare {
    pub fn new(k: usize, cap: u32, bg: usize, seed: u64) -> ZoneShare {
        ZoneShare {
            k,
            cap,
            bg,
            entry: Vec::new(),
            entries: BTreeMap::new(),
            best: BTreeMap::new(),
            hot: VecDeque::new(),
            bgq: VecDeque::new(),
            cold: VecDeque::new(),
            tried: HashSet::new(),
            rng: SmallRng::seed_from_u64(seed ^ 0x7a6f_6e65_7368_6172),
            stats: ShareStats::default(),
            recent: VecDeque::new(),
            lifo: false,
            sample_m: donor_sample(),
            milestones: BTreeMap::new(),
            extra: SampleStats::default(),
        }
    }

    fn since(&self, tree: &Tree, n: u32) -> u32 {
        let e = self.entry[n as usize];
        tree.nodes[n as usize].st.frame - tree.nodes[e as usize].st.frame
    }

    /// `n` or its deepest ancestor within the cap since their entry; None
    /// when that is the entry itself.
    fn capped(&self, tree: &Tree, mut n: u32) -> Option<u32> {
        let e = self.entry[n as usize];
        while n != e && self.since(tree, n) > self.cap {
            n = tree.nodes[n as usize].parent?;
        }
        (n != e).then_some(n)
    }

    /// The edges (node ids, in order) from `donor`'s entry down to `donor`.
    pub fn path(&self, tree: &Tree, donor: u32) -> Vec<u32> {
        let e = self.entry[donor as usize];
        let mut out = Vec::new();
        let mut at = donor;
        while at != e {
            out.push(at);
            at = tree.nodes[at as usize].parent.expect("an entry above every node");
        }
        out.reverse();
        out
    }

    /// Notes a queued job's donor depth in source `src`'s histogram.
    fn note_hist(&mut self, tree: &Tree, donor: u32, src: u8) {
        if !extra_stats() {
            return;
        }
        let f = self.since(tree, donor);
        let x = &mut self.extra;
        let (h, t) = match src {
            0 => (&mut x.hist_route, &mut x.frames_route),
            1 => (&mut x.hist_top, &mut x.frames_top),
            2 => (&mut x.hist_sample, &mut x.frames_sample),
            _ => (&mut x.hist_bg, &mut x.frames_bg),
        };
        h[bucket(f)] += 1;
        *t += f as u64;
    }

    /// Records `node` (in `zone`, `since` frames after its entry) as
    /// reaching milestone `key` when it is the soonest to.
    fn note_milestone(&mut self, zone: (u16, u8), key: u32, since: u32, node: u32) {
        let m = self.milestones.entry(zone).or_default();
        match m.get_mut(&key) {
            Some(v) => {
                if since < v.0 {
                    *v = (since, node);
                }
            }
            None => {
                m.insert(key, (since, node));
                self.extra.milestones += 1;
            }
        }
    }

    fn push(&mut self, job: ShareJob, hot: bool) -> bool {
        if job.from == job.donor || self.entry[job.donor as usize] == job.from || !self.tried.insert((job.from, job.donor)) {
            return false;
        }
        if hot {
            self.hot.push_back(job);
        } else if self.cold.len() < COLD_CAP {
            self.cold.push_back(job);
        } else if self.lifo {
            self.cold.pop_front();
            self.stats.dropped += 1;
            self.cold.push_back(job);
        } else {
            self.stats.dropped += 1;
            return false;
        }
        true
    }

    /// Notes the nodes added since the last call (in id order) and queues
    /// jobs for the zone entries among them; then draws the background jobs.
    /// `route` is the best routes' node set (`hot`).
    pub fn round(&mut self, tree: &Tree, route: &HashSet<u32>) {
        let first = self.entry.len();
        let mut new_entries = Vec::new();
        for id in first..tree.nodes.len() {
            let n = &tree.nodes[id];
            let zone = n.st.zone();
            let e = match n.parent {
                Some(p) if tree.nodes[p as usize].st.zone() == zone => self.entry[p as usize],
                _ => id as u32,
            };
            self.entry.push(e);
            if self.sample_m.is_some() {
                if let Some(p) = n.parent {
                    let ps = &tree.nodes[p as usize].st;
                    if e == id as u32 {
                        // Leaving p's zone for this one.
                        let pe = self.entry[p as usize];
                        if pe != p {
                            let since = ps.frame - tree.nodes[pe as usize].st.frame;
                            let key = (2 << 24) | ((zone.0 as u32) << 8) | zone.1 as u32;
                            self.note_milestone(ps.zone(), key, since, p);
                        }
                    } else {
                        let since = n.st.frame - tree.nodes[e as usize].st.frame;
                        for (i, (&c, &pc)) in n.st.course_stars.iter().zip(&ps.course_stars).enumerate() {
                            let g = c & !pc;
                            for b in 0..8 {
                                if g >> b & 1 != 0 {
                                    self.note_milestone(zone, (i as u32) * 8 + b, since, id as u32);
                                }
                            }
                        }
                        let g = n.st.save_flags & !ps.save_flags;
                        for b in 0..32 {
                            if g >> b & 1 != 0 {
                                self.note_milestone(zone, (1 << 24) | b, since, id as u32);
                            }
                        }
                    }
                }
            }
            if e == id as u32 {
                self.entries.entry(zone).or_default().push(e);
                if id > 0 {
                    new_entries.push(id as u32);
                }
                continue;
            }
            let since = n.st.frame - tree.nodes[e as usize].st.frame;
            if since == 0 || since > self.cap {
                continue;
            }
            let es = &tree.nodes[e as usize].st;
            let gain = (n.st.keys as i32 - es.keys as i32) + (n.st.stars as i32 - es.stars as i32);
            let d = (0..3).map(|k| (n.st.pos[k] - es.pos[k]).powi(2)).sum::<f32>().sqrt();
            let score = (gain, d.min(1e9) as u32);
            let slot = self.best.entry(zone).or_default().entry(e).or_insert((score, id as u32));
            if score > slot.0 {
                *slot = (score, id as u32);
            }
        }
        if let (Some(m), false) = (self.sample_m, new_entries.is_empty()) {
            for &from in &new_entries {
                self.stats.entries += 1;
                let zone = tree.nodes[from as usize].st.zone();
                let Some(ms) = self.milestones.get(&zone) else { continue };
                let paths: Vec<(u32, u32)> = ms.iter().map(|(&k, &(_, n))| (if k >> 24 == 2 { 1 } else { 2 }, n)).collect();
                let total: u32 = paths.iter().map(|p| p.0).sum();
                let hot = route.contains(&from);
                for _ in 0..m {
                    let mut ok = false;
                    for _ in 0..4 {
                        let mut r = self.rng.random_range(0..total);
                        let mut end = paths[0].1;
                        for &(w, n) in &paths {
                            if r < w {
                                end = n;
                                break;
                            }
                            r -= w;
                        }
                        let (lo, hi) = (30f64, self.cap.max(30) as f64);
                        let t = (lo * (hi / lo).powf(self.rng.random::<f64>())).min(self.cap as f64) as u32;
                        let e = self.entry[end as usize];
                        let mut d = end;
                        while d != e && self.since(tree, d) > t {
                            d = tree.nodes[d as usize].parent.expect("an entry above every node");
                        }
                        if d != e && self.push(ShareJob { from, donor: d }, hot) {
                            self.extra.queued_sample += 1;
                            self.note_hist(tree, d, 2);
                            ok = true;
                            break;
                        }
                    }
                    self.extra.sample_miss += !ok as u64;
                }
            }
        } else if !new_entries.is_empty() {
            // Route donors per zone: the deepest route node of each pass
            // (no route child with the same entry), ids ascending.
            let mut ids: Vec<u32> = route.iter().copied().filter(|&i| (i as usize) < self.entry.len()).collect();
            ids.sort_unstable();
            let mut route_donors: BTreeMap<(u16, u8), Vec<u32>> = BTreeMap::new();
            for &i in &ids {
                let e = self.entry[i as usize];
                if e == i {
                    continue;
                }
                let deeper = tree.children[i as usize].iter().any(|c| route.contains(c) && (*c as usize) < self.entry.len() && self.entry[*c as usize] == e);
                if !deeper {
                    if let Some(d) = self.capped(tree, i) {
                        route_donors.entry(tree.nodes[i as usize].st.zone()).or_default().push(d);
                    }
                }
            }
            for &from in &new_entries {
                self.stats.entries += 1;
                let zone = tree.nodes[from as usize].st.zone();
                for &d in route_donors.get(&zone).map_or(&[][..], |v| v.as_slice()) {
                    if self.push(ShareJob { from, donor: d }, true) {
                        self.stats.queued_route += 1;
                        self.note_hist(tree, d, 0);
                    }
                }
                if let Some(m) = self.best.get(&zone) {
                    let mut top: Vec<((i32, u32), u32)> = m.iter().filter(|&(&e, _)| e != from).map(|(_, &v)| v).collect();
                    top.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
                    for &(_, d) in top.iter().take(self.k) {
                        if self.push(ShareJob { from, donor: d }, route.contains(&from)) {
                            self.stats.queued_top += 1;
                            self.note_hist(tree, d, 1);
                        }
                    }
                }
            }
        }
        // Background: a donor furthest into its zone of 8 random nodes, onto
        // a random other live entry of its zone.
        // Last round's untaken ones are dropped (their pairs stay tried).
        self.stats.dropped += self.bgq.len() as u64;
        self.bgq.clear();
        let n = tree.nodes.len();
        for _ in 0..self.bg {
            if n < 2 {
                break;
            }
            let picks: Vec<u32> = (0..8).map(|_| self.rng.random_range(1..n) as u32).collect();
            let donor = (picks.into_iter())
                .filter_map(|i| self.capped(tree, i))
                .max_by_key(|&d| (self.since(tree, d), std::cmp::Reverse(d)));
            let Some(donor) = donor else { continue };
            let (e, zone) = (self.entry[donor as usize], tree.nodes[donor as usize].st.zone());
            let list = &self.entries[&zone];
            for _ in 0..8 {
                let to = list[self.rng.random_range(0..list.len())];
                if to != e && !tree.nodes[to as usize].retired && !self.tried.contains(&(to, donor)) {
                    self.tried.insert((to, donor));
                    self.bgq.push_back(ShareJob { from: to, donor });
                    self.stats.queued_bg += 1;
                    self.note_hist(tree, donor, 3);
                    break;
                }
            }
        }
    }

    /// Up to `n` jobs: route ones, then this round's background ones, then
    /// the rest.
    /// OBRRT_MEM_DIAG: approximate heap bytes of the big tables.
    pub fn mem_diag(&self) -> String {
        let e: usize = self.entries.values().map(|v| v.capacity() * 4).sum();
        let b: usize = self.best.values().map(|m| m.len()).sum();
        let m: usize = self.milestones.values().map(|m| m.len()).sum();
        format!(
            "zs tried {} (cap {}, {} MB) entry {} entries {} MB best {} milestones {} queues {}/{}/{} caps {}/{}/{} recent {}",
            self.tried.len(), self.tried.capacity(), self.tried.capacity() * 9 / 1_000_000, self.entry.len(), e / 1_000_000, b, m,
            self.hot.len(), self.bgq.len(), self.cold.len(), self.hot.capacity(), self.bgq.capacity(), self.cold.capacity(), self.recent.len()
        )
    }

    pub fn take(&mut self, n: usize) -> Vec<ShareJob> {
        let mut out = Vec::new();
        while out.len() < n {
            match self.hot.pop_front().or_else(|| self.bgq.pop_front()).or_else(|| self.cold.pop_front()) {
                Some(j) => out.push(j),
                None => break,
            }
        }
        out
    }

    pub fn queue_len(&self) -> usize {
        self.hot.len() + self.cold.len()
    }

    pub fn note_added(&mut self, id: u32) {
        self.recent.push_back(id);
        if self.recent.len() > RECENT {
            self.recent.pop_front();
        }
    }

    pub fn summary(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(self.stats).unwrap();
        v["k"] = self.k.into();
        v["cap"] = self.cap.into();
        v["bg"] = self.bg.into();
        v["queue_route"] = self.hot.len().into();
        v["queue"] = self.queue_len().into();
        v["zones"] = self.entries.len().into();
        v["recent_added"] = serde_json::json!(self.recent);
        if extra_stats() {
            v["donor"] = (if self.sample_m.is_some() { "sample" } else { "current" }).into();
            v["m"] = self.sample_m.unwrap_or(0).into();
            v["hist_edges"] = serde_json::json!(HIST);
            v["extra"] = serde_json::to_value(&self.extra).unwrap();
        }
        v
    }

    pub fn line(&self) -> String {
        let s = &self.stats;
        let mut out = self.base_line();
        if extra_stats() {
            let x = &self.extra;
            let h = |n: &str, h: &[u64; 8], f: u64| {
                let c: u64 = h.iter().sum();
                format!("{n} {c} mean {} {h:?}", f / c.max(1))
            };
            out += &format!(
                "; donor {} m {}: {} sampled ({} miss, {} milestones); queued frames (<=50,100,200,400,800,1600,3000,>) {}; {}; {}; {}; per job {:.2} added {:.0} frames, {:.4} added/frame",
                if self.sample_m.is_some() { "sample" } else { "current" },
                self.sample_m.unwrap_or(0),
                x.queued_sample,
                x.sample_miss,
                x.milestones,
                h("route", &x.hist_route, x.frames_route),
                h("top", &x.hist_top, x.frames_top),
                h("sample", &x.hist_sample, x.frames_sample),
                h("bg", &x.hist_bg, x.frames_bg),
                s.added as f64 / s.run.max(1) as f64,
                s.frames as f64 / s.run.max(1) as f64,
                s.added as f64 / s.frames.max(1) as f64,
            );
        }
        out
    }

    fn base_line(&self) -> String {
        let s = &self.stats;
        format!(
            "zone share: {} entries; queued {} route {} top {} bg ({} waiting, {} dropped); {} run ({} stale), {} frames, {} boundaries; {} added by {} jobs ({} sooner in cell, {} sooner than donor saving {} frames), {} dup, {} died, {} left zone, {} traps; {} new zones, longest chain {}",
            s.entries, s.queued_route, s.queued_top, s.queued_bg, self.queue_len(), s.dropped, s.run, s.stale, s.frames, s.boundaries,
            s.added, s.jobs_ok, s.sooner, s.matched_sooner, s.matched_frames_saved, s.duplicate, s.died, s.left_zone, s.traps, s.new_zones, s.max_chain
        )
    }
}
