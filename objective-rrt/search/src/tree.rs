//! The state tree: every state reached, how to get there, and how useful it
//! has been as a start.

use std::sync::Arc;

use obrrt_core::{GamePad, Objective, St};

use crate::store::Delta;

/// OBRRT_RETIRE_SUBSET=1: the frontier is every maximal collected set, not
/// just the highest count. A state is behind it (and retired) only when a
/// frontier set has everything it has (`St::collected` superset, keys and
/// stars at least as many), so a fight or a set of coins under way survives
/// another star being collected elsewhere. Off by default.
pub static RETIRE_SUBSET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// OBRRT_RETIRE_SUBSET_CAP: at most this many frontier sets per star count;
/// beyond it, the set that last gained a state longest ago is dropped
/// (its states retired). Default 6.
pub static RETIRE_SUBSET_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(6);

/// OBRRT_KEEP_WIP=1: a state behind the frontier with work under way in
/// its level (`St::wip`: boss damage, red coins, posts) that every frontier
/// state in that level lacks (its wip above theirs) is not retired: it is
/// incomparable, not behind. At most KEEP_WIP_N such states per level (the
/// most wip, then the earliest), and none more than KEEP_WIP_FRAMES game
/// frames before the frontier's earliest state in that level (or anywhere,
/// with none there). They are b candidates and starts. Off by default.
pub static KEEP_WIP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static KEEP_WIP_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(64);
pub static KEEP_WIP_FRAMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(3000);

/// A collected set: (collected bitset, keys, stars).
pub type SetKey = ([u8; 26], u8, i16);

pub fn set_key(st: &St) -> SetKey {
    (st.collected(), st.keys, st.stars)
}

/// Whether `b` has everything `a` has.
pub fn covers(b: &SetKey, a: &SetKey) -> bool {
    b.1 >= a.1 && b.2 >= a.2 && a.0.iter().zip(&b.0).all(|(x, y)| x & !y == 0)
}

/// One maximal collected set on the frontier (OBRRT_RETIRE_SUBSET).
#[derive(Debug, Clone)]
pub struct FrontierSet {
    pub key: SetKey,
    /// The node that last joined it.
    pub last: u32,
    pub states: usize,
}

/// Per-state diagnostics and the state's own failure back-off.
#[derive(Debug, Clone, Default)]
pub struct NodeStats {
    pub as_b: u32,
    pub as_start: u32,
    pub successes: u32,
    pub added: u32,
    /// Attempts started here since it last produced a state.
    pub barren: u32,
    /// Multiplies its start-sampling weight: ×decay per barren attempt,
    /// back to 1 when it produces a state.
    pub backoff: f64,
    /// Soft dedupe: states at the same place (dedupe cell) that got there
    /// sooner; its sampling weight is halved for each.
    pub crowd: u32,
    /// Attempts from it in which the emulated game trapped.
    pub traps: u32,
}

pub struct Node {
    pub parent: Option<u32>,
    pub depth: u32,
    /// Inputs from the parent's state to this one.
    pub inputs: Vec<GamePad>,
    /// This state, with its objects.
    pub st: St,
    pub save: Delta,
    /// The objective kind that produced it ("root" for the root).
    pub via: &'static str,
    pub stats: NodeStats,
    /// Behind the frontier: save state and objects freed (see `retire`).
    pub retired: bool,
    /// RRT* mode: the objective that made it (re-attempted from a faster
    /// parent when rewiring), and whether a faster state has since reached
    /// its place (it is then no longer a start).
    pub made_by: Option<Arc<dyn Objective>>,
    pub dominated: bool,
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub children: Vec<Vec<u32>>,
    /// The best (keys, stars) reached, and the nodes that have it.
    pub frontier_progress: (u8, i16),
    pub frontier: Vec<u32>,
    pub delta_bytes: usize,
    /// While set, an advance of the frontier doesn't retire the states
    /// behind it (resume: a later record may still need their saves).
    pub hold_retire: bool,
    /// OBRRT_RETIRE_SUBSET: the maximal collected sets (an antichain), the
    /// sets dropped by the cap (states in or under them are behind), and
    /// the most sets that coexisted.
    pub subset: bool,
    pub sets: Vec<FrontierSet>,
    pub dropped: Vec<SetKey>,
    pub sets_peak: usize,
    /// OBRRT_KEEP_WIP: each node's `St::wip` (taken when added, as retiring
    /// drops the objects), the states kept for it (sorted), and counts.
    pub keep_wip: bool,
    pub wip: Vec<i64>,
    pub wip_kept: Vec<u32>,
    pub wip_peak: usize,
    pub wip_ever: std::collections::BTreeSet<u32>,
    pub wip_dropped: u64,
}

impl Tree {
    pub fn new(root: St) -> Tree {
        let progress = root.progress();
        let root_wip = root.wip();
        Tree {
            nodes: vec![Node {
                parent: None,
                depth: 0,
                inputs: Vec::new(),
                st: root,
                save: Delta::default(),
                via: "root",
                stats: NodeStats { backoff: 1.0, ..Default::default() },
                retired: false,
                made_by: None,
                dominated: false,
            }],
            children: vec![Vec::new()],
            frontier_progress: progress,
            frontier: vec![0],
            delta_bytes: 0,
            hold_retire: false,
            subset: RETIRE_SUBSET.load(std::sync::atomic::Ordering::Relaxed),
            sets: Vec::new(),
            dropped: Vec::new(),
            sets_peak: 0,
            keep_wip: KEEP_WIP.load(std::sync::atomic::Ordering::Relaxed),
            wip: vec![root_wip],
            wip_kept: Vec::new(),
            wip_peak: 0,
            wip_ever: Default::default(),
            wip_dropped: 0,
        }
        .with_root_set()
    }

    fn with_root_set(mut self) -> Tree {
        if self.subset {
            self.sets.push(FrontierSet { key: set_key(&self.nodes[0].st), last: 0, states: 1 });
            self.sets_peak = 1;
        }
        self
    }

    /// Whether `st` is on the frontier: at its best count, or with
    /// OBRRT_RETIRE_SUBSET in one of its maximal sets.
    pub fn on_frontier(&self, st: &St) -> bool {
        if self.subset {
            let k = set_key(st);
            self.sets.iter().any(|s| s.key == k)
        } else {
            st.progress() == self.frontier_progress
        }
    }

    /// OBRRT_RETIRE_SUBSET: places the new node `id` among the frontier
    /// sets; returns whether it is on the frontier.
    fn add_subset(&mut self, id: u32) -> bool {
        let k = set_key(&self.nodes[id as usize].st);
        if let Some(s) = self.sets.iter_mut().find(|s| s.key == k) {
            s.last = id;
            s.states += 1;
            self.frontier.push(id);
            return true;
        }
        if self.sets.iter().any(|s| covers(&s.key, &k)) || self.dropped.iter().any(|d| covers(d, &k)) {
            if !self.hold_retire && !self.wip_note(id) {
                self.retire_one(id);
            }
            return false;
        }
        // A new maximal set: it replaces the sets it covers.
        self.sets.retain(|s| !covers(&k, &s.key));
        self.sets.push(FrontierSet { key: k, last: id, states: 1 });
        let cap = RETIRE_SUBSET_CAP.load(std::sync::atomic::Ordering::Relaxed).max(1);
        while self.sets.iter().filter(|s| s.key.2 == k.2).count() > cap {
            let (i, _) = self
                .sets
                .iter()
                .enumerate()
                .filter(|(_, s)| s.key.2 == k.2 && s.key != k)
                .min_by_key(|(_, s)| s.last)
                .unwrap();
            let d = self.sets.remove(i);
            self.dropped.push(d.key);
        }
        self.sets_peak = self.sets_peak.max(self.sets.len());
        if !self.hold_retire {
            let frame = self.nodes[id as usize].st.frame;
            let bits: Vec<String> = k.0.iter().enumerate().filter(|(_, b)| **b != 0).map(|(c, b)| format!("{c}:{b:02x}")).collect();
            println!("NEW SET keys {} stars {} at node {id} frame {frame}: [{}]; {} sets, {} dropped", k.1, k.2, bits.join(" "), self.sets.len(), self.dropped.len());
        }
        let keys: std::collections::HashSet<SetKey> = self.sets.iter().map(|s| s.key).collect();
        let nodes = &self.nodes;
        self.frontier.retain(|&i| keys.contains(&set_key(&nodes[i as usize].st)));
        self.frontier.push(id);
        if !self.hold_retire {
            self.retire_subset();
        }
        true
    }

    /// OBRRT_RETIRE_SUBSET: retires every state in no frontier set.
    fn retire_subset(&mut self) {
        let keys: std::collections::HashSet<SetKey> = self.sets.iter().map(|s| s.key).collect();
        let keep = self.wip_select(|t, n| !keys.contains(&set_key(&t.nodes[n as usize].st)));
        let mut freed = 0usize;
        for (i, n) in self.nodes.iter_mut().enumerate() {
            if !n.retired && !keys.contains(&set_key(&n.st)) && keep.binary_search(&(i as u32)).is_err() {
                freed += n.save.bytes();
                n.save = Delta::default();
                n.st.objects = None;
                n.retired = true;
            }
        }
        self.delta_bytes -= freed;
    }

    /// Adds a node; returns its id and whether it advanced the frontier.
    pub fn add(&mut self, parent: u32, inputs: Vec<GamePad>, st: St, save: Delta, via: &'static str) -> (u32, bool) {
        let id = self.nodes.len() as u32;
        let progress = st.progress();
        self.delta_bytes += save.bytes();
        let depth = self.nodes[parent as usize].depth + 1;
        self.nodes.push(Node {
            parent: Some(parent),
            depth,
            inputs,
            st,
            save,
            via,
            stats: NodeStats { backoff: 1.0, ..Default::default() },
            retired: false,
            made_by: None,
            dominated: false,
        });
        self.children.push(Vec::new());
        self.children[parent as usize].push(id);
        self.wip.push(if self.keep_wip { self.nodes[id as usize].st.wip() } else { 0 });
        let advanced = progress > self.frontier_progress;
        if self.subset {
            if advanced {
                self.frontier_progress = progress;
            }
            self.add_subset(id);
            return (id, advanced);
        }
        if advanced {
            self.frontier_progress = progress;
            self.frontier.clear();
            if self.keep_wip {
                // The new frontier state first: the wip comparison needs it.
                self.frontier.push(id);
                if !self.hold_retire {
                    self.retire(progress);
                }
                return (id, advanced);
            }
            if !self.hold_retire {
                self.retire(progress);
            }
        }
        if progress == self.frontier_progress {
            self.frontier.push(id);
        } else if self.keep_wip && !self.hold_retire {
            self.wip_note(id);
        }
        (id, advanced)
    }

    /// Retires every state behind the frontier (after `hold_retire`).
    pub fn retire_behind(&mut self) {
        if self.subset {
            return self.retire_subset();
        }
        self.retire(self.frontier_progress);
    }

    /// Frees the save state and object list of every state behind `progress`:
    /// with b drawn from the frontier and starts never behind b, they can't be
    /// used again. Their inputs stay, for replaying descendants.
    fn retire(&mut self, progress: (u8, i16)) {
        let keep = self.wip_select(|t, n| t.nodes[n as usize].st.progress() < progress);
        let mut freed = 0usize;
        for (i, n) in self.nodes.iter_mut().enumerate() {
            if n.st.progress() < progress && !n.retired && keep.binary_search(&(i as u32)).is_err() {
                freed += n.save.bytes();
                n.save = Delta::default();
                n.st.objects = None;
                n.retired = true;
            }
        }
        self.delta_bytes -= freed;
    }

    /// OBRRT_KEEP_WIP: per level, the frontier's highest wip and earliest
    /// frame there, and the frontier's earliest frame overall.
    fn wip_bounds(&self) -> (std::collections::BTreeMap<u16, (i64, u32)>, u32) {
        let mut per: std::collections::BTreeMap<u16, (i64, u32)> = Default::default();
        let mut first = u32::MAX;
        for &i in &self.frontier {
            let st = &self.nodes[i as usize].st;
            let e = per.entry(st.level).or_insert((i64::MIN, u32::MAX));
            e.0 = e.0.max(self.wip[i as usize]);
            e.1 = e.1.min(st.frame);
            first = first.min(st.frame);
        }
        (per, first)
    }

    /// OBRRT_KEEP_WIP: whether live state `i` (behind the frontier) has more
    /// work under way than every frontier state in its level, recently enough.
    fn wip_qualifies(&self, i: u32, bounds: &(std::collections::BTreeMap<u16, (i64, u32)>, u32)) -> bool {
        let n = &self.nodes[i as usize];
        let w = self.wip[i as usize];
        if i == 0 || n.retired || n.st.modal() || w <= 0 {
            return false;
        }
        let (best, arrival) = bounds.0.get(&n.st.level).copied().unwrap_or((0, bounds.1));
        let m = KEEP_WIP_FRAMES.load(std::sync::atomic::Ordering::Relaxed);
        w > best && (arrival == u32::MAX || n.st.frame.saturating_add(m) >= arrival)
    }

    /// OBRRT_KEEP_WIP: the states to keep among the live ones `behind`
    /// selects (plus those already kept), at most KEEP_WIP_N per level, the
    /// most wip and then the earliest first. Sets `wip_kept`; sorted ids.
    fn wip_select(&mut self, behind: impl Fn(&Tree, u32) -> bool) -> Vec<u32> {
        if !self.keep_wip {
            return Vec::new();
        }
        let bounds = self.wip_bounds();
        let mut by_level: std::collections::BTreeMap<u16, Vec<(i64, u32, u32)>> = Default::default();
        for i in 0..self.nodes.len() as u32 {
            if self.wip[i as usize] > 0 && !self.nodes[i as usize].retired && behind(self, i) && self.wip_qualifies(i, &bounds) {
                let st = &self.nodes[i as usize].st;
                by_level.entry(st.level).or_default().push((-self.wip[i as usize], st.frame, i));
            }
        }
        let cap = KEEP_WIP_N.load(std::sync::atomic::Ordering::Relaxed);
        let mut keep = Vec::new();
        for (_, mut v) in by_level {
            v.sort_unstable();
            keep.extend(v.into_iter().take(cap).map(|x| x.2));
        }
        keep.sort_unstable();
        let before = self.wip_kept.len();
        self.wip_dropped += self.wip_kept.iter().filter(|i| keep.binary_search(i).is_err()).count() as u64;
        self.wip_ever.extend(keep.iter().copied());
        self.wip_kept = keep.clone();
        self.wip_peak = self.wip_peak.max(keep.len());
        if !self.hold_retire && (before > 0 || !keep.is_empty()) {
            let mut lv: std::collections::BTreeMap<u16, (usize, i64)> = Default::default();
            for &i in &keep {
                let e = lv.entry(self.nodes[i as usize].st.level).or_default();
                e.0 += 1;
                e.1 = e.1.max(self.wip[i as usize]);
            }
            let s: Vec<String> = lv.iter().map(|(l, (n, w))| format!("level {l}: {n} (wip up to {w})")).collect();
            println!("WIP kept {} states behind the frontier (was {before}): {}", keep.len(), s.join(", "));
        }
        keep
    }

    /// OBRRT_KEEP_WIP: a state just added behind the frontier; keeps it if it
    /// qualifies (retiring a kept state of its level beyond the cap).
    /// Returns whether it is kept.
    fn wip_note(&mut self, id: u32) -> bool {
        if !self.keep_wip || self.wip[id as usize] <= 0 {
            return false;
        }
        let bounds = self.wip_bounds();
        if !self.wip_qualifies(id, &bounds) {
            return false;
        }
        let level = self.nodes[id as usize].st.level;
        let pos = self.wip_kept.binary_search(&id).unwrap_or_else(|p| p);
        self.wip_kept.insert(pos, id);
        self.wip_ever.insert(id);
        let cap = KEEP_WIP_N.load(std::sync::atomic::Ordering::Relaxed);
        let mut same: Vec<(i64, u32, u32)> = self
            .wip_kept
            .iter()
            .filter(|&&i| self.nodes[i as usize].st.level == level)
            .map(|&i| (-self.wip[i as usize], self.nodes[i as usize].st.frame, i))
            .collect();
        let mut kept = true;
        if same.len() > cap {
            same.sort_unstable();
            for &(_, _, i) in &same[cap..] {
                self.wip_kept.retain(|&k| k != i);
                self.wip_dropped += 1;
                kept &= i != id;
                self.retire_one(i);
            }
        }
        self.wip_peak = self.wip_peak.max(self.wip_kept.len());
        kept
    }

    /// OBRRT_KEEP_WIP: re-checks the kept states (each round): the frontier
    /// may since have the same work in that level, or have moved too far on;
    /// those are retired.
    pub fn wip_refresh(&mut self) {
        if !self.keep_wip || self.wip_kept.is_empty() || self.hold_retire {
            return;
        }
        let bounds = self.wip_bounds();
        let kept = std::mem::take(&mut self.wip_kept);
        let mut out = Vec::with_capacity(kept.len());
        for i in kept {
            if self.wip_qualifies(i, &bounds) && !self.on_frontier(&self.nodes[i as usize].st) {
                out.push(i);
            } else {
                self.wip_dropped += 1;
                if !self.on_frontier(&self.nodes[i as usize].st) {
                    self.retire_one(i);
                }
            }
        }
        self.wip_kept = out;
    }

    /// OBRRT_KEEP_WIP: whether `i` is kept for its work under way.
    pub fn is_wip(&self, i: u32) -> bool {
        self.keep_wip && self.wip_kept.binary_search(&i).is_ok()
    }

    /// Frees one state's save and objects (its inputs stay, for replaying
    /// descendants); it can no longer be a start.
    pub fn retire_one(&mut self, id: u32) {
        let n = &mut self.nodes[id as usize];
        if n.retired || id == 0 {
            return;
        }
        self.delta_bytes -= n.save.bytes();
        n.save = Delta::default();
        n.st.objects = None;
        n.retired = true;
    }

    /// Ids of `id` and its ancestors, nearest first, up to `k` generations up.
    pub fn ancestors(&self, id: u32, k: u32) -> Vec<u32> {
        let mut out = vec![id];
        let mut at = id;
        for _ in 0..k {
            match self.nodes[at as usize].parent {
                Some(p) => {
                    out.push(p);
                    at = p;
                }
                None => break,
            }
        }
        out
    }

    /// Every node sharing an ancestor with `b` at most `k` generations above
    /// both (k = 4 allows third cousins), `b` included.
    pub fn kin_set(&self, b: u32, k: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for a in self.ancestors(b, k) {
            // Descendants of `a` at most k generations below it.
            let mut level = vec![a];
            for _ in 0..=k {
                let mut next = Vec::new();
                for n in level {
                    if seen.insert(n) {
                        out.push(n);
                    }
                    next.extend_from_slice(&self.children[n as usize]);
                }
                level = next;
            }
        }
        out
    }

    /// Whether `a` and `b` share an ancestor at most `k` generations above
    /// each (k = 4 allows third cousins).
    pub fn kin(&self, a: u32, b_ancestors: &[u32], k: u32) -> bool {
        let mut at = a;
        for _ in 0..=k {
            if b_ancestors.contains(&at) {
                return true;
            }
            match self.nodes[at as usize].parent {
                Some(p) => at = p,
                None => return false,
            }
        }
        false
    }

    /// All inputs from the root to `id`.
    pub fn path(&self, id: u32) -> Vec<GamePad> {
        let mut chain = Vec::new();
        let mut at = Some(id);
        while let Some(n) = at {
            chain.push(n);
            at = self.nodes[n as usize].parent;
        }
        chain.iter().rev().flat_map(|&n| self.nodes[n as usize].inputs.iter().copied()).collect()
    }
}
