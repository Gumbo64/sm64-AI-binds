//! Rerouting onto faster duplicates (`OBRRT_REROUTE=1`, off by default).
//!
//! When a state turns up that is "the same" (same key, see [`RerouteKey`]) as
//! a node the tree already has but sooner, the slower node's children are
//! grafted onto it: each child's recorded inputs are replayed from the faster
//! state, and if the end is again the same as the child (same key) and sooner,
//! it is added under the faster state. That new node is itself a faster
//! duplicate of the child, so the child's own children follow, and the saving
//! cascades down the subtree to whatever was found from it. Nothing is
//! dominated or retired: the slow nodes stay starts (RRT* mode's domination
//! stalled the climb, RUNS.md v13-star).
//!
//! Replays are deterministic and cost only their frames; they use no RNG, and
//! the queue order follows merge order, so a run stays reproducible.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use obrrt_core::St;

use crate::tree::Tree;

/// What counts as "the same state".
pub enum RerouteKey {
    /// The engine's duplicate check cell (level, area, stars, keys, coins,
    /// 64-unit position, open dialog or menu, fingerprint).
    Cell,
    /// An objective-scattershot schema's key (`OBRRT_REROUTE_SCHEMA`).
    Schema(Box<objective_scattershot::objective::Tree>),
}

/// Replay `child`'s inputs from node `from`; `depth` counts cascade steps.
#[derive(Clone, Copy, Debug)]
pub struct Graft {
    pub from: u32,
    pub child: u32,
    pub depth: u16,
    /// How it was queued: [`KIND_DUP`], [`KIND_FORWARD`], [`KIND_RELAX`] or
    /// [`KIND_LEG`].
    pub kind: u8,
    /// [`KIND_LEG`]: the leg's last node (the chain follows the path from
    /// `child` down to it); `u32::MAX` otherwise.
    pub leg: u32,
}

/// Graft kinds, for per-kind counts.
pub const KINDS: usize = 5;
pub const KIND_NAMES: [&str; KINDS] = ["dup", "forward", "relax", "leg", "route"];

/// A slower duplicate's child onto a new faster state (the original rule).
pub const KIND_DUP: u8 = 0;
/// `OBRRT_REROUTE_FORWARD`: a new child of a slow copy onto the faster copies
/// of its parent.
pub const KIND_FORWARD: u8 = 1;
/// `OBRRT_REROUTE_RELAX`: between states that differ only in progress (same
/// movement key, see [`move_key`]).
pub const KIND_RELAX: u8 = 2;
/// `OBRRT_LEG_REUSE`: one edge of a known leg (spawn -> next zone change or
/// milestone, see `legs`) replayed from another copy of its spawn; a kept
/// one queues the leg's next edge from the new node.
pub const KIND_LEG: u8 = 3;
/// `OBRRT_POLISH`: one edge of the best route replayed from a sooner copy of
/// an earlier route state (see `polish`); accepted as a leg edge, and the
/// engine queues the next edge from a kept one.
pub const KIND_ROUTE: u8 = 4;

#[derive(Default, Clone, Copy, Debug)]
pub struct RerouteStats {
    pub queued: u64,
    pub dropped: u64,
    pub tried: u64,
    pub ok: u64,
    pub diverged: u64,
    pub not_faster: u64,
    pub duplicate: u64,
    pub died: u64,
    pub traps: u64,
    pub frames: u64,
    pub frames_saved: u64,
    pub max_depth: u16,
    /// Grafts taken whose faster start had been retired by the time they
    /// ran (its save freed): skipped.
    pub stale: u64,
    /// Forward grafts: queued, tried (accept checks), kept, frames saved.
    pub fwd_queued: u64,
    pub fwd_tried: u64,
    pub fwd_ok: u64,
    pub fwd_saved: u64,
    /// Relaxed grafts: queued, tried, kept; rejected as diverged (movement
    /// key), regressed (progress below the start's), longer, duplicate.
    pub relax_queued: u64,
    pub relax_tried: u64,
    pub relax_ok: u64,
    pub relax_diverged: u64,
    pub relax_regress: u64,
    pub relax_longer: u64,
    pub relax_dup: u64,
    /// Kept relaxed grafts that changed zone (e.g. a painting entry).
    pub relax_zone: u64,
    /// Per kind (`KIND_NAMES`): queued, replayed (tried), kept; replayed
    /// frames of kept ones and of rejected ones (diverged, not sooner,
    /// duplicate, died).
    pub k_queued: [u64; KINDS],
    pub k_tried: [u64; KINDS],
    pub k_ok: [u64; KINDS],
    pub k_frames_kept: [u64; KINDS],
    pub k_frames_rejected: [u64; KINDS],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Forward {
    Off,
    Best,
    All,
}

/// `OBRRT_REROUTE_FORWARD`: unset/0 off, `best` the soonest faster copy,
/// `1`/`all` every faster copy.
fn forward_mode() -> Forward {
    static M: std::sync::OnceLock<Forward> = std::sync::OnceLock::new();
    *M.get_or_init(|| match obrrt_core::sys::env::var("OBRRT_REROUTE_FORWARD").as_deref() {
        Ok("best") => Forward::Best,
        Ok("1") | Ok("all") => Forward::All,
        Ok("") | Ok("0") | Err(_) => Forward::Off,
        Ok(v) => panic!("OBRRT_REROUTE_FORWARD={v}: expected 0, 1, all or best"),
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Relax {
    Off,
    /// Peers by [`move_key`], of another full key (another progress): the
    /// segment is replayed as is (no sooner needed), it must reach a state
    /// of a full key the tree doesn't have yet (or later).
    Move,
    /// Peers by [`spawn_key`] and [`move_key`] at least the margin sooner (sooner copies of
    /// the fixed spawns: act select, exit-course lobby, star landing); the
    /// end must match the child's [`move_key`] and be the margin sooner.
    Spawn,
}

/// `OBRRT_REROUTE_RELAX`: unset/0 off, `1`/`move` or `spawn` (see [`Relax`]).
fn relax_mode() -> Relax {
    static M: std::sync::OnceLock<Relax> = std::sync::OnceLock::new();
    *M.get_or_init(|| match obrrt_core::sys::env::var("OBRRT_REROUTE_RELAX").as_deref() {
        Ok("1") | Ok("move") => Relax::Move,
        Ok("spawn") => Relax::Spawn,
        Ok("") | Ok("0") | Err(_) => Relax::Off,
        Ok(v) => panic!("OBRRT_REROUTE_RELAX={v}: expected 0, 1, move or spawn"),
    })
}

fn relax_on() -> bool {
    relax_mode() != Relax::Off
}

/// Relaxed grafts queued at most per noted node (each direction).
const RELAX_CAP: usize = 8;

/// The movement-only key: level, area, 64-unit position cell, action id,
/// speed bucket (8 u/f), facing sector (1/8 turn) and modal state (dialog,
/// menu, act select, transition). No stars, keys, coins or collected set,
/// so a state after a star matches the same spot before it.
pub fn move_key(st: &St) -> u64 {
    let cellp = |v: f32| (v / 64.0).floor() as i32;
    hash(&(
        st.level,
        st.area,
        [cellp(st.pos[0]), cellp(st.pos[1]), cellp(st.pos[2])],
        st.action & 0x1ff,
        (st.forward_vel / 8.0).round() as i32,
        (st.yaw as u16) >> 13,
        (st.dialog, st.menu_mode, st.act_select, st.transition != 0),
    ))
}

/// The fixed-spawn key: level, area, act, star flags and save flags
/// (`progress_set`), keys, 64-unit position cell. No coins, action or speed.
pub fn spawn_key(st: &St) -> u64 {
    let cellp = |v: f32| (v / 64.0).floor() as i32;
    hash(&(st.level, st.area, st.act, st.progress_set(), st.keys, [cellp(st.pos[0]), cellp(st.pos[1]), cellp(st.pos[2])]))
}

fn relax_key(st: &St) -> u64 {
    match relax_mode() {
        // A fixed spawn puts Mario in the same action, speed and facing:
        // peers must share the movement key too (a spawn key alone matches
        // every state passing through the cell, and those diverge).
        Relax::Spawn => hash(&(spawn_key(st), move_key(st))),
        _ => move_key(st),
    }
}

/// Whether node `f` can still be a graft's start (as `replay::loadable`).
fn live(tree: &Tree, f: u32) -> bool {
    f == 0 || !tree.nodes[f as usize].retired || (tree.sparse.is_some() && tree.can_anchor(f))
}

pub struct Reroute {
    pub key: RerouteKey,
    /// Each node's key, by node id (u64::MAX: not keyed yet).
    pub keys: Vec<u64>,
    index: HashMap<u64, Vec<u32>>,
    pub(crate) hot: VecDeque<Graft>,
    pub(crate) cold: VecDeque<Graft>,
    seen: HashSet<(u32, u32)>,
    pub stats: RerouteStats,
    /// Replay budget (`crate::replay`): a full cold queue drops its oldest
    /// graft for a new one instead of the new one.
    pub lifo: bool,
    /// `OBRRT_REROUTE_RELAX`: each node's relaxed key, and nodes by it.
    mkeys: Vec<u64>,
    mindex: HashMap<u64, Vec<u32>>,
    /// The graft last accepted (`accept`), until its node is noted.
    accepted: Option<Graft>,
    /// `OBRRT_LEG_REUSE` (see `legs`).
    pub(crate) legs: Option<crate::legs::Legs>,
    /// `OBRRT_GRAFT_BIAS=star`: nodes on a path that ended in more stars.
    /// OBRRT_POLISH without OBRRT_REROUTE: only route grafts, no duplicate
    /// grafts.
    pub route_only: bool,
}

/// Queued grafts kept at most (beyond it the cold queue drops new ones).
const COLD_CAP: usize = 50_000;

fn hash<T: Hash>(t: &T) -> u64 {
    let mut h = DefaultHasher::new();
    t.hash(&mut h);
    h.finish()
}

/// Loads an objective-scattershot schema (also used by `schemapick`).
pub fn load_schema(path: &str) -> anyhow::Result<Box<objective_scattershot::objective::Tree>> {
    use anyhow::Context;
    let json = obrrt_core::sys::fs::read_to_string(path).with_context(|| format!("schema {path}"))?;
    Ok(Box::new(objective_scattershot::objective::Tree::load(&json)?))
}

/// The hash of `st`'s key under schema `t`.
pub fn schema_hash(t: &objective_scattershot::objective::Tree, st: &St) -> u64 {
    hash(&t.key(st))
}

impl Reroute {
    pub fn new(key: RerouteKey) -> Reroute {
        Reroute { key, keys: Vec::new(), index: HashMap::new(), hot: VecDeque::new(), cold: VecDeque::new(), seen: HashSet::new(), stats: RerouteStats::default(), lifo: false, mkeys: Vec::new(), mindex: HashMap::new(), accepted: None, legs: crate::legs::Legs::from_env(), route_only: false }
    }

    /// `st`'s key; `cell` is the engine's duplicate cell for it.
    pub fn key_of<C: Hash>(&self, st: &St, cell: &C) -> u64 {
        match &self.key {
            RerouteKey::Cell => hash(cell),
            RerouteKey::Schema(t) => schema_hash(t, st),
        }
    }

    /// Notes node `id` (state `st`, duplicate cell `cell`), just added: queues
    /// grafts of the children of every slower node with the same key, by at
    /// least `margin` frames. `depth` is `id`'s own cascade depth (0 for a
    /// node from an ordinary attempt). Grafts of route nodes (`hot`) go first.
    pub fn note<C: Hash>(&mut self, id: u32, st: &St, cell: &C, tree: &Tree, hot: &HashSet<u32>, margin: u32, depth: u16) {
        let k = self.key_of(st, cell);
        if self.keys.len() <= id as usize {
            self.keys.resize(id as usize + 1, u64::MAX);
        }
        self.keys[id as usize] = k;
        if let Some(g) = self.accepted.take() {
            if depth > 0 && tree.nodes[id as usize].parent == Some(g.from) {
                let saved = tree.nodes[g.child as usize].st.frame.saturating_sub(st.frame) as u64;
                match g.kind {
                    KIND_FORWARD => {
                        self.stats.fwd_ok += 1;
                        self.stats.fwd_saved += saved;
                    }
                    KIND_RELAX => {
                        self.stats.relax_ok += 1;
                        self.stats.relax_zone += (tree.nodes[g.from as usize].st.zone() != st.zone()) as u64;
                    }
                    _ => {}
                }
                let n = tree.nodes[g.child as usize].inputs.len() as u64;
                self.stats.k_ok[g.kind as usize] += 1;
                self.stats.k_frames_kept[g.kind as usize] += n;
                self.stats.k_frames_rejected[g.kind as usize] -= n; // counted as rejected until kept
                if g.kind == KIND_LEG {
                    self.leg_next(g, id, tree, hot, depth);
                }
            }
        }
        if let Some(list) = self.index.get(&k).filter(|_| !self.route_only) {
            let mut q = Vec::new();
            for &y in list {
                if tree.nodes[y as usize].st.frame < st.frame + margin {
                    continue;
                }
                let yhot = hot.contains(&y);
                q.extend(tree.children[y as usize].iter().map(|&c| (c, yhot || hot.contains(&c))));
            }
            for (c, h) in q {
                self.push(Graft { from: id, child: c, depth, kind: KIND_DUP, leg: u32::MAX }, h);
            }
        }
        self.forward(id, tree, hot, margin, depth);
        if relax_on() {
            self.relax(id, st, tree, hot, margin, depth);
        }
        self.index.entry(k).or_default().push(id);
        if self.legs.is_some() {
            self.legs_note(id, st, tree, hot, margin, depth);
        }
    }

    /// Queues `g` (once per start and child); `hot` grafts go first.
    ///
    /// `seen` holds the pairs queued now or taken before, not the ones a
    /// full cold queue dropped: a dropped pair forgotten is queued again if
    /// met again (its start and child are rarely both met twice: most
    /// grafts start from the node just noted), and without that a burst of
    /// grafts (a cascade into a cell with thousands of slower copies once
    /// queued 850 M in one run, almost all dropped) kept every pair forever,
    /// ~10 GB the save budget never saw.
    pub(crate) fn push(&mut self, g: Graft, hot: bool) {
        if !self.seen.insert((g.from, g.child)) {
            return;
        }
        if hot {
            self.hot.push_back(g);
        } else if self.cold.len() < COLD_CAP {
            self.cold.push_back(g);
        } else if self.lifo {
            if let Some(old) = self.cold.pop_front() {
                self.seen.remove(&(old.from, old.child));
            }
            self.stats.dropped += 1;
            self.cold.push_back(g);
        } else {
            self.seen.remove(&(g.from, g.child));
            self.stats.dropped += 1;
            return;
        }
        self.stats.queued += 1;
        self.stats.k_queued[g.kind as usize] += 1;
        match g.kind {
            KIND_FORWARD => self.stats.fwd_queued += 1,
            KIND_RELAX => self.stats.relax_queued += 1,
            _ => {}
        }
    }

    /// `OBRRT_REROUTE_FORWARD`: a new child `id` of a node p that already
    /// has faster duplicates (same key, at least `margin` sooner) is grafted
    /// onto them too (`all`), or onto the soonest live one (`best`). Without
    /// it a segment found later from the slow copy of a section's start
    /// never reaches the fast copy, so improvements to successive sections
    /// (start -> lobby -> BoB) never combine.
    fn forward(&mut self, id: u32, tree: &Tree, hot: &HashSet<u32>, margin: u32, depth: u16) {
        let mode = forward_mode();
        if mode == Forward::Off {
            return;
        }
        let Some(p) = tree.nodes[id as usize].parent else { return };
        let Some(&kp) = self.keys.get(p as usize) else { return };
        if kp == u64::MAX {
            return;
        }
        let pf = tree.nodes[p as usize].st.frame;
        let Some(list) = self.index.get(&kp) else { return };
        let mut fs: Vec<u32> = list.iter().copied().filter(|&f| f != id && f != p && tree.nodes[f as usize].st.frame + margin <= pf && live(tree, f)).collect();
        if mode == Forward::Best {
            fs = fs.iter().copied().min_by_key(|&f| (tree.nodes[f as usize].st.frame, f)).into_iter().collect();
        }
        let h = hot.contains(&id) || hot.contains(&p);
        for f in fs {
            self.push(Graft { from: f, child: id, depth, kind: KIND_FORWARD, leg: u32::MAX }, h || hot.contains(&f));
        }
    }

    /// `OBRRT_REROUTE_RELAX`: grafts between states that differ only in
    /// progress (same movement key, another full key). The children of such
    /// peers of `id` are replayed from `id` (a lobby state after a star gets
    /// the painting entry found before it), and, with forward on, the new
    /// child `id` is replayed from the peers of its parent. At most
    /// `RELAX_CAP` each way, the soonest live peer of each other full key.
    fn relax(&mut self, id: u32, st: &St, tree: &Tree, hot: &HashSet<u32>, margin: u32, depth: u16) {
        let spawn = relax_mode() == Relax::Spawn;
        let m = relax_key(st);
        if self.mkeys.len() <= id as usize {
            self.mkeys.resize(id as usize + 1, u64::MAX);
        }
        self.mkeys[id as usize] = m;
        let k = self.keys[id as usize];
        let mut q = Vec::new();
        if let Some(list) = self.mindex.get(&m) {
            'peers: for &y in list.iter().rev() {
                if self.keys[y as usize] == k || (spawn && tree.nodes[y as usize].st.frame < st.frame + margin) {
                    continue;
                }
                for &c in &tree.children[y as usize] {
                    if q.len() >= RELAX_CAP {
                        break 'peers;
                    }
                    if !self.seen.contains(&(id, c)) {
                        q.push((Graft { from: id, child: c, depth, kind: KIND_RELAX, leg: u32::MAX }, hot.contains(&id) || hot.contains(&c)));
                    }
                }
            }
        }
        if forward_mode() != Forward::Off {
            if let Some(p) = tree.nodes[id as usize].parent {
                let (kp, mp) = (self.keys.get(p as usize).copied(), self.mkeys.get(p as usize).copied());
                if let (Some(kp), Some(mp)) = (kp, mp) {
                    if let Some(list) = self.mindex.get(&mp) {
                        let mut best: Vec<(u64, u32, u32)> = Vec::new(); // (full key, frame, node)
                        let pf = tree.nodes[p as usize].st.frame;
                        for &f in list {
                            let fk = self.keys[f as usize];
                            if fk == kp || f == id || !live(tree, f) || (spawn && tree.nodes[f as usize].st.frame + margin > pf) {
                                continue;
                            }
                            let ff = tree.nodes[f as usize].st.frame;
                            match best.iter_mut().find(|b| b.0 == fk) {
                                Some(b) if ff < b.1 => *b = (fk, ff, f),
                                Some(_) => {}
                                None => best.push((fk, ff, f)),
                            }
                        }
                        let h = hot.contains(&id) || hot.contains(&p);
                        for &(_, _, f) in best.iter().take(RELAX_CAP) {
                            q.push((Graft { from: f, child: id, depth, kind: KIND_RELAX, leg: u32::MAX }, h || hot.contains(&f)));
                        }
                    }
                }
            }
        }
        for (g, h) in q {
            self.push(g, h);
        }
        self.mindex.entry(m).or_default().push(id);
    }

    /// Whether graft `g`'s replayed end `st` (cell `cell`) is kept, counting
    /// why not. A duplicate or forward graft must reach the child's key at
    /// least `margin` sooner; a relaxed one the child's movement key, with
    /// progress not below its start's, in no more frames than the child took
    /// from its parent, and no state of its full key at or before it.
    pub fn accept<C: Hash>(&mut self, g: Graft, st: &St, cell: &C, tree: &Tree, margin: u32) -> bool {
        self.accepted = None;
        let child = &tree.nodes[g.child as usize];
        self.stats.k_tried[g.kind as usize] += 1;
        self.stats.k_frames_rejected[g.kind as usize] += child.inputs.len() as u64;
        if g.kind == KIND_LEG || g.kind == KIND_ROUTE {
            return self.leg_accept(g, st, cell, tree);
        }
        if g.kind == KIND_RELAX {
            self.stats.relax_tried += 1;
            if move_key(st) != move_key(&child.st) {
                self.stats.relax_diverged += 1;
                self.stats.diverged += 1;
                return false;
            }
            let (from, parent) = (&tree.nodes[g.from as usize].st, child.parent.map_or(0, |p| tree.nodes[p as usize].st.frame));
            if st.stars < from.stars || st.keys < from.keys {
                self.stats.relax_regress += 1;
                self.stats.diverged += 1;
                return false;
            }
            let longer = match relax_mode() {
                Relax::Spawn => st.frame + margin > child.st.frame,
                _ => st.frame.saturating_sub(from.frame) > child.st.frame.saturating_sub(parent),
            };
            if longer {
                self.stats.relax_longer += 1;
                self.stats.not_faster += 1;
                return false;
            }
            let k = self.key_of(st, cell);
            if self.index.get(&k).is_some_and(|l| l.iter().any(|&i| tree.nodes[i as usize].st.frame <= st.frame)) {
                self.stats.relax_dup += 1;
                self.stats.duplicate += 1;
                return false;
            }
        } else {
            self.stats.fwd_tried += (g.kind == KIND_FORWARD) as u64;
            if self.key_of(st, cell) != self.keys.get(g.child as usize).copied().unwrap_or(u64::MAX) {
                self.stats.diverged += 1;
                return false;
            }
            if st.frame + margin > child.st.frame {
                self.stats.not_faster += 1;
                return false;
            }
        }
        self.accepted = Some(g);
        true
    }

    pub(crate) fn set_accepted(&mut self, g: Graft) {
        self.accepted = Some(g);
    }

    pub(crate) fn full_key_sooner(&self, k: u64, tree: &Tree, frame: u32) -> bool {
        self.index.get(&k).is_some_and(|l| l.iter().any(|&i| tree.nodes[i as usize].st.frame <= frame))
    }

    /// Up to `n` queued grafts, route ones first.
    pub fn take(&mut self, n: usize) -> Vec<Graft> {
        let mut out = Vec::new();
        while out.len() < n {
            match self.hot.pop_front().or_else(|| self.cold.pop_front()) {
                Some(g) => out.push(g),
                None => break,
            }
        }
        out
    }

    /// OBRRT_POLISH: up to `n` queued route grafts (other kinds stay).
    pub fn take_route(&mut self, n: usize) -> Vec<Graft> {
        let mut out = Vec::new();
        for q in [&mut self.hot, &mut self.cold] {
            if !q.iter().any(|g| g.kind == KIND_ROUTE) {
                continue;
            }
            let mut keep = VecDeque::with_capacity(q.len());
            while let Some(g) = q.pop_front() {
                if g.kind == KIND_ROUTE && out.len() < n {
                    out.push(g);
                } else {
                    keep.push_back(g);
                }
            }
            *q = keep;
        }
        out
    }

    pub fn queue_len(&self) -> usize {
        self.hot.len() + self.cold.len()
    }

    /// OBRRT_MEM_DIAG: approximate heap bytes of the big tables.
    pub fn mem_diag(&self) -> String {
        let g = std::mem::size_of::<Graft>();
        let ix: usize = self.index.values().map(|v| v.capacity() * 4).sum::<usize>() + self.index.capacity() * 33;
        let mx: usize = self.mindex.values().map(|v| v.capacity() * 4).sum::<usize>() + self.mindex.capacity() * 33;
        format!(
            "rr seen {} (cap {}, {} MB) index {} keys ({} MB) mindex {} MB keys {} MB hot {}/{} cold {}/{} ({} MB)",
            self.seen.len(), self.seen.capacity(), self.seen.capacity() * 9 / 1_000_000, self.index.len(), ix / 1_000_000, mx / 1_000_000,
            (self.keys.capacity() + self.mkeys.capacity()) * 8 / 1_000_000, self.hot.len(), self.hot.capacity(), self.cold.len(), self.cold.capacity(),
            (self.hot.capacity() + self.cold.capacity()) * g / 1_000_000
        )
    }

    pub fn summary(&self) -> serde_json::Value {
        let s = &self.stats;
        let mut v = serde_json::json!({
            "key": match self.key { RerouteKey::Cell => "cell", RerouteKey::Schema(_) => "schema" },
            "queued": s.queued, "dropped": s.dropped, "queue": self.queue_len(), "tried": s.tried, "ok": s.ok,
            "diverged": s.diverged, "not_faster": s.not_faster, "duplicate": s.duplicate, "died": s.died, "traps": s.traps,
            "frames": s.frames, "frames_saved": s.frames_saved, "max_depth": s.max_depth, "stale": s.stale,
        });
        if forward_mode() != Forward::Off {
            v["forward"] = serde_json::json!(format!("{:?}", forward_mode()).to_lowercase());
            v["fwd"] = serde_json::json!({"queued": s.fwd_queued, "tried": s.fwd_tried, "ok": s.fwd_ok, "saved": s.fwd_saved});
        }
        if s.k_queued[1..].iter().any(|&q| q > 0) {
            let mut kinds = serde_json::Map::new();
            for (i, name) in KIND_NAMES.iter().enumerate() {
                kinds.insert(name.to_string(), serde_json::json!({"queued": s.k_queued[i], "tried": s.k_tried[i], "ok": s.k_ok[i],
                    "frames_kept": s.k_frames_kept[i], "frames_rejected": s.k_frames_rejected[i]}));
            }
            v["by_kind"] = serde_json::Value::Object(kinds);
        }
        if let Some(l) = &self.legs {
            v["legs"] = l.summary();
        }
        if relax_on() {
            v["relax"] = serde_json::json!({"mode": format!("{:?}", relax_mode()).to_lowercase(), "queued": s.relax_queued, "tried": s.relax_tried, "ok": s.relax_ok, "diverged": s.relax_diverged,
                "regress": s.relax_regress, "longer": s.relax_longer, "dup": s.relax_dup, "zone": s.relax_zone});
        }
        v
    }
}
