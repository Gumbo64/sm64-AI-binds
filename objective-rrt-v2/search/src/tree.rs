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

/// OBRRT_STAR_WINDOW=W (default 1): the frontier spans the top W star counts
/// (with the best key count). A state is behind it only when it has fewer
/// stars than best - (W - 1), so with 2 a slower or different k+1th star stays
/// open until some state has k+2: which star comes next is judged by what it
/// leads to. Not with OBRRT_RETIRE_SUBSET, whose sets have their own rule.
pub static STAR_WINDOW: std::sync::atomic::AtomicI16 = std::sync::atomic::AtomicI16::new(1);

/// OBRRT_ZONE_LOCK=1 (per-zone trees of the 120-star plan): the level the
/// tree is locked to (the root's), -1 when off. A state in another level is
/// kept as an exit (its path is a segment out of the zone, with whatever it
/// collected) but is never a start, never on the frontier and never
/// advances it. A Bowser course's arena counts as the course.
pub static ZONE_LOCK: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

/// OBRRT_EXIT_SKIP=<levels>: courses whose entry is not a good exit of the
/// castle (the campaign's list of courses it has given up on for now).
static EXIT_SKIP: std::sync::OnceLock<Vec<u16>> = std::sync::OnceLock::new();

/// Whether `level` is outside the locked zone.
pub fn outside_lock(level: u16) -> bool {
    let l = ZONE_LOCK.load(std::sync::atomic::Ordering::Relaxed);
    if l < 0 || level as i32 == l {
        return false;
    }
    // The castle (inside, grounds, courtyard) is one zone.
    let hub = |x: i32| matches!(x, 6 | 16 | 26);
    if hub(l) && hub(level as i32) {
        return false;
    }
    !matches!((l, level), (17, 30) | (19, 33) | (21, 34))
}

/// The lowest progress still on the frontier when the best is `p`.
fn window_floor(p: (u8, i16)) -> (u8, i16) {
    let w = STAR_WINDOW.load(std::sync::atomic::Ordering::Relaxed);
    (p.0, (p.1 - (w - 1)).max(0).min(p.1))
}

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

/// OBRRT_REPLAY_MAX (sparse saves, objective-scattershot's `replay_max`):
/// a node keeps a save state only when rebuilding it would replay more than
/// this many frames from its nearest saved ancestor (its anchor); the others
/// hold none and are rebuilt by loading the anchor's save and replaying the
/// recorded inputs (`Tree::load_into`). u32::MAX = off (every node saved).
pub static REPLAY_MAX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);
/// OBRRT_ADD (dense adds): the grid of the dense key (`crate::dense::dkey`)
/// every node is indexed by, in units; 0 = off (no index).
pub static DENSE_GRID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// OBRRT_KIN_UNIT=attempt: kinship counts generations in attempts, not
/// nodes. Going up from a node to its parent costs `Tree::gen_step` of the
/// node: 1 for an attempt's result (a segment end, a reroute graft of one,
/// the last node of a zone-share chain), 0 for a dense intermediate ("~d")
/// and the other nodes of a zone-share chain. Off (false): every node is one
/// generation, as before.
pub static KIN_ATTEMPT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// OBRRT_KIN_CAP (attempt unit only): at most this many members per kin set,
/// nearest first (0 = no cap).
pub static KIN_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// OBRRT_KIN_UNIT=frames: kinship generations in game frames of tree-path
/// time: going up from a node to its parent costs its frames since the
/// parent / `KIN_FRAMES` (OBRRT_KIN_FRAMES, 150 frames per generation).
pub static KIN_FRAMES_UNIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static KIN_FRAMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(150);
/// OBRRT_KIN_SOFT=λ (f64 bits; 0 = off): continuous kinship. Instead of the
/// hard set within k generations, every node sharing an ancestor with b
/// within d generations of both (in the unit) has its start weight × λ^d;
/// members below 1e-3 (d > ln 1e-3 / ln λ) are left out. λ = 1: no limit.
pub static KIN_SOFT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Kin-set diagnostics (attempt unit): calls, cache hits, members returned,
/// nodes visited, capped sets, nanoseconds building them.
#[derive(Default)]
pub struct KinStats {
    pub calls: std::sync::atomic::AtomicU64,
    pub hits: std::sync::atomic::AtomicU64,
    pub members: std::sync::atomic::AtomicU64,
    pub visited: std::sync::atomic::AtomicU64,
    pub capped: std::sync::atomic::AtomicU64,
    pub nanos: std::sync::atomic::AtomicU64,
    pub max: std::sync::atomic::AtomicU64,
}

/// Sparse saves: no anchor (the node holds its own save, or none at all).
const SELF: u32 = u32::MAX;

/// Sparse-save bookkeeping (`REPLAY_MAX`), per node and in total.
#[derive(Default)]
pub struct Sparse {
    pub cap: u32,
    /// Per node: whether it holds a save; else its anchor (a saved
    /// ancestor) and the frames from it.
    pub saved: Vec<bool>,
    pub anchor: Vec<u32>,
    pub dist: Vec<u32>,
    /// Per node: live save-less nodes anchored to it. A retired node keeps
    /// its save while any are.
    pub refs: Vec<u32>,
    /// Saves dropped on add (within the cap), kept (beyond it), and nodes
    /// added without a save beyond the cap (their replay is longer).
    pub dropped: u64,
    pub kept: u64,
    pub overshoot: u64,
    pub max_dist: u32,
    /// Saves of retired nodes kept for their dependents.
    pub held: u64,
    /// Rebuilds by `load_into` and the frames they replayed.
    pub rebuilds: std::sync::atomic::AtomicU64,
    pub rebuild_frames: std::sync::atomic::AtomicU64,
}

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
    /// Rate sampler (`Config::sampler_rate`): new states its attempts added
    /// and frames they spent, faded by exp(-Δ/tau) in emulated frames since
    /// `rate_t`. Its start multiplier is this yield per frame against the
    /// tree's, floored (see `Engine::start_mult`), instead of ×state_decay
    /// per barren attempt.
    pub rate_y: f64,
    pub rate_f: f64,
    pub rate_t: u64,
    /// The learned model's view of the state (`OBRRT_MODEL`, model.rs): its
    /// own when its attempt's worker scored it, else its parent's. None
    /// when the model is off.
    pub model: Option<std::sync::Arc<crate::model::NodeModel>>,
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
    /// Route chains (`Config::chain_route`): whether `made_by` was met here
    /// (else the state was kept for progress toward it).
    pub met: bool,
    /// v2: the rolling-horizon controller's plan beyond this state (the best
    /// plan's uncommitted inputs when the segment that made it ended), the
    /// first-generation seed of attempts from here. In memory only: not in
    /// tree.bin, so a resumed run starts without tails.
    pub tail: Vec<GamePad>,
}

pub struct Tree {
    /// OBRRT_ZONE_LOCK: exits of the zone that are worth having (see `add`).
    pub good_exits: u64,
    pub nodes: Vec<Node>,
    pub children: Vec<Vec<u32>>,
    /// The best (keys, stars) reached, and the nodes that have it.
    pub frontier_progress: (u8, i16),
    pub frontier: Vec<u32>,
    pub delta_bytes: usize,
    /// While set, an advance of the frontier doesn't retire the states
    /// behind it (resume: a later record may still need their saves).
    pub hold_retire: bool,
    /// OBRRT_POLISH: nodes kept loadable when retired (save, anchor and
    /// objects stay; they are still not starts for ordinary plans): the best
    /// route, polish results and route-graft chains (see `polish`).
    pub pinned: std::collections::HashSet<u32>,
    /// Pin the nodes `add` makes while set.
    pub pin_next: bool,
    /// OBRRT_POLISH: a frontier advance pins the new frontier's ancestors
    /// before retiring what is behind it (the route there stays loadable).
    pub pin_routes: bool,
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
    /// OBRRT_REPLAY_MAX: sparse saves (None: every node keeps its save).
    pub sparse: Option<Sparse>,
    /// Dense adds: the earliest frame per dense key (`crate::dense::dkey`)
    /// over every node added, and the key's grid (0: not kept).
    pub dkeys: std::collections::HashMap<u64, u32>,
    pub dgrid: f32,
    /// Bytes of the object lists nodes hold (reported; see `mem_bytes`).
    pub obj_bytes: usize,
    /// OBRRT_KIN_UNIT=attempt: per node, the generations going up from it
    /// to its parent costs (see `KIN_ATTEMPT`); set on add from `via`
    /// (dense "~d": 0, else 1), adjusted by the callers for replay chains.
    pub gen_step: Vec<bool>,
    /// Attempt-unit kin sets by (b, k), valid while the node count is the
    /// first field (the tree only changes by adding nodes).
    kin_cache: std::sync::Mutex<(usize, std::collections::HashMap<(u32, u32), std::sync::Arc<Vec<u32>>>)>,
    /// Frames-unit / soft kin sets with weights, cached the same way.
    kin_wcache: std::sync::Mutex<(usize, std::collections::HashMap<(u32, u32), std::sync::Arc<Vec<(u32, f32)>>>)>,
    pub kin_stats: KinStats,
    /// Earliest arrivals near points (obrrt_core::records), when the
    /// generator wants them ([`Tree::enable_records`]): every node, as added.
    pub records: Option<std::sync::Arc<std::sync::RwLock<obrrt_core::records::Records>>>,
}

/// Bytes of `st`'s object list.
fn obj_bytes(st: &St) -> usize {
    st.objects.as_ref().map_or(0, |o| o.len() * std::mem::size_of::<obrrt_core::Obj>())
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
                met: false,
                tail: Vec::new(),
            }],
            children: vec![Vec::new()],
            frontier_progress: progress,
            good_exits: 0,
            frontier: vec![0],
            delta_bytes: 0,
            hold_retire: false,
            pinned: Default::default(),
            pin_next: false,
            pin_routes: false,
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
            sparse: None,
            dkeys: Default::default(),
            dgrid: 0.0,
            obj_bytes: 0,
            gen_step: vec![true],
            kin_cache: Default::default(),
            kin_wcache: Default::default(),
            kin_stats: Default::default(),
            records: None,
        }
        .with_root_set()
        .with_sparse()
    }

    fn with_sparse(mut self) -> Tree {
        let cap = REPLAY_MAX.load(std::sync::atomic::Ordering::Relaxed);
        if cap != u32::MAX {
            self.sparse = Some(Sparse { cap, saved: vec![true], anchor: vec![SELF], dist: vec![0], refs: vec![0], ..Default::default() });
        }
        let g = DENSE_GRID.load(std::sync::atomic::Ordering::Relaxed);
        if g > 0 {
            self.dgrid = g as f32;
            let st = &self.nodes[0].st;
            self.dkeys.insert(crate::dense::dkey(st, self.dgrid), st.frame);
        }
        self.obj_bytes = obj_bytes(&self.nodes[0].st);
        self
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
            st.progress() >= window_floor(self.frontier_progress)
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
        let ids: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| {
                let n = &self.nodes[i as usize];
                !n.retired && !keys.contains(&set_key(&n.st)) && keep.binary_search(&i).is_err()
            })
            .collect();
        for i in ids {
            self.retire_at(i);
        }
    }

    /// Builds the records (obrrt_core::records) from the nodes so far; from
    /// then on every added node goes in too. Returns them, to install.
    pub fn enable_records(&mut self) -> std::sync::Arc<std::sync::RwLock<obrrt_core::records::Records>> {
        let mut r = obrrt_core::records::Records::default();
        for n in &self.nodes {
            r.insert(n.st.zone(), crate::speed::ctx(&n.st), n.st.pos, n.st.frame);
        }
        let r = std::sync::Arc::new(std::sync::RwLock::new(r));
        self.records = Some(r.clone());
        r
    }

    /// Bytes of the records (0 when off).
    pub fn records_bytes(&self) -> usize {
        self.records.as_ref().map_or(0, |r| r.read().unwrap().bytes())
    }

    /// Adds a node; returns its id and whether it advanced the frontier.
    pub fn add(&mut self, parent: u32, inputs: Vec<GamePad>, st: St, save: Delta, via: &'static str) -> (u32, bool) {
        let id = self.nodes.len() as u32;
        if let Some(r) = &self.records {
            r.write().unwrap().insert(st.zone(), crate::speed::ctx(&st), st.pos, st.frame);
        }
        let progress = st.progress();
        // An exit is a state Mario can act from: the menus on the way out or in
        // (star select, the save prompt) stay inside, so they can be answered.
        let outside = outside_lock(st.level) && st.controllable() && !st.modal() && !st.in_dialog();
        self.delta_bytes += save.bytes();
        let depth = self.nodes[parent as usize].depth + 1;
        obrrt_core::hooks::node_added(id, parent, &st);
        self.nodes.push(Node {
            parent: Some(parent),
            depth,
            inputs,
            st,
            save,
            via,
            stats: NodeStats { backoff: 1.0, model: self.nodes[parent as usize].stats.model.clone(), ..Default::default() },
            retired: false,
            made_by: None,
            dominated: false,
            met: false,
            tail: Vec::new(),
        });
        self.children.push(Vec::new());
        self.children[parent as usize].push(id);
        if self.pin_next {
            self.pinned.insert(id);
        }
        self.gen_step.push(!via.ends_with("~d"));
        self.note_added(id, parent);
        self.wip.push(if self.keep_wip { self.nodes[id as usize].st.wip() } else { 0 });
        if outside {
            // An exit of the locked zone: recorded, then retired at once. A
            // good one: more progress than the root's (a course), or any
            // course entered (the castle).
            let l = ZONE_LOCK.load(std::sync::atomic::Ordering::Relaxed);
            let skip = EXIT_SKIP.get_or_init(|| obrrt_core::sys::env::var("OBRRT_EXIT_SKIP").map_or(Vec::new(), |v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()));
            let level = self.nodes[id as usize].st.level;
            // A cap-switch course (TotWC, CotMC, VCutM) whose switch the root
            // hasn't pressed: only an exit with it pressed is good. The star
            // there ends the visit, and campaign t1 left the Tower of the
            // Wing Cap with its star and no wing cap (step 95, seen by the
            // user in the replay, 2026-11-10).
            let cap_bit = match l {
                29 => 0x2u32,
                28 => 0x4,
                18 => 0x8,
                _ => 0,
            };
            let need = cap_bit & !self.nodes[0].st.save_flags;
            // A course tree entering a cap-switch course with its switch
            // unpressed (and not skipped) is good as well: the Cavern of the
            // Metal Cap is entered from HMC, not from the castle, and a cap is a
            // precondition of stars elsewhere.
            let into_cap = match level {
                29 => 0x2u32,
                28 => 0x4,
                18 => 0x8,
                _ => 0,
            } & !self.nodes[id as usize].st.save_flags
                != 0
                && i32::from(level) != l
                && !skip.contains(&level);
            // OBRRT_ANY_EXIT=1: any way out of a course is good. The campaign's tree
            // after its coin legs starts inside the course with the 100-coin star had
            // (that star doesn't end the visit); it only has to leave.
            static ANY_EXIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let any = *ANY_EXIT.get_or_init(|| obrrt_core::sys::env::var("OBRRT_ANY_EXIT").is_ok_and(|v| v == "1"));
            let good = if need != 0 { self.nodes[id as usize].st.save_flags & need != 0 } else { progress > self.nodes[0].st.progress() || into_cap || any };
            // OBRRT_FINAL=1: the run's last search (Bowser in the Sky, Bowser 3). Only the grand star counts: it
            // starts the ending, which leaves the arena for the castle grounds (level 16; the reference movie:
            // 34 -> 16, Peach's scene, then the credits). The grand star raises no star or key count, and a fall
            // or a death leaves for BitS or the castle, so no other exit is good.
            static FINAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let good = if *FINAL.get_or_init(|| obrrt_core::sys::env::var("OBRRT_FINAL").is_ok_and(|v| v == "1")) { level == 16 } else { good };
            // The castle: a course entered that isn't skipped, and nothing else. An exit
            // into a skipped course with a castle star had on the way (Toad's) used to
            // count by its progress: campaign t1 step 166, sent to the Tower of the
            // Wing Cap alone, stopped at 9.9 M frames on an entry into HMC with 28 stars.
            // OBRRT_EXIT_NOSTOP=1: no exit stops the run (the campaign's coin legs:
            // an exit with a star ended a CCM leg at 18.6 M of 25 M, sc1 step 21;
            // only the 100-coin star, by OBRRT_STOP_PROGRESS, should).
            static NOSTOP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let nostop = *NOSTOP.get_or_init(|| obrrt_core::sys::env::var("OBRRT_EXIT_NOSTOP").is_ok_and(|v| v == "1"));
            if !nostop && if matches!(l, 6 | 16 | 26) { !skip.contains(&level) } else { good } {
                self.good_exits += 1;
            }
            self.retire_at(id);
            return (id, false);
        }
        // OBRRT_STOP_PROGRESS=1 (with OBRRT_ZONE_LOCK and OBRRT_STOP_EXIT_GRACE):
        // a state inside the zone with more progress than the root's counts as a
        // good exit, so the run stops on it: the castle's own stars (Toad's,
        // MIPS's) are had without leaving it, and a castle tree only ever
        // stopped on a course entered. Only a state in control counts: the next
        // tree starts from it, and one that ends in the star's dance or at its
        // save prompt never stands still at boot (t1 step 177: "Mario didn't
        // stand still after booting").
        // OBRRT_STOP_FLAG=<hex>: likewise for save flags the root lacks (the campaign's moat step: 0x200,
        // SAVE_FLAG_MOAT_DRAINED, set by ground-pounding both basement pillars; it raises no star count).
        static STOP_FLAG: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let want = *STOP_FLAG.get_or_init(|| obrrt_core::sys::env::var("OBRRT_STOP_FLAG").ok().and_then(|v| u32::from_str_radix(v.trim_start_matches("0x"), 16).ok()).unwrap_or(0));
        if want != 0
            && self.nodes[id as usize].st.save_flags & want & !self.nodes[0].st.save_flags != 0
            && self.nodes[id as usize].st.controllable()
        {
            self.good_exits += 1;
        }
        static STOP_PROGRESS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *STOP_PROGRESS.get_or_init(|| obrrt_core::sys::env::var("OBRRT_STOP_PROGRESS").is_ok_and(|v| v == "1"))
            && progress > self.nodes[0].st.progress()
            && self.nodes[id as usize].st.controllable()
        {
            self.good_exits += 1;
        }
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
            let floor = window_floor(progress);
            let nodes = &self.nodes;
            self.frontier.retain(|&i| nodes[i as usize].st.progress() >= floor);
            if self.keep_wip {
                // The new frontier state first: the wip comparison needs it.
                self.frontier.push(id);
                if !self.hold_retire {
                    self.retire(floor);
                }
                return (id, advanced);
            }
            if !self.hold_retire {
                self.retire(floor);
            }
        }
        if progress >= window_floor(self.frontier_progress) {
            self.frontier.push(id);
        } else if self.keep_wip && !self.hold_retire {
            self.wip_note(id);
        }
        (id, advanced)
    }

    /// Sparse saves, dense keys and object bytes for the node just added.
    fn note_added(&mut self, id: u32, parent: u32) {
        self.obj_bytes += obj_bytes(&self.nodes[id as usize].st);
        if self.dgrid > 0.0 {
            let st = &self.nodes[id as usize].st;
            let k = crate::dense::dkey(st, self.dgrid);
            let f = st.frame;
            let e = self.dkeys.entry(k).or_insert(f);
            *e = (*e).min(f);
        }
        if self.sparse.is_none() {
            return;
        }
        let up = self.anchor_for(parent);
        let len = self.nodes[id as usize].inputs.len() as u32;
        let sp = self.sparse.as_mut().unwrap();
        let own = !self.nodes[id as usize].save.is_none();
        let (saved, anchor, dist) = match up {
            Some((a, d)) if !(own && d + len > sp.cap) => {
                if own {
                    let n = &mut self.nodes[id as usize];
                    self.delta_bytes -= n.save.bytes();
                    n.save = Delta::none();
                    sp.dropped += 1;
                } else if d + len > sp.cap {
                    sp.overshoot += 1;
                }
                sp.refs[a as usize] += 1;
                sp.max_dist = sp.max_dist.max(d + len);
                (false, a, d + len)
            }
            _ if own => {
                sp.kept += 1;
                (true, SELF, 0)
            }
            // No save and nothing to rebuild from: callers check `can_anchor`.
            _ => (false, SELF, 0),
        };
        sp.saved.push(saved);
        sp.anchor.push(anchor);
        sp.dist.push(dist);
        sp.refs.push(0);
    }

    /// Retires every state behind the frontier (after `hold_retire`).
    pub fn retire_behind(&mut self) {
        self.pin_frontier_routes();
        if self.subset {
            return self.retire_subset();
        }
        self.retire(window_floor(self.frontier_progress));
    }

    /// Frees the save state and object list of every state behind `progress`:
    /// with b drawn from the frontier and starts never behind b, they can't be
    /// used again. Their inputs stay, for replaying descendants.
    fn retire(&mut self, progress: (u8, i16)) {
        self.pin_frontier_routes();
        let keep = self.wip_select(|t, n| t.nodes[n as usize].st.progress() < progress);
        let ids: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| {
                let n = &self.nodes[i as usize];
                n.st.progress() < progress && !n.retired && keep.binary_search(&i).is_err()
            })
            .collect();
        for i in ids {
            self.retire_at(i);
        }
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
        let n = &self.nodes[id as usize];
        if n.retired || id == 0 {
            return;
        }
        self.retire_at(id);
    }

    /// Retires `i`: frees its save and objects (without sparse saves, the
    /// save becomes the default delta as before). With sparse saves a node
    /// still anchoring live save-less nodes keeps its save until they go;
    /// a save-less node releases its anchor.
    fn retire_at(&mut self, i: u32) {
        if self.pinned.contains(&i) {
            self.nodes[i as usize].retired = true;
            return;
        }
        self.free_retired(i);
    }

    /// OBRRT_POLISH: pins the ancestors of the frontier's soonest state of
    /// its best progress (the best route).
    fn pin_frontier_routes(&mut self) {
        if !self.pin_routes {
            return;
        }
        let best = self.frontier.iter().copied().min_by_key(|&i| {
            let st = &self.nodes[i as usize].st;
            (std::cmp::Reverse((st.stars, st.keys)), st.frame, i)
        });
        let mut at = best;
        while let Some(i) = at {
            self.pinned.insert(i);
            at = self.nodes[i as usize].parent;
        }
    }

    /// Whether `id` is pinned (`pinned`).
    pub fn is_pinned(&self, id: u32) -> bool {
        self.pinned.contains(&id)
    }

    /// Unpins `id`; if it was retired meanwhile, frees it now.
    pub fn unpin(&mut self, id: u32) {
        if self.pinned.remove(&id) && self.nodes[id as usize].retired {
            self.free_retired(id);
        }
    }

    /// The freeing part of `retire_at`.
    fn free_retired(&mut self, i: u32) {
        let n = &mut self.nodes[i as usize];
        self.obj_bytes -= obj_bytes(&n.st);
        n.st.objects = None;
        n.retired = true;
        n.tail = Vec::new();
        let Some(sp) = self.sparse.as_mut() else {
            self.delta_bytes -= n.save.bytes();
            n.save = Delta::default();
            return;
        };
        let iu = i as usize;
        if sp.saved[iu] {
            if sp.refs[iu] == 0 {
                self.delta_bytes -= n.save.bytes();
                n.save = Delta::none();
                sp.saved[iu] = false;
            } else {
                sp.held += 1;
            }
            return;
        }
        let a = sp.anchor[iu];
        if a != SELF && sp.refs[a as usize] > 0 {
            sp.refs[a as usize] -= 1;
            if sp.refs[a as usize] == 0 && self.nodes[a as usize].retired && sp.saved[a as usize] && !self.pinned.contains(&a) {
                let an = &mut self.nodes[a as usize];
                self.delta_bytes -= an.save.bytes();
                an.save = Delta::none();
                sp.saved[a as usize] = false;
            }
        }
    }

    /// Sparse saves: the saved node `id` can be rebuilt from, and the frames
    /// from it (`id` itself and 0 when it holds a save); None when neither
    /// it nor its anchor holds one (retired and freed).
    pub fn anchor_for(&self, id: u32) -> Option<(u32, u32)> {
        let Some(sp) = self.sparse.as_ref() else { return Some((id, 0)) };
        let i = id as usize;
        if sp.saved[i] {
            return Some((id, 0));
        }
        let a = sp.anchor[i];
        (a != SELF && sp.saved[a as usize]).then(|| (a, sp.dist[i]))
    }

    /// Whether a save-less child of `parent` could be rebuilt (always
    /// without sparse saves).
    pub fn can_anchor(&self, parent: u32) -> bool {
        self.anchor_for(parent).is_some()
    }

    /// Frames from `id`'s anchor (0 when it holds a save or saves aren't sparse).
    pub fn dist(&self, id: u32) -> u32 {
        self.anchor_for(id).map_or(0, |a| a.1)
    }

    /// The base index `id`'s state is packed against (its anchor's).
    pub fn base_of(&self, id: u32) -> u16 {
        let a = self.anchor_for(id).map_or(id, |a| a.0);
        let b = self.nodes[a as usize].save.base;
        if b == u16::MAX { 0 } else { b }
    }

    /// Whether `id` holds a save of its own.
    pub fn has_save(&self, id: u32) -> bool {
        self.sparse.as_ref().map_or(true, |sp| sp.saved[id as usize])
    }

    /// Loads node `id`'s state into `game`: its save, or (sparse saves) its
    /// anchor's save and the recorded inputs from there. Returns the frames
    /// replayed. Without sparse saves exactly the old `save.apply` load.
    pub fn load_into(&self, id: u32, game: &mut crate::game::Game, bases: &[Vec<u8>]) -> anyhow::Result<u32> {
        let Some(sp) = self.sparse.as_ref() else {
            game.load_state(&self.nodes[id as usize].save.apply(bases)?)?;
            return Ok(0);
        };
        let (a, d) = self.anchor_for(id).ok_or_else(|| anyhow::anyhow!("node {id} has no save and no saved anchor"))?;
        game.load_state(&self.nodes[a as usize].save.apply(bases)?)?;
        if a == id {
            return Ok(0);
        }
        let mut chain = Vec::new();
        let mut at = id;
        while at != a {
            chain.push(at);
            at = self.nodes[at as usize].parent.ok_or_else(|| anyhow::anyhow!("anchor {a} not above {id}"))?;
        }
        for &n in chain.iter().rev() {
            for p in &self.nodes[n as usize].inputs {
                game.step_game(*p)?;
            }
        }
        sp.rebuilds.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        sp.rebuild_frames.fetch_add(d as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(d)
    }

    /// Dense adds: whether no node with dense key `key` was reached at or
    /// before `frame`.
    pub fn dense_novel(&self, key: u64, frame: u32) -> bool {
        self.dkeys.get(&key).is_none_or(|&f| frame < f)
    }

    /// Bytes held per node in total: saves, object lists, inputs and the
    /// node records themselves (approximate).
    pub fn mem_bytes(&self) -> usize {
        let inputs: usize = self.nodes.iter().map(|n| n.inputs.capacity() * std::mem::size_of::<GamePad>()).sum();
        self.delta_bytes + self.obj_bytes + inputs + self.nodes.len() * std::mem::size_of::<Node>() + self.records_bytes()
    }

    /// Ids of `id` and its ancestors, nearest first, up to `k` generations up
    /// (in attempts with OBRRT_KIN_UNIT=attempt, see `KIN_ATTEMPT`).
    pub fn ancestors(&self, id: u32, k: u32) -> Vec<u32> {
        if KIN_ATTEMPT.load(std::sync::atomic::Ordering::Relaxed) {
            self.ancestors_attempt(id, k)
        } else {
            self.ancestors_node(id, k)
        }
    }

    /// Every node sharing an ancestor with `b` at most `k` generations above
    /// both (nodes, or attempts with OBRRT_KIN_UNIT=attempt), `b` included.
    pub fn kin_set(&self, b: u32, k: u32) -> Vec<u32> {
        if Self::kin_generic_on() {
            return self.kin_weighted_cached(b, k).iter().map(|x| x.0).collect();
        }
        if KIN_ATTEMPT.load(std::sync::atomic::Ordering::Relaxed) {
            self.kin_set_attempt_cached(b, k).as_ref().clone()
        } else {
            self.kin_set_node(b, k)
        }
    }

    /// Whether `a` and `b` share an ancestor at most `k` generations above
    /// each (`b_ancestors` from `ancestors(b, k)`).
    pub fn kin(&self, a: u32, b_ancestors: &[u32], k: u32) -> bool {
        if KIN_ATTEMPT.load(std::sync::atomic::Ordering::Relaxed) {
            self.kin_attempt(a, b_ancestors, k)
        } else {
            self.kin_node(a, b_ancestors, k)
        }
    }

    /// Whether the frames unit or soft kinship is on (`kin_generic`).
    fn kin_generic_on() -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        KIN_FRAMES_UNIT.load(Relaxed) || KIN_SOFT.load(Relaxed) != 0
    }

    /// `kin_set` with each member's start-weight multiplier (1 except with
    /// OBRRT_KIN_SOFT), for `Engine::pick_start`.
    pub fn kin_weighted(&self, b: u32, k: u32) -> std::sync::Arc<Vec<(u32, f32)>> {
        if Self::kin_generic_on() {
            return self.kin_weighted_cached(b, k);
        }
        let v = if KIN_ATTEMPT.load(std::sync::atomic::Ordering::Relaxed) {
            self.kin_set_attempt_cached(b, k).iter().map(|&i| (i, 1.0)).collect()
        } else {
            self.kin_set_node(b, k).into_iter().map(|i| (i, 1.0)).collect()
        };
        std::sync::Arc::new(v)
    }

    /// `kin_generic`, cached per (b, k) until a node is added.
    fn kin_weighted_cached(&self, b: u32, k: u32) -> std::sync::Arc<Vec<(u32, f32)>> {
        use std::sync::atomic::Ordering::Relaxed;
        let ks = &self.kin_stats;
        ks.calls.fetch_add(1, Relaxed);
        let n = self.nodes.len();
        {
            let mut c = self.kin_wcache.lock().unwrap();
            if c.0 != n {
                c.0 = n;
                c.1.clear();
            }
            if let Some(v) = c.1.get(&(b, k)) {
                ks.hits.fetch_add(1, Relaxed);
                ks.members.fetch_add(v.len() as u64, Relaxed);
                return v.clone();
            }
        }
        let t = obrrt_core::sys::Instant::now();
        let v = std::sync::Arc::new(self.kin_generic(b, k));
        ks.nanos.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        ks.members.fetch_add(v.len() as u64, Relaxed);
        ks.max.fetch_max(v.len() as u64, Relaxed);
        let mut c = self.kin_wcache.lock().unwrap();
        if c.0 == n {
            if c.1.len() >= 4096 {
                c.1.clear();
            }
            c.1.insert((b, k), v.clone());
        }
        v
    }

    /// The generations going up from `id` to its parent costs in the
    /// current unit: frames since the parent / KIN_FRAMES (frames unit),
    /// `gen_step` (attempt unit), else 1.
    fn step_cost(&self, id: u32, frames: bool, attempt: bool, per: f64) -> f64 {
        if frames {
            let n = &self.nodes[id as usize];
            let pf = n.parent.map_or(n.st.frame, |p| self.nodes[p as usize].st.frame);
            n.st.frame.saturating_sub(pf) as f64 / per
        } else if attempt {
            self.step(id) as f64
        } else {
            1.0
        }
    }

    /// Kin set in any unit, hard (within `k`) or soft (OBRRT_KIN_SOFT=λ:
    /// within ln 1e-3 / ln λ, weight λ^d), with each member's weight. A
    /// member's d is the larger of b's and its own distance to their
    /// nearest common ancestor (the first ancestor of b whose search finds
    /// it). The same search as `kin_set_attempt`: ancestors nearest first,
    /// each one's descendants by 0-1 BFS order (a zero-cost child next),
    /// the child toward b skipped; deterministic, each node visited once.
    pub fn kin_generic(&self, b: u32, k: u32) -> Vec<(u32, f32)> {
        use std::sync::atomic::Ordering::Relaxed;
        let frames = KIN_FRAMES_UNIT.load(Relaxed);
        let attempt = KIN_ATTEMPT.load(Relaxed);
        let per = KIN_FRAMES.load(Relaxed).max(1) as f64;
        let lam = f64::from_bits(KIN_SOFT.load(Relaxed));
        let soft = lam > 0.0;
        let limit = if !soft {
            k as f64
        } else if lam >= 1.0 {
            f64::INFINITY
        } else {
            (1e-3f64).ln() / lam.ln()
        };
        let weight = |d: f64| if soft { lam.powf(d) as f32 } else { 1.0 };
        let cap = match KIN_CAP.load(Relaxed) {
            0 => usize::MAX,
            c => c,
        };
        // b's ancestors within the limit, with their distance up from b.
        let mut anc = vec![(b, 0.0f64)];
        let (mut at, mut d) = (b, 0.0f64);
        while let Some(p) = self.nodes[at as usize].parent {
            d += self.step_cost(at, frames, attempt, per);
            if d > limit {
                break;
            }
            anc.push((p, d));
            at = p;
        }
        let mut out = Vec::new();
        let mut dq: std::collections::VecDeque<(u32, f64)> = std::collections::VecDeque::new();
        let mut visited = 0u64;
        'outer: for (j, &(a, up)) in anc.iter().enumerate() {
            let skip = if j > 0 { anc[j - 1].0 } else { u32::MAX };
            dq.clear();
            dq.push_back((a, 0.0));
            while let Some((n, d)) = dq.pop_front() {
                out.push((n, weight(up.max(d))));
                if out.len() >= cap {
                    self.kin_stats.capped.fetch_add(1, Relaxed);
                    break 'outer;
                }
                for &c in &self.children[n as usize] {
                    if c == skip {
                        continue;
                    }
                    visited += 1;
                    let w = self.step_cost(c, frames, attempt, per);
                    if d + w > limit {
                        continue;
                    }
                    if w == 0.0 {
                        dq.push_front((c, d));
                    } else {
                        dq.push_back((c, d + w));
                    }
                }
            }
        }
        self.kin_stats.visited.fetch_add(visited, Relaxed);
        out
    }

    /// The generations going up from `id` to its parent costs (attempt unit).
    #[inline]
    fn step(&self, id: u32) -> u32 {
        self.gen_step.get(id as usize).copied().unwrap_or(true) as u32
    }

    /// `ancestors` in attempts: `id` and every ancestor whose weighted
    /// distance up (the sum of `gen_step` of the nodes left behind) is at
    /// most `k`, nearest first.
    pub fn ancestors_attempt(&self, id: u32, k: u32) -> Vec<u32> {
        let mut out = vec![id];
        let (mut at, mut d) = (id, 0u32);
        while let Some(p) = self.nodes[at as usize].parent {
            d += self.step(at);
            if d > k {
                break;
            }
            out.push(p);
            at = p;
        }
        out
    }

    /// `kin_set` in attempts, cached per (b, k) until a node is added.
    pub fn kin_set_attempt_cached(&self, b: u32, k: u32) -> std::sync::Arc<Vec<u32>> {
        use std::sync::atomic::Ordering::Relaxed;
        let ks = &self.kin_stats;
        ks.calls.fetch_add(1, Relaxed);
        let n = self.nodes.len();
        {
            let mut c = self.kin_cache.lock().unwrap();
            if c.0 != n {
                c.0 = n;
                c.1.clear();
            }
            if let Some(v) = c.1.get(&(b, k)) {
                ks.hits.fetch_add(1, Relaxed);
                ks.members.fetch_add(v.len() as u64, Relaxed);
                return v.clone();
            }
        }
        let t = obrrt_core::sys::Instant::now();
        let v = std::sync::Arc::new(self.kin_set_attempt(b, k));
        ks.nanos.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        ks.members.fetch_add(v.len() as u64, Relaxed);
        ks.max.fetch_max(v.len() as u64, Relaxed);
        let mut c = self.kin_cache.lock().unwrap();
        if c.0 == n {
            if c.1.len() >= 4096 {
                c.1.clear();
            }
            c.1.insert((b, k), v.clone());
        }
        v
    }

    /// `kin_set` in attempts: for each ancestor `a` of `b` within `k`
    /// (nearest first), the descendants of `a` within weighted depth `k`
    /// below it, by 0-1 BFS (a child costing 0 is taken next, one costing 1
    /// after the rest of its distance), so nearer kin come first. The order
    /// depends only on the tree. As in `kin_set_node`, the child on the way
    /// to `b` is skipped (its own search, from one nearer ancestor, already
    /// covered at least as deep below it), which makes the searches disjoint:
    /// each node is visited once. With `KIN_CAP` the set stops at that many.
    pub fn kin_set_attempt(&self, b: u32, k: u32) -> Vec<u32> {
        use std::sync::atomic::Ordering::Relaxed;
        let cap = match KIN_CAP.load(Relaxed) {
            0 => usize::MAX,
            c => c,
        };
        let mut out = Vec::new();
        let anc = self.ancestors_attempt(b, k);
        let mut dq: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();
        let mut visited = 0u64;
        'outer: for (j, &a) in anc.iter().enumerate() {
            let skip = if j > 0 { anc[j - 1] } else { u32::MAX };
            dq.clear();
            dq.push_back((a, 0));
            while let Some((n, d)) = dq.pop_front() {
                out.push(n);
                if out.len() >= cap {
                    self.kin_stats.capped.fetch_add(1, Relaxed);
                    break 'outer;
                }
                for &c in &self.children[n as usize] {
                    if c == skip {
                        continue;
                    }
                    visited += 1;
                    let w = self.step(c);
                    if d + w > k {
                        continue;
                    }
                    if w == 0 {
                        dq.push_front((c, d));
                    } else {
                        dq.push_back((c, d + 1));
                    }
                }
            }
        }
        self.kin_stats.visited.fetch_add(visited, Relaxed);
        out
    }

    /// `kin` in attempts: whether some ancestor of `a` within weighted
    /// distance `k` is in `b_ancestors` (from `ancestors_attempt(b, k)`).
    pub fn kin_attempt(&self, a: u32, b_ancestors: &[u32], k: u32) -> bool {
        let (mut at, mut d) = (a, 0u32);
        loop {
            if b_ancestors.contains(&at) {
                return true;
            }
            match self.nodes[at as usize].parent {
                Some(p) => {
                    d += self.step(at);
                    if d > k {
                        return false;
                    }
                    at = p;
                }
                None => return false,
            }
        }
    }

    /// One line on the attempt-unit kin sets (None when unused).
    pub fn kin_line(&self) -> Option<String> {
        use std::sync::atomic::Ordering::Relaxed;
        let s = &self.kin_stats;
        let calls = s.calls.load(Relaxed);
        if calls == 0 {
            return None;
        }
        let built = calls - s.hits.load(Relaxed);
        Some(format!(
            "kin sets ({}): {calls} calls, {} cached, mean {:.0} members (max {}), {:.0} visited per build, {} capped, {:.2}s building ({:.1} us per build); {} of {} nodes are attempt boundaries",
            {
                let unit = if KIN_FRAMES_UNIT.load(Relaxed) { format!("frames / {}", KIN_FRAMES.load(Relaxed)) } else if KIN_ATTEMPT.load(Relaxed) { "attempt unit".to_string() } else { "node unit".to_string() };
                let l = f64::from_bits(KIN_SOFT.load(Relaxed));
                if l > 0.0 { format!("{unit}, soft {l}") } else { unit }
            },
            s.hits.load(Relaxed),
            s.members.load(Relaxed) as f64 / calls as f64,
            s.max.load(Relaxed),
            s.visited.load(Relaxed) as f64 / built.max(1) as f64,
            s.capped.load(Relaxed),
            s.nanos.load(Relaxed) as f64 * 1e-9,
            s.nanos.load(Relaxed) as f64 * 1e-3 / built.max(1) as f64,
            self.gen_step.iter().filter(|&&g| g).count(),
            self.gen_step.len(),
        ))
    }

    /// Ids of `id` and its ancestors, nearest first, up to `k` generations up.
    pub fn ancestors_node(&self, id: u32, k: u32) -> Vec<u32> {
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
    pub fn kin_set_node(&self, b: u32, k: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let anc = self.ancestors_node(b, k);
        for (j, &a) in anc.iter().enumerate() {
            // Descendants of `a` at most k generations below it. The child
            // on the way to `b` (the previous ancestor) is skipped: its BFS,
            // one level deeper, already found everything below it, in the
            // same order (only unseen nodes are pushed), so the result is
            // the same; with a large k it saves walking those subtrees again.
            let skip = if j > 0 { anc[j - 1] } else { u32::MAX };
            let mut level = vec![a];
            for _ in 0..=k {
                let mut next = Vec::new();
                for n in level {
                    if seen.insert(n) {
                        out.push(n);
                    }
                    if n == a {
                        next.extend(self.children[n as usize].iter().copied().filter(|&c| c != skip));
                    } else {
                        next.extend_from_slice(&self.children[n as usize]);
                    }
                }
                level = next;
            }
        }
        out
    }

    /// Whether `a` and `b` share an ancestor at most `k` generations above
    /// each (k = 4 allows third cousins).
    pub fn kin_node(&self, a: u32, b_ancestors: &[u32], k: u32) -> bool {
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
