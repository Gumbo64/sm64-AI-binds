//! Dense adds (`OBRRT_ADD=last_gen`, on by default): more of the states an attempt
//! samples become nodes, not just the GA winner's segment ends. For every rollout of the
//! last generation, its final state is a candidate.
//!
//! Only states that are new by the *dense key* are kept: (level, area,
//! stars, keys, coins, position on a [`GRID`]-unit grid (64),
//! open dialog or menu, red coins, holding, collected set). It is the duplicate check's cell
//! without the object part of `St::fingerprint` (objects loading and unloading as Mario runs
//! would make every frame a new cell), so with a grid of 64 a state new by
//! the dense key is new by the cell too. A candidate is kept only if no node
//! (of any kind) and no earlier candidate of the attempt reached its key at
//! the same or an earlier frame ("keep the sooner"); uncontrollable states
//! (cutscenes, dialogs, warps) are not candidates.
//!
//! The worker filters against the tree's key index as it was at the round's
//! start (read only: deterministic), and `Engine::merge` checks again, in
//! order, against the tree as it is. Candidates of one rollout form a chain
//! under the attempt's start: each one's parent is the previous kept state of
//! its rollout, its inputs the rollout's pads between them. They hold a save
//! only when needed (sparse saves, 150 frames of replay at most).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use obrrt_core::{GamePad, St};

use crate::store::Delta;

/// The dense key's grid, game units.
pub const GRID: f32 = 64.0;

/// The dense key of `st` on a `grid`-unit grid (see the module doc).
pub fn dkey(st: &St, grid: f32) -> u64 {
    let c = |v: f32| (v / grid).floor() as i32;
    let modal = if st.in_dialog() { st.dialog } else if st.in_save_menu() { -2 } else if st.act_select { -3 } else { -1 };
    let set = st.progress_set();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(
        &(st.level, st.area, st.stars, st.keys, st.coins, c(st.pos[0]), c(st.pos[1]), c(st.pos[2]), modal, st.red_coins, st.holding, set),
        &mut h,
    );
    std::hash::Hasher::finish(&h)
}

/// Whether `st` may be a dense candidate at all.
pub fn eligible(st: &St, start: &St) -> bool {
    st.controllable() && st.alive() && st.lives >= start.lives
}

/// A kept dense candidate, as the worker returns it.
pub struct Cand {
    /// Which rollout chain (in the attempt's order).
    pub chain: u32,
    /// Inputs from the previous candidate of its chain (or the start).
    pub inputs: Vec<GamePad>,
    /// With objects.
    pub st: St,
    pub save: Option<Delta>,
    pub key: u64,
}

/// An attempt's dense candidates and what collecting them cost.
#[derive(Default)]
pub struct Out {
    pub cands: Vec<Cand>,
    pub stats: Stats,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Stats {
    /// States looked at, new against the tree's index, beaten by a sooner
    /// candidate of the same attempt, returned.
    pub seen: u64,
    pub novel: u64,
    pub beaten: u64,
    pub returned: u64,
    /// Saves taken for candidates, and frames replayed to read candidates'
    /// objects and saves (counted in the attempt's frames).
    pub saves: u64,
    pub replay_frames: u64,
}

impl Stats {
    pub fn add(&mut self, o: &Stats) {
        self.seen += o.seen;
        self.novel += o.novel;
        self.beaten += o.beaten;
        self.returned += o.returned;
        self.saves += o.saves;
        self.replay_frames += o.replay_frames;
    }
}

/// Merge-side totals (`Engine::merge`): candidates added, dropped as not
/// new any more, dropped for want of a saved anchor.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct MergeStats {
    pub worker: Stats,
    pub added: u64,
    pub dup: u64,
    pub no_anchor: u64,
    pub frontier: u64,
    pub attempts: u64,
    /// Dense nodes retired by the soft-dedupe memory budget.
    pub pruned: u64,
    /// (total frames, nodes, dense nodes) every 10 M frames: the growth curve.
    pub curve: Vec<(u64, usize, u64)>,
}

/// `via` of dense nodes: the objective kind with "~d" (one leaked string per kind).
pub fn via(kind: &'static str) -> &'static str {
    static VIA: OnceLock<Mutex<HashMap<&'static str, &'static str>>> = OnceLock::new();
    let mut m = VIA.get_or_init(Default::default).lock().unwrap();
    m.entry(kind).or_insert_with(|| Box::leak(format!("{kind}~d").into_boxed_str()))
}
