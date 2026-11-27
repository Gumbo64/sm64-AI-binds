//! Dense adds (`OBRRT_ADD`, off by default): more of the states an attempt
//! samples become nodes, not just the GA winner's segment ends.
//!
//! - `winner_all`: every state along the winning rollout;
//! - `last_gen`: the final state of every rollout of the last generation;
//! - `all`: every state of every rollout of the attempt.
//!
//! (Comma-separated: `winner_all,last_gen`. `winner` = the old behaviour.)
//!
//! Only states that are new by the *dense key* are kept: (level, area,
//! stars, keys, coins, position on a `OBRRT_DENSE_GRID`-unit grid (64),
//! open dialog or menu, red coins, holding, collected set with
//! OBRRT_STARSET_KEY). It is the duplicate check's cell without the object
//! part of `St::fingerprint` (objects loading and unloading as Mario runs
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
//! only when needed (sparse saves, `OBRRT_REPLAY_MAX`, 150 by default with
//! dense adds on).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use obrrt_core::{GamePad, St};

use crate::store::Delta;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Modes {
    pub winner_all: bool,
    pub last_gen: bool,
    pub all: bool,
}

impl Modes {
    pub fn any(&self) -> bool {
        self.winner_all || self.last_gen || self.all
    }

    pub fn parse(s: &str) -> Result<Modes, String> {
        let mut m = Modes::default();
        for t in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            match t {
                "winner" => {}
                "winner_all" => m.winner_all = true,
                "last_gen" => m.last_gen = true,
                "all" => m.all = true,
                _ => return Err(format!("OBRRT_ADD: winner, winner_all, last_gen or all (comma-separated), not {t}")),
            }
        }
        Ok(m)
    }
}

/// Dense-add settings (set once from the environment in main).
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    pub modes: Modes,
    /// Dense key grid, units (`OBRRT_DENSE_GRID`, 64).
    pub grid: f32,
    /// Most candidates kept per attempt, earliest frames first
    /// (`OBRRT_DENSE_CAP`, 0 = no cap).
    pub cap: usize,
    /// Whether dense nodes are noted for the new-eligible queue
    /// (`OBRRT_DENSE_NE`, 1).
    pub ne: bool,
    /// Whether object-list bytes count towards the soft-dedupe memory
    /// budget (`OBRRT_DENSE_MEMOBJ`, 1).
    pub mem_objects: bool,
    /// `winner_all`/`all`: only every this many frames along a rollout (its
    /// last state always) are candidates (`OBRRT_DENSE_EVERY`, 1 = every).
    pub every: u32,
    /// Whether mid-air states are candidates (`OBRRT_DENSE_AIR`, 1). Off:
    /// a state with ACT_FLAG_AIR is one only on its first airborne frame (a
    /// take-off) or as a rollout's last state.
    pub air: bool,
    /// `OBRRT_DENSE_EVERY=obj[:<n>]`: each objective chooses
    /// (`Objective::dense_keep`); `every`/`air` (n, default 15) for those
    /// that leave it to the search.
    pub per_obj: bool,
}

impl Settings {
    /// Whether any thinning applies (else every state is a candidate).
    pub fn thins(&self) -> bool {
        self.per_obj || self.every > 1 || !self.air
    }

    /// The (every n frames, mid-air too) rule for a stretch worked on by
    /// `obj`; n = 0: none but the rollout's end.
    pub fn rule(&self, obj: &dyn obrrt_core::Objective) -> (u32, bool) {
        if !self.per_obj {
            return (self.every, self.air);
        }
        match obj.dense_keep() {
            obrrt_core::DenseKeep::Global => (self.every, self.air),
            obrrt_core::DenseKeep::EndOnly => (0, false),
            obrrt_core::DenseKeep::Every { n, air } => (n.max(1), air),
        }
    }
}

/// Whether the state `t` frames into a rollout of `len` (`st`, after `prev`)
/// passes the rule `(every, air)` of `Settings::rule`.
pub fn thinned(rule: (u32, bool), t: usize, len: usize, st: &St, prev: Option<&St>) -> bool {
    let (every, air_ok) = rule;
    if t == len {
        return true;
    }
    if every == 0 || (every > 1 && t % every as usize != 0) {
        return false;
    }
    if !air_ok {
        let air = |x: &St| x.action_has(obrrt_core::layout::ACT_FLAG_AIR);
        if air(st) && prev.is_none_or(air) {
            return false;
        }
    }
    true
}

/// The dense-add settings a config asks for (None: off). An error names the bad switch.
pub fn from_config(sc: &obrrt_core::config::SearchConfig) -> Result<Option<Settings>, String> {
    let d = &sc.dedup.dense;
    let modes = match &d.add {
        Some(v) => Modes::parse(v)?,
        None => Modes::default(),
    };
    if !modes.any() {
        return Ok(None);
    }
    if !(d.grid >= 64 && d.grid % 64 == 0) {
        return Err("OBRRT_DENSE_GRID: a multiple of 64".into());
    }
    Ok(Some(Settings {
        modes,
        grid: d.grid as f32,
        cap: d.cap.unwrap_or(if modes.all { 256 } else { 0 }),
        ne: d.ne,
        mem_objects: true,
        every: d.every,
        per_obj: d.per_obj,
        air: d.air,
    }))
}

/// The settings of this thread's config ([`obrrt_core::config::current`]; the engine's, inside its threads);
/// None: off. Worked out once per config and thread.
pub fn settings() -> Option<Settings> {
    thread_local! {
        static CACHE: std::cell::RefCell<(usize, Option<Settings>)> = const { std::cell::RefCell::new((0, None)) };
    }
    let sc = obrrt_core::config::current();
    let key = sc as *const _ as usize;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.0 != key {
            *c = (key, from_config(sc).ok().flatten());
        }
        c.1
    })
}

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
    /// candidate of the same attempt, dropped by the cap, returned.
    pub seen: u64,
    pub novel: u64,
    pub beaten: u64,
    pub capped: u64,
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
        self.capped += o.capped;
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
