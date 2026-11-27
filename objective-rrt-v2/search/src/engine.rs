//! The loop: steps 1-6 in rounds. Planning (steps 1-4) and merging (step 6)
//! run on one thread; the attempts (step 5) run one per worker in parallel.

use std::collections::{BTreeMap, HashMap, HashSet};
use obrrt_core::sys::fs::{self, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::{Duration}; use obrrt_core::sys::Instant;

use anyhow::Result;
use obrrt_core::{GamePad, Generator, Objective, St, read_state};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
#[cfg(not(feature = "hooked"))]
use sm64_binds::{SM64GameGenerator, SaveState};

use crate::diag::Diagnostics;
use crate::ga::{Attempt, Worker};
use crate::pool::Pool;
use crate::store::Delta;
use crate::tree::Tree;

mod manual;
pub use manual::ManualInsert;

pub struct Config {
    pub out_dir: String,
    pub seed: u64,
    pub minutes: f64,
    /// Failure decay of an objective's weight (×d per failed attempt).
    pub decay: f64,
    /// Back-off of a state's start weight per barren attempt.
    pub state_decay: f64,
    /// Universal soft factor for health and lives: (wedges/8)^alpha · gamma^lives lost.
    pub alpha: f64,
    pub gamma: f64,
    /// Rounds an objective may stay unattemptable before it expires.
    pub expire_rounds: u32,
    /// Floor on any objective kind's share of samples.
    pub kind_floor: f64,
    /// Initial probability of extending an attempt by one more objective
    /// (then learned), the most objectives per attempt, and how far the next
    /// objective's target may be from where the previous one is expected to
    /// end.
    pub chain_prob: f64,
    pub max_chain: usize,
    pub chain_reach: f32,
    /// Experiment knob: multiply the GA's generations.
    pub ga_scale: f64,
    pub ga_pink: f64,
    /// Share of b drawn from the best routes (the states leading to each
    /// zone's earliest frontier state), so PosTime targets the route.
    pub route_share: f64,
    /// Soft dedupe: keep states that only repeat a place (an earlier state
    /// got there), at half the sampling weight per earlier state, instead of
    /// dropping them; over `mem_budget` bytes of saves, the lowest-weight
    /// repeats are retired. Yield and decay count only new or sooner states.
    /// Attempts planned per round (merged together after it).
    pub round_plans: usize,
    /// Stop after emulating this many frames (None: only the time limit).
    pub max_frames: Option<u64>,
    pub soft_dedup: bool,
    pub mem_budget: usize,
    /// Pick only objectives b (or its parent) satisfies the base criteria
    /// of, so a start always exists.
    pub pick_by_b: bool,
    /// New-eligible boost (OBRRT_NEW_ELIGIBLE=1): a state that can start an
    /// objective its parent couldn't (by key, objectives worth more than
    /// 1) is queued once per such key, and the queue is planned from first,
    /// at most `NE_SHARE` of a round's plans, with b = that state and the
    /// newly startable objective: the highest-weight objective first (the
    /// newest state among equals), since a run may have only a few rounds. Each key is queued at most `NE_PER_KEY`
    /// times in a run. Rare unlocking states (a boss's star spawned) would
    /// otherwise wait in a crowded b cell.
    pub new_eligible: bool,
    /// Progress report (log, nodes.csv, summary.json) every this much wall
    /// time. Reports only; no decision depends on them.
    pub report_every: Duration,
    /// Automatic tuning (`tune`: kind multipliers, chain probability, start
    /// cutoffs) each time this many emulated frames have passed, checked
    /// after each round's merge (`OBRRT_TUNE_FRAMES`). It used to run with
    /// the 30 s report, which made trees depend on machine load; 10 M frames
    /// is about 30 s of a 16-thread run (v35: 500 M frames in 1139 s).
    pub tune_frames: u64,
    /// OBRRT_CHAIN=route: an attempt chases one objective; a follow-on stage
    /// is appended only where the best routes (`hot`) met that objective and
    /// went on with a known next one ("continue along the known route"),
    /// each with the fixed probability `chain_follow`, never learned from
    /// success counts. Objectives met along the way are credited by the
    /// arrival check in `merge`. Off: the random stacking of `extend_chain`
    /// with its learned `chain_prob`.
    pub chain_route: bool,
    /// OBRRT_CHAIN_FOLLOW: probability of each route follow-on (0.5).
    pub chain_follow: f64,
    /// OBRRT_SAMPLER=rate: entry weights are base_weight × clamp(rate /
    /// median rate, floor, cap), the rate being an entry's improvement per
    /// frame spent (novel states, first success, frames saved on its
    /// soonest success, closer kept ends), faded in emulated frames, with a
    /// prior at the pool's overall rate; no ×decay per failure and no
    /// resets. Start states likewise: yield per frame (floored) instead of
    /// ×state_decay per barren attempt. Off: the decay rules.
    pub sampler_rate: bool,
    /// OBRRT_RATE_FLOOR / _CAP / _TAU / _PRIOR / _NOSTART (see RateParams).
    pub rate: crate::pool::RateParams,
    /// OBRRT_RATE_NODE_PRIOR: a start's prior, in frames at the tree's
    /// rate; OBRRT_RATE_NODE_CAP: its multiplier's cap (floor: rate.floor).
    pub node_prior: f64,
    pub node_cap: f64,
    /// OBRRT_KINSHIP_MULT: multiplies each objective's kinship (the
    /// generations pick_start looks up and down from b), for trees whose
    /// edges are short (dense adds). 1 = as the objective says.
    pub kinship_mult: f64,
    /// OBRRT_SPARSE_CHECK: at the end, rebuild this many save-less nodes
    /// (sparse saves) and compare each with its recorded state.
    pub rebuild_check: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            out_dir: "runs/obrrt".into(),
            seed: 1,
            minutes: 30.0,
            decay: 0.7,
            state_decay: 0.85,
            alpha: 1.0,
            gamma: 0.5,
            expire_rounds: 200,
            kind_floor: 0.02,
            chain_prob: 0.5,
            max_chain: 3,
            chain_reach: 2000.0,
            ga_scale: 3.0,
            ga_pink: 0.0,
            // Off by default: v15 (0.2 and on) did worse than v14.
            route_share: 0.0,
            round_plans: 64,
            max_frames: None,
            soft_dedup: true,
            mem_budget: 3_000_000_000,
            pick_by_b: false,
            new_eligible: true,
            report_every: Duration::from_secs(30),
            tune_frames: 10_000_000,
            chain_route: true,
            chain_follow: 0.5,
            sampler_rate: true,
            rate: crate::pool::RateParams { floor: 0.15, cap: 4.0, tau: 50_000_000.0, prior_frames: 200_000.0, nostart_frames: 5_000.0, novel: 1.0 },
            node_prior: 300_000.0,
            node_cap: 2.0,
            kinship_mult: 10.0,
            rebuild_check: 0,
        }
    }
}

/// One attempt: objectives to meet in order (pool entry, objective) from
/// `start`. A single objective is a chain of one.
struct Plan {
    stages: Vec<(usize, Arc<dyn Objective>)>,
    start: u32,
    b: u32,
    /// OBRRT_NEAR_SEED: recorded inputs of a nearby start's attempt of the
    /// first stage, seeding the GA (see near.rs).
    seed: Option<Vec<GamePad>>,
}

/// OBRRT_GOAL_FOCUS (see `index_frontier`): the share of plans drawn toward a
/// goal; each draw adds 10 ETA frames to its cell; half of those plans attempt
/// the goal itself from the drawn state (the rest plan from it as usual); the
/// least objective weight that makes a goal is 20 (stars, bosses, mission
/// spots).
/// OBRRT_GOAL_EXPLORE=1: a goal whose best ETA has stopped dropping keeps its
/// share of the plans, and spends them exploring from its best cells (b there,
/// the objective drawn as usual) instead of attempting the goal: the share of
/// forced attempts falls with the goal's weight (to a tenth). Without it the
/// plans went back to uniform cells: in the SSL pyramid (camp-sslcp-0008, from
/// a ledge at y 3200, the star at 5050) 150 of 3297 attempts started on the
/// ledge, 62 of them straight at the star (off the ledge), 3 explored; the
/// rest built on the 7689 states that had fallen to the floor. The ETA there
/// is flat along the ledge, so the stall stop (`OBRRT_STALL_ETA`) also counts
/// a new cell about as near as the best.
fn goal_explore() -> bool {
    obrrt_core::config::current().goal_focus.explore
}

fn goal_focus() -> (f64, f64, f64, f64) {
    static S: std::sync::OnceLock<(f64, f64, f64, f64)> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        (obrrt_core::config::current().goal_focus.share.clamp(0.0, 1.0), 10.0, 0.5, 20.0)
    })
}

/// How far along the walkable path an approach move aims.
fn geo_ahead() -> f32 {
    700.0
}

/// A frontier cell (see `index_frontier`), with its zone.
type GoalCell = ((u16, u8), (i32, i32, i32, i64, i16, u64));

/// Frames of ETA a unit of work in progress (red coins, chests, boss hits: St::wip) is worth in the goal
/// ranking, so the states furthest along a goal with several steps come first.
const GOAL_WIP_W: f64 = 1500.0;

/// What `Engine::sync_import` did with a path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncImport {
    /// Already in the tree (or imported before).
    Known,
    /// Replayed to a different state than the sender read, or a dead one.
    Bad,
    /// An earlier state at its cell.
    Dup,
    Added(u32),
}

/// One goal of OBRRT_GOAL_FOCUS: its objective and the best state of each of
/// its best-ranked cells.
struct GoalTop {
    key: String,
    obj: Arc<dyn Objective>,
    cells: Vec<(u32, GoalCell)>,
    weight: f64,
    /// The pool weight the goal was chosen by.
    base: f64,
    /// The checkpoints: per amount of work in progress, the grounded,
    /// controllable frontier state with the least ETA to the goal (no wear):
    /// (wip, ETA, node). In summary.json's `goals`, for the campaign to
    /// continue a course from when its tree found no exit.
    cps: Vec<(i64, f64, u32)>,
    /// The ETA of the earliest state on the frontier that has one (the baseline where the root has none: a goal in another area).
    h1: Option<f64>,
    /// The root's ETA to the goal, where it has one.
    h0: Option<f64>,
}

/// `s`'s start rank for `obj` given its start ETA `h`: `h` plus the objective's start bias
/// ([`obrrt_core::start_rank`]); exactly `h` when the bias is 0.
fn rank_of(obj: &dyn Objective, s: &St, h: f64) -> f64 {
    let b = obj.start_bias(s);
    if b == 0.0 { h } else { h + b }
}

/// ETA frames closer a start must be to improve on another ([`improves`]).
const IMPROVE_FRAMES: f64 = 7.0;

/// Whether the start rank `new` improves on `old`: lower by [`IMPROVE_FRAMES`] (v2: h is the start ETA in
/// frames; 7 frames is v1's 0.5 (~200 units at distance/400) at the reference pace).
fn improves(new: f64, old: f64) -> bool {
    new < old - IMPROVE_FRAMES
}

/// Samples an item by weight; None if no weight is positive.
fn weighted<T: Copy>(rng: &mut SmallRng, items: &[(T, f64)]) -> Option<T> {
    let total: f64 = items.iter().map(|i| i.1.max(0.0)).sum();
    if !(total > 0.0) {
        return None;
    }
    let mut x = rng.random_range(0.0..total);
    for (t, w) in items {
        if x < w.max(0.0) {
            return Some(*t);
        }
        x -= w.max(0.0);
    }
    items.last().map(|i| i.0)
}

/// Duplicate check: (level, area, stars, keys, coins, 64-unit cell, open
/// dialog or menu, St::fingerprint). A dialog or menu state and the state
/// after it, or a state after a boss hit and one before, are different
/// states at the same place; only same-progress states compete on time.
type Cell = (u16, u8, i16, u8, i16, i32, i32, i32, i16, u64);



/// The collected set, told apart by the duplicate check and the frontier's b cells (states with the
/// same progress count but different collected sets).
fn starset(st: &St) -> u64 {
    st.progress_set()
}

fn cell(st: &St) -> Cell {
    let c = |v: f32| (v / 64.0).floor() as i32;
    let modal = if st.in_dialog() { st.dialog } else if st.in_save_menu() { -2 } else if st.act_select { -3 } else { -1 };
    let fp = st.fingerprint();
    let fp = {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&(fp, st.progress_set()), &mut h);
        std::hash::Hasher::finish(&h)
    };
    (st.level, st.area, st.stars, st.keys, st.coins, c(st.pos[0]), c(st.pos[1]), c(st.pos[2]), modal, fp)
}

pub struct Engine {
    pub cfg: Config,
    pub tree: Tree,
    pub pool: Pool,
    pub diag: Diagnostics,
    generator: Arc<dyn Generator>,
    workers: Vec<std::sync::Mutex<Worker>>,
    /// Base states deltas are relative to: one per zone (see store.rs).
    bases: Vec<Vec<u8>>,
    zone_base: HashMap<(u16, u8), u16>,
    boot: Vec<GamePad>,
    rng: SmallRng,
    cells: HashMap<Cell, Vec<u32>>,
    first_success: HashSet<&'static str>,
    /// Decay resets by a new state, per objective key (summary.json).
    decay_resets: BTreeMap<String, u32>,
    /// Per [`Objective::diag_tag`]: attempts, successes, added (summary.json).
    tag_stats: BTreeMap<String, [u64; 3]>,
    zones: HashSet<(u16, u8)>,
    /// The frontier grouped by zone, then by 512-unit cell; rebuilt each round.
    b_index: Vec<Vec<Vec<u32>>>,
    /// OBRRT_GOAL_FOCUS: the goals' best cells; draws per (goal, cell); per
    /// goal (best ETA seen, draws since it last dropped, times it dropped).
    goal_top: Vec<GoalTop>,
    goal_wear: HashMap<(String, GoalCell), u32>,
    goal_stats: HashMap<String, (f64, u32, u32)>,
    /// Per b_index zone: its draw weight (all 1).
    zone_weights: Vec<f64>,
    /// The best routes' states (see route_share).
    route: Vec<u32>,
    events: BufWriter<File>,
    total_frames: u64,
    attempts_merged: u64,
    /// Modal states (in a dialog or menu) whose options haven't all been
    /// tried: planned before anything else, one option each time.
    modal_queue: Vec<u32>,
    /// Attempts the emulated game trapped in.
    emulator_errors: u64,
    /// (state, option key) pairs already tried from the queue.
    modal_tried: HashSet<(u32, String)>,
    hot: HashSet<u32>,
    /// New-eligible boost (see `Config::new_eligible`).
    ne: NewEligible,
    /// Wall seconds spent planning (steps 1-4) and merging (step 6).
    diag_plan_secs: (f64, f64),
    /// Wall seconds (report only): [pre-plan refresh+index_frontier, attempts
    /// (parallel), post-merge upkeep, write_outputs, worker load_into (summed
    /// over threads), worker attempt (summed)].
    diag_secs: [f64; 6],
    /// Seconds in index_frontier's goal ETAs (part of pre).
    diag_index_secs: f64,
    /// Seconds in pre's parts: eta snapshot, nostart recheck, rate refresh, wip refresh, index_frontier.
    diag_pre: [f64; 11],
    /// Route chains: per objective key, the best-route (`hot`) nodes where
    /// it was met (ids ascending); rebuilt with `hot`.
    route_met: HashMap<String, Vec<u32>>,
    /// Rate sampler: the tree's new states and frames over all starts
    /// (faded like each node's), and when last faded.
    node_rate: (f64, f64, u64),
    /// Dense adds (OBRRT_ADD, see dense.rs): settings and totals.
    dense: Option<crate::dense::Settings>,
    dstats: crate::dense::MergeStats,
    /// OBRRT_ROAD: the experience roadmap (road.rs).
    road: Option<crate::road::Road>,
    /// Plans of this round that are roadmap legs (by start node), for the stats.
    road_legs: Vec<u32>,
    /// `OBRRT_STALL_ETA`: each goal's best ETA when it last counted as progress, and whether one has since the last stall check.
    stall_eta: HashMap<String, f64>,
    /// OBRRT_GOAL_EXPLORE: per goal, the cells about as near as its best (see `goal_explore`).
    stall_near: HashMap<String, usize>,
    eta_improved: bool,
    /// Called with the engine at the end of every round (the browser build samples the tree for
    /// its visualisation; it rate-limits itself). None: nothing.
    pub on_round: Option<Box<dyn FnMut(&Engine) + Send>>,
    /// Called after `on_round`, with the engine mutable: the browser build's tree sync with the other
    /// workers (`sync_candidates`, `sync_import`).
    pub on_sync: Option<Box<dyn FnMut(&mut Engine) + Send>>,
    /// Nodes already offered to the other workers / paths already imported (by hash).
    sync_sent: HashSet<u32>,
    sync_seen: HashSet<u64>,
    /// Called after `on_sync`, with the engine mutable: the browser build's hybrid mining
    /// (human-played nodes, state requests, position goals; see manual.rs).
    pub on_hybrid: Option<Box<dyn FnMut(&mut Engine) + Send>>,
    /// Human-played nodes' boosts (manual.rs).
    manual: manual::Manual,
    /// Sparse saves: rebuild frames already counted into total_frames.
    rebuild_counted: u64,
    /// Best meets per objective key, Beats, faster-only and improvement
    /// counters (see speed.rs).
    speed: crate::speed::Speed,
}

/// Share of a round's plans the new-eligible queue may take, times a key
/// may be queued, and the queue's length (lowest weight dropped beyond it).
const NE_SHARE: f64 = 0.1;
const NE_PER_KEY: u8 = 2;
const NE_QUEUE: usize = 1024;

#[derive(Default)]
struct NewEligible {
    /// Per node: hashes of the keys it can start (sorted). Modal states
    /// inherit their parent's. Dropped for retired nodes.
    sets: HashMap<u32, Box<[u64]>>,
    /// (weight's bits, sequence number, state, key) to plan from, the
    /// last first (weights are positive, so their bits sort like them).
    queue: std::collections::BTreeSet<(u64, u64, u32, String)>,
    /// Times each key was queued.
    queued_per_key: HashMap<u64, u8>,
    /// Plans left from the queue this round.
    budget: usize,
    rng: Option<SmallRng>,
    queued: u64,
    dropped: u64,
    stale: u64,
    used: u64,
    planned: u64,
    secs: f64,
}

fn key_hash(k: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    k.hash(&mut h);
    h.finish()
}

impl Engine {
    #[cfg(not(feature = "hooked"))]
    pub fn new(
        cfg: Config,
        game_gen: &SM64GameGenerator,
        boot: Vec<GamePad>,
        root_save: SaveState,
        root: St,
        generator: Arc<dyn Generator>,
    ) -> Result<Engine> {
        let threads = rayon::current_num_threads();
        let workers = (0..threads).map(|_| Worker::new(game_gen)).collect::<Result<Vec<_>>>()?;
        Self::with_workers(cfg, workers, boot, root_save, root, generator)
    }

    /// With feature `hooked`: the workers are forks of `proto`, a hooked
    /// game, and every node's save carries the hook's state with the game
    /// memory. `threads` workers.
    #[cfg(feature = "hooked")]
    pub fn new_hooked(
        cfg: Config,
        proto: &crate::game::Game,
        threads: usize,
        boot: Vec<GamePad>,
        root_save: crate::game::GameSave,
        root: St,
        generator: Arc<dyn Generator>,
    ) -> Result<Engine> {
        let workers = (0..threads.max(1)).map(|_| Worker::forked(proto)).collect::<Result<Vec<_>>>()?;
        Self::with_workers(cfg, workers, boot, root_save, root, generator)
    }

    /// Frames emulated so far (the tree's budget counter).
    #[cfg(feature = "hooked")]
    pub fn total_frames_emulated(&self) -> u64 {
        self.total_frames
    }

    fn with_workers(
        cfg: Config,
        workers: Vec<Worker>,
        boot: Vec<GamePad>,
        root_save: crate::game::GameSave,
        root: St,
        generator: Arc<dyn Generator>,
    ) -> Result<Engine> {
        fs::create_dir_all(&cfg.out_dir)?;
        let workers = workers.into_iter().map(std::sync::Mutex::new).collect::<Vec<_>>();
        let events = BufWriter::new(File::create(format!("{}/events.jsonl", cfg.out_dir))?);
        let rng = SmallRng::seed_from_u64(cfg.seed);
        let mut cells = HashMap::new();
        cells.insert(cell(&root), vec![0]);
        let root_zone = root.zone();
        let mut tree = Tree::new(root);
        // The root's save: its pages are the base, its perturbation its own.
        tree.nodes[0].save = Delta::new(0, crate::game::plain(&root_save).as_bytes(), &root_save);
        Ok(Engine {
            tree,
            pool: Pool::default(),
            diag: Diagnostics { chain_prob: cfg.chain_prob, ..Default::default() },
            generator,
            workers,
            bases: vec![crate::game::into_plain(root_save).into_bytes()],
            zone_base: HashMap::from([(root_zone, 0)]),
            boot,
            rng,
            cells,
            first_success: HashSet::new(),
            decay_resets: BTreeMap::new(),
            tag_stats: BTreeMap::new(),
            zones: HashSet::from([root_zone]),
            b_index: vec![vec![vec![0]]],
            goal_top: Vec::new(),
            goal_wear: HashMap::new(),
            goal_stats: HashMap::new(),
            zone_weights: vec![1.0],
            route: Vec::new(),
            events,
            total_frames: 0,
            attempts_merged: 0,
            modal_queue: Vec::new(),
            emulator_errors: 0,
            modal_tried: HashSet::new(),
            hot: HashSet::new(),
            ne: NewEligible { rng: Some(SmallRng::seed_from_u64(cfg.seed ^ 0x6e65_7765_6c69_6769)), ..Default::default() },
            diag_plan_secs: (0.0, 0.0),
            diag_secs: [0.0; 6],
            diag_index_secs: 0.0,
            diag_pre: [0.0; 11],
            route_met: HashMap::new(),
            node_rate: (0.0, 0.0, 0),
            dense: crate::dense::settings().filter(|d| d.modes.any()),
            dstats: Default::default(),
            road: crate::road::setting().map(crate::road::Road::new),
            road_legs: Vec::new(),
            stall_eta: HashMap::new(),
            stall_near: HashMap::new(),
            eta_improved: false,
            on_round: None,
            on_sync: None,
            sync_sent: HashSet::new(),
            sync_seen: HashSet::new(),
            on_hybrid: None,
            manual: Default::default(),
            rebuild_counted: 0,
            speed: Default::default(),
            cfg,
        })
    }

    /// Step 3: picks a kind (by the mean weight of its entries × its learned
    /// multiplier, with a floor; so a kind with many entries doesn't crowd
    /// out the rest), then an entry within it by weight.
    /// From the pool's per-kind tables (`SampleIdx`): the same sums, draws and pick as the scan
    /// (`sample_entry_scan`), without a pass over the pool.
    fn sample_entry(&mut self) -> Option<usize> {
        let idx = self.pool.sidx.get_or_insert_with(Default::default);
        idx.sync(&self.pool.entries);
        if idx.irregular > 0 {
            return self.sample_entry_scan();
        }
        idx.refresh();
        let idx = self.pool.sidx.as_ref().unwrap();
        let kinds: Vec<(&'static str, f64)> = idx
            .by_name
            .iter()
            .map(|(k, &t)| (*k, &idx.kinds[t as usize]))
            .filter(|(_, t)| t.count > 0 && t.sum() > 0.0)
            .map(|(k, t)| (k, t.sum() / t.count as f64 * self.diag.kind(k).multiplier))
            .collect();
        let chosen = self.choose_kind(&kinds)?;
        let idx = self.pool.sidx.as_ref().unwrap();
        let t = &idx.kinds[idx.by_name[chosen] as usize];
        let x = self.rng.random_range(0.0..t.sum());
        Some(t.pick(x) as usize)
    }

    /// The kind stage of `sample_entry`: floor, cap and draw over `kinds`
    /// (name, mean weight × multiplier; all positive), in name order.
    fn choose_kind(&mut self, kinds: &[(&'static str, f64)]) -> Option<&'static str> {
        if kinds.is_empty() {
            return None;
        }
        let total: f64 = kinds.iter().map(|(_, w)| w).sum();
        // The floor guarantees each kind a share of compute, not of picks: an
        // expensive kind's floor is scaled down by its cost per attempt.
        let cost = |k: &str| {
            let ks = self.diag.kinds.get(k);
            ks.filter(|ks| ks.attempts > 0).map_or(5000.0, |ks| ks.frames as f64 / ks.attempts as f64 + crate::diag::ATTEMPT_OVERHEAD)
        };
        let mean_cost = kinds.iter().map(|(k, _)| cost(k)).sum::<f64>() / kinds.len() as f64;
        let weights: Vec<f64> =
            kinds.iter().map(|(k, w)| w.max(self.cfg.kind_floor * total * (mean_cost / cost(k)).min(1.0))).collect();
        let sum: f64 = weights.iter().sum();
        let mut x = self.rng.random_range(0.0..sum);
        let mut chosen = kinds[kinds.len() - 1].0;
        for ((k, _), w) in kinds.iter().zip(&weights) {
            if x < *w {
                chosen = k;
                break;
            }
            x -= w;
        }
        Some(chosen)
    }

    /// `sample_entry` by a pass over the whole pool.
    fn sample_entry_scan(&mut self) -> Option<usize> {
        let eligible: Vec<bool> = self.pool.entries.iter().map(|e| !e.obj.modal()).collect();
        let mut kind_weight: BTreeMap<&'static str, (f64, usize)> = BTreeMap::new();
        for e in self.pool.entries.iter().zip(&eligible).filter(|(_, ok)| **ok).map(|(e, _)| e) {
            let w = kind_weight.entry(e.obj.kind()).or_default();
            w.0 += e.base_weight * e.decay;
            w.1 += 1;
        }
        let kinds: Vec<(&'static str, f64)> = kind_weight
            .into_iter()
            .filter(|(_, (w, _))| *w > 0.0)
            .map(|(k, (w, n))| (k, w / n as f64 * self.diag.kind(k).multiplier))
            .collect();
        let chosen = self.choose_kind(&kinds)?;
        let items: Vec<(usize, f64)> = self
            .pool
            .entries
            .iter()
            .enumerate()
            .filter(|(i, e)| eligible[*i] && e.obj.kind() == chosen)
            .map(|(i, e)| (i, e.base_weight * e.decay))
            .collect();
        weighted(&mut self.rng, &items)
    }

    /// What improvement on a start is judged by (see [`improves`]): the start rank (`start_rank`: the ETA
    /// plus the objective's start bias).
    fn keys_of(&self, obj: &dyn Objective, s: &St) -> f64 {
        rank_of(obj, s, obrrt_core::start_eta(obj, s))
    }

    /// Step 4: the eligible starts for `obj` given `b`, sampled by softmax
    /// over `start_eta` × u(s) × the state's back-off.
    fn pick_start(&mut self, obj: &dyn Objective, b: u32) -> Option<u32> {
        let cutoff = self.diag.kind(obj.kind()).cutoff;
        let k = obj.kinship();
        let k = if self.cfg.kinship_mult == 1.0 { k } else { ((k as f64) * self.cfg.kinship_mult).round().max(0.0) as u32 };
        let b_st = &self.tree.nodes[b as usize].st;
        let (b_keys, b_stars, b_lives) = (b_st.keys, b_st.stars, b_st.lives);
        let mut cands: Vec<(u32, f64, f64)> = Vec::new();
        let kin = self.tree.kin_weighted(b, k);
        for &(i, kw) in kin.iter() {
            let n = &self.tree.nodes[i as usize];
            let s = &n.st;
            // A state kept for its work under way starts whatever b has collected.
            let wip = self.tree.is_wip(i);
            if n.retired
                || !s.alive()
                || (!wip && s.keys < b_keys)
                || (!wip && s.stars < b_stars)
                || s.modal() != obj.modal()
                || !obj.base_criteria(s)
            {
                continue;
            }
            let h = obrrt_core::start_eta(obj, s);
            if !h.is_finite() || h > cutoff {
                continue;
            }
            // The objective's start bias (frames; 0 for most) ranks too.
            let bias = obj.start_bias(s);
            let h = if bias == 0.0 { h } else { h + bias };
            let u = (s.wedges() as f64 / 8.0).powf(self.cfg.alpha)
                * self.cfg.gamma.powi((b_lives as i32 - s.lives as i32).max(0));
            let crowd = 0.5f64.powi(n.stats.crowd as i32);
            cands.push((i as u32, h, u * self.start_mult(&n.stats) * crowd * kw as f64 * self.manual_start_mult(i)));
        }
        if cands.is_empty() {
            return None;
        }
        let hmin = cands.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let t = obj.temperature().max(1e-9);
        let weights: Vec<f64> = cands.iter().map(|c| (-(c.1 - hmin) / t).exp() * c.2).collect();
        let total: f64 = weights.iter().sum();
        if !(total > 0.0) {
            return Some(cands[0].0);
        }
        let mut x = self.rng.random_range(0.0..total);
        for (c, w) in cands.iter().zip(&weights) {
            if x < *w {
                return Some(c.0);
            }
            x -= w;
        }
        Some(cands[cands.len() - 1].0)
    }

    /// Groups the frontier by zone and then by 512-unit cell, so b can be a
    /// zone uniformly, then a cell uniformly (sparse, newly reached regions
    /// are sampled as often as dense ones), then a state.
    fn index_frontier(&mut self) {
        let mut lap = Instant::now();
        macro_rules! tick {
            ($k:expr) => {{
                self.diag_pre[$k] += lap.elapsed().as_secs_f64();
                lap = Instant::now();
            }};
        }
        let mut zones: BTreeMap<(u16, u8), BTreeMap<(i32, i32, i32, i64, i16, u64), Vec<u32>>> = BTreeMap::new();
        // OBRRT_KEEP_WIP: the states kept for their work under way are b
        // candidates too (incomparable with the frontier, not behind it).
        let wip_kept = if self.tree.keep_wip { self.tree.wip_kept.clone() } else { Vec::new() };
        // OBRRT_GOAL_FOCUS=p: this share of plans is drawn toward a goal, an
        // objective in the pool worth at least OBRRT_GOAL_MIN (a star, a boss,
        // a mission spot). b is otherwise uniform over occupied cells, so in a
        // large level the states nearest a star are almost never built on: in
        // the zone-locked BoB tree of 2026-11-09 (150 M frames, no star) the
        // twelve highest states on the mountain were never a start. Each goal
        // ranks the cells by their best state's ETA to it plus
        // OBRRT_GOAL_FOCUS_C frames for every draw the cell has had (so a cell
        // that leads nowhere, under a wall, is flooded and the next-best cells
        // take over), and keeps the best three. Goals are drawn by how often
        // their best ETA has dropped per draw.
        let gf = goal_focus();
        let goals: Vec<(String, Arc<dyn Objective>, f64)> = if gf.0 > 0.0 {
            let mut g: Vec<(f64, String, Arc<dyn Objective>)> = self
                .pool
                .entries
                .iter()
                .filter(|e| e.base_weight >= gf.3 && !e.obj.modal() && e.obj.fixed().is_none())
                .map(|e| (e.base_weight, e.obj.key(), e.obj.clone()))
                .collect();
            g.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            g.dedup_by(|a, b| a.1 == b.1);
            g.truncate(12);
            g.into_iter().map(|(w, k, o)| (k, o, w)).collect()
        } else {
            Vec::new()
        };
        let mut goal_best: Vec<HashMap<GoalCell, (u32, f64)>> = vec![HashMap::new(); goals.len()];
        let mut goal_cp: Vec<std::collections::BTreeMap<i64, (f64, u32)>> = vec![Default::default(); goals.len()];
        let mut goal_first: Vec<Option<(u32, f64)>> = vec![None; goals.len()];
        let ww = GOAL_WIP_W;
        tick!(5);
        if let Some(r) = self.road.as_mut() {
            r.observe(&self.tree);
        }
        tick!(6);
        let skip = |i: u32| {
            let st = &self.tree.nodes[i as usize].st;
            // A course with every star had has nothing left to find, unless the run is only to leave it
            // (OBRRT_ANY_EXIT: the campaign's tree after a star had inside the course; t1 step 464, BitFS's one
            // star held: 1 node, nothing planned, and the star was dropped).
            (!any_exit() && st.course_done(st.course)) || st.modal()
        };
        // Each state's ETA to each goal, on every thread: done on the main
        // thread alone this took 79% of the wall clock in a large tree while
        // the workers waited (t1 step 186, BitFS: pre 6130 s of 7796 s at 85k
        // nodes). Per goal: None if it can't start there, else the ETA and,
        // when the roadmap's field wasn't built yet, the target to ask it.
        let idx_t = Instant::now();
        let items: Vec<u32> = self.tree.frontier.iter().chain(&wip_kept).copied().collect();
        let pre: Vec<Vec<Option<(f64, Option<[f32; 3]>)>>> = {
            let (tree, road, goals) = (&self.tree, self.road.as_ref(), &goals);
            let one = |&i: &u32| -> Vec<Option<(f64, Option<[f32; 3]>)>> {
                if goals.is_empty() || skip(i) {
                    return Vec::new();
                }
                let st = &tree.nodes[i as usize].st;
                goals
                    .iter()
                    .map(|(_, o, _)| {
                        if !st.alive() || !o.exists_in(st) || !o.base_criteria(st) {
                            return None;
                        }
                        let mut h = obrrt_core::start_eta(o.as_ref(), st);
                        // OBRRT_GEO=1: rank by the walkable distance to the goal's
                        // target where the mesh knows it (never less than the ETA): a
                        // state under a cliff below the goal is far, the path up is
                        // near. Straight-line ranking left BoB's Koopa race at the
                        // mountain's foot: 98 attempts, ETA 88-200, none up (2026-11-09).
                        if obrrt_core::geo::enabled() && h.is_finite() {
                            if let Some(t) = o.steer_target(st) {
                                if let Some(d) = obrrt_core::geo::path_dist(st.zone(), st.pos, t) {
                                    h = h.max(obrrt_core::frames_for(d));
                                }
                            }
                        }
                        // OBRRT_ROAD: rank by the roadmap's cost (road.rs), never
                        // less than the ETA.
                        let mut ask = None;
                        if let Some(r) = road.filter(|_| h.is_finite()) {
                            if let Some(t) = o.steer_target(st) {
                                match r.eta_cached(st, t) {
                                    Some(e) => h = h.max(e),
                                    None => ask = Some(t),
                                }
                            }
                        }
                        Some((h, ask))
                    })
                    .collect()
            };
            static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
            let n = self.workers.len().max(1);
            match POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(n).build().ok()) {
                Some(p) if n > 1 => p.install(|| items.par_iter().map(one).collect()),
                _ => items.iter().map(one).collect(),
            }
        };
        self.diag_index_secs += idx_t.elapsed().as_secs_f64();
        lap = Instant::now();
        for (k, &i) in items.iter().enumerate() {
            let st = &self.tree.nodes[i as usize].st;
            // A course whose stars are all collected has nothing left to find;
            // dialogs and menus are planned from the modal queue instead.
            if skip(i) {
                continue;
            }
            let c = |v: f32| (v / 512.0).floor() as i32;
            // Level changes (a pounded post, the Chomp freed) get cells of
            // their own: the states after one would otherwise be a few among
            // hundreds in the cell where they happened, and almost never b.
            // Red coins too, for the same reason: in v32-chomp-s3-1b the
            // states with the most red coins (3-4 of 8) were picked as b 4x
            // less often per state than those with none, and no level ever
            // got past 5 of 8.
            // With OBRRT_STARSET_KEY=1, each collected set too: a state with
            // a different star than the hundreds around it at the same count
            // (a castle exit after WF star 2 among star-6 exits) gets a cell.
            let cell = (c(st.pos[0]), c(st.pos[1]), c(st.pos[2]), st.level_changes(), st.red_coins, starset(st));
            for gi in 0..goals.len() {
                let Some((mut h, ask)) = pre[k].get(gi).copied().flatten() else { continue };
                if let (Some(t), Some(r)) = (ask, self.road.as_mut()) {
                    h = h.max(r.eta(st, t));
                }
                if h.is_finite() {
                    let e = goal_best[gi].entry((st.zone(), cell)).or_insert((i, h));
                    if h < e.1 {
                        *e = (i, h);
                    }
                    if i != 0 && goal_first[gi].is_none_or(|f| i < f.0) {
                        goal_first[gi] = Some((i, h));
                    }
                    let grp = st.action & 0x1C0;
                    if i != 0 && (grp == 0x000 || grp == 0x040) && st.controllable() && st.health >= 0x300 {
                        let w = self.tree.wip.get(i as usize).copied().unwrap_or(0);
                        let e = goal_cp[gi].entry(w).or_insert((h, i));
                        if h < e.0 {
                            *e = (h, i);
                        }
                    }
                }
            }
            zones.entry(st.zone()).or_default().entry(cell).or_default().push(i);
        }
        tick!(7);
        self.goal_top.clear();
        for ((((key, obj, base), best), cp), first) in goals.into_iter().zip(goal_best).zip(goal_cp).zip(goal_first) {
            if best.is_empty() {
                continue;
            }
            let h0 = {
                let r = &self.tree.nodes[0].st;
                (r.alive() && obj.exists_in(r) && obj.base_criteria(r)).then(|| obrrt_core::start_eta(obj.as_ref(), r)).filter(|h| h.is_finite()).map(|h| {
                    match (self.road.as_mut(), obj.steer_target(r)) {
                        (Some(rd), Some(t)) => h.max(rd.eta(r, t)),
                        _ => h,
                    }
                })
            };
            let hmin = best.values().map(|v| v.1).fold(f64::INFINITY, f64::min);
            // OBRRT_STALL_ETA=<fraction>: a goal's best ETA falling by this
            // fraction (and 10 frames) since it last did is progress for
            // OBRRT_STOP_STALL. A course that is one long way (BitDW: campaign
            // t1 step 142 stopped at 94.5 M with 2 red coins while the tree
            // was two thirds up the course) has no work in progress to show.
            if let Some(f) = obrrt_core::config::current().goal_focus.stall_eta {
                let near = |v: f64| best.values().filter(|b| b.1 <= v + (f * v).max(10.0)).count();
                match self.stall_eta.get_mut(&key) {
                    Some(v) => {
                        if hmin < *v * (1.0 - f) && hmin < *v - 10.0 {
                            *v = hmin;
                            self.eta_improved = true;
                            self.stall_near.insert(key.clone(), near(hmin));
                        } else if goal_explore() {
                            // A new cell about as near as the best is progress too: the way on
                            // from a ledge runs level, and often away from the goal.
                            let (n, e) = (near(*v), self.stall_near.entry(key.clone()).or_insert(0));
                            if n > *e {
                                self.eta_improved |= *e > 0;
                                *e = n;
                            }
                        }
                    }
                    None => {
                        self.stall_eta.insert(key.clone(), hmin);
                    }
                }
            }
            let mut ranked: Vec<(f64, u32, GoalCell)> = best
                .into_iter()
                .map(|(cell, (i, h))| {
                    let wip = self.tree.wip.get(i as usize).copied().unwrap_or(0) as f64;
                    (h + gf.1 * self.goal_wear.get(&(key.clone(), cell)).copied().unwrap_or(0) as f64 - ww * wip, i, cell)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            ranked.truncate(3);
            let st = self.goal_stats.entry(key.clone()).or_insert((f64::INFINITY, 0, 0));
            if hmin < st.0 - 5.0 {
                st.0 = hmin;
                st.1 = 0;
                st.2 += 1;
            }
            // 1 while its best ETA keeps dropping; 8 / draws since the last drop
            // once it stalls (a goal the level's layout keeps out of reach,
            // like a box on an island with no cannon open).
            let weight = 1.0 / (1.0 + st.1 as f64 / 8.0);
            self.goal_top.push(GoalTop { key, obj, cells: ranked.into_iter().map(|(_, i, c)| (i, c)).collect(), weight, base, cps: cp.into_iter().map(|(w, (h, i))| (w, h, i)).collect(), h1: first.map(|f| f.1), h0 });
        }
        tick!(8);
        // Zone weights: uniform (the value-weighted and bandit draws are gone; the weighted draw stays so seeds replay).
        self.zone_weights = vec![1.0; zones.len()];
        self.b_index = zones.into_values().map(|cells| cells.into_values().collect()).collect();
        tick!(9);
        // The best routes: the states leading to each zone's earliest
        // frontier state (b for route_share of plans; RRT* rewires them first).
        {
            self.hot.clear();
            let mut best: HashMap<(u16, u8), u32> = HashMap::new();
            for &i in &self.tree.frontier {
                let st = &self.tree.nodes[i as usize].st;
                let e = best.entry(st.zone()).or_insert(i);
                if st.frame < self.tree.nodes[*e as usize].st.frame {
                    *e = i;
                }
            }
            for &i in best.values() {
                let mut at = Some(i);
                while let Some(n) = at {
                    if !self.hot.insert(n) {
                        break;
                    }
                    at = self.tree.nodes[n as usize].parent;
                }
            }
            self.route = self
                .hot
                .iter()
                .copied()
                .filter(|&i| {
                    let n = &self.tree.nodes[i as usize];
                    !n.retired && !n.st.modal() && self.tree.on_frontier(&n.st)
                })
                .collect();
            self.route.sort_unstable();
            // Route chains: where the best routes met each objective.
            if self.cfg.chain_route {
                self.route_met.clear();
                let mut ids: Vec<u32> = self.hot.iter().copied().collect();
                ids.sort_unstable();
                let mut known = 0;
                for i in ids {
                    let n = &self.tree.nodes[i as usize];
                    let Some(m) = n.made_by.as_ref() else { continue };
                    known += 1;
                    if n.met {
                        self.route_met.entry(m.key()).or_default().push(i);
                    }
                }
                self.diag.route_chain.route_nodes = known;
                self.diag.route_chain.route_keys = self.route_met.len() as u64;
            }
        }
        tick!(10);
    }

    fn plan_modal(&mut self, b: u32, queued: bool) -> Option<Plan> {
        // A state retired since it was queued (the frontier moved on) has no
        // save any more: starting from it would start from the root.
        if self.tree.nodes[b as usize].retired {
            return None;
        }
        let b_st = self.tree.nodes[b as usize].st.clone();
        for o in self.generator.generate(&b_st, &mut self.rng) {
            self.pool.insert(o, &b_st);
        }
        let options: Vec<(usize, f64)> = self
            .pool
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.obj.modal() && e.obj.base_criteria(&b_st))
            .filter(|(_, e)| !queued || !self.modal_tried.contains(&(b, e.obj.key())))
            .map(|(i, e)| (i, if queued { 1.0 } else { e.base_weight * e.decay + 1e-9 }))
            .collect();
        let entry = weighted(&mut self.rng, &options)?;
        let obj = self.pool.entries[entry].obj.clone();
        if queued {
            self.modal_tried.insert((b, obj.key()));
            if options.len() > 1 {
                self.modal_queue.push(b);
            }
        }
        let kind = obj.kind();
        self.diag.kind(kind).samples += 1;
        self.diag.kind(kind).window_samples += 1;
        self.pool.entries[entry].unattemptable = 0;
        Some(Plan { stages: vec![(entry, obj)], start: b, b, seed: None })
    }

    /// New-eligible boost: the keys (with their text) of the objectives `st`
    /// can start that are worth more than 1 (self objectives, everywhere,
    /// are the baseline). Generated twice with different draws, keeping
    /// keys in both, so randomly placed objectives don't count as new.
    fn ne_keys(&mut self, st: &St) -> Vec<(u64, String, f64)> {
        if st.modal() {
            return Vec::new();
        }
        let mut rng = self.ne.rng.take().unwrap_or_else(|| SmallRng::seed_from_u64(0));
        let mut sets: Vec<Vec<(u64, String, f64)>> = (0..2)
            .map(|_| {
                let mut v: Vec<(u64, String, f64)> = self
                    .generator
                    .generate(st, &mut rng)
                    .into_iter()
                    .filter(|o| !o.modal() && o.obj_weight(st) > 1.0 && o.exists_in(st) && o.base_criteria(st))
                    .map(|o| {
                        let k = o.key();
                        (key_hash(&k), k, o.obj_weight(st))
                    })
                    .collect();
                v.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                v.dedup_by(|a, b| a.0 == b.0);
                v
            })
            .collect();
        self.ne.rng = Some(rng);
        let second = sets.pop().unwrap_or_default();
        let mut first = sets.pop().unwrap_or_default();
        first.retain(|(h, _, _)| second.binary_search_by(|x| x.0.cmp(h)).is_ok());
        first
    }

    /// `Diagnostics::note_best` for node `id` (made by `via`), and a `best_improved` event (output only)
    /// when it set a new best time for its (keys, stars).
    fn note_best_at(&mut self, id: u32, progress: (u8, i16), zone: (u16, u8), frame: u32, via: &'static str) -> Result<()> {
        if let Some(old) = self.diag.note_best(progress, zone, frame) {
            let saved = (old != u32::MAX).then(|| old - frame);
            if let Some(d) = saved {
                self.speed.note_progress_imp(via, d as u64);
            }
            writeln!(
                self.events,
                "{}",
                serde_json::json!({
                    "event": "best_improved", "milestone": "progress", "progress": [progress.0, progress.1],
                    "old": (old != u32::MAX).then_some(old), "new": frame, "saved": saved, "node": id, "via": via,
                    "frames_total": self.total_frames, "from": null, "added": null,
                })
            )?;
        }
        Ok(())
    }

    /// Node `id` (at game frame `frame`, made by `via`) met `obj`: the best meet per objective key (output
    /// only) and a `best_improved` event when sooner than a previous meet.
    fn note_met(&mut self, id: u32, obj: &Arc<dyn Objective>, c: crate::speed::Ctx, frame: u32, via: &'static str) -> Result<()> {
        let key = obj.key();
        let ck = (key.clone(), c);
        let old = self.speed.met_best.get(&ck).copied();
        if old.is_some_and(|o| o <= frame) {
            return Ok(());
        }
        self.speed.met_best.insert(ck, frame);
        if let Some(o) = old {
            self.speed.note_met_imp(via, (o - frame) as u64);
            writeln!(
                self.events,
                "{}",
                serde_json::json!({
                    "event": "best_improved", "milestone": "objective", "obj_key": key, "obj_kind": obj.kind(), "ctx": [c.0, c.1, c.2],
                    "old": o, "new": frame, "saved": o - frame, "node": id, "via": via,
                    "frames_total": self.total_frames, "from": null, "added": null,
                })
            )?;
        }
        Ok(())
    }

    /// New-eligible boost, on adding `id` (child of `parent`): records its
    /// key set and queues the keys its parent's set lacks.
    fn ne_note(&mut self, id: u32, parent: u32) {
        let t = Instant::now();
        let parent_set: Box<[u64]> = match self.ne.sets.get(&parent) {
            Some(s) => s.clone(),
            None => {
                let p = &self.tree.nodes[parent as usize];
                if p.retired {
                    Box::default()
                } else {
                    let st = p.st.clone();
                    self.ne_keys(&st).into_iter().map(|k| k.0).collect()
                }
            }
        };
        let st = self.tree.nodes[id as usize].st.clone();
        if st.modal() {
            // A dialog or menu can start nothing but its options: its
            // children compare with what came before it.
            self.ne.sets.insert(id, parent_set);
        } else {
            let keys = self.ne_keys(&st);
            for (h, k, w) in &keys {
                if parent_set.binary_search(h).is_ok() {
                    continue;
                }
                let n = self.ne.queued_per_key.entry(*h).or_default();
                if *n >= NE_PER_KEY {
                    continue;
                }
                *n += 1;
                self.ne.queued += 1;
                self.ne.queue.insert((w.to_bits(), self.ne.queued, id, k.clone()));
                if self.ne.queue.len() > NE_QUEUE {
                    self.ne.queue.pop_first();
                    self.ne.dropped += 1;
                }
            }
            self.ne.sets.insert(id, keys.into_iter().map(|k| k.0).collect());
        }
        self.ne.secs += t.elapsed().as_secs_f64();
    }

    /// New-eligible boost: the next queued (state, key) as b and objective,
    /// within this round's budget.
    fn plan_new_eligible(&mut self) -> Option<Plan> {
        let t = Instant::now();
        let mut out = None;
        while self.ne.budget > 0 {
            let Some((_, _, b, key)) = self.ne.queue.pop_last() else { break };
            let n = &self.tree.nodes[b as usize];
            if n.retired
                || n.st.modal()
                || !(self.tree.on_frontier(&n.st) || self.tree.is_wip(b))
                || (!any_exit() && n.st.course_done(n.st.course))
            {
                self.ne.stale += 1;
                continue;
            }
            self.ne.budget -= 1;
            self.ne.used += 1;
            self.tree.nodes[b as usize].stats.as_b += 1;
            let b_st = self.tree.nodes[b as usize].st.clone();
            for o in self.generator.generate(&b_st, &mut self.rng) {
                self.pool.insert(o, &b_st);
            }
            let Some(entry) = self.pool.find(&key) else { continue };
            let obj = self.pool.entries[entry].obj.clone();
            if obj.modal() || !obj.exists_in(&b_st) || !obj.base_criteria(&b_st) {
                continue;
            }
            let kind = obj.kind();
            self.diag.kind(kind).samples += 1;
            self.diag.kind(kind).window_samples += 1;
            match self.pick_start(obj.as_ref(), b) {
                Some(start) => {
                    self.pool.entries[entry].unattemptable = 0;
                    self.ne.planned += 1;
                    let stages = self.extend_chain(entry, obj);
                    out = Some(Plan { stages, start, b, seed: None });
                    break;
                }
                None => {
                    self.diag.kind(kind).no_start += 1;
                    self.diag.kind(kind).window_no_start += 1;
                }
            }
        }
        self.ne.secs += t.elapsed().as_secs_f64();
        out
    }

    fn plan(&mut self) -> Option<Plan> {
        if let Some(b) = self.modal_queue.pop() {
            return self.plan_modal(b, true);
        }
        if self.cfg.new_eligible && self.ne.budget > 0 {
            if let Some(p) = self.plan_new_eligible() {
                return Some(p);
            }
        }
        // Step 1: b from the frontier: a zone uniformly, then an occupied
        // 512-unit cell uniformly, then a state in it (see index_frontier).
        if self.b_index.is_empty() {
            return None;
        }
        // The bubble: its share of the plans draws b from inside it (over every other b draw below).
        let bubble_b = self.plan_bubble();
        // OBRRT_GOAL_FOCUS: a goal by its weight, then one of its best cells'
        // states; the goal itself from there, or b as usual.
        let mut goal_b = None;
        // The share falls with the best goal's weight (to a twentieth), so
        // stalled goals give the plans back.
        let goal_share = goal_focus().0 * if goal_explore() { 1.0 } else { self.goal_top.iter().map(|g| g.weight).fold(0.0, f64::max).clamp(0.05, 1.0) };
        if bubble_b.is_none() && !self.goal_top.is_empty() && self.rng.random_bool(goal_share) {
            let gw: Vec<(usize, f64)> = self.goal_top.iter().enumerate().map(|(i, g)| (i, g.weight)).collect();
            let gi = weighted(&mut self.rng, &gw).unwrap_or(0);
            let g = &self.goal_top[gi];
            let (b, cell) = g.cells[self.rng.random_range(0..g.cells.len())];
            let (key, obj) = (g.key.clone(), g.obj.clone());
            *self.goal_wear.entry((key.clone(), cell)).or_default() += 1;
            self.goal_stats.entry(key.clone()).or_insert((f64::INFINITY, 0, 0)).1 += 1;
            let n = &self.tree.nodes[b as usize];
            if !n.retired && !n.st.modal() {
                if self.rng.random_bool(goal_focus().2 * if goal_explore() { g.weight.clamp(0.1, 1.0) } else { 1.0 }) {
                    let st = n.st.clone();
                    // OBRRT_GEO=1: from beyond the goal's reach, a move to a
                    // point OBRRT_GEO_AHEAD (default 700) units along the
                    // walkable path instead (the goal's own straight-line
                    // steering would walk into the cliff below it).
                    let way = if obrrt_core::geo::enabled() {
                        obj.steer_target(&st)
                            .filter(|&t| obrrt_core::geo::path_dist(st.zone(), st.pos, t).is_some_and(|d| d > 1.5 * geo_ahead()))
                            .and_then(|t| obrrt_core::geo::waypoint(st.zone(), st.pos, t, geo_ahead()))
                    } else {
                        None
                    };
                    // OBRRT_ROAD: where the roadmap's path from here beats the
                    // straight line, its first edge instead (a move to the
                    // edge's end, seeded with the inputs that made it).
                    let leg = self.road.as_mut().and_then(|r| obj.steer_target(&st).and_then(|t| r.way(&st, t))).filter(|l| obrrt_core::dist(l.to, st.pos) > 150.0);
                    // No known way and the goal beyond one attempt (the roadmap's
                    // cost over 1.5 x the attempt's horizon): progress under the
                    // roadmap's direct cost instead (`Road::Climb`). BitDW's
                    // `BowserStage::Enter` took 27-31% of the frames from half
                    // a course away, 760 attempts, none in.
                    let far = match (&leg, self.road.as_mut(), obj.steer_target(&st)) {
                        (None, Some(r), Some(t)) => {
                            let hz = obj.ga_budget().map_or(200, |g| g.horizon) as f64;
                            (r.eta(&st, t) > 1.5 * hz).then(|| (t, r.climb()))
                        }
                        _ => None,
                    };
                    let mut seed = None;
                    let (key, obj): (String, Arc<dyn Objective>) = match (leg, way) {
                        (Some(l), _) => {
                            let m: Arc<dyn Objective> = Arc::new(obrrt_objectives::mario::PosDist::leg(&st, l.to, l.frames));
                            seed = l.node.and_then(|n| self.tree.nodes.get(n as usize)).filter(|n| !n.inputs.is_empty()).map(|n| n.inputs.clone());
                            if let Some(r) = self.road.as_mut() {
                                r.stats.legs += 1;
                                r.stats.legs_seeded += seed.is_some() as u64;
                            }
                            self.road_legs.push(b);
                            (m.key(), m)
                        }
                        (None, None) if far.is_some() => {
                            let (t, (per, jump)) = far.unwrap();
                            let m: Arc<dyn Objective> = Arc::new(obrrt_objectives::mario::PosDist::climb(&st, t, per, jump));
                            if let Some(r) = self.road.as_mut() {
                                r.stats.climbs += 1;
                            }
                            (m.key(), m)
                        }
                        (None, Some(p)) if obrrt_core::dist(p, st.pos) > 150.0 => {
                            let m: Arc<dyn Objective> = Arc::new(obrrt_objectives::mario::PosDist::new(&st, p));
                            (m.key(), m)
                        }
                        _ => (key, obj),
                    };
                    self.pool.insert(obj.clone(), &st);
                    if let Some(entry) = self.pool.find(&key) {
                        self.tree.nodes[b as usize].stats.as_b += 1;
                        return Some(Plan { stages: vec![(entry, obj)], start: b, b, seed });
                    }
                }
                goal_b = Some(b);
            }
        }
        let b = if let Some(b) = bubble_b {
            b
        } else if let Some(b) = goal_b {
            b
        } else if let Some(b) = self.manual_pick() {
            b
        } else if !self.route.is_empty() && self.rng.random_bool(self.cfg.route_share) {
            self.route[self.rng.random_range(0..self.route.len())]
        } else {
            let zw: Vec<(usize, f64)> = self.zone_weights.iter().copied().enumerate().collect();
            let zi = weighted(&mut self.rng, &zw).unwrap_or(0);
            let zone = &self.b_index[zi];
            let cell = &zone[self.rng.random_range(0..zone.len())];
            if self.cfg.soft_dedup {
                let w: Vec<(u32, f64)> = cell.iter().map(|&i| (i, 0.5f64.powi(self.tree.nodes[i as usize].stats.crowd as i32))).collect();
                weighted(&mut self.rng, &w).unwrap_or(cell[0])
            } else {
                cell[self.rng.random_range(0..cell.len())]
            }
        };
        self.tree.nodes[b as usize].stats.as_b += 1;
        if self.tree.nodes[b as usize].st.modal() {
            return self.plan_modal(b, false);
        }
        // Step 2: objectives from b.
        let b_st = self.tree.nodes[b as usize].st.clone();
        for o in self.generator.generate(&b_st, &mut self.rng) {
            self.pool.insert(o, &b_st);
        }
        // Steps 3-4, retrying a few times when a pick has no eligible start.
        for _ in 0..4 {
            // Wall time per pick (diagnostics only, never decisions).
            let pick_t = Instant::now();
            let Some(entry) = self.sample_entry() else { break };
            self.pool.rate.sample_secs += pick_t.elapsed().as_secs_f64();
            let obj = self.pool.entries[entry].obj.clone();
            let kind = obj.kind();
            self.diag.kind(kind).samples += 1;
            self.diag.kind(kind).window_samples += 1;
            let picked = self.pick_start(obj.as_ref(), b);
            let secs = pick_t.elapsed().as_secs_f64();
            if picked.is_some() { self.pool.rate.pick_secs += secs } else { self.pool.rate.nostart_secs += secs }
            match picked {
                Some(start) => {
                    self.pool.entries[entry].unattemptable = 0;
                    let stages = self.extend_chain(entry, obj);
                    return Some(Plan { stages, start, b, seed: None });
                }
                None => {
                    self.diag.kind(kind).no_start += 1;
                    self.diag.kind(kind).window_no_start += 1;
                    // Nowhere to start from decays it like a failure; it
                    // recovers when a state it could start from is added
                    // (the reset in merge).
                    self.pool.entries[entry].unattemptable += 1;
                    // Rate sampler: a pick with nowhere to start costs a few frames' worth of rate
                    // (faded like the rest).
                    let p = self.cfg.rate;
                    self.pool.rate_note(entry, 0.0, p.nostart_frames, self.total_frames, &p);
                    self.pool.rate_one(entry, self.total_frames, &p);
                    self.pool.rate.nostart_charges += 1;
                    if let Some(idx) = self.pool.sidx.as_mut() {
                        idx.touch(&self.pool.entries, entry);
                    }
                }
            }
        }
        self.pool.rate.plans_none += 1;
        None
    }

    /// Route chains (`Config::chain_route`): after `first`, the objective the
    /// best routes took next where they met it, if any (each follow-on with
    /// probability `chain_follow`), up to `max_chain` stages. Nothing is
    /// stacked that the routes didn't do, and nothing is learned from how
    /// chained attempts fared.
    fn extend_chain(&mut self, entry: usize, first: Arc<dyn Objective>) -> Vec<(usize, Arc<dyn Objective>)> {
        let mut stages = vec![(entry, first)];
        if stages[0].1.fixed().is_some() {
            return stages;
        }
        self.diag.route_chain.considered += 1;
        let mut key = stages[0].1.key();
        let mut keys = vec![key.clone()];
        while stages.len() < self.cfg.max_chain {
            let Some(hs) = self.route_met.get(&key) else { break };
            if stages.len() == 1 {
                self.diag.route_chain.on_route += 1;
            }
            // The next distinct objective below each node where the routes
            // met `key`, with the state it started from on the route.
            let mut cands: Vec<(String, Arc<dyn Objective>, u32)> = Vec::new();
            for &h in hs {
                self.route_next(h, &key, &mut cands);
            }
            cands.retain(|(k, o, from)| {
                let st = &self.tree.nodes[*from as usize].st;
                !keys.contains(k) && !o.modal() && o.fixed().is_none() && o.exists_in(st) && o.base_criteria(st)
            });
            if cands.is_empty() {
                break;
            }
            if stages.len() == 1 {
                self.diag.route_chain.next_known += 1;
            }
            if !self.rng.random_bool(self.cfg.chain_follow) {
                break;
            }
            let (k, obj, from) = cands.swap_remove(self.rng.random_range(0..cands.len()));
            let next = match self.pool.find(&k) {
                Some(e) => e,
                None => {
                    let st = self.tree.nodes[from as usize].st.clone();
                    self.pool.insert(obj.clone(), &st);
                    self.pool.entries.len() - 1
                }
            };
            if stages.len() == 1 {
                self.diag.route_chain.planned += 1;
            }
            self.diag.route_chain.appended += 1;
            keys.push(k.clone());
            stages.push((next, obj));
            key = k;
        }
        stages
    }

    /// Route chains: from route node `h` (where `key` was met), down its
    /// best-route children past further `key` nodes, the first other
    /// objective each branch was made by (distinct keys, in id order), with
    /// the node it started from. A branch through a node of unknown origin
    /// (a graft of an unknown one, the root) ends there.
    fn route_next(&self, h: u32, key: &str, out: &mut Vec<(String, Arc<dyn Objective>, u32)>) {
        let mut stack = vec![h];
        while let Some(at) = stack.pop() {
            for &c in self.tree.children[at as usize].iter().rev() {
                if !self.hot.contains(&c) {
                    continue;
                }
                let Some(m) = self.tree.nodes[c as usize].made_by.as_ref() else { continue };
                let k = m.key();
                if k == key {
                    stack.push(c);
                } else if !out.iter().any(|o| o.0 == k) {
                    out.push((k, m.clone(), at));
                }
            }
        }
    }

    /// A start's sampling multiplier: its back-off, or with the rate
    /// sampler, its trap back-off × its recent new states per frame over the
    /// tree's (prior `node_prior` frames at the tree's rate), clamped to
    /// [rate.floor, node_cap].
    fn start_mult(&self, s: &crate::tree::NodeStats) -> f64 {
        if !self.cfg.sampler_rate {
            return s.backoff;
        }
        let (gy, gf, _) = self.node_rate;
        if !(gf > 0.0) {
            return s.backoff;
        }
        let r0 = (gy / gf).max(1e-9);
        let k = (-(self.total_frames.saturating_sub(s.rate_t) as f64) / self.cfg.rate.tau).exp();
        let r = (s.rate_y * k + r0 * self.cfg.node_prior) / (s.rate_f * k + self.cfg.node_prior);
        s.backoff * (r / r0).clamp(self.cfg.rate.floor, self.cfg.node_cap)
    }

    /// Arrival credit (route chains, rate sampler): pool entries other than
    /// the plan's own stages, worth more than 1, that `start` could start
    /// (base criteria: a collectible's is just "in its zone and there", so
    /// one passed on the way to something else counts) and `start` -> `end` met
    /// (`success` over the two states) or whose object was there at the
    /// start and gone at the end in the same zone.
    ///
    /// Only the two end states are seen, so an objective whose success is
    /// generic (a door's is "left the zone") would match every objective of
    /// its kind: a kind with more than one hit in a segment is ambiguous and
    /// credits none.
    fn arrivals(&self, plan: &Plan, start: &St, end: &St) -> Vec<usize> {
        let pair = [start.clone(), end.clone()];
        // v2: met at the end of the pair (a fold from the start).
        let same_zone = start.zone() == end.zone();
        let hits: Vec<usize> = self.pool
            .entries
            .iter()
            .enumerate()
            .filter(|(i, e)| {
                e.base_weight > 1.0
                    && !e.obj.modal()
                    && !plan.stages.iter().any(|s| s.0 == *i)
                    && e.obj.exists_in(start)
                    && e.obj.base_criteria(start)
                    && (obrrt_core::met_at(e.obj.as_ref(), &pair).is_some_and(|i| i > 0) || (same_zone && !e.obj.exists_in(end)))
            })
            .map(|(i, _)| i)
            .collect();
        let mut per_kind: BTreeMap<&'static str, u32> = BTreeMap::new();
        for &i in &hits {
            *per_kind.entry(self.pool.entries[i].obj.kind()).or_default() += 1;
        }
        hits.into_iter().filter(|&i| per_kind[self.pool.entries[i].obj.kind()] == 1).collect()
    }

    fn write_m64(&self, id: u32, name: &str) -> Result<String> {
        let mut path = self.boot.clone();
        path.extend(self.tree.path(id));
        let file = format!("{}/{name}.m64", self.cfg.out_dir);
        let mut f = BufWriter::new(File::create(&file)?);
        for p in &path {
            f.write_all(&p.button.to_be_bytes())?;
            f.write_all(&[p.stick_x as u8, p.stick_y as u8])?;
        }
        Ok(file)
    }

    /// `tree.bin`: the boot inputs, then every node's parent and inputs, so
    /// any node can be replayed (`nodepath`). Little endian: u32 count, then
    /// per record i32 parent (-1 = boot) and u32 input count, 4 bytes each.
    fn write_tree(&self) -> Result<()> {
        let mut f = BufWriter::new(File::create(format!("{}/tree.bin", self.cfg.out_dir))?);
        f.write_all(&(self.tree.nodes.len() as u32 + 1).to_le_bytes())?;
        let mut record = |parent: i64, inputs: &[GamePad]| -> Result<()> {
            f.write_all(&(parent as i32).to_le_bytes())?;
            f.write_all(&(inputs.len() as u32).to_le_bytes())?;
            for p in inputs {
                f.write_all(&p.button.to_be_bytes())?;
                f.write_all(&[p.stick_x as u8, p.stick_y as u8])?;
            }
            Ok(())
        };
        record(-1, &self.boot)?;
        for n in &self.tree.nodes {
            record(n.parent.map_or(-1, |p| p as i64), &n.inputs)?;
        }
        Ok(())
    }

    /// Step 6 for one attempt: each segment (one per objective attempted)
    /// is merged against its own objective, entry and start; each added
    /// state is the parent of the next segment's.
    fn merge(&mut self, plan: &Plan, att: Attempt<Delta>) -> Result<()> {
        let dense = att.dense;
        self.total_frames += att.frames;
        let chain = self.attempts_merged;
        self.attempts_merged += 1;
        let len = plan.stages.len();
        let class = {
            let stages: Vec<&dyn Objective> = plan.stages.iter().map(|(_, o)| o.as_ref()).collect();
            crate::ga::budget_class(&stages)
        };
        let share = att.frames / att.segments.len().max(1) as u64;
        let last_seg = att.segments.len().saturating_sub(1);
        let met = att.segments.iter().filter(|s| s.success).count();
        let mut parent = plan.start;
        let mut carry: Vec<GamePad> = Vec::new();
        let mut seg_start = self.tree.nodes[plan.start as usize].st.clone();
        let (mut chain_added, mut chain_frontier, mut chain_zones, mut chain_value) = (0, 0, 0, 0.0);
        for (i, seg) in att.segments.into_iter().enumerate() {
            let (entry, obj) = (plan.stages[i].0, plan.stages[i].1.clone());
            let kind = obj.kind();
            // v2: progress is minus the ETA (frames), the start's from fresh
            // scratch, the end's as the controller folded it.
            let start_h = obrrt_core::start_eta(obj.as_ref(), &seg_start);
            let start_progress = -start_h;
            // Frames this segment's committed inputs play (to Done for a
            // success; the automatic tail not counted): with `start_h`, what
            // tools/analysis/eta_calib.py calibrates ETAs by.
            let seg_frames = seg.played;
            // What improvement on the start is judged by (`improves`).
            let start_k = self.keys_of(obj.as_ref(), &seg_start);
            let died = att.died && i == last_seg;
            // The node this segment's inputs start from (its state's parent).
            let from = parent;
            let mut added = None;
            // Added and new (no earlier state at its place): what yield,
            // decay and back-off count. Without soft dedupe, added = novel.
            let mut novel = false;
            let mut advanced = false;
            let mut next_start = None;
            let mut end_k = f64::INFINITY;
            carry.extend(seg.inputs);
            let seg_tail = seg.tail;
            let success = seg.success;
            if let Some((st, packed)) = seg.end {
                end_k = self.keys_of(obj.as_ref(), &st);
                // Progress alone counts only for objectives worth at least 1:
                // a failed attempt at a sign or a hidden 1-up trigger would
                // otherwise lay down legs of the route toward it.
                let weight = self.pool.entries.get(entry).map_or(1.0, |e| e.base_weight);
                let worth = success || (seg.progress > start_progress + 1e-6 && weight >= 1.0);
                let c = cell(&st);
                // When an attempt meets two or more goals, every met goal's
                // state is kept: each was reached set up for the next goal,
                // which an older state at the same place may not be.
                let chained = success && met >= 2;
                let dup = !chained
                    && self.cells.get(&c).is_some_and(|ids| ids.iter().any(|&i| self.tree.nodes[i as usize].st.frame <= st.frame));
                next_start = Some(st.clone());
                if worth {
                    novel = !dup;
                    let zone = st.zone();
                    // The worker packed the end against its start's base:
                    // kept as is when that is this zone's base (the same
                    // delta Delta::new makes here), else unpacked and redone.
                    let base = match self.zone_base.get(&zone) {
                        Some(&b) => b,
                        None => {
                            self.bases.push(crate::game::into_plain(packed.apply(&self.bases)?).into_bytes());
                            let b = (self.bases.len() - 1) as u16;
                            self.zone_base.insert(zone, b);
                            b
                        }
                    };
                    let delta = if packed.base == base { packed } else { Delta::new(base, &self.bases[base as usize], &packed.apply(&self.bases)?) };
                    if self.zones.insert(zone) {
                        self.diag.kind(kind).new_zones += 1;
                        chain_zones += 1;
                        println!("new zone {zone:?} via {kind} at node {}", self.tree.nodes.len());
                    }
                    let modal = st.modal();
                    let frame = st.frame;
                    let progress = st.progress();
                    let zone_now = st.zone();
                    let (id, adv) = self.tree.add(parent, std::mem::take(&mut carry), st, delta, kind);
                    // v2: the controller's plan from here seeds attempts from it.
                    self.tree.nodes[id as usize].tail = seg_tail;
                    if self.cfg.chain_route {
                        let n = &mut self.tree.nodes[id as usize];
                        n.made_by = Some(obj.clone());
                        n.met = seg.success;
                    }
                    if self.cfg.new_eligible {
                        self.ne_note(id, parent);
                    }
                    // Best game time per milestone, for the anytime curve.
                    self.note_best_at(id, progress, zone_now, frame, kind)?;
                    if seg.success {
                        self.note_met(id, &obj, crate::speed::ctx(&seg_start), frame, kind)?;
                    }
                    if modal {
                        self.modal_queue.push(id);
                    }
                    // Crowd: earlier states at its place; later ones gain one.
                    let mut crowd = 0;
                    for &o in self.cells.get(&c).map_or(&[][..], |v| v.as_slice()) {
                        if self.tree.nodes[o as usize].st.frame <= frame {
                            crowd += 1;
                        } else {
                            self.tree.nodes[o as usize].stats.crowd += 1;
                        }
                    }
                    self.tree.nodes[id as usize].stats.crowd = crowd;
                    self.cells.entry(c).or_default().push(id);
                    added = Some(id);
                    advanced = adv;
                    parent = id;
                }
            }
            // Stats.
            obj.note_result(success, seg_frames, added.map(|id| self.tree.nodes[id as usize].st.forward_vel));
            if let Some(t) = obj.diag_tag() {
                let e = self.tag_stats.entry(t).or_default();
                e[0] += 1;
                e[1] += success as u64;
                e[2] += novel as u64;
            }
            {
                let k = self.diag.kind(kind);
                k.attempts += 1;
                k.frames += share;
                k.died += died as u64;
                if success {
                    k.successes += 1;
                    k.success_starts.push(start_h);
                    if i > 0 {
                        k.chained_successes += 1;
                    }
                } else {
                    k.fails += 1;
                    // A start or end with a Lost ETA has no finite gain.
                    let gain = seg.progress - start_progress;
                    if gain.is_finite() {
                        k.fail_progress += gain;
                    }
                }
                if novel {
                    k.added += 1;
                    chain_added += 1;
                    if success {
                        k.value += obj.success_value();
                        chain_value += obj.success_value();
                    }
                }
                if advanced {
                    k.frontier += 1;
                    chain_frontier += 1;
                }
            }
            {
                let e = &mut self.pool.entries[entry];
                e.attempts += 1;
                // Improved on its start: met, or kept a state nearer by the objective's own heuristic.
                e.best_start = e.best_start.min(start_k);
                // Rate sampler: credited below, no decay.
                e.successes += success as u32;
            }
            let end_frame = next_start.as_ref().map(|s| s.frame);
            {
                // Improvement: a new state, the first success, frames saved
                // on its soonest success (1 per second, at most 5), or a
                // kept end closer (or sooner) than any before (0.5).
                let p = self.cfg.rate;
                let e = &mut self.pool.entries[entry];
                let (mut imp, mut first, mut sooner, mut closer) = (0.0, 0.0, 0.0, 0.0);
                let novel_i = if novel { p.novel } else { 0.0 };
                if let (true, Some(f)) = (success, end_frame) {
                    if e.best_hit == u32::MAX {
                        first = 1.0;
                    } else if f + 10 <= e.best_hit {
                        sooner = ((e.best_hit - f) as f64 / 30.0).min(5.0);
                    }
                    e.best_hit = e.best_hit.min(f);
                } else if added.is_some() && improves(end_k, e.best_end) {
                    closer = 0.5;
                }
                if added.is_some() {
                    e.best_end = e.best_end.min(end_k);
                }
                imp += novel_i + first + sooner + closer;
                let r = &mut self.pool.rate;
                r.imp_novel += novel_i;
                r.imp_first += first;
                r.imp_sooner += sooner;
                r.imp_closer += closer;
                self.pool.rate_note(entry, imp, share as f64, self.total_frames, &p);
            }
            if i > 0 {
                let rc = &mut self.diag.route_chain;
                rc.tried += 1;
                if success {
                    rc.met += 1;
                    let route_best = self.route_met.get(&obj.key()).and_then(|hs| hs.iter().map(|&h| self.tree.nodes[h as usize].st.frame).min());
                    if let (Some(f), Some(r)) = (end_frame, route_best) {
                        if f < r {
                            rc.sooner += 1;
                            rc.frames_saved += (r - f) as u64;
                        }
                    }
                }
            }
            // Arrival credit: other objectives this kept segment met on the
            // way (an attempt chases one objective; what it passes counts).
            if added.is_some() {
                if let Some(end) = next_start.as_ref() {
                    let hits = self.arrivals(plan, &seg_start, end);
                    let f = end.frame;
                    for j in hits {
                        let k = self.pool.entries[j].obj.kind();
                        self.diag.route_chain.arrivals += 1;
                        *self.diag.route_chain.arrivals_by_kind.entry(k).or_default() += 1;
                        let e = &mut self.pool.entries[j];
                        e.successes += 1;
                        let p = self.cfg.rate;
                        let imp = if e.best_hit == u32::MAX {
                            1.0
                        } else if f + 10 <= e.best_hit {
                            ((e.best_hit - f) as f64 / 30.0).min(5.0)
                        } else {
                            0.0
                        };
                        e.best_hit = e.best_hit.min(f);
                        if imp > 0.0 {
                            self.pool.rate.imp_first += imp;
                            self.pool.rate_note(j, imp, 0.0, self.total_frames, &p);
                        }
                    }
                }
            }
            if i == 0 {
                let ns = &mut self.tree.nodes[plan.start as usize].stats;
                ns.as_start += 1;
                if success {
                    ns.successes += 1;
                }
            }
            let ev = serde_json::json!({
                    "kind": kind, "key": obj.key(), "b": plan.b, "start": plan.start,
                    "chain": chain, "stage": i, "len": len, "from": from,
                    "success": seg.success, "progress": seg.progress - start_progress, "frames": share,
                    "start_eta": start_h.is_finite().then_some(start_h), "seg_frames": seg_frames,
                    "died": died, "added": added, "frontier": advanced,
                    "ga": att.ga.filter(|_| i == 0).map(|g| serde_json::json!({
                        "generations": g.generations, "winner_gen": g.winner_gen, "origin": g.winner_origin,
                        "gen0_met": g.gen0.0, "gen0_tie": g.gen0.1, "best_met": g.best.0, "best_tie": g.best.1,
                        "moves": g.moves,
                    })),
                });
            writeln!(self.events, "{ev}")?;
            if let Some(id) = added.filter(|_| novel) {
                if advanced {
                    let (keys, stars) = self.tree.frontier_progress;
                    let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
                    println!("FRONTIER keys {keys} stars {stars} at node {id} via {kind}: {f}");
                }
                if success && self.first_success.insert(kind) {
                    let f = self.write_m64(id, &format!("first-{}", kind.replace("::", "-")))?;
                    println!("first success of {kind} at node {id}: {f}");
                }
            }
            if let Some(st) = next_start {
                seg_start = st;
            }
        }
        if let Some(d) = dense {
            self.merge_dense(plan, d)?;
        }
        {
            // Rate sampler: the start's yield per frame (see start_mult);
            // its back-off only carries emulator traps, cleared by a yield.
            let (now, tau) = (self.total_frames, self.cfg.rate.tau);
            let fade = |t: u64| (-(now.saturating_sub(t) as f64) / tau).exp();
            let g = &mut self.node_rate;
            let k = fade(g.2);
            *g = (g.0 * k + chain_added as f64, g.1 * k + att.frames as f64, now);
            let ns = &mut self.tree.nodes[plan.start as usize].stats;
            let k = fade(ns.rate_t);
            ns.rate_y = ns.rate_y * k + chain_added as f64;
            ns.rate_f = ns.rate_f * k + att.frames as f64;
            ns.rate_t = now;
            if chain_added > 0 {
                ns.added += chain_added;
                ns.barren = 0;
                ns.backoff = 1.0;
            } else {
                ns.barren += 1;
            }
        }
        self.diag.window_starts.push(plan.start);
        let c = self.diag.chains.entry(len).or_default();
        c.met.resize(len + 1, 0);
        c.attempts += 1;
        c.frames += att.frames;
        c.met[met.min(len)] += 1;
        c.added += chain_added as u64;
        c.frontier += chain_frontier;
        c.new_zones += chain_zones;
        c.value += chain_value;
        let b = self.diag.budgets.entry(class).or_default();
        if b.met.len() <= len {
            b.met.resize(len + 1, 0);
        }
        b.attempts += 1;
        b.frames += att.frames;
        b.met[met.min(len)] += 1;
        b.added += chain_added as u64;
        b.frontier += chain_frontier;
        b.new_zones += chain_zones;
        b.value += chain_value;
        Ok(())
    }

    /// Dense adds: an attempt's candidates (see dense.rs), in order, each
    /// added if still new (dense key and cell, against the tree as it is now)
    /// under the previous kept candidate of its chain (or the start), with
    /// the inputs of those dropped carried on. Bookkeeping as for a kept
    /// segment end; yield, decay and kind statistics don't count them.
    fn merge_dense(&mut self, plan: &Plan, out: Box<crate::dense::Out>) -> Result<()> {
        let obj = plan.stages[0].1.clone();
        let via = crate::dense::via(obj.kind());
        let out = *out;
        self.dstats.attempts += 1;
        self.dstats.worker.add(&out.stats);
        let n = out.cands.iter().map(|c| c.chain as usize + 1).max().unwrap_or(0);
        let mut last = vec![plan.start; n];
        let mut carry: Vec<Vec<GamePad>> = vec![Vec::new(); n];
        let ne = self.cfg.new_eligible && self.dense.is_none_or(|d| d.ne);
        for c in out.cands {
            let ci = c.chain as usize;
            carry[ci].extend(c.inputs);
            let parent = last[ci];
            let st = c.st;
            let cc = cell(&st);
            let dup = !self.tree.dense_novel(c.key, st.frame)
                || self.cells.get(&cc).is_some_and(|ids| ids.iter().any(|&i| self.tree.nodes[i as usize].st.frame <= st.frame));
            if dup {
                self.dstats.dup += 1;
                continue;
            }
            if c.save.is_none() && !self.tree.can_anchor(parent) {
                self.dstats.no_anchor += 1;
                continue;
            }
            let zone = st.zone();
            let delta = match c.save {
                Some(packed) => {
                    let base = match self.zone_base.get(&zone) {
                        Some(&b) => b,
                        None => {
                            self.bases.push(crate::game::into_plain(packed.apply(&self.bases)?).into_bytes());
                            let b = (self.bases.len() - 1) as u16;
                            self.zone_base.insert(zone, b);
                            b
                        }
                    };
                    if packed.base == base { packed } else { Delta::new(base, &self.bases[base as usize], &packed.apply(&self.bases)?) }
                }
                None => Delta::none(),
            };
            if self.zones.insert(zone) {
                println!("new zone {zone:?} via {via} at node {}", self.tree.nodes.len());
            }
            let (modal, frame, progress) = (st.modal(), st.frame, st.progress());
            let (id, adv) = self.tree.add(parent, std::mem::take(&mut carry[ci]), st, delta, via);
            self.tree.nodes[id as usize].made_by = Some(obj.clone());
            if ne {
                self.ne_note(id, parent);
            }
            self.note_best_at(id, progress, zone, frame, via)?;
            if modal {
                self.modal_queue.push(id);
            }
            if self.cfg.soft_dedup {
                let mut crowd = 0;
                for &o in self.cells.get(&cc).map_or(&[][..], |v| v.as_slice()) {
                    if self.tree.nodes[o as usize].st.frame <= frame {
                        crowd += 1;
                    } else {
                        self.tree.nodes[o as usize].stats.crowd += 1;
                    }
                }
                self.tree.nodes[id as usize].stats.crowd = crowd;
            }
            self.cells.entry(cc).or_default().push(id);
            self.dstats.added += 1;
            last[ci] = id;
            if adv {
                self.dstats.frontier += 1;
                let (keys, stars) = self.tree.frontier_progress;
                let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
                println!("FRONTIER keys {keys} stars {stars} at node {id} via {via}: {f}");
            }
        }
        Ok(())
    }

    /// Bytes the soft-dedupe memory budget is checked against: saves, and
    /// with dense adds (OBRRT_DENSE_MEMOBJ, on) the nodes' object lists too.
    fn mem_used(&self) -> usize {
        match self.dense {
            Some(d) if d.mem_objects => self.tree.delta_bytes + self.tree.obj_bytes,
            _ => self.tree.delta_bytes,
        }
    }

    /// Automatic tuning, tier 1: each kind's multiplier follows its yield
    /// relative to the mean.
    fn tune(&mut self) {
        // Chains: extend more often while they yield more per frame than
        // single objectives (ratio 1 -> 0.5).
        let single = self.diag.chains.get(&1).filter(|c| c.attempts >= 20).map(|c| c.yield_per_kframe());
        let mut multi = crate::diag::ChainStats::default();
        for c in self.diag.chains.range(2..).map(|(_, c)| c) {
            multi.attempts += c.attempts;
            multi.frames += c.frames;
            multi.added += c.added;
            multi.frontier += c.frontier;
            multi.new_zones += c.new_zones;
            multi.value += c.value;
        }
        // Route chains learn nothing from how chains fared.
        if let Some(single) = single.filter(|_| multi.attempts >= 20 && !self.cfg.chain_route) {
            let ratio = (multi.yield_per_kframe() + 0.05) / (single + 0.05);
            self.diag.chain_prob = (0.7 * self.diag.chain_prob + 0.3 * (0.5 * ratio)).clamp(0.1, 0.9);
        }
        let mean = self.diag.mean_yield();
        for k in self.diag.kinds.values_mut() {
            if k.attempts < 10 {
                continue;
            }
            {
                let y = k.yield_per_kframe();
                let target = ((y + 0.05) / (mean + 0.05)).clamp(0.1, 5.0);
                let old = k.multiplier;
                k.multiplier = 0.7 * k.multiplier + 0.3 * target;
                if (k.multiplier - old).abs() > 0.05 {
                    k.reason = format!("yield {y:.2} vs mean {mean:.2}");
                }
            }
            // Low success: only start where successes started (90th
            // percentile of their start ETA, plus a margin: v2 ETAs are in
            // frames, so a second rather than v1's 1.0 of distance/400).
            if k.success_starts.len() >= 3 && k.success_rate() < 0.25 {
                let mut hs = k.success_starts.clone();
                hs.sort_by(f64::total_cmp);
                let p90 = hs[((hs.len() - 1) as f64 * 0.9) as usize];
                let learned = p90 + 0.25 * p90.abs() + 30.0;
                if learned < k.cutoff {
                    k.cutoff = learned;
                    k.reason = format!("success {:.0}%: starts cut to h <= {learned:.1}", 100.0 * k.success_rate());
                    k.window_samples = 0;
                    k.window_no_start = 0;
                }
            }
            // Starved by its cutoff: loosen it.
            if k.cutoff.is_finite() && k.window_samples >= 20 && k.window_no_start * 2 > k.window_samples {
                k.cutoff *= 1.5;
                k.reason = format!("starved: cutoff loosened to {:.1}", k.cutoff);
                k.window_samples = 0;
                k.window_no_start = 0;
            }
        }
    }

    fn write_outputs(&mut self, secs: f64) -> Result<()> {
        for k in self.diag.kinds.values_mut() {
            k.pool_size = 0;
        }
        let mut sizes: BTreeMap<&'static str, usize> = BTreeMap::new();
        for e in &self.pool.entries {
            *sizes.entry(e.obj.kind()).or_default() += 1;
        }
        for (k, n) in sizes {
            self.diag.kind(k).pool_size = n;
        }
        print!("{}", self.diag.report(&self.tree, secs, self.total_frames));
        self.events.flush()?;
        let mut csv = String::from("id,parent,depth,x,y,z,level,area,stars,keys,coins,frame,via,as_b,as_start,successes,added,barren,hspeed,fwd_vel,action,red_coins,act,star_flags,retired,caps,save_flags,health\n");
        for (i, n) in self.tree.nodes.iter().enumerate() {
            let s = &n.st;
            let st = &n.stats;
            csv += &format!(
                "{i},{},{},{:.0},{:.0},{:.0},{},{},{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:#x},{},{},{},{},{},{:#x},{:#x}\n",
                n.parent.map_or(-1, |p| p as i64),
                n.depth,
                s.pos[0],
                s.pos[1],
                s.pos[2],
                s.level,
                s.area,
                s.stars,
                s.keys,
                s.coins,
                s.frame,
                n.via,
                st.as_b,
                st.as_start,
                st.successes,
                st.added,
                st.barren,
                (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt(),
                s.forward_vel,
                s.action,
                s.red_coins,
                s.act,
                // Collected stars per course, as course:flags (hex).
                s.course_stars
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| **f != 0)
                    .map(|(c, f)| format!("{}:{f:x}", c + 1))
                    .collect::<Vec<_>>()
                    .join(" "),
                n.retired as u8,
                // Cap switches pressed (save flags: 2 wing, 4 metal, 8 vanish).
                s.save_flags & 0xE,
                // The whole save flags word (the moat drained 0x200, keys, doors, castle stars).
                s.save_flags,
                // Health (0x880 full; one segment is 0x100).
                s.health
            );
        }
        fs::write(format!("{}/nodes.csv", self.cfg.out_dir), csv)?;
        self.write_tree()?;
        // Per kind: entries and mean objective weight multiplier (decay).
        let mut weights: BTreeMap<&'static str, (usize, f64)> = BTreeMap::new();
        for e in &self.pool.entries {
            let w = weights.entry(e.obj.kind()).or_default();
            w.0 += 1;
            w.1 += e.decay;
        }
        let weights: BTreeMap<&str, serde_json::Value> = weights
            .into_iter()
            .map(|(k, (n, d))| (k, serde_json::json!({"entries": n, "mean_decay": d / n as f64})))
            .collect();
        let summary = serde_json::json!({
            "secs": secs, "nodes": self.tree.nodes.len(), "frames": self.total_frames,
            "frontier": {"keys": self.tree.frontier_progress.0, "stars": self.tree.frontier_progress.1, "states": self.tree.frontier.len()},
            "effective_starts": self.diag.effective_starts(),
            "kinds": &self.diag.kinds,
            "budgets": &self.diag.budgets,
            "entry_weights": weights,
        });
        let mut summary = summary;
        if self.tree.keep_wip {
            let t = &self.tree;
            summary["frontier"]["wip"] = serde_json::json!({
                "kept": t.wip_kept.len(), "peak": t.wip_peak, "ever": t.wip_ever.len(), "dropped": t.wip_dropped,
                "kept_list": t.wip_kept.iter().map(|&i| serde_json::json!({"id": i, "level": t.nodes[i as usize].st.level, "stars": t.nodes[i as usize].st.stars, "frame": t.nodes[i as usize].st.frame, "wip": t.wip[i as usize]})).collect::<Vec<_>>(),
            });
        }
        summary["decay_resets"] = serde_json::json!(self.decay_resets);
        self.generator.report(&self.cfg.out_dir);
        for (name, text) in self.generator.summary_blocks() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                summary[name.as_str()] = v;
            }
        }
        for (tag, [a, s, n]) in &self.tag_stats {
            let (blk, sub) = tag.split_once('/').unwrap_or(("tags", tag.as_str()));
            summary[blk]["attempts"][sub] = serde_json::json!({"attempts": a, "successes": s, "added": n});
        }
        summary["speed"] = self.speed.summary();
        if let Some(sp) = &self.tree.sparse {
            use std::sync::atomic::Ordering::Relaxed;
            let n = self.tree.nodes.len();
            let saved = sp.saved.iter().filter(|&&x| x).count();
            let (rb, rf) = (sp.rebuilds.load(Relaxed), sp.rebuild_frames.load(Relaxed));
            println!(
                "   sparse saves (cap {}): {saved} of {n} nodes hold a save; dropped {} kept {} overshoot {} held {} max dist {}; {rb} rebuilds, {rf} frames; mem {} MB ({} B/node; saves {} MB, objects {} MB)",
                sp.cap, sp.dropped, sp.kept, sp.overshoot, sp.held, sp.max_dist,
                self.tree.mem_bytes() / 1_000_000, self.tree.mem_bytes() / n.max(1), self.tree.delta_bytes / 1_000_000, self.tree.obj_bytes / 1_000_000
            );
            summary["sparse"] = serde_json::json!({
                "cap": sp.cap, "nodes": n, "saved": saved, "dropped": sp.dropped, "kept": sp.kept, "overshoot": sp.overshoot,
                "held": sp.held, "max_dist": sp.max_dist, "rebuilds": rb, "rebuild_frames": rf,
                "mem_bytes": self.tree.mem_bytes(), "delta_bytes": self.tree.delta_bytes, "obj_bytes": self.tree.obj_bytes,
            });
        }
        if self.dense.is_some() {
            let d = &self.dstats;
            println!(
                "   dense adds: {} attempts, {} added ({} frontier), {} dup at merge, {} no anchor, {} pruned; worker seen {} novel {} beaten {} capped {} returned {} saves {} replay {} frames; keys {}",
                d.attempts, d.added, d.frontier, d.dup, d.no_anchor, d.pruned, d.worker.seen, d.worker.novel, d.worker.beaten, d.worker.capped, d.worker.returned, d.worker.saves, d.worker.replay_frames, self.tree.dkeys.len()
            );
            summary["dense"] = serde_json::json!(d);
            summary["dense"]["keys"] = serde_json::json!(self.tree.dkeys.len());
        }
        if let Some(l) = self.tree.kin_line() {
            println!("   {l}");
        }
        summary["timing"] =serde_json::json!({"plan_secs": self.diag_plan_secs.0, "merge_secs": self.diag_plan_secs.1});
        {
            let d = &self.diag_secs;
            for (k, v) in ["pre_secs", "run_secs", "post_secs", "write_secs", "worker_load_secs", "worker_attempt_secs"].iter().zip(d) {
                summary["timing"][*k] = serde_json::json!(v);
            }
            summary["timing"]["index_secs"] = serde_json::json!(self.diag_index_secs);
            let q = self.diag_pre;
            println!("   pre parts: eta {:.1}s nostart {:.1}s rate {:.1}s wip {:.1}s index {:.1}s (setup {:.1}s road {:.1}s loop {:.1}s goals {:.1}s zones {:.1}s routes {:.1}s)", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10]);
            println!("   timing: pre {:.1}s (goal etas {:.1}s) plan {:.1}s run {:.1}s merge {:.1}s post {:.1}s write {:.1}s; workers load {:.1}s attempt {:.1}s",
                d[0], self.diag_index_secs, self.diag_plan_secs.0, d[1], self.diag_plan_secs.1, d[2], d[3], d[4], d[5]);
        }
        if crate::movegenome::enabled() {
            println!("   {}", crate::movegenome::stats_line());
        }
        println!("   population: {}", crate::ga::makeup_line());
        let d = crate::diag::door_overshoots();
        if d.attempts > 0 {
            println!("   door attempts {} met {} overshoots {} (of met {})", d.attempts, d.met, d.overshoots, d.overshoots_met);
        }
        if self.cfg.chain_route || self.cfg.sampler_rate {
            println!("   {}", self.diag.route_chain.line());
            summary["chain_route"] = serde_json::json!(&self.diag.route_chain);
        }
        if self.cfg.sampler_rate {
            let r = &self.pool.rate;
            // Start multipliers over live, non-modal states.
            let mut m: Vec<f64> = self.tree.nodes.iter().filter(|n| !n.retired && !n.st.modal()).map(|n| self.start_mult(&n.stats)).collect();
            m.sort_by(f64::total_cmp);
            let q = |f: f64| if m.is_empty() { 0.0 } else { m[((m.len() - 1) as f64 * f).round() as usize] };
            let floor = self.cfg.rate.floor;
            let node_floor = m.iter().filter(|&&x| x <= floor + 1e-12).count();
            println!(
                "   rate sampler: entries {} weight min {:.2} p25 {:.2} median {:.2} p75 {:.2} max {:.2}, at floor {} at cap {}; r0 {:.2}/Mframe median {:.2}/Mframe; imp novel {:.0} first {:.0} sooner {:.1} closer {:.1}; no-start charges {}; starts {} mult min {:.2} median {:.2} max {:.2}, at floor {}",
                r.entries, r.min, r.p25, r.p50, r.p75, r.max, r.at_floor, r.at_cap, r.r0 * 1e6, r.median * 1e6,
                r.imp_novel, r.imp_first, r.imp_sooner, r.imp_closer, r.nostart_charges, m.len(), q(0.0), q(0.5), q(1.0), node_floor
            );
            println!(
                "   rate sampler: weight p10 {:.2} p90 {:.2}; picks {:.1}s started {:.1}s no start (sampling {:.1}s), plans none {}",
                r.p10, r.p90, r.pick_secs, r.nostart_secs, r.sample_secs, r.plans_none
            );
            summary["rate"] = serde_json::json!({
                "entries": r, "starts": {"n": m.len(), "min": q(0.0), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": q(1.0), "at_floor": node_floor},
            });
        }
        if self.cfg.new_eligible {
            summary["new_eligible"] = serde_json::json!({
                "queued": self.ne.queued, "dropped": self.ne.dropped, "stale": self.ne.stale, "used": self.ne.used,
                "planned": self.ne.planned, "queue": self.ne.queue.len(), "keys": self.ne.queued_per_key.len(),
                "sets": self.ne.sets.len(), "secs": self.ne.secs,
            });
        }
        if crate::movegenome::enabled() {
            summary["move_ga"] = crate::movegenome::stats_json();
        }
        // OBRRT_GA_QD (qd.rs): only when on.
        if let Some(q) = crate::qd::summary() {
            summary["ga_qd"] = q;
        }
        if let Some(r) = self.road.as_mut() {
            summary["road"] = serde_json::json!(r.summary());
            r.save();
        }
        // The goals' checkpoints (GoalTop::cp), heaviest goal first.
        let mut goals: Vec<serde_json::Value> = Vec::new();
        for g in &self.goal_top {
            let cps: Vec<serde_json::Value> = g
                .cps
                .iter()
                .map(|&(wip, eta, n)| {
                    let st = &self.tree.nodes[n as usize].st;
                    serde_json::json!({ "node": n, "eta": eta, "wip": wip, "level": st.level, "area": st.area, "frame": st.frame, "pos": st.pos })
                })
                .collect();
            if !cps.is_empty() {
                goals.push(serde_json::json!({
                    "key": g.key, "kind": g.obj.kind(), "weight": g.base, "eta0": g.h0, "eta1": g.h1,
                    "wip0": self.tree.wip.first().copied().unwrap_or(0), "cps": cps,
                }));
            }
        }
        goals.sort_by(|a, b| b["weight"].as_f64().unwrap_or(0.0).total_cmp(&a["weight"].as_f64().unwrap_or(0.0)));
        summary["goals"] = serde_json::json!(goals);
        summary["door_overshoot"] = serde_json::json!(crate::diag::door_overshoots());
        fs::write(format!("{}/summary.json", self.cfg.out_dir), serde_json::to_string_pretty(&summary)?)?;
        self.diag.window_starts.clear();
        Ok(())
    }

    /// Continues a finished run: replays its tree.bin (the same boot) to
    /// rebuild every state, in creation order, so each parent is ready before
    /// its children. Objectives and statistics start fresh; outputs go to
    /// this run's out_dir and include the replayed states.
    ///
    /// Every record is replayed from its parent's own save: in the original
    /// run an attempt always started from a live save (states are retired
    /// only between planning and the next round), so states behind the
    /// frontier are kept until their last child is replayed, and retired at
    /// the end. Each replayed state is checked against the run's nodes.csv;
    /// a mismatch is retried from the root's save, which is how resumes
    /// before this fix replayed a child of a state retired earlier in the
    /// replay (so a tree made by such a resume replays as it was run).
    pub fn resume(&mut self, dir: &str) -> Result<()> {
        let data = fs::read(format!("{dir}/tree.bin"))?;
        let csv = fs::read_to_string(format!("{dir}/nodes.csv"))?;
        let header: Vec<&str> = csv.lines().next().unwrap_or("").split(',').collect();
        let col = |name: &str| header.iter().position(|h| *h == name);
        let (cx, cy, cz, clevel, carea, cstars, cvia) =
            (col("x"), col("y"), col("z"), col("level"), col("area"), col("stars"), col("via").unwrap_or(12));
        // Per node: via, and (level, area, stars, pos) to check the replay by.
        let rows: Vec<(&'static str, Option<(u16, u8, i16, [f32; 3])>)> = csv
            .lines()
            .skip(1)
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                let via = &*Box::leak(f.get(cvia).copied().unwrap_or("resumed").to_string().into_boxed_str());
                let get = |c: Option<usize>| c.and_then(|c| f.get(c)).and_then(|v| v.parse::<f64>().ok());
                let check = (|| {
                    Some((
                        get(clevel)? as u16,
                        get(carea)? as u8,
                        get(cstars)? as i16,
                        [get(cx)? as f32, get(cy)? as f32, get(cz)? as f32],
                    ))
                })();
                (via, check)
            })
            .collect();
        let u32_at = |p: usize| u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
        let count = u32_at(0) as usize;
        let mut pos = 4;
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            let parent = u32_at(pos) as i32;
            let n = u32_at(pos + 4) as usize;
            pos += 8;
            let inputs: Vec<GamePad> = (0..n)
                .map(|k| {
                    let b = &data[pos + 4 * k..pos + 4 * k + 4];
                    GamePad::new(u16::from_be_bytes([b[0], b[1]]), b[2] as i8, b[3] as i8)
                })
                .collect();
            pos += 4 * n;
            records.push((parent, inputs));
        }
        anyhow::ensure!(records.first().is_some_and(|r| r.1 == self.boot), "{dir} was booted differently");
        // The last record that is each node's child: after it, a node behind
        // the frontier can be retired.
        let mut last_child = vec![0usize; count];
        for (i, (parent, _)) in records.iter().enumerate().skip(2) {
            last_child[*parent as usize] = i;
        }
        let started = Instant::now();
        let (mut via_root, mut mismatched) = (0usize, 0usize);
        self.tree.hold_retire = true;
        // Record 0 is the boot, record 1 the root (node 0); node i is record i + 1.
        for (i, (parent, inputs)) in records.into_iter().enumerate().skip(2) {
            let id = i - 1;
            let parent = parent as u32;
            let pnode = &self.tree.nodes[parent as usize];
            let frame = pnode.st.frame + inputs.len() as u32;
            let check = rows.get(id).and_then(|r| r.1);
            let matches = |st: &St| {
                check.is_none_or(|(level, area, stars, p)| {
                    st.level == level
                        && st.area == area
                        && st.stars == stars
                        && (0..3).all(|k| (st.pos[k] - p[k]).abs() <= 1.0)
                })
            };
            let worker = self.workers[0].get_mut().unwrap();
            let tree = &self.tree;
            // From the parent (`Tree::load_into`: its save, or with sparse
            // saves its anchor's and the inputs), or from the root's save.
            #[cfg(feature = "hooked")]
            let mut replay = |_: Option<u32>, _: &[Vec<u8>]| -> Result<(St, crate::game::GameSave)> { unimplemented!("state tree: hooked feature") };
            #[cfg(not(feature = "hooked"))]
            let mut replay = |from: Option<u32>, bases: &[Vec<u8>]| -> Result<(St, SaveState)> {
                match from {
                    Some(p) => {
                        tree.load_into(p, &mut worker.base, bases)?;
                    }
                    None => worker.base.load_state(&tree.nodes[0].save.apply(bases)?)?,
                }
                for p in &inputs {
                    worker.base.step_game(*p)?;
                }
                Ok((read_state(&worker.base, frame, true), worker.base.save_state()))
            };
            let _ = pnode;
            let (mut st, mut save) = if tree.sparse.is_some() && !tree.can_anchor(parent) { replay(None, &self.bases)? } else { replay(Some(parent), &self.bases)? };
            if !matches(&st) && parent != 0 {
                let (st0, save0) = replay(None, &self.bases)?;
                if matches(&st0) {
                    (st, save) = (st0, save0);
                    via_root += 1;
                } else {
                    mismatched += 1;
                }
            } else if !matches(&st) {
                mismatched += 1;
            }
            let zone = st.zone();
            let base = match self.zone_base.get(&zone) {
                Some(&b) => b,
                None => {
                    self.bases.push(crate::game::plain(&save).as_bytes().to_vec());
                    let b = (self.bases.len() - 1) as u16;
                    self.zone_base.insert(zone, b);
                    b
                }
            };
            let delta = Delta::new(base, &self.bases[base as usize], &save);
            self.zones.insert(zone);
            self.diag.note_best(st.progress(), zone, frame);
            let c = cell(&st);
            let via = rows.get(id).map_or("resumed", |r| r.0);
            let frame = st.frame;
            let (nid, _) = self.tree.add(parent, inputs, st, delta, via);
            debug_assert_eq!(nid as usize, id);
            if self.cfg.soft_dedup {
                // Crowd as merge counts it: earlier states at the same place.
                let mut crowd = 0;
                for &o in self.cells.get(&c).map_or(&[][..], |v| v.as_slice()) {
                    if self.tree.nodes[o as usize].st.frame <= frame {
                        crowd += 1;
                    } else {
                        self.tree.nodes[o as usize].stats.crowd += 1;
                    }
                }
                self.tree.nodes[nid as usize].stats.crowd = crowd;
            }
            self.cells.entry(c).or_default().push(nid);
            // Saves no later record needs: states behind the frontier after
            // their last child (or with none to come).
            // OBRRT_KEEP_WIP: states with work under way wait for the
            // selection after the replay.
            let wip_hold = |t: &crate::tree::Tree, n: usize| t.keep_wip && t.wip[n] > 0;
            if last_child[parent as usize] == i && !self.tree.on_frontier(&self.tree.nodes[parent as usize].st) && !wip_hold(&self.tree, parent as usize) {
                self.tree.retire_one(parent);
            }
            if last_child[id] <= i && !self.tree.on_frontier(&self.tree.nodes[id].st) && !wip_hold(&self.tree, id) {
                self.tree.retire_one(nid);
            }
        }
        self.tree.hold_retire = false;
        self.tree.retire_behind();
        println!(
            "resumed {} states from {dir} in {:.0}s; frontier keys {} stars {} ({} states); {via_root} replayed from the root's save, {mismatched} not matching nodes.csv",
            self.tree.nodes.len(),
            started.elapsed().as_secs_f64(),
            self.tree.frontier_progress.0,
            self.tree.frontier_progress.1,
            self.tree.frontier.len()
        );
        Ok(())
    }

    /// Soft dedupe over the memory budget: retires repeated states (an
    /// earlier state got to their place), lowest sampling weight first,
    /// until saves are under 90% of the budget. The earliest state at each
    /// place, the root and the best routes are never retired.
    fn prune_repeats(&mut self) {
        let mut cands: Vec<(f64, u32)> = self
            .tree
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i > 0 && !n.retired && n.stats.crowd > 0 && !self.hot.contains(&(*i as u32)))
            .map(|(i, n)| (self.start_mult(&n.stats) * 0.5f64.powi(n.stats.crowd as i32), i as u32))
            .collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let target = self.cfg.mem_budget / 10 * 9;
        let mut freed = 0;
        for (_, i) in cands {
            if self.mem_used() <= target {
                break;
            }
            self.tree.retire_one(i);
            freed += 1;
        }
        // Dense adds: still over, dense nodes go too (never a start first,
        // then the lowest start weight, the latest first among equals), not
        // on the best routes or kept for work under way. Their object lists
        // are most of the memory (about 10 KB a node).
        if self.dense.is_some() && self.mem_used() > target {
            let mut cands: Vec<(bool, f64, std::cmp::Reverse<u32>)> = self
                .tree
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| *i > 0 && !n.retired && n.via.ends_with("~d") && !self.hot.contains(&(*i as u32)) && !self.tree.is_wip(*i as u32))
                .map(|(i, n)| (n.stats.as_start > 0, self.start_mult(&n.stats) * 0.5f64.powi(n.stats.crowd as i32), std::cmp::Reverse(i as u32)))
                .collect();
            cands.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
            let mut dense = 0;
            for (_, _, std::cmp::Reverse(i)) in cands {
                if self.mem_used() <= target {
                    break;
                }
                self.tree.retire_one(i);
                dense += 1;
            }
            freed += dense;
            self.dstats.pruned += dense as u64;
        }
        println!("soft dedupe: retired {freed} repeated states; saves now {} MB", self.mem_used() / 1_000_000);
    }

    /// The nodes worth offering to other searches of the same step (at most `k`, each once): the best
    /// state of the engine's top goals' best cells (`goal_top`, its own ETA priority), taken round-robin
    /// across goals, then frontier states by progress and sooner frame, at least `sep` units apart from
    /// each other so the offer covers several branches. Imported nodes are never offered back.
    pub fn sync_candidates(&mut self, k: usize, sep: f32) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        let ok = |s: &Self, i: u32| i > 0 && (i as usize) < s.tree.nodes.len() && !s.tree.nodes[i as usize].retired && s.tree.nodes[i as usize].via != "sync" && !s.sync_sent.contains(&i);
        let far = |s: &Self, out: &[u32], i: u32| {
            let p = s.tree.nodes[i as usize].st.pos;
            out.iter().all(|&o| {
                let q = s.tree.nodes[o as usize].st.pos;
                (p[0] - q[0]).hypot(p[2] - q[2]) + (p[1] - q[1]).abs() >= sep
            })
        };
        let goals = self.goal_top.iter().map(|g| g.cells.len()).max().unwrap_or(0);
        let mut order: Vec<usize> = (0..self.goal_top.len()).collect();
        order.sort_by(|&a, &b| self.goal_top[b].weight.total_cmp(&self.goal_top[a].weight));
        'g: for r in 0..goals {
            for &g in &order {
                if out.len() >= k.div_ceil(2) {
                    break 'g;
                }
                if let Some(&(i, _)) = self.goal_top[g].cells.get(r) {
                    if ok(self, i) && !out.contains(&i) && far(self, &out, i) {
                        out.push(i);
                    }
                }
            }
        }
        let mut fr: Vec<u32> = self.tree.frontier.iter().copied().filter(|&i| ok(self, i)).collect();
        fr.sort_by(|&a, &b| {
            let (x, y) = (&self.tree.nodes[a as usize].st, &self.tree.nodes[b as usize].st);
            y.progress().cmp(&x.progress()).then(x.frame.cmp(&y.frame)).then(a.cmp(&b))
        });
        for i in fr.into_iter().take(4 * k) {
            if out.len() >= k {
                break;
            }
            if !out.contains(&i) && far(self, &out, i) {
                out.push(i);
            }
        }
        self.sync_sent.extend(out.iter().copied());
        out
    }

    /// Adds the state another search reached with `pads` from this tree's root, as this search's own
    /// expansion would: the deepest node on the path already in the tree is the parent (its save is
    /// loaded and the rest of the pads replayed, game frames counted), the result is checked against
    /// `expect` (level, position, progress, as the other search read it) and dropped when it differs,
    /// died or has an earlier state at its cell (the usual duplicate check). Added through the same
    /// bookkeeping as `merge_share`'s chain nodes.
    pub fn sync_import(&mut self, pads: &[GamePad], expect: Option<(u16, [f32; 3], (u8, i16))>) -> Result<SyncImport> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        for p in pads {
            (p.button, p.stick_x, p.stick_y).hash(&mut h);
        }
        if !self.sync_seen.insert(h.finish()) {
            return Ok(SyncImport::Known);
        }
        // The deepest node of the tree on this path.
        let (mut at, mut off) = (0u32, 0usize);
        'down: loop {
            for &c in &self.tree.children[at as usize] {
                let e = &self.tree.nodes[c as usize].inputs;
                if !e.is_empty() && pads.len() - off >= e.len() && pads[off..off + e.len()].iter().zip(e).all(|(a, b)| (a.button, a.stick_x, a.stick_y) == (b.button, b.stick_x, b.stick_y)) {
                    (at, off) = (c, off + e.len());
                    continue 'down;
                }
            }
            break;
        }
        while at != 0 && (self.tree.nodes[at as usize].retired || !self.tree.has_save(at)) {
            let n = &self.tree.nodes[at as usize];
            off -= n.inputs.len();
            at = n.parent.unwrap_or(0);
        }
        if off == pads.len() {
            return Ok(SyncImport::Known);
        }
        let rest = &pads[off..];
        let frame = self.tree.nodes[at as usize].st.frame + rest.len() as u32;
        let (st, packed, base) = {
            let mut w = loop {
                if let Some(w) = self.workers.iter().find_map(|w| w.try_lock().ok()) {
                    break w;
                }
                std::thread::yield_now();
            };
            let base = self.tree.base_of(at);
            self.tree.load_into(at, &mut w.base, &self.bases)?;
            for p in rest {
                w.base.step_game(*p)?;
            }
            let st = read_state(&w.base, frame, true);
            let packed = Delta::new(base, &self.bases[base as usize], &w.base.save_state());
            (st, packed, base)
        };
        let _ = base;
        self.total_frames += rest.len() as u64;
        if !st.alive() {
            return Ok(SyncImport::Bad);
        }
        if let Some((level, pos, progress)) = expect {
            if st.level != level || st.pos != pos || st.progress() != progress {
                return Ok(SyncImport::Bad);
            }
        }
        let c = cell(&st);
        if self.cells.get(&c).is_some_and(|ids| ids.iter().any(|&i| self.tree.nodes[i as usize].st.frame <= st.frame)) {
            return Ok(SyncImport::Dup);
        }
        let via = "sync";
        let zone = st.zone();
        let base = match self.zone_base.get(&zone) {
            Some(&b) => b,
            None => {
                self.bases.push(crate::game::into_plain(packed.apply(&self.bases)?).into_bytes());
                let b = (self.bases.len() - 1) as u16;
                self.zone_base.insert(zone, b);
                b
            }
        };
        let delta = if packed.base == base { packed } else { Delta::new(base, &self.bases[base as usize], &packed.apply(&self.bases)?) };
        if self.zones.insert(zone) {
            println!("new zone {zone:?} via tree sync at node {}", self.tree.nodes.len());
        }
        let (modal, fr, progress) = (st.modal(), st.frame, st.progress());
        let (id, adv) = self.tree.add(at, rest.to_vec(), st, delta, via);
        if self.cfg.new_eligible {
            self.ne_note(id, at);
        }
        self.note_best_at(id, progress, zone, fr, via)?;
        if modal {
            self.modal_queue.push(id);
        }
        if self.cfg.soft_dedup {
            let mut crowd = 0;
            for &o in self.cells.get(&c).map_or(&[][..], |v| v.as_slice()) {
                if self.tree.nodes[o as usize].st.frame <= fr {
                    crowd += 1;
                } else {
                    self.tree.nodes[o as usize].stats.crowd += 1;
                }
            }
            self.tree.nodes[id as usize].stats.crowd = crowd;
        }
        self.cells.entry(c).or_default().push(id);
        self.tree.gen_step[id as usize] = true;
        self.sync_sent.insert(id);
        if adv {
            let (keys, stars) = self.tree.frontier_progress;
            let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
            println!("FRONTIER keys {keys} stars {stars} at node {id} via tree sync: {f}");
        }
        Ok(SyncImport::Added(id))
    }

    pub fn run(&mut self) -> Result<()> {
        let begin = Instant::now();
        let mut last_report = Instant::now();
        let mut last_tune = self.total_frames;
        let mut round = 0u64;
        // Records for the generator (Mario::PosTime near b): kept by the
        // tree as nodes are added, read during planning on this thread.
        if self.generator.wants_records() {
            obrrt_core::records::set_ctx_fn(crate::speed::ctx);
            let r = match &self.tree.records {
                Some(r) => r.clone(),
                None => self.tree.enable_records(),
            };
            obrrt_core::records::install(Some(r));
        }
        // Stop on wall-clock time, or on emulated frames (max_frames) so runs
        // compare equally whatever the thread count or CPU sharing.
        // OBRRT_STOP_EXIT_GRACE=<frames> (with OBRRT_ZONE_LOCK): stop this many
        // emulated frames after the first good exit of the zone, or at once
        // when it is 0; the frames after it are for a faster exit.
        let grace: Option<u64> = obrrt_core::config::current().stop.exit_grace;
        let mut stop_at: Option<u64> = None;
        // OBRRT_STOP_STALL=<frames>: stop when the tree's work in progress
        // (its best St::wip: red coins, chests, boss hits) and its progress
        // (keys, stars) haven't improved for this many emulated frames. The
        // budget then follows the tree: a course with nothing to build on
        // gives up early; one gaining red coins keeps going (a fixed 150 M
        // gave 4 of BoB's 8 red coins, 2026-11-09 w4-bob).
        let stall: Option<u64> = obrrt_core::config::current().stop.stall;
        let (mut best_wip, mut best_prog, mut improved_at) = (i64::MIN, (0u8, 0i16), 0u64);
        let mut idle_rounds = 0u32;
        // Rounds in a row that emulated no frames: attempts from a doomed start (campaign t1 step 398: a coin leg
        // from LLL with Mario in hard knockback at 1 health segment ran 4.5 M attempts of 0 frames for 7.6 hours,
        // since the frame budget never came).
        let (mut dead_rounds, mut last_frames) = (0u32, u64::MAX);
        while begin.elapsed() < Duration::from_secs_f64(self.cfg.minutes * 60.0)
            && self.cfg.max_frames.is_none_or(|m| self.total_frames < m)
            && stop_at.is_none_or(|s| self.total_frames < s)
        {
            if let (Some(g), None) = (grace, stop_at) {
                if self.tree.good_exits > 0 {
                    stop_at = Some(self.total_frames + g);
                    println!("zone lock: first good exit at {} frames; stopping at {}", self.total_frames, self.total_frames + g);
                }
            }
            if let Some(st) = stall {
                let wip = self.tree.wip.iter().copied().max().unwrap_or(0);
                let prog = (self.tree.frontier_progress.0, self.tree.frontier_progress.1);
                if wip > best_wip || prog > best_prog || std::mem::take(&mut self.eta_improved) {
                    (best_wip, best_prog, improved_at) = (wip.max(best_wip), prog.max(best_prog), self.total_frames);
                } else if self.total_frames > improved_at + st + st * best_wip.max(0) as u64 / 4 && stop_at.is_none() {
                    // The wait grows with the work done (a quarter of the stall
                    // per unit of wip): BoB with 6 of 8 red coins stopped 50 M
                    // after the sixth (campaign t1 step 101).
                    println!("stall: no new work in progress since {} frames (best wip {best_wip}); stopping at {}", improved_at, self.total_frames);
                    break;
                }
            }
            round += 1;
            let pre_t = Instant::now();
            let mut lap = Instant::now();
            let mut tick = |k: usize, d: &mut [f64; 11]| {
                d[k] += lap.elapsed().as_secs_f64();
                lap = Instant::now();
            };
            let p = self.cfg.rate;
            self.pool.rate_refresh(self.total_frames, &p);
            tick(2, &mut self.diag_pre);
            self.tree.wip_refresh();
            tick(3, &mut self.diag_pre);
            self.index_frontier();
            tick(4, &mut self.diag_pre);
            self.ne.budget = if self.cfg.new_eligible { ((self.cfg.round_plans as f64 * NE_SHARE).ceil() as usize).max(1) } else { 0 };
            self.diag_secs[0] += pre_t.elapsed().as_secs_f64();
            let plan_t = Instant::now();
            // More attempts than workers, so a slow attempt doesn't leave the
            // others idle: each takes whichever game is free.
            // A fixed number of attempts per round, whatever the thread
            // count: the tree only updates between rounds, so the round size
            // shapes the search (how stale planning is); 64 = 16 threads × 4.
            obrrt_core::mover::publish(); // movers see caches observed before this round only
            // sample_entry's tables get this round's weights (kept between
            // rounds only for their kinds and members).
            if let Some(idx) = self.pool.sidx.as_mut() {
                idx.reweigh(&self.pool.entries);
            }
            let plans: Vec<Plan> = (0..self.cfg.round_plans).filter_map(|_| self.plan()).collect();
            // Nothing to plan (a course with every star had: no frontier state
            // is a base) spun forever before 2026-11-09 (campaign t1 step 97,
            // TotWC, 40 minutes at one core, no frames): a few empty rounds end
            // the run.
            if plans.is_empty() {
                idle_rounds += 1;
                if idle_rounds >= 20 {
                    println!("nothing to plan for {idle_rounds} rounds; stopping");
                    break;
                }
            } else {
                idle_rounds = 0;
            }
            if self.total_frames == last_frames {
                dead_rounds += 1;
                if dead_rounds >= 50 {
                    println!("no frames emulated for {dead_rounds} rounds; stopping");
                    break;
                }
            } else {
                (dead_rounds, last_frames) = (0, self.total_frames);
            }
            self.diag_plan_secs.0 += plan_t.elapsed().as_secs_f64();
            if plans.is_empty() {
                continue;
            }
            let bases = &self.bases;
            let tree = &self.tree;
            let seeds: Vec<u64> = plans.iter().map(|_| self.rng.random()).collect();
            let workers = &self.workers;
            let params = crate::ga::GaParams { scale: self.cfg.ga_scale, pink: self.cfg.ga_pink };
            let dense = self.dense;
            let replay_max = self.tree.sparse.as_ref().map_or(0, |sp| sp.cap);
            let run_t = Instant::now();
            let (load_ns, att_ns) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
            let results: Vec<Result<Attempt<Delta>>> = plans
                .par_iter()
                .zip(seeds.par_iter())
                .map(|(plan, seed)| {
                    let mut w = loop {
                        if let Some(w) = workers.iter().find_map(|w| w.try_lock().ok()) {
                            break w;
                        }
                        std::thread::yield_now();
                    };
                    let node = &tree.nodes[plan.start as usize];
                    // The start's state into the worker's base game: its save,
                    // or (sparse saves) its anchor's and the inputs since.
                    let lt = Instant::now();
                    tree.load_into(plan.start, &mut w.base, bases)?;
                    load_ns.fetch_add(lt.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    let mut rng = SmallRng::seed_from_u64(*seed);
                    let stages: Vec<&dyn Objective> = plan.stages.iter().map(|(_, o)| o.as_ref()).collect();
                    let base = tree.base_of(plan.start);
                    let pack = |s: &crate::game::GameSave| Delta::new(base, &bases[base as usize], s);
                    let novel = |k: u64, f: u32| tree.dense_novel(k, f);
                    let req = dense.map(|d| crate::ga::DenseReq {
                        modes: d.modes,
                        grid: d.grid,
                        cap: d.cap,
                        novel: &novel,
                        replay_max,
                        start_dist: tree.dist(plan.start),
                        pack: &pack,
                    });
                    let seed_in = plan.seed.as_deref();
                    crate::ga::set_tail((!node.tail.is_empty()).then(|| node.tail.clone()));
                    let at = Instant::now();
                    let packed = w.attempt_full(&stages, &node.st, None, &mut rng, params, seed_in, req.as_ref()).map(|a| a.map_saves(|s| pack(&s)));
                    att_ns.fetch_add(at.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    anyhow::Context::with_context(packed, || {
                        format!(
                            "attempt from node {} (via {}, retired {}, frame {}, zone {:?}, base {}) for {}",
                            plan.start,
                            node.via,
                            node.retired,
                            node.st.frame,
                            node.st.zone(),
                            tree.base_of(plan.start),
                            plan.stages.iter().map(|(_, o)| o.key()).collect::<Vec<_>>().join(" then ")
                        )
                    })
                })
                .collect();
            self.diag_secs[1] += run_t.elapsed().as_secs_f64();
            self.diag_secs[4] += load_ns.into_inner() as f64 * 1e-9;
            self.diag_secs[5] += att_ns.into_inner() as f64 * 1e-9;
            let merge_t = Instant::now();
            for (plan, r) in plans.iter().zip(results) {
                if plan.stages[0].1.kind() == "Road::Leg" {
                    if let (Some(road), Ok(att)) = (self.road.as_mut(), r.as_ref()) {
                        road.stats.legs_ok += att.segments.first().is_some_and(|s| s.success) as u64;
                    }
                }
                match r {
                    Ok(att) => {
                        self.merge(plan, att)?;
                    }
                    // The emulated game itself can trap (an out-of-bounds
                    // access in SM64's code from some states and inputs; v19
                    // died of one in JRB). Count it, back the start off hard
                    // (retired after 3), decay the objective, and go on: the
                    // next attempt on that game reloads a whole save state.
                    Err(e) => {
                        self.emulator_errors += 1;
                        let n = &mut self.tree.nodes[plan.start as usize];
                        n.stats.backoff *= 0.05;
                        n.stats.barren += 1;
                        n.stats.traps += 1;
                        let traps = n.stats.traps;
                        if traps >= 3 {
                            self.tree.retire_one(plan.start);
                        }
                        for (entry, obj) in &plan.stages {
                            if *entry < self.pool.entries.len() {
                                let p = self.cfg.rate;
                                self.pool.rate_note(*entry, 0.0, p.nostart_frames, self.total_frames, &p);
                            }
                            self.diag.kind(obj.kind()).died += 1;
                        }
                        if self.emulator_errors <= 20 {
                            eprintln!("emulator trap #{} (node {} traps {}): {:#}", self.emulator_errors, plan.start, traps, e);
                        }
                    }
                }
            }
            self.diag_plan_secs.1 += merge_t.elapsed().as_secs_f64();
            let post_t = Instant::now();
            // Sparse saves: frames replayed to rebuild starts count as
            // emulated frames.
            if let Some(sp) = &self.tree.sparse {
                let f = sp.rebuild_frames.load(std::sync::atomic::Ordering::Relaxed);
                self.total_frames += f - self.rebuild_counted;
                self.rebuild_counted = f;
            }
            if self.dense.is_some() {
                let mark = (self.dstats.curve.len() as u64 + 1) * 10_000_000;
                if self.total_frames >= mark {
                    self.dstats.curve.push((self.total_frames, self.tree.nodes.len(), self.dstats.added));
                }
            }
            if self.mem_used() > self.cfg.mem_budget {
                self.prune_repeats();
            }
            if self.cfg.new_eligible && round % 10 == 0 {
                let nodes = &self.tree.nodes;
                self.ne.sets.retain(|&i, _| !nodes[i as usize].retired);
            }
            if round % 10 == 0 {
                self.pool.expire(self.cfg.expire_rounds);
            }
            // Tuning on emulated frames, not wall time, so a run is the same
            // whatever the machine load.
            if self.total_frames - last_tune >= self.cfg.tune_frames {
                last_tune = self.total_frames;
                self.tune();
            }
            self.diag_secs[2] += post_t.elapsed().as_secs_f64();
            if let Some(mut f) = self.on_round.take() {
                f(self);
                self.on_round = Some(f);
            }
            if let Some(mut f) = self.on_sync.take() {
                f(self);
                self.on_sync = Some(f);
            }
            if let Some(mut f) = self.on_hybrid.take() {
                f(self);
                self.on_hybrid = Some(f);
            }
            if last_report.elapsed() > self.cfg.report_every {
                last_report = Instant::now();
                self.write_outputs(begin.elapsed().as_secs_f64())?;
                self.diag_secs[3] += last_report.elapsed().as_secs_f64();
            }
        }
        self.write_outputs(begin.elapsed().as_secs_f64())?;
        Ok(())
    }
}

/// OBRRT_ANY_EXIT=1: this run is only to leave the course (search/src/tree.rs).
fn any_exit() -> bool {
    // OBRRT_FINAL=1 (the last search, BitS and the Bowser 3 arena): at 120 stars every course counts as done, which
    // planned nothing at all ("nothing to plan for 20 rounds", b3 test from a BitS state with its star had).
    let s = &obrrt_core::config::current().stop;
    s.any_exit || s.final_search
}
