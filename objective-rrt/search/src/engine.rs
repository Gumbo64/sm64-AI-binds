//! The loop: steps 1-6 in rounds. Planning (steps 1-4) and merging (step 6)
//! run on one thread; the attempts (step 5) run one per worker in parallel.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use obrrt_core::{Budget, GamePad, Generator, Objective, St, read_state};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sm64_binds::{SM64GameGenerator, SaveState};

use crate::diag::Diagnostics;
use crate::ga::{Attempt, Worker};
use crate::pool::Pool;
use crate::store::Delta;
use crate::tree::Tree;

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
    /// Experiment knobs: multiply the GA's generations; fix the chain
    /// probability instead of learning it.
    pub ga_scale: f64,
    pub chain_fixed: Option<f64>,
    pub fitness: crate::ga::Fitness,
    pub ga_new_ops: bool,
    pub ga_pink: f64,
    /// Per-objective GA budgets (see `GaParams::per_objective_budget`).
    pub ga_per_objective_budget: bool,
    /// RRT* mode (speed): rank starts by estimated arrival time, rewire
    /// (re-attempt the objectives that grew from a state from any state that
    /// reaches its place sooner), and stop starting from slower duplicates.
    pub rrt_star: bool,
    /// Arrival ranking (OBRRT_ARRIVAL=1): for objectives whose heuristic is
    /// a distance (`Objective::arrival_scale`), starts are ranked, and
    /// improvement (decay resets, wear, route recovery) is judged, by the
    /// estimated arrival frame/30 + distance/(pace·30) in seconds, not by
    /// distance alone. No pruning or rewiring (unlike `rrt_star`).
    pub arrival: bool,
    /// Arrival ranking: the reference pace, in units per frame (20 beat 15
    /// and 30 on milestone times in the arrival sweep, RUNS.md).
    pub arrival_pace: f64,
    /// Arrival ranking: the start softmax temperature, in seconds.
    pub arrival_t: f64,
    /// Arrival ranking: seconds sooner that count as an improvement.
    pub arrival_margin: f64,
    /// Arrival ranking: only starts with base_heuristic within this of the
    /// closest one are ranked (None: all).
    pub arrival_band: Option<f64>,
    /// Arrival ranking (OBRRT_ARRIVAL_DECAY): what counts as improving on a
    /// start, for wear, route recovery and decay resets. `either` (default):
    /// h better by 0.5 or the estimated arrival sooner by `arrival_margin`,
    /// so getting closer always remains a way back; `arrival`: only sooner.
    pub arrival_decay_either: bool,
    /// Arrival ranking (OBRRT_ARRIVAL_MIX): the share of start picks ranked
    /// by estimated arrival; the rest rank by the objective's heuristic (one
    /// draw from the engine's seeded RNG per pick, only when below 1).
    /// 0.5 (default) beat T=15 s and 1 on power-on stars and kept a way to
    /// the WF room from the v44 stall (RUNS.md arrival row).
    pub arrival_mix: f64,
    /// Share of b drawn from the best routes (the states leading to each
    /// zone's earliest frontier state), so PosTime targets the route.
    pub route_share: f64,
    /// Draw b's zone by the most valuable objective available there, not
    /// uniformly.
    pub value_zones: bool,
    /// Soft dedupe: keep states that only repeat a place (an earlier state
    /// got there), at half the sampling weight per earlier state, instead of
    /// dropping them; over `mem_budget` bytes of saves, the lowest-weight
    /// repeats are retired. Yield and decay count only new or sooner states.
    /// Attempts planned per round (merged together after it).
    pub round_plans: usize,
    /// Stop after emulating this many frames (None: only the time limit).
    pub max_frames: Option<u64>,
    /// Cap on one objective kind's share of picks (0 = off).
    pub kind_cap: f64,
    /// Draw b's zone by recent new states per attempt there.
    pub zone_bandit: bool,
    pub soft_dedup: bool,
    /// Start wear (OBRRT_START_WEAR): a start's weight for an objective is
    /// ×0.5 per two attempts of it from there that didn't improve on the
    /// start, so dead-end starts burn out and other kin states get picked.
    pub start_wear: bool,
    /// Decay recovery from other routes (OBRRT_DECAY_ROUTE): an objective's
    /// decay also recovers (to at least 0.5, a bounded number of times)
    /// when an attempt from a start not used for it before adds a state
    /// that improves on that start's heuristic, even if not on best_start.
    pub decay_route: bool,
    /// Entry bandit: objective weights are Thompson samples of each entry's
    /// discounted improvement rate (redrawn every round), replacing the
    /// ×decay per failure and its reset rules.
    pub entry_bandit: bool,
    /// Entry bandit: evidence fades by this per attempt at the entry.
    pub bandit_gamma: f64,
    /// Kind bandit (OBRRT_KIND_BANDIT): each kind's multiplier is its
    /// posterior mean improvements per kframe (the entry bandit's signal),
    /// normalised to mean 1, replacing the yield EMA in `tune`.
    pub kind_bandit: bool,
    /// Kind bandit: evidence fades by this per attempt of the kind.
    pub kind_gamma: f64,
    pub mem_budget: usize,
    /// Pick only objectives b (or its parent) satisfies the base criteria
    /// of, so a start always exists.
    pub pick_by_b: bool,
    /// Share of plans spent on rewiring while there is rewiring to do.
    pub rewire_share: f64,
    /// Start cost per frame of game time, in base_heuristic units (h is
    /// mostly distance/400; Mario runs ~30 units a frame, so 30/400 would
    /// weigh time and distance equally).
    pub time_cost: f64,
    /// Frames sooner a state must be to count as reaching a place faster.
    pub rewire_margin: u32,
    /// New-eligible boost (OBRRT_NEW_ELIGIBLE=1): a state that can start an
    /// objective its parent couldn't (by key, objectives worth more than
    /// 1) is queued once per such key, and the queue is planned from first,
    /// at most `NE_SHARE` of a round's plans, with b = that state and the
    /// newly startable objective: the highest-weight objective first (the
    /// newest state among equals), since a run may have only a few rounds. Each key is queued at most `NE_PER_KEY`
    /// times in a run. Rare unlocking states (a boss's star spawned) would
    /// otherwise wait in a crowded b cell.
    pub new_eligible: bool,
    /// Earliest-arrival boost (OBRRT_EARLIEST=1): a state that is the
    /// earliest ever (by frame) in its zone tier (level, area, stars, keys,
    /// collected set with OBRRT_STARSET_KEY), or that beats its b cell's
    /// earliest state by `EA_CELL_MARGIN` frames, is queued for up to
    /// `earliest_cap` attempts started from itself (the objective drawn
    /// as for b among those it can start), at most `earliest_share` of a
    /// round's plans and `EA_PER_ROUND` per state per round: zone records
    /// first (earliest frame first), then cell records (largest lead
    /// first). A new fastest arrival would otherwise tie on distance with
    /// hundreds of later states and get a few attempts. A state no longer
    /// within the margin of its record when served is skipped.
    pub earliest: bool,
    pub earliest_cap: u32,
    pub earliest_share: f64,
    /// Whether b-cell records are queued too (OBRRT_EARLIEST_CELL, on).
    pub earliest_cell: bool,
    /// Progress report (log, nodes.csv, summary.json) every this much wall
    /// time. Reports only; no decision depends on them.
    pub report_every: Duration,
    /// Automatic tuning (`tune`: kind multipliers, chain probability, start
    /// cutoffs) each time this many emulated frames have passed, checked
    /// after each round's merge (`OBRRT_TUNE_FRAMES`). It used to run with
    /// the 30 s report, which made trees depend on machine load; 10 M frames
    /// is about 30 s of a 16-thread run (v35: 500 M frames in 1139 s).
    pub tune_frames: u64,
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
            chain_fixed: None,
            fitness: crate::ga::Fitness::Steps,
            ga_new_ops: true,
            ga_pink: 0.0,
            ga_per_objective_budget: false,
            rrt_star: false,
            arrival: false,
            arrival_pace: 20.0,
            arrival_t: 1.0,
            arrival_margin: 0.1,
            arrival_band: None,
            arrival_decay_either: true,
            arrival_mix: 0.5,
            // Off by default: v15 (0.2 and on) did worse than v14.
            route_share: 0.0,
            value_zones: false,
            round_plans: 64,
            max_frames: None,
            kind_cap: 0.0,
            zone_bandit: false,
            soft_dedup: false,
            start_wear: false,
            decay_route: false,
            entry_bandit: false,
            bandit_gamma: 0.97,
            kind_bandit: false,
            kind_gamma: 0.995,
            mem_budget: 3_000_000_000,
            pick_by_b: true,
            rewire_share: 0.3,
            // 40% of "a frame is worth 30 units": at full weight the start is
            // almost always the single earliest state (1.9 effective starts in
            // a test) and the search stops spreading out.
            time_cost: 0.4 * 30.0 / 400.0,
            rewire_margin: 10,
            new_eligible: std::env::var("OBRRT_NEW_ELIGIBLE").is_ok_and(|v| v == "1"),
            earliest: false,
            earliest_cap: 12,
            earliest_share: 0.15,
            earliest_cell: true,
            report_every: Duration::from_secs(30),
            tune_frames: 10_000_000,
        }
    }
}

/// One attempt: objectives to meet in order (pool entry, objective) from
/// `start`. A single objective is a chain of one.
struct Plan {
    stages: Vec<(usize, Arc<dyn Objective>)>,
    start: u32,
    b: u32,
    /// RRT* rewiring: the slower state whose objective this re-attempts.
    rewire: Option<u32>,
}

/// RRT* rewiring work: re-attempt `target`'s objective from `from`, a state
/// that reached `target`'s parent's place sooner.
struct Rewire {
    from: u32,
    target: u32,
}

/// Arrival ranking (`Config::arrival`): `s`'s estimated arrival at `obj`'s
/// target in seconds, game time plus the straight-line distance left at
/// the reference pace, given h = `obj.base_heuristic(s)`. None when off or
/// when the heuristic isn't a distance.
fn arrival_est(cfg: &Config, obj: &dyn Objective, s: &St, h: f64) -> Option<f64> {
    let k = obj.arrival_scale().filter(|_| cfg.arrival)?;
    // Unreachable (f64::MAX) stays unreachable.
    if h >= f64::MAX / 2.0 {
        return Some(h);
    }
    Some((s.frame as f64 + h * k / cfg.arrival_pace) / 30.0)
}

/// Whether `new` improves on `old`, each (base_heuristic, estimated arrival
/// if arrival ranking applies): h lower by 0.5, or (arrival ranking) the
/// arrival sooner by `arrival_margin`, alone or either (`arrival_decay_either`).
fn improves(cfg: &Config, new: (f64, Option<f64>), old: (f64, Option<f64>)) -> bool {
    let closer = new.0 < old.0 - 0.5;
    match (new.1, old.1) {
        (Some(a), Some(b)) => {
            let sooner = a < b - cfg.arrival_margin;
            sooner || (cfg.arrival_decay_either && closer)
        }
        _ => closer,
    }
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

/// Whether the duplicate check includes St::fingerprint (OBRRT_FINGERPRINT=0
/// turns it off, to compare with runs from before it).
pub static FINGERPRINT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// OBRRT_SETTLE_RANK=1: in pick_start, a standing state (hspeed < 2, not
/// rising or falling) weighs half as a start for a non-modal objective, so
/// moving states are extended first (a restart from rest costs frames).
pub static SETTLE_RANK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the duplicate check and the frontier's b cells tell apart states
/// with the same progress count but different collected sets
/// (St::progress_set; OBRRT_STARSET_KEY=1, off by default).
pub static STARSET: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn starset(st: &St) -> u64 {
    if STARSET.load(std::sync::atomic::Ordering::Relaxed) { st.progress_set() } else { 0 }
}

fn cell(st: &St) -> Cell {
    let c = |v: f32| (v / 64.0).floor() as i32;
    let modal = if st.in_dialog() { st.dialog } else if st.in_save_menu() { -2 } else if st.act_select { -3 } else { -1 };
    let fp = if FINGERPRINT.load(std::sync::atomic::Ordering::Relaxed) { st.fingerprint() } else { 0 };
    // Off, the last field is exactly the fingerprint as before.
    let fp = if STARSET.load(std::sync::atomic::Ordering::Relaxed) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&(fp, st.progress_set()), &mut h);
        std::hash::Hasher::finish(&h)
    } else {
        fp
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
    zones: HashSet<(u16, u8)>,
    /// The frontier grouped by zone, then by 512-unit cell; rebuilt each round.
    b_index: Vec<Vec<Vec<u32>>>,
    /// Per b_index zone: its draw weight (value_zones).
    zone_weights: Vec<f64>,
    /// Zone bandit: per zone, (new states, attempts), decaying.
    zone_stats: HashMap<(u16, u8), (f64, f64)>,
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
    /// RRT*: rewiring to do, on the way to the best states first (`hot`).
    rewire_hot: std::collections::VecDeque<Rewire>,
    rewire_cold: std::collections::VecDeque<Rewire>,
    hot: HashSet<u32>,
    /// New-eligible boost (see `Config::new_eligible`).
    ne: NewEligible,
    /// Earliest-arrival boost (see `Config::earliest`).
    ea: Earliest,
    /// Backward chaining (OBRRT_BACKCHAIN, see backchain.rs).
    bc: crate::backchain::Backchain,
    /// Wall seconds spent planning (steps 1-4) and merging (step 6).
    diag_plan_secs: (f64, f64),
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

/// Earliest-arrival boost: frames a state must beat its b cell's record by,
/// and plans per queued state per round.
const EA_CELL_MARGIN: u32 = 30;
const EA_PER_ROUND: u32 = 2;
const EA_QUEUE: usize = 2048;

/// Zone tier: level, area, stars, keys, collected set (STARSET only).
type Tier = (u16, u8, i16, u8, u64);
/// b cell within a tier (as index_frontier groups them).
type BCell = (Tier, i32, i32, i32, i64, i16);

fn tier(st: &St) -> Tier {
    (st.level, st.area, st.stars, st.keys, starset(st))
}

fn bcell(st: &St) -> BCell {
    let c = |v: f32| (v / 512.0).floor() as i32;
    (tier(st), c(st.pos[0]), c(st.pos[1]), c(st.pos[2]), st.level_changes(), st.red_coins)
}

#[derive(Default)]
struct Earliest {
    /// Earliest frame (and state) per tier and per b cell.
    tier_best: HashMap<Tier, (u32, u32)>,
    cell_best: HashMap<BCell, (u32, u32)>,
    /// Priority (higher served first), state; attempts left per state.
    queue: std::collections::BTreeSet<(u64, u32)>,
    left: HashMap<u32, (u32, u64, bool)>,
    /// Plans left this round, and per state this round.
    budget: usize,
    round_used: HashMap<u32, u32>,
    queued_tier: u64,
    queued_cell: u64,
    dropped: u64,
    stale: u64,
    planned: u64,
    no_objective: u64,
}

fn key_hash(k: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::hash::DefaultHasher::new();
    k.hash(&mut h);
    h.finish()
}

impl Engine {
    pub fn new(
        cfg: Config,
        game_gen: &SM64GameGenerator,
        boot: Vec<GamePad>,
        root_save: SaveState,
        root: St,
        generator: Arc<dyn Generator>,
    ) -> Result<Engine> {
        fs::create_dir_all(&cfg.out_dir)?;
        let threads = rayon::current_num_threads();
        let workers = (0..threads)
            .map(|_| Ok(std::sync::Mutex::new(Worker { base: game_gen.create_game()?, tmp: game_gen.create_game()? })))
            .collect::<Result<Vec<_>>>()?;
        let events = BufWriter::new(File::create(format!("{}/events.jsonl", cfg.out_dir))?);
        let rng = SmallRng::seed_from_u64(cfg.seed);
        let mut cells = HashMap::new();
        cells.insert(cell(&root), vec![0]);
        let root_zone = root.zone();
        let root_st = root.clone();
        Ok(Engine {
            tree: Tree::new(root),
            pool: Pool::default(),
            diag: Diagnostics { chain_prob: cfg.chain_fixed.unwrap_or(cfg.chain_prob), ..Default::default() },
            generator,
            workers,
            bases: vec![root_save.into_bytes()],
            zone_base: HashMap::from([(root_zone, 0)]),
            boot,
            rng,
            cells,
            first_success: HashSet::new(),
            decay_resets: BTreeMap::new(),
            zones: HashSet::from([root_zone]),
            b_index: vec![vec![vec![0]]],
            zone_weights: vec![1.0],
            zone_stats: HashMap::new(),
            route: Vec::new(),
            events,
            total_frames: 0,
            attempts_merged: 0,
            modal_queue: Vec::new(),
            emulator_errors: 0,
            modal_tried: HashSet::new(),
            rewire_hot: Default::default(),
            rewire_cold: Default::default(),
            hot: HashSet::new(),
            ne: NewEligible { rng: Some(SmallRng::seed_from_u64(cfg.seed ^ 0x6e65_7765_6c69_6769)), ..Default::default() },
            ea: Earliest {
                tier_best: HashMap::from([(tier(&root_st), (root_st.frame, 0))]),
                cell_best: HashMap::from([(bcell(&root_st), (root_st.frame, 0))]),
                ..Default::default()
            },
            diag_plan_secs: (0.0, 0.0),
            bc: crate::backchain::Backchain::new(cfg.seed),
            cfg,
        })
    }

    /// Step 3: picks a kind (by the mean weight of its entries × its learned
    /// multiplier, with a floor; so a kind with many entries doesn't crowd
    /// out the rest), then an entry within it by weight.
    /// With `strict` (earliest-arrival boost), only objectives b itself can
    /// start (within its kind's cutoff), whatever `pick_by_b`.
    fn sample_entry(&mut self, b: u32, strict: bool) -> Option<usize> {
        // Only objectives b (or its parent) can start: then b's kin set is
        // sure to hold a start, and no pick is wasted (pick_by_b).
        let eligible: Vec<bool> = if strict {
            let b_st = &self.tree.nodes[b as usize].st;
            let diag = &self.diag;
            self.pool
                .entries
                .iter()
                .map(|e| {
                    let o = e.obj.as_ref();
                    !o.modal()
                        && o.exists_in(b_st)
                        && o.base_criteria(b_st)
                        && o.base_heuristic(b_st) <= diag.kinds.get(o.kind()).map_or(f64::INFINITY, |k| k.cutoff)
                })
                .collect()
        } else if self.cfg.pick_by_b {
            let b_st = &self.tree.nodes[b as usize].st;
            let parent = self.tree.nodes[b as usize].parent.map(|p| &self.tree.nodes[p as usize]);
            let ok = |o: &dyn Objective, s: &St| o.exists_in(s) && o.base_criteria(s);
            self.pool
                .entries
                .iter()
                .map(|e| {
                    !e.obj.modal()
                        && (ok(e.obj.as_ref(), b_st) || parent.is_some_and(|p| !p.retired && !p.st.modal() && ok(e.obj.as_ref(), &p.st)))
                })
                .collect()
        } else {
            self.pool.entries.iter().map(|e| !e.obj.modal()).collect()
        };
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
        let mut weights: Vec<f64> =
            kinds.iter().map(|(k, w)| w.max(self.cfg.kind_floor * total * (mean_cost / cost(k)).min(1.0))).collect();
        // Cap any one kind's share of picks (kind_cap): its excess goes to the
        // others in proportion (a few rounds, since capping raises the rest).
        if self.cfg.kind_cap > 0.0 && weights.len() > 1 {
            let cap = self.cfg.kind_cap.max(1.0 / weights.len() as f64);
            for _ in 0..4 {
                let sum: f64 = weights.iter().sum();
                let mut changed = false;
                for w in &mut weights {
                    if *w > cap * sum {
                        *w = cap * sum;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }
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

    /// Arrival ranking: `s`'s estimated arrival at `obj`'s target in seconds
    /// (game time plus the straight-line distance left at the reference
    /// pace), given h = `obj.base_heuristic(s)`. None when off or when the
    /// heuristic isn't a distance.
    fn arrival_of(&self, obj: &dyn Objective, s: &St, h: f64) -> Option<f64> {
        arrival_est(&self.cfg, obj, s, h)
    }

    /// What improvement on a start is judged by (see [`improves`]): the
    /// heuristic, and the estimated arrival (seconds) with arrival ranking.
    fn keys_of(&self, obj: &dyn Objective, s: &St) -> (f64, Option<f64>) {
        let h = obj.base_heuristic(s);
        (h, self.arrival_of(obj, s, h))
    }

    /// Step 4: the eligible starts for `obj` given `b`, sampled by softmax
    /// over `base_heuristic` (or the estimated arrival, `arrival`) × u(s) ×
    /// the state's back-off.
    fn pick_start(&mut self, entry: usize, obj: &dyn Objective, b: u32) -> Option<u32> {
        let arrival = self.cfg.arrival
            && obj.arrival_scale().is_some()
            && (self.cfg.arrival_mix >= 1.0 || self.rng.random::<f64>() < self.cfg.arrival_mix);
        let cutoff = self.diag.kind(obj.kind()).cutoff;
        let k = obj.kinship();
        let b_st = &self.tree.nodes[b as usize].st;
        let (b_keys, b_stars, b_lives) = (b_st.keys, b_st.stars, b_st.lives);
        // OBRRT_RETIRE_SUBSET: a start has everything b has, not just as many.
        let b_set = self.tree.subset.then(|| crate::tree::set_key(b_st));
        let mut cands: Vec<(u32, f64, f64)> = Vec::new();
        // Arrival band: each candidate's own heuristic, to keep only those
        // within `arrival_band` of the closest.
        let mut raw_h: Vec<f64> = Vec::new();
        for i in self.tree.kin_set(b, k) {
            let n = &self.tree.nodes[i as usize];
            let s = &n.st;
            // OBRRT_KEEP_WIP: a state kept for its work under way starts
            // whatever b has collected.
            let wip = self.tree.is_wip(i);
            if n.retired
                || (self.cfg.rrt_star && n.dominated)
                || !s.alive()
                || (!wip && s.keys < b_keys)
                || (!wip && s.stars < b_stars)
                || (!wip && b_set.as_ref().is_some_and(|k| !crate::tree::covers(&crate::tree::set_key(s), k)))
                || s.modal() != obj.modal()
                || !obj.base_criteria(s)
            {
                continue;
            }
            let h = obj.base_heuristic(s);
            if h > cutoff {
                continue;
            }
            // RRT*: the cost so far (game time) counts too, so attempts
            // extend the fastest ways there.
            // Arrival ranking: the estimated arrival in seconds instead.
            raw_h.push(h);
            let h = if arrival {
                self.arrival_of(obj, s, h).unwrap_or(h)
            } else if self.cfg.rrt_star {
                h + self.cfg.time_cost * s.frame as f64
            } else {
                h
            };
            let u = (s.wedges() as f64 / 8.0).powf(self.cfg.alpha)
                * self.cfg.gamma.powi((b_lives as i32 - s.lives as i32).max(0));
            let crowd = if self.cfg.soft_dedup { 0.5f64.powi(n.stats.crowd as i32) } else { 1.0 };
            let wear = match self.pool.entries[entry].wear.get(&(i as u32)) {
                Some(&w) if self.cfg.start_wear => 0.5f64.powf(w as f64 / 2.0),
                _ => 1.0,
            };
            let settle = if SETTLE_RANK.load(std::sync::atomic::Ordering::Relaxed) && !obj.modal() && s.vel[0].hypot(s.vel[2]) < 2.0 && s.vel[1].abs() < 1.0 { 0.5 } else { 1.0 };
            cands.push((i as u32, h, u * n.stats.backoff * crowd * wear * settle));
        }
        if cands.is_empty() {
            return None;
        }
        // Arrival band (OBRRT_ARRIVAL_BAND): arrival ranks only the starts
        // about as close as the closest; the rest get no weight.
        if let Some(band) = self.cfg.arrival_band.filter(|_| arrival) {
            let rmin = raw_h.iter().cloned().fold(f64::INFINITY, f64::min);
            for (c, &r) in cands.iter_mut().zip(&raw_h) {
                if r > rmin + band {
                    c.2 = 0.0;
                    c.1 = f64::INFINITY;
                }
            }
        }
        let hmin = cands.iter().map(|c| c.1).fold(f64::INFINITY, f64::min);
        let t = if arrival { self.cfg.arrival_t } else { obj.temperature() }.max(1e-9);
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
        let mut zones: BTreeMap<(u16, u8), BTreeMap<(i32, i32, i32, i64, i16, u64), Vec<u32>>> = BTreeMap::new();
        // OBRRT_KEEP_WIP: the states kept for their work under way are b
        // candidates too (incomparable with the frontier, not behind it).
        let wip_kept = if self.tree.keep_wip { self.tree.wip_kept.clone() } else { Vec::new() };
        for &i in self.tree.frontier.iter().chain(&wip_kept) {
            let st = &self.tree.nodes[i as usize].st;
            // A course whose stars are all collected has nothing left to find;
            // dialogs and menus are planned from the modal queue instead.
            if st.course_done(st.course) || st.modal() || (self.cfg.rrt_star && self.tree.nodes[i as usize].dominated) {
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
            zones.entry(st.zone()).or_default().entry(cell).or_default().push(i);
        }
        // Zone weights: 1 + the most valuable objective (weight × decay ×
        // kind multiplier) some of the zone's frontier states can start.
        // Self objectives (weight 1, everywhere) are the baseline and skipped.
        let bandit: Vec<f64> = zones
            .keys()
            .map(|z| self.zone_stats.get(z).map_or(1.0, |(novel, tries)| (1.0 + novel) / (1.0 + tries)))
            .collect();
        let bandit_max = bandit.iter().cloned().fold(0.0, f64::max);
        self.zone_weights = zones
            .iter()
            .enumerate()
            .map(|(zi, (_, cells))| {
                // Zone bandit: recent new states per attempt, floored at 15%
                // of the best zone's so no zone is starved.
                if self.cfg.zone_bandit {
                    return bandit[zi].max(0.15 * bandit_max);
                }
                if !self.cfg.value_zones {
                    return 1.0;
                }
                let reps: Vec<&St> = cells.values().take(4).map(|c| &self.tree.nodes[c[0] as usize].st).collect();
                let best = self
                    .pool
                    .entries
                    .iter()
                    .filter(|e| e.base_weight > 1.0 && !e.obj.modal())
                    .filter(|e| reps.iter().any(|s| e.obj.exists_in(s) && e.obj.base_criteria(s)))
                    .map(|e| e.base_weight * e.decay * self.diag.kinds.get(e.obj.kind()).map_or(1.0, |k| k.multiplier))
                    .fold(0.0, f64::max);
                1.0 + best
            })
            .collect();
        self.b_index = zones.into_values().map(|cells| cells.into_values().collect()).collect();
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
                    !n.retired && !n.dominated && !n.st.modal() && self.tree.on_frontier(&n.st)
                })
                .collect();
            self.route.sort_unstable();
        }
    }

    /// A dialog or menu state as b: its options have infinite weight (and
    /// everything else none), so one of them is attempted from it. From the
    /// queue, each option once, untried first; otherwise by weight.
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
        Some(Plan { stages: vec![(entry, obj)], start: b, b, rewire: None })
    }

    /// RRT*: the next rewiring attempt, if any is still valid.
    fn plan_rewire(&mut self) -> Option<Plan> {
        while let Some(r) = self.rewire_hot.pop_front().or_else(|| self.rewire_cold.pop_front()) {
            let from = &self.tree.nodes[r.from as usize];
            let Some(obj) = self.tree.nodes[r.target as usize].made_by.clone() else { continue };
            if from.retired || from.dominated || from.st.modal() || !obj.exists_in(&from.st) || !obj.base_criteria(&from.st) {
                continue;
            }
            let entry = match self.pool.find(&obj.key()) {
                Some(e) => e,
                None => {
                    self.pool.insert(obj.clone(), &from.st);
                    self.pool.entries.len() - 1
                }
            };
            self.diag.rewire.attempts += 1;
            return Some(Plan { stages: vec![(entry, obj)], start: r.from, b: r.from, rewire: Some(r.target) });
        }
        None
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
                || n.st.course_done(n.st.course)
                || (self.cfg.rrt_star && n.dominated)
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
            match self.pick_start(entry, obj.as_ref(), b) {
                Some(start) => {
                    self.pool.entries[entry].unattemptable = 0;
                    self.ne.planned += 1;
                    let stages = self.extend_chain(entry, obj, start);
                    out = Some(Plan { stages, start, b, rewire: None });
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

    /// Earliest-arrival boost, on adding `id`: updates the tier and b-cell
    /// records and queues `id` if it set one (see `Config::earliest`).
    fn ea_note(&mut self, id: u32) {
        let st = &self.tree.nodes[id as usize].st;
        if st.modal() || !st.alive() {
            return;
        }
        let (t, c, f) = (tier(st), bcell(st), st.frame);
        let tier_rec = self.ea.tier_best.get(&t).is_none_or(|&(bf, _)| f < bf);
        let old_cell = self.ea.cell_best.get(&c).map(|&(bf, _)| bf);
        if old_cell.is_none_or(|bf| f < bf) {
            self.ea.cell_best.insert(c, (f, id));
        }
        if tier_rec {
            self.ea.tier_best.insert(t, (f, id));
        }
        let cell_rec = self.cfg.earliest_cell && old_cell.is_some_and(|bf| f + EA_CELL_MARGIN <= bf);
        if !(tier_rec || cell_rec) || self.ea.left.contains_key(&id) {
            return;
        }
        let prio = if tier_rec {
            self.ea.queued_tier += 1;
            (1u64 << 63) | (u32::MAX - f) as u64
        } else {
            self.ea.queued_cell += 1;
            (old_cell.unwrap() - f) as u64
        };
        self.ea.queue.insert((prio, id));
        self.ea.left.insert(id, (self.cfg.earliest_cap, prio, tier_rec));
        if self.ea.queue.len() > EA_QUEUE {
            if let Some((_, j)) = self.ea.queue.pop_first() {
                self.ea.left.remove(&j);
                self.ea.dropped += 1;
            }
        }
    }

    /// Earliest-arrival boost: the next queued state as b and start, with
    /// an objective it can start, within this round's budget.
    fn plan_earliest(&mut self) -> Option<Plan> {
        while self.ea.budget > 0 {
            let used = &self.ea.round_used;
            let Some(&(prio, b)) = self.ea.queue.iter().rev().find(|(_, i)| used.get(i).copied().unwrap_or(0) < EA_PER_ROUND) else {
                break;
            };
            let n = &self.tree.nodes[b as usize];
            let rec = if self.ea.left[&b].2 {
                self.ea.tier_best.get(&tier(&n.st)).map(|r| r.0)
            } else {
                self.ea.cell_best.get(&bcell(&n.st)).map(|r| r.0)
            };
            if n.retired
                || n.st.modal()
                || !n.st.alive()
                || !(self.tree.on_frontier(&n.st) || self.tree.is_wip(b))
                || n.st.course_done(n.st.course)
                || (self.cfg.rrt_star && n.dominated)
                || rec.is_some_and(|r| r + EA_CELL_MARGIN <= n.st.frame)
            {
                self.ea.queue.remove(&(prio, b));
                self.ea.left.remove(&b);
                self.ea.stale += 1;
                continue;
            }
            *self.ea.round_used.entry(b).or_default() += 1;
            let left = &mut self.ea.left.get_mut(&b).unwrap().0;
            *left -= 1;
            if *left == 0 {
                self.ea.queue.remove(&(prio, b));
                self.ea.left.remove(&b);
            }
            self.tree.nodes[b as usize].stats.as_b += 1;
            let b_st = self.tree.nodes[b as usize].st.clone();
            for o in self.generator.generate(&b_st, &mut self.rng) {
                self.pool.insert(o, &b_st);
            }
            let Some(entry) = self.sample_entry(b, true) else {
                self.ea.no_objective += 1;
                continue;
            };
            self.ea.budget -= 1;
            self.ea.planned += 1;
            let obj = self.pool.entries[entry].obj.clone();
            let kind = obj.kind();
            self.diag.kind(kind).samples += 1;
            self.diag.kind(kind).window_samples += 1;
            self.pool.entries[entry].unattemptable = 0;
            let stages = self.extend_chain(entry, obj, b);
            return Some(Plan { stages, start: b, b, rewire: None });
        }
        None
    }

    fn plan(&mut self) -> Option<Plan> {
        if let Some(b) = self.modal_queue.pop() {
            return self.plan_modal(b, true);
        }
        if self.bc.budget > 0 {
            if let Some((start, obj)) = self.bc.next(&self.tree) {
                self.pool.insert(obj.clone(), &self.tree.nodes[start as usize].st);
                let entry = self.pool.find(&obj.key()).unwrap();
                return Some(Plan { stages: vec![(entry, obj)], start, b: start, rewire: None });
            }
        }
        if self.cfg.rrt_star
            && !(self.rewire_hot.is_empty() && self.rewire_cold.is_empty())
            && self.rng.random_bool(self.cfg.rewire_share)
        {
            if let Some(p) = self.plan_rewire() {
                return Some(p);
            }
        }
        if self.cfg.earliest && self.ea.budget > 0 {
            if let Some(p) = self.plan_earliest() {
                return Some(p);
            }
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
        let b = if !self.route.is_empty() && self.rng.random_bool(self.cfg.route_share) {
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
            let entry = self.sample_entry(b, false)?;
            let obj = self.pool.entries[entry].obj.clone();
            let kind = obj.kind();
            self.diag.kind(kind).samples += 1;
            self.diag.kind(kind).window_samples += 1;
            match self.pick_start(entry, obj.as_ref(), b) {
                Some(start) => {
                    self.pool.entries[entry].unattemptable = 0;
                    let stages = self.extend_chain(entry, obj, start);
                    return Some(Plan { stages, start, b, rewire: None });
                }
                None => {
                    self.diag.kind(kind).no_start += 1;
                    self.diag.kind(kind).window_no_start += 1;
                    // Nowhere to start from decays it like a failure; it
                    // recovers when a state it could start from is added
                    // (the reset in merge).
                    let e = &mut self.pool.entries[entry];
                    e.unattemptable += 1;
                    if self.cfg.entry_bandit {
                        e.observe(false, self.cfg.bandit_gamma);
                    } else {
                        e.decay *= self.cfg.decay;
                    }
                }
            }
        }
        None
    }

    /// Chains more objectives after `first` (step 3 again, for where the
    /// previous objective is expected to end): with probability `chain_prob`
    /// each, from GA objectives in the pool that are eligible there and whose
    /// target is within `chain_reach`. The kind is picked first, as in
    /// `sample_entry`, so many-entry kinds don't crowd the rest out.
    fn extend_chain(&mut self, entry: usize, first: Arc<dyn Objective>, start: u32) -> Vec<(usize, Arc<dyn Objective>)> {
        let mut stages = vec![(entry, first)];
        if !matches!(stages[0].1.budget(), Budget::Ga { .. }) {
            return stages;
        }
        let mut predicted = self.tree.nodes[start as usize].st.clone();
        let mut keys = vec![stages[0].1.key()];
        while stages.len() < self.cfg.max_chain && self.rng.random_bool(self.diag.chain_prob) {
            let Some(target) = stages.last().unwrap().1.steer_target(&predicted) else { break };
            predicted.pos = target;
            let mut by_kind: BTreeMap<&'static str, Vec<(usize, f64)>> = BTreeMap::new();
            for (i, e) in self.pool.entries.iter().enumerate() {
                let w = e.base_weight * e.decay;
                if w <= 0.0 || !matches!(e.obj.budget(), Budget::Ga { .. }) {
                    continue;
                }
                let Some(t) = e.obj.steer_target(&predicted) else { continue };
                if obrrt_core::dist(t, predicted.pos) > self.cfg.chain_reach
                    || !e.obj.exists_in(&predicted)
                    || !e.obj.base_criteria(&predicted)
                {
                    continue;
                }
                by_kind.entry(e.obj.kind()).or_default().push((i, w));
            }
            by_kind.retain(|_, es| {
                es.retain(|(i, _)| !keys.contains(&self.pool.entries[*i].obj.key()));
                !es.is_empty()
            });
            let kinds: Vec<(&'static str, f64)> = by_kind
                .iter()
                .map(|(k, es)| (*k, es.iter().map(|e| e.1).sum::<f64>() / es.len() as f64 * self.diag.kind(k).multiplier))
                .collect();
            let Some(kind) = weighted(&mut self.rng, &kinds) else { break };
            let Some(next) = weighted(&mut self.rng, &by_kind[kind]) else { break };
            let obj = self.pool.entries[next].obj.clone();
            keys.push(obj.key());
            stages.push((next, obj));
        }
        stages
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
            let start_progress = obj.progress(std::slice::from_ref(&seg_start)).score;
            let start_h = obj.base_heuristic(&seg_start);
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
            let mut end_k = (f64::INFINITY, None);
            carry.extend(seg.inputs);
            if let Some((st, packed)) = seg.end {
                if self.cfg.start_wear || self.cfg.decay_route || self.cfg.entry_bandit {
                    end_k = self.keys_of(obj.as_ref(), &st);
                }
                // Progress alone counts only for objectives worth at least 1:
                // a failed attempt at a sign or a hidden 1-up trigger would
                // otherwise lay down legs of the route toward it.
                let weight = self.pool.entries.get(entry).map_or(1.0, |e| e.base_weight);
                let worth = seg.success || (seg.progress > start_progress + 1e-6 && weight >= 1.0);
                let c = cell(&st);
                // When an attempt meets two or more goals, every met goal's
                // state is kept: each was reached set up for the next goal,
                // which an older state at the same place may not be.
                let chained = seg.success && met >= 2;
                let dup = !chained
                    && self.cells.get(&c).is_some_and(|ids| ids.iter().any(|&i| self.tree.nodes[i as usize].st.frame <= st.frame));
                next_start = Some(st.clone());
                if worth && (!dup || self.cfg.soft_dedup) {
                    novel = !dup;
                    let zone = st.zone();
                    // The worker packed the end against its start's base:
                    // kept as is when that is this zone's base (the same
                    // delta Delta::new makes here), else unpacked and redone.
                    let base = match self.zone_base.get(&zone) {
                        Some(&b) => b,
                        None => {
                            self.bases.push(packed.apply(&self.bases)?.into_bytes());
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
                    if self.cfg.new_eligible {
                        self.ne_note(id, parent);
                    }
                    if self.cfg.earliest {
                        self.ea_note(id);
                    }
                    // Best game time per milestone, for the anytime curve.
                    self.diag.note_best(progress, zone_now, frame);
                    if self.cfg.rrt_star {
                        self.tree.nodes[id as usize].made_by = Some(obj.clone());
                        // Everything already at this place got here later (the
                        // duplicate check only lets a sooner state in): it
                        // stops being a start, and what grew from it is
                        // re-attempted from here.
                        let slower: Vec<u32> = self.cells.get(&c).map_or(Vec::new(), |ids| {
                            ids.iter()
                                .copied()
                                .filter(|&i| !self.tree.nodes[i as usize].dominated && self.tree.nodes[i as usize].st.frame >= frame + self.cfg.rewire_margin)
                                .collect()
                        });
                        for y in slower {
                            self.tree.nodes[y as usize].dominated = true;
                            self.diag.rewire.dominated += 1;
                            for &child in &self.tree.children[y as usize] {
                                if self.tree.nodes[child as usize].made_by.is_none() {
                                    continue;
                                }
                                let r = Rewire { from: id, target: child };
                                if self.hot.contains(&child) {
                                    self.rewire_hot.push_back(r);
                                } else if self.rewire_cold.len() < 50_000 {
                                    self.rewire_cold.push_back(r);
                                }
                                self.diag.rewire.queued += 1;
                            }
                        }
                        if let Some(t) = plan.rewire {
                            let target = &self.tree.nodes[t as usize].st;
                            if cell(target) == c && frame + self.cfg.rewire_margin <= target.frame {
                                self.diag.rewire.improved += 1;
                                self.diag.rewire.frames_saved += (target.frame - frame) as u64;
                            }
                        }
                    }
                    if modal {
                        self.modal_queue.push(id);
                    }
                    if self.cfg.soft_dedup {
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
                    }
                    self.cells.entry(c).or_default().push(id);
                    added = Some(id);
                    advanced = adv;
                    parent = id;
                    if seg.success && self.bc.on() {
                        self.bc.note_success(&self.tree, id, &obj, weight, adv);
                    }
                }
            }
            // Stats.
            {
                let k = self.diag.kind(kind);
                k.attempts += 1;
                k.frames += share;
                k.died += died as u64;
                if seg.success {
                    k.successes += 1;
                    k.success_starts.push(start_h);
                    if i > 0 {
                        k.chained_successes += 1;
                    }
                } else {
                    k.fails += 1;
                    k.fail_progress += seg.progress - start_progress;
                }
                if novel {
                    k.added += 1;
                    chain_added += 1;
                    if seg.success {
                        k.value += obj.success_value();
                        chain_value += obj.success_value();
                    }
                }
                if advanced {
                    k.frontier += 1;
                    chain_frontier += 1;
                }
                // Kind bandit: the entry bandit's improvement signal (no-start
                // picks spend no frames and aren't counted). Modal kinds are
                // never picked through `sample_entry`'s kind stage: they stay
                // unobserved, so out of the normalisation, at multiplier 1.
                if self.cfg.kind_bandit && !obj.modal() {
                    k.observe((seg.success && novel) || (added.is_some() && improves(&self.cfg, end_k, start_k)), self.cfg.kind_gamma);
                    self.diag.refresh_kind_bandit();
                }
            }
            {
                let cfg = &self.cfg;
                let e = &mut self.pool.entries[entry];
                e.attempts += 1;
                // Improved on its start: met, or kept a state nearer by the
                // objective's own heuristic (with `arrival`: sooner by the
                // estimated arrival, so a slow walk inching closer doesn't
                // count; or either, `arrival_decay_either`).
                let improved = added.is_some() && improves(cfg, end_k, start_k);
                let first_from_here = !e.wear.contains_key(&from);
                if self.cfg.start_wear || self.cfg.decay_route {
                    let w = e.wear.entry(from).or_insert(0);
                    if seg.success || improved {
                        *w = 0;
                    } else {
                        *w += 1;
                    }
                }
                // Progress along another route than the best start's: the
                // entry's decay partly recovers (bounded per entry), after
                // this failure's own decay below.
                let recover = self.cfg.decay_route
                    && !seg.success
                    && improved
                    && novel
                    && first_from_here
                    && improves(cfg, (e.best_start, start_k.1.map(|_| e.best_arrival)), start_k)
                    && e.recoveries > 0;
                e.best_start = e.best_start.min(start_k.0);
                if let Some(f) = start_k.1 {
                    e.best_arrival = e.best_arrival.min(f);
                }
                if self.cfg.entry_bandit {
                    e.successes += seg.success as u32;
                    e.observe((seg.success && novel) || improved, self.cfg.bandit_gamma);
                } else if seg.success && novel {
                    e.successes += 1;
                    e.decay = 1.0;
                } else {
                    // A success that only reproduced an existing state decays too.
                    e.successes += seg.success as u32;
                    e.decay *= self.cfg.decay;
                }
                if recover && !self.cfg.entry_bandit && e.decay < 0.5 {
                    e.decay = 0.5;
                    e.recoveries -= 1;
                    self.diag.kind(kind).route_recoveries += 1;
                }
            }
            if i == 0 {
                let ns = &mut self.tree.nodes[plan.start as usize].stats;
                ns.as_start += 1;
                if seg.success {
                    ns.successes += 1;
                }
            }
            writeln!(
                self.events,
                "{}",
                serde_json::json!({
                    "kind": kind, "key": obj.key(), "b": plan.b, "start": plan.start,
                    "chain": chain, "stage": i, "len": len, "from": from,
                    "success": seg.success, "progress": seg.progress - start_progress, "frames": share,
                    "died": died, "added": added, "frontier": advanced,
                    "ga": att.ga.filter(|_| i == 0).map(|g| serde_json::json!({
                        "generations": g.generations, "winner_gen": g.winner_gen, "origin": g.winner_origin,
                        "gen0_met": g.gen0.0, "gen0_tie": g.gen0.1, "best_met": g.best.0, "best_tie": g.best.1,
                        "moves": g.moves,
                    })),
                })
            )?;
            if let Some(id) = added.filter(|_| novel) {
                // A new state may make decayed objectives worth trying again
                // (the entry bandit has no resets: its evidence fades instead).
                // With `arrival`, "better" is judged by `improves` (sooner,
                // and with `arrival_decay_either` also closer).
                let st = &self.tree.nodes[id as usize].st;
                let cfg = &self.cfg;
                for e in self.pool.entries.iter_mut().filter(|_| !cfg.entry_bandit) {
                    if e.decay < 1.0 && e.obj.exists_in(st) && e.obj.base_criteria(st) && {
                        let h = e.obj.base_heuristic(st);
                        let f = arrival_est(cfg, e.obj.as_ref(), st, h);
                        improves(cfg, (h, f), (e.best_start, f.map(|_| e.best_arrival)))
                    } {
                        e.decay = 1.0;
                        *self.decay_resets.entry(e.obj.key()).or_insert(0) += 1;
                    }
                }
                if advanced {
                    let (keys, stars) = self.tree.frontier_progress;
                    let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
                    println!("FRONTIER keys {keys} stars {stars} at node {id} via {kind}: {f}");
                }
                if seg.success && self.first_success.insert(kind) {
                    let f = self.write_m64(id, &format!("first-{}", kind.replace("::", "-")))?;
                    println!("first success of {kind} at node {id}: {f}");
                }
            }
            if let Some(st) = next_start {
                seg_start = st;
            }
        }
        {
            let ns = &mut self.tree.nodes[plan.start as usize].stats;
            if chain_added > 0 {
                ns.added += chain_added;
                ns.barren = 0;
                ns.backoff = 1.0;
            } else {
                ns.barren += 1;
                ns.backoff *= self.cfg.state_decay;
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
        // Zone bandit statistics: per zone of b, attempts and new states,
        // decaying so the weights follow recent progress.
        if self.cfg.zone_bandit {
            for v in self.zone_stats.values_mut() {
                v.0 *= 0.998;
                v.1 *= 0.998;
            }
            let z = self.tree.nodes[plan.b as usize].st.zone();
            let e = self.zone_stats.entry(z).or_insert((0.0, 0.0));
            e.0 += chain_added as f64;
            e.1 += 1.0;
        }
        Ok(())
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
        if let Some(single) = single.filter(|_| multi.attempts >= 20 && self.cfg.chain_fixed.is_none()) {
            let ratio = (multi.yield_per_kframe() + 0.05) / (single + 0.05);
            self.diag.chain_prob = (0.7 * self.diag.chain_prob + 0.3 * (0.5 * ratio)).clamp(0.1, 0.9);
        }
        let mean = self.diag.mean_yield();
        let kind_bandit = self.cfg.kind_bandit;
        for k in self.diag.kinds.values_mut() {
            if k.attempts < 10 {
                continue;
            }
            // Kind bandit: the multiplier is set at every merge
            // (`Diagnostics::refresh_kind_bandit`), not here.
            if !kind_bandit {
                let y = k.yield_per_kframe();
                let target = ((y + 0.05) / (mean + 0.05)).clamp(0.1, 5.0);
                let old = k.multiplier;
                k.multiplier = 0.7 * k.multiplier + 0.3 * target;
                if (k.multiplier - old).abs() > 0.05 {
                    k.reason = format!("yield {y:.2} vs mean {mean:.2}");
                }
            }
            // Low success: only start where successes started (90th
            // percentile of their base_heuristic, plus a margin).
            if k.success_starts.len() >= 3 && k.success_rate() < 0.25 {
                let mut hs = k.success_starts.clone();
                hs.sort_by(f64::total_cmp);
                let p90 = hs[((hs.len() - 1) as f64 * 0.9) as usize];
                let learned = p90 + 0.25 * p90.abs() + 1.0;
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
        self.bc.write(&self.cfg.out_dir, &self.tree)?;
        self.events.flush()?;
        let mut csv = String::from("id,parent,depth,x,y,z,level,area,stars,keys,coins,frame,via,as_b,as_start,successes,added,barren,hspeed,fwd_vel,action,red_coins,act,star_flags,retired\n");
        for (i, n) in self.tree.nodes.iter().enumerate() {
            let s = &n.st;
            let st = &n.stats;
            csv += &format!(
                "{i},{},{},{:.0},{:.0},{:.0},{},{},{},{},{},{},{},{},{},{},{},{},{:.1},{:.1},{:#x},{},{},{},{}\n",
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
                n.retired as u8
            );
        }
        fs::write(format!("{}/nodes.csv", self.cfg.out_dir), csv)?;
        self.write_tree()?;
        // Per kind: entries, mean objective weight multiplier (decay, or the
        // entry bandit's draw) and discounted improvement rate.
        let mut weights: BTreeMap<&'static str, (usize, f64, f64, f64)> = BTreeMap::new();
        for e in &self.pool.entries {
            let w = weights.entry(e.obj.kind()).or_default();
            w.0 += 1;
            w.1 += e.decay;
            w.2 += e.bandit_k;
            w.3 += e.bandit_n;
        }
        let weights: BTreeMap<&str, serde_json::Value> = weights
            .into_iter()
            .map(|(k, (n, d, bk, bn))| (k, serde_json::json!({"entries": n, "mean_decay": d / n as f64, "improve_rate": if bn > 0.0 { bk / bn } else { 0.0 }})))
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
        if self.tree.subset {
            summary["frontier"]["sets"] = serde_json::json!(self.tree.sets.len());
            summary["frontier"]["sets_peak"] = serde_json::json!(self.tree.sets_peak);
            summary["frontier"]["sets_dropped"] = serde_json::json!(self.tree.dropped.len());
            summary["frontier"]["set_list"] = self.tree.sets.iter().map(|f| serde_json::json!({"keys": f.key.1, "stars": f.key.2, "states": f.states, "last": f.last, "frontier_states": self.tree.frontier.iter().filter(|&&i| crate::tree::set_key(&self.tree.nodes[i as usize].st) == f.key).count()})).collect();
        }
        if self.tree.keep_wip {
            let t = &self.tree;
            summary["frontier"]["wip"] = serde_json::json!({
                "kept": t.wip_kept.len(), "peak": t.wip_peak, "ever": t.wip_ever.len(), "dropped": t.wip_dropped,
                "kept_list": t.wip_kept.iter().map(|&i| serde_json::json!({"id": i, "level": t.nodes[i as usize].st.level, "stars": t.nodes[i as usize].st.stars, "frame": t.nodes[i as usize].st.frame, "wip": t.wip[i as usize]})).collect::<Vec<_>>(),
            });
        }
        summary["decay_resets"] = serde_json::json!(self.decay_resets);
        summary["timing"] = serde_json::json!({"plan_secs": self.diag_plan_secs.0, "merge_secs": self.diag_plan_secs.1});
        if self.cfg.new_eligible {
            summary["new_eligible"] = serde_json::json!({
                "queued": self.ne.queued, "dropped": self.ne.dropped, "stale": self.ne.stale, "used": self.ne.used,
                "planned": self.ne.planned, "queue": self.ne.queue.len(), "keys": self.ne.queued_per_key.len(),
                "sets": self.ne.sets.len(), "secs": self.ne.secs,
            });
        }
        if self.cfg.earliest {
            summary["earliest"] = serde_json::json!({
                "queued_tier": self.ea.queued_tier, "queued_cell": self.ea.queued_cell, "dropped": self.ea.dropped,
                "stale": self.ea.stale, "planned": self.ea.planned, "no_objective": self.ea.no_objective,
                "queue": self.ea.queue.len(), "tiers": self.ea.tier_best.len(), "cells": self.ea.cell_best.len(),
            });
        }
        if self.cfg.kind_bandit {
            let kb: BTreeMap<&str, serde_json::Value> = self
                .diag
                .kinds
                .iter()
                .map(|(name, k)| (*name, serde_json::json!({"k": k.bandit_k, "n": k.bandit_n, "m": k.bandit_mean(), "multiplier": k.multiplier})))
                .collect();
            summary["kind_bandit"] = serde_json::json!(kb);
        }
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
            let mut replay = |save: &Delta, bases: &[Vec<u8>]| -> Result<(St, SaveState)> {
                worker.base.load_state(&save.apply(bases)?)?;
                for p in &inputs {
                    worker.base.step_game(*p)?;
                }
                Ok((read_state(&worker.base, frame, true), worker.base.save_state()))
            };
            let (mut st, mut save) = replay(&pnode.save, &self.bases)?;
            if !matches(&st) && parent != 0 {
                let (st0, save0) = replay(&Delta::default(), &self.bases)?;
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
                    self.bases.push(save.as_bytes().to_vec());
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
            .map(|(i, n)| (n.stats.backoff * 0.5f64.powi(n.stats.crowd as i32), i as u32))
            .collect();
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let target = self.cfg.mem_budget / 10 * 9;
        let mut freed = 0;
        for (_, i) in cands {
            if self.tree.delta_bytes <= target {
                break;
            }
            self.tree.retire_one(i);
            freed += 1;
        }
        println!("soft dedupe: retired {freed} repeated states; saves now {} MB", self.tree.delta_bytes / 1_000_000);
    }

    pub fn run(&mut self) -> Result<()> {
        let begin = Instant::now();
        let mut last_report = Instant::now();
        let mut last_tune = self.total_frames;
        let mut round = 0u64;
        // Stop on wall-clock time, or on emulated frames (max_frames) so runs
        // compare equally whatever the thread count or CPU sharing.
        while begin.elapsed() < Duration::from_secs_f64(self.cfg.minutes * 60.0)
            && self.cfg.max_frames.is_none_or(|m| self.total_frames < m)
        {
            round += 1;
            if self.cfg.entry_bandit {
                self.pool.redraw(&mut self.rng);
            }
            self.tree.wip_refresh();
            self.index_frontier();
            self.ne.budget = if self.cfg.new_eligible { ((self.cfg.round_plans as f64 * NE_SHARE).ceil() as usize).max(1) } else { 0 };
            if self.cfg.earliest {
                self.ea.budget = ((self.cfg.round_plans as f64 * self.cfg.earliest_share).ceil() as usize).max(1);
                self.ea.round_used.clear();
            }
            let plan_t = Instant::now();
            // More attempts than workers, so a slow attempt doesn't leave the
            // others idle: each takes whichever game is free.
            // A fixed number of attempts per round, whatever the thread
            // count: the tree only updates between rounds, so the round size
            // shapes the search (how stale planning is); 64 = 16 threads × 4.
            self.bc.begin_round(self.cfg.round_plans);
            let plans: Vec<Plan> = (0..self.cfg.round_plans).filter_map(|_| self.plan().inspect(|_| self.bc.made += 1)).collect();
            self.diag_plan_secs.0 += plan_t.elapsed().as_secs_f64();
            if plans.is_empty() {
                continue;
            }
            let bases = &self.bases;
            let tree = &self.tree;
            let seeds: Vec<u64> = plans.iter().map(|_| self.rng.random()).collect();
            let workers = &self.workers;
            let bc = &self.bc;
            let params = crate::ga::GaParams { scale: self.cfg.ga_scale, fitness: self.cfg.fitness, new_ops: self.cfg.ga_new_ops, pink: self.cfg.ga_pink, per_objective_budget: self.cfg.ga_per_objective_budget };
            let results: Vec<Result<Attempt<Delta>>> = plans
                .par_iter()
                .zip(seeds.par_iter())
                .enumerate()
                .map(|(i, (plan, seed))| {
                    let mut w = loop {
                        if let Some(w) = workers.iter().find_map(|w| w.try_lock().ok()) {
                            break w;
                        }
                        std::thread::yield_now();
                    };
                    let node = &tree.nodes[plan.start as usize];
                    let save = node.save.apply(bases)?;
                    let mut rng = SmallRng::seed_from_u64(*seed);
                    let stages: Vec<&dyn Objective> = plan.stages.iter().map(|(_, o)| o.as_ref()).collect();
                    let packed = w.attempt_seeded(&stages, &node.st, &save, &mut rng, params, bc.seed(i)).map(|a| {
                        let base = node.save.base;
                        a.map_saves(|s| Delta::new(base, &bases[base as usize], &s))
                    });
                    anyhow::Context::with_context(packed, || {
                        format!(
                            "attempt from node {} (via {}, retired {}, frame {}, zone {:?}, base {}) for {}",
                            plan.start,
                            node.via,
                            node.retired,
                            node.st.frame,
                            node.st.zone(),
                            node.save.base,
                            plan.stages.iter().map(|(_, o)| o.key()).collect::<Vec<_>>().join(" then ")
                        )
                    })
                })
                .collect();
            let merge_t = Instant::now();
            for (i, (plan, r)) in plans.iter().zip(results).enumerate() {
                let bc = self.bc.is_pending(i).then(|| (self.tree.nodes.len(), r.as_ref().ok().map(crate::backchain::outcome)));
                match r {
                    Ok(att) => self.merge(plan, att)?,
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
                            if let Some(e) = self.pool.entries.get_mut(*entry) {
                                if self.cfg.entry_bandit {
                                    e.observe(false, self.cfg.bandit_gamma);
                                } else {
                                    e.decay *= self.cfg.decay;
                                }
                            }
                            self.diag.kind(obj.kind()).died += 1;
                        }
                        if self.emulator_errors <= 20 {
                            eprintln!("emulator trap #{} (node {} traps {}): {:#}", self.emulator_errors, plan.start, traps, e);
                        }
                    }
                }
                if let Some((before, out)) = bc {
                    self.bc.merged(i, &self.tree, before, out);
                }
            }
            self.diag_plan_secs.1 += merge_t.elapsed().as_secs_f64();
            if self.cfg.soft_dedup && self.tree.delta_bytes > self.cfg.mem_budget {
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
            if last_report.elapsed() > self.cfg.report_every {
                last_report = Instant::now();
                self.write_outputs(begin.elapsed().as_secs_f64())?;
            }
        }
        self.write_outputs(begin.elapsed().as_secs_f64())?;
        Ok(())
    }
}
