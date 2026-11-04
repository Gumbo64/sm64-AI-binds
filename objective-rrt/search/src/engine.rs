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
    /// OBRRT_REROUTE=1: graft slower duplicates' subtrees onto faster ones by
    /// replaying their inputs (see `reroute`). Off by default.
    pub reroute: bool,
    /// OBRRT_REROUTE_SCHEMA: an objective-scattershot schema whose key says
    /// what "the same state" is; without it, the duplicate check's cell.
    pub reroute_schema: Option<String>,
    /// OBRRT_REROUTE_SHARE: at most this share of a round's slots for grafts.
    pub reroute_share: f64,
    /// OBRRT_POLISH=<share>: at most this share of a round's plans re-attempt
    /// the best route's earlier milestones (see `polish`); 0 = off.
    pub polish: f64,
    /// OBRRT_REROUTE_MARGIN: frames a graft must save.
    pub reroute_margin: u32,
    /// OBRRT_ZONE_SHARE=1: replay the inputs states took since entering
    /// their zone from other entries of it (see `share`). Off by default.
    pub zone_share: bool,
    /// OBRRT_ZONE_SHARE_K: top donors per zone entry (4).
    pub zone_share_k: usize,
    /// OBRRT_ZONE_SHARE_CAP: frames a replayed path is at most (3000).
    pub zone_share_cap: u32,
    /// OBRRT_ZONE_SHARE_SHARE: at most this share of a round's slots (0.2).
    pub zone_share_share: f64,
    /// OBRRT_ZONE_SHARE_BG: background jobs drawn per round (2).
    pub zone_share_bg: usize,
    /// OBRRT_REPLAY_BUDGET=<fraction>: replay jobs (reroute grafts, zone
    /// share) take no plan slots; after each round's plans they run until
    /// their frames reach this fraction of the plans' (see `replay`).
    /// None (default): the slot shares above.
    pub replay_budget: Option<f64>,
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
            reroute: false,
            reroute_schema: None,
            reroute_share: 0.25,
            polish: 0.0,
            reroute_margin: 10,
            zone_share: false,
            zone_share_k: 4,
            zone_share_cap: 3000,
            zone_share_share: 0.2,
            zone_share_bg: 2,
            replay_budget: None,
            chain_route: false,
            chain_follow: 0.5,
            sampler_rate: false,
            rate: crate::pool::RateParams { floor: 0.15, cap: 4.0, tau: 50_000_000.0, prior_frames: 200_000.0, nostart_frames: 5_000.0, hier: false, kind_scale: 10.0, nostart_skip: false, skip_budget: 2_000_000, route: 0.0, novel: 1.0 },
            node_prior: 300_000.0,
            node_cap: 2.0,
            kinship_mult: 1.0,
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
    /// RRT* rewiring: the slower state whose objective this re-attempts.
    rewire: Option<u32>,
    /// OBRRT_NEAR_SEED: recorded inputs of a nearby start's attempt of the
    /// first stage, seeding the GA (see near.rs).
    seed: Option<Vec<GamePad>>,
    /// OBRRT_POLISH: the route job this plan is (see `polish`).
    polish: Option<crate::polish::Job>,
    /// OBRRT_MODEL_REPAIR: the seed's recorded states with model data.
    targets: Option<crate::ga::RepairTargets>,
}

/// OBRRT_POLISH: route-graft waves per round at most.
const POLISH_WAVES: usize = 64;

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReplayEnd {
    Done,
    /// Dead at an edge boundary (that state isn't recorded).
    Died,
    /// Recorded a boundary outside the start's zone, and stopped there.
    LeftZone,
}

/// A replay job's states, one per edge boundary passed.
struct Replayed {
    states: Vec<(St, Delta)>,
    frames: u64,
    end: ReplayEnd,
}

/// Replays the recorded inputs of `edges` (node ids) in order from node
/// `from`'s save, recording the state (and its save, against `from`'s base)
/// at the end of each; stops at a dead boundary and, with `stay`, after a
/// boundary outside `from`'s zone.
fn replay_edges(game: &mut sm64_binds::SM64Game, tree: &Tree, bases: &[Vec<u8>], from: u32, edges: &[u32], stay: bool) -> Result<Replayed> {
    let base = tree.base_of(from);
    tree.load_into(from, game, bases)?;
    let from = &tree.nodes[from as usize];
    let mut out = Replayed { states: Vec::with_capacity(edges.len()), frames: 0, end: ReplayEnd::Done };
    for &e in edges {
        let inputs = &tree.nodes[e as usize].inputs;
        for p in inputs {
            game.step_game(*p)?;
        }
        out.frames += inputs.len() as u64;
        let st = read_state(game, from.st.frame + out.frames as u32, true);
        if !st.alive() {
            out.end = ReplayEnd::Died;
            break;
        }
        let left = stay && st.zone() != from.st.zone();
        out.states.push((st, Delta::new(base, &bases[base as usize], &game.save_state())));
        if left {
            out.end = ReplayEnd::LeftZone;
            break;
        }
    }
    Ok(out)
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
    /// Rerouting onto faster duplicates (`Config::reroute`).
    rr: Option<crate::reroute::Reroute>,
    polish: Option<crate::polish::Polish>,
    /// Zone-entry input sharing (`Config::zone_share`).
    zs: Option<crate::share::ZoneShare>,
    /// Replay budget (`Config::replay_budget`).
    rb: Option<crate::replay::ReplayBudget>,
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
    /// Wall seconds (report only): [pre-plan refresh+index_frontier, attempts
    /// (parallel), post-merge upkeep, write_outputs, worker load_into (summed
    /// over threads), worker attempt (summed)].
    diag_secs: [f64; 6],
    /// Route chains: per objective key, the best-route (`hot`) nodes where
    /// it was met (ids ascending); rebuilt with `hot`.
    route_met: HashMap<String, Vec<u32>>,
    /// Rate sampler: the tree's new states and frames over all starts
    /// (faded like each node's), and when last faded.
    node_rate: (f64, f64, u64),
    /// OBRRT_RATE_NOSTART_SKIP: live state ids at the round's start, the
    /// node count then, and the re-check's round-robin cursor.
    live: Vec<u32>,
    live_upto: u32,
    skip_rr: usize,
    /// Dense adds (OBRRT_ADD, see dense.rs): settings and totals.
    dense: Option<crate::dense::Settings>,
    dstats: crate::dense::MergeStats,
    /// Seeds from nearby starts (OBRRT_NEAR_SEED, see near.rs).
    near: crate::near::NearSeed,
    /// OBRRT_PICK=schema|schema_kin: start selection by schema cells (schemapick.rs).
    spick: Option<crate::schemapick::SchemaPick>,
    /// Sparse saves: rebuild frames already counted into total_frames.
    rebuild_counted: u64,
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
        let rr = if cfg.reroute {
            let key = match &cfg.reroute_schema {
                Some(path) => crate::reroute::RerouteKey::Schema(Box::new(objective_scattershot::objective::Tree::load(&fs::read_to_string(path)?)?)),
                None => crate::reroute::RerouteKey::Cell,
            };
            Some(crate::reroute::Reroute::new(key))
        } else if cfg.polish > 0.0 {
            // Route grafts only.
            let mut r = crate::reroute::Reroute::new(crate::reroute::RerouteKey::Cell);
            r.route_only = true;
            Some(r)
        } else {
            None
        };
        let mut zs = cfg.zone_share.then(|| crate::share::ZoneShare::new(cfg.zone_share_k, cfg.zone_share_cap, cfg.zone_share_bg, cfg.seed));
        let mut rr = rr;
        let rb = cfg.replay_budget.map(crate::replay::ReplayBudget::new);
        if rb.is_some() {
            if let Some(r) = rr.as_mut() {
                r.lifo = true;
            }
            if let Some(z) = zs.as_mut() {
                z.lifo = true;
            }
        }
        Ok(Engine {
            rr,
            polish: (cfg.polish > 0.0).then(|| crate::polish::Polish::new(cfg.polish)),
            zs,
            rb,
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
            diag_secs: [0.0; 6],
            route_met: HashMap::new(),
            node_rate: (0.0, 0.0, 0),
            live: Vec::new(),
            live_upto: 0,
            skip_rr: 0,
            bc: crate::backchain::Backchain::new(cfg.seed),
            dense: crate::dense::settings().filter(|d| d.modes.any()),
            dstats: Default::default(),
            near: crate::near::NearSeed::new(cfg.seed),
            spick: crate::schemapick::from_env()?,
            rebuild_counted: 0,
            cfg,
        })
    }

    /// Step 3: picks a kind (by the mean weight of its entries × its learned
    /// multiplier, with a floor; so a kind with many entries doesn't crowd
    /// out the rest), then an entry within it by weight.
    /// With `strict` (earliest-arrival boost), only objectives b itself can
    /// start (within its kind's cutoff), whatever `pick_by_b`.
    fn sample_entry(&mut self, b: u32, strict: bool) -> Option<usize> {
        if strict || self.cfg.pick_by_b {
            return self.sample_entry_scan(b, strict);
        }
        // OBRRT_SAMPLE_CHECK=1: the scan on a copy of the RNG must agree
        // (pick and RNG state) with the per-kind tables.
        static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *CHECK.get_or_init(|| std::env::var("OBRRT_SAMPLE_CHECK").is_ok_and(|v| v == "1")) {
            let rng0 = self.rng.clone();
            let want = self.sample_entry_scan(b, strict);
            let mut rng1 = std::mem::replace(&mut self.rng, rng0);
            let got = self.sample_entry_fast(b);
            let mut rng2 = self.rng.clone();
            assert!(got == want && rng1.random::<u64>() == rng2.random::<u64>(), "sample_entry tables {got:?} vs scan {want:?}");
            return got;
        }
        self.sample_entry_fast(b)
    }

    /// `sample_entry` without b-dependent eligibility, from the pool's
    /// per-kind tables (`SampleIdx`): the same sums, draws and pick as the
    /// scan, without a pass over the pool.
    fn sample_entry_fast(&mut self, b: u32) -> Option<usize> {
        let idx = self.pool.sidx.get_or_insert_with(Default::default);
        idx.sync(&self.pool.entries);
        if idx.irregular > 0 {
            return self.sample_entry_scan(b, false);
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
        Some(chosen)
    }

    /// `sample_entry` by a pass over the whole pool (any eligibility).
    fn sample_entry_scan(&mut self, b: u32, strict: bool) -> Option<usize> {
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
                        && e.ns_upto == 0
                        && (ok(e.obj.as_ref(), b_st) || parent.is_some_and(|p| !p.retired && !p.st.modal() && ok(e.obj.as_ref(), &p.st)))
                })
                .collect()
        } else {
            self.pool.entries.iter().map(|e| !e.obj.modal() && e.ns_upto == 0).collect()
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
        let k = if self.cfg.kinship_mult == 1.0 { k } else { ((k as f64) * self.cfg.kinship_mult).round().max(0.0) as u32 };
        if let Some(sp) = self.spick.as_mut() {
            let kin = (sp.mode == crate::schemapick::Mode::SchemaKin).then(|| self.tree.kin_set(b, k));
            let cx = crate::schemapick::Ctx { obj, entry, b, cutoff, rrt_star: self.cfg.rrt_star, kin, epoch: (self.tree.nodes.len(), self.total_frames) };
            return sp.pick(&self.tree, cx, &mut self.rng);
        }
        let b_st = &self.tree.nodes[b as usize].st;
        let (b_keys, b_stars, b_lives) = (b_st.keys, b_st.stars, b_st.lives);
        // OBRRT_RETIRE_SUBSET: a start has everything b has, not just as many.
        let b_set = self.tree.subset.then(|| crate::tree::set_key(b_st));
        let mut cands: Vec<(u32, f64, f64)> = Vec::new();
        // Arrival band: each candidate's own heuristic, to keep only those
        // within `arrival_band` of the closest.
        let mut raw_h: Vec<f64> = Vec::new();
        // OBRRT_KIN_SOFT: each member's weight × λ^distance (else 1).
        let kin = self.tree.kin_weighted(b, k);
        for &(i, kw) in kin.iter() {
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
            cands.push((i as u32, h, u * self.start_mult(&n.stats) * crowd * wear * settle * kw as f64));
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
        let mut weights: Vec<f64> = cands.iter().map(|c| (-(c.1 - hmin) / t).exp() * c.2).collect();
        // OBRRT_MODEL (model.rs): starts the model puts fewer frames from a
        // star they haven't collected weigh more.
        if let Some(mt) = crate::model::settings().map(|s| s.start_t).filter(|&t| t > 0.0) {
            let score = |i: u32| self.tree.nodes[i as usize].stats.model.as_ref().map(|m| m.score as f64);
            let smin = cands.iter().filter_map(|c| score(c.0)).fold(f64::INFINITY, f64::min);
            if smin.is_finite() {
                for (w, c) in weights.iter_mut().zip(&cands) {
                    if let Some(sc) = score(c.0) {
                        *w *= (-(sc - smin) / mt).exp();
                    }
                }
            }
        }
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
    }

    /// A dialog or menu state as b: its options have infinite weight (and
    /// everything else none), so one of them is attempted from it. From the
    /// queue, each option once, untried first; otherwise by weight.
    /// The model's learned objectives proposed at b (OBRRT_MODEL_LEARNED).
    fn insert_learned(&mut self, b: u32, b_st: &St) {
        let Some(m) = self.tree.nodes[b as usize].stats.model.clone() else { return };
        for o in &m.proposals {
            self.pool.insert(o.clone(), b_st);
        }
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
        self.insert_learned(b, &b_st);
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
        Some(Plan { stages: vec![(entry, obj)], start: b, b, rewire: None, seed: None, polish: None, targets: None })
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
            return Some(Plan { stages: vec![(entry, obj)], start: r.from, b: r.from, rewire: Some(r.target), seed: None, polish: None, targets: None });
        }
        None
    }

    /// OBRRT_POLISH: the next polish plan: a queued repair, else a route
    /// milestone drawn by its leg's length (fewer draws after failed tries),
    /// attempted from a route state up to `back` nodes above it, seeded with
    /// the route's own inputs (see `polish`).
    fn plan_polish(&mut self) -> Option<Plan> {
        let job = loop {
            let pl = self.polish.as_mut()?;
            if let Some(j) = pl.repairs.pop_front() {
                break j;
            }
            let route = &pl.route;
            let ms = pl.milestones(&self.tree);
            if ms.is_empty() {
                pl.stats.no_target += 1;
                return None;
            }
            let mut prev = 0usize;
            let mut w: Vec<(usize, usize, f64)> = Vec::with_capacity(ms.len());
            for &j in &ms {
                let f = self.tree.nodes[route[j] as usize].st.frame.saturating_sub(self.tree.nodes[route[prev] as usize].st.frame).max(30) as f64;
                let t = pl.tries.get(&route[j]).copied().unwrap_or(0) as f64;
                w.push((j, prev, f / (1.0 + t / 4.0)));
                prev = j;
            }
            let total: f64 = w.iter().map(|x| x.2).sum();
            let mut x = self.rng.random::<f64>() * total;
            let mut pick = w[w.len() - 1];
            for &c in &w {
                if x < c.2 {
                    pick = c;
                    break;
                }
                x -= c.2;
            }
            let (j, prev, _) = pick;
            let span = (j - prev).min(pl.back).max(1);
            let k = self.rng.random_range(1..=span);
            let from = route[j - k];
            break crate::polish::Job { from, anchor: from, target: route[j], end: pl.end, repair: false };
        };
        let pl = self.polish.as_mut().unwrap();
        let n = &self.tree.nodes[job.from as usize];
        let Some(obj) = self.tree.nodes[job.target as usize].made_by.clone() else {
            pl.stats.no_target += 1;
            return None;
        };
        let loadable = job.from == 0 || !n.retired || self.tree.is_pinned(job.from);
        if !loadable || n.st.modal() || n.st.objects.is_none() || !obj.exists_in(&n.st) || !obj.base_criteria(&n.st) {
            pl.stats.no_target += 1;
            return None;
        }
        let Some(seed) = crate::polish::Polish::inputs(&self.tree, job.anchor, job.target) else {
            pl.stats.no_target += 1;
            return None;
        };
        pl.stats.plans += 1;
        if job.repair {
            pl.stats.repairs += 1;
        } else {
            *pl.tries.entry(job.target).or_default() += 1;
        }
        let entry = match self.pool.find(&obj.key()) {
            Some(e) => e,
            None => {
                self.pool.insert(obj.clone(), &n.st);
                self.pool.entries.len() - 1
            }
        };
        let targets = crate::model::settings().filter(|m| m.repair > 0).map(|_| {
            // The route's states from the anchor down to the target, each
            // with its own model data, at their frames after the anchor.
            let mut path = Vec::new();
            let mut at = job.target;
            while at != job.anchor {
                path.push(at);
                match self.tree.nodes[at as usize].parent {
                    Some(p) => at = p,
                    None => break,
                }
            }
            let f0 = self.tree.nodes[job.anchor as usize].st.frame;
            let mut out: Vec<(u32, Arc<crate::model::NodeModel>)> = Vec::new();
            for &i in path.iter().rev() {
                let n = &self.tree.nodes[i as usize];
                if let Some(m) = &n.stats.model {
                    if out.last().is_none_or(|(_, l)| !Arc::ptr_eq(l, m)) {
                        out.push((n.st.frame.saturating_sub(f0), m.clone()));
                    }
                }
            }
            Arc::new(out)
        });
        Some(Plan { stages: vec![(entry, obj)], start: job.from, b: job.from, rewire: None, seed: Some(seed), polish: Some(job), targets })
    }

    /// OBRRT_POLISH: a polish or repair plan's first segment ended at `end`
    /// (its node, and whether it met the objective). A result of the target's
    /// progress set at least the margin sooner continues the route: grafted
    /// edge by edge if it is the target's movement key, else a repair toward
    /// the next milestone.
    fn polish_result(&mut self, job: crate::polish::Job, end: Option<(u32, bool)>) {
        let Some(pl) = self.polish.as_mut() else { return };
        let Some((n, true)) = end else {
            pl.stats.failed += 1;
            return;
        };
        pl.pinned(n);
        let (ns, ts) = (&self.tree.nodes[n as usize].st, &self.tree.nodes[job.target as usize].st);
        if ns.progress_set() != ts.progress_set() || ns.stars != ts.stars || ns.keys != ts.keys {
            pl.stats.other_set += 1;
            return;
        }
        let bar = pl.best.get(&job.target).copied().unwrap_or(ts.frame).min(ts.frame);
        if ns.frame + pl.margin > bar {
            pl.stats.not_sooner += 1;
            return;
        }
        pl.best.insert(job.target, ns.frame);
        let saved = (ts.frame - ns.frame) as u64;
        pl.stats.sooner += 1;
        pl.stats.sooner_frames += saved;
        pl.tries.remove(&job.target);
        if job.target == job.end {
            pl.stats.done += 1;
            pl.stats.done_frames += saved;
            return;
        }
        let same = crate::reroute::move_key(ns) == crate::reroute::move_key(ts);
        if pl.stats.sooner <= 30 {
            let tn = &self.tree.nodes[job.target as usize];
            println!(
                "POLISH sooner #{}: target {} ({} via {}, frame {}) from {} (frame {}) repair {}: node {} frame {} (-{saved}), same key {same}, end {}",
                pl.stats.sooner, job.target, tn.made_by.as_ref().map_or(String::new(), |m| m.key()), tn.via, ts.frame, job.from, self.tree.nodes[job.from as usize].st.frame, job.repair, n, ns.frame, job.end
            );
        }
        let Some(child) = crate::legs::below(&self.tree, job.target, job.end) else {
            pl.stats.no_child += 1;
            return;
        };
        if same {
            pl.stats.same_key += 1;
            pl.stats.chains += 1;
            if let Some(rr) = self.rr.as_mut() {
                rr.push(crate::reroute::Graft { from: n, child, depth: 1, kind: crate::reroute::KIND_ROUTE, leg: job.end }, true);
            }
        } else if let Some(t) = crate::polish::Polish::next_milestone(&self.tree, child, job.end) {
            pl.stats.other_key += 1;
            pl.push_repair(crate::polish::Job { from: n, anchor: job.target, target: t, end: job.end, repair: true });
        } else {
            pl.stats.no_next += 1;
        }
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

    /// Rate sampler, OBRRT_RATE_ROUTE: node `id` set a new best time for its
    /// milestone (`old`: the previous best, u32::MAX if none). The entries
    /// that made the states on its path since the previous star share the
    /// credit, so objectives that lead to stars rate above dead ends.
    fn route_credit(&mut self, id: u32, old: u32) {
        let c = self.cfg.rate.route;
        if !self.cfg.sampler_rate || c <= 0.0 {
            return;
        }
        let n = &self.tree.nodes[id as usize];
        let imp = c * if old == u32::MAX { 5.0 } else { (old.saturating_sub(n.st.frame) as f64 / 30.0).min(5.0) };
        if imp <= 0.0 {
            return;
        }
        let floor = n.st.stars - 1;
        let mut js: Vec<usize> = Vec::new();
        let mut cur = Some(id);
        while let Some(i) = cur {
            let n = &self.tree.nodes[i as usize];
            if n.st.stars < floor {
                break;
            }
            if let Some(j) = n.made_by.as_ref().and_then(|m| self.pool.find(&m.key())) {
                if !js.contains(&j) {
                    js.push(j);
                }
            }
            cur = n.parent;
        }
        if js.is_empty() {
            return;
        }
        let p = self.cfg.rate;
        let share = imp / js.len() as f64;
        for j in js {
            self.pool.rate_note(j, share, 0.0, self.total_frames, &p, false);
        }
        self.pool.rate.imp_route += imp;
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
            self.insert_learned(b, &b_st);
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
                    out = Some(Plan { stages, start, b, rewire: None, seed: None, polish: None, targets: None });
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
            self.insert_learned(b, &b_st);
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
            return Some(Plan { stages, start: b, b, rewire: None, seed: None, polish: None, targets: None });
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
                return Some(Plan { stages: vec![(entry, obj)], start, b: start, rewire: None, seed: None, polish: None, targets: None });
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
        self.insert_learned(b, &b_st);
        // Steps 3-4, retrying a few times when a pick has no eligible start.
        for _ in 0..4 {
            // Wall time per pick (diagnostics only, never decisions).
            let pick_t = Instant::now();
            let entry = self.sample_entry(b, false)?;
            self.pool.rate.sample_secs += pick_t.elapsed().as_secs_f64();
            let obj = self.pool.entries[entry].obj.clone();
            let kind = obj.kind();
            self.diag.kind(kind).samples += 1;
            self.diag.kind(kind).window_samples += 1;
            let picked = self.pick_start(entry, obj.as_ref(), b);
            let secs = pick_t.elapsed().as_secs_f64();
            if picked.is_some() { self.pool.rate.pick_secs += secs } else { self.pool.rate.nostart_secs += secs }
            match picked {
                Some(start) => {
                    self.pool.entries[entry].unattemptable = 0;
                    let stages = self.extend_chain(entry, obj, start);
                    return Some(Plan { stages, start, b, rewire: None, seed: None, polish: None, targets: None });
                }
                None => {
                    self.diag.kind(kind).no_start += 1;
                    self.diag.kind(kind).window_no_start += 1;
                    // Nowhere to start from decays it like a failure; it
                    // recovers when a state it could start from is added
                    // (the reset in merge).
                    let e = &mut self.pool.entries[entry];
                    e.unattemptable += 1;
                    if self.cfg.sampler_rate {
                        // Rate sampler: a pick with nowhere to start costs a
                        // few frames' worth of rate (faded like the rest).
                        let p = self.cfg.rate;
                        if p.nostart_skip && !self.any_start(obj.as_ref(), 0, &mut { u64::MAX }).unwrap_or(true) {
                            // No live state could start it, whatever b:
                            // skipped (not charged) until a newer one can.
                            self.pool.entries[entry].ns_upto = self.live_upto.max(1);
                            self.pool.rate.nostart_global += 1;
                        } else {
                            self.pool.rate.nostart_kin += p.nostart_skip as u64;
                            self.pool.rate_note(entry, 0.0, p.nostart_frames, self.total_frames, &p, true);
                            self.pool.rate_one(entry, self.total_frames, &p);
                            self.pool.rate.nostart_charges += 1;
                        }
                    } else if self.cfg.entry_bandit {
                        e.observe(false, self.cfg.bandit_gamma);
                    } else {
                        e.decay *= self.cfg.decay;
                    }
                    if let Some(idx) = self.pool.sidx.as_mut() {
                        idx.touch(&self.pool.entries, entry);
                    }
                }
            }
        }
        self.pool.rate.plans_none += 1;
        None
    }

    /// OBRRT_RATE_NOSTART_SKIP: whether some live state with id >= `from`
    /// could start `obj` for some b (pick_start's tests that don't depend on
    /// b: live, not dominated, alive, modal alike, base criteria), spending
    /// one of `budget` per state tried; None when the budget ran out first.
    fn any_start(&self, obj: &dyn Objective, from: u32, budget: &mut u64) -> Option<bool> {
        let at = self.live.partition_point(|&i| i < from);
        for &i in &self.live[at..] {
            if *budget == 0 {
                return None;
            }
            *budget -= 1;
            let n = &self.tree.nodes[i as usize];
            if !n.retired && !(self.cfg.rrt_star && n.dominated) && n.st.modal() == obj.modal() && obj.base_criteria(&n.st) {
                return Some(true);
            }
        }
        Some(false)
    }

    /// OBRRT_RATE_NOSTART_SKIP, at the start of a round: the live states,
    /// then skipped entries re-checked against the states added since their
    /// check (round-robin, at most `skip_budget` states tried a round): one
    /// a newer state could start is released. A skipped entry counts a
    /// round unattemptable (so one dead everywhere expires).
    fn nostart_recheck(&mut self) {
        let t = Instant::now();
        self.live = (0..self.tree.nodes.len() as u32)
            .filter(|&i| {
                let n = &self.tree.nodes[i as usize];
                !n.retired && !(self.cfg.rrt_star && n.dominated) && n.st.alive()
            })
            .collect();
        self.live_upto = self.tree.nodes.len() as u32;
        let skipped: Vec<usize> = (0..self.pool.entries.len()).filter(|&i| self.pool.entries[i].ns_upto > 0).collect();
        let mut budget = self.cfg.rate.skip_budget;
        let n = skipped.len();
        let start = if n == 0 { 0 } else { self.skip_rr % n };
        let mut released = 0;
        for j in 0..n {
            let i = skipped[(start + j) % n];
            self.pool.entries[i].unattemptable += 1;
            if budget == 0 {
                continue;
            }
            let obj = self.pool.entries[i].obj.clone();
            match self.any_start(obj.as_ref(), self.pool.entries[i].ns_upto, &mut budget) {
                Some(true) => {
                    self.pool.entries[i].ns_upto = 0;
                    released += 1;
                }
                Some(false) => self.pool.entries[i].ns_upto = self.live_upto.max(1),
                None => self.skip_rr = start + j,
            }
        }
        let r = &mut self.pool.rate;
        r.released += released;
        r.skipped = n - released as usize;
        r.recheck_secs += t.elapsed().as_secs_f64();
    }

    /// Chains more objectives after `first` (step 3 again, for where the
    /// previous objective is expected to end): with probability `chain_prob`
    /// each, from GA objectives in the pool that are eligible there and whose
    /// target is within `chain_reach`. The kind is picked first, as in
    /// `sample_entry`, so many-entry kinds don't crowd the rest out.
    fn extend_chain(&mut self, entry: usize, first: Arc<dyn Objective>, start: u32) -> Vec<(usize, Arc<dyn Objective>)> {
        if self.cfg.chain_route {
            return self.extend_route(entry, first);
        }
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

    /// Route chains (`Config::chain_route`): after `first`, the objective the
    /// best routes took next where they met it, if any (each follow-on with
    /// probability `chain_follow`), up to `max_chain` stages. Nothing is
    /// stacked that the routes didn't do, and nothing is learned from how
    /// chained attempts fared.
    fn extend_route(&mut self, entry: usize, first: Arc<dyn Objective>) -> Vec<(usize, Arc<dyn Objective>)> {
        let mut stages = vec![(entry, first)];
        if !matches!(stages[0].1.budget(), Budget::Ga { .. }) {
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
                !keys.contains(k) && !o.modal() && matches!(o.budget(), Budget::Ga { .. }) && o.exists_in(st) && o.base_criteria(st)
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
                    && (e.obj.success(&pair).is_some() || (same_zone && !e.obj.exists_in(end)))
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
        // OBRRT_POLISH: a polish plan's states stay loadable (pinned).
        self.tree.pin_next = plan.polish.is_some();
        let mut polish_end: Option<(u32, bool)> = None;
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
                if self.cfg.start_wear || self.cfg.decay_route || self.cfg.entry_bandit || self.cfg.sampler_rate {
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
                    if let Some(m) = seg.model.clone() {
                        self.tree.nodes[id as usize].stats.model = Some(m);
                    }
                    if self.cfg.chain_route {
                        let n = &mut self.tree.nodes[id as usize];
                        n.made_by = Some(obj.clone());
                        n.met = seg.success;
                    }
                    if self.cfg.new_eligible {
                        self.ne_note(id, parent);
                    }
                    if self.cfg.earliest {
                        self.ea_note(id);
                    }
                    // Best game time per milestone, for the anytime curve.
                    if let Some(old) = self.diag.note_best(progress, zone_now, frame) {
                        self.route_credit(id, old);
                    }
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
                    if let Some(rr) = self.rr.as_mut() {
                        rr.note(id, &self.tree.nodes[id as usize].st, &c, &self.tree, &self.hot, self.cfg.reroute_margin, 0);
                    }
                    added = Some(id);
                    advanced = adv;
                    parent = id;
                    if i == 0 && plan.polish.is_some() {
                        polish_end = Some((id, seg.success));
                    }
                    if i == 0 && self.near.on.is_some() {
                        let rank = if seg.success { (frame - seg_start.frame) as f64 } else { obj.base_heuristic(&self.tree.nodes[id as usize].st) };
                        self.near.note(obj.key(), plan.start, id, seg.success, rank);
                    }
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
                if self.cfg.sampler_rate {
                    // Rate sampler: credited below, no decay.
                    e.successes += seg.success as u32;
                } else if self.cfg.entry_bandit {
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
                if recover && !self.cfg.entry_bandit && !self.cfg.sampler_rate && e.decay < 0.5 {
                    e.decay = 0.5;
                    e.recoveries -= 1;
                    self.diag.kind(kind).route_recoveries += 1;
                }
            }
            let end_frame = next_start.as_ref().map(|s| s.frame);
            if self.cfg.sampler_rate {
                // Improvement: a new state, the first success, frames saved
                // on its soonest success (1 per second, at most 5), or a
                // kept end closer (or sooner) than any before (0.5).
                let p = self.cfg.rate;
                let e = &mut self.pool.entries[entry];
                let (mut imp, mut first, mut sooner, mut closer) = (0.0, 0.0, 0.0, 0.0);
                let novel_i = if novel { p.novel } else { 0.0 };
                if let (true, Some(f)) = (seg.success, end_frame) {
                    if e.best_hit == u32::MAX {
                        first = 1.0;
                    } else if f + 10 <= e.best_hit {
                        sooner = ((e.best_hit - f) as f64 / 30.0).min(5.0);
                    }
                    e.best_hit = e.best_hit.min(f);
                } else if added.is_some() && improves(&self.cfg, end_k, e.best_end) {
                    closer = 0.5;
                }
                if added.is_some() {
                    e.best_end = (e.best_end.0.min(end_k.0), match (e.best_end.1, end_k.1) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    });
                }
                imp += novel_i + first + sooner + closer;
                let r = &mut self.pool.rate;
                r.imp_novel += novel_i;
                r.imp_first += first;
                r.imp_sooner += sooner;
                r.imp_closer += closer;
                self.pool.rate_note(entry, imp, share as f64, self.total_frames, &p, false);
            }
            if self.cfg.chain_route && i > 0 {
                let rc = &mut self.diag.route_chain;
                rc.tried += 1;
                if seg.success {
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
            if (self.cfg.chain_route || self.cfg.sampler_rate) && added.is_some() {
                if let Some(end) = next_start.as_ref() {
                    let hits = self.arrivals(plan, &seg_start, end);
                    let f = end.frame;
                    for j in hits {
                        let k = self.pool.entries[j].obj.kind();
                        self.diag.route_chain.arrivals += 1;
                        *self.diag.route_chain.arrivals_by_kind.entry(k).or_default() += 1;
                        let e = &mut self.pool.entries[j];
                        e.successes += 1;
                        if self.cfg.sampler_rate {
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
                                self.pool.rate_note(j, imp, 0.0, self.total_frames, &p, false);
                            }
                        } else if !self.cfg.entry_bandit && novel {
                            // Met for the first time from a new state: as a
                            // stage's novel success would.
                            e.decay = 1.0;
                        }
                    }
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
                for e in self.pool.entries.iter_mut().filter(|_| !cfg.entry_bandit && !cfg.sampler_rate) {
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
        self.tree.pin_next = false;
        if let Some(job) = plan.polish {
            self.polish_result(job, polish_end);
        }
        if let Some(d) = dense {
            self.merge_dense(plan, d)?;
        }
        if self.cfg.sampler_rate {
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
        } else {
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
                            self.bases.push(packed.apply(&self.bases)?.into_bytes());
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
            if self.cfg.chain_route || self.cfg.rrt_star {
                self.tree.nodes[id as usize].made_by = Some(obj.clone());
            }
            if ne {
                self.ne_note(id, parent);
            }
            if self.cfg.earliest {
                self.ea_note(id);
            }
            if let Some(old) = self.diag.note_best(progress, zone, frame) {
                self.route_credit(id, old);
            }
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
            if let Some(rr) = self.rr.as_mut() {
                rr.note(id, &self.tree.nodes[id as usize].st, &cc, &self.tree, &self.hot, self.cfg.reroute_margin, 0);
            }
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

    /// OBRRT_SPARSE_CHECK: rebuilds up to `n` live save-less nodes (spread
    /// over the ids) and compares each with the state recorded when it was
    /// added. Returns (checked, mismatched).
    fn check_rebuilds(&mut self, n: usize) -> Result<(usize, usize)> {
        let ids: Vec<u32> = (1..self.tree.nodes.len() as u32)
            .filter(|&i| !self.tree.nodes[i as usize].retired && !self.tree.has_save(i) && self.tree.can_anchor(i))
            .collect();
        if ids.is_empty() || n == 0 {
            return Ok((0, 0));
        }
        let step = (ids.len() as f64 / n as f64).max(1.0);
        let pick: Vec<u32> = (0..n.min(ids.len())).map(|k| ids[((k as f64 * step) as usize).min(ids.len() - 1)]).collect();
        let w = self.workers[0].get_mut().unwrap();
        let mut bad = 0;
        for &i in &pick {
            self.tree.load_into(i, &mut w.base, &self.bases)?;
            let rec = &self.tree.nodes[i as usize].st;
            let st = read_state(&w.base, rec.frame, rec.objects.is_some());
            if !same_state(rec, &st) {
                bad += 1;
                if bad <= 5 {
                    eprintln!("sparse check: node {i} rebuilt differently: pos {:?} vs {:?}, action {:#x} vs {:#x}", rec.pos, st.pos, rec.action, st.action);
                }
            }
        }
        Ok((pick.len(), bad))
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
        if let Some(single) = single.filter(|_| multi.attempts >= 20 && self.cfg.chain_fixed.is_none() && !self.cfg.chain_route) {
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
        if let Some(pl) = &self.polish {
            summary["polish"] = serde_json::to_value(&pl.stats).unwrap_or_default();
            summary["polish"]["route_end"] = serde_json::json!(pl.end);
            summary["polish"]["pinned"] = serde_json::json!(self.tree.pinned.len());
            println!("{}", pl.line(&self.tree));
        }
        if let Some(rr) = &self.rr {
            summary["reroute"] = rr.summary();
            let st = &rr.stats;
            let extra = self.rb.as_ref().map_or(String::new(), |rb| format!("; {}", crate::replay::ReplayBudget::kind_line(&rb.stats.grafts)));
            println!("   reroute: {} queued ({} waiting), {} tried, {} ok, {} diverged, {} not faster, {} dup, {} died; {} frames replayed, {} saved, depth {}{extra}",
                st.queued, rr.queue_len(), st.tried, st.ok, st.diverged, st.not_faster, st.duplicate, st.died, st.frames, st.frames_saved, st.max_depth);
        }
        if let Some(zs) = &self.zs {
            summary["zone_share"] = zs.summary();
            let extra = self.rb.as_ref().map_or(String::new(), |rb| format!("; {}", crate::replay::ReplayBudget::kind_line(&rb.stats.shares)));
            println!("   {}{extra}", zs.line());
        }
        if let Some(rb) = &self.rb {
            summary["replay_budget"] = rb.summary();
            println!("   {}", rb.line());
        }
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
        if let Some(sp) = self.spick.as_mut() {
            summary["pick"] = sp.summary();
            println!("   {}", sp.line());
        }
        if self.near.on.is_some() {
            let s = &self.near.stats;
            println!("   near seed: recorded {} asked {} no record {} skipped {} seeded {} (from successes {}) seed met {} successes {}", s.recorded, s.asked, s.no_record, s.skipped_p, s.seeded, s.seeded_success, s.seed_met, s.successes);
            summary["near_seed"] = serde_json::json!(s);
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
            println!("   timing: pre {:.1}s plan {:.1}s run {:.1}s merge {:.1}s post {:.1}s write {:.1}s; workers load {:.1}s attempt {:.1}s",
                d[0], self.diag_plan_secs.0, d[1], self.diag_plan_secs.1, d[2], d[3], d[4], d[5]);
        }
        if crate::movegenome::enabled() {
            println!("   {}", crate::movegenome::stats_line());
        }
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
                "   rate sampler: weight p10 {:.2} p90 {:.2}; kinds {} rate/median min {:.2} median {:.2} max {:.2}; picks {:.1}s started {:.1}s no start (sampling {:.1}s), plans none {}; skip: no-start global {} kin {}, skipped {} released {} recheck {:.1}s; route credit {:.1}",
                r.p10, r.p90, r.kinds, r.kind_min, r.kind_p50, r.kind_max, r.pick_secs, r.nostart_secs, r.sample_secs, r.plans_none,
                r.nostart_global, r.nostart_kin, r.skipped, r.released, r.recheck_secs, r.imp_route
            );
            summary["rate"] = serde_json::json!({
                "entries": r, "starts": {"n": m.len(), "min": q(0.0), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "max": q(1.0), "at_floor": node_floor},
            });
            if self.cfg.rate.hier {
                // Each kind's rate over the median entry's, and its frames.
                let med = r.median.max(1e-12);
                let kinds: serde_json::Map<String, serde_json::Value> =
                    self.pool.kind_rate.iter().map(|(k, v)| (k.to_string(), serde_json::json!({"rel": v.r / med, "frames": v.f, "imp": v.i}))).collect();
                summary["rate"]["kinds"] = serde_json::Value::Object(kinds);
            }
        }
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
        if crate::movegenome::enabled() {
            summary["move_ga"] = crate::movegenome::stats_json();
        }
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
            let mut replay = |from: Option<u32>, bases: &[Vec<u8>]| -> Result<(St, SaveState)> {
                match from {
                    Some(p) => {
                        tree.load_into(p, &mut worker.base, bases)?;
                    }
                    None => worker.base.load_state(&Delta::default().apply(bases)?)?,
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

    /// Replays each graft's child inputs from its faster start in parallel, then
    /// merges the results in order (see `reroute`).
    /// Replay jobs, in parallel, then merged in order: reroute grafts first
    /// (see `reroute`), then zone-share jobs (see `share`).
    /// `checked`: the jobs' starts were checked loadable when taken (replay
    /// budget), so a retired one is rebuilt from its anchor, not skipped.
    fn run_replays(&mut self, grafts: &[crate::reroute::Graft], shares: &[crate::share::ShareJob], checked: bool) -> Result<()> {
        // A share job's start may have been retired by this round's merges.
        let mut share_jobs: Vec<(crate::share::ShareJob, Vec<u32>)> = Vec::new();
        if let Some(zs) = self.zs.as_mut() {
            for j in shares {
                if !checked && j.from != 0 && self.tree.nodes[j.from as usize].retired {
                    zs.stats.stale += 1;
                    continue;
                }
                share_jobs.push((*j, zs.path(&self.tree, j.donor)));
            }
        }
        // So may a graft's faster start: its save is then freed (the
        // default delta, the root's state), so it is skipped as stale.
        let grafts: Vec<crate::reroute::Graft> = match self.rr.as_mut() {
            Some(rr) => grafts
                .iter()
                .copied()
                .filter(|g| {
                    let ok = checked || g.from == 0 || !self.tree.nodes[g.from as usize].retired || self.tree.is_pinned(g.from);
                    rr.stats.stale += !ok as u64;
                    ok
                })
                .collect(),
            None => grafts.to_vec(),
        };
        let grafts = grafts.as_slice();
        let jobs: Vec<(u32, &[u32], bool)> = (grafts.iter().map(|g| (g.from, std::slice::from_ref(&g.child), false)))
            .chain(share_jobs.iter().map(|(j, p)| (j.from, p.as_slice(), true)))
            .collect();
        let (tree, bases, workers) = (&self.tree, &self.bases, &self.workers);
        let mut outs: Vec<Result<Replayed>> = jobs
            .par_iter()
            .map(|&(from, edges, stay)| {
                let mut w = loop {
                    if let Some(w) = workers.iter().find_map(|w| w.try_lock().ok()) {
                        break w;
                    }
                    std::thread::yield_now();
                };
                replay_edges(&mut w.base, tree, bases, from, edges, stay)
            })
            .collect();
        let share_outs = outs.split_off(grafts.len());
        for (g, r) in grafts.iter().zip(outs) {
            let r = r.map(|mut r| if r.end == ReplayEnd::Died { None } else { r.states.pop() });
            self.merge_graft(*g, r)?;
        }
        for ((j, path), r) in share_jobs.into_iter().zip(share_outs) {
            self.merge_share(j, &path, r)?;
        }
        Ok(())
    }

    /// Adds a zone-share job's boundary states as a chain under its start:
    /// each that has no earlier state at its place (the duplicate check, in
    /// both dedupe modes) is added as merge adds a kept state, its parent the
    /// previous one added; one that does is skipped and its inputs carried
    /// on to the next added state (as merge carries a segment's).
    fn merge_share(&mut self, j: crate::share::ShareJob, path: &[u32], r: Result<Replayed>) -> Result<()> {
        let zs = self.zs.as_mut().expect("share jobs only with zone share on");
        zs.stats.run += 1;
        let r = match r {
            Ok(r) => r,
            Err(_) => {
                zs.stats.traps += 1;
                // Charged as if replayed whole: the trap's frame is unknown.
                let n: u64 = path.iter().map(|&e| self.tree.nodes[e as usize].inputs.len() as u64).sum();
                zs.stats.frames += n;
                self.total_frames += n;
                return Ok(());
            }
        };
        zs.stats.frames += r.frames;
        zs.stats.boundaries += r.states.len() as u64;
        match r.end {
            ReplayEnd::Died => zs.stats.died += 1,
            ReplayEnd::LeftZone => zs.stats.left_zone += 1,
            ReplayEnd::Done => {}
        }
        self.total_frames += r.frames;
        let mut parent = j.from;
        let mut carry: Vec<GamePad> = Vec::new();
        let mut chain = 0u64;
        for (k, (st, packed)) in r.states.into_iter().enumerate() {
            let e = path[k];
            let (via, donor_cell, donor_frame) = {
                let n = &self.tree.nodes[e as usize];
                carry.extend_from_slice(&n.inputs);
                (n.via, cell(&n.st), n.st.frame)
            };
            let c = cell(&st);
            let earlier = self.cells.get(&c);
            if earlier.is_some_and(|ids| ids.iter().any(|&i| self.tree.nodes[i as usize].st.frame <= st.frame)) {
                self.zs.as_mut().unwrap().stats.duplicate += 1;
                continue;
            }
            let sooner = earlier.is_some_and(|ids| !ids.is_empty());
            let zone = st.zone();
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
            let new_zone = self.zones.insert(zone);
            if new_zone {
                self.diag.kind(via).new_zones += 1;
                println!("new zone {zone:?} via zone share of {via} at node {}", self.tree.nodes.len());
            }
            let (modal, frame, progress) = (st.modal(), st.frame, st.progress());
            let (id, adv) = self.tree.add(parent, std::mem::take(&mut carry), st, delta, via);
            if self.cfg.new_eligible {
                self.ne_note(id, parent);
            }
            if self.cfg.earliest {
                self.ea_note(id);
            }
            if let Some(old) = self.diag.note_best(progress, zone, frame) {
                self.route_credit(id, old);
            }
            if modal {
                self.modal_queue.push(id);
            }
            if self.cfg.soft_dedup {
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
            if let Some(rr) = self.rr.as_mut() {
                rr.note(id, &self.tree.nodes[id as usize].st, &c, &self.tree, &self.hot, self.cfg.reroute_margin, 0);
            }
            let zs = self.zs.as_mut().unwrap();
            zs.stats.added += 1;
            zs.stats.sooner += sooner as u64;
            zs.stats.new_zones += new_zone as u64;
            if c == donor_cell && frame < donor_frame {
                zs.stats.matched_sooner += 1;
                zs.stats.matched_frames_saved += (donor_frame - frame) as u64;
            }
            zs.note_added(id);
            // OBRRT_KIN_UNIT=attempt: the chain is one generation (its
            // last node, set below).
            self.tree.gen_step[id as usize] = false;
            chain += 1;
            parent = id;
            if adv {
                let (keys, stars) = self.tree.frontier_progress;
                let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
                println!("FRONTIER keys {keys} stars {stars} at node {id} via zone share of {via}: {f}");
            }
        }
        if chain > 0 {
            self.tree.gen_step[parent as usize] = true;
        }
        let zs = self.zs.as_mut().unwrap();
        if chain > 0 {
            zs.stats.jobs_ok += 1;
            zs.stats.max_chain = zs.stats.max_chain.max(chain);
        }
        Ok(())
    }

    /// Keeps a graft's end if it is the same as the child (same key) and at
    /// least the margin sooner, as a node under the faster start; its own
    /// children then follow (`Reroute::note`).
    fn merge_graft(&mut self, g: crate::reroute::Graft, r: Result<Option<(St, Delta)>>) -> Result<()> {
        let child = &self.tree.nodes[g.child as usize];
        let (n, child_frame, kind) = (child.inputs.len() as u64, child.st.frame, child.via);
        let inputs = child.inputs.clone();
        self.total_frames += n;
        let rr = self.rr.as_mut().expect("grafts only with reroute on");
        rr.stats.tried += 1;
        rr.stats.frames += n;
        let (st, packed) = match r {
            Ok(Some(x)) => x,
            Ok(None) => {
                rr.stats.died += 1;
                self.route_diverged(g);
                return Ok(());
            }
            Err(_) => {
                rr.stats.traps += 1;
                self.route_diverged(g);
                return Ok(());
            }
        };
        let c = cell(&st);
        if !rr.accept(g, &st, &c, &self.tree, self.cfg.reroute_margin) {
            self.route_diverged(g);
            return Ok(());
        }
        let dup = self.cells.get(&c).is_some_and(|ids| ids.iter().any(|&i| self.tree.nodes[i as usize].st.frame <= st.frame));
        if dup && !self.cfg.soft_dedup {
            rr.stats.duplicate += 1;
            return Ok(());
        }
        // Added as merge adds a kept state.
        let zone = st.zone();
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
            println!("new zone {zone:?} via reroute of {kind} at node {}", self.tree.nodes.len());
        }
        let (modal, frame, progress) = (st.modal(), st.frame, st.progress());
        let route = g.kind == crate::reroute::KIND_ROUTE;
        self.tree.pin_next = route;
        let (id, _adv) = self.tree.add(g.from, inputs, st, delta, kind);
        self.tree.pin_next = false;
        // OBRRT_KIN_UNIT=attempt: a graft is the generation its child was.
        self.tree.gen_step[id as usize] = self.tree.gen_step[g.child as usize];
        if self.cfg.chain_route {
            // The same inputs from a faster start: the same objective.
            let (m, met) = (self.tree.nodes[g.child as usize].made_by.clone(), self.tree.nodes[g.child as usize].met);
            let n = &mut self.tree.nodes[id as usize];
            n.made_by = m;
            n.met = met;
        }
        if self.cfg.new_eligible {
            self.ne_note(id, g.from);
        }
        if self.cfg.earliest {
            self.ea_note(id);
        }
        if let Some(old) = self.diag.note_best(progress, zone, frame) {
            self.route_credit(id, old);
        }
        if modal {
            self.modal_queue.push(id);
        }
        if self.cfg.soft_dedup {
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
        let rr = self.rr.as_mut().unwrap();
        rr.stats.ok += 1;
        rr.stats.frames_saved += child_frame.saturating_sub(frame) as u64;
        rr.stats.max_depth = rr.stats.max_depth.max(g.depth + 1);
        rr.note(id, &self.tree.nodes[id as usize].st, &c, &self.tree, &self.hot, self.cfg.reroute_margin, g.depth + 1);
        if route {
            // A kept route edge: the next one from here, or the chain is done.
            let next = (g.child != g.leg).then(|| crate::legs::below(&self.tree, g.child, g.leg)).flatten();
            if let Some(next) = next {
                rr.push(crate::reroute::Graft { from: id, child: next, depth: g.depth + 1, kind: crate::reroute::KIND_ROUTE, leg: g.leg }, true);
            }
            if let Some(pl) = self.polish.as_mut() {
                pl.pinned(id);
                pl.stats.edges_kept += 1;
                if g.child == g.leg {
                    pl.stats.done += 1;
                    pl.stats.done_frames += child_frame.saturating_sub(frame) as u64;
                    let f = self.write_m64(id, "polish-best")?;
                    println!("POLISH chain done at node {id}: frame {frame}, {} sooner than node {}: {f}", child_frame.saturating_sub(frame), g.leg);
                }
            }
        }
        Ok(())
    }

    /// OBRRT_POLISH: route edge `g` was not kept (diverged, slower, died):
    /// a repair attempt at the next route milestone from its start.
    fn route_diverged(&mut self, g: crate::reroute::Graft) {
        if g.kind != crate::reroute::KIND_ROUTE {
            return;
        }
        let Some(pl) = self.polish.as_mut() else { return };
        pl.stats.diverged += 1;
        let Some(anchor) = self.tree.nodes[g.child as usize].parent else { return };
        if let Some(t) = crate::polish::Polish::next_milestone(&self.tree, g.child, g.leg) {
            pl.push_repair(crate::polish::Job { from: g.from, anchor, target: t, end: g.leg, repair: true });
        }
    }

    /// Replay budget: after the round's plans (frames since `round_t0`, plus
    /// their start rebuilds), replay jobs in waves until their frames reach
    /// the budget's fraction of the plans' (see `replay`).
    fn budget_replays(&mut self, round: u64, round_t0: u64) -> Result<()> {
        let rebuilt = |t: &Tree| t.sparse.as_ref().map_or(0, |sp| sp.rebuild_frames.load(std::sync::atomic::Ordering::Relaxed));
        let r0 = rebuilt(&self.tree);
        let plan_frames = self.total_frames - round_t0 + (r0 - self.rebuild_counted);
        if let Some(zs) = self.zs.as_mut() {
            zs.round(&self.tree, &self.hot);
        }
        let budget = (plan_frames as f64 * self.rb.as_ref().unwrap().fraction) as u64;
        let t0 = self.total_frames;
        let mut spent = 0u64;
        let (mut waves, mut ng, mut ns) = (0u32, 0u32, 0u32);
        while spent < budget && (waves as usize) < crate::replay::MAX_WAVES {
            let rb = self.rb.as_mut().unwrap();
            let (grafts, shares) = rb.select(self.rr.as_mut(), self.zs.as_mut(), &self.tree, &self.hot, budget - spent);
            if grafts.is_empty() && shares.is_empty() {
                break;
            }
            waves += 1;
            ng += grafts.len() as u32;
            ns += shares.len() as u32;
            self.run_replays(&grafts, &shares, true)?;
            spent = self.total_frames - t0 + (rebuilt(&self.tree) - r0);
        }
        let queue = self.rr.as_ref().map_or(0, |r| r.queue_len()) + self.zs.as_ref().map_or(0, |z| z.queue_len());
        self.rb.as_mut().unwrap().end_round(round, plan_frames, spent, waves, ng, ns, queue as u64);
        Ok(())
    }

    pub fn run(&mut self) -> Result<()> {
        let begin = Instant::now();
        let mut last_report = Instant::now();
        let mut last_tune = self.total_frames;
        let mut round = 0u64;
        self.tree.pin_routes = self.polish.is_some();
        // Stop on wall-clock time, or on emulated frames (max_frames) so runs
        // compare equally whatever the thread count or CPU sharing.
        while begin.elapsed() < Duration::from_secs_f64(self.cfg.minutes * 60.0)
            && self.cfg.max_frames.is_none_or(|m| self.total_frames < m)
        {
            round += 1;
            let pre_t = Instant::now();
            if self.cfg.sampler_rate {
                let p = self.cfg.rate;
                if p.nostart_skip {
                    self.nostart_recheck();
                }
                self.pool.rate_refresh(self.total_frames, &p);
            } else if self.cfg.entry_bandit {
                self.pool.redraw(&mut self.rng);
            }
            self.tree.wip_refresh();
            self.index_frontier();
            self.ne.budget = if self.cfg.new_eligible { ((self.cfg.round_plans as f64 * NE_SHARE).ceil() as usize).max(1) } else { 0 };
            if self.cfg.earliest {
                self.ea.budget = ((self.cfg.round_plans as f64 * self.cfg.earliest_share).ceil() as usize).max(1);
                self.ea.round_used.clear();
            }
            self.diag_secs[0] += pre_t.elapsed().as_secs_f64();
            let plan_t = Instant::now();
            // More attempts than workers, so a slow attempt doesn't leave the
            // others idle: each takes whichever game is free.
            // A fixed number of attempts per round, whatever the thread
            // count: the tree only updates between rounds, so the round size
            // shapes the search (how stale planning is); 64 = 16 threads × 4.
            self.bc.begin_round(self.cfg.round_plans);
            obrrt_core::mover::publish(); // movers see caches observed before this round only
            // Grafts take up to reroute_share of the round's slots (none when off).
            let grafts = match self.rr.as_mut() {
                Some(rr) => {
                    // Key the nodes made before the engine ran (the root, a resumed tree).
                    for id in rr.keys.len()..self.tree.nodes.len() {
                        let st = &self.tree.nodes[id].st;
                        rr.note(id as u32, st, &cell(st), &self.tree, &self.hot, self.cfg.reroute_margin, 0);
                    }
                    if self.rb.is_some() || rr.route_only {
                        Vec::new()
                    } else {
                        rr.take((self.cfg.round_plans as f64 * self.cfg.reroute_share).ceil() as usize)
                    }
                }
                None => Vec::new(),
            };
            // Zone-share jobs take up to zone_share_share of them (none when off).
            let shares = match self.zs.as_mut() {
                // Replay budget: jobs are noted and taken after the plans.
                Some(_) if self.rb.is_some() => Vec::new(),
                Some(zs) => {
                    zs.round(&self.tree, &self.hot);
                    let n = (self.cfg.round_plans as f64 * self.cfg.zone_share_share).ceil() as usize;
                    zs.take(n.min(self.cfg.round_plans.saturating_sub(grafts.len())))
                }
                None => Vec::new(),
            };
            // sample_entry's tables get this round's weights (kept between
            // rounds only for their kinds and members).
            if let Some(idx) = self.pool.sidx.as_mut() {
                idx.reweigh(&self.pool.entries);
            }
            // Polish takes up to its share of the plans, after the others.
            let n_polish = match self.polish.as_mut() {
                Some(pl) => {
                    pl.refresh(&mut self.tree, round);
                    ((self.cfg.round_plans as f64 * pl.share).ceil() as usize).min(self.cfg.round_plans.saturating_sub(grafts.len() + shares.len()))
                }
                None => 0,
            };
            let mut plans: Vec<Plan> = (0..self.cfg.round_plans.saturating_sub(grafts.len() + shares.len() + n_polish))
                .filter_map(|_| self.plan().inspect(|_| self.bc.made += 1))
                .collect();
            for _ in 0..n_polish {
                if let Some(p) = self.plan_polish() {
                    plans.push(p);
                }
            }
            if self.near.on.is_some() {
                for (i, p) in plans.iter_mut().enumerate() {
                    let o = &p.stages[0].1;
                    if self.bc.is_pending(i) || p.rewire.is_some() || p.polish.is_some() || o.modal() || !matches!(o.budget(), Budget::Ga { .. }) {
                        continue;
                    }
                    p.seed = self.near.pick(&self.tree, &o.key(), p.start);
                }
            }
            self.diag_plan_secs.0 += plan_t.elapsed().as_secs_f64();
            if plans.is_empty() && grafts.is_empty() && shares.is_empty() {
                continue;
            }
            let bases = &self.bases;
            let tree = &self.tree;
            let seeds: Vec<u64> = plans.iter().map(|_| self.rng.random()).collect();
            let workers = &self.workers;
            let bc = &self.bc;
            let params = crate::ga::GaParams { scale: self.cfg.ga_scale, fitness: self.cfg.fitness, new_ops: self.cfg.ga_new_ops, pink: self.cfg.ga_pink, per_objective_budget: self.cfg.ga_per_objective_budget };
            let dense = self.dense;
            let replay_max = self.tree.sparse.as_ref().map_or(0, |sp| sp.cap);
            let round_t0 = self.total_frames;
            let run_t = Instant::now();
            let (load_ns, att_ns) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
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
                    // The start's state into the worker's base game: its save,
                    // or (sparse saves) its anchor's and the inputs since.
                    let lt = Instant::now();
                    tree.load_into(plan.start, &mut w.base, bases)?;
                    load_ns.fetch_add(lt.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
                    let mut rng = SmallRng::seed_from_u64(*seed);
                    let stages: Vec<&dyn Objective> = plan.stages.iter().map(|(_, o)| o.as_ref()).collect();
                    let base = tree.base_of(plan.start);
                    let pack = |s: &SaveState| Delta::new(base, &bases[base as usize], s);
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
                    let seed_in = bc.seed(i).or(plan.seed.as_deref());
                    crate::ga::set_repair_targets(plan.targets.clone());
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
            for (i, (plan, r)) in plans.iter().zip(results).enumerate() {
                let bc = self.bc.is_pending(i).then(|| (self.tree.nodes.len(), r.as_ref().ok().map(crate::backchain::outcome)));
                if plan.seed.is_some() {
                    if let Ok(att) = r.as_ref() {
                        self.near.stats.seed_met += att.ga.and_then(|g| g.seed).is_some_and(|s| s.0) as u64;
                        self.near.stats.successes += att.segments.first().is_some_and(|s| s.success) as u64;
                    }
                }
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
                            if self.cfg.sampler_rate && *entry < self.pool.entries.len() {
                                let p = self.cfg.rate;
                                self.pool.rate_note(*entry, 0.0, p.nostart_frames, self.total_frames, &p, true);
                            } else if let Some(e) = self.pool.entries.get_mut(*entry) {
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
            let post_t = Instant::now();
            if !grafts.is_empty() || !shares.is_empty() {
                self.run_replays(&grafts, &shares, false)?;
            }
            if self.rb.is_some() {
                self.budget_replays(round, round_t0)?;
            }
            // Polish: route-graft chains, several edges a round.
            if self.polish.is_some() {
                for _ in 0..POLISH_WAVES {
                    let gs = self.rr.as_mut().map_or(Vec::new(), |rr| rr.take_route(self.cfg.round_plans));
                    if gs.is_empty() {
                        break;
                    }
                    self.run_replays(&gs, &[], false)?;
                }
            }
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
            if self.cfg.soft_dedup && self.mem_used() > self.cfg.mem_budget {
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
            if last_report.elapsed() > self.cfg.report_every {
                last_report = Instant::now();
                self.write_outputs(begin.elapsed().as_secs_f64())?;
                self.diag_secs[3] += last_report.elapsed().as_secs_f64();
            }
        }
        if self.cfg.rebuild_check > 0 && self.tree.sparse.is_some() {
            let (n, bad) = self.check_rebuilds(self.cfg.rebuild_check)?;
            println!("sparse check: rebuilt {n} save-less nodes, {bad} differ from their recorded state");
        }
        self.write_outputs(begin.elapsed().as_secs_f64())?;
        Ok(())
    }
}

/// Whether a rebuilt state is the recorded one (sparse-save check): Mario's
/// fields bit for bit, and each object's identity, position and action.
fn same_state(a: &St, b: &St) -> bool {
    let f = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
    let objs = match (&a.objects, &b.objects) {
        (Some(x), Some(y)) => {
            x.len() == y.len()
                && x.iter().zip(y.iter()).all(|(o, p)| o.slot == p.slot && o.bhv == p.bhv && f(&o.pos, &p.pos) && o.action == p.action && o.health == p.health)
        }
        (None, None) => true,
        _ => false,
    };
    a.frame == b.frame
        && f(&a.pos, &b.pos)
        && f(&a.vel, &b.vel)
        && a.forward_vel.to_bits() == b.forward_vel.to_bits()
        && a.yaw == b.yaw
        && a.action == b.action
        && a.action_timer == b.action_timer
        && a.health == b.health
        && a.coins == b.coins
        && a.stars == b.stars
        && a.cam_yaw == b.cam_yaw
        && a.level == b.level
        && a.area == b.area
        && objs
}
