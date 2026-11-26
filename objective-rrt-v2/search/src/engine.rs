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
    pub fitness: crate::ga::Fitness,
    pub ga_new_ops: bool,
    pub ga_pink: f64,
    /// Per-objective GA budgets (see `GaParams::per_objective_budget`).
    pub ga_per_objective_budget: bool,
    /// RRT* mode (speed): rank starts by estimated arrival time, rewire
    /// (re-attempt the objectives that grew from a state from any state that
    /// reaches its place sooner), and stop starting from slower duplicates.
    pub rrt_star: bool,
    /// Arrival ranking (OBRRT_ARRIVAL=1): starts are ranked, and
    /// improvement (decay resets, wear, route recovery) is judged, by the
    /// estimated arrival (frame + start ETA)/30 in seconds, not by the ETA
    /// alone (v2: every objective has one). No pruning or rewiring (unlike `rrt_star`).
    pub arrival: bool,
    /// Arrival ranking: the reference pace, in units per frame (v2: moot,
    /// ETAs are already at core's PACE; 20 beat 15
    /// and 30 on milestone times in the arrival sweep, RUNS.md).
    pub arrival_pace: f64,
    /// Arrival ranking: the start softmax temperature, in seconds.
    pub arrival_t: f64,
    /// Arrival ranking: seconds sooner that count as an improvement.
    pub arrival_margin: f64,
    /// Arrival ranking: only starts with a start ETA within this many frames
    /// of the closest one are ranked (None: all).
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
    /// Share of plans spent on rewiring while there is rewiring to do.
    pub rewire_share: f64,
    /// Start cost per frame of game time, in start-ETA frames (v2: h is the
    /// ETA in frames, so 1 would weigh time and distance equally; 0.4 is
    /// v1's 0.4·30/400 per base_heuristic unit of distance/400).
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
            round_plans: 64,
            max_frames: None,
            soft_dedup: false,
            mem_budget: 3_000_000_000,
            pick_by_b: true,
            rewire_share: 0.3,
            // 40% of "a frame is worth 30 units": at full weight the start is
            // almost always the single earliest state (1.9 effective starts in
            // a test) and the search stops spreading out.
            time_cost: 0.4,
            rewire_margin: 10,
            new_eligible: obrrt_core::config::current().sampler.new_eligible,
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
    /// OBRRT_PARENT_WINDOW: the selected node, when the attempt starts from
    /// its parent (`start`) with `window` first; it takes the attempt's stats.
    window_b: Option<u32>,
    window: Vec<GamePad>,
    /// OBRRT_ATTEMPT_MODEL: what the model predicted for the first stage.
    mv: Option<crate::attempt_model::Out>,
}

/// The macro goals of OBRRT_WIP_FOCUS (see `index_frontier`).
const MACROS: usize = 5;

/// OBRRT_WIP_FOCUS (share of b draws); levels of wip kept: 2.
fn wip_focus() -> (f64, i64) {
    static S: std::sync::OnceLock<(f64, i64)> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        let p: f64 = obrrt_core::sys::env::var("OBRRT_WIP_FOCUS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        (p.clamp(0.0, 1.0), 2)
    })
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

/// OBRRT_GOAL_WIP_W (default 1500): frames of ETA a unit of work in progress
/// (red coins, chests, boss hits: St::wip) is worth in the goal ranking, so
/// the states furthest along a goal with several steps come first.
fn goal_wip_w() -> f64 {
    static WW: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *WW.get_or_init(|| obrrt_core::sys::env::var("OBRRT_GOAL_WIP_W").ok().and_then(|v| v.parse().ok()).unwrap_or(1500.0))
}

/// OBRRT_PARENT_WINDOW_FRAMES: the copied prefix's minimum length (0: one parent).
fn parent_window_frames() -> usize {
    obrrt_core::config::current().goal_focus.parent_window.frames
}

/// OBRRT_PARENT_WINDOW_POP: the population from which the minimum applies in full.
fn parent_window_pop() -> f64 {
    obrrt_core::config::current().goal_focus.parent_window.pop
}

/// OBRRT_PARENT_WINDOW_HORIZONS: the copied prefix's minimum in GA horizons.
fn parent_window_horizons() -> f64 {
    obrrt_core::config::current().goal_focus.parent_window.horizons
}

/// OBRRT_PARENT_WINDOW (share of attempts), OBRRT_PARENT_WINDOW_MAX (frames).
fn parent_window() -> Option<(f64, usize)> {
    let w = &obrrt_core::config::current().goal_focus.parent_window;
    let share = w.share?;
    (share > 0.0).then_some((share.min(1.0), w.max))
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
/// goal in seconds, game time plus h = `start_eta(obj, s)` (frames; with
/// OBRRT_ETA_LEARN the caller passes it times its kind's c). v2:
/// every objective has one (the ETA is already at the reference pace, so
/// `arrival_pace` is moot). None when off.
fn arrival_est(cfg: &Config, _obj: &dyn Objective, s: &St, h: f64) -> Option<f64> {
    if !cfg.arrival {
        return None;
    }
    // Unreachable (Lost) stays unreachable.
    if !h.is_finite() {
        return Some(h);
    }
    Some((s.frame as f64 + h) / 30.0)
}

/// `s`'s start rank for `obj` given its pure start ETA `h` (with
/// OBRRT_ETA_LEARN, times its kind's c): `h` plus the objective's start bias
/// ([`obrrt_core::start_rank`]); exactly `h` when the bias is 0.
fn rank_of(obj: &dyn Objective, s: &St, h: f64) -> f64 {
    let b = obj.start_bias(s);
    if b == 0.0 { h } else { h + b }
}

/// ETA frames closer a start must be to improve on another ([`improves`]).
const IMPROVE_FRAMES: f64 = 7.0;

/// Whether `new` improves on `old`, each (start ETA, estimated arrival
/// if arrival ranking applies): h lower by [`IMPROVE_FRAMES`], or (arrival ranking) the
/// arrival sooner by `arrival_margin`, alone or either (`arrival_decay_either`).
fn improves(cfg: &Config, new: (f64, Option<f64>), old: (f64, Option<f64>)) -> bool {
    // v2: h is the start ETA in frames; 7 frames is v1's 0.5 (~200 units
    // at distance/400) at the reference pace.
    let closer = new.0 < old.0 - IMPROVE_FRAMES;
    match (new.1, old.1) {
        (Some(a), Some(b)) => {
            let sooner = a < b - cfg.arrival_margin;
            sooner || closer
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
fn replay_edges(game: &mut crate::game::Game, tree: &Tree, bases: &[Vec<u8>], from: u32, edges: &[u32], stay: bool) -> Result<Replayed> {
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
    let fp = st.fingerprint();
    // Off, the last field is exactly the fingerprint as before.
    let fp = if STARSET.load(std::sync::atomic::Ordering::Relaxed) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&(fp, st.progress_set()), &mut h);
        std::hash::Hasher::finish(&h)
    } else {
        fp
    };
    // OBRRT_BLJ_FAST: BLJ setup and chain states (a crouch slide, a long
    // jump or its landing moving backwards) get their own cells by speed
    // and facing, so the tree keeps several per spot (off: unchanged).
    let fp = match blj_mark(st) {
        Some(m) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&(fp, m), &mut h);
            std::hash::Hasher::finish(&h)
        }
        None => fp,
    };
    // OBRRT_CELL_MOTION: coarse motion (see [`motion`]) in the cell too.
    let fp = match motion(st) {
        Some(m) => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&(fp, m), &mut h);
            std::hash::Hasher::finish(&h)
        }
        None => fp,
    };
    (st.level, st.area, st.stars, st.keys, st.coins, c(st.pos[0]), c(st.pos[1]), c(st.pos[2]), modal, fp)
}

/// OBRRT_CELL_MOTION=1 (default off: cells and dense keys unchanged): a
/// coarse motion field, (horizontal speed / 16, facing >> 13), in the
/// duplicate cell ([`cell`]) and the dense key (dense::dkey), so states at
/// one place with a different runway speed or facing (BLJ and other
/// technique setups) are not merged by the hard / soft dedupe.
pub fn motion(st: &St) -> Option<(i32, i32)> {
    if !obrrt_core::config::current().dedup.cell_motion {
        return None;
    }
    let h = st.vel[0].hypot(st.vel[2]);
    Some(((h / 16.0) as i32, (st.yaw as u16 >> 13) as i32))
}

/// OBRRT_BLJ_FAST's cell marker (see [`cell`]): (kind, speed bucket of 25,
/// facing octant) for a crouch slide or a backwards long jump / landing.
fn blj_mark(st: &St) -> Option<(u8, i32, i32)> {
    if !obrrt_core::config::current().objectives.blj_fast {
        return None;
    }
    const CROUCH_SLIDE: u32 = 0x0480_8459;
    const LONG_JUMP: u32 = 0x0300_0888;
    const LONG_JUMP_LAND: u32 = 0x0000_0479;
    let oct = ((st.yaw as i32 + 0x1000).rem_euclid(0x10000)) / 0x2000;
    match st.action {
        CROUCH_SLIDE => Some((1, 0, oct)),
        LONG_JUMP | LONG_JUMP_LAND if st.forward_vel < -1.0 => Some((2, (-st.forward_vel / 25.0) as i32, oct)),
        _ => None,
    }
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
    /// Per [`Objective::diag_tag`]: attempts, successes, added (summary.json).
    tag_stats: BTreeMap<String, [u64; 3]>,
    zones: HashSet<(u16, u8)>,
    /// The frontier grouped by zone, then by 512-unit cell; rebuilt each round.
    b_index: Vec<Vec<Vec<u32>>>,
    /// OBRRT_WIP_FOCUS: per macro goal (red coins, coins toward 100, level
    /// changes, bosses, cap switches), the frontier states furthest along it.
    wip_top: Vec<Vec<u32>>,
    /// OBRRT_GOAL_FOCUS: the goals' best cells; draws per (goal, cell); per
    /// goal (best ETA seen, draws since it last dropped, times it dropped).
    goal_top: Vec<GoalTop>,
    goal_wear: HashMap<(String, GoalCell), u32>,
    goal_stats: HashMap<String, (f64, u32, u32)>,
    /// OBRRT_ATTEMPT_MODEL: attempts so far per (kind, start state) and per start state.
    tried_kind: HashMap<(&'static str, u32), u32>,
    tried_node: HashMap<u32, u32>,
    /// Per b_index zone: its draw weight (all 1).
    zone_weights: Vec<f64>,
    /// The best routes' states (see route_share).
    route: Vec<u32>,
    events: BufWriter<File>,
    total_frames: u64,
    attempts_merged: u64,
    window_attempts: u64,
    window_frames: u64,
    /// Why the walk back stopped: retired, empty, max, zone, control, lost.
    window_stops: [u64; 6],
    /// Merged window attempts; those whose copy survived exactly.
    window_results: [u64; 2],
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
    /// OBRRT_PICK=schema|schema_kin: start selection by schema cells (schemapick.rs).
    spick: Option<crate::schemapick::SchemaPick>,
    /// Sparse saves: rebuild frames already counted into total_frames.
    rebuild_counted: u64,
    /// Best meets per objective key, Beats, faster-only and improvement
    /// counters (see speed.rs).
    speed: crate::speed::Speed,
    /// In-run ETA calibration per kind (OBRRT_ETA_LEARN, see etalearn.rs).
    eta: crate::etalearn::EtaLearn,
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
        let mut tree = Tree::new(root);
        // The root's save: its pages are the base, its perturbation its own.
        tree.nodes[0].save = Delta::new(0, crate::game::plain(&root_save).as_bytes(), &root_save);
        Ok(Engine {
            rr,
            polish: (cfg.polish > 0.0).then(|| crate::polish::Polish::new(cfg.polish)),
            zs,
            rb,
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
            wip_top: Vec::new(),
            goal_top: Vec::new(),
            goal_wear: HashMap::new(),
            goal_stats: HashMap::new(),
            tried_kind: HashMap::new(),
            tried_node: HashMap::new(),
            zone_weights: vec![1.0],
            route: Vec::new(),
            events,
            total_frames: 0,
            attempts_merged: 0,
            window_attempts: 0,
            window_frames: 0,
            window_stops: [0; 6],
            window_results: [0; 2],
            modal_queue: Vec::new(),
            emulator_errors: 0,
            modal_tried: HashSet::new(),
            rewire_hot: Default::default(),
            rewire_cold: Default::default(),
            hot: HashSet::new(),
            ne: NewEligible { rng: Some(SmallRng::seed_from_u64(cfg.seed ^ 0x6e65_7765_6c69_6769)), ..Default::default() },
            diag_plan_secs: (0.0, 0.0),
            diag_secs: [0.0; 6],
            diag_index_secs: 0.0,
            diag_pre: [0.0; 11],
            route_met: HashMap::new(),
            node_rate: (0.0, 0.0, 0),
            live: Vec::new(),
            live_upto: 0,
            skip_rr: 0,
            dense: crate::dense::settings().filter(|d| d.modes.any()),
            dstats: Default::default(),
            near: crate::near::NearSeed::new(cfg.seed),
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
            spick: crate::schemapick::from_env()?,
            rebuild_counted: 0,
            speed: Default::default(),
            eta: crate::etalearn::EtaLearn::from_env(),
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
        if obrrt_core::config::current().research.sample_check {
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
                        && { let h = obrrt_core::start_eta(o, b_st); h.is_finite() && h <= diag.kinds.get(o.kind()).map_or(f64::INFINITY, |k| k.cutoff) }
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

    /// Arrival ranking: `s`'s estimated arrival at `obj`'s goal in seconds
    /// (game time plus h = `start_eta(obj, s)`). None when off.
    fn arrival_of(&self, obj: &dyn Objective, s: &St, h: f64) -> Option<f64> {
        arrival_est(&self.cfg, obj, s, h)
    }

    /// What improvement on a start is judged by (see [`improves`]): the
    /// start rank (`start_rank`: the ETA plus the objective's start bias),
    /// and the estimated arrival (seconds, from the pure ETA) with arrival
    /// ranking.
    fn keys_of(&self, obj: &dyn Objective, s: &St) -> (f64, Option<f64>) {
        // OBRRT_ETA_LEARN: the calibrated ETA (× 1.0 exactly when off).
        let h = obrrt_core::start_eta(obj, s) * self.eta.c(obj.kind());
        (rank_of(obj, s, h), self.arrival_of(obj, s, h))
    }

    /// Step 4: the eligible starts for `obj` given `b`, sampled by softmax
    /// over `start_eta` (or the estimated arrival, `arrival`) × u(s) ×
    /// the state's back-off.
    fn pick_start(&mut self, entry: usize, obj: &dyn Objective, b: u32) -> Option<u32> {
        let arrival = self.cfg.arrival
            && (self.cfg.arrival_mix >= 1.0 || self.rng.random::<f64>() < self.cfg.arrival_mix);
        let cutoff = self.diag.kind(obj.kind()).cutoff;
        // OBRRT_ETA_LEARN: ranks and arrivals by c × ETA; the cutoff stays on
        // the raw ETA (it is learned on raw start ETAs).
        let eta_c = self.eta.c(obj.kind());
        let k = obj.kinship();
        let k = if self.cfg.kinship_mult == 1.0 { k } else { ((k as f64) * self.cfg.kinship_mult).round().max(0.0) as u32 };
        if let Some(sp) = self.spick.as_mut() {
            let kin = (sp.mode == crate::schemapick::Mode::SchemaKin).then(|| self.tree.kin_set(b, k));
            let cx = crate::schemapick::Ctx { obj, entry, b, cutoff, eta_c, rrt_star: self.cfg.rrt_star, kin, epoch: (self.tree.nodes.len(), self.total_frames) };
            return sp.pick(&self.tree, cx, &mut self.rng);
        }
        let b_st = &self.tree.nodes[b as usize].st;
        let (b_keys, b_stars, b_lives) = (b_st.keys, b_st.stars, b_st.lives);
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
                || s.modal() != obj.modal()
                || !obj.base_criteria(s)
            {
                continue;
            }
            let h = obrrt_core::start_eta(obj, s);
            if !h.is_finite() || h > cutoff {
                continue;
            }
            let h = h * eta_c;
            // RRT*: the cost so far (game time) counts too, so attempts
            // extend the fastest ways there.
            // Arrival ranking: the estimated arrival in seconds instead.
            raw_h.push(h);
            // The objective's start bias (frames; 0 for most) ranks too:
            // in seconds on top of the (pure-ETA) arrival.
            let bias = obj.start_bias(s);
            let h = if arrival {
                let a = self.arrival_of(obj, s, h).unwrap_or(h);
                if bias == 0.0 { a } else { a + bias / 30.0 }
            } else {
                let h = if bias == 0.0 { h } else { h + bias };
                if self.cfg.rrt_star { h + self.cfg.time_cost * s.frame as f64 } else { h }
            };
            let u = (s.wedges() as f64 / 8.0).powf(self.cfg.alpha)
                * self.cfg.gamma.powi((b_lives as i32 - s.lives as i32).max(0));
            let crowd = if self.cfg.soft_dedup { 0.5f64.powi(n.stats.crowd as i32) } else { 1.0 };
            cands.push((i as u32, h, u * self.start_mult(&n.stats) * crowd * kw as f64 * self.manual_start_mult(i)));
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
        // OBRRT_WIP_FOCUS=p (with OBRRT_KEEP_WIP, which stores each state's wip):
        // this share of b draws come from the states furthest along a macro
        // goal, so a counted goal is built on instead of restarted. In the
        // zone-locked VCutM tree of 2026-11-09, 10357 states had no red coin
        // and 3 had 4. The macro goals each have their own measure and their
        // own top states (within OBRRT_WIP_FOCUS_W - 1 of the best, default 2
        // levels), so one goal's easy progress doesn't crowd out another's:
        // red coins; coins toward the 100-coin star (tens); level changes
        // (posts and the like); bosses (damage, a spawned star); cap switches.
        let focus = wip_focus();
        let mut top: Vec<(i64, Vec<(u32, i64)>)> = vec![(0, Vec::new()); MACROS];
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
        let ww = goal_wip_w();
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
            (!any_exit() && st.course_done(st.course)) || st.modal() || (self.cfg.rrt_star && self.tree.nodes[i as usize].dominated)
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
            if focus.0 > 0.0 {
                let w = self.tree.wip.get(i as usize).copied().unwrap_or(0);
                let (red, changes, caps) = (st.red_coins.max(0) as i64, st.level_changes(), (st.save_flags & 0xE).count_ones() as i64);
                let coins = if (1..=15).contains(&st.course) && !st.has_star(st.course, 6) { (st.coins.clamp(0, 100) / 10) as i64 } else { 0 };
                for (k, v) in [red, coins, changes, w - red - changes - caps, caps].into_iter().enumerate() {
                    if v > 0 {
                        top[k].0 = top[k].0.max(v);
                        top[k].1.push((i, v));
                    }
                }
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
        self.wip_top = top
            .into_iter()
            .map(|(best, v)| v.into_iter().filter(|&(_, x)| x >= (best - (focus.1 - 1)).max(1)).map(|(i, _)| i).collect::<Vec<u32>>())
            .filter(|v| !v.is_empty())
            .collect();
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
        tick!(10);
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
        Some(Plan { stages: vec![(entry, obj)], start: b, b, rewire: None, seed: None, polish: None, targets: None, window_b: None, window: Vec::new(), mv: None })
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
            return Some(Plan { stages: vec![(entry, obj)], start: r.from, b: r.from, rewire: Some(r.target), seed: None, polish: None, targets: None, window_b: None, window: Vec::new(), mv: None });
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
        Some(Plan { stages: vec![(entry, obj)], start: job.from, b: job.from, rewire: None, seed: Some(seed), polish: Some(job), targets, window_b: None, window: Vec::new(), mv: None })
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

    /// `Diagnostics::note_best` for node `id` (made by `via`), with its
    /// route credit, and a `best_improved` event (output only) when it set
    /// a new best time for its (keys, stars).
    fn note_best_at(&mut self, id: u32, progress: (u8, i16), zone: (u16, u8), frame: u32, via: &'static str) -> Result<()> {
        if let Some(old) = self.diag.note_best(progress, zone, frame) {
            self.route_credit(id, old);
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

    /// The objective a pool objective stands for: a Beat's own objective
    /// (see speed.rs), else itself.
    fn base_obj(&self, obj: &Arc<dyn Objective>) -> Arc<dyn Objective> {
        if self.speed.beat_inner.is_empty() {
            return obj.clone();
        }
        self.speed.beat_inner.get(&obj.key()).cloned().unwrap_or_else(|| obj.clone())
    }

    /// Node `id` (at game frame `frame`, made by `via`) met `obj`: the best
    /// meet per objective key (output only, unless a speed switch reads it),
    /// a `best_improved` event when sooner than a previous meet, and with
    /// OBRRT_BEAT a (tighter) Beat of it.
    fn note_met(&mut self, id: u32, obj: &Arc<dyn Objective>, c: crate::speed::Ctx, frame: u32, via: &'static str) -> Result<()> {
        let base = self.base_obj(obj);
        let key = base.key();
        let ck = (key.clone(), c);
        let old = self.speed.met_best.get(&ck).copied();
        if old.is_some_and(|o| o <= frame) {
            return Ok(());
        }
        Arc::make_mut(&mut self.speed.met_best).insert(ck, frame);
        if let Some(o) = old {
            self.speed.note_met_imp(via, (o - frame) as u64);
            writeln!(
                self.events,
                "{}",
                serde_json::json!({
                    "event": "best_improved", "milestone": "objective", "obj_key": key, "obj_kind": base.kind(), "ctx": [c.0, c.1, c.2],
                    "old": o, "new": frame, "saved": o - frame, "node": id, "via": via,
                    "frames_total": self.total_frames, "from": null, "added": null,
                })
            )?;
        }
        if crate::speed::settings().beat && !base.modal() && base.fixed().is_none() {
            self.beat_spawn(id, base, key, c, frame);
        }
        Ok(())
    }

    /// OBRRT_BEAT: `Beat(base, frame)` into the pool, retiring the live Beat
    /// of `key` (weight 0, expires at the next pool expiry).
    fn beat_spawn(&mut self, id: u32, base: Arc<dyn Objective>, key: String, c: crate::speed::Ctx, frame: u32) {
        let bk = crate::speed::beat_key(&key, c, frame);
        let lk = (key, c);
        let old = self.speed.beat_live.get(&lk).cloned();
        if let Some((d, ok)) = &old {
            if *d <= frame {
                return;
            }
            if *ok == bk {
                self.speed.beat.same_bucket += 1;
                return;
            }
        }
        if self.pool.find(&bk).is_some() {
            self.speed.beat.same_bucket += 1;
            return;
        }
        let opt = self.eta.opt(base.kind(), crate::speed::settings().opt);
        let b: Arc<dyn Objective> = Arc::new(crate::speed::Beat::with_opt(base.clone(), c, frame, opt));
        self.pool.insert(b, &self.tree.nodes[id as usize].st);
        self.speed.beat_inner.insert(bk.clone(), base);
        self.speed.beat.spawned += 1;
        if let Some((_, ok)) = old {
            if let Some(j) = self.pool.find(&ok) {
                let e = &mut self.pool.entries[j];
                e.base_weight = 0.0;
                e.unattemptable = u32::MAX / 2;
            }
            self.speed.beat_inner.remove(&ok);
            self.speed.beat.replaced += 1;
        }
        self.speed.beat_live.insert(lk, (frame, bk));
    }

    /// OBRRT_FASTER_ONLY: whether a success of `obj` (stage `i` of `plan`)
    /// at `st` counts: always unless it is a polish attempt or its key was
    /// met before; then only strictly sooner than the best meet (and the
    /// polish target's best), or at a new (keys, stars) best.
    fn faster_ok(&mut self, plan: &Plan, i: usize, obj: &Arc<dyn Objective>, from: &St, st: &St) -> bool {
        let key = self.base_obj(obj).key();
        let mut bar = self.speed.met_best.get(&(key, crate::speed::ctx(from))).copied();
        if let (0, Some(job), Some(pl)) = (i, plan.polish, self.polish.as_ref()) {
            let tf = self.tree.nodes[job.target as usize].st.frame;
            let pb = pl.best.get(&job.target).copied().unwrap_or(tf).min(tf);
            bar = Some(bar.map_or(pb, |b| b.min(pb)));
        }
        let Some(bar) = bar else { return true };
        let fo = &mut self.speed.fo;
        fo.judged += 1;
        let k = fo.by_kind.entry(obj.kind()).or_default();
        if st.frame < bar {
            fo.sooner += 1;
            fo.sooner_frames += (bar - st.frame) as u64;
            k.0 += 1;
            k.1 += (bar - st.frame) as u64;
            return true;
        }
        if self.diag.best_progress.get(&st.progress()).is_none_or(|b| st.frame < b.0) {
            fo.new_progress += 1;
            return true;
        }
        fo.rejected += 1;
        k.2 += 1;
        false
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
                || (!any_exit() && n.st.course_done(n.st.course))
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
                    out = Some(Plan { stages, start, b, rewire: None, seed: None, polish: None, targets: None, window_b: None, window: Vec::new(), mv: None });
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
        if self.cfg.rrt_star
            && !(self.rewire_hot.is_empty() && self.rewire_cold.is_empty())
            && self.rng.random_bool(self.cfg.rewire_share)
        {
            if let Some(p) = self.plan_rewire() {
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
                        return Some(Plan { stages: vec![(entry, obj)], start: b, b, rewire: None, seed, polish: None, targets: None, window_b: None, window: Vec::new(), mv: None });
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
        } else if !self.wip_top.is_empty() && self.rng.random_bool(wip_focus().0) {
            // A macro goal uniformly, then one of its top states.
            let m = &self.wip_top[self.rng.random_range(0..self.wip_top.len())];
            m[self.rng.random_range(0..m.len())]
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
        // OBRRT_ATTEMPT_MODEL: the sampler proposes up to K (objective, start)
        // pairs and one is drawn by what the model expects of it per frame.
        let am = crate::attempt_model::model();
        let (am_k, am_beta) = crate::attempt_model::settings();
        let mut proposals: Vec<(usize, Arc<dyn Objective>, u32, crate::attempt_model::Out)> = Vec::new();
        for _ in 0..if am.is_some() { 2 * am_k } else { 4 } {
            // Wall time per pick (diagnostics only, never decisions).
            let pick_t = Instant::now();
            let Some(entry) = self.sample_entry(b, false) else { break };
            self.pool.rate.sample_secs += pick_t.elapsed().as_secs_f64();
            let obj = self.pool.entries[entry].obj.clone();
            let kind = obj.kind();
            self.diag.kind(kind).samples += 1;
            self.diag.kind(kind).window_samples += 1;
            let picked = self.pick_start(entry, obj.as_ref(), b);
            let secs = pick_t.elapsed().as_secs_f64();
            if picked.is_some() { self.pool.rate.pick_secs += secs } else { self.pool.rate.nostart_secs += secs }
            match picked {
                Some(start) if am.is_some() => {
                    let out = am.unwrap().eval(&self.model_input(obj.as_ref(), start));
                    proposals.push((entry, obj, start, out));
                    if proposals.len() >= am_k {
                        break;
                    }
                }
                Some(start) => {
                    self.pool.entries[entry].unattemptable = 0;
                    let stages = self.extend_chain(entry, obj, start);
                    return Some(Plan { stages, start, b, rewire: None, seed: None, polish: None, targets: None, window_b: None, window: Vec::new(), mv: None });
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
                    } else {
                        e.decay *= self.cfg.decay;
                    }
                    if let Some(idx) = self.pool.sidx.as_mut() {
                        idx.touch(&self.pool.entries, entry);
                    }
                }
            }
        }
        if !proposals.is_empty() {
            // Per frame at this run's own mean cost of the kind (the model's
            // cost head is from runs before the reach budget and the gate).
            let cost = |k: &str| self.diag.kinds.get(k).filter(|ks| ks.attempts >= 3).map_or(30000.0, |ks| ks.frames as f64 / ks.attempts as f64) + crate::diag::ATTEMPT_OVERHEAD;
            let w: Vec<(usize, f64)> = proposals.iter().enumerate().map(|(i, p)| (i, (crate::attempt_model::score(&p.3).max(1e-4) / cost(p.1.kind())).powf(am_beta))).collect();
            let (entry, obj, start, out) = proposals.swap_remove(weighted(&mut self.rng, &w).unwrap_or(0));
            self.pool.entries[entry].unattemptable = 0;
            let stages = self.extend_chain(entry, obj, start);
            return Some(Plan { stages, start, b, rewire: None, seed: None, polish: None, targets: None, window_b: None, window: Vec::new(), mv: Some(out) });
        }
        self.pool.rate.plans_none += 1;
        None
    }

    /// OBRRT_ATTEMPT_MODEL: what the model sees of attempting `obj` from `start`.
    fn model_input<'a>(&self, obj: &'a dyn Objective, start: u32) -> crate::attempt_model::Input<'a> {
        let n = &self.tree.nodes[start as usize];
        let s = &n.st;
        let kind = obj.kind();
        let h = obrrt_core::start_eta(obj, s);
        let (kind_n, kind_s) = self.diag.kinds.get(kind).map_or((0.0, 0.0), |k| (k.attempts as f64, k.successes as f64));
        crate::attempt_model::Input {
            kind,
            level: s.level,
            action: s.action,
            start_eta: h.is_finite().then_some(h),
            stage: 0.0,
            plan_len: 1.0,
            kind_n,
            kind_s,
            kn_n: self.tried_kind.get(&(kind, start)).copied().unwrap_or(0) as f64,
            node_n: self.tried_node.get(&start).copied().unwrap_or(0) as f64,
            frames_before: self.total_frames as f64,
            y: s.pos[1] as f64,
            hspeed: (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt() as f64,
            fwd_vel: s.forward_vel as f64,
            stars: s.stars as f64,
            coins: s.coins as f64,
            red: s.red_coins as f64,
            route_frame: s.frame as f64 - self.tree.nodes[0].st.frame as f64,
            depth: n.depth as f64,
        }
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
        if stages[0].1.fixed().is_some() {
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
                if w <= 0.0 || e.obj.fixed().is_some() {
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
        // OBRRT_POLISH: a polish plan's states stay loadable (pinned).
        self.tree.pin_next = plan.polish.is_some();
        let mut polish_end: Option<(u32, bool)> = None;
        for (i, mut seg) in att.segments.into_iter().enumerate() {
            let (entry, obj) = (plan.stages[i].0, plan.stages[i].1.clone());
            let kind = obj.kind();
            // OBRRT_PARENT_WINDOW: the copy survived exactly, so this goes on from
            // the selected node: a child of it. A changed copy is another way
            // there: it stays where it started, a sibling.
            if let (0, Some(b)) = (i, plan.window_b) {
                let n = plan.window.len();
                self.window_results[0] += 1;
                if seg.inputs.len() > n && seg.inputs[..n] == plan.window[..] {
                    seg.inputs.drain(..n);
                    seg.played = seg.played.saturating_sub(n);
                    parent = b;
                    seg_start = self.tree.nodes[b as usize].st.clone();
                    self.window_results[1] += 1;
                }
            }
            // v2: progress is minus the ETA (frames), the start's from fresh
            // scratch, the end's as the controller folded it.
            let start_h = obrrt_core::start_eta(obj.as_ref(), &seg_start);
            let start_progress = -start_h;
            // Frames this segment's committed inputs play (to Done for a
            // success; the automatic tail not counted): with `start_h`, what
            // tools/analysis/eta_calib.py calibrates ETAs by.
            let seg_frames = seg.played;
            self.eta.note(kind, seg.success, start_h, seg_frames);
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
            let seg_tail = seg.tail;
            // OBRRT_FASTER_ONLY: a met-but-not-sooner re-attempt is a failure.
            let fo_reject = crate::speed::settings().faster_only && seg.success && seg.end.as_ref().is_some_and(|(st, _)| !self.faster_ok(plan, i, &obj, &seg_start, st));
            let success = seg.success && !fo_reject;
            if let Some((st, packed)) = seg.end {
                if self.cfg.sampler_rate {
                    end_k = self.keys_of(obj.as_ref(), &st);
                }
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
                if worth && (!dup || (self.cfg.soft_dedup && !fo_reject)) {
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
                    if let Some(m) = seg.model.clone() {
                        self.tree.nodes[id as usize].stats.model = Some(m);
                    }
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
                        // (Before note_met, which may replace this Beat.)
                        if self.speed.beat_inner.contains_key(&obj.key()) {
                            self.speed.beat.successes += 1;
                        }
                        self.note_met(id, &obj, crate::speed::ctx(&seg_start), frame, kind)?;
                    }
                    self.speed.fo.rejected_added += fo_reject as u64;
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
                        polish_end = Some((id, success));
                    }
                    if i == 0 && self.near.on.is_some() {
                        let rank = if success { (frame - seg_start.frame) as f64 } else { self.keys_of(obj.as_ref(), &self.tree.nodes[id as usize].st).0 };
                        self.near.note(obj.key(), plan.start, id, success, rank);
                    }
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
                // OBRRT_GAMMA_SET: the first stage's outcome under its γ.
                if i == 0 && crate::gammaset::set().is_some() {
                    crate::gammaset::observe(obj.kind(), (success && novel) || (added.is_some() && improves(&self.cfg, end_k, start_k)));
                }
            }
            {
                let e = &mut self.pool.entries[entry];
                e.attempts += 1;
                // Improved on its start: met, or kept a state nearer by the
                // objective's own heuristic (with `arrival`: sooner by the
                // estimated arrival, so a slow walk inching closer doesn't
                // count; or either, `arrival_decay_either`).
                e.best_start = e.best_start.min(start_k.0);
                if let Some(f) = start_k.1 {
                    e.best_arrival = e.best_arrival.min(f);
                }
                if self.cfg.sampler_rate {
                    // Rate sampler: credited below, no decay.
                    e.successes += success as u32;
                } else if success && novel {
                    e.successes += 1;
                    e.decay = 1.0;
                } else {
                    // A success that only reproduced an existing state decays too.
                    e.successes += success as u32;
                    e.decay *= self.cfg.decay;
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
                if let (true, Some(f)) = (success, end_frame) {
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
                        } else if novel {
                            // Met for the first time from a new state: as a
                            // stage's novel success would.
                            e.decay = 1.0;
                        }
                    }
                }
            }
            if i == 0 {
                let ns = &mut self.tree.nodes[plan.window_b.unwrap_or(plan.start) as usize].stats;
                ns.as_start += 1;
                if success {
                    ns.successes += 1;
                }
            }
            let mut ev = serde_json::json!({
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
            if crate::speed::settings().faster_only {
                ev["accepted"] = serde_json::json!(success);
            }
            if let Some(m) = plan.mv.filter(|_| i == 0) {
                ev["mv"] = serde_json::json!({"success": m.success, "pays": m.pays, "star": m.star, "cost": m.cost[1], "seg": m.seg[1], "value": m.value});
            }
            writeln!(self.events, "{ev}")?;
            if let Some(id) = added.filter(|_| novel) {
                // A new state may make decayed objectives worth trying again
                // (the entry bandit has no resets: its evidence fades instead).
                // With `arrival`, "better" is judged by `improves` (sooner,
                // and with `arrival_decay_either` also closer).
                let st = &self.tree.nodes[id as usize].st;
                let cfg = &self.cfg;
                let eta = &self.eta;
                for e in self.pool.entries.iter_mut().filter(|_| !cfg.sampler_rate) {
                    if e.decay < 1.0 && e.obj.exists_in(st) && e.obj.base_criteria(st) && {
                        let h = obrrt_core::start_eta(e.obj.as_ref(), st) * eta.c(e.obj.kind());
                        let f = arrival_est(cfg, e.obj.as_ref(), st, h);
                        improves(cfg, (rank_of(e.obj.as_ref(), st, h), f), (e.best_start, f.map(|_| e.best_arrival)))
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
                if success && self.first_success.insert(kind) {
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
            let ns = &mut self.tree.nodes[plan.window_b.unwrap_or(plan.start) as usize].stats;
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
            let ns = &mut self.tree.nodes[plan.window_b.unwrap_or(plan.start) as usize].stats;
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
            if self.cfg.chain_route || self.cfg.rrt_star {
                self.tree.nodes[id as usize].made_by = Some(obj.clone());
            }
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

    /// OBRRT_MEM_DIAG=1, at each report: RSS against the accounted bytes,
    /// glibc's heap (in use and free), and the sizes of the tables the save
    /// budget doesn't count.
    fn mem_diag(&self) {
        let rss = obrrt_core::sys::fs::read_to_string("/proc/self/status").ok().and_then(|t| t.lines().find(|l| l.starts_with("VmRSS:")).map(|l| l[6..].trim().to_string())).unwrap_or_default();
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        {
            #[repr(C)]
            #[derive(Default)]
            struct MallInfo2 {
                arena: usize,
                ordblks: usize,
                smblks: usize,
                hblks: usize,
                hblkhd: usize,
                usmblks: usize,
                fsmblks: usize,
                uordblks: usize,
                fordblks: usize,
                keepcost: usize,
            }
            unsafe extern "C" {
                fn mallinfo2() -> MallInfo2;
            }
            // SAFETY: returns a plain struct.
            let m = unsafe { mallinfo2() };
            println!(
                "   memdiag: malloc heap {} MB (in use {} MB, free {} MB, {} free chunks), mmapped {} MB in {} blocks",
                m.arena / 1_000_000, m.uordblks / 1_000_000, m.fordblks / 1_000_000, m.ordblks, m.hblkhd / 1_000_000, m.hblks
            );
        }
        {
            let t = &self.tree;
            let mb = |x: usize| x / 1_000_000;
            let save_len: usize = t.nodes.iter().map(|n| n.save.bytes()).sum();
            let save_cap: usize = t.nodes.iter().map(|n| n.save.heap_bytes()).sum();
            let tail: usize = t.nodes.iter().map(|n| n.tail.capacity() * std::mem::size_of::<GamePad>()).sum();
            let inputs: usize = t.nodes.iter().map(|n| n.inputs.capacity() * std::mem::size_of::<GamePad>()).sum();
            let made_by = t.nodes.iter().filter(|n| n.made_by.is_some()).count();
            let ne: usize = self.ne.sets.values().map(|b| b.len() * 8 + 16).sum::<usize>() + self.ne.sets.capacity() * 21;
            println!(
                "   memdiag: saves len {} MB cap {} MB; tail {} MB inputs {} MB; nodes vec {} MB (cap {} x {} B); made_by {}; pool {} x {} B; ne sets {} MB; dkeys cap {}; frontier cap {}",
                mb(save_len), mb(save_cap), mb(tail), mb(inputs), mb(t.nodes.capacity() * std::mem::size_of::<crate::tree::Node>()), t.nodes.capacity(), std::mem::size_of::<crate::tree::Node>(),
                made_by, self.pool.entries.capacity(), std::mem::size_of::<crate::pool::Entry>(), mb(ne), t.dkeys.capacity(), t.frontier.capacity()
            );
        }
        let cells: usize = self.cells.values().map(|v| v.capacity() * 4).sum();
        let ch: usize = self.tree.children.iter().map(|v| v.capacity() * 4).sum();
        println!(
            "   memdiag: rss {rss}; acct {} MB; nodes {} pool {} met_best {} beat_live {} beat_inner {} cells {} ({} MB) modal_tried {} rewire {}/{} ne sets {} queue {} children {} MB dkeys {} hot {} route_met {} decay_resets {}",
            self.tree.mem_bytes() / 1_000_000, self.tree.nodes.len(), self.pool.entries.len(), self.speed.met_best.len(), self.speed.beat_live.len(), self.speed.beat_inner.len(),
            self.cells.len(), cells / 1_000_000, self.modal_tried.len(), self.rewire_hot.len(), self.rewire_cold.len(), self.ne.sets.len(), self.ne.queue.len(),
            ch / 1_000_000, self.tree.dkeys.len(), self.hot.len(), self.route_met.len(), self.decay_resets.len()
        );
        if let Some(r) = &self.tree.records {
            let r = r.read().unwrap();
            println!("   memdiag: records {} nodes in {} cells, {} MB", r.len(), r.cells(), r.bytes() / 1_000_000);
        }
        if let Some(rr) = &self.rr {
            println!("   memdiag: {}", rr.mem_diag());
        }
        if let Some(zs) = &self.zs {
            println!("   memdiag: {}", zs.mem_diag());
        }
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
        print!("{}", self.eta.report(8));
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
        if self.eta.on {
            summary["eta_learn"] = self.eta.summary();
        }
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
        if obrrt_core::sys::env::var("OBRRT_MEM_DIAG").is_ok_and(|v| v == "1") {
            self.mem_diag();
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
        if self.window_attempts > 0 {
            println!("   parent window: {} attempts, mean copy {:.0} frames; stops retired/empty/max/zone/control/lost {:?}; merged {} copy kept {}", self.window_attempts, self.window_frames as f64 / self.window_attempts as f64, self.window_stops, self.window_results[0], self.window_results[1]);
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
        if crate::movegenome::enabled() {
            summary["move_ga"] = crate::movegenome::stats_json();
        }
        // OBRRT_GA_QD (qd.rs) and OBRRT_GAMMA_SET (gammaset.rs): only when on.
        if let Some(q) = crate::qd::summary() {
            summary["ga_qd"] = q;
        }
        if let Some(g) = crate::gammaset::summary() {
            summary["gamma_set"] = g;
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
                    self.bases.push(crate::game::into_plain(packed.apply(&self.bases)?).into_bytes());
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
            self.note_best_at(id, progress, zone, frame, via)?;
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
        if let Some(rr) = self.rr.as_mut() {
            rr.note(id, &self.tree.nodes[id as usize].st, &c, &self.tree, &self.hot, self.cfg.reroute_margin, 0);
        }
        self.tree.gen_step[id as usize] = true;
        self.sync_sent.insert(id);
        if adv {
            let (keys, stars) = self.tree.frontier_progress;
            let f = self.write_m64(id, &format!("frontier-k{keys}-s{stars}"))?;
            println!("FRONTIER keys {keys} stars {stars} at node {id} via tree sync: {f}");
        }
        Ok(SyncImport::Added(id))
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
                self.bases.push(crate::game::into_plain(packed.apply(&self.bases)?).into_bytes());
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
        self.note_best_at(id, progress, zone, frame, kind)?;
        if let (true, Some(m)) = (self.tree.nodes[id as usize].met, self.tree.nodes[id as usize].made_by.clone()) {
            let c = crate::speed::ctx(&self.tree.nodes[g.from as usize].st);
            self.note_met(id, &m, c, frame, kind)?;
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
            // OBRRT_ETA_LEARN: this round's factors (merges only add samples).
            let mut lap = Instant::now();
            let mut tick = |k: usize, d: &mut [f64; 11]| {
                d[k] += lap.elapsed().as_secs_f64();
                lap = Instant::now();
            };
            self.eta.snapshot();
            tick(0, &mut self.diag_pre);
            if self.cfg.sampler_rate {
                let p = self.cfg.rate;
                if p.nostart_skip {
                    self.nostart_recheck();
                }
                tick(1, &mut self.diag_pre);
                self.pool.rate_refresh(self.total_frames, &p);
            }
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
                .filter_map(|_| {
                    let p = self.plan();
                    // OBRRT_ATTEMPT_MODEL: counted when planned, so the plans of
                    // one round see each other (the model was trained on counts
                    // in attempt order).
                    if let Some(p) = p.as_ref().filter(|_| crate::attempt_model::model().is_some()) {
                        *self.tried_kind.entry((p.stages[0].1.kind(), p.start)).or_default() += 1;
                        *self.tried_node.entry(p.start).or_default() += 1;
                    }
                    p
                })
                .collect();
            // Nothing to plan (a course with every star had: no frontier state
            // is a base) spun forever before 2026-11-09 (campaign t1 step 97,
            // TotWC, 40 minutes at one core, no frames): a few empty rounds end
            // the run.
            if plans.is_empty() && grafts.is_empty() && shares.is_empty() && n_polish == 0 {
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
            for _ in 0..n_polish {
                if let Some(p) = self.plan_polish() {
                    plans.push(p);
                }
            }
            if self.near.on.is_some() {
                for p in plans.iter_mut() {
                    let o = &p.stages[0].1;
                    if p.rewire.is_some() || p.polish.is_some() || p.seed.is_some() || o.modal() || !o.fixed().is_none() {
                        continue;
                    }
                    p.seed = self.near.pick(&self.tree, &o.key(), p.start);
                }
            }
            // OBRRT_PARENT_WINDOW=p: this share of the attempts start from the
            // selected node's parent, every first-generation genome beginning
            // with the recorded parent-to-node inputs (at most
            // OBRRT_PARENT_WINDOW_MAX frames, same zone, parent still loadable,
            // and the goal not already met by the copy).
            if let Some((share, max)) = parent_window() {
                for p in plans.iter_mut() {
                    let o = p.stages[0].1.clone();
                    if p.rewire.is_some() || p.polish.is_some() || p.seed.is_some() || o.modal() || o.fixed().is_some() {
                        continue;
                    }
                    // Back through the ancestors until the copy is at least
                    // OBRRT_PARENT_WINDOW_FRAMES long (0: just the parent), so
                    // short segments don't make a window of a few frames. Stops
                    // early at anything a window can't start from or cross; the
                    // last good ancestor is used.
                    // OBRRT_PARENT_WINDOW_HORIZONS=k: also at least k x this objective's
                    // GA horizon, so the copy scales with what the attempt can search.
                    // Both scale down with the population (full from OBRRT_PARENT_WINDOW_POP,
                    // default 48): a population of 16 can't search a long copy.
                    let (horizon, pop) = o.ga_budget().map_or((150, 16), |b| (b.horizon, b.population));
                    let scale = (pop as f64 / parent_window_pop()).min(1.0);
                    let want = parent_window_frames().max((parent_window_horizons() * horizon as f64) as usize);
                    let (start_id, want) = (p.start, (want as f64 * scale) as usize);
                    let mut at = start_id;
                    let (mut anc, mut segs, mut len): (Option<u32>, Vec<u32>, usize) = (None, Vec::new(), 0);
                    loop {
                        let node = &self.tree.nodes[at as usize];
                        let Some(pid) = node.parent else { break };
                        let par = &self.tree.nodes[pid as usize];
                        let why = if par.retired { 0 } else if node.inputs.is_empty() { 1 } else if len + node.inputs.len() > max { 2 } else if par.st.zone() != node.st.zone() { 3 } else if !par.st.controllable() || par.st.in_dialog() { 4 } else if !obrrt_core::start_eta(o.as_ref(), &par.st).is_finite() { 5 } else { 6 };
                        if why < 6 {
                            self.window_stops[why] += 1;
                            break;
                        }
                        segs.push(at);
                        len += node.inputs.len();
                        anc = Some(pid);
                        at = pid;
                        if len >= want {
                            break;
                        }
                    }
                    let Some(pid) = anc else { continue };
                    if !self.rng.random_bool(share) {
                        continue;
                    }
                    p.window = segs.iter().rev().flat_map(|&n| self.tree.nodes[n as usize].inputs.iter().copied()).collect();
                    p.window_b = Some(start_id);
                    self.window_frames += p.window.len() as u64;
                    p.start = pid;
                    self.window_attempts += 1;
                }
            }
            self.diag_plan_secs.0 += plan_t.elapsed().as_secs_f64();
            if plans.is_empty() && grafts.is_empty() && shares.is_empty() {
                continue;
            }
            let bases = &self.bases;
            let tree = &self.tree;
            let seeds: Vec<u64> = plans.iter().map(|_| self.rng.random()).collect();
            // OBRRT_GAMMA_SET (gammaset.rs): each plan's γ, from its seed and
            // the per-kind counts as of now (empty when off).
            let gam_v: Vec<Option<u8>> = if crate::gammaset::set().is_some() {
                plans.iter().zip(&seeds).map(|(p, s)| crate::gammaset::choose(p.stages[0].1.kind(), *s)).collect()
            } else {
                Vec::new()
            };
            let gam = &gam_v;
            // OBRRT_BNB: per plan and stage, the best meet of its key so far
            // (fixed now, before the round's attempts run: deterministic).
            let bnb_v: Vec<Option<crate::speed::Bounds>> = if crate::speed::settings().bnb {
                plans
                    .iter()
                    .map(|p| {
                        let keys: Vec<Option<String>> =
                            p.stages.iter().map(|(_, o)| (!o.modal() && o.fixed().is_none()).then(|| self.base_obj(o).key())).collect();
                        keys.iter()
                            .any(|k| k.is_some())
                            .then(|| {
                                let opt = p.stages.iter().map(|(_, o)| self.eta.opt(self.base_obj(o).kind(), crate::speed::settings().opt)).collect();
                                Arc::new(crate::speed::BoundsIn { keys, best: self.speed.met_best.clone(), opt })
                            })
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let bnb = &bnb_v;
            let workers = &self.workers;
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
                    crate::ga::set_repair_targets(plan.targets.clone());
                    crate::ga::set_tail((!node.tail.is_empty() && plan.window.is_empty()).then(|| node.tail.clone()));
                    crate::ga::set_window(plan.window.clone());
                    if !bnb.is_empty() {
                        crate::speed::set_bounds(bnb[i].clone());
                    }
                    if !gam.is_empty() {
                        crate::gammaset::set_current(gam[i]);
                    }
                    let at = Instant::now();
                    let packed = w.attempt_full(&stages, &node.st, None, &mut rng, params, seed_in, req.as_ref()).map(|a| a.map_saves(|s| pack(&s)));
                    if !gam.is_empty() {
                        crate::gammaset::set_current(None);
                    }
                    if !bnb.is_empty() {
                        crate::speed::set_bounds(None);
                    }
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
            drop(bnb_v); // the snapshot's last holders: met_best is updated in place again
            self.diag_secs[1] += run_t.elapsed().as_secs_f64();
            self.diag_secs[4] += load_ns.into_inner() as f64 * 1e-9;
            self.diag_secs[5] += att_ns.into_inner() as f64 * 1e-9;
            let merge_t = Instant::now();
            for (i, (plan, r)) in plans.iter().zip(results).enumerate() {
                if plan.stages[0].1.kind() == "Road::Leg" {
                    if let (Some(road), Ok(att)) = (self.road.as_mut(), r.as_ref()) {
                        road.stats.legs_ok += att.segments.first().is_some_and(|s| s.success) as u64;
                    }
                } else if plan.seed.is_some() {
                    if let Ok(att) = r.as_ref() {
                        self.near.stats.seed_met += att.ga.and_then(|g| g.seed).is_some_and(|s| s.0) as u64;
                        self.near.stats.successes += att.segments.first().is_some_and(|s| s.success) as u64;
                    }
                }
                if !gam_v.is_empty() {
                    crate::gammaset::set_current(gam_v[i]);
                }
                match r {
                    Ok(att) => {
                        self.merge(plan, att)?;
                        crate::gammaset::set_current(None);
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
                            if self.cfg.sampler_rate && *entry < self.pool.entries.len() {
                                let p = self.cfg.rate;
                                self.pool.rate_note(*entry, 0.0, p.nostart_frames, self.total_frames, &p, true);
                            } else if let Some(e) = self.pool.entries.get_mut(*entry) {
                                e.decay *= self.cfg.decay;
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

/// OBRRT_ANY_EXIT=1: this run is only to leave the course (search/src/tree.rs).
fn any_exit() -> bool {
    // OBRRT_FINAL=1 (the last search, BitS and the Bowser 3 arena): at 120 stars every course counts as done, which
    // planned nothing at all ("nothing to plan for 20 rounds", b3 test from a BitS state with its star had).
    let s = &obrrt_core::config::current().stop;
    s.any_exit || s.final_search
}
