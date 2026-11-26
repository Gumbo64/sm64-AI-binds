//! [`SearchConfig`]: every setting of the objective-RRT search as plain data.
//!
//! One value, built by an adapter (the environment, [`crate::from_env`]; a TOML file,
//! [`crate::from_toml`]; or in code), handed to the search. The defaults are the search's
//! defaults, so `SearchConfig::default()` is "no switch set". Field docs name the legacy
//! `OBRRT_*` variable each field replaces.
//!
//! Grouping: what the search does ([`Budget`], [`Sampler`], [`Chain`], [`Dedup`], [`Zone`],
//! [`GoalFocus`], [`Kin`], [`Ga`], [`Priors`], [`Stop`]), what the domain knowledge does
//! ([`ObjectiveConfig`]) and the switches only research scripts set ([`Research`]).
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub budget: Budget,
    pub sampler: Sampler,
    pub chain: Chain,
    pub dedup: Dedup,
    pub zone: Zone,
    pub goal_focus: GoalFocus,
    pub kin: Kin,
    pub ga: Ga,
    pub priors: Priors,
    pub objectives: ObjectiveConfig,
    pub stop: Stop,
    pub research: Research,
}

/// How much the search may spend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Budget {
    /// OBRRT_GA_SCALE: multiplies the GA's generations.
    pub ga_scale: f64,
    /// OBRRT_MAX_FRAMES: stop after emulating this many frames.
    pub max_frames: Option<u64>,
    /// OBRRT_MEM_MB: memory budget of saves, in MB (None: the engine's 3 GB).
    pub mem_mb: Option<usize>,
    /// OBRRT_ROUND: attempts planned per round.
    pub round_plans: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget { ga_scale: 3.0, max_frames: None, mem_mb: None, round_plans: 64 }
    }
}

/// The hierarchical rate sampler's parameters (OBRRT_RATE_*).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rate {
    /// OBRRT_RATE_PRIOR
    pub prior_frames: f64,
    pub kind_scale: f64,
    pub route: f64,
    pub novel: f64,
    /// OBRRT_RATE_MODE=hier
    pub hier: bool,
    /// OBRRT_RATE_NOSTART_SKIP=1
    pub nostart_skip: bool,
    pub skip_budget: u64,
}

impl Default for Rate {
    fn default() -> Self {
        Rate {
            prior_frames: 200_000.0,
            kind_scale: 10.0,
            route: 0.0,
            novel: 1.0,
            hier: false,
            nostart_skip: false,
            skip_budget: 2_000_000,
        }
    }
}

/// Arrival ranking (OBRRT_ARRIVAL*).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Arrival {
    pub on: bool,
    pub pace: f64,
    pub band: Option<f64>,
}

impl Default for Arrival {
    fn default() -> Self {
        Arrival { on: false, pace: 20.0, band: None }
    }
}

/// Which objective, from which start: weights and who gets picked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampler {
    /// OBRRT_SAMPLER=rate (false: decay).
    pub rate_sampler: bool,
    pub rate: Rate,
    /// OBRRT_PICK_BY_B
    pub pick_by_b: bool,
    /// OBRRT_NEW_ELIGIBLE
    pub new_eligible: bool,
    /// OBRRT_STAR: RRT* mode.
    pub rrt_star: bool,
    pub arrival: Arrival,
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler {
            rate_sampler: true,
            rate: Rate::default(),
            pick_by_b: false,
            new_eligible: true,
            rrt_star: false,
            arrival: Arrival::default(),
        }
    }
}

/// How attempts chain objectives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Chain {
    /// OBRRT_CHAIN=route (false: stack).
    pub route: bool,
}

impl Default for Chain {
    fn default() -> Self {
        Chain { route: true }
    }
}

/// OBRRT_REPLAY_MAX: when a node keeps a save.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReplayMax {
    /// Unset: 150 with dense adds on, every node saved otherwise.
    #[default]
    Auto,
    /// "off": every save kept.
    Off,
    /// A node keeps a save only beyond this many frames from its saved ancestor.
    Frames(u32),
}

/// Dense adds (OBRRT_ADD, OBRRT_DENSE_*).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dense {
    /// OBRRT_ADD=winner|winner_all|last_gen|all (comma-separated); None: off.
    pub add: Option<String>,
    pub grid: u32,
    /// OBRRT_DENSE_CAP (None: 256 with `all`, else 0).
    pub cap: Option<usize>,
    pub ne: bool,
    pub every: u32,
    /// OBRRT_DENSE_EVERY=obj[:n]
    pub per_obj: bool,
    pub air: bool,
}

impl Default for Dense {
    fn default() -> Self {
        Dense { add: Some("last_gen".into()), grid: 64, cap: None, ne: true, every: 1, per_obj: false, air: true }
    }
}

/// OBRRT_NEAR_SEED and its settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Near {
    pub seed: bool,
    pub radius: f32,
    pub p: f64,
}

impl Default for Near {
    fn default() -> Self {
        Near { seed: false, radius: 300.0, p: 0.5 }
    }
}

/// What counts as the same state, and what the tree retires.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dedup {
    /// OBRRT_SOFT_DEDUP
    pub soft_dedup: bool,
    /// OBRRT_STARSET_KEY: dedupe and b cells by the collected set.
    pub starset_key: bool,
    /// OBRRT_STAR_WINDOW (None: 1, or 99 with the zone lock).
    pub star_window: Option<i16>,
    /// OBRRT_KEEP_WIP / _N / _FRAMES
    pub keep_wip: bool,
    /// OBRRT_CELL_MOTION
    pub cell_motion: bool,
    pub replay_max: ReplayMax,
    /// OBRRT_SPARSE_CHECK
    pub sparse_check: usize,
    pub dense: Dense,
    pub near: Near,
}

impl Default for Dedup {
    fn default() -> Self {
        Dedup {
            soft_dedup: true,
            starset_key: true,
            star_window: None,
            keep_wip: true,
            cell_motion: false,
            replay_max: ReplayMax::Auto,
            sparse_check: 0,
            dense: Dense::default(),
            near: Near::default(),
        }
    }
}

/// Zones: the tree's level lock, replays between entries, grafts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Zone {
    /// OBRRT_ZONE_LOCK: the tree stays in the root's level.
    pub lock: bool,
    /// OBRRT_ZONE_PATH: castle hops, "l.a>l.a@x,y,z;..." (empty: none).
    pub path: String,
    /// OBRRT_ZONE_SHARE / _K / _CAP / _SHARE / _BG
    pub share: bool,
    /// OBRRT_ZONE_SHARE_DONOR=sample
    pub donor_sample: bool,
    /// OBRRT_ZONE_SHARE_STATS
    pub share_stats: bool,
    /// OBRRT_REROUTE / _SCHEMA / _SHARE / _MARGIN
    pub reroute: bool,
    pub reroute_schema: Option<String>,
    /// OBRRT_POLISH: share of a round's plans re-attempting the best route's milestones.
    pub polish: f64,
    /// OBRRT_REPLAY_BUDGET (None / 0: the slot shares).
    pub replay_budget: Option<f64>,
}

impl Default for Zone {
    fn default() -> Self {
        Zone {
            lock: false,
            path: String::new(),
            share: false,
            donor_sample: false,
            share_stats: false,
            reroute: false,
            reroute_schema: None,
            polish: 0.0,
            replay_budget: None,
        }
    }
}

/// Parent windows (OBRRT_PARENT_WINDOW*).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParentWindow {
    /// OBRRT_PARENT_WINDOW: share of attempts (None / not positive: off).
    pub share: Option<f64>,
    pub max: usize,
    pub frames: usize,
    pub pop: f64,
    pub horizons: f64,
}

impl Default for ParentWindow {
    fn default() -> Self {
        ParentWindow { share: None, max: 300, frames: 0, pop: 48.0, horizons: 0.0 }
    }
}

/// Goal focus: plans drawn toward goals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GoalFocus {
    /// OBRRT_GOAL_FOCUS: share of plans drawn toward a goal (clamped to 0..1 where used).
    pub share: f64,
    /// OBRRT_GOAL_EXPLORE
    pub explore: bool,
    /// OBRRT_GOAL_WIDE: far objects still give their goals.
    pub wide: bool,
    /// OBRRT_STALL_ETA (not positive: None).
    pub stall_eta: Option<f64>,
    pub parent_window: ParentWindow,
}

impl Default for GoalFocus {
    fn default() -> Self {
        GoalFocus { share: 0.0, explore: false, wide: false, stall_eta: None, parent_window: ParentWindow::default() }
    }
}

/// OBRRT_KIN_UNIT.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KinUnit {
    #[default]
    Node,
    Attempt,
    Frames,
}

/// Kinship generations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Kin {
    /// OBRRT_KINSHIP_MULT
    pub mult: f64,
    /// OBRRT_KIN_UNIT
    pub unit: KinUnit,
    /// OBRRT_KIN_FRAMES (None: the tree's 150).
    pub frames: Option<u32>,
    /// OBRRT_KIN_SOFT (None: the hard set).
    pub soft: Option<f64>,
    /// OBRRT_KIN_CAP (None: 0 = none).
    pub cap: Option<usize>,
}

impl Default for Kin {
    fn default() -> Self {
        Kin { mult: 10.0, unit: KinUnit::Node, frames: None, soft: None, cap: None }
    }
}

/// OBRRT_FITNESS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FitnessKind {
    Met,
    #[default]
    Steps,
}

/// The GA that runs each attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Ga {
    /// OBRRT_FITNESS
    pub fitness: FitnessKind,
    /// OBRRT_GA_NEW
    pub new_ops: bool,
    /// OBRRT_PINK
    pub pink: f64,
    /// OBRRT_GA_BUDGET=1: per-objective GA budgets.
    pub per_objective_budget: bool,
    /// OBRRT_GA_GAMMA: the open GA's discount per frame.
    pub gamma: f64,
    /// OBRRT_GA_REGRET (frames; 0: off)
    pub regret: usize,
    /// OBRRT_GA_CBUTTONS
    pub cbuttons: f64,
    /// OBRRT_GA_KEEP_SEED
    pub keep_seed: bool,
    /// OBRRT_TAIL
    pub tail: bool,
    /// OBRRT_CALM
    pub calm: bool,
    /// OBRRT_MOVE_GA / OBRRT_MOVE_GA_EXTRA
    pub move_ga: bool,
    pub move_ga_extra: bool,
}

impl Default for Ga {
    fn default() -> Self {
        Ga {
            fitness: FitnessKind::Steps,
            new_ops: true,
            pink: 0.0,
            per_objective_budget: false,
            gamma: 0.99,
            regret: 0,
            cbuttons: 0.0,
            keep_seed: false,
            tail: false,
            calm: true,
            move_ga: true,
            move_ga_extra: true,
        }
    }
}

/// Learned and recorded priors the search reads from files or servers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Priors {
    /// OBRRT_REACH_PRIOR=<file>
    pub reach_prior: Option<String>,
    /// OBRRT_RESCUE_PRIOR=<file>
    pub rescue_prior: Option<String>,
    /// OBRRT_KIND_WEIGHTS=<file>
    pub kind_weights: Option<String>,
    /// OBRRT_ROAD: "" / "0": off, "1": on without a directory, else the directory.
    pub road: Option<String>,
    /// OBRRT_ETA_LEARN / _N0
    pub eta_learn: bool,
    pub eta_learn_n0: f64,
    /// OBRRT_MODEL=<socket> (None / empty: off) and OBRRT_MODEL_START_T / _SERVERS / _ZONES.
    pub model: Option<String>,
    pub model_start_t: f64,
    pub model_servers: usize,
    pub model_zones: Option<String>,
}

impl Default for Priors {
    fn default() -> Self {
        Priors {
            reach_prior: None,
            rescue_prior: None,
            kind_weights: None,
            road: None,
            eta_learn: false,
            eta_learn_n0: 20.0,
            model: None,
            model_start_t: 300.0,
            model_servers: 1,
            model_zones: None,
        }
    }
}

/// When the run stops and what counts as a good exit.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Stop {
    /// OBRRT_ANY_EXIT
    pub any_exit: bool,
    /// OBRRT_FINAL: the run's last search (only the grand star exits).
    pub final_search: bool,
    /// OBRRT_EXIT_NOSTOP
    pub exit_nostop: bool,
    /// OBRRT_EXIT_SKIP=<levels>
    pub exit_skip: Vec<u32>,
    /// OBRRT_STOP_FLAG=<hex save flags> (0: none)
    pub flag: u32,
    /// OBRRT_STOP_PROGRESS
    pub progress: bool,
    /// OBRRT_STOP_STALL=<frames>
    pub stall: Option<u64>,
    /// OBRRT_STOP_EXIT_GRACE=<frames>
    pub exit_grace: Option<u64>,
    /// OBRRT_GOOD_MAX_FRAME (None: no limit)
    pub good_max_frame: Option<u32>,
}

/// OBRRT_MOVES_UPHILL.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Uphill {
    #[default]
    Kick,
    Jump,
    AirKick,
}

/// The domain knowledge's switches (the objectives and the movement library).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectiveConfig {
    /// OBRRT_TECH: technique objectives on ("all", or a list; see extra/tech/mod.rs).
    pub tech: String,
    /// OBRRT_FAST_MODAL (None: the library's default).
    pub fast_modal: Option<bool>,
    /// OBRRT_MOVES / OBRRT_MOVES_EXTRA / OBRRT_MOVES_MODES / OBRRT_MOVES_UPHILL
    pub moves: bool,
    pub moves_extra: bool,
    pub moves_modes: bool,
    pub moves_uphill: Uphill,
    pub swim_hold: bool,
    pub swim_targets: bool,
    pub gp_catch: bool,
    pub lakitu_arc: bool,
    pub lakitu_through_door: bool,
    pub painting_fast: bool,
    pub grounds_fix: bool,
    /// OBRRT_MOVE_WEIGHT=<x>: the 0.5-weight techniques weigh x (None: unchanged).
    pub move_weight: Option<f64>,
    pub talk_nil: bool,
    pub king_landing: bool,
    /// OBRRT_BLJ_*
    pub blj_detector: Option<String>,
    pub blj_fast: bool,
    pub blj_risers: bool,
    pub blj_scan: bool,
    pub blj_spots: bool,
    pub blj_spots_catalog: Option<String>,
    pub blj_walls: bool,
    pub fastexit: bool,
    pub moatwarp: bool,
    pub pause_buf: bool,
    pub waterwarp: bool,
    pub twostar: bool,
    /// OBRRT_STATIC_ID (core): objects keyed by their static id.
    pub static_id: bool,
    /// OBRRT_GEO (core): walkable distances over the level's floors.
    pub geo: bool,
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        ObjectiveConfig {
            tech: "all".into(),
            fast_modal: None,
            moves: true,
            moves_extra: false,
            moves_modes: false,
            moves_uphill: Uphill::Kick,
            swim_hold: true,
            swim_targets: false,
            gp_catch: true,
            lakitu_arc: true,
            lakitu_through_door: false,
            painting_fast: true,
            grounds_fix: false,
            move_weight: Some(0.75),
            talk_nil: false,
            king_landing: false,
            blj_detector: None,
            blj_fast: false,
            blj_risers: false,
            blj_scan: false,
            blj_spots: false,
            blj_spots_catalog: None,
            blj_walls: false,
            fastexit: false,
            moatwarp: false,
            pause_buf: false,
            waterwarp: false,
            twostar: false,
            static_id: false,
            geo: false,
        }
    }
}

/// Faster-only, Beat and branch-and-bound bounds (OBRRT_FASTER_ONLY, _BEAT, _BNB).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Speed {
    pub faster_only: bool,
    pub beat: bool,
    pub bnb: bool,
}

/// The rolling-horizon controller (OBRRT_RH, OBRRT_RH_*).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Rh {
    pub on: bool,
    pub pop: usize,
    pub h: usize,
    pub k: usize,
    pub g: usize,
    pub g_precise: usize,
    pub gamma: f64,
    pub bonus: f64,
    pub death: f64,
    pub stall: usize,
    pub hot: usize,
    pub maxlen: usize,
}

impl Default for Rh {
    fn default() -> Self {
        Rh { on: false, pop: 32, h: 240, k: 15, g: 1, g_precise: 3, gamma: 0.99, bonus: 300.0, death: 600.0, stall: 6, hot: 3, maxlen: 900 }
    }
}

/// Switches only research scripts set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Research {
    pub speed: Speed,
    pub rh: Rh,
    /// OBRRT_GA_QD=1 and OBRRT_GA_GAMMAS=<g,g,...>
    pub qd: bool,
    pub qd_gammas: Option<String>,
    /// OBRRT_PICK=schema|schema_kin (None: the kinship softmax), OBRRT_PICK_SCHEMA, _RAND, _KIN_W.
    pub pick: Option<String>,
    pub pick_schema: Option<String>,
    pub pick_rand: f64,
    pub pick_kin_w: f64,
    /// OBRRT_SAMPLE_CHECK
    pub sample_check: bool,
}

impl Default for Research {
    fn default() -> Self {
        Research {
            speed: Speed::default(),
            rh: Rh::default(),
            qd: false,
            qd_gammas: None,
            pick: None,
            pick_schema: None,
            pick_rand: 0.1,
            pick_kin_w: 4.0,
            sample_check: false,
        }
    }
}
