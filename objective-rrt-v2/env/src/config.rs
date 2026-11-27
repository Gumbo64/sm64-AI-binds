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
    /// OBRRT_MAX_FRAMES: stop after emulating this many frames.
    pub max_frames: Option<u64>,
    /// OBRRT_MEM_MB: memory budget of saves, in MB (None: the engine's 3 GB).
    pub mem_mb: Option<usize>,
    /// OBRRT_ROUND: attempts planned per round.
    pub round_plans: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget { max_frames: None, mem_mb: None, round_plans: 64 }
    }
}

/// Which objective, from which start: weights and who gets picked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampler {
    /// OBRRT_PICK_BY_B
    pub pick_by_b: bool,
    /// OBRRT_NEW_ELIGIBLE
    pub new_eligible: bool,
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler {
            pick_by_b: false,
            new_eligible: true,
        }
    }
}

/// What counts as the same state, and what the tree retires.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dedup {
    /// Dense adds (OBRRT_ADD=last_gen): the final state of every rollout of an attempt's last
    /// generation is a candidate node too, kept when new by the 64-unit dense key.
    pub dense_adds: bool,
}

impl Default for Dedup {
    fn default() -> Self {
        Dedup { dense_adds: true }
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
}

impl Default for Zone {
    fn default() -> Self {
        Zone {
            lock: false,
            path: String::new(),
        }
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
}

impl Default for GoalFocus {
    fn default() -> Self {
        GoalFocus { share: 0.0, explore: false, wide: false, stall_eta: None }
    }
}

/// Kinship generations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Kin {
    /// OBRRT_KINSHIP_MULT
    pub mult: f64,
}

impl Default for Kin {
    fn default() -> Self {
        Kin { mult: 10.0 }
    }
}

/// The GA that runs each attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Ga {
    /// OBRRT_PINK
    pub pink: f64,
    /// OBRRT_GA_GAMMA: the open GA's discount per frame.
    pub gamma: f64,
    /// OBRRT_GA_REGRET (frames; 0: off)
    pub regret: usize,
    /// OBRRT_GA_KEEP_SEED
    pub keep_seed: bool,
    /// OBRRT_CALM
    pub calm: bool,
    /// OBRRT_MOVE_GA / OBRRT_MOVE_GA_EXTRA
    pub move_ga: bool,
    pub move_ga_extra: bool,
}

impl Default for Ga {
    fn default() -> Self {
        Ga {
            pink: 0.0,
            gamma: 0.99,
            regret: 0,
            keep_seed: false,
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
    /// OBRRT_ROAD: "" / "0": off, "1": on without a directory, else the directory.
    pub road: Option<String>,
}

impl Default for Priors {
    fn default() -> Self {
        Priors {
            reach_prior: None,
            rescue_prior: None,
            road: None,
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

/// The domain knowledge's switches (the objectives and the movement library).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectiveConfig {
    /// OBRRT_TECH: technique objectives on ("all", or a list; see extra/tech/mod.rs).
    pub tech: String,
    /// OBRRT_MOVES / OBRRT_MOVES_EXTRA / OBRRT_MOVES_MODES / OBRRT_MOVES_UPHILL
    pub moves: bool,
    pub moves_extra: bool,
    pub moves_modes: bool,
    pub swim_hold: bool,
    pub swim_targets: bool,
    pub gp_catch: bool,
    pub lakitu_arc: bool,
    pub lakitu_through_door: bool,
    pub painting_fast: bool,
    pub grounds_fix: bool,
    /// OBRRT_MOVE_WEIGHT=<x>: the 0.5-weight techniques weigh x (None: unchanged).
    pub move_weight: Option<f64>,
    pub king_landing: bool,
    /// OBRRT_BLJ_*
    pub blj_detector: Option<String>,
    pub blj_spots: bool,
    pub blj_spots_catalog: Option<String>,
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
            moves: true,
            moves_extra: false,
            moves_modes: false,
            swim_hold: true,
            swim_targets: false,
            gp_catch: true,
            lakitu_arc: true,
            lakitu_through_door: false,
            painting_fast: true,
            grounds_fix: false,
            move_weight: Some(0.75),
            king_landing: false,
            blj_detector: None,
            blj_spots: false,
            blj_spots_catalog: None,
            twostar: false,
            static_id: false,
            geo: false,
        }
    }
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
    pub rh: Rh,
    /// OBRRT_GA_QD=1 and OBRRT_GA_GAMMAS=<g,g,...>
    pub qd: bool,
    pub qd_gammas: Option<String>,
}

impl Default for Research {
    fn default() -> Self {
        Research {
            rh: Rh::default(),
            qd: false,
            qd_gammas: None,
        }
    }
}
