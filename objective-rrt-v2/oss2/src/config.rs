//! Search settings in one place. Defaults can be overridden by environment
//! variables (kept from objective-scattershot, so scripts still work):
//! `OSS_REPLAY_MAX`, `OSS_FIT`, `OSS_SELECT=flat`, `OSS_K`; and oss2's GA
//! switches (see [`GaConfig`] and README.md): `OSS_GA`, `OSS_GA_BY`,
//! `OSS_GA_HARVEST`, `OSS_GA_LEARN`, plus obrrt2's own GA knobs
//! `OBRRT_GA_SCALE`, `OBRRT_GA_NEW`, `OBRRT_PINK`, `OBRRT_GA_BUDGET`
//! (read as obrrt2 and goalbench read them, same defaults).

use obrrt_search::ga::GaParams;
use objective_scattershot::noise::Variant;

/// Which states of a GA expansion are offered to the archive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Harvest {
    /// Every state along the winner's committed inputs (what obrrt2 keeps as
    /// the attempt's segment, cut at its fitness maximum, plus the automatic
    /// tail through cutscenes).
    Winner,
    /// Those, and the end state of every rollout of the GA's last generation.
    Ends,
    /// Those, and every state along every last-generation rollout.
    All,
}

impl Harvest {
    pub fn name(self) -> &'static str {
        match self {
            Harvest::Winner => "winner",
            Harvest::Ends => "ends",
            Harvest::All => "all",
        }
    }
}

/// The GA expansion's settings.
#[derive(Clone, Copy, Debug)]
pub struct GaConfig {
    /// `OSS_GA=<p>` (0): the share of expansions made by the GA; the rest are
    /// pellets as in objective-scattershot. Decided per batch (all of a
    /// batch's expansions are GA or none are), so the threads stay balanced: a
    /// GA expansion plays ~200x a pellet's frames.
    pub share: f64,
    /// `OSS_GA_BY=frames`: `share` is the share of emulated frames instead
    /// (a GA batch whenever GA frames so far < share x all frames).
    pub by_frames: bool,
    /// `OSS_GA_HARVEST=winner|ends|all` (winner).
    pub harvest: Harvest,
    /// `OSS_GA_LEARN=0|1` (1): weight objectives by `obj_weight` x their
    /// kind's success rate in this run, (successes + 1) / (expansions + 2),
    /// a success being an expansion that added or improved another cell.
    pub learn: bool,
    /// The GA's own parameters, as obrrt2 uses them.
    pub params: GaParams,
}

#[derive(Clone, Copy)]
pub struct Config {
    /// How the random inputs of a rollout correlate over time (pink by default).
    pub variant: Variant,
    /// A state gets a full save only when rebuilding it would replay more than
    /// this many frames from its nearest saved ancestor; the others are rebuilt
    /// by replaying inputs (the game is deterministic). Measured flat from 64 to
    /// 128; 128 uses half the memory of 64.
    pub replay_max: u32,
    /// Share of rollouts whose noise is fitted to the inputs that led to the
    /// source cell; the rest start fresh.
    pub fit_prob: f64,
    /// Frames of earlier inputs the noise is fitted to.
    pub fit_frames: usize,
    /// Candidates drawn per Thompson selection.
    pub candidates: usize,
    /// Two-level selection: a stage group first, then a cell in it.
    pub grouped: bool,
    /// Rollout length range, in frames.
    pub min_len: usize,
    pub max_len: usize,
    pub ga: GaConfig,
    /// Speed mode's switches (`OSS_CORRIDOR`, `OSS_SELECT=route`,
    /// `OSS_GA_TARGET`, `OSS_GA_SEED`, `OSS_PROP`; see `speed`). All off by default.
    pub speed: crate::speed::SpeedConfig,
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

/// obrrt2's GA parameters: `Config::default` in search/src/engine.rs with
/// run/src/main.rs's environment switches (as run/src/goalbench.rs reads them).
pub fn obrrt2_ga_params() -> GaParams {
    GaParams { scale: 3.0, pink: env("OBRRT_PINK").unwrap_or(0.0) }
}

impl Config {
    pub fn new(variant: Variant) -> Config {
        Config {
            variant,
            replay_max: 128,
            fit_prob: 0.75,
            fit_frames: 256,
            candidates: 256,
            grouped: true,
            min_len: 60,
            max_len: 200,
            ga: GaConfig { share: 0.0, by_frames: false, harvest: Harvest::Winner, learn: true, params: obrrt2_ga_params() },
            speed: crate::speed::SpeedConfig { ga_seed: true, ..Default::default() },
        }
    }

    /// [`Config::new`] with the `OSS_*` environment overrides applied.
    pub fn from_env(variant: Variant) -> anyhow::Result<Config> {
        let mut c = Config::new(variant);
        if let Some(v) = env("OSS_REPLAY_MAX") {
            c.replay_max = v;
        }
        if let Some(v) = env("OSS_FIT") {
            c.fit_prob = v;
        }
        if let Some(v) = env("OSS_K") {
            c.candidates = v;
        }
        if std::env::var("OSS_SELECT").is_ok_and(|v| v == "flat") {
            c.grouped = false;
        }
        if let Ok(v) = std::env::var("OSS_GA") {
            let p: f64 = v.parse()?;
            anyhow::ensure!((0.0..=1.0).contains(&p), "OSS_GA: a share in 0..=1, not {v}");
            c.ga.share = p;
        }
        match std::env::var("OSS_GA_BY").as_deref() {
            Err(_) | Ok("expansions") => {}
            Ok("frames") => c.ga.by_frames = true,
            Ok(v) => anyhow::bail!("OSS_GA_BY: expansions or frames, not {v}"),
        }
        c.ga.harvest = match std::env::var("OSS_GA_HARVEST").as_deref() {
            Err(_) | Ok("winner") => Harvest::Winner,
            Ok("ends") => Harvest::Ends,
            Ok("all") => Harvest::All,
            Ok(v) => anyhow::bail!("OSS_GA_HARVEST: winner, ends or all, not {v}"),
        };
        if let Ok(v) = std::env::var("OSS_GA_LEARN") {
            c.ga.learn = v != "0";
        }
        c.speed = crate::speed::SpeedConfig::from_env()?;
        Ok(c)
    }
}
