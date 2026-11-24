//! Search settings in one place. Defaults can be overridden by environment
//! variables (kept from the earlier version, so scripts still work):
//! `OSS_REPLAY_MAX`, `OSS_FIT`, `OSS_SELECT=flat`, `OSS_K`.

use sm64_ai::noise::Variant;

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
}

fn env<T: std::str::FromStr>(name: &str) -> Option<T> {
    obrrt_core::sys::env::var(name).ok().and_then(|s| s.parse().ok())
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
        }
    }

    /// [`Config::new`] with the `OSS_*` environment overrides applied.
    pub fn from_env(variant: Variant) -> Config {
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
        if obrrt_core::sys::env::var("OSS_SELECT").is_ok_and(|v| v == "flat") {
            c.grouped = false;
        }
        c
    }
}
