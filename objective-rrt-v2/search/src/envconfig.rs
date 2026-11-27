//! The tree's settings as one typed [`SearchConfig`] (the `obrrt-env` crate): [`from_env`] reads
//! the `OBRRT_*` variables, as `obrrt2` does (run/src/main.rs), for every program that runs the
//! engine (obrrt2, sm64-crypto3's miner); [`apply_globals`] puts a config into the engine's
//! [`Config`]. The engine runs its threads under the config ([`Config::search`]), so nothing here
//! is process-wide state.
use anyhow::Result;
use obrrt_core::St;
use obrrt_core::config::SearchConfig;

use crate::Config;

/// Sets `cfg` from `OBRRT_*` and hands it the environment's config. Also installs that config as the
/// process's ([`obrrt_core::config::install`]), for the programs that read it outside an engine.
pub fn from_env(cfg: &mut Config) -> Result<()> {
    let sc = obrrt_core::config::from_env(SearchConfig::default())?;
    apply_globals(obrrt_core::config::install(sc), cfg)
}

/// Puts `sc` into `cfg`: its engine-level settings, and the config itself for the engine's threads.
/// Call before the engine is built.
pub fn apply_globals(sc: &'static SearchConfig, cfg: &mut Config) -> Result<()> {
    let (b, s, k, ga) = (&sc.budget, &sc.sampler, &sc.kin, &sc.ga);
    cfg.search = Some(sc);
    // budget
    cfg.max_frames = b.max_frames;
    if let Some(mb) = b.mem_mb {
        cfg.mem_budget = mb * 1_000_000;
    }
    cfg.round_plans = b.round_plans;

    // GA
    cfg.ga_pink = ga.pink;

    // sampler
    cfg.new_eligible = s.new_eligible;
    cfg.kinship_mult = k.mult;
    Ok(())
}

/// Kept for callers that still call it with the root state: the engine reads the zone lock and the
/// walkable-mesh switch (`OBRRT_GEO`) from its config itself, so there is nothing left to do.
pub fn after_root(_root: &St) {}
