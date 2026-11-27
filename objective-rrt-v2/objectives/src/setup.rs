//! The domain set-up obrrt2 does from its environment, shared with the other
//! drivers (spec's `legs`): the steering controller, the movement library
//! (`OBRRT_MOVES=1`) or only its policies (`OBRRT_MOVE_GA=1` without it),
//! and the technique set
//! (`OBRRT_TECH`, "all" by default). Switches read lazily where they are used
//! (OBRRT_CALM, _LAKITU_ARC, _PAINTING_FAST, _GP_CATCH, _SWIM_HOLD, the
//! search's OBRRT_MOVE_GA*) need nothing here.

/// Installs it all; returns the OBRRT_TECH value and the enabled technique names.
pub fn from_env() -> Result<(String, Vec<&'static str>), String> {
    // The environment's switches become the process's config (the search's `envconfig::from_env`
    // does the same; both read the one environment).
    let cfg = obrrt_core::config::from_env(obrrt_core::config::SearchConfig::default()).map_err(|e| format!("{e:#}"))?;
    from_config(obrrt_core::config::install(cfg))
}

/// The same set-up from a config (`cfg` is the installed one).
pub fn from_config(cfg: &obrrt_core::config::SearchConfig) -> Result<(String, Vec<&'static str>), String> {
    let o = &cfg.objectives;
    crate::moves::install_controller();
    // OBRRT_MOVES=1: steering's ground/air mode through the movement library.
    if o.moves {
        crate::moves::install();
    }
    // OBRRT_MOVE_GA=1 without OBRRT_MOVES: the library offers only its
    // policies (search/src/movegenome.rs), the default steering stays generic.
    if cfg.ga.move_ga && !o.moves {
        crate::moves::install_policies();
    }
    // OBRRT_TECH: which speedrun-technique objectives are on (extra/tech/mod.rs).
    crate::extra::tech::configure(&o.tech)?;
    Ok((o.tech.clone(), crate::extra::tech::enabled_names()))
}
