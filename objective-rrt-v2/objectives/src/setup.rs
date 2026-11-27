//! The domain set-up obrrt2 does from its environment, shared with the other
//! drivers (spec's `legs`): the steering controller, the movement library
//! and the technique set (`OBRRT_TECH`, "all" by default). Switches read
//! lazily where they are used need nothing here.

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
    // Steering's ground/air mode goes through the movement library.
    crate::moves::install();
    // OBRRT_TECH: which speedrun-technique objectives are on (extra/tech/mod.rs).
    crate::extra::tech::configure(&o.tech)?;
    Ok((o.tech.clone(), crate::extra::tech::enabled_names()))
}
