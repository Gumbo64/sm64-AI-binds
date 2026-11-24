//! The domain set-up obrrt2 does from its environment, shared with the other
//! drivers (spec's `legs`): the steering controller, the movement library
//! (`OBRRT_MOVES=1`) or only its policies (`OBRRT_MOVE_GA=1` without it),
//! the dialog/menu policies (`OBRRT_FAST_MODAL`) and the technique set
//! (`OBRRT_TECH`, "all" by default). Switches read lazily where they are used
//! (OBRRT_CALM, _LAKITU_ARC, _PAINTING_FAST, _GP_CATCH, _SWIM_HOLD, the
//! search's OBRRT_MOVE_GA*) need nothing here.

/// Installs it all; returns the OBRRT_TECH value and the enabled technique names.
pub fn from_env() -> Result<(String, Vec<&'static str>), String> {
    crate::moves::install_controller();
    // OBRRT_FAST_MODAL=0: the old dialog/menu macros (A every 4th frame)
    // instead of the frame-optimal policies (menu.rs).
    if let Ok(v) = obrrt_core::sys::env::var("OBRRT_FAST_MODAL") {
        crate::set_fast_modal(v != "0");
    }
    let on = |k: &str| obrrt_core::sys::env::var(k).is_ok_and(|v| v == "1");
    // OBRRT_MOVES=1: steering's ground/air mode through the movement library.
    if on("OBRRT_MOVES") {
        crate::moves::install();
    }
    // OBRRT_MOVE_GA=1 without OBRRT_MOVES: the library offers only its
    // policies (search/src/movegenome.rs), the default steering stays generic.
    if on("OBRRT_MOVE_GA") && !on("OBRRT_MOVES") {
        crate::moves::install_policies();
    }
    // OBRRT_TECH: which speedrun-technique objectives are on (extra/tech/mod.rs).
    let tech = obrrt_core::sys::env::var("OBRRT_TECH").unwrap_or_else(|_| "all".into());
    crate::extra::tech::configure(&tech)?;
    Ok((tech, crate::extra::tech::enabled_names()))
}
