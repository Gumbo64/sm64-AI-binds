//! The game type the rolling-horizon path plays. By default plain
//! `SM64Game`; with feature `sm64coin`, `SM64MiningGame` (the game plus the
//! consensus perturbation), so every step/copy/sync/save/load a worker does
//! is the perturbed one. Passing a mining game where `&mut SM64Game` is
//! typed would silently step it unperturbed, so the RH path uses these.
#[cfg(not(feature = "sm64coin"))]
pub type Game = sm64_binds::SM64Game;
#[cfg(not(feature = "sm64coin"))]
pub type GameSave = sm64_binds::SaveState;
#[cfg(feature = "sm64coin")]
pub type Game = sm64pow::game::SM64MiningGame;
#[cfg(feature = "sm64coin")]
pub type GameSave = sm64pow::game::SM64MiningSave;

/// The plain save inside a [`GameSave`] (for the default-only state tree; it
/// is never reached with `sm64coin`, whose `Worker::new` fails).
#[cfg(not(feature = "sm64coin"))]
pub fn plain(s: &GameSave) -> &sm64_binds::SaveState {
    s
}
#[cfg(feature = "sm64coin")]
pub fn plain(s: &GameSave) -> &sm64_binds::SaveState {
    &s.game
}

/// The plain save of a [`GameSave`], by value.
#[cfg(not(feature = "sm64coin"))]
pub fn into_plain(s: GameSave) -> sm64_binds::SaveState {
    s
}
#[cfg(feature = "sm64coin")]
pub fn into_plain(s: GameSave) -> sm64_binds::SaveState {
    s.game
}
/// The perturbation a save carries (with `sm64coin`; nothing otherwise).
#[cfg(not(feature = "sm64coin"))]
pub type Pert = ();
#[cfg(feature = "sm64coin")]
pub type Pert = sm64pow::game::Perturber;
#[cfg(not(feature = "sm64coin"))]
pub fn pert_of(_: &GameSave) -> Pert {}
#[cfg(feature = "sm64coin")]
pub fn pert_of(s: &GameSave) -> Pert {
    s.pert.clone()
}
#[cfg(not(feature = "sm64coin"))]
pub fn wrap(s: sm64_binds::SaveState, _: &Pert) -> GameSave {
    s
}
#[cfg(feature = "sm64coin")]
pub fn wrap(s: sm64_binds::SaveState, p: &Pert) -> GameSave {
    sm64pow::game::SM64MiningSave { game: s, pert: p.clone() }
}
