//! Reactive movement policies.
//!
//! Layers (one module each, see PLAN.md):
//! - [`menu`]: the reduced input space. Per state, which button sets and which
//!   stick regions can change what happens next; everything else is removed.
//! - [`skills`]: a library of small reactive controllers (each frame: observe,
//!   return a [`types::Choice`]).
//! - [`mpc`]: derivative-free (CEM etc.) refinement of the continuous part of
//!   the stick inside a stick region, by short rollouts.
//! - [`policy`]: the genome (rules that switch skills from observations each
//!   frame) and its GA; [`bench`] compares it with the pink-noise GA.

pub mod actions;
pub mod bench;
pub mod dsl;
pub mod eval;
pub mod evo;
#[cfg(feature = "ga")]
pub mod hybrid;
pub mod inv;
pub mod menu;
pub mod mpc;
#[cfg(feature = "ga")]
pub mod plga;
pub mod policy;
#[cfg(feature = "ga")]
pub mod repolicy;
#[cfg(feature = "ga")]
pub mod seed;
pub mod sense;
pub mod skills;
pub mod types;
