//! oss2, "scattershot v2": objective-scattershot's Go-Explore archive (one
//! state per schema key, Thompson selection by stage group, first-or-faster
//! merge, sparse checkpoint saves) with a share of its random-input rollouts
//! ("pellets") replaced by objective-rrt-v2's GA towards a generated
//! objective, whose fitness is the discounted decrease of the objective's
//! per-state ETA. See README.md.
//!
//! - `objective`, `objectives`, `util`: the schema tree and the key-writing
//!   kinds, re-exported from the vendored objective-scattershot crate
//!   (`../scattershot`, which obrrt-search uses too): not duplicated here.
//! - `archive`: cells, the input tree, checkpoint saves, groups, selection.
//! - `store`: compact save states.
//! - `inputs`: pellet input operators.
//! - `ga`: the GA expansion (`OSS_GA`).
//! - `search`: the loop, stats and output.
//! - `speed`: speed mode (`--pipe`, `OSS_CORRIDOR`, `OSS_SELECT=route`,
//!   `OSS_GA_TARGET=route`, `OSS_PROP`): make a piped obrrt2 route faster.
//! - `share`: input sharing between zone entries (the `oss_share` example).
//! - `config`, `run`: settings and the command-line run.

pub use objective_scattershot::{objective, objectives, util};

pub mod archive;
pub mod config;
pub mod ga;
pub mod inputs;
pub mod run;
pub mod search;
pub mod share;
pub mod speed;
pub mod store;

#[cfg(test)]
pub mod testutil;
#[cfg(test)]
mod tests;
