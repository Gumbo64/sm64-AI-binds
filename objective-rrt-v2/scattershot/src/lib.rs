//! objective-scattershot: a Go-Explore style search over SM64 states. The
//! archive keeps one state per distinct key (what the schema's objectives
//! write, see `objective`), expands cells chosen by Thompson sampling within
//! stage groups with random-ish inputs (`inputs`), and keeps whatever reaches a
//! key first or faster.
//!
//! - `objective`, `objectives`: the schema tree and the objective kinds.
//! - `archive`: cells, the input tree, checkpoint saves, groups, selection.
//! - `store`: compact save states.
//! - `inputs`: rollout input operators.
//! - `search`: the loop, stats and output.
//! - `share`: input sharing between zone entries (the `oss_share` example).
//! - `config`, `run`: settings and the command-line run.

pub mod archive;
pub mod config;
pub mod inputs;
pub mod noise;
pub mod objective;
pub mod objectives;
pub mod run;
pub mod search;
pub mod share;
pub mod store;
pub mod util;

#[cfg(test)]
pub mod testutil;
#[cfg(test)]
mod tests;
