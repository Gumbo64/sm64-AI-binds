//! A position goal set from outside the search (the SM64Coin web miner's visualiser: shift + right
//! click on level geometry): the generator draws `Mario::PosTime` objectives to this point from the
//! states of its zone (see `objectives/src/lib.rs` `Registry::generate`, `mario::PosTime::goal`).
//! It biases which objectives get drawn; nothing forces a route.

use std::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosGoal {
    /// (level, area) the point is in.
    pub zone: (u16, u8),
    pub pos: [f32; 3],
}

static GOAL: RwLock<Option<PosGoal>> = RwLock::new(None);

/// Sets (or clears) the goal.
pub fn set(g: Option<PosGoal>) {
    *GOAL.write().unwrap() = g;
}

/// The current goal, if any.
pub fn get() -> Option<PosGoal> {
    *GOAL.read().unwrap()
}
