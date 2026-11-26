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

// The goal's strength, set by the page (SM64Coin web miner: the Mine page's settings). Without a goal nothing here is read, so
// native miners are unchanged.
use std::sync::atomic::{AtomicU64, Ordering};
static WEIGHT: AtomicU64 = AtomicU64::new(0);
static SHARE: AtomicU64 = AtomicU64::new(0);

/// The pool weight of the goal's objective (`Mario::UserGoal`) among the pool's entries (a regular objective weighs 1):
/// 0 (unset) reads `OBRRT_POSGOAL_WEIGHT`, default 10.
pub fn weight() -> f64 {
    let w = f64::from_bits(WEIGHT.load(Ordering::Relaxed));
    if w > 0.0 {
        w
    } else {
        crate::sys::env::var("OBRRT_POSGOAL_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0)
    }
}
pub fn set_weight(w: f64) {
    WEIGHT.store(w.max(0.0).to_bits(), Ordering::Relaxed);
}

/// The share of plans (0..1) that start from the tree's state nearest the goal (in the goal's zone) and run the goal's
/// objective; `OBRRT_POSGOAL_SHARE`, default 0 (the web miner sets it from its goal-strength setting).
pub fn share() -> f64 {
    let w = f64::from_bits(SHARE.load(Ordering::Relaxed));
    if w > 0.0 {
        w.min(1.0)
    } else {
        crate::sys::env::var("OBRRT_POSGOAL_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0f64).clamp(0.0, 1.0)
    }
}
pub fn set_share(s: f64) {
    SHARE.store(s.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test (the goal is a process-wide static): no goal and no setting change nothing; the strength is what was set.
    #[test]
    fn goal_and_strength() {
        assert_eq!(get(), None);
        assert_eq!(share(), 0.0, "native default: no forced goal plans");
        assert_eq!(weight(), 10.0);
        set(Some(PosGoal { zone: (6, 1), pos: [1.0, 2.0, 3.0] }));
        assert_eq!(get().unwrap().zone, (6, 1));
        set_weight(24.0);
        set_share(0.4);
        assert_eq!((weight(), share()), (24.0, 0.4));
        set_share(7.0);
        assert_eq!(share(), 1.0, "clamped");
        set(None);
        assert_eq!(get(), None);
    }
}
