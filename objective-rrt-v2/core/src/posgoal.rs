//! A sphere bubble set from outside the search (the SM64Coin web miner's visualiser: hold left click on level
//! geometry and scroll): a share (`share()`) of the plans draw their base state only from live tree nodes inside it
//! (`Engine::plan_bubble`). Without nodes inside, or without a bubble, nothing changes.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosGoal {
    /// (level, area) the point is in.
    pub zone: (u16, u8),
    /// The bubble's centre.
    pub pos: [f32; 3],
    /// The bubble's radius (game units).
    pub radius: f32,
}

pub const MIN_RADIUS: f32 = 50.0;
pub const MAX_RADIUS: f32 = 20000.0;
pub const DEFAULT_RADIUS: f32 = 300.0;

/// One position goal and its strength: what a [`crate::runtime::Runtime`] holds.
pub struct GoalState {
    goal: RwLock<Option<PosGoal>>,
    // The goal's strength, set by the page (SM64Coin web miner: the Mine page's settings). Without a goal nothing here is read, so
    // native miners are unchanged.
    weight: AtomicU64,
    share: AtomicU64,
}

impl Default for GoalState {
    fn default() -> Self {
        GoalState { goal: RwLock::new(None), weight: AtomicU64::new(0), share: AtomicU64::new(0) }
    }
}

impl GoalState {
    /// Sets (or clears) the goal.
    pub fn set(&self, g: Option<PosGoal>) {
        *self.goal.write().unwrap() = g;
    }

    /// The current goal, if any.
    pub fn get(&self) -> Option<PosGoal> {
        *self.goal.read().unwrap()
    }

    /// The pool weight of the goal's objective (`Mario::UserGoal`) among the pool's entries (a regular objective weighs 1):
    /// 0 (unset) reads `OBRRT_POSGOAL_WEIGHT`, default 10.
    pub fn weight(&self) -> f64 {
        let w = f64::from_bits(self.weight.load(Ordering::Relaxed));
        if w > 0.0 {
            w
        } else {
            crate::sys::env::var("OBRRT_POSGOAL_WEIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0)
        }
    }

    pub fn set_weight(&self, w: f64) {
        self.weight.store(w.max(0.0).to_bits(), Ordering::Relaxed);
    }

    /// The share of plans (0..1) whose base state is drawn from inside the bubble (when it holds nodes);
    /// `OBRRT_POSGOAL_SHARE`, default 0 (the web miner sets it from its bubble-strength setting).
    pub fn share(&self) -> f64 {
        let w = f64::from_bits(self.share.load(Ordering::Relaxed));
        if w > 0.0 {
            w.min(1.0)
        } else {
            crate::sys::env::var("OBRRT_POSGOAL_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0f64).clamp(0.0, 1.0)
        }
    }

    pub fn set_share(&self, s: f64) {
        self.share.store(s.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
}

// The module-level functions are thin wrappers over this thread's runtime's goal ([`crate::runtime::with`]): the engine's inside
// its threads, the process-wide one elsewhere.

/// Sets (or clears) the goal.
pub fn set(g: Option<PosGoal>) {
    crate::runtime::with(|rt| rt.goal.set(g));
}

/// The current goal, if any.
pub fn get() -> Option<PosGoal> {
    crate::runtime::with(|rt| rt.goal.get())
}

/// See [`GoalState::weight`].
pub fn weight() -> f64 {
    crate::runtime::with(|rt| rt.goal.weight())
}

pub fn set_weight(w: f64) {
    crate::runtime::with(|rt| rt.goal.set_weight(w));
}

/// See [`GoalState::share`].
pub fn share() -> f64 {
    crate::runtime::with(|rt| rt.goal.share())
}

pub fn set_share(s: f64) {
    crate::runtime::with(|rt| rt.goal.set_share(s));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test (it uses the process-wide runtime): no goal and no setting change nothing; the strength is what was set.
    #[test]
    fn goal_and_strength() {
        assert_eq!(get(), None);
        assert_eq!(share(), 0.0, "native default: no forced goal plans");
        assert_eq!(weight(), 10.0);
        set(Some(PosGoal { zone: (6, 1), pos: [1.0, 2.0, 3.0], radius: 300.0 }));
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
