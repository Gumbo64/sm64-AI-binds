//! How a rollout's inputs are made from the cell it starts at.
//!
//! - **PressThrough** for a busy state (a door animation, a dialog, a menu):
//!   the stick neutral or held for yes / no, A every 4th frame, as
//!   objective-rrt does for dialogs.
//! - **Fitted** noise: the noise generator fitted to the inputs that led to the
//!   cell, then sampled, so the new inputs carry on from the old ones.
//! - **Noise**: a fresh generator.
//!
//! Each operator's rollouts and successes are counted in `Search::operators`;
//! more operators (e.g. steering toward a goal) slot in here.

use rand::Rng;
use rand::rngs::SmallRng;
use sm64_ai::noise::Slot;
use sm64_binds::GamePad;

use crate::archive::Archive;
use crate::config::Config;
use crate::objectives::util::press_through;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Operator {
    Noise,
    Fitted,
    PressThrough,
}

impl Operator {
    pub const ALL: [Operator; 3] = [Operator::Noise, Operator::Fitted, Operator::PressThrough];

    pub fn name(self) -> &'static str {
        match self {
            Operator::Noise => "noise",
            Operator::Fitted => "fitted",
            Operator::PressThrough => "press_through",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// The inputs for a rollout from cell `src`, and the operator that made them.
pub fn plan(archive: &Archive, cfg: &Config, src: usize, rng: &mut SmallRng) -> (Operator, Vec<GamePad>) {
    let len = rng.random_range(cfg.min_len..=cfg.max_len);
    let cell = &archive.cells[src];
    if cell.busy {
        let x = [0, -127, 127][rng.random_range(0..3)];
        return (Operator::PressThrough, press_through(x, 0, len));
    }
    let (op, mut slot) = if rng.random_bool(cfg.fit_prob) {
        (Operator::Fitted, Slot::fit(rng, cfg.variant, &archive.tail(cell.node as usize, cfg.fit_frames)))
    } else {
        (Operator::Noise, Slot::new(rng, cfg.variant))
    };
    (op, (0..len).map(|_| slot.next(rng)).collect())
}
