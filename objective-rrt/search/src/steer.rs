//! The steering controller's entry: inputs that take Mario towards an
//! objective's target. The controller itself (per control mode, the
//! movement library, the GA's move policies) is the registered
//! [`obrrt_core::mover::Mover`] (obrrt-objectives' `moves`, which the run
//! binary always registers); the search only asks it. Without one, a
//! neutral pad.

use obrrt_core::{GamePad, Objective, St};

/// Inputs towards `obj`'s target for one frame from state `st`; `frame`
/// counts frames into the rollout (for button rhythms), `jump_every` is a
/// period for the controller's generic A rhythm.
pub fn steer(obj: &dyn Objective, st: &St, frame: usize, jump_every: usize) -> GamePad {
    steer_policy(obj, st, frame, jump_every, None)
}

/// [`steer`] playing the mover's policy `policy` where it applies (the GA's
/// policy-seeded rollouts, movegenome.rs). `None` is [`steer`].
pub fn steer_policy(obj: &dyn Objective, st: &St, frame: usize, jump_every: usize, policy: Option<usize>) -> GamePad {
    let Some(target) = obj.steer_target(st) else { return GamePad::new(0, 0, 0) };
    match obrrt_core::mover::mover() {
        Some(m) => m.control(st, target, obj.arrival(st), obj.steer_button(st), frame, jump_every, policy),
        None => GamePad::new(0, 0, 0),
    }
}
