//! `OBRRT_FREE_SETUP=1` (off by default; off, every objective is exactly as
//! before): scripted technique objectives with less script.
//!
//! What changes under the switch (GOALS.md, "Free setups", has the audit
//! table and the per-objective list):
//! - fixed setup points become targets with a tolerance: a band on the
//!   state (position along a line, facing, speed) the hand-off has to be in,
//!   not one exact point;
//! - the overlays keep only the frame-critical inputs (an exact-frame press,
//!   the stick on a key frame); the approach (run, jumps, dives) is the
//!   genome's, so the γ-discounted ETA rewards a fast one;
//! - the setup's part of the ETA is in real phases: the approach at PACE,
//!   the turn to the needed facing at the game's turn rates, the run-up
//!   still needed for the hand-off speed, instead of fixed SETUP constants;
//! - each objective gets a bounded setup variant (at most [`MAX_VARIANTS`])
//!   in its key: a side, an approach band, a runway or speed bucket, so the
//!   pool keeps several setups apiece.
//!
//! This file holds the switch and the shared phase helpers (decomp rates).

use std::sync::OnceLock;

use obrrt_core::St;
use obrrt_core::layout::ACT_FLAG_AIR;

/// Whether `OBRRT_FREE_SETUP=1`.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_FREE_SETUP").is_ok_and(|v| v == "1"))
}

/// Setup variants per objective at most (the key's variant field).
pub const MAX_VARIANTS: usize = 4;

/// The walking turn rate: update_walking_speed turns the facing towards the
/// stick by at most 0x800 a frame (mario_actions_moving.c:466), at any speed.
pub const WALK_TURN: f64 = 2048.0;
/// analog_stick_held_back: more than 0x471C (100°) off the facing at 16+
/// starts ACT_TURNING_AROUND, which loses 2 a frame (apply_slope_decel(m, 2)
/// on flat ground) and ends in ACT_FINISH_TURNING_AROUND facing the stick.
const TURN_BACK: i32 = 0x471C;
const TURN_BACK_FV: f32 = 16.0;
const SKID_DECEL: f64 = 2.0;
/// ACT_FINISH_TURNING_AROUND's frames (its animation; rough).
const FINISH_TURN_FRAMES: f64 = 2.0;
/// Top walking speed (update_walking_speed's maxTargetSpeed at full tilt).
pub const RUN: f32 = 32.0;

/// SM64 yaw (s16 range, as i32) of the direction from `a` to `b` (xz).
pub fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// |`a` − `b`| as s16 yaws.
pub fn yaw_diff(a: i32, b: i32) -> i32 {
    ((a - b) as i16 as i32).abs()
}

fn airborne(s: &St) -> bool {
    s.action & ACT_FLAG_AIR != 0
}

/// Frames for Mario in `s` to face `want` on the ground at the game's turn
/// rates: standing (|fv| < 1) one frame (act_idle → begin_walking_action
/// sets the facing to the stick at once); walking 0x800 a frame; more than
/// 100° round at 16+ the skid (fv / 2 frames) and the finish. In the air
/// the facing doesn't turn (update_air_without_turn): the same figure, as
/// the turn waits for the landing.
pub fn turn_frames(s: &St, want: i32) -> f64 {
    let d = yaw_diff(s.yaw as i32, want);
    if d < 0x100 {
        return 0.0;
    }
    if !airborne(s) && s.forward_vel.abs() < 1.0 {
        return 1.0;
    }
    if !airborne(s) && d > TURN_BACK && s.forward_vel >= TURN_BACK_FV {
        return s.forward_vel as f64 / SKID_DECEL + FINISH_TURN_FRAMES;
    }
    d as f64 / WALK_TURN
}

/// Walking speed after `dist` more units from `v0` (update_walking_speed:
/// +1.1 − v/43 a frame up to 32; +1.1 below 0) and the frames that took.
pub fn run_up(v0: f32, dist: f32) -> (f32, f64) {
    let (mut v, mut d, mut k) = (v0.clamp(0.0, RUN), 0.0f32, 0.0f64);
    while d < dist && k < 200.0 {
        if v >= RUN {
            return (v, k + ((dist - d) / RUN) as f64);
        }
        v = (v + 1.1 - v / 43.0).min(RUN);
        d += v.max(1.0);
        k += 1.0;
    }
    (v, k)
}

/// Frames a walk from `v0` needs to reach `need` (update_walking_speed's
/// acceleration), however far that takes.
pub fn speed_frames(v0: f32, need: f32) -> f64 {
    let need = need.min(RUN - 0.1);
    let (mut v, mut k) = (v0.max(0.0), 0.0);
    while v < need && k < 200.0 {
        v = (v + 1.1 - v / 43.0).min(RUN);
        k += 1.0;
    }
    k
}

/// The setup-phase ETA of a hand-off at `at` with facing `yaw` and at least
/// `speed`: the approach at PACE (distance less what the run-up itself
/// covers is the same thing at PACE ~ RUN), the turn onto the hand-off's
/// facing from where Mario will arrive, and any run-up frames the speed
/// still needs beyond the approach. Rough on purpose: it has to fall as the
/// setup progresses and be zero at a hand-off.
pub fn handoff_frames(s: &St, at: [f32; 3], yaw: i32, speed: f32) -> f64 {
    let d = s.hdist(at);
    let approach = obrrt_core::frames_for(d);
    // Facing on arrival: the way to the hand-off when it is still far, else
    // the current facing.
    let arrive = if d > 200.0 { yaw_to(s.pos, at) } else { s.yaw as i32 };
    let turn_now = turn_frames(s, arrive);
    let turn_then = if d > 200.0 { yaw_diff(arrive, yaw) as f64 / WALK_TURN } else { turn_frames(s, yaw) };
    let (v, _) = run_up(s.forward_vel.max(0.0), d);
    let more = if v + 0.05 < speed { speed_frames(v, speed) } else { 0.0 };
    approach + turn_now.min(8.0) + turn_then + more
}

/// The variant index in `0..n` a key field shows (`" fs<i>"`).
pub fn tag(i: usize) -> String {
    format!(" fs{i}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::tech::testkit::{ACT_WALKING, st};

    #[test]
    fn phases_fall() {
        let mut s = st(16, 1, [0.0, 0.0, 0.0], ACT_WALKING);
        s.forward_vel = 32.0;
        // Facing +z (yaw 0), hand-off 1000 ahead facing +z at 30.
        let far = handoff_frames(&s, [0.0, 0.0, 1000.0], 0, 30.0);
        s.pos = [0.0, 0.0, 500.0];
        let mid = handoff_frames(&s, [0.0, 0.0, 1000.0], 0, 30.0);
        s.pos = [0.0, 0.0, 1000.0];
        let at = handoff_frames(&s, [0.0, 0.0, 1000.0], 0, 30.0);
        assert!(far > mid && mid > at && at < 0.5, "{far} {mid} {at}");
        // A skid costs more than a walking turn.
        assert!(turn_frames(&s, 0x8000u16 as i16 as i32) > turn_frames(&s, 0x2000));
        // Standing turns at once.
        s.forward_vel = 0.0;
        assert_eq!(turn_frames(&s, 0x6000), 1.0);
        assert!(speed_frames(0.0, 31.0) > 20.0);
    }
}
