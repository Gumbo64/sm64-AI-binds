//! Poles and trees as GA policies (`OBRRT_MOVE_GA`), plausible only on a
//! pole (ACT_FLAG_ON_POLE).
//!
//! Inputs (mario_actions_automatic.c), all on the raw stick:
//!
//! - holding (act_holding_pole): Z (or low health) lets go (soft bonk); A
//!   turns him around (`yaw += 0x8000`) and kicks off (ACT_WALL_KICK_AIR:
//!   62 up, at least 24 forward); stickY > 16 climbs while below the top
//!   (`hitboxHeight - 100`), and at the top > 50 stands on it
//!   (ACT_TOP_OF_POLE_TRANSITION, not on bhvGiantPole); stickY < -16 slides
//!   down, spinning (the spin `-= 2 stickY` a frame up to 0x1000, the drop
//!   `spin / 0x100` up to 16 a frame); otherwise `yaw -= 16 stickX`.
//! - climbing (act_climbing_pole): A kicks off as above; stickY < 8 goes
//!   back to holding (the same frame); else up `stickY / 8` (8 a frame at
//!   full tilt) and the facing turns 0x400 a frame towards the camera's yaw
//!   (the only camera dependence on a pole). Holding with stickY > 16 starts
//!   climbing in the same frame, so a frame of stick release costs one
//!   frame's 8; but only a holding frame stands Mario up at the top, and the
//!   state doesn't show the top. The existing controller (modes.rs `pole`)
//!   releases one frame in 4 (6 a frame); the policies one in 16 (7.5).
//! - top (act_top_of_pole): A jumps along the facing (ACT_TOP_OF_POLE_JUMP,
//!   same speeds as the kick-off), stickY < -16 goes back down, else
//!   `yaw -= 16 stickX`.
//!
//! The kick-off arc (measured, run/examples/modes_lab.rs `pole`, saved in
//! ref/arcs/modes.json, WF's pole and a castle-grounds tree): 62 up less 4
//! a frame with A held (peak 512 at 15 frames, back level at 31), the
//! forward speed 24 growing to 35 with the stick forward: 881 out at 27
//! frames (266 up), 1191 at 35 (280 down), a mean of 32-34 a frame.
//! Climbing measured 174 in 18 frames to the handstand on both (short
//! poles: every release pattern alike).
//!
//! - `pole-top`: climb to the top at 7.5 a frame, stand, turn to the target
//!   and jump from the handstand;
//! - `pole-kick`: climb (or slide) to the height the kick-off arc reaches
//!   the target from, face away from it, kick off;
//! - `pole-drop`: a target below: slide down at full speed to its height,
//!   then let go (Z) over it.

use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::layout::{
    ACT_CLIMBING_POLE, ACT_FLAG_ON_POLE, ACT_HOLDING_POLE, ACT_TOP_OF_POLE,
};
use obrrt_core::{GamePad, St};

use super::modes::{angle, pad, wrap};

/// Kick-off speeds (set_mario_action_airborne, ACT_WALL_KICK_AIR /
/// ACT_TOP_OF_POLE_JUMP) and the mean forward speed over the arc with the
/// stick forward (measured).
pub const KICK_VY: f32 = 62.0;
pub const KICK_FV: f32 = 32.0;

/// Height of the kick-off arc `t` frames out (A held: gravity 4).
pub fn kick_h(t: f32) -> f32 {
    KICK_VY * t - 2.0 * t * (t - 1.0)
}

/// The height to kick off from to come down at `target` (horizontal
/// distance `hd`), at most 600 of fall assumed.
pub fn kick_height(target: [f32; 3], hd: f32) -> f32 {
    let t = hd / KICK_FV;
    target[1] - kick_h(t).max(-600.0)
}

/// The pole policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pole {
    Top,
    Kick,
    Drop,
}

pub fn on_pole(st: &St) -> bool {
    st.action & ACT_FLAG_ON_POLE != 0
}

/// Stick x for `yaw -= 16 stickX` to close `err` within a frame or two.
fn turn(err: i16) -> f32 {
    (-(err as f32) / 16.0).clamp(-64.0, 64.0)
}

/// Climbing at full tilt with one release frame in 16.
fn climb(frame: usize) -> GamePad {
    pad(0, 0.0, if frame % 16 == 15 { 0.0 } else { 64.0 })
}

pub fn pole_pad(k: Pole, st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    if !on_pole(st) {
        return None;
    }
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt();
    let to_target = angle(dx, dz) as i32;
    let press = frame % 2 == 0;
    let a = st.action;
    match k {
        Pole::Top => match a {
            ACT_TOP_OF_POLE => {
                let err = wrap(to_target - st.yaw as i32);
                let aimed = err.unsigned_abs() < 0x600;
                Some(pad(if aimed && press { A_BUTTON } else { 0 }, turn(err), 0.0))
            }
            ACT_HOLDING_POLE | ACT_CLIMBING_POLE => Some(climb(frame)),
            _ => Some(GamePad::new(0, 0, 0)),
        },
        Pole::Kick => {
            let dh = kick_height(target, hd) - st.pos[1];
            match a {
                ACT_TOP_OF_POLE => {
                    let err = wrap(to_target - st.yaw as i32);
                    let aimed = err.unsigned_abs() < 0x600;
                    Some(pad(if aimed && press { A_BUTTON } else { 0 }, turn(err), 0.0))
                }
                ACT_HOLDING_POLE | ACT_CLIMBING_POLE => {
                    if dh > 30.0 {
                        return Some(climb(frame));
                    }
                    if dh < -150.0 && a == ACT_HOLDING_POLE {
                        return Some(pad(0, 0.0, if dh < -300.0 { -64.0 } else { -20.0 }));
                    }
                    // Face away from the target (A turns him round), kick.
                    let err = wrap(to_target + 0x8000 - st.yaw as i32);
                    let aimed = err.unsigned_abs() < 0x600;
                    Some(pad(if aimed && press && a == ACT_HOLDING_POLE { A_BUTTON } else { 0 }, turn(err), 0.0))
                }
                _ => Some(GamePad::new(0, 0, 0)),
            }
        }
        Pole::Drop => {
            let dh = target[1] - st.pos[1];
            match a {
                ACT_TOP_OF_POLE => Some(pad(0, 0.0, -64.0)),
                ACT_CLIMBING_POLE => Some(pad(0, 0.0, 0.0)),
                ACT_HOLDING_POLE => {
                    // Slide at full speed, easing off within 200; let go
                    // within 60 of the target's height (or it beside the
                    // pole and below).
                    if dh > -60.0 || hd > 150.0 && dh > -250.0 {
                        Some(GamePad::new(if press { Z_TRIG } else { 0 }, 0, 0))
                    } else {
                        Some(pad(0, 0.0, if dh < -200.0 { -64.0 } else { -24.0 }))
                    }
                }
                _ => Some(GamePad::new(0, 0, 0)),
            }
        }
    }
}

pub fn pole_plausible(k: Pole, st: &St, target: [f32; 3]) -> bool {
    if !on_pole(st) {
        return false;
    }
    let dy = target[1] - st.pos[1];
    let hd = st.hdist(target);
    match k {
        // Up: a target above, or far (height for the jump).
        Pole::Top => dy > 100.0 || hd > 400.0,
        Pole::Kick => hd > 60.0,
        Pole::Drop => dy < -100.0 && hd < 600.0,
    }
}
