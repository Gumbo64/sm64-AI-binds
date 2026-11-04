//! The 120-star TAS's most used speed techniques missing from the policies
//! (ref/techniques/movement.md part C), as GA policies (`OBRRT_MOVE_GA`),
//! behind the same switch as the mode policies (`OBRRT_MOVES_MODES=1`).
//!
//! - `speed-kick` (A29; ~77 TAS uses): walking at fv >= 29 with A held
//!   (not pressed: a press jumps) and B pressed with stickMag <= 48
//!   (check_ground_dive_or_punch: above 48 it dives) starts a move punch,
//!   which with A down in its first frame becomes a jump kick
//!   (act_move_punching) that keeps the speed. A stays held through the
//!   kick and its landing, so the next walking frame kicks again.
//! - `edge-dive` (A28 / G05 "edge dive hop", 48 TAS chains): run off an
//!   edge without jumping, then B in the freefall (an air dive at fv > 28:
//!   fv + 15, at most 48), A in the dive slide (rollout).
//! - `high-dive` (G17, 36 TAS chains): A from walking at fv >= 28 (vy
//!   42 + fv/4), B on the jump's next frame: the dive keeps the jump's
//!   vy with fv + 15 (at most 48); A in the dive slide (rollout).
//!
//! C-up variants: the braking slide is slide.rs `slide-brake`. A C-up
//! backflip (first person turns the facing without walking) and C-up VSC
//! (landing actions under INPUT_FIRST_PERSON go to their end action
//! without resetting vel.y, common_landing_cancels) need a first-person
//! turn / a landing with stored vy, which no single reactive rule sets up
//! from the states the search meets: not offered.

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::{Arrival, GamePad, St};

use super::{Ctx, camera_stick};

const ACT_WALKING: u32 = 0x0400_0440;
const ACT_JUMP: u32 = 0x0300_0880;
const ACT_DOUBLE_JUMP: u32 = 0x0300_0881;
const ACT_FREEFALL: u32 = 0x0100_088C;
const ACT_JUMP_KICK: u32 = 0x0180_08AC;
const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;
const ACT_DIVE: u32 = 0x0188_088A;
const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
const ACT_STOMACH_SLIDE: u32 = 0x008C_0453;
const ACT_JUMP_LAND: u32 = 0x0400_0470;
const ACT_FREEFALL_LAND: u32 = 0x0400_0471;
const ACT_DOUBLE_JUMP_LAND: u32 = 0x0400_0472;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tech {
    SpeedKick,
    EdgeDive,
    HighDive,
}

/// The stick at the target scaled to `mag` of full tilt (camera-relative).
fn stick_scaled(st: &St, target: [f32; 3], mag: f32) -> (i8, i8) {
    let (x, y) = camera_stick(st, target);
    ((x as f32 * mag) as i8, (y as f32 * mag) as i8)
}

pub fn tech_pad(k: Tech, st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    let a = st.action;
    let even = frame % 2 == 0;
    let (sx, sy) = camera_stick(st, target);
    let full = |b: u16| Some(GamePad::new(b, sx, sy));
    match k {
        Tech::SpeedKick => match a {
            ACT_WALKING if st.forward_vel >= 29.0 => {
                // stickMag <= 48 (raw about 54 of 127 after the dead zone).
                let (x, y) = stick_scaled(st, target, 0.42);
                Some(GamePad::new(A_BUTTON | if even { B_BUTTON } else { 0 }, x, y))
            }
            ACT_WALKING | ACT_MOVE_PUNCHING | ACT_JUMP_LAND | ACT_FREEFALL_LAND | ACT_DOUBLE_JUMP_LAND => full(A_BUTTON),
            ACT_JUMP | ACT_DOUBLE_JUMP | ACT_FREEFALL | ACT_JUMP_KICK => full(A_BUTTON),
            _ => None,
        },
        Tech::EdgeDive => match a {
            ACT_FREEFALL if st.forward_vel > 28.0 => full(if even { B_BUTTON } else { 0 }),
            ACT_DIVE | ACT_FREEFALL => full(0),
            ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE => full(if even { A_BUTTON } else { 0 }),
            _ if a & 0x800 == 0 => full(0),
            _ => None,
        },
        Tech::HighDive => match a {
            ACT_WALKING if st.forward_vel >= 28.0 => full(if even { A_BUTTON } else { 0 }),
            ACT_JUMP if st.vel[1] > 0.0 => full(A_BUTTON | B_BUTTON),
            ACT_DIVE => full(0),
            ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE => full(if even { A_BUTTON } else { 0 }),
            _ => None,
        },
    }
}

pub fn tech_plausible(k: Tech, st: &St, target: [f32; 3], arrival: Arrival) -> bool {
    let a = st.action;
    let ctx = Ctx::with(st, target, arrival);
    let far = arrival == Arrival::Pass && ctx.hd > 300.0 || ctx.hd > 600.0;
    let ground = matches!(a, ACT_WALKING | ACT_JUMP_LAND | ACT_FREEFALL_LAND | ACT_DOUBLE_JUMP_LAND);
    match k {
        Tech::SpeedKick => {
            far && ctx.dy.abs() < 200.0
                && (ground && st.forward_vel >= 25.0 || matches!(a, ACT_JUMP | ACT_DOUBLE_JUMP | ACT_FREEFALL | ACT_JUMP_KICK) && st.forward_vel >= 29.0)
        }
        Tech::EdgeDive => {
            // An edge ahead (a drop of 150+) before the target, at speed.
            far && (a == ACT_FREEFALL && st.forward_vel > 28.0
                || a == ACT_WALKING && st.forward_vel >= 20.0 && ctx.ahead.as_ref().is_some_and(|ah| ah.drop_depth(ctx.base).is_some_and(|(e, _)| e <= 400.0)))
        }
        Tech::HighDive => {
            far && a == ACT_WALKING && st.forward_vel >= 25.0 && ctx.dy < 150.0 && st.ceil_y - st.pos[1] > 400.0
        }
    }
}
