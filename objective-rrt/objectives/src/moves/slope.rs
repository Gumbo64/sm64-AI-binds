//! Slope and high-speed techniques from the 120-star TAS
//! (ref/tas/slope_techniques.md, "Specs for the movement library"), as
//! GA policies ([`super::policies`]): each plays its inputs where Mario's
//! action and speed allow and leaves other frames to the default steering
//! (`None`), so combinations ("slope dive, then surf, then double jump")
//! are found by simulation.
//!
//! - S1 + S2 [`slope_dive`]: dive onto a slippery/steep downhill (A+B on a
//!   steep landing: steep jump + dive), ride the dive slide with the stick
//!   along the slide (lossless), roll out (A) on the first flat frame
//!   (ny > 0.99) or at 95+; a butt slide jumps out at 60+.
//! - S3 [`surf`]: at 60+, on landings A held + B with a short stick (jump
//!   kick: on a slope it lands at once into FREEFALL_LAND, no friction, no
//!   48 cap), refreshed on FREEFALL_LAND's last frame; A held in the air.
//! - S4 [`air_chain`]: at 60+, Z (crouch slide), Z+B (slide kick), A in
//!   the slide-kick slide (rollout), A held + B on the landing (jump
//!   kick), Z on FREEFALL_LAND's last frame: one ground frame a hop.
//! - S5 [`speed_double_jump`]: at 60+, A released in the air, a fresh A
//!   on FREEFALL_LAND / JUMP_LAND (double jump: vy 52 + fv/4, range about
//!   37.4 fv - 500), A held through it.
//! - S6 [`steep_jump`]: at 40+ on a steep floor not facing downhill, a
//!   fresh A (steep jump: vy 42 + fv'/4), A held while rising.
//!
//! [`plausible`] gates them on a slippery or steep floor under or ahead of
//! Mario (where speed comes from) or the speed already there.

use std::f32::consts::TAU;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::{GamePad, St};

use super::*;

const ACT_DIVE: u32 = 0x0188_088A;
const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
const ACT_BUTT_SLIDE: u32 = 0x0084_0452;
const ACT_STEEP_JUMP: u32 = 0x0300_0885;
const ACT_JUMP_LAND_STOP: u32 = 0x0C00_0230;
const ACT_DOUBLE_JUMP_LAND_STOP: u32 = 0x0C00_0231;
const ACT_FREEFALL_LAND_STOP: u32 = 0x0C00_0232;
const ACT_SIDE_FLIP_LAND_STOP: u32 = 0x0C00_0233;

/// Speed the high-speed techniques start at (u/f).
pub const FAST: f32 = 60.0;

/// Stick tilt that keeps B a punch (jump kick with A held), not the
/// speed-kick dive (stick magnitude over 48).
const SHORT: f32 = 0.38;

fn yaw_to(st: &St, target: [f32; 3]) -> i16 {
    angle(target[0] - st.pos[0], target[2] - st.pos[2])
}

/// Full stick towards the target.
fn at(st: &St, target: [f32; 3], b: u16) -> GamePad {
    let (x, y) = camera_stick(st, target);
    GamePad::new(b, x, y)
}

/// Short stick towards the target.
fn short(st: &St, target: [f32; 3], b: u16) -> GamePad {
    let (x, y) = arcs::stick(st.cam_yaw, yaw_to(st, target), SHORT);
    GamePad::new(b, x, y)
}

/// Full stick along the horizontal velocity (the dive slide's lossless
/// steering), towards the target if barely moving.
fn along(st: &St, target: [f32; 3]) -> GamePad {
    let (vx, vz) = (st.vel[0], st.vel[2]);
    if vx * vx + vz * vz < 1.0 {
        return at(st, target, 0);
    }
    let (x, y) = stick_dir(st, vx.atan2(vz));
    GamePad::new(0, x, y)
}

fn landing_stop(a: u32) -> bool {
    matches!(a, ACT_JUMP_LAND_STOP | ACT_DOUBLE_JUMP_LAND_STOP | ACT_FREEFALL_LAND_STOP | ACT_SIDE_FLIP_LAND_STOP)
}

fn steep_floor(st: &St) -> bool {
    st.action & ACT_FLAG_AIR == 0 && st.floor_normal[1] < 0.866
}

/// S1 + S2.
pub fn slope_dive(st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    let a = st.action;
    let fv = st.forward_vel;
    let even = frame % 2 == 0;
    match a {
        ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE => {
            let exit = st.floor_normal[1] > 0.99 || fv >= 95.0;
            Some(if exit { at(st, target, A_BUTTON) } else { along(st, target) })
        }
        ACT_BUTT_SLIDE if fv >= FAST => Some(at(st, target, if even { A_BUTTON } else { 0 })),
        ACT_BUTT_SLIDE => Some(along(st, target)),
        ACT_DIVE => Some(along(st, target)),
        _ if air_actionable(a) && fv > 28.0 => Some(at(st, target, B_BUTTON)),
        ACT_WALKING if fv >= 29.0 => Some(at(st, target, if even { B_BUTTON } else { 0 })),
        // A steep landing: A+B, a steep jump that dives the same frame.
        _ if (is_landing(a) || a == ACT_LONG_JUMP_LAND) && steep_floor(st) && fv >= 25.0 => Some(at(st, target, A_BUTTON | B_BUTTON)),
        _ => None,
    }
}

/// S3.
pub fn surf(st: &St, target: [f32; 3], _frame: usize) -> Option<GamePad> {
    let a = st.action;
    if st.forward_vel < FAST {
        return None;
    }
    if a & ACT_FLAG_AIR != 0 {
        return Some(at(st, target, A_BUTTON));
    }
    match a {
        _ if landing_stop(a) => Some(short(st, target, A_BUTTON | B_BUTTON)),
        ACT_FREEFALL_LAND if st.action_timer >= 3 => Some(short(st, target, A_BUTTON | B_BUTTON)),
        _ if is_landing(a) && st.action_timer == 0 => Some(short(st, target, A_BUTTON | B_BUTTON)),
        _ if is_landing(a) => Some(short(st, target, A_BUTTON)),
        _ => None,
    }
}

/// S4.
pub fn air_chain(st: &St, target: [f32; 3], _frame: usize) -> Option<GamePad> {
    let a = st.action;
    if st.forward_vel < FAST {
        return None;
    }
    match a {
        ACT_CROUCH_SLIDE => Some(at(st, target, Z_TRIG | B_BUTTON)),
        ACT_SLIDE_KICK => Some(at(st, target, 0)),
        ACT_SLIDE_KICK_SLIDE => Some(at(st, target, A_BUTTON)),
        ACT_FREEFALL_LAND if st.action_timer >= 3 => Some(at(st, target, Z_TRIG)),
        _ if landing_stop(a) || is_landing(a) && st.action_timer == 0 => Some(short(st, target, A_BUTTON | B_BUTTON)),
        _ if is_landing(a) => Some(short(st, target, A_BUTTON)),
        _ if a & ACT_FLAG_AIR != 0 => Some(at(st, target, A_BUTTON)),
        ACT_WALKING | ACT_DECELERATING | ACT_BRAKING => Some(at(st, target, Z_TRIG)),
        _ => None,
    }
}

/// S5.
pub fn speed_double_jump(st: &St, target: [f32; 3], _frame: usize) -> Option<GamePad> {
    let a = st.action;
    if st.forward_vel < FAST && a != ACT_DOUBLE_JUMP {
        return None;
    }
    match a {
        ACT_DOUBLE_JUMP => Some(at(st, target, A_BUTTON)),
        ACT_FREEFALL_LAND | ACT_JUMP_LAND => Some(at(st, target, A_BUTTON)),
        _ if a & ACT_FLAG_AIR != 0 => Some(at(st, target, 0)),
        _ => None,
    }
}

/// S6.
pub fn steep_jump(st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    let a = st.action;
    if a == ACT_STEEP_JUMP {
        return Some(at(st, target, if st.vel[1] > 20.0 { A_BUTTON } else { 0 }));
    }
    if !steep_floor(st) || st.forward_vel < 40.0 || is_sliding(a) {
        return None;
    }
    // Not facing downhill (else it's a plain jump down the slope).
    let n = st.floor_normal;
    let f = st.yaw as f32 * TAU / 65536.0;
    let downhill = n[0] * f.sin() + n[2] * f.cos() > 0.1;
    if downhill {
        return None;
    }
    Some(at(st, target, if frame % 2 == 0 { A_BUTTON } else { 0 }))
}

/// A floor speed comes from: slippery (slope accel 8+) or steeper than
/// walking holds, under Mario or within 600 ahead, descending along the
/// target direction.
fn speed_floor(st: &St, ctx: &Ctx) -> bool {
    let n = st.floor_normal;
    let fast_kind = |kind: i16, ny: f32| slope_accel(kind) >= 8.0 || ny < 0.87;
    let down = |n: [f32; 3]| n[0] * ctx.dir[0] + n[2] * ctx.dir[1] > 0.1;
    if !ctx.air && fast_kind(st.floor_type, n[1]) && n[1] < 0.99 && down(n) {
        return true;
    }
    ctx.ahead.as_ref().is_some_and(|ah| {
        ah.low.iter().filter(|(d, _)| *d <= 600.0).any(|(_, f)| f.is_some_and(|f| f.normal[1] < 0.99 && fast_kind(f.kind, f.normal[1]) && down(f.normal)))
    })
}

/// Which technique: 0 slope-dive, 1 surf, 2 air-chain,
/// 3 speed-dj, 4 steep-jump.
pub fn plausible(which: usize, st: &St, ctx: &Ctx) -> bool {
    let fast = ctx.fv >= FAST;
    // Speed now, a slide that builds it, or a floor ahead that does.
    let fuel = fast || is_sliding(st.action) || speed_floor(st, ctx);
    match which {
        0 => speed_floor(st, ctx) && ctx.hd > 400.0 || matches!(st.action, ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE | ACT_BUTT_SLIDE),
        1 => fuel && ctx.hd > 300.0,
        2 => fuel && ctx.hd > 600.0,
        3 => fuel && ctx.hd > 1000.0 && st.ceil_y - st.pos[1] > 800.0 && ctx.dy < 700.0,
        4 => {
            let steep_ahead = ctx.ahead.as_ref().is_some_and(|ah| ah.steepest(ctx.dir, ctx.hd.clamp(60.0, 600.0)) > 0.6);
            (fuel || ctx.fv >= 40.0) && (steep_floor(st) || steep_ahead) && ctx.dy > 100.0
        }
        _ => false,
    }
}
