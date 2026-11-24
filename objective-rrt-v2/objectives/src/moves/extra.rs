//! The TAS catalogue's movement techniques the policies were missing or had
//! wrong (ref/tas/techniques/CATALOGUE.md; the audit is in mod.rs's docs), as
//! GA policies behind `OBRRT_MOVES_EXTRA=1` (on by default in `legs` only;
//! unset, they aren't even counted, so obrrt2's plans draw as before).
//!
//! Every pad is a pure function of the state, target and frame. A button's
//! edge comes from the action, as in mod.rs: a press on the frame an action
//! starts taking it, released while the action before it ran.
//!
//! Decomp facts these rules rest on (mario_actions_*.c, mario.c):
//!
//! - A fresh A beats everything on the ground (`set_jump_from_landing`: a
//!   jump at fv x 0.8, or a steep jump) and in a landing action. A *held*
//!   and B: walking's `check_ground_dive_or_punch` (fv < 29 or stickMag <=
//!   48) gives MOVE_PUNCHING, whose first frame with A down is a JUMP_KICK
//!   in the same frame: vy 20, fv untouched, no steep test. So a kick chain
//!   needs A down since the air, and no walking frame. Landing actions
//!   ignore B: they run out after `numFrames` (4; long jump 6), and on that
//!   last frame the same input runs in *_LAND_STOP, where any stick is
//!   WALKING, where B kicks. No stick there on a slippery floor is
//!   BEGIN_SLIDING. A held in the air costs nothing.
//! - The air has no turning (update_air_without_turn), and neither has a kick
//!   chain's single ground frame: a chain flies along the facing it started
//!   with, so a walking frame without B turns it (0x800 a frame).
//! - A fresh A + B in one frame runs the jump and its B in that frame: a steep
//!   jump or a side flip or a freefall dives at any speed; ACT_JUMP dives
//!   only above fv 28 (after the 0.8: fv > 35 on the ground) and else kicks.
//!   A dive keeps vy, so A then B is a high dive only with fv > 28 in the
//!   jump (the air's stick adds 1.15 a frame up to 32).
//! - DIVE_SLIDE rolls out on A or B only without INPUT_ABOVE_SLIDE
//!   (geom::slippery_limit): there is no ladder on a floor that slides.
//!   A rollout lands straight in FREEFALL_LAND_STOP.
//! - Punching from standing sets fv from a table: 10 at once (walking starts
//!   at 8), any stick then walks on with it. A crouch slide's long jump needs
//!   fv > 10 on its A frame, so the walk goes on to 12 first.
//! - AIR_HIT_WALL and the late kicks want an A *press*: modes.rs's `kick` holds
//!   A in WALL_KICK_AIR towards a target above, so chains only ever kicked from
//!   the soft bonk (fv -8, then the kick's minimum 24). `wall-chain` lets go.
//!
//! The policies, by catalogue id:
//!
//! - `jump-kick` (G15): A held, B at each landing's last frame and on
//!   walking frames: the A+B kick, keeping speed, uphill and on the flat.
//! - `kick-start`, `punch-start` (G18): from standing.
//! - `steep-climb`, `dive-ladder` (G19): steep jumps (a dive at the lip),
//!   and dive / rollout on floors that don't slide.
//! - `edge-hop` (G05): B in FREEFALL at any speed, A in the slide at 45+
//!   (tech.rs `edge-dive` only dives above 28 and only from 20+ walking).
//! - `jump-dive`, `turn-dive` (G17): tech.rs `high-dive` presses B on the
//!   jump's next frame, which at walking speed (under 35) is a kick.
//! - `butt-boost`, `kick-boost` (G07): get onto the slope's slide, ride it
//!   along the velocity (lossless) and leave it fast. slope.rs `slope-dive`
//!   needs 25+ to start, and has no slide kick.
//! - `flip-kick`, `wall-chain` (G10, G09).
//! - `hyper-walk`, `hyper-launch` (G01, G04): only at hyperspeed.

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::{Arrival, GamePad, St};

use super::slope::{along, at, short, speed_floor};
use super::*;

const ACT_DIVE: u32 = 0x0188_088A;
const ACT_FORWARD_ROLLOUT: u32 = 0x0100_08A6;
const ACT_JUMP_KICK: u32 = 0x0180_08AC;
const ACT_PUNCHING: u32 = 0x0080_0380;
const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;
const ACT_STEEP_JUMP: u32 = 0x0300_0885;
const ACT_BUTT_SLIDE: u32 = 0x0084_0452;
const ACT_BUTT_SLIDE_AIR: u32 = 0x0300_088E;
const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
const ACT_WALL_KICK_AIR: u32 = 0x0300_0886;
const ACT_BACKWARD_AIR_KB: u32 = 0x0102_08B0;
const ACT_STANDING_AGAINST_WALL: u32 = 0x0C40_0209;
const ACT_PANTING: u32 = 0x0C40_0205;

/// The extra policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extra {
    JumpKick,
    KickStart,
    PunchStart,
    SteepClimb,
    DiveLadder,
    EdgeHop,
    JumpDive,
    TurnDive,
    ButtBoost,
    KickBoost,
    FlipKick,
    WallChain,
    HyperWalk,
    HyperLaunch,
}

/// `OBRRT_MOVES_EXTRA=1`: the policies are offered (`legs` sets it unless
/// told otherwise).
pub fn extra_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_MOVES_EXTRA").is_ok_and(|v| v == "1"))
}

/// Landing lag, in frames (`LandingAction.numFrames`).
fn landing_frames(a: u32) -> i32 {
    if a == ACT_LONG_JUMP_LAND { 6 } else { 4 }
}

fn is_land(a: u32) -> bool {
    is_landing(a) || a == ACT_LONG_JUMP_LAND
}

/// The *_LAND_STOP actions (check_common_landing_cancels).
fn land_stop(a: u32) -> bool {
    matches!(a, 0x0C00_0230 | 0x0C00_0231 | 0x0C00_0232 | 0x0C00_0233 | 0x0800_022F | 0x0800_023A | 0x0800_023B)
}

fn stationary(a: u32) -> bool {
    matches!(a, ACT_IDLE | ACT_PANTING | ACT_STANDING_AGAINST_WALL) || land_stop(a)
}

fn walks(a: u32) -> bool {
    matches!(a, ACT_WALKING | ACT_DECELERATING | ACT_BRAKING)
}

fn dive_slides(a: u32) -> bool {
    matches!(a, ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE)
}

/// INPUT_ABOVE_SLIDE, for the floor under him.
fn above_slide(st: &St) -> bool {
    st.floor_normal[1] <= geom::slippery_limit(st.floor_type)
}

/// mario_floor_is_steep, facing aside (the game's test also wants him not
/// facing downhill).
fn steep_here(st: &St) -> bool {
    st.floor_normal[1] <= steep_limit(st.floor_type)
}

/// Yaw off the target by more than a kick chain should fly: a walking frame
/// without B turns him.
fn off_course(ctx: &Ctx) -> bool {
    ctx.yaw_err > 0x1000
}

/// A rollout's press: A, with a B now and then in case A is already down
/// (either rolls out).
fn rollout_press(frame: usize) -> u16 {
    if frame % 3 == 2 { B_BUTTON } else { A_BUTTON }
}

/// A floor to land on: flat enough, not far below him, within the jump's reach
/// (`along`) or the target's height near him.
fn lip_ahead(st: &St, ctx: &Ctx) -> bool {
    if ctx.dy.abs() < 150.0 && ctx.hd < 800.0 {
        return true;
    }
    let reach = (ctx.fv.abs() * 8.0).max(250.0);
    ctx.ahead.as_ref().is_some_and(|ah| {
        ah.low.iter().filter(|(d, _)| *d <= reach).any(|(_, f)| f.is_some_and(|f| f.normal[1] > 0.9 && f.y > st.pos[1] - 100.0))
    })
}

fn slope_slide_ride(st: &St, target: [f32; 3], ctx: &Ctx, exit: impl Fn(&St) -> bool, frame: usize) -> GamePad {
    // Leaving as soon as the rest of the way is a jump: its airtime is about
    // vy / 2 frames at 0.8 fv.
    let vy = 42.0 + ctx.fv.max(0.0) / 4.0;
    let jump_reach = ctx.fv.max(0.0) * 0.8 * vy / 2.0;
    if exit(st) || ctx.fv >= 30.0 && ctx.hd < jump_reach && ctx.dy > -50.0 {
        at(st, target, if frame % 2 == 0 { A_BUTTON } else { 0 })
    } else {
        along(st, target)
    }
}

/// Policy `k`'s pad (None: the frame goes to the default steering).
pub fn extra_pad(k: Extra, st: &St, target: [f32; 3], arrival: Arrival, frame: usize) -> Option<GamePad> {
    let a = st.action;
    let air = a & ACT_FLAG_AIR != 0;
    let fv = st.forward_vel;
    let even = frame % 2 == 0;
    let ctx = Ctx::with(st, target, arrival);
    let rising = if st.vel[1] > 20.0 { A_BUTTON } else { 0 };
    match k {
        Extra::JumpKick => {
            // A short stick only where a long one would dive.
            let kick = |b: u16| if fv >= 28.0 { short(st, target, b) } else { at(st, target, b) };
            if air {
                // A held all the way down, so the landing has no fresh press.
                return (a != ACT_DIVE && a != ACT_AIR_HIT_WALL).then(|| at(st, target, A_BUTTON));
            }
            if is_land(a) {
                let last = st.action_timer as i32 + 1 >= landing_frames(a);
                return Some(if last && !off_course(&ctx) { kick(A_BUTTON | B_BUTTON) } else { at(st, target, A_BUTTON) });
            }
            if land_stop(a) || walks(a) || a == ACT_IDLE {
                if off_course(&ctx) && !stationary(a) {
                    return Some(at(st, target, A_BUTTON));
                }
                return Some(if even || land_stop(a) { kick(A_BUTTON | B_BUTTON) } else { at(st, target, A_BUTTON) });
            }
            matches!(a, ACT_MOVE_PUNCHING | ACT_PUNCHING).then(|| at(st, target, A_BUTTON))
        }
        Extra::KickStart => match a {
            // A + B fresh with the stick: a jump, kicked the same frame.
            _ if stationary(a) || walks(a) && fv < 16.0 => {
                if off_course(&ctx) {
                    Some(at(st, target, 0))
                } else {
                    Some(at(st, target, if even || stationary(a) { A_BUTTON | B_BUTTON } else { 0 }))
                }
            }
            ACT_JUMP_KICK => Some(at(st, target, 0)),
            // The landing's fresh A: a double jump at the speed the kick built.
            ACT_JUMP_LAND | ACT_FREEFALL_LAND => Some(at(st, target, A_BUTTON)),
            ACT_DOUBLE_JUMP => Some(at(st, target, rising)),
            _ => None,
        },
        Extra::PunchStart => match a {
            _ if stationary(a) && ctx.yaw_err <= 0x1000 => Some(GamePad::new(B_BUTTON, 0, 0)),
            _ if stationary(a) => Some(at(st, target, 0)),
            ACT_PUNCHING => Some(at(st, target, 0)),
            ACT_WALKING if fv < 12.0 => Some(at(st, target, 0)),
            ACT_WALKING => Some(at(st, target, Z_TRIG)),
            ACT_CROUCH_SLIDE => Some(at(st, target, Z_TRIG | A_BUTTON)),
            ACT_LONG_JUMP | ACT_LONG_JUMP_LAND => Some(pad_for_with(MoveKind::LongJumpChain, st, target, arrival, frame)),
            _ => None,
        },
        Extra::SteepClimb => match a {
            ACT_STEEP_JUMP | ACT_JUMP | ACT_DOUBLE_JUMP if lip_ahead(st, &ctx) && st.vel[1] < 25.0 && (a == ACT_STEEP_JUMP || fv > 28.0) => {
                Some(at(st, target, B_BUTTON))
            }
            ACT_DIVE | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            _ if dive_slides(a) => Some(if above_slide(st) { along(st, target) } else { at(st, target, rollout_press(frame)) }),
            _ if air => (!matches!(a, ACT_AIR_HIT_WALL | ACT_WALL_KICK_AIR)).then(|| at(st, target, rising)),
            ACT_BUTT_SLIDE => Some(at(st, target, if even { A_BUTTON } else { 0 })),
            ACT_BEGIN_SLIDING => Some(at(st, target, 0)),
            // Stalled on the slope a steep jump goes straight up and down: dive
            // out of it (a steep jump dives at any speed).
            _ if (is_land(a) || land_stop(a)) && fv.abs() < 4.0 && steep_here(st) => Some(at(st, target, A_BUTTON | B_BUTTON)),
            _ if is_land(a) || land_stop(a) => Some(at(st, target, A_BUTTON)),
            _ if walks(a) || a == ACT_IDLE => Some(at(st, target, if even { A_BUTTON } else { 0 })),
            _ => None,
        },
        Extra::DiveLadder => match a {
            ACT_DIVE | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            ACT_STEEP_JUMP | ACT_SIDE_FLIP | ACT_FREEFALL => Some(at(st, target, B_BUTTON)),
            ACT_JUMP | ACT_DOUBLE_JUMP if fv > 28.0 => Some(at(st, target, B_BUTTON)),
            ACT_JUMP | ACT_DOUBLE_JUMP => Some(at(st, target, rising)),
            _ if dive_slides(a) => Some(if above_slide(st) { along(st, target) } else { at(st, target, rollout_press(frame)) }),
            _ if is_land(a) || land_stop(a) || walks(a) || a == ACT_IDLE => {
                // On the ground an even-frame press unless this action only
                // lasts the frame: a dive with vy 20 at 29+ on easy floors,
                // else jump and dive in one frame.
                let press = if !steep_here(st) && fv >= 29.0 { B_BUTTON } else { A_BUTTON | B_BUTTON };
                Some(at(st, target, if even || !walks(a) { press } else { 0 }))
            }
            _ => None,
        },
        Extra::EdgeHop => match a {
            // act_freefall dives on B at any speed; the slide steps off again.
            ACT_FREEFALL => Some(at(st, target, B_BUTTON)),
            ACT_DIVE | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            _ if dive_slides(a) => {
                let edge = ctx.ahead.as_ref().and_then(|ah| ah.drop_depth(ctx.base)).is_some_and(|(e, _)| e <= (fv.abs() * 3.0).max(60.0));
                Some(if fv >= 45.0 || !edge { at(st, target, rollout_press(frame)) } else { along(st, target) })
            }
            _ if !air && !is_sliding(a) => Some(at(st, target, 0)),
            _ => None,
        },
        Extra::JumpDive => match a {
            // A + B fresh dives in the jump's own frame only above 35 (0.8 fv
            // > 28). Below, A now and B once the air's stick has it over 28.
            _ if is_land(a) || land_stop(a) || walks(a) => {
                let press = if fv > 35.5 {
                    A_BUTTON | B_BUTTON
                } else if fv >= 20.0 {
                    A_BUTTON
                } else {
                    0
                };
                Some(at(st, target, if even || !walks(a) { press } else { 0 }))
            }
            ACT_JUMP | ACT_DOUBLE_JUMP if fv > 28.0 && st.vel[1] > 0.0 => Some(at(st, target, B_BUTTON | rising)),
            ACT_JUMP | ACT_DOUBLE_JUMP => Some(at(st, target, rising)),
            ACT_DIVE | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            _ if dive_slides(a) => Some(if above_slide(st) { along(st, target) } else { at(st, target, rollout_press(frame)) }),
            _ => None,
        },
        Extra::TurnDive => match a {
            ACT_WALKING if ctx.yaw_err > 0x471C && fv >= 16.0 => Some(at(st, target, 0)),
            ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND => Some(at(st, target, A_BUTTON | B_BUTTON)),
            ACT_SIDE_FLIP => Some(at(st, target, B_BUTTON)),
            ACT_DIVE | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            _ if dive_slides(a) => Some(if above_slide(st) { along(st, target) } else { at(st, target, rollout_press(frame)) }),
            _ => None,
        },
        Extra::ButtBoost => match a {
            ACT_BUTT_SLIDE => Some(slope_slide_ride(st, target, &ctx, |s| s.forward_vel >= 90.0 || s.forward_vel >= 40.0 && s.floor_normal[1] > 0.98, frame)),
            ACT_BEGIN_SLIDING | ACT_BUTT_SLIDE_AIR => Some(along(st, target)),
            ACT_CROUCH_SLIDE => Some(along(st, target)),
            ACT_JUMP => Some(at(st, target, rising)),
            _ if walks(a) && ctx.grade < -0.05 => Some(at(st, target, if even { Z_TRIG } else { 0 })),
            _ if walks(a) => Some(at(st, target, 0)),
            _ => None,
        },
        Extra::KickBoost => match a {
            ACT_CROUCH_SLIDE if fv >= 10.0 => Some(at(st, target, Z_TRIG | B_BUTTON)),
            ACT_SLIDE_KICK | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            ACT_SLIDE_KICK_SLIDE => Some(slope_slide_ride(st, target, &ctx, |s| s.forward_vel >= 75.0 || s.forward_vel >= 30.0 && s.floor_normal[1] > 0.98, frame)),
            _ if land_stop(a) => Some(at(st, target, Z_TRIG)),
            _ if is_land(a) => Some(at(st, target, 0)),
            _ if walks(a) && fv >= 10.0 => Some(at(st, target, if even { Z_TRIG } else { 0 })),
            _ if walks(a) => Some(at(st, target, 0)),
            _ => None,
        },
        Extra::FlipKick => match a {
            ACT_WALKING if ctx.yaw_err > 0x471C && fv >= 16.0 => Some(at(st, target, 0)),
            ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND => Some(at(st, target, A_BUTTON)),
            ACT_SIDE_FLIP => Some(at(st, target, 0)),
            ACT_AIR_HIT_WALL => Some(at(st, target, A_BUTTON)),
            ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB => Some(at(st, target, if even { A_BUTTON } else { 0 })),
            _ => None,
        },
        Extra::WallChain => match a {
            ACT_WALL_KICK_AIR => {
                // A let go in time for the next wall's press.
                let f = facing_dir(st);
                let wall_soon = geom::wall_ahead(st, f, (fv.abs() * 2.0).max(60.0), st.pos[1] - 30.0, 0.5).is_some();
                let b = if wall_soon { 0 } else { rising };
                // The next wall is on along his facing while the target is
                // more above than away.
                if ctx.dy > ctx.hd || ctx.hd < 300.0 {
                    let (x, y) = stick_dir(st, st.yaw as f32 * TAU / 65536.0);
                    Some(GamePad::new(b, x, y))
                } else {
                    Some(at(st, target, b))
                }
            }
            ACT_AIR_HIT_WALL => Some(at(st, target, A_BUTTON)),
            ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB => Some(at(st, target, if even { A_BUTTON } else { 0 })),
            _ if air || walks(a) || is_land(a) || land_stop(a) || a == ACT_IDLE => Some(pad_for_with(MoveKind::WallKick, st, target, arrival, frame)),
            _ => None,
        },
        Extra::HyperWalk => {
            if fv > -10.0 || is_sliding(a) {
                return None;
            }
            if air {
                // Air drag only: any stick along the facing would slow him.
                return Some(GamePad::new(0, 0, 0));
            }
            let d = (target[0] - st.pos[0]).atan2(target[2] - st.pos[2]) + TAU / 2.0;
            let (x, y) = stick_dir(st, d);
            Some(GamePad::new(0, x, y))
        }
        Extra::HyperLaunch => match a {
            ACT_JUMP | ACT_DOUBLE_JUMP | ACT_STEEP_JUMP => {
                // A dive once the rest of the way fits under the dive's 48:
                // the frames left until he falls back to the target's height.
                let vy = st.vel[1];
                let disc = vy * vy - 8.0 * ctx.dy;
                let dive = disc >= 0.0 && ctx.hd <= 0.95 * 48.0 * (vy + disc.sqrt()) / 4.0;
                Some(at(st, target, if dive { B_BUTTON } else { rising }))
            }
            ACT_DIVE | ACT_FORWARD_ROLLOUT => Some(at(st, target, 0)),
            ACT_AIR_HIT_WALL => Some(at(st, target, A_BUTTON)),
            _ if !air && fv >= 100.0 && !is_sliding(a) => Some(at(st, target, A_BUTTON)),
            _ if a == ACT_CROUCH_SLIDE || a == ACT_SLIDE_KICK_SLIDE || a == ACT_BUTT_SLIDE => Some(at(st, target, A_BUTTON)),
            _ => None,
        },
    }
}

/// Obvious nonsense only (policies.rs's rule): not a prediction of which
/// is best. A policy that doesn't apply yet falls to the default steering until
/// it does, so on a far target most stay plausible.
pub fn plausible(k: Extra, st: &St, ctx: &Ctx) -> bool {
    let a = st.action;
    // Ground and air only: other groups have their own controllers.
    if a & obrrt_core::layout::ACT_GROUP_MASK > 0x80 && a != ACT_PUNCHING {
        return false;
    }
    let pass = ctx.arrival == Arrival::Pass;
    let away = pass || ctx.hd > 300.0;
    let ground = !ctx.air;
    match k {
        Extra::JumpKick => pass || ctx.hd > 150.0,
        Extra::KickStart => away && (ground && !is_sliding(a) && ctx.fv < 16.0 || a == ACT_JUMP_KICK),
        Extra::PunchStart => away && stationary(a) && ctx.fv.abs() < 5.0 && st.ceil_y - st.pos[1] > 200.0,
        Extra::SteepClimb => {
            ctx.dy > 100.0
                || floor_too_steep(st, ctx)
                || ctx.ahead.as_ref().is_some_and(|ah| ah.steepest(ctx.dir, ctx.hd.clamp(60.0, 800.0)) > STEEP_GRADE)
        }
        Extra::DiveLadder => {
            let steep = floor_too_steep(st, ctx) || ctx.ahead.as_ref().is_some_and(|ah| ah.steepest(ctx.dir, ctx.hd.clamp(60.0, 800.0)) > STEEP_GRADE);
            (ctx.dy > 50.0 || steep) && !(ground && above_slide(st))
        }
        Extra::EdgeHop => away && (a == ACT_FREEFALL || !ctx.air && !is_sliding(a)) && (ctx.hd > 800.0 || ctx.ahead.as_ref().is_none_or(|ah| ah.drop_depth(ctx.base).is_some())),
        Extra::JumpDive => away && ctx.dy < 1500.0 && st.ceil_y - st.pos[1] > 300.0,
        Extra::TurnDive => matches!(a, ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND | ACT_SIDE_FLIP) || ground && ctx.yaw_err > 0x471C && ctx.fv >= 16.0,
        Extra::ButtBoost => away && (matches!(a, ACT_BUTT_SLIDE | ACT_BEGIN_SLIDING) || speed_floor(st, ctx)),
        Extra::KickBoost => away && (matches!(a, ACT_SLIDE_KICK | ACT_SLIDE_KICK_SLIDE | ACT_CROUCH_SLIDE) || speed_floor(st, ctx)),
        Extra::FlipKick => {
            let wall = wall_along(st, ctx, ctx.dir, 700.0);
            matches!(a, ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND | ACT_SIDE_FLIP) || ground && ctx.yaw_err > 0x471C && ctx.fv >= 16.0 && (wall || ctx.dy > 150.0)
        }
        Extra::WallChain => {
            matches!(a, ACT_WALL_KICK_AIR | ACT_AIR_HIT_WALL | ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB)
                || ctx.dy > 150.0 && (st.wall_normal.is_some() || wall_along(st, ctx, ctx.dir, 700.0) || wall_along(st, ctx, facing_dir(st), 700.0))
        }
        Extra::HyperWalk => ctx.fv <= -50.0 && !is_sliding(a),
        Extra::HyperLaunch => ctx.fv >= 100.0 && ctx.dy > 100.0,
    }
}
