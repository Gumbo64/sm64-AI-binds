//! The steering controller: inputs that take Mario towards a point, in
//! whatever control mode his action is in (moved here from
//! search/src/steer.rs, which now asks the registered mover).
//!
//! On the ground and in the air the stick is camera-relative: `intendedYaw =
//! atan2s(-stickY, stickX) + camera yaw` (update_mario_joystick_inputs,
//! mario.c), so stick up walks away from the camera. Other actions read the
//! stick differently, and steering them camera-relative goes nowhere:
//!
//! - swimming: stick x turns Mario, stick y pitches him (up dives);
//! - poles and trees: stick y climbs, stick x turns him around the pole;
//! - the cannon: stick x and y aim it; A fires;
//! - flying: stick x turns, stick y pitches (up dives), like a plane;
//! - hanging from a ceiling: camera-relative, but only while A is held;
//! - riding Hoot: stick x turns him; A must stay held;
//! - swinging Bowser: rotating the stick spins him; B lets go.
//!
//! Riding a shell, ledge grabs, sliding and metal-cap walking underwater use
//! `intendedYaw` like walking (update_shell_speed, mario_actions_moving.c;
//! act_ledge_grab, mario_actions_automatic.c), so they share the ground code.
//!
//! Conventions. `GamePad::stick_y` is the N64's raw stick y: positive is up
//! (away from the player). The game turns raw values into `controller->stickX`
//! / `stickY` by moving them 6 towards 0 (|raw| < 8 reads as 0) and scaling
//! the vector down to length 64 (adjust_analog_stick, game_init.c); [`pad`]
//! inverts that. Angles are s16 units (65536 a turn); yaw 0 faces +z and
//! 0x4000 faces +x, so `atan2s(dz, dx)` is `dx.atan2(dz)` here. Positive pitch
//! (`faceAngle[0]`) is nose up.

use std::f32::consts::TAU;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::*;
use obrrt_core::{Arrival, GamePad, St};

use super::Library;

/// The frame's inputs towards `target` from `st`, in whatever control mode
/// Mario's action is in (module docs); `frame` counts frames into the
/// rollout (for button rhythms), `jump_every` is the generic ground rule's
/// A period, `policy` a GA policy to ask first in the ground and air mode.
#[allow(clippy::too_many_arguments)]
pub fn control(lib: &Library, st: &St, target: [f32; 3], arrival: Arrival, button: u16, frame: usize, jump_every: usize, policy: Option<usize>) -> GamePad {
    // A GA policy of the mode controllers' own (cannon, swimming, flying,
    // poles: policies.rs `mode_policy`) first; None elsewhere.
    if let Some(p) = policy {
        if let Some(pad) = super::policies::mode_policy_pad(p, st, target, arrival, button, frame) {
            return pad;
        }
    }
    match st.action {
        a if a & ACT_FLAG_SWIMMING != 0 => swim(st, target, button, frame),
        a if a & ACT_FLAG_ON_POLE != 0 => pole(st, target, frame),
        a if a & ACT_FLAG_HANGING != 0 => hang(st, target),
        ACT_IN_CANNON => cannon(st, target, frame),
        // In flight from the cannon nothing reads the stick
        // (act_shot_from_cannon, mario_actions_airborne.c).
        ACT_SHOT_FROM_CANNON => GamePad::new(0, 0, 0),
        ACT_FLYING => fly(st, target),
        ACT_WALL_KICK_AIR | ACT_TOP_OF_POLE_JUMP if button == 0 => kick(st, target),
        ACT_RIDING_HOOT => hoot(st, target),
        ACT_HOLDING_BOWSER => swing_bowser(st, target, frame),
        _ => ground(lib, st, target, arrival, button, frame, jump_every, policy),
    }
}

/// Ground and air: a GA policy's move if one is asked for and applies,
/// then the movement library's (`OBRRT_MOVES`), else the generic rule: the
/// stick towards the target (camera-relative); the objective's button on
/// alternate frames if it names one, else A on a pattern while the target
/// is above Mario.
#[allow(clippy::too_many_arguments)]
fn ground(lib: &Library, st: &St, target: [f32; 3], arrival: Arrival, button: u16, frame: usize, jump_every: usize, policy: Option<usize>) -> GamePad {
    if let Some(p) = policy {
        if let Some(pad) = super::policies::policy_pad(p, st, target, arrival, button, frame) {
            return pad;
        }
    }
    if let Some(p) = lib.library_pad(st, target, arrival, button, frame) {
        return p;
    }
    let (sx, sy) = camera_stick(st, target);
    if button != 0 {
        return GamePad::new(if frame % 2 == 0 { button } else { 0 }, sx, sy);
    }
    let mut jump = target[1] > st.pos[1] + 50.0 && frame % jump_every < jump_every / 2;
    // Calm: the rhythm only towards a raised target close by, not
    // under a low ceiling, and not while turning towards one behind
    // (a jump stops the turn).
    if jump && st.action & ACT_FLAG_AIR == 0 {
        let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
        let behind = wrap(angle(dx, dz) as i32 - st.yaw as i32).unsigned_abs() > super::policies::BEHIND;
        let close = st.hdist(target) < 500.0 && target[1] > st.pos[1] + 100.0;
        jump = close && !behind && st.ceil_y - st.pos[1] > 300.0;
    }
    GamePad::new(if jump { A_BUTTON } else { 0 }, sx, sy)
}

/// The raw stick that makes `intendedYaw` point from Mario to `target`, full
/// tilt; centred within 20 units.
fn camera_stick(st: &St, target: [f32; 3]) -> (i8, i8) {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    if dx.abs() + dz.abs() < 20.0 {
        return (0, 0);
    }
    let desired = dx.atan2(dz);
    let camera = st.cam_yaw as f32 * TAU / 65536.0;
    let rel = desired - camera;
    ((127.0 * rel.sin()) as i8, (-127.0 * rel.cos()) as i8)
}

/// Swimming (every ACT_FLAG_SWIMMING action, mario_actions_submerged.c).
///
/// - Yaw: `targetYawVel = -10 * stickX`, which `angleVel[1]` ramps towards
///   (update_swimming_yaw): stick right turns right (yaw down), relative to
///   Mario, not the camera.
/// - Pitch: `targetPitch = -252 * stickY`, approached 0x100-0x200 a frame
///   (update_swimming_pitch): stick up (forward) points him down, stick down
///   up; ±64 is about ±88°. He moves along his facing at `forwardVel`
///   (update_swimming_speed).
/// - Strokes: A starts a breaststroke; pressing A again in its frames 2-5
///   restarts it at frame 9 (act_breaststroke), so a press every 4 frames
///   keeps stroking, but each restart resets the strength (about 17 u/f).
///   Holding A instead flutter-kicks at a slower 12. `OBRRT_SWIM_HOLD=1`
///   uses the TAS cadence instead (28 u/f, [`hold_a`]).
/// - At the surface Mario floats at `waterLevel - 80`; A in a stroke with
///   pitch ≥ 0 and stickY < -60 jumps out (check_water_jump). Otherwise the
///   stick is kept above -60 there so he doesn't.
fn swim(st: &St, target: [f32; 3], button: u16, frame: usize) -> GamePad {
    let surface = st.water_level as f32 - 80.0;
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt();
    // Out of the water: swim along the surface towards it and jump out when
    // close (the jump carries 42 up and his forward speed).
    if target[1] > st.water_level as f32 {
        let at_surface = st.pos[1] + 1.5 >= surface;
        if at_surface && hd < 350.0 {
            return pad(if frame % 4 == 0 { A_BUTTON } else { 0 }, 0.0, -64.0);
        }
    }
    let aim_y = target[1].min(surface);
    let pitch = angle(aim_y - st.pos[1], hd).clamp(-0x3F00, 0x3F00);
    let cy = (-(pitch as f32) / 252.0).clamp(-58.0, 64.0);
    let yaw_err = wrap(angle(dx, dz) as i32 - st.yaw as i32);
    // Turn at a sixth of the error a frame (at most 0x280, stick at 64).
    let cx = if hd < 20.0 { 0.0 } else { (-(yaw_err as f32) / 60.0).clamp(-64.0, 64.0) };
    // The turn rate doesn't depend on speed: at a stroke's 17 a turn takes
    // about 100 frames, a circle 280 across. Coast while turning towards a
    // target that close, or he circles it.
    let want = yaw_err.unsigned_abs() < 0x2000 || hd > 600.0;
    let stroke = want && hold_a(st, frame);
    let mut b = if stroke { A_BUTTON } else { 0 };
    if button & !A_BUTTON != 0 && frame % 4 == 2 {
        // B punches or grabs underwater (act_water_punch).
        b |= button & !A_BUTTON;
    }
    pad(b, cx, cy)
}

const ACT_BREASTSTROKE: u32 = 0x300024D0;
const ACT_SWIMMING_END: u32 = 0x300024D1;
const ACT_FLUTTER_KICK: u32 = 0x300024D2;
const ACT_HOLD_BREASTSTROKE: u32 = 0x300024D3;
const ACT_HOLD_SWIMMING_END: u32 = 0x300024D4;
const ACT_HOLD_FLUTTER_KICK: u32 = 0x300024D5;

/// A for the 20-frame stroke at the 28 cap (13 BREASTSTROKE + 7
/// SWIMMING_END, mario_actions_submerged.c), from the action at the start of
/// the frame:
///
/// - BREASTSTROKE reads only A presses: one in its frames 2-5 restarts it with
///   the strength reset to 160. So A stays up there; at frame 14 he flutter
///   kicks, and A up then goes to SWIMMING_END with strength +10.
/// - SWIMMING_END: A down at frame 7 strokes again, +10 (max 280, 28 u/f);
///   A up resets the strength. So A is held there throughout.
/// - FLUTTER_KICK (A held): A up leaves it.
/// - Anything else (idle, action end) starts a stroke on a press, so A on
///   alternate frames for an edge.
pub fn hold_a(st: &St, frame: usize) -> bool {
    match st.action {
        ACT_BREASTSTROKE | ACT_HOLD_BREASTSTROKE | ACT_FLUTTER_KICK | ACT_HOLD_FLUTTER_KICK => false,
        ACT_SWIMMING_END | ACT_HOLD_SWIMMING_END => true,
        _ => frame % 2 == 0,
    }
}

/// Poles and trees (mario_actions_automatic.c). Mario sits at the pole's
/// centre, facing it.
///
/// - Holding (act_holding_pole): stickY > 16 climbs, and at the top > 50
///   stands on it (ACT_TOP_OF_POLE_TRANSITION, not on giant poles); stickY
///   < -16 slides down, spinning; otherwise `yaw -= 16 * stickX`. A turns
///   him around (`yaw += 0x8000`) and kicks off: ACT_WALL_KICK_AIR, 62 up and
///   at least 24 forward (set_mario_action_airborne, mario.c), so he jumps
///   away from the way he faced.
/// - Climbing (act_climbing_pole): rises `stickY / 8` a frame, turning him
///   towards the camera's yaw; under 8 he holds; A kicks off as above.
/// - On top (act_top_of_pole): `yaw -= 16 * stickX`; A jumps the way he faces
///   (ACT_TOP_OF_POLE_JUMP, same speeds); stickY < -16 goes back down.
///
/// So: climb to the height the jump reaches the target from (sliding down
/// only for targets near the pole), turn his back to it, kick off. A target
/// on the pole itself is climbed or slid to.
fn pole(st: &St, target: [f32; 3], frame: usize) -> GamePad {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt();
    let to_target = angle(dx, dz) as i32;
    let off = hd >= 60.0;
    // Where the kick-off's arc is at the target's distance: about 28 a frame
    // out (24 plus air control), 62 up less 4 a frame of gravity.
    let height = if off {
        let t = hd / 28.0;
        target[1] - (62.0 * t - 2.0 * t * (t + 1.0)).max(-600.0)
    } else {
        target[1]
    };
    let dh = height - st.pos[1];
    let press = frame % 2 == 0;
    match st.action {
        ACT_TOP_OF_POLE => {
            if !off && dh < -100.0 {
                return pad(0, 0.0, -64.0);
            }
            let err = wrap(to_target - st.yaw as i32);
            let aimed = err.unsigned_abs() < 0x600;
            pad(if off && aimed && press { A_BUTTON } else { 0 }, turn(err), 0.0)
        }
        ACT_HOLDING_POLE | ACT_CLIMBING_POLE => {
            if dh > 30.0 {
                // Climbing stops at the top without standing on it; only
                // holding with stickY > 50 does that. Let go of the stick a
                // frame in four so he's holding (and stands up if at the
                // top, else climbs on the same frame).
                return pad(0, 0.0, if frame % 4 == 3 { 0.0 } else { 64.0 });
            }
            // Above that height: from higher up the arc is longer, and air
            // control can shorten it, so only slide down to a target close
            // to the pole (or on it).
            if (!off && dh < -30.0) || (off && dh < -150.0 && hd < 300.0) {
                // Sliding speeds up to 16 a frame; ease off near the height.
                return pad(0, 0.0, if dh < -200.0 { -64.0 } else { -20.0 });
            }
            if !off {
                return pad(0, 0.0, 0.0);
            }
            let err = wrap(to_target + 0x8000 - st.yaw as i32);
            let aimed = err.unsigned_abs() < 0x600;
            // Climbing turns him to the camera: hold still to turn.
            pad(if aimed && press && st.action == ACT_HOLDING_POLE { A_BUTTON } else { 0 }, turn(err), 0.0)
        }
        // Grabbing and the handstand transitions read no stick.
        _ => GamePad::new(0, 0, 0),
    }
}

/// Kicked off a pole (or a wall): ACT_WALL_KICK_AIR and ACT_TOP_OF_POLE_JUMP
/// have ACT_FLAG_CONTROL_JUMP_HEIGHT, so releasing A while rising faster
/// than 20 quarters the speed (should_strengthen_gravity_for_jump_ascent,
/// mario_step.c). Holds A for the full arc [`pole`] plans with unless the
/// target is below; the stick is camera-relative as in the air.
fn kick(st: &St, target: [f32; 3]) -> GamePad {
    let (sx, sy) = camera_stick(st, target);
    GamePad::new(if target[1] > st.pos[1] - 100.0 { A_BUTTON } else { 0 }, sx, sy)
}

/// Stick x for `yaw -= 16 * stickX` to close `err` within a frame or two.
fn turn(err: i16) -> f32 {
    (-(err as f32) / 16.0).clamp(-64.0, 64.0)
}

/// Hanging from a hangable ceiling (act_start_hanging, act_hanging,
/// act_hang_moving): the stick moves him camera-relative (update_hang_moving
/// turns him to `intendedYaw`), but letting go of A drops him. Drops on
/// purpose above a target close below.
fn hang(st: &St, target: [f32; 3]) -> GamePad {
    let (sx, sy) = camera_stick(st, target);
    let drop = target[1] < st.pos[1] - 150.0 && st.hdist(target) < 150.0;
    GamePad::new(if drop { 0 } else { A_BUTTON }, sx, sy)
}

/// In a cannon (act_in_cannon, mario_actions_automatic.c): once it has
/// opened, `pitch -= 10 * stickY` (clamped to 0..0x38E3) and the yaw offset
/// `-= 10 * stickX` (±0x4000 around the cannon's). A fires: 100 along the
/// aim from 120 ahead. In flight gravity is 1 a frame (apply_gravity,
/// mario_step.c), so the low ballistic arc to the target is
/// `tan p = (d - sqrt(d² - 4a(a + dy))) / 2a` with `a = d² / 20000`; out of
/// range, 45°. Fires once aimed, or after 300 frames anyway (a target
/// outside the cannon's yaw range is never aimed at; the cannon itself takes
/// about 120 frames to open after Mario gets in).
fn cannon(st: &St, target: [f32; 3], frame: usize) -> GamePad {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt().max(1.0);
    let dy = target[1] + 50.0 - st.pos[1];
    let a = hd * hd / 20000.0;
    let disc = hd * hd - 4.0 * a * (a + dy);
    let pitch = if disc >= 0.0 { angle((hd - disc.sqrt()) / (2.0 * a), 1.0) } else { 0x2000 };
    let pitch = pitch.clamp(0, 0x38E3) as i32;
    let yaw_err = wrap(angle(dx, dz) as i32 - st.yaw as i32);
    let pitch_err = wrap(pitch - st.pitch as i32);
    let aimed = yaw_err.unsigned_abs() < 0x100 && pitch_err.unsigned_abs() < 0x100;
    let fire = (aimed || frame >= 300) && frame % 2 == 0;
    let cx = (-(yaw_err as f32) / 10.0).clamp(-64.0, 64.0);
    let cy = (-(pitch_err as f32) / 10.0).clamp(-64.0, 64.0);
    pad(if fire { A_BUTTON } else { 0 }, cx, cy)
}

/// Flying with the wing cap (update_flying, mario_actions_airborne.c).
/// `angleVel` ramps towards `-stickY * forwardVel / 5` (pitch; stick up
/// dives) and `-stickX * forwardVel / 4` (yaw; stick right turns right). On
/// top of that the nose drops by `(32 - forwardVel) * 6` a frame (×10 under
/// 16), climbing costs speed and diving gains it. Aims the nose at the
/// target, but dives (pitch -0x1000 or lower) while slower than 24.
fn fly(st: &St, target: [f32; 3]) -> GamePad {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt();
    let fv = st.forward_vel.max(1.0);
    let mut want = angle(target[1] - st.pos[1], hd).clamp(-0x2000, 0x2000) as f32;
    if st.forward_vel < 24.0 {
        want = want.min(-0x1000 as f32);
    }
    let drift = if st.forward_vel > 16.0 {
        (st.forward_vel - 32.0) * 6.0
    } else if st.forward_vel > 4.0 {
        (st.forward_vel - 32.0) * 10.0
    } else {
        -1024.0
    };
    let pitch_vel = (want - st.pitch as f32) / 8.0 - drift;
    let cy = (-5.0 * pitch_vel / fv).clamp(-64.0, 64.0);
    let yaw_err = wrap(angle(dx, dz) as i32 - st.yaw as i32) as f32;
    let cx = (-4.0 * (yaw_err / 8.0) / fv).clamp(-64.0, 64.0);
    pad(0, cx, cy)
}

/// Riding Hoot (hoot_player_set_yaw, hoot.inc.c): his yaw `-= 5 * rawStickX`
/// (|raw| < 10 counts as 0) and he flies along it; Mario's yaw is `0x4000 -
/// hoot yaw` (act_riding_hoot, mario_actions_airborne.c). Letting go of A
/// drops off: done above a target close below.
fn hoot(st: &St, target: [f32; 3]) -> GamePad {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hoot_yaw = 0x4000 - st.yaw as i32;
    let err = wrap(angle(dx, dz) as i32 - hoot_yaw) as f32;
    let raw = (-err / 40.0).clamp(-127.0, 127.0);
    let raw = if raw.abs() < 10.0 { 0 } else { raw as i8 };
    let drop = target[1] < st.pos[1] - 100.0 && st.hdist(target) < 200.0;
    GamePad::new(if drop { 0 } else { A_BUTTON }, raw, 0)
}

/// Holding Bowser (act_holding_bowser, mario_actions_object.c): each frame
/// the spin speeds up by the stick's turn since the last frame / 0x80 (at
/// most 0x80), so the stick goes round a quarter turn a frame; B lets go the
/// way Mario faces, far once spinning at 0xE00 or more.
fn swing_bowser(st: &St, target: [f32; 3], frame: usize) -> GamePad {
    let spin = st.angle_vel[1].unsigned_abs();
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let err = wrap(angle(dx, dz) as i32 - st.yaw as i32).unsigned_abs();
    if spin >= 0xE00 && err < spin {
        return GamePad::new(B_BUTTON, 0, 0);
    }
    // A quarter turn a frame, counter-clockwise (intendedYaw growing).
    let rel = (frame % 4) as f32 * TAU / 4.0;
    GamePad::new(0, (127.0 * rel.sin()) as i8, (-127.0 * rel.cos()) as i8)
}

/// `atan2s(x, y)`'s angle in s16 units of the vector (`y` along the 0
/// direction): `angle(dx, dz)` is a yaw, `angle(dy, hdist)` a pitch.
pub(super) fn angle(x: f32, y: f32) -> i16 {
    wrap((x.atan2(y) * 65536.0 / TAU).round() as i32)
}

pub(super) fn wrap(a: i32) -> i16 {
    a as i16
}

/// A pad whose stick reads as `(cx, cy)` in `controller->stickX/stickY`
/// (±64): the inverse of adjust_analog_stick's ±6 dead zone. Values under 2
/// can't be produced and read as 0.
pub(super) fn pad(button: u16, cx: f32, cy: f32) -> GamePad {
    let raw = |c: f32| {
        let c = c.round();
        if c.abs() < 2.0 { 0 } else { (c + 6.0 * c.signum()).clamp(-127.0, 127.0) as i8 }
    };
    GamePad::new(button, raw(cx), raw(cy))
}
