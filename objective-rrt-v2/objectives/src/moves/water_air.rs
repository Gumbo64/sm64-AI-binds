//! Swimming and wing-cap flying as GA policies (`OBRRT_MOVE_GA`),
//! plausible only in their control modes. Both read the raw stick
//! (`controller->stickX/Y`), never the camera.
//!
//! Swimming (ACT_FLAG_SWIMMING without the metal cap,
//! mario_actions_submerged.c): `targetYawVel = -10 stickX` (ramped 0x10-0x40
//! a frame), `targetPitch = -252 stickY` (0x100 a frame below level, 0x200
//! above), speed along the facing; buoyancy +1.25 within 400 of the
//! surface. Speeds measured in run/examples/modes_lab.rs (`swim`, saved in
//! ref/arcs/modes.json): the held-A TAS cadence ([`super::modes::hold_a`])
//! reaches the 28 cap, A every 4 frames about 17, flutter kicking 12.
//!
//! - `swim-tas` (offered only without `OBRRT_SWIM_HOLD=1`, which makes the
//!   default swimming this cadence): the TAS cadence and straight steering, turning at full
//!   stick (stroking through the turn: the circle is wide only at a
//!   stroke's speed for targets close by, where it coasts);
//! - `swim-surface`: up to the surface first and along it, then down at
//!   the target once it is closer than twice its depth (the surface's
//!   buoyancy and no ceilings: for long swims under a shallow obstacle);
//! - `swim-exit`: a target above the water: to the surface and a water
//!   jump (A with pitch >= 0 and stickY < -60 at the surface,
//!   check_water_jump: 42 up, the swim speed kept, at least 15) timed so
//!   its arc (gravity 4) comes down at the target's height at the target,
//!   or, for a target higher than the arc, at the shore (a floor up to 200
//!   above the surface near the arc's top). `swim-tas` exits like the
//!   default controller (a water jump within 350).
//!
//! Flying (ACT_FLYING, update_flying, mario_actions_airborne.c): pitch and
//! yaw velocities ramp towards `-stickY fv/5` and `-stickX fv/4`; the nose
//! drifts by `(fv - 32) * 6` a frame (×10 under 16) and `fv -= 2 sin-ish
//! (pitch / 0x4000) + 0.1` (diving gains speed, climbing costs it), turning
//! costs `0.5 (1 - cos yawVel)`; no gravity. [`fly_step`] is that update
//! exactly, so the policies plan by prediction ([`fly_plan`]: 25 constant
//! sticks, 48 frames each, against the zone's floors):
//!
//! - `fly-plan`: the plan that reaches the target soonest (or ends
//!   nearest, by distance over speed);
//! - `fly-high`: the same towards a point 400 above the target until within
//!   1200 (height is speed later: dive onto it);
//! - `fly-land`: over the target (within 80) with it below, Z: a ground
//!   pound straight down (act_flying); before that, plans towards a point
//!   above it.

use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_SWIMMING, ACT_FLYING};
use obrrt_core::{Arrival, GamePad, St};

use super::geom;
use super::modes::{angle, hold_a, pad, wrap};

const MARIO_METAL_CAP: u32 = 0x4;

/// In the swimming mode (not metal-cap walking).
pub fn swimming(st: &St) -> bool {
    st.action & ACT_FLAG_SWIMMING != 0 && st.mario_flags & MARIO_METAL_CAP == 0
}

/// Where Mario floats at the surface.
pub fn surface(st: &St) -> f32 {
    st.water_level as f32 - 80.0
}

/// Water jump: initial rise and gravity (act_water_jump; set_mario_action
/// airborne).
pub const WATER_JUMP_VY: f32 = 42.0;
pub const GRAVITY: f32 = 4.0;

/// Height of the water jump's arc after `t` frames.
pub fn water_jump_h(t: f32) -> f32 {
    WATER_JUMP_VY * t - GRAVITY / 2.0 * t * (t - 1.0)
}

/// Frames until the arc comes down to `dy` (descending root), None if it
/// never gets that high.
pub fn water_jump_down(dy: f32) -> Option<f32> {
    // -2t² + 44t - dy = 0
    let (a, b, c) = (-GRAVITY / 2.0, WATER_JUMP_VY + GRAVITY / 2.0, -dy);
    let disc = b * b - 4.0 * a * c;
    (disc >= 0.0).then(|| (-b - disc.sqrt()) / (2.0 * a))
}

/// The swim policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Swim {
    Tas,
    Surface,
    Exit,
}

/// Stick x/y and stroke towards a point (aim height `aim_y`).
fn swim_towards(st: &St, target: [f32; 3], aim_y: f32, frame: usize, coast_near: bool) -> GamePad {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt();
    let pitch = angle(aim_y.min(surface(st)) - st.pos[1], hd).clamp(-0x3F00, 0x3F00);
    let cy = (-(pitch as f32) / 252.0).clamp(-58.0, 64.0);
    let yaw_err = wrap(angle(dx, dz) as i32 - st.yaw as i32);
    // Full stick turns at 0x280 a frame; proportional within 0x800.
    let cx = if hd < 20.0 { 0.0 } else { (-(yaw_err as f32) / 32.0).clamp(-64.0, 64.0) };
    let want = !coast_near || yaw_err.unsigned_abs() < 0x2000 || hd > 400.0;
    pad(if want && hold_a(st, frame) { A_BUTTON } else { 0 }, cx, cy)
}

pub fn swim_pad(k: Swim, st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    if !swimming(st) {
        return None;
    }
    let surf = surface(st);
    let at_surface = st.pos[1] + 1.5 >= surf;
    let hd = st.hdist(target);
    match k {
        Swim::Tas => {
            // Out of the water: the default's exit (a water jump within 350).
            if target[1] > st.water_level as f32 && at_surface && hd < 350.0 {
                return Some(pad(if frame % 2 == 0 { A_BUTTON } else { 0 }, 0.0, -64.0));
            }
            Some(swim_towards(st, target, target[1], frame, true))
        }
        Swim::Surface => {
            let depth = surf - target[1];
            let aim = if hd > 2.0 * depth.max(0.0) + 200.0 { surf + 10.0 } else { target[1] };
            Some(swim_towards(st, target, aim, frame, true))
        }
        Swim::Exit => {
            if target[1] <= st.water_level as f32 {
                return None;
            }
            if at_surface {
                let fv = st.forward_vel.max(15.0);
                let dy = target[1] - st.pos[1];
                let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
                let aligned = wrap(angle(dx, dz) as i32 - st.yaw as i32).unsigned_abs() < 0x1000;
                // Jump once the arc's landing point at the target's height
                // is no further than the target (it comes down on it or
                // just short and runs on); for a target higher than the
                // arc, at the shore: a floor 0-200 above the surface where
                // the arc is near its top (8-14 frames out).
                let jump = match water_jump_down(dy) {
                    Some(t) => hd <= fv * t,
                    None => shore_ahead(st, target, fv),
                };
                if aligned && jump && st.pitch >= 0 {
                    return Some(pad(if frame % 2 == 0 { A_BUTTON } else { 0 }, 0.0, -64.0));
                }
            }
            // To the surface, then along it towards the target.
            Some(swim_towards(st, target, surf + 10.0, frame, false))
        }
    }
}

/// A floor 0-200 above the surface 8-14 frames of travel towards the
/// target (the water jump's arc is 260-280 up there).
fn shore_ahead(st: &St, target: [f32; 3], fv: f32) -> bool {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt().max(1.0);
    let wl = st.water_level as f32;
    geom::with_grid(st.zone(), |g| {
        [8.0f32, 11.0, 14.0].iter().any(|t| {
            let d = (fv * t).min(hd);
            g.floor_below(st.pos[0] + dx / hd * d, st.pos[2] + dz / hd * d, wl + 200.0).is_some_and(|f| f.y > wl && f.y < wl + 200.0)
        })
    })
    .unwrap_or(false)
}

fn swim_hold_env() -> bool {
    obrrt_core::config::current().objectives.swim_hold
}

pub fn swim_plausible(k: Swim, st: &St, target: [f32; 3]) -> bool {
    if !swimming(st) {
        return false;
    }
    let surf = surface(st);
    match k {
        // With OBRRT_SWIM_HOLD=1 the default swimming is this cadence
        // already (the lab: no wins, a slot taken): offered without it.
        Swim::Tas => !swim_hold_env(),
        // A target well below the surface and far: else it is swim-tas.
        Swim::Surface => target[1] < surf - 300.0 && st.hdist(target) > 1200.0 && st.pos[1] < surf - 200.0,
        // A target out of the water the jump reaches.
        Swim::Exit => {
            target[1] > st.water_level as f32
                && st.hdist(target) < 4000.0
                // Only where the jump's arc reaches the target's height (the
                // lab: a higher target's shore exits lost to the default).
                && water_jump_down(target[1] - surf).is_some()
        }
    }
}

// ---------------------------------------------------------------------------
// Flying.

/// The flight state [`fly_step`] advances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flyer {
    pub pos: [f32; 3],
    pub fv: f32,
    pub pitch: i16,
    pub yaw: i16,
    pub pitch_vel: i16,
    pub yaw_vel: i16,
}

impl Flyer {
    pub fn of(st: &St) -> Flyer {
        Flyer { pos: st.pos, fv: st.forward_vel, pitch: st.pitch, yaw: st.yaw, pitch_vel: st.angle_vel[0], yaw_vel: st.angle_vel[1] }
    }
}

fn approach_s32(cur: i32, target: i32, inc: i32, dec: i32) -> i32 {
    if cur < target { (cur + inc).min(target) } else { (cur - dec).max(target) }
}

/// One angle-velocity ramp (update_flying_pitch / _yaw): towards `target`
/// with (`up`, `down`) the step sizes, reversing at 0x40 a frame.
fn ramp(cur: i16, target: i16, reverse_cap: i32, a: i32, b: i32) -> i16 {
    let (cur, t) = (cur as i32, target as i32);
    let v = if t > 0 {
        if cur < 0 { (cur + 0x40).min(reverse_cap) } else { approach_s32(cur, t, a, b) }
    } else if t < 0 {
        if cur > 0 { (cur - 0x40).max(-reverse_cap) } else { approach_s32(cur, t, b, a) }
    } else {
        approach_s32(cur, 0, 0x40, 0x40)
    };
    v as i16
}

/// One frame of update_flying with the stick (`cx`, `cy`) as the game reads
/// it (±64), no collisions.
pub fn fly_step(f: &mut Flyer, cx: f32, cy: f32) {
    let tp = -((cy * (f.fv / 5.0)) as i32 as i16);
    f.pitch_vel = ramp(f.pitch_vel, tp, 0x20, 0x20, 0x40);
    let ty = -((cx * (f.fv / 4.0)) as i32 as i16);
    f.yaw_vel = ramp(f.yaw_vel, ty, 0x10, 0x10, 0x20);
    f.yaw = f.yaw.wrapping_add(f.yaw_vel);
    f.fv -= 2.0 * (f.pitch as f32 / 16384.0) + 0.1;
    f.fv -= 0.5 * (1.0 - super::cannon::coss(f.yaw_vel as i32));
    if f.fv < 0.0 {
        f.fv = 0.0;
    }
    let drift = if f.fv > 16.0 {
        (f.fv - 32.0) * 6.0
    } else if f.fv > 4.0 {
        (f.fv - 32.0) * 10.0
    } else {
        -1024.0
    };
    let mut p = (f.pitch as f32 + drift) as i32 as i16 as i32;
    p += f.pitch_vel as i32;
    f.pitch = p.clamp(-0x2AAA, 0x2AAA) as i16;
    let (sp, cp) = (super::cannon::sins(f.pitch as i32), super::cannon::coss(f.pitch as i32));
    let (sy, cyw) = (super::cannon::sins(f.yaw as i32), super::cannon::coss(f.yaw as i32));
    f.pos[0] += f.fv * cp * sy;
    f.pos[1] += f.fv * sp;
    f.pos[2] += f.fv * cp * cyw;
}

/// Frames a plan looks ahead, and its sticks (each held constant).
pub const PLAN_FRAMES: usize = 48;
pub const PLAN_STICKS: [f32; 5] = [-64.0, -32.0, 0.0, 32.0, 64.0];

/// The constant stick (cx, cy) whose predicted flight gets within `reach`
/// of `aim` soonest (else ends nearest, by distance over speed), avoiding
/// predicted floors (sampled every 4 frames). Ties go to the first.
pub fn fly_plan(st: &St, aim: [f32; 3], reach: f32) -> (f32, f32) {
    let zone = st.zone();
    let mut best = (f32::MAX, 0.0, 0.0);
    for &cy in &PLAN_STICKS {
        for &cx in &PLAN_STICKS {
            let mut f = Flyer::of(st);
            let mut score = None;
            for k in 0..PLAN_FRAMES {
                fly_step(&mut f, cx, cy);
                let d = obrrt_core::dist(f.pos, aim);
                if d < reach {
                    score = Some(k as f32);
                    break;
                }
                if k % 4 == 3 && geom::floor_y(zone, f.pos[0], f.pos[2], f.pos[1]).is_some_and(|y| f.pos[1] < y + 20.0) && d > 300.0 {
                    score = Some(f32::MAX / 2.0); // crash
                    break;
                }
            }
            let s = score.unwrap_or_else(|| PLAN_FRAMES as f32 + obrrt_core::dist(f.pos, aim) / f.fv.max(10.0));
            if s < best.0 {
                best = (s, cx, cy);
            }
        }
    }
    (best.1, best.2)
}

/// The fly policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fly {
    Plan,
    High,
    Land,
}

pub fn fly_pad(k: Fly, st: &St, target: [f32; 3], arrival: Arrival, frame: usize) -> Option<GamePad> {
    if st.action != ACT_FLYING {
        return None;
    }
    let hd = st.hdist(target);
    let dy = target[1] - st.pos[1];
    let aim = match k {
        Fly::Plan => target,
        Fly::High if hd > 1200.0 => [target[0], target[1] + 400.0, target[2]],
        Fly::High => target,
        Fly::Land => {
            if hd < 80.0 && dy < -60.0 {
                return Some(GamePad::new(if frame % 2 == 0 { Z_TRIG } else { 0 }, 0, 0));
            }
            [target[0], target[1] + 250.0, target[2]]
        }
    };
    let reach = if arrival == Arrival::Pass || k == Fly::Land { 60.0 } else { 100.0 };
    let (cx, cy) = fly_plan(st, aim, reach);
    Some(pad(0, cx, cy))
}

pub fn fly_plausible(k: Fly, st: &St, target: [f32; 3]) -> bool {
    if st.action != ACT_FLYING {
        return false;
    }
    match k {
        Fly::Plan => true,
        Fly::High => st.hdist(target) > 2000.0,
        // A target below to come down on.
        Fly::Land => target[1] < st.pos[1] - 100.0,
    }
}
