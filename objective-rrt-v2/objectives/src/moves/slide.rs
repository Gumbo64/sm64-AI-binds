//! Slide-course steering as GA policies (`OBRRT_MOVE_GA`), plausible only
//! while Mario slides (butt, stomach or dive slide) on a slide course or a
//! very slippery floor.
//!
//! Physics (update_sliding / update_sliding_angle, mario_actions_moving.c).
//! Every slide action runs the same update; the stick is camera-relative
//! (intendedYaw / intendedMag). With θ the angle between the stick and
//! the slide's velocity and m = intendedMag / 32 (1 at full tilt):
//!
//! - the velocity turns by `0.05 m sin θ` radians a frame (towards the
//!   stick);
//! - it is multiplied by `0.98 + 0.02 m cos θ` (very slippery; 0.96 / 0.92
//!   base on slippery / default floors): the stick straight along the
//!   slide loses nothing, a neutral stick 2% a frame, a full sideways stick
//!   2%, backwards up to 4% (halved by the 10k rule while sliding forwards);
//! - the floor adds `accel * steepness` downhill (10 very slippery).
//!
//! So a heading change Δ held at θ takes `Δ / (0.05 sin θ)` frames and
//! keeps `exp(-0.4 Δ tan(θ/2))` of the speed (on top of the slope's
//! gain): a 90° bend at θ = 90° keeps 53% (31 frames), at 45° 77% (44
//! frames), at 30° 85% (63 frames). The policies differ in that cap.
//!
//! Butt slide vs stomach slide (front vs back): the same update, the same
//! speed, the same turning; they differ only at the ends. Leaving the
//! ground, a butt slide goes to BUTT_SLIDE_AIR and lands back in the butt
//! slide (no loss), a stomach / dive slide to FREEFALL, which lands in a
//! freefall landing (the slide stops). A in a butt slide (after 5 frames)
//! jumps (ACT_JUMP: 42 + fv/4 up, the speed × 0.8); A or B in a stomach /
//! dive slide rolls out (ACT_FORWARD_ROLLOUT). Switching form (jump, dive,
//! land in a dive slide) costs the jump's 20% and the landing, so the
//! policies keep whichever form Mario is in (run/examples/modes_lab.rs
//! `slide` measures both on PSS).
//!
//! - `slide-hold`: the stick along the slide's velocity: no loss, no turn
//!   (offered where the floor probe sees the course run straight on);
//! - `slide-course`: towards the course direction (of the headings within
//!   60° of the velocity, the one whose floor runs on furthest, nearest
//!   the floor's downhill blended with the target's direction), at most
//!   35° off the velocity;
//! - `slide-sharp`: the same, at most 90° (turns 2.5× faster, loses more),
//!   probing only 900 ahead (tight courses: CCM's slide; `slide-course`
//!   probes 30 frames of travel ahead: PSS's long bends);
//! - `slide-jump` (offered only with the shortcut there): `slide-course`,
//!   and A in a butt slide when a bend's
//!   wall is ahead along the velocity and the floor towards the target
//!   continues at least 150 lower within 300-800 (a shortcut over it).
//! - `slide-brake`: the TAS's first-person braking slide (ref/tas/techniques
//!   CATALOGUE G14), offered only walking / decelerating / braking / crouch
//!   sliding down a slide-course or very slippery slope, facing downhill:
//!   C-up enters first person (mario.c: only in actions with
//!   ACT_FLAG_ALLOW_FIRST_PERSON), which makes walking and the crouch slide
//!   brake (begin_braking_action at fv >= 16; act_crouch_slide), and
//!   act_braking under first person skips its exits: `apply_slope_decel`
//!   takes 0.4 a frame on very slippery floors and the slope adds
//!   `5.3 * steepness` (apply_slope_accel) with no 100 cap, along the facing
//!   (no steering). C-up is held while braking; B + A leaves (move punch ->
//!   jump kick, the speed kept) once the facing is 0x2000 off the course,
//!   the floor flattens, or the target is within 600.

use std::f32::consts::{FRAC_PI_2, TAU};

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, U_CBUTTONS};
use obrrt_core::{GamePad, St};

use super::geom;

pub const ACT_BUTT_SLIDE: u32 = 0x0084_0452;
pub const ACT_HOLD_BUTT_SLIDE: u32 = 0x0084_0454;
pub const ACT_STOMACH_SLIDE: u32 = 0x008C_0453;
pub const ACT_HOLD_STOMACH_SLIDE: u32 = 0x008C_0455;
pub const ACT_DIVE_SLIDE: u32 = 0x0088_0456;

pub fn slide_action(a: u32) -> bool {
    matches!(a, ACT_BUTT_SLIDE | ACT_HOLD_BUTT_SLIDE | ACT_STOMACH_SLIDE | ACT_HOLD_STOMACH_SLIDE | ACT_DIVE_SLIDE)
}

/// Zones whose terrain is a slide (every floor very slippery,
/// mario_get_floor_class): PSS, CCM's slide, TTM's slide areas.
pub fn slide_zone(zone: (u16, u8)) -> bool {
    matches!(zone, (27, 1) | (5, 2) | (36, 2) | (36, 3) | (36, 4))
}

/// Whether the slide policies apply: sliding on a slide course or a very
/// slippery floor.
pub fn on_slide(st: &St) -> bool {
    slide_action(st.action) && (slide_zone(st.zone()) || geom::slippery_limit(st.floor_type) > 0.98)
}

/// The slide policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slide {
    Hold,
    Course,
    Sharp,
    Jump,
    Brake,
}

const ACT_WALKING: u32 = 0x0400_0440;
const ACT_DECELERATING: u32 = 0x0400_044A;
const ACT_BRAKING: u32 = 0x0400_0445;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;

/// The braking slide's states: on a downhill slide-course / very slippery
/// floor (ny < 0.99) at 16+ (20+ to start), facing within 0x3000 of
/// downhill.
pub fn brake_state(st: &St) -> bool {
    let a = st.action;
    if !matches!(a, ACT_WALKING | ACT_DECELERATING | ACT_BRAKING | ACT_CROUCH_SLIDE) {
        return false;
    }
    let n = st.floor_normal;
    if !(slide_zone(st.zone()) || geom::slippery_limit(st.floor_type) > 0.98) || n[1] >= 0.99 || n[1] < 0.17 {
        return false;
    }
    let downhill = super::modes::angle(n[0], n[2]) as i32;
    let off = super::modes::wrap(downhill - st.yaw as i32).unsigned_abs();
    off < 0x3000 && st.forward_vel >= if a == ACT_BRAKING { 0.0 } else { 20.0 }
}

fn brake_pad(st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    if !brake_state(st) {
        return None;
    }
    let even = frame % 2 == 0;
    if st.action != ACT_BRAKING {
        // C-up as an edge: first person, then braking.
        return Some(GamePad::new(if even { U_CBUTTONS } else { 0 }, 0, 0));
    }
    let want = course_heading(st, target);
    let face = st.yaw as f32 * TAU / 65536.0;
    let off = wrap_rad(want - face).abs();
    let leave = off > 0.785 || st.floor_normal[1] > 0.985 || st.hdist(target) < 600.0;
    if leave {
        return Some(GamePad::new(U_CBUTTONS | if even { A_BUTTON | B_BUTTON } else { A_BUTTON }, 0, 0));
    }
    Some(GamePad::new(U_CBUTTONS, 0, 0))
}

/// World heading (radians, 0 = +z) of a horizontal vector.
fn heading(x: f32, z: f32) -> f32 {
    x.atan2(z)
}

fn wrap_rad(a: f32) -> f32 {
    let mut a = a % TAU;
    if a > TAU / 2.0 {
        a -= TAU;
    } else if a < -TAU / 2.0 {
        a += TAU;
    }
    a
}

/// The course direction: the floor's downhill (if it slopes) plus the
/// target's direction, as a heading.
pub fn course_heading(st: &St, target: [f32; 3]) -> f32 {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt().max(1.0);
    let n = st.floor_normal;
    let steep = (n[0] * n[0] + n[2] * n[2]).sqrt();
    let (mut x, mut z) = (dx / hd, dz / hd);
    if steep > 0.05 {
        // Downhill is along the normal's horizontal part, weighted up to 1
        // by steepness (a 15° slope counts fully).
        let w = (steep / 0.26).min(1.0);
        x += w * n[0] / steep;
        z += w * n[2] / steep;
    }
    heading(x, z)
}

/// The course direction from the zone's floors: of the headings within
/// ±60° of the velocity (every 7.5°), the ones whose floor continues
/// furthest (ten samples three frames of travel apart, each at most 60
/// above and 0.8 of the spacing below the last: not a lower part of the
/// course), nearest the [`course_heading`] among them; None without a grid.
pub fn course_heading_geo(st: &St, target: [f32; 3], vel_h: f32, long: bool) -> Option<f32> {
    let want = course_heading(st, target);
    let base = if st.floor_y > -10000.0 { st.floor_y } else { st.pos[1] };
    // Ten samples, three frames of travel apart (150-300): 30 frames ahead,
    // what a 90° bend takes at full turn.
    // (`long`; else six 150 apart: 900, for tight courses).
    let (step, n) = if long { ((st.forward_vel.abs() * 3.0).clamp(150.0, 300.0), 10) } else { (150.0, 6) };
    geom::with_grid(st.zone(), |g| {
        let mut best: Option<(usize, f32, f32)> = None;
        for i in -8i32..=8 {
            let h = vel_h + i as f32 * 7.5f32.to_radians();
            let (dx, dz) = (h.sin(), h.cos());
            let mut y = base;
            let mut run = 0;
            for i in 1..=n {
                let d = i as f32 * step;
                match g.floor_below(st.pos[0] + dx * d, st.pos[2] + dz * d, y + 60.0) {
                    Some(f) if f.y > y - 0.8 * step => {
                        y = f.y;
                        run += 1;
                    }
                    _ => break,
                }
            }
            let off = wrap_rad(h - want).abs();
            if best.is_none_or(|(r, o, _)| run > r || run == r && off < o) {
                best = Some((run, off, h));
            }
        }
        best.map(|b| b.2)
    })
    .flatten()
}

/// The stick heading for a desired heading with at most `cap` radians off
/// the velocity.
fn capped(vel_h: f32, want: f32, cap: f32) -> f32 {
    let off = wrap_rad(want - vel_h).clamp(-cap, cap);
    vel_h + off
}

pub fn slide_pad(k: Slide, st: &St, target: [f32; 3], frame: usize) -> Option<GamePad> {
    if k == Slide::Brake {
        return brake_pad(st, target, frame);
    }
    if !on_slide(st) {
        return None;
    }
    let (vx, vz) = (st.vel[0], st.vel[2]);
    let v = (vx * vx + vz * vz).sqrt();
    let vel_h = if v > 1.0 { heading(vx, vz) } else { st.yaw as f32 * TAU / 65536.0 };
    let want = course_heading_geo(st, target, vel_h, k != Slide::Sharp).unwrap_or_else(|| course_heading(st, target));
    let h = match k {
        Slide::Hold => vel_h,
        Slide::Course | Slide::Jump => capped(vel_h, want, 35f32.to_radians()),
        Slide::Sharp => capped(vel_h, want, FRAC_PI_2),
        Slide::Brake => return None,
    };
    let (sx, sy) = super::stick_dir(st, h);
    let mut b = 0;
    if k == Slide::Jump && matches!(st.action, ACT_BUTT_SLIDE | ACT_HOLD_BUTT_SLIDE) && frame % 2 == 0 && shortcut(st, target, vel_h) {
        b = A_BUTTON;
    }
    Some(GamePad::new(b, sx, sy))
}

/// A bend ahead worth jumping: a wall along the velocity within 8 frames
/// of travel, and towards the target a floor 150+ lower 300-800 out.
fn shortcut(st: &St, target: [f32; 3], vel_h: f32) -> bool {
    let fv = st.forward_vel.abs().max(20.0);
    let vd = [vel_h.sin(), vel_h.cos()];
    let base = st.floor_y.max(st.pos[1] - 50.0);
    if geom::wall_ahead(st, vd, fv * 8.0, base, 0.3).is_none() {
        return false;
    }
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    let hd = (dx * dx + dz * dz).sqrt().max(1.0);
    let td = [dx / hd, dz / hd];
    geom::with_grid(st.zone(), |g| {
        [300.0f32, 400.0, 500.0, 650.0, 800.0]
            .iter()
            .filter(|&&d| d < hd + 100.0)
            .any(|&d| g.floor_below(st.pos[0] + td[0] * d, st.pos[2] + td[1] * d, base + 50.0).is_some_and(|f| f.y < base - 150.0 && f.y > base - 900.0))
    })
    .unwrap_or(false)
}

pub fn slide_plausible(k: Slide, st: &St, target: [f32; 3]) -> bool {
    if k == Slide::Brake {
        return brake_state(st) && st.action != ACT_BRAKING || st.action == ACT_BRAKING && brake_state(st) && st.hdist(target) > 600.0;
    }
    if !on_slide(st) {
        return false;
    }
    let (vx, vz) = (st.vel[0], st.vel[2]);
    let vel_h = if vx * vx + vz * vz > 1.0 { heading(vx, vz) } else { st.yaw as f32 * TAU / 65536.0 };
    match k {
        // Holding straight only where the course runs straight on (the
        // floor probe's heading within 7.5° of the velocity): on a bend it
        // slides off (the lab).
        Slide::Hold => course_heading_geo(st, target, vel_h, true).is_some_and(|h| wrap_rad(h - vel_h).abs() < 0.14),
        Slide::Course | Slide::Sharp => true,
        // Only with a shortcut there now (else it is slide-course).
        Slide::Jump => matches!(st.action, ACT_BUTT_SLIDE | ACT_HOLD_BUTT_SLIDE) && st.hdist(target) > 600.0 && shortcut(st, target, vel_h),
        Slide::Brake => false,
    }
}
