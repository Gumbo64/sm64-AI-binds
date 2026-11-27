//! `Tech::PoleRedirect` (`poleredirect`, TAS catalogue G03): hyperspeed
//! into a tree or pole low on its trunk, then turn the stored speed into
//! forward speed in a new direction.
//!
//! The mechanism (decomp):
//! - interact_pole (interaction.c:1530) needs an air action with id
//!   0x80-0x9F and zeroes only `forwardVel` and `vel[1]`: `vel[0]`, `vel[2]`
//!   and `slideVelX/Z` keep the hyperspeed.
//! - When the grab point is below the floor, set_pole_position
//!   (mario_actions_automatic.c:58-93) gives ACT_IDLE at once, with Mario at
//!   the trunk.
//! - On that IDLE frame, stick + Z goes idle -> walking (facing = stick) ->
//!   crouch slide in one frame, and update_sliding recomputes `forwardVel`
//!   as +|slideVel| when the facing is within 90° of the stored travel.
//! - C-up on the same frame puts the camera in C-up mode, so the next frame
//!   the crouch slide becomes ACT_BRAKING (mario_actions_moving.c:1502),
//!   which re-aims the velocity along the facing and loses only ~4 a frame.
//!   In BRAKING with first person on, the stick and A are ignored; B goes to
//!   a move punch, and with A down that becomes a jump kick (fv kept). A on
//!   the crouch slide frame instead jumps (vy 42 + fv/4, G04).
//!
//! TAS 2026 f22607-22611 (BoB tree (4208, 927, 3772)): LONG_JUMP at -415;
//! IDLE fv 0, |v| 415.5; Z + C-up with the stick -> CROUCH_SLIDE +379.7;
//! BRAKING; A + B -> JUMP_KICK 371.8.
//!
//! Aimed variants: per pole also one objective per aim point (the nearest
//! 2 higher poles within 12000 and the climb places of hyperclimb.rs) whose
//! IDLE stick points exactly at it (±2° from the genome), success also
//! needing the travel within 30° of it: a 9300-unit hop to the next tree
//! (TAS BoB) needs the facing right to about 0.5°.
//!
//! Objective: one per pole object (INTERACT_POLE: trees, poles) ahead on
//! Mario's velocity ray (within 25° and 150 to the side, within 3000) while
//! he is at |vel_xz| >= 150 in a grabbing air action, or just after the
//! touch (fv 0 with the speed still in `vel`).
//! - ETA: before the touch the xz distance to the trunk / |vel_xz| + 1 +
//!   [`REDIRECT_FRAMES`] (`Lost` once the speed is gone), after it the
//!   redirect's frames left (`Lost` past the window);
//! - success: within 10 frames of the touch, crouch slide, braking or a
//!   jump with `forward_vel >= 0.8 x` the stored |vel_xz|;
//! - inputs ([`PoleRedirect::pad_overlay`]): before the touch the genome's
//!   stick and no buttons; on the pole (grabbed above the floor) stick down
//!   to slide to the floor; on the IDLE frame the genome's stick at full tilt
//!   (its direction is the redirect; a key frame) + Z + C-up; in the crouch
//!   slide Z + C-up held (or A if the genome presses it: jump now); in
//!   BRAKING A + B.

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, U_CBUTTONS, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_ON_POLE, INTERACT_POLE};
use obrrt_core::{Eta, GamePad, Mem, Objective, PadOverlay, Precision, St};

const ACT_BRAKING: u32 = 0x0400_0445;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_JUMP: u32 = 0x0300_0880;
const ACT_DOUBLE_JUMP: u32 = 0x0300_0881;
const ACT_JUMP_KICK: u32 = 0x0180_08AC;
const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;
/// Actions that carry the redirected speed on.
const CARRY: [u32; 6] = [ACT_CROUCH_SLIDE, ACT_BRAKING, ACT_JUMP, ACT_DOUBLE_JUMP, ACT_JUMP_KICK, ACT_MOVE_PUNCHING];

/// Hyperspeed: below this nothing is worth storing.
pub(super) const MIN_SPEED: f32 = 150.0;
const MAX_DIST: f32 = 3000.0;
/// cos 25°.
const MIN_COS: f32 = 0.906;
/// Sideways miss of the velocity ray (the tree's hitbox radius is 80, and
/// air steering moves Mario sideways at most 10 a frame).
const MAX_MISS: f32 = 150.0;
/// Frames after the touch for the redirect.
const WINDOW: usize = 10;
/// Frames from the touch to the carrying action: IDLE + stick + Z -> crouch
/// slide in one frame, braking the next (TAS f22608-22610).
const REDIRECT_FRAMES: f64 = 2.0;

/// Horizontal speed (|vel_xz|): the stored speed survives the touch here.
pub(super) fn hspeed(s: &St) -> f32 {
    s.vel[0].hypot(s.vel[2])
}

/// The pole objects in `b` (trees, poles: INTERACT_POLE).
pub(super) fn poles(b: &St) -> Vec<[f32; 3]> {
    b.objects().iter().filter(|o| o.interact_type == INTERACT_POLE).map(|o| o.pos).collect()
}

/// Mario is flying at `min_speed`+ in an action interact_pole accepts, with
/// `pole` ahead on his velocity ray.
pub(super) fn approaching(s: &St, pole: [f32; 3], min_speed: f32) -> bool {
    let id = s.action & 0x1FF;
    let v = hspeed(s);
    if s.action & ACT_FLAG_AIR == 0 || !(0x80..0xA0).contains(&id) || v < min_speed {
        return false;
    }
    let (dx, dz) = (pole[0] - s.pos[0], pole[2] - s.pos[2]);
    let d = dx.hypot(dz);
    if !(1.0..=MAX_DIST).contains(&d) || (s.pos[1] - (pole[1] + 150.0)).abs() > 600.0 {
        return false;
    }
    let (ux, uz) = (s.vel[0] / v, s.vel[2] / v);
    let along = dx * ux + dz * uz;
    let miss = (dx * uz - dz * ux).abs();
    along >= MIN_COS * d && miss <= MAX_MISS
}

/// Just touched `pole`: at its trunk, off the pole (IDLE: the grab point
/// was below the floor), with fv 0 and the speed still in `vel`.
pub(super) fn touched(s: &St, pole: [f32; 3], min_speed: f32) -> bool {
    s.action & ACT_FLAG_ON_POLE == 0 && at_trunk(s, pole, min_speed)
}

/// Grabbed `pole` above the floor with the speed still in `vel` (pole
/// actions never write it): sliding down to the floor gives the IDLE of
/// [`touched`] (set_pole_position's floor test; TAS f22607: GRAB_POLE_SLOW
/// before the IDLE).
pub(super) fn on_pole(s: &St, pole: [f32; 3], min_speed: f32) -> bool {
    s.action & ACT_FLAG_ON_POLE != 0 && at_trunk(s, pole, min_speed)
}

fn at_trunk(s: &St, pole: [f32; 3], min_speed: f32) -> bool {
    s.forward_vel.abs() < 0.5 && hspeed(s) >= min_speed && s.hdist(pole) < 60.0 && s.alive()
}

/// Full-tilt camera-relative stick along world yaw `yaw` (s16).
/// adjust_analog_stick (game_init.c) moves each raw axis 6 towards 0 and
/// intendedYaw comes from the adjusted values: aim those, add the 6 back.
fn stick_yaw(s: &St, yaw: f32) -> (i8, i8) {
    let rel = yaw - s.cam_yaw as f32 * std::f32::consts::TAU / 65536.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(121.0 * rel.sin()), raw(-121.0 * rel.cos()))
}

/// Aim points for a redirect at `pole`: the nearest 2 other poles 300+
/// higher within 12000 xz (the next redirect: TAS BoB tree 1 -> tree 2,
/// 9300 away, so the facing must be right to ~0.5°) and the climb places
/// of [`super::hyperclimb::PLACES`] in the zone.
fn aims(b: &St, pole: [f32; 3]) -> Vec<[f32; 3]> {
    let hd = |p: [f32; 3]| (p[0] - pole[0]).hypot(p[2] - pole[2]);
    let mut v: Vec<[f32; 3]> = poles(b).into_iter().filter(|&p| p[1] - pole[1] >= 300.0 && hd(p) <= 12000.0).collect();
    v.sort_by(|a, c| hd(*a).total_cmp(&hd(*c)));
    v.truncate(2);
    v.extend(
        super::hyperclimb::PLACES
            .iter()
            .filter(|p| p.0 == b.level as u32 && p.1 == b.area && p.3[1] - pole[1] >= 300.0)
            .map(|p| p.3),
    );
    v
}

pub struct PoleRedirect {
    pole: [f32; 3],
    /// Aim the redirect here (the IDLE frame's stick points at it, the
    /// genome's stick x bends it ±2°); None: the genome's direction.
    aim: Option<[f32; 3]>,
}

impl PoleRedirect {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if hspeed(b) < MIN_SPEED || !b.controllable() {
            return;
        }
        for pole in poles(b) {
            let o = PoleRedirect { pole, aim: None };
            if o.start(b) {
                out.push(Arc::new(o));
                for aim in aims(b, pole) {
                    out.push(Arc::new(PoleRedirect { pole, aim: Some(aim) }));
                }
            }
        }
    }

    fn start(&self, s: &St) -> bool {
        s.alive()
            && (approaching(s, self.pole, MIN_SPEED) || touched(s, self.pole, MIN_SPEED) || on_pole(s, self.pole, MIN_SPEED))
    }
}

impl Objective for PoleRedirect {
    fn kind(&self) -> &'static str {
        "Tech::PoleRedirect"
    }
    fn key(&self) -> String {
        let p = self.pole;
        match self.aim {
            None => format!("Tech::PoleRedirect {:.0},{:.0},{:.0}", p[0], p[1], p[2]),
            Some(a) => format!("Tech::PoleRedirect {:.0},{:.0},{:.0} aim {:.0},{:.0},{:.0}", p[0], p[1], p[2], a[0], a[1], a[2]),
        }
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if self.start(b) { super::hyperwalk::SPEND_WEIGHT } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.start(s)
    }
    // 3000 at 150 is 20 frames, plus the redirect.
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 5, horizon: 40 })
    }
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        // mem: flag M_TOUCH once the touch happened (the first touched
        // state), f[0] the stored speed there, f[1] frames since it.
        const M_TOUCH: u32 = 1;
        if mem.flags & M_TOUCH == 0 {
            if touched(s, self.pole, MIN_SPEED) {
                mem.flags |= M_TOUCH;
                mem.f[0] = hspeed(s) as f64;
                mem.f[1] = 0.0;
                return Eta::Frames(REDIRECT_FRAMES);
            }
            if !s.alive() || !(hspeed(s) >= MIN_SPEED || on_pole(s, self.pole, MIN_SPEED)) {
                // The speed is gone before the touch: nothing to store.
                return Eta::Lost;
            }
            if on_pole(s, self.pole, MIN_SPEED) {
                // Grabbed above the floor: the slide down to the IDLE touch
                // isn't counted (it would raise the ETA at the grab).
                return Eta::Frames(1.0 + REDIRECT_FRAMES);
            }
            return Eta::Frames(s.hdist(self.pole) as f64 / hspeed(s).max(1.0) as f64 + 1.0 + REDIRECT_FRAMES);
        }
        mem.f[1] += 1.0;
        if mem.f[1] > WINDOW as f64 || !s.alive() {
            return Eta::Lost;
        }
        let stored = mem.f[0] as f32;
        // With an aim, travelling within 30° of it.
        let aimed = self.aim.is_none_or(|a| {
            let (dx, dz) = (a[0] - s.pos[0], a[2] - s.pos[2]);
            let d = dx.hypot(dz).max(1.0);
            (dx * s.vel[0] + dz * s.vel[2]) / (d * hspeed(s).max(1.0)) >= 0.866
        });
        if CARRY.contains(&s.action) && s.forward_vel >= 0.8 * stored && aimed {
            return Eta::Done;
        }
        Eta::Frames((REDIRECT_FRAMES - mem.f[1]).max(1.0))
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Seed: at the tree, then on along the stored travel (+fv).
        if touched(s, self.pole, MIN_SPEED) || s.hdist(self.pole) < 60.0 {
            let v = hspeed(s).max(1.0);
            Some([s.pos[0] + 1000.0 * s.vel[0] / v, s.pos[1], s.pos[2] + 1000.0 * s.vel[2] / v])
        } else {
            Some(self.pole)
        }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        let (pole, aim) = (self.pole, self.aim);
        Some(PadOverlay::keyed(move |traj, played, g| {
            let s = traj.last().unwrap();
            if on_pole(s, pole, MIN_SPEED) {
                // Slide down to the floor (stick down; Z would drop off and
                // lose the speed).
                return (GamePad::new(0, 0, -127), false);
            }
            let Some(t) = traj.iter().position(|s| touched(s, pole, MIN_SPEED)) else {
                // Approach: the genome's stick (it drifts Mario sideways onto
                // the trunk); no buttons (Z or B could end the flight).
                return (GamePad::new(0, g.stick_x, g.stick_y), false);
            };
            let held = |b: u16| played.last().is_some_and(|p| p.button & b != 0);
            if t == traj.len() - 1 {
                // The IDLE frame: stick (full tilt in the genome's direction,
                // else along the stored travel) + Z + C-up.
                let yaw = if let Some(a) = aim {
                    (a[0] - s.pos[0]).atan2(a[2] - s.pos[2]) + g.stick_x as f32 / 127.0 * 2f32.to_radians()
                } else if (g.stick_x as i32).abs() + (g.stick_y as i32).abs() >= 16 {
                    let rel = (g.stick_x as f32).atan2(-(g.stick_y as f32));
                    rel + s.cam_yaw as f32 * std::f32::consts::TAU / 65536.0
                } else {
                    s.vel[0].atan2(s.vel[2])
                };
                let (x, y) = stick_yaw(s, yaw);
                return (GamePad::new(Z_TRIG | U_CBUTTONS, x, y), true);
            }
            match s.action {
                ACT_CROUCH_SLIDE if g.button & A_BUTTON != 0 && !held(A_BUTTON) => {
                    (GamePad::new(A_BUTTON, g.stick_x, g.stick_y), true)
                }
                ACT_CROUCH_SLIDE => (GamePad::new(Z_TRIG | U_CBUTTONS, g.stick_x, g.stick_y), false),
                // A down + a new B: move punch -> jump kick.
                ACT_BRAKING => (GamePad::new(A_BUTTON | if held(B_BUTTON) { 0 } else { B_BUTTON }, 0, 0), false),
                _ => (g, false),
            }
        }))
    }
}
