//! `Tech::SpeedChain` (`speedchain`): BoB's spawn slope into about 90 speed,
//! kept all the way to the elevator ramp, as the 2026 TAS (f22397-22548:
//! 151 frames from control to the ramp; a walked Ramp leg takes about 296).
//!
//! The slope: the slippery face on the right of the spawn; a dive slide down
//! it reaches 90+ at (-5357, 30, 7051). The genome (seeded from the TAS, turned
//! to our camera) finds the way onto it. Once fast, the speed stays with the
//! same chain as the Lakitu skip's hill (lakituskip.rs): no friction, no 48 cap:
//! rollout -> landing -> crouch slide -> slide kick -> its slide -> rollout,
//! or a landing refresh (A held + B on a short stick). Only the buttons are
//! set, and only at speed; the stick and every slow state are the genome's.
//!
//! ETA: to the slope at walking pace, then the rest at 90; once fast, the
//! distance at Mario's closing speed on the setup point, so losing speed or
//! heading away from it costs ETA. Done: grounded
//! at the ramp and controllable.

use std::sync::Arc;

use crate::moves::arcs;
use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::LEVEL_BOB;
use obrrt_core::{Eta, GamePad, Mem, Objective, PadOverlay, Precision, St};

const SLOPE: [f32; 3] = [-5802.0, 571.0, 7625.0];
const RAMP: [f32; 3] = [1600.0, 300.0, 4550.0];
const RAMP_R: f32 = 400.0;
/// Speed above which the chain's rules play.
const FAST: f32 = 55.0;
/// The rules stop here, for the ramp's own approach.
const HANDOFF: f32 = 900.0;
const CHAIN_PACE: f64 = 90.0;

const ACT_FORWARD_ROLLOUT: u32 = 0x0100_08A6;
const ACT_FREEFALL_LAND: u32 = 0x0400_0471;
const ACT_FREEFALL_LAND_STOP: u32 = 0x0C00_0232;
const ACT_JUMP_KICK: u32 = 0x0180_08AC;
const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;
const ACT_PUNCHING: u32 = 0x0080_0380;

pub struct SpeedChain;

impl SpeedChain {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if b.level as u32 == LEVEL_BOB as u32 && b.controllable() && b.pos[0] < RAMP[0] - RAMP_R {
            out.push(Arc::new(SpeedChain));
        }
    }
}

/// Frames in the current action, from the last B press.
fn in_action(traj: &[St], played: &[GamePad]) -> usize {
    let a = traj.last().map_or(0, |s| s.action);
    let mut n = 0;
    for i in (0..traj.len()).rev() {
        if traj[i].action != a {
            break;
        }
        n += 1;
        if i >= 1 && played.get(i - 1).is_some_and(|p| p.button & B_BUTTON != 0) {
            break;
        }
    }
    n
}

fn overlay(traj: &[St], played: &[GamePad], g: GamePad) -> GamePad {
    let Some(s) = traj.last() else { return g };
    if !s.controllable() || s.forward_vel <= FAST || s.hdist(RAMP) < HANDOFF {
        return g;
    }
    let last = played.last().copied().unwrap_or(GamePad::new(0, 0, 0));
    let last_a = last.button & A_BUTTON != 0;
    // The buttons are the chain's; the stick stays the genome's, so the GA
    // can bend the path (scaled down for the landing refresh's short stick).
    let pad = |b: u16, tilt: f32| GamePad::new(b, (g.stick_x as f32 * tilt) as i8, (g.stick_y as f32 * tilt) as i8);
    let kick = || if last_a { pad(A_BUTTON | B_BUTTON, 0.38) } else { pad(A_BUTTON, 1.0) };
    match s.action {
        arcs::ACT_DIVE_SLIDE => {
            if s.floor_normal[1] > 0.99 {
                pad(if last_a { 0 } else { A_BUTTON }, 1.0)
            } else {
                pad(0, 1.0)
            }
        }
        ACT_FORWARD_ROLLOUT | ACT_JUMP_KICK => pad(A_BUTTON, 1.0),
        arcs::ACT_SLIDE_KICK => pad(0, 1.0),
        arcs::ACT_SLIDE_KICK_SLIDE => pad(if last_a { 0 } else { A_BUTTON }, 1.0),
        arcs::ACT_CROUCH_SLIDE => pad(Z_TRIG | B_BUTTON, 1.0),
        ACT_FREEFALL_LAND_STOP | ACT_PUNCHING | ACT_MOVE_PUNCHING => kick(),
        ACT_FREEFALL_LAND => {
            if in_action(traj, played) >= 4 {
                if s.floor_normal[1] < 0.8 { kick() } else { pad(Z_TRIG, 1.0) }
            } else {
                pad(A_BUTTON, 1.0)
            }
        }
        _ => g,
    }
}

impl Objective for SpeedChain {
    fn kind(&self) -> &'static str {
        "Tech::SpeedChain"
    }
    fn key(&self) -> String {
        "Tech::SpeedChain bob-ramp".into()
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if self.base_criteria(b) { 1.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_BOB as u32 && s.controllable() && s.pos[0] < RAMP[0] - RAMP_R
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        // A tree's budget (a route leg gives its own): in the zone-locked BoB tree
        // of 2026-11-09 the old 128 x 60 took 17.5% of the frames in 2 attempts.
        Some(obrrt_core::GaBudget { population: 24, generations: 5, horizon: 320 })
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        const M_FAST: u32 = 1;
        if s.level as u32 != LEVEL_BOB as u32 || !s.alive() {
            return Eta::Lost;
        }
        if s.forward_vel > FAST {
            mem.flags |= M_FAST;
        }
        let d = s.hdist(RAMP);
        if d <= RAMP_R && s.action & 0x800 == 0 && s.controllable() {
            return Eta::Done;
        }
        let rest = (d - RAMP_R).max(0.0) as f64;
        if mem.flags & M_FAST != 0 {
            // Closing speed, not speed: going fast the wrong way isn't progress,
            // so the direct line to the setup point is what pays.
            let (dx, dz) = (RAMP[0] - s.pos[0], RAMP[2] - s.pos[2]);
            let closing = (s.vel[0] * dx + s.vel[2] * dz) / dx.hypot(dz).max(1.0);
            return Eta::Frames(rest / closing.max(obrrt_core::PACE as f32) as f64);
        }
        Eta::Frames(s.hdist(SLOPE) as f64 / obrrt_core::PACE + (obrrt_core::dist(SLOPE, RAMP) as f64 - RAMP_R as f64) / CHAIN_PACE)
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(if s.forward_vel > FAST || s.pos[0] > SLOPE[0] + 1500.0 { RAMP } else { SLOPE })
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(PadOverlay::new(overlay))
    }
}
