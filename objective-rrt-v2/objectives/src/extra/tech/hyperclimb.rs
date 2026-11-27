//! `Tech::HyperClimb` (`hyperclimb`, TAS catalogue G04): turn positive
//! hyperspeed (fv >= 150, after a pole redirect, G03) into height.
//!
//! The mechanism (decomp):
//! - set_mario_y_vel_based_on_fspeed (mario.c:788-829) adds fv / 4 to a
//!   jump's, double jump's or steep jump's vy with no cap (vy = 42 or 52 +
//!   fv / 4), and the jump keeps 0.8 fv. Gravity is 4 a frame, so the apex
//!   is about vy² / 8 up: +2300 at fv 380.
//! - ACT_DIVE caps fv at 48 (mario.c:859) and keeps vy: a dive near the
//!   target ends the flight steeply instead of overshooting.
//! - A wall kick sets vy 62 and keeps fv >= 24 (mario.c:831): a kick off
//!   AIR_HIT_WALL (needs fv > 16, airborne.c:393) keeps the speed going up.
//! - From the redirect's crouch slide A jumps at once (act_crouch_slide
//!   reads A before first person); in BRAKING with first person on only B
//!   acts (a move punch, a jump kick with A down); on a jump's landing a new
//!   A is a double jump.
//!
//! TAS 2026 f22609-22672 (BoB): redirect +380 -> jump kick -> landing A:
//! JUMP vy 131 fv 298 -> second tree at (6799, 2008, -5587), +1080 ->
//! redirect +278 -> double jump vy 115 -> wall kick -> double jump ->
//! summit (2359, 4293, -6242).
//!
//! Objectives from a base with fv >= 150 on the ground (any non-air,
//! non-water action: crouch slide, braking, landings, walking):
//! - one generic: +1000 height (a point 1500 up along the travel);
//! - one per high target in the zone with dy >= 300 within max(6000, 35 fv)
//!   xz: stars
//!   and their spawn points, ! boxes, King Bob-omb, plus the places in
//!   [`PLACES`] (BoB summit, floating island), and poles / trees (a touch at
//!   hyperspeed is the next redirect): the nearest 3 objects, the nearest 2
//!   poles and every place in reach.
//! - ETA: xz / |vel_xz| + dy / PACE (+ the launch's and landing's frames);
//! - success within 90 frames, alive: standing or hanging on the target
//!   (within its radius xz, at most 150 below it; for a pole, the IDLE
//!   touch at its trunk with the speed kept) or a star more; for the
//!   generic one, standing / hanging 1000+ above the start;
//! - only after a launch (in the air rising at 80+: a jump at fv >= 110;
//!   no credit for a plain jump, a wall kick alone or walking up) do
//!   height, target and success count;
//! - inputs (a PadOverlay over the genome): on the ground a new A (the
//!   genome's A) launches, in BRAKING A + B; in the air A on the first
//!   AIR_HIT_WALL frames (wall kick), the genome's A otherwise (height
//!   control), B (dive) once the target's xz is within
//!   48 x frames-to-apex and above-ish; Z never (no long jump: capped at
//!   48, and no ground pound); the stick is the genome's (seeded at the
//!   target).

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_HANGING, ACT_FLAG_ON_POLE, ACT_FLAG_SWIMMING, LEVEL_BOB};
use obrrt_core::{Eta, GamePad, Mem, Objective, PadOverlay, Precision, St, frames_for};

const ACT_BRAKING: u32 = 0x0400_0445;
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;

/// Speed for a hyperspeed launch.
const MIN_FV: f32 = 150.0;
/// Frames a climb gets.
const WINDOW: usize = 90;
const MIN_DY: f32 = 300.0;
const MAX_XZ: f32 = 6000.0;
/// The generic success: this much higher.
const HEIGHT: f32 = 1000.0;
/// Behaviours of high targets.
const TARGETS: &[&str] =
    &["bhvStar", "bhvStarSpawnCoordinates", "bhvSpawnedStar", "bhvSpawnedStarNoLevelExit", "bhvHiddenStar", "bhvExclamationBox", "bhvKingBobomb"];
/// Fixed high places (level, area, name, floor point, radius).
pub(super) const PLACES: &[(u32, u8, &str, [f32; 3], f32)] = &[
    // King Bob-omb's arena (TAS landing (2359, 4293, -6242), slides to
    // (1973, 4293, -5869); the King at (1636, 4242, -5567)).
    (LEVEL_BOB, 1, "bob-summit", [1900.0, 4293.0, -5900.0], 900.0),
    // The floating island's top (TAS dive landing, CATALOGUE 3.2).
    (LEVEL_BOB, 1, "bob-island", [5501.0, 3072.0, 559.0], 400.0),
    // The TAS's high ledge between its wall kick and its last double jump
    // (2026 f22664: (3368, 3617, -7073)).
    (LEVEL_BOB, 1, "bob-north-ledge", [3368.0, 3613.0, -7073.0], 400.0),
    // Flat ground just past the slippery east slope.
    (LEVEL_BOB, 1, "bob-above-slope", [7642.0, 1971.0, -6140.0], 500.0),
];

fn grounded(s: &St) -> bool {
    s.action & (ACT_FLAG_AIR | ACT_FLAG_SWIMMING) == 0
}

/// Settled on a floor, ledge or pole. A grounded state still moving up
/// fast is no landing (Mario squeezed up the steep walls at the level's
/// edge, x 8192 in BoB, reads as JUMP_LAND with vy 90+).
fn standing(s: &St) -> bool {
    s.alive()
        && s.action != 0
        && ((grounded(s) && s.vel[1].abs() < 10.0) || s.action & (ACT_FLAG_HANGING | ACT_FLAG_ON_POLE) != 0)
}

#[derive(Clone, Copy)]
struct Target {
    name: &'static str,
    pos: [f32; 3],
    radius: f32,
}

pub struct HyperClimb {
    zone: (u16, u8),
    /// None: the generic +1000.
    target: Option<Target>,
}

impl HyperClimb {
    pub fn applies(b: &St) -> bool {
        b.controllable() && b.alive() && !b.holding && b.forward_vel >= MIN_FV && grounded(b)
    }

    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if !Self::applies(b) {
            return;
        }
        let zone = b.zone();
        out.push(Arc::new(HyperClimb { zone, target: None }));
        let reach = |t: &Target| t.pos[1] - b.pos[1] >= MIN_DY && b.hdist(t.pos) <= max_xz(b.forward_vel);
        let mut ts: Vec<Target> = b
            .objects()
            .iter()
            .filter(|o| TARGETS.contains(&o.bhv_name()))
            .map(|o| Target { name: o.bhv_name(), pos: o.pos, radius: 300.0 })
            .filter(reach)
            .collect();
        // High poles and trees: touching one at hyperspeed is the next
        // redirect (G03; TAS: the second BoB tree, +1080).
        let mut poles: Vec<Target> = super::poleredirect::poles(b)
            .into_iter()
            .map(|pos| Target { name: "pole", pos, radius: 60.0 })
            .filter(reach)
            .collect();
        poles.sort_by(|x, y| cost(b, x.pos).total_cmp(&cost(b, y.pos)));
        poles.truncate(2);
        ts.sort_by(|x, y| cost(b, x.pos).total_cmp(&cost(b, y.pos)));
        ts.truncate(3);
        ts.extend(poles);
        // The curated places always (they're what the climbs are for).
        ts.extend(
            PLACES
                .iter()
                .filter(|p| p.0 == zone.0 as u32 && p.1 == zone.1)
                .map(|p| Target { name: p.2, pos: p.3, radius: p.4 })
                // Height only: the summit is 9740 xz from the BoB tree
                // redirect (+278, reach 9730), and the TAS gets there via
                // the second tree.
                .filter(|t| t.pos[1] - b.pos[1] >= MIN_DY),
        );
        for t in ts {
            out.push(Arc::new(HyperClimb { zone, target: Some(t) }));
        }
    }

    fn goal(&self, s: &St) -> [f32; 3] {
        match self.target {
            Some(t) => t.pos,
            None => {
                let v = s.vel[0].hypot(s.vel[2]).max(1.0);
                [s.pos[0] + 1500.0 * s.vel[0] / v, s.pos[1] + 1500.0, s.pos[2] + 1500.0 * s.vel[2] / v]
            }
        }
    }

    fn on_target(&self, s: &St) -> bool {
        self.target.is_some_and(|t| {
            if t.name == "pole" {
                // The touch that keeps the speed for the next redirect: IDLE
                // at the trunk (a grab above the floor spins and slides for
                // tens of frames first).
                super::poleredirect::touched(s, t.pos, super::poleredirect::MIN_SPEED)
            } else {
                standing(s) && s.hdist(t.pos) <= t.radius && s.pos[1] >= t.pos[1] - 150.0
            }
        })
    }
}

/// A hyperspeed launch: in the air rising at 80+ (a jump's 42/52 + fv / 4
/// at fv >= 110; a plain jump or wall kick is 62 at most).
const LAUNCH_VY: f32 = 80.0;

/// A launch frame: credit (height, target, success) only counts from the
/// first one, so a normal jump or a walk up earns nothing.
fn launched(s: &St) -> bool {
    s.action & ACT_FLAG_AIR != 0 && s.vel[1] >= LAUNCH_VY
}

/// Frames to the launch from a hyperspeed ground state: a new A (from
/// BRAKING a jump kick, then a landing A), ~3 frames (TAS f22609-22612).
const LAUNCH_FRAMES: f64 = 3.0;
/// Frames to settle after coming down on the target (a landing; rough).
const LAND_FRAMES: f64 = 2.0;

/// How far a launch at `fv` reaches (TAS: 9300 in 31 frames at 298).
fn max_xz(fv: f32) -> f32 {
    MAX_XZ.max(35.0 * fv)
}

fn cost(s: &St, p: [f32; 3]) -> f32 {
    s.hdist(p) + 2.0 * (p[1] - s.pos[1]).abs()
}

impl Objective for HyperClimb {
    fn kind(&self) -> &'static str {
        "Tech::HyperClimb"
    }
    fn key(&self) -> String {
        match self.target {
            Some(t) => format!("Tech::HyperClimb {}:{} {} {:.0},{:.0},{:.0}", self.zone.0, self.zone.1, t.name, t.pos[0], t.pos[1], t.pos[2]),
            None => format!("Tech::HyperClimb {}:{} up", self.zone.0, self.zone.1),
        }
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if self.base_criteria(b) { super::hyperwalk::SPEND_WEIGHT } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        Self::applies(s)
            && s.zone() == self.zone
            && self.target.is_none_or(|t| t.pos[1] - s.pos[1] >= MIN_DY && s.hdist(t.pos) <= max_xz(s.forward_vel))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 5, horizon: WINDOW })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // mem: f[0] the rollout's frame index, flag M_LAUNCH once a launch
        // frame was seen (credit only counts from there).
        const M_SEEN: u32 = 1;
        const M_LAUNCH: u32 = 2;
        let i = if mem.flags & M_SEEN == 0 { 0.0 } else { mem.f[0] + 1.0 };
        mem.f[0] = i;
        mem.flags |= M_SEEN;
        if launched(s) {
            mem.flags |= M_LAUNCH;
        }
        let launched = mem.flags & M_LAUNCH != 0;
        if i > WINDOW as f64 || s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if launched
            && i >= 1.0
            && (self.on_target(s)
                || (self.target.is_none() && standing(s) && s.pos[1] >= start.pos[1] + HEIGHT)
                || (self.target.is_some_and(|t| t.name.contains("Star")) && s.stars > start.stars))
        {
            return Eta::Done;
        }
        // The flight: xz at Mario's horizontal speed (at least PACE), the
        // height left at PACE; before the launch, its frames too.
        let (xz, up) = match self.target {
            Some(t) => (s.hdist(t.pos), t.pos[1] - s.pos[1]),
            None => (0.0, start.pos[1] + HEIGHT - s.pos[1]),
        };
        let v = s.vel[0].hypot(s.vel[2]).max(obrrt_core::PACE as f32);
        let flight = (xz / v) as f64 + frames_for(up.max(0.0)) + if standing(s) { 0.0 } else { LAND_FRAMES };
        Eta::Frames(flight + if launched { 0.0 } else { LAUNCH_FRAMES })
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(match self.target {
            Some(t) => t.pos,
            None => self.goal(s),
        })
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        let target = self.target.map(|t| t.pos);
        Some(PadOverlay::keyed(move |traj, played, g| {
            let s = traj.last().unwrap();
            let held = |b: u16| played.last().is_some_and(|p| p.button & b != 0);
            let (sx, sy) = (g.stick_x, g.stick_y);
            if grounded(s) {
                if s.action == ACT_BRAKING {
                    // First person: only B acts (move punch -> jump kick
                    // with A down); without it A jumps.
                    if g.button & (A_BUTTON | B_BUTTON) != 0 && !held(B_BUTTON) {
                        return (GamePad::new(A_BUTTON | B_BUTTON, sx, sy), true);
                    }
                    return (GamePad::new(0, sx, sy), false);
                }
                // A new A launches (jump / double jump / steep jump); the
                // genome picks the frame. Never Z (long jump caps at 48).
                if g.button & A_BUTTON != 0 && !held(A_BUTTON) && s.forward_vel >= 100.0 {
                    return (GamePad::new(A_BUTTON, sx, sy), true);
                }
                return (GamePad::new(0, sx, sy), false);
            }
            if s.action == ACT_AIR_HIT_WALL {
                // Wall kick: a new A within the wall frames (release first
                // if A was held for the jump's height).
                return if held(A_BUTTON) { (GamePad::new(0, sx, sy), false) } else { (GamePad::new(A_BUTTON, sx, sy), true) };
            }
            if let Some(t) = target {
                // Dive once the rest of the way is within the apex's reach
                // at the dive's 48, with the target below the apex.
                let apex_frames = (s.vel[1] / 4.0).max(0.0);
                let apex = s.pos[1] + s.vel[1] * s.vel[1] / 8.0;
                if s.forward_vel > 48.0 && s.hdist(t) <= 48.0 * apex_frames.max(3.0) && apex >= t[1] && g.button & B_BUTTON != 0 {
                    return (GamePad::new(B_BUTTON, sx, sy), true);
                }
            }
            // The genome's A (releasing it while rising cuts the jump's vy
            // to a quarter: the TAS does that to come down on the second BoB
            // tree, f22626); no Z or B otherwise.
            (GamePad::new(g.button & A_BUTTON, sx, sy), false)
        }))
    }
}
