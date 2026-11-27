//! `Tech::BljAny` (`bljany`): a backwards long jump wherever one can build
//! speed, not only at the preset spots of [`super::blj`].
//!
//! Base predicate [`blj_ok`] (ref/blj-conditions.md section 4.2): growth needs
//! each jump to come down within n < |v| / 1.7 air frames (n <= 9 at the
//! natural start, -16). The gate: controllable, holding nothing, above
//! water, grounded or already in the long-jump chain, and the floor not
//! slippery for its class at its tilt (a landing at fv <= -1 there starts a
//! slide: should_begin_sliding / mario_floor_is_slippery, mario.c). Then the
//! first that holds of:
//! 1. **Elevator:** the floor is an object rising at `oVelY >= 7.5` (each
//!    jump's first quarter step, +7.5, lands on it at once); base states
//!    only need it rising (> 2), as elevators accelerate during the setup.
//! 2. **Ceiling:** headroom `ceil - floor - 160 < 50` (n <= 9).
//! 3. **Slope** (the doc's stand-in for its arc probe, from St alone): a
//!    NOT_SLIPPERY floor with rise s = tan >= 1.25, which lands every jump in
//!    its jump frame from -16 (s |v| >= 20); Mario faces down it.
//! 4. **Fast:** fv <= -53 already: even flat ground grows.
//!
//! Not covered: the doc's clause 3 proper (an arc / riser probe over static
//! geometry). It needs floor and ceiling queries (core::geo::floor_at /
//! ceil_at take game memory), which objectives don't have when they're
//! generated; stairs like the lobby steps are left to the preset.
//!
//! At most 2 objectives per state and one per 400-unit cell, clause and
//! facing quadrant (the key). Elevator and ceiling states get the wall-stall
//! facing (into the wall Mario touches) and one of a 4-way fan.
//!
//! **More kinds** (CATALOGUE G02; switch `bljkinds`, off = the four above
//! only, unchanged):
//! 5. **RisingFloor:** an object floor that rises without oVelY (rotating or
//!    moved by its parent: the LLL drawbridge, Bowser 2's arena, the BitDW
//!    Ferris wheel). "floor_y rose since the parent" needs the parent, which
//!    a base state doesn't carry, so the stand-ins are a tilted object floor
//!    (ny < 0.98) or a floor object whose behaviour is a known platform.
//! 6. **WaterSlope:** the shallow-water band water - 100 < y < water - 40,
//!    where INPUT_ABOVE_SLIDE is never set (mario.c:1346) so no floor class
//!    slides: a slope with ny > 0.2924 (HMC entrance pool, 0x13, ny 0.33).
//!    With the switch, the water gate is y > water - 100 instead of y > water.
//! 7. **BlockedStep:** a blocked quarter step (TTC corner, CCM east): in
//!    ACT_LONG_JUMP at the floor's height (within 2) with vy -2 / -4 (the
//!    jump's air step found no room, so it ends in the next frame), moving
//!    backwards (fv <= -3). A normal long jump is at vy -2 only at its apex,
//!    far above the floor.
//! 8. **StepRiser:** a step riser, under a wall or not (HMC (-6725, 1312,
//!    -2640) by a wall, castle basement (3277, -1279, -2569) without one
//!    until -70): not "a wall behind" (that alone only stalls Mario,
//!    mario_step.c:500, and is everywhere), but growth already seen: in
//!    ACT_LONG_JUMP_LAND at -53 < fv <= -20 on a flat floor (ny > 0.9). A
//!    plain long jump lands at about -16 (air drag +2 a frame below -16), so
//!    a landing at -20 or faster means the jump ended early, on a riser.
//!    Facing as is (the chain's). TAS check (hwlab kinds): HMC lower-step
//!    landings, basement landings from -25, CCM east from -6.6, TTC corner
//!    from -3.3; LLL drawbridge and Bowser 2 as RisingFloor; HMC pool as
//!    WaterSlope (in-band frames).
//!
//! The corner variant (two wall normals within 3 frames) and the geometric
//! step probe need history or geometry queries a base state lacks.
//!
//! Success: a chain landing at <= -100 at least 800 from the start (a new
//! place reached at hyperspeed), or the point 2500 behind the facing, the way
//! a BLJ carries Mario, within dist / 60 frames. ETA: super::blj::build_eta to the success's
//! speed, plus the rest of the 800 at that speed for the place clause. Inputs: the GA with the BLJ overlay
//! (super::blj::overlay), no riser point, so every landing is a press; the
//! genome's stick is used on the press frames.
//!
//! With `hyperwalk` on, BljAny only builds speed: its one success is the
//! **speed** hand-off (the place clauses above are off): in
//! ACT_LONG_JUMP(_LAND), alive, fv <= -150 grown by a BLJ, from a start
//! slower than that. That state is the
//! hand-off: `Tech::HyperWalk` starts from hyperspeed states and steers the
//! speed to a goal, instead of BljAny walking blind along the facing.

use std::sync::Arc;

pub mod params;
pub mod spots;

use obrrt_core::{Eta, Mem, Objective, PadOverlay, Precision, St};

use super::{blj, enabled};

const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
/// ACT_FLAG_AIR (include/sm64.h).
const ACT_FLAG_AIR: u32 = 0x0000_0800;

/// Distance behind the facing of the target point.
const REACH: f32 = 2500.0;
/// Speed the target must be reached at (frames = dist / FAST).
const FAST: f32 = 60.0;
/// A chain landing at least this fast, this far from the start, succeeds.
const HYPER: f32 = 100.0;
const MOVED: f32 = 800.0;
/// A rising floor object faster than this makes a base state (see blj_ok).
const ELEVATOR_BASE_VY: f32 = 2.0;
/// The speed success's landing speed (with `hyperwalk` on).
const HANDOFF: f32 = 150.0;
/// Platform behaviours whose floor can rise without oVelY (CATALOGUE G02).
const PLATFORMS: &[&str] = &[
    "bhvCheckerboardPlatformSub",
    "bhvFerrisWheelPlatform",
    "bhvActivatedBackAndForthPlatform",
    "bhvSquishablePlatform",
    "bhvTiltingBowserLavaPlatform",
    "bhvLllDrawbridge",
    "bhvHmcElevatorPlatform",
    "bhvWdwExpressElevator",
    "bhvWdwSquareFloatingPlatform",
    "bhvRrElevatorPlatform",
    "bhvPlatformOnTrack",
    "bhvFallingBowserPlatform",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Elevator,
    Ceiling,
    Slope,
    Fast,
    RisingFloor,
    WaterSlope,
    BlockedStep,
    StepRiser,
}

/// The floor's class (mario_get_floor_class, mario.c; types from
/// include/surface_terrains.h), and the normal.y at or below which a
/// landing with backward speed slides (mario_floor_is_slippery).
fn slides_below(floor_type: i16) -> (bool, f32) {
    match floor_type {
        // NOT_SLIPPERY, HARD_NOT_SLIPPERY, SWITCH: never slides.
        0x15 | 0x37 | 0x7A => (true, 0.0),
        // SLIPPERY classes.
        0x14 | 0x2A | 0x35 | 0x79 => (false, 0.9397),
        // VERY_SLIPPERY classes.
        0x13 | 0x2E | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => (false, 0.9848),
        _ => (false, 0.7880),
    }
}

/// Which BLJ condition `s` meets, if any, and the facing for it.
pub fn blj_ok(s: &St) -> Option<(Kind, i32)> {
    let more = enabled("bljkinds");
    let grounded = s.action & ACT_FLAG_AIR == 0 || matches!(s.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND);
    let water = s.water_level as f32;
    let under = if more { s.pos[1] < water - 100.0 } else { s.pos[1] < water };
    if !s.controllable() || s.holding || !grounded || under {
        return None;
    }
    // The shallow-water band: no slide test there (mario.c:1346).
    let band = more && s.pos[1] < water - 40.0;
    let (not_slippery, slide_ny) = slides_below(s.floor_type);
    let ny = s.floor_normal[1];
    if ny <= if band { 0.2924 } else { slide_ny } {
        return None;
    }
    let yaw = s.yaw as i32;
    // The jump-frame condition is oVelY >= 7.5, but elevators accelerate
    // (the BoB checkerboard: 4.2, 6.2, then 10 a frame) and the setup takes
    // ~25 frames, so a base state only needs the floor already rising.
    if s.floor_obj_vy.is_some_and(|v| v > ELEVATOR_BASE_VY) {
        return Some((Kind::Elevator, yaw));
    }
    if more && rising_floor(s) {
        return Some((Kind::RisingFloor, yaw));
    }
    let headroom = s.ceil_y - s.floor_y - 160.0;
    if (0.0..50.0).contains(&headroom) {
        return Some((Kind::Ceiling, yaw));
    }
    let rise = (1.0 - ny * ny).max(0.0).sqrt() / ny.max(1e-3);
    // Face down the slope (the normal's horizontal part points downhill).
    let down = (s.floor_normal[0].atan2(s.floor_normal[2]) * 32768.0 / std::f32::consts::PI) as i32;
    if not_slippery && rise >= 1.25 && ny > 0.2924 {
        return Some((Kind::Slope, down));
    }
    if band && rise >= 1.25 {
        return Some((Kind::WaterSlope, down));
    }
    if more && s.action == ACT_LONG_JUMP && s.forward_vel <= -3.0 && (s.pos[1] - s.floor_y).abs() < 2.0 && (-4.5..-1.5).contains(&s.vel[1]) {
        return Some((Kind::BlockedStep, yaw));
    }
    if more && s.action == ACT_LONG_JUMP_LAND && s.forward_vel > -53.0 && s.forward_vel <= -20.0 && ny > 0.9 {
        return Some((Kind::StepRiser, yaw));
    }
    if s.forward_vel <= -53.0 {
        return Some((Kind::Fast, yaw));
    }
    None
}

/// An object floor that may rise without oVelY (see the module doc, 5):
/// tilted, or a known platform behaviour (when `s` has its objects).
fn rising_floor(s: &St) -> bool {
    let Some(slot) = s.floor_obj else { return false };
    s.floor_normal[1] < 0.98 || s.objects().iter().any(|o| o.slot == slot && PLATFORMS.contains(&o.bhv_name()))
}

fn cell(p: [f32; 3]) -> [i32; 2] {
    [(p[0] / 400.0).round() as i32, (p[2] / 400.0).round() as i32]
}

#[derive(Clone, Copy)]
pub struct BljAny {
    zone: (u16, u8),
    cell: [i32; 2],
    kind: Kind,
    yaw: i32,
    start: [f32; 3],
}

impl BljAny {
    /// At most 2 per state. Slopes face downhill and fast states keep their
    /// facing; on an elevator or under a ceiling any facing may work, so:
    /// the one that backs Mario into the wall he touches (a wall stall keeps
    /// him on the platform, as in the BoB TAS), and one of a 4-way fan around
    /// his facing (by frame, so states cover the fan between them).
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        let Some((kind, yaw)) = blj_ok(b) else { return };
        let mut yaws = vec![];
        if matches!(kind, Kind::Elevator | Kind::Ceiling | Kind::RisingFloor) {
            if let Some(n) = b.wall_normal.filter(|n| n[0].abs() + n[2].abs() > 0.1) {
                // The wall's normal points out of it; facing along it, backwards is into it.
                yaws.push((n[0].atan2(n[2]) * 32768.0 / std::f32::consts::PI) as i32);
            }
            yaws.push(yaw + 0x4000 * (b.frame % 4) as i32);
        } else {
            yaws.push(yaw);
        }
        for yaw in yaws {
            out.push(Arc::new(BljAny { zone: b.zone(), cell: cell(b.pos), kind, yaw, start: b.pos }));
        }
    }

    /// The point a BLJ facing `yaw` from `from` heads for.
    fn target(&self, from: [f32; 3]) -> [f32; 3] {
        let r = self.yaw as f32 * std::f32::consts::TAU / 65536.0;
        [from[0] - REACH * r.sin(), from[1], from[2] - REACH * r.cos()]
    }
}

impl Objective for BljAny {
    fn kind(&self) -> &'static str {
        "Tech::BljAny"
    }
    fn key(&self) -> String {
        // One per cell, clause and facing quadrant.
        let quadrant = ((self.yaw + 0x2000).rem_euclid(0x10000)) / 0x4000;
        format!("Tech::BljAny {:?} {:?} {:?} q{quadrant}", self.zone, self.cell, self.kind)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if self.base_criteria(b) { 4.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && cell(s.pos) == self.cell && blj_ok(s).is_some_and(|(k, _)| k == self.kind)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 32, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        let first = blj::is_start(mem);
        let sp = blj::chain_step(s, mem);
        // The rollout's frame index (f[3] is ours in blj's tracker).
        let i = if first { 0.0 } else { mem.f[3] + 1.0 };
        mem.f[3] = i;
        let origin = start.pos;
        let t = self.target(origin);
        let deadline = (REACH / FAST) as f64;
        // The hand-off to Tech::HyperWalk (module doc).
        let speed = HANDOFF;
        let handoff = enabled("hyperwalk") && start.forward_vel > -speed;
        // Only with speed built here: a BLJ re-jump in the rollout and the
        // speed past one x1.5 of the start's (a start already fast carrying
        // on, or a walk to the target, is no BLJ).
        let grown = sp > 0.0 && s.forward_vel < 1.5 * start.forward_vel.min(0.0);
        // With hyperwalk on, BljAny only builds speed: the hand-off is its
        // only success (no place clause).
        let places = !enabled("hyperwalk");
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if (places && s.action == ACT_LONG_JUMP_LAND && s.forward_vel <= -HYPER && s.hdist(origin) >= MOVED && grown)
            || (places && i <= deadline && s.hdist(t) < 400.0 && sp >= HYPER)
            || (handoff && i > 0.0 && blj::long_jumping(s) && s.forward_vel <= -speed && grown)
        {
            return Eta::Done;
        }
        // The chain to the success's speed (set up where Mario is), then for
        // the place clause the rest of the MOVED distance at the chain's speed.
        let target = if handoff { speed } else { HYPER };
        let mut eta = blj::build_eta(s, None, target);
        if places {
            let v = (-s.forward_vel).max(HYPER);
            eta += ((MOVED - s.hdist(origin)).max(0.0) / v) as f64;
        }
        Eta::Frames(eta)
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(blj::steer(s, self.target(s.pos)))
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        // Run-up 0: start where Mario is (the rule stops him, turns him to
        // the facing and runs him out from there).
        Some(blj::overlay(self.start, self.yaw, 0.0, None, 1000.0))
    }
}
