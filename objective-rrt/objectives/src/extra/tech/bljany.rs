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
//! **Riser** (switch `OBRRT_BLJ_RISERS=1`, default off; needs the mover's
//! published geometry, i.e. `OBRRT_MOVES=1`): a stand-in for clause 3 on
//! stairs, from an external script's riser probe ([`riser`],
//! ref/blj-conditions.md section 6): the nearest short static riser (15-60
//! high, >= 100 long, lower floor in front, higher behind) within 1500 of a
//! grounded base gives a `Stair` (facing away, 60 in front) and a `Side`
//! (facing along it, 13 in front) objective whose overlay walks to the
//! setup point, stops, walks, Z, Z+A, and runs the chain; the GA picks the
//! facing offset, walk length, drift, each A gap and press stick. Lab
//! (run/examples/riser_lab.rs): 1 of 3 risers to -388 with a 300-combo
//! sweep, 0 of 29 with 40 combos; 50 M smoke: 20 attempts, 0 successes.
//!
//! **Wall** (switch `OBRRT_BLJ_WALLS=1`, default off; also needs
//! `OBRRT_MOVES=1`): the wall-pinned BLJ, ref/blj-conditions.md 6.6: a
//! short riser ending against a tall wall ([`riser::corners`]); the 3
//! nearest corners per grounded base. The overlay walks out from the pin,
//! long-jumps back into it, and chains by rule: press from the foot floor
//! whenever A is up, stick back turned toward the step by max(asin((d +
//! 1) / 10), a speed schedule); the genome nudges facing, drift and
//! offsets on its B frames only. Lab: one GA attempt reaches the -150
//! hand-off in 7 / 57 attempts (lobby, WF, basement).
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
//! a BLJ carries Mario, within dist / 60 frames. Progress: the graded BLJ
//! metric (super::blj::metric). Inputs: the GA with the BLJ overlay
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

pub mod riser;

use obrrt_core::{Budget, Objective, PadOverlay, Precision, Progress, St};

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
/// The `Wall` kind's speed hand-off.
const WALL_HANDOFF: f32 = 300.0;
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
    /// A short static riser near Mario (OBRRT_BLJ_RISERS=1; see [`riser`]).
    Riser,
    /// A wall-pinned BLJ: a short riser ending against a tall wall
    /// (OBRRT_BLJ_WALLS=1; [`riser::Corner`]).
    Wall,
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

pub struct BljAny {
    zone: (u16, u8),
    cell: [i32; 2],
    kind: Kind,
    yaw: i32,
    start: [f32; 3],
    /// The `Riser` clause's riser and mode (None for the other clauses).
    riser: Option<(riser::Riser, RMode)>,
    /// The `Wall` clause's pinned spot.
    pin: Option<riser::Corner>,
}

impl BljAny {
    /// At most 2 per state. Slopes face downhill and fast states keep their
    /// facing; on an elevator or under a ceiling any facing may work, so:
    /// the one that backs Mario into the wall he touches (a wall stall keeps
    /// him on the platform, as in the BoB TAS), and one of a 4-way fan around
    /// his facing (by frame, so states cover the fan between them).
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if risers_on() {
            riser_generate(b, out);
        }
        if walls_on() {
            wall_generate(b, out);
        }
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
            out.push(Arc::new(BljAny { zone: b.zone(), cell: cell(b.pos), kind, yaw, start: b.pos, riser: None, pin: None }));
        }
    }

    /// The point a BLJ facing `yaw` from `from` heads for.
    fn target(&self, from: [f32; 3]) -> [f32; 3] {
        let r = self.yaw as f32 * std::f32::consts::TAU / 65536.0;
        [from[0] - REACH * r.sin(), from[1], from[2] - REACH * r.cos()]
    }
}

impl Objective for BljAny {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        "Tech::BljAny"
    }
    fn key(&self) -> String {
        if let Some(c) = &self.pin {
            return format!("Tech::BljAny {:?} Wall ({:.0}, {:.0}, {:.0})", self.zone, c.pin[0], c.pin[1], c.pin[2]);
        }
        if let Some((r, mode)) = &self.riser {
            let c = r.mid();
            return format!("Tech::BljAny {:?} Riser ({:.0}, {:.0}, {:.0}) {mode:?}", self.zone, c[0], c[1], c[2]);
        }
        // One per cell, clause and facing quadrant.
        let quadrant = ((self.yaw + 0x2000).rem_euclid(0x10000)) / 0x4000;
        format!("Tech::BljAny {:?} {:?} {:?} q{quadrant}", self.zone, self.cell, self.kind)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if self.base_criteria(b) { 4.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        if self.riser.is_some() {
            return s.zone() == self.zone && riser_base(s) && s.hdist(self.start) < RISER_REACH && (s.pos[1] - self.start[1]).abs() < 300.0;
        }
        s.zone() == self.zone && cell(s.pos) == self.cell && blj_ok(s).is_some_and(|(k, _)| k == self.kind)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.start) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let s0 = &traj[0];
        let t = self.target(s0.pos);
        let deadline = (REACH / FAST) as usize;
        // The hand-off to Tech::HyperWalk (module doc).
        // The Wall chain is stationary and keeps growing: its hand-off is
        // at -300 (-150 stays progress, the BLJ metric's).
        let speed = if self.pin.is_some() { WALL_HANDOFF } else { HANDOFF };
        let handoff = enabled("hyperwalk") && s0.forward_vel > -speed;
        // Only with speed built here: a BLJ re-jump in the trajectory and
        // the speed past one x1.5 of the start's (a start already fast
        // carrying on, or a walk to the target, is no BLJ).
        let sp = blj::chain_speed(traj);
        let grown = |i: usize, s: &St| sp[i] > 0.0 && s.forward_vel < 1.5 * s0.forward_vel.min(0.0);
        // With hyperwalk on, BljAny only builds speed: the hand-off is its
        // only success (no place clause).
        let places = !enabled("hyperwalk");
        traj.iter().enumerate().position(|(i, s)| {
            s.zone() == self.zone
                && s.alive()
                && ((places && s.action == ACT_LONG_JUMP_LAND && s.forward_vel <= -HYPER && s.hdist(s0.pos) >= MOVED && grown(i, s))
                    || (places && i <= deadline && s.hdist(t) < 400.0 && sp[i] >= HYPER)
                    || (handoff && i > 0 && blj::long_jumping(s) && s.forward_vel <= -speed && grown(i, s)))
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // The chain metric, at the best chain's state (none: the start).
        blj::progress(traj)
    }
    fn budget(&self) -> Budget {
        if self.riser.is_some() {
            // Room for the walk to the setup point.
            return Budget::Ga { population: 32, generations: 5, horizon: 150 + RISER_SETUP_FRAMES };
        }
        Budget::Ga { population: 32, generations: 5, horizon: 150 }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if self.riser.is_some() && !blj::long_jumping(s) && s.forward_vel > -blj::CHAIN_BUILT {
            // Before the chain: the setup point (the overlay walks the last 100).
            return Some(self.start);
        }
        Some(blj::steer(s, self.target(s.pos)))
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        if let Some(c) = self.pin {
            return Some(wall_overlay(c, self.start));
        }
        if let Some((r, mode)) = self.riser {
            return Some(riser_overlay(r, mode, self.start));
        }
        // Run-up 0: start where Mario is (the rule stops him, turns him to
        // the facing and runs him out from there).
        Some(blj::overlay(self.start, self.yaw, 0.0, None, 1000.0))
    }
}

// ---- The Riser clause (OBRRT_BLJ_RISERS=1; default off: nothing changes) ----
//
// From ref/blj-conditions.md section 6 (an external script's riser probe,
// labbed in run/examples/riser_lab.rs). The objective gives only the rough
// place (a riser found by [`riser`] over the zone's published static
// geometry near Mario, and a setup point in front of it); the overlay does
// what the decomp fixes (stop at the point, walk, Z, Z+A, the chain's Z and
// fresh A presses) and the GA picks the rest: the facing offset, the walk
// length, the first jump's drift, and in the chain each press's frame (the
// A gap) and stick offset.

/// OBRRT_BLJ_RISERS=1: the `Riser` clause.
fn risers_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_BLJ_RISERS").is_ok_and(|v| v == "1"))
}

/// The setup: facing away from the riser (stair: backwards into it) or
/// along it (side, either way: the stick's sideways strain carries the
/// jump onto the step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RMode {
    Stair,
    SideL,
    SideR,
    Wall,
}

/// Bases within this of the setup point (horizontally).
const RISER_REACH: f32 = 1500.0;
/// Horizon added for the walk to the setup point.
const RISER_SETUP_FRAMES: usize = 150;
/// The setup point: half way along, this far in front (blj.py's landing
/// shapes: stair 0..60 facing into it, side -1..8 parallel).
const STAIR_D0: f32 = 60.0;
const SIDE_D0: f32 = 13.0;
/// Idle within this of the setup point is the pose.
const RISER_POSE: f32 = 10.0;
/// Side mode only on risers this low.
const SIDE_H: f32 = 30.0;
const ACT_IDLE: u32 = 0x0C40_0201;

fn riser_base(s: &St) -> bool {
    s.controllable()
        && !s.holding
        && s.action & ACT_FLAG_AIR == 0
        && s.pos[1] > s.water_level as f32
        && s.forward_vel > -FAST
}

fn riser_generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    obrrt_core::geo::want_static_floors();
    if !riser_base(b) {
        return;
    }
    let Some(all) = riser::zone_risers(b.zone()) else { return };
    let Some(r) = riser::rank(&all, b.pos, RISER_REACH, None, 1).into_iter().next() else { return };
    // Side: the direction along the riser nearer Mario's facing.
    let away = r.away_yaw();
    let side = if ((b.yaw as i32 - (away + 0x4000)) as i16).unsigned_abs() < 0x4000 { RMode::SideL } else { RMode::SideR };
    // Side only on risers <= 30: the ground wall probe (y + 30, radius
    // 24) keeps Mario 24 out of a taller one (ref/blj-conditions.md 6.2).
    let modes: &[RMode] = if r.h() <= SIDE_H { &[RMode::Stair, side] } else { &[RMode::Stair] };
    for &mode in modes {
        let p = r.point(0.5, if mode == RMode::Stair { STAIR_D0 } else { SIDE_D0 });
        if b.hdist(p) >= RISER_REACH {
            continue;
        }
        let yaw = match mode {
            RMode::Stair | RMode::Wall => away,
            RMode::SideL => away + 0x4000,
            RMode::SideR => away - 0x4000,
        };
        out.push(Arc::new(BljAny { zone: b.zone(), cell: cell(p), kind: Kind::Riser, yaw, start: p, riser: Some((r, mode)), pin: None }));
    }
}

/// The full-tilt (or `mag` 0..32 intendedMag) raw stick along `yaw`
/// (camera-relative; adjust_analog_stick's 6 added back).
fn stick_mag(s: &St, yaw: i32, mag: f32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    // intendedMag = stickMag^2 / 128, stickMag clamped to 64.
    let t = if mag >= 32.0 { 80.0 } else { (128.0 * mag.max(0.0)).sqrt() };
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// The genome stick's x as a yaw offset of up to `max`.
fn offset(g: obrrt_core::GamePad, max: i32) -> i32 {
    g.stick_x as i32 * max / 128
}

fn riser_overlay(r: riser::Riser, mode: RMode, p: [f32; 3]) -> PadOverlay {
    use obrrt_core::GamePad;
    use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
    let away = r.away_yaw();
    PadOverlay::keyed(move |traj, played, g| {
        let s = traj.last().unwrap();
        let held_a = played.last().is_some_and(|q| q.button & A_BUTTON != 0);
        let pose = traj.iter().position(|t| t.action == ACT_IDLE && t.forward_vel == 0.0 && t.hdist(p) < RISER_POSE);
        let Some(kp) = pose else {
            // Setup: the genome far off (the GA finds the way; the
            // controller steers to the point), then walk in slowly and stop.
            let d = s.hdist(p);
            if d > 100.0 && !traj.iter().any(|t| t.hdist(p) < 100.0) {
                return (g, false);
            }
            if d <= RISER_POSE.min(5.0f32.max(2.0 * s.forward_vel)) || (d < RISER_POSE && s.action != ACT_IDLE) {
                return (GamePad::new(0, 0, 0), false);
            }
            if s.action & ACT_FLAG_AIR != 0 {
                return (GamePad::new(0, 0, 0), false);
            }
            let (x, y) = stick_mag(s, yaw_to(s.pos, p), (d / 5.0).clamp(1.5, 32.0));
            return (GamePad::new(0, x, y), false);
        };
        let run = &traj[kp..];
        let begun = run.iter().any(|t| t.action == blj::ACT_LONG_JUMP);
        let back = |off: i32| stick_mag(s, s.yaw as i32 + 0x8000 + off, 32.0);
        if !begun {
            if s.action == blj::ACT_CROUCH_SLIDE {
                let (x, y) = stick_mag(s, s.yaw as i32, 32.0);
                return (GamePad::new(Z_TRIG | if held_a { 0 } else { A_BUTTON }, x, y), false);
            }
            if s.action == ACT_IDLE && s.hdist(p) < RISER_POSE {
                // From idle the stick sets the facing at once: the GA's offset.
                let face = match mode {
                    RMode::Stair | RMode::Wall => away + offset(g, 0x400),
                    RMode::SideL => away + 0x4000 + offset(g, 0x100),
                    RMode::SideR => away - 0x4000 + offset(g, 0x100),
                };
                let (x, y) = stick_mag(s, face, 32.0);
                return (GamePad::new(0, x, y), true);
            }
            if s.action & obrrt_core::layout::ACT_FLAG_MOVING != 0 {
                // Walk on; Z (a new press) once fast enough for the long
                // jump (act_crouch_slide: > 10), when the GA says (A) or at 14.
                let (x, y) = stick_mag(s, s.yaw as i32, 32.0);
                let held_z = played.last().is_some_and(|q| q.button & Z_TRIG != 0);
                let z = !held_z && s.forward_vel >= 10.5 && (g.button & A_BUTTON != 0 || s.forward_vel >= 14.0);
                return (GamePad::new(if z { Z_TRIG } else { 0 }, x, y), z);
            }
            return (GamePad::new(0, 0, 0), false);
        }
        if s.action == blj::ACT_LONG_JUMP {
            // The first jump: drift back toward the riser (+ the GA's
            // offset); in the chain: straight back (+ offset).
            let (x, y) = back(offset(g, 0x3000));
            return (GamePad::new(Z_TRIG, x, y), false);
        }
        if s.action == blj::ACT_LONG_JUMP_LAND {
            // A fresh press when the GA says (its gap), at the latest on
            // the 5th landing frame (the 6th ends it); stick back + the
            // GA's offset on the press; neutral between (no landing accel).
            let since = played.iter().rev().take_while(|q| q.button & A_BUTTON == 0).count();
            let landed_for = traj.iter().rev().take(since).take_while(|t| t.action == blj::ACT_LONG_JUMP_LAND).count();
            // From the foot floor only (a press from the step top flies
            // over the next step, ref/blj-conditions.md 2.2), or the 5th
            // landing frame (the last chance).
            let foot = s.pos[1] <= r.ylo + 2.0;
            let press = !held_a && s.forward_vel < 0.0 && ((foot && g.button & A_BUTTON != 0) || landed_for >= 4);
            if press {
                let (x, y) = back(offset(g, 0x3000));
                return (GamePad::new(Z_TRIG | A_BUTTON, x, y), true);
            }
            // Gap frames: the side BLJ needs the stick (the landing accel
            // resets the drift along the riser), the stair BLJ neutral (no
            // x0.98) (lab, section 6.2).
            if mode != RMode::Stair {
                let (x, y) = back(offset(g, 0x3000));
                return (GamePad::new(Z_TRIG, x, y), false);
            }
            return (GamePad::new(Z_TRIG, 0, 0), false);
        }
        (GamePad::new(0, 0, 0), false)
    })
}

// ---- The Wall clause (OBRRT_BLJ_WALLS=1; default off) ----
//
// The wall-pinned BLJ (ref/blj-conditions.md 6.6): a short riser ending
// against a tall static wall ([`riser::corners`]). Mario lands on the
// riser's seam with his back to the wall; the wall pushes cancel the
// backward travel, the sideways strain puts each jump frame's quarter step
// onto the step top (a landing at once), and the next ground frame drops
// him to the foot. The rule: press from the foot whenever A is up, the
// stick back turned toward the step by an offset that grows with speed;
// the GA adjusts the facing, walk length, first-flight drift and offsets.

/// OBRRT_BLJ_WALLS=1: the `Wall` clause.
fn walls_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_BLJ_WALLS").is_ok_and(|v| v == "1"))
}

/// The pose: this far out of the pinned point (along the wall's normal),
/// and in front of the riser.
const PIN_X: f32 = 100.0;
const PIN_D0: f32 = 3.0;
/// Wall objectives per base state (the nearest corners).
const WALL_PER_STATE: usize = 3;

/// The genome perturbs the rule only on its B frames (8% of random
/// segments, search/src/ga.rs): mostly the lab's rule, sometimes a nudge.
fn nudge(g: obrrt_core::GamePad, max: i32) -> i32 {
    if g.button & obrrt_core::buttons::B_BUTTON != 0 { offset(g, max) } else { 0 }
}

fn zone_corners(zone: (u16, u8)) -> Option<Arc<Vec<riser::Corner>>> {
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};
    static C: OnceLock<RwLock<HashMap<(u16, u8), Arc<Vec<riser::Corner>>>>> = OnceLock::new();
    let c = C.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(r) = c.read().unwrap().get(&zone) {
        return Some(r.clone());
    }
    let rs = riser::zone_risers(zone)?;
    let walls = obrrt_core::geo::static_walls(zone)?;
    let v = Arc::new(riser::corners(&walls, &rs));
    Some(c.write().unwrap().entry(zone).or_insert(v).clone())
}

fn wall_generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    obrrt_core::geo::want_static_floors();
    if !riser_base(b) {
        return;
    }
    let Some(all) = zone_corners(b.zone()) else { return };
    // The nearest pinned spots at Mario's level (within 300 up or down):
    // only some corners work (lab: 3 of 172 with the rule; section 6.6),
    // so several, and the sampler's yield picks among them.
    let mut near: Vec<_> = all
        .iter()
        .map(|c| (c, c.out(PIN_X, PIN_D0)))
        .filter(|(c, p)| c.push_off && (p[1] - b.pos[1]).abs() < 300.0 && b.hdist(*p) < RISER_REACH)
        .collect();
    near.sort_by(|x, y| b.hdist(x.1).total_cmp(&b.hdist(y.1)).then(x.1[0].total_cmp(&y.1[0])).then(x.1[2].total_cmp(&y.1[2])));
    for (c, p) in near.into_iter().take(WALL_PER_STATE) {
        out.push(Arc::new(BljAny { zone: b.zone(), cell: cell(p), kind: Kind::Wall, yaw: c.face_yaw(), start: p, riser: Some((c.riser, RMode::Wall)), pin: Some(*c) }));
    }
}

/// `back + |off|` turned toward the riser's step side (-n); a negative
/// `off` turns away from it.
fn toward_step(c: &riser::Corner, back: i32, off: i32) -> i32 {
    let n = c.riser.n;
    let d = |y: i32| {
        let r = y as f32 * std::f32::consts::TAU / 65536.0;
        r.sin() * n[0] + r.cos() * n[1]
    };
    let (p, m) = (back + off.abs(), back - off.abs());
    let toward = if d(p) <= d(m) { p } else { m };
    if off >= 0 { toward } else if toward == p { m } else { p }
}

/// The chain's press offset by speed (the lab chain: 0x800 to -80, 0x1000
/// to -180, then 0x3000).
fn pin_sched(fv: f32) -> i32 {
    if fv > -80.0 {
        0x800
    } else if fv > -180.0 {
        0x1000
    } else {
        0x3000
    }
}

fn wall_overlay(c: riser::Corner, p: [f32; 3]) -> PadOverlay {
    use obrrt_core::GamePad;
    use obrrt_core::buttons::{A_BUTTON, Z_TRIG};
    let setup = riser_overlay(c.riser, RMode::Wall, p);
    let face0 = c.face_yaw();
    PadOverlay::keyed(move |traj, played, g| {
        let s = traj.last().unwrap();
        let pose = traj.iter().position(|t| t.action == ACT_IDLE && t.forward_vel == 0.0 && t.hdist(p) < RISER_POSE);
        let Some(kp) = pose else { return setup.pad_keyed(traj, played, g) };
        let run = &traj[kp..];
        let begun = run.iter().any(|t| t.action == blj::ACT_LONG_JUMP);
        let held_a = played.last().is_some_and(|q| q.button & A_BUTTON != 0);
        if !begun {
            if s.action == ACT_IDLE && s.hdist(p) < RISER_POSE {
                let (x, y) = stick_mag(s, face0 + nudge(g, 0x100), 32.0);
                return (GamePad::new(0, x, y), true);
            }
            // Walk, Z, Z+A: the Riser rule's.
            return setup.pad_keyed(traj, played, g);
        }
        let landed = run.iter().any(|t| t.action == blj::ACT_LONG_JUMP_LAND);
        let back = s.yaw as i32 + 0x8000;
        if s.action == blj::ACT_LONG_JUMP {
            // First flight: back to the wall, the GA's drift (+-0x300);
            // then the chain's stick.
            let yaw = if landed { toward_step(&c, back, pin_sched(s.forward_vel)) } else { toward_step(&c, back, nudge(g, 0x300)) };
            let (x, y) = stick_mag(s, yaw, 32.0);
            return (GamePad::new(Z_TRIG, x, y), false);
        }
        if s.action == blj::ACT_LONG_JUMP_LAND {
            let since = played.iter().rev().take_while(|q| q.button & A_BUTTON == 0).count();
            let landed_for = traj.iter().rev().take(since).take_while(|t| t.action == blj::ACT_LONG_JUMP_LAND).count();
            let foot = s.pos[1] <= c.riser.ylo + 2.0;
            let off = pin_sched(s.forward_vel);
            if !held_a && s.forward_vel < 0.0 && (foot || landed_for >= 4) {
                // The strain must carry the jump frame across the riser
                // plane (d + 1 in 10 sin(off) a frame).
                let need = pin_off(c.riser.front(s.pos[0], s.pos[2])).max(off);
                let (x, y) = stick_mag(s, toward_step(&c, back, need + nudge(g, 0x400)), 32.0);
                return (GamePad::new(Z_TRIG | A_BUTTON, x, y), true);
            }
            let (x, y) = stick_mag(s, toward_step(&c, back, off), 32.0);
            return (GamePad::new(Z_TRIG, x, y), false);
        }
        (GamePad::new(0, 0, 0), false)
    })
}

/// The press offset whose sideways strain (10 sin(off) a frame over the
/// jump frame's four quarter steps) crosses `d` + 1 to the riser plane.
fn pin_off(d: f32) -> i32 {
    let s = ((d.max(0.0) + 1.0) / 10.0).min(1.0);
    (s.asin() * 32768.0 / std::f32::consts::PI) as i32
}
