//! Mario's movement: everything about how inputs move him lives here
//! (search/ only asks the registered [`obrrt_core::mover::Mover`]).
//!
//! | file | what |
//! |------|------|
//! | `mod.rs` | the move library (this doc): [`MoveKind`]s, [`key`], [`classify`] / `select`, the pads ([`pad_for`]), [`Library`] (the registered controller) and the `install*` switches |
//! | `modes.rs` | the per-control-mode controllers: ground/air (policy, then library, then the generic rule with its A rhythm), swimming (`OBRRT_SWIM_HOLD`), poles, wall kicks, hanging, cannon, flying, Hoot, Bowser |
//! | `cannon.rs` | cannon aim (closed-form shot solve, Arrival-aware) and firing policies (ACT_IN_CANNON only) |
//! | `water_air.rs` | swimming (TAS stroke, surface, water-jump exit) and wing-cap flying (predictive plan, high, land) policies |
//! | `slide.rs` | slide-course policies (hold, course, sharp, jump, first-person braking slide) |
//! | `pole.rs` | pole / tree policies (top, kick-off, drop) |
//! | `tech.rs` | TAS speed techniques as policies: speed kick, edge dive hop, high dive |
//! | `slope.rs` | the TAS's slope / high-speed techniques S1-S6 (ref/tas/slope_techniques.md) as policies: slope dive slide + exit, landing surf, air chain, speed double jump, steep jump |
//! | `policies.rs` | the moves as forcible policies for the GA (`OBRRT_MOVE_GA`, search/src/movegenome.rs), their plausibility pre-filter, and the `OBRRT_CALM` helpers (jump starts, how useful a jump is) |
//! | `extra.rs` | the TAS catalogue's movement techniques that were missing or wrong (audit below), as policies behind `OBRRT_MOVES_EXTRA=1`: A+B jump kick, standing starts, steep climb and dive ladder, edge hop, high and turnaround dive, slope boosts, side-flip wall kick, kick chain, hyperspeed walk and launch |
//! | `arcs.rs` | measured jump/dive/kick arcs and their predictors; the approach planner (doors) |
//! | `doors.rs` | doors seen in states, for door approaches |
//! | `geom.rs` | static-floor probes ahead of Mario (steps, gaps, walls, slopes) |
//!
//! Switches: none (the run binary registers [`Library`] with the library
//! off: the per-mode controllers and the generic rule, as before the
//! library); `OBRRT_MOVES=1` ([`install`]) the library steers the ground
//! and air mode; `OBRRT_MOVE_GA=1` the GA also plays the policies;
//! `OBRRT_MOVES_MODES=1` the GA also offers the control modes' and the
//! speed techniques' policies (cannon.rs, water_air.rs, slide.rs, pole.rs,
//! tech.rs); `OBRRT_MOVES_EXTRA=1` the GA also offers extra.rs's (`legs`
//! defaults both on, obrrt2 neither); `OBRRT_CALM=1` fewer pointless jumps. How a target is reached comes
//! from the objective ([`obrrt_core::Arrival`]: pass, stop, door).
//!
//! ## Audit: the TAS catalogue's movement techniques against the policies
//!
//! ref/tas/techniques/CATALOGUE.md, checked against the decomp. "Offered"
//! is what a `legs` run saw before `OBRRT_MOVES_MODES` and `_EXTRA` became
//! its defaults: every mode policy, tech.rs's included, was silently off.
//!
//! | Id | Technique | Policy before | Was it right? | Now |
//! |----|-----------|---------------|---------------|-----|
//! | G01 | Hyperspeed walk | none: every rule's stick faces the target, so at fv < 0 he backs away from it | missing | `hyper-walk` |
//! | G04 | Hyperspeed to height | none (only `Tech::HyperClimb`, an objective) | missing | `hyper-launch` |
//! | G05 | Edge dive hop | `edge-dive` (mode, off in legs) | wrong: B only above fv 28, though act_freefall dives at any speed; plausible only walking at 20+, though the use is slow starts; A in the slide at once, not at 45+ | `edge-hop` |
//! | G07 | Slope dive / butt slide gain | `slope-dive`, `surf`, `air-chain` | incomplete: no start under 25, nothing gets onto a butt slide, no slide-kick boost | `butt-boost`, `kick-boost` |
//! | G09 | Multi-kick climbs | `wall-kick` | partly: modes.rs's `kick` holds A in WALL_KICK_AIR towards a target above, so AIR_HIT_WALL gets no press: every kick of a chain was a late one from the soft bonk, at the kick's minimum speed | `wall-chain` |
//! | G10 | Single kicks, side-flip entry | `wall-kick` (reversal, into a wall ahead); `side-flip` has no kick | missing: the side-flip kick | `flip-kick` |
//! | G14 | C-up braking slide | `slide-brake` (mode, off in legs) | right | on in legs |
//! | G15 | Landing and walking A+B kick | `uphill-kick` lets A go while falling, so a landing's A is fresh: a (steep) jump at 0.8 fv, the kick only on walking frames; `speed-kick` (mode, off in legs) walking only at 25+ near level | wrong for the landing, and speed lost at every landing | `jump-kick` |
//! | G17 | High dive; turnaround dive | `high-dive` (mode, off in legs) | wrong below fv 35: A then B the next frame is a kick, 0.8 fv being under 28 | `jump-dive`, `turn-dive` |
//! | G18 | Standing starts | none | missing | `punch-start`, `kick-start` |
//! | G19 | Steep climb, dive ladder | `steep-jump` at fv 40+ only, no dive; `uphill-kick` jumps but never dives | incomplete; no ladder | `steep-climb`, `dive-ladder` |
//! | G23 | Crouch-slide long jump re-entry | `long-jump` | right | as was |
//! | G27 | Pole jump | `pole-kick`, `pole-top` (mode, off in legs) | right | on in legs |
//!
//! Not movement policies: G02 / G03 (BLJ, redirects: native `Tech` objectives),
//! G06, G08, G11-G13 (their own rules or objectives, G11 is `swim-tas`), G16, G20-G22,
//! G24-G26, G28-G33 (course-specific or out of the bot's reach).
//!
//! The movement library: named movement primitives, each matched on Mario's
//! state and the local geometry between him and the steering target, that
//! take over the steering controller's generic ground/air mode ("stick at
//! the target, A every few frames") when `OBRRT_MOVES=1` ([`install`]).
//!
//! Each frame [`classify`] picks one [`MoveKind`] by priority from a
//! [`MoveKey`] (start-state class, geometry pattern, target class; see
//! [`key`]) plus a few continuous gates (speed, facing, predicted landing),
//! and the move's pad rule turns it into inputs. Everything is a pure
//! function of the state, the target and the frame number: button edges
//! come from the action (a press on the frame an action starts accepting
//! it, released while the previous action ran), never from memory of the
//! previous pad. So a rollout replays exactly, and the GA's mutations act
//! around the moves as before. No randomness, no logging.
//!
//! Button edges: A is held in the air only while rising (full jump
//! height; ACT_FLAG_CONTROL_JUMP_HEIGHT) and released when falling, so a
//! press on the first landing frame is an edge; on other ground frames
//! presses go on even frames.
//!
//! The moves (match rule; pad):
//! - `run`: default on the ground; stick at the target, no buttons.
//! - `uphill-kick`: the floor under or just ahead of Mario rises along the
//!   target direction steeper than walking holds (apply_slope_accel's
//!   `accel * steepness` beats the walk's 1.1 a frame, or the floor is
//!   steep for jumps), or a floor ahead is that steep. "Hold A and press B
//!   whenever you touch the ground": A held on the ground (its press lands
//!   on the landing frame, as A is let go while falling), B on even ground
//!   frames: a landing jumps again (a steep jump on steep floors, which
//!   carries far uphill), and walking frames kick (a hop up). In the moves
//!   lab this climbs the castle grounds' 49° very slippery hill that plain
//!   steering slides down. `OBRRT_MOVES_UPHILL=airkick` (jump, kick in the
//!   air) and `=jump` (jumps only) are the variants it beat.
//! - `long-jump`: long level stretch towards a far, level target with
//!   speed built up and facing it, landing short of the target: Z (crouch
//!   slide), Z+A (long jump), on LONG_JUMP_LAND Z+A again.
//! - `ledge-jump`: a floor 45-350 above within reach ahead (or a target
//!   just above): A on landings, so single -> double -> triple.
//! - `side-flip` / `backflip`: a tall step (140-350) while running away
//!   from it (turn-around then A) / standing still facing away (Z, then A
//!   while crouching).
//! - `gap-jump` / `gap-long-jump`: no floor ahead (or far below) before a
//!   far side at about the same height: jump (or long jump at speed) near
//!   the edge, A held for distance. `stop`: a pit with no far side and the
//!   target not below: let go of the stick at the edge.
//! - `wall-kick`: ACT_AIR_HIT_WALL (A released on its first frame, pressed
//!   on the next: the kick window is actionTimer <= 2) unless the target is
//!   well below on this side; on the ground or in the air, also the
//!   approach: jump into a wall ahead when the target is behind (a
//!   reversal faster than braking), or when it is above and a second wall
//!   faces this one within 500 behind (kicks chain between them;
//!   ACT_WALL_KICK_AIR itself is steered by search/src/steer.rs's `kick`).
//! - `wall-slide`: grounded against a wall (or a cliff no jump climbs)
//!   facing the target: stick along the wall towards the target's side.
//! - `dive-rollout`: fast (>= 29) on a level stretch at mid range or
//!   downhill, or in the air over a gap at > 28: B (dive; +15 speed, capped
//!   at 48), then A on DIVE_SLIDE / STOMACH_SLIDE (rollout).
//!   `belly-slide`: the same slides ridden (stick only) while the floor runs
//!   downhill towards a target below.
//! - `slide-kick`: "hold Z, press B while moving": Z from walking (crouch
//!   slide), Z+B in it (slide kick: 12 up, at least 32 forward; its first
//!   landing while falling bounces at half the speed, act_slide_kick), A
//!   in SLIDE_KICK_SLIDE hops out (forward rollout). Downhill towards a
//!   far, lower target at 10-29, or over a gap at most 150 wide at speed.
//!   `slide-ride` rides a slide-kick slide or butt slide that heads at a
//!   lower target downhill.
//! - `ledge-climb`: ACT_LEDGE_GRAB with the target not far below: A.
//! - `slide-escape`: butt sliding or beginning to, with the target uphill,
//!   near, or off the slide's heading: A (slide jump) towards it.
//! - `ground-pound`: in the air over the target spot (within 80) with the
//!   target 250+ below, when the pound to walking (arcs::ground_pound: spin
//!   15, drop, get-up) beats falling on and walking back the overshoot,
//!   and it hurts no more than the plain fall: Z.
//! - `fall-safe`: at the edge of a drop that would hurt (arcs::fall_damage
//!   of the floor under the edge: > 3000 is 4 wedges and an 82-frame
//!   knockback, always avoided, as is the void; > 1150 is 2 wedges,
//!   avoided at 3 wedges or fewer): let go of the stick. (A ground pound
//!   doesn't help: its spin rise counts towards the fall.)
//! - door approach: a target at a door the library has seen (doors.rs)
//!   is reached by arcs::plan_approach_with along the door's normal (the
//!   door opens only to walking, within ~110), within 2500 of it.
//! - `air`: anything else in the air: stick at the target, A while rising.

pub mod arcs;
pub mod cannon;
pub mod doors;
pub mod extra;
pub mod geom;
pub mod modes;
pub mod pole;
pub mod policies;
pub mod slide;
pub mod slope;
pub mod tech;
pub mod water_air;

use std::f32::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::mover::Mover;
use obrrt_core::{Arrival, GamePad, St};

use geom::{Ahead, STEP_REACH, slope_accel, steep_limit};

const ACT_FLAG_AIR: u32 = 0x0000_0800;
const ACT_FLAG_BUTT_OR_STOMACH_SLIDE: u32 = 0x0004_0000;
const ACT_IDLE: u32 = 0x0C40_0201;
const ACT_WALKING: u32 = 0x0400_0440;
const ACT_DECELERATING: u32 = 0x0400_044A;
const ACT_BRAKING: u32 = 0x0400_0445;
const ACT_TURNING_AROUND: u32 = 0x0000_0443;
const ACT_FINISH_TURNING_AROUND: u32 = 0x0000_0444;
const ACT_START_CROUCHING: u32 = 0x0C00_8221;
const ACT_CROUCHING: u32 = 0x0C00_8220;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_BEGIN_SLIDING: u32 = 0x0000_0050;
const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080_045A;
const ACT_JUMP_LAND: u32 = 0x0400_0470;
const ACT_FREEFALL_LAND: u32 = 0x0400_0471;
const ACT_DOUBLE_JUMP_LAND: u32 = 0x0400_0472;
const ACT_SIDE_FLIP_LAND: u32 = 0x0400_0473;
const ACT_TRIPLE_JUMP_LAND: u32 = 0x0400_0478;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_BACKFLIP_LAND: u32 = 0x0400_047A;
const ACT_JUMP: u32 = 0x0300_0880;
const ACT_DOUBLE_JUMP: u32 = 0x0300_0881;
const ACT_FREEFALL: u32 = 0x0100_088C;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;
const ACT_LEDGE_GRAB: u32 = 0x0800_034B;

/// The target's height (for fall-damage estimates).
fn target_y(ctx: &Ctx, st: &St) -> f32 {
    st.pos[1] + ctx.dy
}
const ACT_SOFT_BONK: u32 = 0x0102_08B6;
const ACT_STOMACH_SLIDE: u32 = 0x008C_0453;
const ACT_TRIPLE_JUMP: u32 = 0x0100_0882;
const ACT_BACKFLIP: u32 = 0x0100_0883;
const ACT_SIDE_FLIP: u32 = 0x0100_0887;

/// The movement primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MoveKind {
    Run,
    UphillKick,
    LongJumpChain,
    LedgeJump,
    SideFlip,
    Backflip,
    GapJump,
    GapLongJump,
    Stop,
    WallKick,
    WallSlide,
    DiveRollout,
    SlideKick,
    LedgeClimb,
    SlideEscape,
    SlideRide,
    BellySlide,
    GroundPound,
    FallSafe,
    Air,
}

impl MoveKind {
    pub const ALL: [MoveKind; 20] = [
        MoveKind::Run,
        MoveKind::UphillKick,
        MoveKind::LongJumpChain,
        MoveKind::LedgeJump,
        MoveKind::SideFlip,
        MoveKind::Backflip,
        MoveKind::GapJump,
        MoveKind::GapLongJump,
        MoveKind::Stop,
        MoveKind::WallKick,
        MoveKind::WallSlide,
        MoveKind::DiveRollout,
        MoveKind::SlideKick,
        MoveKind::LedgeClimb,
        MoveKind::SlideEscape,
        MoveKind::SlideRide,
        MoveKind::BellySlide,
        MoveKind::GroundPound,
        MoveKind::FallSafe,
        MoveKind::Air,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MoveKind::Run => "run",
            MoveKind::UphillKick => "uphill-kick",
            MoveKind::LongJumpChain => "long-jump",
            MoveKind::LedgeJump => "ledge-jump",
            MoveKind::SideFlip => "side-flip",
            MoveKind::Backflip => "backflip",
            MoveKind::GapJump => "gap-jump",
            MoveKind::GapLongJump => "gap-long-jump",
            MoveKind::Stop => "stop",
            MoveKind::WallKick => "wall-kick",
            MoveKind::WallSlide => "wall-slide",
            MoveKind::DiveRollout => "dive-rollout",
            MoveKind::SlideKick => "slide-kick",
            MoveKind::LedgeClimb => "ledge-climb",
            MoveKind::SlideEscape => "slide-escape",
            MoveKind::SlideRide => "slide-ride",
            MoveKind::BellySlide => "belly-slide",
            MoveKind::GroundPound => "ground-pound",
            MoveKind::FallSafe => "fall-safe",
            MoveKind::Air => "air",
        }
    }
}

// ---------------------------------------------------------------------------
// Keys: the discrete part of the match, for a later outcome table (which
// move gets from start class X to Y under geometry pattern G).

/// Mario's state, coarsely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StartClass {
    /// On the ground, forward speed under 16.
    GroundSlow,
    GroundFast,
    /// The first frames of a landing action (jumps chain from here).
    Landing,
    Crouching,
    AirRising,
    AirFalling,
    LongJumping,
    HitWall,
    /// Butt/stomach/dive/slide-kick sliding, or beginning to slide.
    Sliding,
    LedgeGrab,
    Other,
}

/// What lies ahead along the target direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeomClass {
    /// No floor snapshot for the zone.
    Unknown,
    Level,
    /// Rising faster than walking holds (or the current floor is steep).
    SteepUp,
    GentleUp,
    Down,
    /// A floor that takes a jump: height above Mario's floor in 50s.
    StepUp(u8),
    /// A step no jump climbs (or a wall).
    Cliff,
    /// A gap with a far side / without one.
    Gap,
    Pit,
}

/// The target's place relative to Mario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetClass {
    Near,
    Mid,
    Far,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MoveKey {
    pub start: StartClass,
    pub geom: GeomClass,
    pub target: TargetClass,
    /// Target height relative to Mario: -1 below (< -150), 0 level, 1 above (> 100).
    pub dy: i8,
}

/// Everything a match looks at, computed once per frame.
pub struct Ctx {
    /// Horizontal unit vector to the target (x, z).
    pub dir: [f32; 2],
    pub hd: f32,
    pub dy: f32,
    pub fv: f32,
    /// |facing - direction to target|, s16 units.
    pub yaw_err: u16,
    pub air: bool,
    /// Mario's floor height on the ground, his height in the air.
    pub base: f32,
    /// Rise per unit of the floor under Mario along `dir` (ground only).
    pub grade: f32,
    pub ahead: Option<Ahead>,
    /// How the target must be reached (the objective's): with
    /// [`Arrival::Pass`] a long jump may land just past it (its arc comes
    /// down through it) instead of short of it, and a ground pound needn't
    /// beat walking back an overshoot.
    pub arrival: Arrival,
}

impl Ctx {
    /// [`Ctx::with`] for a target to stop at (the library's rules before
    /// arrivals: every jump kept short of the target).
    pub fn new(st: &St, target: [f32; 3]) -> Ctx {
        Ctx::with(st, target, Arrival::Stop)
    }

    pub fn with(st: &St, target: [f32; 3], arrival: Arrival) -> Ctx {
        let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
        let hd = (dx * dx + dz * dz).sqrt();
        let dir = if hd > 1e-3 { [dx / hd, dz / hd] } else { [0.0, 1.0] };
        let air = st.action & ACT_FLAG_AIR != 0;
        let n = st.floor_normal;
        let grade = if !air && n[1] > 0.01 { -(n[0] * dir[0] + n[2] * dir[1]) / n[1] } else { 0.0 };
        let base = if air || st.floor_y < -10000.0 { st.pos[1] } else { st.floor_y };
        let yaw_err = (angle(dx, dz) as i32 - st.yaw as i32) as i16;
        Ctx {
            dir,
            hd,
            dy: target[1] - st.pos[1],
            fv: st.forward_vel,
            yaw_err: yaw_err.unsigned_abs(),
            air,
            base,
            grade,
            ahead: geom::ahead(st, dir, hd.min(1000.0), base),
            arrival,
        }
    }
}

fn is_landing(a: u32) -> bool {
    matches!(
        a,
        ACT_JUMP_LAND | ACT_FREEFALL_LAND | ACT_DOUBLE_JUMP_LAND | ACT_SIDE_FLIP_LAND | ACT_TRIPLE_JUMP_LAND | ACT_BACKFLIP_LAND
    )
}

fn is_sliding(a: u32) -> bool {
    a & ACT_FLAG_BUTT_OR_STOMACH_SLIDE != 0 || matches!(a, ACT_BEGIN_SLIDING | ACT_DIVE_SLIDE | ACT_SLIDE_KICK_SLIDE)
}

/// Whether walking up the current floor along the target direction fails:
/// the slope accel (`accel * steepness`, apply_slope_accel) at least the
/// walk's 1.1 - fv/43 at a walking pace of 10, or a jump from here is a
/// steep jump.
fn floor_too_steep(st: &St, ctx: &Ctx) -> bool {
    if ctx.air || ctx.grade <= 0.05 {
        return false;
    }
    let n = st.floor_normal;
    let steepness = (n[0] * n[0] + n[2] * n[2]).sqrt();
    let kind = st.floor_type;
    slope_accel(kind) * steepness >= 1.1 - 10.0 / 43.0 || n[1] < steep_limit(kind)
}

/// Rise per unit ahead that walking doesn't hold (about 27° on default
/// floors: 1.7 sin θ = 0.87 - the walk's gain at 10 u/f).
const STEEP_GRADE: f32 = 0.5;

pub fn key(st: &St, target: [f32; 3]) -> MoveKey {
    key_of(st, &Ctx::new(st, target))
}

fn key_of(st: &St, ctx: &Ctx) -> MoveKey {
    let a = st.action;
    let start = match a {
        ACT_LEDGE_GRAB => StartClass::LedgeGrab,
        ACT_AIR_HIT_WALL => StartClass::HitWall,
        ACT_LONG_JUMP => StartClass::LongJumping,
        ACT_START_CROUCHING | ACT_CROUCHING | ACT_CROUCH_SLIDE => StartClass::Crouching,
        _ if is_sliding(a) => StartClass::Sliding,
        _ if is_landing(a) || a == ACT_LONG_JUMP_LAND => StartClass::Landing,
        _ if a & ACT_FLAG_AIR != 0 => {
            if st.vel[1] > 0.0 {
                StartClass::AirRising
            } else {
                StartClass::AirFalling
            }
        }
        _ if a & obrrt_core::layout::ACT_GROUP_MASK <= 0x40 => {
            if ctx.fv >= 16.0 {
                StartClass::GroundFast
            } else {
                StartClass::GroundSlow
            }
        }
        _ => StartClass::Other,
    };
    let geom = match &ctx.ahead {
        None => {
            if floor_too_steep(st, ctx) {
                GeomClass::SteepUp
            } else {
                GeomClass::Unknown
            }
        }
        Some(ah) => {
            let reach = ctx.hd.min(600.0);
            if let Some((edge, far)) = ah.gap(ctx.base, 150.0, 120.0).filter(|g| g.0 <= reach) {
                let _ = edge;
                if far.is_some() { GeomClass::Gap } else { GeomClass::Pit }
            } else if floor_too_steep(st, ctx) || ah.steepest(ctx.dir, 300f32.min(ctx.hd.max(60.0))) > STEEP_GRADE
            {
                GeomClass::SteepUp
            } else if ah.cliff(ctx.base).is_some_and(|d| d <= 150.0 && d <= ctx.hd + 50.0)
                || ah.wall.is_some_and(|w| w <= 150.0 && w <= ctx.hd)
                    && ah.step_up(ctx.base, 45.0).is_none_or(|s| s.0 > ah.wall.unwrap_or(0.0) + 100.0)
            {
                GeomClass::Cliff
            } else if let Some((_, h)) = ah.step_up(ctx.base, 45.0).filter(|s| s.0 <= 300.0 && s.0 <= ctx.hd + 50.0) {
                GeomClass::StepUp((h / 50.0).min(7.0) as u8)
            } else {
                match ah.grade(ctx.base, 400.0) {
                    Some(g) if g < -0.25 => GeomClass::Down,
                    Some(g) if g > 0.1 => GeomClass::GentleUp,
                    _ => GeomClass::Level,
                }
            }
        }
    };
    let target = if ctx.hd < 300.0 {
        TargetClass::Near
    } else if ctx.hd < 900.0 {
        TargetClass::Mid
    } else {
        TargetClass::Far
    };
    let dy = if ctx.dy < -150.0 {
        -1
    } else if ctx.dy > 100.0 {
        1
    } else {
        0
    };
    MoveKey { start, geom, target, dy }
}

/// Jumps whose landing [`reach`] predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Jump {
    Single,
    Long,
}

/// Horizontal distance a jump started now at forward speed `fv` covers
/// before landing on a floor `dy` above its takeoff (below: negative), from
/// the measured arcs ([`arcs::predict_with`], A held, stick forward;
/// `chained`: a long jump pressed on LONG_JUMP_LAND), plus the slide of
/// its landing lag. A floor the arc never comes down to counts as the flat
/// landing.
pub fn reach_on(kind: Jump, fv: f32, chained: bool, dy: f32) -> f32 {
    let k = match kind {
        Jump::Single => arcs::ArcKind::Single,
        Jump::Long => arcs::ArcKind::LongJump,
    };
    let arc = arcs::predict_with(k, fv.max(0.0), 255, arcs::AirStick::Fwd, chained);
    arcs::land_on(&arc, dy).map_or(arc.land_dist, |(_, d)| d) + arc.lag_dist
}

/// [`reach_on`] from a run onto level ground.
pub fn reach(kind: Jump, fv: f32) -> f32 {
    reach_on(kind, fv, false, 0.0)
}

/// Whether a jump started now lands (and its lag slides) at least `margin`
/// short of the target (so it doesn't overshoot it, e.g. into the wall
/// beside a door), landing at the target's height relative to his floor
/// (clamped to what a jump comes down on).
pub fn lands_short(kind: Jump, ctx: &Ctx, chained: bool, margin: f32) -> bool {
    reach_on(kind, ctx.fv, chained, ctx.dy.clamp(-300.0, 100.0)) + margin <= ctx.hd
}

/// The move for this frame (see the module docs for the rules), for a
/// target to stop at.
pub fn classify(st: &St, target: [f32; 3]) -> MoveKind {
    classify_with(st, target, Arrival::Stop)
}

/// [`classify`] for a target reached as `arrival` says.
pub fn classify_with(st: &St, target: [f32; 3], arrival: Arrival) -> MoveKind {
    let ctx = Ctx::with(st, target, arrival);
    select(st, &key_of(st, &ctx), &ctx)
}

fn facing(ctx: &Ctx, within: u16) -> bool {
    ctx.yaw_err < within
}

/// Whether the wall Mario touches faces against the target direction.
fn wall_against(st: &St, ctx: &Ctx) -> Option<[f32; 2]> {
    let n = st.wall_normal?;
    let l = (n[0] * n[0] + n[2] * n[2]).sqrt();
    if l < 0.3 {
        return None;
    }
    let n2 = [n[0] / l, n[2] / l];
    (n2[0] * ctx.dir[0] + n2[1] * ctx.dir[1] < -0.5).then_some(n2)
}

/// Facing as a unit vector (x, z).
fn facing_dir(st: &St) -> [f32; 2] {
    let a = st.yaw as f32 * TAU / 65536.0;
    [a.sin(), a.cos()]
}

/// A wall met nearly head-on (within about 45°: what a wall kick takes)
/// along `dir` within `within`: the wall Mario touches, a static wall, or
/// a cliff no jump climbs.
fn wall_along(st: &St, ctx: &Ctx, dir: [f32; 2], within: f32) -> bool {
    if let Some(n) = st.wall_normal {
        let l = (n[0] * n[0] + n[2] * n[2]).sqrt().max(1e-3);
        if (n[0] * dir[0] + n[2] * dir[1]) / l < -0.7 && !ctx.air {
            return true;
        }
    }
    let base = if ctx.air { st.floor_y.max(st.pos[1] - 100.0) } else { ctx.base };
    geom::tall_wall_ahead(st, dir, within, if ctx.air { st.pos[1] - 30.0 } else { base }, WALL_KICK_COS).is_some()
        || geom::ahead(st, dir, within, ctx.base).is_some_and(|ah| ah.cliff(ctx.base).is_some_and(|d| d <= within))
}

/// Frames to fall `drop` (> 0) from vertical speed `vy` at gravity 4.
fn fall_frames(vy: f32, drop: f32) -> f32 {
    (vy + (vy * vy + 8.0 * drop.max(0.0)).sqrt()) / 4.0
}

/// Whether a drop of `depth` from Mario's floor would hurt enough to
/// avoid it (arcs::fall_damage, a plain landing; a ground pound doesn't
/// help, its spin rise counts too): a hard knockback (> 3000: 4 wedges and
/// 82 frames) always, plain damage (> 1150: 2 wedges) only when health is
/// short (3 wedges or fewer: health as a resource).
fn drop_hurts(st: &St, depth: f32) -> bool {
    let (loss, lag) = arcs::fall_damage(depth, 0.0, ACT_FREEFALL_LAND);
    lag > 20 || (loss > 0 && st.wedges() <= 3)
}

/// cos of the widest angle off a wall's normal a kick takes
/// (arcs::WALL_KICK_MAX_ANGLE, measured: 44° kicks, 45° slides along).
const WALL_KICK_COS: f32 = 0.7193398; // cos 44°

/// Air actions that take B (kick/dive) and Z (ground pound).
fn air_actionable(a: u32) -> bool {
    matches!(a, ACT_JUMP | ACT_DOUBLE_JUMP | ACT_FREEFALL | ACT_TRIPLE_JUMP | ACT_BACKFLIP | ACT_SIDE_FLIP)
}

fn select(st: &St, key: &MoveKey, ctx: &Ctx) -> MoveKind {
    use GeomClass as G;
    use StartClass as S;
    let a = st.action;
    match key.start {
        S::LedgeGrab => return if ctx.dy > -150.0 { MoveKind::LedgeClimb } else { MoveKind::Air },
        S::HitWall => {
            // Kick off unless the target is well below on this side: it
            // turns him around with 52 up (a reversal, or the next wall
            // of a chain).
            return if ctx.dy > -150.0 || wall_against(st, ctx).is_some() || ctx.yaw_err > 0x5000 {
                MoveKind::WallKick
            } else {
                MoveKind::Air
            };
        }
        S::LongJumping => return MoveKind::LongJumpChain,
        // A soft bonk off a wall still takes a kick for 5 frames.
        _ if a == ACT_SOFT_BONK && st.wall_normal.is_some() && (ctx.dy > -150.0 || ctx.yaw_err > 0x5000) => {
            return MoveKind::WallKick;
        }
        S::Sliding => {
            // Slide heading vs the target direction.
            let vh = (st.vel[0] * st.vel[0] + st.vel[2] * st.vel[2]).sqrt();
            let along = if vh > 1.0 { (st.vel[0] * ctx.dir[0] + st.vel[2] * ctx.dir[1]) / vh } else { 0.0 };
            let downhill = ctx.grade < -0.15 || ctx.dy < -100.0;
            let ride = downhill && ctx.hd > 250.0 && along > 0.5 && ctx.dy < 50.0;
            return match a {
                // A rollout carries 400+ at this speed: near a level target,
                // slide in on the belly instead of flying past.
                ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE => {
                    if ride || (ctx.hd < 450.0 && ctx.dy.abs() < 150.0) { MoveKind::BellySlide } else { MoveKind::DiveRollout }
                }
                ACT_SLIDE_KICK_SLIDE => {
                    if ride && ctx.fv >= 20.0 { MoveKind::SlideRide } else { MoveKind::SlideKick }
                }
                ACT_BEGIN_SLIDING => if ctx.dy > 50.0 || ctx.grade > 0.1 { MoveKind::SlideEscape } else { MoveKind::SlideRide },
                _ => if ride { MoveKind::SlideRide } else { MoveKind::SlideEscape },
            };
        }
        S::AirRising | S::AirFalling => {
            let act = air_actionable(a);
            // Over the target spot with speed that would carry him past it:
            // pound straight down (it kills the horizontal speed), when the
            // pound to walking (arcs::ground_pound: spin 15, the drop, the
            // get-up) beats falling on, landing, and walking back the
            // overshoot (a turn-around and about 24 a frame), and the pound
            // doesn't hurt more than the fall would.
            if act && ctx.dy < -250.0 && ctx.hd < 80.0 {
                let h = -ctx.dy;
                let t = fall_frames(st.vel[1], h);
                // Passing the target costs nothing with Arrival::Pass.
                let overshoot = if ctx.arrival == Arrival::Pass { 0.0 } else { (ctx.fv.max(0.0) * t - ctx.hd).max(0.0) };
                let plain = t + 4.0 + overshoot / 24.0 + if overshoot > 100.0 { 8.0 } else { 0.0 };
                let gp = arcs::ground_pound(h).total as f32;
                let hurts = |action| arcs::fall_damage(st.pos[1], target_y(ctx, st), action).0;
                if gp < plain && hurts(arcs::ACT_GROUND_POUND) <= hurts(ACT_FREEFALL_LAND) {
                    return MoveKind::GroundPound;
                }
            }
            if act && ctx.fv <= 28.0 && key.geom == G::SteepUp {
                return MoveKind::UphillKick;
            }
            // Turning back with a wall ahead: keep flying into it to kick.
            if ctx.yaw_err > 0x6000 && ctx.fv >= 16.0 && wall_along(st, ctx, facing_dir(st), (ctx.fv * 4.0).max(100.0)) {
                return MoveKind::WallKick;
            }
            // Dive over a gap whose far side the jump falls short of.
            let over_gap = matches!(key.geom, G::Gap) || st.floor_y < ctx.base - 400.0;
            if act && ctx.fv > 28.0 && facing(ctx, 0x1800) && ctx.dy < 100.0 && over_gap && key.start == S::AirFalling {
                return MoveKind::DiveRollout;
            }
            return MoveKind::Air;
        }
        S::Other => return MoveKind::Run,
        S::GroundSlow | S::GroundFast | S::Landing | S::Crouching => {}
    }
    // On the ground. A crouch slide is always one of ours (Z): finish what
    // it started, a slide kick (B) or a long jump (A; a plain jump under
    // 10 speed).
    if a == ACT_CROUCH_SLIDE {
        // What walking here would pick decides which.
        let mut walking = st.clone();
        walking.action = ACT_WALKING;
        let k = MoveKey { start: StartClass::GroundFast, ..*key };
        return if select(&walking, &k, ctx) == MoveKind::SlideKick { MoveKind::SlideKick } else { MoveKind::LongJumpChain };
    }
    // A target to pass may be landed just past (an arc that comes down by
    // it passes through it low); one to stop at must be landed short of.
    let pass = ctx.arrival == Arrival::Pass;
    let margin = |m: f32| if pass { -100.0 } else { m };
    if a == ACT_LONG_JUMP_LAND && key.geom == G::Level && key.dy == 0 && lands_short(Jump::Long, ctx, true, margin(100.0)) {
        return MoveKind::LongJumpChain;
    }
    // A wall straight ahead with the target behind: jump into it and kick
    // back (faster than braking and turning at speed).
    // (Within about 10 frames; the pad runs on into it and jumps at 4.)
    if ctx.fv >= 20.0
        && ctx.yaw_err > 0x6000
        && ctx.hd > 300.0
        && arcs::predict_wall_kick(ctx.fv, 0.0, 0).is_some()
        && wall_along(st, ctx, facing_dir(st), ctx.fv * 10.0)
    {
        return MoveKind::WallKick;
    }
    match key.geom {
        G::SteepUp if ctx.dy > -50.0 => return MoveKind::UphillKick,
        G::Pit => {
            // A drop that would hurt (by the floor below the edge): stop
            // at the edge rather than take it.
            if let Some((edge, depth)) = ctx.ahead.as_ref().and_then(|ah| ah.drop_depth(ctx.base)) {
                if edge <= (ctx.fv * 3.0).max(60.0) && drop_hurts(st, depth) {
                    return MoveKind::FallSafe;
                }
            }
            if ctx.dy < -150.0 {
                return MoveKind::Run;
            }
            let edge = ctx.ahead.as_ref().and_then(|ah| ah.gap(ctx.base, 150.0, 120.0)).map_or(0.0, |g| g.0);
            return if edge <= 100.0 { MoveKind::Stop } else { MoveKind::Run };
        }
        G::Gap => {
            let Some((edge, Some(far))) = ctx.ahead.as_ref().and_then(|ah| ah.gap(ctx.base, 150.0, 120.0)) else {
                return MoveKind::Run;
            };
            // Jump near the edge: within about 3 frames of it.
            if edge > (ctx.fv * 3.0).max(60.0) {
                return MoveKind::Run;
            }
            // A narrow gap to a far side no higher, at speed: slide kick
            // (low, fast, and its first landing bounces). TODO(arcs):
            // the measured slide-kick arc.
            if far - edge <= 150.0 && ctx.fv >= 16.0 && ctx.dy < 50.0 && facing(ctx, 0x1000) {
                return MoveKind::SlideKick;
            }
            // The long jump only where it reaches further than a jump
            // (with measured arcs, TODO(arcs)).
            let long = reach(Jump::Long, ctx.fv);
            return if ctx.fv >= 20.0 && facing(ctx, 0x1000) && long > reach(Jump::Single, ctx.fv) && long >= far + 50.0 {
                MoveKind::GapLongJump
            } else {
                MoveKind::GapJump
            };
        }
        G::Cliff => {
            // Two facing walls (one ahead, one behind) and the target
            // above: jump into this one and chain kicks between them.
            // (arcs::wall_kicks_to_climb: the chain gains height at this gap.)
            let back = [-ctx.dir[0], -ctx.dir[1]];
            if ctx.dy > 150.0 && ctx.fv > 16.0 && facing(ctx, 0x2000) {
                let ahead = geom::tall_wall_ahead(st, ctx.dir, 200.0, ctx.base, WALL_KICK_COS);
                let behind = geom::tall_wall_ahead(st, back, 600.0, ctx.base, WALL_KICK_COS);
                if let (Some(a), Some(b)) = (ahead, behind) {
                    if arcs::wall_kicks_to_climb(a + b, ctx.dy).is_some() {
                        return MoveKind::WallKick;
                    }
                }
            }
            if wall_against(st, ctx).is_some() || ctx.fv < 4.0 {
                return MoveKind::WallSlide;
            }
            return MoveKind::Run;
        }
        G::StepUp(h50) => {
            let h = h50 as f32 * 50.0;
            let d = ctx.ahead.as_ref().and_then(|ah| ah.step_up(ctx.base, 45.0)).map_or(0.0, |s| s.0);
            if h >= 140.0 && ctx.fv >= 16.0 && ctx.yaw_err > 0x6000 {
                return MoveKind::SideFlip;
            }
            if h >= 200.0 && ctx.fv < 4.0 && ctx.yaw_err > 0x6000 && d < 150.0 {
                return MoveKind::Backflip;
            }
            // Take off within about eight frames of the step (a jump at speed
            // clears 120 within 3 frames and peaks after 12).
            // Only for a target not below Mario (a rise past a lower target
            // is a detour).
            if ctx.dy > -50.0 && (d <= (ctx.fv * 8.0).max(100.0) || key.start == S::Landing && h >= 140.0) {
                return MoveKind::LedgeJump;
            }
            return MoveKind::Run;
        }
        // Downhill towards a lower, far target: a dive into a belly slide
        // at running speed >= 29, else a slide kick (Z, then B).
        G::Down if ctx.dy < -100.0 && ctx.hd > 500.0 && facing(ctx, 0x1000) => {
            if ctx.fv >= 29.0 && a == ACT_WALKING {
                return MoveKind::DiveRollout;
            }
            if ctx.fv >= 10.0 {
                return MoveKind::SlideKick;
            }
        }
        G::Unknown => {
            if wall_against(st, ctx).is_some() && ctx.fv < 4.0 {
                return MoveKind::WallSlide;
            }
        }
        _ => {}
    }
    // Level-ish ground.
    if key.geom == G::Level && ctx.dy.abs() < 200.0 && facing(ctx, 0x1000) {
        let strip = ctx.ahead.as_ref().is_some_and(|ah| ah.level(ctx.base, ctx.hd.min(800.0), 60.0));
        if strip && ctx.fv >= 20.0 && ctx.hd > 900.0 && lands_short(Jump::Long, ctx, false, margin(150.0)) && st.ceil_y - st.pos[1] > 300.0 {
            return MoveKind::LongJumpChain;
        }
        if strip && ctx.fv >= 29.0 && ctx.hd > 700.0 && ctx.hd <= 900.0 && ctx.dy.abs() < 100.0 {
            return MoveKind::DiveRollout;
        }
    }
    if ctx.hd < 250.0 && ctx.dy > 60.0 && ctx.dy < STEP_REACH {
        return MoveKind::LedgeJump;
    }
    MoveKind::Run
}

// ---------------------------------------------------------------------------
// Pads.

#[derive(Clone, Copy, PartialEq, Eq)]
enum UphillStyle {
    /// A held on the ground (pressed on landing), B tapped on ground
    /// frames: the user's "hold A, press B whenever you touch the ground".
    Kick,
    /// A on landings and even ground frames, B in the air (jump kicks).
    AirKick,
    /// Jumps only.
    Jump,
}

fn uphill_style() -> UphillStyle {
    use obrrt_core::config::Uphill;
    match obrrt_core::config::current().objectives.moves_uphill {
        Uphill::Jump => UphillStyle::Jump,
        Uphill::AirKick => UphillStyle::AirKick,
        Uphill::Kick => UphillStyle::Kick,
    }
}

/// The pad for `kind` this frame, for a target to stop at.
pub fn pad_for(kind: MoveKind, st: &St, target: [f32; 3], frame: usize) -> GamePad {
    pad_for_with(kind, st, target, Arrival::Stop, frame)
}

/// [`pad_for`] for a target reached as `arrival` says. (Jumps are cut
/// short near the target either way: a jump that flies over a target
/// doesn't pass through it.)
pub fn pad_for_with(kind: MoveKind, st: &St, target: [f32; 3], arrival: Arrival, frame: usize) -> GamePad {
    let a = st.action;
    let (sx, sy) = camera_stick(st, target);
    let air = a & ACT_FLAG_AIR != 0;
    let even = frame % 2 == 0;
    // A held while rising: full height and distance; released before the
    // fall, so the landing frame's press is an edge.
    // A only matters above 20 up: letting go then quarters the rise
    // (should_strengthen_gravity_for_jump_ascent); below, it is free to let
    // go, which readies the next press (a wall kick's single-frame window).
    let rising_a = if air && st.vel[1] > 20.0 { A_BUTTON } else { 0 };
    // For jumps that needn't go far: A held while rising only below the
    // target's height + 100 or while the target is further than the rest
    // of the arc carries (about fv * vy / 2), so a jump near the target is
    // cut short (letting go above 20 up quarters the rise,
    // ACT_FLAG_CONTROL_JUMP_HEIGHT) instead of overshooting it.
    let hd = st.hdist(target);
    let _ = arrival;
    let short_a = if rising_a != 0 && (st.pos[1] < target[1] + 100.0 || hd > st.forward_vel.max(0.0) * st.vel[1] / 2.0) {
        A_BUTTON
    } else {
        0
    };
    // On the ground: on the first frames of a landing, else on even frames.
    let ground_a = if is_landing(a) || even { A_BUTTON } else { 0 };
    let pad = |b: u16| GamePad::new(b, sx, sy);
    // The stick along Mario's facing (into the wall ahead).
    let fwd = || {
        let (fx, fy) = stick_dir(st, st.yaw as f32 * TAU / 65536.0);
        move |b: u16| GamePad::new(b, fx, fy)
    };
    match kind {
        MoveKind::Run => pad(0),
        MoveKind::Air => pad(short_a),
        MoveKind::Stop => GamePad::new(0, 0, 0),
        MoveKind::UphillKick => match uphill_style() {
            UphillStyle::Kick => {
                if air {
                    pad(rising_a)
                } else if is_landing(a) && (st.floor_normal[1] >= steep_limit(st.floor_type) || st.forward_vel.abs() > 3.0) {
                    // The jump first: B on a landing frame kicks instead. Only
                    // a stall on a steep floor (a steep jump straight up and
                    // down at 0 speed) takes the kick's hop there.
                    pad(A_BUTTON)
                } else {
                    pad(A_BUTTON | if even { B_BUTTON } else { 0 })
                }
            }
            style => {
                if air {
                    let kick = style == UphillStyle::AirKick
                        && matches!(a, ACT_JUMP | ACT_DOUBLE_JUMP | ACT_FREEFALL)
                        && st.forward_vel <= 28.0;
                    pad(rising_a | if kick { B_BUTTON } else { 0 })
                } else {
                    pad(ground_a)
                }
            }
        },
        MoveKind::LedgeJump => pad(if air { short_a } else { ground_a }),
        MoveKind::GapJump => pad(if air { rising_a } else { ground_a }),
        MoveKind::LongJumpChain | MoveKind::GapLongJump => {
            if air {
                pad(rising_a)
            } else {
                match a {
                    // A only counts with Z down here (act_long_jump_land).
                    ACT_LONG_JUMP_LAND | ACT_CROUCH_SLIDE => pad(Z_TRIG | A_BUTTON),
                    _ if is_landing(a) => pad(0),
                    _ => pad(if even { Z_TRIG } else { 0 }),
                }
            }
        }
        MoveKind::SideFlip => {
            if air {
                pad(rising_a)
            } else if matches!(a, ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND) {
                pad(if even { A_BUTTON } else { 0 })
            } else {
                pad(0)
            }
        }
        MoveKind::Backflip => {
            if air {
                pad(rising_a)
            } else if matches!(a, ACT_START_CROUCHING | ACT_CROUCHING) {
                GamePad::new(Z_TRIG | if even { A_BUTTON } else { 0 }, 0, 0)
            } else if matches!(a, ACT_IDLE) {
                GamePad::new(if even { Z_TRIG } else { 0 }, 0, 0)
            } else {
                // Walking: let go to stop first.
                GamePad::new(0, 0, 0)
            }
        }
        MoveKind::WallKick => {
            if a == ACT_AIR_HIT_WALL {
                // act_air_hit_wall runs twice a frame (its missing return), so
                // the kick's window is the one frame after the hit: A now
                // (released since vy fell under 20).
                pad(A_BUTTON)
            } else if a == ACT_SOFT_BONK {
                // Missed it: A within wallKickTimer's 5 frames still kicks
                // (an edge on even frames).
                pad(if even { A_BUTTON } else { 0 })
            } else {
                // Approach: jump (A held rising) and keep flying into the wall
                // ahead; facing the target unless reversing.
                let reversing = {
                    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
                    ((angle(dx, dz) as i32 - st.yaw as i32) as i16).unsigned_abs() > 0x6000
                };
                // On the ground, jump once the wall is within about 4 frames
                // (the jump rises past 20 up for 6, so the hit comes with A
                // already let go or within the bonk's late window).
                let b = if air {
                    rising_a
                } else {
                    let f = facing_dir(st);
                    let near = geom::wall_ahead(st, f, (st.forward_vel * 4.0).max(80.0), st.floor_y, WALL_KICK_COS).is_some()
                        || st.wall_normal.is_some();
                    if near || !reversing { ground_a } else { 0 }
                };
                if reversing { fwd()(b) } else { pad(b) }
            }
        }
        MoveKind::WallSlide => {
            let ctx_dir = {
                let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
                let l = (dx * dx + dz * dz).sqrt().max(1e-3);
                [dx / l, dz / l]
            };
            let n = st.wall_normal.unwrap_or([-ctx_dir[0], 0.0, -ctx_dir[1]]);
            let l = (n[0] * n[0] + n[2] * n[2]).sqrt().max(1e-3);
            let n2 = [n[0] / l, n[2] / l];
            // The direction along the wall nearest the target's.
            let dot = ctx_dir[0] * n2[0] + ctx_dir[1] * n2[1];
            let mut t = [ctx_dir[0] - dot * n2[0], ctx_dir[1] - dot * n2[1]];
            if t[0].abs() + t[1].abs() < 0.05 {
                t = [-n2[1], n2[0]];
            }
            let (sx, sy) = stick_dir(st, t[0].atan2(t[1]));
            GamePad::new(0, sx, sy)
        }
        MoveKind::DiveRollout => {
            if matches!(a, ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE) {
                // A or B rolls out (once off the slide-steep floor).
                pad(if even { A_BUTTON } else { 0 })
            } else if air {
                pad(if air_actionable(a) { B_BUTTON } else { 0 })
            } else if matches!(a, ACT_WALKING) {
                // check_ground_dive_or_punch: B at >= 29 with the stick out dives.
                pad(if even { B_BUTTON } else { 0 })
            } else {
                pad(0)
            }
        }
        MoveKind::BellySlide => pad(0),
        MoveKind::SlideKick => {
            // "Hold Z, press B while moving": Z (an edge on even frames) from
            // walking starts the crouch slide; B in it slide kicks (>= 10);
            // the kick's first landing bounces (act_slide_kick); in
            // SLIDE_KICK_SLIDE A hops out (forward rollout).
            match a {
                ACT_CROUCH_SLIDE => pad(Z_TRIG | B_BUTTON),
                ACT_SLIDE_KICK_SLIDE => pad(if even { A_BUTTON } else { 0 }),
                ACT_WALKING | ACT_DECELERATING | ACT_BRAKING => pad(if even { Z_TRIG } else { 0 }),
                _ if air => pad(rising_a),
                _ => pad(0),
            }
        }
        MoveKind::GroundPound => {
            // Z in the air pounds (edge: Z isn't held in the air otherwise);
            // the pound itself reads no stick.
            if air && air_actionable(a) { GamePad::new(Z_TRIG, 0, 0) } else { pad(0) }
        }
        // Let go of the stick at the edge (decelerates; the GA can still
        // try the drop).
        MoveKind::FallSafe => GamePad::new(0, 0, 0),
        MoveKind::LedgeClimb => pad(if even { A_BUTTON } else { 0 }),
        MoveKind::SlideEscape => pad(if even { A_BUTTON } else { 0 }),
        MoveKind::SlideRide => pad(0),
    }
}

/// The steering controller as a [`Mover`]: the per-mode controllers
/// ([`modes`]) with, in the ground and air mode, the GA's move policies
/// ([`policies`]) and, when `steer` is on (`OBRRT_MOVES=1`), the move
/// library before the generic rule.
pub struct Library {
    steer: AtomicBool,
}

impl Library {
    /// Whether the move library steers the ground and air mode.
    pub fn steering(&self) -> bool {
        self.steer.load(Ordering::Relaxed)
    }

    /// The move library's pad for a frame of the ground and air mode
    /// (None: the generic rule).
    pub fn library_pad(&self, st: &St, target: [f32; 3], arrival: Arrival, button: u16, frame: usize) -> Option<GamePad> {
        if !self.steering() {
            return None;
        }
        // An objective's own button (grab, punch) keeps the generic rule;
        // so do states outside play and a target right here.
        if button != 0 || !st.controllable() || st.hdist(target) < 20.0 && (target[1] - st.pos[1]).abs() < 50.0 {
            return None;
        }
        doors::observe(st);
        policies::classify_pad(st, target, arrival, frame)
    }
}

impl Mover for Library {
    fn control(&self, st: &St, target: [f32; 3], arrival: Arrival, button: u16, frame: usize, jump_every: usize, policy: Option<usize>) -> GamePad {
        modes::control(self, st, target, arrival, button, frame, jump_every, policy)
    }

    fn policies(&self) -> usize {
        policies::offered()
    }

    fn policy_name(&self, p: usize) -> &str {
        policies::POLICIES.get(p).map_or("?", |p| p.name())
    }

    fn policy_plausible(&self, p: usize, st: &St, target: [f32; 3], arrival: Arrival) -> bool {
        policies::plausible(p, st, target, arrival)
    }

    fn is_jump_start(&self, prev: &St, cur: &St) -> bool {
        policies::is_jump_start(prev, cur)
    }

    fn in_flight(&self, st: &St) -> bool {
        st.action & ACT_FLAG_AIR != 0 || is_landing(st.action) || st.action == ACT_LONG_JUMP_LAND
    }

    fn jump_useful(&self, st: &St, target: [f32; 3], arrival: Arrival) -> f32 {
        policies::jump_useful(st, target, arrival)
    }
}

/// Doors open only while WALKING/DECELERATING within ~110 of their
/// centre, and a landing's lag-slide into one bonks: for a target at a
/// known door (within 50; [`Arrival::Door`] targets), the arcs planner's approach (run, jumps that
/// land short, walk in along the door's normal). Only within 2500 of the
/// door, so the planner's cost (~30 µs) stays on the last stretch.
fn door_pad(st: &St, target: [f32; 3]) -> Option<GamePad> {
    let (pos, yaw) = doors::door_at(st.zone(), target)?;
    if st.hdist(pos) > 2500.0 {
        return None;
    }
    let (_, dir) = arcs::door_approach(pos, yaw, st.pos, 0.0);
    let zone = st.zone();
    let floor = |x: f32, y: f32, z: f32| geom::floor_y(zone, x, z, y);
    let plan = arcs::plan_approach_with(st, pos, Some(dir), arcs::DOOR_STANDOFF, Some(&floor))?;
    Some(plan.now.pad(st))
}

/// Remembers what `st` shows that rollouts can't (its doors); a tool that
/// steers single-threaded calls this and then [`obrrt_core::mover::publish`].
pub fn observe(st: &St) {
    doors::observe(st);
}

fn library() -> &'static Arc<Library> {
    static L: OnceLock<Arc<Library>> = OnceLock::new();
    L.get_or_init(|| {
        let l = Arc::new(Library { steer: AtomicBool::new(false) });
        obrrt_core::mover::set_mover(l.clone());
        l
    })
}

/// Registers the steering controller with the move library off: the
/// per-mode controllers and the generic ground rule only (the steering
/// before the library). Every binary that steers calls this (or
/// [`install`] / [`install_policies`]).
pub fn install_controller() {
    library();
}

/// [`install_controller`] with the move library steering, and static floor
/// snapshots on (`OBRRT_MOVES=1`, run/src/main.rs).
pub fn install() {
    obrrt_core::geo::want_static_floors();
    library().steer.store(true, Ordering::Relaxed);
}

/// [`install_controller`] with static floor snapshots for the GA's move
/// policies (`OBRRT_MOVE_GA=1` without `OBRRT_MOVES=1`): the default
/// steering stays generic.
pub fn install_policies() {
    obrrt_core::geo::want_static_floors();
    library();
}

/// Turns the move library's steering off again (labs comparing both).
pub fn set_steering(on: bool) {
    library().steer.store(on, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Stick math (as search/src/steer.rs).

/// The raw stick that makes `intendedYaw` point from Mario to `target`,
/// full tilt; centred within 20 units.
pub fn camera_stick(st: &St, target: [f32; 3]) -> (i8, i8) {
    let (dx, dz) = (target[0] - st.pos[0], target[2] - st.pos[2]);
    if dx.abs() + dz.abs() < 20.0 {
        return (0, 0);
    }
    stick_dir(st, dx.atan2(dz))
}

/// Full-tilt raw stick for a world direction (radians, 0 = +z, TAU/4 = +x).
pub fn stick_dir(st: &St, desired: f32) -> (i8, i8) {
    let camera = st.cam_yaw as f32 * TAU / 65536.0;
    let rel = desired - camera;
    ((127.0 * rel.sin()) as i8, (-127.0 * rel.cos()) as i8)
}

fn angle(x: f32, y: f32) -> i16 {
    ((x.atan2(y) * 65536.0 / TAU).round() as i32) as i16
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod modes_tests;
