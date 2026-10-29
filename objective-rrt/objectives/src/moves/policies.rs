//! The library's moves as forcible policies for the GA's policy-seeded
//! rollouts (`OBRRT_MOVE_GA`, search/src/movegenome.rs): instead of the
//! classifier picking one move a frame, policy `p` plays its move whenever
//! the move can physically be done from Mario's action (and speed, where
//! the game gates the move on it), and otherwise leaves the frame to the
//! default steering (`None`). Which policy (and which sequence of them)
//! gets where soonest is left to simulation.
//!
//! [`plausible`] is a cheap pre-filter on the start state, the local
//! geometry and the measured arcs, for obvious nonsense only (a ground
//! pound towards a target level with Mario, a backflip with nothing tall
//! ahead): it is no prediction of which policy is best.
//!
//! Everything here is a pure function of the state, the target and the
//! frame (the [`obrrt_core::mover::Mover`] rule).

use obrrt_core::{Arrival, GamePad, St};

use super::*;

/// The policies, in index order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// The classifier's choice every frame (the library's `pad`; differs
    /// from the default steering only when the library isn't steering,
    /// `OBRRT_MOVES` unset).
    Classify,
    Run,
    UphillKick,
    LongJump,
    /// Jumps on every landing: single, double, triple (A cut short near
    /// the target).
    LedgeJump,
    SideFlip,
    Backflip,
    /// Jumps with A held while rising (full distance).
    GapJump,
    WallKick,
    DiveRollout,
    BellySlide,
    /// Z, then Z+B (slide kick), A in its slide (hop).
    SlideKick,
    GroundPound,
    /// arcs::plan_approach_with towards a door target.
    DoorApproach,
    /// The TAS's slope and high-speed techniques (slope.rs, S1-S6).
    SlopeDive,
    Surf,
    AirChain,
    SpeedDoubleJump,
    SteepJump,
    /// The control modes' own policies (cannon.rs, water_air.rs, slide.rs,
    /// pole.rs, tech.rs), plausible only in their states, offered with
    /// `OBRRT_MOVES_MODES=1`.
    CannonAim,
    CannonLob,
    CannonEdge,
    SwimTas,
    SwimSurface,
    SwimExit,
    FlyPlan,
    FlyHigh,
    FlyLand,
    SlideHold,
    SlideCourse,
    SlideSharp,
    SlideJump,
    SlideBrake,
    PoleTop,
    PoleKick,
    PoleDrop,
    SpeedKick,
    EdgeDive,
    HighDive,
}

pub const POLICIES: [Policy; 39] = [
    Policy::Classify,
    Policy::Run,
    Policy::UphillKick,
    Policy::LongJump,
    Policy::LedgeJump,
    Policy::SideFlip,
    Policy::Backflip,
    Policy::GapJump,
    Policy::WallKick,
    Policy::DiveRollout,
    Policy::BellySlide,
    Policy::SlideKick,
    Policy::GroundPound,
    Policy::DoorApproach,
    Policy::SlopeDive,
    Policy::Surf,
    Policy::AirChain,
    Policy::SpeedDoubleJump,
    Policy::SteepJump,
    Policy::CannonAim,
    Policy::CannonLob,
    Policy::CannonEdge,
    Policy::SwimTas,
    Policy::SwimSurface,
    Policy::SwimExit,
    Policy::FlyPlan,
    Policy::FlyHigh,
    Policy::FlyLand,
    Policy::SlideHold,
    Policy::SlideCourse,
    Policy::SlideSharp,
    Policy::SlideJump,
    Policy::SlideBrake,
    Policy::PoleTop,
    Policy::PoleKick,
    Policy::PoleDrop,
    Policy::SpeedKick,
    Policy::EdgeDive,
    Policy::HighDive,
];

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::Classify => "classify",
            Policy::Run => "run",
            Policy::UphillKick => "uphill-kick",
            Policy::LongJump => "long-jump",
            Policy::LedgeJump => "ledge-jump",
            Policy::SideFlip => "side-flip",
            Policy::Backflip => "backflip",
            Policy::GapJump => "gap-jump",
            Policy::WallKick => "wall-kick",
            Policy::DiveRollout => "dive-rollout",
            Policy::BellySlide => "belly-slide",
            Policy::SlideKick => "slide-kick",
            Policy::GroundPound => "ground-pound",
            Policy::DoorApproach => "door-approach",
            Policy::SlopeDive => "slope-dive",
            Policy::Surf => "surf",
            Policy::AirChain => "air-chain",
            Policy::SpeedDoubleJump => "speed-dj",
            Policy::SteepJump => "steep-jump",
            Policy::CannonAim => "cannon-aim",
            Policy::CannonLob => "cannon-lob",
            Policy::CannonEdge => "cannon-edge",
            Policy::SwimTas => "swim-tas",
            Policy::SwimSurface => "swim-surface",
            Policy::SwimExit => "swim-exit",
            Policy::FlyPlan => "fly-plan",
            Policy::FlyHigh => "fly-high",
            Policy::FlyLand => "fly-land",
            Policy::SlideHold => "slide-hold",
            Policy::SlideCourse => "slide-course",
            Policy::SlideSharp => "slide-sharp",
            Policy::SlideJump => "slide-jump",
            Policy::SlideBrake => "slide-brake",
            Policy::PoleTop => "pole-top",
            Policy::PoleKick => "pole-kick",
            Policy::PoleDrop => "pole-drop",
            Policy::SpeedKick => "speed-kick",
            Policy::EdgeDive => "edge-dive",
            Policy::HighDive => "high-dive",
        }
    }
}

const ACT_DIVE: u32 = 0x0188_088A;
const ACT_SLIDE_KICK: u32 = 0x0180_08AA;

fn walking(a: u32) -> bool {
    matches!(a, ACT_WALKING | ACT_DECELERATING | ACT_BRAKING)
}

/// The move policy `p` plays this frame, if it can be done from here.
fn forced(p: Policy, st: &St) -> Option<MoveKind> {
    let a = st.action;
    let air = a & ACT_FLAG_AIR != 0;
    let fv = st.forward_vel;
    use MoveKind as M;
    match p {
        Policy::Classify | Policy::DoorApproach => None,
        Policy::SlopeDive | Policy::Surf | Policy::AirChain | Policy::SpeedDoubleJump | Policy::SteepJump => None,
        Policy::Run => (!air && !is_sliding(a)).then_some(M::Run),
        Policy::UphillKick => Some(M::UphillKick),
        Policy::LongJump => (a == ACT_LONG_JUMP
            || !air && (walking(a) || is_landing(a) || matches!(a, ACT_IDLE | ACT_CROUCH_SLIDE | ACT_LONG_JUMP_LAND)))
        .then_some(M::LongJumpChain),
        Policy::LedgeJump => Some(M::LedgeJump),
        Policy::GapJump => Some(M::GapJump),
        Policy::SideFlip => (a == ACT_SIDE_FLIP || !air && !is_sliding(a)).then_some(M::SideFlip),
        Policy::Backflip => (a == ACT_BACKFLIP
            || !air && (walking(a) || is_landing(a) || matches!(a, ACT_IDLE | ACT_START_CROUCHING | ACT_CROUCHING)))
        .then_some(M::Backflip),
        Policy::WallKick => Some(M::WallKick),
        // The game's gates: a ground dive at >= 29 (else B punches), an air
        // dive above 28 (else a jump kick).
        Policy::DiveRollout | Policy::BellySlide => {
            let ok = a == ACT_WALKING && fv >= 29.0
                || air_actionable(a) && fv > 28.0
                || matches!(a, ACT_DIVE | ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE);
            ok.then_some(if p == Policy::BellySlide { M::BellySlide } else { M::DiveRollout })
        }
        // B in a crouch slide kicks at >= 10 (else punches).
        Policy::SlideKick => (walking(a) && fv >= 10.0 || matches!(a, ACT_CROUCH_SLIDE | ACT_SLIDE_KICK | ACT_SLIDE_KICK_SLIDE))
            .then_some(M::SlideKick),
        Policy::GroundPound => air_actionable(a).then_some(M::GroundPound),
        _ => None,
    }
}

/// Policy `p`'s pad this frame (None: default steering).
pub fn policy_pad(p: usize, st: &St, target: [f32; 3], arrival: Arrival, button: u16, frame: usize) -> Option<GamePad> {
    let p = *POLICIES.get(p)?;
    if button != 0 || !st.controllable() || st.hdist(target) < 20.0 && (target[1] - st.pos[1]).abs() < 50.0 {
        return None;
    }
    doors::observe(st);
    match p {
        Policy::Classify => classify_pad(st, target, arrival, frame),
        Policy::DoorApproach => door_pad(st, target),
        Policy::SlopeDive => super::slope::slope_dive(st, target, frame),
        Policy::Surf => super::slope::surf(st, target, frame),
        Policy::AirChain => super::slope::air_chain(st, target, frame),
        Policy::SpeedDoubleJump => super::slope::speed_double_jump(st, target, frame),
        Policy::SteepJump => super::slope::steep_jump(st, target, frame),
        _ if p.mode().is_some() => mode_pad(p, st, target, arrival, frame),
        _ => {
            let kind = forced(p, st)?;
            // The belly slide is a dive whose slide is ridden.
            let kind = if kind == MoveKind::BellySlide && !matches!(st.action, ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE) {
                MoveKind::DiveRollout
            } else {
                kind
            };
            Some(pad_for_with(kind, st, target, arrival, frame))
        }
    }
}

/// Whether policy `p` is worth simulating from `st` towards `target`
/// (module docs). Without geometry for the zone, only the state-based
/// rules apply.
pub fn plausible(p: usize, st: &St, target: [f32; 3], arrival: Arrival) -> bool {
    let Some(&p) = POLICIES.get(p) else { return false };
    if p.mode().is_some() {
        return modes_on() && mode_plausible(p, st, target, arrival);
    }
    let ctx = Ctx::with(st, target, arrival);
    // A target to pass may be overshot: no overshoot-based pruning.
    let pass = arrival == Arrival::Pass;
    let a = st.action;
    let ah = ctx.ahead.as_ref();
    let reach_d = ctx.hd.min(800.0);
    match p {
        Policy::Classify | Policy::Run | Policy::LedgeJump => true,
        Policy::DoorApproach => arrival == Arrival::Door && doors::door_at(st.zone(), target).is_some(),
        Policy::UphillKick => {
            // Steep floor here or ahead (towards the target).
            floor_too_steep(st, &ctx)
                || ah.is_some_and(|ah| ah.steepest(ctx.dir, ctx.hd.clamp(60.0, 1000.0)) > STEEP_GRADE * 0.8)
                || ah.is_none() && !ctx.air && st.floor_normal[1] < 0.85 && ctx.grade > 0.2
                // A target up a slope the probe (floors within a jump's
                // reach) doesn't see whole.
                || ctx.dy > 150.0 && ctx.dy / ctx.hd.max(1.0) > 0.3
        }
        Policy::LongJump => {
            // Room to build speed, a floor to land on and head room; the
            // shortest long jump (at the speed it will have, landing lag
            // slide included) not far past the target.
            let shortest = reach_on(Jump::Long, ctx.fv.max(15.0), false, ctx.dy.clamp(-300.0, 100.0));
            (if pass { ctx.hd > 300.0 } else { ctx.hd > 600.0 && shortest <= 1.5 * ctx.hd + 200.0 })
                && ctx.dy < 150.0
                && st.ceil_y - st.pos[1] > 200.0
                && ah.is_none_or(|ah| ah.gap(ctx.base, 150.0, 120.0).is_none_or(|(e, far)| far.is_some() || e > 400.0))
                && ah.is_none_or(|ah| ah.wall.is_none_or(|w| w > 300.0))
        }
        Policy::SideFlip | Policy::Backflip => {
            // Something tall ahead, and Mario turned away from it (both
            // flips go the other way from where he faces or runs).
            let tall = ctx.dy > 150.0
                || ah.is_some_and(|ah| ah.step_up(ctx.base, 120.0).is_some_and(|(d, h)| d <= 500.0 && h >= 120.0));
            let away = ctx.yaw_err > 0x5000;
            let ground = !ctx.air || a == if p == Policy::SideFlip { ACT_SIDE_FLIP } else { ACT_BACKFLIP };
            tall && away && ground && (p == Policy::SideFlip || ctx.fv < 20.0)
        }
        Policy::GapJump => {
            // A gap, a step or a target above; not a long overshoot of a
            // near level target.
            let overshoot = !pass && ctx.dy.abs() < 100.0 && reach(Jump::Single, ctx.fv) > ctx.hd + 400.0;
            let feature = ah.is_none_or(|ah| {
                ah.gap(ctx.base, 150.0, 120.0).is_some_and(|(e, _)| e <= reach_d)
                    || ah.step_up(ctx.base, 45.0).is_some_and(|(d, _)| d <= reach_d)
            });
            !overshoot && (feature || ctx.dy > 50.0)
        }
        Policy::WallKick => {
            // A wall ahead (towards the target), or behind with the target
            // behind (reversal), or touched now.
            matches!(a, ACT_AIR_HIT_WALL | ACT_SOFT_BONK)
                || st.wall_normal.is_some()
                || wall_along(st, &ctx, ctx.dir, reach_d.max(200.0))
                || ctx.yaw_err > 0x6000 && wall_along(st, &ctx, facing_dir(st), (ctx.fv * 10.0).max(300.0))
        }
        Policy::DiveRollout => {
            // A dive + rollout carries 300-400: not for a near target, nor
            // one well above.
            (pass || ctx.hd > 300.0) && ctx.dy < 200.0
        }
        Policy::BellySlide => {
            matches!(a, ACT_DIVE_SLIDE | ACT_STOMACH_SLIDE)
                || (pass || ctx.hd > 300.0) && (ctx.dy < -100.0 || ctx.grade < -0.1 || ah.is_some_and(|ah| ah.grade(ctx.base, 400.0).is_some_and(|g| g < -0.15)))
        }
        Policy::SlideKick => {
            // On or near a slope down, or a short gap.
            let slope = ctx.dy < -100.0
                || ctx.grade < -0.1
                || ah.is_some_and(|ah| ah.grade(ctx.base, 400.0).is_some_and(|g| g < -0.15));
            let short_gap = ah.is_some_and(|ah| {
                ah.gap(ctx.base, 150.0, 120.0).is_some_and(|(e, far)| e <= 300.0 && far.is_some_and(|f| f - e <= 200.0))
            });
            matches!(a, ACT_SLIDE_KICK | ACT_SLIDE_KICK_SLIDE) || (pass || ctx.hd > 300.0) && (slope || short_gap)
        }
        Policy::GroundPound => ctx.dy < -250.0 && ctx.hd < 300.0,
        Policy::SlopeDive => super::slope::plausible(0, st, &ctx),
        Policy::Surf => super::slope::plausible(1, st, &ctx),
        Policy::AirChain => super::slope::plausible(2, st, &ctx),
        Policy::SpeedDoubleJump => super::slope::plausible(3, st, &ctx),
        Policy::SteepJump => super::slope::plausible(4, st, &ctx),
        _ => modes_on() && mode_plausible(p, st, target, arrival),
    }
}

/// The classifier's pad (what [`Library::pad`] plays when steering): a
/// door target's approach, else the classified move.
pub fn classify_pad(st: &St, target: [f32; 3], arrival: Arrival, frame: usize) -> Option<GamePad> {
    if arrival == Arrival::Door {
        if let Some(p) = door_pad(st, target) {
            return Some(p);
        }
    }
    let ctx = Ctx::with(st, target, arrival);
    let key = key_of(st, &ctx);
    if key.geom == GeomClass::Unknown && !ctx.air && ctx.dy > 50.0 {
        return None;
    }
    let mut kind = select(st, &key, &ctx);
    // OBRRT_CALM: with the target well behind on the ground, turn (stick
    // at it, no A: a jump stops the turn) instead of jumping on; a wall
    // kick reversal and a side flip onto a tall step keep their reasons.
    if calm() && !ctx.air && ctx.yaw_err > BEHIND && JUMPY.contains(&kind) {
        kind = MoveKind::Run;
    }
    Some(pad_for_with(kind, st, target, arrival, frame))
}

/// More than 120° off Mario's facing (s16 units).
pub const BEHIND: u16 = 0x5555;

/// Moves that jump forwards (along his facing) from the ground.
const JUMPY: [MoveKind; 7] = [
    MoveKind::LedgeJump,
    MoveKind::GapJump,
    MoveKind::GapLongJump,
    MoveKind::LongJumpChain,
    MoveKind::DiveRollout,
    MoveKind::SlideKick,
    MoveKind::UphillKick,
];

/// `OBRRT_CALM=1`: fewer pointless jumps (module docs of mod.rs).
pub fn calm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_CALM").is_ok_and(|v| v == "1"))
}

const ACT_STEEP_JUMP: u32 = 0x0300_0885;
const ACT_HOLD_JUMP: u32 = 0x0300_08A0;

/// Whether `prev` -> `cur` starts a jump from the ground.
pub fn is_jump_start(prev: &St, cur: &St) -> bool {
    prev.action & ACT_FLAG_AIR == 0
        && matches!(
            cur.action,
            ACT_JUMP | ACT_DOUBLE_JUMP | ACT_TRIPLE_JUMP | ACT_BACKFLIP | ACT_SIDE_FLIP | ACT_LONG_JUMP | ACT_STEEP_JUMP | ACT_HOLD_JUMP
        )
}

/// How useful jumping is from `st` towards `target` (a multiplier on
/// random genomes' A presses, `OBRRT_CALM`): lower indoors (a ceiling
/// within 3000), near a door or a spot to stop at (within 800), with
/// nothing to jump onto, over or off ahead, and with the target behind
/// (jumping stops a turn, unless a wall ahead makes it a wall-kick
/// reversal). At least 0.15.
pub fn jump_useful(st: &St, target: [f32; 3], arrival: Arrival) -> f32 {
    let ctx = Ctx::with(st, target, arrival);
    let mut m = 1.0f32;
    if st.ceil_y - st.pos[1] < 3000.0 {
        m *= 0.6;
    }
    if arrival != Arrival::Pass && ctx.hd < 800.0 {
        m *= 0.4;
    }
    // Target behind: a jump stops the turn, unless a wall ahead (along his
    // facing) makes it a wall-kick reversal.
    let reversal = wall_along(st, &ctx, facing_dir(st), (ctx.fv * 10.0).max(300.0));
    if ctx.yaw_err > BEHIND && !ctx.air && !reversal {
        m *= 0.4;
    }
    let reach = ctx.hd.min(800.0);
    let feature = match &ctx.ahead {
        Some(ah) => {
            ah.step_up(ctx.base, 45.0).is_some_and(|(d, _)| d <= reach)
                || ah.gap(ctx.base, 150.0, 120.0).is_some_and(|(e, _)| e <= reach)
                || ah.wall.is_some_and(|w| w <= reach)
                || ah.steepest(ctx.dir, reach.max(60.0)) > STEEP_GRADE
        }
        None => true,
    };
    if !feature && ctx.dy < 100.0 {
        m *= 0.4;
    }
    m.max(0.15)
}

// ---------------------------------------------------------------------------
// The control modes' policies (cannon.rs, water_air.rs, slide.rs, pole.rs).

/// Which mode file a policy belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Cannon(super::cannon::Kind),
    Swim(super::water_air::Swim),
    Fly(super::water_air::Fly),
    Slide(super::slide::Slide),
    Pole(super::pole::Pole),
    /// Ground / air speed techniques (tech.rs).
    Tech(super::tech::Tech),
}

impl Policy {
    pub fn mode(self) -> Option<Mode> {
        use super::cannon::Kind as C;
        use super::pole::Pole as P;
        use super::slide::Slide as S;
        use super::water_air::{Fly as F, Swim as W};
        Some(match self {
            Policy::CannonAim => Mode::Cannon(C::Aim),
            Policy::CannonLob => Mode::Cannon(C::Lob),
            Policy::CannonEdge => Mode::Cannon(C::Edge),
            Policy::SwimTas => Mode::Swim(W::Tas),
            Policy::SwimSurface => Mode::Swim(W::Surface),
            Policy::SwimExit => Mode::Swim(W::Exit),
            Policy::FlyPlan => Mode::Fly(F::Plan),
            Policy::FlyHigh => Mode::Fly(F::High),
            Policy::FlyLand => Mode::Fly(F::Land),
            Policy::SlideHold => Mode::Slide(S::Hold),
            Policy::SlideCourse => Mode::Slide(S::Course),
            Policy::SlideSharp => Mode::Slide(S::Sharp),
            Policy::SlideJump => Mode::Slide(S::Jump),
            Policy::SlideBrake => Mode::Slide(S::Brake),
            Policy::PoleTop => Mode::Pole(P::Top),
            Policy::PoleKick => Mode::Pole(P::Kick),
            Policy::PoleDrop => Mode::Pole(P::Drop),
            Policy::SpeedKick => Mode::Tech(super::tech::Tech::SpeedKick),
            Policy::EdgeDive => Mode::Tech(super::tech::Tech::EdgeDive),
            Policy::HighDive => Mode::Tech(super::tech::Tech::HighDive),
            _ => return None,
        })
    }
}

/// `OBRRT_MOVES_MODES=1` (with `OBRRT_MOVE_GA=1`): the mode and technique
/// policies are offered. Unset, they are never plausible and never play:
/// the GA's plans are exactly what they were before them.
pub fn modes_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_MOVES_MODES").is_ok_and(|v| v == "1"))
}

fn mode_pad(p: Policy, st: &St, target: [f32; 3], arrival: Arrival, frame: usize) -> Option<GamePad> {
    if !modes_on() {
        return None;
    }
    match p.mode()? {
        Mode::Cannon(k) => super::cannon::pad_for(k, st, target, arrival, frame),
        Mode::Swim(k) => super::water_air::swim_pad(k, st, target, frame),
        Mode::Fly(k) => super::water_air::fly_pad(k, st, target, arrival, frame),
        Mode::Slide(k) => super::slide::slide_pad(k, st, target, frame),
        Mode::Pole(k) => super::pole::pole_pad(k, st, target, frame),
        Mode::Tech(k) => super::tech::tech_pad(k, st, target, frame),
    }
}

/// A mode policy's own plausibility (without the switch).
pub fn mode_plausible(p: Policy, st: &St, target: [f32; 3], arrival: Arrival) -> bool {
    match p.mode() {
        Some(Mode::Cannon(k)) => super::cannon::plausible(k, st, target, arrival),
        Some(Mode::Swim(k)) => super::water_air::swim_plausible(k, st, target),
        Some(Mode::Fly(k)) => super::water_air::fly_plausible(k, st, target),
        Some(Mode::Slide(k)) => super::slide::slide_plausible(k, st, target),
        Some(Mode::Pole(k)) => super::pole::pole_plausible(k, st, target),
        Some(Mode::Tech(k)) => super::tech::tech_plausible(k, st, target, arrival),
        None => false,
    }
}

/// The pad of policy `p` if it is a mode policy (cannon, swimming, flying,
/// poles: modes.rs asks before its per-mode controllers, which ignore the
/// other policies), else None.
pub fn mode_policy_pad(p: usize, st: &St, target: [f32; 3], arrival: Arrival, button: u16, frame: usize) -> Option<GamePad> {
    let pol = *POLICIES.get(p)?;
    match pol.mode()? {
        Mode::Slide(_) | Mode::Tech(_) => None, // ground mode: through policy_pad
        _ => policy_pad(p, st, target, arrival, button, frame),
    }
}
