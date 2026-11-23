//! Shared vocabulary of the layers. Owned by the integrator: change only by agreement.

use obrrt_core::{GamePad, St};
use obrrt_gp::lang::Ctx;

/// Which of A, B, Z are DOWN this frame (levels, not edges: whether that is a
/// fresh press or a hold follows from the previous pad).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Btn(pub u8);
impl Btn {
    pub const A: u8 = 1;
    pub const B: u8 = 2;
    pub const Z: u8 = 4;
    pub fn a(self) -> bool { self.0 & Self::A != 0 }
    pub fn b(self) -> bool { self.0 & Self::B != 0 }
    pub fn z(self) -> bool { self.0 & Self::Z != 0 }
}

/// Reference direction a stick angle is measured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ref {
    /// Direction to the task's current target (Ctx.pts[0]).
    Target,
    /// Mario's facing (faceAngle[1]).
    Face,
    /// The slide / velocity direction (atan2 of vel).
    Vel,
    /// Directly up the floor's fall line (uphill) / downhill.
    Uphill,
    Downhill,
    /// The camera's forward (stick straight up on the pad).
    Cam,
    /// Fixed world yaw (the skill gives it in `Choice.yaw`).
    World,
}

/// Magnitude class, by what the game does with the stick magnitude
/// (deadzone, intendedMag/32, the 48 speed-kick limit, the 61+ full range).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mag {
    Neutral,
    /// 0 < mag <= 48 (never a speed-kick dive; walking below full speed).
    Short,
    /// mag > 48 up to full (speed kick dive, full air steering).
    Full,
}

/// Angle class relative to `Ref`. Bounds come from the guards (held back is
/// |dyaw| > 0x471C, wall angles, ...). `Aim` is the continuous part: the
/// refiner (mpc) moves `fine` inside the class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ang {
    /// Within a few degrees of the reference.
    Aim,
    /// Mild left/right (turn while keeping most of the forward component).
    Left,
    Right,
    /// Sideways.
    SideLeft,
    SideRight,
    /// Held back (|dyaw| > 0x471C): turn-around / braking / bleed.
    Back,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StickCls {
    pub mag: Mag,
    pub ang: Ang,
    pub r: Ref,
}

/// One cell of the reduced menu: a button set and a stick region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cell {
    pub btn: Btn,
    pub stick: StickCls,
}

/// What a skill returns for one frame: a cell plus the continuous part.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub cell: Cell,
    /// World yaw for `Ref::World`.
    pub yaw: i16,
    /// Position inside the class, -1..1 (angle) and 0..1 (magnitude);
    /// 0 = the class's default point.
    pub fine_ang: f32,
    pub fine_mag: f32,
}

/// What the game's surroundings look like from Mario (sense.rs fills it, once
/// per frame, from emulator memory and the static geometry). Lengths are in
/// game units; `f32::INFINITY` = nothing within range.
#[derive(Clone, Debug, Default)]
pub struct Sense {
    /// Floor probes along Mario's heading (faceAngle) at 24, 48, 96, 192, 384, 768 units.
    pub ahead: [FloorProbe; 6],
    /// Same along the velocity direction (slides) at 96, 384.
    pub vel_ahead: [FloorProbe; 2],
    /// Wall raycasts (nearest wall hit) forward, left, right, back at wall height ~ 30..160.
    pub wall: [f32; 4],
    /// Yaw of the forward-hit wall's normal.
    pub wall_normal_yaw: i16,
    /// Headroom above Mario.
    pub ceil_clear: f32,
    /// Distance ahead to a floor drop > 100 (ledge / edge), if any.
    pub edge_ahead: f32,
    /// Distance ahead to a ledge-grabbable lip, if any.
    pub ledge_ahead: f32,
    /// Airborne: ballistic prediction of the landing under the current pad kept.
    pub land: Option<Landing>,
    /// The nearest objects by category (enemy, coin, star, pole, platform/moving floor, bomb, other), relative to Mario.
    pub objs: Vec<ObjProbe>,
    /// Zone flags from the task (e.g. Lakitu box): Mario's position inside, and the predicted landing inside.
    pub in_zone: bool,
    /// The predicted landing (`land`) lies in a zone.
    pub land_zone: bool,
    /// On the ground: where each jump would land if pressed now (full stick along the facing, A held):
    /// double jump, jump kick, single jump, long jump (index = `HYP_*`).
    pub hyp: [Option<Landing>; 4],
    /// Each of those landings lies in a zone.
    pub hyp_zone: [bool; 4],
}

pub const HYP_DJ: usize = 0;
pub const HYP_KICK: usize = 1;
pub const HYP_JUMP: usize = 2;
pub const HYP_LJ: usize = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct FloorProbe {
    pub dist: f32,
    pub y: f32,
    pub ny: f32,
    /// Surface type (SURFACE_*) and class (0 very slippery .. 3 not slippery).
    pub ty: u16,
    pub class: u8,
    pub has_floor: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Landing {
    pub frames: u32,
    pub pos: [f32; 3],
    pub fv: f32,
    pub probe: FloorProbe,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ObjProbe {
    pub cat: u8,
    pub dist: f32,
    /// Angle of the object relative to Mario's heading, as yaw.
    pub rel_yaw: i16,
    pub dy: f32,
    pub radius: f32,
}

/// What a controller sees: the gp crate's context (Mario's own state in `s`:
/// action, speed, flags, health, held object, caps, floor, wall) plus the
/// surroundings.
pub struct Obs<'a> {
    pub ctx: &'a Ctx<'a>,
    pub sense: &'a Sense,
}

/// Turn a choice into the pad for this frame.
pub fn realize(c: &Choice, o: &Obs) -> GamePad {
    crate::menu::realize(c, o)
}

pub fn act_name(s: &St) -> u32 { s.action }
