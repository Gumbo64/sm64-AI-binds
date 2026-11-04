//! Techniques: BLJs and hyper speed, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: objective-rrt/objectives/src/extra/tech/{blj,bljany,lblj,hyperwalk,hyperclimb}.rs
//!
//! rrt kind -> scattershot kind:
//! - `Tech::Blj` / `Tech::BljRun` (blj.rs; the preset BLJ spots: BoB elevator,
//!   WF stairs, the lobby steps, ...) -> [`TechBlj`] with a `center`/`radius`
//!   (or `min`/`max`) around each spot. The spot table itself (poses, setup
//!   overlays, run-up) is search machinery and not ported. What the rrt
//!   objective reads, the BLJ chain (re-jump from the long-jump landing, the
//!   backwards speed growing x1.5), becomes the fields `phase` and `bspeed`;
//!   the destination of the BLJ (the summit, the ledge) is a plain `Lens` or
//!   `Stage` further on in the schema.
//!   A BLJ-only region with no BLJ-in-progress distinction is a plain `Lens`
//!   (`speed_step` 16, `yaw_sectors` 16, `actions` [`long_jump`]): that is
//!   what the zero-star Lobby BLJ needs; use `TechBlj` when the chain's speed
//!   steps (x1.5) and the kind of support should be cells of their own.
//! - `Tech::BljAny` (bljany.rs, the eight `Kind`s of `blj_ok`) -> [`TechBljAny`]
//!   (`TechBlj` with no region, `extra_kinds` and `need_setup` on): the field
//!   `support` is the `Kind` (1 Elevator, 2 Ceiling, 3 Slope, 4 Fast,
//!   5 RisingFloor, 6 WaterSlope, 7 BlockedStep, 8 StepRiser). Not ported
//!   from it: the 400-unit cell / facing-quadrant dedup (the key itself does
//!   that), the wall-stall facing fan, and the success/hand-off rules.
//! - `Tech::Lblj` (lblj.rs) -> [`TechLblj`]: `TechBlj` gated to the lobby
//!   states in front of the 8-star door (fewer than `max_stars`, no key),
//!   plus the single-cell `behind` state for the corridor behind the door.
//!   BitDW (level 17) is the next level and is `Stage`'s business.
//! - `Tech::HyperWalk` (hyperwalk.rs) -> [`TechHyperWalk`]: a hyperspeed
//!   (|fv| >= 100) state: speed level, motion heading (relative to a goal
//!   direction if one is given), air/ground. The per-goal objective
//!   generation (zones, 200-unit goal cells, star/warp goals), the stick
//!   overlay, hops and stops are not ported: goals are other objectives, and
//!   inputs are the search's.
//! - `Tech::HyperClimb` (hyperclimb.rs) -> [`TechHyperClimb`]: hyperspeed
//!   forward (fv >= 150) with the launch (rising at 80+) and the height
//!   gained as cells. The target list (stars, poles, ! boxes, places), the
//!   climb heuristic and the pad overlay are not ported.
//! - Not ported anywhere: blj.rs `metric`/`chain_speed`/`progress` (they score
//!   trajectories, the key has no trajectory), `Tech::PoleRedirect` hand-offs
//!   (that is `tech_misc`), steering and button hints (`steer_*`, overlays),
//!   budgets and kinship.
//!
//! All distances are straight-line; no walking metric is used.

use anyhow::Result;
use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist, in_box, pack_pos, yaw_bucket, yaw_sector};
use crate::objective::{Cx, Objective};

const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_FLAG_AIR: u32 = 0x0000_0800;
const ACT_FLAG_SWIMMING: u32 = 0x0000_2000;

/// Platform behaviours whose floor can rise without oVelY (bljany.rs `PLATFORMS`).
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

/// Highest backwards-speed level (16 * 1.5^11 is about 2900).
const MAX_LEVEL: i32 = 11;

/// The `support` field's values (bljany.rs `Kind`, 0 = none).
pub mod support {
    pub const ELEVATOR: i32 = 1;
    pub const CEILING: i32 = 2;
    pub const SLOPE: i32 = 3;
    pub const FAST: i32 = 4;
    pub const RISING_FLOOR: i32 = 5;
    pub const WATER_SLOPE: i32 = 6;
    pub const BLOCKED_STEP: i32 = 7;
    pub const STEP_RISER: i32 = 8;
}

fn controllable(st: &St) -> bool {
    // obrrt_core::St::controllable, as the rrt objectives gate on it.
    st.controllable()
}

fn long_jumping(st: &St) -> bool {
    matches!(st.action, ACT_LONG_JUMP | ACT_LONG_JUMP_LAND)
}

/// 0 other, 1 long jump (air), 2 its landing, 3 crouch slide (the setup).
fn phase(st: &St) -> i32 {
    match st.action {
        ACT_LONG_JUMP => 1,
        ACT_LONG_JUMP_LAND => 2,
        ACT_CROUCH_SLIDE => 3,
        _ => 0,
    }
}

/// 1 + the x1.5 speed level of a backwards speed (16 is level 0): the BLJ
/// grows x1.5 a jump (blj.rs `metric`: log1.5(max(16, v) / 16)).
fn speed_level(back: f32) -> i32 {
    let l = ((back.max(16.0) / 16.0).ln() / 1.5f32.ln() + 1e-4).floor() as i32;
    1 + l.clamp(0, MAX_LEVEL)
}

/// The floor-class slide test of bljany.rs `slides_below`: (never slides,
/// the normal.y at or below which a landing at backward speed slides).
fn slides_below(floor_type: i16) -> (bool, f32) {
    match floor_type {
        0x15 | 0x37 | 0x7A => (true, 0.0),
        0x14 | 0x2A | 0x35 | 0x79 => (false, 0.9397),
        0x13 | 0x2E | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => (false, 0.9848),
        _ => (false, 0.7880),
    }
}

/// bljany.rs `blj_ok`: which condition lets a backwards long jump build speed
/// here (a `support` value, 0 if none). `platforms` are the behaviour
/// addresses of [`PLATFORMS`].
fn support_of(s: &St, extra: bool, platforms: &[u32]) -> i32 {
    let grounded = s.action & ACT_FLAG_AIR == 0 || long_jumping(s);
    let water = s.water_level as f32;
    let under = if extra { s.pos[1] < water - 100.0 } else { s.pos[1] < water };
    if !controllable(s) || s.holding || !grounded || under {
        return 0;
    }
    let band = extra && s.pos[1] < water - 40.0;
    let (not_slippery, slide_ny) = slides_below(s.floor_type);
    let ny = s.floor_normal[1];
    if ny <= if band { 0.2924 } else { slide_ny } {
        return 0;
    }
    if s.floor_obj_vy.is_some_and(|v| v > 2.0) {
        return support::ELEVATOR;
    }
    if extra && rising_floor(s, platforms) {
        return support::RISING_FLOOR;
    }
    let headroom = s.ceil_y - s.floor_y - 160.0;
    if (0.0..50.0).contains(&headroom) {
        return support::CEILING;
    }
    let rise = (1.0 - ny * ny).max(0.0).sqrt() / ny.max(1e-3);
    if not_slippery && rise >= 1.25 && ny > 0.2924 {
        return support::SLOPE;
    }
    if band && rise >= 1.25 {
        return support::WATER_SLOPE;
    }
    if extra
        && s.action == ACT_LONG_JUMP
        && s.forward_vel <= -3.0
        && (s.pos[1] - s.floor_y).abs() < 2.0
        && (-4.5..-1.5).contains(&s.vel[1])
    {
        return support::BLOCKED_STEP;
    }
    if extra && s.action == ACT_LONG_JUMP_LAND && s.forward_vel > -53.0 && s.forward_vel <= -20.0 && ny > 0.9 {
        return support::STEP_RISER;
    }
    if s.forward_vel <= -53.0 {
        return support::FAST;
    }
    0
}

/// An object floor that may rise without oVelY: tilted, or a known platform.
fn rising_floor(s: &St, platforms: &[u32]) -> bool {
    let Some(slot) = s.floor_obj else { return false };
    if s.floor_normal[1] < 0.98 {
        return true;
    }
    let objs: &[Obj] = s.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[]);
    objs.iter().any(|o| o.slot == slot && platforms.contains(&o.bhv))
}

fn platform_addrs() -> Vec<u32> {
    PLATFORMS.iter().filter_map(|n| bhv_addr(n)).collect()
}

fn seven() -> u32 {
    7
}
fn sixteen() -> u32 {
    16
}
fn three() -> u32 {
    3
}
fn forty_five() -> f32 {
    45.0
}
fn hundred() -> f32 {
    100.0
}
fn fifty() -> f32 {
    50.0
}
fn yes() -> bool {
    true
}

/// The lobby's 8-star door pair (lblj.rs `DOOR`, `N`): midpoint, and the
/// front normal (towards the lobby).
const DOOR: [f32; 3] = [-2652.0, 512.0, -1463.0];
const DOOR_N: [f32; 2] = [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2];
const LEVEL_CASTLE: u16 = 6;

/// (signed distance past the door plane, offset across the corridor) of `p`.
fn behind(p: [f32; 3]) -> (f32, f32) {
    let (dx, dz) = (p[0] - DOOR[0], p[2] - DOOR[2]);
    (-(dx * DOOR_N[0] + dz * DOOR_N[1]), dx * DOOR_N[1] - dz * DOOR_N[0])
}

/// The corridor behind the 8-star door (lblj.rs `behind_door`).
fn behind_door(s: &St) -> bool {
    let (along, across) = behind(s.pos);
    s.level == LEVEL_CASTLE && s.area == 1 && (150.0..4600.0).contains(&along) && across.abs() < 600.0 && s.pos[1] > 400.0
}

/// A backwards long jump: the BLJ-in-progress state and what makes a BLJ
/// possible, in a region.
///
/// Gate: Mario controllable and not holding (and inside the region, a sphere
/// `center`/`radius` or box `min`/`max`; with neither, everywhere), and with
/// `need_setup` only where a BLJ can build (`support` != 0) or a chain is
/// going (`phase` 1/2 at a backwards speed). Fields:
/// - `pos`: position on the 2^`grid_log2` grid, only while the backwards
///   speed is below `pos_speed` (at hyperspeed where he is doesn't matter).
/// - `phase`: 0 other, 1 in the long jump, 2 in its landing, 3 crouch slide
///   (the Z setup).
/// - `bspeed`: 0, or in a long jump / landing moving backwards, 1 + the x1.5
///   speed level (level 0 is up to 16; one cell per jump of the chain).
/// - `yaw`: the facing, `yaw_sectors` equal sectors; or with `yaw_toward`
///   (x, z) fine within `yaw_fine_deg` of it (`yaw_fine_sectors`) and coarse
///   (`yaw_coarse_sectors`) elsewhere. For a BLJ, give the direction the
///   stairs/wall is to be climbed, Mario faces down it.
/// - `support`: see [`support`]; `extra_kinds` off keeps the four basic ones
///   (elevator, ceiling, slope, fast).
///
/// ```json
/// {"kind": "TechBlj", "name": "BobElevatorBLJ", "params": {"center": [3100, 600, 2900], "radius": 700, "grid_log2": 7}}
/// ```
#[derive(Deserialize)]
pub struct TechBlj {
    pub center: Option<[f32; 3]>,
    pub radius: Option<f32>,
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
    #[serde(default = "seven")]
    pub grid_log2: u32,
    #[serde(default = "hundred")]
    pub pos_speed: f32,
    #[serde(default = "sixteen")]
    pub yaw_sectors: u32,
    pub yaw_toward: Option<[f32; 2]>,
    #[serde(default = "forty_five")]
    pub yaw_fine_deg: f32,
    #[serde(default = "seven")]
    pub yaw_fine_sectors: u32,
    #[serde(default = "three")]
    pub yaw_coarse_sectors: u32,
    #[serde(default)]
    pub extra_kinds: bool,
    #[serde(default)]
    pub need_setup: bool,
    /// Lobby gate (set by [`TechLblj`] only).
    #[serde(skip)]
    lobby_max_stars: Option<i32>,
    #[serde(skip)]
    lobby_radius: f32,
    #[serde(skip)]
    platforms: Vec<u32>,
}

impl TechBlj {
    fn finish(mut self) -> Self {
        self.platforms = platform_addrs();
        self
    }

    fn inside(&self, p: [f32; 3]) -> bool {
        let sphere = match (self.center, self.radius) {
            (Some(c), Some(r)) => Some(dist(p, c) < r),
            _ => None,
        };
        let boxed = match (self.min, self.max) {
            (Some(lo), Some(hi)) => Some(in_box(p, lo, hi)),
            _ => None,
        };
        match (sphere, boxed) {
            (None, None) => true,
            (a, b) => a == Some(true) || b == Some(true),
        }
    }

    /// The key proper, once the region/lobby gate has passed.
    fn write(&self, st: &St, cx: &mut Cx) {
        if !controllable(st) || st.holding {
            return;
        }
        let ph = phase(st);
        let chain = long_jumping(st) && st.forward_vel < -1.0;
        let bs = if chain { speed_level(-st.forward_vel) } else { 0 };
        let sup = support_of(st, self.extra_kinds, &self.platforms);
        if self.need_setup && sup == 0 && !chain {
            return;
        }
        if -st.forward_vel < self.pos_speed || !chain {
            cx.put(0, pack_pos(st.pos, self.grid_log2));
        }
        cx.put(1, ph);
        cx.put(2, bs);
        cx.put(3, match self.yaw_toward {
            Some([x, z]) => {
                let target = (x.atan2(z) / std::f32::consts::TAU * 65536.0).round() as i32 as i16;
                let range = (self.yaw_fine_deg / 360.0 * 65536.0) as i32;
                yaw_bucket(st.yaw, target, range, self.yaw_fine_sectors, self.yaw_coarse_sectors)
            }
            None => yaw_sector(st.yaw, self.yaw_sectors),
        });
        cx.put(4, sup);
        cx.all_children();
    }
}

impl Objective for TechBlj {
    fn fields(&self) -> &[&'static str] {
        &["pos", "phase", "bspeed", "yaw", "support", "behind"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        self.extra_kinds
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if let Some(max_stars) = self.lobby_max_stars {
            // Lobby gate (lblj.rs `start`): in front of the 8-star door.
            if st.level != LEVEL_CASTLE || st.area != 1 {
                return;
            }
            if behind_door(st) {
                cx.put(5, 1);
                return;
            }
            let h = ((st.pos[0] - DOOR[0]).powi(2) + (st.pos[2] - DOOR[2]).powi(2)).sqrt();
            if i32::from(st.stars) >= max_stars
                || st.keys != 0
                || st.pos[1] <= -60.0
                || behind(st.pos).0 >= 0.0
                || h >= self.lobby_radius
            {
                return;
            }
        } else if !self.inside(st.pos) {
            return;
        }
        self.write(st, cx);
    }
}

/// `Tech::BljAny`: a BLJ wherever one can build speed. [`TechBlj`] with no
/// region, `extra_kinds` and `need_setup` on (the params of `TechBlj` are
/// accepted and may override, e.g. `grid_log2` 8 to coarsen).
///
/// ```json
/// {"kind": "TechBljAny", "params": {"grid_log2": 8}}
/// ```
pub struct TechBljAny;

#[derive(Deserialize)]
struct AnyParams {
    #[serde(default = "eight")]
    grid_log2: u32,
    #[serde(default = "yes")]
    extra_kinds: bool,
    #[serde(default = "yes")]
    need_setup: bool,
    #[serde(default = "sixteen")]
    yaw_sectors: u32,
    #[serde(default = "hundred")]
    pos_speed: f32,
}

fn eight() -> u32 {
    8
}

impl TechBljAny {
    fn build(p: Value) -> Result<TechBlj> {
        let a: AnyParams = serde_json::from_value(p)?;
        Ok(TechBlj {
            center: None,
            radius: None,
            min: None,
            max: None,
            grid_log2: a.grid_log2,
            pos_speed: a.pos_speed,
            yaw_sectors: a.yaw_sectors,
            yaw_toward: None,
            yaw_fine_deg: 45.0,
            yaw_fine_sectors: 7,
            yaw_coarse_sectors: 3,
            extra_kinds: a.extra_kinds,
            need_setup: a.need_setup,
            lobby_max_stars: None,
            lobby_radius: 0.0,
            platforms: Vec::new(),
        }
        .finish())
    }
}

/// `Tech::Lblj`: the lobby BLJ past the 8-star door with fewer than
/// `max_stars` (default 8) stars and no key, within `radius` (default 2600)
/// of the door, in front of it: the [`TechBlj`] fields (`yaw_toward` defaults
/// to the direction from the door to the trap floor, so Mario faces away
/// from where the BLJ carries him). In the corridor behind the door (past
/// the plane, within its width) the key is just `behind` = 1: one cell.
///
/// ```json
/// {"kind": "TechLblj", "name": "LobbyBLJ", "params": {"max_stars": 8}}
/// ```
pub struct TechLblj;

#[derive(Deserialize)]
struct LbljParams {
    #[serde(default = "eight_i")]
    max_stars: i32,
    #[serde(default = "radius_2600")]
    radius: f32,
    #[serde(default = "seven")]
    grid_log2: u32,
    #[serde(default = "sixteen")]
    yaw_sectors: u32,
    yaw_toward: Option<[f32; 2]>,
}

fn eight_i() -> i32 {
    8
}
fn radius_2600() -> f32 {
    2600.0
}

impl TechLblj {
    fn build(p: Value) -> Result<TechBlj> {
        let a: LbljParams = serde_json::from_value(p)?;
        Ok(TechBlj {
            center: None,
            radius: None,
            min: None,
            max: None,
            grid_log2: a.grid_log2,
            pos_speed: 100.0,
            yaw_sectors: a.yaw_sectors,
            yaw_toward: a.yaw_toward,
            yaw_fine_deg: 45.0,
            yaw_fine_sectors: 7,
            yaw_coarse_sectors: 3,
            extra_kinds: false,
            need_setup: false,
            lobby_max_stars: Some(a.max_stars),
            lobby_radius: a.radius,
            platforms: Vec::new(),
        }
        .finish())
    }
}

/// Mario's direction of motion: the facing, reversed at negative speed.
fn motion_yaw(st: &St) -> i16 {
    if st.forward_vel < 0.0 { st.yaw.wrapping_add(i16::MIN) } else { st.yaw }
}

/// Hyperspeed on the ground or in the air: |forward speed| >= `min_speed`
/// (default 100), controllable, not holding, in one of the `HyperWalk` base
/// actions (walking, braking, decelerating, the landings, long jump, jump
/// kick, jump, freefall, butt slide air, move punching) and optionally in a
/// region (sphere `center`/`radius` or box `min`/`max`). Fields:
/// - `speed`: floor(|fv| / `speed_step`) (default 50);
/// - `heading`: the direction of motion in `yaw_sectors` (default 16), or
///   with `goal_toward` (x, z) the same uneven buckets as `TechBlj`'s `yaw`
///   (fine within `yaw_fine_deg`);
/// - `air`: 1 when airborne (the stick steers differently), else 0;
/// - `pos`: a coarse (`grid_log2` 9) position.
///
/// ```json
/// {"kind": "TechHyperWalk", "params": {"min_speed": 100, "grid_log2": 9}}
/// ```
#[derive(Deserialize)]
pub struct TechHyperWalk {
    pub center: Option<[f32; 3]>,
    pub radius: Option<f32>,
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
    #[serde(default = "hundred")]
    pub min_speed: f32,
    #[serde(default = "fifty")]
    pub speed_step: f32,
    #[serde(default = "nine")]
    pub grid_log2: u32,
    #[serde(default = "sixteen")]
    pub yaw_sectors: u32,
    pub goal_toward: Option<[f32; 2]>,
    #[serde(default = "forty_five")]
    pub yaw_fine_deg: f32,
    #[serde(default = "seven")]
    pub yaw_fine_sectors: u32,
    #[serde(default = "three")]
    pub yaw_coarse_sectors: u32,
}

fn nine() -> u32 {
    9
}

/// The hyperwalk.rs base actions (low bits and flags of the action).
fn hyper_action(a: u32) -> bool {
    matches!(
        a,
        0x0400_0440 // walking
            | 0x0400_044A // decelerating
            | 0x0400_0445 // braking
            | 0x0400_0470 // jump land
            | 0x0400_0471 // freefall land
            | 0x0000_0479 // long jump land
            | 0x0300_0888 // long jump
            | 0x0300_0880 // jump
            | 0x0180_08AC // jump kick
            | 0x0100_088C // freefall
            | 0x0300_088E // butt slide air
            | 0x0080_0457 // move punching
    )
}

fn region_has(
    center: Option<[f32; 3]>,
    radius: Option<f32>,
    min: Option<[f32; 3]>,
    max: Option<[f32; 3]>,
    p: [f32; 3],
) -> bool {
    let sphere = center.zip(radius).map(|(c, r)| dist(p, c) < r);
    let boxed = min.zip(max).map(|(lo, hi)| in_box(p, lo, hi));
    match (sphere, boxed) {
        (None, None) => true,
        (a, b) => a == Some(true) || b == Some(true),
    }
}

fn heading(yaw: i16, toward: Option<[f32; 2]>, sectors: u32, deg: f32, fine: u32, coarse: u32) -> i32 {
    match toward {
        Some([x, z]) => {
            let target = (x.atan2(z) / std::f32::consts::TAU * 65536.0).round() as i32 as i16;
            yaw_bucket(yaw, target, (deg / 360.0 * 65536.0) as i32, fine, coarse)
        }
        None => yaw_sector(yaw, sectors),
    }
}

impl Objective for TechHyperWalk {
    fn fields(&self) -> &[&'static str] {
        &["pos", "speed", "heading", "air"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.forward_vel.abs() < self.min_speed
            || !controllable(st)
            || st.holding
            || !hyper_action(st.action)
            || !region_has(self.center, self.radius, self.min, self.max, st.pos)
        {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, bucket(st.forward_vel.abs(), self.speed_step));
        cx.put(
            2,
            heading(
                motion_yaw(st),
                self.goal_toward,
                self.yaw_sectors,
                self.yaw_fine_deg,
                self.yaw_fine_sectors,
                self.yaw_coarse_sectors,
            ),
        );
        cx.put(3, (st.action & ACT_FLAG_AIR != 0) as i32);
        cx.all_children();
    }
}

/// Positive hyperspeed turned into height: forward speed >= `min_speed`
/// (default 150), controllable, not holding, on the ground or in the air (not
/// swimming), optionally in a region. Fields:
/// - `launched`: 1 when airborne and rising at `launch_vy` (default 80)
///   or more (a jump at hyperspeed; plain jumps, wall kicks alone don't
///   count), else 0;
/// - `height`: floor((y - `base_y`) / `height_step`) (defaults 0 and 250);
/// - `vspeed`: vertical speed in `vspeed_step` buckets (default 40), only
///   when airborne, else 0;
/// - `speed`: floor(fv / `speed_step`) (default 50);
/// - `pos`: coarse position (`grid_log2` 9) so different places differ.
///
/// ```json
/// {"kind": "TechHyperClimb", "params": {"min_speed": 150, "height_step": 250}}
/// ```
#[derive(Deserialize)]
pub struct TechHyperClimb {
    pub center: Option<[f32; 3]>,
    pub radius: Option<f32>,
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
    #[serde(default = "one_fifty")]
    pub min_speed: f32,
    #[serde(default = "eighty")]
    pub launch_vy: f32,
    #[serde(default)]
    pub base_y: f32,
    #[serde(default = "two_fifty")]
    pub height_step: f32,
    #[serde(default = "forty")]
    pub vspeed_step: f32,
    #[serde(default = "fifty")]
    pub speed_step: f32,
    #[serde(default = "nine")]
    pub grid_log2: u32,
}

fn one_fifty() -> f32 {
    150.0
}
fn eighty() -> f32 {
    80.0
}
fn two_fifty() -> f32 {
    250.0
}
fn forty() -> f32 {
    40.0
}

impl Objective for TechHyperClimb {
    fn fields(&self) -> &[&'static str] {
        &["pos", "launched", "height", "vspeed", "speed"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.forward_vel < self.min_speed
            || !controllable(st)
            || st.holding
            || st.action & ACT_FLAG_SWIMMING != 0
            || !region_has(self.center, self.radius, self.min, self.max, st.pos)
        {
            return;
        }
        let air = st.action & ACT_FLAG_AIR != 0;
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, (air && st.vel[1] >= self.launch_vy) as i32);
        cx.put(2, bucket(st.pos[1] - self.base_y, self.height_step));
        cx.put(3, if air { bucket(st.vel[1], self.vspeed_step) } else { 0 });
        cx.put(4, bucket(st.forward_vel, self.speed_step));
        cx.all_children();
    }
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    let p = params.clone();
    Some(match kind {
        "TechBlj" => serde_json::from_value::<TechBlj>(p).map(|b| Box::new(b.finish()) as Box<dyn Objective>).map_err(Into::into),
        "TechBljAny" => TechBljAny::build(p).map(|b| Box::new(b) as Box<dyn Objective>),
        "TechLblj" => TechLblj::build(p).map(|b| Box::new(b) as Box<dyn Objective>),
        "TechHyperWalk" => serde_json::from_value::<TechHyperWalk>(p).map(|b| Box::new(b) as Box<dyn Objective>).map_err(Into::into),
        "TechHyperClimb" => serde_json::from_value::<TechHyperClimb>(p).map(|b| Box::new(b) as Box<dyn Objective>).map_err(Into::into),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, st};

    fn schema(kind: &str, params: &str) -> String {
        format!(r#"{{"kind": "{kind}", "params": {params}}}"#)
    }

    fn lj(pos: [f32; 3], fv: f32, action: u32) -> St {
        let mut s = st(pos, fv, 9, 1, 0);
        s.action = action;
        s.floor_obj_vy = Some(8.0);
        s
    }

    const BLJ: &str = r#"{"center": [0, 0, 0], "radius": 500}"#;

    #[test]
    fn gate_region_and_control() {
        let j = schema("TechBlj", BLJ);
        assert!(!key_of(&j, 0, &lj([0.0, 0.0, 0.0], -20.0, ACT_LONG_JUMP)).is_empty());
        assert!(key_of(&j, 0, &lj([900.0, 0.0, 0.0], -20.0, ACT_LONG_JUMP)).is_empty());
        let mut s = lj([0.0, 0.0, 0.0], -20.0, ACT_LONG_JUMP);
        s.holding = true;
        assert!(key_of(&j, 0, &s).is_empty());
        s.holding = false;
        s.play_mode = 4;
        assert!(key_of(&j, 0, &s).is_empty());
    }

    #[test]
    fn chain_speed_levels() {
        let j = schema("TechBlj", BLJ);
        let b = |fv: f32| {
            let k = key_of(&j, 0, &lj([0.0, 0.0, 0.0], fv, ACT_LONG_JUMP_LAND));
            k.iter().find(|e| e.0 == 2).unwrap().1
        };
        assert_eq!(b(-10.0), 1);
        assert_eq!(b(-16.0), 1);
        assert_eq!(b(-24.5), 2); // 16 * 1.5 = 24 -> level 1
        assert_eq!(b(-36.5), 3);
        assert!(b(-3000.0) <= 1 + MAX_LEVEL);
        // A forward long jump is no chain.
        assert_eq!(b(20.0), 0);
        // Same speed level = same cell, next level = another.
        let k = |fv| key_of(&j, 0, &lj([0.0, 0.0, 0.0], fv, ACT_LONG_JUMP_LAND));
        assert_eq!(k(-17.0), k(-20.0));
        assert_ne!(k(-20.0), k(-30.0));
    }

    #[test]
    fn phases_and_pos_dropped_at_speed() {
        let j = schema("TechBlj", BLJ);
        let air = key_of(&j, 0, &lj([0.0, 0.0, 0.0], -30.0, ACT_LONG_JUMP));
        let land = key_of(&j, 0, &lj([0.0, 0.0, 0.0], -30.0, ACT_LONG_JUMP_LAND));
        assert_ne!(air, land);
        // At pos_speed and beyond the position is not in the key.
        let fast = |x: f32| key_of(&j, 0, &lj([x, 0.0, 0.0], -150.0, ACT_LONG_JUMP_LAND));
        assert!(fast(0.0).iter().all(|e| e.0 != 0));
        assert_eq!(fast(0.0), fast(300.0));
        // Slow: position matters.
        let slow = |x: f32| key_of(&j, 0, &lj([x, 0.0, 0.0], -30.0, ACT_LONG_JUMP_LAND));
        assert_ne!(slow(0.0), slow(300.0));
    }

    #[test]
    fn support_kinds() {
        let mut s = lj([0.0, 0.0, 0.0], 0.0, 0x0400_0440);
        s.floor_obj_vy = None;
        assert_eq!(support_of(&s, false, &[]), 0);
        s.floor_obj_vy = Some(3.0);
        assert_eq!(support_of(&s, false, &[]), support::ELEVATOR);
        s.floor_obj_vy = None;
        s.floor_y = 0.0;
        s.ceil_y = 200.0; // headroom 40
        assert_eq!(support_of(&s, false, &[]), support::CEILING);
        s.ceil_y = 20000.0;
        s.floor_type = 0x15;
        s.floor_normal = [0.6, 0.8, 0.0]; // ny 0.8: rise 0.75 -> no
        assert_eq!(support_of(&s, false, &[]), 0);
        s.floor_normal = [0.8, 0.6, 0.0]; // rise 1.33
        assert_eq!(support_of(&s, false, &[]), support::SLOPE);
        s.floor_type = 0; // default class slides below ny 0.788
        assert_eq!(support_of(&s, false, &[]), 0);
        // Fast chain on flat ground.
        let mut f = lj([0.0, 0.0, 0.0], -60.0, ACT_LONG_JUMP_LAND);
        f.floor_obj_vy = None;
        assert_eq!(support_of(&f, false, &[]), support::FAST);
        // Step riser only with the extra kinds.
        f.forward_vel = -30.0;
        assert_eq!(support_of(&f, false, &[]), 0);
        assert_eq!(support_of(&f, true, &[]), support::STEP_RISER);
    }

    #[test]
    fn bljany_needs_setup() {
        let j = schema("TechBljAny", "{}");
        let flat = st([0.0, 0.0, 0.0], 0.0, 9, 1, 0);
        assert!(key_of(&j, 0, &flat).is_empty());
        let on_elevator = lj([0.0, 0.0, 0.0], 0.0, 0x0400_0440);
        let k = key_of(&j, 0, &on_elevator);
        assert_eq!(k.iter().find(|e| e.0 == 4).unwrap().1, support::ELEVATOR);
        // A chain in progress counts even with no support left.
        let mut chain = lj([0.0, 0.0, 0.0], -40.0, ACT_LONG_JUMP);
        chain.floor_obj_vy = None;
        assert!(!key_of(&j, 0, &chain).is_empty());
    }

    #[test]
    fn yaw_toward_buckets() {
        let j = schema("TechBlj", r#"{"center": [0,0,0], "radius": 500, "yaw_toward": [0, 1]}"#);
        let mut a = lj([0.0, 0.0, 0.0], -20.0, ACT_LONG_JUMP);
        let mut b = a.clone();
        a.yaw = 0;
        b.yaw = 0x4000; // 90 degrees off: coarse
        assert_ne!(key_of(&j, 0, &a), key_of(&j, 0, &b));
        let mut c = a.clone();
        c.yaw = 0x4800;
        assert_eq!(key_of(&j, 0, &b), key_of(&j, 0, &c));
    }

    #[test]
    fn lblj_lobby() {
        let j = schema("TechLblj", r#"{"max_stars": 8}"#);
        let mut s = st([-2000.0, 512.0, -800.0], 0.0, 6, 1, 0); // in front of the door
        assert!(!key_of(&j, 0, &s).is_empty());
        s.stars = 8;
        assert!(key_of(&j, 0, &s).is_empty(), "with 8 stars the door just opens");
        s.stars = 0;
        s.keys = 1;
        assert!(key_of(&j, 0, &s).is_empty());
        s.keys = 0;
        s.level = 9;
        assert!(key_of(&j, 0, &s).is_empty());
        // Behind the door: one cell, wherever.
        let mut b1 = st([-3300.0, 717.0, -2100.0], 0.0, 6, 1, 0);
        let mut b2 = st([-3500.0, 717.0, -2300.0], -200.0, 6, 1, 0);
        b1.yaw = 0;
        b2.yaw = 1234;
        assert_eq!(key_of(&j, 0, &b1), vec![(5, 1)]);
        assert_eq!(key_of(&j, 0, &b1), key_of(&j, 0, &b2));
    }

    #[test]
    fn hyperwalk_gate_and_buckets() {
        let j = schema("TechHyperWalk", "{}");
        let mut s = st([0.0, 0.0, 0.0], 99.0, 9, 1, 0);
        assert!(key_of(&j, 0, &s).is_empty());
        s.forward_vel = -100.0;
        assert!(!key_of(&j, 0, &s).is_empty());
        let sp = |fv: f32| {
            let mut s = s.clone();
            s.forward_vel = fv;
            key_of(&j, 0, &s).iter().find(|e| e.0 == 1).unwrap().1
        };
        assert_eq!(sp(-149.0), 2);
        assert_eq!(sp(-150.0), 3);
        // Moving backwards at yaw y is the same heading as forwards at y + 0x8000.
        let mut a = s.clone();
        a.forward_vel = -200.0;
        a.yaw = 0;
        let mut b = s.clone();
        b.forward_vel = 200.0;
        b.yaw = i16::MIN;
        assert_eq!(key_of(&j, 0, &a), key_of(&j, 0, &b));
        // Not an allowed action: silent.
        a.action = 0x0C40_0201;
        assert!(key_of(&j, 0, &a).is_empty());
    }

    #[test]
    fn hyperclimb_launch_and_height() {
        let j = schema("TechHyperClimb", "{}");
        let mut s = st([0.0, 0.0, 0.0], 149.0, 9, 1, 0);
        assert!(key_of(&j, 0, &s).is_empty());
        s.forward_vel = 300.0;
        let ground = key_of(&j, 0, &s);
        assert_eq!(ground.iter().find(|e| e.0 == 1).unwrap().1, 0);
        let mut up = s.clone();
        up.action = 0x0300_0880;
        up.vel[1] = 120.0;
        up.pos[1] = 600.0;
        let k = key_of(&j, 0, &up);
        assert_eq!(k.iter().find(|e| e.0 == 1).unwrap().1, 1);
        assert_eq!(k.iter().find(|e| e.0 == 2).unwrap().1, 2);
        up.vel[1] = 79.0;
        assert_eq!(key_of(&j, 0, &up).iter().find(|e| e.0 == 1).unwrap().1, 0);
        // Swimming: silent.
        up.action = 0x300022B8;
        assert!(key_of(&j, 0, &up).is_empty());
    }

    #[test]
    fn unknown_kind_is_none() {
        assert!(build("TechNope", &Value::Null).is_none());
    }
}
