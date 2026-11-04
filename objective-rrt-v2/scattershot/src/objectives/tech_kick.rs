//! Techniques: kicks, wall kicks, clips and hits, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: src/extra/tech/wallkick.rs, kicks.rs, flip.rs, bombclip.rs, whomphits.rs, kinghits.rs,
//! wfclimb.rs, wftop.rs, wfcless.rs, src/extra/enemies/clip.rs.
//!
//! A scattershot objective only decides which states are different cells, so a technique becomes
//! "where it happens (region / gate) + the few quantities it needs". The rrt success tests, progress
//! scores, deadlines, kick counts and start-cell keys have no counterpart (a cell holds one state,
//! not a trajectory); what a trajectory showed (a kick happened) is replaced by the action phase
//! the state is in.
//!
//! | rrt kind | scattershot kind |
//! |---|---|
//! | `WallKick::Climb` | [`WallKick`] as `WallKickClimb` |
//! | `TripleJumpWallKick::Climb` | `TripleJumpWallKickClimb` (same struct; the triple-jump phase is always keyed) |
//! | `WallKick::SideFlip` | `WallKickSideFlip` (same struct; speed step 4, the side-flip phase is always keyed) |
//! | `WallKick::Redirect` | `WallKickRedirect` (same struct; speed step 4 so 0.9x the contact speed is told apart, facing the wall) |
//! | `WallKick::Chain` | `WallKickChain` (same struct; adds the wall's direction in 8 sectors so two walls are two cells) |
//! | `Tech::SlideKick` | [`KickMoves`] as `TechSlideKick` (adds the slope flag for the downhill Boost) |
//! | `Tech::JumpKick` | [`KickMoves`] as `TechJumpKick` (landing-frame punch/kick phase) |
//! | `Tech::FlipHeight` | [`FlipHeight`] as `TechFlipHeight` |
//! | `Tech::BombClip` | [`BombClip`] as `TechBombClip` |
//! | `Bobomb::GateClip` | `BobombGateClip` (BombClip with the BoB gate as the wall) |
//! | `Boss::GrabRelease` (kinghits) | [`GrabRelease`] as `BossGrabRelease` |
//! | `Boss::WhompFast` (whomphits) | [`WhompFast`] as `BossWhompFast` |
//! | `Wf::FortressClimb` (wfclimb) | [`FortressClimb`] as `WfFortressClimb` |
//! | `Wf::TopKick` (wftop) | [`TopKick`] as `WfTopKick` |
//! | `Wf::Cless` (wfcless) | [`Cless`] as `WfCless` |
//!
//! Not ported: the objectives' pad overlays, steering hints, GA budgets, weights, `OBRRT_KICKS_OFF`
//! fixes, the per-start-cell dedupe keys, success tests and progress scores (search machinery);
//! `Tech::SlideKick`'s and `Tech::JumpKick`'s target points (a goal, not a state distinction);
//! `WallKick`'s kick counter and the wall-normal history of `WallKick::Chain` (a cell has no
//! history: the phase and the wall sector stand in); the bomb-clip "known floor on the far side"
//! map (needs the engine's geometry); `FortressClimb`'s deadline (frame counts are not keyed);
//! the "star is loaded" test of `Wf::TopKick` / `Wf::Cless` (the star flag being missing stands in).
//!
//! What a plain `Lens` already says: a speed/vertical-speed/yaw/action cell in a box. The ground part
//! of `SlideKick` / `JumpKick` (run-up at 27-32, a dive rollout or long jump chain) is
//! `{"kind":"Lens","params":{"min":..,"max":..,"speed_step":4,"actions":["walking","long_jump","slide_kick","dive","jump","freefall"]}}`;
//! use [`KickMoves`] when the kick phases (crouch slide, rollout, punch landing) must be told apart.
//! `WfTopKick`/`WfCless` positions without the star-gating are `Lens` with `center`/`radius` around
//! the tower (0,3584,0) / the drop (642,2560,2723); the kinds below add the star-missing gate and
//! the ledge/ring phases a Lens cannot see.

use anyhow::Result;
use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist, pack_pos, yaw_bucket, yaw_sector};
use crate::objective::{Cx, Objective};

const ACT_FLAG_AIR: u32 = 0x0000_0800;
const ACT_FLAG_MOVING: u32 = 0x0000_0400;

const ACT_SIDE_FLIP: u32 = 0x0100_0887;
const ACT_BACKFLIP: u32 = 0x0100_0883;
const ACT_TRIPLE_JUMP: u32 = 0x0100_0882;
const ACT_WALL_KICK_AIR: u32 = 0x0300_0886;
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;
const ACT_SOFT_BONK: u32 = 0x0102_08B6;
const ACT_BACKWARD_AIR_KB: u32 = 0x0102_08B0;
const ACT_FORWARD_AIR_KB: u32 = 0x0102_08B1;
const ACT_LEDGE_GRAB: u32 = 0x0800_034B;
const ACT_TURNING_AROUND: u32 = 0x0000_0443;
const ACT_FINISH_TURNING_AROUND: u32 = 0x0000_0444;
const ACT_START_CROUCHING: u32 = 0x0C00_8221;
const ACT_CROUCHING: u32 = 0x0C00_8220;
const ACT_STOP_CROUCHING: u32 = 0x0C00_8222;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080_045A;
const ACT_FORWARD_ROLLOUT: u32 = 0x0100_08A6;
const ACT_DIVE: u32 = 0x0188_088A;
const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_JUMP_KICK: u32 = 0x0180_08AC;
const ACT_PUNCHING: u32 = 0x0080_0380;
const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;
const ACT_GROUND_POUND: u32 = 0x0080_08A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;
const ACT_SQUISHED: u32 = 0x0002_0339;
const ACT_GRABBED: u32 = 0x0002_0370;

const LEVEL_BOB: u16 = 9;
const LEVEL_WF: u16 = 24;

fn alive(st: &St) -> bool {
    (st.health >> 8) >= 1
}

fn ledge_action(a: u32) -> bool {
    a == ACT_LEDGE_GRAB || (0x0000_054C..=0x0000_054F).contains(&a)
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn objects(st: &St) -> &[Obj] {
    st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[])
}

/// A sm64 yaw (s16, 0 faces +z, grows towards +x) of the horizontal vector (x, z).
fn yaw_of(x: f32, z: f32) -> i16 {
    (x.atan2(z) / std::f32::consts::TAU * 65536.0).round() as i32 as i16
}

/// Where a technique applies: a sphere (`center`/`radius`) or a box (`min`/`max`); with neither,
/// everywhere (cells everywhere, so give one).
#[derive(Deserialize, Default, Clone)]
pub struct Region {
    pub center: Option<[f32; 3]>,
    pub radius: Option<f32>,
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
}

impl Region {
    fn contains(&self, p: [f32; 3]) -> bool {
        let sphere = self.center.zip(self.radius).map(|(c, r)| dist(p, c) < r);
        let boxed = self.min.zip(self.max).map(|(lo, hi)| (0..3).all(|i| lo[i] <= p[i] && p[i] <= hi[i]));
        match (sphere, boxed) {
            (None, None) => true,
            (a, b) => a == Some(true) || b == Some(true),
        }
    }
}

/// The horizontal wall Mario last hit (`wall_normal` with |y| < 0.35, wallkick.rs `WALL_NY`).
fn wall_of(st: &St) -> Option<[f32; 3]> {
    st.wall_normal.filter(|n| n[1].abs() < 0.35 && (n[0] != 0.0 || n[2] != 0.0))
}

// ---------------------------------------------------------------------------------------------
// Wall kicks
// ---------------------------------------------------------------------------------------------

/// Wall-kick climbs (`WallKick::Climb`, `TripleJumpWallKick::Climb`, `WallKick::SideFlip`,
/// `WallKick::Redirect`, `WallKick::Chain`; wallkick.rs). In the region, the cell is the position
/// on a `grid_log2` grid, forward speed (`speed_step`, 0 = off), and the kick *phase*: 0 not in
/// the air, 1 airborne, 2 triple jump, 3 side flip, 4 the kick window (air hit wall, soft bonk,
/// knocked back), 5 wall kick, 6 ledge grab/climb. With a wall contact: `wall` (1, or 1 + the
/// wall's direction in `wall_sectors`) and `facing` (Mario's yaw relative to straight into the
/// wall, fine within 45 degrees; the 45 degree cone of the kick).
///
/// Kind names set the defaults: `WallKickClimb`, `TripleJumpWallKickClimb` (`speed_step` 8),
/// `WallKickSideFlip` and `WallKickRedirect` (`speed_step` 4), `WallKickChain` (`wall_sectors` 8).
/// `WallKick` is the bare kind with every default.
///
/// ```json
/// {"kind":"WallKickClimb","params":{"center":[3276,620,3890],"radius":900}}
/// ```
#[derive(Deserialize)]
pub struct WallKick {
    #[serde(flatten)]
    pub region: Region,
    #[serde(default = "seven")]
    pub grid_log2: u32,
    #[serde(default = "eight_f")]
    pub speed_step: f32,
    /// Vertical speed per bucket; 0 (default) leaves it out (the phase already says rising).
    #[serde(default)]
    pub vspeed_step: f32,
    /// Directions of the wall's normal told apart; 0 = just "touching a wall".
    #[serde(default)]
    pub wall_sectors: u32,
    /// Key Mario's facing relative to the wall while touching one.
    #[serde(default = "yes")]
    pub facing: bool,
}

fn seven() -> u32 {
    7
}
fn eight_f() -> f32 {
    8.0
}
fn yes() -> bool {
    true
}

fn wk_phase(a: u32) -> i32 {
    match a {
        ACT_TRIPLE_JUMP => 2,
        ACT_SIDE_FLIP => 3,
        ACT_AIR_HIT_WALL | ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB | ACT_FORWARD_AIR_KB => 4,
        ACT_WALL_KICK_AIR => 5,
        a if ledge_action(a) => 6,
        a if a & ACT_FLAG_AIR != 0 => 1,
        _ => 0,
    }
}

impl Objective for WallKick {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "vspeed", "phase", "wall", "facing"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.region.contains(st.pos) {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        if self.speed_step > 0.0 {
            cx.put(1, bucket(st.forward_vel, self.speed_step));
        }
        if self.vspeed_step > 0.0 {
            cx.put(2, bucket(st.vel[1], self.vspeed_step));
        }
        cx.put(3, wk_phase(st.action));
        match wall_of(st) {
            Some(n) => {
                let dir = yaw_of(n[0], n[2]);
                cx.put(4, 1 + if self.wall_sectors > 0 { yaw_sector(dir, self.wall_sectors) } else { 0 });
                if self.facing {
                    cx.put(5, yaw_bucket(st.yaw, dir.wrapping_add(i16::MIN), 0x2000, 3, 3));
                }
            }
            None => cx.put(4, 0),
        }
        cx.all_children();
    }
}

fn with_defaults(params: &Value, defaults: Value) -> Value {
    let mut p = if params.is_object() { params.clone() } else { Value::Object(Default::default()) };
    for (k, v) in defaults.as_object().unwrap() {
        p.as_object_mut().unwrap().entry(k.clone()).or_insert(v.clone());
    }
    p
}

// ---------------------------------------------------------------------------------------------
// Kicks
// ---------------------------------------------------------------------------------------------

/// `Tech::SlideKick` and `Tech::JumpKick` (kicks.rs): in the region, forward speed
/// (`speed_step`; the technique lives on keeping 27-32, or 75+ on a slope), the move Mario is in
/// (`move`: 0 other, 1 walking/running, 2 crouch slide, 3 slide kick (air), 4 slide kick slide,
/// 5 forward rollout, 6 dive or dive slide, 7 long jump, 8 jump kick, 9 punch (the landing-frame
/// punch that A turns into a jump kick, or a move punch), 10 other air) and, with `slope_cos` > 0,
/// `slope` = 1 on a floor steeper than that cosine (the downhill Boost). Position is on a
/// `grid_log2` grid.
///
/// `TechSlideKick` defaults `slope_cos` to 0.95; `TechJumpKick` to 0 (off).
///
/// ```json
/// {"kind":"TechSlideKick","params":{"min":[-6000,-500,-6000],"max":[6000,2500,6000],"grid_log2":8}}
/// ```
#[derive(Deserialize)]
pub struct KickMoves {
    #[serde(flatten)]
    pub region: Region,
    #[serde(default = "seven")]
    pub grid_log2: u32,
    #[serde(default = "four_f")]
    pub speed_step: f32,
    #[serde(default)]
    pub slope_cos: f32,
}

fn four_f() -> f32 {
    4.0
}

fn kick_move(a: u32) -> i32 {
    match a {
        ACT_CROUCH_SLIDE => 2,
        ACT_SLIDE_KICK => 3,
        ACT_SLIDE_KICK_SLIDE => 4,
        ACT_FORWARD_ROLLOUT => 5,
        ACT_DIVE | ACT_DIVE_SLIDE => 6,
        ACT_LONG_JUMP => 7,
        ACT_JUMP_KICK => 8,
        // ACT_JUMP_LAND_STOP, ACT_DOUBLE_JUMP_LAND_STOP, ACT_FREEFALL_LAND_STOP (the A+B landing kick).
        ACT_PUNCHING | ACT_MOVE_PUNCHING | 0x0C00_0230 | 0x0C00_0231 | 0x0C00_0232 => 9,
        a if a & ACT_FLAG_AIR != 0 => 10,
        a if a & ACT_FLAG_MOVING != 0 => 1,
        _ => 0,
    }
}

impl Objective for KickMoves {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "move", "slope"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.region.contains(st.pos) {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        if self.speed_step > 0.0 {
            cx.put(1, bucket(st.forward_vel, self.speed_step));
        }
        cx.put(2, kick_move(st.action));
        if self.slope_cos > 0.0 {
            cx.put(3, (st.floor_normal[1] < self.slope_cos) as i32);
        }
        cx.all_children();
    }
}

/// `Tech::FlipHeight` (flip.rs): height with no run-up from a side flip or a backflip. In the
/// region: position on a `grid_log2` grid (its y is the height gained), forward speed
/// (`speed_step`, 0 = off; a side flip needs 16+ to turn round), and the phase: 0 other,
/// 1 turning round (the side flip's A window), 2 crouching (the backflip's), 3 side flip,
/// 4 backflip, 5 ledge grab/climb, 6 other air.
///
/// ```json
/// {"kind":"TechFlipHeight","params":{"center":[0,3584,0],"radius":600}}
/// ```
#[derive(Deserialize)]
pub struct FlipHeight {
    #[serde(flatten)]
    pub region: Region,
    #[serde(default = "seven")]
    pub grid_log2: u32,
    #[serde(default = "eight_f")]
    pub speed_step: f32,
}

fn flip_phase(a: u32) -> i32 {
    match a {
        ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND => 1,
        ACT_START_CROUCHING | ACT_CROUCHING | ACT_STOP_CROUCHING => 2,
        ACT_SIDE_FLIP => 3,
        ACT_BACKFLIP => 4,
        a if ledge_action(a) => 5,
        a if a & ACT_FLAG_AIR != 0 => 6,
        _ => 0,
    }
}

impl Objective for FlipHeight {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "phase"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.region.contains(st.pos) {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        if self.speed_step > 0.0 {
            cx.put(1, bucket(st.forward_vel, self.speed_step));
        }
        cx.put(2, flip_phase(st.action));
        cx.all_children();
    }
}

// ---------------------------------------------------------------------------------------------
// Bomb clip
// ---------------------------------------------------------------------------------------------

/// `Tech::BombClip` / `Bobomb::GateClip` (bombclip.rs, enemies/clip.rs): a bloated bob-omb shoves
/// Mario through a thin wall. Active in the region (and within `setup_radius` of the wall point,
/// if there is one). Fields: `pos` (`grid_log2`), `held` (holding a bob-omb), `bomb` (the held or
/// nearest bob-omb within `fetch`: -1 none, 0 idle/other, 1 launched, 2 bloated = exploding),
/// `bomb_dist` (its distance in `bomb_step` buckets, at most 12; -1 none), `depth` (signed
/// distance past the wall plane in `depth_step` buckets clamped to -6..6; negative = the far
/// side; 0 without a wall) and `wall` (touching a horizontal wall).
///
/// The wall is a plane: `wall_point` (or the position of the first object with behaviour
/// `wall_bhv`) and the horizontal unit `wall_normal` [x, z] pointing to the *near* side; or, with
/// `toward_bhv`, the normal is derived to point from the nearest such object (e.g. the star)
/// through the wall, ie away from it. `course` gates on the course.
///
/// `BobombGateClip` defaults: `wall_bhv` "bhvChainChompGate", `toward_bhv` "bhvStar", `course` 1,
/// `setup_radius` 2500, and (BoB star 6) is silent once that star (`star_course` 1, `star_index`
/// 5) is collected.
///
/// ```json
/// {"kind":"BobombGateClip","params":{}}
/// ```
#[derive(Deserialize)]
pub struct BombClip {
    #[serde(flatten)]
    pub region: Region,
    #[serde(default = "seven")]
    pub grid_log2: u32,
    #[serde(default = "bomb_bhv")]
    pub bomb_bhv: String,
    #[serde(default = "fetch")]
    pub fetch: f32,
    #[serde(default = "fifty")]
    pub bomb_step: f32,
    #[serde(default = "forty")]
    pub depth_step: f32,
    pub wall_point: Option<[f32; 3]>,
    pub wall_normal: Option<[f32; 2]>,
    pub wall_bhv: Option<String>,
    pub toward_bhv: Option<String>,
    pub setup_radius: Option<f32>,
    pub course: Option<i16>,
    pub star_course: Option<i16>,
    pub star_index: Option<u8>,
}

fn bomb_bhv() -> String {
    "bhvBobomb".into()
}
fn fetch() -> f32 {
    1200.0
}
fn fifty() -> f32 {
    50.0
}
fn forty() -> f32 {
    40.0
}

impl BombClip {
    /// The wall plane (point, normal x z), if there is one.
    fn wall(&self, st: &St) -> Option<([f32; 3], [f32; 2])> {
        let point = match (&self.wall_bhv, self.wall_point) {
            (Some(name), _) => {
                let b = bhv_addr(name)?;
                objects(st).iter().find(|o| o.bhv == b)?.pos
            }
            (None, p) => p?,
        };
        let n = match (&self.toward_bhv, self.wall_normal) {
            (_, Some(n)) => n,
            (Some(name), None) => {
                let b = bhv_addr(name)?;
                let t = objects(st).iter().filter(|o| o.bhv == b).min_by(|a, c| dist(a.pos, point).total_cmp(&dist(c.pos, point)))?;
                // From the star through the wall to Mario's side.
                let (dx, dz) = (point[0] - t.pos[0], point[2] - t.pos[2]);
                let l = (dx * dx + dz * dz).sqrt().max(1.0);
                [dx / l, dz / l]
            }
            (None, None) => return None,
        };
        Some((point, n))
    }
}

impl Objective for BombClip {
    fn fields(&self) -> &[&'static str] {
        &["pos", "held", "bomb", "bomb_dist", "depth", "wall"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if self.course.is_some_and(|c| c != st.course) || !self.region.contains(st.pos) {
            return;
        }
        if let (Some(c), Some(i)) = (self.star_course, self.star_index) {
            if st.has_star(c, i) {
                return;
            }
        }
        let wall = self.wall(st);
        let wants_wall = self.wall_point.is_some() || self.wall_bhv.is_some();
        if wants_wall && wall.is_none() {
            return;
        }
        if let (Some((p, _)), Some(r)) = (wall, self.setup_radius) {
            if dist(st.pos, p) >= r {
                return;
            }
        }
        let bhv = bhv_addr(&self.bomb_bhv);
        let bombs = objects(st).iter().filter(|o| Some(o.bhv) == bhv);
        let held = bombs.clone().find(|o| Some(o.slot) == st.held_slot);
        let bomb = held.or_else(|| bombs.filter(|o| dist(o.pos, st.pos) < self.fetch).min_by(|a, b| dist(a.pos, st.pos).total_cmp(&dist(b.pos, st.pos))));
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, held.is_some() as i32);
        match bomb {
            Some(b) => {
                cx.put(2, match b.action {
                    1 => 1,
                    3 => 2,
                    _ => 0,
                });
                cx.put(3, bucket(dist(b.pos, st.pos), self.bomb_step).min(12));
            }
            None => {
                cx.put(2, -1);
                cx.put(3, -1);
            }
        }
        cx.put(4, match wall {
            Some((p, n)) => bucket((st.pos[0] - p[0]) * n[0] + (st.pos[2] - p[2]) * n[1], self.depth_step).clamp(-6, 6),
            None => 0,
        });
        cx.put(5, wall_of(st).is_some() as i32);
        cx.all_children();
    }
}

// ---------------------------------------------------------------------------------------------
// Boss hits
// ---------------------------------------------------------------------------------------------

/// Mario's bearing from `o` relative to its facing, in `n` sectors (sector 0 starts at the facing
/// and they grow clockwise seen from above; the middle ones are behind it).
fn bearing_sector(st: &St, o: &Obj, n: u32) -> i32 {
    let to = yaw_of(st.pos[0] - o.pos[0], st.pos[2] - o.pos[2]);
    yaw_sector(to.wrapping_sub(o.yaw as i16), n)
}

/// `Boss::GrabRelease` (kinghits.rs, BoB star 1): King Bob-omb hit by grabbing him from behind and
/// letting go at once. Active in BoB (`level`) while the star (`course`, `index`) is missing and a
/// `bhv` object exists. Fields: `held` (Mario holds him), `grabbed` (he holds Mario),
/// `dist` (`step` buckets, at most 16), `behind` (Mario's bearing from the king relative to the
/// king's facing, `sectors` of them: the middle ones are behind), `kact` (his action: idle,
/// walking, thrown ...), `health` and `above` (0 when thrown more than 100 below his home: off
/// the hill he takes no damage). Mario's position is not keyed (the distance and bearing are).
///
/// ```json
/// {"kind":"BossGrabRelease","params":{}}
/// ```
#[derive(Deserialize)]
pub struct GrabRelease {
    #[serde(default = "king_bhv")]
    pub bhv: String,
    #[serde(default = "bob")]
    pub level: u16,
    #[serde(default = "one_i")]
    pub course: i16,
    #[serde(default)]
    pub index: u8,
    #[serde(default = "step150")]
    pub step: f32,
    #[serde(default = "eight_u")]
    pub sectors: u32,
}

fn king_bhv() -> String {
    "bhvKingBobomb".into()
}
fn bob() -> u16 {
    LEVEL_BOB
}
fn one_i() -> i16 {
    1
}
fn step150() -> f32 {
    150.0
}
fn eight_u() -> u32 {
    8
}

impl Objective for GrabRelease {
    fn fields(&self) -> &[&'static str] {
        &["held", "grabbed", "dist", "behind", "kact", "health", "above"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.level != self.level || st.has_star(self.course, self.index) {
            return;
        }
        let Some(bhv) = bhv_addr(&self.bhv) else { return };
        let kings = objects(st).iter().filter(|o| o.bhv == bhv);
        let held = kings.clone().find(|o| Some(o.slot) == st.held_slot);
        let Some(k) = held.or_else(|| kings.min_by(|a, b| dist(a.pos, st.pos).total_cmp(&dist(b.pos, st.pos)))) else { return };
        cx.put(0, held.is_some() as i32);
        cx.put(1, (st.action == ACT_GRABBED) as i32);
        cx.put(2, bucket(dist(k.pos, st.pos), self.step).min(16));
        cx.put(3, bearing_sector(st, k, self.sectors));
        cx.put(4, k.action);
        cx.put(5, k.health);
        cx.put(6, (k.pos[1] - k.home[1] >= -100.0) as i32);
        cx.all_children();
    }
}

/// `Boss::WhompFast` (whomphits.rs, WF star 1): the Whomp King beaten at the game's fastest
/// cadence: Mario in his attack cone the first frame he can attack, above his back as he lands
/// flat, ground pound on his first flat frame. Active in WF (`level`) while the star (`course`,
/// `index`) is missing, within `reach` (3000) of the King. Fields: `dist` (`step` buckets, at most
/// 20), `cone` (Mario within 290 and 0x1E00 of the King's facing: where he can attack), `back`
/// (Mario within 300 horizontally of the point 600 ahead of the King, where his flat back is),
/// `mario` (0 other, 1 airborne, 2 ground pound, 3 ground pound landing, 4 squished), `wact`
/// (the King's action: chase 2, prepare 3, jump 4, land 5, flat 6, dead 8, done 9) and `health`.
///
/// ```json
/// {"kind":"BossWhompFast","params":{}}
/// ```
#[derive(Deserialize)]
pub struct WhompFast {
    #[serde(default = "whomp_bhv")]
    pub bhv: String,
    #[serde(default = "wf")]
    pub level: u16,
    #[serde(default = "two_i")]
    pub course: i16,
    #[serde(default)]
    pub index: u8,
    #[serde(default = "step150")]
    pub step: f32,
    #[serde(default = "reach3000")]
    pub reach: f32,
}

fn whomp_bhv() -> String {
    "bhvWhompKingBoss".into()
}
fn wf() -> u16 {
    LEVEL_WF
}
fn two_i() -> i16 {
    2
}
fn reach3000() -> f32 {
    3000.0
}

fn ahead(o: &Obj, forward: f32) -> [f32; 3] {
    let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
    [o.pos[0] + forward * a.sin(), o.pos[1], o.pos[2] + forward * a.cos()]
}

fn in_cone(st: &St, o: &Obj) -> bool {
    if hdist(st.pos, o.pos) >= 290.0 {
        return false;
    }
    let to = ((st.pos[0] - o.pos[0]).atan2(st.pos[2] - o.pos[2]) * 65536.0 / std::f32::consts::TAU) as i32;
    let diff = (to - o.yaw + 0x8000).rem_euclid(0x10000) - 0x8000;
    diff.abs() < 0x1E00
}

impl Objective for WhompFast {
    fn fields(&self) -> &[&'static str] {
        &["dist", "cone", "back", "mario", "wact", "health"]
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.level != self.level || st.has_star(self.course, self.index) {
            return;
        }
        let Some(bhv) = bhv_addr(&self.bhv) else { return };
        let Some(k) = objects(st).iter().filter(|o| o.bhv == bhv).min_by(|a, b| dist(a.pos, st.pos).total_cmp(&dist(b.pos, st.pos))) else { return };
        if dist(k.pos, st.pos) >= self.reach {
            return;
        }
        cx.put(0, bucket(dist(k.pos, st.pos), self.step).min(20));
        cx.put(1, in_cone(st, k) as i32);
        cx.put(2, (hdist(st.pos, ahead(k, 600.0)) < 300.0) as i32);
        cx.put(3, match st.action {
            ACT_GROUND_POUND => 2,
            ACT_GROUND_POUND_LAND => 3,
            ACT_SQUISHED => 4,
            a if a & ACT_FLAG_AIR != 0 => 1,
            _ => 0,
        });
        cx.put(4, k.action);
        cx.put(5, k.health);
        cx.all_children();
    }
}

// ---------------------------------------------------------------------------------------------
// Whomp's Fortress location techniques
// ---------------------------------------------------------------------------------------------

/// Common gate of the three WF techniques: WF area 1, alive, star (2, `index`) missing.
fn wf_gate(st: &St, index: u8) -> bool {
    st.level == LEVEL_WF && st.area == 1 && alive(st) && !st.has_star(2, index)
}

/// `Wf::FortressClimb` (wfclimb.rs): from the painting landing to the fortress top as a race, only
/// fast moves. Active in WF area 1 while star 1 (`index` 0) is missing. Fields: `pos`
/// (`grid_log2`), `dist` (straight line to the arena: the disc of radius 1250 round `king`,
/// `step` buckets), `sig` (the move: 0 other, 1 long jump, 2 wall kick, 3 side flip, 4 backflip,
/// 5 triple jump, 6 dive, 7 forward rollout), `hspeed` (`speed_step`, 0 = off) and `top`
/// (standing on the arena within 1450 of the King, not in the air).
///
/// ```json
/// {"kind":"WfFortressClimb","params":{"grid_log2":8}}
/// ```
#[derive(Deserialize)]
pub struct FortressClimb {
    #[serde(default = "eight_u")]
    pub grid_log2: u32,
    #[serde(default = "step250")]
    pub step: f32,
    #[serde(default = "eight_f")]
    pub speed_step: f32,
    #[serde(default = "king_pos")]
    pub king: [f32; 3],
    #[serde(default)]
    pub index: u8,
}

fn step250() -> f32 {
    250.0
}
fn king_pos() -> [f32; 3] {
    [0.0, 3584.0, 0.0]
}

fn climb_sig(a: u32) -> i32 {
    match a {
        ACT_LONG_JUMP => 1,
        ACT_WALL_KICK_AIR => 2,
        ACT_SIDE_FLIP => 3,
        ACT_BACKFLIP => 4,
        ACT_TRIPLE_JUMP => 5,
        ACT_DIVE | ACT_DIVE_SLIDE => 6,
        ACT_FORWARD_ROLLOUT => 7,
        _ => 0,
    }
}

impl Objective for FortressClimb {
    fn fields(&self) -> &[&'static str] {
        &["pos", "dist", "sig", "hspeed", "top"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !wf_gate(st, self.index) {
            return;
        }
        let r = (hdist(st.pos, self.king) - 1250.0).max(0.0);
        let d = (r * r + (st.pos[1] - self.king[1]).powi(2)).sqrt();
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, bucket(d, self.step));
        cx.put(2, climb_sig(st.action));
        if self.speed_step > 0.0 {
            cx.put(3, bucket(st.forward_vel, self.speed_step));
        }
        cx.put(4, (hdist(st.pos, self.king) < 1450.0 && st.pos[1] >= self.king[1] - 20.0 && st.action & ACT_FLAG_AIR == 0) as i32);
        cx.all_children();
    }
}

/// `Wf::TopKick` (wftop.rs): WF star 2 by jumping and wall-kicking up the tower's side. Active in
/// WF area 1 while star 2 (`index` 1) is missing, on the fortress top and above (y >= `fort_y`)
/// within `start_reach` of the tower `axis`. Fields: `pos` (`grid_log2`), `phase` (0 nothing yet,
/// 1 wall-kicking high by the tower (y > 4400, within 900), 2 ledge grab, 3 on the top ring (the
/// ring's floor at `top_y`, within `top_reach` + 60 of the axis, standing or grabbing)) and `air`.
///
/// ```json
/// {"kind":"WfTopKick","params":{}}
/// ```
#[derive(Deserialize)]
pub struct TopKick {
    #[serde(default = "seven")]
    pub grid_log2: u32,
    #[serde(default = "tower")]
    pub axis: [f32; 3],
    #[serde(default = "top_y")]
    pub top_y: f32,
    #[serde(default = "top_reach")]
    pub top_reach: f32,
    #[serde(default = "fort_y")]
    pub fort_y: f32,
    #[serde(default = "start_reach")]
    pub start_reach: f32,
    #[serde(default = "one_u8")]
    pub index: u8,
}

fn tower() -> [f32; 3] {
    [0.0, 3584.0, 0.0]
}
fn top_y() -> f32 {
    5376.0
}
fn top_reach() -> f32 {
    520.0
}
fn fort_y() -> f32 {
    3500.0
}
fn start_reach() -> f32 {
    1700.0
}
fn one_u8() -> u8 {
    1
}

impl Objective for TopKick {
    fn fields(&self) -> &[&'static str] {
        &["pos", "phase", "air"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !wf_gate(st, self.index) || st.pos[1] < self.fort_y || hdist(st.pos, self.axis) >= self.start_reach {
            return;
        }
        let air = st.action & ACT_FLAG_AIR != 0;
        let ledge = ledge_action(st.action);
        let on_top = hdist(st.pos, self.axis) < self.top_reach + 60.0
            && ((ledge && st.pos[1] > self.top_y - 100.0) || (!air && st.pos[1] >= self.top_y - 10.0 && st.forward_vel.abs() < 10.0));
        let phase = if on_top {
            3
        } else if ledge {
            2
        } else if st.action == ACT_WALL_KICK_AIR && st.pos[1] > 4400.0 && hdist(st.pos, self.axis) < 900.0 {
            1
        } else {
            0
        };
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, phase);
        cx.put(2, air as i32);
        cx.all_children();
    }
}

/// `Wf::Cless` (wfcless.rs): WF star 6 without the cannon, by falling off the walkway beside the
/// star and ledge-grabbing the wall. Active in WF area 1 while star 6 (`index` 5) is missing,
/// above y 1800 and within `radius` (2000) of the `drop` point. Fields: `pos` (`grid_log2`),
/// `phase` (0 on the walkway, y >= `top_y`; 1 below its edge; 2 ledge grab/climb within 300 of the
/// `star`) and `dstar` (below the edge: straight-line distance to the star in `step` buckets;
/// else -1).
///
/// ```json
/// {"kind":"WfCless","params":{}}
/// ```
#[derive(Deserialize)]
pub struct Cless {
    #[serde(default = "six")]
    pub grid_log2: u32,
    #[serde(default = "hundred")]
    pub step: f32,
    #[serde(default = "drop_pt")]
    pub drop: [f32; 3],
    #[serde(default = "star_pt")]
    pub star: [f32; 3],
    #[serde(default = "walk_y")]
    pub top_y: f32,
    #[serde(default = "two_k")]
    pub radius: f32,
    #[serde(default = "five_u8")]
    pub index: u8,
}

fn six() -> u32 {
    6
}
fn hundred() -> f32 {
    100.0
}
fn drop_pt() -> [f32; 3] {
    [642.0, 2560.0, 2723.0]
}
fn star_pt() -> [f32; 3] {
    [590.0, 2450.0, 2650.0]
}
fn walk_y() -> f32 {
    2540.0
}
fn two_k() -> f32 {
    2000.0
}
fn five_u8() -> u8 {
    5
}

impl Objective for Cless {
    fn fields(&self) -> &[&'static str] {
        &["pos", "phase", "dstar"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !wf_gate(st, self.index) || st.pos[1] <= 1800.0 || dist(st.pos, self.drop) >= self.radius {
            return;
        }
        let d = dist(st.pos, self.star);
        let below = st.pos[1] < self.top_y;
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, if !below {
            0
        } else if ledge_action(st.action) && d < 300.0 {
            2
        } else {
            1
        });
        cx.put(2, if below { bucket(d, self.step) } else { -1 });
        cx.all_children();
    }
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    fn mk<T: Objective + for<'de> Deserialize<'de> + 'static>(p: Value) -> Option<Result<Box<dyn Objective>>> {
        Some(serde_json::from_value::<T>(p).map(|o| Box::new(o) as Box<dyn Objective>).map_err(Into::into))
    }
    let p = params.clone();
    let d = |j: Value| with_defaults(params, j);
    match kind {
        "WallKick" | "WallKickClimb" | "TripleJumpWallKickClimb" => mk::<WallKick>(p),
        "WallKickSideFlip" | "WallKickRedirect" => mk::<WallKick>(d(serde_json::json!({"speed_step": 4.0}))),
        "WallKickChain" => mk::<WallKick>(d(serde_json::json!({"wall_sectors": 8}))),
        "TechSlideKick" => mk::<KickMoves>(d(serde_json::json!({"slope_cos": 0.95}))),
        "TechJumpKick" => mk::<KickMoves>(p),
        "TechFlipHeight" => mk::<FlipHeight>(p),
        "TechBombClip" => mk::<BombClip>(p),
        "BobombGateClip" => mk::<BombClip>(d(serde_json::json!({
            "wall_bhv": "bhvChainChompGate", "toward_bhv": "bhvStar", "course": 1, "setup_radius": 2500.0,
            "star_course": 1, "star_index": 5
        }))),
        "BossGrabRelease" => mk::<GrabRelease>(p),
        "BossWhompFast" => mk::<WhompFast>(p),
        "WfFortressClimb" => mk::<FortressClimb>(p),
        "WfTopKick" => mk::<TopKick>(p),
        "WfCless" => mk::<Cless>(p),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn key(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&format!(r#"{{"kind":"Root","children":[{{"kind":"{kind}","params":{params}}}]}}"#), 1, s)
    }
    fn at(p: [f32; 3], action: u32) -> St {
        let mut s = st(p, 0.0, LEVEL_WF, 1, 0);
        s.action = action;
        s
    }
    const BOX: &str = r#"{"min":[-1000,-1000,-1000],"max":[1000,1000,1000]}"#;

    #[test]
    fn unknown_kind_is_none() {
        assert!(build("Nope", &Value::Null).is_none());
        assert!(build("WallKickChain", &Value::Null).is_some());
    }

    #[test]
    fn wallkick_gated_by_region_and_phase() {
        assert!(key("WallKickClimb", BOX, &at([5000.0, 0.0, 0.0], 0x0300_0880)).is_empty());
        let air = key("WallKickClimb", BOX, &at([0.0, 0.0, 0.0], 0x0300_0880));
        let kick = key("WallKickClimb", BOX, &at([0.0, 0.0, 0.0], ACT_WALL_KICK_AIR));
        let window = key("WallKickClimb", BOX, &at([0.0, 0.0, 0.0], ACT_AIR_HIT_WALL));
        assert!(!air.is_empty());
        assert!(air != kick && kick != window && air != window);
    }

    #[test]
    fn wallkick_wall_facing_and_chain_sectors() {
        let mut a = at([0.0, 0.0, 0.0], ACT_AIR_HIT_WALL);
        a.wall_normal = Some([0.0, 0.0, -1.0]); // wall faces -z; Mario facing +z (yaw 0) runs into it
        let mut away = a.clone();
        away.yaw = i16::MIN; // facing -z, away from the wall
        assert_ne!(key("WallKickClimb", BOX, &a), key("WallKickClimb", BOX, &away));
        assert_ne!(key("WallKickClimb", BOX, &a), key("WallKickClimb", BOX, &at([0.0, 0.0, 0.0], ACT_AIR_HIT_WALL)));
        // Two perpendicular walls: one cell for the plain kind (both are "a wall", same facing
        // relative to it), two for Chain.
        let mut b = a.clone();
        b.wall_normal = Some([-1.0, 0.0, 0.0]);
        b.yaw = 0x4000; // still straight into it
        assert_eq!(key("WallKickClimb", BOX, &a), key("WallKickClimb", BOX, &b));
        assert_ne!(key("WallKickChain", BOX, &a), key("WallKickChain", BOX, &b));
        // A slope is not a wall.
        let mut f = a.clone();
        f.wall_normal = Some([0.0, 0.9, -0.4]);
        assert_eq!(key("WallKickClimb", BOX, &f), key("WallKickClimb", BOX, &at([0.0, 0.0, 0.0], ACT_AIR_HIT_WALL)));
    }

    #[test]
    fn wallkick_speed_buckets() {
        let k = |fv: f32| {
            let mut s = at([0.0, 0.0, 0.0], ACT_WALL_KICK_AIR);
            s.forward_vel = fv;
            key("WallKickRedirect", BOX, &s)
        };
        assert_eq!(k(32.0), k(35.9));
        assert_ne!(k(35.9), k(36.0));
        // Climb has 8-wide buckets.
        let c = |fv: f32| {
            let mut s = at([0.0, 0.0, 0.0], ACT_WALL_KICK_AIR);
            s.forward_vel = fv;
            key("WallKickClimb", BOX, &s)
        };
        assert_eq!(c(32.0), c(39.0));
    }

    #[test]
    fn kick_moves_and_slope() {
        let mut slide = at([0.0, 0.0, 0.0], ACT_SLIDE_KICK_SLIDE);
        slide.forward_vel = 40.0;
        let mut rollout = slide.clone();
        rollout.action = ACT_FORWARD_ROLLOUT;
        assert_ne!(key("TechSlideKick", BOX, &slide), key("TechSlideKick", BOX, &rollout));
        let mut down = slide.clone();
        down.floor_normal = [0.5, 0.8, 0.0];
        assert_ne!(key("TechSlideKick", BOX, &slide), key("TechSlideKick", BOX, &down));
        // JumpKick leaves the slope out.
        assert_eq!(key("TechJumpKick", BOX, &slide), key("TechJumpKick", BOX, &down));
        let punch = at([0.0, 0.0, 0.0], 0x0C00_0230);
        let walk = at([0.0, 0.0, 0.0], 0x0400_0440);
        assert_ne!(key("TechJumpKick", BOX, &punch), key("TechJumpKick", BOX, &walk));
        assert!(key("TechJumpKick", BOX, &at([9999.0, 0.0, 0.0], 0x0400_0440)).is_empty());
    }

    #[test]
    fn flip_phases_and_height() {
        let side = at([0.0, 100.0, 0.0], ACT_SIDE_FLIP);
        let back = at([0.0, 100.0, 0.0], ACT_BACKFLIP);
        let turn = at([0.0, 100.0, 0.0], ACT_TURNING_AROUND);
        let k = |s: &St| key("TechFlipHeight", BOX, s);
        assert!(k(&side) != k(&back) && k(&side) != k(&turn));
        let high = at([0.0, 400.0, 0.0], ACT_SIDE_FLIP);
        assert_ne!(k(&side), k(&high));
    }

    #[test]
    fn bombclip_held_bloated_and_side() {
        let gate = r#"{"wall_point":[0,0,0],"wall_normal":[0,1],"grid_log2":10}"#;
        let bomb = |a: i32, z: f32| {
            let mut o = obj("bhvBobomb", 3, [0.0, 0.0, z]);
            o.action = a;
            o
        };
        let s = |z: f32, bombs: Vec<Obj>, held: bool| {
            let mut s = with_objects(st([0.0, 0.0, z], 0.0, 9, 1, 0), bombs);
            if held {
                s.held_slot = Some(3);
                s.holding = true;
            }
            s
        };
        let near = s(300.0, vec![bomb(0, 400.0)], false);
        assert!(!key("TechBombClip", gate, &near).is_empty());
        // Held, bloated and idle differ.
        let heldk = key("TechBombClip", gate, &s(300.0, vec![bomb(0, 400.0)], true));
        assert_ne!(heldk, key("TechBombClip", gate, &near));
        assert_ne!(key("TechBombClip", gate, &s(300.0, vec![bomb(3, 400.0)], false)), key("TechBombClip", gate, &near));
        // Near and far side of the plane differ; no bomb within fetch still keys.
        assert_ne!(key("TechBombClip", gate, &s(60.0, vec![], false)), key("TechBombClip", gate, &s(-60.0, vec![], false)));
        // Far-away bomb = none.
        assert_eq!(key("TechBombClip", gate, &s(300.0, vec![bomb(0, 5000.0)], false)), key("TechBombClip", gate, &s(300.0, vec![], false)));
    }

    #[test]
    fn gate_clip_silent_without_gate_or_after_star() {
        let gate = obj("bhvChainChompGate", 1, [1000.0, 0.0, 0.0]);
        let star = obj("bhvStar", 2, [1300.0, 0.0, 0.0]);
        let mut s = with_objects(st([700.0, 0.0, 0.0], 0.0, 9, 1, 0), vec![gate.clone(), star.clone()]);
        s.course = 1;
        assert!(!key("BobombGateClip", "{}", &s).is_empty());
        let before = key("BobombGateClip", "{}", &s);
        // Mario past the bars (x > 1000): the depth differs.
        let mut past = s.clone();
        past.pos[0] = 1100.0;
        assert_ne!(before, key("BobombGateClip", "{}", &past));
        // No gate loaded, other course, star collected, too far.
        let mut none = s.clone();
        none.objects = None;
        assert!(key("BobombGateClip", "{}", &none).is_empty());
        let mut c = s.clone();
        c.course = 2;
        assert!(key("BobombGateClip", "{}", &c).is_empty());
        let mut got = s.clone();
        got.course_stars[0] = 1 << 5;
        assert!(key("BobombGateClip", "{}", &got).is_empty());
        let mut far = s.clone();
        far.pos[0] = -3000.0;
        assert!(key("BobombGateClip", "{}", &far).is_empty());
    }

    #[test]
    fn king_grab_release() {
        let mut king = obj("bhvKingBobomb", 4, [0.0, 100.0, 500.0]);
        king.home = [0.0, 100.0, 500.0];
        king.health = 3;
        king.yaw = 0; // faces +z
        let s = |p: [f32; 3]| with_objects(st(p, 0.0, LEVEL_BOB, 1, 0), vec![king.clone()]);
        let behind = key("BossGrabRelease", "{}", &s([0.0, 100.0, 300.0]));
        let front = key("BossGrabRelease", "{}", &s([0.0, 100.0, 700.0]));
        assert!(!behind.is_empty() && behind != front);
        // Held vs free.
        let mut h = s([0.0, 100.0, 450.0]);
        let free = key("BossGrabRelease", "{}", &h);
        h.held_slot = Some(4);
        h.holding = true;
        assert_ne!(free, key("BossGrabRelease", "{}", &h));
        // Silent in another level, without the king, or once the star is taken.
        let mut other = s([0.0, 100.0, 300.0]);
        other.level = 6;
        assert!(key("BossGrabRelease", "{}", &other).is_empty());
        assert!(key("BossGrabRelease", "{}", &st([0.0; 3], 0.0, LEVEL_BOB, 1, 0)).is_empty());
        let mut got = s([0.0, 100.0, 300.0]);
        got.course_stars[0] = 1;
        assert!(key("BossGrabRelease", "{}", &got).is_empty());
        // Thrown off the hill takes no damage: different cell.
        let mut off = king.clone();
        off.pos[1] = -50.0;
        assert_ne!(key("BossGrabRelease", "{}", &with_objects(st([0.0, 100.0, 300.0], 0.0, LEVEL_BOB, 1, 0), vec![off])), behind);
    }

    #[test]
    fn whomp_cone_and_back() {
        let mut k = obj("bhvWhompKingBoss", 2, [0.0, 3584.0, 0.0]);
        k.yaw = 0;
        k.health = 3;
        let s = |p: [f32; 3], a: u32| {
            let mut s = with_objects(st(p, 0.0, LEVEL_WF, 1, 0), vec![k.clone()]);
            s.action = a;
            s
        };
        let cone = key("BossWhompFast", "{}", &s([0.0, 3584.0, 200.0], 0x0400_0440));
        let behind = key("BossWhompFast", "{}", &s([0.0, 3584.0, -200.0], 0x0400_0440));
        assert_ne!(cone, behind);
        // Above his back, in the air vs ground pounding.
        let air = key("BossWhompFast", "{}", &s([0.0, 3784.0, 600.0], 0x0300_0880));
        let gp = key("BossWhompFast", "{}", &s([0.0, 3784.0, 600.0], ACT_GROUND_POUND));
        assert_ne!(air, gp);
        // Boss action changes the cell.
        let mut flat = k.clone();
        flat.action = 6;
        let mut f = s([0.0, 3584.0, 200.0], 0x0400_0440);
        f.objects = Some(std::sync::Arc::new(vec![flat]));
        assert_ne!(key("BossWhompFast", "{}", &f), cone);
        // Far away: silent.
        assert!(key("BossWhompFast", "{}", &s([0.0, 3584.0, 5000.0], 0x0400_0440)).is_empty());
    }

    #[test]
    fn wf_techniques_gate_and_phases() {
        // Top kick: gated by star 2, height, area.
        let p = [900.0, 4700.0, -200.0];
        assert!(!key("WfTopKick", "{}", &at(p, 0x0300_0880)).is_empty());
        let mut got = at(p, 0x0300_0880);
        got.course_stars[1] = 1 << 1;
        assert!(key("WfTopKick", "{}", &got).is_empty());
        assert!(key("WfTopKick", "{}", &at([900.0, 1000.0, -200.0], 0)).is_empty());
        let mut area2 = at(p, 0);
        area2.area = 2;
        assert!(key("WfTopKick", "{}", &area2).is_empty());
        let kick = key("WfTopKick", "{}", &at([569.0, 4715.0, -227.0], ACT_WALL_KICK_AIR));
        let plain = key("WfTopKick", "{}", &at([569.0, 4715.0, -227.0], 0x0300_0880));
        assert_ne!(kick, plain);
        let ring = key("WfTopKick", "{}", &at([378.0, 5380.0, -334.0], 0x0C40_0201));
        assert_eq!(ring.iter().find(|e| e.0 == 1).unwrap().1, 3);

        // Cless: walkway vs below the edge vs ledge grab by the star.
        let walk = key("WfCless", "{}", &at([642.0, 2560.0, 2723.0], 0x0400_0440));
        let fall = key("WfCless", "{}", &at([642.0, 2300.0, 2723.0], 0x0100_088C));
        let grab = key("WfCless", "{}", &at([628.0, 2319.0, 2724.0], ACT_LEDGE_GRAB));
        let ph = |k: &Vec<(u8, i32)>| k.iter().find(|e| e.0 == 1).unwrap().1;
        assert_eq!((ph(&walk), ph(&fall), ph(&grab)), (0, 1, 2));
        assert!(key("WfCless", "{}", &at([5000.0, 2560.0, 2723.0], 0)).is_empty());
        let mut dead = at([642.0, 2560.0, 2723.0], 0);
        dead.health = 0xFF;
        assert!(key("WfCless", "{}", &dead).is_empty());

        // Fortress climb: distance buckets and signature moves; top flag.
        let a = key("WfFortressClimb", "{}", &at([2600.0, 256.0, 5120.0], 0x0400_0440));
        let b = key("WfFortressClimb", "{}", &at([2600.0, 256.0, 5120.0], ACT_LONG_JUMP));
        assert!(!a.is_empty() && a != b);
        let top = key("WfFortressClimb", "{}", &at([500.0, 3584.0, 500.0], 0x0C40_0201));
        assert_eq!(top.iter().find(|e| e.0 == 4).unwrap().1, 1);
        let mut won = at([500.0, 3584.0, 500.0], 0);
        won.course_stars[1] = 1;
        assert!(key("WfFortressClimb", "{}", &won).is_empty());
    }
}
