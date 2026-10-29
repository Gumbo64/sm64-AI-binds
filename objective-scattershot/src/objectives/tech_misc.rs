//! Techniques: the rest, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: src/extra/tech/jrbcless.rs, jrbcaveclip.rs, islandhop.rs, owlless.rs, poleredirect.rs, openers.rs, travel.rs, ccmslideskip.rs, lakituskip.rs, mod.rs, src/extra/mod.rs
//!
//! # How a technique becomes a key
//! An rrt technique is a goal (reach the platform, keep the speed through a
//! pole) with a progress measure. Here only the state distinctions it cares
//! about survive: *where* (a coarse `pos` grid, `grid_log2`), the few
//! quantities the trick needs (a speed bucket, an action class, facing
//! relative to the way the trick goes, vertical speed in the air), gated to
//! where the rrt objective's `base_criteria` holds (nothing is written
//! elsewhere), plus a `done` tag: the rrt `success` states (on the platform,
//! in the cave, in the cage) form one cell each, so the payoff is never
//! merged with the states that did not make it. All distances are straight
//! line. The places and constants are the rrt ones (each kind says which
//! file); every one can be overridden in the schema.
//!
//! # Mapping of every rrt kind in the sources
//! | rrt kind (file) | scattershot kind |
//! |---|---|
//! | `Jrb::PillarCannonless` (jrbcless.rs) | [`JrbPillarCannonless`] |
//! | `Jrb::PillarHyper` (jrbcless.rs) | [`JrbPillarHyper`] (the BLJ metric and the dock rule overlay are not ported: progress/pad machinery; the BLJ itself is `tech_blj`'s) |
//! | `Jrb::CaveClip` (jrbcaveclip.rs, +`jrbcavewide` as `wide`) | [`JrbCaveClip`] |
//! | `IslandHop::Jump` (islandhop.rs) | [`IslandHopJump`] |
//! | `IslandHop::Star` (islandhop.rs) | [`IslandHopStar`] |
//! | `Owlless::Cage` (owlless.rs) | [`OwllessCage`] |
//! | `Tech::PoleRedirect` (poleredirect.rs; plain and aimed) | [`TechPoleRedirect`] (`aim` optional; the aim list from hyperclimb's `PLACES` is not ported: give `aim` in the schema) |
//! | `Opener::Wf`, `Opener::Bob` (openers.rs) | [`Opener`] as kinds `OpenerWf`, `OpenerBob` (the 180/240 frame deadlines are not keyed: a faster state at the same cell replaces a slower one, which is the deadline) |
//! | `Tech::DiveRollout` (travel.rs; Plain, Hop, High) | [`TechDash`] as kind `TechDiveRollout` (Hop and High are the same kind: they differ by the action and vertical speed the key already has) |
//! | `Tech::LongJumpChain` (travel.rs) | [`TechDash`] as kind `TechLongJumpChain` |
//! | `Tech::DiveRollout` Boost variant, `Tech::SlideKick`, `Tech::JumpKick` (travel.rs hands them to kicks.rs) | not ported here: `tech_kick` (kicks.rs) |
//! | aimed travel (`generate_aim`, `aim_goals`, `aim_progress`; `OBRRT_TRAVEL_AIM`) | `TechDash`'s optional `toward` (a point: yaw relative to it and a distance bucket); choosing route goals is search machinery |
//! | `Ccm::SlideSkip` (ccmslideskip.rs) | [`CcmSlideSkip`] |
//! | `Tech::LakituSkip`, both modes (lakituskip.rs) | not ported: already `Lens` (see below) |
//! | `TECHNIQUES`, `Tag`, `parse`, `configure`, `enabled`, `OBRRT_TECH` (mod.rs) | not ported: an on/off switch for the search's objective list; a scattershot schema is static, a technique is on when its node is in the schema. The table is kept as data in [`TECHNIQUES`] (name, kind, tags, course) for tools |
//! | `generate`/`generate_level` (extra/mod.rs, tech/mod.rs) | not ported: they instantiate objectives per object or per level from the start state. Per-object choices became per-behaviour (box, star, ship, pole) lookups in `eval` |
//! | `wftop`, `wfcless`, `wallkick`, ... other registry entries | other families (`tech_kick`, `tech_blj`) |
//!
//! Not ported anywhere in this file, because they are search machinery and
//! not state distinctions: `precision`, `budget`, `obj_weight`, `kinship`,
//! `steer_target`/`steer_button` (steering hints), `pad_overlay` (the dock
//! rule, the pole stick/Z/C-up rule), `success_heuristic`, `exists_in`, and
//! `Progress` (the best-so-far scores, such as the "kicked" bonus: the
//! wall-kick action classes in the key keep those states apart instead).
//!
//! # `Tech::LakituSkip` is a `Lens`
//! `schemas/zero_star.json` node `LakituSkip` is a [`crate::objectives::lens::Lens`]:
//! ```json
//! {"kind": "Lens", "name": "LakituSkip", "params": {
//!   "min": [-1400, 500, -2300], "max": [1400, 1600, 1500],
//!   "actions": ["walking", "long_jump", "slide_kick", "dive"],
//!   "yaw_toward": [0, -1], "speed_step": 0, "vspeed_step": 8,
//!   "while_object": "bhvCameraLakitu", "while_param2": 1,
//!   "caught_actions": ["dialog"], "caught_dialog": 34}}
//! ```
//! `min`/`max` is the trigger box (-544..545, z -2000..-177, y > 800) widened for slack and
//! the run-up; `while_object` + `while_param2` is the `lakituskip.rs` generate-per-object
//! condition (the intro Lakitu exists only before the castle is first entered);
//! `caught_*` is "caught" (dialog 34, `ACT_WAITING_FOR_DIALOG`/`ACT_READING_NPC_DIALOG`),
//! one cell; `actions`, `yaw_toward` (the castle, -z) and `vspeed_step` are the air and edge
//! mode's long jump / dive / jump arc. The two rrt modes (`Edge`, `Air`) differ only in
//! how they steer, not in the states, so one lens covers both.
//!
//! # Example schema fragment
//! ```json
//! {"kind": "Stage", "name": "JRB", "params": {"level": 12, "area": 1}, "children": [
//!   {"kind": "JrbCaveClip", "params": {"wide": true}},
//!   {"kind": "JrbPillarCannonless", "params": {"radius": 3000}},
//!   {"kind": "OwllessCage"}]}
//! ```

use std::f32::consts::TAU;

use anyhow::Result;
use obrrt_core::layout::{
    ACT_FLAG_AIR, ACT_FLAG_HANGING, ACT_FLAG_MOVING, ACT_FLAG_ON_POLE, ACT_FLAG_SWIMMING, ACT_GROUP_MASK, ACT_IN_CANNON,
    ACT_RIDING_HOOT, ACT_SHOT_FROM_CANNON, INTERACT_POLE, LEVEL_BOB, LEVEL_CCM, LEVEL_WF,
};
use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist, pack_pos, yaw_bucket};
use crate::objective::{Cx, Objective};

/// The techniques of this family, as the rrt registry lists them
/// (extra/tech/mod.rs `TECHNIQUES`): (rrt name, scattershot kind, tags, course in
/// gLevelToCourseNumTable numbering, if any).
pub const TECHNIQUES: &[(&str, &str, &[&str], Option<i16>)] = &[
    ("ccmslideskip", "CcmSlideSkip", &["star"], Some(4)),
    ("islandhop", "IslandHopJump", &["star"], Some(1)),
    ("islandhop", "IslandHopStar", &["star"], Some(1)),
    ("owlless", "OwllessCage", &["star", "location"], Some(2)),
    ("jrbcless", "JrbPillarCannonless", &["star"], Some(3)),
    ("jrbcaveclip", "JrbCaveClip", &["location"], Some(3)),
    ("jrbclesshyper", "JrbPillarHyper", &["star"], Some(3)),
    ("diverollout", "TechDiveRollout", &["movement"], None),
    ("ljchain", "TechLongJumpChain", &["movement"], None),
    ("poleredirect", "TechPoleRedirect", &["movement"], None),
    ("wfopener", "OpenerWf", &["location"], Some(2)),
    ("bobopener", "OpenerBob", &["location"], Some(1)),
];

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    fn p<T: for<'de> Deserialize<'de> + Objective + 'static>(v: &Value) -> Result<Box<dyn Objective>> {
        Ok(Box::new(serde_json::from_value::<T>(v.clone())?))
    }
    Some(match kind {
        "JrbPillarCannonless" => p::<JrbPillarCannonless>(params),
        "JrbPillarHyper" => p::<JrbPillarHyper>(params),
        "JrbCaveClip" => p::<JrbCaveClip>(params),
        "IslandHopJump" => p::<IslandHopJump>(params),
        "IslandHopStar" => p::<IslandHopStar>(params),
        "OwllessCage" => p::<OwllessCage>(params),
        "TechPoleRedirect" => p::<TechPoleRedirect>(params),
        "CcmSlideSkip" => p::<CcmSlideSkip>(params),
        "OpenerWf" => serde_json::from_value::<OpenerParams>(params.clone())
            .map(|o| Box::new(Opener::new(Which::Wf, o)) as Box<dyn Objective>)
            .map_err(Into::into),
        "OpenerBob" => serde_json::from_value::<OpenerParams>(params.clone())
            .map(|o| Box::new(Opener::new(Which::Bob, o)) as Box<dyn Objective>)
            .map_err(Into::into),
        "TechDiveRollout" => serde_json::from_value::<DashParams>(params.clone())
            .map(|o| Box::new(TechDash::new(Move::DiveRollout, o)) as Box<dyn Objective>)
            .map_err(Into::into),
        "TechLongJumpChain" => serde_json::from_value::<DashParams>(params.clone())
            .map(|o| Box::new(TechDash::new(Move::LongJumpChain, o)) as Box<dyn Objective>)
            .map_err(Into::into),
        _ => return None,
    })
}

// ---------------------------------------------------------------- helpers

/// Action ids (the low 9 bits, as in sm64.h) the classes below are made of.
mod act {
    pub const WALKING: u32 = 0x040;
    pub const BRAKING: u32 = 0x045;
    pub const DECELERATING: u32 = 0x04A;
    pub const BUTT_SLIDE: u32 = 0x052;
    pub const DIVE_SLIDE: u32 = 0x056;
    pub const MOVE_PUNCHING: u32 = 0x057;
    pub const CROUCH_SLIDE: u32 = 0x059;
    pub const LONG_JUMP_LAND: u32 = 0x079;
    pub const JUMP: u32 = 0x080;
    pub const DOUBLE_JUMP: u32 = 0x081;
    pub const WALL_KICK_AIR: u32 = 0x086;
    pub const LONG_JUMP: u32 = 0x088;
    pub const DIVE: u32 = 0x08A;
    pub const FREEFALL: u32 = 0x08C;
    pub const FORWARD_ROLLOUT: u32 = 0x0A6;
    pub const AIR_HIT_WALL: u32 = 0x0A7;
    pub const JUMP_KICK: u32 = 0x0AC;
    pub const GRABBED: u32 = 0x170;
    pub const LEDGE_GRAB: u32 = 0x14B;
}

/// 1 + the index of the first row of `table` holding Mario's action id, else 0.
fn class(action: u32, table: &[&[u32]]) -> i32 {
    let id = action & 0x1FF;
    table.iter().position(|r| r.contains(&id)).map_or(0, |i| i as i32 + 1)
}

fn air(st: &St) -> bool {
    st.action & ACT_FLAG_AIR != 0
}

fn swimming(st: &St) -> bool {
    st.action & ACT_FLAG_SWIMMING != 0
}

/// Horizontal speed of the velocity vector (not `forward_vel`).
fn hs(st: &St) -> f32 {
    st.vel[0].hypot(st.vel[2])
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).hypot(a[2] - b[2])
}

fn in_cannon(st: &St) -> bool {
    st.action == ACT_IN_CANNON || st.action == ACT_SHOT_FROM_CANNON
}

/// sm64 yaw (0 faces +z, grows towards +x) of the direction from `from` to `to`.
fn bearing(from: [f32; 3], to: [f32; 3]) -> i16 {
    ((to[0] - from[0]).atan2(to[2] - from[2]) / TAU * 65536.0).round() as i32 as i16
}

/// Mario's facing relative to `target`: 7 fine buckets within 45 degrees, 3 coarse for the rest.
fn yaw_to(st: &St, target: i16) -> i32 {
    yaw_bucket(st.yaw, target, 0x2000, 7, 3)
}

fn objs(st: &St) -> &[Obj] {
    st.objects()
}

/// An object of behaviour `name` whose behaviour-parameter byte `shift` (24 or 16) equals `v`.
fn find_param<'a>(st: &'a St, name: &str, shift: u32, v: u32) -> Option<&'a Obj> {
    let b = bhv_addr(name)?;
    objs(st).iter().find(|o| o.bhv == b && (o.beh_params >> shift) & 0xFF == v)
}

fn has_star(st: &St, course: i16, index: u8) -> bool {
    (1..=25).contains(&course) && st.course_stars[course as usize - 1] & (1 << index) != 0
}

fn g6() -> u32 {
    6
}
fn g7() -> u32 {
    7
}
fn g8() -> u32 {
    8
}
fn g9() -> u32 {
    9
}
fn speed8() -> f32 {
    8.0
}
fn vspeed16() -> f32 {
    16.0
}
fn d400() -> f32 {
    400.0
}

// ---------------------------------------------------------------- JRB star 5 and the cave

const JRB_COURSE: i16 = 3;
const JRB_AREA: u8 = 1;
const JRB_STAR: u8 = 4;
const BOX_ABOVE_FLOOR: f32 = 214.0;
const PLATFORM_REACH: f32 = 1100.0;

/// The star's ! box in JRB (`bhvExclamationBox`, behaviour byte 0 = 4), if the
/// state is in JRB area 1, star 5 is not had, Mario is alive and not in the cannon.
fn jrb_box(st: &St) -> Option<[f32; 3]> {
    if st.course != JRB_COURSE || st.area != JRB_AREA || has_star(st, JRB_COURSE, JRB_STAR) || !st.alive() || in_cannon(st) {
        return None;
    }
    find_param(st, "bhvExclamationBox", 24, JRB_STAR as u32).map(|o| o.pos)
}

/// jrbcless.rs `on_platform`: standing on the stone platform that the box floats over.
fn on_platform(st: &St, boxpos: [f32; 3]) -> bool {
    let floor = boxpos[1] - BOX_ABOVE_FLOOR;
    st.alive()
        && st.action & (ACT_FLAG_AIR | ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE) == 0
        && st.pos[1] >= floor - 10.0
        && st.pos[1] <= floor + 400.0
        && hdist(st.pos, boxpos) < PLATFORM_REACH
}

fn jrb_ledge() -> [f32; 3] {
    [777.0, 1946.0, 2380.0]
}
fn jrb_spire() -> [f32; 3] {
    [127.0, 1100.0, 2779.0]
}
fn offer_radius() -> f32 {
    4000.0
}
fn launch_radius() -> f32 {
    300.0
}

/// JRB star 5 (the stone pillar) without the cannon (`Jrb::PillarCannonless`,
/// jrbcless.rs). Applies in JRB area 1 while the star is not had, its ! box is
/// loaded, Mario is alive, not in the cannon and within `radius` of the
/// platform's `ledge`. Fields: `pos` (grid); `action` (1 swimming, 2 wall
/// kick in the air, 3 other air, 4 ledge grab, 0 on foot); `launch` (1 at the
/// stone `spire`, 2 at the sunken ship's stern `bhvInSunkenShip3`, within
/// `launch_radius`, 0 elsewhere: where the climb starts); `done` (1 on the
/// platform: the success cell; nothing else is written then).
///
/// `{"kind": "JrbPillarCannonless", "params": {"radius": 3000}}`
#[derive(Deserialize)]
pub struct JrbPillarCannonless {
    #[serde(default = "jrb_ledge")]
    pub ledge: [f32; 3],
    #[serde(default = "jrb_spire")]
    pub spire: [f32; 3],
    #[serde(default = "offer_radius")]
    pub radius: f32,
    #[serde(default = "launch_radius")]
    pub launch_radius: f32,
    #[serde(default = "g7")]
    pub grid_log2: u32,
}

/// jrbcless.rs `SHIP_STERN`: (height above, z offset) from the ship's origin.
const SHIP_STERN: [f32; 2] = [1536.0, -1840.0];

impl Objective for JrbPillarCannonless {
    fn fields(&self) -> &[&'static str] {
        &["pos", "action", "launch", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(boxpos) = jrb_box(st) else { return };
        if on_platform(st, boxpos) {
            cx.put(3, 1);
            return;
        }
        if st.dist(self.ledge) >= self.radius {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        let a = if swimming(st) {
            1
        } else if st.action & 0x1FF == act::WALL_KICK_AIR {
            2
        } else if air(st) {
            3
        } else if st.action & 0x1FF == act::LEDGE_GRAB {
            4
        } else {
            0
        };
        cx.put(1, a);
        let mut launch = (st.dist(self.spire) < self.launch_radius) as i32;
        if let Some(ship) = bhv_addr("bhvInSunkenShip3").and_then(|b| objs(st).iter().find(|o| o.bhv == b)) {
            let stern = [ship.pos[0], ship.pos[1] + SHIP_STERN[0], ship.pos[2] + SHIP_STERN[1]];
            if st.dist(stern) < self.launch_radius {
                launch = 2;
            }
        }
        cx.put(2, launch);
    }
}

fn hyper() -> f32 {
    150.0
}
fn slope() -> [f32; 3] {
    [-5115.0, 947.0, 1230.0]
}
fn spire_axis() -> [f32; 3] {
    [60.0, 0.0, 2730.0]
}
fn spire_base_y() -> f32 {
    980.0
}
fn spire_radius() -> f32 {
    350.0
}
fn dock_x() -> f32 {
    -4800.0
}
fn dock_y() -> f32 {
    900.0
}
fn speed50() -> f32 {
    50.0
}

/// Wet: swimming or in the submerged action group (jrbcless.rs `wet`).
fn wet(st: &St) -> bool {
    swimming(st) || st.action & ACT_GROUP_MASK == 0x0C0
}

/// JRB star 5 the TAS way (`Jrb::PillarHyper`, jrbcless.rs): the dock-slope BLJ
/// under the overhang, then the backwards-hyperspeed dive and rollout, a braking
/// run up the spire and a steep jump onto the platform. Applies in the same
/// place as [`JrbPillarCannonless`] and, as the rrt `base_criteria`, only
/// on the dock (x < `dock_x`, y > `dock_y`, dry, |forward speed| < `hyper`,
/// within 2500 of the dock `slope`), in backwards hyperspeed (forward speed
/// < -`hyper`, dry, y > `dock_y`, west of the spire), or on the spire
/// (within `spire_radius` of the axis, above the base) at < -`hyper`/2.
/// Fields: `phase` (1 dock, 2 fast, 3 spire); `pos` (grid); `hspeed`
/// (forward speed in `speed_step`s, so hyperspeed has a few cells); `action`
/// (1 walking/braking/decelerating, 2 long jump, 3 long-jump landing or
/// crouch slide, 4 freefall, 5 dive, 6 butt slide, 0 other: the BLJ chain and
/// the dive/rollout); `done` (1 on the platform).
///
/// `{"kind": "JrbPillarHyper", "params": {"speed_step": 50}}`
#[derive(Deserialize)]
pub struct JrbPillarHyper {
    #[serde(default = "hyper")]
    pub hyper: f32,
    #[serde(default = "slope")]
    pub slope: [f32; 3],
    #[serde(default = "spire_axis")]
    pub spire_axis: [f32; 3],
    #[serde(default = "spire_base_y")]
    pub spire_base_y: f32,
    #[serde(default = "spire_radius")]
    pub spire_radius: f32,
    #[serde(default = "dock_x")]
    pub dock_x: f32,
    #[serde(default = "dock_y")]
    pub dock_y: f32,
    #[serde(default = "speed50")]
    pub speed_step: f32,
    #[serde(default = "g7")]
    pub grid_log2: u32,
}

impl Objective for JrbPillarHyper {
    fn fields(&self) -> &[&'static str] {
        &["phase", "pos", "hspeed", "action", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(boxpos) = jrb_box(st) else { return };
        if on_platform(st, boxpos) {
            cx.put(4, 1);
            return;
        }
        let dry = !wet(st);
        let on_dock = st.pos[0] < self.dock_x
            && st.pos[1] > self.dock_y
            && dry
            && st.forward_vel.abs() < 100.0
            && st.hdist(self.slope) < 2500.0;
        let fast = st.forward_vel < -self.hyper && dry && st.pos[1] > self.dock_y && st.pos[0] < self.spire_axis[0] + 600.0;
        let on_spire = dry && st.pos[1] > self.spire_base_y - 30.0 && hdist(st.pos, self.spire_axis) < self.spire_radius;
        let phase = if on_dock {
            1
        } else if fast {
            2
        } else if on_spire && st.forward_vel < -self.hyper / 2.0 {
            3
        } else {
            return;
        };
        cx.put(0, phase);
        cx.put(1, pack_pos(st.pos, self.grid_log2));
        cx.put(2, bucket(st.forward_vel, self.speed_step));
        let table: &[&[u32]] = &[
            &[act::WALKING, act::BRAKING, act::DECELERATING],
            &[act::LONG_JUMP],
            &[act::LONG_JUMP_LAND, act::CROUCH_SLIDE],
            &[act::FREEFALL],
            &[act::DIVE, act::DIVE_SLIDE],
            &[act::BUTT_SLIDE],
        ];
        cx.put(3, class(st.action, table));
    }
}

fn bay_z() -> f32 {
    -669.0
}
fn cave_z() -> f32 {
    -907.0
}
fn cave_ceil() -> f32 {
    -918.0
}
fn mouth_y() -> f32 {
    -3850.0
}
fn entry() -> [f32; 3] {
    [5214.0, -500.0, -300.0]
}
fn submerged_hs() -> f32 {
    26.0
}
fn cave_radius() -> f32 {
    3000.0
}
fn speed10() -> f32 {
    10.0
}
fn pitch_step() -> f32 {
    4096.0
}

/// Swim-clipping into the JRB cave (`Jrb::CaveClip`, jrbcaveclip.rs): full-speed
/// swimming into the steep faces above the tunnel's mouth, which clips into
/// the rock and downwarps into the cave. Applies in JRB area 1 while swimming
/// in the bay (z > `bay_z`), above the mouth (y > `mouth_y`) and within
/// `radius` of the tunnel `entry`; with `wide` (the rrt `jrbcavewide` mode) any
/// submerged-group action at horizontal speed > `submerged_hs` counts as in
/// the bay. Fields: `pos` (grid); `hspeed` (velocity's horizontal speed in
/// `speed_step`s: the clip needs ~full speed); `yaw` (relative to straight -z,
/// at the cave, 7 fine buckets + 3 coarse); `pitch` (nose up/down, `pitch_step`
/// units: the swim has to be aimed at the faces' height); `action` (1 swimming,
/// 2 other submerged); `done` (1 in the cave: z < `cave_z`, y < `cave_ceil`; the success cell).
/// Falling below the mouth in the bay (the failed clip) writes nothing.
///
/// `{"kind": "JrbCaveClip", "params": {"wide": true}}`
#[derive(Deserialize)]
pub struct JrbCaveClip {
    #[serde(default)]
    pub wide: bool,
    #[serde(default = "bay_z")]
    pub bay_z: f32,
    #[serde(default = "cave_z")]
    pub cave_z: f32,
    #[serde(default = "cave_ceil")]
    pub cave_ceil: f32,
    #[serde(default = "mouth_y")]
    pub mouth_y: f32,
    #[serde(default = "entry")]
    pub entry: [f32; 3],
    #[serde(default = "submerged_hs")]
    pub submerged_hs: f32,
    #[serde(default = "cave_radius")]
    pub radius: f32,
    #[serde(default = "speed10")]
    pub speed_step: f32,
    #[serde(default = "pitch_step")]
    pub pitch_step: f32,
    #[serde(default = "g7")]
    pub grid_log2: u32,
}

impl Objective for JrbCaveClip {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "yaw", "pitch", "action", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.course != JRB_COURSE || st.area != JRB_AREA || !st.alive() {
            return;
        }
        if st.pos[2] < self.cave_z && st.pos[1] < self.cave_ceil {
            cx.put(5, 1);
            return;
        }
        let submerged = st.action & ACT_GROUP_MASK == 0x0C0;
        let in_bay = st.pos[2] > self.bay_z && (swimming(st) || (self.wide && submerged && hs(st) > self.submerged_hs));
        if !in_bay || st.pos[1] <= self.mouth_y || st.dist(self.entry) >= self.radius {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, bucket(hs(st), self.speed_step));
        cx.put(2, yaw_to(st, bearing([0.0; 3], [0.0, 0.0, -1.0])));
        cx.put(3, bucket(st.pitch as f32, self.pitch_step));
        cx.put(4, if swimming(st) { 1 } else { 2 });
    }
}

// ---------------------------------------------------------------- BoB star 3, the floating island

const BOB_COURSE: i16 = 1;
const BOB_STAR: u8 = 2;
const BOX_CONTENTS: u32 = 11;
const MARIO_WING_CAP: u32 = 0x08;

fn island_ledge() -> [f32; 3] {
    [3341.0, 3007.0, -1564.0]
}
fn island_summit() -> [f32; 3] {
    [3072.0, 4097.0, -2559.0]
}
fn island_landing() -> [f32; 3] {
    [3700.0, 3072.0, 150.0]
}
fn island_box() -> [f32; 3] {
    [5540.0, 3350.0, 1200.0]
}
fn island_x() -> [f32; 2] {
    [2800.0, 5900.0]
}
fn island_z() -> [f32; 2] {
    [-120.0, 2580.0]
}
fn island_top() -> f32 {
    3072.0
}

/// islandhop.rs `on_island`: standing on the island or hanging on its ledge.
fn on_island(st: &St, x: [f32; 2], z: [f32; 2], top: f32) -> bool {
    let [px, py, pz] = st.pos;
    let inside = (x[0]..=x[1]).contains(&px) && (z[0]..=z[1]).contains(&pz);
    inside && ((!air(st) && py >= top - 20.0) || (st.action & 0x1FF == act::LEDGE_GRAB && py >= top - 200.0))
}

/// BoB star 3 without cannon or wing cap, the jump onto the floating island
/// (`IslandHop::Jump`, islandhop.rs). Applies in BoB area 1, star 3 not had,
/// in control and alive, no wing cap, not in the cannon or grabbed, and
/// within a launch region: `ledge` (the TAS ledge, from y `ledge_y`, within
/// `ledge_radius` horizontally) or `summit` (the summit corner, from y
/// `summit_y`, within `summit_radius`). Fields: `launch` (1 ledge, 2 summit);
/// `pos` (grid); `hspeed` (forward speed per `speed_step`); `action` (1 walking,
/// 2 crouch slide, 3 long jump, 4 dive, 5 other air, 0 other); `yaw` (relative to
/// the `landing` spot on the island); `vspeed` (while airborne, `vspeed_step`);
/// `done` (1 on the island: the success cell, written anywhere in the zone).
///
/// `{"kind": "IslandHopJump", "params": {"summit_radius": 2500}}`
#[derive(Deserialize)]
pub struct IslandHopJump {
    #[serde(default = "island_ledge")]
    pub ledge: [f32; 3],
    #[serde(default = "island_summit")]
    pub summit: [f32; 3],
    #[serde(default = "island_landing")]
    pub landing: [f32; 3],
    #[serde(default = "ledge_y")]
    pub ledge_y: f32,
    #[serde(default = "ledge_radius")]
    pub ledge_radius: f32,
    #[serde(default = "summit_y")]
    pub summit_y: f32,
    #[serde(default = "summit_radius")]
    pub summit_radius: f32,
    #[serde(default = "island_x")]
    pub island_x: [f32; 2],
    #[serde(default = "island_z")]
    pub island_z: [f32; 2],
    #[serde(default = "island_top")]
    pub island_top: f32,
    #[serde(default = "speed8")]
    pub speed_step: f32,
    #[serde(default = "vspeed16")]
    pub vspeed_step: f32,
    #[serde(default = "g7")]
    pub grid_log2: u32,
}

fn ledge_y() -> f32 {
    2850.0
}
fn ledge_radius() -> f32 {
    2000.0
}
fn summit_y() -> f32 {
    3900.0
}
fn summit_radius() -> f32 {
    3500.0
}

/// The BoB gate shared by both island kinds: area 1, star not had, alive, in control.
fn bob_gate(st: &St) -> bool {
    st.zone() == (LEVEL_BOB as u16, 1) && st.alive() && st.controllable() && !has_star(st, BOB_COURSE, BOB_STAR)
}

impl Objective for IslandHopJump {
    fn fields(&self) -> &[&'static str] {
        &["launch", "pos", "hspeed", "action", "yaw", "vspeed", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !bob_gate(st) || st.mario_flags & MARIO_WING_CAP != 0 || in_cannon(st) || st.action & 0x1FF == act::GRABBED {
            return;
        }
        if on_island(st, self.island_x, self.island_z, self.island_top) {
            cx.put(6, 1);
            return;
        }
        let launch = if st.pos[1] >= self.summit_y && hdist(st.pos, self.summit) < self.summit_radius {
            2
        } else if st.pos[1] >= self.ledge_y && hdist(st.pos, self.ledge) < self.ledge_radius {
            1
        } else {
            return;
        };
        cx.put(0, launch);
        cx.put(1, pack_pos(st.pos, self.grid_log2));
        cx.put(2, bucket(st.forward_vel, self.speed_step));
        let a = class(st.action, &[&[act::WALKING], &[act::CROUCH_SLIDE], &[act::LONG_JUMP], &[act::DIVE, act::DIVE_SLIDE]]);
        cx.put(3, if a == 0 && air(st) { 5 } else { a });
        cx.put(4, yaw_to(st, bearing(st.pos, self.landing)));
        if air(st) && self.vspeed_step > 0.0 {
            cx.put(5, bucket(st.vel[1], self.vspeed_step));
        }
    }
}

/// From the island, the star (`IslandHop::Star`, islandhop.rs). Applies in BoB
/// area 1, star 3 not had, in control and alive, once Mario is on the island
/// (same box as [`IslandHopJump`]) or the star has spawned (`bhvSpawnedStar`, star
/// index 2). Fields: `pos` (grid); `dist` (straight-line distance to the star, else
/// to the loaded ! box with contents 11, else to `box`, in `step`s); `spawned`
/// (1 once the star is out); `action` (0 on foot, 1 airborne).
///
/// `{"kind": "IslandHopStar", "params": {"step": 200}}`
#[derive(Deserialize)]
pub struct IslandHopStar {
    #[serde(default = "island_box", rename = "box")]
    pub box_pos: [f32; 3],
    #[serde(default = "island_x")]
    pub island_x: [f32; 2],
    #[serde(default = "island_z")]
    pub island_z: [f32; 2],
    #[serde(default = "island_top")]
    pub island_top: f32,
    #[serde(default = "d400")]
    pub step: f32,
    #[serde(default = "g6")]
    pub grid_log2: u32,
}

impl Objective for IslandHopStar {
    fn fields(&self) -> &[&'static str] {
        &["pos", "dist", "spawned", "action"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !bob_gate(st) {
            return;
        }
        let star = find_param(st, "bhvSpawnedStar", 24, BOB_STAR as u32).map(|o| o.home);
        if star.is_none() && !on_island(st, self.island_x, self.island_z, self.island_top) {
            return;
        }
        let target = star
            .or_else(|| find_param(st, "bhvExclamationBox", 16, BOX_CONTENTS).map(|o| o.pos))
            .unwrap_or(self.box_pos);
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        cx.put(1, bucket(st.dist(target), self.step));
        cx.put(2, star.is_some() as i32);
        cx.put(3, air(st) as i32);
    }
}

// ---------------------------------------------------------------- WF star 5 without Hoot

const WF_COURSE: i16 = 2;
const WF_STAR: u8 = 4;

fn owl_star() -> [f32; 3] {
    [2880.0, 4300.0, 190.0]
}

/// WF star 5 without Hoot, into the cage (`Owlless::Cage`, owlless.rs). Applies
/// in WF area 1, star 5 not had, in control and alive, not riding Hoot, from
/// y `start_y` and within `start_radius` of the star (`bhvStar` with star index 4,
/// else `star`), and not yet in the cage (within `cage_radius` of the star at
/// floor height or higher: `done`, the success cell) nor below it
/// (within `under_radius`, below the floor: the failed side, nothing written).
/// Fields: `pos` (grid); `action` (1 jump or double jump, 2 wall kick in the
/// air, 3 hitting a wall in the air, 4 freefall, 5 other air, 0 on foot);
/// `hspeed` (forward speed, `speed_step`); `vspeed` (in the air, `vspeed_step`); `done`.
///
/// `{"kind": "OwllessCage", "params": {"start_radius": 1600}}`
#[derive(Deserialize)]
pub struct OwllessCage {
    #[serde(default = "owl_star")]
    pub star: [f32; 3],
    #[serde(default = "floor_4096")]
    pub floor: f32,
    #[serde(default = "cage_radius")]
    pub cage_radius: f32,
    #[serde(default = "start_y")]
    pub start_y: f32,
    #[serde(default = "start_radius")]
    pub start_radius: f32,
    #[serde(default = "under_radius")]
    pub under_radius: f32,
    #[serde(default = "speed8")]
    pub speed_step: f32,
    #[serde(default = "vspeed16")]
    pub vspeed_step: f32,
    #[serde(default = "g6")]
    pub grid_log2: u32,
}

fn floor_4096() -> f32 {
    4096.0
}
fn cage_radius() -> f32 {
    260.0
}
fn start_y() -> f32 {
    3300.0
}
fn start_radius() -> f32 {
    1600.0
}
fn under_radius() -> f32 {
    450.0
}

impl Objective for OwllessCage {
    fn fields(&self) -> &[&'static str] {
        &["pos", "action", "hspeed", "vspeed", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.zone() != (LEVEL_WF as u16, 1)
            || !st.alive()
            || !st.controllable()
            || has_star(st, WF_COURSE, WF_STAR)
            || st.action == ACT_RIDING_HOOT
        {
            return;
        }
        let star = find_param(st, "bhvStar", 24, WF_STAR as u32).map_or(self.star, |o| o.pos);
        let h = hdist(st.pos, star);
        if h < self.cage_radius && st.pos[1] >= self.floor - 10.0 {
            cx.put(4, 1);
            return;
        }
        if h < self.under_radius && st.pos[1] < self.floor - 10.0 {
            return;
        }
        if st.pos[1] < self.start_y || h >= self.start_radius {
            return;
        }
        cx.put(0, pack_pos(st.pos, self.grid_log2));
        let a = class(
            st.action,
            &[&[act::JUMP, act::DOUBLE_JUMP], &[act::WALL_KICK_AIR], &[act::AIR_HIT_WALL], &[act::FREEFALL]],
        );
        cx.put(1, if a == 0 && air(st) { 5 } else { a });
        cx.put(2, bucket(st.forward_vel, self.speed_step));
        if air(st) && self.vspeed_step > 0.0 {
            cx.put(3, bucket(st.vel[1], self.vspeed_step));
        }
    }
}

// ---------------------------------------------------------------- pole redirect

/// poleredirect.rs `CARRY`: the actions that keep the stored speed after the pole.
const POLE_CARRY: [u32; 6] = [act::CROUCH_SLIDE, act::BRAKING, act::JUMP, act::DOUBLE_JUMP, act::JUMP_KICK, act::MOVE_PUNCHING];

fn pole_min_speed() -> f32 {
    150.0
}
fn pole_max_dist() -> f32 {
    3000.0
}
fn carry_radius() -> f32 {
    1600.0
}
fn d100() -> f32 {
    100.0
}

/// Hyperspeed through a tree or pole (`Tech::PoleRedirect`, poleredirect.rs):
/// at |velocity| >= `min_speed` into a pole (`INTERACT_POLE` object) low on its
/// trunk, the speed stays in the velocity while Mario holds it; on the idle
/// frame stick + Z + C-up crouch-slides along the stick at that speed; success
/// is forward speed >= 0.8 of the stored one soon after, in a carrying action.
/// Applies when |velocity| >= `min_speed`, Mario is alive, and for the nearest pole
/// one of the phases holds: 1 approaching (a jump in the air, within `max_dist`
/// of it, within 600 of its height + 150, velocity within ~25 degrees and 150 of
/// the pole), 2 just touched it (forward speed ~0, within 60, not yet on it),
/// 3 holding it, 4 carrying after it (an action of `POLE_CARRY`: crouch slide,
/// braking, jump, double jump, jump kick, punching; forward speed >= 0.8
/// `min_speed`; within `carry_radius` of the nearest pole). Fields: `phase`;
/// `pole` (the pole's position, on a grid of `grid_log2`); `dist` (horizontal
/// distance to it in `dist_step`s); `hspeed` (|velocity| in `speed_step`s);
/// `miss` (approaching: the sideways miss in 50-unit steps, else 0); `action`
/// (the carry action 1..6 in the order above, 0 other); `aimed` (with `aim`: 1
/// when the velocity is within 30 degrees of the direction to it, else 0).
///
/// `{"kind": "TechPoleRedirect", "params": {"aim": [2000, 3500, -1000]}}`
#[derive(Deserialize)]
pub struct TechPoleRedirect {
    #[serde(default = "pole_min_speed")]
    pub min_speed: f32,
    #[serde(default = "pole_max_dist")]
    pub max_dist: f32,
    #[serde(default = "carry_radius")]
    pub carry_radius: f32,
    pub aim: Option<[f32; 3]>,
    #[serde(default = "d100")]
    pub dist_step: f32,
    #[serde(default = "speed50")]
    pub speed_step: f32,
    #[serde(default = "g8")]
    pub grid_log2: u32,
}

impl TechPoleRedirect {
    fn at_trunk(&self, st: &St, pole: [f32; 3]) -> bool {
        st.forward_vel.abs() < 0.5 && hs(st) >= self.min_speed && st.hdist(pole) < 60.0
    }
    /// Approach: (miss in units) when the velocity points at the pole (poleredirect.rs `approaching`).
    fn approaching(&self, st: &St, pole: [f32; 3]) -> Option<f32> {
        let id = st.action & 0x1FF;
        let v = hs(st);
        if !air(st) || !(0x80..0xA0).contains(&id) || v < self.min_speed {
            return None;
        }
        let (dx, dz) = (pole[0] - st.pos[0], pole[2] - st.pos[2]);
        let d = dx.hypot(dz);
        if !(1.0..=self.max_dist).contains(&d) || (st.pos[1] - (pole[1] + 150.0)).abs() > 600.0 {
            return None;
        }
        let (ux, uz) = (st.vel[0] / v, st.vel[2] / v);
        let (along, miss) = (dx * ux + dz * uz, (dx * uz - dz * ux).abs());
        (along >= 0.906 * d && miss <= 150.0).then_some(miss)
    }
}

impl Objective for TechPoleRedirect {
    fn fields(&self) -> &[&'static str] {
        &["phase", "pole", "dist", "hspeed", "miss", "action", "aimed"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !st.alive() {
            return;
        }
        let v = hs(st);
        let carrying = POLE_CARRY.contains(&(st.action & 0x1FF)) && st.forward_vel >= 0.8 * self.min_speed;
        if v < self.min_speed && !carrying {
            return;
        }
        let mut poles: Vec<[f32; 3]> = objs(st).iter().filter(|o| o.interact_type == INTERACT_POLE).map(|o| o.pos).collect();
        poles.sort_by(|a, b| st.hdist(*a).total_cmp(&st.hdist(*b)));
        let mut found = None;
        for &pole in &poles {
            let on = st.action & ACT_FLAG_ON_POLE != 0;
            let phase = if on && self.at_trunk(st, pole) {
                Some((3, 0.0))
            } else if !on && self.at_trunk(st, pole) {
                Some((2, 0.0))
            } else {
                self.approaching(st, pole).map(|m| (1, m))
            };
            if let Some((ph, miss)) = phase {
                found = Some((pole, ph, miss));
                break;
            }
        }
        if found.is_none() && carrying {
            if let Some(&pole) = poles.first().filter(|p| st.hdist(**p) < self.carry_radius) {
                found = Some((pole, 4, 0.0));
            }
        }
        let Some((pole, phase, miss)) = found else { return };
        cx.put(0, phase);
        cx.put(1, pack_pos(pole, self.grid_log2));
        cx.put(2, bucket(st.hdist(pole), self.dist_step));
        cx.put(3, bucket(v, self.speed_step));
        cx.put(4, bucket(miss, 50.0));
        cx.put(5, POLE_CARRY.iter().position(|&a| a == st.action & 0x1FF).map_or(0, |i| i as i32 + 1));
        let aimed = self.aim.is_some_and(|a| {
            let (dx, dz) = (a[0] - st.pos[0], a[2] - st.pos[2]);
            let d = dx.hypot(dz).max(1.0);
            (dx * st.vel[0] + dz * st.vel[2]) / (d * v.max(1.0)) >= 0.866
        });
        cx.put(6, aimed as i32);
    }
}

// ---------------------------------------------------------------- level openers

#[derive(Clone, Copy, PartialEq, Debug)]
enum Which {
    Wf,
    Bob,
}

/// The parameters of [`Opener`]; each unset field has the rrt value of the
/// kind (openers.rs `Which`).
#[derive(Deserialize, Default)]
pub struct OpenerParams {
    /// Where the painting drops Mario in (default WF (2600, 256, 5120), BoB (-6558, 0, 6464)).
    pub spawn: Option<[f32; 3]>,
    /// The high ground to reach (WF (3300, 1600, 2000), BoB (1600, 768, 4150)).
    pub target: Option<[f32; 3]>,
    /// Horizontal reach of the target (400).
    pub reach: Option<f32>,
    /// At the target only from this height (WF 1400, BoB none).
    pub y_min: Option<f32>,
    /// At the target only within this of its height (BoB 200, WF none).
    pub y_band: Option<f32>,
    /// A waypoint to aim for first (WF: the wall to kick at (3276, 640, 3890)), while
    /// Mario is below `way_y` and beyond `way_z` (700, 3700).
    pub waypoint: Option<[f32; 3]>,
    pub way_y: Option<f32>,
    pub way_z: Option<f32>,
    /// Active within this of the target (1.25 x the spawn-target distance).
    pub radius: Option<f32>,
    pub grid_log2: Option<u32>,
    pub speed_step: Option<f32>,
}

/// From the BoB or WF painting landing to the high ground in 6 s
/// (`Opener::Wf`, `Opener::Bob`, openers.rs; kinds `OpenerWf`, `OpenerBob`).
/// Applies in the level's area 1 while Mario is in control, alive, not
/// swimming or hanging, and within `radius` of the `target`, but not at it yet
/// (the rrt deadline is not keyed: the faster state of a cell wins, which is
/// the deadline). Fields: `pos` (grid); `hspeed` (forward speed, `speed_step`);
/// `action` (1 walking, 2 crouch slide, 3 long jump or landing, 4 dive or dive
/// slide, 5 rollout, 6 wall kick, 7 other air, 0 other); `yaw` (relative to the
/// target, or the waypoint while below it); `stage` (1 while below the waypoint's
/// wall, else 0); `done` (1 at the target: written alone).
///
/// `{"kind": "OpenerBob", "params": {"radius": 9000}}`
pub struct Opener {
    which: Which,
    p: OpenerParams,
    spawn: [f32; 3],
    target: [f32; 3],
    radius: f32,
    grid: u32,
    step: f32,
}

impl Opener {
    fn new(which: Which, p: OpenerParams) -> Opener {
        let spawn = p.spawn.unwrap_or(match which {
            Which::Wf => [2600.0, 256.0, 5120.0],
            Which::Bob => [-6558.0, 0.0, 6464.0],
        });
        let target = p.target.unwrap_or(match which {
            Which::Wf => [3300.0, 1600.0, 2000.0],
            Which::Bob => [1600.0, 768.0, 4150.0],
        });
        let radius = p.radius.unwrap_or(1.25 * dist(spawn, target));
        let (grid, step) = (p.grid_log2.unwrap_or(7), p.speed_step.unwrap_or(8.0));
        let mut p = p;
        if which == Which::Wf {
            p.waypoint = p.waypoint.or(Some([3276.0, 640.0, 3890.0]));
        }
        Opener { which, p, spawn, target, radius, grid, step }
    }

    fn at(&self, st: &St) -> bool {
        let (y_min, y_band) = match self.which {
            Which::Wf => (self.p.y_min.or(Some(1400.0)), self.p.y_band),
            Which::Bob => (self.p.y_min, self.p.y_band.or(Some(200.0))),
        };
        st.hdist(self.target) <= self.p.reach.unwrap_or(400.0)
            && st.alive()
            && y_min.is_none_or(|y| st.pos[1] >= y)
            && y_band.is_none_or(|b| (st.pos[1] - self.target[1]).abs() <= b)
    }
}

impl Objective for Opener {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "action", "yaw", "stage", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        let zone = match self.which {
            Which::Wf => (LEVEL_WF as u16, 1),
            Which::Bob => (LEVEL_BOB as u16, 1),
        };
        if st.zone() != zone {
            return;
        }
        if self.at(st) {
            cx.put(5, 1);
            return;
        }
        if !st.controllable() || !st.alive() || st.action & (ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE | ACT_FLAG_HANGING) != 0 {
            return;
        }
        if st.dist(self.target) >= self.radius {
            return;
        }
        let before_wall = self.p.waypoint.is_some()
            && st.pos[1] < self.p.way_y.unwrap_or(700.0)
            && st.pos[2] > self.p.way_z.unwrap_or(3700.0);
        let aim = if before_wall { self.p.waypoint.unwrap_or(self.target) } else { self.target };
        cx.put(0, pack_pos(st.pos, self.grid));
        cx.put(1, bucket(st.forward_vel, self.step));
        let a = class(
            st.action,
            &[
                &[act::WALKING, act::BRAKING, act::DECELERATING],
                &[act::CROUCH_SLIDE],
                &[act::LONG_JUMP, act::LONG_JUMP_LAND],
                &[act::DIVE, act::DIVE_SLIDE],
                &[act::FORWARD_ROLLOUT],
                &[act::WALL_KICK_AIR],
            ],
        );
        cx.put(2, if a == 0 && air(st) { 7 } else { a });
        cx.put(3, yaw_to(st, bearing(st.pos, aim)));
        cx.put(4, before_wall as i32);
        let _ = (self.spawn, ACT_FLAG_MOVING);
    }
}

// ---------------------------------------------------------------- travel: dive rollouts and long-jump chains

#[derive(Clone, Copy, PartialEq, Debug)]
enum Move {
    DiveRollout,
    LongJumpChain,
}

/// The parameters of [`TechDash`]; unset fields have the kind's default.
#[derive(Deserialize, Default)]
pub struct DashParams {
    /// Only at |velocity| >= this (dive rollout 24, which is the edge-hop's pace; long-jump chain 30,
    /// above running's 32 minus the first slow frames).
    pub min_speed: Option<f32>,
    /// Speed bucket (forward speed): 6.
    pub speed_step: Option<f32>,
    /// Vertical speed bucket while airborne; 0 leaves it out (dive rollout 16: the high dive keeps
    /// a rising jump's vertical speed; long-jump chain 0).
    pub vspeed_step: Option<f32>,
    /// Coarse position grid (9, 512 units).
    pub grid_log2: Option<u32>,
    /// Restricts it to a box (like [`crate::objectives::lens::Lens`]).
    pub min: Option<[f32; 3]>,
    pub max: Option<[f32; 3]>,
    /// A target point (the aimed travel of travel.rs): adds the facing relative to it and the
    /// straight-line distance in `toward_step`s.
    pub toward: Option<[f32; 3]>,
    pub toward_step: Option<f32>,
}

/// Fast travel with a technique, anywhere (`Tech::DiveRollout`,
/// `Tech::LongJumpChain`, travel.rs; kinds `TechDiveRollout`,
/// `TechLongJumpChain`). rrt asks for a point 900-2000 away reached at 30+
/// (dive rollout) or 33+ (long jumps, two or more) units a frame with the
/// trick in the segment; here what carries over is the state: Mario in
/// control (not swimming, on a pole or hanging), moving at |velocity| >=
/// `min_speed`. Fields: `pos` (coarse grid); `hspeed` (forward speed,
/// `speed_step`); `action` (dive rollout: 1 dive, 2 dive slide, 3 rollout, 4
/// freefall, 5 jump or double jump, 6 walking, 0 other; long-jump chain: 1
/// long jump, 2 long-jump landing, 3 crouch slide, 4 walking, 5 jump,
/// 0 other); `vspeed` (airborne, `vspeed_step`); with `toward`: `yaw` (relative
/// to it) and `dist`.
///
/// `{"kind": "TechLongJumpChain", "params": {"min": [-3000, 0, -3000], "max": [3000, 2000, 3000]}}`
pub struct TechDash {
    mv: Move,
    p: DashParams,
    min_speed: f32,
    step: f32,
    vstep: f32,
    grid: u32,
}

impl TechDash {
    fn new(mv: Move, p: DashParams) -> TechDash {
        let (ms, vs) = match mv {
            Move::DiveRollout => (24.0, 16.0),
            Move::LongJumpChain => (30.0, 0.0),
        };
        TechDash {
            mv,
            min_speed: p.min_speed.unwrap_or(ms),
            step: p.speed_step.unwrap_or(6.0),
            vstep: p.vspeed_step.unwrap_or(vs),
            grid: p.grid_log2.unwrap_or(9),
            p,
        }
    }
}

impl Objective for TechDash {
    fn fields(&self) -> &[&'static str] {
        &["pos", "hspeed", "action", "vspeed", "yaw", "dist"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if !st.controllable() || st.action & (ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE | ACT_FLAG_HANGING) != 0 {
            return;
        }
        if hs(st) < self.min_speed {
            return;
        }
        if let (Some(lo), Some(hi)) = (self.p.min, self.p.max) {
            if !super::util::in_box(st.pos, lo, hi) {
                return;
            }
        }
        cx.put(0, pack_pos(st.pos, self.grid));
        cx.put(1, bucket(st.forward_vel, self.step));
        let a = match self.mv {
            Move::DiveRollout => class(
                st.action,
                &[
                    &[act::DIVE],
                    &[act::DIVE_SLIDE],
                    &[act::FORWARD_ROLLOUT],
                    &[act::FREEFALL],
                    &[act::JUMP, act::DOUBLE_JUMP],
                    &[act::WALKING, act::BRAKING, act::DECELERATING],
                ],
            ),
            Move::LongJumpChain => class(
                st.action,
                &[
                    &[act::LONG_JUMP],
                    &[act::LONG_JUMP_LAND],
                    &[act::CROUCH_SLIDE],
                    &[act::WALKING, act::BRAKING, act::DECELERATING],
                    &[act::JUMP, act::DOUBLE_JUMP],
                ],
            ),
        };
        cx.put(2, a);
        if air(st) && self.vstep > 0.0 {
            cx.put(3, bucket(st.vel[1], self.vstep));
        }
        if let Some(t) = self.p.toward {
            cx.put(4, yaw_to(st, bearing(st.pos, t)));
            cx.put(5, bucket(st.dist(t), self.p.toward_step.unwrap_or(400.0)));
        }
    }
}

// ---------------------------------------------------------------- CCM slide skip

const CCM_COURSE: i16 = 4;
const CCM_STAR: u8 = 0;
const CCM_AREA: u8 = 2;

fn ccm_start() -> [f32; 3] {
    [-5836.0, 6656.0, -6143.0]
}
fn ccm_exit_door() -> [f32; 3] {
    [-7628.0, -5836.0, -6988.0]
}
fn ccm_start_y() -> f32 {
    6300.0
}
fn ccm_start_reach() -> f32 {
    2000.0
}
fn ccm_bottom_y() -> f32 {
    -5650.0
}
fn exit_reach() -> f32 {
    1100.0
}
fn fall_speed() -> f32 {
    60.0
}

/// CCM star 1 skipping the slide: a wall kick out of the tube at the top and a
/// ~12,500 fall to the exit room (`Ccm::SlideSkip`, ccmslideskip.rs). Applies
/// in CCM area 2 (the slide) while star 1 is not had, Mario alive, in two
/// phases: 0, at the top (y > `start_y`, within `start_reach` of `start`);
/// 1, falling (airborne, y between `bottom_y` and `start_y`, vertical speed
/// below -`fall_speed`: a free fall, not the slide's own jumps). `done` is 1 in
/// the exit room (y < `bottom_y`, within `exit_reach` of `exit_door`) or after
/// leaving the slide for CCM area 1. Fields: `phase`; `pos` (grid; at the top
/// fine, in the fall coarse); `action` (1 wall kick in the air, 2 hitting a wall in the
/// air, 3 other air, 0 on foot); `hspeed` (forward speed, `speed_step`); `vspeed`
/// (fall, `vspeed_step`: 50); `done`.
///
/// `{"kind": "CcmSlideSkip"}`
#[derive(Deserialize)]
pub struct CcmSlideSkip {
    #[serde(default = "ccm_start")]
    pub start: [f32; 3],
    #[serde(default = "ccm_exit_door")]
    pub exit_door: [f32; 3],
    #[serde(default = "ccm_start_y")]
    pub start_y: f32,
    #[serde(default = "ccm_start_reach")]
    pub start_reach: f32,
    #[serde(default = "ccm_bottom_y")]
    pub bottom_y: f32,
    #[serde(default = "exit_reach")]
    pub exit_reach: f32,
    #[serde(default = "fall_speed")]
    pub fall_speed: f32,
    #[serde(default = "speed8")]
    pub speed_step: f32,
    #[serde(default = "speed50")]
    pub vspeed_step: f32,
    #[serde(default = "g7")]
    pub grid_log2: u32,
    #[serde(default = "g9")]
    pub fall_grid_log2: u32,
}

impl Objective for CcmSlideSkip {
    fn fields(&self) -> &[&'static str] {
        &["phase", "pos", "action", "hspeed", "vspeed", "done"]
    }
    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }
    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.level as u32 != LEVEL_CCM || !st.alive() || has_star(st, CCM_COURSE, CCM_STAR) {
            return;
        }
        if st.area == 1 || (st.area == CCM_AREA && st.pos[1] < self.bottom_y && st.hdist(self.exit_door) < self.exit_reach) {
            cx.put(5, 1);
            return;
        }
        if st.area != CCM_AREA {
            return;
        }
        let phase = if st.pos[1] > self.start_y && st.dist(self.start) < self.start_reach {
            0
        } else if air(st) && st.pos[1] > self.bottom_y && st.pos[1] <= self.start_y && st.vel[1] < -self.fall_speed {
            1
        } else {
            return;
        };
        cx.put(0, phase);
        let grid = if phase == 0 { self.grid_log2 } else { self.fall_grid_log2 };
        cx.put(1, pack_pos(st.pos, grid));
        let a = class(st.action, &[&[act::WALL_KICK_AIR], &[act::AIR_HIT_WALL]]);
        cx.put(2, if a == 0 && air(st) { 3 } else { a });
        cx.put(3, bucket(st.forward_vel, self.speed_step));
        if phase == 1 {
            cx.put(4, bucket(st.vel[1], self.vspeed_step));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn schema(kind: &str, params: &str) -> String {
        format!(r#"{{"kind":"Root","children":[{{"kind":"{kind}","params":{params}}}]}}"#)
    }

    /// (field name -> value) of the objective of `kind` for `s`.
    fn key(kind: &str, params: &str, s: &St) -> Vec<(String, i32)> {
        let fields = super::super::build(kind, &serde_json::from_str(params).unwrap()).unwrap().fields().to_vec();
        key_of(&schema(kind, params), 1, s).into_iter().map(|(f, v)| (fields[f as usize].to_string(), v)).collect()
    }

    fn get(k: &[(String, i32)], f: &str) -> Option<i32> {
        k.iter().find(|e| e.0 == f).map(|e| e.1)
    }

    const ACT_WALKING: u32 = 0x0400_0440;
    const ACT_LONG_JUMP: u32 = 0x0300_0888;
    const ACT_DIVE: u32 = 0x0188_088A;
    const ACT_WALL_KICK_AIR: u32 = 0x0300_0886;
    const ACT_FREEFALL: u32 = 0x0100_088C;

    #[test]
    fn every_listed_kind_builds() {
        for (_, kind, _, _) in TECHNIQUES {
            let o = super::super::build(kind, &Value::Null);
            assert!(o.is_ok(), "{kind}");
        }
        assert!(build("Nope", &Value::Null).is_none());
    }

    fn jrb(pos: [f32; 3], action: u32) -> St {
        let mut s = st(pos, 0.0, 12, 1, 0);
        s.course = 3;
        s.action = action;
        let mut b = obj("bhvExclamationBox", 5, [1540.0, 2160.0, 2130.0]);
        b.beh_params = 4 << 24;
        with_objects(s, vec![b])
    }

    #[test]
    fn jrb_cannonless_gate_and_cells() {
        let k = |s: &St| key("JrbPillarCannonless", "{}", s);
        // Silent: wrong course, star had, box missing, far from the ledge.
        let mut s = jrb([0.0, 900.0, 2700.0], ACT_WALKING);
        assert!(!k(&s).is_empty());
        s.course = 4;
        assert!(k(&s).is_empty());
        let mut s = jrb([0.0, 900.0, 2700.0], ACT_WALKING);
        s.course_stars[2] = 1 << 4;
        assert!(k(&s).is_empty());
        let mut s = jrb([0.0, 900.0, 2700.0], ACT_WALKING);
        s.objects = None;
        assert!(k(&s).is_empty());
        assert!(k(&jrb([-9000.0, 900.0, 2700.0], ACT_WALKING)).is_empty());
        // Wall kick vs plain air vs on foot are three cells; the spire is a launch spot.
        let (a, b, c) = (jrb([200.0, 1200.0, 2700.0], ACT_WALL_KICK_AIR), jrb([200.0, 1200.0, 2700.0], ACT_FREEFALL), jrb([200.0, 1200.0, 2700.0], ACT_WALKING));
        assert_eq!((get(&k(&a), "action"), get(&k(&b), "action"), get(&k(&c), "action")), (Some(2), Some(3), Some(0)));
        let spire = jrb([127.0, 1100.0, 2779.0], ACT_WALKING);
        assert_eq!(get(&k(&spire), "launch"), Some(1));
        // On the platform: the success cell, one cell wherever on it.
        let p1 = k(&jrb([700.0, 1946.0, 2300.0], ACT_WALKING));
        let p2 = k(&jrb([1000.0, 1950.0, 2500.0], ACT_WALKING));
        assert_eq!(p1, p2);
        assert_eq!(get(&p1, "done"), Some(1));
        assert_eq!(p1.len(), 1);
        // Cannon: silent.
        assert!(k(&jrb([200.0, 1200.0, 2700.0], obrrt_core::layout::ACT_IN_CANNON)).is_empty());
    }

    #[test]
    fn jrb_hyper_phases() {
        let k = |s: &St| key("JrbPillarHyper", "{}", s);
        // The dock: slow, west of -4800 and above 900.
        let mut dock = jrb([-5200.0, 1000.0, 1300.0], ACT_WALKING);
        assert_eq!(get(&k(&dock), "phase"), Some(1));
        // Hyperspeed backwards, just past the dock: phase 2, speed buckets of 50.
        dock.forward_vel = -160.0;
        dock.action = ACT_LONG_JUMP;
        let fast = k(&dock);
        assert_eq!((get(&fast, "phase"), get(&fast, "hspeed"), get(&fast, "action")), (Some(2), Some(-4), Some(2)));
        dock.forward_vel = -210.0;
        assert_ne!(k(&dock), fast);
        // Walking at 60 in the middle of nowhere does not apply.
        assert!(k(&jrb([2000.0, 1000.0, 2000.0], ACT_WALKING)).is_empty());
        // The spire at half hyperspeed: phase 3.
        let mut sp = jrb([60.0, 1200.0, 2730.0], ACT_WALKING);
        sp.forward_vel = -80.0;
        assert_eq!(get(&k(&sp), "phase"), Some(3));
        sp.forward_vel = -10.0;
        assert!(k(&sp).is_empty());
        // Swimming is wet: never.
        let mut w = jrb([-5200.0, 1000.0, 1300.0], 0x300022C0);
        w.forward_vel = -200.0;
        assert!(k(&w).is_empty());
    }

    #[test]
    fn jrb_cave_clip() {
        let k = |s: &St, wide: bool| key("JrbCaveClip", &format!(r#"{{"wide":{wide}}}"#), s);
        let mut s = st([5000.0, -400.0, -200.0], 0.0, 12, 1, 0);
        s.course = 3;
        s.action = 0x300022C0 & !0x0C0 | 0x2000; // swimming flag
        assert!(swimming(&s));
        s.vel = [0.0, 0.0, -35.0];
        let k0 = k(&s, false);
        assert_eq!(get(&k0, "hspeed"), Some(3));
        s.vel = [0.0, 0.0, -45.0];
        assert_eq!(get(&k(&s, false), "hspeed"), Some(4));
        // Pitch changes the cell; far from the entry or past the bay does not apply.
        let mut up = s.clone();
        up.pitch = 5000;
        assert_ne!(k(&up, false), k(&s, false));
        let mut behind = s.clone();
        behind.pos[2] = -700.0;
        assert!(k(&behind, false).is_empty());
        let mut fell = s.clone();
        fell.pos[1] = -3900.0;
        assert!(k(&fell, false).is_empty());
        // In the cave: one done cell.
        let mut cave = s.clone();
        cave.pos = [5200.0, -3000.0, -1000.0];
        assert_eq!(k(&cave, false), vec![("done".to_string(), 1)]);
        // Wide: a submerged (non-swimming) action at speed counts, but not at rest.
        let mut sub = s.clone();
        sub.action = 0x0000_00C1;
        assert!(k(&sub, false).is_empty());
        assert!(!k(&sub, true).is_empty());
        sub.vel = [0.0, 0.0, -10.0];
        assert!(k(&sub, true).is_empty());
    }

    fn bob(pos: [f32; 3], action: u32) -> St {
        let mut s = st(pos, 20.0, 9, 1, 0);
        s.action = action;
        s
    }

    #[test]
    fn island_hop() {
        let k = |s: &St| key("IslandHopJump", "{}", s);
        // On the ledge: launch 1; at the summit: 2; down in the level: silent.
        assert_eq!(get(&k(&bob([3341.0, 3007.0, -1564.0], ACT_WALKING)), "launch"), Some(1));
        assert_eq!(get(&k(&bob([3072.0, 4097.0, -2559.0], ACT_WALKING)), "launch"), Some(2));
        assert!(k(&bob([0.0, 100.0, 0.0], ACT_WALKING)).is_empty());
        // Wing cap or star had: silent.
        let mut w = bob([3341.0, 3007.0, -1564.0], ACT_WALKING);
        w.mario_flags = 8;
        assert!(k(&w).is_empty());
        let mut h = bob([3341.0, 3007.0, -1564.0], ACT_WALKING);
        h.course_stars[0] = 1 << 2;
        assert!(k(&h).is_empty());
        // Walking vs dive vs air; the island is one cell.
        let dive = k(&bob([3341.0, 3007.0, -1500.0], ACT_DIVE));
        assert_eq!(get(&dive, "action"), Some(4));
        assert!(get(&dive, "vspeed").is_some());
        let i1 = k(&bob([4000.0, 3072.0, 500.0], ACT_WALKING));
        let i2 = k(&bob([5000.0, 3080.0, 2000.0], ACT_WALKING));
        assert_eq!(i1, i2);
        assert_eq!(i1, vec![("done".to_string(), 1)]);
        // Same spot on the island: the Star kind is active, with the box as target.
        let ks = |s: &St| key("IslandHopStar", "{}", s);
        assert!(ks(&bob([0.0, 100.0, 0.0], ACT_WALKING)).is_empty());
        let on = ks(&bob([4000.0, 3072.0, 500.0], ACT_WALKING));
        assert_eq!(get(&on, "spawned"), Some(0));
        let mut sp = obj("bhvSpawnedStar", 7, [5540.0, 3350.0, 1200.0]);
        sp.beh_params = 2 << 24;
        sp.home = [5540.0, 3350.0, 1200.0];
        let off_island = with_objects(bob([0.0, 100.0, 0.0], ACT_WALKING), vec![sp]);
        assert_eq!(get(&ks(&off_island), "spawned"), Some(1));
        // Nearer the box is a different bucket.
        let near = ks(&bob([5400.0, 3072.0, 1100.0], ACT_WALKING));
        assert!(get(&near, "dist") < get(&on, "dist"));
    }

    #[test]
    fn owlless_cage() {
        let k = |s: &St| key("OwllessCage", "{}", s);
        let wf = |pos: [f32; 3], action: u32| {
            let mut s = st(pos, 10.0, 24, 1, 0);
            s.action = action;
            s
        };
        assert!(!k(&wf([1800.0, 3500.0, 0.0], ACT_FREEFALL)).is_empty());
        assert!(k(&wf([2500.0, 3000.0, 0.0], ACT_FREEFALL)).is_empty(), "below the start height");
        assert!(k(&wf([2880.0, 3900.0, 190.0], ACT_FREEFALL)).is_empty(), "under the cage");
        assert!(k(&wf([2880.0, 3500.0, 3000.0], ACT_FREEFALL)).is_empty(), "far");
        let mut hoot = wf([1800.0, 3500.0, 0.0], obrrt_core::layout::ACT_RIDING_HOOT);
        assert!(k(&hoot).is_empty());
        hoot.action = ACT_WALL_KICK_AIR;
        assert_eq!(get(&k(&hoot), "action"), Some(2));
        // In the cage: one cell.
        let c1 = k(&wf([2880.0, 4100.0, 190.0], ACT_WALKING));
        let c2 = k(&wf([2900.0, 4300.0, 200.0], ACT_WALKING));
        assert_eq!(c1, c2);
        assert_eq!(c1, vec![("done".to_string(), 1)]);
        // The star object overrides the default position.
        let mut star = obj("bhvStar", 3, [0.0, 4300.0, 0.0]);
        star.beh_params = 4 << 24;
        let s = with_objects(wf([100.0, 4100.0, 0.0], ACT_WALKING), vec![star]);
        assert_eq!(k(&s), vec![("done".to_string(), 1)]);
    }

    #[test]
    fn pole_redirect() {
        let k = |s: &St, p: &str| key("TechPoleRedirect", p, s);
        let mut pole = obj("bhvPoleGrabbing", 9, [0.0, 0.0, 1000.0]);
        pole.interact_type = INTERACT_POLE;
        // Jumping at the pole at 200 u/f.
        let mut s = st([0.0, 100.0, 0.0], 100.0, 4, 1, 0);
        s.action = 0x0300_0880;
        s.vel = [0.0, 0.0, 200.0];
        let s = with_objects(s, vec![pole.clone()]);
        let a = k(&s, "{}");
        assert_eq!(get(&a, "phase"), Some(1));
        // Too slow, or pointing away: silent.
        let mut slow = s.clone();
        slow.vel = [0.0, 0.0, 100.0];
        assert!(k(&slow, "{}").is_empty());
        let mut away = s.clone();
        away.vel = [200.0, 0.0, 0.0];
        assert!(k(&away, "{}").is_empty());
        // Holding it with the speed stored: phase 3; touched but not on it: phase 2.
        let mut on = st([0.0, 100.0, 980.0], 0.0, 4, 1, 0);
        on.action = obrrt_core::layout::ACT_HOLDING_POLE;
        on.vel = [0.0, 0.0, 200.0];
        let on = with_objects(on, vec![pole.clone()]);
        assert_eq!(get(&k(&on, "{}"), "phase"), Some(3));
        let mut touch = on.clone();
        touch.action = 0x0300_0880;
        assert_eq!(get(&k(&touch, "{}"), "phase"), Some(2));
        // Carried out: crouch slide at 190 next to the pole.
        let mut out = st([0.0, 100.0, 1500.0], 190.0, 4, 1, 0);
        out.action = 0x0480_8459;
        out.vel = [190.0, 0.0, 0.0];
        let out = with_objects(out, vec![pole]);
        let o = k(&out, "{}");
        assert_eq!((get(&o, "phase"), get(&o, "action")), (Some(4), Some(1)));
        // Aimed: velocity at the aim point vs not.
        assert_eq!(get(&k(&out, r#"{"aim":[1000,100,1500]}"#), "aimed"), Some(1));
        assert_eq!(get(&k(&out, r#"{"aim":[0,100,5000]}"#), "aimed"), Some(0));
    }

    #[test]
    fn openers() {
        let k = |kind: &str, s: &St| key(kind, "{}", s);
        let mut s = st([2600.0, 256.0, 5120.0], 0.0, 24, 1, 0);
        let spawn = k("OpenerWf", &s);
        assert!(!spawn.is_empty());
        assert_eq!(get(&spawn, "stage"), Some(1), "WF spawn is below the wall waypoint");
        // Speed buckets and facing change the cell.
        s.forward_vel = 40.0;
        assert_ne!(k("OpenerWf", &s), spawn);
        // BoB kind does not fire in WF; the right level does.
        assert!(k("OpenerBob", &s).is_empty());
        let b = st([-6558.0, 0.0, 6464.0], 0.0, 9, 1, 0);
        assert!(!k("OpenerBob", &b).is_empty());
        assert_eq!(get(&k("OpenerBob", &b), "stage"), Some(0));
        // At the target: done, one cell. WF needs height, BoB a band.
        let hi = st([3300.0, 1600.0, 2000.0], 0.0, 24, 1, 0);
        let hi2 = st([3400.0, 1700.0, 2100.0], 5.0, 24, 1, 0);
        assert_eq!(k("OpenerWf", &hi), k("OpenerWf", &hi2));
        assert_eq!(k("OpenerWf", &hi), vec![("done".to_string(), 1)]);
        let low = st([3300.0, 800.0, 2000.0], 0.0, 24, 1, 0);
        assert_ne!(k("OpenerWf", &low), vec![("done".to_string(), 1)]);
        let plateau = st([1600.0, 800.0, 4150.0], 0.0, 9, 1, 0);
        assert_eq!(k("OpenerBob", &plateau), vec![("done".to_string(), 1)]);
        let above = st([1600.0, 1500.0, 4150.0], 0.0, 9, 1, 0);
        assert_ne!(k("OpenerBob", &above), vec![("done".to_string(), 1)]);
        // Far from everything: silent.
        assert!(k("OpenerBob", &st([-20000.0, 0.0, 0.0], 0.0, 9, 1, 0)).is_empty());
    }

    #[test]
    fn dash_techniques() {
        let k = |kind: &str, p: &str, s: &St| key(kind, p, s);
        let mut s = st([0.0, 0.0, 0.0], 30.0, 9, 1, 0);
        s.action = ACT_DIVE;
        s.vel = [0.0, 10.0, 30.0];
        let dive = k("TechDiveRollout", "{}", &s);
        assert_eq!((get(&dive, "action"), get(&dive, "hspeed")), (Some(1), Some(5)));
        // The high dive (rising vy) and the plain one are two cells.
        let mut hi = s.clone();
        hi.vel[1] = 40.0;
        assert_ne!(k("TechDiveRollout", "{}", &hi), dive);
        // Slow: silent. Swimming: silent.
        let mut slow = s.clone();
        slow.vel = [0.0, 0.0, 10.0];
        assert!(k("TechDiveRollout", "{}", &slow).is_empty());
        // Long-jump chain needs 30; a walking run at 32 is class 4 and still keyed.
        let mut lj = st([0.0, 0.0, 0.0], 32.0, 9, 1, 0);
        lj.vel = [0.0, 0.0, 32.0];
        assert_eq!(get(&k("TechLongJumpChain", "{}", &lj), "action"), Some(4));
        lj.action = ACT_LONG_JUMP;
        assert_eq!(get(&k("TechLongJumpChain", "{}", &lj), "action"), Some(1));
        lj.vel = [0.0, 0.0, 25.0];
        assert!(k("TechLongJumpChain", "{}", &lj).is_empty());
        assert!(!k("TechLongJumpChain", r#"{"min_speed":20}"#, &lj).is_empty());
        // Coarse position: nearby states share a cell, far ones do not.
        let mut a = s.clone();
        a.pos = [100.0, 0.0, 100.0];
        assert_eq!(k("TechDiveRollout", "{}", &a), dive);
        a.pos = [1100.0, 0.0, 100.0];
        assert_ne!(k("TechDiveRollout", "{}", &a), dive);
        // Box and target.
        let boxed = r#"{"min":[-10,-10,-10],"max":[10,10,10],"toward":[0,0,1000]}"#;
        let b = k("TechDiveRollout", boxed, &s);
        assert_eq!(get(&b, "yaw"), Some(3));
        assert_eq!(get(&b, "dist"), Some(2));
        a.pos = [100.0, 0.0, 0.0];
        assert!(k("TechDiveRollout", boxed, &a).is_empty());
    }

    #[test]
    fn ccm_slide_skip() {
        let k = |s: &St| key("CcmSlideSkip", "{}", s);
        let ccm = |pos: [f32; 3], area: u8, action: u32| {
            let mut s = st(pos, 20.0, 5, area, 0);
            s.action = action;
            s
        };
        let top = k(&ccm([-5600.0, 6600.0, -6200.0], 2, ACT_FREEFALL));
        assert_eq!((get(&top, "phase"), get(&top, "action")), (Some(0), Some(3)));
        let kick = k(&ccm([-5600.0, 6600.0, -6200.0], 2, ACT_WALL_KICK_AIR));
        assert_eq!(get(&kick, "action"), Some(1));
        // Falling fast mid-way is phase 1; the normal slide (not fast) is silent.
        let mut fall = ccm([-6500.0, 0.0, -6500.0], 2, ACT_FREEFALL);
        fall.vel = [0.0, -75.0, 0.0];
        let f = k(&fall);
        assert_eq!((get(&f, "phase"), get(&f, "vspeed")), (Some(1), Some(-2)));
        fall.vel[1] = -20.0;
        assert!(k(&fall).is_empty());
        // Exit room and leaving the slide: done. Star had: silent.
        let exit = ccm([-7600.0, -5836.0, -6900.0], 2, ACT_WALKING);
        assert_eq!(k(&exit), vec![("done".to_string(), 1)]);
        assert_eq!(k(&ccm([0.0, 0.0, 0.0], 1, ACT_WALKING)), vec![("done".to_string(), 1)]);
        let mut had = ccm([-5600.0, 6600.0, -6200.0], 2, ACT_FREEFALL);
        had.course_stars[3] = 1;
        assert!(k(&had).is_empty());
        // Another level: silent.
        assert!(k(&st([-5600.0, 6600.0, -6200.0], 0.0, 9, 2, 0)).is_empty());
    }
}
