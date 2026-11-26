//! `Tech::LakituSkip` (technique `lakituskip`): from the castle grounds
//! before the intro, reach the castle door without starting Lakitu's intro
//! (dialog 34, ~8 s plus the cutscene).
//!
//! Where (camera_lakitu.inc.c, camera_lakitu_intro_act_trigger_cutscene;
//! mario_actions_cutscene.c, set_mario_npc_dialog / mario_ready_to_speak):
//! - The intro Lakitu (`bhvCameraLakitu`, oBehParams2ndByte 1) exists only
//!   while `gNeverEnteredCastle` (bhv_camera_lakitu_init); in its act 0 it
//!   waits for Mario in the box -544 < x < 545, y > 800, -2000 < z < -177
//!   (the bridge, floor ~808-966; the moat below).
//! - The box takes Mario through set_mario_npc_dialog, which only takes him
//!   in a stationary or moving action (not invulnerable). Lakitu
//!   (OBJ_LIST_DEFAULT) updates after Mario (OBJ_LIST_PLAYER) in the same
//!   frame, and an air step's landing ends that frame in the landing action
//!   (act_jump etc. return FALSE after set_mario_action), so **any ground
//!   frame in the box is caught**: there is no same-frame landing cancel.
//! - Airborne Mario passes untouched, but no air chain covers the box's
//!   1823 units at normal speed: a long jump (vy 30, gravity 2, 48 u/f) is
//!   ~1270 over the rising bridge, a triple jump + dive ~1300.
//! - The bridge is wider than the box ("slightly smaller than the actual
//!   bridge bounds, allowing the RTA speedrunning method"): its floor runs
//!   to x ~ -556 / 552, a strip ~10 wide each side outside the box. Every
//!   landing in the box's z range has to be on that strip. The one success
//!   seen (tech-lakitu-on4-s1 node 529): onto the strip from the bank at
//!   x -596 (a wall puts Mario at x -549.2), jump / double / triple jump
//!   along it landing at x -549..-550, dive out past z -2000.
//! - The door: the two castle-door warps at (-76, 803, -3155) and
//!   (77, 803, -3155) (castle_grounds/areas/1/collision.inc.c).
//!
//! Two modes (keys), the same success and ETA:
//! - [`Mode::Edge`]: steer onto the nearer strip and along it (walk, jumps).
//! - [`Mode::Air`]: crouch-slide long jump at the box's start, aimed to land
//!   on the strip ~1270 on, then long jumps (Z + A on each landing) out.
//!
//! Success: in the castle (level 6), or within [`DOOR_R`] of the door, with
//! Mario never caught (`ACT_WAITING_FOR_DIALOG` / `ACT_READING_NPC_DIALOG`,
//! dialog 34) nor fallen into the moat.
//! ETA (v2): the straight-line frames to the door, Lost from the first
//! frame Mario is caught or falls in (a latch in `mem`; v1 cut the
//! trajectory there). A walk into the box is caught in its first frame, so
//! it is Lost at once; an airborne state over the box gets its distance
//! credit and loses it if it lands there.

use obrrt_core::layout::{
    ACT_READING_NPC_DIALOG, ACT_WAITING_FOR_DIALOG, LEVEL_CASTLE, LEVEL_CASTLE_GROUNDS,
};
use obrrt_core::{Eta, GamePad, Mem, Obj, ObjId, Objective, PadOverlay, St, buttons, frames_for};

/// Between the two door warps, at their height.
const DOOR: [f32; 3] = [0.0, 803.0, -3155.0];
const DOOR_R: f32 = 350.0;
/// The trigger box (x, z) and its height bound.
const BOX_X: (f32, f32) = (-544.0, 545.0);
const BOX_Z: (f32, f32) = (-2000.0, -177.0);
const BOX_Y: f32 = 800.0;
/// Starts lie within this of the bridge's south end (and not past the box).
const START_R: f32 = 2500.0;
const BRIDGE_SOUTH: [f32; 3] = [0.0, 811.0, -150.0];
/// oBehParams2ndByte of the intro Lakitu; its waiting act.
const CAMERA_LAKITU_BP_INTRO: u32 = 1;
const CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE: i32 = 0;
const DIALOG_034: i16 = 34;
/// Strip centre lines outside the box (castle_grounds collision: the bridge
/// floor to x ~ -556 / 552; Mario held at x -549.2 in the success above).
const EDGE_X: [f32; 2] = [-549.5, 548.5];
/// Within this of the strip's centre line counts as on it (steering).
const ON_STRIP: f32 = 40.0;
/// Edge mode: the steering point this far ahead on the strip's line.
const STRIP_AHEAD: f32 = 150.0;
/// Air mode: where the first long jump aims to land (on the strip).
const AIR_LAND_Z: f32 = -1450.0;
/// Air mode: the crouch slide starts this far before the box at the latest.
const TAKEOFF: (f32, f32) = (-177.0, -40.0);
/// Below the bridge and the forecourt (floors 803-966): in the moat.
const FELL_Y: f32 = 700.0;
/// Frames ahead for [`entering`].
const ENTER_FRAMES: f32 = 4.0;

const ACT_WALKING: u32 = 0x0400_0440;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_JUMP: u32 = 0x0300_0880;
const ACT_DOUBLE_JUMP: u32 = 0x0300_0881;
const ACT_TRIPLE_JUMP: u32 = 0x0100_0882;
const ACT_SIDE_FLIP: u32 = 0x0100_0887;
const ACT_BACKFLIP: u32 = 0x0100_0883;
const ACT_FREEFALL: u32 = 0x0100_088C;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Edge,
    Air,
    /// `OBRRT_LAKITU_ARC=1` only: two long jumps on one straight line,
    /// takeoff on the outer bank → the railing strip → the castle door.
    Arc,
    /// `OBRRT_LAKITU_ARC=1` only: the hill slide's speed (~100 u/f) and a
    /// double jump from the bridge start over the whole box ([`hill`]).
    Hill,
}

/// `OBRRT_LAKITU_ARC=1`: the Arc mode, and the Edge / Air modes' direct
/// approach and door aim (see [`arc`]). Off by default (byte-identical).
pub fn arc_on() -> bool {
    obrrt_core::config::current().objectives.lakitu_arc
}

pub struct LakituSkip {
    lakitu: ObjId,
    mode: Mode,
    /// `OBRRT_FREE_SETUP=1` only: the setup variant ([`free`]); 0 off.
    fs: u8,
}

fn is_intro_lakitu(o: &Obj) -> bool {
    o.bhv_name() == "bhvCameraLakitu" && (o.beh_params >> 16) & 0xFF == CAMERA_LAKITU_BP_INTRO
}

fn talking(s: &St) -> bool {
    matches!(s.action, ACT_WAITING_FOR_DIALOG | ACT_READING_NPC_DIALOG) || s.dialog == DIALOG_034
}

fn grounds(s: &St) -> bool {
    s.level as u32 == LEVEL_CASTLE_GROUNDS && s.area == 1
}

fn arrived(s: &St) -> bool {
    s.level as u32 == LEVEL_CASTLE || (grounds(s) && s.dist(DOOR) < DOOR_R)
}

/// `OBRRT_LAKITU_THROUGH_DOOR=1` (off by default): the Hill mode is met only
/// inside the castle, not within [`DOOR_R`] of the door. The chain reaches
/// the door at ~100 u/f, so a state 350 away is already committed to hitting
/// it (doors open only to ACT_WALKING / ACT_DECELERATING, interaction.c); as
/// one objective, the arrival is judged by whether the door opens.
fn through_door() -> bool {
    obrrt_core::config::current().objectives.lakitu_through_door
}

/// Frames of the door's opening animation, until the warp (door pull ~30).
const DOOR_ANIM_FRAMES: f64 = 30.0;

/// In the trigger box (camera_lakitu_intro_act_trigger_cutscene's test).
fn in_box(s: &St) -> bool {
    grounds(s)
        && s.pos[0] > BOX_X.0
        && s.pos[0] < BOX_X.1
        && s.pos[1] > BOX_Y
        && s.pos[2] > BOX_Z.0
        && s.pos[2] < BOX_Z.1
}

/// Outside the box over a floor of the bridge, banks or forecourt (not
/// over the moat, floor -511, where an airborne Mario only falls in).
fn safe(s: &St) -> bool {
    !in_box(s) && (s.level as u32 != LEVEL_CASTLE_GROUNDS || s.floor_y > FELL_Y)
}

/// Before the box, inside its x range, and moving into it within
/// [`ENTER_FRAMES`]: as good as caught (a start there can only jump in).
fn entering(s: &St) -> bool {
    s.pos[0] > BOX_X.0
        && s.pos[0] < BOX_X.1
        && s.pos[2] >= BOX_Z.1
        && s.pos[2] + ENTER_FRAMES * s.vel[2] < BOX_Z.1
}

/// The nearer strip's centre line.
fn side(s: &St) -> f32 {
    if s.pos[0] < 0.0 { EDGE_X[0] } else { EDGE_X[1] }
}

/// Near the bridge, before the box ends: where the hints apply.
fn near_bridge(s: &St) -> bool {
    grounds(s) && s.pos[0].abs() < 800.0 && s.pos[2] < 400.0 && s.pos[2] > BOX_Z.0 - 100.0 && s.pos[1] > FELL_Y
}

/// Fallen off the bridge into the moat (from the bridge or past it).
fn fell(s: &St) -> bool {
    grounds(s) && s.pos[2] < BOX_Z.1 && s.pos[1] < FELL_Y
}

/// `mem.flags`: caught by a dialog or fallen into the moat on some frame
/// since the start (v1's `clean` prefix ended there).
const CAUGHT: u32 = 1;

/// `OBRRT_GROUNDS_FIX=1`: also offered from states farther than the
/// registry's object radius (5000) from the intro Lakitu, e.g. the spawn
/// (-1328, 260, 4354). Without it the skip is generated only once some
/// state gets within 5000, so the search's first ~130 attempts on the
/// grounds went to what is near the spawn (the grilled cannon at 4400,
/// trees, doors) instead (runs/gs-off-s1: first LakituSkip at event 132).
pub fn generate_level(b: &St, out: &mut Vec<std::sync::Arc<dyn Objective>>) {
    if !crate::extra::warps::grounds_fix() || !grounds(b) {
        return;
    }
    for o in b.objects() {
        if is_intro_lakitu(o) && obrrt_core::dist(o.pos, b.pos) > 5000.0 {
            generate(b, o, out);
        }
    }
}

/// Generated per object: for the intro Lakitu still waiting, both modes.
pub fn generate(b: &St, o: &Obj, out: &mut Vec<std::sync::Arc<dyn Objective>>) {
    if grounds(b) && is_intro_lakitu(o) && o.action == CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE {
        let fs_on = super::freesetup::on();
        let mut modes = vec![Mode::Edge, Mode::Air];
        if arc_on() {
            modes.extend([Mode::Arc, Mode::Hill]);
        }
        for mode in modes {
            let n = if fs_on { free::variants(mode) } else { 1 };
            for fs in 0..n {
                out.push(std::sync::Arc::new(LakituSkip { lakitu: o.id, mode, fs }));
            }
        }
    }
}

impl Objective for LakituSkip {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn kind(&self) -> &'static str {
        "Tech::LakituSkip"
    }
    fn key(&self) -> String {
        let k = match self.mode {
            Mode::Edge => "Tech::LakituSkip".to_string(),
            Mode::Air => "Tech::LakituSkip air".to_string(),
            Mode::Arc => "Tech::LakituSkip arc".to_string(),
            Mode::Hill => "Tech::LakituSkip hill".to_string(),
        };
        if super::freesetup::on() { k + &super::freesetup::tag(self.fs as usize) } else { k }
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // On the route to every star (Door::Enter is 8), and only offered
        // until the intro plays: high enough that the pool samples it
        // before then (at 4, after a few failures' decay, 1-4 attempts in
        // 60 M frames).
        // OBRRT_FREE_SETUP: a mode's variants share its weight.
        if super::freesetup::on() { 12.0 / free::variants(self.mode) as f64 } else { 12.0 }
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Anywhere outside the box (the south end, the strip) not past it.
        if self.mode == Mode::Hill {
            // In the chain at speed, or on the ground within a run of the
            // hill's run-up (from further the search's other objectives get
            // there; the attempt's horizon covers run-up + chain).
            let lakitu = s.find(self.lakitu).is_some_and(|o| o.action == CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE);
            let in_chain = s.forward_vel > 55.0 && s.pos[2] > BOX_Z.1 && s.pos[2] < 4600.0;
            let near = s.action & 0x800 == 0 && s.pos[2] > 2500.0 && s.dist(hill::C) < 2600.0;
            return grounds(s) && !talking(s) && !in_box(s) && lakitu && (in_chain || near);
        }
        if self.mode == Mode::Arc {
            return grounds(s)
                && !talking(s)
                && (safe(s) || arc::hop1_over_moat(s))
                && !arc::caught_soon(s)
                && s.pos[1] > FELL_Y
                && s.pos[2] > BOX_Z.0
                && s.dist(BRIDGE_SOUTH) < START_R
                && s.find(self.lakitu).is_some_and(|o| o.action == CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE);
        }
        grounds(s)
            && !talking(s)
            && safe(s)
            && !entering(s)
            && s.pos[1] > FELL_Y
            && s.pos[2] > BOX_Z.0
            && s.dist(BRIDGE_SOUTH) < START_R
            && s.find(self.lakitu).is_some_and(|o| o.action == CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        // Hill: the run-up (~60-120 frames) plus the chain (~120).
        let horizon = match self.mode {
            Mode::Hill if through_door() => 360,
            Mode::Hill => 300,
            _ => 200,
        };
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon })
    }
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        // v1's success on the clean prefix: arrived, never caught by the
        // dialog nor fallen into the moat on any frame from the start.
        if mem.flags & CAUGHT != 0 {
            return Eta::Lost;
        }
        if talking(s) || fell(s) {
            mem.flags |= CAUGHT;
            return Eta::Lost;
        }
        if self.mode == Mode::Hill && through_door() {
            if s.level as u32 == LEVEL_CASTLE {
                return Eta::Done;
            }
            if !grounds(s) {
                return Eta::Lost;
            }
            return Eta::Frames(frames_for(s.dist(DOOR)) + DOOR_ANIM_FRAMES);
        }
        if arrived(s) {
            return Eta::Done;
        }
        if !grounds(s) {
            return Eta::Lost;
        }
        if super::freesetup::on() {
            return Eta::Frames(free::eta(self.mode, self.fs, s, mem));
        }
        // Straight-line to the door warps (through the box: a walk into it
        // is caught at once, which is Lost above).
        Eta::Frames(frames_for((s.dist(DOOR) - DOOR_R).max(0.0)))
    }
    fn exists_in(&self, s: &St) -> bool {
        grounds(s) && s.find(self.lakitu).is_some()
    }
    fn precision(&self) -> obrrt_core::Precision {
        // A ~10-unit strip.
        obrrt_core::Precision::Precise
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if super::freesetup::on() {
            return Some(free::target(self.mode, self.fs, s));
        }
        if arc_on() {
            if self.mode == Mode::Arc {
                return Some(arc::target(s));
            }
            if self.mode == Mode::Hill {
                return Some(hill::target(s));
            }
            if let Some(t) = arc::edge_air_target(self.mode == Mode::Air, s) {
                return Some(t);
            }
        }
        let ex = side(s);
        if !near_bridge(s) || s.pos[2] <= BOX_Z.0 + 100.0 {
            return Some(DOOR);
        }
        Some(match self.mode {
            // Onto the strip at the box's start, then along it.
            Mode::Edge if s.pos[2] > BOX_Z.1 && (s.pos[0] - ex).abs() > ON_STRIP => [ex, 830.0, BOX_Z.1 - 20.0],
            Mode::Edge => [ex, 960.0, s.pos[2].min(BOX_Z.1) - STRIP_AHEAD],
            // At the long jump's landing spot on the strip, then out.
            Mode::Air if s.pos[2] > AIR_LAND_Z + 100.0 => [ex, 960.0, AIR_LAND_Z],
            Mode::Air => [ex, 960.0, BOX_Z.0 - 200.0],
            Mode::Arc => arc::target(s),
            Mode::Hill => hill::target(s),
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Over the box's far half in a jump that would land in it: dive
        // (B; +15 u/f, capped at 48) to carry past z -2000.
        use buttons::B_BUTTON;
        if matches!(self.mode, Mode::Arc | Mode::Hill) {
            // The overlay plays the hops / the chain.
            return 0;
        }
        let far = near_bridge(s) && s.pos[2] < -1300.0 && s.pos[2] > BOX_Z.0;
        if far
            && s.vel[1] < 0.0
            && matches!(s.action, ACT_JUMP | ACT_DOUBLE_JUMP | ACT_TRIPLE_JUMP | ACT_SIDE_FLIP | ACT_BACKFLIP | ACT_FREEFALL)
        {
            return B_BUTTON;
        }
        0
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        if super::freesetup::on() {
            return Some(free::overlay(self.mode, self.fs));
        }
        if self.mode == Mode::Arc {
            return Some(PadOverlay::new(arc::overlay));
        }
        if self.mode == Mode::Hill {
            return Some(PadOverlay::new(hill::overlay));
        }
        let air = self.mode == Mode::Air;
        Some(PadOverlay::new(move |traj, played, genome| lj_rule(air, traj, played, genome)))
    }
}

/// Long-jump timing near the bridge (exact frames, which alternate-frame
/// steering buttons can't give):
/// - on a long jump's landing (ACT_LONG_JUMP_LAND, a frame on the strip):
///   Z held + A newly pressed, another long jump (act_long_jump_land);
/// - in a long jump: Z held, A released (ready for that press);
/// - air mode, walking at speed just before the box: Z (crouch slide), and
///   in the crouch slide Z + A (long jump, act_crouch_slide).
fn lj_rule(air: bool, traj: &[St], played: &[GamePad], genome: GamePad) -> GamePad {
    use buttons::{A_BUTTON, Z_TRIG};
    let Some(s) = traj.last() else { return genome };
    if !near_bridge(s) || !s.controllable() {
        return genome;
    }
    let last_a = played.last().is_some_and(|p| p.button & A_BUTTON != 0);
    let mut pad = genome;
    let set = |pad: &mut GamePad, b: u16| pad.button = (pad.button & !(A_BUTTON | Z_TRIG | buttons::B_BUTTON)) | b;
    match s.action {
        ACT_LONG_JUMP_LAND if !last_a => set(&mut pad, Z_TRIG | A_BUTTON),
        ACT_LONG_JUMP_LAND => set(&mut pad, Z_TRIG),
        ACT_LONG_JUMP => set(&mut pad, Z_TRIG),
        ACT_CROUCH_SLIDE if air && s.pos[2] > TAKEOFF.0 - 60.0 && !last_a => set(&mut pad, Z_TRIG | A_BUTTON),
        ACT_WALKING
            if air && s.forward_vel >= 20.0 && s.vel[2] < 0.0 && s.pos[2] > TAKEOFF.0 && s.pos[2] < TAKEOFF.1 =>
        {
            set(&mut pad, Z_TRIG)
        }
        _ => {}
    }
    pad
}

/// The Arc mode (`OBRRT_LAKITU_ARC=1`): Mario runs straight at a point on
/// the nearer railing strip and long-jumps onto it, then long-jumps again
/// (Z + A on the landing) through the forecourt gate towards the door.
/// A long jump never turns Mario (update_air_without_turn) and the one
/// from LONG_JUMP_LAND keeps that facing (act_long_jump_land only allows
/// that jump), so run-up, strip landing and second hop lie on one line;
/// only the air stick's sideways 10 u/f (and ±1.5 u/f² forward) bends it.
///
/// Geometry (castle grounds static floors / walls, `arc_lab column`,
/// `lakitu_lab boot walls`):
/// - the railing top is a floor, x -553..-456 / 456..554 (outer faces are
///   walls down to the moat at x -553 / 554), y 964 at z -300 falling to
///   957 at z -2000 (a ramp from the bridge, 811, over z -177..-300);
///   outside the box only x -553..-544 / 545..554: the strip *is* the
///   railing top's outer edge;
/// - the banks south of the moat (z > -121) fall away from the bridge:
///   y ~ 811 - 0.2 (|x| - 560) west, 807 - 0.245 (|x| - 560) east; along
///   the moat edge (z -162..-121) a parapet, top 882 at x -600 down to 600
///   at x -2047 (east 877 at 605 to 447 at 2314); between it and the bridge
///   (x -600..-553, 554..605) a gap at bank level;
/// - the forecourt gate is open for |x| < ~330 over z -2040..-2180 (walls
///   to 2800 either side), then the forecourt (803) to the doors at z -3155.
///
/// Planner ([`plan_from`], every frame): for each strip and landing z (and,
/// as fallback, run-up lines through gate aim points) the takeoff at the
/// from-a-run long jump's distance (bank height up to the railing), the
/// parapet cleared, the chained hop's landing past the gate; cost = frames
/// (run to the takeoff, turn, both hops, slide, walk to the door). Direct
/// lines (straight from Mario's position) need no turn: Mario heads
/// straight for the rail. The overlay plays it closed-loop: Z (crouch
/// slide) on the walking frame whose predicted long jump reaches the strip
/// along the heading on a line through the gate, Z + A, in hop 1 the air
/// stick (32 directions + neutral) whose simulated flight (quarter steps,
/// [`floor`]) lands nearest the strip's centre, Z + A on the strip, in hop 2
/// the stick whose landing + slide ends nearest the door, clear of the gate.
pub mod arc {
    use super::*;
    use crate::moves::arcs::{self, AirStick, ArcKind};

    /// Strip centre lines and half-widths (outside the box, on the railing).
    pub const STRIP: [f32; 2] = [-549.0, 550.0];
    const BAND: f32 = 3.5;
    /// The railing floor's outer edges (faces at x -553 / 554; find_floor
    /// truncates to integers, so the floor reaches x -553.99 / 554.99).
    const RAIL_OUT: [f32; 2] = [-554.0, 555.0];
    const DOOR_Z: f32 = -3155.0;
    /// The second landing's slide should end this far in front of the doors.
    const DOOR_FRONT: f32 = 100.0;
    /// Gate: max |x| for Mario's centre (radius 50 off the ~330 opening).
    const GATE_X: f32 = 285.0;
    const GATE_Z: (f32, f32) = (-2185.0, -2035.0);
    /// The parapet along the moat's south edge, and the bank behind it.
    const PARAPET_Z: (f32, f32) = (-162.0, -121.0);
    const GAP_X: [f32; 2] = [-600.0, 605.0];
    /// Planned run-up speed.
    const RUN: f32 = 32.0;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Kind {
        /// The railing top outside the box.
        Strip,
        /// A floor inside the box (caught).
        Box,
        Moat,
        /// Outside the box, not the strip (banks, bridge end, forecourt).
        Safe,
        /// A wall (the gate).
        Wall,
    }

    pub fn railing_y(z: f32) -> f32 {
        if z <= -300.0 { 964.0 + (z + 300.0) * 0.00412 } else { (838.0 + (-180.0 - z) * 1.05).clamp(814.0, 964.0) }
    }

    fn bridge_y(z: f32) -> f32 {
        if z > -800.0 {
            (817.0 + (-200.0 - z) * 0.15).clamp(811.0, 906.0)
        } else if z > -1500.0 {
            906.0
        } else {
            906.0 - (-1500.0 - z) * 0.164
        }
    }

    pub fn bank_y(x: f32) -> f32 {
        if x < -560.0 {
            811.0 - 0.2 * (-560.0 - x)
        } else if x > 560.0 {
            807.0 - 0.245 * (x - 560.0)
        } else {
            811.0
        }
    }

    /// The parapet's top at x, if it is there.
    fn parapet_top(x: f32) -> Option<f32> {
        if (-2047.0..=GAP_X[0]).contains(&x) {
            Some(600.0 + (x + 2047.0) * 0.1949)
        } else if (GAP_X[1]..=2314.0).contains(&x) {
            Some(877.0 - (x - GAP_X[1]) * 0.2516)
        } else {
            None
        }
    }

    fn forecourt_y(z: f32) -> f32 {
        (824.0 + (z - BOX_Z.0) * 0.167).max(803.0)
    }

    /// The floor model under (x, z).
    pub fn floor(x: f32, z: f32) -> (f32, Kind) {
        let ax = x.abs();
        let on_bridge = x > RAIL_OUT[0] && x < RAIL_OUT[1];
        if z >= PARAPET_Z.1 {
            (if on_bridge { 811.0 } else { bank_y(x) }, Kind::Safe)
        } else if z >= PARAPET_Z.0 {
            match parapet_top(x) {
                Some(y) => (y, Kind::Safe),
                None if x > -2047.0 && x < 2314.0 => (if on_bridge { 811.0 } else { 808.0 }, Kind::Safe),
                None => (-511.0, Kind::Moat),
            }
        } else if z >= BOX_Z.1 {
            if on_bridge { (811.0, Kind::Safe) } else { (-511.0, Kind::Moat) }
        } else if z > BOX_Z.0 {
            if ax < 456.0 {
                (bridge_y(z), Kind::Box)
            } else if on_bridge {
                let k = if x > BOX_X.0 && x < BOX_X.1 { Kind::Box } else { Kind::Strip };
                (railing_y(z), k)
            } else {
                (-511.0, Kind::Moat)
            }
        } else if z > GATE_Z.0 && z < GATE_Z.1 && ax > GATE_X {
            (2800.0, Kind::Wall)
        } else if on_bridge || z <= GATE_Z.0 {
            (forecourt_y(z), Kind::Safe)
        } else {
            (-511.0, Kind::Moat)
        }
    }

    /// A simulated long-jump flight.
    #[derive(Clone, Copy, Debug)]
    #[allow(dead_code)]
    pub struct Fly {
        pub frames: u32,
        pub pos: [f32; 3],
        pub fv: f32,
        pub kind: Kind,
        /// Hit a wall (the gate, a floor's side).
        pub blocked: bool,
    }

    /// The rest of a long jump from `st` with the stick held at world
    /// direction `dir` (radians, atan2(x, z)), or neutral: update_air_
    /// without_turn and perform_air_step's quarter steps on [`floor`]
    /// (a floor up to 78 above the feet is landed on, find_floor's reach).
    pub fn sim(st: &St, dir: Option<f32>) -> Fly {
        let yaw = st.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let (s, c) = yaw.sin_cos();
        let (fwd, side) = dir.map_or((0.0, 0.0), |d| ((d - yaw).cos(), (d - yaw).sin()));
        let (mut fv, mut vy, mut p) = (st.forward_vel, st.vel[1], st.pos);
        let mut blocked = false;
        for k in 1..=60u32 {
            fv = if fv > 0.0 { (fv - 0.35).max(0.0) } else { (fv + 0.35).min(0.0) };
            fv += 1.5 * fwd;
            let sv = 10.0 * side;
            if fv > 48.0 {
                fv -= 1.0;
            }
            if fv < -16.0 {
                fv += 2.0;
            }
            let v = [fv * s + sv * c, vy, fv * c - sv * s];
            for _ in 0..4 {
                for i in 0..3 {
                    p[i] += v[i] / 4.0;
                }
                let (fy, kind) = floor(p[0], p[2]);
                if kind == Kind::Wall {
                    if p[1] < fy {
                        blocked = true;
                    }
                } else if p[1] <= fy {
                    // Up to 78 below a floor find_floor still finds it and
                    // Mario lands on it, but below ~30 the wall check at
                    // y + 30 pushes him off its side first.
                    if fy - p[1] < 30.0 {
                        p[1] = fy;
                        return Fly { frames: k, pos: p, fv, kind, blocked };
                    }
                    return Fly { frames: k, pos: p, fv, kind: Kind::Wall, blocked: true };
                }
            }
            vy = (vy - 2.0).max(-75.0);
        }
        Fly { frames: 60, pos: p, fv, kind: Kind::Moat, blocked }
    }

    pub(super) fn side_ix(x: f32) -> usize {
        if x < 0.0 { 0 } else { 1 }
    }

    /// Distance of a long jump (from a run, or chained) landing `dy` above
    /// its takeoff, stick forward.
    pub fn hop_dist(fv: f32, dy: f32, chained: bool) -> Option<(u32, f32)> {
        let a = arcs::predict_with(ArcKind::LongJump, fv, 1, AirStick::Fwd, chained);
        arcs::land_on(&a, dy)
    }

    /// The planner's long-jump numbers at [`RUN`]: hop 1 (frames, distance)
    /// by dy / 5 (0..300), its (along, height) profile, hop 2 (frames,
    /// distance) from the railing down to the forecourt.
    struct Tables {
        /// By takeoff speed 10, 12, .. 32 (12 rows), then dy / 5.
        hop1: Vec<Vec<Option<(u32, f32)>>>,
        hop2: (u32, f32),
    }

    fn tables() -> &'static Tables {
        static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
        T.get_or_init(|| Tables {
            hop1: (0..12).map(|v| (0..=60).map(|i| hop_dist(10.0 + 2.0 * v as f32, i as f32 * 5.0, false)).collect()).collect(),
            hop2: hop_dist(48.0, 806.0 - 960.0, true).unwrap_or((34, 1700.0)),
        })
    }

    fn hop1_at(v: f32, dy: f32) -> Option<(u32, f32)> {
        let i = (dy / 5.0).round();
        if !(0.0..=60.0).contains(&i) {
            return None;
        }
        let r = (((v - 10.0) / 2.0).round().clamp(0.0, 11.0)) as usize;
        tables().hop1[r][i as usize]
    }

    /// Walking speed after `dist` more units from `v0` (update_walking_
    /// speed: +1.1 - v/43 a frame up to 32): (speed, frames).
    pub fn run_up(v0: f32, dist: f32) -> (f32, f32) {
        let (mut v, mut d, mut k) = (v0.clamp(0.0, 32.0), 0.0, 0.0);
        while d < dist && k < 200.0 {
            if v >= 32.0 {
                return (v, k + (dist - d) / 32.0);
            }
            v = (v + 1.1 - v / 43.0).min(32.0);
            d += v.max(1.0);
            k += 1.0;
        }
        (v, k)
    }

    /// Height of a long jump taken off at walking speed `v`, `along` units
    /// on: vy 30, gravity 2 (k (31 - k) after k air frames), at 1.5 v
    /// (capped 48) a frame after the crouch slide's frame.
    fn height_at(along: f32, v: f32) -> f32 {
        let k = (along / (1.5 * v).clamp(15.0, 48.0) - 1.0).max(0.0);
        k * (31.0 - k)
    }

    /// Ground under a takeoff point (outside the box, not the parapet).
    fn ground_at(x: f32, z: f32) -> Option<f32> {
        let on_bridge = x > RAIL_OUT[0] && x < RAIL_OUT[1];
        if z >= PARAPET_Z.1 {
            Some(if on_bridge { 811.0 } else { bank_y(x) })
        } else if z >= BOX_Z.1 && (on_bridge || (z >= PARAPET_Z.0 && x > GAP_X[0] && x < GAP_X[1])) {
            Some(if on_bridge { 811.0 } else { 808.0 })
        } else {
            None
        }
    }

    /// A planned skip line.
    #[derive(Clone, Copy, Debug)]
    #[allow(dead_code)]
    pub struct Line {
        pub side: usize,
        pub takeoff: [f32; 3],
        pub land: [f32; 3],
        /// Unit heading (x, z).
        pub h: [f32; 2],
        /// Estimated frames to the door.
        pub frames: f32,
        /// Run straight at the strip from here (no turn).
        pub direct: bool,
    }

    /// The hop line from takeoff `t` (on the ground) along unit `h` onto the
    /// strip of `side`: strip landing and frames from the takeoff to the
    /// door, if the line works (lands on the strip in range, clears the
    /// parapet, hop 2 passes the gate).
    fn hop_line(t: [f32; 3], h: [f32; 2], side: usize, v: f32) -> Option<([f32; 3], f32)> {
        let ex = STRIP[side];
        if h[0].abs() < 1e-3 || h[1] > -0.2 {
            return None;
        }
        let s = (ex - t[0]) / h[0];
        if s <= 0.0 {
            return None;
        }
        let zl = t[2] + s * h[1];
        if !(-1950.0..=-420.0).contains(&zl) {
            return None;
        }
        let ly = railing_y(zl);
        let (k1, _) = hop1_at(v, ly - t[1])?;
        // Parapet: over its whole width, above its top.
        if t[2] > PARAPET_Z.0 {
            for zc in [PARAPET_Z.1, (PARAPET_Z.0 + PARAPET_Z.1) / 2.0, PARAPET_Z.0] {
                if zc > t[2] {
                    continue;
                }
                let a = (zc - t[2]) / h[1];
                let xc = t[0] + a * h[0];
                if parapet_top(xc).is_some_and(|top| t[1] + height_at(a, v) < top + 10.0) {
                    return None;
                }
            }
        }
        // Hop 2 from the strip, same heading: past the gate, inside it.
        let (k2, d2) = tables().hop2;
        let x_at = |z: f32| ex + (z - zl) / h[1] * h[0];
        if x_at(GATE_Z.1).abs() > GATE_X + 60.0 || x_at(GATE_Z.0).abs() > GATE_X + 120.0 {
            return None;
        }
        let hz = zl + d2 * h[1];
        if hz > GATE_Z.0 + 100.0 {
            return None;
        }
        let ex2 = ex + (d2 + 290.0) * h[0];
        let ez2 = (zl + (d2 + 290.0) * h[1]).max(DOOR_Z + 60.0);
        let gx = if ex2 < 0.0 { -76.0 } else { 77.0 };
        let walk = (ex2 - gx).hypot(ez2 - (DOOR_Z + 60.0));
        Some(([ex, ly, zl], (k1 + k2) as f32 + 8.0 + walk / 28.0))
    }

    /// The fastest skip line from `p` (see the module doc).
    #[allow(dead_code)]
    pub fn plan_from(p: [f32; 3]) -> Option<Line> {
        plan_from_moving(p, 0.0, None)
    }

    /// [`plan_from`] for Mario moving at `v0` facing `yaw` (radians): the
    /// run-up speed and the turn onto each line count.
    pub fn plan_from_moving(p: [f32; 3], v0: f32, yaw: Option<f32>) -> Option<Line> {
        plan_from_moving_in(p, v0, yaw, None, None)
    }

    /// [`plan_from_moving`] on one strip only (`side`) and with the takeoff
    /// speed in `band` (OBRRT_FREE_SETUP's variants); None, None is
    /// [`plan_from_moving`] itself.
    pub fn plan_from_moving_in(p: [f32; 3], v0: f32, yaw: Option<f32>, only: Option<usize>, band: Option<(f32, f32)>) -> Option<Line> {
        // Frames for turning from the facing onto `h` (11.25 degrees a
        // frame walking, a skid beyond ~100), and the speed kept.
        let turn = |h: [f32; 2]| -> (f32, f32) {
            let Some(y) = yaw else { return (0.0, 0.0) };
            let c = (y.sin() * h[0] + y.cos() * h[1]).clamp(-1.0, 1.0);
            let a = c.acos();
            if a > 1.75 { (12.0 + a / 0.2, 0.0) } else { (a / 0.2, v0 * c.max(0.0)) }
        };
        let mut best: Option<Line> = None;
        let mut keep = |l: Line| {
            if best.is_none_or(|b| l.frames < b.frames) {
                best = Some(l);
            }
        };
        for (side, &ex) in STRIP.iter().enumerate() {
            if only.is_some_and(|o| o != side) {
                continue;
            }
            for zi in 0..=61 {
                let zl = -420.0 - 25.0 * zi as f32;
                let ly = railing_y(zl);
                // Direct: the line from Mario through the strip point.
                let (dx, dz) = (ex - p[0], zl - p[2]);
                let n = dx.hypot(dz);
                if n > 1.0 {
                    let h = [dx / n, dz / n];
                    // Takeoff at the hop's distance back from the strip, at
                    // the speed the run there reaches.
                    let (tf, v1) = turn(h);
                    let mut d1 = 1000.0;
                    let mut ok = None;
                    let mut run_f = 0.0;
                    let mut vt = RUN;
                    for _ in 0..3 {
                        let (tx, tz) = (ex - d1 * h[0], zl - d1 * h[1]);
                        let Some(ty) = ground_at(tx, tz) else { break };
                        let (v, k) = run_up(v1, (n - d1).max(0.0));
                        let Some((_, d)) = hop1_at(v, ly - ty) else { break };
                        d1 = d;
                        run_f = k;
                        vt = v;
                        ok = Some([ex - d1 * h[0], ty, zl - d1 * h[1]]);
                    }
                    if let Some(t) = ok
                        && ground_at(t[0], t[2]).is_some()
                        && n >= d1 - 300.0
                        && band.is_none_or(|(lo, hi)| vt >= lo && vt < hi)
                    {
                        let t = if n < d1 { [p[0], p[1], p[2]] } else { t };
                        if let Some((land, f)) = hop_line(t, h, side, vt) {
                            keep(Line { side, takeoff: t, land, h, frames: f + run_f + tf, direct: true });
                        }
                    }
                }
            }
        }
        best
    }

    /// Per side, the fastest line from a full-speed run-up (aimed through
    /// the gate): the fallback when no direct line works from Mario's spot
    /// (the bridge's middle, too close to the parapet, too slow).
    pub fn canonical(side: usize) -> Line {
        static C: std::sync::OnceLock<[Line; 2]> = std::sync::OnceLock::new();
        C.get_or_init(|| {
            [0, 1].map(|side| {
                let ex = STRIP[side];
                let mut best: Option<Line> = None;
                for zi in 0..=61 {
                    let zl = -420.0 - 25.0 * zi as f32;
                    let ly = railing_y(zl);
                    for gi in 0..=24 {
                        let gx = -240.0 + 20.0 * gi as f32;
                        let (gdx, gdz) = (gx - ex, -2110.0 - zl);
                        let gn = gdx.hypot(gdz);
                        let h = [gdx / gn, gdz / gn];
                        let mut d1 = 1000.0;
                        let mut t = None;
                        for _ in 0..3 {
                            let (tx, tz) = (ex - d1 * h[0], zl - d1 * h[1]);
                            let Some(ty) = ground_at(tx, tz) else { break };
                            let Some((_, d)) = hop1_at(RUN, ly - ty) else { break };
                            d1 = d;
                            t = Some([ex - d1 * h[0], ty, zl - d1 * h[1]]);
                        }
                        let Some(t) = t else { continue };
                        // A run-up on the bank behind the takeoff.
                        let back = [t[0] - 700.0 * h[0], t[2] - 700.0 * h[1]];
                        if ground_at(t[0], t[2]).is_none() || ground_at(back[0], back[1]).is_none() {
                            continue;
                        }
                        let Some((land, f)) = hop_line(t, h, side, RUN) else { continue };
                        if best.is_none_or(|b| f < b.frames) {
                            best = Some(Line { side, takeoff: t, land, h, frames: f, direct: false });
                        }
                    }
                }
                best.expect("a canonical lakitu-skip line")
            })
        })[side]
    }

    pub(super) fn airborne(s: &St) -> bool {
        s.action & arcs::ACT_FLAG_AIR != 0
    }

    /// On the ground in the box's x range about to walk into it (a long
    /// jump needs the crouch slide's frame first).
    pub fn caught_soon(s: &St) -> bool {
        !airborne(s)
            && s.pos[0] > BOX_X.0
            && s.pos[0] < BOX_X.1
            && s.pos[2] >= BOX_Z.1
            && s.pos[2] + 2.0 * s.vel[2] < BOX_Z.1
    }

    /// In the first hop, over the moat beside the bridge.
    pub fn hop1_over_moat(s: &St) -> bool {
        grounds(s) && s.action == ACT_LONG_JUMP && s.pos[0].abs() > 545.0 && s.pos[2] < BOX_Z.1 && s.pos[2] > BOX_Z.0
    }

    pub(super) fn on_strip(s: &St) -> bool {
        s.pos[2] < BOX_Z.1 && s.pos[2] > BOX_Z.0 && !in_box(s) && s.pos[0].abs() < 556.0 && s.pos[1] > 900.0
    }

    /// The door aim for a side's second hop.
    pub(super) fn aim(x: f32) -> [f32; 3] {
        [if x < 0.0 { 77.0 } else { -76.0 }, 803.0, DOOR_Z]
    }

    pub(super) fn plan_of(s: &St) -> Option<Line> {
        let moving = !airborne(s) && s.forward_vel > 1.0;
        let yaw = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
        plan_from_moving(s.pos, s.forward_vel.max(0.0), moving.then_some(yaw))
    }

    /// Where to steer (the stick in the hops comes from [`overlay`]): before
    /// the box, straight at the planned strip point (direct line) or onto
    /// the planned run-up line.
    pub fn target(s: &St) -> [f32; 3] {
        if !grounds(s) || s.pos[2] <= BOX_Z.0 + 100.0 {
            return DOOR;
        }
        if airborne(s) || s.pos[2] < BOX_Z.1 {
            return aim(s.pos[0]);
        }
        if let Some(l) = plan_of(s) {
            return [l.land[0], s.pos[1], l.land[2]];
        }
        // No direct line: onto the side's canonical line well behind its
        // takeoff, then along it (pure pursuit) until a direct line works.
        let c = canonical(side_ix(s.pos[0]));
        let t = c.takeoff;
        let (dx, dz) = (s.pos[0] - t[0], s.pos[2] - t[2]);
        let a = -(dx * c.h[0] + dz * c.h[1]);
        let e = (dx * c.h[1] - dz * c.h[0]).abs();
        let along = if a > 150.0 && e < 150.0 + 0.2 * a { a - 350.0 } else { 700.0 };
        [t[0] - along * c.h[0], s.pos[1], t[2] - along * c.h[1]]
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(super) enum Hop {
        One,
        Two,
    }

    pub(super) fn hop_of(traj: &[St]) -> Hop {
        let s = traj.last().unwrap();
        match traj.iter().rev().find(|g| !airborne(g)) {
            Some(g) if g.pos[2] < BOX_Z.1 - 40.0 => Hop::Two,
            Some(_) => Hop::One,
            None if s.pos[0].abs() > 545.0 && s.pos[2] < BOX_Z.1 => Hop::One,
            None if s.pos[2] < BOX_Z.1 => Hop::Two,
            None => Hop::One,
        }
    }

    pub(super) fn hop1_cost(f: &Fly, side: usize) -> f32 {
        let dx = (f.pos[0] - STRIP[side]).abs();
        let mut c = if f.kind == Kind::Strip && dx <= BAND { 0.01 * f.pos[2] + 2.0 * dx } else { 1000.0 + 10.0 * dx };
        if f.kind == Kind::Box {
            c += 500.0;
        }
        if f.blocked {
            c += 1e6;
        }
        c
    }

    pub(super) fn hop2_cost(f: &Fly, yaw: f32) -> f32 {
        let mut c = 0.0;
        if f.blocked {
            c += 1e6;
        }
        match f.kind {
            Kind::Safe if f.pos[2] < BOX_Z.0 => {
                let lag = 5.7 * f.fv;
                let ex = f.pos[0] + lag * yaw.sin();
                let ez = (f.pos[2] + lag * yaw.cos()).max(DOOR_Z + 60.0);
                let gx = if ex < 0.0 { -76.0 } else { 77.0 };
                c + (ex - gx).hypot(ez - (DOOR_Z + DOOR_FRONT))
            }
            Kind::Strip | Kind::Safe => c + 5e4 - f.pos[2].min(0.0).abs(),
            _ => c + 1e5 - f.pos[2].min(0.0).abs(),
        }
    }

    /// The best stick for the rest of a long jump by `cost` over 32 world
    /// directions (relative to the facing) and neutral.
    pub(super) fn best_stick(s: &St, cost: impl Fn(&Fly) -> f32) -> (i8, i8, Fly) {
        let yaw = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let mut best = (f32::MAX, None, sim(s, None));
        for k in 0..33 {
            let dir = if k == 32 { None } else { Some(yaw + k as f32 * std::f32::consts::TAU / 32.0) };
            let f = sim(s, dir);
            let c = cost(&f);
            if c < best.0 {
                best = (c, dir, f);
            }
        }
        let (x, y) = match best.1 {
            Some(d) => arcs::stick(s.cam_yaw, rad_to_yaw(d), 1.0),
            None => (0, 0),
        };
        (x, y, best.2)
    }

    pub fn debug() -> bool {
        static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *D.get_or_init(|| obrrt_core::sys::env::var("LAKITU_DEBUG").is_ok())
    }

    pub(super) fn rad_to_yaw(r: f32) -> i16 {
        ((r * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
    }

    /// Walking before the box: the frame to start the crouch slide (the
    /// predicted long jump from here, on a line that works, reaches the
    /// strip along the heading, with up to ~300 to take off in the air).
    pub fn takeoff_now(s: &St) -> bool {
        takeoff_now_in(s, None, None)
    }

    /// [`takeoff_now`] onto one strip (`only`) at a takeoff speed in `band`.
    pub fn takeoff_now_in(s: &St, only: Option<usize>, band: Option<(f32, f32)>) -> bool {
        if band.is_some_and(|(lo, hi)| s.forward_vel < lo || s.forward_vel >= hi) {
            return false;
        }
        if s.forward_vel < 18.0 || s.pos[2] < BOX_Z.1 {
            return false;
        }
        let yaw = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let h = [yaw.sin(), yaw.cos()];
        // In the box's x range the crouch slide's frame must stay before it.
        if s.pos[0] > BOX_X.0 && s.pos[0] < BOX_X.1 && s.pos[2] + 2.0 * s.vel[2] < BOX_Z.1 {
            return false;
        }
        (0..2).filter(|&side| only.is_none_or(|o| o == side)).any(|side| {
            let Some((land, _)) = hop_line(s.pos, h, side, s.forward_vel) else { return false };
            let t = (land[0] - s.pos[0]).hypot(land[2] - s.pos[2]);
            let Some((_, d1)) = hop_dist(s.forward_vel, land[1] - s.pos[1], false) else { return false };
            t <= d1 + 10.0 && t >= d1 - 300.0
        })
    }

    /// Per-frame pad: the hops' buttons and air stick (see the module doc).
    pub fn overlay(traj: &[St], played: &[GamePad], genome: GamePad) -> GamePad {
        use buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
        let Some(s) = traj.last() else { return genome };
        if !grounds(s) || !s.controllable() || s.pos[2] < BOX_Z.0 - 300.0 {
            return genome;
        }
        let last = played.last().copied().unwrap_or(GamePad::new(0, 0, 0));
        let (last_a, last_z) = (last.button & A_BUTTON != 0, last.button & Z_TRIG != 0);
        let mut pad = genome;
        let set = |pad: &mut GamePad, b: u16| pad.button = (pad.button & !(A_BUTTON | Z_TRIG | B_BUTTON)) | b;
        let facing = |pad: &mut GamePad| {
            let (x, y) = arcs::stick(s.cam_yaw, s.yaw, 1.0);
            pad.stick_x = x;
            pad.stick_y = y;
        };
        match s.action {
            ACT_LONG_JUMP => {
                set(&mut pad, Z_TRIG);
                let yaw = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
                let hop = hop_of(traj);
                let (x, y, fl) = match hop {
                    Hop::One => best_stick(s, |f| hop1_cost(f, side_ix(s.pos[0]))),
                    Hop::Two => best_stick(s, |f| hop2_cost(f, yaw)),
                };
                if debug() {
                    eprintln!("DBG {hop:?} pos {:?} -> stick ({x},{y}) predicts {:?}", s.pos, fl);
                }
                pad.stick_x = x;
                pad.stick_y = y;
            }
            // On the strip, or a parapet top before the box: chain.
            ACT_LONG_JUMP_LAND if on_strip(s) || s.pos[2] >= BOX_Z.1 => {
                set(&mut pad, if last_a { Z_TRIG } else { Z_TRIG | A_BUTTON });
                facing(&mut pad);
            }
            ACT_CROUCH_SLIDE if s.pos[2] >= BOX_Z.1 && s.forward_vel > 10.0 => {
                set(&mut pad, if last_a { Z_TRIG } else { Z_TRIG | A_BUTTON });
                facing(&mut pad);
            }
            ACT_WALKING if takeoff_now(s) => {
                set(&mut pad, if last_z { 0 } else { Z_TRIG });
                facing(&mut pad);
            }
            ACT_WALKING | arcs::ACT_IDLE | arcs::ACT_DECELERATING if s.pos[2] >= BOX_Z.1 => {
                // Near the takeoff: the stick on the plan, no buttons.
                let near = |t: [f32; 3], r: f32| (s.pos[0] - t[0]).hypot(s.pos[2] - t[2]) < r;
                let c = canonical(side_ix(s.pos[0])).takeoff;
                if near(c, 700.0) || (near(c, 1600.0) && plan_of(s).is_some_and(|l| near(l.takeoff, 700.0))) {
                    let t = target(s);
                    let d = (t[0] - s.pos[0]).atan2(t[2] - s.pos[2]);
                    let (x, y) = arcs::stick(s.cam_yaw, rad_to_yaw(d), 1.0);
                    pad.stick_x = x;
                    pad.stick_y = y;
                    set(&mut pad, 0);
                }
            }
            _ => {}
        }
        pad
    }

    /// Edge / Air with the switch: straight for the nearer strip from
    /// anywhere before the box, and at the door once a jump's landing is
    /// predicted past the box.
    pub fn edge_air_target(air: bool, s: &St) -> Option<[f32; 3]> {
        if !grounds(s) {
            return None;
        }
        let ex = side(s);
        if airborne(s) && s.pos[2] < BOX_Z.1 + 50.0 && s.pos[2] > BOX_Z.0 - 100.0 {
            let (_, land, _) = arcs::fly(s, 806.0, 1.0, 0.0);
            if land[2] < BOX_Z.0 - 30.0 {
                return Some(aim(s.pos[0]));
            }
        }
        if s.pos[2] > BOX_Z.1 && !near_bridge(s) {
            return Some(if air { [ex, 960.0, AIR_LAND_Z] } else { [ex, 830.0, BOX_Z.1 - 20.0] });
        }
        None
    }
}

/// The Hill mode (`OBRRT_LAKITU_ARC=1`), after the 2026 120-star TAS's
/// grounds hill (ref/tas/slope_techniques.md): a crouch-slide long jump
/// onto the hill's very slippery (0x13) west face, A+B+Z on the landing (a
/// steep jump turned into a rising dive), the dive slide down the face
/// (48 → ~100 u/f; the facing turns 0x200 a frame towards the slide, so
/// it ends near −z while the slide runs west), a rollout on the first flat
/// frame, a landing-action surf up the ramp (A held + B with a short stick
/// on FREEFALL_LAND_STOP / FREEFALL_LAND's 4th frame: jump kicks that land
/// at once, no friction, no 48 cap), Z then Z+B at the top (slide kick), A
/// on the slide-kick slide (rollout), a jump kick on its landing, and on
/// that FREEFALL_LAND a fresh A: a double jump at ~90 (vy 52 + fv/4, range
/// ~37.4 fv − 500) over the whole box, A held, landing near the door. If
/// the double jump would land short of the box's end, another jump-kick
/// hop first. Everything is event-driven from the action (and the frames
/// spent in it), with the stick at world −z, except the approach and the
/// slide, which steer closed-loop.
pub mod hill {
    use super::*;
    use crate::moves::arcs;
    use buttons::{A_BUTTON, B_BUTTON, Z_TRIG};

    const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
    const ACT_DIVE: u32 = 0x0188_088A;
    const ACT_STEEP_JUMP: u32 = 0x0300_0885;
    const ACT_FORWARD_ROLLOUT: u32 = 0x0100_08A6;
    const ACT_FREEFALL_LAND: u32 = 0x0400_0471;
    const ACT_FREEFALL_LAND_STOP: u32 = 0x0C00_0232;
    const ACT_JUMP_KICK: u32 = 0x0180_08AC;
    const ACT_SLIDE_KICK: u32 = 0x0180_08AA;
    const ACT_SLIDE_KICK_SLIDE: u32 = 0x0080_045A;
    const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;
    const ACT_PUNCHING: u32 = 0x0080_0380;

    /// The TAS's run-up: the crouch slide at C facing RUN_YAW (its long
    /// jump lands on the face at (89, 377, 4463)).
    pub const C: [f32; 3] = [-79.4, 260.0, 4558.6];
    /// The slide's exit at the hill's foot (the TAS's first flat frame).
    const EXIT: [f32; 2] = [-38.0, 3882.0];
    pub const RUN_YAW: i16 = 23818;
    /// The approach starts this far behind C on the run-up line.
    /// Not yet behind C (the grounds' south wall is ~400 behind it, so
    /// there is no room to turn in behind it): first this point west of it,
    /// from where the pursuit curves onto the run-up line as from the spawn.
    pub(super) const WEST: [f32; 2] = [-1400.0, 4400.0];
    /// South of this the chain is on the ramp (surf), north on the bridge
    /// approach (jump kick, double jump).
    pub(super) const RAMP_Z: f32 = 2500.0;
    /// World yaw towards the castle.
    const NORTH: i16 = 0x8000u16 as i16;

    pub(super) fn dir(yaw: i16) -> [f32; 2] {
        let r = yaw as f32 * std::f32::consts::TAU / 65536.0;
        [r.sin(), r.cos()]
    }

    pub(super) fn yaw_of(dx: f32, dz: f32) -> i16 {
        ((dx.atan2(dz) * 65536.0 / std::f32::consts::TAU).round() as i64 as i32) as i16
    }

    /// Frames the last state has been in its action (1 = just entered),
    /// counting from the last B press (a landing refresh restarts it).
    fn in_action(traj: &[St], played: &[GamePad]) -> usize {
        let a = traj.last().map_or(0, |s| s.action);
        let mut n = 0;
        for i in (0..traj.len()).rev() {
            if traj[i].action != a {
                break;
            }
            n += 1;
            // traj[i] is the state after played[i - 1].
            if i >= 1 && played.get(i - 1).is_some_and(|p| p.button & B_BUTTON != 0) {
                break;
            }
        }
        n
    }

    // --- The hops' air physics (decomp: `update_air_without_turn`,
    // `perform_air_step`, `apply_gravity`; mario_actions_airborne.c,
    // mario_step.c). Checked against 2033 double jumps, 2954 jump kicks and
    // 4152 slide kicks of the hill-bottom fuzz: landing z within +-0.5 (+-18 on
    // the odd step that lands on the other side of a floor-table kink) for the
    // double jump, within ~24 (the kick's first frame) and ~20 for the others.

    /// Floor height of the bridge approach near x = 0 by z (ascending;
    /// measured on landings; the box's hump is only ever a failed landing).
    const FLOOR_Z: [(f32, f32); 18] = [
        (-2127.0, 803.0), (-1900.0, 842.0), (-500.0, 859.0),
        (-400.0, 846.0), (-300.0, 830.0), (-200.0, 817.0), (-100.0, 811.0), (0.0, 810.0), (200.0, 809.0),
        (300.0, 808.0), (400.0, 807.0), (600.0, 806.0), (800.0, 805.0), (1000.0, 804.0), (1400.0, 799.0),
        (1450.0, 796.0), (1550.0, 785.0), (1700.0, 772.0),
    ];

    fn floor_at(z: f32) -> f32 {
        let t = &FLOOR_Z;
        if z <= t[0].0 {
            return t[0].1;
        }
        for w in t.windows(2) {
            if z <= w[1].0 {
                return w[0].1 + (w[1].1 - w[0].1) * (z - w[0].0) / (w[1].0 - w[0].0);
            }
        }
        t[t.len() - 1].1
    }

    fn rad(a: i32) -> f32 {
        (a as i16) as f32 * std::f32::consts::TAU / 65536.0
    }

    /// `intendedMag` of a raw stick (adjust_analog_stick: the +-8 dead zone
    /// takes 6 off, magnitude capped at 64; then (mag/64)^2 * 32).
    fn intended_mag(x: i8, y: i8) -> f32 {
        let adj = |v: i8| {
            let v = v as f32;
            if v > -8.0 && v < 8.0 { 0.0 } else if v > 0.0 { v - 6.0 } else { v + 6.0 }
        };
        let m = adj(x).hypot(adj(y)).min(64.0);
        m * m / 128.0
    }

    struct Landing {
        x: f32,
        z: f32,
        fv: f32,
    }

    /// Flight from `p` (vertical speed `vy`, forward speed `fv`, facing
    /// `face`; gravity `grav`) to the first ground contact: the stick is
    /// `first_mag` (intendedMag) on the first frame, then full towards
    /// north (or the door). `bounce`: the slide kick's first landing
    /// (vel[1] < 0 -> -vel[1] / 2, the action stays airborne).
    fn fly(mut p: [f32; 3], mut vy: f32, mut fv: f32, face: i16, grav: f32, bounce: bool, first_mag: f32, door: bool) -> Option<Landing> {
        let mut bounced = false;
        for fr in 0..90 {
            let want = if door && fr > 0 { yaw_of(-40.0 - p[0], DOOR[2] - p[2]) } else { NORTH };
            let dyaw = want as i32 - face as i32;
            let mag = if fr == 0 { first_mag } else { 32.0 } / 32.0;
            fv = if fv > 0.35 { fv - 0.35 } else if fv < -0.35 { fv + 0.35 } else { 0.0 };
            fv += mag * rad(dyaw).cos() * 1.5;
            let side = mag * rad(dyaw).sin() * 10.0;
            if fv > 32.0 {
                fv -= 1.0;
            }
            if fv < -16.0 {
                fv += 2.0;
            }
            let (sf, cf) = (rad(face as i32).sin(), rad(face as i32).cos());
            let (sp, cp) = (rad(face as i32 + 0x4000).sin(), rad(face as i32 + 0x4000).cos());
            let (vx, vz) = (fv * sf + side * sp, fv * cf + side * cp);
            let mut landed = false;
            for _ in 0..4 {
                let (nx, ny, nz) = (p[0] + vx / 4.0, p[1] + vy / 4.0, p[2] + vz / 4.0);
                let fl = floor_at(nz);
                if ny <= fl {
                    p = [nx, fl, nz];
                    landed = true;
                    break;
                }
                p = [nx, ny, nz];
            }
            vy = (vy - grav).max(-75.0);
            if landed {
                if bounce && !bounced && vy < 0.0 {
                    vy = -vy / 2.0;
                    bounced = true;
                    continue;
                }
                return Some(Landing { x: p[0], z: p[2], fv });
            }
        }
        None
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Hop {
        Double,
        Kick,
        Slide,
        /// Speed kick (ground B with fv >= 29, stick > 48): ACT_DIVE, vy 20,
        /// fv = min(fv + 15, 48) (set_mario_action_airborne), gravity 4.
        Dive,
    }

    /// `OBRRT_HILL_DIVE=1`: the dive-drop option (off by default; see
    /// policy/NOTES-divedrop.md).
    fn dive_enabled() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| obrrt_core::sys::env::var("OBRRT_HILL_DIVE").is_ok_and(|v| v != "0"))
    }

    /// Where hop `h`, started now from the ground state `s`, lands.
    fn hop_land(h: Hop, s: &St) -> Option<Landing> {
        let fv = s.forward_vel;
        match h {
            // set_mario_action_airborne: vy 52 + fv/4, fv x 0.8.
            Hop::Double => fly(s.pos, 52.0 + 0.25 * fv, 0.8 * fv, s.yaw, 4.0, false, 32.0, true),
            // vy 20; the first frame's stick is `kick()`'s.
            Hop::Kick => {
                let (x, y) = arcs::stick(s.cam_yaw, NORTH, KICK_TILT);
                fly(s.pos, 20.0, fv, s.yaw, 4.0, false, intended_mag(x, y), false)
            }
            // vy 12, fv >= 32, gravity 2 (apply_gravity), bounce on the first landing.
            Hop::Slide => fly(s.pos, 12.0, fv.max(32.0), s.yaw, 2.0, true, 32.0, false),
            Hop::Dive => fly(s.pos, 20.0, (fv + 15.0).min(48.0), s.yaw, 4.0, false, 32.0, false),
        }
    }

    /// The stick tilt of the jump kick's first frame (stickMag ~46 < 48:
    /// not a speed-kick dive).
    const KICK_TILT: f32 = 0.38;
    /// The sim's worst error on the jump kick / slide kick (units).
    const HOP_ERR: f32 = 30.0;
    /// ... on the double jump.
    const DJ_ERR: f32 = 20.0;

    /// After one ground frame of a landing (apply_landing_accel: 0.98 per
    /// frame).
    fn after_landing_frame(s: &St) -> St {
        let fv = 0.98 * s.forward_vel;
        St { pos: [s.pos[0] + fv * rad(s.yaw as i32).sin(), s.pos[1], s.pos[2] + fv * rad(s.yaw as i32).cos()], forward_vel: fv, ..s.clone() }
    }

    /// The crouch slide's first frame (Z on the landing's last frame; the
    /// slide's own speed loss, ~4%).
    fn after_crouch_frame(s: &St) -> St {
        let fv = 0.959 * s.forward_vel;
        St { pos: [s.pos[0] + fv * rad(s.yaw as i32).sin(), s.pos[1], s.pos[2] + fv * rad(s.yaw as i32).cos()], forward_vel: fv, ..s.clone() }
    }

    /// The hop lands outside the trigger box (before it, or past it).
    fn clear_of_box(z: f32) -> bool {
        z > BOX_Z.1 + HOP_ERR || z < BOX_Z.0 - HOP_ERR
    }

    /// The double jump from here clears the box (by `err`) and doesn't hit
    /// the castle (fv ≤ ~105; x −400..500; outside the box).
    fn dj_ok_by(s: &St, err: f32) -> bool {
        s.pos[2] > BOX_Z.1 + 2.0
            && s.forward_vel < 106.0
            && s.pos[0] > -400.0
            && s.pos[0] < 500.0
            && hop_land(Hop::Double, s).is_some_and(|l| l.z < BOX_Z.0 - err)
    }

    /// ... by the sim's error: a quarter step (~fv/4) on the odd landing.
    fn dj_ok(s: &St) -> bool {
        dj_ok_by(s, DJ_ERR)
    }

    /// Single jump range (from a rollout's LAND_STOP): vy 42 + fv/4, fv x0.8.
    fn single_range(fv: f32) -> f32 {
        0.8 * fv * (2.0 * (42.0 + fv / 4.0) / 4.0) * 1.05
    }

    /// Where an airborne Mario (gravity 4) comes down to `floor`.
    fn land_z(s: &St, floor: f32) -> f32 {
        let (mut y, mut vy, mut z) = (s.pos[1], s.vel[1], s.pos[2]);
        for _ in 0..60 {
            y += vy;
            z += s.vel[2];
            if y <= floor && vy < 0.0 {
                break;
            }
            vy = (vy - 4.0).max(-75.0);
        }
        z
    }

    /// The rollout before the bridge: kick then double jump (true), or a
    /// single jump straight from the landing (false)?
    fn kick_plan(s: &St) -> bool {
        let z = land_z(s, s.floor_y.max(700.0));
        let fv = s.forward_vel;
        // The kick from the rollout's landing, then the double jump from the
        // kick's (pessimistically: HOP_ERR short) landing.
        let g = St { pos: [s.pos[0], floor_at(z), z], forward_vel: fv, vel: [0.0; 3], ..s.clone() };
        if let Some(k) = hop_land(Hop::Kick, &g) {
            let g2 = St { pos: [k.x, floor_at(k.z), k.z + HOP_ERR], forward_vel: k.fv, ..g.clone() };
            if k.z > BOX_Z.1 + HOP_ERR && dj_ok(&g2) {
                return true;
            }
        }
        !(z > BOX_Z.1 + 5.0 && z - single_range(fv) < BOX_Z.0 - 30.0)
    }

    /// Steer target: the run-up line, then (in the chain) the castle.
    pub fn target(s: &St) -> [f32; 3] {
        if !grounds(s) || s.pos[2] < BOX_Z.1 || s.forward_vel > 50.0 {
            return [-40.0, 803.0, DOOR[2]];
        }
        let d = dir(RUN_YAW);
        let (dx, dz) = (s.pos[0] - C[0], s.pos[2] - C[2]);
        let a = -(dx * d[0] + dz * d[1]);
        let e = (dx * d[1] - dz * d[0]).abs();
        let _ = e;
        // Behind C: pure pursuit on the line; in front: back round to
        // BACK behind it.
        let aligned = ((s.yaw as i32 - RUN_YAW as i32) as i16 as i32).abs() < 0x1800 && s.forward_vel > 10.0;
        // Only with room to reach full walking speed by C (else round again).
        let speed_ok = a <= 0.0 || super::arc::run_up(s.forward_vel.max(0.0), a).0 >= 31.0;
        if speed_ok && (a > 250.0 || (a > -60.0 && aligned)) {
            let along = a - (0.5 * a).clamp(150.0, 600.0);
            return [C[0] - along * d[0], s.pos[1], C[2] - along * d[1]];
        }
        [WEST[0], s.pos[1], WEST[1]]
    }

    /// Per-frame pad (see the module doc).
    pub fn overlay(traj: &[St], played: &[GamePad], genome: GamePad) -> GamePad {
        let Some(s) = traj.last() else { return genome };
        if !grounds(s) || !s.controllable() || s.pos[2] < BOX_Z.0 - 200.0 {
            return genome;
        }
        let last = played.last().copied().unwrap_or(GamePad::new(0, 0, 0));
        let (last_a, last_z) = (last.button & A_BUTTON != 0, last.button & Z_TRIG != 0);
        let pad = |b: u16, yaw: i16, tilt: f32| {
            let (x, y) = arcs::stick(s.cam_yaw, yaw, tilt);
            GamePad::new(b, x, y)
        };
        let fast = s.forward_vel > 55.0;
        let north = |b: u16| pad(b, NORTH, 1.0);
        // A held + B with a short stick (jump kick / landing refresh); A
        // first if it isn't held yet.
        let kick = || if last_a { pad(A_BUTTON | B_BUTTON, NORTH, 0.38) } else { north(A_BUTTON) };
        // The same presses with the stick past 48: a speed kick (dive).
        let dive = || if last_a { pad(A_BUTTON | B_BUTTON, NORTH, 1.0) } else { north(A_BUTTON) };
        match s.action {
            // --- Run-up: crouch slide at C, long jump onto the face.
            arcs::ACT_WALKING if s.pos[2] > RAMP_Z + 1500.0 && !fast => {
                let d = dir(RUN_YAW);
                let ahead = (C[0] - s.pos[0]) * d[0] + (C[2] - s.pos[2]) * d[1];
                let near_line = ((s.pos[0] - C[0]) * d[1] - (s.pos[2] - C[2]) * d[0]).abs() < 40.0;
                let on_yaw = ((s.yaw as i32 - RUN_YAW as i32) as i16 as i32).abs() < 0x300;
                if near_line && on_yaw && s.forward_vel > 30.5 && ahead < 0.95 * s.forward_vel && ahead > -10.0 {
                    return pad(if last_z { 0 } else { Z_TRIG }, s.yaw, 1.0);
                }
                // The last stretch: onto the line (lookahead 150), then
                // the exact run-up yaw for the final frames.
                let (dx, dz) = (s.pos[0] - C[0], s.pos[2] - C[2]);
                let e = dx * d[1] - dz * d[0];
                let aligned = ((s.yaw as i32 - RUN_YAW as i32) as i16 as i32).abs() < 0x2000 && s.forward_vel > 10.0;
                if aligned && e.abs() < 120.0 && ahead > 0.0 && ahead < 700.0 {
                    let corr = if ahead > 2.0 * s.forward_vel { -(e / 150.0).atan() } else { 0.0 };
                    let want = RUN_YAW as i32 + (corr * 65536.0 / std::f32::consts::TAU) as i32;
                    if super::arc::debug() {
                        eprintln!("HILL run pos {:?} ahead {ahead:.0} e {e:.1} yaw {:#x} want {want:#x}", s.pos, s.yaw as u16);
                    }
                    return pad(0, want as i16, 1.0);
                }
                genome
            }
            arcs::ACT_CROUCH_SLIDE if s.pos[2] > RAMP_Z + 1500.0 && !fast && ((s.yaw as i32 - RUN_YAW as i32) as i16 as i32).abs() < 0x400 => {
                pad(if last_a { Z_TRIG } else { Z_TRIG | A_BUTTON }, s.yaw, 1.0)
            }
            arcs::ACT_LONG_JUMP if s.pos[2] > RAMP_Z + 1500.0 => pad(Z_TRIG, s.yaw, 1.0),
            // On the face: A + B + Z (steep jump → rising dive).
            arcs::ACT_LONG_JUMP_LAND if s.floor_type == 0x13 => pad(A_BUTTON | B_BUTTON | Z_TRIG, s.yaw, 1.0),
            ACT_STEEP_JUMP | ACT_DIVE if s.pos[2] > RAMP_Z + 1000.0 && !fast => pad(A_BUTTON | B_BUTTON | Z_TRIG, s.yaw, 1.0),
            // --- Slide down the face: the stick at the TAS's exit point at
            // the foot (along the slide velocity loses less speed but ends
            // too far east), A on the first flat frame.
            // Dive-drop: the dive slide on the bridge approach: A at once (rollout).
            // (a dive-drop's slide has fv <= 48; the ramp's is faster)
            ACT_DIVE_SLIDE if dive_enabled() && s.forward_vel <= 52.0 && s.pos[2] < RAMP_Z && s.pos[2] > BOX_Z.1 - 50.0 => north(if last_a { 0 } else { A_BUTTON }),
            ACT_DIVE_SLIDE => {
                if s.floor_normal[1] > 0.99 && s.forward_vel > 60.0 {
                    return pad(if last_a { 0 } else { A_BUTTON }, s.yaw, 1.0);
                }
                pad(0, yaw_of(EXIT[0] - s.pos[0], EXIT[1] - s.pos[2]), 1.0)
            }
            // --- Air phases at speed: A held (for the landing's jump kick)
            // except a jump kick whose landing should take the double jump.
            // Before the bridge: A held for a jump kick on the landing, or
            // released for a single jump from it.
            ACT_FORWARD_ROLLOUT if fast && s.pos[2] < RAMP_Z && !kick_plan(s) => north(0),
            ACT_FORWARD_ROLLOUT if fast => north(A_BUTTON),
            ACT_JUMP if fast && s.pos[2] < RAMP_Z => north(A_BUTTON),
            ACT_JUMP_KICK if fast => {
                if s.pos[2] > RAMP_Z {
                    return north(A_BUTTON);
                }
                // Where it lands (flat ~808, gravity 4): release A when the
                // double jump from there would clear the box.
                let Some(l) = fly(s.pos, s.vel[1], s.forward_vel, s.yaw, 4.0, false, 32.0, false) else { return north(A_BUTTON) };
                let land = St { pos: [l.x, floor_at(l.z), l.z], forward_vel: l.fv, ..s.clone() };
                if dj_ok(&land) { north(0) } else { north(A_BUTTON) }
            }
            ACT_SLIDE_KICK if fast => north(0),
            ACT_SLIDE_KICK_SLIDE if fast => north(if last_a { 0 } else { A_BUTTON }),
            arcs::ACT_CROUCH_SLIDE if fast => north(Z_TRIG | B_BUTTON),
            ACT_DOUBLE_JUMP => {
                let t = [-40.0, DOOR[2]];
                pad(A_BUTTON, yaw_of(t[0] - s.pos[0], t[1] - s.pos[2]), 1.0)
            }
            // --- Landings at speed.
            ACT_FREEFALL_LAND_STOP if fast && s.pos[2] > BOX_Z.1 && s.pos[2] < RAMP_Z && !last_a => north(A_BUTTON),
            ACT_FREEFALL_LAND_STOP | ACT_PUNCHING | ACT_MOVE_PUNCHING if fast && s.pos[2] > BOX_Z.1 => {
                // Dive-drop (opt-in): the jump kick would land in the box,
                // the dive lands short of it.
                if dive_enabled()
                    && last_a
                    && s.pos[2] < RAMP_Z
                    && !hop_land(Hop::Kick, s).is_some_and(|l| clear_of_box(l.z))
                    && hop_land(Hop::Dive, s).is_some_and(|l| l.z > BOX_Z.1 + HOP_ERR)
                {
                    return dive();
                }
                kick()
            }
            ACT_FREEFALL_LAND if fast && s.pos[2] > BOX_Z.1 => {
                let n = in_action(traj, played);
                if s.pos[2] > RAMP_Z {
                    // The ramp: refresh on the 4th frame while it is
                    // steep, else Z (crouch slide → slide kick).
                    if n >= 4 {
                        return if s.floor_normal[1] < 0.8 { kick() } else { north(Z_TRIG) };
                    }
                    return north(A_BUTTON);
                }
                // A fresh A: released first if it is still held (that
                // frame is a ground frame too, and the jump comes from
                // the state after it).
                if dj_ok(s) && (!last_a || dj_ok(&after_landing_frame(s))) {
                    return north(if last_a { 0 } else { A_BUTTON });
                }
                // Too far out: on the 4th frame Z (crouch slide → slide
                // kick → rollout → the chain again), if the slide kick
                // lands outside the box; else, with A held, the jump kick
                // if that does; else (nothing lands clear) the Z.
                if n >= 4 && s.pos[2] > 300.0 {
                    let sk_clear = hop_land(Hop::Slide, &after_crouch_frame(s)).is_some_and(|l| clear_of_box(l.z));
                    if !sk_clear && last_a && hop_land(Hop::Kick, s).is_some_and(|l| clear_of_box(l.z)) {
                        return kick();
                    }
                    // Nothing lands clear: the double jump if the sim has it
                    // landing past the box at all (it is the best gamble), else Z.
                    if dj_ok_by(s, 2.0) && (!last_a || dj_ok_by(&after_landing_frame(s), 2.0)) {
                        return north(if last_a { 0 } else { A_BUTTON });
                    }
                    if dive_enabled() && !sk_clear && last_a && hop_land(Hop::Dive, s).is_some_and(|l| l.z > BOX_Z.1 + HOP_ERR) {
                        return dive();
                    }
                    return north(Z_TRIG);
                }
                north(0)
            }
            _ => genome,
        }
    }
}

/// `OBRRT_FREE_SETUP=1` (super::freesetup): the modes' setups as targets
/// with a tolerance, overlays cut to the frame-critical inputs, a setup ETA
/// in phases, and setup variants in the key:
/// - Edge / Air: the strip's side (0 west, 1 east) instead of the nearer
///   one; ETA = the route through the strip's entry (Edge) or the long
///   jump's landing on it (Air), then along it and out to the door, at
///   PACE, plus the turn onto it; overlay unchanged (lj_rule is already
///   only the exact-frame Z / Z + A presses).
/// - Arc: the side (0 west, 1 east; [`ARC_BANDS`] has the canonical-line
///   runway too, off: [`ARC_RUNWAYS`]). ETA
///   before the takeoff = the planner's own frames for the variant's best
///   direct line (run, turn, hops, walk), or the canonical line's frames
///   plus the approach and turn to its takeoff; in the skip, the distance
///   left at the canonical line's pace. Overlay: the crouch slide on the
///   frame a line works from the current heading (on the canonical line's
///   heading for its variants), Z + A, the hops' air stick and the strip
///   landing's Z + A; the forced stick and button lock near the takeoff
///   (the slow walk into position) is gone: the genome runs there.
/// - Hill: the TAS's crouch-slide window at C ([`HILL_WINDOWS`], one
///   variant, the line's tolerance as before: wider ones fell off the face
///   in the steer lab); the genome runs the approach and only the last
///   ~10 frames before the window get the stick on the line (key frames;
///   it was forced over 700 units with the buttons locked). ETA: the
///   approach, the turn to RUN_YAW and the run-up still needed for 30.5
///   (freesetup::handoff_frames), then the chain at [`HILL_PACE`].
pub mod free {
    use super::*;
    use crate::extra::tech::freesetup as fsu;
    use crate::moves::arcs;
    use buttons::{A_BUTTON, B_BUTTON, Z_TRIG};

    /// Arc's runway variants (per side): a direct line (Mario runs
    /// straight at the strip from where he is, takeoff at 18+:
    /// act_crouch_slide long-jumps above 10, `takeoff_now` needs 18), or the
    /// side's canonical line (a full-speed run-up along it, 29+: a walk
    /// reaches 29 after ~20 frames, ~400 units from rest), heading within
    /// [`CANON_YAW`] of it. Only the direct one is offered ([`ARC_RUNWAYS`]
    /// = 1): in the steer lab (5 starts) direct lines took off and skipped
    /// 10/10 on both sides, canonical ones 1/10 (the pursuit overshoots the
    /// takeoff and turns back); a speed band on direct lines alone made the
    /// lab circle (the plan comes and goes as the turn eats the speed).
    pub const ARC_BANDS: [(f32, f32); 2] = [(18.0, 1000.0), (29.0, 1000.0)];
    const ARC_RUNWAYS: u8 = 1;
    /// (Steer lab: lines that work take off up to ~0x500 off its heading.)
    const CANON_YAW: i32 = 0x800;
    /// The stick is on the line over this much before the takeoff (~10
    /// frames at 32).
    const ARC_PURSUIT: f32 = 320.0;
    /// Hill: the crouch slide's window along the run-up line, ahead of C in
    /// frames of the current speed: the TAS's (C within the next frame).
    /// One variant only: in the steer lab (lakitu_lab boot steer, 4 starts)
    /// a window 1-3 frames earlier never got onto the face, and a line 60
    /// wide / 0x400 in facing took off ~17 off it and fell in the moat after
    /// the slide; the face landing wants the TAS's line to a few units.
    pub const HILL_WINDOWS: [(f32, f32); 1] = [(-0.3, 0.95)];
    /// The stick is on the line (key frames) over this much of the run-up
    /// before the window (~10 frames at 32; hill::overlay forced it over 700
    /// and stopped the genome's buttons there).
    const HILL_PURSUIT: f32 = 320.0;
    /// The line's tolerance at the crouch slide (hill::overlay's).
    const HILL_LATERAL: f32 = 40.0;
    const HILL_YAW: i32 = 0x300;
    /// The crouch slide's speed (hill::overlay: > 30.5; the face landing
    /// needs about the full run).
    const HILL_SPEED: f32 = 30.5;
    /// The hill chain's pace from C to the door (TAS run-up long jump at C
    /// to the double jump's landing at the door: ~7700 units in ~100
    /// frames; runs/q6-p-s1 hill successes take 111-152 frames incl. the
    /// run-up).
    pub const HILL_PACE: f64 = 75.0;
    /// `mem.flags`: Hill's chain began (the run-up long jump or the slide's
    /// speed); Arc's first hop began.
    const GO: u32 = 2;

    pub fn variants(mode: Mode) -> u8 {
        match mode {
            Mode::Edge | Mode::Air => 2,
            Mode::Arc => 2 * ARC_RUNWAYS,
            Mode::Hill => HILL_WINDOWS.len() as u8,
        }
    }

    fn hd(a: [f32; 3], b: [f32; 3]) -> f32 {
        (a[0] - b[0]).hypot(a[2] - b[2])
    }

    fn rad(yaw: i16) -> f32 {
        yaw as f32 * std::f32::consts::TAU / 65536.0
    }

    fn door_left(p: [f32; 3]) -> f32 {
        (hd(p, DOOR) - DOOR_R).max(0.0)
    }

    /// Arc's skip pace: the canonical line's distance to the door over its
    /// planned frames (hops + slide + walk; ~38 u/f).
    fn arc_pace() -> f64 {
        static P: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
        *P.get_or_init(|| {
            let c = arc::canonical(0);
            (door_left(c.takeoff) as f64 / c.frames.max(1.0) as f64).max(obrrt_core::PACE)
        })
    }

    fn canon_only(fs: u8) -> bool {
        fs % ARC_RUNWAYS == 1
    }

    fn arc_side_band(fs: u8) -> (usize, (f32, f32)) {
        ((fs / ARC_RUNWAYS) as usize, ARC_BANDS[(fs % ARC_RUNWAYS) as usize])
    }

    /// Along the strip of `ex` from z, then out to the door.
    fn strip_rest(ex: f32, p: [f32; 3]) -> f32 {
        let exit = [ex, 960.0, BOX_Z.0];
        (p[2] - BOX_Z.0).max(0.0) + (p[0] - ex).abs() + door_left(exit)
    }

    /// The setup-phase ETA (module doc), after LakituSkip::eta's latches.
    pub fn eta(mode: Mode, fs: u8, s: &St, mem: &mut Mem) -> f64 {
        let air = s.action & arcs::ACT_FLAG_AIR != 0;
        match mode {
            Mode::Edge | Mode::Air => {
                let ex = EDGE_X[fs as usize & 1];
                if s.pos[2] <= BOX_Z.0 {
                    return frames_for(door_left(s.pos));
                }
                let entry = if mode == Mode::Air { [ex, 960.0, AIR_LAND_Z] } else { [ex, 830.0, BOX_Z.1] };
                let on_way = if mode == Mode::Air { s.pos[2] > AIR_LAND_Z + 100.0 } else { s.pos[2] > BOX_Z.1 };
                if !on_way {
                    return frames_for(strip_rest(ex, s.pos));
                }
                let d = hd(s.pos, entry) + strip_rest(ex, entry);
                let turn = if air || s.pos[2] < BOX_Z.1 { 0.0 } else { fsu::turn_frames(s, fsu::yaw_to(s.pos, entry)).min(8.0) };
                frames_for(d) + turn
            }
            Mode::Arc => {
                if s.action == ACT_LONG_JUMP {
                    mem.flags |= GO;
                }
                if mem.flags & GO != 0 || s.pos[2] < BOX_Z.1 {
                    return door_left(s.pos) as f64 / arc_pace();
                }
                let (side, band) = arc_side_band(fs);
                let moving = !air && s.forward_vel > 1.0;
                let plan = if canon_only(fs) {
                    f64::INFINITY
                } else {
                    arc::plan_from_moving_in(s.pos, s.forward_vel.max(0.0), moving.then(|| rad(s.yaw)), Some(side), Some(band))
                        .map_or(f64::INFINITY, |l| l.frames as f64)
                };
                let c = arc::canonical(side);
                let cy = fsu::yaw_to([0.0; 3], [c.h[0], 0.0, c.h[1]]);
                let canon = frames_for(hd(s.pos, c.takeoff)) + fsu::turn_frames(s, cy).min(12.0) + c.frames as f64;
                plan.min(canon)
            }
            Mode::Hill => {
                let run = s.action == ACT_LONG_JUMP && s.pos[2] > hill::RAMP_Z + 1500.0;
                if run || s.forward_vel > 55.0 {
                    mem.flags |= GO;
                }
                if mem.flags & GO != 0 {
                    return door_left(s.pos) as f64 / HILL_PACE;
                }
                let at = hill_point(fs, HILL_SPEED);
                fsu::handoff_frames(s, at, hill::RUN_YAW as i32, HILL_SPEED) + door_left(at) as f64 / HILL_PACE
            }
        }
    }

    /// The point on the run-up line where variant `fs`'s window opens at
    /// speed `v` (its far end: the crouch slide there or later).
    fn hill_point(fs: u8, v: f32) -> [f32; 3] {
        let d = hill::dir(hill::RUN_YAW);
        let back = HILL_WINDOWS[fs as usize].1 * v;
        [hill::C[0] - back * d[0], hill::C[1], hill::C[2] - back * d[1]]
    }

    pub fn target(mode: Mode, fs: u8, s: &St) -> [f32; 3] {
        match mode {
            Mode::Edge | Mode::Air => {
                let ex = EDGE_X[fs as usize & 1];
                if !near_bridge(s) || s.pos[2] <= BOX_Z.0 + 100.0 {
                    if s.pos[2] > BOX_Z.1 && grounds(s) {
                        return if mode == Mode::Air { [ex, 960.0, AIR_LAND_Z] } else { [ex, 830.0, BOX_Z.1 - 20.0] };
                    }
                    return DOOR;
                }
                match mode {
                    Mode::Edge if s.pos[2] > BOX_Z.1 && (s.pos[0] - ex).abs() > ON_STRIP => [ex, 830.0, BOX_Z.1 - 20.0],
                    Mode::Edge => [ex, 960.0, s.pos[2].min(BOX_Z.1) - STRIP_AHEAD],
                    _ if s.pos[2] > AIR_LAND_Z + 100.0 => [ex, 960.0, AIR_LAND_Z],
                    _ => [ex, 960.0, BOX_Z.0 - 200.0],
                }
            }
            Mode::Arc => {
                if !grounds(s) || s.pos[2] <= BOX_Z.0 + 100.0 {
                    return DOOR;
                }
                if arc::airborne(s) || s.pos[2] < BOX_Z.1 {
                    return arc::aim(s.pos[0]);
                }
                let (side, band) = arc_side_band(fs);
                let moving = s.forward_vel > 1.0;
                if !canon_only(fs)
                    && let Some(l) = arc::plan_from_moving_in(s.pos, s.forward_vel.max(0.0), moving.then(|| rad(s.yaw)), Some(side), Some(band))
                {
                    return [l.land[0], s.pos[1], l.land[2]];
                }
                // Onto the side's canonical line behind its takeoff (pure
                // pursuit), as arc::target.
                let c = arc::canonical(side);
                let t = c.takeoff;
                let (dx, dz) = (s.pos[0] - t[0], s.pos[2] - t[2]);
                let a = -(dx * c.h[0] + dz * c.h[1]);
                let e = (dx * c.h[1] - dz * c.h[0]).abs();
                let along = if a > 150.0 && e < 150.0 + 0.2 * a { a - 350.0 } else { 700.0 };
                [t[0] - along * c.h[0], s.pos[1], t[2] - along * c.h[1]]
            }
            Mode::Hill => hill::target(s),
        }
    }

    pub fn overlay(mode: Mode, fs: u8) -> PadOverlay {
        match mode {
            Mode::Edge | Mode::Air => {
                let air = mode == Mode::Air;
                PadOverlay::new(move |traj, played, genome| lj_rule(air, traj, played, genome))
            }
            Mode::Arc => PadOverlay::keyed(move |traj, played, genome| arc_rule(fs, traj, played, genome)),
            Mode::Hill => PadOverlay::keyed(move |traj, played, genome| hill_rule(fs, traj, played, genome)),
        }
    }

    /// arc::overlay without the takeoff lock; the crouch slide only onto the
    /// variant's strip at its band's speed.
    fn arc_rule(fs: u8, traj: &[St], played: &[GamePad], genome: GamePad) -> (GamePad, bool) {
        let Some(s) = traj.last() else { return (genome, false) };
        if !grounds(s) || !s.controllable() || s.pos[2] < BOX_Z.0 - 300.0 {
            return (genome, false);
        }
        match s.action {
            ACT_WALKING | arcs::ACT_IDLE | arcs::ACT_DECELERATING => {
                let (side, band) = arc_side_band(fs);
                let on_canon = !canon_only(fs) || {
                    let c = arc::canonical(side);
                    fsu::yaw_diff(s.yaw as i32, fsu::yaw_to([0.0; 3], [c.h[0], 0.0, c.h[1]])) < CANON_YAW
                };
                if s.action == ACT_WALKING && on_canon && arc::takeoff_now_in(s, Some(side), Some(band)) {
                    let last_z = played.last().is_some_and(|p| p.button & Z_TRIG != 0);
                    let (x, y) = arcs::stick(s.cam_yaw, s.yaw, 1.0);
                    return (GamePad::new(if last_z { 0 } else { Z_TRIG }, x, y), true);
                }
                // The last ~10 frames before the takeoff: the stick on the
                // line (key frames; the hop line wants the heading to a
                // degree or so). arc::overlay forced it, buttons off, from
                // 700 (1600) out.
                let t = if canon_only(fs) {
                    Some(arc::canonical(side).takeoff)
                } else {
                    let moving = s.forward_vel > 1.0;
                    arc::plan_from_moving_in(s.pos, s.forward_vel.max(0.0), moving.then(|| rad(s.yaw)), Some(side), Some(band)).map(|l| l.takeoff)
                };
                if t.is_some_and(|t| hd(s.pos, t) < ARC_PURSUIT) && s.forward_vel > 10.0 {
                    let tg = target(Mode::Arc, fs, s);
                    let d = (tg[0] - s.pos[0]).atan2(tg[2] - s.pos[2]);
                    let (x, y) = arcs::stick(s.cam_yaw, arc::rad_to_yaw(d), 1.0);
                    return (GamePad::new(0, x, y), true);
                }
                (genome, false)
            }
            ACT_LONG_JUMP_LAND | ACT_CROUCH_SLIDE => (arc::overlay(traj, played, genome), true),
            _ => (arc::overlay(traj, played, genome), false),
        }
    }

    /// hill::overlay with the run-up's walking frames freed (module doc).
    fn hill_rule(fs: u8, traj: &[St], played: &[GamePad], genome: GamePad) -> (GamePad, bool) {
        let Some(s) = traj.last() else { return (genome, false) };
        if !grounds(s) || !s.controllable() || s.pos[2] < BOX_Z.0 - 200.0 {
            return (genome, false);
        }
        let fast = s.forward_vel > 55.0;
        if s.action != arcs::ACT_WALKING || s.pos[2] <= hill::RAMP_Z + 1500.0 || fast {
            let crouch = s.action == arcs::ACT_CROUCH_SLIDE && !fast;
            return (hill::overlay(traj, played, genome), crouch);
        }
        let d = hill::dir(hill::RUN_YAW);
        let ahead = (hill::C[0] - s.pos[0]) * d[0] + (hill::C[2] - s.pos[2]) * d[1];
        let e = (s.pos[0] - hill::C[0]) * d[1] - (s.pos[2] - hill::C[2]) * d[0];
        let yaw_err = fsu::yaw_diff(s.yaw as i32, hill::RUN_YAW as i32);
        let (lo, hi) = HILL_WINDOWS[fs as usize];
        let v = s.forward_vel;
        if e.abs() < HILL_LATERAL && yaw_err < HILL_YAW && v > HILL_SPEED && ahead > lo * v && ahead < hi * v {
            let last_z = played.last().is_some_and(|p| p.button & Z_TRIG != 0);
            let (x, y) = arcs::stick(s.cam_yaw, s.yaw, 1.0);
            return (GamePad::new(if last_z { 0 } else { Z_TRIG }, x, y), true);
        }
        // The last frames before the window: the stick on the line (key
        // frames, the genome's x bends it a little); the rest is the
        // genome's.
        if yaw_err < 0x2000 && e.abs() < 120.0 && v > 10.0 && ahead > hi * v && ahead < hi * v + HILL_PURSUIT {
            let corr = if ahead > 2.0 * v { -(e / 150.0).atan() } else { 0.0 };
            let want = hill::RUN_YAW as i32 + (corr * 65536.0 / std::f32::consts::TAU) as i32;
            let (x, y) = arcs::stick(s.cam_yaw, want as i16, 1.0);
            return (GamePad::new(genome.button & !(A_BUTTON | B_BUTTON | Z_TRIG), x, y), true);
        }
        (genome, false)
    }
}

#[cfg(test)]
mod tests {
    use super::arc;

    /// `cargo test -p obrrt-objectives lakitu_arc_plan -- --nocapture`
    #[test]
    fn lakitu_arc_plan() {
        for p in [[-1328.0, 260.0, 4354.0], [-900.0, 760.0, 200.0], [0.0, 811.0, -100.0], [-400.0, 811.0, -100.0], [800.0, 760.0, 100.0], [-1500.0, 700.0, 0.0]] {
            println!("from {p:?}: {:?}", arc::plan_from(p));
        }
        for side in 0..2 {
            println!("canonical {side}: {:?}", arc::canonical(side));
        }
        assert!(arc::plan_from([-1328.0, 260.0, 4354.0]).is_some());
    }

    /// OBRRT_FREE_SETUP's setup ETAs fall along a setup (free::eta is
    /// called directly: the switch itself is read once per process).
    #[test]
    fn free_setup_eta_falls() {
        use super::{Mode, free};
        use crate::extra::tech::testkit::{ACT_WALKING, st};
        use obrrt_core::Mem;
        let at = |p: [f32; 3], yaw: i16, fv: f32| {
            let mut s = st(16, 1, p, ACT_WALKING);
            s.yaw = yaw;
            s.forward_vel = fv;
            s
        };
        // Edge, west strip: walking north (-z, yaw 0x8000) from the bank to
        // its entry, then along it.
        let north = 0x8000u16 as i16;
        let path = [[-800.0, 811.0, 300.0], [-650.0, 811.0, 0.0], [-549.0, 900.0, -200.0], [-549.0, 960.0, -1000.0], [-549.0, 960.0, -1900.0]];
        let f: Vec<f64> = path.iter().map(|&p| free::eta(Mode::Edge, 0, &at(p, north, 32.0), &mut Mem::default())).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "edge {f:?}");
        // Hill: closer to the run-up's crouch slide, faster, on its facing.
        let c = super::hill::C;
        let y = super::hill::RUN_YAW;
        let far = free::eta(Mode::Hill, 0, &at([-1328.0, 260.0, 4354.0], 0, 0.0), &mut Mem::default());
        let near = free::eta(Mode::Hill, 0, &at([c[0] - 300.0 * 0.75, 260.0, c[2] + 300.0 * 0.66], y, 32.0), &mut Mem::default());
        assert!(near < far, "hill {far} {near}");
        // Arc: the planner's frames fall as Mario runs at the strip.
        let a0 = free::eta(Mode::Arc, 0, &at([-1500.0, 700.0, 600.0], north, 32.0), &mut Mem::default());
        let a1 = free::eta(Mode::Arc, 0, &at([-1000.0, 760.0, 300.0], north, 32.0), &mut Mem::default());
        assert!(a1 < a0 && a0.is_finite(), "arc {a0} {a1}");
    }
}
