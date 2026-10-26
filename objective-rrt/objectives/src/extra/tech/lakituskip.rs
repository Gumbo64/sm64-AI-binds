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
//! Two modes (keys), the same success and progress:
//! - [`Mode::Edge`]: steer onto the nearer strip and along it (walk, jumps).
//! - [`Mode::Air`]: crouch-slide long jump at the box's start, aimed to land
//!   on the strip ~1270 on, then long jumps (Z + A on each landing) out.
//!
//! Success: in the castle (level 6), or within [`DOOR_R`] of the door, with
//! Mario never caught (`ACT_WAITING_FOR_DIALOG` / `ACT_READING_NPC_DIALOG`,
//! dialog 34) nor fallen into the moat.
//! Progress rewards the skip itself, not walking at the door: each safe
//! state (outside the box) scores how far it is through the box's z range
//! (0 before it; only past the box does closeness to the door add),
//! and an airborne state over the box scores half its own depth plus half
//! the last safe state's, with the kept state the last safe one (an
//! airborne state over the box is about to be caught on landing; one over
//! the moat, floor -511, to fall in: neither is safe). A walk
//! into the box is caught in its first frame, so it scores nothing.

use obrrt_core::layout::{
    ACT_READING_NPC_DIALOG, ACT_WAITING_FOR_DIALOG, LEVEL_CASTLE, LEVEL_CASTLE_GROUNDS,
};
use obrrt_core::{Budget, GamePad, Obj, ObjId, Objective, PadOverlay, Progress, St, buttons};

/// Between the two door warps, at their height.
const DOOR: [f32; 3] = [0.0, 803.0, -3155.0];
const DOOR_R: f32 = 350.0;
/// The trigger box (x, z) and its height bound.
const BOX_X: (f32, f32) = (-544.0, 545.0);
const BOX_Z: (f32, f32) = (-2000.0, -177.0);
const BOX_Y: f32 = 800.0;
const BOX_LEN: f32 = BOX_Z.1 - BOX_Z.0;
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
/// Progress values: past the box, plus closeness to the door up to this.
const PAST_BONUS: f32 = 1500.0;

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
}

pub struct LakituSkip {
    lakitu: ObjId,
    mode: Mode,
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

/// How far `s` is through the box's z range: 0 anywhere before it, so
/// walking up to the box earns nothing.
fn depth(s: &St) -> f32 {
    (BOX_Z.1 - s.pos[2]).clamp(0.0, BOX_LEN)
}

/// A state outside the box: how far along the skip it is.
fn value(s: &St) -> f32 {
    if arrived(s) {
        BOX_LEN + PAST_BONUS
    } else if s.level as u32 == LEVEL_CASTLE_GROUNDS && s.pos[2] <= BOX_Z.0 {
        BOX_LEN + (PAST_BONUS - s.dist(DOOR)).max(0.0)
    } else {
        depth(s)
    }
}

/// The nearer strip's centre line.
fn side(s: &St) -> f32 {
    if s.pos[0] < 0.0 { EDGE_X[0] } else { EDGE_X[1] }
}

/// Near the bridge, before the box ends: where the hints apply.
fn near_bridge(s: &St) -> bool {
    grounds(s) && s.pos[0].abs() < 800.0 && s.pos[2] < 400.0 && s.pos[2] > BOX_Z.0 - 100.0 && s.pos[1] > FELL_Y
}

/// The prefix of `traj` before Mario is first caught by a dialog, or falls
/// off the bridge into the moat (from the bridge or past it).
fn clean(traj: &[St]) -> &[St] {
    let fell = |s: &St| grounds(s) && s.pos[2] < BOX_Z.1 && s.pos[1] < FELL_Y;
    let n = traj.iter().position(|s| talking(s) || fell(s)).unwrap_or(traj.len());
    &traj[..n]
}

/// Generated per object: for the intro Lakitu still waiting, both modes.
pub fn generate(b: &St, o: &Obj, out: &mut Vec<std::sync::Arc<dyn Objective>>) {
    if grounds(b) && is_intro_lakitu(o) && o.action == CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE {
        for mode in [Mode::Edge, Mode::Air] {
            out.push(std::sync::Arc::new(LakituSkip { lakitu: o.id, mode }));
        }
    }
}

impl Objective for LakituSkip {
    fn kind(&self) -> &'static str {
        "Tech::LakituSkip"
    }
    fn key(&self) -> String {
        match self.mode {
            Mode::Edge => "Tech::LakituSkip".to_string(),
            Mode::Air => "Tech::LakituSkip air".to_string(),
        }
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // On the route to every star (Door::Enter is 8), and only offered
        // until the intro plays: high enough that the pool samples it
        // before then (at 4, after a few failures' decay, 1-4 attempts in
        // 60 M frames).
        12.0
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Anywhere outside the box (the south end, the strip) not past it.
        grounds(s)
            && !talking(s)
            && safe(s)
            && !entering(s)
            && s.pos[1] > FELL_Y
            && s.pos[2] > BOX_Z.0
            && s.dist(BRIDGE_SOUTH) < START_R
            && s.find(self.lakitu).is_some_and(|o| o.action == CAMERA_LAKITU_INTRO_ACT_TRIGGER_CUTSCENE)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Distance left along the skip (straight-line through the box).
        (BOX_LEN + PAST_BONUS - value(s)) as f64 / 200.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        clean(traj).iter().position(arrived)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let c = clean(traj);
        let mut last = (0usize, c.first().map_or(f32::MIN, value));
        let mut best = (0usize, f32::MIN);
        for (i, s) in c.iter().enumerate() {
            let v = if !safe(s) {
                0.5 * depth(s) + 0.5 * last.1
            } else {
                last = (i, value(s));
                last.1
            };
            if v > best.1 {
                best = (last.0, v);
            }
        }
        Progress { score: best.1 as f64, at: best.0 }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
    }
    fn exists_in(&self, s: &St) -> bool {
        grounds(s) && s.find(self.lakitu).is_some()
    }
    fn precision(&self) -> obrrt_core::Precision {
        // A ~10-unit strip.
        obrrt_core::Precision::Precise
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
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
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Over the box's far half in a jump that would land in it: dive
        // (B; +15 u/f, capped at 48) to carry past z -2000.
        use buttons::B_BUTTON;
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
