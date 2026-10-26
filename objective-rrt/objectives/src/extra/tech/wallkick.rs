//! `WallKick::Climb` and `TripleJumpWallKick::Climb`: reach a place high
//! above where Mario stands, higher than a jump can take him, which is what
//! wall kicks make reachable (techniques `wallkick` and `tjwk`).
//!
//! The mechanism (decomp, mario_actions_airborne.c / mario.c / mario_step.c):
//! - An airborne action that runs into a wall within ±45° of Mario's facing
//!   with `forwardVel > 16` turns him round and enters `ACT_AIR_HIT_WALL`
//!   (common_air_action_step). Because of act_air_hit_wall's missing return
//!   it runs twice on that frame, so A pressed on the contact frame itself
//!   kicks at once (a "firstie"). Otherwise the next frame sets
//!   `wallKickTimer = 5` and Mario soft-bonks (or, at `forwardVel >= 38`,
//!   is knocked back), and a fresh A press in those 5 frames still kicks
//!   (check_wall_kick in act_soft_bonk / act_backward_air_kb /
//!   act_forward_air_kb).
//! - `ACT_WALL_KICK_AIR` starts at vertical speed 62 (gravity 4/frame, so
//!   about 480 units of rise), with forward speed at least 24, and it can
//!   ledge-grab. Kicks chain between facing walls.
//! - A triple jump (vertical speed 69, about 595 units) can't ledge-grab;
//!   kicking off a wall out of it adds another ~480: about 1,075 in all.
//!
//! What a jump reaches without a kick, landing or ledge-grabbing (a ledge
//! grab lifts Mario to a floor up to 160 above him, mario_step.c
//! check_ledge_grab): a double jump (52) about 340 + 160 = 500, a side flip
//! (62) about 480 + 160 = 640, a triple jump (no grab) about 595. So a gain
//! of [`NO_KICK_MAX`] or more needs a wall kick (or a climb some other way,
//! which is just as good: success is the new height, by any means). Below
//! that the objective also asks for a kick in the trajectory, which keeps
//! it apart from `Mario::PosDist` (whose targets sit up to 900 above b but
//! accept any way there).
//!
//! Targets. Walls aren't in `St`, so the high places come from the loaded
//! objects: an object whose spawn point (`oHome`) sits well above b and
//! near it marks a floor or platform top up there (coins, enemies, boxes,
//! stars are placed on or just above the geometry). Mario runs at it, which
//! takes him into the face below it; the GA chooses the jump and kick
//! timing. With no such object, one generic target straight ahead of b.
//!
//! Success: Mario supported (on a floor or a ledge, not airborne, swimming,
//! on a pole or hanging), alive and in control, within [`REACH`] of b
//! horizontally, at least the target's gain above b. That's a new place for
//! the dedupe; a kick alone (which comes back down) isn't.
//!
//! Three more objectives (TAS catalogue G09 / G10), each its own technique:
//!
//! - `wksideflip`, `WallKick::SideFlip`: a side-flip wall kick (WF fortress
//!   and cage walls in the TAS). Run at the wall, reverse the stick about
//!   200 from it (ACT_TURNING_AROUND at 16+), A with the stick back at the
//!   wall: the side flip (62 up, 8 forward, facing the stick on the press
//!   frame, mario.c), stick on (+1.15 a frame in the air), A on contact
//!   (TAS f16065-16083: +500 in 18 frames).
//!   Same targets and success as `WallKick::Climb` (gain 350-1150), plus a
//!   wall target (+650) from b's `wall_normal` when b stands at a wall.
//! - `wkredirect`, `WallKick::Redirect`: a kick as a 180° turn. A kick
//!   keeps `forwardVel` (at least 24; a contact-frame kick keeps it all),
//!   so a fast Mario into a wall comes back out fast. From b at a wall
//!   (`wall_normal`), a goal 1000 back out along the normal; success a few
//!   frames after a kick at >= 0.9x the speed before contact (and >= 32)
//!   with the goal distance falling.
//! - `wkchain`, `WallKick::Chain`: corridor, chimney and concave-corner
//!   chains. Success: +N (700 or 1500) over 3+ kicks, with two horizontal
//!   wall normals at least 60° apart (opposite: a corridor or chimney;
//!   perpendicular: a corner) seen within [`CHAIN_WINDOW`] frames of each
//!   other in the trajectory; supported, or still on the walls (kicking or
//!   in the kick window), so a chain can go on from the success. Made from
//!   any b at a wall.
//!
//! All three force a fresh A press on every kick-window frame (pad
//! overlay); the steering holds A through a kick's rise (steer.rs).

use std::sync::Arc;

use obrrt_core::buttons::A_BUTTON;
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_HANGING, ACT_FLAG_ON_POLE, ACT_FLAG_SWIMMING, ACT_WALL_KICK_AIR};
use obrrt_core::{Budget, GamePad, Objective, PadOverlay, Precision, Progress, St};

const ACT_SIDE_FLIP: u32 = 0x0100_0887;
const ACT_TURNING_AROUND: u32 = 0x0000_0443;
const ACT_FINISH_TURNING_AROUND: u32 = 0x0000_0444;
const ACT_TRIPLE_JUMP: u32 = 0x0100_0882;
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;
const ACT_SOFT_BONK: u32 = 0x0102_08B6;
const ACT_BACKWARD_AIR_KB: u32 = 0x0102_08B0;
const ACT_FORWARD_AIR_KB: u32 = 0x0102_08B1;
const ACT_JUMP_LAND: u32 = 0x0400_0470;
const ACT_DOUBLE_JUMP_LAND: u32 = 0x0400_0472;

/// Gains below this are within a jump's reach (side flip + ledge grab), so
/// they count only with a wall kick in the trajectory.
pub const NO_KICK_MAX: f32 = 650.0;
/// The smallest gain offered: above a double jump plus a ledge grab.
const MIN_GAIN: f32 = 350.0;
/// The most a wall-kick climb is asked for (chimney chains in the TAS
/// climb 2,500-2,800).
const MAX_GAIN_WK: f32 = 3000.0;
/// A side-flip wall kick: the kick comes ~500 up the flip and rises ~480
/// more (TAS WF fortress: 2560 -> 3064 kick -> 3540), +160 with a ledge grab.
const MAX_GAIN_SF: f32 = 1150.0;
/// The side flip's turn: within this of the wall (TAS: 200), and not
/// closer than the second.
const SF_TURN_DIST: f32 = 350.0;
const SF_MIN_DIST: f32 = 100.0;
/// `forwardVel` that turns a reversed stick into ACT_TURNING_AROUND.
const SF_TURN_FV: f32 = 16.0;
/// The fastest b a side flip starts from (TAS: 10-40).
const SF_MAX_FV: f32 = 40.0;
/// Redirect: the goal's distance back out from the wall, the slowest
/// contact that counts, and the speed kept.
const RD_DIST: f32 = 1000.0;
const RD_MIN_FV: f32 = 32.0;
const RD_KEEP: f32 = 0.9;
/// Chain: the heights asked for, the kicks they need, how far apart (in
/// frames) the two walls' normals may be seen, and how far from b (and
/// from the first wall) the chain may go.
const CHAIN_GAINS: [f32; 2] = [700.0, 1500.0];
const CHAIN_MIN_KICKS: u32 = 3;
const CHAIN_WINDOW: usize = 60;
const CHAIN_REACH: f32 = 1200.0;
/// Two horizontal normals this far apart in direction (cos) are two walls.
const CHAIN_COS: f32 = 0.5;
/// A wall normal with |y| under this is a wall, not a slope.
const WALL_NY: f32 = 0.35;
/// The longest trajectory any of these is given.
const MAX_HORIZON: usize = 300;
/// A triple jump wall kick: up to about 1,075.
const MIN_GAIN_TJ: f32 = 650.0;
const MAX_GAIN_TJ: f32 = 1150.0;
/// A marker object this far below its spawn point is a floor top: coins
/// and the like float a little above the ground.
const MARKER_DROP: f32 = 120.0;
/// How far from b (horizontally) the marker and the success may be.
const MARKER_RADIUS: f32 = 1500.0;
const REACH: f32 = 1800.0;
/// Starts this close to b horizontally, and no higher than this above it.
const START_RADIUS: f32 = 700.0;
const START_ABOVE: f32 = 100.0;
/// Marker targets per b, per technique.
const PER_B: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Any number of wall kicks.
    WallKick,
    /// A triple jump, then a wall kick.
    Tjwk,
    /// A side flip off the ground, then a wall kick.
    SideFlip,
}

/// b's wall, if it has one: the horizontal unit normal (pointing away from
/// the wall, towards Mario).
fn wall_dir(s: &St) -> Option<[f32; 2]> {
    let n = s.wall_normal?;
    let l = (n[0] * n[0] + n[2] * n[2]).sqrt();
    (n[1].abs() < WALL_NY && l > 0.5).then(|| [n[0] / l, n[2] / l])
}

fn facing(s: &St) -> [f32; 2] {
    let a = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
    [a.sin(), a.cos()]
}

/// The raw stick that makes `intendedYaw` = `yaw` (camera-relative, full
/// tilt; adjust_analog_stick's ±6 added back, as blj.rs's stick_tilt).
fn stick_toward(s: &St, yaw: i32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = 121.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// The octant of a horizontal direction, for keys.
fn octant(d: [f32; 2]) -> i32 {
    ((d[0].atan2(d[1]) / std::f32::consts::FRAC_PI_4).round() as i32).rem_euclid(8)
}

fn cell(v: f32) -> i32 {
    (v / 400.0).round() as i32
}

/// A fresh A press on every kick-window frame (A_PRESSED needs A up the
/// frame before, so it alternates while the window lasts). The genome
/// keeps the rest, including A during a kick's rise: held, it keeps the
/// full rise (a corridor); let go, A can be pressed again on the next
/// contact frame (a corner, where contacts come 1-2 frames apart).
fn kick_overlay() -> PadOverlay {
    kick_rule(None)
}

/// [`kick_overlay`], and in ACT_TURNING_AROUND / ACT_FINISH_TURNING_AROUND
/// a fresh A with the stick at `target` (the side flip faces the stick on
/// the press frame, mario.c); on a frame A can't be pressed (held), the
/// stick stays back so the turn goes on.
fn side_flip_overlay(target: [f32; 3]) -> PadOverlay {
    kick_rule(Some(target))
}

fn kick_rule(flip_at: Option<[f32; 3]>) -> PadOverlay {
    PadOverlay::new(move |traj: &[St], played: &[GamePad], g: GamePad| {
        let Some(s) = traj.last() else { return g };
        let mut pad = g;
        let held = played.last().is_some_and(|p| p.button & A_BUTTON != 0);
        if kick_window(s) {
            pad.button = if held { pad.button & !A_BUTTON } else { pad.button | A_BUTTON };
        } else if let Some(t) = flip_at
            && matches!(s.action, ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND)
        {
            let to = (t[0] - s.pos[0]).atan2(t[2] - s.pos[2]) * 65536.0 / std::f32::consts::TAU;
            let to = to.round() as i32;
            let (yaw, button) = if held { (to + 0x8000, pad.button & !A_BUTTON) } else { (to, pad.button | A_BUTTON) };
            let (x, y) = stick_toward(s, yaw);
            pad = GamePad::new(button, x, y);
        }
        pad
    })
}

/// Whether Mario stands on something (a floor, a ledge he holds), not in
/// the air, the water, on a pole or hanging.
fn supported(s: &St) -> bool {
    s.action & (ACT_FLAG_AIR | ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE | ACT_FLAG_HANGING) == 0
}

/// States from which A starts a wall kick (in the window after contact).
fn kick_window(s: &St) -> bool {
    matches!(s.action, ACT_AIR_HIT_WALL | ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB | ACT_FORWARD_AIR_KB)
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Kicks seen so far along a trajectory (for `Tjwk`, only kicks after a
/// triple jump), updated state by state.
#[derive(Default)]
struct Seen {
    triple: bool,
    flip: bool,
    kicks: u32,
    prev: u32,
}

impl Seen {
    fn step(&mut self, mode: Mode, s: &St) {
        if s.action == ACT_TRIPLE_JUMP {
            self.triple = true;
        }
        if s.action == ACT_SIDE_FLIP {
            self.flip = true;
        }
        if s.action == ACT_WALL_KICK_AIR && self.prev != ACT_WALL_KICK_AIR && (mode != Mode::Tjwk || self.triple) {
            self.kicks += 1;
        }
        self.prev = s.action;
    }
}

pub struct Climb {
    mode: Mode,
    zone: (u16, u8),
    /// b's position: where the climb starts from.
    anchor: [f32; 3],
    /// The high place to run at (x, z), and its height.
    target: [f32; 3],
    /// Success at or above this height.
    goal_y: f32,
    /// Whether the target came from an object (else straight ahead of b).
    marked: bool,
}

impl Climb {
    fn gain(&self) -> f32 {
        self.goal_y - self.anchor[1]
    }

    /// Climbs from `b` (in control, standing) for `mode`: towards the
    /// nearest objects that sit high enough above b, or one straight ahead
    /// if none does.
    pub fn generate(b: &St, mode: Mode, out: &mut Vec<Arc<dyn Objective>>) {
        if !b.controllable() || !b.alive() || !supported(b) || b.action & ACT_FLAG_AIR != 0 {
            return;
        }
        if mode == Mode::SideFlip && b.forward_vel.abs() > SF_MAX_FV {
            return;
        }
        let (lo, hi) = match mode {
            Mode::WallKick => (MIN_GAIN, MAX_GAIN_WK),
            Mode::Tjwk => (MIN_GAIN_TJ, MAX_GAIN_TJ),
            Mode::SideFlip => (MIN_GAIN, MAX_GAIN_SF),
        };
        let mut marks: Vec<([f32; 3], f32)> = b
            .objects()
            .iter()
            .filter(|o| o.id.home[0] != i32::MIN && Some(o.slot) != b.held_slot)
            .filter_map(|o| {
                let gain = o.home[1] - MARKER_DROP - b.pos[1];
                let d = hdist(o.home, b.pos);
                (gain >= lo && gain <= hi && d >= 100.0 && d <= MARKER_RADIUS).then_some((o.home, d))
            })
            .collect();
        marks.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut cells: Vec<[i32; 2]> = Vec::new();
        for (p, _) in marks {
            let cell = [(p[0] / 400.0).round() as i32, (p[2] / 400.0).round() as i32];
            if cells.contains(&cell) {
                continue;
            }
            cells.push(cell);
            let goal_y = p[1] - MARKER_DROP;
            out.push(Arc::new(Climb { mode, zone: b.zone(), anchor: b.pos, target: [p[0], goal_y, p[2]], goal_y, marked: true }));
            if cells.len() == PER_B {
                return;
            }
        }
        if mode == Mode::SideFlip {
            // b at a wall: the face behind its normal, high enough to need
            // the kick (with or without marker targets). Else, with no
            // marker, straight ahead as for the climb.
            if let Some(n) = wall_dir(b) {
                let goal_y = b.pos[1] + NO_KICK_MAX;
                let target = [b.pos[0] - 100.0 * n[0], goal_y, b.pos[2] - 100.0 * n[1]];
                out.push(Arc::new(Climb { mode, zone: b.zone(), anchor: b.pos, target, goal_y, marked: false }));
                return;
            }
        }
        if cells.is_empty() {
            // Straight ahead: a face in front of b, high enough to need a kick.
            let (d, gain) = match mode {
                Mode::Tjwk => (900.0, 800.0),
                _ => (400.0, NO_KICK_MAX),
            };
            let a = b.yaw as f32 * std::f32::consts::TAU / 65536.0;
            let goal_y = b.pos[1] + gain;
            let target = [b.pos[0] + d * a.sin(), goal_y, b.pos[2] + d * a.cos()];
            out.push(Arc::new(Climb { mode, zone: b.zone(), anchor: b.pos, target, goal_y, marked: false }));
        }
    }

    /// The side flip's steering point for state `s` (TAS, WF fortress
    /// f16062: running at the wall at 32, stick back 200 from it, A with the
    /// stick at the wall on the first ACT_TURNING_AROUND frame). Run at the
    /// wall; within [`SF_TURN_DIST`] at 16+, reverse the stick; the A press
    /// and the stick back at the wall on that frame are the overlay's
    /// ([`side_flip_overlay`]: the controller's alternate-frame buttons
    /// can't set the stick with the press). Ground points sit at Mario's
    /// height so the controller doesn't jump.
    fn side_flip(&self, s: &St) -> [f32; 3] {
        let up = [self.target[0], self.goal_y + 200.0, self.target[2]];
        let d = hdist(s.pos, self.target);
        if s.action & ACT_FLAG_AIR != 0 || d < 1.0 {
            return up;
        }
        let w = [(self.target[0] - s.pos[0]) / d, (self.target[2] - s.pos[2]) / d];
        let toward = dot2(facing(s), w) > 0.5;
        let away = [s.pos[0] - 400.0 * w[0], s.pos[1], s.pos[2] - 400.0 * w[1]];
        let turning = matches!(s.action, ACT_TURNING_AROUND | ACT_FINISH_TURNING_AROUND);
        if turning || (toward && d <= SF_TURN_DIST && s.forward_vel >= SF_TURN_FV) || d < SF_MIN_DIST {
            away
        } else {
            [self.target[0], s.pos[1], self.target[2]]
        }
    }

    fn met(&self, s: &St, seen: &Seen) -> bool {
        let kicked = match self.mode {
            // A kick always (audit: 7 of 24 gains >= NO_KICK_MAX had none).
            Mode::WallKick | Mode::Tjwk | Mode::SideFlip => seen.kicks > 0,
        };
        kicked
            && s.zone() == self.zone
            && s.alive()
            && s.controllable()
            && supported(s)
            && s.pos[1] >= self.goal_y
            && hdist(s.pos, self.anchor) <= REACH
    }
}

impl Objective for Climb {
    fn kind(&self) -> &'static str {
        match self.mode {
            Mode::WallKick => "WallKick::Climb",
            Mode::Tjwk => "TripleJumpWallKick::Climb",
            Mode::SideFlip => "WallKick::SideFlip",
        }
    }
    fn key(&self) -> String {
        let c = |v: f32| (v / 400.0).round() as i32;
        format!(
            "{} {:?} from {},{},{} to {},{},{}",
            self.kind(),
            self.zone,
            c(self.anchor[0]),
            (self.anchor[1] / 200.0).round() as i32,
            c(self.anchor[2]),
            c(self.target[0]),
            (self.goal_y / 200.0).round() as i32,
            c(self.target[2])
        )
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        let w = match self.mode {
            Mode::WallKick => 1.0,
            Mode::Tjwk => 0.7,
            Mode::SideFlip => 0.8,
        };
        super::kicks::move_weight(if self.marked { w } else { w * 0.4 })
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.alive()
            && s.controllable()
            && supported(s)
            && s.pos[1] <= self.anchor[1] + START_ABOVE
            && s.pos[1] >= self.anchor[1] - 400.0
            && hdist(s.pos, self.anchor) <= START_RADIUS
            && (self.mode != Mode::SideFlip || s.forward_vel.abs() <= SF_MAX_FV)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let h = hdist(s.pos, self.anchor) as f64 / 500.0 + (self.goal_y - s.pos[1]).max(0.0) as f64 / 1000.0;
        match self.mode {
            // A run-up helps the triple jump (it needs forward speed 20+).
            Mode::Tjwk => h - (s.forward_vel.clamp(0.0, 32.0) as f64) / 64.0,
            Mode::WallKick | Mode::SideFlip => h,
        }
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let mut seen = Seen::default();
        for (i, s) in traj.iter().enumerate() {
            seen.step(self.mode, s);
            if i > 0 && self.met(s, &seen) {
                return Some(i);
            }
        }
        None
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Height towards the goal (straight line), nearer the target, a
        // bonus per kick (and for the triple jump), more when standing.
        // Only from the first kick on: before it every state scores as the
        // start does (audit: a walk or jump there was credited its height).
        let mut seen = Seen::default();
        let (mut best, mut at) = (f64::MIN, 0);
        for (i, s) in traj.iter().enumerate() {
            seen.step(self.mode, s);
            if s.zone() != self.zone || !s.alive() {
                break;
            }
            if i > 0 && seen.kicks == 0 {
                continue;
            }
            // A side flip's kick rises past the edge and comes back over it
            // (the TAS dives onto the WF fortress top), so it counts higher.
            let cap = if self.mode == Mode::SideFlip { 400.0 } else { 100.0 };
            let mut score = s.pos[1].min(self.goal_y + cap) as f64 - 0.25 * hdist(s.pos, self.target) as f64;
            score += 150.0 * seen.kicks.min(4) as f64;
            if self.mode == Mode::Tjwk && seen.triple {
                score += 150.0;
            }
            if self.mode == Mode::SideFlip && seen.flip {
                score += 150.0;
            }
            if supported(s) {
                score += 50.0;
            }
            if score > best {
                (best, at) = (score, i);
            }
        }
        Progress { score: best, at }
    }
    fn budget(&self) -> Budget {
        let kicks = (self.gain() / 480.0).ceil().max(1.0) as usize;
        let horizon = match self.mode {
            Mode::WallKick => (90 + 35 * kicks).min(MAX_HORIZON),
            Mode::Tjwk => 180,
            Mode::SideFlip => 120,
        };
        Budget::Ga { population: 16, generations: 5, horizon }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Above the target: run at it, jump while it's higher, and hold A
        // through a kick (steer.rs's kick mode).
        if s.zone() != self.zone {
            return None;
        }
        if self.mode == Mode::SideFlip {
            return Some(self.side_flip(s));
        }
        Some([self.target[0], self.goal_y + 200.0, self.target[2]])
    }
    fn steer_button(&self, s: &St) -> u16 {
        // A (on alternate frames) in the kick window after wall contact;
        // for the triple jump also on the landings of the first two jumps.
        if self.mode == Mode::SideFlip {
            return if kick_window(s) { A_BUTTON } else { 0 };
        }
        let jump_land = self.mode == Mode::Tjwk && matches!(s.action, ACT_JUMP_LAND | ACT_DOUBLE_JUMP_LAND);
        if kick_window(s) || jump_land { A_BUTTON } else { 0 }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        (self.mode == Mode::SideFlip).then(|| side_flip_overlay(self.target))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
}

/// Not in the water, on a pole or hanging: where a kick can happen.
fn free(s: &St) -> bool {
    s.action & (ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE | ACT_FLAG_HANGING) == 0
}

/// `WallKick::Redirect` (`wkredirect`): a wall kick as a 180° turn that
/// keeps the speed. See the module doc.
pub struct Redirect {
    zone: (u16, u8),
    /// b's position (at the wall) and the wall's horizontal normal.
    anchor: [f32; 3],
    n: [f32; 2],
    /// The goal, [`RD_DIST`] back out along the normal.
    goal: [f32; 3],
}

/// The kick in a trajectory: its index and Mario's speed then, and his
/// speed on the last frame before the contact.
fn first_kick(traj: &[St]) -> Option<(usize, f32, f32)> {
    let mut before = 0.0f32;
    for (i, s) in traj.iter().enumerate() {
        if s.action == ACT_WALL_KICK_AIR && i > 0 && traj[i - 1].action != ACT_WALL_KICK_AIR {
            return Some((i, s.forward_vel, before));
        }
        if s.action & ACT_FLAG_AIR != 0 && !kick_window(s) && s.action != ACT_WALL_KICK_AIR {
            before = s.forward_vel;
        }
    }
    None
}

impl Redirect {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if !b.controllable() || !b.alive() || !free(b) {
            return;
        }
        let Some(n) = wall_dir(b) else { return };
        let goal = [b.pos[0] + RD_DIST * n[0], b.pos[1], b.pos[2] + RD_DIST * n[1]];
        out.push(Arc::new(Redirect { zone: b.zone(), anchor: b.pos, n, goal }));
    }

    /// The wall point to run (or jump) at.
    fn wall(&self) -> [f32; 3] {
        [self.anchor[0] - 60.0 * self.n[0], self.anchor[1], self.anchor[2] - 60.0 * self.n[1]]
    }
}

impl Objective for Redirect {
    fn kind(&self) -> &'static str {
        "WallKick::Redirect"
    }
    fn key(&self) -> String {
        format!(
            "WallKick::Redirect {:?} {},{},{} {}",
            self.zone,
            cell(self.anchor[0]),
            (self.anchor[1] / 200.0).round() as i32,
            cell(self.anchor[2]),
            octant(self.n)
        )
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        super::kicks::move_weight(0.4)
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Near the wall, facing into it, already fast.
        s.zone() == self.zone
            && s.alive()
            && s.controllable()
            && free(s)
            && !kick_window(s)
            && s.action != ACT_WALL_KICK_AIR
            && s.forward_vel >= RD_MIN_FV
            && dot2(facing(s), self.n) < -0.5
            && hdist(s.pos, self.anchor) <= 600.0
            && (s.pos[1] - self.anchor[1]).abs() <= 300.0
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        hdist(s.pos, self.anchor) as f64 / 300.0 - s.forward_vel.clamp(0.0, 100.0) as f64 / 50.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let (k, fv, before) = first_kick(traj)?;
        if before < RD_MIN_FV || fv < RD_KEEP * before {
            return None;
        }
        let d0 = hdist(traj[k].pos, self.goal);
        let i = k + 3;
        let s = traj.get(i)?;
        (s.zone() == self.zone && s.alive() && hdist(s.pos, self.goal) < d0 - 30.0).then_some(i)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Before the kick: speed and closeness to the wall; after it, the
        // speed kept, and closeness to the goal.
        let kick = first_kick(traj);
        let (mut best, mut at) = (f64::MIN, 0);
        for (i, s) in traj.iter().enumerate() {
            if s.zone() != self.zone || !s.alive() {
                break;
            }
            let score = match kick {
                Some((k, fv, _)) if i >= k => {
                    1000.0 + 10.0 * fv as f64 - 0.25 * hdist(s.pos, self.goal) as f64
                }
                _ => 5.0 * s.forward_vel.clamp(0.0, 100.0) as f64 - 0.25 * hdist(s.pos, self.wall()) as f64,
            };
            if score > best {
                (best, at) = (score, i);
            }
        }
        Progress { score: best, at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 4, horizon: 90 }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // At the wall (a little above, so the controller jumps: a long or
        // double jump into it), then out to the goal once turned.
        if s.zone() != self.zone {
            return None;
        }
        let out = s.action & ACT_FLAG_AIR != 0 && !kick_window(s) && dot2(facing(s), self.n) > 0.3;
        let w = self.wall();
        Some(if out { self.goal } else { [w[0], s.pos[1] + 150.0, w[2]] })
    }
    fn steer_button(&self, s: &St) -> u16 {
        if kick_window(s) { A_BUTTON } else { 0 }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(kick_overlay())
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
}

/// `WallKick::Chain` (`wkchain`): kicks between two walls (a corridor, a
/// chimney, a concave corner). See the module doc.
pub struct Chain {
    zone: (u16, u8),
    /// b's position (at the first wall) and that wall's normal.
    anchor: [f32; 3],
    n: [f32; 2],
    gain: f32,
}

/// Kicks and two-wall evidence along a trajectory, state by state.
#[derive(Default)]
struct ChainSeen {
    seen: Seen,
    /// Distinct horizontal wall normals and the frame each was last seen.
    walls: Vec<([f32; 2], usize)>,
    two: bool,
}

impl ChainSeen {
    fn step(&mut self, i: usize, s: &St) {
        self.seen.step(Mode::WallKick, s);
        let Some(n) = wall_dir(s) else { return };
        self.walls.retain(|w| i - w.1 <= CHAIN_WINDOW);
        if self.walls.iter().any(|w| dot2(w.0, n) < CHAIN_COS) {
            self.two = true;
        }
        match self.walls.iter_mut().find(|w| dot2(w.0, n) > 0.95) {
            Some(w) => w.1 = i,
            None => self.walls.push((n, i)),
        }
    }
}

impl Chain {
    pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
        if !b.controllable() || !b.alive() || !free(b) {
            return;
        }
        let Some(n) = wall_dir(b) else { return };
        for gain in CHAIN_GAINS {
            out.push(Arc::new(Chain { zone: b.zone(), anchor: b.pos, n, gain }));
        }
    }

    fn kicks(&self) -> u32 {
        // TAS chains climb 100-320 a kick, net.
        ((self.gain / 250.0).ceil() as u32).max(CHAIN_MIN_KICKS)
    }

    fn near(&self, s: &St) -> bool {
        s.zone() == self.zone && s.alive() && hdist(s.pos, self.anchor) <= CHAIN_REACH
    }
}

impl Objective for Chain {
    fn kind(&self) -> &'static str {
        "WallKick::Chain"
    }
    fn key(&self) -> String {
        format!(
            "WallKick::Chain {:?} {},{},{} {} +{}",
            self.zone,
            cell(self.anchor[0]),
            (self.anchor[1] / 200.0).round() as i32,
            cell(self.anchor[2]),
            octant(self.n),
            self.gain as i32
        )
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        super::kicks::move_weight(if self.gain <= CHAIN_GAINS[0] { 0.5 } else { 0.35 })
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Near the wall, standing or in the air (a chain under way).
        s.zone() == self.zone
            && s.alive()
            && s.controllable()
            && free(s)
            && hdist(s.pos, self.anchor) <= 500.0
            && s.pos[1] <= self.anchor[1] + 200.0
            && s.pos[1] >= self.anchor[1] - 300.0
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let at_wall = if wall_dir(s).is_some() { -1.0 } else { 0.0 };
        hdist(s.pos, self.anchor) as f64 / 300.0 + (self.anchor[1] - s.pos[1]).max(0.0) as f64 / 500.0 + at_wall
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let mut c = ChainSeen::default();
        let goal = self.anchor[1] + self.gain;
        for (i, s) in traj.iter().enumerate() {
            c.step(i, s);
            let on = (supported(s) && s.controllable()) || s.action == ACT_WALL_KICK_AIR || kick_window(s);
            if i > 0 && c.two && c.seen.kicks >= CHAIN_MIN_KICKS && on && s.pos[1] >= goal && self.near(s) {
                return Some(i);
            }
        }
        None
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Height (straight up), a bonus per kick and for the second wall.
        let mut c = ChainSeen::default();
        let (mut best, mut at) = (f64::MIN, 0);
        let top = self.anchor[1] + self.gain + 100.0;
        for (i, s) in traj.iter().enumerate() {
            c.step(i, s);
            if !self.near(s) {
                break;
            }
            // Nothing before the first kick (as Climb).
            if i > 0 && c.seen.kicks == 0 {
                continue;
            }
            let mut score = s.pos[1].min(top) as f64 + 120.0 * c.seen.kicks.min(12) as f64;
            if c.two {
                score += 200.0;
            }
            if score > best {
                (best, at) = (score, i);
            }
        }
        Progress { score: best, at }
    }
    fn budget(&self) -> Budget {
        let horizon = (90 + 30 * self.kicks() as usize).min(MAX_HORIZON);
        Budget::Ga { population: 16, generations: 5, horizon }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Standing: at the wall (jump into it). In the air: straight on in
        // the facing (after a kick that's at the other wall, or the other
        // face of the corner), up; in the kick window, the way the kick
        // will face. Holding A through the rise is steer.rs's kick mode.
        if s.zone() != self.zone {
            return None;
        }
        if s.action & ACT_FLAG_AIR == 0 {
            return Some([self.anchor[0] - 80.0 * self.n[0], s.pos[1] + 400.0, self.anchor[2] - 80.0 * self.n[1]]);
        }
        let f = facing(s);
        let f = if kick_window(s) { [-f[0], -f[1]] } else { f };
        Some([s.pos[0] + 300.0 * f[0], s.pos[1] + 400.0, s.pos[2] + 300.0 * f[1]])
    }
    fn steer_button(&self, s: &St) -> u16 {
        if kick_window(s) { A_BUTTON } else { 0 }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(kick_overlay())
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
}
