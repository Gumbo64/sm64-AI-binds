//! `Tech::HyperWalk` (`hyperwalk`): steer a hyperspeed walk (after a BLJ,
//! |fv| >= 100) to a goal (ref/tas/techniques/CATALOGUE.md G01). Every BLJ
//! in the TAS is followed by this; without it a BLJ only carries Mario
//! backwards along a fixed facing.
//!
//! Mechanism (decomp):
//! - Walking with fv <= 0 adds 1.1 a frame with no cap
//!   (update_walking_speed, mario_actions_moving.c), and the facing turns
//!   towards the stick by at most 0x800 a frame at any speed. The motion is
//!   `yaw + 0x8000`, so the stick points **away** from the goal.
//! - In the air the facing doesn't turn (update_air_without_turn,
//!   mario_actions_airborne.c): a stick opposite the facing (towards the
//!   goal, the way Mario moves) cuts the drag from +2.35 to +0.85 a frame,
//!   and its sideways part (10 sin dYaw) steers.
//! - Landing actions keep the speed with a neutral stick (no 0.98 friction,
//!   common_landing_action) but end in *_LAND_STOP (speed 0) unless the stick
//!   is held on their last frame (common_landing_cancels: 6 frames for a
//!   long jump, 4 for jumps and falls), which goes on walking at full speed.
//!   An A press in ACT_LONG_JUMP_LAND without Z is masked (act_long_jump_land),
//!   so A gets held there without a jump (TAS f4843).
//! - **Hops:** with A held, B while walking is ACT_MOVE_PUNCHING, which with
//!   A down on its first frame is a jump kick (act_move_punching:852): vy 20,
//!   fv unchanged, over gaps and risers. B without A held is a move punch
//!   that loses 8 a frame, so B is only pressed with A held. A fresh A
//!   while walking jumps (vy 42 + fv / 4: a hop above -168), so A is
//!   pressed only in the air or in a long-jump landing, then held.
//! - **Stops** (near the goal only): Z (crouch slide into a wall on the
//!   ground, ground pound in the air) or B (move-punch grab) when the GA's
//!   genome has them.
//!
//! Goals, from `b`'s objects and the castle painting table: stars not yet
//! had, warps, doors, holdable objects, red coins, the paintings of the
//! castle area; at 800-9000 straight-line, the nearest 4 stars / entries
//! and the nearest 2 others, plus the 2 farthest objects at 2500-10000
//! (the far points of the area), less those behind a moving `b` within two
//! turning circles (the facing turns 0x800 a frame). One
//! objective per zone and 200-unit goal cell (the key).
//!
//! - **Base:** the goal's zone, controllable, holding nothing, |fv| >= 100
//!   in a walking, landing, long-jump, jump-kick, freefall or butt-slide-air
//!   action, at least 800 from the goal. Made from hyperspeed `b`s and from
//!   BLJ spots (`bljany::blj_ok`), whose descendants (the `bljany` speed
//!   hand-off) are the starts.
//! - **Heuristic:** distance / |fv| (frames).
//! - **Success**, within 20 + d0 / (0.5 |fv0|) frames, alive, with |fv| > 60
//!   in every state up to it: within 300 (horizontal, 500 vertical) of the
//!   goal; or a new place far away: on the ground, 3000+ from the start and
//!   1500+ closer to the goal; or (the speed held up to the frame before)
//!   a star more (star goals) or another zone without a life lost (warp,
//!   door and painting goals).
//! - **Progress:** (d0 - dmin) / d0 + 0.2 min(|fv| / |fv0|, 1) at the
//!   closest state of the stretch that keeps |fv| > 60: once the speed is
//!   lost nothing more counts, so a plain walk scores no distance.
//! - **Inputs:** the GA with an overlay ([`Rule`]) while |fv| >= 50: the
//!   landing stick (neutral, then the steering line on the last frame), A
//!   held, B only with A held (hops), Z and B only near the goal (stops);
//!   elsewhere the genome's stick direction at full tilt, a centred one
//!   walks straight on (ACT_DECELERATING keeps the facing). The steering
//!   line (away from the goal on the ground, at it in the air) seeds the
//!   controller's genomes ([`Objective::steer_target`]). Below 50 the
//!   genome plays as is.
//!
//! Lab (scratchpad hwlab: GA attempts from 24 TAS post-BLJ landings, 112
//! goals x 3 seeds): 144 / 336 successes vs 119 for the same objective with
//! no overlay and the controller steering at the goal; forcing a hop on
//! every stalled frame (it fires at walls) cost 17 of them, and playing
//! only half the genome's angle off the line 8 more.

use std::f32::consts::{PI, TAU};
use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::*;
use obrrt_core::{Budget, GamePad, Objective, PadOverlay, Precision, Progress, St, dist};

use super::bljany;

const ACT_WALKING: u32 = 0x0400_0440;
const ACT_DECELERATING: u32 = 0x0400_044A;
const ACT_BRAKING: u32 = 0x0400_0445;
const ACT_JUMP_LAND: u32 = 0x0400_0470;
const ACT_FREEFALL_LAND: u32 = 0x0400_0471;
const ACT_LONG_JUMP_LAND: u32 = 0x0000_0479;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_JUMP: u32 = 0x0300_0880;
const ACT_JUMP_KICK: u32 = 0x0180_08AC;
const ACT_FREEFALL: u32 = 0x0100_088C;
const ACT_BUTT_SLIDE_AIR: u32 = 0x0300_088E;
const ACT_MOVE_PUNCHING: u32 = 0x0080_0457;

/// Speed a start needs.
const HYPER: f32 = 100.0;
/// The overlay steers at this speed and above.
const STEER: f32 = 50.0;
/// The objective weight of the speed-spending kinds (HyperWalk, HyperClimb,
/// PoleRedirect): they start only from rare hyperspeed states.
pub const SPEND_WEIGHT: f64 = 25.0;
/// Speed at the goal for a success.
const ARRIVE: f32 = 60.0;
/// Goal radius (horizontal, vertical).
const RADIUS: f32 = 300.0;
const RADIUS_Y: f32 = 500.0;
/// Goal distances from b.
const NEAR: (f32, f32) = (800.0, 9000.0);
const FAR: (f32, f32) = (2500.0, 10000.0);
/// A fresh A while walking above this speed would jump.
const A_SAFE: f32 = 168.0;
/// Whether A may be pressed fresh while walking below -A_SAFE (it jumps
/// with vy 42 + fv / 4 < 0, a one-frame hop).
const A_WALK: bool = false;
/// Whether a stalled walking frame forces the hop (a wall stall too, where
/// it only wastes frames).
const STALL_HOP: bool = false;
/// Whether the genome's Z and B are played near the goal (stops).
const NEAR_STOP: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Goal {
    Star,
    /// A warp, door or painting: entering it changes the zone.
    Entry,
    Holdable,
    /// A red coin or a far point of the area.
    Place,
}

/// The actions HyperWalk starts from (CATALOGUE G01 preconditions).
fn hyper_action(a: u32) -> bool {
    matches!(
        a,
        ACT_WALKING
            | ACT_DECELERATING
            | ACT_FREEFALL_LAND
            | ACT_JUMP_LAND
            | ACT_LONG_JUMP_LAND
            | ACT_LONG_JUMP
            | ACT_JUMP_KICK
            | ACT_FREEFALL
            | ACT_BUTT_SLIDE_AIR
    )
}

fn hyper(s: &St) -> bool {
    s.controllable() && !s.holding && s.forward_vel.abs() >= HYPER && hyper_action(s.action)
}

/// The length of the trajectory's prefix whose states all keep |fv| >
/// ARRIVE (the start included): states from this index on are slow or
/// after a slow one.
fn fast_len(traj: &[St]) -> usize {
    traj.iter().position(|s| s.forward_vel.abs() <= ARRIVE).unwrap_or(traj.len())
}

fn hdist_y(s: &St, g: [f32; 3]) -> (f32, f32) {
    (s.hdist(g), (s.pos[1] - g[1]).abs())
}

/// Castle goals behind star doors (area, stars the door needs, point):
/// painting centres (painting_table.rs corners + size / 2; the rooms' doors
/// need 1 (WF), 3 (CCM, JRB)), the PSS entrance (TAS lobby-left exit,
/// behind the 1-star door) and the BitDW trap floor behind the 8-star door
/// (lblj.rs).
const BEHIND_STAR_DOORS: &[(u8, i16, [f32; 3])] = &[
    (1, 3, [-2304.2, 0.0, -4352.0]),
    (1, 1, [255.8, 102.4, -4505.6]),
    (1, 3, [4300.8, 716.6, -230.6]),
    (1, 1, [1982.0, 768.0, -2095.0]),
    (1, 8, [-5513.0, 512.0, -4324.0]),
];

/// Objectives from `b`: when `b` is at hyperspeed, or a BLJ spot.
pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if !(hyper(b) || bljany::blj_ok(b).is_some()) {
        return;
    }
    let zone = b.zone();
    let mut typed: Vec<(Goal, [f32; 3])> = vec![];
    for o in b.objects() {
        let g = match o.interact_type {
            INTERACT_STAR_OR_KEY => {
                if crate::star::star_of(b, o).is_some_and(|(c, i)| b.has_star(c, i)) {
                    continue;
                }
                (Goal::Star, crate::star::rest_pos(o))
            }
            _ if crate::star::spawned_star(o) => (Goal::Star, crate::star::rest_pos(o)),
            INTERACT_WARP | INTERACT_WARP_DOOR | INTERACT_DOOR | INTERACT_BBH_ENTRANCE => (Goal::Entry, o.pos),
            INTERACT_GRABBABLE => (Goal::Holdable, o.pos),
            INTERACT_COIN if o.bhv_name() == "bhvRedCoin" => (Goal::Place, o.pos),
            _ => continue,
        };
        typed.push(g);
    }
    if b.level as u32 == LEVEL_CASTLE {
        for p in crate::painting_table::PAINTINGS.iter().filter(|p| p.area == b.area) {
            typed.push((Goal::Entry, p.centre()));
        }
    }
    let within = |p: [f32; 3], (lo, hi): (f32, f32)| (lo..hi).contains(&dist(b.pos, p));
    typed.retain(|g| within(g.1, NEAR));
    typed.sort_by(|x, y| dist(b.pos, x.1).total_cmp(&dist(b.pos, y.1)));
    // The nearest 4 stars / entries, and the nearest 2 others (holdables,
    // red coins), so a field of red coins doesn't crowd out the warps.
    let (mut typed, others): (Vec<_>, Vec<_>) = typed.into_iter().partition(|g| matches!(g.0, Goal::Star | Goal::Entry));
    typed.truncate(4);
    typed.extend(others.into_iter().take(2));
    // Entries only speed reaches (the TAS's BLJ exits): castle paintings
    // and places behind a star door b lacks the stars for, and star doors
    // b can't open, always (not among the nearest 4): through the lobby's
    // back wall to CCM, the WF / JRB rooms, the PSS entrance, the BitDW
    // corridor past the 8-star door.
    if b.level as u32 == LEVEL_CASTLE {
        for &(area, needs, goal) in BEHIND_STAR_DOORS {
            if area == b.area && b.stars < needs && within(goal, NEAR) {
                typed.push((Goal::Entry, goal));
            }
        }
        for o in b.objects() {
            let needs = if matches!(o.bhv_name(), "bhvDoor" | "bhvStarDoor") { (o.beh_params >> 24) as i16 } else { 0 };
            if b.stars < needs && within(o.pos, NEAR) {
                typed.push((Goal::Place, o.pos));
            }
        }
    }
    let mut far: Vec<[f32; 3]> = b.objects().iter().map(|o| o.pos).filter(|&p| within(p, FAR)).collect();
    far.sort_by(|x, y| dist(b.pos, *y).total_cmp(&dist(b.pos, *x)));
    far.dedup_by(|x, y| dist(*x, *y) < 1000.0);
    typed.extend(far.into_iter().take(2).map(|p| (Goal::Place, p)));
    // Not a goal behind a moving b within two turning circles (the facing
    // turns 0x800 a frame: radius |fv| / 0.196), which the walk can't make.
    let motion = if b.forward_vel < 0.0 { b.yaw as i32 + 0x8000 } else { b.yaw as i32 };
    let behind = |g: [f32; 3]| {
        let off = ((yaw_to(b.pos, g) - motion) as i16).unsigned_abs();
        hyper(b) && off > 0x4000 && b.hdist(g) < 2.0 * b.forward_vel.abs() / 0.196
    };
    typed.retain(|g| !behind(g.1));
    // The budget's horizon, from b's speed (at least HYPER: BLJ spots are slow).
    let speed = b.forward_vel.abs().max(HYPER);
    let mut keys = std::collections::HashSet::new();
    for (kind, goal) in typed {
        let deadline = 20 + (dist(b.pos, goal) / (0.5 * speed)) as usize;
        let o = HyperWalk { zone, goal, kind, horizon: (deadline + 20).clamp(60, 360) };
        if keys.insert(o.key()) {
            out.push(Arc::new(o));
        }
    }
}

pub struct HyperWalk {
    zone: (u16, u8),
    goal: [f32; 3],
    kind: Goal,
    horizon: usize,
}

impl HyperWalk {
    fn arrived(&self, s: &St) -> bool {
        let (h, y) = hdist_y(s, self.goal);
        s.zone() == self.zone && h < RADIUS && y < RADIUS_Y && s.forward_vel.abs() > ARRIVE
    }
}

impl Objective for HyperWalk {
    fn kind(&self) -> &'static str {
        "Tech::HyperWalk"
    }
    fn key(&self) -> String {
        let c = |v: f32| (v / 200.0).round() as i32;
        format!("Tech::HyperWalk {:?} {:?} ({}, {}, {})", self.zone, self.kind, c(self.goal[0]), c(self.goal[1]), c(self.goal[2]))
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // The pool keeps the weight of the entry's first b: an entry made at
        // a BLJ spot (no speed yet) must already weigh what a hyperspeed
        // start is worth, or the speed is never spent (40 M trees: 1-3
        // attempts from 9 BLJ successes at weight 4 / 1, NEW_ELIGIBLE
        // serving highest weight first never reached them).
        if self.base_criteria(b) || (b.zone() == self.zone && bljany::blj_ok(b).is_some()) { SPEND_WEIGHT } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && hyper(s) && s.dist(self.goal) >= NEAR.0
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        (s.dist(self.goal) / s.forward_vel.abs().max(1.0)) as f64
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let s0 = &traj[0];
        let d0 = s0.dist(self.goal);
        let deadline = 20 + (d0 / (0.5 * s0.forward_vel.abs().max(HYPER))) as usize;
        // Only the stretch still at speed counts: every state before the
        // success above ARRIVE (a walk that lost the speed is no HyperWalk).
        let fast = fast_len(traj);
        traj.iter().enumerate().skip(1).take(deadline).take_while(|&(i, _)| i <= fast).find_map(|(i, s)| {
            if !s.alive() {
                return None;
            }
            // A star grab or a warp stops Mario in that frame: the speed
            // must hold up to it.
            let star = self.kind == Goal::Star && s.stars > s0.stars;
            let entry = self.kind == Goal::Entry && s.zone() != self.zone && s.lives >= s0.lives;
            if i == fast && !(star || entry) {
                return None;
            }
            let far = s.zone() == self.zone
                && s.action & ACT_FLAG_AIR == 0
                && s.hdist(s0.pos) >= 3000.0
                && s.dist(self.goal) <= d0 - 1500.0;
            (self.arrived(s) || star || entry || far).then_some(i)
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let s0 = &traj[0];
        let d0 = s0.dist(self.goal).max(1.0);
        let v0 = s0.forward_vel.abs().max(1.0);
        // Distance credit only while the speed holds (|fv| > ARRIVE): the
        // closest state of that stretch. Past it, nothing more counts, so a
        // walk that loses the speed scores no distance.
        let fast = fast_len(traj);
        let (at, dmin) = traj[..fast]
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == self.zone && s.alive())
            .map(|(i, s)| (i, s.dist(self.goal)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, d0));
        let v = traj[at].forward_vel.abs();
        let score = ((d0 - dmin) / d0) as f64 + 0.2 * (v / v0).min(1.0) as f64;
        Progress { score, at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 32, generations: 6, horizon: self.horizon }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // The controller-seeded genomes: the stick away from the goal on the
        // ground (backwards speed), at it in the air (the overlay's lines).
        let g = self.goal;
        if s.action & ACT_FLAG_AIR == 0 && s.forward_vel < 0.0 {
            Some([2.0 * s.pos[0] - g[0], s.pos[1], 2.0 * s.pos[2] - g[2]])
        } else {
            Some(g)
        }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        let r = Rule { goal: self.goal, grab: self.kind == Goal::Holdable };
        Some(PadOverlay::new(move |traj, played, genome| r.pad(traj, played, genome)))
    }
}

/// SM64 yaw (s16 units) from `a` to `b`.
fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / PI) as i32
}

/// The full-tilt raw stick whose intendedYaw is `yaw` (camera-relative,
/// update_mario_joystick_inputs), with adjust_analog_stick's 6-unit dead
/// band added back (as blj.rs).
fn stick_along(s: &St, yaw: i32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * TAU / 65536.0;
    let t = 121.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

/// The intendedYaw of a raw stick, if it's out of the centre.
fn stick_yaw(s: &St, x: i8, y: i8) -> Option<i32> {
    let (x, y) = (x as f32, y as f32);
    (x * x + y * y >= 30.0 * 30.0).then(|| (x.atan2(-y) * 32768.0 / PI) as i32 + s.cam_yaw as i32)
}

#[derive(Clone, Copy)]
struct Rule {
    goal: [f32; 3],
    grab: bool,
}

impl Rule {
    fn pad(&self, traj: &[St], played: &[GamePad], genome: GamePad) -> GamePad {
        let s = traj.last().unwrap();
        let fv = s.forward_vel;
        if fv.abs() < STEER || !s.controllable() {
            return genome;
        }
        let last = played.last().map_or(0, |p| p.button);
        let held_a = last & A_BUTTON != 0;
        let air = s.action & ACT_FLAG_AIR != 0;
        let d = s.dist(self.goal);
        let near = d < 1.5 * fv.abs() + RADIUS;
        let to_goal = yaw_to(s.pos, self.goal);
        // Steering line: on the ground the facing, so away from the goal at
        // backwards speed; in the air the stick against the facing, i.e. at
        // the goal.
        let line = if !air && fv < 0.0 { to_goal + 0x8000 } else { to_goal };
        // Elsewhere the genome's direction (the controller seeds it along
        // the line; half the genome's angle off the line did worse in the
        // lab), and a centred genome stick is no stick.
        let bent = stick_yaw(s, genome.stick_x, genome.stick_y);
        let landing = matches!(s.action, ACT_LONG_JUMP_LAND | ACT_JUMP_LAND | ACT_FREEFALL_LAND);
        let stick = if landing {
            // Neutral (no friction) until the landing's last frame, then
            // the stick so it goes on walking instead of *_LAND_STOP.
            let frames = if s.action == ACT_LONG_JUMP_LAND { 6 } else { 4 };
            if s.action_timer as i32 + 1 >= frames { stick_along(s, line) } else { (0, 0) }
        } else {
            bent.map_or((0, 0), |y| stick_along(s, y))
        };
        // A: held from where a press is harmless.
        let a_ok = held_a || air || s.action == ACT_LONG_JUMP_LAND || (A_WALK && fv < -A_SAFE);
        let mut button = if a_ok { A_BUTTON } else { 0 };
        let walking = matches!(s.action, ACT_WALKING | ACT_DECELERATING | ACT_BRAKING);
        // A stall: a walking frame that moved less than an eighth of the speed
        // (a quarter step found no floor, or a wall).
        let stalled = walking && traj.len() >= 2 && traj[traj.len() - 2].hdist(s.pos) < fv.abs() / 8.0;
        let fresh_b = last & B_BUTTON == 0;
        if walking && held_a && fresh_b && (genome.button & B_BUTTON != 0 || (STALL_HOP && stalled)) {
            // The jump kick (A held + B).
            button |= B_BUTTON;
        }
        if near && NEAR_STOP {
            // Stops: the genome's Z (crouch slide / ground pound) and B.
            button |= genome.button & Z_TRIG;
            if fresh_b && (genome.button & B_BUTTON != 0 || (self.grab && !air && d < 2.0 * fv.abs())) {
                button |= B_BUTTON;
            }
        }
        if s.action == ACT_MOVE_PUNCHING || s.action == ACT_JUMP {
            // Not the hop's own frames: keep A as it is.
            button = (button & !A_BUTTON) | (last & A_BUTTON);
        }
        GamePad::new(button, stick.0, stick.1)
    }
}
