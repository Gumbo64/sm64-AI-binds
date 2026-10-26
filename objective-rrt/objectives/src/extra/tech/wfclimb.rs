//! `Wf::FortressClimb` (technique `wfclimb`): from the WF painting landing
//! to the fortress top (the Whomp King's arena) as a race against the clock,
//! with progress only for fast movement.
//!
//! The TAS (ref/tas/dumps/full-120star-2026-U.csv, first WF visit, control
//! at f15926 by the spawn (2685, 409, 5090)) is on the arena (y 3584,
//! within 1450 of the King at (0, 3584, 0)) at f16101, **175 frames**: jump
//! kick, long jump up the first ramp, two wall kicks off the wall at
//! (3276, 620, 3890), dive and rollout onto the high ground (3355, 1535,
//! 2185, +84), long jump to the steep ramp's foot (2348, 1787, 1347), two
//! steep jumps and a dive slide onto the lower top (1957, 2560, 1016), a
//! side flip into the wall at (1586, 3064, 1013), a wall kick and a dive
//! onto the top (1528, 3584, 621, +170). The v46 trees take 668-1325
//! frames from control to the top (Mission::ReachStar / WhompKing::Pound
//! segments walking a 2-4x longer path at ~22 u/frame); `Opener::Wf` gets
//! them to the high ground in time but nothing carries on from there.
//!
//! Location technique, so only a rough start and the endpoint are given:
//! - start: an entry state `b` by the spawn point, in control, not yet
//!   running (as the openers'); its frame is the race's start and the
//!   deadline is [`DEADLINES`] frames later (in the key, one race per
//!   entry). Starts are WF states from after the entry on the high ground
//!   (`Opener::Wf`'s target, the rough start: from the spawn itself the
//!   straight line runs under the fortress's walls, where every test's
//!   best states wall-kicked to y ~1500-1900 and stopped) or anywhere
//!   above it (y 2000+, the climb), that can still make
//!   it at [`MAX_SPEED`] straight-line; ranked by straight-line distance.
//! - endpoint ([`at_top`]): on the fortress top within [`TOP_REACH`] of
//!   the King (horizontal), below the tower: inside `Boss::WhompFast`'s
//!   Init start radius (1500), so every success is a whomphits start.
//!
//! Distances are straight-line to the arena: to the nearest point of the
//! disc of radius 1250 round the King at y 3584 ([`aim`]), so height is
//! what's left to gain once below its rim. (To a point on the arena,
//! the trees' best progress states sat under the fortress's walls at
//! y ~1500, nearer the point than the steep ramp the TAS climbs.)
//!
//! Progress, gated on fast movement (no walk credit): the straight-line
//! distance to the arena gained on frames of a signature move
//! ([`signature`]: long jump, wall kick, side flip, backflip, triple jump,
//! steep jump, dive, rollout), while every frame's loss counts (so walking
//! away and long-jumping back nets nothing), minus [`FRAME_COST`] a frame.
//! Walking, running and plain jumps at the target earn nothing. (A first
//! version scored minus the predicted arrival `frame + dist / 34`; from a
//! standstill at the spawn no GA genome beat its start: the TAS itself
//! averages 33.6 straight-line.) Success: at the endpoint by the deadline
//! with a signature move in the segment. Success heuristic: frame (the
//! default).
//!
//! Hints only (the GA does the timing): stick at the endpoint; A in the
//! wall-kick window and to roll out of dive slides; A out of a crouch slide
//! (long jump), Z+A on long-jump landings (chain); Z when running at 29+
//! far from the endpoint and not far below it (crouch slide for a long
//! jump); B in the air at fv > 28 when the endpoint isn't high above (dive);
//! otherwise the controller's default A while the endpoint is higher
//! (jumps, double and triple jumps).

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_FLAG_MOVING, LEVEL_WF};
use obrrt_core::{Budget, Objective, Precision, Progress, St};

use super::travel::{ACT_CROUCH_SLIDE, ACT_DIVE, ACT_DIVE_SLIDE, ACT_FORWARD_ROLLOUT, ACT_LONG_JUMP, ACT_LONG_JUMP_LAND, can_start};

const ZONE: (u16, u8) = (LEVEL_WF as u16, 1);
/// MARIO_POS of WF area 1 (levels/wf/script.c).
const SPAWN: [f32; 3] = [2600.0, 256.0, 5120.0];
/// An entry state: this close (horizontally) to the spawn point, about at
/// its height, not yet running (as the openers').
const ENTRY_RADIUS: f32 = 250.0;
const ENTRY_HEIGHT: f32 = 200.0;
const ENTRY_SPEED: f32 = 20.0;
/// Frames after the entry the top must be reached by: 8 s, and 11 s and
/// 15 s steps (TAS 175; the trees' best 668).
const DEADLINES: [u32; 3] = [240, 330, 450];
/// The Whomp King's home (bhvWhompKingBoss, levels/wf/script.c) and the
/// arena floor.
const KING: [f32; 3] = [0.0, 3584.0, 0.0];
const TOP_Y: f32 = 3584.0;
/// On the arena: within this of the King horizontally (whomphits' Init
/// starts need dist < 1500).
const TOP_REACH: f32 = 1450.0;
/// Distances are to the arena: the disc of this radius round the King at
/// the arena's height ([`aim`]).
const AIM_REACH: f32 = 1250.0;
/// Straight-line speed no route beats, for which starts can still make the
/// deadline.
const MAX_SPEED: f32 = 60.0;
/// The rough start region: the high ground `Opener::Wf` races to (its
/// target and reach), or anything higher than it (the climb).
const HIGH: [f32; 3] = [3300.0, 1600.0, 2000.0];
const HIGH_REACH: f32 = 700.0;
const HIGH_Y: f32 = 1350.0;
const CLIMB_Y: f32 = 2000.0;

/// A start: on the high ground, or above it.
fn on_route(s: &St) -> bool {
    (s.hdist(HIGH) < HIGH_REACH && s.pos[1] >= HIGH_Y) || s.pos[1] >= CLIMB_Y
}

/// Progress charge per frame (distance units).
const FRAME_COST: f64 = 1.0;
/// Starts this far from the target at most (the spawn is ~5600 away).
const START_MAX: f32 = 7000.0;

/// Actions (include/sm64.h).
const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;
const ACT_SOFT_BONK: u32 = 0x0102_08B6;
const ACT_BACKWARD_AIR_KB: u32 = 0x0102_08B0;
const ACT_FORWARD_AIR_KB: u32 = 0x0102_08B1;
const ACT_WALL_KICK_AIR: u32 = 0x0300_0886;
const ACT_SIDE_FLIP: u32 = 0x0100_0887;
const ACT_BACKFLIP: u32 = 0x0100_0883;
const ACT_TRIPLE_JUMP: u32 = 0x0100_0882;
const ACT_STEEP_JUMP: u32 = 0x0300_0885;

/// A fast move or kick: the actions whose frames earn progress.
fn signature(a: u32) -> bool {
    matches!(
        a,
        ACT_LONG_JUMP | ACT_WALL_KICK_AIR | ACT_SIDE_FLIP | ACT_BACKFLIP | ACT_TRIPLE_JUMP | ACT_STEEP_JUMP | ACT_DIVE | ACT_FORWARD_ROLLOUT
    )
}

/// The nearest point of the arena disc to `s` (its endpoint).
fn aim(s: &St) -> [f32; 3] {
    let (dx, dz) = (s.pos[0] - KING[0], s.pos[2] - KING[2]);
    let h = (dx * dx + dz * dz).sqrt();
    if h <= AIM_REACH {
        return [s.pos[0], TOP_Y, s.pos[2]];
    }
    let k = AIM_REACH / h;
    [KING[0] + dx * k, TOP_Y, KING[2] + dz * k]
}

/// Straight-line distance to the arena (to [`aim`]).
fn gap(s: &St) -> f32 {
    s.dist(aim(s))
}

fn kick_window(s: &St) -> bool {
    matches!(s.action, ACT_AIR_HIT_WALL | ACT_SOFT_BONK | ACT_BACKWARD_AIR_KB | ACT_FORWARD_AIR_KB)
}

/// On the fortress top's arena (or just above it), below the tower.
pub fn at_top(s: &St) -> bool {
    s.zone() == ZONE
        && s.alive()
        && s.pos[1] >= TOP_Y - 30.0
        && s.pos[1] < TOP_Y + 400.0
        && s.floor_y >= TOP_Y - 30.0
        && s.hdist(KING) < TOP_REACH
}

pub struct FortressClimb {
    entry: u32,
    limit: u32,
}

/// One race per deadline per entry state `b` by the WF spawn.
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if b.zone() == ZONE
        && b.controllable()
        && b.alive()
        && b.hdist(SPAWN) < ENTRY_RADIUS
        && (b.pos[1] - SPAWN[1]).abs() < ENTRY_HEIGHT
        && b.forward_vel.abs() < ENTRY_SPEED
    {
        for limit in DEADLINES {
            out.push(Arc::new(FortressClimb { entry: b.frame, limit }));
        }
    }
}

impl FortressClimb {
    fn deadline(&self) -> u32 {
        self.entry + self.limit
    }
}

impl Objective for FortressClimb {
    fn kind(&self) -> &'static str {
        "Wf::FortressClimb"
    }
    fn key(&self) -> String {
        format!("Wf::FortressClimb from {} in {}", self.entry, self.limit)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if b.zone() == ZONE { 12.0 * 240.0 / self.limit as f64 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == ZONE
            && can_start(s)
            && s.alive()
            && s.frame >= self.entry
            && !at_top(s)
            && s.pos[1] < TOP_Y - 30.0
            && on_route(s)
            && gap(s) < START_MAX
            && (s.frame as f32 + gap(s) / MAX_SPEED) < self.deadline() as f32
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Straight-line distance (the deadline filter keeps starts on
        // time; ranking by predicted arrival at 34 sent every attempt
        // back to the spawn, where the GA's gains are slower than that).
        gap(s) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let mut seen = false;
        for (i, s) in traj.iter().enumerate() {
            if s.frame > self.deadline() || s.zone() != ZONE || !s.alive() {
                return None;
            }
            seen |= signature(s.action);
            if seen && at_top(s) {
                return Some(i);
            }
        }
        None
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Distance to the target gained on signature-move frames only;
        // every frame's loss counts. Minus a small charge per frame, so of
        // two equal gains the sooner is better.
        let Some(s0) = traj.first() else { return Progress { score: f64::MIN, at: 0 } };
        let mut best = Progress { score: 0.0, at: 0 };
        let mut credit = 0.0;
        let mut last = gap(s0) as f64;
        for (i, s) in traj.iter().enumerate().skip(1) {
            if s.frame > self.deadline() || s.zone() != ZONE || !s.alive() {
                break;
            }
            let d = gap(s) as f64;
            let gain = last - d;
            last = d;
            credit += if signature(s.action) { gain } else { gain.min(0.0) };
            let score = credit - FRAME_COST * i as f64;
            if score > best.score {
                best = Progress { score, at: i };
            }
        }
        best
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 6, horizon: self.limit as usize + 10 }
    }
    fn precision(&self) -> Precision {
        Precision::Precise
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == ZONE && s.frame <= self.deadline()
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(aim(s))
    }
    fn steer_button(&self, s: &St) -> u16 {
        if kick_window(s) {
            return A_BUTTON;
        }
        let air = s.action & ACT_FLAG_AIR != 0;
        let moving = s.action & ACT_FLAG_MOVING != 0;
        let t = aim(s);
        let dy = t[1] - s.pos[1];
        let far = s.hdist(t) > 800.0;
        match s.action {
            // The wall kick's own controller (no button) steers it.
            ACT_WALL_KICK_AIR | ACT_DIVE => 0,
            ACT_DIVE_SLIDE | ACT_FORWARD_ROLLOUT => A_BUTTON,
            ACT_CROUCH_SLIDE => A_BUTTON,
            ACT_LONG_JUMP_LAND => Z_TRIG | A_BUTTON,
            _ if !air && moving && s.forward_vel >= 29.0 && far && dy < 600.0 => Z_TRIG,
            _ if air && s.forward_vel > 28.0 && dy < 300.0 && s.vel[1] < 0.0 => B_BUTTON,
            // Otherwise the controller's A while the target is higher.
            _ => 0,
        }
    }
}
