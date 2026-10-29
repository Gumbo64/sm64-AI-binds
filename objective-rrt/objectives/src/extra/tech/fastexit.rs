//! `Tech::FastExit` (technique `fastexit`, on with `OBRRT_FASTEXIT=1`):
//! grab a star while a death warp is already pending, so the course is left
//! by the death exit instead of the star dance, the star-exit warp, the
//! kick-out and the save menu (catalogue G13, "death-fade star grab").
//!
//! The mechanics (sm64-port src/game):
//! - A normal grab: `interact_star_or_key` (interaction.c) saves the star at
//!   once (`save_file_collect_star_or_key`) and starts ACT_STAR_DANCE_EXIT
//!   (ACT_FALL_AFTER_STAR_GRAB in the air, then the dance on landing). The
//!   dance calls `level_trigger_warp(WARP_OP_STAR_EXIT)` on its 80th frame
//!   (general_star_dance_handler, mario_actions_cutscene.c), the warp fires
//!   32 frames later (level_update.c), Mario is kicked out of the painting
//!   (ACT_EXIT_AIRBORNE, then ACT_EXIT_LAND_SAVE_DIALOG) and the save menu
//!   must be answered.
//! - `level_trigger_warp` does nothing while another warp is pending
//!   (`sDelayedWarpOp != WARP_OP_NONE`). So a star touched with a death warp
//!   queued is saved, the dance starts, its STAR_EXIT is ignored, and the
//!   death warp fires on its own timer: the castle arrival is the death
//!   exit (ACT_DEATH_EXIT / ACT_FALLING_DEATH_EXIT, one life lost, no save
//!   menu; act_death_exit decrements the lives).
//! - Death warps that leave Mario in control while they count down:
//!   - over a SURFACE_DEATH_PLANE floor, below its height + 2048
//!     (`check_death_barrier`, from `mario_handle_special_floors`, which
//!     skips cutscene actions): WARP_OP_WARP_FLOOR, 20 frames;
//!   - no floor under him (out of bounds; `update_mario_geometry_inputs`,
//!     mario.c, every action): WARP_OP_DEATH, 48 frames.
//!   Both are queued in `execute_mario_action` before the frame's
//!   interactions, from the position the previous frame ended at, so a
//!   state that ends a frame doomed (and not in a cutscene action) has the
//!   warp queued by the next frame's star touch. With 0 lives either warp
//!   is a game over, so the trick needs a spare life.
//! - "Star dance in the sky" (a star grabbed above a pit) is not a fast
//!   exit: ACT_FALL_AFTER_STAR_GRAB is a cutscene action, so the death plane
//!   never triggers during the fall; Mario lands on it and dances, then the
//!   normal star exit runs, after the extra fall.
//!
//! Where it applies: a star within a few hundred units (20 frames of free
//! fall and air steering) of a point below a death plane's line, or a star
//! reachable right after an out-of-bounds frame (BoB's star 6 through the
//! gate, `gateclip`, which with Star::Collect already gets it in the window
//! from the doomed nodes). `run/examples/fastexit_lab.rs scan` lists, per
//! level, the death planes near each star; [`SPOTS`] are the rough drop
//! regions found that way (rough-location rule: a region to drop from, the
//! GA finds the run-up and the fall). Anywhere else the objective is also
//! offered from a state that is already over a death plane near a star.
//!
//! Progress requires the trick: a trajectory scores in tiers, doomed (then
//! closeness to the star within the warp's frames) above over a death plane
//! (then height over its line and distance to the star) above neither (then,
//! at a known spot, distance to the drop region only). Success is the star
//! with a doomed state before the grab; a plain grab is no success.

use std::sync::{Arc, OnceLock};

use obrrt_core::layout::{ACT_GROUP_CUTSCENE, ACT_GROUP_MASK};
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use crate::star::{rest_pos, spawned_star, star_blocked, star_of};

/// `SURFACE_DEATH_PLANE` (surface_terrains.h).
const SURFACE_DEATH_PLANE: i16 = 0x0A;
/// `check_death_barrier`: the warp is queued below the plane's height + this.
const LINE: f32 = 2048.0;
/// Frames between the queue and the warp (level_update.c, level_trigger_warp).
const PLANE_FRAMES: usize = 20;
const OOB_FRAMES: usize = 48;
/// Offered from states this close (straight line) to the star.
const RADIUS: f32 = 2500.0;
/// Within this of a spot's drop point, steer at the star.
const DROP_REACH: f32 = 250.0;

/// On with `OBRRT_FASTEXIT=1` (and `fastexit` in `OBRRT_TECH`).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_FASTEXIT").is_ok_and(|v| v == "1"))
}

/// A rough drop region for one star: from about `drop`, fall past the death
/// line and reach the star within the 20 frames.
struct Spot {
    level: u16,
    area: u8,
    /// Course star index (behaviour param byte 0).
    index: u8,
    drop: [f32; 3],
}

/// Known spots (from `fastexit_lab scan` and the 2026 TAS; see the module
/// doc). Our own levels (BoB, WF, CCM) have none: their death planes' lines
/// are 2500-5300 under every star (WF -1023, CCM -6143), and BoB's star 6
/// (out of bounds at the gate) is `gateclip`'s.
/// - TTM star 2 (index 1, the spawned star by the pool, ~(2500, -1200,
///   1294)): the TAS (f87270-87319) falls in a jump down the pit by the pool
///   at (2545, -4028, 890) past the line (-6143) at f87299, is put back up at
///   the pool's surface (y -1772, WATER_PLUNGE) the next frame, double-jumps
///   and touches the star at f87319, 20 frames after the doom; the death
///   exit follows.
/// - (HMC's four, f59196-61380, come from parallel-universe positions,
///   x ~583000, which the search doesn't make; not listed.)
const SPOTS: &[Spot] = &[Spot { level: 36, area: 1, index: 1, drop: [2545.0, -4000.0, 890.0] }];

/// Whether the frame after `s` has a death warp queued that leaves Mario in
/// control, and the warp's frames: None if not (or if it would be a game
/// over, or isn't a course).
pub fn doom(s: &St) -> Option<usize> {
    if !(1..=24).contains(&s.course) || s.lives < 1 || s.play_mode != 0 || s.action == 0 {
        return None;
    }
    if s.action & ACT_GROUP_MASK == ACT_GROUP_CUTSCENE {
        return None;
    }
    if s.floor_type == SURFACE_DEATH_PLANE && s.pos[1] < s.floor_y + LINE {
        Some(PLANE_FRAMES)
    } else if s.floor_type == -1 {
        Some(OOB_FRAMES)
    } else {
        None
    }
}

fn over_plane(s: &St) -> bool {
    s.floor_type == SURFACE_DEATH_PLANE && s.action & ACT_GROUP_MASK != ACT_GROUP_CUTSCENE
}

/// How far above the death line a star can be and still be touched once
/// doomed: Mario is falling by then (star hitbox height 50 plus Mario's 160
/// hurtbox, and a little of a jump's rise left).
const ABOVE_LINE: f32 = 300.0;

/// Over a death plane whose line is close enough under `star` for the
/// trick (a star 3500 above the line, WF's, would take 45+ frames of fall).
fn plane_near(s: &St, star: [f32; 3]) -> bool {
    over_plane(s) && star[1] < s.floor_y + LINE + ABOVE_LINE
}

pub struct FastExit {
    id: ObjId,
    pos: [f32; 3],
    zone: (u16, u8),
    star: (i16, u8),
    drop: Option<[f32; 3]>,
}

impl FastExit {
    fn new(b: &St, o: &Obj) -> Option<FastExit> {
        let star = star_of(b, o)?;
        let exit = o.interact_type == obrrt_core::layout::INTERACT_STAR_OR_KEY || spawned_star(o);
        // The 100-coin star (and secret stars) don't exit anyway.
        if !exit || o.bhv_name() == "bhvSpawnedStarNoLevelExit" {
            return None;
        }
        let (level, area) = b.zone();
        let drop = SPOTS.iter().find(|p| p.level == level && p.area == area && p.index == star.1).map(|p| p.drop);
        Some(FastExit { id: o.id, pos: rest_pos(o), zone: b.zone(), star, drop })
    }

    fn here(&self, s: &St) -> bool {
        s.find(self.id).is_some()
    }

    /// Index of the first doomed state and its warp's frames.
    fn doomed(traj: &[St]) -> Option<(usize, usize)> {
        traj.iter().enumerate().find_map(|(i, s)| doom(s).map(|w| (i, w)))
    }
}

/// From `tech::generate`, per object near `b`.
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    if !on() {
        return;
    }
    if let Some(x) = FastExit::new(b, o) {
        // Without a known spot, only from a state already over a death
        // plane (or doomed) near the star.
        if x.drop.is_some() || plane_near(b, x.pos) || doom(b) == Some(OOB_FRAMES) {
            out.push(Arc::new(x));
        }
    }
}

impl Objective for FastExit {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Tech::FastExit"
    }
    fn key(&self) -> String {
        format!("Tech::FastExit {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // A known spot, or Mario already over the pit: worth a star's exit
        // (~160 frames); else a long shot.
        if self.drop.is_some() || plane_near(b, self.pos) { 12.0 } else { 3.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        let (c, i) = self.star;
        s.zone() == self.zone
            && self.here(s)
            && !s.has_star(c, i)
            && !star_blocked(s, c, i)
            && s.lives >= 1
            && s.alive()
            && s.controllable()
            && s.dist(self.pos) < RADIUS
            && (self.drop.is_some() || plane_near(s, self.pos) || doom(s) == Some(OOB_FRAMES))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        match self.drop {
            Some(d) if doom(s).is_none() && !over_plane(s) => s.dist(d) as f64 / 400.0,
            _ => s.dist(self.pos) as f64 / 400.0,
        }
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let (c, i) = self.star;
        let (d, _) = Self::doomed(traj)?;
        traj.iter().enumerate().skip(d + 1).find(|(_, s)| s.zone() == self.zone && s.has_star(c, i)).map(|(k, _)| k)
    }
    fn success_heuristic(&self, traj: &[St]) -> f64 {
        // Rank by when the warp fires (the castle arrival), not the grab.
        match Self::doomed(traj) {
            Some((d, w)) => traj[d].frame as f64 + w as f64,
            None => f64::MAX,
        }
    }
    fn progress(&self, traj: &[St]) -> Progress {
        if let Some((d, w)) = Self::doomed(traj) {
            let end = (d + w + 1).min(traj.len());
            let (at, near) = (d..end)
                .map(|k| (k, dist(traj[k].pos, self.pos)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap_or((d, f32::MAX));
            return Progress { score: 40000.0 - near as f64, at };
        }
        let over = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == self.zone && over_plane(s))
            .map(|(k, s)| (k, (s.pos[1] - (s.floor_y + LINE)).max(0.0) + s.hdist(self.pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((at, g)) = over {
            return Progress { score: 20000.0 - g as f64, at };
        }
        match self.drop {
            Some(p) => {
                let (at, d) = traj
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.zone() == self.zone)
                    .map(|(k, s)| (k, dist(s.pos, p)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap_or((0, f32::MAX));
                Progress { score: -(d as f64), at }
            }
            None => Progress { score: -1e6, at: 0 },
        }
    }
    fn success_value(&self) -> f64 {
        50.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Doomed: the star. Over the plane, above its line: straight down.
        // Else a spot's drop region, then the star.
        if doom(s).is_some() {
            return Some(self.pos);
        }
        if over_plane(s) {
            return Some([s.pos[0], s.floor_y + LINE - 200.0, s.pos[2]]);
        }
        match self.drop {
            Some(d) if s.hdist(d) > DROP_REACH => Some(d),
            _ => Some(self.pos),
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.here(s)
    }
}
