//! `IslandHop::Jump` (technique `islandhop`): BoB star 3 ("Shoot to the
//! Island in the Sky") without the cannon or the wing cap, by a long jump
//! (or jump-dive) off the mountain onto the floating island.
//!
//! Where (levels/bob/areas/1/collision.inc.c, macro.inc.c):
//! - The island's top is the floor at y 3072, a hexagon with corners
//!   (2816, 1230), (3584, -99), (5120, -99), (5888, 1230) and the far side
//!   at z 2560. The star's ! box (`macro_box_star_3`, contents 11 = star
//!   index 2, sExclamationBoxContents in exclamation_box.inc.c) is at
//!   (5540, 3350, 1200).
//! - Two launches ([`Launch`]):
//!   - [`LEDGE`] (3341, 3007, -1564), on the mountain side: the 70-star TAS
//!     (ref/tas/ANALYSIS.md, "Island in the Sky", 20.5 s) side-flips and
//!     wall-kicks up the mountain side and dives off this ledge onto the
//!     island, ~1500 units (horizontal) from its corner (3584, 3072, -99)
//!     and 65 below its top. Starts: y >= 2850 within 2000 of it (the
//!     trees pass there rarely, e.g. v35-nofp-s2 (2058, 2917, -2189),
//!     v36-bandit-s2 (2799, 2919, -2581)). Hints: A at speed near the
//!     edge, B in the air (jump-dive).
//!   - [`SUMMIT`] (3072, 4097, -2559), the summit's lower step (collision
//!     data): 2513 from the island corner and 1025 above it. A long jump
//!     leaves at most 48 u/frame, vy 30, gravity 2 (mario.c, mario_step.c
//!     apply_gravity): about 52 frames and 2500 units with a ledge grab
//!     150 below the top, so only just in range (**unverified**; Ukikipedia:
//!     "simply long-jumping from the mountain summit"). Starts: y >= 3900
//!     (the trees reach the King's arena at y 4293, 1600-3000 away).
//!     Hints: Z then A (crouch slide into a long jump) near the edge.
//!
//! Success is Mario on the island (or hanging from its edge), or the star.
//! The GA does the inputs; the hints run Mario at the launch, then at the
//! island.

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_IN_CANNON, ACT_SHOT_FROM_CANNON, LEVEL_BOB};
use obrrt_core::{Budget, Objective, Progress, St, dist};

const COURSE: i16 = 1;
/// "Shoot to the Island in the Sky".
const INDEX: u8 = 2;
const ZONE: (u16, u8) = (LEVEL_BOB as u16, 1);
/// The ! box's contents byte for star index 2 (sExclamationBoxContents).
const BOX_CONTENTS: u32 = 11;
/// The box from the level data, if it isn't loaded.
const BOX: [f32; 3] = [5540.0, 3350.0, 1200.0];
/// The TAS's ledge on the mountain side (ref/tas/ANALYSIS.md).
const LEDGE: [f32; 3] = [3341.0, 3007.0, -1564.0];
/// The summit's corner nearest the island (collision data, see above).
const SUMMIT: [f32; 3] = [3072.0, 4097.0, -2559.0];
/// Where to land: a little inside the island's corner nearest the launches.
const LANDING: [f32; 3] = [3700.0, 3072.0, 150.0];
/// The island top's height and its bounding box (x, z).
const TOP: f32 = 3072.0;
const ISLAND_X: [f32; 2] = [2800.0, 5900.0];
const ISLAND_Z: [f32; 2] = [-120.0, 2580.0];
/// Starts near the ledge / on the summit: at least this high and this near.
const LEDGE_Y: f32 = 2850.0;
const LEDGE_RADIUS: f32 = 2000.0;
const SUMMIT_Y: f32 = 3900.0;
const SUMMIT_RADIUS: f32 = 3500.0;
/// The hop has failed once Mario is this far below both the island top and
/// his start (the ground under the island and the gap is ~700-1900).
const FAILED_DROP: f32 = 100.0;
/// Within this (horizontal) of the launch, run for the island.
const RUNUP: f32 = 350.0;

const MARIO_WING_CAP: u32 = 0x08;
const ACT_GRABBED: u32 = 0x0002_0370;
const ACT_WALKING: u32 = 0x0400_0440;
const ACT_CROUCH_SLIDE: u32 = 0x0480_8459;
const ACT_LONG_JUMP: u32 = 0x0300_0888;
const ACT_LEDGE_GRAB: u32 = 0x0800_034B;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Launch {
    Ledge,
    Summit,
}

impl Launch {
    fn spot(self) -> [f32; 3] {
        match self {
            Launch::Ledge => LEDGE,
            Launch::Summit => SUMMIT,
        }
    }

    /// The launch a state at `p` would use, if it is near one.
    fn of(p: [f32; 3]) -> Option<Launch> {
        if p[1] >= SUMMIT_Y && hdist(p, SUMMIT) < SUMMIT_RADIUS {
            Some(Launch::Summit)
        } else if p[1] >= LEDGE_Y && hdist(p, LEDGE) < LEDGE_RADIUS {
            Some(Launch::Ledge)
        } else {
            None
        }
    }
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The star's ! box in `s` (loaded, unbroken), else the level data's spot.
fn box_pos(s: &St) -> [f32; 3] {
    s.objects()
        .iter()
        .find(|o| o.bhv_name() == "bhvExclamationBox" && (o.beh_params >> 16) & 0xFF == BOX_CONTENTS)
        .map_or(BOX, |o| o.pos)
}

/// Mario on the island: standing on its top, or hanging from its edge.
fn on_island(s: &St) -> bool {
    let [x, y, z] = s.pos;
    let inside = (ISLAND_X[0]..=ISLAND_X[1]).contains(&x) && (ISLAND_Z[0]..=ISLAND_Z[1]).contains(&z);
    inside && ((s.action & ACT_FLAG_AIR == 0 && y >= TOP - 20.0) || (s.action == ACT_LEDGE_GRAB && y >= TOP - 200.0))
}

pub struct IslandHop;

impl IslandHop {
    pub fn new(b: &St) -> Option<IslandHop> {
        (b.zone() == ZONE && !b.has_star(COURSE, INDEX)).then_some(IslandHop)
    }
}

impl Objective for IslandHop {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "IslandHop::Jump"
    }
    fn key(&self) -> String {
        "IslandHop::Jump".to_string()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        15.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == ZONE
            && s.alive()
            && s.controllable()
            && !s.has_star(COURSE, INDEX)
            && s.mario_flags & MARIO_WING_CAP == 0
            && !matches!(s.action, ACT_IN_CANNON | ACT_SHOT_FROM_CANNON | ACT_GRABBED)
            && Launch::of(s.pos).is_some()
            // On the island already: met at the start, nothing to attempt.
            && !on_island(s)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        Launch::of(s.pos).map_or(f64::MAX, |l| s.dist(l.spot()) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // This star only (has_star), not any star collected on the way.
        traj.iter().position(|s| s.zone() == ZONE && (s.has_star(COURSE, INDEX) || on_island(s)))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Nearer the landing point (straight line), scored only until Mario
        // drops below the island: after that the hop has failed.
        let failed = traj[0].pos[1].min(TOP) - FAILED_DROP;
        let end = traj.iter().position(|s| s.zone() != ZONE || s.pos[1] < failed).unwrap_or(traj.len());
        let (at, d) = traj[..end]
            .iter()
            .enumerate()
            .map(|(i, s)| (i, dist(s.pos, LANDING)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        false
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Run to the launch corner, then at the island; over the island,
        // at the box.
        if on_island(s) || s.pos[2] > ISLAND_Z[0] {
            return Some(box_pos(s));
        }
        let grounded = s.action & ACT_FLAG_AIR == 0;
        Some(match Launch::of(s.pos) {
            Some(l) if grounded && hdist(s.pos, l.spot()) > RUNUP => l.spot(),
            _ => LANDING,
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        if on_island(s) || s.pos[2] > ISLAND_Z[0] {
            return 0;
        }
        let launch = Launch::of(s.pos);
        let at_edge = launch.is_some_and(|l| hdist(s.pos, l.spot()) <= RUNUP) && s.forward_vel >= 24.0;
        match s.action {
            // Summit: near the edge at speed, crouch slide, then long jump
            // out of it. Ledge: jump at speed near the edge.
            ACT_WALKING if at_edge && launch == Some(Launch::Summit) => Z_TRIG,
            ACT_WALKING if at_edge => A_BUTTON,
            ACT_CROUCH_SLIDE if launch == Some(Launch::Summit) => A_BUTTON,
            ACT_LONG_JUMP => 0,
            // In a plain jump's arc off the edge, dive for reach.
            a if a & ACT_FLAG_AIR != 0 && s.vel[1] < 0.0 && s.pos[2] > LEDGE[2] - 200.0 => B_BUTTON,
            a if a & ACT_FLAG_AIR != 0 && s.vel[1] < 0.0 && s.pos[1] < SUMMIT[1] - 300.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == ZONE.0 && !s.has_star(COURSE, INDEX)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// Where the star the box spawns (`bhvSpawnedStar`, its params copied from
/// the box: index 2 in the top byte) comes to rest (its home: SET_HOME,
/// set_y_home_to_pos), while it is loaded.
fn spawned_star(s: &St) -> Option<[f32; 3]> {
    s.objects()
        .iter()
        .find(|o| o.bhv_name() == "bhvSpawnedStar" && (o.beh_params >> 24) as u8 == INDEX)
        .map(|o| o.home)
}

/// `IslandHop::Star`: from the island, break the box and collect its star.
/// Written when the registry's `Star::Collect` was made only for objects
/// with a star's interaction, which a spawned star gets only once its spawn
/// cutscene is over (action 3, bhv_spawned_star_loop in
/// sparkle_spawn_star.inc.c): the states saved right after the box broke
/// were mid-cutscene, and with pick_by_b only they could start it (v1/v2
/// tests: box broken 2-3 times per run, star never collected). The registry
/// now also makes `Star::Collect` for spawning stars (star.rs), from the
/// spawn on; this one still differs in starting from any island state
/// before the box is broken (or one with the star out), aiming at the box
/// (jumping into it from below breaks it) and then at the star, and it
/// succeeds only on the star.
pub struct IslandStar {
    /// The star's rest point if it was out at generation, else the box.
    pos: [f32; 3],
}

impl IslandStar {
    /// The box (if still loaded) or the star (if out) in `s`, else `pos`.
    fn target(&self, s: &St) -> [f32; 3] {
        spawned_star(s).or_else(|| loaded_box(s)).unwrap_or(self.pos)
    }
}

/// The star's ! box in `s`, if it is loaded (not yet broken).
fn loaded_box(s: &St) -> Option<[f32; 3]> {
    s.objects()
        .iter()
        .find(|o| o.bhv_name() == "bhvExclamationBox" && (o.beh_params >> 16) & 0xFF == BOX_CONTENTS)
        .map(|o| o.pos)
}

impl Objective for IslandStar {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "IslandHop::Star"
    }
    fn key(&self) -> String {
        "IslandHop::Star".to_string()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == ZONE && s.alive() && s.controllable() && !s.has_star(COURSE, INDEX) && (on_island(s) || spawned_star(s).is_some())
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.target(s)) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.has_star(COURSE, INDEX))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == ZONE)
            .map(|(i, s)| (i, dist(s.pos, self.pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        // The spawn cutscene stops time for ~100 frames first.
        Budget::Ga { population: 24, generations: 6, horizon: 300 }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.target(s))
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Under the box: jump into it.
        let t = self.target(s);
        if s.action & ACT_FLAG_AIR == 0 && hdist(s.pos, t) < 120.0 && t[1] > s.pos[1] + 150.0 { A_BUTTON } else { 0 }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == ZONE.0 && !s.has_star(COURSE, INDEX)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// Offered from any BoB state while star 3 is missing (starts are summit
/// states; `base_criteria`), and [`IslandStar`] (starts on the island).
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if let Some(o) = IslandHop::new(b) {
        out.push(Arc::new(o));
    }
    if b.zone() == ZONE && !b.has_star(COURSE, INDEX) {
        out.push(Arc::new(IslandStar { pos: spawned_star(b).unwrap_or(BOX) }));
    }
}
