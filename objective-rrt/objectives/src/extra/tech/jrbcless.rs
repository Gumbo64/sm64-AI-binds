//! `Jrb::PillarCannonless` (technique `jrbcless`): JRB star 5, "Blast to the
//! Stone Pillar", without the cannon: climb out of the water onto something
//! high near the cliff, jump and wall-kick off the cliff face, and ledge-grab
//! the stone platform the star's ! box floats over.
//!
//! The place (decomp, levels/jrb/areas/1/collision.inc.c and script.c):
//! - The platform is a ledge 103 units thick sticking out of the bay's back
//!   cliff: floor y 1946 over the quad (522, 2287), (1762, 1734),
//!   (1853, 2191), (1546, 2703); the cliff behind it is a true wall (normals
//!   (0.06, 0, 1.00) and (0.66, 0, 0.75)) from below the water up to 4096.
//!   The ! box (`bhvExclamationBox`, behaviour byte 0 = 4, star index 4) is
//!   at (1540, 2160, 2130), 214 above the floor, in every act (US; the JP
//!   version places a bare star).
//! - The water surface is 1024, 922 below the floor. A water jump reaches
//!   about 1190, nowhere near.
//!
//! Launch spots (where the climb starts):
//! - The stone spire next to the platform: a thin column at about
//!   (130, -500..2304, 2720) whose faces have normal y 0.04, so they're
//!   floors, not walls (surface_load.c: normal.y > 0.01 is a floor), and
//!   Mario can walk a little way up out of the water. The 120-star TAS by
//!   rikku (ref/tas/dumps/full-120star-rikku-U.csv, frames 7824-7894, act 3)
//!   swims up the spire, walks from y 1033 to 1244 on it, butt-slides off,
//!   jumps at the cliff, hits it (ACT_AIR_HIT_WALL), wall-kicks from y 1324
//!   to 1836 and ledge-grabs the platform at (777, 1946, 2380): 70 frames
//!   from leaving the water. Nothing act-dependent is used, so it works in
//!   act 1 too. That's [`SPIRE`] and [`LEDGE`] below.
//! - The floating ship (`bhvInSunkenShip3`, its collision, acts 2-6 only,
//!   at (4880, 820, 2375), script.c): the catalogue's route (Ukikipedia's
//!   Cless page: long jump off the ship, then wall kick). Its highest deck is
//!   the stern, local y 1536 at local z -1945..-1740
//!   (levels/jrb/wooden_ship/collision.inc.c), so absolute about
//!   (4880, 2356, 535): 410 above the platform but ~3,000 from it, so this
//!   launch is the long shot of the two. Taken from the loaded objects at
//!   runtime when the ship is there (it rocks a little; that's ignored).
//!
//! So the objective rewards getting near the platform from below, with a
//! bonus once a wall kick has happened, and succeeds when Mario stands on
//! (or ledge-grabs) the platform, or has the star. The jumps, the kick timing
//! and the ledge grab are the GA's; the hints only point at the launch spot
//! (in the water) and at the ledge (out of it), with A on alternate frames in
//! the air so a kick is tried at the wall.
//!
//! Hyperspeed mode (`Jrb::PillarHyper`, technique `jrbclesshyper`; c5 §6,
//! the 2026 TAS f82451-82613, 142 frames from control to the platform):
//! 1. long jump onto the dock slope at (-5115, 947, 1230) (type 0x15, ny
//!    0.970) facing downhill (yaw 17561, +x) and BLJ in place, A+Z on
//!    alternate frames, no pause: -9.8 -> -477 in 10 presses (f82522-82541;
//!    the slope ends each jump at once, so x stays -5115 until -150);
//! 2. the BLJ carries him west along the dock at -470; the wall at x -7642
//!    (normal (0.99, 0, 0.14)) flips his facing (f82547 WALKING, f82548
//!    DECELERATING at -429, facing -x), so the speed now carries him east;
//! 3. off the dock edge (FREEFALL f82553), B: DIVE straight into
//!    DIVE_SLIDE on the rock at (-5053, 984, 2125), A: BACKWARD_ROLLOUT at
//!    -405 for 17 frames over the water (y 950-1110);
//! 4. lands at the spire base (-17, 980, 2746), Z: CROUCH_SLIDE, then
//!    BRAKING at -305 whose quarter steps climb the spire's ny 0.04 floor
//!    faces from 980 to 2343 in 4 frames (f82576-82579);
//! 5. BUTT_SLIDE down the far face, A at y ~1910: STEEP_JUMP backwards at
//!    -98, landing on the platform (y 1946) at f82593.
//!
//! The objective: base = on the dock near the slope and slow (the BLJ
//! start), or backwards at |fv| > 150 out of the water between the dock
//! and the spire, or on the spire backwards; heuristic = straight line to
//! the platform (via the slope from the dock); success = the platform test
//! above (or the star), no cannon. Progress in stages: the graded BLJ
//! metric (blj.rs) on the dock, then nearness to the spire base at
//! hyperspeed, then height gained on the spire (+ nearness to the box once
//! above 1900). The dock part is a rule overlay ([`dock_rule`]: the
//! TAS's run-up, two long jumps onto the slope, then A+Z on landings with
//! the genome's stick on the press frames), up to -400; the pinned BLJ
//! wasn't reproduced in the lab (see [`dock_rule`]). After that the genome
//! plays, with B forced on a
//! fast FREEFALL frame and A on a fast DIVE_SLIDE frame (both one-frame
//! windows), and hints for the rest (Z on the spire base, A in the butt
//! slide above 1850).
//!
//! Gating: nothing blocks this star (`star_blocked` covers BoB's star 6
//! only), so there's nothing to soften. The cannon route is excluded from
//! success: a trajectory that went through the cannon doesn't count.

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::*;
use obrrt_core::{Budget, GamePad, Obj, Objective, PadOverlay, Progress, St, dist};

const COURSE: i16 = 3;
const INDEX: u8 = 4;
const AREA: u8 = 1;
const BOX: &str = "bhvExclamationBox";
const SHIP: &str = "bhvInSunkenShip3";
/// The platform's floor is this far below the box (2160 - 1946).
const BOX_ABOVE_FLOOR: f32 = 214.0;
/// The platform reaches this far from the box horizontally (its far corner
/// (522, 2287) is 1030 from it).
const PLATFORM_REACH: f32 = 1100.0;
/// Where the TAS leaves the water onto the spire (rikku, frame 7830).
const SPIRE: [f32; 3] = [127.0, 1100.0, 2779.0];
/// Where the TAS ledge-grabs the platform (rikku, frame 7894).
const LEDGE: [f32; 3] = [777.0, 1946.0, 2380.0];
/// Offered from states this close to the ledge (straight line).
const OFFER_RADIUS: f32 = 4000.0;
/// The ship's stern deck relative to the ship object (local y, local z).
const SHIP_STERN: [f32; 2] = [1536.0, -1840.0];
const BONUS_KICK: f64 = 250.0;

/// The star's ! box, if loaded (it's gone once broken).
fn star_box(s: &St) -> Option<&Obj> {
    s.objects().iter().find(|o| o.bhv_name() == BOX && (o.beh_params >> 24) as u8 == INDEX)
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn in_jrb(s: &St) -> bool {
    s.course == COURSE && s.area == AREA
}

fn cannon(s: &St) -> bool {
    s.action == ACT_IN_CANNON || s.action == ACT_SHOT_FROM_CANNON
}

/// Launch spots: the spire, and the ship's deck when it's afloat.
fn launches(s: &St) -> Vec<[f32; 3]> {
    let mut v = vec![SPIRE];
    if let Some(o) = s.objects().iter().find(|o| o.bhv_name() == SHIP) {
        v.push([o.pos[0], o.pos[1] + SHIP_STERN[0], o.pos[2] + SHIP_STERN[1]]);
    }
    v
}

/// On the platform: supported (not airborne, swimming or on a pole) at its
/// floor height, within its reach of the box.
fn on_platform(s: &St, boxpos: [f32; 3]) -> bool {
    let floor = boxpos[1] - BOX_ABOVE_FLOOR;
    s.alive()
        && s.action & (ACT_FLAG_AIR | ACT_FLAG_SWIMMING | ACT_FLAG_ON_POLE) == 0
        && s.pos[1] >= floor - 10.0
        && s.pos[1] <= floor + 400.0
        && hdist(s.pos, boxpos) < PLATFORM_REACH
}

pub struct PillarCannonless {
    zone: (u16, u8),
    boxpos: [f32; 3],
}

impl PillarCannonless {
    pub fn new(b: &St) -> Option<Arc<dyn Objective>> {
        if !in_jrb(b) || b.has_star(COURSE, INDEX) || !b.alive() {
            return None;
        }
        let o = star_box(b)?;
        let x = PillarCannonless { zone: b.zone(), boxpos: o.pos };
        (b.dist(LEDGE) < OFFER_RADIUS && !on_platform(b, o.pos)).then(|| Arc::new(x) as Arc<dyn Objective>)
    }
}

impl Objective for PillarCannonless {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Jrb::PillarCannonless"
    }
    fn key(&self) -> String {
        format!("Jrb::PillarCannonless {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        4.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.alive()
            && !s.has_star(COURSE, INDEX)
            && !cannon(s)
            && s.dist(LEDGE) < OFFER_RADIUS
            && !on_platform(s, self.boxpos)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Via the nearest launch spot (straight lines), or straight at the
        // ledge from above the water.
        let via = launches(s).iter().map(|l| s.dist(*l) + dist(*l, LEDGE)).fold(f32::MAX, f32::min);
        let direct = if s.pos[1] > s.water_level as f32 { s.dist(LEDGE) } else { f32::MAX };
        via.min(direct) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // This star or the platform only, not any star collected on the way
        // (base_criteria excludes states that already meet either).
        for (i, s) in traj.iter().enumerate() {
            if cannon(s) {
                return None;
            }
            if s.has_star(COURSE, INDEX) || (s.zone() == self.zone && on_platform(s, self.boxpos)) {
                return Some(i);
            }
        }
        None
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Nearer the ledge (straight line), plus a bonus once a wall kick
        // has happened above the water.
        let mut kicked = false;
        let mut best = Progress { score: f64::MIN, at: 0 };
        for (i, s) in traj.iter().enumerate() {
            if s.zone() != self.zone || cannon(s) {
                break;
            }
            if s.action == ACT_WALL_KICK_AIR && s.pos[1] > s.water_level as f32 {
                kicked = true;
            }
            let score = -(s.dist(LEDGE) as f64) + if kicked { BONUS_KICK } else { 0.0 };
            if score > best.score {
                best = Progress { score, at: i };
            }
        }
        best
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // In (or under) the water: the nearest launch spot. Out of it: the
        // ledge, which runs Mario at the cliff below it.
        if s.pos[1] <= s.water_level as f32 + 20.0 || s.action & ACT_FLAG_SWIMMING != 0 {
            return launches(s).into_iter().min_by(|a, b| s.dist(*a).total_cmp(&s.dist(*b)));
        }
        Some(LEDGE)
    }
    fn steer_button(&self, s: &St) -> u16 {
        // A on alternate frames in the air near the cliff, so a kick is tried
        // when he hits it; otherwise the controller's usual jump pattern.
        if s.action & ACT_FLAG_AIR != 0 && hdist(s.pos, LEDGE) < 700.0 && s.pos[1] < LEDGE[1] {
            A_BUTTON
        } else {
            0
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(COURSE, INDEX) && star_box(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}

// ---------------------------------------------------------------------
// Hyperspeed mode (`jrbclesshyper`).

/// The TAS's BLJ spot on the dock slope (f82522) and its facing (downhill).
const SLOPE: [f32; 3] = [-5115.0, 947.0, 1230.0];
const SLOPE_YAW: i32 = 17561;
/// Where the TAS's run-up starts (JRB's spawn on the dock, f82451).
const DOCK_START: [f32; 3] = [-6703.0, 1126.0, 1482.0];
/// The rule stops pressing past this backwards speed (TAS: -320 -> -477).
const BLJ_STOP: f32 = 400.0;
/// Where the backward rollout lands at the spire's foot (TAS f82572).
const SPIRE_BASE: [f32; 3] = [-17.0, 980.0, 2746.0];
/// The spire's axis (xz) and how far from it counts as on it.
const SPIRE_AXIS: [f32; 3] = [60.0, 0.0, 2730.0];
const SPIRE_RADIUS: f32 = 350.0;
/// Backwards speed that counts as hyperspeed here.
const HYPER: f32 = 150.0;
/// The dock: west of this x, above this y.
const DOCK_X: f32 = -4800.0;
const DOCK_Y: f32 = 900.0;
const ACT_FREEFALL: u32 = 0x0100_088C;
const ACT_DIVE_SLIDE: u32 = 0x0088_0456;
const ACT_BUTT_SLIDE: u32 = 0x0084_0452;
const ACT_WALKING: u32 = 0x0400_0440;
const ACT_DECELERATING: u32 = 0x0400_044A;
const ACT_GROUP_SUBMERGED: u32 = 0x0C0;

fn wet(s: &St) -> bool {
    s.action & ACT_FLAG_SWIMMING != 0 || s.action & ACT_GROUP_MASK == ACT_GROUP_SUBMERGED
}

/// On the dock, slow: where the BLJ starts from.
fn on_dock(s: &St) -> bool {
    s.pos[0] < DOCK_X && s.pos[1] > DOCK_Y && !wet(s) && s.forward_vel.abs() < 100.0 && s.hdist(SLOPE) < 2500.0
}

/// Backwards at hyperspeed out of the water (between the dock and the spire).
fn fast(s: &St) -> bool {
    s.forward_vel < -HYPER && !wet(s) && s.pos[1] > DOCK_Y && s.pos[0] < SPIRE_AXIS[0] + 600.0
}

fn on_spire(s: &St) -> bool {
    !wet(s) && s.pos[1] > SPIRE_BASE[1] - 30.0 && hdist(s.pos, SPIRE_AXIS) < SPIRE_RADIUS
}

pub struct PillarHyper {
    zone: (u16, u8),
    boxpos: [f32; 3],
}

impl PillarHyper {
    pub fn new(b: &St) -> Option<Arc<dyn Objective>> {
        if !in_jrb(b) || b.has_star(COURSE, INDEX) || !b.alive() {
            return None;
        }
        let o = star_box(b)?;
        let x = PillarHyper { zone: b.zone(), boxpos: o.pos };
        x.base_criteria(b).then(|| Arc::new(x) as Arc<dyn Objective>)
    }

    /// One state's progress value (see the module doc), if it counts.
    /// `hyper`: the trajectory has been at hyperspeed by `s` (a `fast`
    /// state, a backwards start on the spire, or a BLJ chain); the spire
    /// and platform credit needs it (audit: it counted however he got up).
    fn value(&self, s: &St, hyper: bool) -> Option<f64> {
        if hyper && (on_spire(s) || (s.pos[1] > 1900.0 && hdist(s.pos, self.boxpos) < PLATFORM_REACH && !wet(s))) {
            let near_box = if s.pos[1] > 1900.0 { (1200.0 - hdist(s.pos, self.boxpos)).max(0.0) } else { 0.0 };
            return Some(3000.0 + (s.pos[1] - SPIRE_BASE[1]).clamp(0.0, 1400.0) as f64 + near_box as f64);
        }
        if fast(s) {
            return Some(2000.0 + 1000.0 * (1.0 - s.hdist(SPIRE_BASE) as f64 / 8000.0).max(0.0));
        }
        None
    }
}

impl Objective for PillarHyper {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Jrb::PillarHyper"
    }
    fn key(&self) -> String {
        format!("Jrb::PillarHyper {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        4.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.alive()
            && !s.has_star(COURSE, INDEX)
            && !cannon(s)
            && !on_platform(s, self.boxpos)
            && (on_dock(s) || fast(s) || (on_spire(s) && s.forward_vel < -HYPER / 2.0))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // From the dock via the slope; at speed straight at the ledge.
        let d = if on_dock(s) { s.dist(SLOPE) + dist(SLOPE, LEDGE) } else { s.dist(LEDGE) };
        d as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        for (i, s) in traj.iter().enumerate() {
            if cannon(s) {
                return None;
            }
            if s.has_star(COURSE, INDEX) || (s.zone() == self.zone && on_platform(s, self.boxpos)) {
                return Some(i);
            }
        }
        None
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Up to the first water entry, death or zone change.
        let end = traj
            .iter()
            .position(|s| s.zone() != self.zone || !s.alive() || cannon(s) || wet(s))
            .unwrap_or(traj.len())
            .max(1);
        let traj = &traj[..end];
        let (m, _) = super::blj::metric(traj);
        let mut best = Progress { score: 100.0 * m.min(15.0), at: traj.len() - 1 };
        let mut hyper = m > 0.0 || (on_spire(&traj[0]) && traj[0].forward_vel < -HYPER / 2.0);
        for (i, s) in traj.iter().enumerate() {
            hyper |= fast(s);
            if let Some(v) = self.value(s, hyper) {
                if v > best.score {
                    best = Progress { score: v, at: i };
                }
            }
        }
        best
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if on_dock(s) {
            return Some(SLOPE);
        }
        if on_spire(s) || s.pos[1] > 1900.0 {
            return Some(self.boxpos);
        }
        Some(SPIRE_BASE)
    }
    fn steer_button(&self, s: &St) -> u16 {
        let base = s.hdist(SPIRE_BASE) < 400.0 && s.pos[1] < SPIRE_BASE[1] + 150.0;
        match s.action {
            ACT_FREEFALL if s.forward_vel < -HYPER => B_BUTTON,
            ACT_DIVE_SLIDE if s.forward_vel < -100.0 => A_BUTTON,
            ACT_WALKING | ACT_DECELERATING if base && s.forward_vel < -HYPER => Z_TRIG,
            ACT_BUTT_SLIDE if on_spire(s) && s.pos[1] > 1850.0 => A_BUTTON,
            _ => 0,
        }
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(PadOverlay::keyed(move |traj, played, genome| {
            let s = traj.last().unwrap();
            let held = |b: u16| played.last().is_some_and(|p| p.button & b != 0);
            // The BLJ: from a slow dock start, until -400 or the chain breaks.
            let begun = traj.iter().any(super::blj::long_jumping);
            let broken = begun && !super::blj::long_jumping(s);
            let hyper = traj.iter().any(|t| t.forward_vel < -BLJ_STOP);
            if on_dock(&traj[0]) && !hyper && !broken && s.pos[0] < DOCK_X && !wet(s) {
                return dock_rule(traj, played, genome);
            }
            // One-frame windows at speed: B dives out of a freefall, A rolls
            // out of the dive slide (backwards, forwardVel < 0).
            let forced = match s.action {
                ACT_FREEFALL if s.forward_vel < -HYPER && !held(B_BUTTON) => B_BUTTON,
                ACT_DIVE_SLIDE if s.forward_vel < -100.0 && !held(A_BUTTON) => A_BUTTON,
                _ => 0,
            };
            if forced != 0 {
                return (GamePad::new(forced, genome.stick_x, genome.stick_y), true);
            }
            (genome, false)
        }))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(COURSE, INDEX) && star_box(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// The full-tilt raw stick that points `intendedYaw` along `yaw`
/// (camera-relative; adjust_analog_stick's 6-unit dead band added back).
fn stick_along(s: &St, yaw: i32) -> (i8, i8) {
    let rel = (yaw - s.cam_yaw as i32) as f32 * std::f32::consts::TAU / 65536.0;
    let t = 121.0;
    let raw = |v: f32| if v.abs() < 0.5 { 0 } else { (v + 6.0 * v.signum()).round().clamp(-127.0, 127.0) as i8 };
    (raw(t * rel.sin()), raw(-t * rel.cos()))
}

/// The 1st long jump's stick goes from forward to back at this speed
/// (lab, 2026-10-25: from the spawn the 2nd jump then comes down at
/// (-5114, 947, 1227) at -9.6, the TAS's (-5115, 947, 1230) at -9.8).
const JUMP1_CAP: f32 = 39.0;
/// In the air the stick also pushes sideways toward the spot's z beyond
/// this lateral offset.
const LATERAL_TOL: f32 = 4.0;

fn yaw_to(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// The dock part of the TAS's route as a rule (2026 TAS f82451-82541):
/// - walk to [`DOCK_START`], stop, then run along [`SLOPE_YAW`] (+x, down the
///   slope); at forwardVel 10 Z (CROUCH_SLIDE), then A: a long jump;
/// - the 1st long jump with the stick forward (TAS 16 -> 44) comes down on
///   the slope (f82488, x -5619); A+Z at once: the 2nd long jump, stick
///   back (44 -> -9.8), comes down on the slope again (f82522, x -5115)
///   moving backwards;
/// - then the BLJ: on each LONG_JUMP_LAND frame with forwardVel < 0 a new
///   A (Z held) re-jumps at x1.5. The press frames' stick is the genome's
///   (key frames), neutral between, back in the air.
///
/// Why the spot: the TAS's chain stays at x -5115.0 for 7 presses (-15.8
/// to -150, y +0.2 a jump): it's under the overhang of the bhvRockSolid at
/// (-5332, 1434, 1023), whose underside is a ceiling (a low-ceiling BLJ,
/// ref/blj-conditions.md) and whose top is a floor at y ~1025-1035 a few
/// units further in. Lab (scratchpad hyperlab, from v34-1b-s2 node 8854):
/// the rule lands at (-5114, 947, 1227) at -9.6, but no first-press stick
/// (81 tried) pinned the chain: each press either flies a normal long jump
/// (-15.5, 27 air frames) or snaps up onto the rock's top (y 1031) and
/// butt-slides. So the pin needs a position within a few units that the
/// rule doesn't reach; the GA gets the descent's last frames' stick and
/// the press sticks to find it. Unverified beyond that.
fn dock_rule(traj: &[St], played: &[GamePad], genome: GamePad) -> (GamePad, bool) {
    let s = traj.last().unwrap();
    let held = |b: u16| played.last().is_some_and(|p| p.button & b != 0);
    let jumps = traj.windows(2).filter(|w| w[1].action == super::blj::ACT_LONG_JUMP && w[0].action != super::blj::ACT_LONG_JUMP).count()
        + (traj[0].action == super::blj::ACT_LONG_JUMP) as usize;
    let fwd = stick_along(s, SLOPE_YAW);
    let back = stick_along(s, s.yaw as i32 + 0x8000);
    // In the air, also sideways toward the TAS spot (its z: the BLJ only
    // stays pinned there), 45 degrees off the forward / back stick.
    let side = {
        let f = s.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let (dx, dz) = (SLOPE[0] - s.pos[0], SLOPE[2] - s.pos[2]);
        // Lateral offset, positive toward yaw + 0x4000.
        let lat = dx * f.cos() - dz * f.sin();
        if lat.abs() < LATERAL_TOL { 0 } else if lat > 0.0 { 0x2000 } else { -0x2000 }
    };
    let air_fwd = stick_along(s, s.yaw as i32 + side);
    let air_back = stick_along(s, s.yaw as i32 + 0x8000 - side);
    let new_a = if held(A_BUTTON) { 0 } else { A_BUTTON };
    match s.action {
        super::blj::ACT_LONG_JUMP if jumps <= 1 => {
            // Forward up to JUMP1_CAP, then back, so the 2nd jump comes
            // down on the slope at the TAS's spot moving backwards.
            let st = if s.forward_vel < JUMP1_CAP { air_fwd } else { air_back };
            return (GamePad::new(Z_TRIG, st.0, st.1), false);
        }
        super::blj::ACT_LONG_JUMP if jumps == 2 => {
            // The last part of the descent (falling below 1100) is the
            // genome's stick (key frames): the spot is a few units wide.
            if s.vel[1] < 0.0 && s.pos[1] < 1100.0 && (genome.stick_x, genome.stick_y) != (0, 0) {
                return (GamePad::new(Z_TRIG, genome.stick_x, genome.stick_y), true);
            }
            return (GamePad::new(Z_TRIG, air_back.0, air_back.1), false);
        }
        super::blj::ACT_LONG_JUMP => return (GamePad::new(Z_TRIG, back.0, back.1), false),
        super::blj::ACT_LONG_JUMP_LAND if jumps <= 1 => return (GamePad::new(Z_TRIG | new_a, fwd.0, fwd.1), false),
        super::blj::ACT_LONG_JUMP_LAND => {
            if s.forward_vel < 0.0 && new_a != 0 {
                return (GamePad::new(Z_TRIG | A_BUTTON, genome.stick_x, genome.stick_y), true);
            }
            return (GamePad::new(Z_TRIG, 0, 0), false);
        }
        super::blj::ACT_CROUCH_SLIDE => return (GamePad::new(Z_TRIG | new_a, fwd.0, fwd.1), false),
        _ => {}
    }
    if jumps > 0 {
        // The chain broke: the genome's.
        return (genome, false);
    }
    // Run-up.
    if !traj.iter().any(|t| t.hdist(DOCK_START) < 40.0) {
        let d = s.hdist(DOCK_START);
        let (x, y) = stick_along(s, yaw_to(s.pos, DOCK_START));
        let k = if d > 700.0 { 1.0 } else { (d / 700.0).max(0.25) };
        return (GamePad::new(0, (x as f32 * k) as i8, (y as f32 * k) as i8), false);
    }
    let lined_up = ((s.yaw as i32 - SLOPE_YAW) as i16).unsigned_abs() < 0x180;
    if !lined_up && s.action & ACT_FLAG_MOVING != 0 {
        return (GamePad::new(0, 0, 0), false);
    }
    let z = lined_up && s.forward_vel >= 10.0 && !held(Z_TRIG);
    (GamePad::new(if z { Z_TRIG } else { 0 }, fwd.0, fwd.1), false)
}
