//! `Wf::Cless`: WF star 6 ("Blast Away the Wall") without the cannon.
//!
//! The star (590, 2450, 2650) sits inside the corner wall below the upper
//! walkway (y 2560). The cannon breaks the wall
//! (bhvWfBreakableWallRight breaks only for ACT_SHOT_FROM_CANNON,
//! breakable_wall.inc.c), but a ledge grab reaches it without one:
//! - when an airborne quarter step hits a lower wall with no upper wall,
//!   check_ledge_grab (mario_step.c:352-386) runs if Mario is falling
//!   (vel y <= 0) and the wall pushed him against his velocity;
//! - it looks for a floor 60 units *into* the wall, searching down from
//!   160 above him, and if that floor is more than 100 above him, puts him
//!   on it (ACT_LEDGE_GRAB), inside the corner, touching the star.
//!
//! v19-soft got the star this way by accident (5:22.6): Mario walked off
//! the walkway at about (642, 2560, 2723) in freefall, fell to y 2196 by
//! the corner's face, and the ledge grab put him at (628, 2319, 2724),
//! 155 units from the star, which he collected on the next frame. The TAS
//! grabs higher, at (665, 2477, 2620).
//!
//! So the objective starts from the walkway near the corner, rewards falling
//! off its edge beside the star (more for a ledge grab there), and succeeds
//! only on the star. The fall, the drift into the wall and the grab are left
//! to the GA.

use obrrt_core::layout::LEVEL_WF;
use obrrt_core::{Budget, Objective, Progress, St, dist};

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The best (highest) score `f` gives over the trajectory.
fn best(traj: &[St], f: impl Fn(&St) -> Option<f64>) -> Progress {
    let (at, score) = traj
        .iter()
        .enumerate()
        .filter_map(|(i, s)| f(s).map(|v| (i, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f64::MIN));
    Progress { score, at }
}

/// WF, star 6 (index 5).
const COURSE: i16 = 2;
const INDEX: u8 = 5;
const STAR: [f32; 3] = [590.0, 2450.0, 2650.0];
/// The TAS's ledge grab, beside the star.
const LEDGE: [f32; 3] = [665.0, 2477.0, 2620.0];
/// Where v19-soft walked off the walkway.
const DROP: [f32; 3] = [642.0, 2560.0, 2723.0];
/// The walkway's height above the corner; standing on it is "on top".
const TOP_Y: f32 = 2540.0;
/// Starts this close to the drop (straight line).
const START_RADIUS: f32 = 2000.0;

const ACT_LEDGE_GRAB: u32 = 0x0800_034B;
const ACT_LEDGE_CLIMB_SLOW_1: u32 = 0x0000_054C;
const ACT_LEDGE_CLIMB_FAST: u32 = 0x0000_054F;

fn ledge_action(s: &St) -> bool {
    s.action == ACT_LEDGE_GRAB || (ACT_LEDGE_CLIMB_SLOW_1..=ACT_LEDGE_CLIMB_FAST).contains(&s.action)
}

fn star_loaded(s: &St) -> bool {
    s.objects().iter().any(|o| crate::star::star_of(s, o) == Some((COURSE, INDEX)) && o.bhv_name() == "bhvStar")
}

pub struct Cless {
    zone: (u16, u8),
}

impl Cless {
    /// Offered in WF's main area while star 6 is missing and loaded, from
    /// states near the corner.
    pub fn new(b: &St) -> Option<Cless> {
        (b.level as u32 == LEVEL_WF
            && b.area == 1
            && !b.has_star(COURSE, INDEX)
            && b.pos[1] > 1800.0
            && b.dist(DROP) < START_RADIUS
            && star_loaded(b))
        .then_some(Cless { zone: b.zone() })
    }
}

impl Objective for Cless {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Wf::Cless"
    }
    fn key(&self) -> String {
        format!("Wf::Cless {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        8.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.alive()
            && !s.has_star(COURSE, INDEX)
            && s.pos[1] > TOP_Y - 40.0
            && s.dist(DROP) < START_RADIUS
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(DROP) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // This star only, not any star collected on the way (base_criteria
        // excludes states that already have it).
        traj.iter().position(|s| s.has_star(COURSE, INDEX))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // On the walkway: nearer the drop-off beside the star. Below its
        // edge (falling past the corner): nearer the star, which always
        // beats any state on top (standing right above the star doesn't
        // touch it). A ledge grab by the star is the clip itself.
        best(traj, |s| {
            if s.zone() != self.zone || !s.alive() {
                return None;
            }
            if s.pos[1] >= TOP_Y {
                return Some(-2000.0 - hdist(s.pos, LEDGE) as f64);
            }
            if s.pos[1] < 1800.0 {
                return None;
            }
            let d = dist(s.pos, STAR) as f64;
            let bonus = if ledge_action(s) && d < 300.0 { 500.0 } else { 0.0 };
            Some(-d + bonus)
        })
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 6, horizon: 200 }
    }
    fn success_value(&self) -> f64 {
        50.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // On top, walk out past the edge by the corner; once falling, drift
        // back into the corner's face, towards the star. Both targets are
        // below Mario, so the controller doesn't jump.
        Some(if s.pos[1] >= TOP_Y { [LEDGE[0] + 40.0, LEDGE[1], DROP[2]] } else { STAR })
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(COURSE, INDEX)
    }
    fn kinship(&self) -> u32 {
        10
    }
}
