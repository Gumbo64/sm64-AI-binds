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
use obrrt_core::{Eta, Mem, Objective, St, dist, frames_for};

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
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
/// Below this, past the corner's face, v1 scored nothing: the grab can't
/// happen any more without climbing back up.
const FLOOR_Y: f32 = 1800.0;

/// Stepping off the walkway's edge to the grab: a fall of ~240 (2560 to
/// v19-soft's 2319) at gravity 4, ~11 frames.
const DROP_FRAMES: f64 = 11.0;
/// The ledge grab itself, snapping him onto the floor inside the corner
/// (a frame or two of the airborne quarter steps).
const GRAB_FRAMES: f64 = 2.0;
/// In a ledge grab by the star: v19-soft collected it on the next frame.
const TOUCH_FRAMES: f64 = 1.0;

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
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
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
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 6, horizon: 200 })
    }
    /// `Done` on this star (v1's success). The zone left, dead, or below
    /// [`FLOOR_Y`]: `Lost`. On the walkway: the way to the drop-off by the
    /// ledge, [`DROP_FRAMES`] and [`GRAB_FRAMES`]; below its edge: the way
    /// to the star and the grab, or in a ledge grab by it [`TOUCH_FRAMES`].
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.has_star(COURSE, INDEX) {
            return Eta::Done;
        }
        if s.zone() != self.zone || !s.alive() || s.pos[1] < FLOOR_Y {
            return Eta::Lost;
        }
        if s.pos[1] >= TOP_Y {
            return Eta::Frames(frames_for(hdist(s.pos, LEDGE)) + DROP_FRAMES + GRAB_FRAMES);
        }
        let d = dist(s.pos, STAR);
        if ledge_action(s) && d < 300.0 {
            return Eta::Frames(TOUCH_FRAMES);
        }
        Eta::Frames(frames_for(d) + GRAB_FRAMES)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::tech::testkit::{ACT_WALKING, st};
    use obrrt_core::fold;

    const ACT_FREEFALL: u32 = 0x0100_088C;

    #[test]
    fn walk_off_fall_grab_star() {
        let at = |p, a| st(LEVEL_WF, 1, p, a);
        let obj = Cless { zone: (LEVEL_WF as u16, 1) };
        let mut star = at([628.0, 2319.0, 2724.0], ACT_LEDGE_GRAB);
        star.course_stars[(COURSE - 1) as usize] = 1 << INDEX;
        let traj = vec![
            at([1200.0, 2560.0, 2723.0], ACT_WALKING),
            at(DROP, ACT_WALKING),
            at([642.0, 2400.0, 2723.0], ACT_FREEFALL),
            at([628.0, 2319.0, 2724.0], ACT_LEDGE_GRAB),
            star,
        ];
        let (met, etas) = fold(&obj, &traj);
        assert_eq!(met, Some(4), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
        assert!(obj.eta(&at([628.0, 1700.0, 2724.0], ACT_FREEFALL), &traj[0], &mut Mem::default()) == Eta::Lost);
    }
}
