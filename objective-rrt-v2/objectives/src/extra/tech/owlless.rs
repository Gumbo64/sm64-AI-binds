//! `Owlless::Cage` (technique `owlless`): WF star 5 ("Fall onto the Caged
//! Island") without Hoot, by a triple jump, or a double jump plus a wall
//! kick, from the high ground near the cage into it.
//!
//! Where (levels/wf/script.c, levels/wf/areas/1/collision.inc.c):
//! - The star (`bhvStar`, params 0x04, all acts) is at (2880, 4300, 190).
//!   The cage's floor is a disc at y 4096, radius ~260, round the star; its
//!   fence's top is at ~4360. Under it there is only the ground at
//!   ~2000-2300, which the trees reach and ground-pound from, under the
//!   island (v34/v35 nodes.csv: many states at hdist < 400, y 3000-3900).
//! - High ground: the fortress top (floor 3584 from x ~1380 west) and the
//!   rotating platforms (bhvRotatingPlatform at (2688, 3584, -895) and
//!   (2304, 3584, -2303)), 1100-1500 units from the cage; the trees stand
//!   there (v34-1b-s1, v35-route-s2: states at y 3550-3700 within 500-1000).
//! - A triple jump rises ~595 (vy 69, gravity 4) and a wall kick ~480
//!   (mario.c, mario_step.c), so from y 3584 a triple jump alone tops out
//!   near the fence and the wall kick gets over it (Ukikipedia RTA guide,
//!   "Cage Wallkick"). Hoot only appears in acts 3+ after stars 1-2; this
//!   works in any act.
//!
//! Success is Mario in the cage (on its floor, inside the fence) or the
//! star. Steering aims above the cage, so the controller jumps; the GA does
//! the rest.

use std::sync::Arc;

use obrrt_core::buttons::A_BUTTON;
use obrrt_core::layout::{ACT_FLAG_AIR, ACT_RIDING_HOOT, LEVEL_WF};
use obrrt_core::{Eta, Mem, Objective, St, frames_for};

const COURSE: i16 = 2;
/// "Fall onto the Caged Island".
const INDEX: u8 = 4;
const ZONE: (u16, u8) = (LEVEL_WF as u16, 1);
/// The star from the level data, if it isn't loaded.
const STAR: [f32; 3] = [2880.0, 4300.0, 190.0];
/// The cage's floor and its radius inside the fence.
const FLOOR: f32 = 4096.0;
const CAGE_RADIUS: f32 = 260.0;
/// Aim this far above the cage floor (over the fence).
const ABOVE: f32 = 400.0;
/// Starts: at least this high and this near the star (horizontally).
const START_Y: f32 = 3300.0;
const START_RADIUS: f32 = 1600.0;
/// Under the island: within this of its centre and below its floor. Jumps
/// from here hit its underside, so it is no start and scores lower.
const UNDER_RADIUS: f32 = 450.0;
/// Under the island: getting out from under it before a jump can clear its
/// underside; v1's under-island penalty (800 units) at PACE.
const UNDER_FRAMES: f64 = 27.0;

const ACT_AIR_HIT_WALL: u32 = 0x0000_08A7;

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// The star in `s`, else the level data's spot.
fn star_pos(s: &St) -> [f32; 3] {
    // star_of reads only the first behavior parameter, so it must be a star.
    s.objects()
        .iter()
        .find(|o| o.bhv_name() == "bhvStar" && crate::star::star_of(s, o) == Some((COURSE, INDEX)))
        .map_or(STAR, |o| o.pos)
}

fn in_cage(s: &St, star: [f32; 3]) -> bool {
    hdist(s.pos, star) < CAGE_RADIUS && s.pos[1] >= FLOOR - 10.0
}

fn under(s: &St, star: [f32; 3]) -> bool {
    hdist(s.pos, star) < UNDER_RADIUS && s.pos[1] < FLOOR - 10.0
}

pub struct Owlless;

impl Owlless {
    pub fn new(b: &St) -> Option<Owlless> {
        (b.zone() == ZONE && !b.has_star(COURSE, INDEX)).then_some(Owlless)
    }
}

impl Objective for Owlless {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Owlless::Cage"
    }
    fn key(&self) -> String {
        "Owlless::Cage".to_string()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        15.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        if s.zone() != ZONE || !s.alive() || !s.controllable() || s.has_star(COURSE, INDEX) || s.action == ACT_RIDING_HOOT {
            return false;
        }
        let star = star_pos(s);
        s.pos[1] >= START_Y && hdist(s.pos, star) < START_RADIUS && !under(s, star) && !in_cage(s, star)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 180 })
    }
    /// `Done` on v1's success: in WF's area with this star, or in the cage
    /// (round the star as the start saw it). The zone left or dead: `Lost`.
    /// Else the straight line to the point over the cage, plus
    /// [`UNDER_FRAMES`] under the island.
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        let star = star_pos(start);
        if s.zone() == ZONE && (s.has_star(COURSE, INDEX) || in_cage(s, star)) {
            return Eta::Done;
        }
        if s.zone() != ZONE || !s.alive() {
            return Eta::Lost;
        }
        let aim = [star[0], FLOOR + ABOVE, star[2]];
        Eta::Frames(frames_for(s.dist(aim)) + if under(s, star) { UNDER_FRAMES } else { 0.0 })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Above the cage while outside it (the controller jumps while the
        // target is higher); inside, the star.
        let star = star_pos(s);
        Some(if in_cage(s, star) { star } else { [star[0], FLOOR + ABOVE, star[2]] })
    }
    fn steer_button(&self, s: &St) -> u16 {
        // A against a wall in the air: the wall kick.
        if s.action == ACT_AIR_HIT_WALL && s.action & ACT_FLAG_AIR != 0 { A_BUTTON } else { 0 }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level == ZONE.0 && !s.has_star(COURSE, INDEX)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// Offered from any WF state while star 5 is missing (starts are high
/// states near the cage; `base_criteria`).
pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if let Some(o) = Owlless::new(b) {
        out.push(Arc::new(o));
    }
}
