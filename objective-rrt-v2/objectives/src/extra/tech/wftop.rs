//! `Wf::TopKick` (technique `wftop`): WF star 2, "To the Top of the
//! Fortress", by jumping and wall-kicking up the tower's side to its top
//! instead of riding the platforms around it.
//!
//! The place (decomp, levels/wf/script.c and areas/1/10/collision.inc.c):
//! - the tower (bhvTower, acts 2-6) stands at (0, 3584, 0) on the fortress
//!   top; its top ring is a floor at y 5376 (local 1792) about 475 from the
//!   axis, the cap with the star (300, 5550, 0) at 5581 (local 1997);
//! - the moving platforms around it (bhvTowerPlatformGroup) are what the
//!   bot's trees ride; their frontier route takes 47.5 s in level.
//!
//! The TAS (ref/tas/dumps/il-wf-top-1030-U, frames 8148-8241, 10.3 s in
//! level): from the fortress top at (1489, 3584, 604) it single-jumps onto a
//! platform at (909, 4085, 215), double-jumps into the tower's wall at
//! (569, 4715, -227), wall-kicks and ledge-grabs the top ring at
//! (378, 5376, -334), then long-jumps onto the cap for the star.
//!
//! So the objective starts on the fortress top near the tower (the TAS
//! spot; the v35 trees have tens to ~100 such states per seed), rewards
//! getting near the ledge (straight line, plus a bonus for a wall kick high
//! up, and a larger one on the top ring, after which nearer the star), and
//! succeeds on the star. (Success on the ring alone, tried first, made
//! ring states at 2 stars that were never used as starts: in every test the
//! nearby star 6 put the frontier at 3 stars first. The TAS takes 31 frames
//! from the ring's ledge grab to the star, so one attempt can do both.) A hints in the kick window; the GA does the rest. No gate softens:
//! star 2 has no `star_blocked` rule, and the tower needs act 2+, which the
//! game decides (the objective is offered only where the star is loaded).

use obrrt_core::buttons::A_BUTTON;
use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Objective, St, frames_for};

const COURSE: i16 = 2;
const INDEX: u8 = 1;
/// The tower's axis (and base height).
const AXIS: [f32; 3] = [0.0, 3584.0, 0.0];
/// The top ring's floor height and reach from the axis.
const TOP_Y: f32 = 5376.0;
const TOP_REACH: f32 = 520.0;
/// The TAS's platform landing, wall kick and ledge grab.
const PLATFORM: [f32; 3] = [909.0, 4085.0, 215.0];
const KICK: [f32; 3] = [569.0, 4715.0, -227.0];
const LEDGE: [f32; 3] = [378.0, 5376.0, -334.0];
/// Starts: on the fortress top (y 3584 and up) this close to the axis.
const FORT_Y: f32 = 3500.0;
const START_REACH: f32 = 1700.0;
const STAR: [f32; 3] = [300.0, 5550.0, 0.0];

/// The wall kick high up by the tower still to do: v1's kick bonus (400
/// units) at PACE.
const KICK_FRAMES: f64 = 13.0;
/// From the ring's ledge grab to the star: the TAS takes 31 frames
/// (ledge climb, long jump onto the cap).
const TOP_TO_STAR_FRAMES: f64 = 31.0;
/// `mem.flags`: a wall kick high up by the tower seen (v1's `kicked`).
const KICKED: u32 = 1;
/// `mem.flags`: on the top ring seen (v1's `topped`).
const TOPPED: u32 = 2;

const ACT_LEDGE_GRAB: u32 = 0x0800_034B;
const ACT_LEDGE_CLIMB_SLOW_1: u32 = 0x0000_054C;
const ACT_LEDGE_CLIMB_FAST: u32 = 0x0000_054F;

fn ledge_action(s: &St) -> bool {
    s.action == ACT_LEDGE_GRAB || (ACT_LEDGE_CLIMB_SLOW_1..=ACT_LEDGE_CLIMB_FAST).contains(&s.action)
}

fn star_loaded(s: &St) -> bool {
    s.objects().iter().any(|o| crate::star::star_of(s, o) == Some((COURSE, INDEX)) && o.bhv_name() == "bhvStar")
}

/// On the top ring (or grabbing its edge), and staying there: the ring is
/// only ~175 wide around the cap, and a landing at speed slides off it (the
/// first test's lone top state did: it was back at y 3885 once he stopped).
fn on_top(s: &St) -> bool {
    s.alive()
        && s.hdist(AXIS) < TOP_REACH + 60.0
        && ((ledge_action(s) && s.pos[1] > TOP_Y - 100.0)
            || (s.action & ACT_FLAG_AIR == 0 && s.pos[1] >= TOP_Y - 10.0 && s.forward_vel.abs() < 10.0))
}

pub struct TopKick {
    zone: (u16, u8),
}

impl TopKick {
    /// Offered in WF's main area while star 2 is missing and loaded, from
    /// the fortress top near the tower.
    pub fn new(b: &St) -> Option<TopKick> {
        (b.level as u32 == LEVEL_WF
            && b.area == 1
            && b.alive()
            && !b.has_star(COURSE, INDEX)
            && b.pos[1] >= FORT_Y
            && b.pos[1] < TOP_Y - 50.0
            && b.hdist(AXIS) < START_REACH
            && star_loaded(b))
        .then_some(TopKick { zone: b.zone() })
    }
}

impl Objective for TopKick {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Wf::TopKick"
    }
    fn key(&self) -> String {
        format!("Wf::TopKick {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        6.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.alive()
            && !s.has_star(COURSE, INDEX)
            && s.pos[1] >= FORT_Y
            && s.pos[1] < TOP_Y - 50.0
            && s.hdist(AXIS) < START_REACH
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 240 })
    }
    /// `Done` on this star (v1's success: star 6 nearby isn't this one).
    /// The zone left or dead: `Lost` (v1's progress stopped). Latched in
    /// `mem` as v1's: [`KICKED`] (a wall kick above 4400 within 900 of the
    /// axis) and [`TOPPED`] (`on_top` seen). Topped: the way to the star;
    /// else the way to the ledge, [`KICK_FRAMES`] until kicked, and
    /// [`TOP_TO_STAR_FRAMES`].
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        if s.has_star(COURSE, INDEX) {
            return Eta::Done;
        }
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if s.action == ACT_WALL_KICK_AIR && s.pos[1] > 4400.0 && s.hdist(AXIS) < 900.0 {
            mem.flags |= KICKED;
        }
        if on_top(s) {
            mem.flags |= TOPPED;
        }
        if mem.flags & TOPPED != 0 {
            return Eta::Frames(frames_for(s.dist(STAR)));
        }
        let kick = if mem.flags & KICKED != 0 { 0.0 } else { KICK_FRAMES };
        Eta::Frames(frames_for(s.dist(LEDGE)) + kick + TOP_TO_STAR_FRAMES)
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Low: the TAS's platform; then the kick spot on the wall; above it,
        // the ledge.
        Some(if s.pos[1] < PLATFORM[1] - 100.0 {
            PLATFORM
        } else if s.pos[1] < KICK[1] - 50.0 {
            KICK
        } else if s.pos[1] < TOP_Y - 20.0 {
            LEDGE
        } else {
            STAR
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        // A on alternate frames in the air by the tower's wall below the
        // ring, so a kick is tried when he hits it.
        if s.action & ACT_FLAG_AIR != 0 && s.hdist(AXIS) < 900.0 && s.pos[1] > 4300.0 && s.pos[1] < TOP_Y {
            A_BUTTON
        } else {
            0
        }
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
    use obrrt_core::{fold, start_eta};

    const ACT_DOUBLE_JUMP: u32 = 0x0300_0881;

    fn at(pos: [f32; 3], action: u32) -> St {
        st(LEVEL_WF, 1, pos, action)
    }

    #[test]
    fn kick_and_top_latch_and_eta_drops() {
        let obj = TopKick { zone: (LEVEL_WF as u16, 1) };
        let mut star = at(STAR, ACT_WALKING);
        star.course_stars[(COURSE - 1) as usize] = 1 << INDEX;
        let traj = vec![
            at([1489.0, 3584.0, 604.0], ACT_WALKING),
            at(PLATFORM, ACT_WALKING),
            at([569.0, 4600.0, -227.0], ACT_DOUBLE_JUMP),
            // The kick: latched, so the ETA drops by KICK_FRAMES.
            at(KICK, ACT_WALL_KICK_AIR),
            at([450.0, 5100.0, -300.0], ACT_DOUBLE_JUMP),
            at(LEDGE, ACT_LEDGE_GRAB),
            at([330.0, 5450.0, -100.0], ACT_DOUBLE_JUMP),
            star,
        ];
        let (met, etas) = fold(&obj, &traj);
        assert_eq!(met, Some(7), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
        // The kick is latched: back at the double-jump state after it, the
        // ETA is lower than before it.
        let mut mem = Mem::default();
        let s0 = &traj[0];
        let before = obj.eta(&traj[2], s0, &mut mem).frames();
        obj.eta(&traj[3], s0, &mut mem);
        let after = obj.eta(&traj[2], s0, &mut mem).frames();
        assert!((before - after - KICK_FRAMES).abs() < 1e-9);
        // Leaving the zone is Lost; the start is an approach.
        assert!(obj.eta(&st(LEVEL_WF, 2, LEDGE, ACT_WALKING), s0, &mut Mem::default()) == Eta::Lost);
        assert!(start_eta(&obj, s0) > TOP_TO_STAR_FRAMES);
    }
}
