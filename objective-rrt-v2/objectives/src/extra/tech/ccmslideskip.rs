//! `Ccm::SlideSkip` (technique `ccmslideskip`): CCM star 1, "Slip Slidin'
//! Away", by leaving the slide's tube at the top with a wall kick and
//! falling straight down to the exit room instead of sliding.
//!
//! The place (decomp, levels/ccm/script.c, object_list_processor.c,
//! behaviors/spawn_star_exit.inc.c):
//! - the slide is CCM area 2; Mario arrives at (-5836, 6656, -6143)
//!   (bhvAirborneWarp at y 7465) and leaves by the exit room at y -5836,
//!   whose warp (node 0x14) puts him back in area 1 by the cabin at
//!   (2665, -4607, 4524);
//! - loading area 2 sets gCCMEnteredSlide, and then bhvCcmTouchedStarSpawn
//!   in area 1 spawns the star at (2500, -4350, 5750), which `Star::Collect`
//!   picks up. Nothing gates it (no `star_blocked` rule), so nothing softens.
//!
//! The TAS (ref/tas/dumps/il-ccm-slide-2288-U, frames 2952-3310, 22.9 s in
//! level vs the bot's 65-68 s): from the start it walks east, jumps up to a
//! triple jump into the tube's wall at (-4659, 6614, -6518), wall-kicks back
//! over the tube's rim, falls about 12,500 units in 6.6 s at x -5477..-6850
//! and lands at (-6848, -5836, -6817), runs west into the exit room
//! (-7881, -5836, -7000) and warps out.
//!
//! So the objective starts at the slide's top (the TAS spot; v34-1b-s2 has
//! ~800 states there, v32-chomp-s3-1b ~120), rewards getting near the
//! exit door (straight line, plus a bonus for a wall kick at the top), and
//! succeeds in the exit room, back in area 1 or on the star. Sliding
//! down the normal way takes far longer than the horizon. A hints in the
//! kick window; the GA does the rest.

use obrrt_core::buttons::A_BUTTON;
use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Objective, St, frames_for};

const COURSE: i16 = 4;
const INDEX: u8 = 0;
const AREA: u8 = 2;
/// Where Mario lands at the slide's top.
const START: [f32; 3] = [-5836.0, 6656.0, -6143.0];
/// Starts: above this height, this close to the start (straight line).
const START_Y: f32 = 6300.0;
const START_REACH: f32 = 2000.0;
/// The TAS's wall kick out of the tube.
const KICK: [f32; 3] = [-4659.0, 6614.0, -6518.0];
/// Where the TAS lands after the fall.
const LANDING: [f32; 3] = [-6848.0, -5836.0, -6817.0];
/// The exit room's door (special_wooden_door_warp, areas/2/collision.inc.c);
/// the room's floor is y -5836 over x -8191..-6706, z -7372..-6143. A fall
/// straight down from the TAS kick lands on the slide's last stretch at
/// x ~-5300, about 2300 from the door; counting that as success (v2 of
/// this file, tech-ccm-v2-*) gave many successes but no star in 2 seeds,
/// since nothing then walks those states into the room. So success is the
/// room itself (the TAS lands 800 from the door), and the stretch before it
/// is progress.
const EXIT_DOOR: [f32; 3] = [-7628.0, -5836.0, -6988.0];
const EXIT_REACH: f32 = 1100.0;
const BOTTOM_Y: f32 = -5650.0;

/// Falling speed down the outside of the tube: the TAS falls about 12,500
/// units in 6.6 s (198 frames).
const FALL_PACE: f32 = 63.0;
/// The wall kick out of the tube still to do: v1's kick bonus (500 units)
/// at PACE.
const KICK_FRAMES: f64 = 17.0;
/// `mem.flags`: a wall kick above [`START_Y`] seen (v1's `kicked`).
const KICKED: u32 = 1;

/// Frames from `p` (out of the tube) to the exit room: the fall to its
/// floor at [`FALL_PACE`], then the walk to within [`EXIT_REACH`] of the
/// door.
fn fall_eta(p: [f32; 3]) -> f64 {
    let fall = ((p[1] - BOTTOM_Y).max(0.0) / FALL_PACE) as f64;
    let hd = ((p[0] - EXIT_DOOR[0]).powi(2) + (p[2] - EXIT_DOOR[2]).powi(2)).sqrt();
    fall + frames_for((hd - EXIT_REACH).max(0.0))
}

fn in_slide(s: &St) -> bool {
    s.level as u32 == LEVEL_CCM && s.area == AREA
}

fn in_exit_room(s: &St) -> bool {
    in_slide(s) && s.alive() && s.pos[1] < BOTTOM_Y && s.hdist(EXIT_DOOR) < EXIT_REACH
}

pub struct SlideSkip;

impl SlideSkip {
    /// Offered at the slide's top while star 1 is missing.
    pub fn new(b: &St) -> Option<SlideSkip> {
        SlideSkip.base_criteria(b).then_some(SlideSkip)
    }
}

impl Objective for SlideSkip {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Ccm::SlideSkip"
    }
    fn key(&self) -> String {
        "Ccm::SlideSkip".into()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        6.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        in_slide(s) && s.alive() && !s.has_star(COURSE, INDEX) && s.pos[1] > START_Y && s.dist(START) < START_REACH
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        // The kick takes about 80 frames from the start, the fall 200 more.
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 330 })
    }
    /// `Done` on v1's success: this star, the exit room, or area 1 alive
    /// (reached from area 2 only through the exit door). Out of the slide
    /// or dead: `Lost` (v1's progress stopped). Not yet kicked
    /// ([`KICKED`], latched in `mem`): the way to the TAS's kick spot,
    /// [`KICK_FRAMES`], and the fall from there; kicked: the fall from here
    /// ([`fall_eta`]).
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        if s.has_star(COURSE, INDEX) || in_exit_room(s) || (s.level as u32 == LEVEL_CCM && s.area == 1 && s.alive()) {
            return Eta::Done;
        }
        if !in_slide(s) || !s.alive() {
            return Eta::Lost;
        }
        if s.action == ACT_WALL_KICK_AIR && s.pos[1] > START_Y {
            mem.flags |= KICKED;
        }
        if mem.flags & KICKED != 0 {
            return Eta::Frames(fall_eta(s.pos));
        }
        Eta::Frames(frames_for(s.dist(KICK)) + KICK_FRAMES + fall_eta(KICK))
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // At the top: the kick spot on the tube's wall; once falling below
        // the rim, the TAS's landing spot, then the exit door.
        Some(if s.pos[1] > START_Y {
            KICK
        } else if s.pos[1] > LANDING[1] + 200.0 {
            LANDING
        } else {
            EXIT_DOOR
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        // A on alternate frames in the air near the kick spot.
        if s.action & ACT_FLAG_AIR != 0 && s.pos[1] > START_Y && s.hdist(KICK) < 600.0 {
            A_BUTTON
        } else {
            0
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_CCM && !s.has_star(COURSE, INDEX)
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

    const ACT_TRIPLE_JUMP: u32 = 0x0100_0882;
    const ACT_FREEFALL: u32 = 0x0100_088C;

    #[test]
    fn kick_latches_and_fall_counts_down() {
        let at = |p, a| st(LEVEL_CCM, AREA, p, a);
        let traj = vec![
            at(START, ACT_WALKING),
            at([-5000.0, 6650.0, -6400.0], ACT_TRIPLE_JUMP),
            at(KICK, ACT_WALL_KICK_AIR),
            at([-5500.0, 3000.0, -6600.0], ACT_FREEFALL),
            at([-6000.0, -2000.0, -6700.0], ACT_FREEFALL),
            at([-6600.0, -5000.0, -6800.0], ACT_FREEFALL),
            // The TAS's landing, 800 from the door: in the exit room.
            at(LANDING, ACT_WALKING),
        ];
        let (met, etas) = fold(&SlideSkip, &traj);
        assert_eq!(met, Some(6), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
        // Falling without the kick (sliding down the tube) keeps the kick
        // and the whole fall to go.
        let mut mem = Mem::default();
        let slid = SlideSkip.eta(&traj[4], &traj[0], &mut mem).frames();
        assert!(slid > f[4] + KICK_FRAMES);
        // Out of the slide without success: Lost.
        assert!(SlideSkip.eta(&st(LEVEL_CCM, 3, START, ACT_WALKING), &traj[0], &mut Mem::default()) == Eta::Lost);
    }
}
