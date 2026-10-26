//! `Star::Collect(id)`: touch a star or a Bowser key (`INTERACT_STAR_OR_KEY`).
//! Grabbing one mostly ends the course; the search's automatic tail presses A
//! through the star dance, text and the warp back.
//!
//! Spawned stars count from the frame they spawn: a star from a ! box, a boss,
//! a race or an NPC (`bhvSpawnedStar`, `bhvSpawnedStarNoLevelExit`,
//! sparkle_spawn_star.inc.c; `bhvStarSpawnCoordinates`, spawn_star.inc.c)
//! first plays a spawn cutscene and only then becomes collectable (the first
//! two set their star hitbox in action 3; the last is intangible until it
//! lands, though its interaction is set at init). The states saved right
//! after the spawn are mid-cutscene, and with pick-by-b only they can start
//! an objective made from them, so the objective must exist there: it aims at
//! where the star comes to rest (its home: `spawn_star` sets it to the
//! target, `set_y_home_to_pos` / `set_home_to_mario` for the others) and
//! succeeds only on that star being collected.

use obrrt_core::buttons::Z_TRIG;
use obrrt_core::layout::ACT_FLAG_AIR;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

/// `ACT_GROUND_POUND` (sm64.h).
const ACT_GROUND_POUND: u32 = 0x0080_08A9;

/// The ground-pound star catch (catalogue G06), on with OBRRT_GP_CATCH=1:
/// a Z hint on the final approach and success ranked by the dance start.
///
/// act_ground_pound (mario_actions_airborne.c) zeroes Mario's horizontal
/// speed and lifts him 110 units over 10 frames, so a star a little above
/// or ahead is touched without overshooting it. An airborne grab is then
/// ACT_FALL_AFTER_STAR_GRAB (forward speed 0, gravity 3.2 a frame down to
/// -65, apply_gravity in mario_step.c), and the star dance starts on
/// landing: the frame that matters is that landing, not the touch.
fn gp_catch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_GP_CATCH").is_ok_and(|v| v == "1"))
}

/// Frames from an airborne star grab at `s` to the landing that starts the
/// dance (ACT_FALL_AFTER_STAR_GRAB's fall onto `floor_y`); 0 on the ground.
fn fall_frames(s: &St) -> usize {
    if s.action & ACT_FLAG_AIR == 0 || s.floor_y <= -10000.0 {
        return 0;
    }
    let (mut y, mut vy, mut t) = (s.pos[1], s.vel[1], 0);
    while y > s.floor_y && t < 150 {
        vy = (vy - 3.2).max(-65.0);
        y += vy;
        t += 1;
    }
    t
}

pub struct Collect {
    id: ObjId,
    pos: [f32; 3],
    zone: (u16, u8),
    /// The star's course and index (behaviour param byte 0), for stars.
    star: Option<(i16, u8)>,
    /// A spawned star still in its spawn cutscene at generation.
    spawning: bool,
}

/// A star object that flies in with a spawn cutscene (see the module doc).
/// These are the game's spawn mechanics, shared by every course's event stars.
pub fn spawned_star(o: &Obj) -> bool {
    matches!(o.bhv_name(), "bhvSpawnedStar" | "bhvSpawnedStarNoLevelExit" | "bhvStarSpawnCoordinates")
}

/// A spawned star not yet collectable: in its spawn cutscene (before action
/// 3, for all three behaviours).
pub fn spawning(o: &Obj) -> bool {
    spawned_star(o) && o.action < 3
}

/// Where a star object is, or comes to rest: a spawned star's home.
pub fn rest_pos(o: &Obj) -> [f32; 3] {
    if spawned_star(o) { o.home } else { o.pos }
}

/// The (course, index) of a star object, from its first behaviour byte;
/// None for Bowser keys and outside courses.
pub fn star_of(b: &St, o: &Obj) -> Option<(i16, u8)> {
    let index = (o.beh_params >> 24) as u8;
    ((1..=24).contains(&b.course) && index <= 6 && !o.bhv_name().contains("Key")).then_some((b.course, index))
}

/// Whether star `index` of `course` is walled off for a plain approach from
/// `s`: BoB's star 6 (index 5) sits behind the Chain Chomp's gate
/// (bhvChainChompGate), which is deleted only once the freed Chomp hits it
/// (bhv_chain_chomp_gate_update, chain_chomp.inc.c). Walking at it fails,
/// decays the star's objectives and spends their starts on states in front
/// of the gate, which are nearer the star than the post the Chomp is freed
/// from (the v32 fix).
///
/// The block is soft, though: a gate or wall can be clipped (speedruns push
/// Mario through these bars with a bob-omb; v19 clipped a wall in WF for a
/// star), and a hard block would remove the incentive to find that. So the
/// star stays open while Mario is set up for the clip, holding a bob-omb
/// near the gate or already behind its bars (`clip::clip_setup`). Only stars
/// that truly can't be had otherwise (100 coins) should be blocked outright,
/// and those have it in their success criterion anyway. With the `gateclip`
/// technique off (`OBRRT_TECH`), the block is hard again (v32).
pub fn star_blocked(s: &St, course: i16, index: u8) -> bool {
    course == 1
        && index == 5
        && s.objects().iter().any(|o| o.bhv_name() == "bhvChainChompGate")
        && !(crate::extra::tech::enabled("gateclip") && crate::extra::enemies::clip::clip_setup(s))
}

impl Collect {
    pub fn new(b: &St, o: &Obj) -> Collect {
        Collect { id: o.id, pos: rest_pos(o), zone: b.zone(), star: star_of(b, o), spawning: spawning(o) }
    }

    fn here(&self, s: &St) -> bool {
        s.find(self.id).is_some()
    }
}

impl Objective for Collect {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Star::Collect"
    }
    fn key(&self) -> String {
        format!("Star {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        40.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // A star already collected is worth nothing.
        s.zone() == self.zone && self.here(s) && self.star.is_none_or(|(c, i)| !s.has_star(c, i) && !star_blocked(s, c, i))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // A star: this star only (base_criteria excludes it already had).
        // A key: any key or star gained.
        match self.star {
            Some((c, i)) => traj.iter().position(|s| s.has_star(c, i)),
            None => {
                let p0 = traj[0].progress();
                traj.iter().position(|s| s.progress() > p0)
            }
        }
    }
    fn success_heuristic(&self, traj: &[St]) -> f64 {
        // The grab's frame since power-on, plus (G06) the fall to the dance start.
        let extra = if gp_catch() { traj.last().map_or(0, fall_frames) } else { 0 };
        traj.last().map_or(f64::MAX, |s| (s.frame as usize + extra) as f64)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.zone() == self.zone)
            .map(|(i, s)| (i, dist(s.pos, self.pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn success_value(&self) -> f64 {
        50.0
    }
    fn budget(&self) -> Budget {
        // A spawn cutscene stops time for ~100 frames (a boss star's
        // longer) before the star can be had.
        Budget::Ga { population: 24, generations: 6, horizon: if self.spawning { 360 } else { 240 } }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn steer_button(&self, s: &St) -> u16 {
        // G06: airborne, with the star within reach horizontally and 0-200
        // above, ground-pound into it.
        if !gp_catch() || s.action & ACT_FLAG_AIR == 0 || s.action == ACT_GROUND_POUND {
            return 0;
        }
        let above = self.pos[1] - s.pos[1];
        let reach = (60.0 + s.forward_vel.abs()).min(250.0);
        if (0.0..=200.0).contains(&above) && s.hdist(self.pos) < reach { Z_TRIG } else { 0 }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.here(s)
    }
    fn kinship(&self) -> u32 {
        10
    }
}
