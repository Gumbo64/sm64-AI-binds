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
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, frames_for};

/// `ACT_GROUND_POUND` (sm64.h).
const ACT_GROUND_POUND: u32 = 0x0080_08A9;

/// The ground-pound star catch (catalogue G06), on with OBRRT_GP_CATCH=1:
/// a Z hint on the final approach, and the grab's fall to the dance start
/// counted after `Done` ([`Objective::done_frames`], v1's success ranking
/// by the dance start).
///
/// act_ground_pound (mario_actions_airborne.c) zeroes Mario's horizontal
/// speed and lifts him 110 units over 10 frames, so a star a little above
/// or ahead is touched without overshooting it. An airborne grab is then
/// ACT_FALL_AFTER_STAR_GRAB (forward speed 0, gravity 3.2 a frame down to
/// -65, apply_gravity in mario_step.c), and the star dance starts on
/// landing: the frame that matters is that landing, not the touch.
fn gp_catch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_GP_CATCH").is_ok_and(|v| v == "1"))
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
    /// The Grand Star (see `grand_star`).
    grand: bool,
}

/// A star object that flies in with a spawn cutscene (see the module doc).
/// These are the game's spawn mechanics, shared by every course's event stars.
pub fn spawned_star(o: &Obj) -> bool {
    matches!(o.bhv_name(), "bhvSpawnedStar" | "bhvSpawnedStarNoLevelExit" | "bhvStarSpawnCoordinates")
}

/// A spawned star not yet collectable: in its spawn cutscene (before action
/// 3, for all three behaviours).
pub fn spawning(o: &Obj) -> bool {
    (spawned_star(o) && o.action < 3) || (grand_star(o) && o.action < 2)
}

/// The Grand Star (bhvGrandStar, spawned by Bowser 3's death, bowser.inc.c `bowser_spawn_collectable`): 70
/// frames of sparkles (action 0), then a cutscene arc to the arena's centre (0, 0) and a hop (action 1;
/// grand_star.inc.c `arc_to_goal_pos` to the origin, home height + 200), tangible only from action 2.
pub fn grand_star(o: &Obj) -> bool {
    o.bhv_name() == "bhvGrandStar"
}

/// `ACT_JUMBO_STAR_CUTSCENE`: Mario's action once he touches the Grand Star (interaction.c `interact_star_or_key`).
/// Nothing is saved for it in Bowser 3 (save_file.c: `LEVEL_BOWSER_3: break`), so no star count shows the grab.
pub const ACT_JUMBO_STAR_CUTSCENE: u32 = 0x0000_1909;

/// Where a star object is, or comes to rest: a spawned star's home.
pub fn rest_pos(o: &Obj) -> [f32; 3] {
    if grand_star(o) {
        [0.0, o.home[1] + 200.0, 0.0]
    } else if spawned_star(o) {
        o.home
    } else {
        o.pos
    }
}

/// The (course, index) of a star object, from its first behaviour byte;
/// None for Bowser keys and outside courses.
pub fn star_of(b: &St, o: &Obj) -> Option<(i16, u8)> {
    let index = (o.beh_params >> 24) as u8;
    ((1..=24).contains(&b.course) && index <= 6 && !o.bhv_name().contains("Key") && !grand_star(o)).then_some((b.course, index))
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
        Collect { id: o.id, pos: rest_pos(o), zone: b.zone(), star: star_of(b, o), spawning: spawning(o), grand: grand_star(o) }
    }

    fn here(&self, s: &St) -> bool {
        s.find(self.id).is_some()
    }
}

impl Objective for Collect {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
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
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // A star: this star only (base_criteria excludes it already had).
        // A key: any key or star gained.
        let got = match self.star {
            Some((c, i)) => s.has_star(c, i),
            // The Grand Star: its cutscene action (or the ending's level, the grounds), no count changes.
            None if self.grand => s.action == ACT_JUMBO_STAR_CUTSCENE || s.level == 16,
            None => s.progress() > start.progress(),
        };
        if got {
            return Eta::Done;
        }
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        Eta::Frames(frames_for(crate::walk::dist(s, self.pos).max(s.dist(self.pos))))
    }
    fn done_frames(&self, s: &St, _mem: &Mem) -> f64 {
        // G06: the dance starts on landing, so an airborne grab costs its
        // fall (v1's success_heuristic: grab frame + fall_frames).
        if gp_catch() { fall_frames(s) as f64 } else { 0.0 }
    }
    fn success_value(&self) -> f64 {
        50.0
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        // A spawn cutscene stops time for ~100 frames (a boss star's
        // longer) before the star can be had.
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: if self.spawning { 360 } else { 240 } })
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
