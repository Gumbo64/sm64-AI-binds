//! `Bobomb::GateClip`: the speedrun way to BoB's star 6 ("Behind Chain
//! Chomp's Gate") without freeing the Chomp: a bob-omb shoves Mario through
//! the gate's bars.
//!
//! The mechanism (decomp):
//! - A bob-omb is `INTERACT_GRABBABLE`. When Mario touches one he can't grab
//!   (not punching, or facing away), interact_grabbable calls
//!   push_mario_out_of_object(m, o, -5): Mario is moved to
//!   `o.hitboxRadius + 37 - 5` from the bob-omb's centre, away from it, then
//!   wall collision is checked only once at the new spot (radius 60), and
//!   the move is kept if there's a floor there (interaction.c). A big enough
//!   step past the thin bars isn't pushed back.
//! - When a bob-omb explodes (`BOBOMB_ACT_EXPLODE`, after a throw lands or
//!   its fuse runs out) it first grows for 5 frames (scale 1 to 1.8,
//!   bobomb_act_explode), and obj_set_hitbox scales its hitbox radius with
//!   it (65 to 117, object_helpers.c), so each frame the push reaches
//!   further. A held bob-omb whose fuse runs out is dropped with its action
//!   set to explode, but the explosion only runs once it's free, so a
//!   regrab on the right frame delays it (bobomb_held_loop, bobomb.inc.c).
//! - Mario pushed where there is no floor dies out of bounds
//!   (update_mario_geometry_inputs, mario.c: WARP_OP_DEATH). The death warp
//!   waits 48 frames (level_trigger_warp) and lives drop only on landing
//!   after it (act_death_exit), so a star touched during the fade counts.
//!
//! GateClip is an instance of the general `Tech::BombClip`
//! (extra/tech/bombclip.rs): a rough start region (BoB, within 2500 of the
//! standing gate, a bob-omb held or within 1200) and the endpoint (past the
//! bars, or star 6); the general code does the rest: its rule overlay
//! (fetch, air-throw towards the gate, regrab it as it bloats, carry it
//! so the push takes Mario back to the bars, release window there) and its
//! signature-gated progress (0 until a push frame by a bloated bob-omb,
//! then the displacement through the gate's plane), with the gate as the
//! wall: the plane through the gate, its normal pointing away from the
//! star. The success rule and `star_blocked`'s softening (`clip_setup`)
//! are GateClip's own, as before.

use obrrt_core::{Budget, Obj, Objective, PadOverlay, Progress, St, dist};

use crate::extra::tech::bombclip::{self, Wall};

use super::common::hdist;

const GATE: &str = "bhvChainChompGate";
const BOBOMB: &str = "bhvBobomb";
/// BoB, star 6 (index 5).
const COURSE: i16 = 1;
const INDEX: u8 = 5;
/// How far in front of the bars Mario sets up.
const NEAR: f32 = 120.0;
/// Holding a bob-omb within this of the gate is a clip setup. The TAS's
/// push happens on the slope at (1680, 857, 2711), about 2300 from the gate.
const SETUP_RADIUS: f32 = 2500.0;
/// A bob-omb within this of Mario can be fetched.
const FETCH: f32 = 1200.0;
/// ... and at most this far above or below him.
const FETCH_DY: f32 = 150.0;

/// The gate, the star behind it, and the unit vector (x, z) from the gate
/// towards the star: where "behind" is.
pub struct Gate {
    pub pos: [f32; 3],
    pub star: [f32; 3],
    pub n: [f32; 2],
}

impl Gate {
    /// The standing gate and star 6 in `s`, if both are loaded.
    pub fn find(s: &St) -> Option<Gate> {
        if s.course != COURSE {
            return None;
        }
        let gate = s.objects().iter().find(|o| o.bhv_name() == GATE)?;
        let star = s.objects().iter().find(|o| crate::star::star_of(s, o) == Some((COURSE, INDEX)))?;
        let (dx, dz) = (star.pos[0] - gate.pos[0], star.pos[2] - gate.pos[2]);
        let l = (dx * dx + dz * dz).sqrt().max(1.0);
        Some(Gate { pos: gate.pos, star: star.pos, n: [dx / l, dz / l] })
    }

    /// Signed horizontal distance of `p` past the gate, towards the star
    /// (negative on the near side).
    pub fn depth(&self, p: [f32; 3]) -> f32 {
        (p[0] - self.pos[0]) * self.n[0] + (p[2] - self.pos[2]) * self.n[1]
    }

    /// Where Mario sets up: just in front of the bars, on the line from the
    /// star through the gate.
    pub fn spot(&self) -> [f32; 3] {
        self.wall().front(NEAR)
    }

    /// The gate as a bomb-clip wall: its normal points away from the star.
    pub fn wall(&self) -> Wall {
        Wall { p: self.pos, n: [-self.n[0], -self.n[1]] }
    }

    /// Whether `p` is past the bars, in the enclosure: deeper than the bars'
    /// thickness, no further back than the star plus a margin, roughly in
    /// line with the gate and at its height.
    pub fn behind(&self, p: [f32; 3]) -> bool {
        let d = self.depth(p);
        // The cage's static walls (bombclip lab `geom`) make a box from the
        // gate's line (corners (1024, 171), (1888, 720)) 348 deep, so 512
        // either side of the gate: inside is depth 50-340, lateral < 490.
        // (Was lateral < 400 with depth up to the star + 300: a clip into
        // the cage's side, like the TAS's at lateral 441, didn't count.)
        let reach = (hdist(self.pos, self.star) + 300.0).min(340.0);
        let lateral = ((p[0] - self.pos[0]) * self.n[1] - (p[2] - self.pos[2]) * self.n[0]).abs();
        d > 50.0 && d < reach && lateral < 490.0 && (p[1] - self.pos[1]).abs() < 600.0
    }
}

fn is_bobomb(o: &Obj) -> bool {
    o.bhv_name() == BOBOMB
}

fn holding_bobomb(s: &St) -> bool {
    s.held().is_some_and(is_bobomb)
}

/// The bob-omb nearest Mario.
fn nearest_bobomb(s: &St) -> Option<&Obj> {
    s.objects().iter().filter(|o| is_bobomb(o)).min_by(|a, b| s.dist(a.pos).total_cmp(&s.dist(b.pos)))
}

/// Whether `s` is set up for the clip: holding a bob-omb near the standing
/// gate, or already behind its bars. Used by `star_blocked` to keep star 6
/// open to clips while the gate stands.
pub fn clip_setup(s: &St) -> bool {
    let Some(g) = Gate::find(s) else { return false };
    g.behind(s.pos) || (holding_bobomb(s) && s.dist(g.pos) < SETUP_RADIUS)
}

/// OBRRT_DEATHFADE=1: GateClip succeeds on star 6 only (see `done`).
fn death_fade() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_DEATHFADE").is_ok_and(|v| v == "1"))
}

pub struct GateClip {
    zone: (u16, u8),
    wall: Wall,
}

impl GateClip {
    /// Offered in BoB while the gate stands, star 6 isn't had, and a
    /// bob-omb is loaded.
    pub fn new(b: &St) -> Option<GateClip> {
        let g = Gate::find(b)?;
        (!b.has_star(COURSE, INDEX) && b.objects().iter().any(is_bobomb)).then_some(GateClip { zone: b.zone(), wall: g.wall() })
    }

    /// Star 6 (not any star) or Mario behind the gate; base_criteria
    /// excludes states that already meet either.
    ///
    /// With OBRRT_DEATHFADE=1 (catalogue G13), star 6 only. The TAS enters
    /// the cage through an out-of-bounds frame at the gate rock (f22059-60,
    /// a 108-unit shove: no floor, so update_mario_geometry_inputs queues
    /// WARP_OP_DEATH, 48 frames), falls in and grabs the star at
    /// f22104, 3 frames before the warp fires. Being behind the bars is then
    /// a doomed state: the attempt ends there in control (no tail runs), so
    /// the node is kept with the warp pending, and anything started from it
    /// dies unless it grabs a star in the frames left (then the GA's
    /// end-of-attempt rule keeps it: progress went up, the life is a price
    /// paid; lives drop only in act_death_exit). St has no warp timer, and
    /// the GA asks objectives with objects about the first and last states
    /// only, so the OOB frame can't be seen from the state that's behind;
    /// with the switch the objective goes on to the star itself instead.
    /// Tested from TAS f22055/f22062 (2 seeds each): the star comes at the
    /// same frame either way (Star::Collect / Mission::ReachStar finish it
    /// from the doomed nodes), so this stays off.
    fn done(&self, s: &St) -> bool {
        s.has_star(COURSE, INDEX)
            || (!death_fade() && s.zone() == self.zone && Gate::find(s).is_some_and(|g| g.behind(s.pos)))
    }
}

impl Objective for GateClip {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Bobomb::GateClip"
    }
    fn key(&self) -> String {
        format!("Bobomb::GateClip {:?}", self.zone)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // More with a fresh bob-omb at hand near the gate (the clip saves
        // ~500 frames over the post route; the fuse leaves one window).
        // (base_criteria already wants the bob-omb on Mario's level within
        // 1200; fresh = not yet bloated or thrown.)
        let fresh = match b.held().filter(|o| is_bobomb(o)) {
            Some(o) => o.action != 3,
            None => nearest_bobomb(b).is_some_and(|o| o.action == 0 || o.action == 2),
        };
        if self.base_criteria(b) && fresh { 12.0 } else { 4.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        if s.zone() != self.zone || !s.alive() || s.has_star(COURSE, INDEX) {
            return false;
        }
        let Some(g) = Gate::find(s) else { return false };
        if g.behind(s.pos) {
            return false;
        }
        // Holding a bob-omb, or one within 1200 (not holding anything
        // else) on Mario's level: the rule walks to it in a straight line,
        // so a bob-omb up a cliff is never reached (v46: 4 of 279 attempts
        // succeeded; starts below the plateau, 250-760 under the bob-omb,
        // ended with progress 0, no push; the successes started 0-40 off its
        // level, 270-800 away).
        // Not mid-air with empty hands: a start in a jump off the plateau
        // (v46-mw-s1 node 366) lands below it, out of reach.
        holding_bobomb(s)
            || (!s.holding
                && s.action & 0x800 == 0
                && nearest_bobomb(s).is_some_and(|o| {
                    s.hdist(o.pos) < FETCH && (o.pos[1] - s.pos[1]).abs() < FETCH_DY && dist(o.pos, g.pos) < 4000.0 && bombclip::fetchable(o)
                }))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(g) = Gate::find(s) else { return f64::MAX };
        let spot = g.spot();
        if holding_bobomb(s) {
            return s.dist(spot) as f64 / 400.0;
        }
        nearest_bobomb(s).map_or(f64::MAX, |o| 2.0 + (s.dist(o.pos) + dist(o.pos, spot)) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| self.done(s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // The general bomb clip's signature-gated progress through the gate.
        // The cage spans about 500 either side of the gate.
        bombclip::progress(&self.wall, self.zone, 500.0, traj)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 8, horizon: 240 }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        bombclip::steer_target(&self.wall, s)
    }
    fn pad_overlay(&self) -> Option<PadOverlay> {
        Some(bombclip::overlay(self.wall))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(COURSE, INDEX) && Gate::find(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}
