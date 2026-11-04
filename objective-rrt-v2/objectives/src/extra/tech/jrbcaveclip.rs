//! `Jrb::CaveClip` (technique `jrbcaveclip`): swim through the rock above the
//! ocean cave's tunnel mouth, from the bay into the tunnel or the cave,
//! instead of diving to the mouth at y -3,900 and swimming up the tunnel.
//!
//! Source: Super Mario Wiki's glitch list (ref/mariowiki/
//! List_of_Super_Mario_64_glitches.md): "In Jolly Roger Bay, above the
//! entrance to the cave, by a rock. The wall can be clipped through if the
//! player is swimming with full speed and pressing [A] at the correct time."
//!
//! Checked against the decomp (sm64-port):
//! - The geometry (levels/jrb/areas/1/collision.inc.c). The tunnel is a
//!   square tube from its mouth at the bay's bottom (x 4638..5662, floor
//!   y -4904, ceiling -3900 at z -590) up to the cave's pool (water box 1:
//!   x 4433..5969, z -4253..-669, surface -3016). Above the mouth, from
//!   y -3890 up to y 0, the bay's back is two steep faces meeting in a crease
//!   (shared edge (5167, -3890, 198)-(5221, 0, -669)) with normals
//!   (0.25, 0.21, 0.95) and (-0.49, 0.20, 0.85). Their normal y is above
//!   0.01, so they're *floors*, not walls (surface_load.c); the only wall
//!   is above y 0. The cave's ceiling is the plane y -918 over
//!   z -7563..-907. A `bhvRockSolid` sits on the east face at
//!   (5896, -393, -123) (script.c): the wiki's "by a rock".
//! - Swimming moves Mario in one full step a frame, no quarter steps
//!   (perform_water_step, mario_actions_submerged.c), at up to 28 u/frame.
//!   Walls are resolved once at radius 110, and the floor is found under the
//!   new spot, counting only floors at most 78 above it (find_floor,
//!   surface_collision.c). Below a floor he's put on it.
//! - So a stroke that carries Mario d units into one of those faces puts him
//!   about 4.9·d below its surface (normal y 0.2): past 78, i.e. d > ~16,
//!   the face isn't found and nothing pushes him out; with a floor somewhere
//!   below and no ceiling in the way the step is kept, inside the rock. Once
//!   he is past z -669 he's out of the bay's water box: inside box 1's x
//!   range the water level drops to -3016, the step is clamped to 80 below
//!   it, and the tunnel's ceiling under him downwarps him into the tunnel
//!   (the "water ceiling downwarp" in perform_water_full_step); further west
//!   there's no water and he drops through the cave's one-sided ceiling to
//!   its floor. After the first contact frame the swim pitch follows the
//!   slope (HIT_FLOOR raises faceAngle[0] to the floor's pitch), so the clip
//!   has to happen on contact, at full speed: the wiki's "full speed" and
//!   "pressing A at the correct time".
//! - Two 120-star TASes (ref/tas/dumps/full-120star-2012-U.csv frame 72670,
//!   full-120star-2026-U.csv frame 81720) cross this same rock with
//!   hyperspeed along the surface and land in the tunnel at y ~ -3,900,
//!   z -750..-850, which shows the region behind the faces resolves into the
//!   tunnel. The swimming version itself isn't in any reference movie, so the
//!   exact frame timing is unverified; the objective leaves it to the GA.
//! - Reproduced in the emulator (runs/tech-jrb-cave-on, 2026-10-24, its
//!   first-Jrb-CaveClip.m64): swimming down the crease at ~17 u/frame, at
//!   (5215, -62, -651) one stroke crosses z -669 and Mario lands in the
//!   tunnel at y -4003 (frame 21511), then swims on into the cave.
//!
//! Wide mode (technique `jrbcavewide`, ref/tas/techniques CATALOGUE G12, c5
//! §5): the 2026 TAS crosses this rock from just under the surface, metal,
//! in one 443-unit water step (f81719-81720: (4778, 753, -438) ->
//! (4926, -3869, -854)), which clears the faces and the wall above y 0 at
//! once. So with the switch, any submerged action counts as in the bay
//! (metal walking and falling aren't ACT_FLAG_SWIMMING) if it's moving at
//! hs > 26, and a state that fast (hs > [`FAST_HS`]) aims anywhere in
//! y -850..950. Swimming stays in the old band: checked in the emulator
//! (steertest from runs/tech-jrb-cave-on node 0, 2026-10-25), a swim at 17
//! or 28 u/frame into the crease at y 750, 500, 300 or 150 stops at
//! z -556 against the wall (radius 110) and never crosses z -669, and at
//! y 0 / -100 it gets to z -647 at most; only from y -300 (28 u/frame) did
//! it clip (to y -4358 in the tunnel). So the normal-swim clip works only
//! where the faces are floors, below y ~ -150; the surface clip needs a
//! step longer than the wall's push-out, i.e. metal or plunge speed.
//!
//! Success: alive in the cave or the tunnel (z < -907 and y < -918, which
//! in this area is only the cave side), from the bay, without passing the
//! mouth (no state deeper than [`MOUTH_Y`] while still in the bay). That's
//! a new place: no tree so far has a state below y -1,000 in JRB.

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Objective, St, frames_for};

const COURSE: i16 = 3;
const AREA: u8 = 1;
/// The bay's water box ends here (box 0: z -669..7814).
const BAY_Z: f32 = -669.0;
/// The cave's ceiling plane covers z < this, at y [`CAVE_CEIL`].
const CAVE_Z: f32 = -907.0;
const CAVE_CEIL: f32 = -918.0;
/// The mouth's top (the tunnel ceiling meets the bay at y -3900); deeper
/// than this in the bay is the ordinary way in.
const MOUTH_Y: f32 = -3850.0;
/// In front of the crease between the two faces, at the rock's depth.
const ENTRY: [f32; 3] = [5214.0, -500.0, -300.0];
/// Behind the faces, over the tunnel: what the stroke aims through.
const BEHIND_Z: f32 = -1100.0;
/// The face's depth band worth aiming at (well below the wall at y 0, above
/// the cave ceiling's height).
const AIM_Y: [f32; 2] = [-850.0, -150.0];
/// Wide mode: the band for fast states (the TAS's crossing at y 753).
const AIM_Y_WIDE: [f32; 2] = [-850.0, 950.0];
/// Wide mode: a non-swimming submerged action counts at this hspeed.
const SUBMERGED_HS: f32 = 26.0;
/// Wide mode: at this hspeed (above swimming's 28) the wide band applies.
const FAST_HS: f32 = 60.0;
/// Progress counts states at this hspeed or more (full swim speed is 28).
const CREDIT_HS: f32 = 26.0;
const ACT_GROUP_SUBMERGED: u32 = 0x0C0;
const OFFER_RADIUS: f32 = 3000.0;
/// Below full swim speed ([`CREDIT_HS`], which v1's progress required
/// after the start): getting back to it, about one breaststroke cycle
/// (act_breaststroke's stroke and glide).
const SPEEDUP_FRAMES: f64 = 12.0;
/// `mem.flags`: deeper than [`MOUTH_Y`] in the bay seen (the ordinary way
/// in; v1's success failed from that state on).
const MOUTH: u32 = 1;

fn in_bay_water(s: &St) -> bool {
    s.course == COURSE && s.area == AREA && s.action & ACT_FLAG_SWIMMING != 0 && s.pos[2] > BAY_Z
}

fn hs(s: &St) -> f32 {
    (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt()
}

/// Wide mode's bay test: swimming, or any submerged action at hs > 26.
fn in_bay_wide(s: &St) -> bool {
    in_bay_water(s)
        || (s.course == COURSE
            && s.area == AREA
            && s.action & ACT_GROUP_MASK == ACT_GROUP_SUBMERGED
            && hs(s) > SUBMERGED_HS
            && s.pos[2] > BAY_Z)
}

fn in_cave(s: &St) -> bool {
    s.course == COURSE && s.area == AREA && s.pos[2] < CAVE_Z && s.pos[1] < CAVE_CEIL
}

pub struct CaveClip {
    zone: (u16, u8),
    /// `jrbcavewide`.
    wide: bool,
}

impl CaveClip {
    pub fn new(b: &St) -> Option<Arc<dyn Objective>> {
        let x = CaveClip { zone: b.zone(), wide: super::enabled("jrbcavewide") };
        x.base_criteria(b).then(|| Arc::new(x) as Arc<dyn Objective>)
    }
}

impl Objective for CaveClip {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Jrb::CaveClip"
    }
    fn key(&self) -> String {
        format!("Jrb::CaveClip {:?}", self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        3.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        let bay = if self.wide { in_bay_wide(s) } else { in_bay_water(s) };
        s.zone() == self.zone && s.alive() && bay && s.pos[1] > MOUTH_Y && s.dist(ENTRY) < OFFER_RADIUS
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 6, horizon: 200 })
    }
    /// `Lost` once the rollout went through the mouth ([`MOUTH`], latched),
    /// left the zone or died; `Done` in the cave (v1's success). Else the
    /// straight line to the point behind the faces (wide mode: the start's
    /// aim, as v1's progress), plus [`SPEEDUP_FRAMES`] below full swim
    /// speed.
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        if s.pos[2] > BAY_Z && s.pos[1] < MOUTH_Y {
            mem.flags |= MOUTH;
        }
        if mem.flags & MOUTH != 0 {
            return Eta::Lost;
        }
        if in_cave(s) {
            return Eta::Done;
        }
        let aim = if self.wide { self.aim(start) } else { [ENTRY[0], ENTRY[1], BEHIND_Z] };
        let slow = if hs(s) <= CREDIT_HS { SPEEDUP_FRAMES } else { 0.0 };
        Eta::Frames(frames_for(s.dist(aim)) + slow)
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.aim(s))
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
    fn kinship(&self) -> u32 {
        10
    }
}

impl CaveClip {
    /// The point to swim at: behind the crease, at Mario's depth within the
    /// band, so the stroke goes into the face level and at full speed.
    /// Wide mode: states faster than swimming use [`AIM_Y_WIDE`].
    fn aim(&self, s: &St) -> [f32; 3] {
        let band = if self.wide && hs(s) > FAST_HS { AIM_Y_WIDE } else { AIM_Y };
        [ENTRY[0], s.pos[1].clamp(band[0], band[1]), BEHIND_Z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::tech::testkit::st;
    use obrrt_core::fold;

    const ACT_BREASTSTROKE: u32 = 0x3000_24D0;

    fn swim(p: [f32; 3], hs: f32) -> St {
        let mut s = st(LEVEL_JRB, AREA, p, ACT_BREASTSTROKE);
        s.vel = [0.0, 0.0, -hs];
        s
    }

    #[test]
    fn mouth_latch_and_clip() {
        let obj = CaveClip { zone: (LEVEL_JRB as u16, AREA), wide: false };
        let clip = vec![
            swim([5214.0, -300.0, 400.0], 10.0),
            swim([5214.0, -300.0, 0.0], 28.0),
            swim([5215.0, -300.0, -651.0], 28.0),
            swim([5215.0, -4003.0, -950.0], 20.0),
        ];
        let (met, etas) = fold(&obj, &clip);
        assert_eq!(met, Some(3), "{etas:?}");
        assert!(etas[1].frames() < etas[0].frames() && etas[2].frames() < etas[1].frames());
        // Through the mouth first: the cave after it doesn't count.
        let mouth = vec![clip[0].clone(), swim([5150.0, -3900.0, 100.0], 28.0), clip[3].clone()];
        let (met, etas) = fold(&obj, &mouth);
        assert_eq!(met, None);
        assert!(etas[2] == Eta::Lost);
    }
}
