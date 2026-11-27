//! The monkeys of TTM (bhvUkiki / bhvMacroUkiki; ukiki.inc.c):
//!
//! - `Ukiki::ToCage`: the cage Ukiki (2nd param byte `UKIKI_CAGE` = 0,
//!   act 2) is caught (grabbed), asks to be let go (DIALOG_079; "yes" is the
//!   default answer), then runs to the cage (`UKIKI_ACT_GO_TO_CAGE` = 4),
//!   opens it and frees "Mystery of the Monkey Cage" (TTM star 2;
//!   cage_ukiki_held_loop, ukiki_cage.inc.c).
//! - `Ukiki::RetrieveCap`: the cap Ukiki (`UKIKI_CAP` = 1) steals the cap
//!   when Mario picks him up with it on; catching him again gives it back
//!   (cap_ukiki_held_loop: DIALOG_101, mario_retrieve_cap).

use obrrt_core::buttons::{A_BUTTON, B_BUTTON};
use obrrt_core::layout::LEVEL_TTM;
use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::{MARIO_CAP_ON_HEAD, bp2, capless, find, holding};

const UKIKI_ACT_GO_TO_CAGE: i32 = 4;
const UKIKI_CAGE: u32 = 0;
const UKIKI_CAP: u32 = 1;
/// TTM's monkey cage star: course 12, index 1.
const CAGE_STAR: (i16, u8) = (12, 1);

/// Close to him to holding him: lining up and the B grab (~1/3 s; he runs).
const GRAB_FRAMES: f64 = 10.0;
/// Holding him to his dialog opening (DIALOG_079 / DIALOG_101: Mario must
/// stand still first, ~1 s).
const DIALOG_WAIT_FRAMES: f64 = 30.0;
/// The dialog open to its effect: paging through with A ("yes" is the
/// default), then he heads for the cage or the cap comes back (~1 s).
const DIALOG_FRAMES: f64 = 30.0;

#[derive(Clone, Copy, PartialEq)]
enum Goal {
    ToCage,
    RetrieveCap,
}

pub struct Ukiki {
    id: ObjId,
    zone: (u16, u8),
    goal: Goal,
}

impl Ukiki {
    pub fn new(b: &St, o: &Obj) -> Option<Ukiki> {
        if b.level as u32 != LEVEL_TTM {
            return None;
        }
        let goal = match bp2(o) {
            UKIKI_CAGE if !b.has_star(CAGE_STAR.0, CAGE_STAR.1) && o.action != UKIKI_ACT_GO_TO_CAGE => Goal::ToCage,
            UKIKI_CAP if capless(b) => Goal::RetrieveCap,
            _ => return None,
        };
        Some(Ukiki { id: o.id, zone: b.zone(), goal })
    }

    fn done(&self, s: &St) -> bool {
        match self.goal {
            Goal::ToCage => find(s, self.id, false).is_none_or(|o| o.action == UKIKI_ACT_GO_TO_CAGE),
            Goal::RetrieveCap => s.mario_flags & MARIO_CAP_ON_HEAD != 0,
        }
    }
}

impl Objective for Ukiki {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        match self.goal {
            Goal::ToCage => "Ukiki::ToCage",
            Goal::RetrieveCap => "Ukiki::RetrieveCap",
        }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        match self.goal {
            Goal::ToCage => 20.0,
            Goal::RetrieveCap => 4.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && find(s, self.id, false).is_some()
            && !self.done(s)
            && (self.goal != Goal::ToCage || !s.has_star(CAGE_STAR.0, CAGE_STAR.1))
            && (!s.holding || holding(s, self.id, false))
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // ToCage counts him gone as done; a state whose objects weren't read
        // can't tell.
        if (s.objects.is_some() || self.goal == Goal::RetrieveCap) && self.done(s) {
            return Eta::Done;
        }
        if holding(s, self.id, false) {
            return Eta::Frames(if s.in_dialog() { DIALOG_FRAMES } else { DIALOG_WAIT_FRAMES + DIALOG_FRAMES });
        }
        let Some(o) = find(s, self.id, false) else { return Eta::Lost };
        Eta::Frames(frames_for(dist(s.pos, o.pos)) + GRAB_FRAMES + DIALOG_WAIT_FRAMES + DIALOG_FRAMES)
    }
    fn success_value(&self) -> f64 {
        match self.goal {
            Goal::ToCage => 10.0,
            Goal::RetrieveCap => 2.0,
        }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Holding him: stand still so the dialog can start.
        if holding(s, self.id, false) {
            return Some(s.pos);
        }
        find(s, self.id, false).map(|o| o.pos)
    }
    fn steer_button(&self, s: &St) -> u16 {
        if s.in_dialog() {
            // Advance the text; A on the default "yes" lets him go to the cage.
            return A_BUTTON;
        }
        if holding(s, self.id, false) {
            return 0;
        }
        match find(s, self.id, false) {
            Some(o) if dist(s.pos, o.pos) < 180.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && find(s, self.id, false).is_some() && !self.done(s)
    }
}
