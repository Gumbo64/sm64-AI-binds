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
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

use super::common::{MARIO_CAP_ON_HEAD, best, bp2, capless, find, holding};

const UKIKI_ACT_GO_TO_CAGE: i32 = 4;
const UKIKI_CAGE: u32 = 0;
const UKIKI_CAP: u32 = 1;
/// TTM's monkey cage star: course 12, index 1.
const CAGE_STAR: (i16, u8) = (12, 1);

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
    fn base_heuristic(&self, s: &St) -> f64 {
        if holding(s, self.id, false) {
            return 0.0;
        }
        find(s, self.id, false).map_or(f64::MAX, |o| 5.0 + dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && self.done(s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |s| {
            if holding(s, self.id, false) {
                return Some(1000.0 + if s.in_dialog() { 100.0 } else { 0.0 });
            }
            find(s, self.id, false).map(|o| -(dist(s.pos, o.pos) as f64))
        })
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 240 }
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
