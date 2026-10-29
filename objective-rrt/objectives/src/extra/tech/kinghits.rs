//! `Boss::GrabRelease` (technique `kinghits`): BoB star 1 ("Big Bob-omb on
//! the Summit"), King Bob-omb hit by grabbing him and letting go of him as
//! soon as possible, instead of a full heavy throw.
//!
//! The mechanics (sm64-port src/game/behaviors/king_bobomb.inc.c,
//! interaction.c, mario_actions_object.c):
//! - Mario can only grab him with a punch (`able_to_grab_object`: a dive
//!   doesn't take objects with `INT_SUBTYPE_GRABS_MARIO`), facing him, and
//!   the King grabs Mario instead if *he* faces Mario
//!   (`check_object_grab_mario`, King act 3, Mario `ACT_GRABBED`): so from
//!   behind. He is holdable only after his first dialog (act 0 sets
//!   `OBJ_FLAG_HOLDABLE` on going to act 2).
//! - Mario holds him from the end of the pick-up animation
//!   (`mario_grab_used_object` in `act_picking_up`; the King is `HELD_HELD`).
//! - Any release counts the same: `bhv_king_bobomb_loop` handles
//!   `HELD_THROWN` and `HELD_DROPPED` alike with
//!   `cur_obj_get_thrown_or_placed(20, 50, 4)`, so a drop (the heavy throw
//!   itself is a `mario_drop_held_object` on its 13th frame, and leaving the
//!   floor or a stomp drop him too) puts him in act 4, "thrown", at 20 u/f
//!   forward and 50 up.
//! - In act 4 he loses 1 `oHealth` when he lands (`OBJ_MOVE_LANDED`), unless
//!   he is over 100 below his home (thrown off the hill: act 5, back home,
//!   no damage). Then act 6 (recoil) or, at 0, act 7 (the star).
//!
//! So a hit is: get behind him, punch-grab, release at once (B starts the
//! heavy throw; the TAS releases within 2 frames, by a means not verified
//! here), and he lands on the summit. Success is his health dropping (or
//! act 6/7, which only follow it). The progress rewards each stage: near
//! behind him, holding him (the fewer frames held the better), him in act
//! 4 still above his home's height - 100. Hints only: run behind him, B
//! when close, B while holding; the GA does the inputs.

use std::sync::Arc;

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::layout::LEVEL_BOB;
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

const COURSE: i16 = 1;
/// "Big Bob-omb on the Summit".
const INDEX: u8 = 0;
const KING: &str = "bhvKingBobomb";
/// King Bob-omb's actions (sKingBobombActions).
const KING_ACT_IDLE: i32 = 1;
const KING_ACT_WALK: i32 = 2;
const KING_ACT_GRABBED_MARIO: i32 = 3;
const KING_ACT_THROWN: i32 = 4;
const KING_ACT_RECOIL: i32 = 6;
const KING_ACT_DEFEATED: i32 = 7;
/// Thrown more than this below his home, he takes no damage.
const OFF_HILL: f32 = -100.0;
const ACT_GRABBED: u32 = 0x0002_0370;
/// Starts within this (straight line) of him.
const START_RADIUS: f32 = 2500.0;
/// The point behind him the approach aims at.
const BEHIND: f32 = 150.0;

pub struct GrabRelease {
    id: ObjId,
    zone: (u16, u8),
}

impl GrabRelease {
    pub fn new(b: &St, o: &Obj) -> Option<GrabRelease> {
        (b.level == LEVEL_BOB as u16 && o.bhv_name() == KING && !b.has_star(COURSE, INDEX))
            .then(|| GrabRelease { id: o.id, zone: b.zone() })
    }

    fn king<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        s.find(self.id)
    }

    fn holding(&self, s: &St) -> bool {
        s.held().is_some_and(|h| h.id == self.id)
    }
}

impl Objective for GrabRelease {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Boss::GrabRelease"
    }
    fn key(&self) -> String {
        format!("BossGrabRelease {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        if s.zone() != self.zone || !s.alive() || !s.controllable() || s.has_star(COURSE, INDEX) || s.action == ACT_GRABBED {
            return false;
        }
        let Some(k) = self.king(s) else { return false };
        // Fightable (talked to, on the summit), or already in Mario's hands.
        k.health > 0
            && (self.holding(s) || matches!(k.action, KING_ACT_IDLE | KING_ACT_WALK | KING_ACT_RECOIL))
            && dist(s.pos, k.pos) < START_RADIUS
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(k) = self.king(s) else { return f64::MAX };
        // Fewer hits left first, then holding him, then near behind him.
        k.health as f64 * 10.0 + if self.holding(s) { 0.0 } else { 5.0 + dist(s.pos, k.behind(BEHIND)) as f64 / 400.0 }
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let hp0 = self.king(&traj[0])?.health;
        // He must be in the state: a frame without him (not read, or out of
        // the object list) is not a hit.
        traj.iter().position(|s| {
            self.king(s).is_some_and(|k| k.health < hp0 || (k.action == KING_ACT_DEFEATED && hp0 > 0))
                || s.has_star(COURSE, INDEX)
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let mut best = Progress { score: f64::MIN, at: 0 };
        let mut held_from: Option<usize> = None;
        let mut held_len = 0usize;
        for (i, s) in traj.iter().enumerate() {
            // Grabbed by the King: the rest of the attempt is lost.
            if s.action == ACT_GRABBED {
                break;
            }
            let Some(k) = self.king(s) else { continue };
            if k.action == KING_ACT_GRABBED_MARIO {
                break;
            }
            let score = if self.holding(s) {
                let from = *held_from.get_or_insert(i);
                held_len = i - from;
                1000.0 - 2.0 * held_len as f64
            } else if held_from.is_some() && k.action == KING_ACT_THROWN {
                if k.pos[1] - k.home[1] < OFF_HILL {
                    // Off the hill: no damage. Worth no more than holding him.
                    0.0
                } else {
                    2000.0 - 2.0 * held_len as f64
                }
            } else {
                -(dist(s.pos, k.behind(BEHIND)) as f64)
            };
            if score > best.score {
                best = Progress { score, at: i };
            }
        }
        best
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let k = self.king(s)?;
        if self.holding(s) || dist(s.pos, k.behind(BEHIND)) < 120.0 {
            Some(k.pos)
        } else {
            Some(k.behind(BEHIND))
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Punch-grab when close behind him; B again once holding him (the
        // heavy throw, which drops him).
        match self.king(s) {
            Some(k) if self.holding(s) || dist(s.pos, k.pos) < 200.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(COURSE, INDEX) && self.king(s).is_some_and(|k| k.health > 0)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// Offered per loaded King Bob-omb while BoB star 1 is missing.
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    if let Some(x) = GrabRelease::new(b, o) {
        out.push(Arc::new(x));
    }
}
