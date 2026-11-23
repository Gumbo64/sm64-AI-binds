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
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

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

/// From close behind him to holding him: the punch and the pick-up
/// animation (act_picking_up, ~10 frames) plus a few frames of lining up.
const GRAB_FRAMES: f64 = 15.0;
/// From holding him to letting go: the heavy throw drops him on its 13th
/// frame (act_heavy_throw; the TAS releases in 2, so this is an upper bound).
const RELEASE_FRAMES: f64 = 13.0;
/// His thrown arc: 50 u/f up at gravity 4 (cur_obj_get_thrown_or_placed(20,
/// 50, 4)), ~2·50/4 = 25 frames to land on the summit.
const FLIGHT_FRAMES: f64 = 25.0;
/// After his landing hit, back close behind him: his recoil (act 6) and
/// walk, Mario round to his back (~1.5 s).
const REGRAB_FRAMES: f64 = 45.0;
/// One more hit after this one: the regrab, the grab, the release and the
/// flight.
const HIT_CYCLE_FRAMES: f64 = REGRAB_FRAMES + GRAB_FRAMES + RELEASE_FRAMES + FLIGHT_FRAMES;

/// `mem.flags`: he has been in Mario's hands this attempt.
const HELD: u32 = 1;
/// `mem.flags`: released (thrown act seen after holding); `mem.f[0]` the
/// frame count since.
const RELEASED: u32 = 2;

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

impl GrabRelease {
    /// This hit's ETA (the phases up to its `Done`).
    fn hit_eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // v1's success, with start = traj[0]: he must be in the state (a
        // frame without him is not a hit) and his health below the start's.
        let Some(hp0) = self.king(start).map(|k| k.health) else { return Eta::Lost };
        if s.has_star(COURSE, INDEX) {
            return Eta::Done;
        }
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let Some(k) = self.king(s) else {
            // Not read this frame: no hit, keep the phase estimate.
            return Eta::Frames(GRAB_FRAMES + RELEASE_FRAMES + FLIGHT_FRAMES);
        };
        if k.health < hp0 || (k.action == KING_ACT_DEFEATED && hp0 > 0) {
            return Eta::Done;
        }
        // Grabbed by the King: the attempt is lost (v1's progress stopped).
        if s.action == ACT_GRABBED || k.action == KING_ACT_GRABBED_MARIO {
            return Eta::Lost;
        }
        let approach = || frames_for(dist(s.pos, k.behind(BEHIND))) + GRAB_FRAMES + RELEASE_FRAMES + FLIGHT_FRAMES;
        if self.holding(s) {
            mem.flags |= HELD;
            mem.flags &= !RELEASED;
            return Eta::Frames(RELEASE_FRAMES + FLIGHT_FRAMES);
        }
        if mem.flags & HELD != 0 && k.action == KING_ACT_THROWN {
            if k.pos[1] - k.home[1] < OFF_HILL {
                // Off the hill: no damage; he walks back home.
                return Eta::Frames(approach());
            }
            if mem.flags & RELEASED == 0 {
                mem.flags |= RELEASED;
                mem.f[0] = 0.0;
            } else {
                mem.f[0] += 1.0;
            }
            return Eta::Frames((FLIGHT_FRAMES - mem.f[0]).max(1.0));
        }
        Eta::Frames(approach())
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
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 240 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // The fight after this hit (v1 ranked starts by hits left, 10
        // heuristic units a hit): from the start's health, which only
        // changes at a hit (Done), so the term is constant in an attempt.
        let rest = self.king(start).map_or(0.0, |k| (k.health - 1).max(0) as f64 * HIT_CYCLE_FRAMES);
        match self.hit_eta(s, start, mem) {
            Eta::Frames(f) => Eta::Frames(f + rest),
            e => e,
        }
    }
    fn success_value(&self) -> f64 {
        10.0
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn king(id: ObjId, pos: [f32; 3], action: i32, health: i32) -> Obj {
        Obj {
            id,
            slot: 7,
            bhv: id.bhv,
            pos,
            home: [0.0, 4000.0, 0.0],
            interact_type: 0,
            interact_status: 0,
            action,
            health,
            value: 0,
            loot_coins: 0,
            beh_params: 0,
            held_state: 0,
            intangible_timer: 0,
            yaw: 0,
            raw_1ac: [0; 4],
        }
    }

    fn st(pos: [f32; 3], k: Obj, holding: bool) -> St {
        St {
            frame: 0,
            pos,
            vel: [0.0; 3],
            forward_vel: 0.0,
            yaw: 0,
            pitch: 0,
            angle_vel: [0; 3],
            water_level: -11000,
            cam_yaw: 0,
            action: 0x0C40_0201, // ACT_IDLE
            mario_flags: 0,
            holding,
            held_slot: holding.then_some(k.slot),
            red_coins: 0,
            level: LEVEL_BOB as u16,
            area: 1,
            act: 1,
            course: 1,
            stars: 0,
            keys: 0,
            coins: 0,
            lives: 4,
            health: 0x880,
            play_mode: 0,
            transition: 0,
            dialog: -1,
            dialog_choice: false,
            menu_line: 0,
            menu_mode: -1,
            act_select: false,
            act_init: 0,
            act_cursor: 0,
            dialog_state: 0,
            dialog_last_page: false,
            menu_timer: 0,
            floor_y: pos[1],
            ceil_y: 20000.0,
            floor_type: 0,
            floor_normal: [0.0, 1.0, 0.0],
            floor_obj: None,
            floor_obj_vy: None,
            floor_obj_y: None,
            action_timer: 0, cap_timer: 0,
            wall_normal: None,
            course_stars: [0; 25],
            save_flags: 0,
            objects: Some(Arc::new(vec![k])),
        }
    }

    #[test]
    fn phases_drop_and_done_on_hit() {
        let id = ObjId { bhv: 1, home: [0, 4000, 0] };
        let obj = GrabRelease { id, zone: (LEVEL_BOB as u16, 1) };
        let kp = [0.0, 4000.0, 0.0];
        let traj = [
            st([0.0, 4000.0, -1000.0], king(id, kp, KING_ACT_WALK, 3), false),
            st([0.0, 4000.0, -160.0], king(id, kp, KING_ACT_WALK, 3), false),
            st([0.0, 4000.0, -100.0], king(id, kp, KING_ACT_WALK, 3), true),
            st([0.0, 4000.0, -100.0], king(id, [0.0, 4050.0, 50.0], KING_ACT_THROWN, 3), false),
            st([0.0, 4000.0, -100.0], king(id, [0.0, 4060.0, 80.0], KING_ACT_THROWN, 3), false),
            st([0.0, 4000.0, -100.0], king(id, [0.0, 4000.0, 200.0], KING_ACT_RECOIL, 2), false),
        ];
        let (met, etas) = obrrt_core::fold(&obj, &traj);
        assert_eq!(met, Some(5));
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        for w in f.windows(2) {
            assert!(w[1] < w[0], "{f:?}");
        }
        // The start is an approach, not Done.
        assert!(!obrrt_core::start_eta(&obj, &traj[0]).is_nan() && obrrt_core::start_eta(&obj, &traj[0]) > 0.0);
    }
}
