//! `Boss::Hit(id)`: lower a boss's health (`oHealth`), and once it is at 0,
//! finish the fight up to the star. For King Bob-omb that means grabbing him
//! from behind and throwing him; starts where Mario already holds the boss
//! are preferred, so `Object::Grab` and this chain.
//!
//! Stages (king_bobomb.inc.c):
//! - alive (hp > 0): success is his hp below the start's while he is in the
//!   state (a frame without him, objects not read or not in the list, is not
//!   a hit), or his death act, or the star.
//! - beaten (act 7, hp 0): he waits for Mario to be ready to speak
//!   (`cur_obj_update_dialog_with_cutscene`, DIALOG_116); success is that
//!   dialog opening (a modal state; the dialog objectives take it on), or
//!   act 8, or the star.
//! - done (act 8: hidden, the star spawned, `cur_obj_spawn_star_at_y_offset`
//!   flying to its home): success is the star collected.
//!
//! Keeping the post-defeat stages in this objective matters for the search:
//! with pick-by-b an objective is picked only if b can start it, and a b with
//! the King alive (any summit state) then picks its start by heuristic among
//! its kin, which ranks the beaten and done states first. As separate
//! objectives they could start only when the one beaten state itself was b,
//! one state among many in its 512-unit cell (tech-king-summit-on: never).

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, dist, frames_for};

/// King Bob-omb's act while flying from Mario's throw (king_bobomb_act_4):
/// he loses health on landing.
const THROWN_ACT: i32 = 4;
/// Close behind him to holding him: circling to his back and the B grab
/// (~2/3 s).
const GRAB_FRAMES: f64 = 20.0;
/// Holding him to letting go: turning towards the summit and the throw
/// animation (~1/3 s).
const THROW_FRAMES: f64 = 10.0;
/// Thrown to his landing hit (act 4's flight, ~2/3 s).
const LAND_FRAMES: f64 = 20.0;
/// Beaten to his dialog: he waits for Mario to stand still on the ground
/// (~1 s).
const DIALOG_FRAMES: f64 = 30.0;
/// The done stage without the star in the list yet: its spawn (~1 s).
const STAR_SPAWN_FRAMES: f64 = 30.0;
/// After his landing hit, back to close behind him: he recoils and walks
/// again (acts 6, 2) while Mario runs round to his back (~1.5 s).
const REGRAB_FRAMES: f64 = 45.0;
/// One more hit after this one: the regrab, the grab, the throw and the
/// landing (v1 ranked starts by hits left, 10 heuristic units a hit; the
/// ETA counts them as the rest of the fight after this hit's `Done`).
const HIT_CYCLE_FRAMES: f64 = REGRAB_FRAMES + GRAB_FRAMES + THROW_FRAMES + LAND_FRAMES;

/// `OBRRT_KING_LANDING=1` (off by default): a thrown King predicted to land
/// off the hill (core's `king::lands_on_stage`, from his position this
/// frame and last, kept in `mem`), or already off it, costs his way back and
/// another throw: no better than not throwing, so the GA aims at the summit.
pub fn king_landing() -> bool {
    obrrt_core::config::current().objectives.king_landing
}
/// Off the hill to home: his landing animation (act 4's sub-action 1), the
/// jump back (act 5: oVelY 100 up to home height, then a 100-up / -4 arc,
/// ~50 frames) and the landing animation (~3.5 s in all, king_bobomb.inc.c).
const KING_RETURN_FRAMES: f64 = 105.0;
/// The dialog he gives when Mario comes back within 500 (DIALOG_128),
/// pressed through (~3 s).
const KING_DIALOG_FRAMES: f64 = 90.0;
/// mem.flags bit: mem.f[0..3] hold the King's position last frame.
const KING_PREV: u32 = 1;

/// The boss's (beaten, done) actions: beaten = hp 0 and waiting to give the
/// star, done = the star spawned. King Bob-omb: acts 7 and 8
/// (`sKingBobombActions`).
fn death_acts(bhv: &str) -> Option<(i32, i32)> {
    match bhv {
        "bhvKingBobomb" => Some((7, 8)),
        _ => None,
    }
}

/// A spawned star flies to its home (spawn_star, spawn_star.inc.c).
const STAR_BHV: &str = "bhvStarSpawnCoordinates";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Alive(i32),
    Beaten,
    Done,
}

pub struct Hit {
    id: ObjId,
    zone: (u16, u8),
    /// The star beating it gives, if known.
    star: Option<(i16, u8)>,
    /// Its (beaten, done) actions, if known.
    death: Option<(i32, i32)>,
}

/// The (course, index) of the star a boss gives.
pub fn boss_star(bhv: &str) -> Option<(i16, u8)> {
    match bhv {
        "bhvKingBobomb" => Some((1, 0)),
        "bhvWhompKingBoss" => Some((2, 0)),
        "bhvBigBully" => Some((7, 0)),
        "bhvBigBullyWithMinions" => Some((7, 1)),
        // levels/sl/script.c places it with behParam 0x01000000: star 2.
        "bhvBigChillBully" => Some((10, 1)),
        _ => None,
    }
}

impl Hit {
    pub fn new(b: &St, o: &Obj) -> Hit {
        Hit { id: o.id, zone: b.zone(), star: boss_star(o.bhv_name()), death: death_acts(o.bhv_name()) }
    }

    /// The fight's stage in `s`; None without the boss in the state, or
    /// with it at hp 0 in no known death act.
    fn stage(&self, s: &St) -> Option<Stage> {
        let o = s.find(self.id)?;
        match self.death {
            Some((beaten, _)) if o.action == beaten => Some(Stage::Beaten),
            Some((_, done)) if o.action == done => Some(Stage::Done),
            _ if o.health > 0 => Some(Stage::Alive(o.health)),
            _ => None,
        }
    }

    fn got_star(&self, s: &St) -> bool {
        self.star.is_some_and(|(c, i)| s.has_star(c, i))
    }

    /// Where the boss's spawned star is headed (its home), if loaded.
    fn star_target(&self, s: &St) -> Option<[f32; 3]> {
        let (_, index) = self.star?;
        s.objects().iter().find(|o| o.bhv_name() == STAR_BHV && (o.beh_params >> 24) as u8 == index).map(|o| o.home)
    }

    fn holding(&self, s: &St) -> bool {
        s.held().is_some_and(|h| h.id == self.id)
    }
}

impl Objective for Hit {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Boss::Hit"
    }
    fn key(&self) -> String {
        format!("BossHit {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // No need to beat a boss whose star is already collected.
        s.zone() == self.zone
            && s.alive()
            && !self.got_star(s)
            && match self.stage(s) {
                Some(Stage::Alive(_) | Stage::Beaten) => true,
                // Done counts only with a star to collect.
                Some(Stage::Done) => self.star_target(s).is_some(),
                None => false,
            }
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 200 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // This boss's star only (got_star), not any star or key; once had it
        // counts either way.
        if self.got_star(s) {
            return Eta::Done;
        }
        // The boss must be in the start and in the state: a frame without it
        // (not in the list) is not a hit. Its death never removes it (the
        // King hides in act 8).
        let Some(stage0) = self.stage(start) else { return Eta::Lost };
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let Some(stage) = self.stage(s) else { return Eta::Lost };
        let met = match (stage0, stage) {
            (Stage::Alive(hp0), Stage::Alive(hp)) => hp < hp0,
            (Stage::Alive(_), Stage::Beaten | Stage::Done) => true,
            (Stage::Beaten, Stage::Beaten) => s.in_dialog(),
            (Stage::Beaten, Stage::Done) => true,
            _ => false,
        };
        if met {
            return Eta::Done;
        }
        // The hits after this one, from the start's health (it only changes
        // at Done, so this term is constant within an attempt).
        let rest = match stage0 {
            Stage::Alive(hp0) => (hp0 - 1).max(0) as f64 * HIT_CYCLE_FRAMES,
            _ => 0.0,
        };
        Eta::Frames(rest + match stage {
            // Near the star first (it may still be spawning).
            Stage::Done => self.star_target(s).map_or(STAR_SPAWN_FRAMES, |t| frames_for(dist(s.pos, t))),
            Stage::Beaten => DIALOG_FRAMES,
            Stage::Alive(_) => {
                let Some(o) = s.find(self.id) else { return Eta::Lost };
                // OBRRT_KING_LANDING: an off-hill throw costs the way back.
                if king_landing() {
                    let prev = (mem.flags & KING_PREV != 0).then(|| [mem.f[0] as f32, mem.f[1] as f32, mem.f[2] as f32]);
                    mem.f[0] = o.pos[0] as f64;
                    mem.f[1] = o.pos[1] as f64;
                    mem.f[2] = o.pos[2] as f64;
                    mem.flags |= KING_PREV;
                    let off = obrrt_core::king::is_off(o.action, o.pos[1], o.home[1])
                        || (o.action == THROWN_ACT && prev.is_some_and(|p| !obrrt_core::king::lands_on_stage(p, o.pos, o.home[1])));
                    if off {
                        let back = if o.action == 5 { KING_RETURN_FRAMES / 2.0 } else { KING_RETURN_FRAMES };
                        return Eta::Frames(rest + back + KING_DIALOG_FRAMES + frames_for(dist(s.pos, o.home)) + GRAB_FRAMES + THROW_FRAMES + LAND_FRAMES);
                    }
                }
                if o.action == THROWN_ACT {
                    LAND_FRAMES
                } else if self.holding(s) {
                    THROW_FRAMES + LAND_FRAMES
                } else {
                    frames_for(dist(s.pos, o.pos)) + GRAB_FRAMES + THROW_FRAMES + LAND_FRAMES
                }
            }
        })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        match self.stage(s)? {
            Stage::Done => return self.star_target(s),
            // Let him talk: no steering (Mario must be on the ground).
            Stage::Beaten => return None,
            Stage::Alive(_) => {}
        }
        let o = s.find(self.id)?;
        // OBRRT_KING_LANDING: holding him, face the summit's middle.
        if king_landing() && self.holding(s) {
            return Some(obrrt_core::king::ARENA_CENTRE);
        }
        if self.holding(s) || dist(s.pos, o.behind(150.0)) < 120.0 {
            Some(o.pos)
        } else {
            Some(o.behind(150.0))
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Grab when close behind him; throw (B again) once holding him.
        match s.find(self.id) {
            Some(o) if o.health > 0 && (self.holding(s) || dist(s.pos, o.pos) < 180.0) => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grab::Grab;
    use obrrt_core::layout::LEVEL_BOB;
    use obrrt_core::{bhv_addr, fold};
    use std::sync::Arc;

    const SUMMIT: [f32; 3] = [1900.0, 4300.0, -5900.0];

    fn king(action: i32, health: i32, pos: [f32; 3]) -> Obj {
        let bhv = bhv_addr("bhvKingBobomb").expect("bhvKingBobomb in the layout");
        Obj {
            id: ObjId { bhv, home: [1636, 4242, -5567] },
            slot: 7,
            bhv,
            pos,
            home: [1636.0, 4242.0, -5567.0],
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
            action: 0x0C400201,
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

    /// Far, close, holding, thrown, landed with one less health: the ETA
    /// falls at every phase change and the hit is `Done`.
    #[test]
    fn king_hit_phases() {
        let far = [SUMMIT[0] + 900.0, SUMMIT[1], SUMMIT[2]];
        let near = [SUMMIT[0] + 100.0, SUMMIT[1], SUMMIT[2]];
        let traj = vec![
            st(far, king(2, 3, SUMMIT), false),
            st(near, king(2, 3, SUMMIT), false),
            st(near, king(2, 3, SUMMIT), true),
            st(near, king(THROWN_ACT, 3, SUMMIT), false),
            st(near, king(6, 2, SUMMIT), false),
        ];
        let hit = Hit::new(&traj[0], &traj[0].objects()[0]);
        let (met, etas) = fold(&hit, &traj);
        assert_eq!(met, Some(4));
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA must fall: {f:?}");
        assert!(f[3].is_finite() && f[3] > 0.0);
    }

    /// Fewer hits left is closer (v1's start ranking): the rest of the
    /// fight is in the ETA, and beaten is closer still.
    #[test]
    fn king_fewer_hits_left_is_closer() {
        let p = [SUMMIT[0] + 500.0, SUMMIT[1], SUMMIT[2]];
        let at = |hp| st(p, king(2, hp, SUMMIT), false);
        let hit = Hit::new(&at(3), &at(3).objects()[0]);
        let e: Vec<f64> = (1..=3).map(|hp| obrrt_core::start_eta(&hit, &at(hp))).collect();
        assert!(e[0] < e[1] && e[1] < e[2], "{e:?}");
        assert_eq!(e[2] - e[1], HIT_CYCLE_FRAMES);
        let beaten = st(p, king(7, 0, SUMMIT), false);
        assert!(obrrt_core::start_eta(&hit, &beaten) < e[0]);
    }

    /// Beaten: Done on his dialog; the done stage only on the star.
    #[test]
    fn king_beaten_and_star() {
        let p = [SUMMIT[0] + 200.0, SUMMIT[1], SUMMIT[2]];
        let beaten = st(p, king(7, 0, SUMMIT), false);
        let hit = Hit::new(&beaten, &beaten.objects()[0]);
        let mut talking = beaten.clone();
        talking.dialog = 116;
        assert_eq!(fold(&hit, &[beaten.clone(), beaten.clone()]).0, None);
        assert_eq!(fold(&hit, &[beaten.clone(), talking]).0, Some(1));
        let mut starred = st(p, king(8, 0, SUMMIT), false);
        assert_eq!(fold(&hit, &[starred.clone(), starred.clone()]).0, None);
        starred.course_stars[0] = 1;
        assert_eq!(fold(&hit, &[beaten.clone(), starred]).0, Some(1));
        // Leaving the zone without it is Lost.
        let mut away = beaten.clone();
        away.area = 2;
        assert_eq!(fold(&hit, &[beaten, away]).1[1], Eta::Lost);
    }

    /// Object::Grab on the King: Done once held, closer is less ETA.
    #[test]
    fn king_grab() {
        let far = st([SUMMIT[0] + 900.0, SUMMIT[1], SUMMIT[2]], king(2, 3, SUMMIT), false);
        let near = st([SUMMIT[0] + 100.0, SUMMIT[1], SUMMIT[2]], king(2, 3, SUMMIT), false);
        let held = st([SUMMIT[0] + 100.0, SUMMIT[1], SUMMIT[2]], king(2, 3, SUMMIT), true);
        let grab = Grab::new(&far, &far.objects()[0], 20.0, true);
        let (met, etas) = fold(&grab, &[far, near, held]);
        assert_eq!(met, Some(2));
        assert!(etas[1].frames() < etas[0].frames());
    }
}
