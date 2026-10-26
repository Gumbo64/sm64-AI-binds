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
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

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

    fn boss(&self, s: &St) -> Option<(i32, [f32; 3])> {
        s.find(self.id).map(|o| (o.health, o.pos))
    }

    fn holding(&self, s: &St) -> bool {
        s.held().is_some_and(|h| h.id == self.id)
    }
}

impl Objective for Hit {
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
    fn base_heuristic(&self, s: &St) -> f64 {
        let (Some(stage), Some((_, p))) = (self.stage(s), self.boss(s)) else { return f64::MAX };
        // Done (near the star first), then beaten, then fewer hits left,
        // holding him, close to him.
        match stage {
            Stage::Done => self.star_target(s).map_or(5.0, |t| dist(s.pos, t) as f64 / 400.0),
            Stage::Beaten => 10.0,
            Stage::Alive(hp) => 10.0 + hp as f64 * 10.0 + if self.holding(s) { 0.0 } else { 5.0 + dist(s.pos, p) as f64 / 400.0 },
        }
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // The boss must be in the state: a frame without it (objects not
        // read, or not in the list) is not a hit. Its death never removes it
        // (the King hides in act 8); the star, once had, counts either way.
        let stage0 = self.stage(&traj[0])?;
        // Done: this boss's star only (got_star), not any star or key.
        traj.iter().position(|s| {
            self.got_star(s)
                || match (stage0, self.stage(s)) {
                    (Stage::Alive(hp0), Some(Stage::Alive(hp))) => hp < hp0,
                    (Stage::Alive(_), Some(Stage::Beaten | Stage::Done)) => true,
                    (Stage::Beaten, Some(Stage::Beaten)) => s.in_dialog(),
                    (Stage::Beaten, Some(Stage::Done)) => true,
                    _ => false,
                }
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, score) = traj
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let (_, p) = self.boss(s)?;
                Some((
                    i,
                    match self.stage(s)? {
                        Stage::Done => -(dist(s.pos, self.star_target(s)?) as f64),
                        Stage::Beaten => 0.0,
                        Stage::Alive(_) if self.holding(s) => 1000.0,
                        Stage::Alive(_) => -(dist(s.pos, p) as f64),
                    },
                ))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f64::MIN));
        Progress { score, at }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 200 }
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
