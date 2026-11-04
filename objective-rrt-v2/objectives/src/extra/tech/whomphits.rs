//! `Boss::WhompFast` (technique `whomphits`): WF star 1 ("Chip Off Whomp's
//! Block"), the Whomp King beaten the TAS way: be in his attack cone the
//! first frame he can attack, be in the air above his back as he lands flat,
//! and ground pound it on his first flat frame. Every cycle then takes the
//! game's minimum of 114 frames (TAS 2026 f16212-16510: hits at f16282,
//! f16396, f16510; ref/tas/techniques/c2.md "Whomp King").
//!
//! The mechanics (sm64-port src/game/behaviors/whomp.inc.c, King: behParam2
//! != 0), with the TAS's frame counts (state frames, `oAction` as read):
//! - act 0 (init): within 600 of Mario he talks (DIALOG_114), then act 2.
//! - act 2 (`king_whomp_chase`): turns toward Mario 0x200 a frame; only once
//!   `oTimer > 30` (the act-3 state is 32 frames after the act-2 one) and
//!   with Mario within 0x2000 of his facing and 300 away does he attack.
//! - act 3 (prepare, 15 frames), act 4 (jump: vy 40, then pitches forward,
//!   19 frames), act 5 (land, 3 frames), act 6 (flat): 37 frames from the
//!   act-3 state to the act-6 one. His body lies 100..860 ahead of his base,
//!   its top 200 above it; Mario under it on landing is squished.
//! - act 6, subaction 0: `cur_obj_is_mario_ground_pounding_platform`
//!   (Mario's platform is him and Mario in ACT_GROUND_POUND_LAND) takes 1
//!   `oHealth` (3 to start). The hit shows the next state. He is flat for
//!   at most 101 frames (`oTimer > 100`), then rises (0x200 pitch a frame:
//!   32 frames) back to act 2; after a hit he shakes 11 frames first (hit
//!   to act 2: 44 frames).
//! - hp 0: act 8, waiting for Mario to be ready for DIALOG_115; after it,
//!   act 9, the star spawned flying to (180, 3880, 340).
//!
//! Nothing in [`Obj`] shows his timer or subaction, so the ETA tracks his
//! action changes inside the rollout (in `Mem`) and is optimistic about what
//! it hasn't seen (the start state's timers): a later state can't look
//! better than the start by learning that time has passed.
//!
//! ETA (v2): the predicted frame of the next hit minus now: when he can next
//! attack, when Mario is in his cone (straight-line at core's `PACE`), when
//! he lies flat, when Mario is above his back with a ground pound. Waiting
//! in place never beats getting into position (the prediction is an
//! absolute frame, so idling in a slow spot makes it later); getting
//! into position faster than `PACE` does. Mario squished is `Lost`.
//!
//! Staged like `Boss::Hit` (boss.rs): init (success: the dialog opens or he
//! chases), alive (success: his hp below the start's), beaten (act 8:
//! success: DIALOG_115 opens or act 9), done (act 9: success: the star).
//! While on, the old
//! `WhompKing::Pound` keeps only the approach (act 0; extra/bosses.rs).

use std::sync::Arc;

use obrrt_core::buttons::Z_TRIG;
use obrrt_core::layout::{ACT_FLAG_AIR, LEVEL_WF};
use obrrt_core::{Eta, Mem, Obj, ObjId, Objective, St, frames_for};

const COURSE: i16 = 2;
const INDEX: u8 = 0;
const KING: &str = "bhvWhompKingBoss";
const STAR_BHV: &str = "bhvStarSpawnCoordinates";

const ACT_GROUND_POUND: u32 = 0x0080_08A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;
const ACT_SQUISHED: u32 = 0x0002_0339;

// sWhompActions.
const W_INIT: i32 = 0;
const W_CHASE: i32 = 2;
const W_PREPARE: i32 = 3;
const W_JUMP: i32 = 4;
const W_LAND: i32 = 5;
const W_FLAT: i32 = 6;
const W_DIE: i32 = 8;
const W_DONE: i32 = 9;

/// Act-2 state to the first possible act-3 state.
const CHASE_MIN: f64 = 32.0;
/// Act-3, act-4 and act-5 lengths.
const PREPARE: f64 = 15.0;
const JUMP: f64 = 19.0;
const LAND: f64 = 3.0;
/// Flat without a hit: the last vulnerable frame after the act-6 state.
const FLAT_MAX: f64 = 101.0;
/// Rising back up (subaction 10) to the act-2 state.
const RISE: f64 = 33.0;
/// Flat to the pound landing on his back (the jump up, the pound's spin and
/// drop, ~25 frames as bosses.rs POUND_FRAMES).
const POUND_FRAMES: f64 = 25.0;
/// One more hit after this one: he rises, chases (at least CHASE_MIN),
/// prepares, jumps, lands, and the pound (one cycle of `next_hit`).
const HIT_CYCLE_FRAMES: f64 = RISE + CHASE_MIN + PREPARE + JUMP + LAND + POUND_FRAMES;
/// Within this of him an attempt may start (the fight's arena).
const START_RADIUS: f32 = 3000.0;
/// Init: he talks within 600 (oDistanceToMario).
const TALK: f32 = 550.0;
/// Beaten, Mario in the air: landing before DIALOG_115 can open (v1's
/// estimate).
const BEATEN_LAND_FRAMES: f64 = 10.0;
/// Beaten: Mario's speed lost per frame coming to a stop (v1's estimate).
const BEATEN_DECEL: f64 = 4.0;
/// From ready (or in range) to the dialog actually opening.
const DIALOG_FRAMES: f64 = 1.0;
/// The star's flight target (whomp.inc.c, the King's star spawn), if its
/// spawner isn't loaded.
const STAR_LANDING: [f32; 3] = [180.0, 3880.0, 340.0];

/// `mem.flags` of the action tracking (v1's `predictions`): an action seen
/// (`f[0]` the last one), an action change seen (`f[1]` the frame it
/// happened, `f[2]` the action it came from).
const M_LAST: u32 = 1;
const M_ENTRY: u32 = 2;

/// A point `forward` ahead of `o` along its facing, `up` above.
fn ahead(o: &Obj, forward: f32, up: f32) -> [f32; 3] {
    let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
    [o.pos[0] + forward * a.sin(), o.pos[1] + up, o.pos[2] + forward * a.cos()]
}

/// Where Mario stands to be attacked: inside the 300 x 0x2000 cone.
fn cone_point(o: &Obj) -> [f32; 3] {
    ahead(o, 200.0, 0.0)
}

/// Above the middle of his back once flat (body 100..860 ahead, top +200).
fn back_point(o: &Obj) -> [f32; 3] {
    ahead(o, 600.0, 200.0)
}

fn in_cone(s: &St, o: &Obj) -> bool {
    let d = s.hdist(o.pos);
    if d >= 290.0 {
        return false;
    }
    let to = ((s.pos[0] - o.pos[0]).atan2(s.pos[2] - o.pos[2]) * 65536.0 / std::f32::consts::TAU) as i32;
    let diff = (to - o.yaw + 0x8000).rem_euclid(0x10000) - 0x8000;
    diff.abs() < 0x1E00
}

fn in_air(s: &St) -> bool {
    s.action & ACT_FLAG_AIR != 0
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Stage {
    Init,
    Alive(i32),
    Beaten,
    Done,
}

pub struct WhompFast {
    id: ObjId,
    zone: (u16, u8),
}

impl WhompFast {
    pub fn new(b: &St, o: &Obj) -> Option<WhompFast> {
        (b.level == LEVEL_WF as u16 && o.bhv_name() == KING && !b.has_star(COURSE, INDEX))
            .then(|| WhompFast { id: o.id, zone: b.zone() })
    }

    fn king<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        s.find(self.id)
    }

    fn stage(&self, s: &St) -> Option<Stage> {
        let o = self.king(s)?;
        match o.action {
            W_INIT => Some(Stage::Init),
            W_DIE => Some(Stage::Beaten),
            W_DONE => Some(Stage::Done),
            a if (W_CHASE..W_DIE).contains(&a) && (1..=3).contains(&o.health) => Some(Stage::Alive(o.health)),
            _ => None,
        }
    }

    fn got_star(&self, s: &St) -> bool {
        s.has_star(COURSE, INDEX)
    }

    fn star_target(&self, s: &St) -> Option<[f32; 3]> {
        s.objects().iter().find(|o| o.bhv_name() == STAR_BHV && (o.beh_params >> 24) as u8 == INDEX).map(|o| o.home)
    }

    /// Frames from `s` until Mario, above or on his back, lands a ground
    /// pound on it (straight line to the back point at core's `PACE`).
    fn to_pound(s: &St, o: &Obj) -> f64 {
        let back = back_point(o);
        let top = o.pos[1] + 200.0;
        let fall = |y: f32| ((y - top).max(0.0) / 50.0) as f64;
        let gp = match s.action {
            ACT_GROUND_POUND_LAND if s.floor_obj == Some(o.slot) => return 0.0,
            ACT_GROUND_POUND => fall(s.pos[1]),
            _ if in_air(s) => 12.0 + fall(s.pos[1]),
            _ => 20.0,
        };
        frames_for(s.hdist(back)) + gp
    }

    /// Frames from `s` until Mario is in his cone.
    fn to_cone(s: &St, o: &Obj) -> f64 {
        if in_cone(s, o) { 0.0 } else { frames_for(s.dist(cone_point(o))) }
    }

    /// The predicted frame of the next hit in an alive state, with `entry`
    /// the frame his current action began if the trajectory saw it begin
    /// (and where from), else None (then the soonest it can be).
    fn next_hit(s: &St, o: &Obj, entry: Option<(u32, i32)>) -> f64 {
        let now = s.frame as f64;
        let from_chase = |ready: f64| ready.max(now + 1.0 + Self::to_cone(s, o)) + PREPARE + JUMP + LAND + 1.0;
        let falling = |flat: f64| flat.max(now + Self::to_pound(s, o)) + 1.0;
        let e = entry.map(|(f, _)| f as f64);
        match o.action {
            W_CHASE => from_chase(e.map_or(now, |e| e + CHASE_MIN)),
            W_PREPARE => falling(e.map_or(now + 1.0 + JUMP + LAND, |e| e + PREPARE + JUMP + LAND)),
            W_JUMP => falling(e.map_or(now + 1.0 + LAND, |e| e + JUMP + LAND)),
            W_LAND => falling(e.map_or(now + 1.0, |e| e + LAND)),
            W_FLAT => {
                let pound = now + Self::to_pound(s, o) + 1.0;
                match entry {
                    // Seen landing: flat (vulnerable) until e + FLAT_MAX.
                    Some((f, W_LAND)) => {
                        let e = f as f64;
                        if pound <= e + FLAT_MAX + 1.0 { pound } else { e + FLAT_MAX + RISE + CHASE_MIN + PREPARE + JUMP + LAND + 1.0 }
                    }
                    // Unseen (or after a hit): flat now, or about to rise.
                    _ => pound.min(from_chase(now + 1.0 + CHASE_MIN)),
                }
            }
            // Any other alive action (act 7, turning): as a fresh chase.
            _ => from_chase(now + CHASE_MIN),
        }
    }

    /// Folds `s`'s King action into `mem` (v1's `predictions` loop): the
    /// frame his current action began and the one before it, once the
    /// rollout has seen a change.
    fn track(o: &Obj, s: &St, mem: &mut Mem) -> Option<(u32, i32)> {
        if mem.flags & M_LAST != 0 && mem.f[0] as i32 != o.action {
            mem.flags |= M_ENTRY;
            mem.f[1] = s.frame as f64;
            mem.f[2] = mem.f[0];
        }
        mem.flags |= M_LAST;
        mem.f[0] = o.action as f64;
        (mem.flags & M_ENTRY != 0).then(|| (mem.f[1] as u32, mem.f[2] as i32))
    }
}

impl WhompFast {
    /// This hit's ETA (the phases up to its `Done`).
    fn hit_eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        if self.got_star(s) {
            return Eta::Done;
        }
        let Some(stage0) = self.stage(start) else { return Eta::Lost };
        if s.zone() != self.zone || !s.alive() || s.action == ACT_SQUISHED {
            return Eta::Lost;
        }
        let Some(o) = self.king(s) else { return Eta::Lost };
        let entry = Self::track(o, s, mem);
        let Some(stage) = self.stage(s) else { return Eta::Lost };
        let done = match (stage0, stage) {
            (Stage::Init, Stage::Init) => s.in_dialog(),
            (Stage::Init, _) => true,
            (Stage::Alive(hp0), Stage::Alive(hp)) => hp < hp0,
            (Stage::Alive(_), Stage::Beaten | Stage::Done) => true,
            (Stage::Beaten, Stage::Beaten) => s.in_dialog(),
            (Stage::Beaten, Stage::Done) => true,
            _ => false,
        };
        if done {
            return Eta::Done;
        }
        let now = s.frame as f64;
        Eta::Frames(match stage {
            Stage::Init => frames_for((s.dist(o.pos) - TALK).max(0.0)) + DIALOG_FRAMES,
            Stage::Alive(_) => (Self::next_hit(s, o, entry) - now).max(0.0),
            Stage::Beaten => {
                (if in_air(s) { BEATEN_LAND_FRAMES } else { 0.0 }) + s.forward_vel.abs() as f64 / BEATEN_DECEL + DIALOG_FRAMES
            }
            Stage::Done => frames_for(s.dist(self.star_target(s).unwrap_or(STAR_LANDING))),
        })
    }
}

impl Objective for WhompFast {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): technique setups are frame-precise: every state, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Boss::WhompFast"
    }
    fn key(&self) -> String {
        format!("BossWhompFast {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        if s.zone() != self.zone || !s.alive() || !s.controllable() || self.got_star(s) {
            return false;
        }
        let Some(o) = self.king(s) else { return false };
        match self.stage(s) {
            Some(Stage::Init) => s.dist(o.pos) < 1500.0,
            Some(Stage::Alive(_) | Stage::Beaten) => s.dist(o.pos) < START_RADIUS,
            Some(Stage::Done) => self.star_target(s).is_some(),
            None => false,
        }
    }
    /// `Done` on v1's success for the start's stage: this star, or Init:
    /// the dialog opens or he chases; Alive: his health below the start's
    /// (or beaten); Beaten: DIALOG_115 opens or act 9. `Lost`: the zone
    /// left, dead, squished (v1's progress stopped), the start without a
    /// stage, or the King out of a stage. Else the frames to the stage's
    /// goal, v1's predicted frame minus now: Init the walk into talking
    /// range; Alive [`WhompFast::next_hit`] (with his action changes
    /// tracked in `mem`); Beaten landing and stopping; Done the way to the
    /// star.
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        // The fight after this hit (v1 ranked starts by hits left, 10
        // heuristic units a hit): from the start's health, which only
        // changes at a hit (Done), so the term is constant in an attempt.
        let rest = match self.stage(start) {
            // His dialog first: all three hits after it (whomp_init sets 3).
            Some(Stage::Init) => 3.0 * HIT_CYCLE_FRAMES,
            Some(Stage::Alive(hp0)) => (hp0 - 1).max(0) as f64 * HIT_CYCLE_FRAMES,
            _ => 0.0,
        };
        match self.hit_eta(s, start, mem) {
            Eta::Frames(f) => Eta::Frames(f + rest),
            e => e,
        }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 24, generations: 6, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.king(s)?;
        match self.stage(s)? {
            Stage::Init => Some(o.pos),
            Stage::Beaten => None,
            Stage::Done => self.star_target(s),
            Stage::Alive(_) => Some(match o.action {
                W_CHASE => cone_point(o),
                // Falling or flat: above his back (the default A while the
                // target is higher jumps out from under him).
                _ => back_point(o),
            }),
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Above his back as he lands or lies flat: ground pound.
        match self.king(s) {
            Some(o)
                if matches!(o.action, W_JUMP | W_LAND | W_FLAT)
                    && in_air(s)
                    && s.action != ACT_GROUND_POUND
                    && s.hdist(back_point(o)) < 300.0
                    && s.pos[1] > o.pos[1] + 230.0 =>
            {
                Z_TRIG
            }
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !self.got_star(s) && self.stage(s).is_some()
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// Offered per loaded Whomp King while WF star 1 is missing.
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    if let Some(x) = WhompFast::new(b, o) {
        out.push(Arc::new(x));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::tech::testkit::{ACT_WALKING, obj, st, with};
    use obrrt_core::fold;

    const ACT_JUMP: u32 = 0x0300_0880;

    fn at(frame: u32, p: [f32; 3], action: u32, king_act: i32, hp: i32) -> St {
        let mut s = with(st(LEVEL_WF, 1, p, action), vec![obj(KING, 9, [0.0, 3584.0, 0.0], king_act, hp)]);
        s.frame = frame;
        s
    }

    #[test]
    fn one_cycle_counts_down_to_the_hit() {
        let traj = vec![
            at(0, [0.0, 3584.0, 1150.0], ACT_WALKING, W_CHASE, 3),
            at(40, [0.0, 3584.0, 150.0], ACT_WALKING, W_CHASE, 3),
            at(41, [0.0, 3584.0, 150.0], ACT_WALKING, W_PREPARE, 3),
            at(60, [0.0, 3900.0, 600.0], ACT_JUMP, W_JUMP, 3),
            at(79, [0.0, 3850.0, 600.0], ACT_GROUND_POUND, W_LAND, 3),
            at(82, [0.0, 3800.0, 600.0], ACT_GROUND_POUND, W_FLAT, 3),
            at(83, [0.0, 3784.0, 600.0], ACT_GROUND_POUND_LAND, W_FLAT, 2),
        ];
        let king = WhompFast { id: traj[0].objects()[0].id, zone: (LEVEL_WF as u16, 1) };
        let (met, etas) = fold(&king, &traj);
        assert_eq!(met, Some(6), "{etas:?}");
        let f: Vec<f64> = etas.iter().map(|e| e.frames()).collect();
        assert!(f.windows(2).all(|w| w[1] < w[0]), "ETA not decreasing: {f:?}");
        // Squished under him: Lost.
        let mut sq = traj[4].clone();
        sq.action = ACT_SQUISHED;
        assert!(king.eta(&sq, &traj[0], &mut Mem::default()) == Eta::Lost);
    }
}
