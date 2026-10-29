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
//! Nothing in [`Obj`] shows his timer or subaction, so the progress tracks
//! his action changes inside the trajectory and is optimistic about what it
//! hasn't seen (the start state's timers): a later state can't look better
//! than the start by learning that time has passed.
//!
//! Progress: hits taken (real `oHealth` drops only), then minus the
//! predicted frame of the next hit: when he can next attack, when Mario is
//! in his cone (straight-line at [`PACE`]), when he lies flat, when Mario
//! is above his back with a ground pound. Waiting in place never improves
//! it (the prediction is an absolute frame, so idling in a slow spot makes
//! it later); getting into position faster than [`PACE`] does. Mario
//! squished ends the attempt's score.
//!
//! Staged like `Boss::Hit` (boss.rs): init (success: the dialog opens or he
//! chases), alive (success: his hp below the start's), beaten (act 8:
//! success: DIALOG_115 opens or act 9), done (act 9: success: the star).
//! Success heuristic: frames, so faster fights win. While on, the old
//! `WhompKing::Pound` keeps only the approach (act 0; extra/bosses.rs).

use std::sync::Arc;

use obrrt_core::buttons::Z_TRIG;
use obrrt_core::layout::{ACT_FLAG_AIR, LEVEL_WF};
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St};

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
/// Reference pace for straight-line distances (u/frame): getting somewhere
/// faster than this is progress, slower is a loss.
const PACE: f64 = 24.0;
/// Within this of him an attempt may start (the fight's arena).
const START_RADIUS: f32 = 3000.0;
/// Init: he talks within 600 (oDistanceToMario).
const TALK: f64 = 550.0;

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
    /// pound on it (straight line to the back point at [`PACE`]).
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
        s.hdist(back) as f64 / PACE + gp
    }

    /// Frames from `s` until Mario is in his cone.
    fn to_cone(s: &St, o: &Obj) -> f64 {
        if in_cone(s, o) { 0.0 } else { s.dist(cone_point(o)) as f64 / PACE }
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
            _ => f64::MAX,
        }
    }

    /// Per-state frame estimates over a trajectory: (index, predicted frame
    /// of the stage's goal), stopping where Mario is squished.
    fn predictions(&self, traj: &[St]) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        let mut last: Option<i32> = None;
        let mut entry: Option<(u32, i32)> = None;
        let stage0 = traj.first().and_then(|s| self.stage(s));
        for (i, s) in traj.iter().enumerate() {
            if s.action == ACT_SQUISHED || !s.alive() {
                break;
            }
            let Some(o) = self.king(s) else { continue };
            if s.zone() != self.zone {
                continue;
            }
            if let Some(a) = last
                && a != o.action
            {
                entry = Some((s.frame, a));
            }
            last = Some(o.action);
            let now = s.frame as f64;
            let h = match (stage0, self.stage(s)) {
                (_, Some(Stage::Init)) => now + (s.dist(o.pos) as f64 - TALK).max(0.0) / PACE,
                (_, Some(Stage::Alive(_))) => Self::next_hit(s, o, entry),
                // Ready to talk: on the ground, slow.
                (_, Some(Stage::Beaten)) => now + if in_air(s) { 10.0 } else { 0.0 } + s.forward_vel.abs() as f64 / 4.0,
                (_, Some(Stage::Done)) => match self.star_target(s) {
                    Some(t) => now + s.dist(t) as f64 / PACE,
                    None => continue,
                },
                _ => continue,
            };
            out.push((i, h));
        }
        out
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
    fn base_heuristic(&self, s: &St) -> f64 {
        let (Some(stage), Some(o)) = (self.stage(s), self.king(s)) else { return f64::MAX };
        match stage {
            Stage::Done => self.star_target(s).map_or(5.0, |t| s.dist(t) as f64 / 400.0),
            Stage::Beaten => 10.0,
            Stage::Alive(hp) => 10.0 + hp as f64 * 10.0 + (Self::next_hit(s, o, None) - s.frame as f64) / 100.0,
            Stage::Init => 50.0 + s.dist(o.pos) as f64 / 400.0,
        }
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // The King must be in the state: a frame without him is not a hit.
        let stage0 = self.stage(&traj[0])?;
        traj.iter().position(|s| {
            self.got_star(s)
                || match (stage0, self.stage(s)) {
                    (Stage::Init, Some(Stage::Init)) => s.in_dialog(),
                    (Stage::Init, Some(_)) => true,
                    (Stage::Alive(hp0), Some(Stage::Alive(hp))) => hp < hp0,
                    (Stage::Alive(_), Some(Stage::Beaten | Stage::Done)) => true,
                    (Stage::Beaten, Some(Stage::Beaten)) => s.in_dialog(),
                    (Stage::Beaten, Some(Stage::Done)) => true,
                    _ => false,
                }
        })
    }
    fn success_heuristic(&self, traj: &[St]) -> f64 {
        // Frame since power-on, so chains rank by when they finish.
        traj.last().map_or(f64::MAX, |s| s.frame as f64)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let Some(s0) = traj.first() else { return Progress { score: f64::MIN, at: 0 } };
        let f0 = s0.frame as f64;
        let hp0 = self.king(s0).map_or(0, |o| o.health);
        let mut best = Progress { score: f64::MIN, at: 0 };
        for (i, h) in self.predictions(traj) {
            // Hits only from real health drops.
            let hits = match self.stage(&traj[i]) {
                Some(Stage::Alive(hp)) if hp < hp0 => (hp0 - hp) as f64,
                _ => 0.0,
            };
            let score = 100_000.0 * hits - (h - f0);
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
        Budget::Ga { population: 24, generations: 6, horizon: 300 }
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
