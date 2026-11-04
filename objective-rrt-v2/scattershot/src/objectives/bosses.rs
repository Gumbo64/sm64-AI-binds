//! Bosses, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: `objective-rrt/objectives/src/boss.rs` and `src/extra/bosses.rs`.
//!
//! # What `Boss` already covers
//! The generic [`Boss`](super::boss::Boss) kind (edit-locked) keys Mario's
//! position, whether he holds the boss, the straight-line distance to the held
//! or nearest boss, the boss's position, health, raw action and facing, plus
//! Mario's facing and spin while he holds it. The kinds below do not repeat
//! that. They add what the rrt objectives read beyond it: the fight's phase
//! (Alive/Beaten/Done, hit points left, eye open, lying flat, thrown), where
//! Mario is relative to the boss (behind it, in front, on its back, above it),
//! and boss-specific targets (the spawned star, Bowser's tail, the bombs, how
//! far a bully has been pushed). Use one as a child of `Boss`, or next to it,
//! for the same behaviour; they write no Mario position of their own.
//!
//! Every kind is gated as rrt's `Fight::base` is: silent when no matching
//! boss is loaded, when it is not in the fighting state, or when the boss's
//! star is already owned. Hit points are written as they are (the number of
//! hits left), actions as a few classes, and angles as sectors centred on the
//! boss's facing (0 in front, `sectors / 2` behind). All distances are
//! straight-line.
//!
//! # Mapping of the rrt kinds
//! | rrt kind | scattershot kind |
//! |---|---|
//! | `Boss::Hit` (boss.rs) | [`BossHit`]: stage, hp, held, side, dist (boss or spawned star) |
//! | `WhompKing::Pound` | [`WhompKingPound`] |
//! | `BigBoo::Hit` | [`BigBooHit`] |
//! | `Wiggler::Stomp` | [`WigglerStomp`] |
//! | `Eyerok::PunchEye` | [`EyerokPunchEye`] |
//! | `Eyerok::Wake` | [`EyerokWake`] |
//! | `BigBully::PushOff` | [`BigBullyPushOff`] |
//! | `Bowser::GrabTail` | [`BowserGrabTail`] |
//! | `Bowser::ThrowAtBomb` | [`BowserThrowAtBomb`] |
//!
//! Not ported (search machinery, not a state distinction): the budgets (GA
//! population, generations, horizon), `obj_weight`/success values, kinship,
//! `Precision::Precise`, `steer_target`/`steer_button` (dodge and approach
//! points, B/A/Z hints), `progress` scores and heuristics (they turn into the
//! bucketed distance fields), `exists_in`, and `success` as a test (it turns
//! into fields that make the success state a cell of its own: hp, phase). Also
//! left out: the `whomphits` / `Boss::WhompFast` switch (not in these
//! sources: `WhompKingPound` always covers the whole fight), the objective
//! `key()`/`Fight` identity (per-object objectives become the nearest matching
//! object), and the "hp is junk (2048) before init" guard of `hurt_or`
//! (junk hit points are simply written as they are, or as 3 for Big Boo, as
//! rrt's heuristic does).
//!
//! Every kind has `step` (distance bucket size, default 250, rrt's
//! `arrival_scale` is 400) and, where an angle matters, `sectors` (default 8).
//! `star` is `[course, index]` of the boss's star (the fight ends once it is
//! owned); left out, it is read from the boss's behaviour parameter as rrt's
//! `star_of` does.

use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist};
use crate::objective::{Cx, Objective};

// Mario actions (include/sm64.h), as in rrt's extra/bosses.rs.
const ACT_PICKING_UP_BOWSER: u32 = 0x0000_0390;
const ACT_HOLDING_BOWSER: u32 = 0x0000_0391;
const ACT_GROUND_POUND: u32 = 0x0080_08A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;

// Boss actions (whomp.inc.c, boo.inc.c, wiggler, eyerok, bully, bowser).
const WHOMP_INIT: i32 = 0;
const WHOMP_PREPARE_JUMP: i32 = 3;
const WHOMP_JUMP: i32 = 4;
const WHOMP_LAND: i32 = 5;
const WHOMP_ON_GROUND: i32 = 6;
const WHOMP_DIE: i32 = 8;
const BIG_BOO_INACTIVE: i32 = 0;
const BIG_BOO_HURT: i32 = 3;
const BIG_BOO_DEAD: i32 = 4;
const WIGGLER_UNINITIALIZED: i32 = 0;
const WIGGLER_JUMPED_ON: i32 = 3;
const WIGGLER_SHRINK: i32 = 4;
const WIGGLER_FALL_THROUGH_FLOOR: i32 = 5;
const EYEROK_BOSS_SLEEP: i32 = 0;
const EYEROK_HAND_SLEEP: i32 = 0;
const EYEROK_HAND_SHOW_EYE: i32 = 3;
const EYEROK_HAND_DIE: i32 = 15;
const BULLY_KNOCKBACK: i32 = 2;
const BULLY_INACTIVE: i32 = 4;
const BULLY_ACTIVATE_AND_FALL: i32 = 5;
const BULLY_LAVA_DEATH: i32 = 100;
const BULLY_DEATH_PLANE_DEATH: i32 = 101;
const BOWSER_THROWN: i32 = 1;
const BOWSER_DEAD: i32 = 4;

// Bowser levels (obrrt_core::layout::LEVEL_BOWSER_*).
const LEVEL_BOWSER_1: u32 = 30;
const LEVEL_BOWSER_2: u32 = 33;
const LEVEL_BOWSER_3: u32 = 34;

/// A spawned star flies to its home (spawn_star.inc.c).
const STAR_BHV: &str = "bhvStarSpawnCoordinates";

/// One behaviour name or a list of them.
#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub enum Bhvs {
    One(String),
    Many(Vec<String>),
}

impl Bhvs {
    fn addrs(&self) -> Vec<u32> {
        match self {
            Bhvs::One(s) => bhv_addr(s).into_iter().collect(),
            Bhvs::Many(v) => v.iter().filter_map(|s| bhv_addr(s)).collect(),
        }
    }
}

fn one(s: &str) -> Bhvs {
    Bhvs::One(s.to_string())
}
fn step() -> f32 {
    250.0
}
fn sectors() -> u32 {
    8
}

// ---------------------------------------------------------------------------
// Shared helpers.

fn objs(st: &St) -> &[Obj] {
    st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[])
}

/// The matching object nearest to Mario that satisfies `ok`.
fn nearest<'a>(st: &'a St, addrs: &[u32], ok: impl Fn(&Obj) -> bool) -> Option<&'a Obj> {
    objs(st)
        .iter()
        .filter(|o| addrs.contains(&o.bhv) && ok(o))
        .min_by(|a, b| dist(a.pos, st.pos).total_cmp(&dist(b.pos, st.pos)))
}

/// Whether the boss's star is owned: `star` = [course, index], else the star
/// the boss's behaviour parameter names in the current course (rrt `star_of`).
fn star_owned(st: &St, o: &Obj, star: &Option<[i32; 2]>) -> bool {
    match star {
        Some([c, i]) => st.has_star(*c as i16, *i as u8),
        None => {
            let index = (o.beh_params >> 24) as u8;
            (1..=15).contains(&st.course) && index <= 5 && st.has_star(st.course, index)
        }
    }
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    (a[0] - b[0]).hypot(a[2] - b[2])
}

fn yaw_to_vec(yaw: i32) -> (f32, f32) {
    let a = yaw as f32 * std::f32::consts::TAU / 65536.0;
    (a.sin(), a.cos())
}

/// SM64 yaw (0 = +z, 0x4000 = +x) of the direction from `a` to `b`.
fn yaw_between(a: [f32; 3], b: [f32; 3]) -> i32 {
    ((b[0] - a[0]).atan2(b[2] - a[2]) * 65536.0 / std::f32::consts::TAU) as i32
}

/// Signed difference of two yaws, -0x8000..0x8000.
fn yaw_diff(a: i32, b: i32) -> i32 {
    (a - b + 0x8000).rem_euclid(0x10000) - 0x8000
}

fn offset(p: [f32; 3], yaw: i32, forward: f32, right: f32, up: f32) -> [f32; 3] {
    let (sx, cz) = yaw_to_vec(yaw);
    [p[0] + forward * sx + right * cz, p[1] + up, p[2] + forward * cz - right * sx]
}

/// Where Mario is around `o`, in `n` sectors centred on its facing: 0 in
/// front, `n / 2` directly behind.
fn around(st: &St, o: &Obj, n: u32) -> i32 {
    let d = yaw_diff(yaw_between(o.pos, st.pos), o.yaw);
    let u = (d + 0x10000 / (2 * n as i32)).rem_euclid(0x10000) as u32;
    ((u * n) >> 16) as i32
}

/// `bucket` limited to 0..=max, so far-away states share one value.
fn cb(v: f32, step: f32, max: i32) -> i32 {
    bucket(v, step).clamp(0, max)
}

// ---------------------------------------------------------------------------
// King Bob-omb (Boss::Hit).

/// `Boss::Hit` (boss.rs): the stages of a boss fight that ends in a star. For
/// King Bob-omb: Alive (hp > 0), Beaten (act 7: hp 0, waiting to speak), Done
/// (act 8: hidden, the star spawned and flying to its home). Gate: the boss is
/// loaded, its star is not owned, and (for Done) the spawned star exists.
///
/// Fields: `stage` (0 alive, 1 beaten, 2 done); `hp` (alive: hit points left,
/// else 0); `held` (Mario holds the boss); `side` (alive: Mario's sector
/// around the boss, 0 in front, `sectors / 2` behind, which is where he is
/// grabbed from; else -1); `dist` (alive: bucketed distance to the boss; done:
/// to the spawned star; beaten: -1).
///
/// ```json
/// {"kind": "BossHit", "params": {"bhv": "bhvKingBobomb", "star": [1, 0]}}
/// ```
#[derive(Deserialize)]
pub struct BossHit {
    #[serde(default = "king")]
    pub bhv: Bhvs,
    /// Boss action for Beaten.
    #[serde(default = "beaten_act")]
    pub beaten_act: i32,
    /// Boss action for Done.
    #[serde(default = "done_act")]
    pub done_act: i32,
    #[serde(default = "king_star")]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default = "sectors")]
    pub sectors: u32,
}

fn king() -> Bhvs {
    one("bhvKingBobomb")
}
fn beaten_act() -> i32 {
    7
}
fn done_act() -> i32 {
    8
}
fn king_star() -> Option<[i32; 2]> {
    Some([1, 0])
}

impl Objective for BossHit {
    fn fields(&self) -> &[&'static str] {
        &["stage", "hp", "held", "side", "dist"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let addrs = self.bhv.addrs();
        // The held one, else the nearest.
        let held = objs(st).iter().find(|o| addrs.contains(&o.bhv) && Some(o.slot) == st.held_slot);
        let Some(b) = held.or_else(|| nearest(st, &addrs, |_| true)) else { return };
        if star_owned(st, b, &self.star) {
            return;
        }
        let index = self.star.map_or((b.beh_params >> 24) as u8, |s| s[1] as u8);
        let spawned = || {
            let star = bhv_addr(STAR_BHV)?;
            objs(st).iter().find(|o| o.bhv == star && (o.beh_params >> 24) as u8 == index)
        };
        let (stage, hp, d) = if b.action == self.beaten_act {
            (1, 0, -1)
        } else if b.action == self.done_act {
            // Done counts only with a star to collect.
            let Some(s) = spawned() else { return };
            (2, 0, bucket(dist(st.pos, s.home), self.step))
        } else if b.health > 0 {
            (0, b.health, bucket(dist(st.pos, b.pos), self.step))
        } else {
            return;
        };
        cx.put(0, stage);
        cx.put(1, hp);
        cx.put(2, held.is_some() as i32);
        cx.put(3, if stage == 0 { around(st, b, self.sectors) } else { -1 });
        cx.put(4, d);
    }
}

// ---------------------------------------------------------------------------
// Whomp King.

/// `WhompKing::Pound`: lure him into falling flat (stand in front of him),
/// dodge, get on his back and ground pound it (whomp.inc.c). Gate: hp 1-3,
/// action below 8 (dying), star not owned.
///
/// Fields: `hp`; `phase` (0 waiting to start, 1 chasing/other, 2 falling
/// (prepare-jump/jump), 3 lying flat); `dist` (lying: horizontal distance to
/// the middle of his back, bucketed by `back_step`; else straight-line
/// distance to him by `step`); `on_top` (lying: above his back and within
/// 350 of it, else 0); `pound` (lying: Mario's action, 0 other, 1 ground
/// pounding, 2 landed from a pound; else 0).
///
/// ```json
/// {"kind": "WhompKingPound", "params": {"back_step": 150}}
/// ```
#[derive(Deserialize)]
pub struct WhompKingPound {
    #[serde(default = "whomp")]
    pub bhv: Bhvs,
    #[serde(default)]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default = "back_step")]
    pub back_step: f32,
}

fn whomp() -> Bhvs {
    one("bhvWhompKingBoss")
}
fn back_step() -> f32 {
    150.0
}

impl Objective for WhompKingPound {
    fn fields(&self) -> &[&'static str] {
        &["hp", "phase", "dist", "on_top", "pound"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let fighting = |o: &Obj| (1..=3).contains(&o.health) && o.action < WHOMP_DIE;
        let Some(o) = nearest(st, &self.bhv.addrs(), fighting) else { return };
        if star_owned(st, o, &self.star) {
            return;
        }
        cx.put(0, o.health);
        let lying = matches!(o.action, WHOMP_LAND | WHOMP_ON_GROUND);
        let phase = match o.action {
            _ if lying => 3,
            WHOMP_PREPARE_JUMP | WHOMP_JUMP => 2,
            WHOMP_INIT => 0,
            _ => 1,
        };
        cx.put(1, phase);
        if lying {
            // The centre of his back while he lies flat (rrt `WhompPound::back`).
            let back = offset(o.pos, o.yaw, 480.0, 0.0, 200.0);
            let h = hdist(st.pos, back);
            let on_top = st.pos[1] > back[1] - 30.0 && h < 350.0;
            cx.put(2, cb(h, self.back_step, 20));
            cx.put(3, on_top as i32);
            cx.put(4, match st.action {
                ACT_GROUND_POUND => 1,
                ACT_GROUND_POUND_LAND => 2,
                _ => 0,
            });
        } else {
            cx.put(2, cb(dist(st.pos, o.pos), self.step, 20));
            cx.put(3, 0);
            cx.put(4, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Big Boo.

/// `BigBoo::Hit`: punch or kick a Big Boo from its side while it is visible
/// (boo.inc.c). Gate: not dead, and not hurt with its last hit point, star
/// not owned.
///
/// Fields: `hp` (1-3; before it is activated, 3); `phase` (0 inactive, 1 hurt,
/// 2 otherwise); `tangible` (it can be hit); `dist`; `side` (Mario's sector
/// around the Boo, 0 in front, `sectors / 2` behind; it is hit from the side).
///
/// ```json
/// {"kind": "BigBooHit", "params": {"step": 250}}
/// ```
#[derive(Deserialize)]
pub struct BigBooHit {
    #[serde(default = "big_boos")]
    pub bhv: Bhvs,
    #[serde(default)]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default = "sectors")]
    pub sectors: u32,
}

fn big_boos() -> Bhvs {
    Bhvs::Many(["bhvGhostHuntBigBoo", "bhvBalconyBigBoo", "bhvMerryGoRoundBigBoo"].map(String::from).to_vec())
}

impl Objective for BigBooHit {
    fn fields(&self) -> &[&'static str] {
        &["hp", "phase", "tangible", "dist", "side"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let fighting = |o: &Obj| o.action != BIG_BOO_DEAD && !(o.action == BIG_BOO_HURT && o.health <= 1);
        let Some(o) = nearest(st, &self.bhv.addrs(), fighting) else { return };
        if star_owned(st, o, &self.star) {
            return;
        }
        cx.put(0, if (1..=3).contains(&o.health) { o.health } else { 3 });
        cx.put(1, match o.action {
            BIG_BOO_INACTIVE => 0,
            BIG_BOO_HURT => 1,
            _ => 2,
        });
        cx.put(2, (o.interact_type != 0) as i32);
        cx.put(3, cb(dist(st.pos, o.pos), self.step, 20));
        cx.put(4, around(st, o, self.sectors));
    }
}

// ---------------------------------------------------------------------------
// Wiggler.

/// `Wiggler::Stomp`: stomp or ground pound the Wiggler's head (wiggler.inc.c).
/// Gate: hp 2-4 and not uninitialised, shrinking or falling; star not owned.
///
/// Fields: `hp`; `jumped` (a stomp has landed: action 3); `dist`; `above`
/// (Mario's height over the head, in `dy_step` buckets, -2..=4).
///
/// ```json
/// {"kind": "WigglerStomp", "params": {"dy_step": 100}}
/// ```
#[derive(Deserialize)]
pub struct WigglerStomp {
    #[serde(default = "wiggler")]
    pub bhv: Bhvs,
    #[serde(default)]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default = "dy_step")]
    pub dy_step: f32,
}

fn wiggler() -> Bhvs {
    one("bhvWigglerHead")
}
fn dy_step() -> f32 {
    100.0
}

impl Objective for WigglerStomp {
    fn fields(&self) -> &[&'static str] {
        &["hp", "jumped", "dist", "above"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let fighting = |o: &Obj| {
            !matches!(o.action, WIGGLER_UNINITIALIZED | WIGGLER_SHRINK | WIGGLER_FALL_THROUGH_FLOOR) && (2..=4).contains(&o.health)
        };
        let Some(o) = nearest(st, &self.bhv.addrs(), fighting) else { return };
        if star_owned(st, o, &self.star) {
            return;
        }
        cx.put(0, o.health);
        cx.put(1, (o.action == WIGGLER_JUMPED_ON) as i32);
        cx.put(2, cb(dist(st.pos, o.pos), self.step, 20));
        cx.put(3, bucket(st.pos[1] - o.pos[1], self.dy_step).clamp(-2, 4));
    }
}

// ---------------------------------------------------------------------------
// Eyerok.

fn eyerok_star() -> Option<[i32; 2]> {
    Some([8, 3])
}

/// `Eyerok::PunchEye`: punch a hand's eye while it is open and Mario is in
/// front of it (eyerok.inc.c). Gate: a hand awake and not dying, hp 2-4, the
/// Eyerok star (SSL, 3) not owned.
///
/// Fields: `hp`; `eye` (the eye is open: action 3); `front` (Mario within
/// 0x3000 of the hand's facing); `dist`.
///
/// ```json
/// {"kind": "EyerokPunchEye", "params": {"step": 150}}
/// ```
#[derive(Deserialize)]
pub struct EyerokPunchEye {
    #[serde(default = "eyerok_hand")]
    pub bhv: Bhvs,
    #[serde(default = "eyerok_star")]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
}

fn eyerok_hand() -> Bhvs {
    one("bhvEyerokHand")
}

impl Objective for EyerokPunchEye {
    fn fields(&self) -> &[&'static str] {
        &["hp", "eye", "front", "dist"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let fighting = |o: &Obj| !matches!(o.action, EYEROK_HAND_SLEEP | EYEROK_HAND_DIE) && (2..=4).contains(&o.health);
        let Some(o) = nearest(st, &self.bhv.addrs(), fighting) else { return };
        if star_owned(st, o, &self.star) {
            return;
        }
        cx.put(0, o.health);
        cx.put(1, (o.action == EYEROK_HAND_SHOW_EYE) as i32);
        cx.put(2, (yaw_diff(yaw_between(o.pos, st.pos), o.yaw).abs() < 0x3000) as i32);
        cx.put(3, cb(dist(st.pos, o.pos), self.step, 20));
    }
}

/// `Eyerok::Wake`: come within 500 of the sleeping Eyerok to start the fight.
/// Gate: the boss is loaded and the star not owned.
///
/// Fields: `awake` (its action is not 0); `dist` (asleep: bucketed distance
/// to it, else -1; the default `step` 250 puts the 500 wake radius on a
/// bucket edge).
///
/// ```json
/// {"kind": "EyerokWake"}
/// ```
#[derive(Deserialize)]
pub struct EyerokWake {
    #[serde(default = "eyerok_boss")]
    pub bhv: Bhvs,
    #[serde(default = "eyerok_star")]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
}

fn eyerok_boss() -> Bhvs {
    one("bhvEyerokBoss")
}

impl Objective for EyerokWake {
    fn fields(&self) -> &[&'static str] {
        &["awake", "dist"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(o) = nearest(st, &self.bhv.addrs(), |_| true) else { return };
        if star_owned(st, o, &self.star) {
            return;
        }
        let asleep = o.action == EYEROK_BOSS_SLEEP;
        cx.put(0, !asleep as i32);
        cx.put(1, if asleep { cb(dist(st.pos, o.pos), self.step, 20) } else { -1 });
    }
}

// ---------------------------------------------------------------------------
// Big bullies.

/// `BigBully::PushOff`: push a bully off its platform into lava or the death
/// plane (bully.inc.c). Bullies have no hit points; the fight is how far it
/// has been pushed. Gate: active (not inactive, not activating), star not
/// owned. The two death actions are written too, so the defeat is a cell.
///
/// Fields: `phase` (0 fighting, 1 knocked back, 2 dead); `pushed` (distance
/// from its home across plus three times the drop below it, in `push_step`
/// buckets, 0..=10); `dist` (straight-line to it); `behind` (Mario is behind
/// it along the push direction: away from home, or away from Mario near home).
///
/// ```json
/// {"kind": "BigBullyPushOff", "params": {"push_step": 300}}
/// ```
#[derive(Deserialize)]
pub struct BigBullyPushOff {
    #[serde(default = "bullies")]
    pub bhv: Bhvs,
    #[serde(default)]
    pub star: Option<[i32; 2]>,
    #[serde(default = "step")]
    pub step: f32,
    #[serde(default = "push_step")]
    pub push_step: f32,
}

fn bullies() -> Bhvs {
    Bhvs::Many(["bhvBigBully", "bhvBigBullyWithMinions", "bhvBigChillBully"].map(String::from).to_vec())
}
fn push_step() -> f32 {
    300.0
}

impl Objective for BigBullyPushOff {
    fn fields(&self) -> &[&'static str] {
        &["phase", "pushed", "dist", "behind"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let active = |o: &Obj| !matches!(o.action, BULLY_INACTIVE | BULLY_ACTIVATE_AND_FALL);
        let Some(o) = nearest(st, &self.bhv.addrs(), active) else { return };
        if star_owned(st, o, &self.star) {
            return;
        }
        let phase = match o.action {
            BULLY_LAVA_DEATH | BULLY_DEATH_PLANE_DEATH => 2,
            BULLY_KNOCKBACK => 1,
            _ => 0,
        };
        cx.put(0, phase);
        let pushed = hdist(o.pos, o.home) + (o.home[1] - o.pos[1]).max(0.0) * 3.0;
        cx.put(1, cb(pushed, self.push_step, 10));
        cx.put(2, cb(dist(st.pos, o.pos), self.step, 20));
        // Push direction: away from home, or (near home) away from Mario.
        let (mut dx, mut dz) = (o.pos[0] - o.home[0], o.pos[2] - o.home[2]);
        if dx.hypot(dz) < 60.0 {
            (dx, dz) = (o.pos[0] - st.pos[0], o.pos[2] - st.pos[2]);
        }
        let n = dx.hypot(dz).max(1e-3);
        let along = ((st.pos[0] - o.pos[0]) * dx + (st.pos[2] - o.pos[2]) * dz) / n;
        cx.put(3, (along < -50.0) as i32);
    }
}

// ---------------------------------------------------------------------------
// Bowser.

fn holding_bowser(st: &St) -> bool {
    matches!(st.action, ACT_PICKING_UP_BOWSER | ACT_HOLDING_BOWSER)
}

/// Bowser alive and its key not yet had (the first key is BITDW's, the
/// second BITFS's; `keys` counts keys held or used).
fn bowser_fighting(st: &St, o: &Obj) -> bool {
    let key_done = match st.level as u32 {
        LEVEL_BOWSER_1 => st.keys >= 1,
        LEVEL_BOWSER_2 => st.keys >= 2,
        LEVEL_BOWSER_3 => false,
        _ => return false,
    };
    !key_done && o.health > 0 && o.action != BOWSER_DEAD
}

/// Bowser's tail anchor nearest him, else a point behind him.
fn bowser_tail(st: &St, o: &Obj) -> [f32; 3] {
    let anchor = bhv_addr("bhvBowserTailAnchor");
    objs(st)
        .iter()
        .filter(|t| Some(t.bhv) == anchor)
        .map(|t| t.pos)
        .filter(|p| dist(*p, o.pos) < 700.0)
        .min_by(|a, b| dist(*a, o.pos).total_cmp(&dist(*b, o.pos)))
        .unwrap_or_else(|| o.behind(300.0))
}

fn bowser_of<'a>(st: &'a St, bhv: &Bhvs) -> Option<&'a Obj> {
    nearest(st, &bhv.addrs(), |o| bowser_fighting(st, o))
}

fn bowser() -> Bhvs {
    one("bhvBowser")
}
fn tail_step() -> f32 {
    150.0
}

/// `Bowser::GrabTail`: get behind Bowser and grab his tail (B) while facing
/// his way (mario_check_object_grab: within 0x5000 here). Gate: a Bowser level
/// whose key is not yet had, Bowser alive. Picking him up always ends in
/// holding him, so `held` makes that success a cell.
///
/// Fields: `hp`; `held` (Mario is picking up or holding Bowser); `behind`
/// (Mario is on his tail's side: more than 150 behind him); `tail` (distance
/// to the tail anchor in `tail_step` buckets, 0..=12); `facing` (not held:
/// Mario's yaw within 0x5000 of Bowser's, else -1).
///
/// ```json
/// {"kind": "BowserGrabTail", "params": {"tail_step": 150}}
/// ```
#[derive(Deserialize)]
pub struct BowserGrabTail {
    #[serde(default = "bowser")]
    pub bhv: Bhvs,
    #[serde(default = "tail_step")]
    pub tail_step: f32,
}

fn behind_bowser(st: &St, o: &Obj) -> bool {
    let (fx, fz) = yaw_to_vec(o.yaw);
    (st.pos[0] - o.pos[0]) * fx + (st.pos[2] - o.pos[2]) * fz < -150.0
}

impl Objective for BowserGrabTail {
    fn fields(&self) -> &[&'static str] {
        &["hp", "held", "behind", "tail", "facing"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(o) = bowser_of(st, &self.bhv) else { return };
        let held = holding_bowser(st);
        cx.put(0, o.health);
        cx.put(1, held as i32);
        cx.put(2, behind_bowser(st, o) as i32);
        cx.put(3, cb(dist(st.pos, bowser_tail(st, o)), self.tail_step, 12));
        cx.put(4, if held { -1 } else { (yaw_diff(st.yaw as i32, o.yaw).abs() < 0x5000) as i32 });
    }
}

/// `Bowser::ThrowAtBomb`: (grab him,) spin him with the stick and throw him
/// into a bomb at the arena's edge (bowser.inc.c). Gate as `BowserGrabTail`.
///
/// Fields: `hp`; `mode` (0 free, 1 holding him, 2 thrown: his action is the
/// thrown one); `spin` (holding: Mario's yaw rate `angle_vel[1]` in
/// `spin_step` buckets, clamped -4..=4; else 0, since only the full spin gives
/// the 2.5x throw); `aim` (holding: Mario's yaw within 0x1400 of a bomb, else
/// 0); `bomb` (thrown: distance from Bowser to the nearest `bhvBowserBomb`
/// in `bomb_step` buckets, 0..=8, else -1); `tail` (free: distance to the tail
/// anchor in `tail_step` buckets, else -1).
///
/// ```json
/// {"kind": "BowserThrowAtBomb", "params": {"spin_step": 0x400}}
/// ```
#[derive(Deserialize)]
pub struct BowserThrowAtBomb {
    #[serde(default = "bowser")]
    pub bhv: Bhvs,
    #[serde(default = "spin_step")]
    pub spin_step: f32,
    #[serde(default = "bomb_step")]
    pub bomb_step: f32,
    #[serde(default = "tail_step")]
    pub tail_step: f32,
}

fn spin_step() -> f32 {
    1024.0
}
fn bomb_step() -> f32 {
    500.0
}

impl Objective for BowserThrowAtBomb {
    fn fields(&self) -> &[&'static str] {
        &["hp", "mode", "spin", "aim", "bomb", "tail"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(o) = bowser_of(st, &self.bhv) else { return };
        let bomb = bhv_addr("bhvBowserBomb");
        let bombs = || objs(st).iter().filter(move |b| Some(b.bhv) == bomb);
        cx.put(0, o.health);
        if holding_bowser(st) {
            cx.put(1, 1);
            cx.put(2, bucket(st.angle_vel[1] as f32, self.spin_step).clamp(-4, 4));
            cx.put(3, bombs().any(|b| yaw_diff(yaw_between(st.pos, b.pos), st.yaw as i32).abs() < 0x1400) as i32);
            cx.put(4, -1);
            cx.put(5, -1);
        } else if o.action == BOWSER_THROWN {
            cx.put(1, 2);
            cx.put(2, 0);
            cx.put(3, 0);
            let d = bombs().map(|b| dist(b.pos, o.pos)).fold(f32::MAX, f32::min);
            cx.put(4, if d == f32::MAX { 8 } else { cb(d, self.bomb_step, 8) });
            cx.put(5, -1);
        } else {
            cx.put(1, 0);
            cx.put(2, 0);
            cx.put(3, 0);
            cx.put(4, -1);
            cx.put(5, cb(dist(st.pos, bowser_tail(st, o)), self.tail_step, 12));
        }
    }
}

fn mk<T: Objective + serde::de::DeserializeOwned + 'static>(p: &Value) -> Option<anyhow::Result<Box<dyn Objective>>> {
    Some(serde_json::from_value::<T>(p.clone()).map(|t| Box::new(t) as Box<dyn Objective>).map_err(Into::into))
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<anyhow::Result<Box<dyn Objective>>> {
    match kind {
        "BossHit" => mk::<BossHit>(params),
        "WhompKingPound" => mk::<WhompKingPound>(params),
        "BigBooHit" => mk::<BigBooHit>(params),
        "WigglerStomp" => mk::<WigglerStomp>(params),
        "EyerokPunchEye" => mk::<EyerokPunchEye>(params),
        "EyerokWake" => mk::<EyerokWake>(params),
        "BigBullyPushOff" => mk::<BigBullyPushOff>(params),
        "BowserGrabTail" => mk::<BowserGrabTail>(params),
        "BowserThrowAtBomb" => mk::<BowserThrowAtBomb>(params),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn schema(kind: &str, params: &str) -> String {
        format!(r#"{{"kind":"{kind}","params":{params}}}"#)
    }
    fn k(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&schema(kind, params), 0, s)
    }
    fn at(pos: [f32; 3], objects: Vec<Obj>) -> St {
        with_objects(st(pos, 0.0, 9, 1, 0), objects)
    }

    #[test]
    fn silent_without_boss_or_when_star_owned() {
        for kind in ["BossHit", "WhompKingPound", "BigBooHit", "WigglerStomp", "EyerokPunchEye", "EyerokWake", "BigBullyPushOff", "BowserGrabTail", "BowserThrowAtBomb"] {
            assert!(k(kind, "{}", &at([0.0; 3], vec![])).is_empty(), "{kind}");
        }
        let mut king = obj("bhvKingBobomb", 1, [0.0, 0.0, 500.0]);
        king.health = 1;
        let mut s = at([0.0; 3], vec![king]);
        assert!(!k("BossHit", "{}", &s).is_empty());
        s.course_stars[0] = 1; // course 1, star 0
        assert!(k("BossHit", "{}", &s).is_empty());
    }

    #[test]
    fn boss_hit_stages() {
        let mut king = obj("bhvKingBobomb", 1, [0.0, 0.0, 500.0]);
        king.health = 2;
        let alive = k("BossHit", "{}", &at([0.0; 3], vec![king.clone()]));
        assert_eq!(alive, vec![(0, 0), (1, 2), (2, 0), (3, 4), (4, 2)]); // Mario is behind the king (he faces +z, Mario is at -z of him)
        // One more hit left is another cell; holding is another.
        king.health = 1;
        assert_ne!(alive, k("BossHit", "{}", &at([0.0; 3], vec![king.clone()])));
        let mut s = at([0.0; 3], vec![king.clone()]);
        s.held_slot = Some(1);
        assert_eq!(k("BossHit", "{}", &s)[2], (2, 1));
        // Beaten.
        king.health = 0;
        king.action = 7;
        let beaten = k("BossHit", "{}", &at([0.0; 3], vec![king.clone()]));
        assert_eq!(beaten, vec![(0, 1), (1, 0), (2, 0), (3, -1), (4, -1)]);
        // Done needs the spawned star; its distance is keyed.
        king.action = 8;
        assert!(k("BossHit", "{}", &at([0.0; 3], vec![king.clone()])).is_empty());
        let mut star = obj("bhvStarSpawnCoordinates", 2, [0.0; 3]);
        star.home = [0.0, 0.0, 600.0];
        let done = k("BossHit", "{}", &at([0.0; 3], vec![king.clone(), star.clone()]));
        assert_eq!(done, vec![(0, 2), (1, 0), (2, 0), (3, -1), (4, 2)]);
        let near = k("BossHit", "{}", &at([0.0, 0.0, 400.0], vec![king, star]));
        assert_eq!(near[4], (4, 0));
    }

    #[test]
    fn boss_hit_side_is_relative_to_facing() {
        let mut king = obj("bhvKingBobomb", 1, [0.0, 0.0, 0.0]);
        king.health = 1;
        // King faces +z (yaw 0): Mario at +z is in front (sector 0), at -z behind (4).
        assert_eq!(k("BossHit", "{}", &at([0.0, 0.0, 300.0], vec![king.clone()]))[3], (3, 0));
        assert_eq!(k("BossHit", "{}", &at([0.0, 0.0, -300.0], vec![king.clone()]))[3], (3, 4));
        assert_eq!(k("BossHit", "{}", &at([300.0, 0.0, 0.0], vec![king]))[3], (3, 2));
    }

    #[test]
    fn nearest_boss_is_chosen() {
        let mut a = obj("bhvKingBobomb", 1, [0.0, 0.0, 2000.0]);
        a.health = 1;
        let mut b = obj("bhvKingBobomb", 2, [0.0, 0.0, 100.0]);
        b.health = 2;
        let s = at([0.0; 3], vec![a, b]);
        assert_eq!(k("BossHit", "{}", &s)[1], (1, 2));
    }

    #[test]
    fn whomp_phases_and_back() {
        let mut w = obj("bhvWhompKingBoss", 1, [0.0, 0.0, 0.0]);
        w.health = 3;
        assert!(k("WhompKingPound", "{}", &at([0.0, 0.0, 300.0], vec![w.clone()])).len() == 5);
        w.action = WHOMP_DIE;
        assert!(k("WhompKingPound", "{}", &at([0.0, 0.0, 300.0], vec![w.clone()])).is_empty());
        w.action = WHOMP_ON_GROUND;
        // His back is 480 ahead and 200 up; standing on it vs far away.
        let on = k("WhompKingPound", "{}", &at([0.0, 250.0, 480.0], vec![w.clone()]));
        assert_eq!(on, vec![(0, 3), (1, 3), (2, 0), (3, 1), (4, 0)]);
        let mut pounding = at([0.0, 250.0, 480.0], vec![w.clone()]);
        pounding.action = ACT_GROUND_POUND;
        assert_eq!(k("WhompKingPound", "{}", &pounding)[4], (4, 1));
        let far = k("WhompKingPound", "{}", &at([0.0, 0.0, -500.0], vec![w.clone()]));
        assert_eq!(far[3], (3, 0));
        assert_ne!(far[2], on[2]);
        w.action = WHOMP_JUMP;
        assert_eq!(k("WhompKingPound", "{}", &at([0.0, 0.0, 300.0], vec![w]))[1], (1, 2));
    }

    #[test]
    fn big_boo_gate_and_sides() {
        let mut b = obj("bhvBalconyBigBoo", 1, [0.0; 3]);
        b.health = 3;
        b.interact_type = 1;
        let front = k("BigBooHit", "{}", &at([0.0, 0.0, 200.0], vec![b.clone()]));
        assert_eq!(front, vec![(0, 3), (1, 0), (2, 1), (3, 0), (4, 0)]);
        assert_eq!(k("BigBooHit", "{}", &at([0.0, 0.0, -200.0], vec![b.clone()]))[4], (4, 4));
        b.action = BIG_BOO_HURT;
        b.health = 1;
        assert!(k("BigBooHit", "{}", &at([0.0; 3], vec![b.clone()])).is_empty());
        b.health = 2;
        assert_eq!(k("BigBooHit", "{}", &at([0.0; 3], vec![b.clone()]))[1], (1, 1));
        b.action = BIG_BOO_DEAD;
        assert!(k("BigBooHit", "{}", &at([0.0; 3], vec![b])).is_empty());
    }

    #[test]
    fn wiggler_stomp() {
        let mut w = obj("bhvWigglerHead", 1, [0.0; 3]);
        w.health = 4;
        w.action = 1;
        assert!(k("WigglerStomp", "{}", &at([0.0; 3], vec![obj("bhvWigglerHead", 1, [0.0; 3])])).is_empty()); // hp 0
        let low = k("WigglerStomp", "{}", &at([0.0, 0.0, 300.0], vec![w.clone()]));
        let high = k("WigglerStomp", "{}", &at([0.0, 250.0, 300.0], vec![w.clone()]));
        assert_eq!(low[3], (3, 0));
        assert_eq!(high[3], (3, 2));
        w.action = WIGGLER_JUMPED_ON;
        assert_eq!(k("WigglerStomp", "{}", &at([0.0, 0.0, 300.0], vec![w.clone()]))[1], (1, 1));
        w.action = WIGGLER_SHRINK;
        assert!(k("WigglerStomp", "{}", &at([0.0; 3], vec![w])).is_empty());
    }

    #[test]
    fn eyerok_hand_and_wake() {
        let mut h = obj("bhvEyerokHand", 1, [0.0; 3]);
        h.health = 4;
        h.action = 1;
        let closed = k("EyerokPunchEye", "{}", &at([0.0, 0.0, 200.0], vec![h.clone()]));
        assert_eq!(closed, vec![(0, 4), (1, 0), (2, 1), (3, 0)]);
        h.action = EYEROK_HAND_SHOW_EYE;
        assert_eq!(k("EyerokPunchEye", "{}", &at([0.0, 0.0, 200.0], vec![h.clone()]))[1], (1, 1));
        assert_eq!(k("EyerokPunchEye", "{}", &at([0.0, 0.0, -200.0], vec![h.clone()]))[2], (2, 0));
        h.action = EYEROK_HAND_DIE;
        assert!(k("EyerokPunchEye", "{}", &at([0.0; 3], vec![h.clone()])).is_empty());
        // Star (8, 3) owned: silent.
        h.action = 1;
        let mut s = at([0.0; 3], vec![h]);
        s.course_stars[7] = 1 << 3;
        assert!(k("EyerokPunchEye", "{}", &s).is_empty());

        let mut boss = obj("bhvEyerokBoss", 2, [0.0; 3]);
        let asleep_near = k("EyerokWake", "{}", &at([0.0, 0.0, 400.0], vec![boss.clone()]));
        let asleep_far = k("EyerokWake", "{}", &at([0.0, 0.0, 900.0], vec![boss.clone()]));
        assert_eq!(asleep_near, vec![(0, 0), (1, 1)]);
        assert_ne!(asleep_near, asleep_far);
        boss.action = 1;
        let awake_a = k("EyerokWake", "{}", &at([0.0, 0.0, 400.0], vec![boss.clone()]));
        let awake_b = k("EyerokWake", "{}", &at([0.0, 0.0, 900.0], vec![boss]));
        assert_eq!(awake_a, vec![(0, 1), (1, -1)]);
        assert_eq!(awake_a, awake_b);
    }

    #[test]
    fn bully_push() {
        let mut b = obj("bhvBigBully", 1, [0.0, 0.0, 0.0]);
        b.action = 1;
        let home = k("BigBullyPushOff", "{}", &at([0.0, 0.0, 300.0], vec![b.clone()]));
        assert_eq!(home[0], (0, 0));
        assert_eq!(home[1], (1, 0));
        b.pos = [0.0, 0.0, 700.0]; // pushed 700 from home (pos == home at spawn above)
        b.home = [0.0, 0.0, 0.0];
        let pushed = k("BigBullyPushOff", "{}", &at([0.0, 0.0, 300.0], vec![b.clone()]));
        assert_eq!(pushed[1], (1, 2));
        // Away from home is +z; Mario at z < pos.z - 50 is behind it, at z > is not.
        assert_eq!(pushed[3], (3, 1));
        assert_eq!(k("BigBullyPushOff", "{}", &at([0.0, 0.0, 1000.0], vec![b.clone()]))[3], (3, 0));
        b.action = BULLY_KNOCKBACK;
        assert_eq!(k("BigBullyPushOff", "{}", &at([0.0, 0.0, 300.0], vec![b.clone()]))[0], (0, 1));
        b.action = BULLY_LAVA_DEATH;
        assert_eq!(k("BigBullyPushOff", "{}", &at([0.0, 0.0, 300.0], vec![b.clone()]))[0], (0, 2));
        b.action = BULLY_INACTIVE;
        assert!(k("BigBullyPushOff", "{}", &at([0.0, 0.0, 300.0], vec![b])).is_empty());
    }

    fn bowser_state(pos: [f32; 3], objects: Vec<Obj>, level: u16, keys: u8) -> St {
        with_objects(st(pos, 0.0, level, 1, keys), objects)
    }

    #[test]
    fn bowser_gate_and_grab() {
        let mut b = obj("bhvBowser", 1, [0.0; 3]);
        b.health = 1;
        // Not a Bowser level: silent. Key already had: silent. Dead: silent.
        assert!(k("BowserGrabTail", "{}", &bowser_state([0.0; 3], vec![b.clone()], 9, 0)).is_empty());
        assert!(k("BowserGrabTail", "{}", &bowser_state([0.0; 3], vec![b.clone()], 30, 1)).is_empty());
        assert!(!k("BowserGrabTail", "{}", &bowser_state([0.0; 3], vec![b.clone()], 34, 3)).is_empty());
        b.action = BOWSER_DEAD;
        assert!(k("BowserGrabTail", "{}", &bowser_state([0.0; 3], vec![b.clone()], 34, 0)).is_empty());
        b.action = 0;
        // No tail anchor: the tail is 300 behind him (-z). Mario there, behind, facing him (yaw 0).
        let at_tail = k("BowserGrabTail", "{}", &bowser_state([0.0, 0.0, -300.0], vec![b.clone()], 30, 0));
        assert_eq!(at_tail, vec![(0, 1), (1, 0), (2, 1), (3, 0), (4, 1)]);
        // In front of him: not behind, farther from the tail.
        let front = k("BowserGrabTail", "{}", &bowser_state([0.0, 0.0, 500.0], vec![b.clone()], 30, 0));
        assert_eq!(front[2], (2, 0));
        assert!(front[3].1 > 0);
        // Holding him is another cell and drops the facing.
        let mut held = bowser_state([0.0, 0.0, -300.0], vec![b], 30, 0);
        held.action = ACT_HOLDING_BOWSER;
        let h = k("BowserGrabTail", "{}", &held);
        assert_eq!((h[1], h[4]), ((1, 1), (4, -1)));
    }

    #[test]
    fn bowser_throw() {
        let mut b = obj("bhvBowser", 1, [0.0; 3]);
        b.health = 3;
        let bomb = obj("bhvBowserBomb", 2, [0.0, 0.0, 1500.0]);
        let free = k("BowserThrowAtBomb", "{}", &bowser_state([0.0, 0.0, -300.0], vec![b.clone(), bomb.clone()], 34, 0));
        assert_eq!(free[1], (1, 0));
        assert_eq!(free[5], (5, 0));
        // Holding: spin bucketed, aim true when facing the bomb (yaw 0 = +z).
        let mut s = bowser_state([0.0; 3], vec![b.clone(), bomb.clone()], 34, 0);
        s.action = ACT_HOLDING_BOWSER;
        s.angle_vel[1] = 0x1000;
        let a = k("BowserThrowAtBomb", "{}", &s);
        assert_eq!((a[1], a[2], a[3]), ((1, 1), (2, 4), (3, 1)));
        s.angle_vel[1] = 0x500;
        assert_eq!(k("BowserThrowAtBomb", "{}", &s)[2], (2, 1));
        s.yaw = 0x4000;
        assert_eq!(k("BowserThrowAtBomb", "{}", &s)[3], (3, 0));
        // Thrown: distance of Bowser to the bomb.
        b.action = BOWSER_THROWN;
        b.pos = [0.0, 0.0, 700.0];
        let thrown = k("BowserThrowAtBomb", "{}", &bowser_state([0.0, 0.0, -300.0], vec![b.clone(), bomb.clone()], 34, 0));
        assert_eq!((thrown[1], thrown[4]), ((1, 2), (4, 1)));
        b.pos = [0.0, 0.0, 1400.0];
        let closer = k("BowserThrowAtBomb", "{}", &bowser_state([0.0, 0.0, -300.0], vec![b, bomb], 34, 0));
        assert_eq!(closer[4], (4, 0));
    }

    #[test]
    fn build_knows_its_kinds() {
        assert!(build("BossHit", &serde_json::json!({})).is_some());
        assert!(build("Nope", &serde_json::json!({})).is_none());
        assert!(build("BigBooHit", &serde_json::json!({"bhv": "bhvBalconyBigBoo"})).unwrap().is_ok());
    }
}
