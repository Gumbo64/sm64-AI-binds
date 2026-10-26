//! Bosses and big fights: Whomp King (ground pound his back), Big Boo, Wiggler, Eyerok, the Big Bullies, Big Chill Bully, Bowser (grab tail, spin, throw into bombs), King Bob-omb is already covered; every boss's sequence of hits as objectives.
//!
//! Generated objectives for this category; see `extra/mod.rs`.
//!
//! What each boss needs, from the decomp (sm64-port):
//! - `WhompKing::Pound` (bhvWhompKingBoss): `oHealth` 3, lowered only by
//!   `ACT_GROUND_POUND_LAND` on his back while he lies flat (whomp.inc.c,
//!   `king_whomp_on_ground`, action 6). He falls forward when Mario is within
//!   300 in front of him (`king_whomp_chase` -> action 3).
//! - `BigBoo::Hit` (bhvGhostHuntBigBoo, bhvBalconyBigBoo,
//!   bhvMerryGoRoundBigBoo): `oHealth` 3, lowered by any attack that isn't
//!   from above (punch, kick, dive; boo.inc.c `boo_get_attack_status`,
//!   `big_boo_act_3`). It is only tangible while visible, and it vanishes when
//!   it faces Mario while Mario faces it (`boo_vanish_or_appear`), so it is hit
//!   from the side.
//! - `Wiggler::Stomp` (bhvWigglerHead): `oHealth` 4, each stomp/ground pound
//!   enters `WIGGLER_ACT_JUMPED_ON` (3); health drops after the dialog and at 1
//!   he shrinks and drops the star (wiggler.inc.c).
//! - `Eyerok::PunchEye` (bhvEyerokHand): each hand has `oHealth` 4 and is hurt
//!   only while its eye is open (`EYEROK_HAND_ACT_SHOW_EYE`, 3) and Mario is in
//!   front of it (eyerok.inc.c `eyerok_hand_check_attacked`); 3 hits each.
//!   `Eyerok::Wake`: the boss wakes when Mario is within 500 of it.
//! - `BigBully::PushOff` (bhvBigBully, bhvBigBullyWithMinions,
//!   bhvBigChillBully): bullies have no health at all (their hitbox sets
//!   `oHealth` 0, bully.inc.c); defeat is falling into lava / below y 1030
//!   (`BULLY_ACT_LAVA_DEATH` 100) or off the level (`BULLY_ACT_DEATH_PLANE_DEATH`
//!   101), so the generic `Boss::Hit` (needs hp > 0) never applies to them.
//! - `Bowser::GrabTail` / `Bowser::ThrowAtBomb` (bhvBowser): touch the tail
//!   anchor so Bowser becomes tangible, punch (B) facing his way
//!   (interaction.c `mario_check_object_grab`: Mario's yaw within 0x5555 of
//!   his) -> `ACT_PICKING_UP_BOWSER` -> `ACT_HOLDING_BOWSER`; rotating the
//!   stick spins him (mario_actions_object.c `act_holding_bowser`, angle
//!   velocity up to 0x1000 per frame); B throws him along Mario's yaw at
//!   `angleVel / 3000 * 70` units/frame, times 2.5 above 90
//!   (bowser.inc.c `bowser_thrown_dropped_update`). While thrown, any
//!   bhvBowserBomb within 800 explodes and takes one hp (`bowser_act_thrown`,
//!   `bowser_check_hit_mine`). hp is 1, 1, 3 (`sBowserHealth`).
//!
//! Stars and keys these bosses drop are `INTERACT_STAR_OR_KEY` (the Bowser key
//! once it lands, bowser_key.inc.c; the grand star) and so already
//! `Star::Collect`. A boss's star index is its first behaviour byte
//! (`spawn_default_star` -> `spawn_star` copies the spawner's params).

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, B_BUTTON, Z_TRIG};
use obrrt_core::layout::{ACT_FLAG_AIR, LEVEL_BOWSER_1, LEVEL_BOWSER_2, LEVEL_BOWSER_3};
use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St, dist};

// Mario actions (include/sm64.h).
const ACT_PICKING_UP_BOWSER: u32 = 0x0000_0390;
const ACT_HOLDING_BOWSER: u32 = 0x0000_0391;
const ACT_GROUND_POUND: u32 = 0x0080_08A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080_023C;

// Whomp actions (whomp.inc.c, `sWhompActions`).
const WHOMP_INIT: i32 = 0;
const WHOMP_PREPARE_JUMP: i32 = 3;
const WHOMP_JUMP: i32 = 4;
const WHOMP_LAND: i32 = 5;
const WHOMP_ON_GROUND: i32 = 6;
const WHOMP_DIE: i32 = 8;

// Big Boo actions (boo.inc.c, `sBooGivingStarActions`).
const BIG_BOO_INACTIVE: i32 = 0;
const BIG_BOO_HURT: i32 = 3;
const BIG_BOO_DEAD: i32 = 4;

// include/object_constants.h.
const WIGGLER_ACT_UNINITIALIZED: i32 = 0;
const WIGGLER_ACT_JUMPED_ON: i32 = 3;
const WIGGLER_ACT_SHRINK: i32 = 4;
const WIGGLER_ACT_FALL_THROUGH_FLOOR: i32 = 5;
const EYEROK_BOSS_ACT_SLEEP: i32 = 0;
const EYEROK_HAND_ACT_SLEEP: i32 = 0;
const EYEROK_HAND_ACT_SHOW_EYE: i32 = 3;
const EYEROK_HAND_ACT_DIE: i32 = 15;
const BULLY_ACT_KNOCKBACK: i32 = 2;
const BULLY_ACT_INACTIVE: i32 = 4;
const BULLY_ACT_ACTIVATE_AND_FALL: i32 = 5;
const BULLY_ACT_LAVA_DEATH: i32 = 100;
const BULLY_ACT_DEATH_PLANE_DEATH: i32 = 101;
const BOWSER_ACT_THROWN: i32 = 1;
const BOWSER_ACT_DEAD: i32 = 4;

/// Objectives `o` offers, seen from base state `b` (called for every
/// object within range of `b`; `b` is never a dialog or menu state).
pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let star = star_of(b, o);
    if star.is_some_and(|(c, i)| b.has_star(c, i)) {
        return;
    }
    match o.bhv_name() {
        "bhvWhompKingBoss" if (1..=3).contains(&o.health) && o.action < WHOMP_DIE => {
            out.push(Arc::new(WhompPound { f: Fight::new(b, o, star) }))
        }
        "bhvGhostHuntBigBoo" | "bhvBalconyBigBoo" | "bhvMerryGoRoundBigBoo"
            if o.action != BIG_BOO_DEAD && !(o.action == BIG_BOO_HURT && o.health <= 1) =>
        {
            out.push(Arc::new(BigBooHit { f: Fight::new(b, o, star) }))
        }
        "bhvWigglerHead" if wiggler_fighting(o) => out.push(Arc::new(WigglerStomp { f: Fight::new(b, o, star) })),
        "bhvEyerokHand" if eyerok_hand_fighting(o) => {
            // The hands don't carry the star's params; it's the boss's 0x03
            // (levels/ssl/script.c: bhvEyerokBoss behParam 0x03000000).
            let star = (b.course == 8).then_some((8, 3));
            if !star.is_some_and(|(c, i)| b.has_star(c, i)) {
                out.push(Arc::new(EyerokPunchEye { f: Fight::new(b, o, star) }))
            }
        }
        "bhvEyerokBoss" if o.action == EYEROK_BOSS_ACT_SLEEP => out.push(Arc::new(EyerokWake { f: Fight::new(b, o, star) })),
        "bhvBigBully" | "bhvBigBullyWithMinions" | "bhvBigChillBully" if bully_fighting(o) => {
            out.push(Arc::new(BullyPushOff { f: Fight::new(b, o, star) }))
        }
        "bhvBowser" if bowser_fighting(b, o) => {
            out.push(Arc::new(BowserGrabTail { f: Fight::new(b, o, None) }));
            out.push(Arc::new(BowserThrow { f: Fight::new(b, o, None) }));
        }
        _ => {}
    }
}

/// Objectives that come from the level rather than one object (called once
/// per `b`).
pub fn generate_level(_b: &St, _out: &mut Vec<Arc<dyn Objective>>) {}

// ---------------------------------------------------------------------------
// Shared helpers.

/// The star a boss gives: its first behaviour byte in its course.
fn star_of(b: &St, o: &Obj) -> Option<(i16, u8)> {
    let index = (o.beh_params >> 24) as u8;
    ((1..=15).contains(&b.course) && index <= 5).then_some((b.course, index))
}

fn wiggler_fighting(o: &Obj) -> bool {
    !matches!(o.action, WIGGLER_ACT_UNINITIALIZED | WIGGLER_ACT_SHRINK | WIGGLER_ACT_FALL_THROUGH_FLOOR) && (2..=4).contains(&o.health)
}

fn eyerok_hand_fighting(o: &Obj) -> bool {
    !matches!(o.action, EYEROK_HAND_ACT_SLEEP | EYEROK_HAND_ACT_DIE) && (2..=4).contains(&o.health)
}

fn bully_fighting(o: &Obj) -> bool {
    !matches!(o.action, BULLY_ACT_INACTIVE | BULLY_ACT_ACTIVATE_AND_FALL | BULLY_ACT_LAVA_DEATH | BULLY_ACT_DEATH_PLANE_DEATH)
}

/// Bowser alive and his key not yet had (the first key is BITDW's, the
/// second BITFS's; `St::keys` counts keys held or used).
fn bowser_fighting(s: &St, o: &Obj) -> bool {
    let key_done = match s.level as u32 {
        LEVEL_BOWSER_1 => s.keys >= 1,
        LEVEL_BOWSER_2 => s.keys >= 2,
        LEVEL_BOWSER_3 => false,
        _ => return false,
    };
    !key_done && o.health > 0 && o.action != BOWSER_ACT_DEAD
}

/// One boss instance in one zone.
struct Fight {
    id: ObjId,
    zone: (u16, u8),
    star: Option<(i16, u8)>,
}

impl Fight {
    fn new(b: &St, o: &Obj, star: Option<(i16, u8)>) -> Fight {
        Fight { id: o.id, zone: b.zone(), star }
    }

    fn get<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        s.find(self.id)
    }

    /// In the zone, the boss loaded, Mario alive, the star not had.
    fn base<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone || !s.alive() || self.star.is_some_and(|(c, i)| s.has_star(c, i)) {
            return None;
        }
        self.get(s)
    }

    fn key(&self, kind: &str) -> String {
        format!("{kind} {:?} {}", self.zone, self.id)
    }

    /// The first index where the boss's health is below traj[0]'s, or `extra`
    /// holds (per state, so the GA can check it every frame).
    fn hurt_or(&self, traj: &[St], extra: impl Fn(&Obj, &Obj) -> bool) -> Option<usize> {
        let o0 = self.get(&traj[0])?;
        let (hp0, o0) = (o0.health, o0.clone());
        // oHealth is junk (2048) before some bosses initialise it; a drop from
        // there is not a hit.
        let hp_known = (1..=4).contains(&hp0);
        traj.iter().position(|s| s.zone() == self.zone && self.get(s).is_some_and(|o| (hp_known && o.health < hp0) || extra(&o0, o)))
    }
}

/// The best (index, score) of a per-state score.
fn best(traj: &[St], f: impl Fn(usize, &St) -> Option<f64>) -> Progress {
    let (at, score) = traj
        .iter()
        .enumerate()
        .filter_map(|(i, s)| f(i, s).map(|v| (i, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f64::MIN));
    Progress { score, at }
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
    // `right` is along the perpendicular (cos, -sin); which side it is on
    // screen doesn't matter to its users (dodging, going around).
    [p[0] + forward * sx + right * cz, p[1] + up, p[2] + forward * cz - right * sx]
}

fn in_air(s: &St) -> bool {
    s.action & ACT_FLAG_AIR != 0
}

macro_rules! common {
    () => {
        fn needs_objects(&self) -> bool {
            true
        }
        fn kinship(&self) -> u32 {
            10
        }
    };
}

// ---------------------------------------------------------------------------
// Whomp King.

/// `WhompKing::Pound(id)`: lure him into falling flat (stand in front of
/// him), dodge sideways, get on his back and ground pound it.
struct WhompPound {
    f: Fight,
}

impl WhompPound {
    /// Centre of his back while he lies flat: his collision box (actors/whomp/
    /// collision.inc.c, y 50..430, z -100..-3) scaled 2x and pitched forward.
    fn back(o: &Obj) -> [f32; 3] {
        offset(o.pos, o.yaw, 480.0, 0.0, 200.0)
    }

    fn lying(o: &Obj) -> bool {
        matches!(o.action, WHOMP_LAND | WHOMP_ON_GROUND)
    }
}

impl Objective for WhompPound {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "WhompKing::Pound"
    }
    fn key(&self) -> String {
        self.f.key("WhompKingPound")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // With whomphits on, Boss::WhompFast takes the fight from his dialog on.
        self.f.base(s).is_some_and(|o| (1..=3).contains(&o.health) && o.action < WHOMP_DIE && (o.action == WHOMP_INIT || !super::tech::enabled("whomphits")))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = self.f.get(s) else { return f64::MAX };
        // Fewer hits left, then him already lying flat (Mario near his back),
        // then the fight started, then distance.
        let phase = if Self::lying(o) {
            s.dist(Self::back(o)) as f64 / 400.0
        } else if o.action == WHOMP_INIT {
            8.0 + s.dist(o.pos) as f64 / 400.0
        } else {
            4.0 + s.dist(o.pos) as f64 / 400.0
        };
        o.health as f64 * 10.0 + phase
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        self.f.hurt_or(traj, |_, o| o.action >= WHOMP_DIE)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |_, s| {
            let o = self.f.get(s)?;
            Some(if Self::lying(o) {
                let back = Self::back(o);
                let on_top = s.pos[1] > back[1] - 30.0 && s.hdist(back) < 350.0;
                let pound = matches!(s.action, ACT_GROUND_POUND | ACT_GROUND_POUND_LAND);
                1000.0 - s.hdist(back) as f64 + if on_top { 500.0 } else { 0.0 } + if on_top && pound { 500.0 } else { 0.0 }
            } else if matches!(o.action, WHOMP_PREPARE_JUMP | WHOMP_JUMP) {
                // He's falling: being out of the way matters.
                500.0
            } else {
                -(s.dist(offset(o.pos, o.yaw, 250.0, 0.0, 0.0)) as f64)
            })
        })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 300 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.f.get(s)?;
        Some(if Self::lying(o) {
            Self::back(o)
        } else if matches!(o.action, WHOMP_PREPARE_JUMP | WHOMP_JUMP) {
            // Out of the way of his fall (he covers 100..860 ahead, +-300 wide).
            offset(o.pos, o.yaw, 400.0, 550.0, 0.0)
        } else {
            // In front of him, within 300: he jumps to slam.
            offset(o.pos, o.yaw, 250.0, 0.0, 0.0)
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        let Some(o) = self.f.get(s) else { return 0 };
        if !Self::lying(o) {
            return 0;
        }
        let back = Self::back(o);
        if s.hdist(back) > 380.0 {
            return 0;
        }
        if in_air(s) {
            // Above his back: ground pound.
            if s.pos[1] > back[1] + 20.0 { Z_TRIG } else { 0 }
        } else {
            // Next to or on him: jump (onto him, or up to pound).
            A_BUTTON
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    common!();
}

// ---------------------------------------------------------------------------
// Big Boo.

/// `BigBoo::Hit(id)`: punch or kick a Big Boo from its side while it's
/// visible.
struct BigBooHit {
    f: Fight,
}

impl BigBooHit {
    fn hits_left(o: &Obj) -> f64 {
        // Before activation oHealth isn't set yet (big_boo_act_0 sets 3).
        if (1..=3).contains(&o.health) { o.health as f64 } else { 3.0 }
    }
}

impl Objective for BigBooHit {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "BigBoo::Hit"
    }
    fn key(&self) -> String {
        self.f.key("BigBooHit")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(|o| o.action != BIG_BOO_DEAD && !(o.action == BIG_BOO_HURT && o.health <= 1))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = self.f.get(s) else { return f64::MAX };
        Self::hits_left(o) * 10.0 + if o.action == BIG_BOO_INACTIVE { 5.0 } else { 0.0 } + s.dist(o.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        self.f.hurt_or(traj, |o0, o| (o.action == BIG_BOO_HURT && o0.action != BIG_BOO_HURT) || o.action == BIG_BOO_DEAD)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |_, s| {
            let o = self.f.get(s)?;
            // Close to it while it's tangible (visible) is what counts.
            let visible = if o.interact_type != 0 { 300.0 } else { 0.0 };
            Some(visible - s.dist(o.pos) as f64)
        })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.f.get(s)?;
        // Come in from its side (it vanishes if Mario faces it head-on), then
        // straight at it.
        if s.hdist(o.pos) > 260.0 { Some(offset(o.pos, o.yaw, 0.0, 220.0, 0.0)) } else { Some(o.pos) }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match self.f.get(s) {
            Some(o) if s.hdist(o.pos) < 230.0 && o.interact_type != 0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    common!();
}

// ---------------------------------------------------------------------------
// Wiggler.

/// `Wiggler::Stomp(id)`: jump on Wiggler's head (3 times in all).
struct WigglerStomp {
    f: Fight,
}

impl Objective for WigglerStomp {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Wiggler::Stomp"
    }
    fn key(&self) -> String {
        self.f.key("WigglerStomp")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(wiggler_fighting)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = self.f.get(s) else { return f64::MAX };
        // Mid-squish (dialog pending) counts as one hit fewer.
        let hits = o.health as f64 - if o.action == WIGGLER_ACT_JUMPED_ON { 1.0 } else { 0.0 };
        hits * 10.0 + s.dist(o.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        self.f.hurt_or(traj, |o0, o| o.action == WIGGLER_ACT_JUMPED_ON && o0.action != WIGGLER_ACT_JUMPED_ON)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |_, s| {
            let o = self.f.get(s)?;
            let above = if s.pos[1] > o.pos[1] + 50.0 { 200.0 } else { 0.0 };
            Some(above - s.hdist(o.pos) as f64)
        })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 6, horizon: 200 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Above his head: the default steering jumps toward it.
        self.f.get(s).map(|o| [o.pos[0], o.pos[1] + 200.0, o.pos[2]])
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    common!();
}

// ---------------------------------------------------------------------------
// Eyerok.

/// `Eyerok::PunchEye(hand)`: punch an Eyerok hand's open eye from in front.
struct EyerokPunchEye {
    f: Fight,
}

impl Objective for EyerokPunchEye {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Eyerok::PunchEye"
    }
    fn key(&self) -> String {
        self.f.key("EyerokPunchEye")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(eyerok_hand_fighting)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = self.f.get(s) else { return f64::MAX };
        let eye = if o.action == EYEROK_HAND_ACT_SHOW_EYE { 0.0 } else { 3.0 };
        o.health as f64 * 10.0 + eye + s.dist(offset(o.pos, o.yaw, 200.0, 0.0, 0.0)) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        self.f.hurt_or(traj, |o0, o| o.action == EYEROK_HAND_ACT_DIE && o0.action != EYEROK_HAND_ACT_DIE)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |_, s| {
            let o = self.f.get(s)?;
            // In front of it (within 0x3000 of its facing) while the eye shows.
            let front = yaw_diff(yaw_between(o.pos, s.pos), o.yaw).abs() < 0x3000;
            let bonus = if front { 200.0 } else { 0.0 } + if o.action == EYEROK_HAND_ACT_SHOW_EYE && front { 300.0 } else { 0.0 };
            Some(bonus - s.dist(o.pos) as f64)
        })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 300 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.f.get(s)?;
        let front = offset(o.pos, o.yaw, 220.0, 0.0, 0.0);
        if o.action == EYEROK_HAND_ACT_SHOW_EYE && s.hdist(front) < 150.0 { Some(o.pos) } else { Some(front) }
    }
    fn steer_button(&self, s: &St) -> u16 {
        match self.f.get(s) {
            Some(o) if o.action == EYEROK_HAND_ACT_SHOW_EYE && s.hdist(o.pos) < 300.0 => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    common!();
}

/// `Eyerok::Wake(boss)`: come within 500 of the sleeping Eyerok to start the
/// fight (eyerok.inc.c `eyerok_boss_act_sleep`).
struct EyerokWake {
    f: Fight,
}

impl Objective for EyerokWake {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Eyerok::Wake"
    }
    fn key(&self) -> String {
        self.f.key("EyerokWake")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        5.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(|o| o.action == EYEROK_BOSS_ACT_SLEEP)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.f.get(s).map_or(f64::MAX, |o| s.dist(o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.f.zone && self.f.get(s).is_some_and(|o| o.action != EYEROK_BOSS_ACT_SLEEP))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |_, s| self.f.get(s).map(|o| -(s.dist(o.pos) as f64)))
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.f.get(s).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    common!();
}

// ---------------------------------------------------------------------------
// Big Bullies.

/// `BigBully::PushOff(id)`: knock a Big Bully (or the Big Chill Bully) off
/// its platform into the lava / water. Pushes away from its home.
struct BullyPushOff {
    f: Fight,
}

impl BullyPushOff {
    /// Unit vector (x, z) to push along: away from home, or (near home) away
    /// from Mario.
    fn push_dir(s: &St, o: &Obj) -> (f32, f32) {
        let (mut dx, mut dz) = (o.pos[0] - o.home[0], o.pos[2] - o.home[2]);
        if dx.hypot(dz) < 60.0 {
            (dx, dz) = (o.pos[0] - s.pos[0], o.pos[2] - s.pos[2]);
        }
        let n = dx.hypot(dz).max(1e-3);
        (dx / n, dz / n)
    }

    fn pushed(o: &Obj) -> f64 {
        let h = (o.pos[0] - o.home[0]).hypot(o.pos[2] - o.home[2]) as f64;
        h + (o.home[1] - o.pos[1]).max(0.0) as f64 * 3.0
    }
}

impl Objective for BullyPushOff {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "BigBully::PushOff"
    }
    fn key(&self) -> String {
        self.f.key("BigBullyPushOff")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(bully_fighting)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        let Some(o) = self.f.get(s) else { return f64::MAX };
        // Already pushed far from home first, then close to him.
        (1500.0 - Self::pushed(o)).max(0.0) / 400.0 + s.dist(o.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Lava/death-plane death, seen while it is still in the state (both
        // last at least a frame before it is deleted: obj_lava_death shrinks
        // it first, the death-plane act deactivates it on its next update,
        // bully.inc.c), or its star. A frame without it (objects not read,
        // or not in the list) is not a defeat.
        traj.iter().position(|s| {
            self.f.star.is_some_and(|(c, i)| s.has_star(c, i))
                || (s.zone() == self.f.zone
                    && self.f.get(s).is_some_and(|o| matches!(o.action, BULLY_ACT_LAVA_DEATH | BULLY_ACT_DEATH_PLANE_DEATH)))
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let p0 = self.f.get(&traj[0]).map_or(0.0, Self::pushed);
        best(traj, |_, s| {
            let o = self.f.get(s)?;
            let kb = if o.action == BULLY_ACT_KNOCKBACK { 150.0 } else { 0.0 };
            Some((Self::pushed(o) - p0) * 2.0 + kb - s.dist(o.pos) as f64 * 0.2)
        })
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 300 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.f.get(s)?;
        let (ux, uz) = Self::push_dir(s, o);
        let rel = (s.pos[0] - o.pos[0], s.pos[2] - o.pos[2]);
        let along = rel.0 * ux + rel.1 * uz;
        let side = rel.0 * uz - rel.1 * ux;
        let at = |a: f32, p: f32| [o.pos[0] + ux * a + uz * p, o.pos[1], o.pos[2] + uz * a - ux * p];
        Some(if along > -80.0 {
            // On the wrong side: go around him.
            at(-280.0, if side >= 0.0 { 300.0 } else { -300.0 })
        } else if s.hdist(o.pos) > 280.0 {
            at(-200.0, 0.0)
        } else {
            // Behind him: run through him, toward the edge.
            at(300.0, 0.0)
        })
    }
    fn steer_button(&self, s: &St) -> u16 {
        let Some(o) = self.f.get(s) else { return 0 };
        let (ux, uz) = Self::push_dir(s, o);
        let along = (s.pos[0] - o.pos[0]) * ux + (s.pos[2] - o.pos[2]) * uz;
        // Punch (or dive, at speed) into him from behind.
        if along < -50.0 && s.hdist(o.pos) < 230.0 { B_BUTTON } else { 0 }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
    common!();
}

// ---------------------------------------------------------------------------
// Bowser.

fn holding_bowser(s: &St) -> bool {
    matches!(s.action, ACT_PICKING_UP_BOWSER | ACT_HOLDING_BOWSER)
}

/// Bowser's tail anchor (bhvBowserTailAnchor) nearest him, else a point
/// behind him.
fn bowser_tail(s: &St, o: &Obj) -> [f32; 3] {
    s.objects()
        .iter()
        .filter(|t| t.bhv_name() == "bhvBowserTailAnchor")
        .map(|t| t.pos)
        .filter(|p| dist(*p, o.pos) < 700.0)
        .min_by(|a, b| dist(*a, o.pos).total_cmp(&dist(*b, o.pos)))
        .unwrap_or_else(|| o.behind(300.0))
}

/// Whether Mario is behind Bowser (on his tail's side).
fn behind_bowser(s: &St, o: &Obj) -> bool {
    let (fx, fz) = yaw_to_vec(o.yaw);
    (s.pos[0] - o.pos[0]) * fx + (s.pos[2] - o.pos[2]) * fz < -150.0
}

/// Steering to grab the tail: around to his back, then onto the tail, B.
fn bowser_grab_target(s: &St, o: &Obj) -> [f32; 3] {
    if behind_bowser(s, o) {
        bowser_tail(s, o)
    } else {
        // Swing round his side to a point well behind him.
        let (fx, fz) = yaw_to_vec(o.yaw);
        let side = (s.pos[0] - o.pos[0]) * fz - (s.pos[2] - o.pos[2]) * fx;
        offset(o.pos, o.yaw, -500.0, if side >= 0.0 { 450.0 } else { -450.0 }, 0.0)
    }
}

fn bowser_grab_button(s: &St, o: &Obj) -> u16 {
    let tail = bowser_tail(s, o);
    // Facing his way (mario_check_object_grab: within 0x5555) and on the tail.
    if s.hdist(tail) < 170.0 && yaw_diff(s.yaw as i32, o.yaw).abs() < 0x5000 { B_BUTTON } else { 0 }
}

/// Whether Mario's grab progress score (not holding): closer to the tail,
/// from behind, is better.
fn bowser_grab_score(s: &St, o: &Obj) -> f64 {
    let tail = bowser_tail(s, o);
    (if behind_bowser(s, o) { 400.0 } else { 0.0 }) - s.dist(tail) as f64
}

fn bowser_heuristic(s: &St, o: &Obj) -> f64 {
    o.health as f64 * 10.0 + if holding_bowser(s) { 0.0 } else { 5.0 + s.dist(bowser_tail(s, o)) as f64 / 400.0 }
}

/// `Bowser::GrabTail(id)`: get behind Bowser and grab his tail (B).
struct BowserGrabTail {
    f: Fight,
}

impl Objective for BowserGrabTail {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Bowser::GrabTail"
    }
    fn key(&self) -> String {
        self.f.key("BowserGrabTail")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        20.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        !s.holding && self.f.base(s).is_some_and(|o| bowser_fighting(s, o))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.f.get(s).map_or(f64::MAX, |o| bowser_heuristic(s, o))
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        // Picking him up always ends in holding him (act_picking_up_bowser).
        traj.iter().position(holding_bowser)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        best(traj, |_, s| self.f.get(s).map(|o| bowser_grab_score(s, o)))
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 6, horizon: 240 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.f.get(s).map(|o| bowser_grab_target(s, o))
    }
    fn steer_button(&self, s: &St) -> u16 {
        self.f.get(s).map_or(0, |o| bowser_grab_button(s, o))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(|o| bowser_fighting(s, o))
    }
    common!();
}

/// `Bowser::ThrowAtBomb(id)`: (grab him,) spin him with the stick and throw
/// him into a bomb at the arena's edge; met when his health drops.
struct BowserThrow {
    f: Fight,
}

impl BowserThrow {
    fn bombs(s: &St) -> impl Iterator<Item = &Obj> {
        s.objects().iter().filter(|b| b.bhv_name() == "bhvBowserBomb")
    }

    /// Mario's yaw change per frame: `angleVel[1]`, the spin while swinging
    /// Bowser (act_holding_bowser adds it to faceAngle[1]). Read from the
    /// state, not remembered across calls: the objective is shared by all
    /// worker threads, so a remembered last yaw (the old Mutex estimate)
    /// made steering depend on thread scheduling.
    fn spin_at(&self, s: &St) -> i32 {
        s.angle_vel[1] as i32
    }
}

impl Objective for BowserThrow {
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Bowser::ThrowAtBomb"
    }
    fn key(&self) -> String {
        self.f.key("BowserThrowAtBomb")
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // Holding him, or free to go grab him (then `Bowser::GrabTail` starts
        // are much preferred via the heuristic).
        (holding_bowser(s) || !s.holding) && self.f.base(s).is_some_and(|o| bowser_fighting(s, o))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.f.get(s).map_or(f64::MAX, |o| bowser_heuristic(s, o) + if holding_bowser(s) { 0.0 } else { 15.0 })
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        self.f.hurt_or(traj, |_, o| o.action == BOWSER_ACT_DEAD)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Not holding: approach the tail; holding: spin speed; thrown (after
        // holding): how close he got to a bomb.
        let mut held = false;
        let mut best_p = Progress { score: f64::MIN, at: 0 };
        for (i, s) in traj.iter().enumerate() {
            let Some(o) = self.f.get(s) else { continue };
            let score = if holding_bowser(s) {
                held = true;
                let spin = match i.checked_sub(1).map(|j| &traj[j]) {
                    Some(p) if holding_bowser(p) => yaw_diff(s.yaw as i32, p.yaw as i32).abs(),
                    _ => 0,
                };
                1000.0 + spin as f64 * 0.25
            } else if held && o.action == BOWSER_ACT_THROWN {
                let d = Self::bombs(s).map(|b| dist(b.pos, o.pos)).fold(f32::MAX, f32::min);
                2500.0 + (4000.0 - d.min(4000.0)) as f64
            } else {
                bowser_grab_score(s, o)
            };
            if score > best_p.score {
                best_p = Progress { score, at: i };
            }
        }
        best_p
    }
    fn success_value(&self) -> f64 {
        15.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 24, generations: 6, horizon: 240 }
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.f.get(s)?;
        if holding_bowser(s) {
            // Spin: a stick direction rotating 0x3000 per frame accelerates
            // him by 0x60 per frame (act_holding_bowser), to the 0x1000 cap in
            // ~45 frames.
            let a = (s.frame as i64 * 0x3000 % 0x10000) as i32;
            Some(offset(s.pos, a, 500.0, 0.0, 0.0))
        } else {
            Some(bowser_grab_target(s, o))
        }
    }
    fn steer_button(&self, s: &St) -> u16 {
        let Some(o) = self.f.get(s) else { return 0 };
        let spin = self.spin_at(s);
        if s.action == ACT_HOLDING_BOWSER {
            // Release along Mario's yaw: when facing a bomb, at near-full spin
            // (only angleVel >= ~0xF12 gives the 2.5x throw speed), or close
            // to the bomb already.
            let facing = Self::bombs(s).find(|b| yaw_diff(yaw_between(s.pos, b.pos), s.yaw as i32).abs() < 0x1400);
            match facing {
                Some(b) if spin.abs() >= 0xE00 || s.hdist(b.pos) < 1300.0 => B_BUTTON,
                _ => 0,
            }
        } else if holding_bowser(s) {
            0
        } else {
            bowser_grab_button(s, o)
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        self.f.base(s).is_some_and(|o| bowser_fighting(s, o))
    }
    common!();
}
