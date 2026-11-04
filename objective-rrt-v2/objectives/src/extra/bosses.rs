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
use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, St, dist, frames_for};

// Mario actions (include/sm64.h).
const ACT_PICKING_UP_BOWSER: u32 = 0x0000_0390;
const ACT_HOLDING_BOWSER: u32 = 0x0000_0391;
const ACT_RELEASING_BOWSER: u32 = 0x0000_0392;
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
const BULLY_ACT_INACTIVE: i32 = 4;
const BULLY_ACT_ACTIVATE_AND_FALL: i32 = 5;
const BULLY_ACT_LAVA_DEATH: i32 = 100;
const BULLY_ACT_DEATH_PLANE_DEATH: i32 = 101;
const BOWSER_ACT_THROWN: i32 = 1;
const BOWSER_ACT_DEAD: i32 = 4;

// ETA phase estimates (v2, GOALS.md rule 3), in frames. Rough: calibrated
// later from runs (tools/analysis/eta_calib.py).

/// Within reach to the hit landing: a punch or kick (~1/6 s).
const PUNCH_FRAMES: f64 = 5.0;
/// Whomp King: his intro dialog (DIALOG_114, whomp_init) before he moves (~2 s).
const DIALOG_FRAMES: f64 = 60.0;
/// Whomp: standing in front of him to his jump windup starting
/// (king_whomp_chase / whomp_patrol turn to face Mario first, ~1/3 s).
const LURE_FRAMES: f64 = 10.0;
/// Whomp: windup to lying flat: whomp_prepare_jump's animation (~20), then
/// whomp_jump's 8 frames and its pitch to 0x4000 at +0x100/frame² (~11),
/// and the landing (whomp.inc.c).
const FALL_FRAMES: f64 = 45.0;
/// Whomp mid-fall (actions 3, 4): about half of FALL_FRAMES left.
const FALLING_LEFT_FRAMES: f64 = 20.0;
/// Whomp lying flat: from beside his back up onto it (a jump of ~200, 1/2 s).
const CLIMB_FRAMES: f64 = 15.0;
/// On his back to the pound landing: a jump (~10), the pound's 10-frame
/// spin and the drop (mario_actions_airborne.c act_ground_pound).
const POUND_FRAMES: f64 = 25.0;
/// Mid-pound over his back to the landing (the rest of the spin, the drop).
const POUND_FALL_FRAMES: f64 = 8.0;
/// Big Boo invisible (intangible): v1's progress bonus for it being visible
/// was 300 units, 10 frames at PACE.
const BOO_INVISIBLE_FRAMES: f64 = 10.0;
/// Big Boo not active yet (big_boo_act_0): v1 ranked such a start 5
/// heuristic units (2000 units, ~67 frames) behind.
const BOO_ACTIVATE_FRAMES: f64 = 60.0;
/// Wiggler from his level: a jump's rise and fall onto his head (~2/3 s).
const STOMP_JUMP_FRAMES: f64 = 20.0;
/// Wiggler with Mario above him (v1's +200 bonus): the drop (~1/4 s).
const STOMP_FALL_FRAMES: f64 = 8.0;
/// Eyerok hand with its eye shut: v1 ranked it 3 heuristic units (1200
/// units, 40 frames) behind an open eye.
const EYE_WAIT_FRAMES: f64 = 40.0;
/// Eyerok wakes when Mario is within 500 (eyerok_boss_act_sleep).
const EYEROK_WAKE_RADIUS: f32 = 500.0;
/// In range to its next update seeing him.
const WAKE_FRAMES: f64 = 1.0;
/// Big Bully: v1's heuristic counted 1500 pushed (home distance + 3x the
/// drop) as off the edge.
const BIG_BULLY_EDGE: f64 = 1500.0;
/// Big Bully: units pushed per frame over hit, knockback, catching up and
/// hitting again: ~1/3 of PACE.
const PUSH_PACE: f64 = 10.0;
/// Over the edge to the lava / death-plane act (~1/2 s of falling).
const BULLY_DEATH_FRAMES: f64 = 15.0;
/// Bowser: on his tail to ACT_PICKING_UP_BOWSER (the B grab, ~1/3 s).
const TAIL_GRAB_FRAMES: f64 = 10.0;
/// ACT_PICKING_UP_BOWSER to ACT_HOLDING_BOWSER (its animation, ~2/3 s).
const PICKUP_FRAMES: f64 = 20.0;
/// Spin to release at: v1's steering threw at |angleVel| >= 0xE00 (the 2.5x
/// throw speed needs ~0xF12).
const SPIN_TARGET: i32 = 0xE00;
/// Spin gained per frame with the stick rotating 0x3000/frame
/// (act_holding_bowser: 0x60).
const SPIN_ACCEL: f64 = 0x60 as f64;
/// Holding him still to SPIN_TARGET.
const SPIN_FRAMES: f64 = SPIN_TARGET as f64 / SPIN_ACCEL;
/// Spun up to facing a bomb: half a turn at 0xE00/frame on average.
const AIM_FRAMES: f64 = 9.0;
/// B to Bowser's thrown act (ACT_RELEASING_BOWSER throws on its first frame).
const RELEASE_FRAMES: f64 = 2.0;
/// Thrown Bowser's speed at SPIN_TARGET: 0xE00 / 3000 * 70 * 2.5
/// (bowser_thrown_dropped_update).
const THROW_PACE: f32 = 209.0;
/// A bomb explodes on thrown Bowser within 800 (bowser_check_hit_mine).
const BOMB_REACH: f32 = 800.0;
/// Flight frames when no bomb is in the list (objects not read).
const NO_BOMB_FLIGHT_FRAMES: f64 = 10.0;
/// `mem.flags` of Bowser::ThrowAtBomb: Mario has held him this attempt.
const HELD_BOWSER: u32 = 1;

// The rest of a fight after this hit (v1 ranked fight starts by hits left,
// 10 heuristic units a hit): (hits left after this one) x one hit's cycle,
// from the start's boss (its health only changes at a hit, which is Done, so
// the term is constant within an attempt). Each cycle is the boss's
// recovery plus this objective's phases once more.

/// Whomp King after a pound: he rises (whomp.inc.c, ~1 s) and turns to
/// chase again.
const WHOMP_RECOVER_FRAMES: f64 = 40.0;
/// One more Whomp King hit: recovery, the lure, his fall, onto his back,
/// the pound.
const WHOMP_HIT_CYCLE_FRAMES: f64 = WHOMP_RECOVER_FRAMES + LURE_FRAMES + FALL_FRAMES + CLIMB_FRAMES + POUND_FRAMES;
/// Big Boo after a hit: its hurt flight and fade back in (big_boo_act_2/3,
/// ~2 s).
const BOO_RECOVER_FRAMES: f64 = 60.0;
/// Back to it from where the hit left Mario (~1 s).
const BOO_REAPPROACH_FRAMES: f64 = 30.0;
/// One more Big Boo hit.
const BOO_HIT_CYCLE_FRAMES: f64 = BOO_RECOVER_FRAMES + BOO_REAPPROACH_FRAMES + PUNCH_FRAMES;
/// Wiggler after a stomp: his squish and dialog (wiggler_act_jumped_on,
/// ~2 s) and the chase to his head again (~1 s).
const WIGGLER_RECOVER_FRAMES: f64 = 90.0;
/// One more Wiggler stomp.
const WIGGLER_HIT_CYCLE_FRAMES: f64 = WIGGLER_RECOVER_FRAMES + STOMP_JUMP_FRAMES;
/// Eyerok hand after a punch: it recoils and attacks before showing its eye
/// again (eyerok.inc.c, ~2 s), and back in front of it (~1 s).
const EYEROK_RECOVER_FRAMES: f64 = 90.0;
/// One more punch on the same hand.
const EYEROK_HIT_CYCLE_FRAMES: f64 = EYEROK_RECOVER_FRAMES + EYE_WAIT_FRAMES + PUNCH_FRAMES;
/// Bowser after a bomb: his hurt act and the walk / jump back to the
/// arena's middle (bowser.inc.c, ~3 s).
const BOWSER_RECOVER_FRAMES: f64 = 90.0;
/// Round to his tail again (~1 s).
const BOWSER_REAPPROACH_FRAMES: f64 = 30.0;
/// One more Bowser throw: recovery, the grab, the spin, the aim and the
/// release, a typical bomb flight.
const BOWSER_HIT_CYCLE_FRAMES: f64 = BOWSER_RECOVER_FRAMES
    + BOWSER_REAPPROACH_FRAMES
    + TAIL_GRAB_FRAMES
    + PICKUP_FRAMES
    + SPIN_FRAMES
    + AIM_FRAMES
    + RELEASE_FRAMES
    + NO_BOMB_FLIGHT_FRAMES;

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

    /// The fight after this hit: `after(start's boss)` hits left once this
    /// one lands, times `cycle` frames (0 without the boss in the start).
    fn rest(&self, start: &St, after: impl Fn(&Obj) -> i32, cycle: f64) -> f64 {
        self.get(start).map_or(0.0, |o0| after(o0).max(0) as f64 * cycle)
    }

    /// v1's `hurt_or`, per state: `Err(Done)` where it held (the boss's
    /// health below the start's, or `extra(start's boss, boss)`), `Err(Lost)`
    /// without the boss in the start or in `s` or out of the zone, else the
    /// boss in `s`.
    fn hit<'a>(&self, s: &'a St, start: &St, extra: impl Fn(&Obj, &Obj) -> bool) -> Result<&'a Obj, Eta> {
        let o0 = self.get(start).ok_or(Eta::Lost)?;
        if s.zone() != self.zone {
            return Err(Eta::Lost);
        }
        let o = self.get(s).ok_or(Eta::Lost)?;
        // oHealth is junk (2048) before some bosses initialise it; a drop from
        // there is not a hit.
        let hp_known = (1..=4).contains(&o0.health);
        if (hp_known && o.health < o0.health) || extra(o0, o) {
            return Err(Eta::Done);
        }
        Ok(o)
    }
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

    /// In front of him, within 300: he jumps to slam.
    fn front(o: &Obj) -> [f32; 3] {
        offset(o.pos, o.yaw, 250.0, 0.0, 0.0)
    }

    /// Out of the way of his fall (he covers 100..860 ahead, +-300 wide).
    fn dodge(o: &Obj) -> [f32; 3] {
        offset(o.pos, o.yaw, 400.0, 550.0, 0.0)
    }

    /// Frames to the pound on his back: (dialog,) lure him from in front,
    /// his fall (dodged), onto his back, the pound. His back is where it
    /// will be once he lies flat (he falls forward about his feet).
    fn pound_eta(s: &St, o: &Obj) -> f64 {
        let back = Self::back(o);
        let from = |p: [f32; 3]| frames_for(dist(p, back)) + CLIMB_FRAMES + POUND_FRAMES;
        if Self::lying(o) {
            let on_top = s.pos[1] > back[1] - 30.0 && s.hdist(back) < 350.0;
            if on_top && matches!(s.action, ACT_GROUND_POUND | ACT_GROUND_POUND_LAND) {
                POUND_FALL_FRAMES
            } else if on_top {
                POUND_FRAMES
            } else {
                from(s.pos)
            }
        } else if matches!(o.action, WHOMP_PREPARE_JUMP | WHOMP_JUMP) {
            FALLING_LEFT_FRAMES + frames_for(dist(s.pos, Self::dodge(o))) + from(Self::dodge(o))
        } else {
            let lure = frames_for(dist(s.pos, Self::front(o))) + LURE_FRAMES + FALL_FRAMES + from(Self::front(o));
            if o.action == WHOMP_INIT { DIALOG_FRAMES + lure } else { lure }
        }
    }
}

impl Objective for WhompPound {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 24, generations: 6, horizon: 300 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        match self.f.hit(s, start, |_, o| o.action >= WHOMP_DIE) {
            // He dies at 0 (whomp.inc.c): health - 1 hits after this one.
            Ok(o) => Eta::Frames(Self::pound_eta(s, o) + self.f.rest(start, |o0| o0.health.clamp(1, 3) - 1, WHOMP_HIT_CYCLE_FRAMES)),
            Err(e) => e,
        }
    }
    fn success_value(&self) -> f64 {
        10.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = self.f.get(s)?;
        Some(if Self::lying(o) {
            Self::back(o)
        } else if matches!(o.action, WHOMP_PREPARE_JUMP | WHOMP_JUMP) {
            Self::dodge(o)
        } else {
            Self::front(o)
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

impl Objective for BigBooHit {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 24, generations: 6, horizon: 240 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        let hurt = |o0: &Obj, o: &Obj| (o.action == BIG_BOO_HURT && o0.action != BIG_BOO_HURT) || o.action == BIG_BOO_DEAD;
        let o = match self.f.hit(s, start, hurt) {
            Ok(o) => o,
            Err(e) => return e,
        };
        // Tangible only while visible; not yet active before its fight.
        let wait = if o.interact_type == 0 { BOO_INVISIBLE_FRAMES } else { 0.0 }
            + if o.action == BIG_BOO_INACTIVE { BOO_ACTIVATE_FRAMES } else { 0.0 };
        // It dies at 0 (boo.inc.c); before activation oHealth isn't set yet
        // (big_boo_act_0 sets 3): v1's hits_left.
        let rest = self.f.rest(start, |o0| if (1..=3).contains(&o0.health) { o0.health - 1 } else { 2 }, BOO_HIT_CYCLE_FRAMES);
        Eta::Frames(frames_for(dist(s.pos, o.pos)) + wait + PUNCH_FRAMES + rest)
    }
    fn success_value(&self) -> f64 {
        10.0
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
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 20, generations: 6, horizon: 200 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        let stomped = |o0: &Obj, o: &Obj| o.action == WIGGLER_ACT_JUMPED_ON && o0.action != WIGGLER_ACT_JUMPED_ON;
        let o = match self.f.hit(s, start, stomped) {
            Ok(o) => o,
            Err(e) => return e,
        };
        let land = if s.pos[1] > o.pos[1] + 50.0 { STOMP_FALL_FRAMES } else { STOMP_JUMP_FRAMES };
        // He shrinks at 1 (`--oHealth == 1`, wiggler.inc.c): health - 1
        // stomps in all, one fewer mid-squish (v1), minus this one.
        let after = |o0: &Obj| o0.health - 1 - (o0.action == WIGGLER_ACT_JUMPED_ON) as i32 - 1;
        Eta::Frames(frames_for(s.hdist(o.pos)) + land + self.f.rest(start, after, WIGGLER_HIT_CYCLE_FRAMES))
    }
    fn success_value(&self) -> f64 {
        10.0
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
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 24, generations: 6, horizon: 300 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        let died = |o0: &Obj, o: &Obj| o.action == EYEROK_HAND_ACT_DIE && o0.action != EYEROK_HAND_ACT_DIE;
        let o = match self.f.hit(s, start, died) {
            Ok(o) => o,
            Err(e) => return e,
        };
        let eye = if o.action == EYEROK_HAND_ACT_SHOW_EYE { 0.0 } else { EYE_WAIT_FRAMES };
        // A hand dies when `--oHealth >= 2` fails (eyerok.inc.c): health - 1
        // punches in all, minus this one.
        let rest = self.f.rest(start, |o0| o0.health - 2, EYEROK_HIT_CYCLE_FRAMES);
        Eta::Frames(frames_for(dist(s.pos, offset(o.pos, o.yaw, 200.0, 0.0, 0.0))) + eye + PUNCH_FRAMES + rest)
    }
    fn success_value(&self) -> f64 {
        10.0
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
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 200 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.f.zone {
            return Eta::Lost;
        }
        let Some(o) = self.f.get(s) else { return Eta::Lost };
        if o.action != EYEROK_BOSS_ACT_SLEEP {
            return Eta::Done;
        }
        Eta::Frames(frames_for((dist(s.pos, o.pos) - EYEROK_WAKE_RADIUS).max(0.0)) + WAKE_FRAMES)
    }
    fn success_value(&self) -> f64 {
        2.0
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
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 24, generations: 6, horizon: 300 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        // Lava/death-plane death, seen while it is still in the state (both
        // last at least a frame before it is deleted: obj_lava_death shrinks
        // it first, the death-plane act deactivates it on its next update,
        // bully.inc.c), or its star. A frame without it (objects not read,
        // or not in the list) is not a defeat.
        if self.f.star.is_some_and(|(c, i)| s.has_star(c, i)) {
            return Eta::Done;
        }
        if s.zone() != self.f.zone {
            return Eta::Lost;
        }
        let Some(o) = self.f.get(s) else { return Eta::Lost };
        if matches!(o.action, BULLY_ACT_LAVA_DEATH | BULLY_ACT_DEATH_PLANE_DEATH) {
            return Eta::Done;
        }
        let push_left = (BIG_BULLY_EDGE - Self::pushed(o)).max(0.0) / PUSH_PACE;
        Eta::Frames(frames_for(dist(s.pos, o.pos)) + push_left + BULLY_DEATH_FRAMES)
    }
    fn success_value(&self) -> f64 {
        10.0
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

/// Frames to holding him: round to his back (via the steering point beside
/// him), onto the tail, the grab.
fn bowser_grab_eta(s: &St, o: &Obj) -> f64 {
    let tail = bowser_tail(s, o);
    let approach = if behind_bowser(s, o) {
        frames_for(dist(s.pos, tail))
    } else {
        let t = bowser_grab_target(s, o);
        frames_for(dist(s.pos, t)) + frames_for(dist(t, tail))
    };
    approach + TAIL_GRAB_FRAMES
}

/// `Bowser::GrabTail(id)`: get behind Bowser and grab his tail (B).
struct BowserGrabTail {
    f: Fight,
}

impl Objective for BowserGrabTail {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 20, generations: 6, horizon: 240 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // Picking him up always ends in holding him (act_picking_up_bowser).
        if holding_bowser(s) {
            return Eta::Done;
        }
        if s.zone() != self.f.zone {
            return Eta::Lost;
        }
        // v1 ranked by his health too (bowser_heuristic): the throws after
        // the one this grab leads to.
        let rest = self.f.rest(start, |o0| o0.health - 1, BOWSER_HIT_CYCLE_FRAMES);
        self.f.get(s).map_or(Eta::Lost, |o| Eta::Frames(bowser_grab_eta(s, o) + rest))
    }
    fn success_value(&self) -> f64 {
        3.0
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

    /// Frames for Bowser thrown from `p` to be within a bomb's reach.
    fn flight(s: &St, p: [f32; 3]) -> f64 {
        let d = Self::bombs(s).map(|b| dist(b.pos, p)).fold(f32::MAX, f32::min);
        if d == f32::MAX { NO_BOMB_FLIGHT_FRAMES } else { ((d - BOMB_REACH).max(0.0) / THROW_PACE) as f64 + 1.0 }
    }
}

impl Objective for BowserThrow {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): boss hits are timing windows: a state every 10 frames, mid-air too.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 10, air: true }
    }
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
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 24, generations: 6, horizon: 240 })
    }
    fn eta(&self, s: &St, start: &St, mem: &mut Mem) -> Eta {
        let o = match self.f.hit(s, start, |_, o| o.action == BOWSER_ACT_DEAD) {
            Ok(o) => o,
            Err(e) => return e,
        };
        if holding_bowser(s) {
            mem.flags |= HELD_BOWSER;
        }
        // A flight counts only after holding him in this attempt (v1's
        // progress latch): a throw from before the start isn't ours.
        let held = mem.flags & HELD_BOWSER != 0;
        let from_mario = AIM_FRAMES + RELEASE_FRAMES + Self::flight(s, s.pos);
        // He dies at 0 (bowser.inc.c): health - 1 throws after this one.
        let rest = self.f.rest(start, |o0| o0.health - 1, BOWSER_HIT_CYCLE_FRAMES);
        Eta::Frames(rest + match s.action {
            ACT_PICKING_UP_BOWSER => PICKUP_FRAMES + SPIN_FRAMES + from_mario,
            ACT_HOLDING_BOWSER => {
                (SPIN_TARGET - (s.angle_vel[1] as i32).abs()).max(0) as f64 / SPIN_ACCEL + from_mario
            }
            _ if held && o.action == BOWSER_ACT_THROWN => Self::flight(s, o.pos),
            ACT_RELEASING_BOWSER if held => RELEASE_FRAMES + Self::flight(s, s.pos),
            _ => bowser_grab_eta(s, o) + PICKUP_FRAMES + SPIN_FRAMES + from_mario,
        })
    }
    fn success_value(&self) -> f64 {
        15.0
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{falls, obj, st};
    use obrrt_core::layout::LEVEL_BOWSER_1;
    use obrrt_core::{fold, start_eta};

    // A Whomp King-like level (not a Bowser one, so the generator's level
    // checks don't matter here).
    const LEVEL: u32 = 24;

    fn king(action: i32, health: i32) -> Obj {
        obj("bhvWhompKingBoss", 5, [0.0; 3], action, health)
    }

    /// Far in front, in front, his windup, mid-fall (dodged), flat, on his
    /// back, pounding, one hit: the ETA falls at every phase change.
    #[test]
    fn whomp_king_pound_phases() {
        let pound = |action, health, pos: [f32; 3], mario: u32| {
            let mut s = st(LEVEL, pos, vec![king(action, health)]);
            if mario != 0 {
                s.action = mario;
            }
            s
        };
        let back = [0.0, 200.0, 480.0];
        let dodge = [550.0, 0.0, 400.0];
        let traj = vec![
            pound(2, 3, [0.0, 0.0, 1500.0], 0),
            pound(2, 3, [0.0, 0.0, 250.0], 0),
            pound(WHOMP_PREPARE_JUMP, 3, [0.0, 0.0, 250.0], 0),
            pound(WHOMP_JUMP, 3, dodge, 0),
            pound(WHOMP_ON_GROUND, 3, dodge, 0),
            pound(WHOMP_ON_GROUND, 3, back, 0),
            pound(WHOMP_ON_GROUND, 3, back, ACT_GROUND_POUND),
            pound(WHOMP_ON_GROUND, 2, back, ACT_GROUND_POUND_LAND),
        ];
        let w = WhompPound { f: Fight::new(&traj[0], &traj[0].objects()[0], None) };
        let (met, etas) = fold(&w, &traj);
        assert_eq!(met, Some(7));
        let f: Vec<f64> = etas[..7].iter().map(|e| e.frames()).collect();
        assert!(falls(&f), "ETA must fall: {f:?}");
        // His intro dialog first costs more; leaving the zone is Lost.
        let init = pound(WHOMP_INIT, 3, [0.0, 0.0, 1500.0], 0);
        assert!(start_eta(&w, &init) > start_eta(&w, &traj[0]));
        let mut away = traj[1].clone();
        away.area = 2;
        assert_eq!(fold(&w, &[traj[0].clone(), away]).1[1], Eta::Lost);
    }

    fn bowser_st(pos: [f32; 3], action: u32, b: Obj, av: i16) -> St {
        let bomb = obj("bhvBowserBomb", 9, [0.0, 0.0, 3000.0], 0, 0);
        let tail = obj("bhvBowserTailAnchor", 10, [0.0, 0.0, -300.0], 0, 0);
        let mut s = st(LEVEL_BOWSER_1, pos, vec![b, bomb, tail]);
        s.action = action;
        s.angle_vel[1] = av;
        s
    }

    fn bowser(action: i32, health: i32, pos: [f32; 3]) -> Obj {
        let mut b = obj("bhvBowser", 8, [0.0; 3], action, health);
        b.pos = pos;
        b
    }

    /// In front, behind at the tail, picking up, spinning, spun, releasing,
    /// flying at the bomb, hit: the ETA falls throughout; the grab alone is
    /// `Bowser::GrabTail`'s Done.
    #[test]
    fn bowser_throw_phases() {
        let idle = ACT_IDLE_TEST;
        let home = [0.0; 3];
        let at_tail = [0.0, 0.0, -350.0];
        let traj = vec![
            bowser_st([0.0, 0.0, 1000.0], idle, bowser(0, 1, home), 0),
            bowser_st(at_tail, idle, bowser(0, 1, home), 0),
            bowser_st(at_tail, ACT_PICKING_UP_BOWSER, bowser(0, 1, home), 0),
            bowser_st(at_tail, ACT_HOLDING_BOWSER, bowser(0, 1, home), 0x400),
            bowser_st(at_tail, ACT_HOLDING_BOWSER, bowser(0, 1, home), 0xE00),
            bowser_st(at_tail, ACT_RELEASING_BOWSER, bowser(0, 1, home), 0),
            bowser_st(at_tail, idle, bowser(BOWSER_ACT_THROWN, 1, [0.0, 0.0, 1000.0]), 0),
            bowser_st(at_tail, idle, bowser(BOWSER_ACT_THROWN, 1, [0.0, 0.0, 2000.0]), 0),
            bowser_st(at_tail, idle, bowser(BOWSER_ACT_THROWN, 0, [0.0, 0.0, 2300.0]), 0),
        ];
        let o = traj[0].objects()[0].clone();
        let throw = BowserThrow { f: Fight::new(&traj[0], &o, None) };
        let (met, etas) = fold(&throw, &traj);
        assert_eq!(met, Some(8));
        let f: Vec<f64> = etas[..8].iter().map(|e| e.frames()).collect();
        assert!(falls(&f), "ETA must fall: {f:?}");

        // A flight not started by Mario in this attempt (no hold seen) is
        // the grab phase, not the throw's.
        let (_, e) = fold(&throw, &traj[6..7]);
        assert!(e[0].frames() > f[6], "unlatched flight {:?} vs {}", e[0], f[6]);

        let grab = BowserGrabTail { f: Fight::new(&traj[0], &o, None) };
        let (met, etas) = fold(&grab, &traj[..3]);
        assert_eq!(met, Some(2));
        assert!(etas[1].frames() < etas[0].frames());
    }

    /// Each fight ranks a start with fewer hits left closer (v1's health
    /// term): the hits after this one are in the ETA.
    #[test]
    fn fewer_hits_left_is_closer() {
        let p = [0.0, 0.0, 800.0];
        let check = |name: &str, o: &dyn Objective, mk: &dyn Fn(i32) -> St, hps: &[i32]| {
            let e: Vec<f64> = hps.iter().map(|&h| start_eta(o, &mk(h))).collect();
            assert!(e.iter().all(|x| x.is_finite()) && falls(&e), "{name}: {e:?} for health {hps:?}");
        };
        let whomp = |h| st(LEVEL, p, vec![king(2, h)]);
        let w = WhompPound { f: Fight::new(&whomp(3), &whomp(3).objects()[0], None) };
        check("whomp", &w, &whomp, &[3, 2, 1]);
        let boo = |h| {
            let mut o = obj("bhvGhostHuntBigBoo", 5, [0.0; 3], 1, h);
            o.interact_type = 1;
            st(LEVEL, p, vec![o])
        };
        let b = BigBooHit { f: Fight::new(&boo(3), &boo(3).objects()[0], None) };
        check("boo", &b, &boo, &[3, 2, 1]);
        let wig = |h| st(LEVEL, p, vec![obj("bhvWigglerHead", 5, [0.0; 3], 1, h)]);
        let g = WigglerStomp { f: Fight::new(&wig(4), &wig(4).objects()[0], None) };
        check("wiggler", &g, &wig, &[4, 3, 2]);
        let hand = |h| st(LEVEL, p, vec![obj("bhvEyerokHand", 5, [0.0; 3], EYEROK_HAND_ACT_SHOW_EYE, h)]);
        let e = EyerokPunchEye { f: Fight::new(&hand(4), &hand(4).objects()[0], None) };
        check("eyerok", &e, &hand, &[4, 3, 2]);
        let bw = |h| bowser_st([0.0, 0.0, 1000.0], ACT_IDLE_TEST, bowser(0, h, [0.0; 3]), 0);
        let o = bw(3).objects()[0].clone();
        check("bowser throw", &BowserThrow { f: Fight::new(&bw(3), &o, None) }, &bw, &[3, 2, 1]);
        check("bowser grab", &BowserGrabTail { f: Fight::new(&bw(3), &o, None) }, &bw, &[3, 2, 1]);
    }

    const ACT_IDLE_TEST: u32 = obrrt_core::layout::ACT_IDLE;
}
