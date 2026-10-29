//! Warps and transitions: warp pipes, warp holes and teleporters
//! (INTERACT_WARP), the BBH entrance (the courtyard's boo cage), tornadoes
//! (Tweesters), star doors and key doors with their requirements, cannons
//! (enter, then aim and fire at a target), the castle's secret entrances
//! (PSS, the aquarium, WMOTR, the cap stages, the sun to the Tower of the
//! Wing Cap, the moat drain) and the Bowser stage entrances (the trap floor
//! behind the 8-star door, the DDD painting room, the endless stairs) and
//! the pipes to the Bowser fights.
//!
//! Not here, on purpose:
//! - Whirlpools (INTERACT_WHIRLPOOL, DDD): being caught is a death warp
//!   (mario_actions_submerged.c act_caught_in_whirlpool:
//!   `level_trigger_warp(m, WARP_OP_DEATH)`), never a transition.
//! - Plain doors and paintings: `Door::Enter` / `Painting::Enter`. The star
//!   and key doors get extra objectives here that only start from states
//!   which meet the door's requirement.
//!
//! NEEDS: `SAVE_FLAG_MOAT_DRAINED` (or the save flags) in `St`: the VCUTM
//! warp at the moat's bottom is behind grills until the moat is drained
//! (moat_grill.inc.c), so it is offered at a reduced weight from every state
//! in the castle grounds instead of only from drained ones; and a drained
//! moat's pillars are told apart only by their lowered position.

use std::sync::Arc;

use obrrt_core::buttons::{A_BUTTON, U_CBUTTONS, Z_TRIG};
use obrrt_core::layout::*;
use obrrt_core::{Budget, GamePad, Obj, ObjId, Objective, Progress, St, course_of_level, dist};

mod table;

use table::{CANNON_TARGETS, Class, LEVEL_BOWSER_1, LEVEL_BOWSER_2, LEVEL_BOWSER_3, Req, SUN_SPOT, SUN_SPOT_RADIUS, Warp};

// Actions missing from the generated layout (include/sm64.h).
const ACT_DISAPPEARED: u32 = 0x0000_1300;
const ACT_TELEPORT_FADE_OUT: u32 = 0x0000_1336;
const ACT_BBH_ENTER_JUMP: u32 = 0x0000_1934;
const ACT_BBH_ENTER_SPIN: u32 = 0x0000_1535;
const ACT_IN_CANNON: u32 = 0x0000_1371;
const ACT_SHOT_FROM_CANNON: u32 = 0x0088_0898;
const ACT_TORNADO_TWIRLING: u32 = 0x1002_0372;
const ACT_GROUND_POUND: u32 = 0x0080_08A9;

/// include/object_constants.h BOO_CAGE_ACT_ON_GROUND: the cage has fallen
/// out of the dead Big Boo and is tangible.
const BOO_CAGE_ACT_ON_GROUND: i32 = 2;

pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let castle = b.level as u32 == LEVEL_CASTLE;
    match o.interact_type {
        INTERACT_WARP => out.push(Arc::new(WarpEnter::new(b, o))),
        INTERACT_CANNON_BASE if o.bhv_name() == "bhvCannon" && !cannon_grilled(b) => {
            out.push(Arc::new(CannonEnter::new(b, o)));
            out.push(Arc::new(CannonFire::far(b, o)));
            if let Some(aim) = cannon_aim_yaw(o) {
                for &(level, area, name, target, star) in CANNON_TARGETS {
                    if (level, area) == (b.level as u32, b.area) && !b.has_star(star.0, star.1) {
                        if let Some(f) = CannonFire::aimed(b, o, aim, name, target, star) {
                            out.push(Arc::new(f));
                        }
                    }
                }
            }
        }
        INTERACT_BBH_ENTRANCE if o.action == BOO_CAGE_ACT_ON_GROUND => out.push(Arc::new(BbhEnter::new(b, o))),
        INTERACT_TORNADO => out.push(Arc::new(TornadoEnter::new(b, o))),
        // interaction.c interact_door: the 1st behaviour byte is the number of
        // stars needed (bhvStarDoor and the castle's 1- and 3-star bhvDoors).
        INTERACT_DOOR if castle && (o.beh_params >> 24) > 0 => out.push(Arc::new(StarDoor::new(b, o))),
        // interaction.c interact_warp_door: warp door 1 (upstairs) needs key 2,
        // door 2 (basement) needs key 1.
        INTERACT_WARP_DOOR if castle && matches!(o.beh_params >> 24, 1 | 2) => out.push(Arc::new(KeyDoor::new(b, o))),
        _ => {}
    }
    // The basement pillars that drain the moat are Pillar::Pound in
    // switches.rs (PillarPound here was the same objective twice).
}

pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    // The sun in the lobby: needs 10 stars (mario_actions_stationary.c
    // act_first_person).
    if b.level as u32 == LEVEL_CASTLE && b.area == 1 && b.stars >= 10 {
        out.push(Arc::new(SunStand));
        out.push(Arc::new(SunLookUp));
    }
}

// ---------------------------------------------------------------- helpers

/// The trajectory's closest approach to `p` among states that pass `ok`.
fn closest(traj: &[St], p: [f32; 3], ok: impl Fn(&St) -> bool) -> Progress {
    let (at, d) = traj
        .iter()
        .enumerate()
        .filter(|(_, s)| ok(s))
        .map(|(i, s)| (i, dist(s.pos, p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f32::MAX));
    Progress { score: -(d as f64), at }
}

/// SM64 yaw (0 = +z, 0x4000 = +x) of the direction (dx, dz), 0..65536.
fn yaw_of(dx: f32, dz: f32) -> i32 {
    ((dx.atan2(dz) * 32768.0 / std::f32::consts::PI) as i32).rem_euclid(65536)
}

/// `a - b` wrapped to -0x8000..0x8000.
fn angle_diff(a: i32, b: i32) -> i32 {
    ((a - b + 0x8000).rem_euclid(0x10000)) - 0x8000
}

/// How worth entering `dest` is from `s`: courses (and Bowser fights) that
/// are finished give little.
fn dest_worth(s: &St, dest: u32) -> f64 {
    let done = match dest {
        LEVEL_BOWSER_1 => s.keys >= 1,
        LEVEL_BOWSER_2 => s.keys >= 2,
        LEVEL_BOWSER_3 => false,
        // A Bowser course is worth its fight as well as its red-coin star.
        LEVEL_BITDW => s.keys >= 1 && s.course_done(16),
        LEVEL_BITFS => s.keys >= 2 && s.course_done(17),
        _ => {
            let c = course_of_level(dest);
            c != 0 && s.course_done(c)
        }
    };
    if done { 0.15 } else { 1.0 }
}

fn req_met(req: Req, s: &St) -> bool {
    match req {
        Req::None => true,
        Req::Stars(n) => s.stars >= n,
        Req::StarsAndStar(n, c, i) => s.stars >= n && s.has_star(c, i),
    }
}

// ---------------------------------------------------------- warp objects

/// `Pipe::Enter` / `WarpHole::Enter` / `Teleporter::Enter` /
/// `SecretEntrance::Enter` / `BowserStage::Enter`: touch an INTERACT_WARP
/// object. Pipes and holes put Mario in ACT_DISAPPEARED and warp
/// (interaction.c interact_warp); teleporters (bhvFadingWarp,
/// INT_SUBTYPE_FADING_WARP) need Mario standing still on them (idle,
/// panting, crouching) and fade him out (ACT_TELEPORT_FADE_OUT).
pub struct WarpEnter {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
    kind: &'static str,
    name: String,
    info: Option<&'static Warp>,
    /// Touch radius: the 1st behaviour byte * 10 (warp.inc.c), pipes 70.
    radius: f32,
    fading: bool,
}

impl WarpEnter {
    pub fn new(b: &St, o: &Obj) -> WarpEnter {
        let bhv = o.bhv_name();
        // bhvWarp has no SET_HOME, so its oHome is (0, 0, 0) (core/src/read.rs
        // gives such objects the id home (i32::MIN, slot, 0): the castle
        // grounds' VCUTM warp is `bhvWarp@-2147483648,93,0`): with
        // OBRRT_GROUNDS_FIX=1 that home is replaced by the position (warps
        // don't move), so the table finds it and the steering aims at it.
        // Before, its WarpHole::Enter steered at (0, 0, 0), the plateau by
        // the castle (its nodes end at (-2, 810, 4)): 8-19% of the grounds
        // frames before the first castle entry, its only "successes" walks
        // through the castle door (any zone change counts without a table
        // entry).
        let home = if grounds_fix() && o.home == [0.0, 0.0, 0.0] { o.pos } else { o.home };
        let info = table::find(b.level, b.area, bhv, home);
        let fading = bhv == "bhvFadingWarp";
        let kind = match info.map(|w| w.class) {
            Some(Class::Secret) => "SecretEntrance::Enter",
            Some(Class::Bowser) => "BowserStage::Enter",
            Some(Class::Teleport) => "Teleporter::Enter",
            _ if bhv == "bhvWarpPipe" => "Pipe::Enter",
            _ if fading => "Teleporter::Enter",
            _ if bhv == "bhvWarp" => "WarpHole::Enter",
            _ => "Warp::Enter",
        };
        let byte0 = (o.beh_params >> 24) & 0xFF;
        let radius = match (bhv, byte0) {
            ("bhvWarpPipe", _) => 70.0,
            (_, 0) if fading => 85.0,
            (_, 0) => 50.0,
            // 0xFF is "anywhere" (10000); cap it for the closeness test.
            (_, 0xFF) => 1000.0,
            (_, r) => r as f32 * 10.0,
        };
        let name = info.map_or_else(|| format!("{:?} {}", b.zone(), o.id), |w| w.name.to_string());
        WarpEnter { id: o.id, zone: b.zone(), pos: home, kind, name, info, radius, fading }
    }

    fn dest(&self) -> Option<(u16, u8)> {
        self.info.map(|w| (w.dest.0 as u16, w.dest.1))
    }

    fn warping(&self, s: &St) -> bool {
        let near = s.hdist(self.pos) < self.radius + 150.0 && (s.pos[1] - self.pos[1]).abs() < 400.0;
        near && (s.action == ACT_DISAPPEARED || s.action == ACT_TELEPORT_FADE_OUT)
    }
}

impl Objective for WarpEnter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn key(&self) -> String {
        format!("Warp {} {}", self.name, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        let base = match self.info.map(|w| w.class) {
            Some(Class::Bowser) => 15.0,
            // VCUTM may well be behind the undrained moat's grills (NEEDS).
            // OBRRT_GROUNDS_FIX=1: behind the grills (moat_grill.inc.c: solid
            // until SAVE_FLAG_MOAT_DRAINED), so only the moat drain or the
            // moat-bank BLJ (Tech::MoatWarpBlj) gets there: low.
            Some(Class::Secret) if self.name == "vcutm" && grounds_fix() => 1.0,
            Some(Class::Secret) if self.name == "vcutm" => 6.0,
            Some(Class::Secret) => 12.0,
            Some(Class::Area) if self.kind == "Pipe::Enter" => 5.0,
            Some(Class::Area) => 6.0,
            Some(Class::Teleport) => 2.0,
            None if self.fading => 2.0,
            None => 4.0,
        };
        base * self.info.map_or(1.0, |w| if w.dest.0 != b.level as u32 { dest_worth(b, w.dest.0) } else { 1.0 })
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && self.info.is_none_or(|w| {
                req_met(w.req, s)
                    // lblj: the trap floor checks no stars; behind the 8-star door it's open.
                    || (w.name == "bitdw" && super::tech::enabled("lblj") && super::tech::lblj::behind_door(s))
            })
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| {
            let arrived = match self.dest() {
                Some(d) => s.zone() == d && d != self.zone,
                None => s.zone() != self.zone && s.alive(),
            };
            arrived || self.warping(s)
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, self.pos, |s| s.zone() == self.zone)
    }
    fn budget(&self) -> Budget {
        match self.info.map(|w| w.class) {
            Some(Class::Secret | Class::Bowser) => Budget::Ga { population: 20, generations: 6, horizon: 300 },
            _ => Budget::Ga { population: 16, generations: 5, horizon: 200 },
        }
    }
    fn success_value(&self) -> f64 {
        match self.info.map(|w| w.class) {
            Some(Class::Bowser) => 6.0,
            Some(Class::Secret) => 5.0,
            Some(Class::Area) => 2.0,
            _ if self.fading => 0.5,
            _ => 1.0,
        }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn kinship(&self) -> u32 {
        if matches!(self.info.map(|w| w.class), Some(Class::Secret | Class::Bowser)) { 10 } else { 6 }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some()
    }
}

// ------------------------------------------------------------ BBH entrance

/// `BbhEntrance::Enter`: jump into the boo cage that falls out of the
/// courtyard's Big Boo (boo_cage.inc.c, INTERACT_BBH_ENTRANCE); Mario
/// jumps / spins in (ACT_BBH_ENTER_JUMP / _SPIN) and warps to BBH.
pub struct BbhEnter {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl BbhEnter {
    pub fn new(b: &St, o: &Obj) -> BbhEnter {
        BbhEnter { id: o.id, zone: b.zone(), pos: o.pos }
    }
}

impl Objective for BbhEnter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "BbhEntrance::Enter"
    }
    fn key(&self) -> String {
        format!("BbhEntrance {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        15.0 * dest_worth(b, LEVEL_BBH)
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some_and(|o| o.action == BOO_CAGE_ACT_ON_GROUND)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| {
            s.level as u32 == LEVEL_BBH || matches!(s.action, ACT_BBH_ENTER_JUMP | ACT_BBH_ENTER_SPIN)
        })
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, self.pos, |s| s.zone() == self.zone)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some()
    }
}

// ----------------------------------------------------------------- tornado

/// `Tornado::Enter`: get picked up by a Tweester (tweester.inc.c,
/// INTERACT_TORNADO), which spins Mario up (ACT_TORNADO_TWIRLING) and
/// throws him high: a way up in SSL. It wanders, so it is tracked.
pub struct TornadoEnter {
    id: ObjId,
    zone: (u16, u8),
}

impl TornadoEnter {
    pub fn new(b: &St, o: &Obj) -> TornadoEnter {
        TornadoEnter { id: o.id, zone: b.zone() }
    }

    fn pos_in(&self, s: &St) -> Option<[f32; 3]> {
        s.find(self.id).map(|o| o.pos)
    }
}

impl Objective for TornadoEnter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Tornado::Enter"
    }
    fn key(&self) -> String {
        format!("Tornado {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        3.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.action != ACT_TORNADO_TWIRLING && self.pos_in(s).is_some()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.pos_in(s).map_or(f64::MAX, |p| s.dist(p) as f64 / 400.0)
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.action == ACT_TORNADO_TWIRLING)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter_map(|(i, s)| self.pos_in(s).map(|p| (i, s.hdist(p))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 12, generations: 4, horizon: 150 }
    }
    fn success_value(&self) -> f64 {
        0.5
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.pos_in(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.pos_in(s).is_some()
    }
}

// ----------------------------------------------------- star and key doors

fn in_door_action(s: &St) -> bool {
    matches!(
        s.action,
        ACT_PULLING_DOOR | ACT_PUSHING_DOOR | ACT_ENTERING_STAR_DOOR | ACT_UNLOCKING_STAR_DOOR | ACT_UNLOCKING_KEY_DOOR
    )
}

/// `StarDoor::Enter`: go through a castle door that needs stars
/// (interaction.c interact_door: `numStars >= oBehParams >> 24`, else it
/// only shows the "you need N stars" text), only from states with enough.
/// `Door::Enter` tries these doors from any state.
pub struct StarDoor {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
    stars: i16,
}

impl StarDoor {
    pub fn new(b: &St, o: &Obj) -> StarDoor {
        StarDoor { id: o.id, zone: b.zone(), pos: o.pos, stars: (o.beh_params >> 24) as i16 }
    }
}

impl Objective for StarDoor {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "StarDoor::Enter"
    }
    fn key(&self) -> String {
        format!("StarDoor {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if b.stars < self.stars {
            // Nothing to try yet; the pool keeps it for later states.
            0.0
        } else if self.stars >= 8 {
            // 8, 30, 50, 70: the Bowser stages and the upper floors.
            12.0
        } else {
            10.0
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.stars >= self.stars
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() != self.zone || (in_door_action(s) && s.dist(self.pos) < 300.0))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, self.pos, |s| s.zone() == self.zone)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 180 }
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Door
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

/// `KeyDoor::Enter`: the basement (warp door 2, needs key 1) and upstairs
/// (warp door 1, needs key 2) doors, only from states that have the key
/// (`St::keys` counts keys held or already used on their door).
pub struct KeyDoor {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
    keys: u8,
}

impl KeyDoor {
    pub fn new(b: &St, o: &Obj) -> KeyDoor {
        let keys = if o.beh_params >> 24 == 1 { 2 } else { 1 };
        KeyDoor { id: o.id, zone: b.zone(), pos: o.pos, keys }
    }
}

impl Objective for KeyDoor {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "KeyDoor::Enter"
    }
    fn key(&self) -> String {
        format!("KeyDoor {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        if b.keys >= self.keys { 12.0 } else { 0.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.keys >= self.keys
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() != self.zone || (in_door_action(s) && s.dist(self.pos) < 300.0))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, self.pos, |s| s.zone() == self.zone)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 180 }
    }
    fn success_value(&self) -> f64 {
        4.0
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Door
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn kinship(&self) -> u32 {
        10
    }
}

// ----------------------------------------------------------------- cannons

/// `OBRRT_GROUNDS_FIX=1`: the castle grounds' cannon is grilled until 120
/// stars (see [`cannon_grilled`]); bhvWarp's unset oHome is replaced by its
/// position and the grilled VCUTM warp weighs 1 (WarpEnter::new); the
/// grounds' trees weigh a quarter (switches.rs). Off by default: it changes
/// runs.
pub fn grounds_fix() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_GROUNDS_FIX").is_ok_and(|v| v == "1"))
}

/// Whether this zone's cannon is covered (`OBRRT_GROUNDS_FIX=1` only). The
/// castle grounds' cannon is placed open (macro_cannon_open, behaviour
/// param 0x40, at (2384, 70, 1961)), so a bhvCannon exists from the start,
/// but castle_grounds/script.c puts a grill over it (MODEL_CASTLE_GROUNDS_
/// CANNON_GRILL, bhvHiddenAt120Stars, a solid surface object) that
/// castle_cannon_grate.inc.c only deactivates at load once the save has 120
/// stars. Before that Mario stands on the grill (y 65) and never touches the
/// cannon: in runs before this fix Cannon::Enter there was 22-30% of the
/// grounds frames before the first castle entry with 0 successes ever.
fn cannon_grilled(b: &St) -> bool {
    grounds_fix() && b.level as u32 == LEVEL_CASTLE_GROUNDS && b.stars < 120
}

/// `Cannon::Enter`: drop into an open cannon (bhvCannon only exists once a
/// Bob-omb Buddy has opened the course's cannons: cannon_door.inc.c spawns it
/// in place of the lid). It is tangible only within 500 units
/// (cannon.inc.c opened_cannon_act_0); touching it sets ACT_IN_CANNON
/// (interaction.c interact_cannon_base).
pub struct CannonEnter {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl CannonEnter {
    pub fn new(b: &St, o: &Obj) -> CannonEnter {
        CannonEnter { id: o.id, zone: b.zone(), pos: o.home }
    }
}

impl Objective for CannonEnter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Cannon::Enter"
    }
    fn key(&self) -> String {
        format!("CannonEnter {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        let c = course_of_level(b.level as u32);
        if c != 0 && b.course_done(c) { 1.0 } else { 8.0 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.action != ACT_IN_CANNON && s.find(self.id).is_some()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.action == ACT_IN_CANNON && s.hdist(self.pos) < 300.0)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        closest(traj, self.pos, |s| s.zone() == self.zone)
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 180 }
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some()
    }
}

/// The yaw the cannon will aim at when Mario can start aiming
/// (`oMarioCannonObjectYaw`, copied from the cannon's oMoveAngleYaw when it
/// reaches action 1), if it can be told from this object's state.
///
/// cannon.inc.c: action 0 (idle) sets yaw = 2nd behaviour byte << 8 and
/// action 4 (rising) keeps it; action 6 then swings it to
/// `sins(0x3C00) * 0x4000 + base` (its last update, at oTimer 21, with
/// oCannonUnkF4 = 15 * 0x400), i.e. base + 16305, which actions 5 and 1
/// keep. Mid-swing (action 6) it can't be told. Face yaw follows move yaw
/// (OBJ_FLAG_SET_FACE_YAW_TO_MOVE_YAW).
fn cannon_aim_yaw(o: &Obj) -> Option<i32> {
    match o.action {
        0 | 4 => Some((o.yaw + 16305).rem_euclid(65536)),
        1 | 5 => Some(o.yaw.rem_euclid(65536)),
        _ => None,
    }
}

/// Mario's flight out of a cannon aimed at (yaw, pitch): its closest
/// approach to `target`. mario_actions_automatic.c act_in_cannon: he starts
/// 350 above the cannon base plus 120 along the barrel, with forward speed
/// 100 cos(pitch) and vertical speed 100 sin(pitch); act_shot_from_cannon
/// keeps the forward speed, and gravity is 1 per frame down to -75
/// (mario_step.c apply_gravity), applied after each frame's move.
fn flight_miss(cannon: [f32; 3], yaw: i32, pitch: i32, target: [f32; 3]) -> f32 {
    let to_rad = std::f32::consts::TAU / 65536.0;
    let (y, p) = (yaw as f32 * to_rad, pitch as f32 * to_rad);
    let mut pos = [
        cannon[0] + 120.0 * p.cos() * y.sin(),
        cannon[1] + 350.0 + 120.0 * p.sin(),
        cannon[2] + 120.0 * p.cos() * y.cos(),
    ];
    let fv = 100.0 * p.cos();
    let mut vy = 100.0 * p.sin();
    let mut best = dist(pos, target);
    for _ in 0..600 {
        pos[0] += fv * y.sin();
        pos[1] += vy;
        pos[2] += fv * y.cos();
        vy = (vy - 1.0).max(-75.0);
        best = best.min(dist(pos, target));
        if pos[1] < target[1] - 3000.0 {
            break;
        }
    }
    best
}

/// Pads that move an aim value up by `amount` angle units: the value moves
/// by 10 per processed stick unit per frame, and a raw stick value `r >= 8`
/// processes to `r - 6` (game_init.c adjust_analog_stick), at most 64.
fn stick_steps(amount: i32, pad: impl Fn(i8) -> GamePad, out: &mut Vec<GamePad>) {
    out.extend(std::iter::repeat_n(pad(70), (amount / 640) as usize));
    let units = (amount % 640 + 5) / 10;
    if units >= 2 {
        out.push(pad((units + 6) as i8));
    }
}

/// `Cannon::Fire`: from inside a cannon, shoot somewhere. Two flavours:
/// - aimed at a known target (a star on a floating island, behind a wall):
///   a fixed macro that waits out the cannon's rise, clamps the aim to its
///   lower-left limits, turns and raises it by computed amounts, fires, and
///   flies; the aim is solved by simulating the flight (`flight_miss`);
/// - "far": a GA that just needs Mario to land far from the cannon.
pub struct CannonFire {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
    target: Option<(&'static str, [f32; 3], (i16, u8))>,
    pads: Vec<GamePad>,
}

impl CannonFire {
    pub fn far(b: &St, o: &Obj) -> CannonFire {
        CannonFire { id: o.id, zone: b.zone(), pos: o.home, target: None, pads: Vec::new() }
    }

    pub fn aimed(
        b: &St,
        o: &Obj,
        aim_yaw: i32,
        name: &'static str,
        target: [f32; 3],
        star: (i16, u8),
    ) -> Option<CannonFire> {
        let cannon = o.home;
        // act_in_cannon: the stick turns the yaw at most 0x4000 either side of
        // the cannon's, and the pitch between 0 and 0x38E3.
        let want = yaw_of(target[0] - cannon[0], target[2] - cannon[2]);
        let input = angle_diff(want, aim_yaw);
        if input.abs() > 0x4000 {
            return None;
        }
        let yaw = aim_yaw + input;
        let (pitch, miss) = (0..=0x38E3)
            .step_by(16)
            .map(|p| (p, flight_miss(cannon, yaw, p, target)))
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        if miss > 300.0 {
            return None;
        }
        let n = GamePad::new(0, 0, 0);
        // The cannon rises and looks around for ~120 frames before Mario can
        // aim (cannon.inc.c actions 4, 6, 5: 69 + 27 + 26 frames).
        let mut pads = vec![n; 160];
        // act_in_cannon: yaw input -= stickX * 10, pitch -= stickY * 10.
        // Clamp to input yaw -0x4000 (0x8000 / 640 = 52 frames at most) and
        // pitch 0 (0x38E3 / 640 = 23 frames).
        pads.extend(std::iter::repeat_n(GamePad::new(0, 70, 0), 55));
        pads.extend(std::iter::repeat_n(GamePad::new(0, 0, 70), 25));
        stick_steps(input + 0x4000, |v| GamePad::new(0, -v, 0), &mut pads);
        stick_steps(pitch, |v| GamePad::new(0, 0, -v), &mut pads);
        pads.push(n);
        pads.push(GamePad::new(A_BUTTON, 0, 0));
        pads.extend(std::iter::repeat_n(n, 240));
        Some(CannonFire { id: o.id, zone: b.zone(), pos: cannon, target: Some((name, target, star)), pads })
    }

    fn hit(&self, s: &St) -> bool {
        match self.target {
            Some((_, t, _)) => {
                dist(s.pos, t) < 500.0
                    || (!s.action_has(ACT_FLAG_AIR) && s.hdist(t) < 800.0 && (s.pos[1] - t[1]).abs() < 500.0)
            }
            // Out of the cannon (only firing gets him out), landed, far away.
            None => {
                s.action != ACT_IN_CANNON
                    && !s.action_has(ACT_FLAG_AIR)
                    && s.alive()
                    && s.zone() == self.zone
                    && s.hdist(self.pos) > 1500.0
            }
        }
    }
}

impl Objective for CannonFire {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Cannon::Fire"
    }
    fn key(&self) -> String {
        format!("CannonFire {:?} {} {}", self.zone, self.id, self.target.map_or("far", |t| t.0))
    }
    fn obj_weight(&self, b: &St) -> f64 {
        match self.target {
            Some(_) => 10.0,
            None => {
                let c = course_of_level(b.level as u32);
                if c != 0 && b.course_done(c) { 0.5 } else { 3.0 }
            }
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && s.action == ACT_IN_CANNON
            && s.hdist(self.pos) < 250.0
            && self.target.is_none_or(|(_, _, (c, i))| !s.has_star(c, i))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Earliest in the cannon: a fresh entry, with the aim untouched.
        s.frame as f64 / 1e6
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| self.hit(s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        match self.target {
            Some((_, t, _)) => closest(traj, t, |s| s.zone() == self.zone),
            None => {
                let (at, d) = traj
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.action == ACT_SHOT_FROM_CANNON || s.action_has(ACT_FLAG_AIR))
                    .map(|(i, s)| (i, s.hdist(self.pos)))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap_or((0, 0.0));
                Progress { score: d as f64, at }
            }
        }
    }
    fn budget(&self) -> Budget {
        if self.pads.is_empty() {
            Budget::Ga { population: 16, generations: 5, horizon: 300 }
        } else {
            Budget::Macro(self.pads.clone())
        }
    }
    fn success_value(&self) -> f64 {
        if self.target.is_some() { 4.0 } else { 1.0 }
    }
    fn temperature(&self) -> f64 {
        0.5
    }
}

// --------------------------------------------------------------- the moat

/// `Moat::Drain`: ground-pound a basement pillar (water_pillar.inc.c: action
/// 0 -> 1 on `cur_obj_is_mario_ground_pounding_platform`); with both down
/// the moat drains (SAVE_FLAG_MOAT_DRAINED), opening the way to VCUTM.
pub struct PillarPound {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl PillarPound {
    pub fn new(b: &St, o: &Obj) -> PillarPound {
        PillarPound { id: o.id, zone: b.zone(), pos: o.pos }
    }
}

impl Objective for PillarPound {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Moat::Drain"
    }
    fn key(&self) -> String {
        format!("MoatPillar {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        6.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some_and(|o| o.action == 0 && o.pos[1] > o.home[1] - 40.0)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.pos) as f64 / 400.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.find(self.id).is_some_and(|o| o.action >= 1))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Above the pillar is what counts; a ground pound there is the rest.
        let (at, d) = traj
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let below = (self.pos[1] - s.pos[1]).max(0.0);
                let pound = if s.action == ACT_GROUND_POUND { -100.0 } else { 0.0 };
                (i, s.hdist(self.pos) + below + pound)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn steer_button(&self, s: &St) -> u16 {
        // Z in the air above it starts a ground pound.
        if s.action_has(ACT_FLAG_AIR) && s.hdist(self.pos) < 120.0 && s.pos[1] > self.pos[1] { Z_TRIG } else { 0 }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.find(self.id).is_some()
    }
}

// ----------------------------------------------------------------- the sun

/// mario_actions_stationary.c act_first_person: the warp needs Mario's head
/// yaw plus facing in [0x7000, 0x9000] (facing -z). With the stick's x left
/// alone the head yaw is 0, so his facing must be there; keep a margin.
fn faces_sun(s: &St) -> bool {
    angle_diff(s.yaw as i32, 0x8000).abs() <= 0x0E00
}

fn on_sun_spot(s: &St) -> bool {
    s.level as u32 == LEVEL_CASTLE
        && s.area == 1
        && s.hdist(SUN_SPOT) < SUN_SPOT_RADIUS
        && (s.pos[1] - SUN_SPOT[1]).abs() < 30.0
}

/// `SunSpot::Stand`: stand still in the lobby's sunlight (the
/// SURFACE_LOOK_UP_WARP floor) facing the sun window (-z), ready for
/// `SecretEntrance::Enter(totwc)`. Walking onto it from its +z side leaves
/// Mario facing -z, so the steering goes there first.
pub struct SunStand;

impl Objective for SunStand {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "SunSpot::Stand"
    }
    fn key(&self) -> String {
        "SunSpot stand".to_string()
    }
    fn obj_weight(&self, b: &St) -> f64 {
        8.0 * dest_worth(b, LEVEL_TOTWC)
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_CASTLE && s.area == 1 && s.stars >= 10 && !(on_sun_spot(s) && s.action == ACT_IDLE && faces_sun(s))
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(SUN_SPOT) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| on_sun_spot(s) && s.action == ACT_IDLE && faces_sun(s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| s.level as u32 == LEVEL_CASTLE && s.area == 1)
            .map(|(i, s)| {
                let facing = angle_diff(s.yaw as i32, 0x8000).abs() as f32 / 64.0;
                let still = if s.action == ACT_IDLE { 0.0 } else { 100.0 };
                (i, s.dist(SUN_SPOT) + facing + still)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 200 }
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        if s.pos[2] > SUN_SPOT[2] + 100.0 {
            Some(SUN_SPOT)
        } else {
            Some([SUN_SPOT[0], SUN_SPOT[1], SUN_SPOT[2] + 400.0])
        }
    }
}

/// `SecretEntrance::Enter(totwc)`: from the sun spot, C-up into first
/// person and look up until the head pitch is -0x1800 (camera.c
/// move_mario_head_c_up: head pitch = 3/4 of the C-up pitch, which the stick
/// moves by stickY * 10 per frame down to its -0x2000 limit; the pitch and
/// head yaw reset to 0 when C-up mode starts, set_camera_mode). Then
/// act_first_person warps (WARP_OP_UNKNOWN_01 -> node 0xF2, LEVEL_TOTWC).
pub struct SunLookUp;

impl Objective for SunLookUp {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): warps and entrances: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "SecretEntrance::Enter"
    }
    fn key(&self) -> String {
        "SecretEntrance totwc".to_string()
    }
    fn obj_weight(&self, b: &St) -> f64 {
        12.0 * dest_worth(b, LEVEL_TOTWC)
    }
    fn base_criteria(&self, s: &St) -> bool {
        on_sun_spot(s) && s.action == ACT_IDLE && faces_sun(s) && s.stars >= 10
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        angle_diff(s.yaw as i32, 0x8000).abs() as f64 / 1024.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.level as u32 == LEVEL_TOTWC)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        Progress { score: 0.0, at: traj.len() - 1 }
    }
    fn budget(&self) -> Budget {
        let n = GamePad::new(0, 0, 0);
        let mut pads = vec![n, GamePad::new(U_CBUTTONS, 0, 0)];
        pads.extend(std::iter::repeat_n(n, 20));
        // stickY -64 per frame: -0x2000 takes 13 frames.
        pads.extend(std::iter::repeat_n(GamePad::new(0, 0, -70), 30));
        // The warp's fade (30 frames) and the level load.
        pads.extend(std::iter::repeat_n(n, 120));
        Budget::Macro(pads)
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn temperature(&self) -> f64 {
        0.5
    }
}
