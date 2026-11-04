//! Warps, doors and paintings, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: src/extra/warps.rs, src/extra/warps/table.rs, src/painting.rs
//! (+ painting_table.rs, the data of the paintings).
//!
//! In objective-rrt these are goals ("get through that door"); here each is a
//! gated lens: it is silent unless the goal would apply (the rrt
//! `base_criteria`: zone, stars, keys, course not done ...) and then writes
//! how far Mario is from the nearest candidate (a straight-line distance
//! bucket), where he is (a packed position, only near it), the facing or side
//! where the way in depends on it, and the "it is happening" flag the rrt
//! `success` reads (warping, in a door action, in the cannon ...).
//! The tables (warp pads, paintings, cannon targets, the sun spot) are data
//! copied from objective-rrt and stay tables here.
//!
//! Already covered elsewhere (reused, not duplicated):
//! - `Busy` (busy.rs) makes door animations and dialogs states of their own
//!   (a door action counts as having entered the door, like rrt's `in_door`).
//!   The door kinds here add only the `entering` flag, which says *which*
//!   door (within 300, as rrt's success test) and are otherwise independent.
//! - `Side` (lens.rs) is a fixed plane; the door kinds compute the same
//!   `util::side` from each door's own facing, the painting kind from the
//!   painting's, so one schema node serves every door of the level.
//! - `Lens` (lens.rs) is the way to get fine position/speed around one
//!   hand-picked spot; the kinds here pick their spots from the data.
//!
//! # Mapping of every rrt kind in the sources
//!
//! | rrt kind | scattershot kind |
//! |---|---|
//! | `SecretEntrance::Enter` (table class Secret) | [`WarpEnter`] -> `SecretEntranceEnter` |
//! | `BowserStage::Enter` (class Bowser) | `BowserStageEnter` |
//! | `Teleporter::Enter` (class Teleport, bhvFadingWarp) | `TeleporterEnter` |
//! | `Pipe::Enter` (bhvWarpPipe, incl. the Area pipes) | `PipeEnter` |
//! | `WarpHole::Enter` (bhvWarp) | `WarpHoleEnter` |
//! | `Warp::Enter` (any table warp) | `WarpEnter` |
//! | `SunSpot::Stand` | `SunSpotStand` |
//! | `SecretEntrance::Enter` (key `totwc`, `SunLookUp`) | `SunLookUp` |
//! | `BbhEntrance::Enter` | `BbhEntranceEnter` ([`ObjEnter`]) |
//! | `Tornado::Enter` | `TornadoEnter` ([`ObjEnter`]) |
//! | `StarDoor::Enter` | `StarDoorEnter` ([`DoorEnter`]) |
//! | `KeyDoor::Enter` | `KeyDoorEnter` ([`DoorEnter`]) |
//! | `Painting::Enter` | `PaintingEnter` |
//! | `Cannon::Enter` | `CannonEnter` ([`ObjEnter`]) |
//! | `Cannon::Fire` (far and aimed) | `CannonFire` |
//! | `Moat::Drain` (`PillarPound`) | not ported: dead code in rrt, never generated (replaced by `Pillar::Pound` in switches.rs, which is another family) |
//! | `Door::Enter` (door.rs, not in these sources) | another agent's `DoorEnter`-family kind in door.rs |
//!
//! Not ported from the rrt objective machinery (search, not state):
//! weights (`obj_weight`, `dest_worth`, `success_value`), budgets, steering
//! (`steer_target`, `steer_button`, the cannon's `pads` macro, the sun
//! macro), kinship, `exists_in`, progress scores. The destination (`dest`)
//! is not needed either: arriving in the destination zone is already a
//! different `Root` key (level, area). Not ported either:
//! - warp objects *not in the table* (rrt also offers them from any
//!   INTERACT_WARP object); the table lists every pad the levels place;
//! - the `lblj` waiver of the 8-star requirement of the `bitdw` warp (it is
//!   technique knowledge): use `ignore_req` on a second node instead;
//! - the head pitch of the sun look-up (not in `St`): only `first_person`;
//! - the VCUTM moat-drained requirement (rrt's own NEEDS note, `St` has
//!   no save flag read): the warp is simply always a candidate.

use obrrt_core::layout::ACT_IN_CANNON;
use obrrt_core::{Obj, St, bhv_addr, course_of_level};
use serde::Deserialize;
use serde_json::Value;

use anyhow::Result;

use super::util::{ActionGroup, bucket, dist, pack_pos, side, yaw_bucket};
use crate::objective::{Cx, Objective};

/// Tables copied from objective-rrt/objectives/src/extra/warps/table.rs,
/// src/painting_table.rs and the constants of src/extra/warps.rs.
mod table {
    use obrrt_core::layout::*;
    use serde::Deserialize;

    /// What else (besides being in the zone) entering needs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Req {
        None,
        /// Total stars (the castle's star doors in front of it).
        Stars(i16),
        /// Total stars and a given star (course, index).
        StarsAndStar(i16, i16, u8),
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Class {
        /// A hole / light / hidden warp to a secret level of the castle.
        Secret,
        /// The castle's warps into a Bowser course, and the course pipes to
        /// the Bowser fights.
        Bowser,
        /// A pipe or hole between two areas of one course.
        Area,
        /// A teleporter pad within one area.
        Teleport,
    }

    pub struct Warp {
        pub name: &'static str,
        pub level: u32,
        pub area: u8,
        pub bhv: &'static str,
        pub pos: [i32; 3],
        /// Where it leads (kept as data; arriving is a different `Root` key).
        #[allow(dead_code)]
        pub dest: (u32, u8),
        pub class: Class,
        pub req: Req,
    }

    const fn w(
        name: &'static str,
        level: u32,
        area: u8,
        bhv: &'static str,
        pos: [i32; 3],
        dest: (u32, u8),
        class: Class,
        req: Req,
    ) -> Warp {
        Warp { name, level, area, bhv, pos, dest, class, req }
    }

    use Class::*;
    use Req::*;

    pub const WARPS: &[Warp] = &[
        // Bowser courses' end pipes / hole to the fights.
        w("bowser_1", LEVEL_BITDW, 1, "bhvWarpPipe", [6816, 2860, -7], (LEVEL_BOWSER_1, 1), Bowser, None),
        w("bowser_2", LEVEL_BITFS, 1, "bhvWarp", [6735, 3681, 99], (LEVEL_BOWSER_2, 1), Bowser, None),
        w("bowser_3", LEVEL_BITS, 1, "bhvWarpPipe", [351, 6652, -6030], (LEVEL_BOWSER_3, 1), Bowser, None),
        // The castle's hidden warps (star requirements are the star doors in front of them).
        w("pss", LEVEL_CASTLE, 1, "bhvWarp", [2013, 768, -2014], (LEVEL_PSS, 1), Secret, Stars(1)),
        // Under the trap floor (bhvCastleFloorTrap) behind the 8-star door.
        w("bitdw", LEVEL_CASTLE, 1, "bhvWarp", [-5513, 512, -4324], (LEVEL_BITDW, 1), Bowser, Stars(8)),
        // The aquarium wall in the JRB room.
        w("sa", LEVEL_CASTLE, 1, "bhvWarp", [1963, 819, 1280], (LEVEL_SA, 1), Secret, Stars(3)),
        // Third floor (behind the 50-star door), the light by the window.
        w("wmotr", LEVEL_CASTLE, 2, "bhvWarp", [3002, 2816, 5886], (LEVEL_WMOTR, 1), Secret, Stars(50)),
        // Top of the endless stairs (70-star door).
        w("bits", LEVEL_CASTLE, 2, "bhvWarp", [-230, 4813, -3352], (LEVEL_BITS, 1), Bowser, Stars(70)),
        // Behind the DDD painting (moves back once Bowser's sub is gone: DDD star 1); 30-star door.
        w("bitfs", LEVEL_CASTLE, 3, "bhvWarp", [4147, -1280, 1997], (LEVEL_BITFS, 1), Bowser, StarsAndStar(30, 9, 0)),
        // The moat's bottom; reachable only once the moat is drained.
        w("vcutm", LEVEL_CASTLE_GROUNDS, 1, "bhvWarp", [-3379, -815, -2025], (LEVEL_VCUTM, 1), Secret, None),
        // HMC's pool down to the Cavern of the Metal Cap.
        w("cotmc", LEVEL_HMC, 1, "bhvWarp", [3351, -4690, 4773], (LEVEL_COTMC, 1), Secret, None),
        // Area changes inside courses.
        w("ccm_slide", LEVEL_CCM, 1, "bhvWarp", [-181, 2918, -1486], (LEVEL_CCM, 2), Area, None),
        w("lll_volcano", LEVEL_LLL, 1, "bhvWarp", [0, 105, 0], (LEVEL_LLL, 2), Area, None),
        w("sl_igloo", LEVEL_SL, 1, "bhvWarp", [569, 2150, 1336], (LEVEL_SL, 2), Area, None),
        w("sl_igloo_exit", LEVEL_SL, 2, "bhvWarp", [0, 0, 3277], (LEVEL_SL, 1), Area, None),
        w("ssl_pyramid", LEVEL_SSL, 1, "bhvWarp", [-2048, 0, 56], (LEVEL_SSL, 2), Area, None),
        w("ssl_pyramid_top", LEVEL_SSL, 1, "bhvWarp", [-2048, 768, -1024], (LEVEL_SSL, 2), Area, None),
        w("thi_wiggler", LEVEL_THI, 1, "bhvWarp", [410, -512, 717], (LEVEL_THI, 3), Area, None),
        w("thi_wiggler_top", LEVEL_THI, 1, "bhvWarp", [0, 3170, -1570], (LEVEL_THI, 3), Area, None),
        w("thi_wiggler_exit", LEVEL_THI, 3, "bhvWarp", [512, 1024, 2355], (LEVEL_THI, 1), Area, None),
        w("thi_pipe_a", LEVEL_THI, 1, "bhvWarpPipe", [6656, -1536, -5632], (LEVEL_THI, 2), Area, None),
        w("thi_pipe_b", LEVEL_THI, 1, "bhvWarpPipe", [-5888, -2048, -5888], (LEVEL_THI, 2), Area, None),
        w("thi_pipe_c", LEVEL_THI, 1, "bhvWarpPipe", [-3072, 512, -3840], (LEVEL_THI, 2), Area, None),
        w("thi_pipe_a_back", LEVEL_THI, 2, "bhvWarpPipe", [1997, -461, -1690], (LEVEL_THI, 1), Area, None),
        w("thi_pipe_b_back", LEVEL_THI, 2, "bhvWarpPipe", [-1766, -614, -1766], (LEVEL_THI, 1), Area, None),
        w("thi_pipe_c_back", LEVEL_THI, 2, "bhvWarpPipe", [-922, 154, -1152], (LEVEL_THI, 1), Area, None),
        // Teleporter pads (bhvFadingWarp), in pairs within an area.
        w("bob_tp_a", LEVEL_BOB, 1, "bhvFadingWarp", [583, 2683, -5387], (LEVEL_BOB, 1), Teleport, None),
        w("bob_tp_b", LEVEL_BOB, 1, "bhvFadingWarp", [1680, 3835, -5523], (LEVEL_BOB, 1), Teleport, None),
        w("bob_tp_c", LEVEL_BOB, 1, "bhvFadingWarp", [-6612, 1024, -3351], (LEVEL_BOB, 1), Teleport, None),
        w("bob_tp_d", LEVEL_BOB, 1, "bhvFadingWarp", [1980, 768, 6618], (LEVEL_BOB, 1), Teleport, None),
        w("ccm_tp_a", LEVEL_CCM, 1, "bhvFadingWarp", [-1847, 2815, -321], (LEVEL_CCM, 1), Teleport, None),
        w("ccm_tp_b", LEVEL_CCM, 1, "bhvFadingWarp", [3349, -4694, -183], (LEVEL_CCM, 1), Teleport, None),
        w("lll_tp_a", LEVEL_LLL, 1, "bhvFadingWarp", [-3200, 11, 3456], (LEVEL_LLL, 1), Teleport, None),
        w("lll_tp_b", LEVEL_LLL, 1, "bhvFadingWarp", [-5888, 154, 6656], (LEVEL_LLL, 1), Teleport, None),
        w("rr_tp_a", LEVEL_RR, 1, "bhvFadingWarp", [-7092, 2364, -63], (LEVEL_RR, 1), Teleport, None),
        w("rr_tp_b", LEVEL_RR, 1, "bhvFadingWarp", [-4213, 3379, -2815], (LEVEL_RR, 1), Teleport, None),
        w("sl_tp_a", LEVEL_SL, 1, "bhvFadingWarp", [5468, 1056, -5400], (LEVEL_SL, 1), Teleport, None),
        w("sl_tp_b", LEVEL_SL, 1, "bhvFadingWarp", [-3698, 1024, -1237], (LEVEL_SL, 1), Teleport, None),
        w("ssl_tp_a", LEVEL_SSL, 1, "bhvFadingWarp", [6930, 0, -4871], (LEVEL_SSL, 1), Teleport, None),
        w("ssl_tp_b", LEVEL_SSL, 1, "bhvFadingWarp", [-5943, 0, -4903], (LEVEL_SSL, 1), Teleport, None),
        w("ssl_tp_c", LEVEL_SSL, 2, "bhvFadingWarp", [3070, 1280, 2900], (LEVEL_SSL, 2), Teleport, None),
        w("ssl_tp_d", LEVEL_SSL, 2, "bhvFadingWarp", [2546, 1150, -2647], (LEVEL_SSL, 2), Teleport, None),
        w("thi_tp_a", LEVEL_THI, 2, "bhvFadingWarp", [280, -767, -4180], (LEVEL_THI, 2), Teleport, None),
        w("thi_tp_b", LEVEL_THI, 2, "bhvFadingWarp", [-1638, 0, -1988], (LEVEL_THI, 2), Teleport, None),
        w("ttm_tp_a", LEVEL_TTM, 1, "bhvFadingWarp", [2267, -3006, -3788], (LEVEL_TTM, 1), Teleport, None),
        w("ttm_tp_b", LEVEL_TTM, 1, "bhvFadingWarp", [-557, -3448, -4146], (LEVEL_TTM, 1), Teleport, None),
        w("wdw_tp_a", LEVEL_WDW, 1, "bhvFadingWarp", [818, 0, 3634], (LEVEL_WDW, 1), Teleport, None),
        w("wdw_tp_b", LEVEL_WDW, 1, "bhvFadingWarp", [-2865, 3328, 3065], (LEVEL_WDW, 1), Teleport, None),
        w("wf_tp_a", LEVEL_WF, 1, "bhvFadingWarp", [-2925, 2560, -947], (LEVEL_WF, 1), Teleport, None),
        w("wf_tp_b", LEVEL_WF, 1, "bhvFadingWarp", [2548, 1075, -3962], (LEVEL_WF, 1), Teleport, None),
    ];

    /// Cannon targets worth a dedicated aimed shot: (level, area, name,
    /// point, the star it leads to as (course, index)).
    pub const CANNON_TARGETS: &[(u32, u8, &str, [f32; 3], (i16, u8))] = &[
        // BoB: macro_box_star_3 on the floating island ("Shoot to the Island in the Sky").
        (LEVEL_BOB, 1, "bob_island", [5540.0, 3350.0, 1200.0], (1, 2)),
        // WF: bhvStar behind the wall ("Blast Away the Wall").
        (LEVEL_WF, 1, "wf_wall", [590.0, 2450.0, 2650.0], (2, 5)),
        // RR: macro_box_star_6.
        (LEVEL_RR, 1, "rr_box_6", [5000.0, 4100.0, 4440.0], (15, 5)),
    ];

    /// The floor that warps to the Tower of the Wing Cap when Mario looks up
    /// at the sun from it (SURFACE_LOOK_UP_WARP).
    pub const SUN_SPOT: [f32; 3] = [-1023.0, -50.0, 717.0];
    pub const SUN_SPOT_RADIUS: f32 = 200.0;

    /// A castle painting (painting_table.rs).
    pub struct Painting {
        pub name: &'static str,
        /// Castle area it hangs in: 1 first floor, 2 upstairs, 3 basement.
        pub area: u8,
        /// Level it warps to.
        pub dest: u32,
        pub yaw: f32,
        /// Bottom corner, as in the level data.
        pub corner: [f32; 3],
        pub size: f32,
    }

    impl Painting {
        /// Its centre: the corner plus half its size along its width and height.
        pub fn centre(&self) -> [f32; 3] {
            let yaw = self.yaw.to_radians();
            let half = self.size / 2.0;
            [self.corner[0] + half * yaw.cos(), self.corner[1] + half, self.corner[2] - half * yaw.sin()]
        }
    }

    macro_rules! p {
        ($n:expr, $a:expr, $d:expr, $y:expr, $c:expr, $s:expr) => {
            Painting { name: $n, area: $a, dest: $d, yaw: $y, corner: $c, size: $s }
        };
    }

    pub const PAINTINGS: &[Painting] = &[
        p!("bob", 1, LEVEL_BOB, 90.0, [-5222.4, 409.6, -153.6], 614.0),
        p!("ccm", 1, LEVEL_CCM, 0.0, [-2611.2, -307.2, -4352.0], 614.0),
        p!("wf", 1, LEVEL_WF, 0.0, [-51.2, -204.8, -4505.6], 614.0),
        p!("jrb", 1, LEVEL_JRB, 270.0, [4300.8, 409.6, -537.6], 614.0),
        p!("lll", 3, LEVEL_LLL, 0.0, [-1689.6, -1126.4, -3942.4], 614.0),
        p!("ssl", 3, LEVEL_SSL, 180.0, [-2611.2, -1177.6, -1075.2], 614.0),
        p!("hmc", 3, LEVEL_HMC, 0.0, [2099.2, -1484.8, -2278.4], 768.0),
        p!("ddd", 3, LEVEL_DDD, 270.0, [3456.0, -1075.2, 1587.2], 819.2),
        p!("wdw", 2, LEVEL_WDW, 0.0, [-966.656, 1305.6, -143.36], 614.0),
        p!("thi_tiny", 2, LEVEL_THI, 180.0, [-4598.7842, 1354.752, 3005.44], 393.216),
        p!("ttm", 2, LEVEL_TTM, 180.0, [-546.816, 1356.8, 3813.376], 256.0),
        p!("ttc", 2, LEVEL_TTC, 180.0, [0.0, 2713.6, 7232.5122], 409.6),
        p!("sl", 1, LEVEL_SL, 0.0, [3179.52, 1408.0, -271.36], 716.8),
        p!("thi_huge", 2, LEVEL_THI, 0.0, [-5614.5918, 1510.4, -3292.16], 1638.4),
    ];
}

use table::{CANNON_TARGETS, Class, PAINTINGS, Painting, Req, SUN_SPOT, SUN_SPOT_RADIUS, WARPS, Warp};

// Actions missing from the generated layout (include/sm64.h), as in rrt's warps.rs.
const ACT_DISAPPEARED: u32 = 0x0000_1300;
const ACT_TELEPORT_FADE_OUT: u32 = 0x0000_1336;
const ACT_BBH_ENTER_JUMP: u32 = 0x0000_1934;
const ACT_BBH_ENTER_SPIN: u32 = 0x0000_1535;
const ACT_SHOT_FROM_CANNON: u32 = 0x0088_0898;
const ACT_TORNADO_TWIRLING: u32 = 0x1002_0372;
const ACT_FIRST_PERSON: u32 = 0x0C00_0227;
/// include/object_constants.h BOO_CAGE_ACT_ON_GROUND: the cage has fallen out
/// of the dead Big Boo and is tangible.
const BOO_CAGE_ACT_ON_GROUND: i32 = 2;
const LEVEL_CASTLE: u16 = 6;

fn objs(st: &St) -> &[Obj] {
    st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[])
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn nearest<T>(st: &St, items: impl Iterator<Item = T>, pos: impl Fn(&T) -> [f32; 3]) -> Option<T> {
    items.min_by(|a, b| dist(pos(a), st.pos).total_cmp(&dist(pos(b), st.pos)))
}

/// Standing still: idle, panting, crouching (what a teleporter pad needs).
fn still(action: u32) -> bool {
    matches!(action & 0x1FF, 0x001 | 0x005 | 0x020)
}

/// The yaw (s16 angle) of the ground direction (dx, dz); sm64 yaw 0 faces +z.
fn yaw_of(dx: f32, dz: f32) -> i16 {
    (dx.atan2(dz) / std::f32::consts::TAU * 65536.0).round() as i32 as i16
}

fn default_step() -> f32 {
    200.0
}
fn default_grid() -> u32 {
    7
}
fn default_near() -> f32 {
    1200.0
}

// ------------------------------------------------------------------ warps

/// Pads of the warp table: pipes, holes, teleporters, the castle's secret
/// entrances and the Bowser stage entrances. Active in a zone with candidate
/// pads (those that match the filters and whose requirement `Req` of stars
/// (and star) is met, unless `ignore_req`). Keys on the nearest candidate:
/// `which` (its index in the table), `dist` (straight line, buckets of
/// `step`), within `near` of it `pos` (Mario, grid `grid_log2`), `warping`
/// (Mario is disappearing or fading out at the pad, rrt's success test) and,
/// for teleporters, `still` (idle, panting or crouching, which they need).
///
/// Kinds: `WarpEnter` (all), `SecretEntranceEnter` (class secret),
/// `BowserStageEnter` (bowser), `TeleporterEnter` (teleport), `PipeEnter`
/// (bhvWarpPipe), `WarpHoleEnter` (bhvWarp); params narrow them further.
/// ```json
/// {"kind": "SecretEntranceEnter", "params": {"name": "pss", "step": 200}}
/// ```
#[derive(Deserialize)]
pub struct WarpEnter {
    /// Only the table entry with this name (e.g. `pss`, `thi_tp_a`).
    pub name: Option<String>,
    /// Only these table classes (`secret`, `bowser`, `area`, `teleport`).
    #[serde(default)]
    pub classes: Vec<Class>,
    /// Only this pad behaviour (`bhvWarp`, `bhvWarpPipe`, `bhvFadingWarp`).
    pub bhv: Option<String>,
    /// Offer the pad whatever the star requirement says.
    #[serde(default)]
    pub ignore_req: bool,
    #[serde(default = "default_step")]
    pub step: f32,
    #[serde(default = "default_grid")]
    pub grid_log2: u32,
    /// Distance within which Mario's own position is keyed.
    #[serde(default = "default_near")]
    pub near: f32,
}

fn req_met(req: Req, s: &St) -> bool {
    match req {
        Req::None => true,
        Req::Stars(n) => s.stars >= n,
        Req::StarsAndStar(n, c, i) => s.stars >= n && s.has_star(c, i),
    }
}

/// Touch radius of a pad (warp.inc.c): pipes 70, teleporters 85, holes 50.
fn warp_radius(w: &Warp) -> f32 {
    match w.bhv {
        "bhvWarpPipe" => 70.0,
        "bhvFadingWarp" => 85.0,
        _ => 50.0,
    }
}

fn warp_pos(w: &Warp) -> [f32; 3] {
    [w.pos[0] as f32, w.pos[1] as f32, w.pos[2] as f32]
}

impl WarpEnter {
    fn accepts(&self, w: &Warp) -> bool {
        self.name.as_deref().is_none_or(|n| n == w.name)
            && (self.classes.is_empty() || self.classes.contains(&w.class))
            && self.bhv.as_deref().is_none_or(|b| b == w.bhv)
    }

    /// The nearest candidate pad for `st`: (table index, pad).
    fn pick(&self, st: &St) -> Option<(usize, &'static Warp)> {
        let cands = WARPS.iter().enumerate().filter(|(_, w)| {
            self.accepts(w) && (w.level, w.area) == (st.level as u32, st.area) && (self.ignore_req || req_met(w.req, st))
        });
        nearest(st, cands, |(_, w)| warp_pos(w))
    }
}

impl Objective for WarpEnter {
    fn fields(&self) -> &[&'static str] {
        &["which", "dist", "pos", "warping", "still"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some((i, w)) = self.pick(st) else { return };
        let p = warp_pos(w);
        let d = dist(st.pos, p);
        cx.put(0, i as i32);
        cx.put(1, bucket(d, self.step));
        if d < self.near {
            cx.put(2, pack_pos(st.pos, self.grid_log2));
            if w.class == Class::Teleport {
                cx.put(4, still(st.action) as i32);
            }
        }
        let near_pad = hdist(st.pos, p) < warp_radius(w) + 150.0 && (st.pos[1] - p[1]).abs() < 400.0;
        cx.put(3, (near_pad && (st.action == ACT_DISAPPEARED || st.action == ACT_TELEPORT_FADE_OUT)) as i32);
        cx.all_children();
    }
}

// ------------------------------------------------------------- the sun spot

fn on_sun_spot(s: &St) -> bool {
    s.level == LEVEL_CASTLE
        && s.area == 1
        && hdist(s.pos, SUN_SPOT) < SUN_SPOT_RADIUS
        && (s.pos[1] - SUN_SPOT[1]).abs() < 30.0
}

/// mario_actions_stationary.c act_first_person: the warp needs facing -z
/// (yaw 0x8000), within 0x0E00 as rrt's `faces_sun`.
fn faces_sun(s: &St) -> bool {
    i32::from(s.yaw.wrapping_sub(i16::MIN)).abs() <= 0x0E00
}

#[derive(Deserialize)]
pub struct SunParams {
    #[serde(default = "default_step")]
    pub step: f32,
    #[serde(default = "default_grid")]
    pub grid_log2: u32,
}

/// Stand still on the lobby's sunlight spot facing the sun window (-z):
/// castle lobby with 10+ stars. `dist` (buckets of `step` to the spot); within
/// 3 radii of the spot `pos`, `yaw` (relative to -z: 5 fine buckets within
/// 0x0E00 and 3 coarse) and `still` (idle, panting, crouching).
/// ```json
/// {"kind": "SunSpotStand"}
/// ```
pub struct SunSpotStand(SunParams);

impl Objective for SunSpotStand {
    fn fields(&self) -> &[&'static str] {
        &["dist", "pos", "yaw", "still"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.0.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !(st.level == LEVEL_CASTLE && st.area == 1 && st.stars >= 10) {
            return;
        }
        let d = dist(st.pos, SUN_SPOT);
        cx.put(0, bucket(d, self.0.step));
        if d < 3.0 * SUN_SPOT_RADIUS {
            cx.put(1, pack_pos(st.pos, self.0.grid_log2));
            cx.put(2, yaw_bucket(st.yaw, i16::MIN, 0x0E00, 5, 3));
            cx.put(3, still(st.action) as i32);
        }
        cx.all_children();
    }
}

/// From the sun spot (idle, facing the window, 10+ stars): C-up into first
/// person and look up to warp to the Tower of the Wing Cap. `first_person`
/// (0/1) is what can be seen of it (the head pitch is not in `St`).
/// ```json
/// {"kind": "SunLookUp"}
/// ```
pub struct SunLookUp(SunParams);

impl Objective for SunLookUp {
    fn fields(&self) -> &[&'static str] {
        &["first_person", "pos"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.0.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let fp = st.action == ACT_FIRST_PERSON;
        if !(st.stars >= 10 && on_sun_spot(st) && faces_sun(st) && (fp || still(st.action))) {
            return;
        }
        cx.put(0, fp as i32);
        cx.put(1, pack_pos(st.pos, self.0.grid_log2));
        cx.all_children();
    }
}

// ---------------------------------------------------- doors (stars, keys)

#[derive(Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum DoorKind {
    /// Castle doors with a star count in their 1st behaviour byte.
    Star,
    /// The castle's key doors: warp door 1 (upstairs) needs key 2, door 2 (basement) key 1.
    Key,
}

/// `StarDoor::Enter` and `KeyDoor::Enter`: the nearest castle door of its
/// kind *that Mario can open* (stars >= its count, or the key held; rrt's
/// base criteria). Keys: `need` (the star count or key number, which door),
/// `dist` (buckets of `step`), within `near` `pos` and `side` (which side of
/// the door's plane, from the door's own facing) and `entering` (a door
/// action within 300 of it, rrt's success test).
/// ```json
/// {"kind": "StarDoorEnter", "params": {"step": 200, "near": 1200}}
/// {"kind": "KeyDoorEnter"}
/// ```
#[derive(Deserialize)]
pub struct DoorEnter {
    #[serde(skip, default = "star_kind")]
    kind: DoorKind,
    /// Behaviours of the doors (default `bhvDoor`, `bhvStarDoor` / `bhvDoorWarp`).
    pub bhv: Option<Vec<String>>,
    #[serde(default = "default_step")]
    pub step: f32,
    #[serde(default = "default_grid")]
    pub grid_log2: u32,
    #[serde(default = "default_near")]
    pub near: f32,
}

fn star_kind() -> DoorKind {
    DoorKind::Star
}

impl DoorEnter {
    fn bhvs(&self) -> Vec<u32> {
        let names: Vec<&str> = match (&self.bhv, self.kind) {
            (Some(v), _) => v.iter().map(String::as_str).collect(),
            (None, DoorKind::Star) => vec!["bhvDoor", "bhvStarDoor"],
            (None, DoorKind::Key) => vec!["bhvDoorWarp"],
        };
        names.into_iter().filter_map(bhv_addr).collect()
    }

    /// What opening `o` needs (stars or key number), if it is a door of this kind.
    fn need(&self, o: &Obj) -> Option<i32> {
        let byte = (o.beh_params >> 24) as i32;
        match self.kind {
            DoorKind::Star => (byte > 0).then_some(byte),
            // interaction.c interact_warp_door: door 1 needs key 2, door 2 key 1.
            DoorKind::Key => matches!(byte, 1 | 2).then_some(if byte == 1 { 2 } else { 1 }),
        }
    }

    fn can_open(&self, st: &St, need: i32) -> bool {
        match self.kind {
            DoorKind::Star => i32::from(st.stars) >= need,
            DoorKind::Key => i32::from(st.keys) >= need,
        }
    }
}

impl Objective for DoorEnter {
    fn fields(&self) -> &[&'static str] {
        &["need", "dist", "pos", "side", "entering"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.level != LEVEL_CASTLE {
            return;
        }
        let bhvs = self.bhvs();
        let doors = objs(st)
            .iter()
            .filter(|o| bhvs.contains(&o.bhv))
            .filter_map(|o| self.need(o).filter(|&n| self.can_open(st, n)).map(|n| (o, n)));
        let Some((o, need)) = nearest(st, doors, |(o, _)| o.pos) else { return };
        let d = dist(st.pos, o.pos);
        cx.put(0, need);
        cx.put(1, bucket(d, self.step));
        if d < self.near {
            cx.put(2, pack_pos(st.pos, self.grid_log2));
            let yaw = o.yaw as f32 / 65536.0 * std::f32::consts::TAU;
            cx.put(3, side(st.pos, o.pos, [yaw.sin(), 0.0, yaw.cos()]));
        }
        cx.put(4, (ActionGroup::Door.contains(st.action) && d < 300.0) as i32);
        cx.all_children();
    }
}

// ------------------------------------------- objects to touch: cage, tornado, cannon

#[derive(Deserialize, Clone, Copy, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ObjKind {
    /// The boo cage of the courtyard's Big Boo, on the ground (`BbhEntrance::Enter`).
    BbhCage,
    /// A Tweester (`Tornado::Enter`); it wanders, so its position is keyed.
    Tornado,
    /// An open cannon (`Cannon::Enter`).
    Cannon,
}

/// `BbhEntranceEnter`, `TornadoEnter`, `CannonEnter`: touch an object that
/// takes Mario somewhere. Active while such an object (`bhv`, nearest) is
/// loaded (the cage only once `action` is 2, on the ground). Keys: `dist` to
/// it (buckets of `step`), within `near` `pos`, `caught` (the success test:
/// a BBH-entering jump or spin / in the tornado's twirl / in the cannon
/// within 300) and, for the tornado, `obj_pos` (where it is, same grid).
/// ```json
/// {"kind": "BbhEntranceEnter"}
/// {"kind": "TornadoEnter", "params": {"grid_log2": 8}}
/// {"kind": "CannonEnter"}
/// ```
#[derive(Deserialize)]
pub struct ObjEnter {
    #[serde(skip, default = "cage_kind")]
    kind: ObjKind,
    pub bhv: Option<String>,
    #[serde(default = "default_step")]
    pub step: f32,
    #[serde(default = "default_grid")]
    pub grid_log2: u32,
    #[serde(default = "default_near")]
    pub near: f32,
}

fn cage_kind() -> ObjKind {
    ObjKind::BbhCage
}

impl ObjEnter {
    fn bhv(&self) -> Option<u32> {
        let default = match self.kind {
            ObjKind::BbhCage => "bhvBooCage",
            ObjKind::Tornado => "bhvTweester",
            ObjKind::Cannon => "bhvCannon",
        };
        bhv_addr(self.bhv.as_deref().unwrap_or(default))
    }

    fn caught(&self, st: &St, p: [f32; 3]) -> bool {
        match self.kind {
            ObjKind::BbhCage => matches!(st.action, ACT_BBH_ENTER_JUMP | ACT_BBH_ENTER_SPIN),
            ObjKind::Tornado => st.action == ACT_TORNADO_TWIRLING,
            ObjKind::Cannon => st.action == ACT_IN_CANNON && hdist(st.pos, p) < 300.0,
        }
    }
}

impl Objective for ObjEnter {
    fn fields(&self) -> &[&'static str] {
        &["dist", "pos", "caught", "obj_pos"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(bhv) = self.bhv() else { return };
        let cands = objs(st)
            .iter()
            .filter(|o| o.bhv == bhv && (self.kind != ObjKind::BbhCage || o.action == BOO_CAGE_ACT_ON_GROUND));
        // A cannon stays where it spawned (`home`).
        let at = |o: &Obj| if self.kind == ObjKind::Cannon { o.home } else { o.pos };
        let Some(o) = nearest(st, cands, |o| at(o)) else { return };
        let p = at(o);
        let d = dist(st.pos, p);
        cx.put(0, bucket(d, self.step));
        if d < self.near {
            cx.put(1, pack_pos(st.pos, self.grid_log2));
        }
        cx.put(2, self.caught(st, p) as i32);
        if self.kind == ObjKind::Tornado {
            cx.put(3, pack_pos(p, self.grid_log2));
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- cannon fire

/// Mario's flight out of a cannon aimed at (yaw, pitch): its closest approach
/// to `target`. mario_actions_automatic.c act_in_cannon: he starts 350 above
/// the cannon base plus 120 along the barrel, with forward speed
/// 100 cos(pitch) and vertical speed 100 sin(pitch); gravity is 1 per frame
/// down to -75 (copied from rrt's warps.rs `flight_miss`).
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

#[derive(Deserialize)]
pub struct CannonFire {
    /// Yaw sectors of the aim (Mario's facing in the cannon).
    #[serde(default = "sectors64")]
    pub yaw_sectors: u32,
    /// Pitch sectors of the aim over the 0..0x38E3 the cannon allows.
    #[serde(default = "sectors16")]
    pub pitch_sectors: u32,
    /// Width of a bucket of the predicted miss distance to the nearest
    /// not-yet-collected target of `CANNON_TARGETS`.
    #[serde(default = "miss_step")]
    pub miss_step: f32,
}

fn sectors64() -> u32 {
    64
}
fn sectors16() -> u32 {
    16
}
fn miss_step() -> f32 {
    250.0
}
fn flight_step() -> f32 {
    500.0
}

/// `Cannon::Fire` (aimed and far): inside a cannon (`ACT_IN_CANNON`, within
/// 250 of a `bhvCannon`) keys the aim: `yaw`, `pitch` (sectors), and `miss`,
/// the simulated flight's closest approach to the nearest known target whose
/// star is missing in this zone (-1 without one), in buckets of `miss_step`
/// (rrt solves the aim by this simulation; here it is what tells aims apart).
/// The flight after the shot is `CannonFlight` (`flight`: horizontal distance
/// from the cannon in buckets of `flight_step`, and `pos`).
/// ```json
/// {"kind": "CannonFire", "params": {"yaw_sectors": 64}}
/// ```
impl Objective for CannonFire {
    fn fields(&self) -> &[&'static str] {
        &["yaw", "pitch", "miss"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let Some(bhv) = bhv_addr("bhvCannon") else { return };
        let cannon = objs(st).iter().filter(|o| o.bhv == bhv).map(|o| o.home).filter(|&h| hdist(st.pos, h) < 250.0);
        let Some(home) = nearest(st, cannon, |&h| h) else { return };
        if st.action == ACT_IN_CANNON {
            cx.put(0, ((st.yaw as u16 as u32 * self.yaw_sectors) >> 16) as i32);
            let pitch = i32::from(st.pitch).clamp(0, 0x38E3);
            cx.put(1, pitch * self.pitch_sectors as i32 / 0x38E4);
            let yaw = i32::from(st.yaw as u16);
            let miss = CANNON_TARGETS
                .iter()
                .filter(|t| (t.0, t.1) == (st.level as u32, st.area) && !st.has_star(t.4.0, t.4.1))
                .map(|t| flight_miss(home, yaw, pitch, t.3))
                .min_by(f32::total_cmp);
            cx.put(2, miss.map_or(-1, |m| bucket(m, self.miss_step)));
            cx.all_children();
        }
    }
}

/// The flight half of `Cannon::Fire` (the "far" flavour): after the shot
/// (`ACT_SHOT_FROM_CANNON`) keys how far Mario is from the nearest cannon.
/// ```json
/// {"kind": "CannonFlight", "params": {"flight_step": 500}}
/// ```
#[derive(Deserialize)]
pub struct CannonFlight {
    #[serde(default = "flight_step")]
    pub flight_step: f32,
    #[serde(default = "default_grid")]
    pub grid_log2: u32,
}

impl Objective for CannonFlight {
    fn fields(&self) -> &[&'static str] {
        &["flight", "pos"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if st.action != ACT_SHOT_FROM_CANNON {
            return;
        }
        let Some(bhv) = bhv_addr("bhvCannon") else { return };
        let Some(home) = nearest(st, objs(st).iter().filter(|o| o.bhv == bhv).map(|o| o.home), |&h| h) else { return };
        cx.put(0, bucket(hdist(st.pos, home), self.flight_step));
        cx.put(1, pack_pos(st.pos, self.grid_log2));
        cx.all_children();
    }
}

// ---------------------------------------------------------------- paintings

/// Stars needed to reach a painting's room: the star doors in front of them
/// (WF's door 1, JRB's and CCM's 3, DDD's in the basement 30, the third floor 50).
fn stars_needed(name: &str) -> i16 {
    match name {
        "wf" => 1,
        "jrb" | "ccm" => 3,
        "ddd" => 30,
        "ttc" | "rr" => 50,
        _ => 0,
    }
}

/// The room in front of a painting behind a star door: (depth of the room
/// plus the door's approach, depth past the door, half width, top y). Only
/// WF's is known (rrt painting.rs `room`).
fn room(name: &str) -> Option<(f32, f32, f32, f32)> {
    match name {
        "wf" => Some((3900.0, 3300.0, 900.0, 450.0)),
        _ => None,
    }
}

/// (Distance in front of the painting's plane, distance across it) of `p`.
fn facing(p: &Painting, centre: [f32; 3], at: [f32; 3]) -> (f32, f32) {
    let yaw = p.yaw.to_radians();
    let (nx, nz) = (yaw.sin(), yaw.cos());
    let (rx, rz) = (at[0] - centre[0], at[2] - centre[2]);
    (rx * nx + rz * nz, (rx * nz - rz * nx).abs())
}

#[derive(Deserialize)]
pub struct PaintingEnter {
    /// Only this painting of the table (`bob`, `wf`, `thi_huge`, ...).
    pub name: Option<String>,
    /// Active only in front of the painting (ahead of the wall, within a cone
    /// widening from its width; behind a star door only in its room or with
    /// the stars): rrt's `OBRRT_PAINT_FRONT=1`.
    #[serde(default = "yes")]
    pub front: bool,
    #[serde(default = "default_step")]
    pub step: f32,
    #[serde(default = "default_grid")]
    pub grid_log2: u32,
    /// Distance within which Mario's position and facing are keyed.
    #[serde(default = "default_near")]
    pub near: f32,
}

fn yes() -> bool {
    true
}

impl PaintingEnter {
    /// Whether the rrt `Painting::Enter` would apply to this painting from `s`.
    fn applies(&self, p: &Painting, s: &St) -> bool {
        if !(s.level == LEVEL_CASTLE && s.area == p.area && !s.course_done(course_of_level(p.dest))) {
            return false;
        }
        if !self.front {
            return s.stars >= stars_needed(p.name);
        }
        let (front, across) = facing(p, p.centre(), s.pos);
        if front < -50.0 || across > p.size + 1.5 * front {
            return false;
        }
        match room(p.name) {
            Some((approach, depth, half, top)) => {
                across < half && front < approach && s.pos[1] < top && (front < depth || s.stars >= stars_needed(p.name))
            }
            None => s.stars >= stars_needed(p.name),
        }
    }
}

/// `Painting::Enter`: jump into a castle painting. Active in the castle area
/// the painting hangs in while its course is not done, Mario has the stars of
/// the door in front of it and (`front`) he is in front of it. Keys on the
/// nearest such painting: `which` (table index), `dist` to its centre
/// (buckets of `step`), within `near` `pos` and `yaw` (Mario's facing
/// relative to straight into the painting: 5 buckets within 45 degrees, 3
/// elsewhere).
/// ```json
/// {"kind": "PaintingEnter", "params": {"name": "wf", "step": 200}}
/// ```
impl Objective for PaintingEnter {
    fn fields(&self) -> &[&'static str] {
        &["which", "dist", "pos", "yaw"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let cands = PAINTINGS
            .iter()
            .enumerate()
            .filter(|(_, p)| self.name.as_deref().is_none_or(|n| n == p.name) && self.applies(p, st));
        let Some((i, p)) = nearest(st, cands, |(_, p)| p.centre()) else { return };
        let c = p.centre();
        let d = dist(st.pos, c);
        cx.put(0, i as i32);
        cx.put(1, bucket(d, self.step));
        if d < self.near {
            cx.put(2, pack_pos(st.pos, self.grid_log2));
            let a = p.yaw.to_radians();
            // Entering faces against the painting's front normal.
            cx.put(3, yaw_bucket(st.yaw, yaw_of(-a.sin(), -a.cos()), 0x2000, 5, 3));
        }
        cx.all_children();
    }
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    fn warp(p: &Value, classes: &[Class], bhv: Option<&str>) -> Result<Box<dyn Objective>> {
        let mut w: WarpEnter = serde_json::from_value(p.clone())?;
        if w.classes.is_empty() {
            w.classes = classes.to_vec();
        }
        if w.bhv.is_none() {
            w.bhv = bhv.map(String::from);
        }
        Ok(Box::new(w))
    }
    fn door(p: &Value, kind: DoorKind) -> Result<Box<dyn Objective>> {
        let mut d: DoorEnter = serde_json::from_value(p.clone())?;
        d.kind = kind;
        Ok(Box::new(d))
    }
    fn obj(p: &Value, kind: ObjKind) -> Result<Box<dyn Objective>> {
        let mut d: ObjEnter = serde_json::from_value(p.clone())?;
        d.kind = kind;
        Ok(Box::new(d))
    }
    fn parse<T: Objective + serde::de::DeserializeOwned + 'static>(p: &Value) -> Result<Box<dyn Objective>> {
        Ok(Box::new(serde_json::from_value::<T>(p.clone())?))
    }
    let p = params;
    Some(match kind {
        "WarpEnter" => warp(p, &[], None),
        "SecretEntranceEnter" => warp(p, &[Class::Secret], None),
        "BowserStageEnter" => warp(p, &[Class::Bowser], None),
        "TeleporterEnter" => warp(p, &[Class::Teleport], Some("bhvFadingWarp")),
        "PipeEnter" => warp(p, &[], Some("bhvWarpPipe")),
        "WarpHoleEnter" => warp(p, &[], Some("bhvWarp")),
        "SunSpotStand" => serde_json::from_value(p.clone()).map(|s| Box::new(SunSpotStand(s)) as _).map_err(Into::into),
        "SunLookUp" => serde_json::from_value(p.clone()).map(|s| Box::new(SunLookUp(s)) as _).map_err(Into::into),
        "StarDoorEnter" => door(p, DoorKind::Star),
        "KeyDoorEnter" => door(p, DoorKind::Key),
        "BbhEntranceEnter" => obj(p, ObjKind::BbhCage),
        "TornadoEnter" => obj(p, ObjKind::Tornado),
        "CannonEnter" => obj(p, ObjKind::Cannon),
        "CannonFire" => parse::<CannonFire>(p),
        "CannonFlight" => parse::<CannonFlight>(p),
        "PaintingEnter" => parse::<PaintingEnter>(p),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn schema(kind: &str, params: &str) -> String {
        format!(r#"{{"kind":"Root","children":[{{"kind":"{kind}","params":{params}}}]}}"#)
    }

    fn key(kind: &str, params: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(&schema(kind, params), 1, s)
    }

    fn in_castle(pos: [f32; 3], area: u8, stars: i16) -> St {
        let mut s = st(pos, 0.0, 6, area, 0);
        s.stars = stars;
        s
    }

    #[test]
    fn warp_gate_and_requirement() {
        // The PSS pad (2013, 768, -2014) needs 1 star.
        let mut s = in_castle([2013.0, 768.0, -1500.0], 1, 0);
        assert!(key("SecretEntranceEnter", "{}", &s).is_empty());
        s.stars = 1;
        assert!(!key("SecretEntranceEnter", "{\"name\":\"pss\"}", &s).is_empty());
        // ignore_req offers it without the stars.
        s.stars = 0;
        assert!(!key("SecretEntranceEnter", r#"{"ignore_req":true,"name":"pss"}"#, &s).is_empty());
        // Wrong zone: silent.
        let s2 = st([2013.0, 768.0, -1500.0], 0.0, 9, 1, 0);
        assert!(key("WarpEnter", "{}", &s2).iter().all(|f| f.0 != 99)); // BoB has teleporters only
        let s3 = st([0.0, 0.0, 0.0], 0.0, 12, 1, 0); // JRB: no pads
        assert!(key("WarpEnter", "{}", &s3).is_empty());
    }

    #[test]
    fn warp_kind_filters() {
        // THI area 1 has holes (bhvWarp) and pipes (bhvWarpPipe).
        let s = st([6600.0, -1536.0, -5600.0], 0.0, 13, 1, 0);
        let pipe = key("PipeEnter", "{}", &s);
        let hole = key("WarpHoleEnter", "{}", &s);
        let tele = key("TeleporterEnter", "{}", &s);
        assert!(!pipe.is_empty() && !hole.is_empty());
        assert!(tele.is_empty(), "no teleporters in THI area 1");
        // Nearest pad is thi_pipe_a for the pipe kind.
        let idx = |name: &str| WARPS.iter().position(|w| w.name == name).unwrap() as i32;
        assert_eq!(pipe[0], (0, idx("thi_pipe_a")));
    }

    #[test]
    fn warp_dist_buckets_and_pos() {
        let at = |dx: f32| st([583.0 + dx, 2683.0, -5387.0], 0.0, 9, 1, 0);
        let d0 = key("TeleporterEnter", "{\"name\":\"bob_tp_a\"}", &at(0.0));
        let d1 = key("TeleporterEnter", "{\"name\":\"bob_tp_a\"}", &at(199.0));
        let d2 = key("TeleporterEnter", "{\"name\":\"bob_tp_a\"}", &at(201.0));
        assert_eq!(d0[1], (1, 0));
        assert_eq!(d1[1], (1, 0));
        assert_eq!(d2[1], (1, 1));
        // Near: pos is keyed, far: it is not.
        assert!(d0.iter().any(|f| f.0 == 2));
        let far = key("TeleporterEnter", "{\"name\":\"bob_tp_a\"}", &at(3000.0));
        assert!(far.iter().all(|f| f.0 != 2));
    }

    #[test]
    fn teleporter_still_and_warping() {
        let walking = st([583.0, 2683.0, -5387.0], 0.0, 9, 1, 0);
        let mut idle = walking.clone();
        idle.action = 0x0C40_0201;
        let mut fade = walking.clone();
        fade.action = ACT_TELEPORT_FADE_OUT;
        let k = |s: &St| key("TeleporterEnter", "{\"name\":\"bob_tp_a\"}", s);
        assert_ne!(k(&walking), k(&idle), "standing still on a pad is another cell");
        assert_ne!(k(&idle), k(&fade));
        assert!(k(&fade).contains(&(3, 1)));
        assert!(k(&walking).contains(&(3, 0)));
        // Fading far from the pad is not "warping" at it.
        let mut away = fade.clone();
        away.pos[0] += 1000.0;
        assert!(k(&away).contains(&(3, 0)));
    }

    #[test]
    fn bowser_stage_via_star_requirement_and_extra_star() {
        // BitFS warp needs 30 stars and DDD star 0 (course 9 index 0).
        let mut s = in_castle([4147.0, -1280.0, 1900.0], 3, 40);
        assert!(key("BowserStageEnter", "{}", &s).is_empty());
        s.course_stars[8] = 1;
        assert!(!key("BowserStageEnter", "{}", &s).is_empty());
    }

    #[test]
    fn star_door_needs_stars_and_picks_nearest_openable() {
        let door = |slot, x, stars: u32| {
            let mut o = obj("bhvStarDoor", slot, [x, 0.0, -500.0]);
            o.beh_params = stars << 24;
            o
        };
        let objs = vec![door(1, 0.0, 8), door(2, 1000.0, 1)];
        let s = |stars| with_objects(in_castle([100.0, 0.0, -300.0], 1, stars), objs.clone());
        // 0 stars: silent. 1 star: only the far 1-star door counts. 8: the near one.
        assert!(key("StarDoorEnter", "{}", &s(0)).is_empty());
        assert_eq!(key("StarDoorEnter", "{}", &s(1))[0], (0, 1));
        assert_eq!(key("StarDoorEnter", "{}", &s(8))[0], (0, 8));
        assert_ne!(key("StarDoorEnter", "{}", &s(1)), key("StarDoorEnter", "{}", &s(8)));
        // Outside the castle: silent.
        let outside = with_objects(st([100.0, 0.0, -300.0], 0.0, 16, 1, 0), objs.clone());
        assert!(key("StarDoorEnter", "{}", &outside).is_empty());
    }

    #[test]
    fn door_side_entering_and_busy_overlap() {
        let mut d = obj("bhvStarDoor", 1, [0.0, 0.0, 0.0]);
        d.beh_params = 1 << 24;
        d.yaw = 0; // faces +z
        let at = |z: f32, action: u32| {
            let mut s = in_castle([0.0, 0.0, z], 1, 5);
            s.action = action;
            with_objects(s, vec![d.clone()])
        };
        let front = key("StarDoorEnter", "{}", &at(200.0, 0x0400_0440));
        let back = key("StarDoorEnter", "{}", &at(-200.0, 0x0400_0440));
        assert!(front.contains(&(3, 1)) && back.contains(&(3, 0)));
        let opening = key("StarDoorEnter", "{}", &at(200.0, 0x0000_1320));
        assert!(opening.contains(&(4, 1)) && front.contains(&(4, 0)));
        // Beyond `near`, no pos / side.
        let far = key("StarDoorEnter", "{}", &at(5000.0, 0x0400_0440));
        assert!(far.iter().all(|f| f.0 != 2 && f.0 != 3));
    }

    #[test]
    fn key_doors() {
        let door = |param: u32| {
            let mut o = obj("bhvDoorWarp", 1, [0.0, 0.0, 0.0]);
            o.beh_params = param << 24;
            o
        };
        let s = |param, keys| {
            let mut s = in_castle([0.0, 0.0, 300.0], 1, 0);
            s.keys = keys;
            with_objects(s, vec![door(param)])
        };
        // Door 2 (basement) needs key 1; door 1 (upstairs) key 2; others are not key doors.
        assert!(key("KeyDoorEnter", "{}", &s(2, 0)).is_empty());
        assert_eq!(key("KeyDoorEnter", "{}", &s(2, 1))[0], (0, 1));
        assert!(key("KeyDoorEnter", "{}", &s(1, 1)).is_empty());
        assert_eq!(key("KeyDoorEnter", "{}", &s(1, 2))[0], (0, 2));
        assert!(key("KeyDoorEnter", "{}", &s(3, 3)).is_empty());
    }

    #[test]
    fn bbh_cage_only_on_the_ground() {
        let cage = |action| {
            let mut o = obj("bhvBooCage", 3, [0.0, 0.0, 0.0]);
            o.action = action;
            o
        };
        let s = |action| with_objects(st([300.0, 0.0, 0.0], 0.0, 26, 1, 0), vec![cage(action)]);
        assert!(key("BbhEntranceEnter", "{}", &s(1)).is_empty());
        let on = key("BbhEntranceEnter", "{}", &s(2));
        assert_eq!(on[0], (0, 1));
        let mut jumping = s(2);
        jumping.action = ACT_BBH_ENTER_JUMP;
        assert!(key("BbhEntranceEnter", "{}", &jumping).contains(&(2, 1)));
        assert!(on.contains(&(2, 0)));
    }

    #[test]
    fn tornado_follows_the_nearest_and_its_position() {
        let t = |slot, pos| obj("bhvTweester", slot, pos);
        let s = |objs| with_objects(st([0.0, 0.0, 0.0], 0.0, 8, 1, 0), objs);
        let a = key("TornadoEnter", "{}", &s(vec![t(1, [500.0, 0.0, 0.0]), t(2, [5000.0, 0.0, 0.0])]));
        let b = key("TornadoEnter", "{}", &s(vec![t(1, [1700.0, 0.0, 0.0])]));
        assert_ne!(a, b, "the tornado being elsewhere is another cell");
        let c = key("TornadoEnter", "{}", &s(vec![t(2, [500.0, 0.0, 0.0])]));
        assert_eq!(a, c, "the same nearest position is the same cell");
        let mut twirl = s(vec![t(1, [500.0, 0.0, 0.0])]);
        twirl.action = ACT_TORNADO_TWIRLING;
        assert_ne!(key("TornadoEnter", "{}", &twirl), a);
        assert!(key("TornadoEnter", "{}", &st([0.0; 3], 0.0, 8, 1, 0)).is_empty());
    }

    #[test]
    fn cannon_enter_and_fire() {
        let mut cannon = obj("bhvCannon", 4, [0.0, 0.0, 0.0]);
        cannon.home = [0.0, 0.0, 0.0];
        let mk = |pos, action, yaw: i16, pitch: i16| {
            let mut s = st(pos, 0.0, 9, 1, 0);
            s.action = action;
            s.yaw = yaw;
            s.pitch = pitch;
            with_objects(s, vec![cannon.clone()])
        };
        let walking = mk([300.0, 0.0, 0.0], 0x0400_0440, 0, 0);
        let inside = mk([0.0, 0.0, 0.0], ACT_IN_CANNON, 0, 0);
        assert!(key("CannonEnter", "{}", &walking).contains(&(2, 0)));
        assert!(key("CannonEnter", "{}", &inside).contains(&(2, 1)));
        // Fire: silent outside the cannon, aims differ inside.
        assert!(key("CannonFire", "{}", &walking).is_empty());
        let a = key("CannonFire", "{}", &inside);
        let b = key("CannonFire", "{}", &mk([0.0, 0.0, 0.0], ACT_IN_CANNON, 0x4000, 0));
        let c = key("CannonFire", "{}", &mk([0.0, 0.0, 0.0], ACT_IN_CANNON, 0, 0x2000));
        assert_ne!(a, b);
        assert_ne!(a, c);
        // Same sector, same cell.
        assert_eq!(a, key("CannonFire", "{}", &mk([0.0, 0.0, 0.0], ACT_IN_CANNON, 100, 50)));
        // In BoB with the island star missing, `miss` is a real distance (not -1).
        assert!(a.iter().any(|f| f.0 == 2 && f.1 >= 0));
        // Flight: bucketed distance from the cannon.
        let shot = mk([1200.0, 100.0, 0.0], ACT_SHOT_FROM_CANNON, 0, 0);
        assert!(key("CannonFlight", "{}", &shot).contains(&(0, 2)));
        assert!(key("CannonFlight", "{}", &inside).is_empty());
    }

    #[test]
    fn flight_miss_prefers_the_right_aim() {
        // Aiming at a point 2000 away along +z: level aim misses by more than the aimed-up one.
        let target = [0.0, 1500.0, 3000.0];
        let best = (0..=0x38E3).step_by(64).map(|p| flight_miss([0.0; 3], 0, p, target)).fold(f32::MAX, f32::min);
        assert!(best < flight_miss([0.0; 3], 0, 0, target));
        assert!(best < 300.0);
        // Aiming the wrong way misses.
        assert!(flight_miss([0.0; 3], 0x8000, 0x1000, target) > best);
    }

    #[test]
    fn paintings_front_and_stars() {
        // BoB painting (faces +x at yaw 90): centre about (-5222+0, ..., -153.6 - 307).
        let p = PAINTINGS.iter().find(|p| p.name == "bob").unwrap();
        let c = p.centre();
        let front = in_castle([c[0] + 800.0, c[1], c[2]], 1, 0);
        let behind = in_castle([c[0] - 800.0, c[1], c[2]], 1, 0);
        assert!(!key("PaintingEnter", "{\"name\":\"bob\"}", &front).is_empty());
        assert!(key("PaintingEnter", "{\"name\":\"bob\"}", &behind).is_empty());
        // Without `front`, behind counts too.
        assert!(!key("PaintingEnter", "{\"name\":\"bob\",\"front\":false}", &behind).is_empty());
        // Wrong area: silent. Course done: silent.
        let up = in_castle([c[0] + 800.0, c[1], c[2]], 2, 0);
        assert!(key("PaintingEnter", "{\"name\":\"bob\"}", &up).is_empty());
        let mut done = front.clone();
        done.course_stars[0] = 0x7F;
        assert!(key("PaintingEnter", "{\"name\":\"bob\"}", &done).is_empty());
        // JRB (3 stars) needs them.
        let j = PAINTINGS.iter().find(|p| p.name == "jrb").unwrap().centre();
        let mut s = in_castle([j[0] - 800.0, j[1], j[2]], 1, 2);
        assert!(key("PaintingEnter", "{\"name\":\"jrb\"}", &s).is_empty());
        s.stars = 3;
        assert!(!key("PaintingEnter", "{\"name\":\"jrb\"}", &s).is_empty());
    }

    #[test]
    fn painting_yaw_and_nearest() {
        let p = PAINTINGS.iter().find(|p| p.name == "bob").unwrap();
        let c = p.centre();
        let mut s = in_castle([c[0] + 500.0, c[1], c[2]], 1, 0);
        // The painting faces +x; entering means facing -x (yaw 0xC000).
        s.yaw = 0xC000u16 as i16;
        let face = key("PaintingEnter", "{}", &s);
        s.yaw = 0x4000;
        let away = key("PaintingEnter", "{}", &s);
        assert_ne!(face, away);
        s.yaw = 0xC000u16 as i16 + 100;
        assert_eq!(face, key("PaintingEnter", "{}", &s));
        // Without a name the nearest painting is chosen.
        assert_eq!(face[0], (0, PAINTINGS.iter().position(|q| q.name == "bob").unwrap() as i32));
    }

    #[test]
    fn sun_spot() {
        let on = |yaw: i16, action: u32, stars: i16| {
            let mut s = in_castle([SUN_SPOT[0], SUN_SPOT[1], SUN_SPOT[2]], 1, stars);
            s.yaw = yaw;
            s.action = action;
            s
        };
        let idle = 0x0C40_0201;
        assert!(key("SunSpotStand", "{}", &on(i16::MIN, idle, 9)).is_empty(), "needs 10 stars");
        let good = key("SunSpotStand", "{}", &on(i16::MIN, idle, 10));
        assert!(!good.is_empty());
        assert_ne!(good, key("SunSpotStand", "{}", &on(0, idle, 10)), "facing away is another cell");
        assert_ne!(good, key("SunSpotStand", "{}", &on(i16::MIN, 0x0400_0440, 10)), "walking is another cell");
        // Look up: only from idle/first person on the spot facing -z.
        assert!(!key("SunLookUp", "{}", &on(i16::MIN, idle, 10)).is_empty());
        assert!(key("SunLookUp", "{}", &on(0, idle, 10)).is_empty());
        assert!(key("SunLookUp", "{}", &on(i16::MIN, 0x0400_0440, 10)).is_empty());
        let fp = key("SunLookUp", "{}", &on(i16::MIN, ACT_FIRST_PERSON, 10));
        assert_ne!(fp, key("SunLookUp", "{}", &on(i16::MIN, idle, 10)));
        let mut off = on(i16::MIN, idle, 10);
        off.pos[0] += 500.0;
        assert!(key("SunLookUp", "{}", &off).is_empty());
    }

    #[test]
    fn unknown_kinds_are_not_ours() {
        assert!(build("Lens", &Value::Null).is_none());
        assert!(build("DoorEnter", &Value::Null).is_none());
    }

    #[test]
    fn table_is_consistent() {
        for w in WARPS {
            assert!(["bhvWarp", "bhvWarpPipe", "bhvFadingWarp"].contains(&w.bhv), "{}", w.name);
            assert!(bhv_addr(w.bhv).is_some());
        }
        for n in ["bhvStarDoor", "bhvDoor", "bhvDoorWarp", "bhvBooCage", "bhvTweester", "bhvCannon"] {
            assert!(bhv_addr(n).is_some(), "{n}");
        }
    }
}
