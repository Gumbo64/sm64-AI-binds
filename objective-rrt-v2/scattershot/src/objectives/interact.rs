//! Switches, interactables and rides, ported from objective-rrt/objectives (see PORTING.md).
//!
//! Sources: `objective-rrt/objectives/src/extra/switches.rs` (`Switch`, kinds `K`),
//! `extra/rides/{platform,chest,ring,shell,common}.rs`.
//!
//! Every kind here is a *state distinction* writer: it finds the nearest
//! matching object, and keys on how far Mario is from it (bucketed straight-line
//! distance), the object's state (what the rrt `ready`/`done`/`success` read) and
//! a `done` flag. They write nothing outside their `level`/`area` (when given),
//! once their `star` (`[course, index]`) is collected, or when no matching
//! object is loaded. All need objects. Common params (all optional): `level`,
//! `area`, `star`, `bhv` (a list of behaviour names replacing the default),
//! `step` (distance per bucket), `grid_log2` (for `tpos`/`ppos`).
//!
//! | rrt kind | scattershot kind |
//! |---|---|
//! | `PurpleSwitch::Press` | `PurpleSwitchPress` (state = oAction, done = 1/2/4) |
//! | `CapSwitch::Press` | `CapSwitchPress` (state = oAction, done >= 2) |
//! | `BlueCoinSwitch::Pound` | `BlueCoinSwitchPound` (done = oAction >= 1) |
//! | `Pillar::Pound` | `PillarPound` (done = pounded or sunk; level default Castle) |
//! | `Post::Pound` | `PostPound` (state = pound depth in 40s; BoB, star 1/5; the post the Chain Chomp is chained to) |
//! | `Post::Circle` | `PostCircle` (state = coins left, `ang` = octant around the post within 400) |
//! | `IslandTop::Pound` | `IslandTopPound` (THI, star 13/5) |
//! | `WaterDiamond::Touch` | `WaterDiamondTouch` (done = oAction >= 2) |
//! | `PyramidPillar::Touch` | `PyramidPillarTouch` (SSL, star 8/3; `extra` = detectors left, done = none left) |
//! | `Book::Punch` | `BookPunch` (param `index` 0-2; BBH, star 5/2; `extra` = bitmask of books sticking out) |
//! | `Block::Push` | `BlockPush` (`tpos` = the box's position, so a pushed box is another cell) |
//! | `Pole::Climb` | `PoleClimb` (any INTERACT_POLE object; state = 0 off the pole, 1+ height in 100s on it; done = at the top) |
//! | `Platform::Board` | `PlatformBoard` (`dist` to the platform top, `on`) |
//! | `Platform::Ride` | `PlatformRide` (as Board, plus `ppos` = the platform's position while Mario is on it, so being carried makes new cells) |
//! | `Seesaw::Board` | `SeesawBoard` (Board restricted to the seesaw table) |
//! | `Chest::Open` | `ChestOpen` (param `set` = JrbCave, JrbShip or Ddd; keys the next chest) |
//! | `WaterRing::SwimThrough` | `WaterRingSwimThrough` |
//! | `Shell::Ride` | `ShellRide` |
//!
//! Not ported: the rrt `precision`, `budget`, `success_value`, `obj_weight`,
//! `kinship`, `steer_*` and `progress` machinery (search, not state); the
//! "success measured from the attempt's start" parts (Post::Pound's "50 lower than
//! at the start", Block::Push's "moved 200", Ride's "carried `travel`"), which
//! become keyed positions/depth instead; `Chest::Open`'s per-chest identity is
//! replaced by "the next chest in order" (the opened count). The BBH-style
//! `generate_level` hook is empty in rrt. Not seen by this port (as in rrt): the
//! spawner's ring count and ring indices.
//!
//! Example: `{"kind": "PurpleSwitchPress", "params": {"level": 9, "step": 200}}`.

use anyhow::{Result, bail};
use obrrt_core::layout::{ACT_FLAG_ON_POLE, ACT_FLAG_RIDING_SHELL, INTERACT_POLE, LEVEL_BBH, LEVEL_BOB, LEVEL_CASTLE, LEVEL_DDD, LEVEL_JRB, LEVEL_SSL, LEVEL_THI};
use obrrt_core::{Obj, St, bhv_addr};
use serde::Deserialize;
use serde_json::Value;

use super::util::{bucket, dist, pack_pos, yaw_bucket};
use crate::objective::{Cx, Objective};

// Mario actions, from include/sm64.h.
const ACT_GROUND_POUND: u32 = 0x008008A9;
const ACT_GROUND_POUND_LAND: u32 = 0x0080023C;
const ACT_TOP_OF_POLE_TRANSITION: u32 = 0x00100344;
const ACT_TOP_OF_POLE: u32 = 0x00100345;
/// Castle basement water pillars stand at y = -1178; pounded or drained they sit 80 lower
/// (rrt switches.rs `PILLAR_Y`).
const PILLAR_Y: f32 = -1178.0;

fn objs(st: &St) -> &[Obj] {
    st.objects.as_deref().map(|v| v.as_slice()).unwrap_or(&[])
}

fn hdist(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn add_y(p: [f32; 3], dy: f32) -> [f32; 3] {
    [p[0], p[1] + dy, p[2]]
}

fn nearest<'a>(it: impl Iterator<Item = &'a Obj>, from: [f32; 3], point: impl Fn(&Obj) -> [f32; 3]) -> Option<&'a Obj> {
    it.min_by(|a, b| dist(point(a), from).total_cmp(&dist(point(b), from)))
}

/// Where the objective applies: a level (and area) and, until its star is collected.
#[derive(Deserialize, Clone, Default)]
struct Gate {
    level: Option<u16>,
    area: Option<u8>,
    star: Option<[i16; 2]>,
}

impl Gate {
    fn open(&self, st: &St) -> bool {
        self.level.is_none_or(|l| l == st.level)
            && self.area.is_none_or(|a| a == st.area)
            && self.star.is_none_or(|[c, i]| !st.has_star(c, i as u8))
    }

    fn default_to(mut self, level: Option<u32>, star: Option<(i16, u8)>) -> Gate {
        self.level = self.level.or(level.map(|l| l as u16));
        self.star = self.star.or(star.map(|(c, i)| [c, i as i16]));
        self
    }
}

fn bhvs(names: &Option<Vec<String>>, default: &[&str]) -> Result<Vec<u32>> {
    let names: Vec<String> = names.clone().unwrap_or_else(|| default.iter().map(|s| s.to_string()).collect());
    names.iter().map(|n| bhv_addr(n).ok_or_else(|| anyhow::anyhow!("unknown behaviour {n}"))).collect()
}

fn step() -> f32 {
    200.0
}
fn grid() -> u32 {
    7
}

// ---------------------------------------------------------------- switches

#[derive(Clone, Copy, PartialEq, Debug)]
enum K {
    Purple,
    Cap,
    BlueCoin,
    Pillar,
    ChompPost,
    PostCircle,
    IslandTop,
    Diamond,
    PyramidPillar,
    Book(u8),
    Box,
    PoleTop,
}

#[derive(Deserialize)]
struct SwitchP {
    #[serde(flatten)]
    gate: Gate,
    bhv: Option<Vec<String>>,
    #[serde(default = "step")]
    step: f32,
    #[serde(default = "grid")]
    grid_log2: u32,
    /// `BookPunch`: which book (0-2) of the BBH bookshelf puzzle.
    #[serde(default)]
    index: u8,
}

/// A switch-like object: the nearest ready one (else the nearest) of the
/// `bhv`s, keyed by distance, its state and whether it is done.
///
/// Fields: `dist` (bucket of the straight-line distance to its point, -1 when
/// done), `state`, `done` (0/1), `act` (ground-pound kinds: 1 pounding, 2 the
/// landing, only near the target), `ang` (`PostCircle`: octant around the post
/// within 400, else -1), `tpos` (`BlockPush`: its packed position), `extra`
/// (`BookPunch`: bitmask of books out; `PyramidPillarTouch`: detectors left).
pub struct Switch {
    k: K,
    gate: Gate,
    bhv: Vec<u32>,
    step: f32,
    grid_log2: u32,
}

fn is_chomp_post(all: &[Obj], o: &Obj) -> bool {
    let chomp = bhv_addr("bhvChainChomp");
    all.iter().any(|c| Some(c.bhv) == chomp && hdist(c.home, o.home) < 50.0)
}

impl Switch {
    fn new(k: K, p: &Value, default: &[&str], level: Option<u32>, star: Option<(i16, u8)>) -> Result<Box<dyn Objective>> {
        let p: SwitchP = serde_json::from_value(p.clone())?;
        let k = if let K::Book(_) = k { K::Book(p.index) } else { k };
        Ok(Box::new(Switch {
            k,
            gate: p.gate.default_to(level, star),
            bhv: if k == K::PoleTop && p.bhv.is_none() { Vec::new() } else { bhvs(&p.bhv, default)? },
            step: p.step,
            grid_log2: p.grid_log2,
        }))
    }

    fn matches(&self, all: &[Obj], o: &Obj) -> bool {
        if self.k == K::PoleTop && self.bhv.is_empty() {
            return o.interact_type == INTERACT_POLE;
        }
        if !self.bhv.contains(&o.bhv) {
            return false;
        }
        match self.k {
            K::Book(i) => (o.beh_params >> 16) & 0xFF == i as u32,
            K::ChompPost => is_chomp_post(all, o),
            K::PostCircle => o.beh_params & 0xFF00 == 0 && !is_chomp_post(all, o),
            _ => true,
        }
    }

    fn books_out(all: &[Obj]) -> impl Iterator<Item = (u32, bool, i32)> + '_ {
        let book = bhv_addr("bhvBookSwitch");
        all.iter().filter(move |o| Some(o.bhv) == book).map(|o| ((o.beh_params >> 16) & 0xFF, o.home[2] - o.pos[2] > 25.0, o.action))
    }

    fn on_pole(st: &St, o: &Obj) -> bool {
        st.action_has(ACT_FLAG_ON_POLE) && hdist(st.pos, o.pos) < 100.0
    }

    fn at_top(st: &St, o: &Obj) -> bool {
        Self::on_pole(st, o) && matches!(st.action, ACT_TOP_OF_POLE | ACT_TOP_OF_POLE_TRANSITION)
    }

    /// rrt `ready`: the object can still be activated.
    fn ready(&self, st: &St, all: &[Obj], o: &Obj) -> bool {
        match self.k {
            K::Purple => o.action == 0,
            K::Cap => o.action == 1,
            K::BlueCoin | K::IslandTop => o.action == 0,
            K::Pillar => o.action == 0 && o.pos[1] > PILLAR_Y - 40.0,
            K::ChompPost => o.pos[1] > o.home[1] - 189.0,
            K::PostCircle => o.loot_coins > 0 && (o.pos[1] - o.home[1]).abs() < 1.0,
            K::Diamond => o.action == 1,
            K::PyramidPillar | K::Box => true,
            K::Book(i) => {
                o.action == 1 && Self::books_out(all).all(|(j, out, a)| j == i as u32 || if j < i as u32 { !out && a == 0 } else { out })
            }
            K::PoleTop => !Self::at_top(st, o),
        }
    }

    /// rrt `done`, from the object alone (and Mario for poles).
    fn done(&self, st: &St, o: &Obj) -> bool {
        match self.k {
            K::Purple => matches!(o.action, 1 | 2 | 4),
            K::Cap => o.action >= 2,
            K::BlueCoin => o.action >= 1,
            K::Pillar => o.action >= 1 || o.pos[1] <= PILLAR_Y - 40.0,
            K::ChompPost => o.pos[1] <= o.home[1] - 189.0,
            K::PostCircle => o.loot_coins == 0,
            K::IslandTop => o.action >= 1,
            K::Diamond => o.action >= 2,
            K::PyramidPillar | K::Box => false,
            K::Book(_) => o.action == 2,
            K::PoleTop => Self::at_top(st, o),
        }
    }

    fn state(&self, st: &St, o: &Obj) -> i32 {
        match self.k {
            K::Purple | K::Cap | K::BlueCoin | K::Pillar | K::IslandTop | K::Diamond | K::Book(_) => o.action,
            K::ChompPost => bucket((o.home[1] - o.pos[1]).max(0.0), 40.0).min(5),
            K::PostCircle => (o.loot_coins > 0) as i32,
            K::PyramidPillar | K::Box => 0,
            K::PoleTop => {
                if Self::on_pole(st, o) { 1 + bucket((st.pos[1] - o.pos[1]).max(0.0), 100.0).min(6) } else { 0 }
            }
        }
    }

    /// The point to reach (rrt `point`).
    fn point(&self, o: &Obj) -> [f32; 3] {
        match self.k {
            K::Cap => add_y(o.pos, 60.0),
            K::BlueCoin | K::ChompPost | K::PoleTop => add_y(o.pos, 100.0),
            _ => o.pos,
        }
    }

    fn pound(&self) -> bool {
        matches!(self.k, K::BlueCoin | K::Pillar | K::ChompPost | K::IslandTop)
    }
}

impl Objective for Switch {
    fn fields(&self) -> &[&'static str] {
        &["dist", "state", "done", "act", "ang", "tpos", "extra"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.open(st) {
            return;
        }
        let all = objs(st);
        let cands = || all.iter().filter(|o| self.matches(all, o));
        let count = cands().count();
        let level_pinned = self.gate.level.is_some();
        // "Gone when touched" kinds are done when none is left (needs the level to be pinned).
        if count == 0 && !(self.k == K::PyramidPillar && level_pinned) {
            return;
        }
        let target = nearest(cands().filter(|o| self.ready(st, all, o)), st.pos, |o| self.point(o))
            .or_else(|| nearest(cands(), st.pos, |o| self.point(o)));
        let Some(o) = target else {
            // PyramidPillar with every detector gone.
            cx.put(0, -1);
            cx.put(1, 0);
            cx.put(2, 1);
            cx.put(6, 0);
            return;
        };
        let done = self.done(st, o) || (self.k == K::PoleTop && !self.ready(st, all, o));
        cx.put(0, if done { -1 } else { bucket(dist(st.pos, self.point(o)), self.step) });
        cx.put(1, self.state(st, o));
        cx.put(2, done as i32);
        if self.pound() {
            let near = if self.k == K::IslandTop { 500.0 } else { 150.0 };
            let act = match st.action {
                _ if hdist(st.pos, o.pos) > near => 0,
                ACT_GROUND_POUND => 1,
                ACT_GROUND_POUND_LAND => 2,
                _ => 0,
            };
            cx.put(3, act);
        }
        match self.k {
            K::PostCircle => {
                let ang = if hdist(st.pos, o.pos) < 400.0 {
                    let a = (st.pos[0] - o.pos[0]).atan2(st.pos[2] - o.pos[2]);
                    ((a / std::f32::consts::TAU * 8.0).floor() as i32).rem_euclid(8)
                } else {
                    -1
                };
                cx.put(4, ang);
            }
            K::Box => cx.put(5, pack_pos(o.pos, self.grid_log2)),
            K::Book(_) => {
                let mask = Self::books_out(all).filter(|b| b.1).fold(0, |m, (j, _, _)| m | 1 << j.min(7));
                cx.put(6, mask);
            }
            K::PyramidPillar => cx.put(6, count as i32),
            _ => {}
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- platforms

#[derive(Clone, Copy)]
struct Shape {
    x: [f32; 2],
    y: [f32; 2],
    z: [f32; 2],
}

const fn sh(x: [f32; 2], y: [f32; 2], z: [f32; 2]) -> Shape {
    Shape { x, y, z }
}

impl Shape {
    /// An object's (x, z) frame: SM64 yaw maps local +z to world (sin, cos) (rrt common.rs `to_local`).
    fn local(o: &Obj, p: [f32; 3]) -> (f32, f32) {
        let (sn, cs) = (o.yaw as f32 * std::f32::consts::TAU / 65536.0).sin_cos();
        let (dx, dz) = (p[0] - o.pos[0], p[2] - o.pos[2]);
        (dx * cs - dz * sn, dx * sn + dz * cs)
    }

    fn contains(&self, o: &Obj, p: [f32; 3], mxz: f32, my: f32) -> bool {
        let (lx, lz) = Self::local(o, p);
        let dy = p[1] - o.pos[1];
        lx >= self.x[0] - mxz && lx <= self.x[1] + mxz && lz >= self.z[0] - mxz && lz <= self.z[1] + mxz && dy >= self.y[0] - my && dy <= self.y[1] + my
    }

    fn top(&self, o: &Obj) -> [f32; 3] {
        let (lx, lz) = ((self.x[0] + self.x[1]) / 2.0, (self.z[0] + self.z[1]) / 2.0);
        let (sn, cs) = (o.yaw as f32 * std::f32::consts::TAU / 65536.0).sin_cos();
        [o.pos[0] + lx * cs + lz * sn, o.pos[1] + self.y[1], o.pos[2] - lx * sn + lz * cs]
    }
}

struct Spec {
    shape: Shape,
    tilt: f32,
    star: Option<(i16, u8)>,
    seesaw: bool,
}

fn ride(shape: Shape) -> Spec {
    Spec { shape, tilt: 0.0, star: None, seesaw: false }
}

/// What `o` is as a platform; shapes are the min/max of each collision model's
/// vertices (rrt rides/platform.rs `spec_of` has the file of each).
fn spec_of(o: &Obj) -> Option<Spec> {
    let p2 = (o.beh_params >> 16) & 0xFF;
    Some(match o.bhv_name() {
        "bhvPlatformOnTrack" => ride(match (p2 & 0x70) >> 4 {
            0 => sh([-306.0, 307.0], [20.0, 20.0], [-204.0, 205.0]),
            1 => sh([-204.0, 205.0], [-255.0, 0.0], [-142.0, 143.0]),
            2 => sh([-255.0, 256.0], [-25.0, 26.0], [-153.0, 154.0]),
            _ => sh([-306.0, 307.0], [0.0, 205.0], [-306.0, 307.0]),
        }),
        "bhvHmcElevatorPlatform" => ride(sh([-255.0, 256.0], [0.0, 102.0], [-255.0, 256.0])),
        "bhvMeshElevator" => ride(sh([-306.0, 307.0], [-101.0, 0.0], [-255.0, 256.0])),
        "bhvRrElevatorPlatform" => ride(sh([-204.0, 205.0], [0.0, 102.0], [-306.0, 307.0])),
        "bhvWdwExpressElevator" => Spec { star: Some((11, 3)), ..ride(sh([-357.0, 512.0], [0.0, 51.0], [-64.0, 384.0])) },
        "bhvArrowLift" => ride(sh([-64.0, 63.0], [0.0, 127.0], [-60.0, 67.0])),
        "bhvPyramidElevator" => ride(sh([-511.0, 512.0], [-50.0, 256.0], [-511.0, 512.0])),
        "bhvTTCElevator" => ride(sh([-255.0, 256.0], [-153.0, 0.0], [-127.0, 128.0])),
        "bhvWfElevatorTowerPlatform" | "bhvWfSlidingTowerPlatform" => ride(sh([-214.0, 215.0], [0.0, 102.0], [-204.0, 205.0])),
        "bhvCheckerboardPlatformSub" => ride(sh([-255.0, 256.0], [-25.0, 26.0], [-153.0, 154.0])),
        "bhvControllablePlatform" => ride(sh([-306.0, 307.0], [0.0, 51.0], [-306.0, 307.0])),
        "bhvDorrie" => ride(sh([-313.0, 307.0], [-192.0, 717.0], [-947.0, 822.0])),
        "bhvActivatedBackAndForthPlatform" => ride(match (o.beh_params >> 24) & 3 {
            0 => sh([-460.0, 461.0], [0.0, 102.0], [-306.0, 307.0]),
            1 => sh([-306.0, 307.0], [0.0, 205.0], [-306.0, 307.0]),
            _ => sh([-204.0, 205.0], [0.0, 154.0], [-204.0, 205.0]),
        }),
        "bhvFerrisWheelPlatform" => ride(sh([-204.0, 205.0], [0.0, 102.0], [-306.0, 307.0])),
        "bhvLllMovingOctagonalMeshPlatform" => ride(sh([-255.0, 256.0], [0.0, 154.0], [-255.0, 256.0])),
        "bhvSquarishPathMoving" => ride(sh([-306.0, 307.0], [-306.0, 102.0], [-306.0, 307.0])),
        "bhvBitfsSinkingCagePlatform" => ride(sh([-306.0, 307.0], [0.0, 154.0], [-306.0, 307.0])),
        "bhvSeesawPlatform" => {
            let shape = match p2 {
                2 => sh([-1381.0, 1382.0], [51.0, 205.0], [-551.0, 779.0]),
                3 => sh([-180.0, 181.0], [0.0, 51.0], [-904.0, 905.0]),
                4 => sh([-511.0, 512.0], [26.0, 179.0], [-306.0, 307.0]),
                6 => sh([-665.0, 666.0], [0.0, 154.0], [-613.0, 410.0]),
                _ => sh([-306.0, 307.0], [0.0, 102.0], [-818.0, 819.0]),
            };
            Spec { shape, tilt: 250.0, star: None, seesaw: true }
        }
        _ => return None,
    })
}

/// A track platform falling off the end of its track (action 4) can't be used.
fn usable(o: &Obj) -> bool {
    !(o.bhv_name() == "bhvPlatformOnTrack" && o.action == 4)
}

/// Mario is standing or walking (not airborne, swimming, hanging, on a pole; rrt common.rs `grounded`).
fn grounded(st: &St) -> bool {
    matches!(st.action & 0x1C0, 0x000 | 0x040) && st.action & 0x800 == 0
}

#[derive(Deserialize)]
struct PlatformP {
    #[serde(flatten)]
    gate: Gate,
    /// Only these behaviours (default: every platform of the table).
    bhv: Option<Vec<String>>,
    #[serde(default = "step150")]
    step: f32,
    #[serde(default = "grid")]
    grid_log2: u32,
}

fn step150() -> f32 {
    150.0
}

/// Get onto a platform (`PlatformBoard`, `SeesawBoard`) or be carried by it
/// (`PlatformRide`, which also keys the platform's position while Mario is on it).
/// Fields: `dist` (to the platform's top), `on` (Mario stands on it), `ppos` (Ride, while on).
pub struct Platform {
    ride: bool,
    seesaw_only: bool,
    gate: Gate,
    bhv: Option<Vec<u32>>,
    step: f32,
    grid_log2: u32,
}

impl Platform {
    fn make(kind: &str, p: &Value) -> Result<Box<dyn Objective>> {
        let p: PlatformP = serde_json::from_value(p.clone())?;
        let bhv = match &p.bhv {
            Some(_) => Some(bhvs(&p.bhv, &[])?),
            None => None,
        };
        Ok(Box::new(Platform {
            ride: kind == "PlatformRide",
            seesaw_only: kind == "SeesawBoard",
            gate: p.gate,
            bhv,
            step: p.step,
            grid_log2: p.grid_log2,
        }))
    }

    fn on(spec: &Spec, st: &St, o: &Obj) -> bool {
        grounded(st) && spec.shape.contains(o, st.pos, 30.0, 40.0 + spec.tilt)
    }
}

impl Objective for Platform {
    fn fields(&self) -> &[&'static str] {
        &["dist", "on", "ppos"]
    }

    fn pos_grid(&self) -> Option<u32> {
        Some(self.grid_log2)
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.open(st) {
            return;
        }
        let cands = objs(st).iter().filter(|o| usable(o) && self.bhv.as_ref().is_none_or(|b| b.contains(&o.bhv))).filter_map(|o| {
            let s = spec_of(o)?;
            (s.seesaw == self.seesaw_only || (!self.seesaw_only && s.seesaw && self.bhv.is_some()))
                .then_some(())?;
            s.star.is_none_or(|(c, i)| !st.has_star(c, i)).then_some((o, s))
        });
        // The one Mario is on, else the nearest top.
        let best = cands.min_by(|a, b| {
            let key = |(o, s): &(&Obj, Spec)| (!Self::on(s, st, o), dist(st.pos, s.shape.top(o)));
            let (ka, kb) = (key(a), key(b));
            ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
        });
        let Some((o, s)) = best else { return };
        let on = Self::on(&s, st, o);
        cx.put(0, bucket(dist(st.pos, s.shape.top(o)), self.step));
        cx.put(1, on as i32);
        if self.ride && on {
            cx.put(2, pack_pos(o.pos, self.grid_log2));
        }
        cx.all_children();
    }
}

// ---------------------------------------------------------------- chests

struct ChestSet {
    name: &'static str,
    level: u32,
    /// Spawn x, z of chests 1 to 4 (spawn_treasure_chest calls).
    chests: [[f32; 2]; 4],
    star: (i16, u8),
}

const SETS: &[ChestSet] = &[
    ChestSet { name: "JrbCave", level: LEVEL_JRB, chests: [[-1700.0, -1150.0], [-1150.0, -1550.0], [-2400.0, -1800.0], [-1800.0, -2100.0]], star: (3, 2) },
    ChestSet { name: "JrbShip", level: LEVEL_JRB, chests: [[400.0, -2700.0], [650.0, -940.0], [-550.0, -770.0], [100.0, -1700.0]], star: (3, 0) },
    ChestSet { name: "Ddd", level: LEVEL_DDD, chests: [[-4500.0, 1300.0], [-1800.0, 1050.0], [-4500.0, -1100.0], [-2400.0, 125.0]], star: (9, 1) },
];

#[derive(Deserialize)]
struct ChestP {
    set: String,
    #[serde(default = "step100")]
    step: f32,
    /// Key the facing needed to open the next chest (within 300 of it): 3 fine buckets
    /// around the opening window plus `facing_coarse` for the rest (0 turns it off).
    #[serde(default = "three")]
    facing_coarse: u32,
}

fn step100() -> f32 {
    100.0
}
fn three() -> u32 {
    3
}

/// Open the 4 chests of a puzzle in order (treasure_chest.inc.c). Fields:
/// `opened` (chests open in order, 4 = done, -1 = one buzzed: wrong order),
/// `dist` (to the point 120 in front of the next chest), `facing` (Mario's yaw
/// against the 0x3000 opening window of the next chest, only within 300 of it, else -1).
pub struct ChestOpen {
    set: &'static ChestSet,
    step: f32,
    facing_coarse: u32,
}

fn chest_at(st: &St, xz: [f32; 2]) -> Option<&Obj> {
    let bottom = bhv_addr("bhvTreasureChestBottom");
    objs(st).iter().find(|o| Some(o.bhv) == bottom && hdist(o.pos, [xz[0], 0.0, xz[1]]) < 40.0)
}

impl Objective for ChestOpen {
    fn fields(&self) -> &[&'static str] {
        &["opened", "dist", "facing"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        let set = self.set;
        if st.level as u32 != set.level || st.has_star(set.star.0, set.star.1) {
            return;
        }
        let chests: Vec<Option<&Obj>> = set.chests.iter().map(|c| chest_at(st, *c)).collect();
        if chests.iter().all(|c| c.is_none()) {
            return;
        }
        if chests.iter().flatten().any(|o| o.action == 2) {
            cx.put(0, -1);
            return;
        }
        let opened = chests.iter().take_while(|c| c.is_some_and(|o| o.action == 1)).count();
        cx.put(0, opened as i32);
        let Some(Some(o)) = chests.get(opened) else { return };
        let a = o.yaw as f32 * std::f32::consts::TAU / 65536.0;
        let front = [o.pos[0] + 120.0 * a.sin(), o.pos[1], o.pos[2] + 120.0 * a.cos()];
        cx.put(1, bucket(dist(st.pos, front).min(dist(st.pos, o.pos)), self.step));
        if self.facing_coarse > 0 && dist(st.pos, o.pos) < 300.0 {
            // Opening needs Mario's yaw within 0x3000 of the chest's yaw + 0x8000.
            cx.put(2, yaw_bucket(st.yaw, (o.yaw as i16).wrapping_add(i16::MIN), 0x3000, 3, self.facing_coarse));
        } else {
            cx.put(2, -1);
        }
    }
}

// ---------------------------------------------------------------- rings and shells

#[derive(Deserialize)]
struct RingP {
    #[serde(flatten)]
    gate: Gate,
    /// Default: the manta ray's and the jet stream's rings.
    bhv: Option<Vec<String>>,
    #[serde(default = "step150")]
    step: f32,
}

/// Swim through the water rings of DDD's manta ray / jet stream (water_ring.inc.c).
/// Fields: `dist` (to the nearest uncollected ring, measured from Mario + 80 up as
/// the game does; -1 if none), `collected` (rings with oAction 1 loaded, 0-9).
pub struct WaterRingSwimThrough {
    gate: Gate,
    rings: Vec<u32>,
    step: f32,
}

impl Objective for WaterRingSwimThrough {
    fn fields(&self) -> &[&'static str] {
        &["dist", "collected"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.open(st) {
            return;
        }
        let rings: Vec<&Obj> = objs(st).iter().filter(|o| self.rings.contains(&o.bhv)).collect();
        if rings.is_empty() {
            return;
        }
        let probe = add_y(st.pos, 80.0);
        let d = rings.iter().filter(|o| o.action == 0).map(|o| dist(o.pos, probe)).min_by(f32::total_cmp);
        cx.put(0, d.map_or(-1, |d| bucket(d, self.step)));
        cx.put(1, rings.iter().filter(|o| o.action == 1).count().min(9) as i32);
        cx.all_children();
    }
}

#[derive(Deserialize)]
struct ShellP {
    #[serde(flatten)]
    gate: Gate,
    #[serde(default = "step")]
    step: f32,
}

/// Get on a Koopa shell (`bhvKoopaShell`, oAction 0 lying free). Fields: `riding`
/// (0/1), `dist` (to the nearest free shell, -1 while riding or with none).
pub struct ShellRide {
    gate: Gate,
    step: f32,
}

impl Objective for ShellRide {
    fn fields(&self) -> &[&'static str] {
        &["riding", "dist"]
    }

    fn needs_objects(&self) -> bool {
        true
    }

    fn eval(&self, st: &St, cx: &mut Cx) {
        if !self.gate.open(st) {
            return;
        }
        let shell = bhv_addr("bhvKoopaShell");
        let riding = st.action_has(ACT_FLAG_RIDING_SHELL);
        let free = nearest(objs(st).iter().filter(|o| Some(o.bhv) == shell && o.action == 0), st.pos, |o| o.pos);
        if !riding && free.is_none() {
            return;
        }
        cx.put(0, riding as i32);
        cx.put(1, match free {
            Some(o) if !riding => bucket(dist(st.pos, o.pos), self.step),
            _ => -1,
        });
        cx.all_children();
    }
}

/// The kinds of this family: `Some(objective)` for a kind name it knows, else `None`.
pub fn build(kind: &str, params: &Value) -> Option<Result<Box<dyn Objective>>> {
    let sw = |k, def: &[&str], level: Option<u32>, star| Switch::new(k, params, def, level, star);
    Some(match kind {
        "PurpleSwitchPress" => sw(
            K::Purple,
            &["bhvFloorSwitchHiddenObjects", "bhvFloorSwitchAnimatesObject", "bhvFloorSwitchGrills", "bhvFloorSwitchHardcodedModel"],
            None,
            None,
        ),
        "CapSwitchPress" => sw(K::Cap, &["bhvCapSwitch"], None, None),
        "BlueCoinSwitchPound" => sw(K::BlueCoin, &["bhvBlueCoinSwitch"], None, None),
        "PillarPound" => sw(K::Pillar, &["bhvWaterLevelPillar"], Some(LEVEL_CASTLE), None),
        "PostPound" => sw(K::ChompPost, &["bhvWoodenPost"], Some(LEVEL_BOB), Some((1, 5))),
        "PostCircle" => sw(K::PostCircle, &["bhvWoodenPost"], None, None),
        "IslandTopPound" => sw(K::IslandTop, &["bhvThiTinyIslandTop"], Some(LEVEL_THI), Some((13, 5))),
        "WaterDiamondTouch" => sw(K::Diamond, &["bhvWaterLevelDiamond"], None, None),
        "PyramidPillarTouch" => sw(K::PyramidPillar, &["bhvPyramidPillarTouchDetector"], Some(LEVEL_SSL), Some((8, 3))),
        "BookPunch" => sw(K::Book(0), &["bhvBookSwitch"], Some(LEVEL_BBH), Some((5, 2))),
        "BlockPush" => sw(K::Box, &["bhvPushableMetalBox"], None, None),
        "PoleClimb" => sw(K::PoleTop, &[], None, None),
        "PlatformBoard" | "PlatformRide" | "SeesawBoard" => Platform::make(kind, params),
        "ChestOpen" => (|| {
            let p: ChestP = serde_json::from_value(params.clone())?;
            let Some(set) = SETS.iter().find(|s| s.name == p.set) else { bail!("unknown chest set {}", p.set) };
            Ok(Box::new(ChestOpen { set, step: p.step, facing_coarse: p.facing_coarse }) as Box<dyn Objective>)
        })(),
        "WaterRingSwimThrough" => (|| {
            let p: RingP = serde_json::from_value(params.clone())?;
            let rings = bhvs(&p.bhv, &["bhvMantaRayWaterRing", "bhvJetStreamWaterRing"])?;
            Ok(Box::new(WaterRingSwimThrough { gate: p.gate.default_to(Some(LEVEL_DDD), None), rings, step: p.step }) as Box<dyn Objective>)
        })(),
        "ShellRide" => serde_json::from_value::<ShellP>(params.clone())
            .map(|p| Box::new(ShellRide { gate: p.gate, step: p.step }) as Box<dyn Objective>)
            .map_err(Into::into),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{key_of, obj, st, with_objects};

    fn k(json: &str, s: &St) -> Vec<(u8, i32)> {
        key_of(json, 0, s)
    }

    const BOB: u16 = LEVEL_BOB as u16;

    #[test]
    fn purple_gating_state_and_buckets() {
        let j = r#"{"kind":"PurpleSwitchPress","params":{"step":200}}"#;
        let plain = st([0.0; 3], 0.0, 9, 1, 0);
        assert!(k(j, &plain).is_empty(), "no objects: silent");
        let sw = |a, p| with_objects(st(p, 0.0, 9, 1, 0), vec![{ let mut o = obj("bhvFloorSwitchGrills", 3, [500.0, 0.0, 0.0]); o.action = a; o }]);
        let near = k(j, &sw(0, [350.0, 0.0, 0.0]));
        assert_eq!(near, vec![(0, 0), (1, 0), (2, 0)]);
        assert_eq!(k(j, &sw(0, [100.0, 0.0, 0.0]))[0], (0, 2));
        assert_eq!(k(j, &sw(0, [100.0, 0.0, 0.0])), k(j, &sw(0, [100.0, 0.0, 0.0])));
        // Pressed: done, distance no longer matters.
        assert_eq!(k(j, &sw(1, [100.0, 0.0, 0.0])), k(j, &sw(1, [-1000.0, 0.0, 0.0])));
        assert_ne!(k(j, &sw(1, [100.0, 0.0, 0.0])), k(j, &sw(0, [100.0, 0.0, 0.0])));
    }

    #[test]
    fn level_and_star_gate() {
        let j = r#"{"kind":"IslandTopPound"}"#;
        let o = || obj("bhvThiTinyIslandTop", 1, [0.0, 0.0, 0.0]);
        assert!(k(j, &with_objects(st([0.0; 3], 0.0, 9, 1, 0), vec![o()])).is_empty(), "wrong level");
        let s = with_objects(st([0.0; 3], 0.0, 13, 1, 0), vec![o()]);
        assert!(!k(j, &s).is_empty());
        let mut done = s.clone();
        done.course_stars = [0x7F; 25];
        done.stars = 255;
        assert!(done.has_star(13, 5), "test setup: star collected");
        assert!(k(j, &done).is_empty());
    }

    #[test]
    fn per_object_choice_prefers_ready() {
        let j = r#"{"kind":"CapSwitchPress"}"#;
        let mut far = obj("bhvCapSwitch", 1, [2000.0, 0.0, 0.0]);
        far.action = 1;
        let mut pressed = obj("bhvCapSwitch", 2, [100.0, 0.0, 0.0]);
        pressed.action = 3;
        let s = with_objects(st([0.0; 3], 0.0, 9, 1, 0), vec![far, pressed]);
        // The ready (unpressed) far one is the target, not the near pressed one.
        let key = k(j, &s);
        assert_eq!(key[1], (1, 1));
        assert_eq!(key[2], (2, 0));
    }

    #[test]
    fn pound_depth_and_action() {
        let j = r#"{"kind":"PostPound","params":{"level":9}}"#;
        let post = |dy: f32| {
            let mut p = obj("bhvWoodenPost", 1, [0.0, 0.0, 0.0]);
            p.pos[1] = -dy;
            vec![p, obj("bhvChainChomp", 2, [0.0, 0.0, 0.0])]
        };
        let s = |dy, a| {
            let mut s = with_objects(st([100.0, 0.0, 0.0], 0.0, BOB, 1, 0), post(dy));
            s.action = a;
            s
        };
        assert_ne!(k(j, &s(0.0, 0x0400_0440)), k(j, &s(100.0, 0x0400_0440)));
        assert_eq!(k(j, &s(10.0, 0x0400_0440)), k(j, &s(30.0, 0x0400_0440)));
        assert_ne!(k(j, &s(0.0, ACT_GROUND_POUND)), k(j, &s(0.0, 0x0400_0440)));
        // Not chained to a chomp: not a Post::Pound target.
        let lone = with_objects(st([300.0, 0.0, 0.0], 0.0, BOB, 1, 0), vec![obj("bhvWoodenPost", 1, [900.0, 0.0, 0.0])]);
        assert!(k(j, &lone).is_empty());
        let circ = r#"{"kind":"PostCircle"}"#;
        let mut post = obj("bhvWoodenPost", 1, [900.0, 0.0, 0.0]);
        post.loot_coins = 5;
        let at = |p| k(circ, &with_objects(st(p, 0.0, BOB, 1, 0), vec![post.clone()]));
        assert_ne!(at([900.0, 0.0, 1100.0]), at([900.0, 0.0, 700.0]));
        assert_eq!(at([2000.0, 0.0, 0.0])[3], (4, -1));
    }

    #[test]
    fn books_and_pyramid_and_box() {
        let book = |i: u32, out: bool, a: i32| {
            let mut b = obj("bhvBookSwitch", i as u16, [0.0, 0.0, 0.0]);
            b.beh_params = i << 16;
            b.home[2] = if out { 50.0 } else { 0.0 };
            b.action = a;
            b
        };
        let j = r#"{"kind":"BookPunch","params":{"index":0}}"#;
        let s = |bs| with_objects(st([100.0, 0.0, 0.0], 0.0, 4, 1, 0), bs);
        let a = k(j, &s(vec![book(0, true, 1), book(1, true, 1), book(2, true, 1)]));
        let b = k(j, &s(vec![book(0, true, 1), book(1, false, 0), book(2, true, 1)]));
        assert_ne!(a, b, "the other books' positions matter");
        let pj = r#"{"kind":"PyramidPillarTouch"}"#;
        let pill = |n: usize| (0..n).map(|i| obj("bhvPyramidPillarTouchDetector", i as u16, [i as f32 * 100.0, 0.0, 0.0])).collect::<Vec<_>>();
        let ps = |n| with_objects(st([0.0; 3], 0.0, 8, 1, 0), pill(n));
        assert_ne!(k(pj, &ps(4)), k(pj, &ps(3)));
        assert_eq!(k(pj, &ps(0)), vec![(0, -1), (1, 0), (2, 1), (6, 0)]);
        let bj = r#"{"kind":"BlockPush"}"#;
        let bx = |x| k(bj, &with_objects(st([0.0; 3], 0.0, 9, 1, 0), vec![obj("bhvPushableMetalBox", 1, [x, 0.0, 0.0])]));
        assert_eq!(bx(1000.0)[1], (1, 0));
        assert_ne!(bx(1000.0), bx(1300.0));
    }

    #[test]
    fn pole_states() {
        let j = r#"{"kind":"PoleClimb"}"#;
        let mut pole = obj("bhvGiantPole", 1, [0.0, 0.0, 0.0]);
        pole.interact_type = INTERACT_POLE;
        let s = |y: f32, a: u32| {
            let mut s = with_objects(st([10.0, y, 0.0], 0.0, 9, 1, 0), vec![pole.clone()]);
            s.action = a;
            s
        };
        let off = k(j, &s(0.0, 0x0400_0440));
        assert_eq!(off[1], (1, 0));
        let climbing = k(j, &s(250.0, 0x0010_0341 | 0x0800_0000));
        assert_eq!(climbing[1].1, 3);
        let top = k(j, &s(900.0, ACT_TOP_OF_POLE));
        assert_eq!(top, vec![(0, -1), (1, 7), (2, 1)]);
    }

    #[test]
    fn platform_on_and_ride() {
        let o = obj("bhvHmcElevatorPlatform", 1, [0.0, 0.0, 0.0]);
        let rj = r#"{"kind":"PlatformRide"}"#;
        let bj = r#"{"kind":"PlatformBoard"}"#;
        let at = |p, y: f32| {
            let mut o = o.clone();
            o.pos[1] = y;
            with_objects(st(p, 0.0, 7, 1, 0), vec![o])
        };
        let off = k(rj, &at([1000.0, 0.0, 0.0], 0.0));
        let on1 = k(rj, &at([0.0, 50.0, 0.0], 0.0));
        let on2 = k(rj, &at([0.0, 350.0, 0.0], 300.0));
        assert_eq!(off.len(), 2);
        assert_eq!(on1[1], (1, 1));
        assert_eq!(on1.len(), 3);
        assert_ne!(on1, on2, "carried to another height: another cell");
        assert_eq!(k(bj, &at([0.0, 50.0, 0.0], 0.0)).len(), 2, "Board has no ppos");
        // Seesaw kinds only see seesaws.
        let see = r#"{"kind":"SeesawBoard"}"#;
        assert!(k(see, &at([0.0, 50.0, 0.0], 0.0)).is_empty());
        let s = with_objects(st([0.0, 50.0, 0.0], 0.0, 7, 1, 0), vec![obj("bhvSeesawPlatform", 1, [0.0; 3])]);
        assert!(!k(see, &s).is_empty());
    }

    #[test]
    fn chests_in_order() {
        let j = r#"{"kind":"ChestOpen","params":{"set":"Ddd"}}"#;
        let set = &SETS[2];
        let chests = |acts: [i32; 4]| {
            (0..4).map(|i| { let mut o = obj("bhvTreasureChestBottom", i as u16, [set.chests[i][0], 0.0, set.chests[i][1]]); o.action = acts[i]; o }).collect::<Vec<_>>()
        };
        let s = |acts, p: [f32; 3]| with_objects(st(p, 0.0, LEVEL_DDD as u16, 1, 0), chests(acts));
        let far = [0.0, 0.0, 0.0];
        assert!(k(j, &with_objects(st(far, 0.0, 9, 1, 0), vec![])).is_empty());
        assert_eq!(k(j, &s([0, 0, 0, 0], far))[0], (0, 0));
        assert_eq!(k(j, &s([1, 0, 0, 0], far))[0], (0, 1));
        assert_eq!(k(j, &s([1, 2, 0, 0], far)), vec![(0, -1)]);
        assert_eq!(k(j, &s([1, 1, 1, 1], far)), vec![(0, 4)]);
        // Facing matters only near the next chest: yaw 0 vs the needed 0x8000.
        let near = [-4500.0, 0.0, 1300.0 + 100.0];
        let mut faced = s([0, 0, 0, 0], near);
        let wrong = k(j, &faced);
        faced.yaw = i16::MIN;
        assert_ne!(wrong, k(j, &faced));
        assert_eq!(k(j, &s([0, 0, 0, 0], far))[2], (2, -1));
    }

    #[test]
    fn rings_and_shell() {
        let j = r#"{"kind":"WaterRingSwimThrough"}"#;
        let ring = |slot, a, p| { let mut o = obj("bhvJetStreamWaterRing", slot, p); o.action = a; o };
        let s = |rs| with_objects(st([0.0, 0.0, 0.0], 0.0, LEVEL_DDD as u16, 1, 0), rs);
        assert!(k(j, &s(vec![])).is_empty());
        let a = k(j, &s(vec![ring(1, 0, [0.0, 500.0, 0.0])]));
        let b = k(j, &s(vec![ring(1, 1, [0.0, 500.0, 0.0])]));
        assert_ne!(a, b);
        assert_eq!(b[0], (0, -1));
        let sj = r#"{"kind":"ShellRide"}"#;
        let shell = obj("bhvKoopaShell", 1, [300.0, 0.0, 0.0]);
        let walk = k(sj, &with_objects(st([0.0; 3], 0.0, 9, 1, 0), vec![shell.clone()]));
        assert_eq!(walk, vec![(0, 0), (1, 1)]);
        let mut riding = with_objects(st([0.0; 3], 0.0, 9, 1, 0), vec![shell]);
        riding.action = 0x20C0_0000 | ACT_FLAG_RIDING_SHELL | 0x0400_0000;
        assert_eq!(k(sj, &riding), vec![(0, 1), (1, -1)]);
        assert!(build("Nope", &Value::Null).is_none());
    }
}
