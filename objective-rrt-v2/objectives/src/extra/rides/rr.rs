//! `RrCarpet::Ride` (Rainbow Ride, course 15): ride a flying carpet (`bhvPlatformOnTrack`, type 0) all the way
//! along its track.
//!
//! The carpets are RR's transport: levels/rr/script.c places four, each on its own path
//! (platform_on_track.inc.c `sPlatformOnTrackPaths`, rr/areas/1/trajectory.inc.c), beh param 2nd byte = path
//! (0x00, 0x01, 0x07, 0x08), none with the return-to-start bit, so each flies from its first waypoint to its last
//! at 10 units a frame and then falls (`PLATFORM_ON_TRACK_ACT_FALL`). It waits for Mario to stand on it 20 frames
//! and carries him only while he stays on (150 frames off and it resets):
//! - path 0, from (4500, -1820, 2050) over the lava-side bridge to (4035, -915, -40) (16 waypoints, ~5.5 k);
//! - path 1, from (1220, -1080, -3650) up to the hub (1100, 920, 20): the carpet that brings Mario to the two hub
//!   carpets (the hub is at y 920, ~2000 above the start's side);
//! - path 7, from the hub (1580, 980, -180) in a spiral up to (1580, 2460, 140) (40 waypoints, ~13 k): the ship;
//! - path 8, from the hub (540, 925, -160) over the big house's walls to (-4200, 6740, -5100) (42 waypoints, ~27 k),
//!   650 from the Big House in the Sky star (-4200, 6700, -4450): this ride IS the route of star index 1.
//! `Platform::Ride` (platform.rs) asks only 600 units of carriage per attempt, measured from where the carpet was
//! at the attempt's start, and finds a moving carpet by its home, which every ball spawn changes. t1 steps 645-648
//! and 699-700 (1.3 B frames) got no star in RR but the red coins'. This objective's ETA is the whole remaining
//! track (arc length / 10, from projecting the carpet's position on the path), so a state further along is
//! rated better however far the attempt started; it finds the carpet by pool slot; Done: standing on it
//! within `END_LEAD` of the track's last waypoint (the jump to whatever it leads to is `Star::Collect`'s / the
//! GA's). The weights: path 8 is a goal (>= OBRRT_GOAL_MIN 20) while the Big House star is missing; the others are
//! stepping stones that are only rated while a star they lead towards (index 0 ship, 1 big house) is missing.

use std::sync::Arc;

use obrrt_core::layout::LEVEL_RR;
use obrrt_core::{Eta, Mem, Obj, Objective, St, dist, frames_for};

use super::common::grounded;

const PATH0: [[f32; 3]; 16] = [
    [4500.0, -1820.0, 2050.0], [5250.0, -1840.0, 2050.0], [5860.0, -1430.0, 2070.0], [6260.0, -1430.0, 2070.0],
    [6970.0, -1220.0, 2000.0], [7300.0, -1160.0, 1745.0], [7490.0, -1125.0, 1450.0], [7560.0, -1065.0, 1040.0],
    [7510.0, -1020.0, 630.0], [7265.0, -935.0, 310.0], [6960.0, -915.0, 90.0], [6085.0, -915.0, -40.0],
    [5420.0, -915.0, -40.0], [4960.0, -915.0, -40.0], [4310.0, -915.0, -40.0], [4035.0, -915.0, -40.0],
];
const PATH1: [[f32; 3]; 27] = [
    [1220.0, -1080.0, -3650.0], [1740.0, -1120.0, -3640.0], [2220.0, -840.0, -3640.0], [2480.0, -300.0, -3640.0],
    [2780.0, -80.0, -3640.0], [3360.0, -100.0, -3640.0], [3820.0, -500.0, -3640.0], [4120.0, -500.0, -3640.0],
    [4520.0, -80.0, -3640.0], [4900.0, 40.0, -3640.0], [5300.0, 100.0, -3580.0], [5520.0, 260.0, -3360.0],
    [5640.0, 380.0, -3040.0], [5700.0, 470.0, -2660.0], [5680.0, 520.0, -2220.0], [5440.0, 570.0, -1920.0],
    [5120.0, 640.0, -1680.0], [4520.0, 720.0, -1620.0], [3900.0, 720.0, -1620.0], [3280.0, 920.0, -1620.0],
    [2460.0, 920.0, -1620.0], [2000.0, 920.0, -1660.0], [1660.0, 920.0, -1620.0], [1340.0, 920.0, -1460.0],
    [1060.0, 920.0, -1100.0], [1020.0, 920.0, -580.0], [1100.0, 920.0, 20.0],
];
const PATH7: [[f32; 3]; 40] = [
    [1580.0, 980.0, -180.0], [2740.0, 1140.0, -140.0], [3440.0, 1340.0, 260.0], [3740.0, 1400.0, 460.0],
    [3960.0, 1440.0, 800.0], [4200.0, 1490.0, 1100.0], [4520.0, 1540.0, 1340.0], [4920.0, 1600.0, 1420.0],
    [5360.0, 1645.0, 1340.0], [5660.0, 1700.0, 1120.0], [5900.0, 1740.0, 780.0], [5960.0, 1740.0, 380.0],
    [5900.0, 1740.0, -20.0], [5640.0, 1740.0, -360.0], [5340.0, 1740.0, -580.0], [4920.0, 1740.0, -620.0],
    [4520.0, 1750.0, -580.0], [4240.0, 1760.0, -340.0], [4080.0, 1785.0, -40.0], [4060.0, 1820.0, 320.0],
    [4140.0, 1855.0, 680.0], [4320.0, 1900.0, 940.0], [4620.0, 1940.0, 1140.0], [4920.0, 2010.0, 1190.0],
    [5240.0, 2050.0, 1140.0], [5500.0, 2100.0, 960.0], [5680.0, 2150.0, 700.0], [5740.0, 2160.0, 400.0],
    [5680.0, 2160.0, 60.0], [5500.0, 2160.0, -200.0], [5260.0, 2160.0, -380.0], [4940.0, 2160.0, -420.0],
    [4620.0, 2160.0, -360.0], [4400.0, 2200.0, -200.0], [4240.0, 2240.0, 20.0], [3900.0, 2260.0, 140.0],
    [3740.0, 2260.0, 140.0], [2700.0, 2260.0, 140.0], [2200.0, 2460.0, 140.0], [1580.0, 2460.0, 140.0],
];
const PATH8: [[f32; 3]; 42] = [
    [540.0, 925.0, -160.0], [-425.0, 1020.0, -160.0], [-1160.0, 1620.0, -160.0], [-1940.0, 2960.0, -160.0],
    [-2550.0, 3320.0, -170.0], [-3060.0, 3500.0, -80.0], [-3500.0, 3500.0, -620.0], [-3500.0, 3500.0, -1100.0],
    [-3380.0, 3500.0, -1600.0], [-3040.0, 3500.0, -1960.0], [-2620.0, 3500.0, -2200.0], [-2100.0, 3500.0, -3100.0],
    [-1660.0, 3500.0, -4160.0], [-1660.0, 3500.0, -4940.0], [-1880.0, 3500.0, -5280.0], [-2220.0, 3500.0, -5500.0],
    [-3060.0, 3380.0, -5560.0], [-3920.0, 3260.0, -5480.0], [-4180.0, 3260.0, -5260.0], [-4300.0, 3260.0, -4940.0],
    [-4300.0, 3440.0, -3780.0], [-4240.0, 3560.0, -2440.0], [-4400.0, 3560.0, -2120.0], [-4740.0, 3560.0, -1860.0],
    [-5320.0, 3600.0, -1800.0], [-6000.0, 3680.0, -1860.0], [-7720.0, 3980.0, -3250.0], [-7810.0, 4140.0, -3990.0],
    [-7670.0, 4320.0, -4645.0], [-7090.0, 4620.0, -5170.0], [-6040.0, 4620.0, -5280.0], [-5360.0, 4620.0, -5220.0],
    [-4980.0, 4100.0, -5140.0], [-4780.0, 3360.0, -5140.0], [-4480.0, 3180.0, -5140.0], [-3960.0, 3180.0, -5140.0],
    [-3580.0, 3580.0, -5140.0], [-3660.0, 4300.0, -5140.0], [-3800.0, 4800.0, -5140.0], [-4200.0, 5220.0, -5120.0],
    [-4200.0, 6560.0, -5100.0], [-4200.0, 6740.0, -5100.0],
];

/// Units a frame a carpet flies (platform_on_track.inc.c: `oForwardVel = 10`).
const SPEED: f64 = 10.0;
/// Standing on it before it starts (`act_wait_for_mario`: oTimer > 20).
const WAIT: f64 = 20.0;
/// Done when Mario is on the carpet this close (along the track) to its last waypoint.
const END_LEAD: f32 = 300.0;
/// Reaching its top from the side: the landing.
const BOARD_FRAMES: f64 = 10.0;
/// The carpet's top above its origin (rr_seg7_collision_07029038: y = 20).
const TOP: f32 = 20.0;
/// A carpet is only an ETA reference while Mario is within this of the track.
const OFF_TRACK: f32 = 400.0;

fn path_of(n: u32) -> Option<&'static [[f32; 3]]> {
    Some(match n {
        0 => &PATH0,
        1 => &PATH1,
        7 => &PATH7,
        8 => &PATH8,
        _ => return None,
    })
}

/// The carpet's path number (beh param 2nd byte & 0xF) if `o` is a carpet (type (2nd byte & 0x70) >> 4 = 0).
fn carpet_path(o: &Obj) -> Option<u32> {
    let p2 = (o.beh_params >> 16) & 0xFF;
    (o.bhv_name() == "bhvPlatformOnTrack" && p2 & 0x70 == 0).then_some(p2 & 0xF)
}

/// Arc length of every waypoint, and the point nearest `p` (3D): its arc length and distance.
fn project(path: &[[f32; 3]], p: [f32; 3]) -> (f32, f32, f32) {
    let mut acc = 0.0;
    let mut best = (0.0, f32::MAX);
    for w in path.windows(2) {
        let d = [w[1][0] - w[0][0], w[1][1] - w[0][1], w[1][2] - w[0][2]];
        let l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let t = (((p[0] - w[0][0]) * d[0] + (p[1] - w[0][1]) * d[1] + (p[2] - w[0][2]) * d[2]) / l2).clamp(0.0, 1.0);
        let q = [w[0][0] + t * d[0], w[0][1] + t * d[1], w[0][2] + t * d[2]];
        let e = dist(p, q);
        if e < best.1 {
            best = (acc + t * l2.sqrt(), e);
        }
        acc += l2.sqrt();
    }
    (best.0, best.1, acc)
}

/// (star indices it leads towards, weight) per path. From 2026-11-21 (the user): every carpet is a goal while any
/// RR star but the 100 coins is missing: the carpets are RR's way between its islands, and once the ship and big
/// house stars were had (t1 steps 749, 751) no carpet objective was generated at all and the trees stopped short
/// of the swing, triangles and cannon (agent rr3: nearest nodes 700-1200 from their starts).
fn leads(path: u32) -> (&'static [u8], f64) {
    match path {
        8 => (&[0, 1, 3, 4, 5], 30.0),
        7 => (&[0, 1, 3, 4, 5], 30.0),
        1 => (&[0, 1, 3, 4, 5], 25.0),
        _ => (&[0, 1, 3, 4, 5], 25.0),
    }
}

/// Off a moving carpet: Mario in the air over it (within `HOP_RADIUS` horizontally) is still riding (hops over
/// obstacles are fine); further off, every unit away costs and the carpet may reset (150 frames off), so leaving
/// it early is rated worse than staying on (the user: ditching the carpet was too rewarding/exploratory).
const HOP_RADIUS: f32 = 200.0;
const LEAVE_PENALTY: f64 = 120.0;

pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if b.level as u32 != LEVEL_RR {
        return;
    }
    for o in b.objects() {
        let Some(path) = carpet_path(o) else { continue };
        if path_of(path).is_none() || !leads(path).0.iter().any(|&i| !b.has_star(15, i)) {
            continue;
        }
        out.push(Arc::new(Ride { zone: b.zone(), path, slot: o.slot }));
    }
}

pub struct Ride {
    zone: (u16, u8),
    path: u32,
    slot: u16,
}

impl Ride {
    fn carpet<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone {
            return None;
        }
        s.objects().iter().find(|o| o.slot == self.slot && carpet_path(o) == Some(self.path))
    }

    fn open(&self, s: &St) -> bool {
        leads(self.path).0.iter().any(|&i| !s.has_star(15, i))
    }

    fn on(s: &St, p: &Obj) -> bool {
        grounded(s) && s.floor_obj == Some(p.slot)
    }

    /// Arc length left to the end of the track, from the carpet's position (a carpet off its track, falling
    /// from the end or not yet initialised, counts as at the start: it resets).
    fn left(&self, p: &Obj) -> (f32, bool) {
        let path = path_of(self.path).unwrap();
        let (a, d, len) = project(path, p.pos);
        // Not yet initialised: still at the object's placement (within a few hundred of waypoint 0).
        if p.action == 4 || d > OFF_TRACK {
            return (len, true);
        }
        (len - a, false)
    }
}

impl Objective for Ride {
    fn kind(&self) -> &'static str {
        "RrCarpet::Ride"
    }
    fn key(&self) -> String {
        format!("RrCarpet {:?} {}", self.zone, self.path)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        leads(self.path).1
    }
    fn base_criteria(&self, s: &St) -> bool {
        self.open(s) && self.carpet(s).is_some_and(|p| p.action != 4 && !(Self::on(s, p) && self.left(p).0 <= END_LEAD))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.open(s) && self.carpet(s).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 12, generations: 5, horizon: 450 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        let Some(p) = self.carpet(s) else { return Eta::Lost };
        let (left, reset) = self.left(p);
        let on = Self::on(s, p);
        if on && !reset && left <= END_LEAD {
            return Eta::Done;
        }
        let ride = (left - END_LEAD).max(0.0) as f64 / SPEED;
        let top = [p.pos[0], p.pos[1] + TOP, p.pos[2]];
        let h = ((s.pos[0] - top[0]).powi(2) + (s.pos[2] - top[2]).powi(2)).sqrt();
        let hop = !grounded(s) && h < HOP_RADIUS && s.pos[1] > top[1] - 50.0;
        if on || hop {
            let wait = if p.action == 1 { WAIT } else { 0.0 };
            return Eta::Frames(wait + ride);
        }
        // Off a carpet that is already flying (it was ridden): the way back plus the risk it resets.
        if p.action >= 2 && !reset {
            let back = frames_for(dist(s.pos, top)) * 2.0 + BOARD_FRAMES;
            return Eta::Frames(back + LEAVE_PENALTY + ride);
        }
        // Off it: walk to it (a straight line: the mesh has no floor in the sky gaps), board, ride. A moving carpet
        // is caught up with (Mario runs 32 a frame, it flies 10) for the 150 frames before it resets.
        let board = frames_for(dist(s.pos, [p.pos[0], p.pos[1] + TOP, p.pos[2]])) + BOARD_FRAMES;
        let wait = if p.action == 1 { WAIT } else { 0.0 };
        Eta::Frames(board + wait + ride)
    }
    fn success_value(&self) -> f64 {
        if self.path == 8 { 3.0 } else { 1.5 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.carpet(s).map(|p| [p.pos[0], p.pos[1] + TOP, p.pos[2]])
    }
}

// ---------------------------------------------------------------------------------------------------------------
// `RrRoute`: the three RR stars whose way is a fixed chain of platforms (waypoint chains read off
// levels/rr/areas/1/collision.inc.c and the object collisions; no inputs, only where the platforms are).

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// idx 4 "Tricky Triangles!" (3700, -400, 6600). The purple switch (4428, -1936, 6564) (bhvFloorSwitchAnimatesObject)
    /// flips bhvAnimatesOnFloorSwitchPress (animated_floor_switch.inc.c, param 2: the triangle platforms at
    /// (5862, -1347, 6564)) between its two collision sets (rr_seg7_collision_0702A6B4 at rest, ..07029924 pressed) for
    /// 250 frames. The union of both sets (5 columns of 410-wide squares, x 4941..6988, z 6360..6769) is a spiral
    /// stair, each step 205 higher: switch platform (-1935) -> c0 -1756 -> c1 -1551 -> c2 -1347 -> c1 -1142 -> c0
    /// -937 -> c0 -733 -> the tower top (y -732, x 3303..4940, z 6052..7076) under the star.
    Triangles,
    /// idx 3 "Swingin' in the Breeze" (-5850, -700, 4950). The pendulum bhvSwingPlatform at (-3557, -809, 4619)
    /// (swing_platform.inc.c: roll +-0x2000 (45 degrees), 256-frame period; deck 1000 below the pivot, x +-507) has
    /// its deck ends at y -1102 at the extremes, level with the platform on each side: (-2482..-1202, -1116, z
    /// 4619..5233) and the star's platform (x -6269..-4632, y -1116, z 4414..5438).
    Swing,
}

impl Route {
    fn star(self) -> u8 {
        match self {
            Route::Triangles => 4,
            Route::Swing => 3,
        }
    }
    fn chain(self) -> &'static [[f32; 3]] {
        match self {
            Route::Triangles => &[
                [4428.0, -1936.0, 6564.0], [4838.0, -1935.0, 6564.0], [5146.0, -1756.0, 6564.0], [5556.0, -1551.0, 6564.0],
                [5965.0, -1347.0, 6564.0], [5556.0, -1142.0, 6564.0], [5146.0, -937.0, 6564.0], [5146.0, -733.0, 6564.0],
                [4700.0, -732.0, 6564.0], [3900.0, -732.0, 6600.0],
            ],
            Route::Swing => &[
                // The stair up to the pendulum's near platform (collision.inc.c: single 205-wide squares at y -1750
                // (-1509..-1304, 6025..6217), -1494 (-1714..-1509, 5705..5910), -1238 (-1509..-1304, 5398..5603),
                // 205 a step, the last 122 under the platform (y -1116)).
                [-1406.0, -1782.0, 6000.0], [-1406.0, -1750.0, 6121.0], [-1611.0, -1494.0, 5807.0], [-1406.0, -1238.0, 5500.0],
                [-1850.0, -1116.0, 4900.0], [-2400.0, -1116.0, 4900.0], [-3557.0, -1730.0, 4900.0], [-4700.0, -1116.0, 4900.0],
                [-5850.0, -1116.0, 4950.0],
            ],
        }
    }
    /// Units a frame along the chain (jumps and climbs, slower than the run).
    fn pace(self) -> f32 {
        match self {
            Route::Triangles => 18.0,
            Route::Swing => 14.0,
        }
    }
    fn weight(self) -> f64 {
        match self {
            _ => 25.0,
        }
    }
    /// Mario's footing is the end of the route.
    fn done(self, s: &St) -> bool {
        if !grounded(s) {
            return false;
        }
        let [x, y, z] = s.pos;
        match self {
            Route::Triangles => (3303.0..4940.0).contains(&x) && (6052.0..7076.0).contains(&z) && (y + 732.0).abs() < 60.0,
            Route::Swing => (-6269.0..-4640.0).contains(&x) && (4414.0..5438.0).contains(&z) && (y + 1116.0).abs() < 60.0,
        }
    }
    /// Where the star's missing-ness is read (course 15).
    fn open(self, s: &St) -> bool {
        !s.has_star(15, self.star())
    }
}

pub fn generate_route(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if b.level as u32 != LEVEL_RR {
        return;
    }
    for r in [Route::Triangles, Route::Swing] {
        if r.open(b) {
            out.push(Arc::new(RrRoute { zone: b.zone(), r }));
        }
    }
}

pub struct RrRoute {
    zone: (u16, u8),
    r: Route,
}

impl Objective for RrRoute {
    fn kind(&self) -> &'static str {
        match self.r {
            Route::Triangles => "RrTriangles::Climb",
            Route::Swing => "RrSwing::Cross",
        }
    }
    fn key(&self) -> String {
        format!("{} {:?}", self.kind(), self.zone)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.r.weight()
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && self.r.open(s) && !self.r.done(s)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.r.open(s)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 12, generations: 5, horizon: 300 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if self.r.done(s) {
            return Eta::Done;
        }
        let chain = self.r.chain();
        let (arc, d, len) = project(chain, s.pos);
        let pace = self.r.pace();
        // Far from the chain: go to its start (the straight line: the walking distance of the mesh adds UNREACHABLE
        // for a platform one has to jump to, which put the swing's ETA out of every start's reach in the tests),
        // then all of it.
        if d > OFF_TRACK {
            let to = dist(s.pos, chain[0]);
            return Eta::Frames(frames_for(to) + (len / pace) as f64);
        }
        Eta::Frames(((len - arc) / pace) as f64 + (d / 32.0) as f64)
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let chain = self.r.chain();
        let (arc, d, _) = project(chain, s.pos);
        if d > OFF_TRACK {
            return Some(chain[0]);
        }
        // The first waypoint further along than Mario's progress.
        let mut acc = 0.0;
        for w in chain.windows(2) {
            acc += dist(w[0], w[1]);
            if acc > arc + 40.0 {
                return Some(w[1]);
            }
        }
        chain.last().copied()
    }
}
