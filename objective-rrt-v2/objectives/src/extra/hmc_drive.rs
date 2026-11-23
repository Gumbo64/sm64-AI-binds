//! Hazy Maze Cave red coins on the boxes: "Elevate for 8 Red Coins" (course 6, star index 1) by driving the arrow
//! platform.
//!
//! Three of the red coins sit inside breakable boxes (macro objects at y 1024, 200 wide and tall: levels/hmc/areas/1/
//! macro.inc.c) at (2960, 5140), (4080, 5760) and (6260, 4960), 1000 above the red coin room's y = 0 floor and
//! between its 1024..1126 high pillars; nothing a jump from the pillars or the 2048 ledge reaches (the pillars stand
//! 500 and more from the boxes). The way in is the metal arrow platform `bhvControllablePlatform` at (1100, 950,
//! 6350) (controllable_platform.inc.c). Its top is 1001, and its four buttons (`bhvControllablePlatformSub`, 204
//! from the centre on each side, lowered when stood on) set its direction: +z, -z, +x, -x at 10 a frame, a button
//! taking 30 frames after the last one was stood on; it stops dead at a wall (probes 300 ahead at 0 and +-250
//! across, radius 50, at its own height, so the 1024+ high pillars stop it) and vanishes and resets to its home if
//! Mario ends up more than 400 from its centre. Standing on it puts Mario's head at the box's bottom (1024): the
//! box is punched from the platform's edge (its hitbox: radius 150, y 1004..1204, switch_hidden_objects.inc.c), and
//! the coin, 100 above the box's base, is jumped for. The earlier red-coin objectives steered straight at the coin
//! (from below, the box is in the way) or rode the checkered track platform, which passes none of the three; t1 got
//! 7 of the 8 coins and never the (4080, 5760) one (RUNS.md 2026-11-15 00:57).
//!
//! `Hmc::DriveCoin`: the ETA is Mario to the platform, then the platform's drive to within [`GOAL_R`] of the nearest
//! box coin left, then [`FINISH`] for the punch and jump. The drive is a shortest path of the platform's centre
//! on a 50-unit grid with the wall probes above, 40 frames for every change of direction (the button's 30-frame
//! delay and the walk to the next one), over the level collision's wall segments that span the platform's height
//! (`WALLS`: the pillars and the room's edge). Done when a red coin is taken. Steering: the platform while Mario
//! is off it; on it, the button of the next leg, then the coin. Generated in HMC with the star missing and a box
//! coin left.
//!
//! Tests (side trees, OBRRT_GEO=1, route frames from the start): from t1's cp-0718b (7 red coins, riding the track
//! platform; hmc2-drive3, 19 M frames): `Hmc::DriveCoin` 39 attempts, 11 successes; `RedCoin::Next` 46, 0; the
//! first success's route rides the track to its end, drops onto the arrow platform (967, 1004, 6321), drives east
//! (x 1813 to 3413 at z 5590, then to the (4080, 5760) box), takes the 8th coin and the star: 94 stars in 1443
//! frames, replayed with tasplay (save flags: course 6 byte 0x39 -> 0x3b). Before the track-aware `board` it was
//! 0 of 13, states jumping off the track far from the platform and falling to the floor. From route-0792 (the
//! spawn, 80 M frames, hmc2-full2): DriveCoin 138 attempts, 45 successes (coins 1 to 3), `RedCoin::Ride` 136/31,
//! `RedCoin::Next` 176/1; no run got past 3 red coins there in 80 M frames.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Arc, OnceLock};

use super::collectibles::coins::{track_len, track_project};
use obrrt_core::layout::LEVEL_HMC;
use obrrt_core::{Eta, GaBudget, Mem, Obj, Objective, St, dist, frames_for};

/// The box coins' (x, z): their boxes are at y 1024, the coins 100 above.
const BOX_COINS: [[f32; 2]; 3] = [[2960.0, 5140.0], [4080.0, 5760.0], [6260.0, 4960.0]];
/// Platform home (hmc/script.c) and the cell grid of its centre.
const HOME: [f32; 2] = [1100.0, 6350.0];
const STEP: f32 = 50.0;
const X0: f32 = 700.0;
const Z0: f32 = 2850.0;
const NX: usize = 133;
const NZ: usize = 92;
/// Frames for a 50-unit step (10 a frame), and for changing the button.
const STEP_FRAMES: f64 = 5.0;
const TURN_FRAMES: f64 = 40.0;
/// The drive is done when the platform's centre is within this of the box.
const GOAL_R: f32 = 250.0;
/// Falling the 1000 from the track to the arrow platform and landing on it.
const DROP_FRAMES: f64 = 60.0;
/// Added when Mario is below the arrow platform's top: back up is a long way (an elevator, the track).
const BELOW: f64 = 1500.0;
/// Horizontal distance a fall from the track's height can cover to the platform: this plus the height over it.
const REACH_BASE: f64 = 300.0;
/// Frames from there: punch the box, take the coin.
const FINISH: f64 = 60.0;
/// The buttons' offsets from the platform's centre.
const BUTTON: f32 = 204.0;
/// Directions: +z, -z, +x, -x (the buttons' 2nd behaviour bytes 1..4).
const DIRS: [[i32; 2]; 4] = [[0, 1], [0, -1], [1, 0], [-1, 0]];

/// Wall segments (x1, z1, x2, z2) of the level collision that span y 960 (`obj_find_wall_displacement` at the
/// platform's y 950 + 10): the room's pillars and edge.
const WALLS: [[f32; 4]; 55] = [
    [410.0, 3584.0, 410.0, 3738.0],
    [410.0, 3584.0, 410.0, 6656.0],
    [410.0, 3584.0, 836.0, 3158.0],
    [410.0, 3584.0, 922.0, 3072.0],
    [410.0, 3738.0, 410.0, 6656.0],
    [410.0, 4096.0, 1434.0, 4096.0],
    [410.0, 4608.0, 922.0, 4608.0],
    [410.0, 6656.0, 1434.0, 6656.0],
    [836.0, 3158.0, 1270.0, 2723.0],
    [922.0, 3072.0, 1270.0, 2723.0],
    [922.0, 3072.0, 1434.0, 2560.0],
    [922.0, 4608.0, 922.0, 5120.0],
    [922.0, 4608.0, 1690.0, 4608.0],
    [922.0, 5120.0, 1690.0, 5120.0],
    [1270.0, 2723.0, 1434.0, 2560.0],
    [1434.0, 2560.0, 3484.0, 2560.0],
    [1434.0, 2560.0, 4150.0, 2560.0],
    [1434.0, 2560.0, 5579.0, 2560.0],
    [1434.0, 2560.0, 6554.0, 2560.0],
    [1434.0, 4096.0, 1434.0, 4608.0],
    [1434.0, 6656.0, 1434.0, 7680.0],
    [1434.0, 7680.0, 6554.0, 7680.0],
    [1690.0, 4608.0, 1690.0, 5120.0],
    [2048.0, 3891.0, 2048.0, 4403.0],
    [2048.0, 3891.0, 2765.0, 3891.0],
    [2048.0, 4403.0, 2765.0, 4403.0],
    [2765.0, 3891.0, 2765.0, 4403.0],
    [2765.0, 5939.0, 2765.0, 6656.0],
    [2765.0, 5939.0, 3277.0, 5939.0],
    [2765.0, 6656.0, 3277.0, 6656.0],
    [3277.0, 5939.0, 3277.0, 6656.0],
    [3484.0, 922.0, 3484.0, 2560.0],
    [4150.0, 922.0, 4150.0, 2560.0],
    [4506.0, 4403.0, 4506.0, 4915.0],
    [4506.0, 4403.0, 5018.0, 4403.0],
    [4506.0, 4915.0, 5018.0, 4915.0],
    [5018.0, 3584.0, 5018.0, 4403.0],
    [5018.0, 3584.0, 5530.0, 3584.0],
    [5018.0, 4403.0, 5018.0, 4915.0],
    [5018.0, 4403.0, 5530.0, 4403.0],
    [5018.0, 6144.0, 5018.0, 6656.0],
    [5018.0, 6144.0, 5530.0, 6144.0],
    [5018.0, 6656.0, 5530.0, 6656.0],
    [5530.0, 3584.0, 5530.0, 4403.0],
    [5530.0, 6144.0, 5530.0, 6656.0],
    [5579.0, 922.0, 5579.0, 2560.0],
    [5579.0, 2458.0, 6244.0, 2458.0],
    [5579.0, 2560.0, 6554.0, 2560.0],
    [6244.0, 922.0, 6244.0, 2560.0],
    [6244.0, 2560.0, 6554.0, 2560.0],
    [6554.0, 2560.0, 6554.0, 3584.0],
    [6554.0, 3584.0, 7578.0, 3584.0],
    [6554.0, 6656.0, 6554.0, 7680.0],
    [6554.0, 6656.0, 7578.0, 6656.0],
    [7578.0, 3584.0, 7578.0, 6656.0],
];

fn seg_dist(p: [f32; 2], w: &[f32; 4]) -> f32 {
    let (dx, dz) = (w[2] - w[0], w[3] - w[1]);
    let l = dx * dx + dz * dz;
    let t = if l > 0.0 { (((p[0] - w[0]) * dx + (p[1] - w[1]) * dz) / l).clamp(0.0, 1.0) } else { 0.0 };
    ((p[0] - w[0] - t * dx).powi(2) + (p[1] - w[1] - t * dz).powi(2)).sqrt()
}

/// Whether the platform at (x, z) moving in `d` is stopped by a wall (the three probes of `case 1..4`).
fn blocked(x: f32, z: f32, d: usize) -> bool {
    let [dx, dz] = DIRS[d];
    let probes: [[f32; 2]; 3] = if dx == 0 {
        let pz = z + 300.0 * dz as f32;
        [[x + 250.0, pz], [x, pz], [x - 250.0, pz]]
    } else {
        let px = x + 300.0 * dx as f32;
        [[px, z + 250.0], [px, z], [px, z - 250.0]]
    };
    probes.iter().any(|p| WALLS.iter().any(|w| seg_dist(*p, w) < 50.0))
}

fn cell(x: f32, z: f32) -> Option<(usize, usize)> {
    let (ix, iz) = (((x - X0) / STEP).round(), ((z - Z0) / STEP).round());
    (ix >= 0.0 && iz >= 0.0 && (ix as usize) < NX && (iz as usize) < NZ).then_some((ix as usize, iz as usize))
}

fn idx(ix: usize, iz: usize, d: usize) -> usize {
    (iz * NX + ix) * 4 + d
}

/// Frames still to drive, per (cell, direction in force), for one box.
struct Field {
    cost: Vec<f64>,
}

impl Field {
    /// Reverse Dijkstra from the cells within `GOAL_R` of the box.
    fn build(box_xz: [f32; 2]) -> Field {
        let mut cost = vec![f64::INFINITY; NX * NZ * 4];
        let mut heap = BinaryHeap::new();
        for iz in 0..NZ {
            for ix in 0..NX {
                let (x, z) = (X0 + ix as f32 * STEP, Z0 + iz as f32 * STEP);
                if ((x - box_xz[0]).powi(2) + (z - box_xz[1]).powi(2)).sqrt() < GOAL_R {
                    for d in 0..4 {
                        cost[idx(ix, iz, d)] = 0.0;
                        heap.push(Reverse((0u32, idx(ix, iz, d))));
                    }
                }
            }
        }
        while let Some(Reverse((v, i))) = heap.pop() {
            if (v as f64) > cost[i] {
                continue;
            }
            let (d2, c) = (i % 4, i / 4);
            let (ix, iz) = ((c % NX) as i32, (c / NX) as i32);
            // The state (n, d2) was entered by a step in d2 from n - d2, held in some direction d before.
            let (cx, cz) = (ix - DIRS[d2][0], iz - DIRS[d2][1]);
            if cx < 0 || cz < 0 || cx >= NX as i32 || cz >= NZ as i32 {
                continue;
            }
            let (cx, cz) = (cx as usize, cz as usize);
            if blocked(X0 + cx as f32 * STEP, Z0 + cz as f32 * STEP, d2) {
                continue;
            }
            for d in 0..4 {
                let nv = cost[i] + STEP_FRAMES + if d == d2 { 0.0 } else { TURN_FRAMES };
                let j = idx(cx, cz, d);
                if nv < cost[j] {
                    cost[j] = nv;
                    heap.push(Reverse((nv as u32, j)));
                }
            }
        }
        Field { cost }
    }

    /// Frames to drive from the centre (x, z), the direction in force unknown (the best); None off the grid or
    /// when it can't get there.
    fn to_go(&self, x: f32, z: f32) -> Option<f64> {
        let (ix, iz) = cell(x, z)?;
        let v = (0..4).map(|d| self.cost[idx(ix, iz, d)]).fold(f64::INFINITY, f64::min);
        v.is_finite().then_some(v)
    }

    /// The direction (button) of the cheapest next step from (x, z).
    fn next_dir(&self, x: f32, z: f32) -> Option<usize> {
        let (ix, iz) = cell(x, z)?;
        let (px, pz) = (X0 + ix as f32 * STEP, Z0 + iz as f32 * STEP);
        (0..4)
            .filter(|&d| !blocked(px, pz, d))
            .filter_map(|d| {
                let (nx, nz) = (ix as i32 + DIRS[d][0], iz as i32 + DIRS[d][1]);
                if nx < 0 || nz < 0 || nx >= NX as i32 || nz >= NZ as i32 {
                    return None;
                }
                let c = self.cost[idx(nx as usize, nz as usize, d)];
                c.is_finite().then_some((d, c))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|a| a.0)
    }
}

fn fields() -> &'static [Field; 3] {
    static F: OnceLock<[Field; 3]> = OnceLock::new();
    F.get_or_init(|| [Field::build(BOX_COINS[0]), Field::build(BOX_COINS[1]), Field::build(BOX_COINS[2])])
}

pub struct DriveCoin;

fn in_room(s: &St) -> bool {
    s.zone() == (LEVEL_HMC as u16, 1)
}

fn platform(s: &St) -> Option<&Obj> {
    s.objects().iter().find(|o| o.bhv_name() == "bhvControllablePlatform")
}

/// The box coins still out (which of `BOX_COINS`).
fn coins_left(s: &St) -> Vec<usize> {
    (0..3)
        .filter(|&i| {
            s.objects().iter().any(|o| {
                o.bhv_name() == "bhvRedCoin"
                    && (o.pos[0] - BOX_COINS[i][0]).abs() < 60.0
                    && (o.pos[2] - BOX_COINS[i][1]).abs() < 60.0
                    && o.pos[1] > 1000.0
                    && o.pos[1] < 1250.0
            })
        })
        .collect()
}

/// Standing on the platform or one of its buttons.
fn on_platform(s: &St, p: &Obj) -> bool {
    s.floor_obj.is_some_and(|f| {
        f == p.slot
            || s.objects().iter().any(|o| {
                o.slot == f
                    && o.bhv_name() == "bhvControllablePlatformSub"
                    && (o.pos[0] - p.pos[0]).abs() < 400.0
                    && (o.pos[2] - p.pos[2]).abs() < 400.0
            })
    })
}

/// Frames to drive to the nearest box coin left, with that coin's index; from the platform's position `at`
/// (None: it is gone and respawns at its home, 200 frames).
fn drive(s: &St, at: Option<[f32; 2]>) -> Option<(f64, usize)> {
    let (pos, wait) = match at {
        Some(p) => (p, 0.0),
        None => (HOME, 200.0),
    };
    coins_left(s)
        .into_iter()
        .filter_map(|i| fields()[i].to_go(pos[0], pos[1]).map(|v| (v + wait, i)))
        .min_by(|a, b| a.0.total_cmp(&b.0))
}

/// Frames for Mario, off the arrow platform, to be on it. The platform hangs 1000 over the room's floor between
/// pillars, so only from above: from the checkered track platform (`RedRide`'s, `HMC_TRACK`), whose last
/// leg (1270, 1990, 7130) to (1272, 2000, 6000) passes 170 east of the arrow platform's home, or from the 2048
/// ledge. A Mario on the track platform is as far as the track to the point nearest the arrow platform, then a drop
/// of [`DROP_FRAMES`]; one elsewhere higher than the platform, the straight way (in the air, only within the reach of a fall); one below its
/// top (the room's floor, `BELOW`) has lost the way up. The first test (hmc2-drive2, a straight line for everything
/// above the platform) filled up with jumps off the track far to the east, falling to the floor.
fn board(s: &St, p: &Obj) -> f64 {
    let top = [p.pos[0], p.pos[1] + 60.0, p.pos[2]];
    let straight = frames_for(dist(s.pos, top)) + 10.0;
    if s.pos[1] < p.pos[1] + 20.0 {
        return straight + BELOW;
    }
    let track = s.objects().iter().find(|o| o.bhv_name() == "bhvPlatformOnTrack" && o.action != 4);
    if let Some(t) = track.filter(|t| s.floor_obj == Some(t.slot)) {
        let len = track_len();
        let (here, d, _) = track_project(t.pos);
        let (there, _, _) = track_project(p.pos);
        if d < 100.0 {
            let arc = if here <= there { there - here } else { len[len.len() - 1] - here + there + 400.0 };
            return arc as f64 / 10.0 + DROP_FRAMES;
        }
    }
    // In the air off both (a jump from the track far from the arrow platform): the fall can't carry it there.
    let horiz = (s.pos[0] - p.pos[0]).hypot(s.pos[2] - p.pos[2]) as f64;
    let air = (s.pos[1] - s.floor_y) as f64;
    if air > 60.0 && horiz > REACH_BASE + (s.pos[1] - p.pos[1]) as f64 {
        return straight + BELOW;
    }
    straight
}

impl Objective for DriveCoin {
    fn kind(&self) -> &'static str {
        "Hmc::DriveCoin"
    }
    fn key(&self) -> String {
        "Hmc::DriveCoin".into()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // A goal (over OBRRT_GOAL_MIN 20) like RedCoin::Next and RedCoin::Ride.
        25.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        in_room(s) && !s.has_star(6, 1) && !coins_left(s).is_empty()
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 400 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if !in_room(s) {
            return Eta::Lost;
        }
        if s.alive() && s.red_coins > start.red_coins {
            return Eta::Done;
        }
        let p = platform(s);
        let at = p.map(|p| [p.pos[0], p.pos[2]]);
        let Some((go, _)) = drive(s, at) else { return Eta::Lost };
        match p {
            // On it: the drive and the punch.
            Some(p) if on_platform(s, p) => Eta::Frames(go + FINISH),
            Some(p) => Eta::Frames(board(s, p) + go + FINISH),
            None => Eta::Frames(go + FINISH + 100.0),
        }
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
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let p = platform(s)?;
        if !on_platform(s, p) {
            // Riding the track platform: stay on it until it is over the arrow platform.
            let riding = s.objects().iter().any(|o| o.bhv_name() == "bhvPlatformOnTrack" && s.floor_obj == Some(o.slot));
            return (!riding || (s.pos[0] - p.pos[0]).hypot(s.pos[2] - p.pos[2]) < 450.0).then_some([p.pos[0], p.pos[1] + 60.0, p.pos[2]]);
        }
        let (go, i) = drive(s, Some([p.pos[0], p.pos[2]]))?;
        if go > 0.0 {
            // The button of the next leg.
            let d = fields()[i].next_dir(p.pos[0], p.pos[2])?;
            return Some([p.pos[0] + BUTTON * DIRS[d][0] as f32, p.pos[1] + 60.0, p.pos[2] + BUTTON * DIRS[d][1] as f32]);
        }
        Some([BOX_COINS[i][0], 1124.0, BOX_COINS[i][1]])
    }
    fn exists_in(&self, s: &St) -> bool {
        self.base_criteria(s)
    }
}

pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if DriveCoin.base_criteria(b) && platform(b).is_some() {
        out.push(Arc::new(DriveCoin));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extra::testkit::{obj, st};
    use obrrt_core::fold;

    fn coin(i: usize) -> Obj {
        obj("bhvRedCoin", 20 + i as u16, [BOX_COINS[i][0], 1124.0, BOX_COINS[i][1]], 0, 0)
    }

    /// The drive from the platform's home to each box (a python BFS over the same walls: 260, 390, 610 frames
    /// without turns), and the field falls as the platform moves along it.
    #[test]
    fn drive_lengths() {
        for (i, (lo, hi)) in [(150.0, 450.0), (250.0, 600.0), (450.0, 900.0)].into_iter().enumerate() {
            let v = fields()[i].to_go(HOME[0], HOME[1]).expect("reachable");
            assert!((lo..hi).contains(&v), "box {i}: {v}");
        }
        let f = &fields()[1];
        let d = f.next_dir(HOME[0], HOME[1]).expect("a step");
        let (nx, nz) = (HOME[0] + 50.0 * DIRS[d][0] as f32, HOME[1] + 50.0 * DIRS[d][1] as f32);
        assert!(f.to_go(nx, nz).unwrap() < f.to_go(HOME[0], HOME[1]).unwrap());
    }

    /// Off the platform costs the walk to it; driving it nearer the coin lowers the ETA; the coin taken is `Done`.
    #[test]
    fn plat_then_drive() {
        let plat = |p: [f32; 3]| obj("bhvControllablePlatform", 5, p, 0, 0);
        let off = st(7, [1100.0, 1500.0, 6000.0], vec![plat([1100.0, 950.0, 6350.0]), coin(1)]);
        let mut on = st(7, [1100.0, 1001.0, 6350.0], vec![plat([1100.0, 950.0, 6350.0]), coin(1)]);
        on.floor_obj = Some(5);
        let mut near = st(7, [3600.0, 1001.0, 5650.0], vec![plat([3600.0, 950.0, 5650.0]), coin(1)]);
        near.floor_obj = Some(5);
        let mut taken = st(7, [4000.0, 1001.0, 5700.0], vec![plat([4000.0, 950.0, 5650.0])]);
        taken.red_coins = 1;
        let e = |s: &St| fold(&DriveCoin, std::slice::from_ref(s)).1[0];
        assert!(e(&off).frames() > e(&on).frames());
        assert!(e(&on).frames() > e(&near).frames() + 100.0);
        assert_eq!(fold(&DriveCoin, &[on.clone(), taken]).0, Some(1));
    }
}
