//! SSL's pyramid (area 2) as a walk with a few leaps the floor graph of
//! [`obrrt_core::geo`] can't see, for `Secrets::Tour` ("Pyramid Puzzle", star 6).
//!
//! The five `bhvHiddenStarTrigger` (levels/ssl/areas/2/macro.inc.c) are two at y 1229 at the ends of the lowest
//! floor's corridors (x -1940, z -600 / 2320) and three on lone 256x256 slabs in the middle of the pyramid
//! (collision.inc.c: x 131..387 / -384..-128 / 131..387, z -716..-460, tops at y 1967 / 2940 / 3913, 128 thick:
//! walls only on their own sides). Nothing climbs to them from below (the main run's 400 M frame tree reached
//! 393 units from the first slab and never touched one): the slabs are the way DOWN. The spiral of corridors
//! (floors at y 1280, 1920, 2560, 3200, 3840 / 4429, each level one corridor of a ring) ends at the platform
//! (x 387..643, z -1125..-409, y 4815, a very slippery lip to y 4927) next to the pyramid elevator
//! (bhvPyramidElevator, pyramid_elevator.inc.c: the platform x -511..512, z -255..768 around the object's x, z
//! 256, top at y + 256, waits at y 4966 until Mario stands on it, then drops 10 a frame to y 128). The slabs
//! are 205 units north of its edge and each is 973 below the one before: ride down, hop off to a slab (the
//! trigger), hop back on as the elevator passes under, next slab. (Area 1's top hole, warp 0x1E, drops Mario on
//! the elevator too, `bhvAirborneWarp` at (0, 5500, 256): [`TopHole`], below.)
//!
//! The floor graph walks drops freely but climbs at most 300, and sees no pole, no moving wall and no
//! elevator, so from the lowest floors the slabs are "unreachable" (straight line + 20000: no gradient but the
//! straight line, which ends at the corridor end under slab 1) and the three sections that only a gap, a pole
//! or a leap joins (this file's [`GATES`], each located on the floors' dump of that collision file) cut the
//! walk. Here: a hand list of links, the walking distance between them from the mesh, and the elevator ride.
//! Moving pyramid walls (bhvSslMovingPyramidWall, pyramid_wall.inc.c): 128 wide, 614 deep (the corridor's width
//! z -2612..-1998), 512 tall, rising and falling 5.12 a frame in 100-frame legs; pairs at x 730 / 858 (y 1927,
//! the gap x 666..922 between the y 1920 corridor and the y 2560 ledge) and 1345 / 1473 (y 2567, the gap
//! x 1280..1536 to the y 3200 corridor): stepping stones whose top is 121 under the next floor at best. Poles:
//! (2867, 640, 2867) height 770 (to the y 1280 floor) and (0, 3200, 1331) height 920 (to y 3942), pole.inc.c.

use obrrt_core::layout::LEVEL_SSL;
use obrrt_core::{PACE, St, dist, frames_for, geo};

pub const ZONE: (u16, u8) = (LEVEL_SSL as u16, 2);

/// Added to a leg that can't be done (the same as the floor graph's `UNREACHABLE`).
const NO_WAY: f32 = 20000.0;
/// The slabs' tops (x, y, z) from the lowest, with their x ranges.
const SLABS: [([f32; 3], f32, f32); 3] = [([260.0, 1967.0, -600.0], 131.0, 387.0), ([-260.0, 2940.0, -600.0], -384.0, -128.0), ([260.0, 3913.0, -600.0], 131.0, 387.0)];
/// The elevator's platform over its object: x +-512, z -255..768 about z 256, top 256 above y; waiting at y 4966.
const ELEV_TOP: f32 = 4966.0 + 256.0;
/// Where Mario leaps onto the elevator from: the platform's lip.
const LIP: [f32; 3] = [405.0, 4910.0, -560.0];

/// One link the floor graph lacks: from a floor point to another, at a cost in distance units (30 per frame).
struct Gate {
    from: [f32; 3],
    to: [f32; 3],
    cost: f32,
}

const GATES: [Gate; 5] = [
    // The pole at (2867, 640, 2867), 770 high: up to the y 1280 floor of the south-east room.
    Gate { from: [2867.0, 640.0, 2867.0], to: [3000.0, 1280.0, 2900.0], cost: 1540.0 },
    // The moving walls at x 730 / 858 between the y 1920 corridor (ends x 666) and the y 2560 ledge (x 922..1280).
    Gate { from: [600.0, 1920.0, -2300.0], to: [1000.0, 2560.0, -2300.0], cost: 1500.0 },
    // The walls at x 1345 / 1473 between the ledge and the y 3200 corridor (x 1536..1946).
    Gate { from: [1200.0, 2560.0, -2300.0], to: [1600.0, 3200.0, -2300.0], cost: 1500.0 },
    // The pole at (0, 3200, 1331), 920 high: up to the y 3942 floor.
    Gate { from: [0.0, 3200.0, 1300.0], to: [0.0, 3942.0, 1200.0], cost: 1840.0 },
    // A double jump from the y 4480 ledge (x 387..512, z -767..-716) onto the y 4815 platform (335 up).
    Gate { from: [450.0, 4480.0, -740.0], to: [500.0, 4815.0, -700.0], cost: 300.0 },
];

/// A leg over the floor graph (straight where it doesn't know the points).
fn walk(a: [f32; 3], b: [f32; 3]) -> f32 {
    geo::path_dist(ZONE, a, b).unwrap_or_else(|| dist(a, b))
}

/// Whether `t` is one of the three slab triggers.
pub fn is_slab_trigger(t: [f32; 3]) -> bool {
    t[1] > 1900.0 && t[1] < 4000.0
}

/// The slab (0 lowest) a point stands on or is just over.
fn slab_of(a: [f32; 3]) -> Option<usize> {
    SLABS.iter().position(|&(t, x0, x1)| a[0] >= x0 - 20.0 && a[0] <= x1 + 20.0 && (a[2] + 588.0).abs() < 150.0 && a[1] >= t[1] - 20.0 && a[1] < t[1] + 100.0)
}

/// The elevator's top y, if its object is loaded.
fn elevator_top(s: &St) -> Option<f32> {
    s.objects().iter().find(|o| o.bhv_name() == "bhvPyramidElevator").map(|o| o.pos[1] + 256.0)
}

/// Distance units from `a` to the elevator's platform, over the gates (the leap from the lip included).
fn to_elevator(a: [f32; 3]) -> f32 {
    let n = GATES.len();
    // suffix[i]: from gate i's start to the lip.
    let mut suffix = [0.0f32; GATES.len()];
    suffix[n - 1] = GATES[n - 1].cost + walk(GATES[n - 1].to, LIP);
    for i in (0..n - 1).rev() {
        suffix[i] = GATES[i].cost + walk(GATES[i].to, GATES[i + 1].from) + suffix[i + 1];
    }
    let mut best = walk(a, LIP);
    for i in 0..n {
        best = best.min(walk(a, GATES[i].from) + suffix[i]);
    }
    best + dist(LIP, [450.0, ELEV_TOP, -200.0])
}

/// Distance units from `a` to the slab trigger `t`: the walk to the elevator, the ride down to the slab's level
/// (the platform falls 10 a frame) and the hop off (205 across); from a higher slab (or the elevator) it is the
/// ride down and the hop; from a lower one, or once the elevator has passed, there is none.
pub fn to_slab(s: &St, a: [f32; 3], t: [f32; 3]) -> f32 {
    let Some(top) = elevator_top(s) else { return dist(a, t) + NO_WAY };
    let hop = 205.0;
    let ride = |from_top: f32| (from_top - t[1]).max(0.0) / 10.0 * PACE as f32;
    let on_elevator = a[0].abs() < 520.0 && a[2] > -260.0 && a[2] < 780.0 && a[1] > top - 80.0 && a[1] < top + 500.0;
    if on_elevator {
        return if top + 40.0 >= t[1] { ride(top) + hop } else { dist(a, t) + NO_WAY };
    }
    if let Some(k) = slab_of(a) {
        // On slab k: the elevator is under it by now or soon; a slab above the target is ridden to.
        return if SLABS[k].0[1] > t[1] + 100.0 { dist(a, t) + 600.0 } else { dist(a, t) + NO_WAY };
    }
    if top < ELEV_TOP - 50.0 {
        // The elevator left without Mario: nothing brings him to the slabs again.
        return dist(a, t) + NO_WAY;
    }
    to_elevator(a) + ride(ELEV_TOP) + hop
}

// ---------------------------------------------------------------------------
// The way in from above: area 1's pyramid-top hole.

/// Area 1 (the outside) of SSL.
pub const OUTSIDE: (u16, u8) = (LEVEL_SSL as u16, 1);
/// Where the hole is: the pit x -2149..-1945, z -1125..-921 in the middle of the pyramid's flat top (y 1280;
/// levels/ssl/areas/1/collision.inc.c, walls down to a floor at y 768), covered by `bhvPyramidTop`'s cap
/// (levels/ssl/script.c: (-2047, 1536, -1023), collision base 1281, x/z +-511 = the whole flat top).
const HOLE: [f32; 3] = [-2047.0, 1280.0, -1023.0];
/// Frames to fall the pit's 512 and touch the warp at its bottom.
const HOLE_FALL: f64 = 20.0;
/// What the cap still needs once the 4th pillar is touched: spins 60 + 90 frames, then explodes at oTimer 150
/// (pyramid_top.inc.c); less once it is already spinning (`oAction` 1).
const CAP_WAIT: f64 = 150.0;

/// Whether `s` has dropped down the hole: area 2 and still up at the airborne warp's height
/// (`bhvAirborneWarp` (0, 5500, 256), area 2's node 0x14, which warp 0x1E of area 1 leads to), over the
/// elevator (waiting at y 4966, top 5222), not the door's node 0x0A at y 300.
pub fn dropped_in(s: &St) -> bool {
    s.zone() == ZONE && s.pos[1] > 4900.0
}

/// `Pyramid::TopHole`: from area 1 (the outside) into area 2 by the hole in the pyramid's top. The warp
/// `bhvWarp` (-2048, 768, -1024) (levels/ssl/script.c; warp id 0x1E: area 2, node 0x14, the `bhvAirborneWarp`
/// at (0, 5500, 256) straight above `bhvPyramidElevator`) sits at the pit's floor, and the pit is under the
/// cap of `bhvPyramidTop`, which is always there (no act mask) and spins and explodes only once Mario has
/// touched all four `bhvPyramidPillarTouchDetector` (pyramid_top.inc.c: (1789, 1024, 764), (1789, 896, -2579),
/// (-5883, 1024, -2579), (-5883, 1024, 764); each deactivated on touch). So the way in is: the four pillars, the
/// pyramid's top (the same plateau "Shining atop the Pyramid" is on, the star at (-2050, 1200, -580)), a wait
/// for the cap, the pit. The pillars' star ("Stand Tall on the Four Pillars") may be had, and then
/// `Switch::PyramidPillar` isn't generated, so this one walks them itself. The ETA is the greedy walk over
/// the pillars left (floor-graph distances where known), the cap's wait, the walk to the pit and the fall;
/// `Done` on arrival in area 2 up at the airborne warp's height, from where `Secrets::Tour` rides the elevator
/// down ([`to_slab`]). `Lost` when Mario leaves area 1 any other way.
pub struct TopHole;

impl TopHole {
    pub fn new(b: &St) -> Option<TopHole> {
        let ok = b.zone() == OUTSIDE && b.objects.is_some() && !b.has_star(8, 5) && b.objects().iter().any(|o| o.bhv_name() == "bhvPyramidTop" || o.bhv_name() == "bhvPyramidPillarTouchDetector");
        ok.then_some(TopHole)
    }

    fn pillars(s: &St) -> Vec<[f32; 3]> {
        s.objects().iter().filter(|o| o.bhv_name() == "bhvPyramidPillarTouchDetector").map(|o| o.pos).collect()
    }

    fn cap(s: &St) -> Option<i32> {
        s.objects().iter().find(|o| o.bhv_name() == "bhvPyramidTop").map(|o| o.action)
    }

    fn walk(a: [f32; 3], b: [f32; 3]) -> f32 {
        geo::path_dist(OUTSIDE, a, b).unwrap_or_else(|| dist(a, b))
    }
}

impl obrrt_core::Objective for TopHole {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Pyramid::TopHole"
    }
    fn key(&self) -> String {
        "PyramidTopHole".into()
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        35.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == OUTSIDE && s.objects.is_some() && s.alive()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 240 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut obrrt_core::Mem) -> obrrt_core::Eta {
        use obrrt_core::Eta;
        if dropped_in(s) {
            return Eta::Done;
        }
        if s.zone() != OUTSIDE {
            return Eta::Lost;
        }
        if s.objects.is_none() {
            return Eta::Frames(frames_for(dist(s.pos, HOLE)) + 3.0 * 600.0);
        }
        let mut left = Self::pillars(s);
        let mut at = s.pos;
        let mut d = 0.0;
        while !left.is_empty() {
            let (i, l) = left.iter().enumerate().map(|(i, &t)| (i, Self::walk(at, t))).min_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
            d += l;
            at = left.swap_remove(i);
        }
        // The cap: waiting on the pillars (action 0), spinning (1), or gone (deactivated at its explosion).
        let wait = match Self::cap(s) {
            Some(0) => CAP_WAIT,
            Some(1) => CAP_WAIT * 0.5,
            _ => 0.0,
        };
        Eta::Frames(frames_for(d + Self::walk(at, HOLE)) + wait + HOLE_FALL)
    }
    fn success_value(&self) -> f64 {
        8.0
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let p = Self::pillars(s).into_iter().min_by(|a, b| Self::walk(s.pos, *a).total_cmp(&Self::walk(s.pos, *b)));
        Some(p.unwrap_or(HOLE))
    }
    fn kinship(&self) -> u32 {
        10
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == OUTSIDE && !s.has_star(8, 5)
    }
}

/// The level-wide objectives of this file (called from `extra::generate_level`).
pub fn generate_level(b: &St, out: &mut Vec<std::sync::Arc<dyn obrrt_core::Objective>>) {
    if let Some(t) = TopHole::new(b) {
        out.push(std::sync::Arc::new(t));
    }
}
