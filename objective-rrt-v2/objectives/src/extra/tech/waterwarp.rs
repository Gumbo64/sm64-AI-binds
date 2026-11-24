//! `Tech::WaterWarp` (switch `OBRRT_WATERWARP=1`, default off): the water-box
//! up- and downwarps (ref/tas/techniques/CATALOGUE.md G12, ref/techniques/
//! glitches.md 2.1 / 2.2, ref/ukikipedia/Upwarp.md), at every place of a
//! level where the geometry lets one happen, found from the level's data:
//! its water boxes (`gEnvironmentRegions`, [`obrrt_core::geo::static_water`])
//! and static floors / ceilings / walls ([`obrrt_core::geo`] snapshots).
//! `jrbcaveclip` is one such place, written by hand.
//!
//! The conditions, from the decomp (sm64-port):
//! - Water boxes: find_water_level (surface_collision.c) is an xz-only test,
//!   `lo < x < hi`, `lo < z < hi` (exclusive), first water box (id < 50) in
//!   the list wins; a box has a surface y and **no bottom**. Mario's
//!   `waterLevel` is that at his position at the start of the frame
//!   (update_mario_geometry_inputs, mario.c).
//! - **Plunge upwarp** ([`Kind::Plunge`]): any non-submerged action (air:
//!   check_common_airborne_cancels; ground: check_common_moving_cancels and
//!   friends, mario_actions_*.c) at y < waterLevel - 100 calls
//!   set_water_plunge_action (mario.c), which sets y = waterLevel - 100
//!   with no collision check. So walking or falling into a box's footprint
//!   from outside it, deeper than its surface - 100, lifts Mario to the
//!   surface through whatever is between (TAS: TTM pool -> lake +2750 x3,
//!   OOB fall -> lake +4430, THI under-mountain -> summit pond). If a
//!   ceiling is just above the new spot the next water step puts him 160
//!   under it (WDW downtown, Upwarp.md).
//! - **Box clamp downwarp** ([`Kind::Clamp`]): check_common_submerged_cancels
//!   (mario_actions_submerged.c): a submerged Mario at y > waterLevel - 80 is
//!   set to waterLevel - 80 at once if that is above his floor (else he
//!   walks out). So swimming from a box into a lower box's footprint (or the
//!   overlap where a lower box comes first) drops him to the lower surface -
//!   80, through any rock between: no floor below him above that height.
//!   (perform_water_step's own clamp uses the frame's start level.)
//! - **Water ceiling downwarp** ([`Kind::Ceiling`]): perform_water_full_step
//!   (mario_actions_submerged.c) resolves walls once (radius 110), finds the
//!   floor under the new spot (find_floor: only floors at most 78 above it
//!   count), then the ceiling from that floor + 80; a ceiling less than 160
//!   above the new spot puts Mario at ceiling - 160 ("water ceiling
//!   downwarp"). Steep faces with 0.01 < ny < 0.45 are floors (surface_load.c),
//!   which never push sideways: a stroke that ends > 78 under one (a
//!   horizontal step d > 78 ny / |n_xz| into it) finds the floor of the room
//!   below the rock, and the room's ceiling, under him, sends him into it
//!   (TAS: DDD (-452, -1092, 76) -> (-426, -3743, 74) x3, JRB metal surface
//!   clip (4778, 753, -438) -> (4926, -3869, -854); jrbcaveclip).
//! - Leaving water (transition_submerged_to_walking, ACT_WATER_JUMP) and
//!   water seams (Water_seam.md: standing underwater exactly on the shared
//!   edge of two flush boxes, a float-exact spot) move nothing through
//!   geometry; not generated.
//!
//! Spots ([`spots`]): a Plunge spot is a box edge point with a floor just
//!   outside it at least [`GAIN`] + 100 below the surface, dry there, with
//!   headroom, a passage across the edge (no wall at walking or falling
//!   height) and a floor under the inside point (else the plunge lands OOB).
//!   A Clamp spot is a box edge point with a higher box's water just outside
//!   and the step's outcome (floor, ceiling, clamp) simulated at the inside
//!   point from a swim height. A Ceiling spot is a steep floor face in water
//!   whose column a stroke's depth behind the face resolves (floor 78 under,
//!   ceiling from it, then the clamp) [`GAIN`]+ lower, with no wall within
//!   110 to push him back. Deduplicated to one per kind, box and 400 cell.
//!
//! Objective (rough location: region + endpoint, the GA finds the inputs):
//! offered from bases within [`OFFER`] of a spot (the 2 nearest whose kind
//! fits: dry for Plunge, submerged for the others). Success: the warp
//! itself, a one-frame jump of y by [`GAIN`]+ in the warp's direction, into
//! (Plunge) or within (downwarps) water, within [`NEAR`] of the spot, same
//! zone, alive for the next frames, and GAIN+ more than the frame's vy
//! explains (swim speed is 28 and plunge entry halves vy, but metal Mario at
//! hyperspeed falls ~400 a frame in water: the TAS's JRB f76711-76724, which
//! the vy term leaves out). ETA (v2, folded frame by frame in `mem`): outside
//! the warp's precondition band (Plunge: dry and deeper than the spot's
//! surface - 100 - [`GAIN`]; Clamp: submerged above the lower surface - 80 +
//! [`GAIN`]; Ceiling: submerged at the spot's stroke speed) the distance to
//! the spot + [`READY_FRAMES`]; in it the distance to the spot's inside
//! point; each + the [`AFTER`] frames that must follow the warp. `Done`
//! comes at the window's last frame, as eta can't see the frames after.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use obrrt_core::geo::{EnvBox, Tri};
use obrrt_core::{Eta, Mem, Objective, St, frames_for};

/// The smallest y change a warp must make (one frame).
pub const GAIN: f32 = 300.0;
/// Bases within this (xz) of a spot are offered it.
pub const OFFER: f32 = 3000.0;
/// A warp counts for a spot when it starts within this (xz).
pub const NEAR: f32 = 700.0;
/// Edge sampling step and the distance in / out of the edge.
const EDGE_STEP: f32 = 150.0;
const EDGE_OFF: f32 = 40.0;
/// Spots per (kind, box) are merged within this cell size.
const CELL: f32 = 400.0;
/// How far from a fall spot its ledge may be.
const LEDGE_R: f32 = 800.0;
/// A box's level counts as moved (spots made again) past this.
const LEVEL_MOVED: f32 = 50.0;
/// Frames after the warp that must stay alive and in the zone.
const AFTER: usize = 20;
/// Getting into the warp's precondition band from outside it (deeper than
/// the surface - 100 - GAIN out of the water, or up at the right swim
/// height / stroke speed): a second of swimming or a drop.
const READY_FRAMES: f64 = 30.0;

// `mem` of WaterWarp::eta: the previous frame's y (f[0]) and hdist to the
// spot (f[2]); a downwarp candidate's y awaiting the next frame (f[1]); the
// frames left of the post-warp window (f[3]).
const M_PREV: u32 = 1;
const M_PREV_SUB: u32 = 2;
const M_PREV_OK: u32 = 4;
const M_CAND: u32 = 8;
const M_WARPED: u32 = 16;

/// In the post-warp window with `left` frames still to see.
fn window(left: f64) -> Eta {
    if left <= 0.0 { Eta::Done } else { Eta::Frames(left) }
}

const ACT_GROUP_SUBMERGED: u32 = 0x0C0;
const ACT_GROUP_MASK: u32 = 0x1C0;
/// SURFACE_DEATH_PLANE (surface_terrains.h).
const SURFACE_DEATH_PLANE: i16 = 0x0A;

/// `OBRRT_WATERWARP=1`.
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| obrrt_core::sys::env::var("OBRRT_WATERWARP").is_ok_and(|v| v == "1"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// Dry Mario into a box's footprint below its surface - 100: up.
    Plunge,
    /// Submerged into a lower box: down to its surface - 80.
    Clamp,
    /// Submerged through a steep floor face onto a room's ceiling: down.
    Ceiling,
}

/// One place a water warp can happen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spot {
    pub kind: Kind,
    /// Index of the (target) water box in the area's list.
    pub water_box: usize,
    /// Where Mario is just before the warp (outside the edge / on the face).
    pub from: [f32; 3],
    /// Where the warp step goes (inside the edge / behind the face).
    pub to: [f32; 3],
    /// Where the warp puts him (`to` at the new height).
    pub end: [f32; 3],
    /// end.y - from.y (positive up).
    pub gain: f32,
    /// The surface of the box whose rule warps him.
    pub level: f32,
    /// The horizontal step needed (Ceiling: depth into the face).
    pub step: f32,
    /// Plunge: the entry is a fall (death-plane pit) rather than a walk.
    pub fall: bool,
}

// ---------------------------------------------------------------------------
// Geometry queries over triangle lists (binned by xz).

const BIN: f32 = 512.0;

struct Binned {
    tris: Arc<[Tri]>,
    bins: HashMap<(i32, i32), Vec<u32>>,
}

impl Binned {
    fn new(tris: Arc<[Tri]>) -> Binned {
        let mut bins: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (i, t) in tris.iter().enumerate() {
            let (mut x0, mut x1, mut z0, mut z1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for v in &t.v {
                x0 = x0.min(v[0]);
                x1 = x1.max(v[0]);
                z0 = z0.min(v[2]);
                z1 = z1.max(v[2]);
            }
            for bx in (x0 / BIN).floor() as i32..=(x1 / BIN).floor() as i32 {
                for bz in (z0 / BIN).floor() as i32..=(z1 / BIN).floor() as i32 {
                    bins.entry((bx, bz)).or_default().push(i as u32);
                }
            }
        }
        Binned { tris, bins }
    }
    fn near(&self, x: f32, z: f32) -> impl Iterator<Item = &Tri> {
        let k = ((x / BIN).floor() as i32, (z / BIN).floor() as i32);
        self.bins.get(&k).into_iter().flatten().map(|&i| &self.tris[i as usize])
    }
    /// Heights (with the triangles) of every non-vertical triangle over (x, z).
    fn heights(&self, x: f32, z: f32) -> Vec<(f32, &Tri)> {
        let mut out = Vec::new();
        for t in self.near(x, z) {
            if let Some(y) = height_at(t, x, z) {
                out.push((y, t));
            }
        }
        out
    }
    /// find_floor's answer: the highest floor at most `y` + 78.
    fn floor_under(&self, x: f32, y: f32, z: f32) -> Option<(f32, i16)> {
        self.heights(x, z).into_iter().filter(|h| h.0 <= y + 78.0).max_by(|a, b| a.0.total_cmp(&b.0)).map(|h| (h.0, h.1.kind))
    }
    /// find_ceil's answer from `y`: the lowest ceiling at least `y` - 78.
    fn ceil_over(&self, x: f32, y: f32, z: f32) -> Option<f32> {
        self.heights(x, z).into_iter().map(|h| h.0).filter(|&h| h >= y - 78.0).min_by(f32::total_cmp)
    }
    /// Whether the horizontal segment a -> b at height y crosses a triangle.
    fn crosses(&self, a: [f32; 3], b: [f32; 3]) -> bool {
        let mid = [(a[0] + b[0]) / 2.0, (a[2] + b[2]) / 2.0];
        let mut seen = std::collections::HashSet::new();
        for p in [[a[0], a[2]], mid, [b[0], b[2]]] {
            let k = ((p[0] / BIN).floor() as i32, (p[1] / BIN).floor() as i32);
            for &i in self.bins.get(&k).into_iter().flatten() {
                if seen.insert(i) && seg_tri(a, b, &self.tris[i as usize]) {
                    return true;
                }
            }
        }
        false
    }
    /// Whether a triangle (a wall) is within `r` of p, in its plane's
    /// direction and over its face.
    fn within(&self, p: [f32; 3], r: f32) -> bool {
        self.near(p[0], p[2]).any(|t| {
            let d = dot(t.n, sub(p, t.v[0]));
            if d.abs() >= r {
                return false;
            }
            let q = sub(p, scale(t.n, d));
            inside3(t, q)
        })
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn scale(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

/// The triangle's plane height at (x, z) if (x, z) is over it.
fn height_at(t: &Tri, x: f32, z: f32) -> Option<f32> {
    let [a, b, c] = t.v;
    let side = |p: [f32; 3], q: [f32; 3]| (q[0] - p[0]) * (z - p[2]) - (q[2] - p[2]) * (x - p[0]);
    let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
    let inside = (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0) || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0);
    if !inside || t.n[1].abs() < 1e-3 {
        return None;
    }
    Some(a[1] - (t.n[0] * (x - a[0]) + t.n[2] * (z - a[2])) / t.n[1])
}

/// Whether q (in t's plane) is inside t.
fn inside3(t: &Tri, q: [f32; 3]) -> bool {
    let [a, b, c] = t.v;
    let s1 = dot(cross(sub(b, a), sub(q, a)), t.n);
    let s2 = dot(cross(sub(c, b), sub(q, b)), t.n);
    let s3 = dot(cross(sub(a, c), sub(q, c)), t.n);
    (s1 >= 0.0 && s2 >= 0.0 && s3 >= 0.0) || (s1 <= 0.0 && s2 <= 0.0 && s3 <= 0.0)
}

/// Segment a-b against triangle t.
fn seg_tri(a: [f32; 3], b: [f32; 3], t: &Tri) -> bool {
    let (da, db) = (dot(t.n, sub(a, t.v[0])), dot(t.n, sub(b, t.v[0])));
    if (da > 0.0) == (db > 0.0) || da == db {
        return false;
    }
    let k = da / (da - db);
    let q = [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
    inside3(t, q)
}

/// find_water_level over `boxes`: the first water box containing (x, z).
pub fn level_at(boxes: &[EnvBox], x: f32, z: f32) -> Option<(usize, f32)> {
    boxes.iter().enumerate().find(|(_, b)| b.is_water() && b.contains(x, z)).map(|(i, b)| (i, b.y))
}

// ---------------------------------------------------------------------------
// Spot finding.

struct Geo<'a> {
    floors: Binned,
    ceils: Binned,
    walls: Binned,
    boxes: &'a [EnvBox],
}

/// Every warp spot of an area from its static floors, ceilings, walls and
/// environment boxes.
pub fn spots(floors: Arc<[Tri]>, ceils: Arc<[Tri]>, walls: Arc<[Tri]>, boxes: &[EnvBox]) -> Vec<Spot> {
    let g = Geo { floors: Binned::new(floors), ceils: Binned::new(ceils), walls: Binned::new(walls), boxes };
    let mut out = Vec::new();
    for (bi, b) in boxes.iter().enumerate() {
        if !b.is_water() {
            continue;
        }
        for (p, outward) in edge_points(b) {
            let i = [p[0] - outward[0] * EDGE_OFF, 0.0, p[1] - outward[1] * EDGE_OFF];
            let o = [p[0] + outward[0] * EDGE_OFF, 0.0, p[1] + outward[1] * EDGE_OFF];
            if level_at(boxes, i[0], i[2]).map(|l| l.0) != Some(bi) {
                continue;
            }
            g.plunge(bi, b.y, i, o, &mut out);
            g.clamp(bi, b.y, i, o, &mut out);
        }
    }
    g.ceiling(&mut out);
    dedup(out)
}

/// Points along a box's edges with their outward xz normal.
fn edge_points(b: &EnvBox) -> Vec<([f32; 2], [f32; 2])> {
    let mut v = Vec::new();
    let mut run = |a: [f32; 2], c: [f32; 2], n: [f32; 2]| {
        let len = ((c[0] - a[0]).powi(2) + (c[1] - a[1]).powi(2)).sqrt();
        let k = (len / EDGE_STEP).floor().max(1.0) as usize;
        for j in 0..k {
            let t = (j as f32 + 0.5) / k as f32;
            v.push(([a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t], n));
        }
    };
    let ([x0, z0], [x1, z1]) = (b.lo, b.hi);
    run([x0, z0], [x1, z0], [0.0, -1.0]);
    run([x0, z1], [x1, z1], [0.0, 1.0]);
    run([x0, z0], [x0, z1], [-1.0, 0.0]);
    run([x1, z0], [x1, z1], [1.0, 0.0]);
    v
}

impl Geo<'_> {
    fn plunge(&self, bi: usize, level: f32, i: [f32; 3], o: [f32; 3], out: &mut Vec<Spot>) {
        // A floor under the inside point below the new height (else OOB).
        if self.floors.floor_under(i[0], level - 100.0 - 78.0, i[2]).is_none() {
            return;
        }
        let outside = level_at(self.boxes, o[0], o[2]);
        let mut hs: Vec<(f32, i16)> = self.floors.heights(o[0], o[2]).into_iter().map(|h| (h.0, h.1.kind)).collect();
        hs.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (h, kind) in hs {
            if h >= level - 100.0 - GAIN {
                continue;
            }
            if outside.is_some_and(|l| h < l.1 - 10.0) {
                continue; // in (other) water there: swimming, no plunge
            }
            // Headroom over that floor.
            if self.ceils.ceil_over(o[0], h + 80.0, o[2]).is_some_and(|c| c - h < 160.0) {
                continue;
            }
            let fall = kind == SURFACE_DEATH_PLANE;
            // Walk in at h + 60, or fall in at the deepest height that
            // still gains GAIN.
            let walk_y = h + 60.0;
            let fall_y = level - 100.0 - GAIN - 20.0;
            let fall_y = if fall { fall_y.min(h + 600.0) } else { fall_y };
            let pass = |y: f32| {
                !self.walls.crosses([o[0], y, o[2]], [i[0], y, i[2]])
                    && self.floors.floor_under(i[0], y, i[2]).is_some()
                    && self.ceils.ceil_over(i[0], y, i[2]).is_none_or(|c| c > y + 80.0)
            };
            // Over a floor: walk in, or jump in over whatever blocks the
            // walk (a rim; TAS THI f99603 / WF f17979 are airborne), the
            // lowest height in 150 steps that passes. Over a pit: fall in.
            let air = || {
                let mut y = h + 210.0;
                while y <= fall_y {
                    if pass(y) {
                        return Some(y);
                    }
                    y += 150.0;
                }
                None
            };
            let y = if !fall && pass(walk_y) {
                walk_y
            } else if !fall && let Some(y) = air() {
                y
            } else if fall && fall_y > h + 100.0 && pass(fall_y) && self.fall_ok(o, h, fall_y, level) {
                fall_y
            } else {
                continue;
            };
            let mut end_y = level - 100.0;
            if let Some((f2, _)) = self.floors.floor_under(i[0], end_y, i[2])
                && let Some(c) = self.ceils.ceil_over(i[0], f2 + 80.0, i[2])
                && c - end_y < 160.0
                && c - f2 >= 160.0
            {
                end_y = c - 160.0;
            }
            if end_y - y < GAIN {
                continue;
            }
            out.push(Spot {
                kind: Kind::Plunge,
                water_box: bi,
                from: [o[0], y, o[2]],
                to: [i[0], y, i[2]],
                end: [i[0], end_y, i[2]],
                gain: end_y - y,
                level,
                step: 0.0,
                fall: y != walk_y,
                // (fall = an airborne entry: a jump over a rim or a pit fall)
            });
            return; // the highest qualifying floor at this edge point
        }
    }

    /// A fall entry over a death-plane pit (into the footprint while falling
    /// past `fall_y`) needs somewhere to fall from: the column at `o` open from `h` up to the surface + 1000 (he
    /// falls straight past the edge), and a dry ledge within [`LEDGE_R`] of
    /// `o`, outside the box, between `fall_y` + 100 and the surface + 2000
    /// (walk or jump off it). Without the ledge rule the lab listed pit
    /// spots nothing can reach (CCM's slide-end pool, 0/5 in the 50M smoke);
    /// with it the TAS's void fall into TTM's lake (f87300) stays.
    fn fall_ok(&self, o: [f32; 3], h: f32, fall_y: f32, level: f32) -> bool {
        let open = self.floors.heights(o[0], o[2]).iter().all(|f| f.0 <= h + 1.0 || f.0 > level + 1000.0);
        if !open {
            return false;
        }
        for r in [200.0f32, 400.0, LEDGE_R] {
            for k in 0..8 {
                let a = k as f32 * std::f32::consts::FRAC_PI_4;
                let (x, z) = (o[0] + r * a.cos(), o[2] + r * a.sin());
                if self.boxes.iter().any(|b| b.is_water() && b.contains(x, z) && (b.y - level).abs() < 1.0) {
                    continue;
                }
                let wet = level_at(self.boxes, x, z);
                if self.floors.heights(x, z).iter().any(|f| {
                    f.1.kind != SURFACE_DEATH_PLANE
                        && f.0 > fall_y + 100.0
                        && f.0 < level + 2000.0
                        && f.1.n[1] > 0.5
                        && wet.is_none_or(|l| f.0 > l.1 - 10.0)
                }) {
                    return true;
                }
            }
        }
        false
    }

    /// The water step + submerged cancel from `y` to `to`: Mario's height
    /// after them, or None if the step is cancelled / he walks out.
    fn water_step(&self, to: [f32; 3], y: f32, level_to: Option<f32>) -> Option<f32> {
        let (f, _) = self.floors.floor_under(to[0], y, to[2])?;
        let c = self.ceils.ceil_over(to[0], f + 80.0, to[2]).unwrap_or(20000.0);
        let mut y2 = if y >= f {
            if c - y >= 160.0 {
                y
            } else if c - f < 160.0 {
                return None;
            } else {
                c - 160.0
            }
        } else {
            if c - f < 160.0 {
                return None;
            }
            f
        };
        let l = level_to?;
        if y2 > l - 80.0 {
            if l - 80.0 > f {
                y2 = l - 80.0;
            } else {
                return None;
            }
        }
        Some(y2)
    }

    fn clamp(&self, bi: usize, level: f32, i: [f32; 3], o: [f32; 3], out: &mut Vec<Spot>) {
        let Some((_, la)) = level_at(self.boxes, o[0], o[2]) else { return };
        if la - level < GAIN + 20.0 {
            return;
        }
        // A swim height at o: from just under the surface down in steps.
        let mut y = la - 100.0;
        while y >= level - 80.0 + GAIN {
            let swim = self.floors.floor_under(o[0], y, o[2]).is_some_and(|(f, _)| f < y - 20.0)
                && self.ceils.ceil_over(o[0], y, o[2]).is_none_or(|c| c > y + 160.0);
            if swim && !self.walls.crosses([o[0], y, o[2]], [i[0], y, i[2]]) && !self.walls.within([i[0], y, i[2]], 110.0) {
                if let Some(y2) = self.water_step(i, y, Some(level))
                    && y - y2 >= GAIN
                {
                    out.push(Spot {
                        kind: Kind::Clamp,
                        water_box: bi,
                        from: [o[0], y, o[2]],
                        to: [i[0], y, i[2]],
                        end: [i[0], y2, i[2]],
                        gain: y2 - y,
                        level,
                        step: 2.0 * EDGE_OFF,
                        fall: false,
                    });
                }
                return;
            }
            y -= 100.0;
        }
    }

    fn ceiling(&self, out: &mut Vec<Spot>) {
        for t in self.floors.tris.iter() {
            let ny = t.n[1];
            if !(0.01 < ny && ny < 0.45) {
                continue;
            }
            let nh = (t.n[0] * t.n[0] + t.n[2] * t.n[2]).sqrt();
            if nh < 1e-3 {
                continue;
            }
            let dir = [t.n[0] / nh, t.n[2] / nh];
            // The horizontal step that ends 78 under the face.
            let need = 78.0 * ny / nh + 4.0;
            let c = scale([t.v[0][0] + t.v[1][0] + t.v[2][0], t.v[0][1] + t.v[1][1] + t.v[2][1], t.v[0][2] + t.v[1][2] + t.v[2][2]], 1.0 / 3.0);
            let samples = [c, mid(c, t.v[0]), mid(c, t.v[1]), mid(c, t.v[2])];
            for p in samples {
                let Some((bi, level)) = level_at(self.boxes, p[0], p[2]) else { continue };
                if p[1] > level - 80.0 - 10.0 {
                    continue; // not under water
                }
                for d in [need.max(10.0), need.max(28.0), 60.0, 110.0] {
                    let q = [p[0] - dir[0] * d, p[1], p[2] - dir[1] * d];
                    if self.walls.within(q, 110.0) {
                        continue;
                    }
                    let lq = level_at(self.boxes, q[0], q[2]).map(|l| l.1);
                    let Some(y2) = self.water_step(q, p[1], lq) else { continue };
                    if p[1] - y2 >= GAIN {
                        out.push(Spot {
                            kind: if lq.is_some_and(|l| (y2 - (l - 80.0)).abs() < 1.0) { Kind::Clamp } else { Kind::Ceiling },
                            water_box: bi,
                            from: p,
                            to: q,
                            end: [q[0], y2, q[2]],
                            gain: y2 - p[1],
                            level,
                            step: d,
                            fall: false,
                        });
                        break;
                    }
                }
            }
        }
    }
}

fn mid(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0]
}

fn cell_of(p: [f32; 3]) -> (i32, i32, i32) {
    ((p[0] / CELL).floor() as i32, (p[1] / CELL).floor() as i32, (p[2] / CELL).floor() as i32)
}

/// One spot per (kind, box, cell of `from`): the biggest |gain|, then the
/// smallest step.
fn dedup(v: Vec<Spot>) -> Vec<Spot> {
    let mut best: HashMap<(Kind, usize, (i32, i32, i32)), Spot> = HashMap::new();
    for s in v {
        let k = (s.kind, s.water_box, cell_of(s.from));
        match best.get(&k) {
            Some(b) if (b.gain.abs(), -b.step) >= (s.gain.abs(), -s.step) => {}
            _ => {
                best.insert(k, s);
            }
        }
    }
    let mut out: Vec<Spot> = best.into_values().collect();
    out.sort_by(|a, b| {
        (a.kind as u8, a.water_box, cell_of(a.from)).cmp(&(b.kind as u8, b.water_box, cell_of(b.from))).then(b.gain.abs().total_cmp(&a.gain.abs()))
    });
    out
}

/// A zone's spots from its published geometry, with the water boxes as
/// `b` sees them: the box Mario is in takes his current water level
/// (St::water_level, to 10) when it is [`LEVEL_MOVED`]+ off the snapshot's (WDW's level
/// moves with the water diamonds after the zone is stored). Cached per
/// zone and box levels; None until the zone is published
/// (obrrt_core::mover), so what a round sees is fixed.
pub fn zone_spots(b: &St) -> Option<Arc<Vec<Spot>>> {
    type Key = ((u16, u8), Vec<i32>);
    static C: OnceLock<RwLock<HashMap<Key, Arc<Vec<Spot>>>>> = OnceLock::new();
    let c = C.get_or_init(|| RwLock::new(HashMap::new()));
    let zone = b.zone();
    if !obrrt_core::mover::published(zone) {
        return None;
    }
    let mut boxes = obrrt_core::geo::static_water(zone)?.to_vec();
    if b.water_level > -11000
        && let Some((i, y)) = level_at(&boxes, b.pos[0], b.pos[2])
        && (y - b.water_level as f32).abs() >= LEVEL_MOVED
    {
        // Rounded: WDW's surface bobs a unit a frame.
        boxes[i].y = (b.water_level as f32 / 10.0).round() * 10.0;
    }
    let key: Key = (zone, boxes.iter().map(|x| x.y as i32).collect());
    if let Some(r) = c.read().unwrap().get(&key) {
        return Some(r.clone());
    }
    let floors = obrrt_core::geo::static_floors(zone)?;
    let ceils = obrrt_core::geo::static_ceils(zone)?;
    let walls = obrrt_core::geo::static_walls(zone)?;
    let r = Arc::new(spots(floors, ceils, walls, &boxes));
    Some(c.write().unwrap().entry(key).or_insert(r).clone())
}

// ---------------------------------------------------------------------------
// The warp test (success) and the objective.

fn submerged(s: &St) -> bool {
    s.action & ACT_GROUP_MASK == ACT_GROUP_SUBMERGED
}

fn hs(s: &St) -> f32 {
    (s.vel[0] * s.vel[0] + s.vel[2] * s.vel[2]).sqrt()
}

/// Whether prev -> cur (consecutive frames) is a water warp, and which:
/// same zone, all alive and in control, y changed GAIN+ (and GAIN+ more
/// than the frame's vy), and the mechanism's own height checks out:
/// - Plunge: dry -> submerged, up, from below cur's water level - 100 to
///   within 120 of it (set_water_plunge_action, then one water step);
/// - downwarp (both submerged): `next` (the frame after) has to show it:
///   Clamp when cur.y = next's water level - 80, Ceiling when next's
///   ceiling (found from cur's position) is 160 above cur.y.
///
/// The height checks leave out metal Mario's hyperspeed falls and floor
/// snaps in water (JRB, TAS f76711-76724: ~400 a frame at vy 0), which
/// move him through the water, not through a warp.
pub fn warp(prev: &St, cur: &St, next: Option<&St>) -> Option<Kind> {
    if prev.zone() != cur.zone() || !prev.alive() || !cur.alive() || !prev.controllable() || !cur.controllable() {
        return None;
    }
    let dy = cur.pos[1] - prev.pos[1];
    if (dy - cur.vel[1]).abs() < GAIN {
        return None;
    }
    if dy >= GAIN && !submerged(prev) && submerged(cur) {
        let l = cur.water_level as f32 - 100.0;
        return (prev.pos[1] < l && (cur.pos[1] - l).abs() < 120.0).then_some(Kind::Plunge);
    }
    if -dy >= GAIN && submerged(prev) && submerged(cur) {
        let n = next?;
        if n.zone() != cur.zone() || !n.alive() {
            return None;
        }
        if (cur.pos[1] - (n.water_level as f32 - 80.0)).abs() < 2.0 {
            return Some(Kind::Clamp);
        }
        if (n.ceil_y - cur.pos[1] - 160.0).abs() < 2.0 {
            return Some(Kind::Ceiling);
        }
    }
    None
}

fn kind_matches(spot: Kind, seen: Kind) -> bool {
    match spot {
        Kind::Plunge => seen == Kind::Plunge,
        Kind::Clamp | Kind::Ceiling => seen != Kind::Plunge,
    }
}

pub struct WaterWarp {
    zone: (u16, u8),
    spot: Spot,
}

impl WaterWarp {
    pub fn new(zone: (u16, u8), spot: Spot) -> WaterWarp {
        WaterWarp { zone, spot }
    }

    /// The precondition band of the spot's warp (progress counts only
    /// states in it).
    fn ready(&self, s: &St) -> bool {
        let sp = &self.spot;
        match sp.kind {
            Kind::Plunge => !submerged(s) && s.pos[1] < sp.level - 100.0 - GAIN,
            Kind::Clamp => submerged(s) && s.pos[1] >= sp.end[1] + GAIN,
            Kind::Ceiling => submerged(s) && hs(s) >= sp.step.min(26.0),
        }
    }

    fn fits_base(&self, s: &St) -> bool {
        match self.spot.kind {
            Kind::Plunge => !submerged(s),
            Kind::Clamp | Kind::Ceiling => submerged(s),
        }
    }
}

/// Up to 2 spots within [`OFFER`] of `b` whose kind fits b's state, ready
/// ones (b in the warp's precondition band) first, then by distance.
pub fn generate(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    if !on() {
        return;
    }
    obrrt_core::geo::want_static_floors();
    if !b.alive() || !b.controllable() {
        return;
    }
    let Some(all) = zone_spots(b) else { return };
    // Spots whose precondition b already meets first (e.g. deep enough
    // under the box's surface for a Plunge), then the nearest.
    let mut near: Vec<(bool, f32, &Spot)> = all
        .iter()
        .map(|s| (!WaterWarp::new(b.zone(), *s).ready(b), b.hdist(s.from), s))
        .filter(|(_, d, s)| *d < OFFER && WaterWarp::new(b.zone(), **s).fits_base(b))
        .collect();
    near.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    for (_, _, s) in near.into_iter().take(2) {
        out.push(Arc::new(WaterWarp::new(b.zone(), *s)));
    }
}

impl Objective for WaterWarp {
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 1, air: true }
    }
    fn precision(&self) -> obrrt_core::Precision {
        obrrt_core::Precision::Precise
    }
    fn kind(&self) -> &'static str {
        "Tech::WaterWarp"
    }
    fn key(&self) -> String {
        let c = cell_of(self.spot.from);
        format!("Tech::WaterWarp {:?} {:?} box{}@{} {},{},{}", self.spot.kind, self.zone, self.spot.water_box, self.spot.level as i32, c.0, c.1, c.2)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        2.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && s.alive() && s.controllable() && self.fits_base(s) && s.hdist(self.spot.from) < OFFER
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 20, generations: 6, horizon: 200 })
    }
    fn eta(&self, s: &St, _start: &St, mem: &mut Mem) -> Eta {
        // v1: no frame after the start may leave the zone or die before the
        // warp and its AFTER frames are over.
        let first = mem.flags & M_PREV == 0;
        if !first && (s.zone() != self.zone || !s.alive()) {
            return Eta::Lost;
        }
        if mem.flags & M_WARPED != 0 {
            // Done once the window's AFTER frames (the warp frame on) have
            // all been alive in the zone.
            mem.f[3] -= 1.0;
            return window(mem.f[3]);
        }
        // A downwarp seen last frame (prev -> cur both submerged, GAIN+ down
        // beyond vy, prev near the spot) is decided with this frame as
        // `warp`'s `next`.
        if mem.flags & M_CAND != 0 {
            mem.flags &= !M_CAND;
            let cy = mem.f[1] as f32;
            let k = if (cy - (s.water_level as f32 - 80.0)).abs() < 2.0 {
                Some(Kind::Clamp)
            } else if (s.ceil_y - cy - 160.0).abs() < 2.0 {
                Some(Kind::Ceiling)
            } else {
                None
            };
            if k.is_some_and(|k| kind_matches(self.spot.kind, k)) {
                mem.flags |= M_WARPED;
                // The warp frame and this one are in the window.
                mem.f[3] = AFTER as f64 - 2.0;
                return window(mem.f[3]);
            }
        }
        // prev -> s, as in `warp` (the parts decidable without `next`).
        if !first && mem.flags & M_PREV_OK != 0 && s.alive() && s.controllable() && mem.f[2] < NEAR as f64 {
            let py = mem.f[0] as f32;
            let dy = s.pos[1] - py;
            let prev_sub = mem.flags & M_PREV_SUB != 0;
            if (dy - s.vel[1]).abs() >= GAIN {
                if dy >= GAIN && !prev_sub && submerged(s) && self.spot.kind == Kind::Plunge {
                    let l = s.water_level as f32 - 100.0;
                    if py < l && (s.pos[1] - l).abs() < 120.0 {
                        mem.flags |= M_WARPED;
                        mem.f[3] = AFTER as f64 - 1.0;
                        return window(mem.f[3]);
                    }
                }
                if -dy >= GAIN && prev_sub && submerged(s) && self.spot.kind != Kind::Plunge {
                    mem.flags |= M_CAND;
                    mem.f[1] = s.pos[1] as f64;
                }
            }
        }
        mem.flags = (mem.flags & !(M_PREV_SUB | M_PREV_OK))
            | M_PREV
            | if submerged(s) { M_PREV_SUB } else { 0 }
            | if s.alive() && s.controllable() { M_PREV_OK } else { 0 };
        mem.f[0] = s.pos[1] as f64;
        mem.f[2] = s.hdist(self.spot.from) as f64;
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // Phases: into the precondition band, to the spot, the warp, and its
        // AFTER frames.
        Eta::Frames(if mem.flags & M_CAND != 0 {
            AFTER as f64 - 1.0
        } else if self.ready(s) {
            frames_for(s.dist(self.spot.to)) + AFTER as f64
        } else {
            frames_for(s.dist(self.spot.from)) + READY_FRAMES + AFTER as f64
        })
    }
    fn success_value(&self) -> f64 {
        4.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let t = self.spot.to;
        Some(match self.spot.kind {
            // Across the edge at his own height (walk / fall in).
            Kind::Plunge | Kind::Clamp => [t[0], s.pos[1], t[2]],
            // Into the face at its depth.
            Kind::Ceiling => t,
        })
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone
    }
    fn kinship(&self) -> u32 {
        10
    }
}
