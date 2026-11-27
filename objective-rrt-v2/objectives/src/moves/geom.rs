//! Local geometry for pattern matching: a per-zone grid index over the
//! static floor triangles ([`obrrt_core::geo::static_floors`]), and the probe
//! along the line from Mario to the target ([`Ahead`]).
//!
//! Read-only caches. A zone's grid exists once its floors have been
//! snapshot (the first state in the zone read with objects, after
//! [`obrrt_core::geo::want_static_floors`]); until then probes come back
//! `None` and moves fall back to state-only rules.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use obrrt_core::St;
use obrrt_core::geo::Tri;

/// Grid cell side.
const CELL: f32 = 256.0;

/// Floors Mario can't stand on or that kill him (death planes, lava,
/// quicksand pits, ...), as in core's mesh.
fn excluded(kind: i16) -> bool {
    matches!(kind, 0x01 | 0x0A | 0x12 | 0x22 | 0x23 | 0x24 | 0x2D | 0x38)
}

/// The slope past which Mario slides (mario_floor_is_slippery, mario.c),
/// by surface class: the normal's y at or below which the floor is
/// "above slide" (38° default, 20° slippery, 10° very slippery, never for
/// not slippery).
pub fn slippery_limit(kind: i16) -> f32 {
    match kind {
        0x13 | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => 0.9848077,
        0x14 | 0x2A | 0x35 | 0x79 | 0x2E => 0.9396926,
        0x15 | 0x37 => 0.0,
        _ => 0.7880108,
    }
}

/// The slope from which a jump is a steep jump (mario_floor_is_steep:
/// 15° very slippery, 20° slippery, 30° default, never not slippery).
pub fn steep_limit(kind: i16) -> f32 {
    match kind {
        0x13 | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => 0.9659258,
        0x14 | 0x2A | 0x35 | 0x79 | 0x2E => 0.9396926,
        0x15 | 0x37 => 0.0,
        _ => 0.8660254,
    }
}

/// The slope accel against walking uphill (apply_slope_accel): per frame
/// `accel * steepness` comes off the forward speed.
pub fn slope_accel(kind: i16) -> f32 {
    match kind {
        0x13 | 0x36 | 0x73 | 0x74 | 0x75 | 0x78 => 5.3,
        0x14 | 0x2A | 0x35 | 0x79 | 0x2E => 2.7,
        0x15 | 0x37 => 0.0,
        _ => 1.7,
    }
}

/// A floor found by a probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Floor {
    pub y: f32,
    pub normal: [f32; 3],
    pub kind: i16,
}

struct Entry {
    v: [[f32; 3]; 3],
    n: [f32; 3],
    kind: i16,
}

/// One zone's floors and walls, bucketed by XZ cell.
pub struct Grid {
    tris: Vec<Entry>,
    cells: HashMap<(i32, i32), Vec<u32>>,
    walls: Vec<Entry>,
    wall_cells: HashMap<(i32, i32), Vec<u32>>,
}

fn bucket(cells: &mut HashMap<(i32, i32), Vec<u32>>, i: u32, v: &[[f32; 3]; 3]) {
    let (mut x0, mut x1, mut z0, mut z1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in v {
        x0 = x0.min(p[0]);
        x1 = x1.max(p[0]);
        z0 = z0.min(p[2]);
        z1 = z1.max(p[2]);
    }
    let (a, b) = (cell(x0, z0), cell(x1, z1));
    for cx in a.0..=b.0 {
        for cz in a.1..=b.1 {
            cells.entry((cx, cz)).or_default().push(i);
        }
    }
}

fn cell(x: f32, z: f32) -> (i32, i32) {
    ((x / CELL).floor() as i32, (z / CELL).floor() as i32)
}

impl Grid {
    pub fn build(tris: &[Tri], walls: &[Tri]) -> Grid {
        let mut out = Grid { tris: Vec::new(), cells: HashMap::new(), walls: Vec::new(), wall_cells: HashMap::new() };
        for t in tris {
            if excluded(t.kind) || t.n[1] < 0.01 {
                continue;
            }
            let i = out.tris.len() as u32;
            out.tris.push(Entry { v: t.v, n: t.n, kind: t.kind });
            bucket(&mut out.cells, i, &t.v);
        }
        for t in walls {
            let i = out.walls.len() as u32;
            out.walls.push(Entry { v: t.v, n: t.n, kind: t.kind });
            bucket(&mut out.wall_cells, i, &t.v);
        }
        out
    }

    /// Distance along the horizontal segment from `p` (dir, length `len`,
    /// at height p[1]) to the first wall it runs into from the wall's front
    /// (walls push Mario out along their normal).
    pub fn wall_hit(&self, p: [f32; 3], dir: [f32; 2], len: f32, min_cos: f32) -> Option<f32> {
        let mut seen: Vec<u32> = Vec::new();
        let mut best: Option<f32> = None;
        let mut t = 0.0;
        while t <= len + 64.0 {
            let c = cell(p[0] + dir[0] * t.min(len), p[2] + dir[1] * t.min(len));
            if let Some(ids) = self.wall_cells.get(&c) {
                for &i in ids {
                    if seen.contains(&i) {
                        continue;
                    }
                    seen.push(i);
                    let w = &self.walls[i as usize];
                    let n = w.n;
                    let toward = n[0] * dir[0] + n[2] * dir[1];
                    if toward > -min_cos {
                        continue;
                    }
                    let o = -(n[0] * w.v[0][0] + n[1] * w.v[0][1] + n[2] * w.v[0][2]);
                    let f0 = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + o;
                    if f0 < -10.0 {
                        continue; // behind it already
                    }
                    let d = f0 / -toward;
                    if d > len || best.is_some_and(|b| d >= b) {
                        continue;
                    }
                    let x = [p[0] + dir[0] * d, p[1], p[2] + dir[1] * d];
                    let inside = {
                        let [a, b, c] = w.v;
                        let side = |a: [f32; 3], b: [f32; 3]| {
                            let e = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                            let q = [x[0] - a[0], x[1] - a[1], x[2] - a[2]];
                            let cr = [e[1] * q[2] - e[2] * q[1], e[2] * q[0] - e[0] * q[2], e[0] * q[1] - e[1] * q[0]];
                            cr[0] * n[0] + cr[1] * n[1] + cr[2] * n[2]
                        };
                        let (s1, s2, s3) = (side(a, b), side(b, c), side(c, a));
                        (s1 >= -1.0 && s2 >= -1.0 && s3 >= -1.0) || (s1 <= 1.0 && s2 <= 1.0 && s3 <= 1.0)
                    };
                    if inside {
                        best = Some(d.max(0.0));
                    }
                }
            }
            t += 64.0;
        }
        best
    }

    /// The highest floor under (x, z) at most `y_max` high.
    pub fn floor_below(&self, x: f32, z: f32, y_max: f32) -> Option<Floor> {
        let ids = self.cells.get(&cell(x, z))?;
        let mut best: Option<Floor> = None;
        for &i in ids {
            let t = &self.tris[i as usize];
            let [a, b, c] = t.v;
            let side = |p: [f32; 3], q: [f32; 3]| (q[0] - p[0]) * (z - p[2]) - (q[2] - p[2]) * (x - p[0]);
            let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
            let inside = (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0) || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0);
            if !inside {
                continue;
            }
            let y = a[1] - (t.n[0] * (x - a[0]) + t.n[2] * (z - a[2])) / t.n[1];
            if y <= y_max && best.is_none_or(|b| y > b.y) {
                best = Some(Floor { y, normal: t.n, kind: t.kind });
            }
        }
        best
    }
}

type Zone = (u16, u8);

#[derive(Default)]
struct Grids(RwLock<HashMap<Zone, Option<Arc<Grid>>>>);

/// This thread's runtime's grids (the engine's own inside its threads).
fn grids() -> Arc<Grids> {
    obrrt_core::runtime::current().ext::<Grids>()
}

/// Installs a grid for a zone (tests).
pub fn insert(zone: Zone, tris: &[Tri]) {
    insert_with_walls(zone, tris, &[]);
}

/// Installs a grid with walls for a zone (tests).
pub fn insert_with_walls(zone: Zone, tris: &[Tri], walls: &[Tri]) {
    grids().0.write().unwrap().insert(zone, Some(Arc::new(Grid::build(tris, walls))));
    obrrt_core::mover::publish_zone(zone);
}

/// Calls `f` with the zone's grid, building it from the floor snapshot the
/// first time one is there.
pub fn with_grid<R>(zone: Zone, f: impl FnOnce(&Grid) -> R) -> Option<R> {
    thread_local! {
        static LAST: std::cell::RefCell<Option<(usize, Zone, Arc<Grid>)>> = const { std::cell::RefCell::new(None) };
    }
    let rt = obrrt_core::runtime::with(|rt| rt.id());
    let hit = LAST.with(|l| l.borrow().as_ref().filter(|(r, z, _)| *r == rt && *z == zone).map(|(_, _, g)| g.clone()));
    let g = match hit {
        Some(g) => g,
        None => {
            // Only zones published at a round boundary (determinism; see
            // obrrt_core::mover). Publication is permanent, so a cached
            // grid stays valid.
            if !obrrt_core::mover::published(zone) {
                return None;
            }
            let grids = grids();
            let known = grids.0.read().unwrap().get(&zone).cloned().flatten();
            let g = match known {
                Some(g) => g,
                None => {
                    let tris = obrrt_core::geo::static_floors(zone)?;
                    let walls = obrrt_core::geo::static_walls(zone).unwrap_or_else(|| Vec::new().into());
                    let g = Arc::new(Grid::build(&tris, &walls));
                    grids.0.write().unwrap().entry(zone).or_insert_with(|| Some(g.clone())).clone().unwrap_or(g)
                }
            };
            LAST.with(|l| *l.borrow_mut() = Some((rt, zone, g.clone())));
            g
        }
    };
    Some(f(&g))
}

/// Distances ahead (along the target direction) the probe samples.
pub const SAMPLES: [f32; 10] = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 650.0, 800.0, 1000.0];

/// What lies between Mario and the target along the straight line.
#[derive(Debug, Clone, Default)]
pub struct Ahead {
    /// Per [`SAMPLES`] distance (only those short of the target, plus one
    /// past): the highest floor at most [`STEP_REACH`] above Mario's floor.
    pub low: Vec<(f32, Option<Floor>)>,
    /// Per sample: the highest floor at most 3000 above (cliffs, walls).
    pub high: Vec<(f32, Option<f32>)>,
    /// The probe's direction (x, z).
    pub dir: [f32; 2],
    /// Distance to the first wall Mario's body (30 and 60 above `base`)
    /// would run into along the probe, within its reach.
    pub wall: Option<f32>,
}

/// Highest step a jump can reach (a triple / side flip about 350).
pub const STEP_REACH: f32 = 350.0;

impl Ahead {
    /// First sample with a floor more than `rise` above `base` (and within
    /// [`STEP_REACH`]): (distance of the last sample short of it, height
    /// above base).
    pub fn step_up(&self, base: f32, rise: f32) -> Option<(f32, f32)> {
        // A step, not a ramp: the floor is more than `rise` above base,
        // and higher than the sample before it by more than its own plane
        // explains (a discontinuity of at least 40).
        let (mut d0, mut y0) = (0.0f32, base);
        for (d, f) in self.low.iter() {
            if let Some(f) = f {
                let n = f.normal;
                let plane = if n[1] > 0.01 { -(n[0] * self.dir[0] + n[2] * self.dir[1]) / n[1] * (d - d0) } else { 0.0 };
                if f.y - base > rise && f.y - y0 - plane > 40.0 {
                    return Some((d0, f.y - base));
                }
                y0 = f.y;
            } else {
                y0 = f32::MAX; // after a gap: a far side is the gap rule's
            }
            d0 = *d;
        }
        None
    }

    /// First sample whose highest floor is more than [`STEP_REACH`] above
    /// `base`: a cliff no jump climbs.
    pub fn cliff(&self, base: f32) -> Option<f32> {
        self.high.iter().find(|(_, y)| y.is_some_and(|y| y - base > STEP_REACH)).map(|(d, _)| *d)
    }

    /// A gap or a drop: the first sample with no floor at all within
    /// `drop` below `base` (none above either, else it is a cliff), as the
    /// distance of the last sample before it that had one (0 if none: the
    /// edge is right here), and the first sample after it with a floor at
    /// most `back` below `base` (the far side), if any.
    pub fn gap(&self, base: f32, drop: f32, back: f32) -> Option<(f32, Option<f32>)> {
        let i = self.low.iter().zip(&self.high).position(|((_, f), (_, h))| {
            f.is_none_or(|f| f.y < base - drop) && h.is_none_or(|h| h < base - drop)
        })?;
        let edge = if i == 0 { 0.0 } else { self.low[i - 1].0 };
        let gap_at = self.low[i].0;
        let far = self.low.iter().find(|(d, f)| *d > gap_at && f.is_some_and(|f| f.y >= base - back)).map(|(d, _)| *d);
        Some((edge, far))
    }

    /// The first drop of more than 150 below `base` (a gap's edge, as
    /// [`Ahead::gap`]): (distance of the last sample before it, how far
    /// down the floor under it is; infinite over no floor).
    pub fn drop_depth(&self, base: f32) -> Option<(f32, f32)> {
        let i = self.low.iter().zip(&self.high).position(|((_, f), (_, h))| {
            f.is_none_or(|f| f.y < base - 150.0) && h.is_none_or(|h| h < base - 150.0)
        })?;
        let edge = if i == 0 { 0.0 } else { self.low[i - 1].0 };
        Some((edge, self.low[i].1.map_or(f32::INFINITY, |f| base - f.y)))
    }

    /// The steepest rise per unit along `dir` of the floors sampled within
    /// `upto` (by each floor's own plane, so a step of flat floors doesn't
    /// count).
    pub fn steepest(&self, dir: [f32; 2], upto: f32) -> f32 {
        let mut best = f32::MIN;
        for (d, f) in &self.low {
            if *d > upto {
                break;
            }
            if let Some(f) = f {
                let n = f.normal;
                if n[1] > 0.01 {
                    best = best.max(-(n[0] * dir[0] + n[2] * dir[1]) / n[1]);
                }
            }
        }
        best
    }

    /// Mean grade (rise per unit) of the floor from Mario to `upto` ahead,
    /// from the samples with a floor; None without any.
    pub fn grade(&self, base: f32, upto: f32) -> Option<f32> {
        let (mut s, mut n) = (0.0, 0);
        for (d, f) in &self.low {
            if *d > upto {
                break;
            }
            if let Some(f) = f {
                s += (f.y - base) / d;
                n += 1;
            }
        }
        (n > 0).then(|| s / n as f32)
    }

    /// Whether every sample up to `upto` has a floor within ±`tol` of
    /// `base` (a level run-up / landing strip).
    pub fn level(&self, base: f32, upto: f32, tol: f32) -> bool {
        self.low.iter().filter(|(d, _)| *d <= upto).all(|(_, f)| f.is_some_and(|f| (f.y - base).abs() <= tol))
    }
}

/// Probes along the horizontal direction `dir` (unit) from `st` out to just
/// past `reach`, relative to `base` (Mario's floor, or his height in the
/// air). None without a grid for the zone.
pub fn ahead(st: &St, dir: [f32; 2], reach: f32, base: f32) -> Option<Ahead> {
    with_grid(st.zone(), |g| {
        let mut a = Ahead { dir, ..Ahead::default() };
        for &d in SAMPLES.iter() {
            let (x, z) = (st.pos[0] + dir[0] * d, st.pos[2] + dir[1] * d);
            a.low.push((d, g.floor_below(x, z, base + STEP_REACH)));
            a.high.push((d, g.floor_below(x, z, base + 3000.0).map(|f| f.y)));
            if d > reach {
                break;
            }
        }
        let len = reach.min(1000.0);
        let lo = g.wall_hit([st.pos[0], base + 30.0, st.pos[2]], dir, len, 0.1);
        let hi = g.wall_hit([st.pos[0], base + 60.0, st.pos[2]], dir, len, 0.1);
        a.wall = match (lo, hi) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        };
        a
    })
}

/// Distance to the first wall met nearly head-on (the angle between `dir`
/// and the wall's normal reversed within acos(`min_cos`)) along `dir`
/// within `len`, at Mario's body height above `base`.
pub fn wall_ahead(st: &St, dir: [f32; 2], len: f32, base: f32, min_cos: f32) -> Option<f32> {
    with_grid(st.zone(), |g| {
        let lo = g.wall_hit([st.pos[0], base + 30.0, st.pos[2]], dir, len, min_cos);
        let hi = g.wall_hit([st.pos[0], base + 60.0, st.pos[2]], dir, len, min_cos);
        match (lo, hi) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        }
    })
    .flatten()
}

/// [`wall_ahead`] for a wall a jump doesn't clear: it is also there 250
/// above `base` (within 60 of the same distance), so a jump into it hits
/// it rather than landing on top.
pub fn tall_wall_ahead(st: &St, dir: [f32; 2], len: f32, base: f32, min_cos: f32) -> Option<f32> {
    let d = wall_ahead(st, dir, len, base, min_cos)?;
    let top = with_grid(st.zone(), |g| g.wall_hit([st.pos[0], base + 250.0, st.pos[2]], dir, len + 60.0, min_cos)).flatten()?;
    ((top - d).abs() <= 60.0).then_some(d)
}

/// The static floor under (x, z) at most 100 above `y`, if the zone's
/// floors are published (a [`super::arcs::FloorFn`]).
pub fn floor_y(zone: Zone, x: f32, z: f32, y: f32) -> Option<f32> {
    with_grid(zone, |g| g.floor_below(x, z, y + 100.0).map(|f| f.y)).flatten()
}
