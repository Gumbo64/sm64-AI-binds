//! The riser probe over static geometry (ref/blj-conditions.md section 6:
//! "Riser probe (from an external script)"): static wall triangles that are
//! a short step's riser, with a lower floor in front and a higher one
//! behind. A BLJ facing away from the riser (stair) or along it (side)
//! lands each jump on the step top within the jump frame.
//!
//! Pure functions of the zone's static floors and walls
//! ([`obrrt_core::geo::static_floors`] / [`obrrt_core::geo::static_walls`]),
//! so the result depends only on published geometry (deterministic, as
//! `moves::geom`). Per zone the list is computed once and cached.
//!
//! The conditions (decomp, checked in run/examples/riser_lab.rs):
//! - **height** 15..=[`H_MAX`]: the air lower-wall probe sits at y + 30 +
//!   7.5 j on quarter step j of the jump frame (radius 50), and a wall
//!   pushes only where that point is inside its triangle (the 5-unit
//!   lowerY / upperY buffer is only the early-out, surface_collision.c:43),
//!   so a riser of height r is invisible to quarter step j when
//!   r <= 30 + 7.5 j: 37.5 at j = 1, 60 at j = 4 (the cap). Above, a start
//!   within 50 of the plane is pushed back out to 50.
//! - **length** >= 100 (room for the setup along it);
//! - **floors**: a floor within -60..+8 of the riser's foot in front (15,
//!   25 or 40 along the normal from its middle) and one 8..h + 60 above
//!   the foot behind; the normal is flipped when only the reverse holds.
//! - **corner** (a ranking preference): an endpoint within 25 of a tall
//!   (> 150) wall at the same height: the lobby steps' side-wall push.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use obrrt_core::geo::Tri;

/// Riser heights considered (see the module doc).
pub const H_MIN: f32 = 15.0;
pub const H_MAX: f32 = 60.0;
/// Preferred: invisible to the air wall probe from quarter step 1.
pub const H_LOW: f32 = 40.0;
/// Shortest riser.
pub const MIN_LEN: f32 = 100.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Riser {
    /// Horizontal end points (x, z) (the lexicographic min / max vertex).
    pub a: [f32; 2],
    pub b: [f32; 2],
    /// Unit horizontal normal, pointing to the lower floor (the front).
    pub n: [f32; 2],
    pub ylo: f32,
    pub yhi: f32,
    /// An end point by a tall wall.
    pub corner: bool,
}

impl Riser {
    pub fn h(&self) -> f32 {
        self.yhi - self.ylo
    }
    pub fn len(&self) -> f32 {
        (self.b[0] - self.a[0]).hypot(self.b[1] - self.a[1])
    }
    pub fn mid(&self) -> [f32; 3] {
        [(self.a[0] + self.b[0]) / 2.0, self.ylo, (self.a[1] + self.b[1]) / 2.0]
    }
    /// The point at fraction `along` of the riser, `d0` in front of it.
    pub fn point(&self, along: f32, d0: f32) -> [f32; 3] {
        [
            self.a[0] + (self.b[0] - self.a[0]) * along + self.n[0] * d0,
            self.ylo,
            self.a[1] + (self.b[1] - self.a[1]) * along + self.n[1] * d0,
        ]
    }
    /// SM64 yaw of the normal (facing away from the riser).
    pub fn away_yaw(&self) -> i32 {
        (self.n[0].atan2(self.n[1]) * 32768.0 / std::f32::consts::PI) as i32
    }
    /// Signed distance of (x, z) in front of the riser's plane.
    pub fn front(&self, x: f32, z: f32) -> f32 {
        (x - self.a[0]) * self.n[0] + (z - self.a[1]) * self.n[1]
    }
    /// Distance along the riser from `a`.
    pub fn along(&self, x: f32, z: f32) -> f32 {
        let l = self.len().max(1e-3);
        ((x - self.a[0]) * (self.b[0] - self.a[0]) + (z - self.a[1]) * (self.b[1] - self.a[1])) / l
    }
}

const CELL: f32 = 256.0;

fn cell(x: f32, z: f32) -> (i32, i32) {
    ((x / CELL).floor() as i32, (z / CELL).floor() as i32)
}

/// Floors Mario can't stand on or that kill him (as moves::geom).
fn excluded(kind: i16) -> bool {
    matches!(kind, 0x01 | 0x0A | 0x12 | 0x22 | 0x23 | 0x24 | 0x2D | 0x38)
}

/// A grid over floor triangles for column queries.
struct Floors<'a> {
    tris: &'a [Tri],
    cells: HashMap<(i32, i32), Vec<u32>>,
}

impl<'a> Floors<'a> {
    fn new(tris: &'a [Tri]) -> Floors<'a> {
        let mut cells: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (i, t) in tris.iter().enumerate() {
            if excluded(t.kind) || t.n[1] < 0.01 {
                continue;
            }
            let (mut x0, mut x1, mut z0, mut z1) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for p in &t.v {
                x0 = x0.min(p[0]);
                x1 = x1.max(p[0]);
                z0 = z0.min(p[2]);
                z1 = z1.max(p[2]);
            }
            let (a, b) = (cell(x0, z0), cell(x1, z1));
            for cx in a.0..=b.0 {
                for cz in a.1..=b.1 {
                    cells.entry((cx, cz)).or_default().push(i as u32);
                }
            }
        }
        Floors { tris, cells }
    }

    /// Whether some floor over (x, z) has its height in `lo..=hi`.
    fn any_in(&self, x: f32, z: f32, lo: f32, hi: f32) -> bool {
        let Some(ids) = self.cells.get(&cell(x, z)) else { return false };
        ids.iter().any(|&i| {
            let t = &self.tris[i as usize];
            let [a, b, c] = t.v;
            let side = |p: [f32; 3], q: [f32; 3]| (q[0] - p[0]) * (z - p[2]) - (q[2] - p[2]) * (x - p[0]);
            let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
            let inside = (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0) || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0);
            if !inside {
                return false;
            }
            let y = a[1] - (t.n[0] * (x - a[0]) + t.n[2] * (z - a[2])) / t.n[1];
            (lo..=hi).contains(&y)
        })
    }
}

fn ends(t: &Tri) -> ([f32; 2], [f32; 2]) {
    let mut pts: Vec<[f32; 2]> = t.v.iter().map(|p| [p[0], p[2]]).collect();
    pts.sort_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
    (pts[0], pts[2])
}

fn yr(t: &Tri) -> (f32, f32) {
    let ys = t.v.map(|p| p[1]);
    (ys[0].min(ys[1]).min(ys[2]), ys[0].max(ys[1]).max(ys[2]))
}

/// Every riser of a zone (unranked, deduplicated per the module doc).
pub fn find(floors: &[Tri], walls: &[Tri], h_max: f32) -> Vec<Riser> {
    // Thresholds (OBRRT_BLJ_DETECTOR; defaults the module doc's).
    let pd = super::params::get();
    let fl = Floors::new(floors);
    let tall: Vec<([f32; 2], [f32; 2], f32, f32)> = walls
        .iter()
        .filter(|t| {
            let (lo, hi) = yr(t);
            hi - lo > pd.corner_tall
        })
        .map(|t| {
            let (a, b) = ends(t);
            let (lo, hi) = yr(t);
            (a, b, lo, hi)
        })
        .collect();
    let corner = |p: [f32; 2], y: f32| {
        tall.iter().any(|&(a, b, lo, hi)| {
            if lo > y + 60.0 || hi < y {
                return false;
            }
            let (dx, dz) = (b[0] - a[0], b[1] - a[1]);
            let l2 = (dx * dx + dz * dz).max(1.0);
            let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / l2).clamp(0.0, 1.0);
            (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dz) < pd.corner_dist
        })
    };
    let mut out: Vec<Riser> = Vec::new();
    for t in walls {
        let (ylo, yhi) = yr(t);
        let h = yhi - ylo;
        if !(pd.riser_h_min..=h_max).contains(&h) {
            continue;
        }
        let (a, b) = ends(t);
        if (b[0] - a[0]).hypot(b[1] - a[1]) < pd.riser_min_len {
            continue;
        }
        let nl = t.n[0].hypot(t.n[2]);
        if nl < 0.5 {
            continue;
        }
        let mut n = [t.n[0] / nl, t.n[2] / nl];
        let c = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
        let lower = |x: f32, z: f32| fl.any_in(x, z, ylo + pd.riser_front_lo, ylo + pd.riser_front_hi);
        let higher = |x: f32, z: f32| fl.any_in(x, z, ylo + (pd.riser_front_hi + 0.001), yhi + pd.riser_behind_extra);
        let probe = |s: f32, f: &dyn Fn(f32, f32) -> bool| [15.0, 25.0, 40.0].iter().any(|e| f(c[0] + s * n[0] * e, c[1] + s * n[1] * e));
        if !(probe(1.0, &lower) && probe(-1.0, &higher)) {
            if probe(-1.0, &lower) && probe(1.0, &higher) {
                n = [-n[0], -n[1]];
            } else {
                continue;
            }
        }
        let r = Riser { a, b, n, ylo, yhi, corner: corner(a, ylo) || corner(b, ylo) };
        // The other triangle of the same quad (same foot, middles close).
        if out.iter().any(|q| (q.ylo - r.ylo).abs() <= 5.0 && dist2(q.mid(), r.mid()) <= 120.0 && q.n[0] * r.n[0] + q.n[1] * r.n[1] > 0.9) {
            continue;
        }
        out.push(r);
    }
    out
}

fn dist2(p: [f32; 3], q: [f32; 3]) -> f32 {
    (p[0] - q[0]).hypot(p[2] - q[2])
}

/// The script's ranking: same level as `near` (within 40) first, then
/// low (<= [`H_LOW`]), then corner, then not facing the goal (the BLJ
/// moves Mario through the riser, against its normal), then straight-line
/// distance. At most `k` within `reach` of `near` (horizontally),
/// deduplicated by foot height (5) and middle (120).
pub fn rank(all: &[Riser], near: [f32; 3], reach: f32, goal: Option<[f32; 3]>, k: usize) -> Vec<Riser> {
    let mut v: Vec<(u32, f32, Riser)> = all
        .iter()
        .filter(|r| dist2(r.mid(), near) <= reach)
        .map(|r| {
            let c = r.mid();
            let level = if (r.ylo - near[1]).abs() <= 40.0 { 0 } else { 4 };
            let pen = match goal {
                Some(g) if r.n[0] * (g[0] - c[0]) + r.n[1] * (g[2] - c[2]) > 0.0 => 2,
                _ => 0,
            };
            (level + if r.h() <= H_LOW { 0 } else { 2 } + if r.corner { 0 } else { 1 } + pen, dist2(c, near), *r)
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut res: Vec<Riser> = Vec::new();
    for (_, _, r) in v {
        if !res.iter().any(|q| (q.ylo - r.ylo).abs() <= 5.0 && dist2(q.mid(), r.mid()) <= 120.0) {
            res.push(r);
        }
        if res.len() >= k {
            break;
        }
    }
    res
}

/// A zone's risers from its published geometry (cached per zone; `None`
/// until the zone is published: obrrt_core::mover).
pub fn zone_risers(zone: (u16, u8)) -> Option<Arc<Vec<Riser>>> {
    static C: OnceLock<RwLock<HashMap<(u16, u8), Arc<Vec<Riser>>>>> = OnceLock::new();
    let c = C.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(r) = c.read().unwrap().get(&zone) {
        return Some(r.clone());
    }
    if !obrrt_core::mover::published(zone) {
        return None;
    }
    let floors = obrrt_core::geo::static_floors(zone)?;
    let walls = obrrt_core::geo::static_walls(zone)?;
    let r = Arc::new(find(&floors, &walls, super::params::get().riser_h_max));
    Some(c.write().unwrap().entry(zone).or_insert(r).clone())
}

/// A wall-pinned BLJ spot (the `Wall` kind): a short riser (<= [`PIN_H`],
/// invisible to the air wall probe from quarter step 1) that ends against
/// a tall static wall. Mario stands on the riser's seam, his back to the
/// wall, held 50 out of it by the wall pushes (radius 50 in the air,
/// mario_step.c:407-408; the upper ground probe too, :271): backward travel
/// is cancelled every quarter step, and the stick's sideways strain puts
/// the jump frame's quarter step onto the step top, so every jump lands in
/// its jump frame and the next ground frame drops him back to the foot
/// (the lab's -388 chain: run/examples/riser_lab.rs, ref/blj-conditions.md
/// section 6.6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Corner {
    pub riser: Riser,
    /// Unit horizontal normal of the tall wall (out of it, toward the riser).
    pub wall_n: [f32; 2],
    /// The pinned point: the riser's end, 50 out of the wall, on the seam.
    pub pin: [f32; 3],
    /// A wall pushes Mario back off the step top after a landing there
    /// ([`pushes_off`]): without it he stays up and the chain stops.
    pub push_off: bool,
}

/// Highest riser for the wall pin (the air probe rule, r <= 30 + 7.5).
pub const PIN_H: f32 = 37.5;

impl Corner {
    /// SM64 yaw of the facing for the chain (away from the wall).
    pub fn face_yaw(&self) -> i32 {
        (self.wall_n[0].atan2(self.wall_n[1]) * 32768.0 / std::f32::consts::PI) as i32
    }
    /// The point `x` out from the pin, along the wall's normal, `d` in
    /// front of the riser.
    pub fn out(&self, x: f32, d: f32) -> [f32; 3] {
        [self.pin[0] + self.wall_n[0] * x + self.riser.n[0] * d, self.pin[1], self.pin[2] + self.wall_n[1] * x + self.riser.n[1] * d]
    }
}

/// Every wall-pinned spot of a zone (see [`Corner`]), from its risers.
pub fn corners(walls: &[Tri], risers: &[Riser]) -> Vec<Corner> {
    let pd = super::params::get();
    let tall: Vec<([f32; 2], [f32; 2], [f32; 2], f32, f32)> = walls
        .iter()
        .filter_map(|t| {
            let (lo, hi) = yr(t);
            let nl = t.n[0].hypot(t.n[2]);
            (hi - lo > pd.corner_tall && nl > 0.5).then(|| {
                let (a, b) = ends(t);
                (a, b, [t.n[0] / nl, t.n[2] / nl], lo, hi)
            })
        })
        .collect();
    let mut out: Vec<Corner> = Vec::new();
    for r in risers.iter().filter(|r| r.h() <= pd.pin_h) {
        let l = r.len().max(1e-3);
        let dir = [(r.b[0] - r.a[0]) / l, (r.b[1] - r.a[1]) / l];
        for (e, inward) in [(r.a, dir), (r.b, [-dir[0], -dir[1]])] {
            for &(a, b, n, lo, hi) in &tall {
                // The wall must span the foot to the probes (60 above).
                if lo > r.ylo + 5.0 || hi < r.ylo + 60.0 {
                    continue;
                }
                // Across the riser's end, facing back along it.
                if n[0] * inward[0] + n[1] * inward[1] < 0.8 {
                    continue;
                }
                let (dx, dz) = (b[0] - a[0], b[1] - a[1]);
                let l2 = (dx * dx + dz * dz).max(1.0);
                let t = (((e[0] - a[0]) * dx + (e[1] - a[1]) * dz) / l2).clamp(0.0, 1.0);
                if (e[0] - a[0] - t * dx).hypot(e[1] - a[1] - t * dz) >= pd.corner_dist {
                    continue;
                }
                let pin = [e[0] + n[0] * 50.0, r.ylo, e[1] + n[1] * 50.0];
                if out.iter().any(|c| dist2(c.pin, pin) < 30.0 && (c.pin[1] - pin[1]).abs() < 5.0) {
                    continue;
                }
                let mut c = Corner { riser: *r, wall_n: n, pin, push_off: false };
                c.push_off = pushes_off(&c, walls);
                out.push(c);
            }
        }
    }
    out
}

/// Whether, after a jump lands on the step top just behind the riser
/// plane (d = -2) at the pinned point, a static wall's ground probe push
/// (y + 30 radius 24, y + 60 radius 50; mario_step.c:270-271, pushes to
/// the wall's front side, surface_collision.c) moves Mario back in front
/// of the plane (d > 0), so the next ground frame drops him to the foot.
/// On the lobby's diagonal steps that is the next riser up.
pub fn pushes_off(c: &Corner, walls: &[Tri]) -> bool {
    let r = &c.riser;
    let top = r.yhi;
    let p = [c.pin[0] - r.n[0] * 2.0, c.pin[2] - r.n[1] * 2.0];
    let d0 = -2.0f32;
    walls.iter().any(|t| {
        let nl = t.n[0].hypot(t.n[2]);
        if nl < 0.5 {
            return false;
        }
        let along = t.n[0] * r.n[0] + t.n[2] * r.n[1];
        if along <= 0.2 {
            return false;
        }
        let (lo, hi) = yr(t);
        let (a, b) = ends(t);
        let (dx, dz) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dz * dz;
        if l2 < 1.0 {
            return false;
        }
        let tt = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / l2;
        if !(0.0..=1.0).contains(&tt) {
            return false;
        }
        let oo = -(t.n[0] * t.v[0][0] + t.n[1] * t.v[0][1] + t.n[2] * t.v[0][2]);
        [(30.0f32, 24.0f32), (60.0, 50.0)].iter().any(|&(dy, rad)| {
            let y = top + dy;
            if y < lo || y > hi {
                return false;
            }
            let off = t.n[0] * p[0] + t.n[1] * y + t.n[2] * p[1] + oo;
            if off < -rad || off > rad {
                return false;
            }
            // Pushed out to `rad` along the wall's normal.
            d0 + along * (rad - off) / nl.max(1e-3) * nl > 0.0
        })
    })
}
