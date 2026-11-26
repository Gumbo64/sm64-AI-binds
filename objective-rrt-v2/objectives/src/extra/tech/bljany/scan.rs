//! The BLJ geometry scan (`OBRRT_BLJ_SCAN=1`, default off): BLJ spots in
//! the collision around a base state, not only under it.
//!
//! [`super::blj_ok`] tests b's own floor, ceiling and speed, so `BljAny`
//! fired only where a tree state already stood on a usable spot. The scan
//! looks for candidate spots within [`scan_r`] (`OBRRT_BLJ_SCAN_R`, 2500) of
//! b in the zone's static collision (floors, walls, ceilings; the snapshots
//! of [`obrrt_core::geo`], stored when a state of the zone is first read with
//! objects and used only once [`obrrt_core::mover::published`], i.e. from
//! the round after: the same zone always gives the same geometry, whatever
//! the thread timing) and among b's objects. One candidate list per zone,
//! built once and cached (bounded by the zones visited).
//!
//! Clauses (ref/blj-conditions.md 4.2 and 6):
//! - **Slope:** a static floor triangle of a NOT_SLIPPERY class with rise
//!   tan >= 1.25 (0.2924 < ny <= 0.6247), the existing slope clause per
//!   triangle; at least [`SLOPE_AREA`] of floor (projected); start at its
//!   centroid, facing down it.
//! - **Ceiling:** a standable floor point (sampled every [`CEIL_STEP`]) whose
//!   lowest static ceiling above leaves headroom `ceil - floor - 160` in
//!   0..50; any facing (the 4-way fan by b's frame, as blj_ok's).
//! - **Riser:** every riser of [`super::riser::find`] (15-60 high, >= 100
//!   long), stair mode (facing away, 60 in front) and, at h <= 30, both side
//!   modes (13 in front); the riser overlay.
//! - **Wall:** [`super::riser::corners`] with a push-off, 100 out of the pin;
//!   the wall overlay.
//! - **Elevator:** an object of b's within R whose behaviour is a known
//!   rising platform ([`super::PLATFORMS`]); the overlay waits until Mario
//!   stands on that object, then runs the BLJ rule from there.
//!
//! Per base state at most [`scan_n`] (`OBRRT_BLJ_SCAN_N`, 3) candidates,
//! nearest first, spread over clauses and 400-unit cells (first pass: a new
//! clause and a new cell; second: a new cell).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use obrrt_core::St;
use obrrt_core::geo::Tri;

use super::riser::{Corner, Riser};
use super::{Kind, RMode};

/// OBRRT_BLJ_SCAN=1.
pub fn on() -> bool {
    obrrt_core::config::current().objectives.blj_scan
}

/// The scan radius around b (3D; the detector's `scan_r`, 2500).
pub fn scan_r() -> f32 {
    super::params::get().scan_r
}

/// Candidates per base state (the detector's `scan_n`, 3).
pub fn scan_n() -> usize {
    super::params::get().scan_n as usize
}

/// Smallest slope triangle (projected area): room for the walk and the
/// first landings.
pub const SLOPE_AREA: f32 = 20000.0;
/// Ceiling sample spacing.
pub const CEIL_STEP: f32 = 100.0;

/// A candidate spot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cand {
    pub kind: Kind,
    /// Where the setup starts.
    pub start: [f32; 3],
    /// The BLJ facing; None: any (the fan).
    pub yaw: Option<i32>,
    pub riser: Option<(Riser, RMode)>,
    pub pin: Option<Corner>,
    /// The Elevator clause's object slot.
    pub slot: Option<u16>,
    /// The point the key's cell is taken from (the start; an object's home).
    pub anchor: [f32; 3],
    /// A spot on an object's collision (OBRRT_BLJ_SCAN_OBJECTS, [`super::objscan`]).
    pub obj: Option<super::objscan::ObjRef>,
}

impl Cand {
    fn plain(kind: Kind, start: [f32; 3], yaw: Option<i32>) -> Cand {
        Cand { kind, start, yaw, riser: None, pin: None, slot: None, anchor: start, obj: None }
    }
}

fn excluded(kind: i16) -> bool {
    matches!(kind, 0x01 | 0x0A | 0x12 | 0x22 | 0x23 | 0x24 | 0x2D | 0x38)
}

fn cell4(p: [f32; 3]) -> [i32; 2] {
    super::cell(p)
}

fn quadrant(yaw: i32) -> i32 {
    ((yaw + 0x2000).rem_euclid(0x10000)) / 0x4000
}

fn area_xz(t: &Tri) -> f32 {
    let [a, b, c] = t.v;
    (((b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])) / 2.0).abs()
}

fn centroid(t: &Tri) -> [f32; 3] {
    let [a, b, c] = t.v;
    [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0]
}

fn inside_xz(t: &Tri, x: f32, z: f32) -> bool {
    let [a, b, c] = t.v;
    let side = |p: [f32; 3], q: [f32; 3]| (q[0] - p[0]) * (z - p[2]) - (q[2] - p[2]) * (x - p[0]);
    let (d1, d2, d3) = (side(a, b), side(b, c), side(c, a));
    (d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0) || (d1 <= 0.0 && d2 <= 0.0 && d3 <= 0.0)
}

/// Whether (x, z) is over `t` (labs, the registry's dumps).
pub fn inside_xz_pub(t: &Tri, x: f32, z: f32) -> bool {
    inside_xz(t, x, z)
}

fn plane_y(t: &Tri, x: f32, z: f32) -> f32 {
    let a = t.v[0];
    a[1] - (t.n[0] * (x - a[0]) + t.n[2] * (z - a[2])) / t.n[1]
}

/// SM64 yaw down a floor's slope (its normal's horizontal part).
pub fn down_yaw(n: [f32; 3]) -> i32 {
    (n[0].atan2(n[2]) * 32768.0 / std::f32::consts::PI) as i32
}

/// The slope clause on one triangle (area aside).
pub fn slope_ok(t: &Tri) -> bool {
    let (not_slippery, _) = super::slides_below(t.kind);
    let pd = super::params::get();
    !excluded(t.kind) && not_slippery && t.n[1] > pd.slope_ny_min && t.n[1] <= pd.slope_ny_max
}

/// Steep non-slippery slope triangles, one per (cell, facing quadrant):
/// the largest.
pub fn slopes(floors: &[Tri]) -> Vec<Cand> {
    slopes_ix(floors, super::params::get().slope_area).into_iter().map(|v| v.0).collect()
}

/// [`slopes`] with a minimum area, and each candidate's triangle index.
pub fn slopes_ix(floors: &[Tri], min_area: f32) -> Vec<(Cand, usize)> {
    let mut best: BTreeMap<([i32; 2], i32), (f32, (Cand, usize))> = BTreeMap::new();
    for (i, t) in floors.iter().enumerate() {
        if !slope_ok(t) {
            continue;
        }
        let a = area_xz(t);
        if a < min_area {
            continue;
        }
        let yaw = down_yaw(t.n);
        // The centroid; with `slope_spacing` > 0 also grid points over the
        // triangle (OBRRT_BLJ_DETECTOR; 0: the centroid only).
        let mut pts = vec![centroid(t)];
        let sp = super::params::get().slope_spacing;
        if sp > 0.0 {
            let (x0, x1) = (t.v.iter().map(|p| p[0]).fold(f32::MAX, f32::min), t.v.iter().map(|p| p[0]).fold(f32::MIN, f32::max));
            let (z0, z1) = (t.v.iter().map(|p| p[2]).fold(f32::MAX, f32::min), t.v.iter().map(|p| p[2]).fold(f32::MIN, f32::max));
            let mut x = (x0 / sp).ceil() * sp;
            while x <= x1 {
                let mut z = (z0 / sp).ceil() * sp;
                while z <= z1 {
                    if inside_xz(t, x, z) {
                        pts.push([x, plane_y(t, x, z), z]);
                    }
                    z += sp;
                }
                x += sp;
            }
        }
        for c in pts {
            let k = (cell4(c), quadrant(yaw));
            if best.get(&k).is_none_or(|(b, _)| a > *b) {
                best.insert(k, (a, (Cand::plain(Kind::Slope, c, Some(yaw)), i)));
            }
        }
    }
    best.into_values().map(|v| v.1).collect()
}

/// Low-ceiling floor points (headroom 0..50 over 160), one per cell: the
/// smallest headroom (then the lowest x, z).
pub fn ceilings(floors: &[Tri], ceils: &[Tri]) -> Vec<Cand> {
    ceilings_ix(floors, ceils).into_iter().map(|v| v.0).collect()
}

/// The lowest ceiling of `ceils` over (x, z) at or above `y`.
pub fn ceil_over(ceils: &[Tri], x: f32, z: f32, y: f32) -> Option<f32> {
    ceils.iter().filter(|t| t.n[1] < -0.01 && inside_xz(t, x, z)).map(|t| plane_y(t, x, z)).filter(|&h| h >= y).min_by(f32::total_cmp)
}

/// The height of floor `t`'s plane at (x, z) if (x, z) is over it.
pub fn floor_y_at(t: &Tri, x: f32, z: f32) -> Option<f32> {
    (t.n[1] > 0.01 && inside_xz(t, x, z)).then(|| plane_y(t, x, z))
}

/// [`ceilings`] with each candidate's floor and ceiling indices.
pub fn ceilings_ix(floors: &[Tri], ceils: &[Tri]) -> Vec<(Cand, usize, usize)> {
    const G: f32 = 256.0;
    let gc = |x: f32, z: f32| ((x / G).floor() as i32, (z / G).floor() as i32);
    let mut grid: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
    for (i, t) in ceils.iter().enumerate() {
        if t.n[1] > -0.01 {
            continue;
        }
        let (x0, x1) = (t.v.iter().map(|p| p[0]).fold(f32::MAX, f32::min), t.v.iter().map(|p| p[0]).fold(f32::MIN, f32::max));
        let (z0, z1) = (t.v.iter().map(|p| p[2]).fold(f32::MAX, f32::min), t.v.iter().map(|p| p[2]).fold(f32::MIN, f32::max));
        let (a, b) = (gc(x0, z0), gc(x1, z1));
        for cx in a.0..=b.0 {
            for cz in a.1..=b.1 {
                grid.entry((cx, cz)).or_default().push(i as u32);
            }
        }
    }
    if grid.is_empty() {
        return Vec::new();
    }
    let ceil_over = |x: f32, z: f32, y: f32| -> Option<(f32, usize)> {
        let ids = grid.get(&gc(x, z))?;
        ids.iter()
            .map(|&i| (i as usize, &ceils[i as usize]))
            .filter(|(_, t)| inside_xz(t, x, z))
            .map(|(i, t)| (plane_y(t, x, z), i))
            .filter(|&(h, _)| h >= y)
            .min_by(|a, b| a.0.total_cmp(&b.0))
    };
    let mut best: BTreeMap<[i32; 2], (f32, (Cand, usize, usize))> = BTreeMap::new();
    for (fi, t) in floors.iter().enumerate() {
        if excluded(t.kind) || t.n[1] <= super::slides_below(t.kind).1.max(0.01) {
            continue;
        }
        let xs = t.v.map(|p| p[0]);
        let zs = t.v.map(|p| p[2]);
        let (x0, x1) = (xs.iter().copied().fold(f32::MAX, f32::min), xs.iter().copied().fold(f32::MIN, f32::max));
        let (z0, z1) = (zs.iter().copied().fold(f32::MAX, f32::min), zs.iter().copied().fold(f32::MIN, f32::max));
        let mut pts = vec![centroid(t)];
        let mut x = (x0 / CEIL_STEP).ceil() * CEIL_STEP;
        while x <= x1 {
            let mut z = (z0 / CEIL_STEP).ceil() * CEIL_STEP;
            while z <= z1 {
                if inside_xz(t, x, z) {
                    pts.push([x, 0.0, z]);
                }
                z += CEIL_STEP;
            }
            x += CEIL_STEP;
        }
        for p in pts {
            let fy = plane_y(t, p[0], p[2]);
            // Mario's query: find_ceil from floor + 80, ceilings >= that - 78.
            let Some((cy, ci)) = ceil_over(p[0], p[2], fy + 2.0) else { continue };
            let head = cy - fy - 160.0;
            let pd = super::params::get();
            if !(pd.ceil_head_lo..pd.ceil_head_hi).contains(&head) {
                continue;
            }
            let q = [p[0], fy, p[2]];
            let k = cell4(q);
            let better = |(h, (c, _, _)): &(f32, (Cand, usize, usize))| head < *h || (head == *h && (q[0], q[2]) < (c.start[0], c.start[2]));
            if best.get(&k).is_none_or(better) {
                best.insert(k, (head, (Cand::plain(Kind::Ceiling, q, None), fi, ci)));
            }
        }
    }
    best.into_values().map(|v| v.1).collect()
}

/// The riser clause's candidates (riser_generate's modes, both sides).
pub fn riser_cands(risers: &[Riser]) -> Vec<Cand> {
    let mut out = Vec::new();
    for r in risers {
        let away = r.away_yaw();
        let mut modes = vec![RMode::Stair];
        if r.h() <= super::SIDE_H {
            modes.extend([RMode::SideL, RMode::SideR]);
        }
        for mode in modes {
            let p = r.point(0.5, if mode == RMode::Stair { super::STAIR_D0 } else { super::SIDE_D0 });
            let yaw = match mode {
                RMode::Stair | RMode::Wall => away,
                RMode::SideL => away + 0x4000,
                RMode::SideR => away - 0x4000,
            };
            out.push(Cand { kind: Kind::Riser, start: p, yaw: Some(yaw), riser: Some((*r, mode)), pin: None, slot: None, anchor: p, obj: None });
        }
    }
    out
}

/// The wall clause's candidates (pinned corners with a push-off).
pub fn wall_cands(corners: &[Corner]) -> Vec<Cand> {
    corners
        .iter()
        .filter(|c| c.push_off)
        .map(|c| {
            let p = c.out(super::PIN_X, super::PIN_D0);
            Cand { kind: Kind::Wall, start: p, yaw: Some(c.face_yaw()), riser: Some((c.riser, RMode::Wall)), pin: Some(*c), slot: None, anchor: p, obj: None }
        })
        .collect()
}

/// Every static candidate of a zone's geometry.
pub fn find_static(floors: &[Tri], walls: &[Tri], ceils: &[Tri]) -> Vec<Cand> {
    let risers = super::riser::find(floors, walls, super::params::get().riser_h_max);
    let corners = super::riser::corners(walls, &risers);
    let mut v = slopes(floors);
    v.extend(ceilings(floors, ceils));
    v.extend(riser_cands(&risers));
    v.extend(wall_cands(&corners));
    v
}

/// Rising-platform objects of `b` within `r` of it.
pub fn elevators(b: &St, r: f32) -> Vec<Cand> {
    b.objects()
        .iter()
        .filter(|o| super::PLATFORMS.contains(&o.bhv_name()) && b.dist(o.pos) <= r)
        .map(|o| Cand { kind: Kind::Elevator, start: o.pos, yaw: None, riser: None, pin: None, slot: Some(o.slot), anchor: o.home, obj: None })
        .collect()
}

/// A zone's static candidates (cached; None until the zone is published).
pub fn zone_cands(zone: (u16, u8)) -> Option<Arc<Vec<Cand>>> {
    static C: OnceLock<RwLock<HashMap<(u16, u8), Arc<Vec<Cand>>>>> = OnceLock::new();
    let c = C.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(r) = c.read().unwrap().get(&zone) {
        return Some(r.clone());
    }
    if !obrrt_core::mover::published(zone) {
        return None;
    }
    let floors = obrrt_core::geo::static_floors(zone)?;
    let walls = obrrt_core::geo::static_walls(zone)?;
    let ceils = obrrt_core::geo::static_ceils(zone)?;
    let v = Arc::new(find_static(&floors, &walls, &ceils));
    let mut w = c.write().unwrap();
    if let Some(r) = w.get(&zone) {
        return Some(r.clone());
    }
    if super::spots::on() {
        for c in v.iter() {
            super::spots::note(zone, "static", &format!("{:?}", c.kind), c.start, c.yaw.unwrap_or(0), c.start, "static", 0, None, false);
        }
    }
    let mut d = diag().lock().unwrap();
    let z = d.zones.entry(zone).or_default();
    for cand in v.iter() {
        *z.entry(format!("{:?}", cand.kind)).or_default() += 1;
    }
    w.insert(zone, v.clone());
    Some(v)
}

/// Picks up to `n` of `cands` (with distances): nearest first, a new
/// clause and a new 400-cell, then a new cell.
pub fn select(mut cands: Vec<(f32, Cand)>, n: usize) -> Vec<Cand> {
    cands.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(a.1.start[0].total_cmp(&b.1.start[0]))
            .then(a.1.start[2].total_cmp(&b.1.start[2]))
            .then((a.1.kind as u8).cmp(&(b.1.kind as u8)))
    });
    let mut picked: Vec<usize> = Vec::new();
    let mut cells: BTreeSet<[i32; 2]> = BTreeSet::new();
    let mut kinds: BTreeSet<u8> = BTreeSet::new();
    for pass in 0..2 {
        for (i, (_, c)) in cands.iter().enumerate() {
            if picked.len() >= n {
                break;
            }
            let cell = cell4(c.anchor);
            if picked.contains(&i) || cells.contains(&cell) || (pass == 0 && kinds.contains(&(c.kind as u8))) {
                continue;
            }
            picked.push(i);
            cells.insert(cell);
            kinds.insert(c.kind as u8);
        }
    }
    picked.into_iter().map(|i| cands[i].1).collect()
}

/// The candidates for base `b` (static ones of its zone and its
/// elevators), selected.
pub fn near(b: &St, all: &[Cand]) -> Vec<Cand> {
    let r = scan_r();
    let mut v: Vec<(f32, Cand)> = all.iter().map(|c| (b.dist(c.start), *c)).filter(|(d, _)| *d <= r).collect();
    v.extend(elevators(b, r).into_iter().map(|c| (b.dist(c.start), c)));
    if obrrt_core::dyngeo::on() {
        v.extend(super::objscan::cands(b).into_iter().map(|c| (b.dist(c.start), c)).filter(|(d, _)| *d <= r));
    }
    // OBRRT_BLJ_SPOTS: observed and catalogue spots on static geometry.
    for (_, clause, p, yaw) in super::spots::extra_cands(b, r) {
        let c = match clause.as_str() {
            "Slope" => Cand::plain(Kind::Slope, p, Some(yaw)),
            "Ceiling" => Cand::plain(Kind::Ceiling, p, None),
            _ => Cand::plain(Kind::Observed, p, Some(yaw)),
        };
        v.push((b.dist(p), c));
    }
    select(v, scan_n())
}

// ---- Diagnostics (summary.json `blj_scan`) ----

#[derive(Default)]
pub struct Diag {
    /// Per zone, candidates found per clause.
    pub zones: BTreeMap<(u16, u8), BTreeMap<String, u64>>,
    /// Objectives emitted per clause (every generation call), and the
    /// distinct keys per clause.
    pub emitted: BTreeMap<String, u64>,
    pub keys: BTreeMap<String, BTreeSet<String>>,
    /// Elevator objects seen (zone, behaviour, home).
    pub elevators: BTreeSet<((u16, u8), String, [i32; 3])>,
    /// OBRRT_BLJ_SCAN_OBJECTS: per zone, the collision objects scanned (by
    /// id), and the distinct object candidates (object id, clause, local
    /// cell) with whether the object was moving.
    pub obj_scanned: BTreeMap<(u16, u8), BTreeSet<String>>,
    pub obj_cands: BTreeMap<(u16, u8), BTreeMap<(String, String, [i32; 3]), bool>>,
}

pub fn diag() -> &'static Mutex<Diag> {
    static D: OnceLock<Mutex<Diag>> = OnceLock::new();
    D.get_or_init(|| Mutex::new(Diag::default()))
}

/// Notes one emitted objective.
pub fn note_emit(kind: Kind, key: &str, zone: (u16, u8), elevator: Option<(&str, [f32; 3])>) {
    let mut d = diag().lock().unwrap();
    let k = format!("{kind:?}");
    *d.emitted.entry(k.clone()).or_default() += 1;
    if !d.keys.get(&k).is_some_and(|s| s.contains(key)) {
        d.keys.entry(k).or_default().insert(key.to_string());
    }
    if let Some((bhv, home)) = elevator {
        d.elevators.insert((zone, bhv.to_string(), home.map(|v| v.round() as i32)));
    }
}

/// summary.json's `blj_scan` block (None when the scan is off).
pub fn summary_block() -> Option<(String, String)> {
    if !on() {
        return None;
    }
    let d = diag().lock().unwrap();
    let obj = |m: &BTreeMap<String, u64>| format!("{{{}}}", m.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect::<Vec<_>>().join(", "));
    let zones = d.zones.iter().map(|((l, a), m)| format!("\"{l}:{a}\": {}", obj(m))).collect::<Vec<_>>().join(", ");
    let keys: BTreeMap<String, u64> = d.keys.iter().map(|(k, s)| (k.clone(), s.len() as u64)).collect();
    let mut el: BTreeMap<String, u64> = BTreeMap::new();
    for (z, _, _) in &d.elevators {
        *el.entry(format!("{}:{}", z.0, z.1)).or_default() += 1;
    }
    let objects = d
        .obj_scanned
        .iter()
        .map(|(z, set)| {
            let c = d.obj_cands.get(z);
            let mut by: BTreeMap<String, u64> = BTreeMap::new();
            let mut moving = 0u64;
            for ((_, k, _), m) in c.into_iter().flatten() {
                *by.entry(k.clone()).or_default() += 1;
                moving += *m as u64;
            }
            let n = c.map_or(0, |c| c.len());
            format!("\"{}:{}\": {{\"objects_scanned\": {}, \"candidates\": {}, \"by_clause\": {}, \"on_moving\": {moving}}}", z.0, z.1, set.len(), n, obj(&by))
        })
        .collect::<Vec<_>>()
        .join(", ");
    let text = format!(
        "{{\"r\": {}, \"n\": {}, \"candidates\": {{{zones}}}, \"elevator_objects\": {}, \"emitted\": {}, \"distinct_keys\": {}, \"objects\": {{{objects}}}}}",
        scan_r(),
        scan_n(),
        obj(&el),
        obj(&d.emitted),
        obj(&keys)
    );
    Some(("blj_scan".to_string(), text))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri(v: [[f32; 3]; 3], kind: i16) -> Tri {
        let (a, b, c) = (v[0], v[1], v[2]);
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let mut n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        n = n.map(|x| x / l);
        Tri { v, n, kind, room: 0 }
    }
    /// An upward-facing floor (normal y > 0) whatever the winding.
    fn floor(v: [[f32; 3]; 3], kind: i16) -> Tri {
        let t = tri(v, kind);
        if t.n[1] < 0.0 { tri([v[0], v[2], v[1]], kind) } else { t }
    }
    fn ceil(v: [[f32; 3]; 3]) -> Tri {
        let t = tri(v, 0);
        if t.n[1] > 0.0 { tri([v[0], v[2], v[1]], 0) } else { t }
    }

    #[test]
    fn steep_not_slippery_slope_found_facing_down() {
        // Rises 600 over 400 along +z (tan 1.5): down is -z (yaw 0x8000).
        let t = floor([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 600.0, 400.0]], 0x15);
        let v = slopes(&[t]);
        assert_eq!(v.len(), 1);
        let c = v[0];
        assert_eq!(c.kind, Kind::Slope);
        let y = c.yaw.unwrap().rem_euclid(0x10000);
        assert!((y - 0x8000).abs() < 0x100, "{y:x}");
        // Slippery (default class) or shallow (tan 1.0): none.
        assert!(slopes(&[floor([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 600.0, 400.0]], 0x00)]).is_empty());
        assert!(slopes(&[floor([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 400.0, 400.0]], 0x15)]).is_empty());
        // Too small.
        assert!(slopes(&[floor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 150.0, 100.0]], 0x15)]).is_empty());
        // Too steep (ny <= 0.2924, a wall-like floor).
        assert!(slopes(&[floor([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [0.0, 2000.0, 400.0]], 0x15)]).is_empty());
    }

    #[test]
    fn low_ceiling_found_only_under_the_low_part() {
        // Flat floor 0..800 x 0..800 at y 0; a ceiling at 190 over x < 400
        // (headroom 30) and one at 400 over x > 400.
        let f = [floor([[0.0, 0.0, 0.0], [800.0, 0.0, 0.0], [800.0, 0.0, 800.0]], 0), floor([[0.0, 0.0, 0.0], [800.0, 0.0, 800.0], [0.0, 0.0, 800.0]], 0)];
        let c = [
            ceil([[0.0, 190.0, 0.0], [400.0, 190.0, 0.0], [400.0, 190.0, 800.0]]),
            ceil([[0.0, 190.0, 0.0], [400.0, 190.0, 800.0], [0.0, 190.0, 800.0]]),
            ceil([[400.0, 400.0, 0.0], [800.0, 400.0, 0.0], [800.0, 400.0, 800.0]]),
            ceil([[400.0, 400.0, 0.0], [800.0, 400.0, 800.0], [400.0, 400.0, 800.0]]),
        ];
        let v = ceilings(&f, &c);
        assert!(!v.is_empty());
        assert!(v.iter().all(|c| c.kind == Kind::Ceiling && c.start[0] <= 400.0 && c.start[1] == 0.0), "{v:?}");
        // One per 400-cell.
        let cells: BTreeSet<[i32; 2]> = v.iter().map(|c| cell4(c.start)).collect();
        assert_eq!(cells.len(), v.len());
        // Headroom 50+ (ceiling 210): none.
        let hi = [ceil([[0.0, 210.0, 0.0], [800.0, 210.0, 0.0], [800.0, 210.0, 800.0]]), ceil([[0.0, 210.0, 0.0], [800.0, 210.0, 800.0], [0.0, 210.0, 800.0]])];
        assert!(ceilings(&f, &hi).is_empty());
    }

    /// A step: lower floor z < 0 at y 0, a riser at z = 0 (h 26, x 0..400),
    /// upper floor z > 0 at y 26, and a tall wall at x = 0.
    fn step() -> (Vec<Tri>, Vec<Tri>) {
        let fl = vec![
            floor([[-50.0, 0.0, -300.0], [450.0, 0.0, -300.0], [450.0, 0.0, 0.0]], 0x15),
            floor([[-50.0, 0.0, -300.0], [450.0, 0.0, 0.0], [-50.0, 0.0, 0.0]], 0x15),
            floor([[-50.0, 26.0, 0.0], [450.0, 26.0, 0.0], [450.0, 26.0, 300.0]], 0x15),
            floor([[-50.0, 26.0, 0.0], [450.0, 26.0, 300.0], [-50.0, 26.0, 300.0]], 0x15),
        ];
        // Riser facing -z (front = lower floor).
        let r = Tri { v: [[0.0, 0.0, 0.0], [400.0, 0.0, 0.0], [400.0, 26.0, 0.0]], n: [0.0, 0.0, -1.0], kind: 0, room: 0 };
        // Tall wall at x = 0 facing +x (toward the riser).
        let w = Tri { v: [[0.0, -50.0, -300.0], [0.0, -50.0, 300.0], [0.0, 400.0, 0.0]], n: [1.0, 0.0, 0.0], kind: 0, room: 0 };
        (fl, vec![r, w])
    }

    #[test]
    fn riser_and_corner_candidates() {
        let (fl, walls) = step();
        let v = find_static(&fl, &walls, &[]);
        let risers: Vec<&Cand> = v.iter().filter(|c| c.kind == Kind::Riser).collect();
        // Stair + both sides (h 26 <= 30).
        assert_eq!(risers.len(), 3, "{v:?}");
        let stair = risers.iter().find(|c| c.riser.unwrap().1 == RMode::Stair).unwrap();
        // 60 in front (z -60), facing away (-z: yaw 0x8000).
        assert!((stair.start[2] + 60.0).abs() < 1.0 && (stair.start[0] - 200.0).abs() < 1.0, "{:?}", stair.start);
        assert!((stair.yaw.unwrap().rem_euclid(0x10000) - 0x8000).abs() < 0x100);
        // The corner (riser end at the tall wall).
        let corners = super::super::riser::corners(&walls, &super::super::riser::find(&fl, &walls, 60.0));
        assert!(!corners.is_empty());
        for c in v.iter().filter(|c| c.kind == Kind::Wall) {
            assert!(c.pin.unwrap().push_off);
        }
    }

    #[test]
    fn selection_spreads_and_is_bounded() {
        let mk = |k: Kind, x: f32| Cand::plain(k, [x, 0.0, 0.0], Some(0));
        let cands = vec![
            (10.0, mk(Kind::Slope, 0.0)),
            (20.0, mk(Kind::Slope, 1000.0)),
            (30.0, mk(Kind::Ceiling, 0.0)), // same cell as the first
            (40.0, mk(Kind::Ceiling, 2000.0)),
            (50.0, mk(Kind::Riser, 3000.0)),
        ];
        let s = select(cands.clone(), 3);
        assert_eq!(s.len(), 3);
        // Pass 1: Slope@0, Ceiling@2000, Riser@3000 (one per clause and cell).
        assert_eq!(s.iter().map(|c| c.start[0] as i32).collect::<Vec<_>>(), vec![0, 2000, 3000]);
        // With room for more: the second slope (a new cell), not the
        // ceiling in the first one's cell.
        let s = select(cands, 5);
        assert_eq!(s.len(), 4);
        assert!(s.iter().any(|c| c.start[0] == 1000.0));
    }
}
