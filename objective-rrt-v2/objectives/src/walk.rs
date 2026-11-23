//! Walking distance for Shifting Sand Land and Snowman's Land.
//!
//! Both courses are climbs whose way up is not the straight line: SSL's
//! pyramid interior is a quicksand lake (deep quicksand, surface 0x22, is
//! left out of the mesh) around a stairway of ledges, and its Eyerok door and
//! the five secrets sit on that stairway; SL's mountain is one spiral up to
//! the snowman's head (levels/sl/areas/1/collision.inc.c: the floor above
//! y 4500 is a plateau reached only by the spiral). A straight-line ETA rates
//! the foot of the pyramid or of the mountain "close" to a star 1300-5000
//! above it (t1's SSL pyramid trees: 5 states above y 1000 in 123 M frames;
//! RUNS.md "SSL pyramid"). With `OBRRT_GEO=1` (`obrrt_core::geo`, the Koopa
//! race recipe, npcs.rs `Race`) these helpers rate by the walkable distance
//! over the area's floors; without it, or in other courses, they are exactly
//! the straight distance and the plain target, so nothing else changes.

use obrrt_core::layout::{LEVEL_SL, LEVEL_SSL};
use obrrt_core::{St, geo};

/// Whether the course is one whose ETAs use the walkable distance.
pub fn applies(s: &St) -> bool {
    matches!(s.level as u32, LEVEL_SSL | LEVEL_SL)
}

/// Distance from Mario in `s` to `p`: along the floors in SSL and SL when
/// `OBRRT_GEO=1` and both points are over the mesh, else straight.
pub fn dist(s: &St, p: [f32; 3]) -> f32 {
    if !applies(s) {
        return s.dist(p);
    }
    let d = s.path_dist(p);
    // SL's head: the brim under the star is 512 above the ledge the walk ends on (extra/slhead.rs), more than a
    // jump of the mesh (`geo::JUMP`), so every walk to it is "unreachable" (straight + 20000, no gradient). Walk to
    // the ledge instead and add the climb as a straight line.
    if s.level as u32 == LEVEL_SL && p[1] > 4300.0 && (p[0] - SL_BRIM[0]).hypot(p[2] - SL_BRIM[2]) < 1100.0 {
        let via = s.path_dist(SL_LEDGE) + dist_3d(SL_LEDGE, p);
        return d.min(via);
    }
    d
}

/// SL: the ledge at the head's foot (3840, collision: HARD (-378, 3840, 898)...) and the brim over it.
const SL_LEDGE: [f32; 3] = [-200.0, 3840.0, 1000.0];
const SL_BRIM: [f32; 3] = [700.0, 4352.0, 690.0];

fn dist_3d(a: [f32; 3], b: [f32; 3]) -> f32 {
    obrrt_core::dist(a, b)
}

/// What to steer at to get to `p`: a point 700 units ahead along the walk
/// (`OBRRT_GEO_AHEAD` in the engine), else `p` itself.
pub fn steer(s: &St, p: [f32; 3]) -> [f32; 3] {
    if applies(s) { geo::waypoint(s.zone(), s.pos, p, 700.0).unwrap_or(p) } else { p }
}
