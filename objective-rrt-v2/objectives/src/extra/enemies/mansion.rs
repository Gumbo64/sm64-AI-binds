//! Walking time through Big Boo's Haunt (BBH), for the objectives that go
//! to its Boos, its Mr. I and its merry-go-round.
//!
//! The straight line misleads in the mansion: the rooms are walled, and the
//! basement is 2300 units under the courtyard. [`walk_frames`] adds up:
//!
//! - `OBRRT_GEO=1`: the walk over the static floors (core/src/geo.rs), which
//!   for BBH keeps out of the walls between rooms (rooms 1-13 of
//!   `bbh_seg7_rooms` meet only through doorway floors, rooms 14+; the 27
//!   `special_haunted_door`s of areas/1/collision.inc.c all open by walking
//!   into them, so a door is a few frames of the walk, not a gate);
//! - the mesh elevator (bhvMeshElevator, levels/bbh/script.c at
//!   (-2985, -205, 5400); elevator.inc.c: `sElevatorHeights[4]` = -2360 to
//!   its home height, speed up to 10 a frame, so ~220 frames) between the
//!   courtyard and the basement, which a floor graph can't know: it is a
//!   moving object, and the basement's doors and the merry-go-round are
//!   below it.
//!
//! - the secret room of the big Mr. I (the "Eye to Eye" star): see [`SECRET_GATE_WEST`].
//!
//! Where the mesh doesn't know a point the straight distance stands in (as
//! `St::path_dist` does).

use obrrt_core::layout::LEVEL_BBH;
use obrrt_core::{St, frames_for};

/// The mesh elevator's top (its home, levels/bbh/script.c) and bottom
/// (`sElevatorHeights[4]`, elevator.inc.c) standing points.
const ELEVATOR_TOP: [f32; 3] = [-2985.0, -205.0, 5400.0];
const ELEVATOR_BOTTOM: [f32; 3] = [-2985.0, -2360.0, 5400.0];
/// Its ride: from rest to 10 a frame (2 a frame²), 2155 down.
const ELEVATOR_FRAMES: f64 = 2155.0 / 10.0 + 5.0;
/// Heights under which a point is in the basement (its floors are at
/// -2457 to -2764; the courtyard's and the mansion's lie above -420).
const BASEMENT_Y: f32 = -1500.0;

/// Whether `s` is in BBH.
pub fn in_bbh(s: &St) -> bool {
    s.level as u32 == LEVEL_BBH
}

pub(super) fn leg(s: &St, a: [f32; 3], b: [f32; 3]) -> f32 {
    obrrt_core::geo::path_dist(s.zone(), a, b).unwrap_or_else(|| obrrt_core::dist(a, b))
}

/// Frames to walk from Mario to `to`, through the elevator when the basement
/// is between them (BBH only; elsewhere the straight distance).
pub fn walk_frames(s: &St, to: [f32; 3]) -> f64 {
    if !in_bbh(s) {
        return frames_for(s.dist(to));
    }
    let (here_below, there_below) = (s.pos[1] < BASEMENT_Y, to[1] < BASEMENT_Y);
    match (here_below, there_below) {
        (false, true) => frames_for(leg(s, s.pos, ELEVATOR_TOP)) + ELEVATOR_FRAMES + frames_for(leg(s, ELEVATOR_BOTTOM, to)),
        (true, false) => frames_for(leg(s, s.pos, ELEVATOR_BOTTOM)) + ELEVATOR_FRAMES + frames_for(leg(s, ELEVATOR_TOP, to)),
        _ => frames_for(leg(s, s.pos, to)),
    }
}

/// A point about `ahead` units along the walk to `to`, for steering (the
/// elevator's end of the way when the basement is between).
pub fn waypoint(s: &St, to: [f32; 3], ahead: f32) -> [f32; 3] {
    let via = if !in_bbh(s) {
        to
    } else {
        match (s.pos[1] < BASEMENT_Y, to[1] < BASEMENT_Y) {
            (false, true) => ELEVATOR_TOP,
            (true, false) => ELEVATOR_BOTTOM,
            _ => to,
        }
    };
    obrrt_core::geo::waypoint(s.zone(), s.pos, via, ahead).unwrap_or(via)
}

/// Big Boo's Haunt's secret room (third floor, "Eye to Eye in the Secret Room"): the big Mr. I's chamber, x 205 to
/// 2048, z -805 to 116 over the 1741 floor, is walled all round (levels/bbh/areas/1/collision.inc.c: type-0 walls
/// at z = 116 / 218, x = 205 / 2048 and z = -805), except a gate in its west wall: two `SURFACE_VANISH_CAP_WALLS`
/// (0x7B) walls at x = 102 and x = 205, z -497 to 116, y 1741 to 2355. Mario walks through them only with the
/// vanish cap (find_wall_collisions skips them for him then), which a vanish `!` box gives for 600 frames. The
/// static walk graph knows nothing of it: the chamber is unreachable by the floors alone.
pub const SECRET_GATE_WEST: [f32; 3] = [20.0, 1741.0, -190.0];
/// Just inside the gate, in the chamber.
pub const SECRET_GATE_EAST: [f32; 3] = [330.0, 1741.0, -190.0];
/// `MARIO_VANISH_CAP` (include/sm64.h).
pub const MARIO_VANISH_CAP: u32 = 0x02;
/// The level's vanish-cap `!` boxes (2nd behaviour byte 2, macro.inc.c), for when the objects aren't loaded: the
/// first floor's west wing and the second floor's east wing. (A third, at (460, 2140, -560), is in the chamber.)
const VANISH_BOXES: [[f32; 3]; 2] = [[-1960.0, 300.0, -120.0], [1268.0, 1050.0, 1860.0]];
/// Frames to break a box (a punch or a ground pound) and pick the cap up.
pub const BOX_FRAMES: f64 = 40.0;

/// The way to the third floor, the gate's floor, is a wall-kick column at the far east of the second floor's wing
/// (the walk graph has no way up 900 units; the trees' routes to the third floor, nodes of agent mri's tests and the
/// main run's balcony star, all climb it): from the platform at its foot ([`COLUMN_BASE`], reached by walking from the
/// second-floor box at (1268, 1050, 1860): 3469 units) four to five wall kicks up the wall at x ~ 3300 to the third
/// floor's landing ([`THIRD_FLOOR_LANDING`]); from there the floors walk to the gate (2945 units).
pub const COLUMN_BASE: [f32; 3] = [3330.0, 1024.0, 520.0];
/// The third floor where the column's wall kicks end (x 2463, z 268).
pub const THIRD_FLOOR_LANDING: [f32; 3] = [2463.0, 1741.0, 268.0];
/// Frames of the wall-kick chain from [`COLUMN_BASE`] to [`THIRD_FLOOR_LANDING`] (the tree's own: 4 kicks of ~30).
pub const COLUMN_FRAMES: f64 = 150.0;
/// Heights from which Mario is on the third floor (its floors are at 1741 and over).
const THIRD_FLOOR_Y: f32 = 1700.0;

/// Whether `p` is in the wall-kick column's air or past its base (second-floor east wing, above its platform).
pub fn in_column(p: [f32; 3]) -> bool {
    p[0] > 2300.0 && p[1] > 1000.0 && p[1] < THIRD_FLOOR_Y
}

/// Walking distance from `a` to `b`, by the ordinary graph or, where that can't reach (the ledge climbs it lacks:
/// first floor to second), the climb mesh ([`obrrt_core::geo::climb_zone`], ledges up to 520).
fn reach_leg(s: &St, a: [f32; 3], b: [f32; 3]) -> f32 {
    let d = leg(s, a, b);
    // `geo::UNREACHABLE` is added to the straight distance where the graph has no walk.
    if d - obrrt_core::dist(a, b) < 19000.0 {
        return d;
    }
    obrrt_core::geo::path_dist(obrrt_core::geo::climb_zone(s.zone()), a, b).unwrap_or(d)
}

/// Frames from Mario to `p` (a vanish box or cap), up ledges the ordinary graph lacks; the elevator for the basement.
pub fn reach_frames(s: &St, p: [f32; 3]) -> f64 {
    if s.pos[1] < BASEMENT_Y {
        return walk_frames(s, p);
    }
    frames_for(reach_leg(s, s.pos, p))
}

/// Frames from `p` to the secret room's west gate by the way above: third-floor floors; else to the column's base,
/// up it, and across the third floor.
pub fn gate_frames(s: &St, p: [f32; 3]) -> f64 {
    let across = frames_for(leg(s, THIRD_FLOOR_LANDING, SECRET_GATE_WEST));
    if p[1] >= THIRD_FLOOR_Y {
        return frames_for(leg(s, p, SECRET_GATE_WEST));
    }
    if in_column(p) {
        let left = ((THIRD_FLOOR_Y - p[1].max(COLUMN_BASE[1])) / (THIRD_FLOOR_Y - COLUMN_BASE[1])).clamp(0.0, 1.0) as f64;
        return COLUMN_FRAMES * left + across;
    }
    frames_for(reach_leg(s, p, COLUMN_BASE)) + COLUMN_FRAMES + across
}

/// [`waypoint`] where the graph walks there, else along the climb mesh ([`reach_leg`]).
pub fn reach_waypoint(s: &St, to: [f32; 3], ahead: f32) -> [f32; 3] {
    if leg(s, s.pos, to) - obrrt_core::dist(s.pos, to) < 19000.0 {
        return waypoint(s, to, ahead);
    }
    obrrt_core::geo::waypoint(obrrt_core::geo::climb_zone(s.zone()), s.pos, to, ahead).unwrap_or(to)
}

/// Whether `p` is inside the secret room (past its west gate).
pub fn in_secret_room(p: [f32; 3]) -> bool {
    p[0] > 150.0 && p[0] < 2048.0 && p[2] > -805.0 && p[2] < 116.0 && p[1] > 1700.0
}

/// Where Mario can get the vanish cap: the cap itself if it is out, else a vanish box standing outside the
/// secret room (those loaded, else the level's known ones), each with its `bhvVanishCap`-or-box position.
pub fn vanish_sources(s: &St) -> Vec<[f32; 3]> {
    let caps: Vec<[f32; 3]> = s.objects().iter().filter(|o| o.bhv_name() == "bhvVanishCap").map(|o| o.pos).collect();
    if !caps.is_empty() {
        return caps;
    }
    let boxes: Vec<[f32; 3]> = s
        .objects()
        .iter()
        .filter(|o| o.bhv_name() == "bhvExclamationBox" && (o.beh_params >> 16) & 0xFF == 2 && o.action <= 2)
        .map(|o| o.pos)
        .filter(|p| !in_secret_room(*p))
        .collect();
    if boxes.is_empty() { VANISH_BOXES.to_vec() } else { boxes }
}
