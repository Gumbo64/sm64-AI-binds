//! `Painting::Enter(course)`: jump into a castle painting. Paintings aren't
//! objects, so their positions come from the level data (painting_table.rs).

use obrrt_core::layout::LEVEL_CASTLE;
use obrrt_core::{Budget, Objective, Progress, St, dist};

pub struct Painting {
    pub name: &'static str,
    /// Castle area it hangs in: 1 first floor, 2 upstairs, 3 basement.
    pub area: u8,
    /// Level it warps to.
    pub dest: u32,
    pub pitch: f32,
    pub yaw: f32,
    /// Bottom corner, as in the level data.
    pub corner: [f32; 3],
    pub size: f32,
}

impl Painting {
    /// The painting's centre: its corner plus half its size along its width
    /// and height.
    pub fn centre(&self) -> [f32; 3] {
        let yaw = self.yaw.to_radians();
        let half = self.size / 2.0;
        [self.corner[0] + half * yaw.cos(), self.corner[1] + half, self.corner[2] - half * yaw.sin()]
    }
}

/// Stars needed to reach a painting's room: the star doors in front of them
/// (levels/castle_inside/script.c door behParams; interaction.c
/// interact_door: 1 for WF's door, 3 for JRB's and CCM's, 30 for DDD's in the
/// basement, 50 for the third floor).
fn stars_needed(name: &str) -> i16 {
    match name {
        "wf" => 1,
        "jrb" | "ccm" => 3,
        "ddd" => 30,
        "ttc" | "rr" => 50,
        _ => 0,
    }
}

/// OBRRT_PAINT_FRONT=1: Painting::Enter starts only in front of the
/// painting (a painting is entered head-on from the room it faces), and a
/// painting behind a star door with a known [`room`] starts only in its room
/// or at its door: the door is `StarDoor::Enter`'s job (one attempt can't
/// do door, cutscene and room within the horizon: 0 / ~100 such attempts
/// succeeded in the v46 trees). A start deep in the room needs no stars (a
/// 0-star BLJ / HyperWalk through the door). Without the switch the
/// straight-line heuristic ranked states behind the painting's wall or in
/// the next corridor (x ~ -766 by the WF room) first.
pub fn paint_front() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_PAINT_FRONT").is_ok_and(|v| v == "1"))
}

/// The room in front of a painting behind a star door, from its centre
/// along its facing and across it: (depth of the room plus the door's
/// approach, depth of the room past the door, half width, top y). WF: the 1-star
/// door at (235, 0, -960) (Mario unlocks it at z -905..-960), 3545 in
/// front of the painting; the room runs x ~ -500..1000 to the painting,
/// beside a corridor at x ~ -766..-886 (1022+ across) that ends at its wall,
/// with its floor at y -306..0, under the stair landing (y 512) and the
/// hallway past the PSS 1-star door (y 614) above it.
fn room(name: &str) -> Option<(f32, f32, f32, f32)> {
    match name {
        "wf" => Some((3900.0, 3300.0, 900.0, 450.0)),
        _ => None,
    }
}

/// (Distance in front of the painting's plane, distance across it) of `p`.
fn facing(painting: &Painting, centre: [f32; 3], p: [f32; 3]) -> (f32, f32) {
    let yaw = painting.yaw.to_radians();
    let (nx, nz) = (yaw.sin(), yaw.cos());
    let (rx, rz) = (p[0] - centre[0], p[2] - centre[2]);
    (rx * nx + rz * nz, (rx * nz - rz * nx).abs())
}

pub struct Enter {
    painting: &'static Painting,
    centre: [f32; 3],
}

impl Enter {
    pub fn new(_b: &St, p: &'static Painting) -> Enter {
        Enter { painting: p, centre: p.centre() }
    }

    fn in_room(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_CASTLE && s.area == self.painting.area
    }
}

impl Objective for Enter {
    fn kind(&self) -> &'static str {
        "Painting::Enter"
    }
    fn key(&self) -> String {
        format!("Painting {}", self.painting.name)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        15.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        // A course whose stars are all collected isn't worth entering, and a
        // painting behind a star door can't be reached without the stars.
        if !(self.in_room(s) && !s.course_done(obrrt_core::course_of_level(self.painting.dest))) {
            return false;
        }
        if !paint_front() {
            return s.stars >= stars_needed(self.painting.name);
        }
        // In front: ahead of the wall, within a cone widening from the
        // painting's own width.
        let (front, across) = facing(self.painting, self.centre, s.pos);
        if front < -50.0 || across > self.painting.size + 1.5 * front {
            return false;
        }
        match room(self.painting.name) {
            Some((approach, depth, half, top)) => {
                across < half
                    && front < approach
                    && s.pos[1] < top
                    && (front < depth || s.stars >= stars_needed(self.painting.name))
            }
            None => s.stars >= stars_needed(self.painting.name),
        }
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        s.dist(self.centre) as f64 / 400.0
    }
    fn arrival_scale(&self) -> Option<f64> {
        Some(400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.level as u32 == self.painting.dest)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, d) = traj
            .iter()
            .enumerate()
            .filter(|(_, s)| self.in_room(s))
            .map(|(i, s)| (i, dist(s.pos, self.centre)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::MAX));
        Progress { score: -(d as f64), at }
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 20, generations: 6, horizon: 300 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.centre)
    }
    fn kinship(&self) -> u32 {
        10
    }
}
