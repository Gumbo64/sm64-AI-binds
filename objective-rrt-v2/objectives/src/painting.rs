//! `Painting::Enter(course)`: jump into a castle painting. Paintings aren't
//! objects, so their positions come from the level data (painting_table.rs).

use obrrt_core::layout::LEVEL_CASTLE;
use obrrt_core::{Eta, Mem, Objective, St, dist, frames_for};

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

/// OBRRT_PAINTING_FAST=1 (off by default): faster painting (re-)entries.
/// - Staged castle navigation: a painting behind a door ([`gate`]) is
///   steered to in two stages, the door first (as `Door::Enter` would:
///   Arrival::Door), then the painting; progress counts the stage, then the
///   straight-line distance to the stage's waypoint. Without it the attempt
///   steered straight at a painting behind walls (WF behind the 1-star door
///   and down the stairs) and 300-frame attempts ended at walls; the routes'
///   lobby -> other painting legs took 609-1579 f (138-210 straight).
///   Horizon 300 -> 420 (door + room in one attempt; the doors are also a
///   phase of the ETA).
/// - One objective per progress level: the key carries the collected set
///   (St::progress_set) and only states with that set may start it. A new
///   star makes a new key, so the new-eligible queue (OBRRT_NEW_ELIGIBLE,
///   at most 2 queueings per key per run) tries the painting from the first
///   post-exit state every time, not just after the first two exits, and
///   failures at earlier levels don't decay it.
/// - Mario::GoToCastle loses its post-star boost (mario/exit.rs).
pub fn painting_fast() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("OBRRT_PAINTING_FAST").is_ok_and(|v| v == "1"))
}

/// A door between the lobby and a painting's room (levels/castle_inside/
/// areas/1/collision.inc.c special objects: the 0-star door to BoB's room,
/// the 1-star door to WF's, the 3-star doors to CCM's and JRB's; yaw in
/// degrees, from the preset's 0-255): (door position, unit normal pointing
/// to the lobby side, half width of the room across the normal, top of the
/// room above the door's floor). A painting in a room has no gate.
struct Gate {
    pos: [f32; 3],
    normal: [f32; 2],
    half: f32,
    top: f32,
}

fn gate(name: &str) -> Option<Gate> {
    let g = |pos: [f32; 3], yaw: f32, half: f32, top: f32| {
        let y = yaw.to_radians();
        Some(Gate { pos, normal: [y.sin(), y.cos()], half, top })
    };
    match name {
        // Yaw 64 (90): the lobby is east of it; the room runs to x -5222.
        "bob" => g([-3122.0, 205.0, -793.0], 90.0, 1500.0, 1500.0),
        // Yaw 0: the lobby is north (+z). WF's room (x ~ -500..1000) and
        // the hallway above it (y 614); CCM's mirrors it about x -1023.
        "wf" => g([256.0, 0.0, -1074.0], 0.0, 900.0, 900.0),
        "ccm" => g([-2303.0, 0.0, -1074.0], 0.0, 900.0, 900.0),
        // Yaw 192 (270): the lobby is west (-x).
        "jrb" => g([1075.0, 205.0, -229.0], 270.0, 1500.0, 1500.0),
        _ => None,
    }
}

impl Gate {
    /// Whether `p` is past the door, in the painting's room.
    fn past(&self, p: [f32; 3]) -> bool {
        let (rx, rz) = (p[0] - self.pos[0], p[2] - self.pos[2]);
        let front = rx * self.normal[0] + rz * self.normal[1];
        let across = (rx * self.normal[1] - rz * self.normal[0]).abs();
        front < 0.0 && across < self.half && p[1] < self.pos[1] + self.top
    }
}

/// Staged mode: frames to go through a door on the way (the push / pull
/// door action, act_pushing_door / act_pulling_door, plays a ~1 s
/// animation). Rough; calibrate from runs (tools/analysis/eta_calib.py).
const DOOR_FRAMES: f64 = 30.0;

pub struct Enter {
    painting: &'static Painting,
    centre: [f32; 3],
    /// OBRRT_PAINTING_FAST: the door on the way, the other rooms' doors
    /// in its area (out of another room first: BoB's exit puts Mario in
    /// BoB's room), and the collected set the objective belongs to.
    gate: Option<Gate>,
    others: Vec<Gate>,
    progress: Option<u64>,
}

impl Enter {
    pub fn new(b: &St, p: &'static Painting) -> Enter {
        let fast = painting_fast();
        // OBRRT_PAINTING_FAST: a wall painting is aimed at 150 above its
        // bottom edge, not its centre: the warp floor is touched anywhere
        // in the painting's plane above it (the routes' triggers: BoB's at
        // y 446-566, bottom 410, centre 716; WF's at -177..174, bottom
        // -205), and the centre of a 614-high painting hung 200 above the
        // floor is a double jump up.
        let mut centre = p.centre();
        if fast && p.pitch == 0.0 {
            centre[1] = centre[1].min(p.corner[1] + 150.0);
        }
        Enter {
            painting: p,
            centre,
            gate: if fast && p.area == 1 { gate(p.name) } else { None },
            others: if fast && p.area == 1 {
                crate::painting_table::PAINTINGS
                    .iter()
                    .filter(|q| q.area == 1 && q.name != p.name)
                    .filter_map(|q| gate(q.name))
                    .collect()
            } else {
                Vec::new()
            },
            progress: fast.then(|| b.progress_set()),
        }
    }

    /// The stage `s` is in and its waypoint: 0, in another painting's room:
    /// its door; 1, before this painting's door: the door; 2: the painting.
    fn stage(&self, s: &St) -> (u32, [f32; 3]) {
        if let Some(g) = self.others.iter().find(|g| g.past(s.pos)) {
            return (0, g.pos);
        }
        match &self.gate {
            Some(g) if !g.past(s.pos) => (1, g.pos),
            _ => (2, self.centre),
        }
    }

    /// Straight-line distance from `s` through the waypoints left.
    fn to_go(&self, s: &St) -> f32 {
        let (k, w) = self.stage(s);
        let mut d = s.dist(w);
        let mut at = w;
        if k == 0 {
            if let Some(g) = &self.gate {
                d += dist(at, g.pos);
                at = g.pos;
            }
        }
        if k < 2 {
            d += dist(at, self.centre);
        }
        d
    }

    /// Doors left to go through from stage `k` ([`Enter::stage`]): out of
    /// another room, then this painting's door, if it has one.
    fn doors_left(&self, k: u32) -> u32 {
        match k {
            0 => 1 + self.gate.is_some() as u32,
            1 => 1,
            _ => 0,
        }
    }

    fn in_room(&self, s: &St) -> bool {
        s.level as u32 == LEVEL_CASTLE && s.area == self.painting.area
    }
}

impl Objective for Enter {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): paintings: only the end matters.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Painting::Enter"
    }
    fn key(&self) -> String {
        match self.progress {
            Some(p) => format!("Painting {} p{:016x}", self.painting.name, p),
            None => format!("Painting {}", self.painting.name),
        }
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
        if self.progress.is_some_and(|p| p != s.progress_set()) {
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
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.level as u32 == self.painting.dest {
            return Eta::Done;
        }
        if !self.in_room(s) {
            return Eta::Lost;
        }
        if self.progress.is_none() {
            return Eta::Frames(frames_for(s.dist(self.centre)));
        }
        // Staged: straight lines through the waypoints left, plus the doors
        // on the way (the sum only drops as a door is passed).
        let (k, _) = self.stage(s);
        Eta::Frames(frames_for(self.to_go(s)) + DOOR_FRAMES * self.doors_left(k) as f64)
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        let horizon = if self.progress.is_some() && self.painting.area == 1 { 420 } else { 300 };
        Some(obrrt_core::GaBudget { population: 20, generations: 6, horizon })
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        Some(self.stage(s).1)
    }
    fn arrival(&self, s: &St) -> obrrt_core::Arrival {
        match self.stage(s) {
            (2, _) => obrrt_core::Arrival::Pass,
            _ => obrrt_core::Arrival::Door,
        }
    }
    fn kinship(&self) -> u32 {
        10
    }
}
