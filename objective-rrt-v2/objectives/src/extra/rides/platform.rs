//! `Platform::Board(id)`: get onto a moving platform; `Platform::Ride(id)`:
//! be on it after it has carried Mario some distance from where it was at the
//! start (most of these only start moving once Mario stands on them, e.g.
//! `platform_act_wait_for_mario` in platform_on_track.inc.c, `elevator_act_0`
//! in elevator.inc.c, arrow_lift.inc.c, express_elevator.inc.c,
//! pyramid_elevator.inc.c, tower_platform.inc.c). `Seesaw::Board(id)`: get
//! onto a seesaw (it tilts in place, so there is nothing to ride).
//!
//! "On it" is inferred (St has no `marioObj->platform`): Mario grounded and
//! inside the platform's collision box (from its collision.inc.c, rotated by
//! its yaw) at a height between the box's bottom and top. A Ride also needs
//! the platform to have moved, which level floor next to it can't fake.

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Eta, Mem, Obj, Objective, St, dist, frames_for};

use super::common::{Ident, Shape, grounded};

/// Units a frame these platforms carry Mario at (elevators and track
/// platforms move ~5-15 u/frame), for the ride's remaining distance.
const PLATFORM_SPEED: f64 = 10.0;
/// Reaching its top to standing on it: the landing (~1/3 s).
const BOARD_FRAMES: f64 = 10.0;

#[derive(Debug, Clone, Copy)]
struct Spec {
    weight: f64,
    shape: Shape,
    /// How far the platform must carry Mario for a Ride; None: Board only.
    travel: Option<f32>,
    /// Whether the object moves (so a home-less one is found by slot).
    moving: bool,
    /// Extra height tolerance (seesaws tilt).
    tilt: f32,
    /// The star this ride leads to, if it's for one.
    star: Option<(i16, u8)>,
    seesaw: bool,
}

const fn ride(weight: f64, shape: Shape, travel: f32) -> Spec {
    Spec { weight, shape, travel: Some(travel), moving: true, tilt: 0.0, star: None, seesaw: false }
}

const fn sh(x: [f32; 2], y: [f32; 2], z: [f32; 2]) -> Shape {
    Shape::new(x, y, z)
}

/// What `o` is as a ride. Shapes are the min/max of each collision model's
/// vertices (file in the comment).
fn spec_of(o: &Obj) -> Option<Spec> {
    let p2 = (o.beh_params >> 16) & 0xFF;
    Some(match o.bhv_name() {
        // platform_on_track.inc.c: type = (2nd byte & 0x70) >> 4, the
        // collision model per type in sPlatformOnTrackCollisionModels.
        "bhvPlatformOnTrack" => match (p2 & 0x70) >> 4 {
            // rr/flying_carpet: the Rainbow Ride carpets.
            0 => ride(4.0, sh([-306.0, 307.0], [20.0, 20.0], [-204.0, 205.0]), 600.0),
            // ccm/ropeway_lift: CCM's ski lift (hangs below its origin).
            1 => ride(3.0, sh([-204.0, 205.0], [-255.0, 0.0], [-142.0, 143.0]), 600.0),
            // actors/checkerboard_platform.
            2 => ride(2.0, sh([-255.0, 256.0], [-25.0, 26.0], [-153.0, 154.0]), 600.0),
            // bitfs/platform_on_track: the grates (BitFS, HMC, LLL).
            _ => ride(2.0, sh([-306.0, 307.0], [0.0, 205.0], [-306.0, 307.0]), 600.0),
        },
        // elevator.inc.c; hmc/elevator_platform, bbh/mesh_elevator,
        // rr/elevator_platform.
        "bhvHmcElevatorPlatform" => ride(3.0, sh([-255.0, 256.0], [0.0, 102.0], [-255.0, 256.0]), 200.0),
        "bhvMeshElevator" => ride(3.0, sh([-306.0, 307.0], [-101.0, 0.0], [-255.0, 256.0]), 200.0),
        "bhvRrElevatorPlatform" => ride(2.5, sh([-204.0, 205.0], [0.0, 102.0], [-306.0, 307.0]), 200.0),
        // express_elevator.inc.c: goes down 20/frame for 132 frames once
        // stood on, waits, comes back. wdw/express_elevator. WDW's 4th star
        // ("Express Elevator--Hurry Up!", behParam 0x03) is placed right by
        // it at (890, 3400, -2040) in wdw/script.c; how exactly the ride
        // leads to it is not modelled here.
        "bhvWdwExpressElevator" => {
            Spec { weight: 5.0, star: Some((11, 3)), ..ride(5.0, sh([-357.0, 512.0], [0.0, 51.0], [-64.0, 384.0]), 1000.0) }
        }
        // arrow_lift.inc.c: moves 384 away and back. wdw/arrow_lift.
        "bhvArrowLift" => ride(3.0, sh([-64.0, 63.0], [0.0, 127.0], [-60.0, 67.0]), 300.0),
        // pyramid_elevator.inc.c; ssl/pyramid_elevator.
        "bhvPyramidElevator" => ride(3.0, sh([-511.0, 512.0], [-50.0, 256.0], [-511.0, 512.0]), 500.0),
        // ttc/elevator_platform.
        "bhvTTCElevator" => ride(2.0, sh([-255.0, 256.0], [-153.0, 0.0], [-127.0, 128.0]), 200.0),
        // tower_platform.inc.c; wf/extending_platform.
        "bhvWfElevatorTowerPlatform" => ride(3.0, sh([-214.0, 215.0], [0.0, 102.0], [-204.0, 205.0]), 200.0),
        "bhvWfSlidingTowerPlatform" => ride(2.0, sh([-214.0, 215.0], [0.0, 102.0], [-204.0, 205.0]), 200.0),
        // checkerboard_platform.inc.c (BitS elevators).
        "bhvCheckerboardPlatformSub" => ride(2.0, sh([-255.0, 256.0], [-25.0, 26.0], [-153.0, 154.0]), 200.0),
        // controllable_platform.inc.c: HMC's arrow platform, steered by its
        // buttons. hmc/arrow_platform.
        "bhvControllablePlatform" => ride(3.0, sh([-306.0, 307.0], [0.0, 51.0], [-306.0, 307.0]), 400.0),
        // dorrie.inc.c: HMC's Dorrie carries Mario across the lake.
        // actors/dorrie.
        "bhvDorrie" => ride(3.0, sh([-313.0, 307.0], [-192.0, 717.0], [-947.0, 822.0]), 600.0),
        // activated_bf_plat.inc.c: type = (2nd byte & 0x300) >> 8 of the
        // u16 >> 16, i.e. byte 0's low 2 bits.
        "bhvActivatedBackAndForthPlatform" => match (o.beh_params >> 24) & 3 {
            // bits/areas/1/25: BitS arrow platform.
            0 => ride(2.0, sh([-460.0, 461.0], [0.0, 102.0], [-306.0, 307.0]), 250.0),
            // bitfs/platform_on_track: mesh platform.
            1 => ride(2.0, sh([-306.0, 307.0], [0.0, 205.0], [-306.0, 307.0]), 250.0),
            // bitfs/elevator.
            _ => ride(2.0, sh([-204.0, 205.0], [0.0, 154.0], [-204.0, 205.0]), 250.0),
        },
        // ferris_wheel.inc.c: both BitDW's and BitS's platforms share a shape.
        "bhvFerrisWheelPlatform" => ride(1.5, sh([-204.0, 205.0], [0.0, 102.0], [-306.0, 307.0]), 300.0),
        // lll/moving_octagonal_mesh_platform.
        "bhvLllMovingOctagonalMeshPlatform" => ride(1.0, sh([-255.0, 256.0], [0.0, 154.0], [-255.0, 256.0]), 300.0),
        // bitdw/square_platform.
        "bhvSquarishPathMoving" => ride(1.0, sh([-306.0, 307.0], [-306.0, 102.0], [-306.0, 307.0]), 300.0),
        // bitfs/sinking_cage_platform.
        "bhvBitfsSinkingCagePlatform" => ride(1.0, sh([-306.0, 307.0], [0.0, 154.0], [-306.0, 307.0]), 200.0),
        // seesaw_platform.inc.c: model by 2nd byte (sSeesawPlatformCollisionModels).
        "bhvSeesawPlatform" => {
            let shape = match p2 {
                2 => sh([-1381.0, 1382.0], [51.0, 205.0], [-551.0, 779.0]),  // bits/areas/1/27 (S-shaped)
                3 => sh([-180.0, 181.0], [0.0, 51.0], [-904.0, 905.0]),      // bob/seesaw_platform
                4 => sh([-511.0, 512.0], [26.0, 179.0], [-306.0, 307.0]),    // bitfs/seesaw_platform
                6 => sh([-665.0, 666.0], [0.0, 154.0], [-613.0, 410.0]),     // rr/l_platform
                _ => sh([-306.0, 307.0], [0.0, 102.0], [-818.0, 819.0]),     // bitdw, bits, rr, vcutm
            };
            Spec { weight: 0.5, shape, travel: None, moving: false, tilt: 250.0, star: None, seesaw: true }
        }
        _ => return None,
    })
}

/// Whether the platform can be used now: a track platform falling off the
/// end of its track (PLATFORM_ON_TRACK_ACT_FALL = 4) can't.
fn usable(o: &Obj) -> bool {
    !(o.bhv_name() == "bhvPlatformOnTrack" && o.action == 4)
}

pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let Some(spec) = spec_of(o) else { return };
    if !usable(o) {
        return;
    }
    let ident = Ident::of(o, spec.moving);
    let zone = b.zone();
    out.push(Arc::new(Board { ident, zone, spec }));
    if spec.travel.is_some() {
        out.push(Arc::new(Ride { ident, zone, spec }));
    }
    // Off: the star was not had from the gap (zt-wdwdrop-1: WdwDrop::Gap 21%, Star::Collect 0 of 251 after it).
    if false && o.bhv_name() == "bhvWdwExpressElevator" && b.level as u32 == LEVEL_WDW && !b.has_star(11, 3) {
        out.push(Arc::new(WdwDrop { zone }));
    }
    if o.bhv_name() == "bhvWdwExpressElevator" && b.level as u32 == LEVEL_WDW && !b.has_star(11, 3) {
        out.push(Arc::new(WdwRide { zone }));
    }
}

fn on(spec: &Spec, s: &St, o: &Obj) -> bool {
    grounded(s) && spec.shape.contains(o, s.pos, 30.0, 40.0 + spec.tilt)
}

fn star_open(spec: &Spec, s: &St) -> bool {
    spec.star.is_none_or(|(c, i)| !s.has_star(c, i))
}

pub struct Board {
    ident: Ident,
    zone: (u16, u8),
    spec: Spec,
}

impl Board {
    fn plat<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone {
            return None;
        }
        self.ident.find(s)
    }
}

impl Objective for Board {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        if self.spec.seesaw { "Seesaw::Board" } else { "Platform::Board" }
    }
    fn key(&self) -> String {
        format!("Board {:?} {}", self.zone, self.ident.key())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // A stepping stone to the Ride.
        if self.spec.seesaw { self.spec.weight } else { self.spec.weight * 0.6 }
    }
    fn base_criteria(&self, s: &St) -> bool {
        star_open(&self.spec, s) && self.plat(s).is_some_and(|p| usable(p) && !on(&self.spec, s, p))
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        let Some(p) = self.plat(s) else { return Eta::Lost };
        if on(&self.spec, s, p) {
            return Eta::Done;
        }
        Eta::Frames(frames_for(dist(s.pos, self.spec.shape.top(p))))
    }
    fn success_value(&self) -> f64 {
        0.3
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.plat(s).map(|p| self.spec.shape.top(p))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.plat(s).is_some()
    }
}

pub struct Ride {
    ident: Ident,
    zone: (u16, u8),
    spec: Spec,
}

impl Ride {
    fn plat<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone {
            return None;
        }
        self.ident.find(s)
    }

    fn travel(&self) -> f32 {
        self.spec.travel.unwrap_or(f32::MAX)
    }
}

impl Objective for Ride {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Platform::Ride"
    }
    fn key(&self) -> String {
        format!("Ride {:?} {}", self.zone, self.ident.key())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        self.spec.weight
    }
    fn base_criteria(&self, s: &St) -> bool {
        star_open(&self.spec, s) && self.plat(s).is_some_and(usable)
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 12, generations: 4, horizon: 300 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        // How far it carried Mario is measured from where it was at the start.
        let Some(p0) = self.plat(start).map(|p| p.pos) else { return Eta::Lost };
        let Some(p) = self.plat(s) else { return Eta::Lost };
        let moved = dist(p.pos, p0);
        let ride = (self.travel() - moved).max(0.0) as f64 / PLATFORM_SPEED;
        if on(&self.spec, s, p) {
            if moved >= self.travel() {
                return Eta::Done;
            }
            return Eta::Frames(ride);
        }
        Eta::Frames(frames_for(dist(s.pos, self.spec.shape.top(p))) + BOARD_FRAMES + ride)
    }
    fn success_value(&self) -> f64 {
        if self.spec.star.is_some() { 3.0 } else { 1.0 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn arrival(&self, _s: &St) -> obrrt_core::Arrival {
        obrrt_core::Arrival::Stop
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        // Onto it, then stay in its middle.
        self.plat(s).map(|p| self.spec.shape.top(p))
    }
    fn exists_in(&self, s: &St) -> bool {
        self.plat(s).is_some()
    }
}


/// `WdwDrop::Gap` (WDW's 4th star, "Express Elevator--Hurry Up!"): send the express elevator down and be back at the
/// top while it is away. The star (890, 3400, -2040) is in a cage at the top of the elevator's shaft
/// (wdw/script.c: the cage, bhvWdwExpressElevatorPlatform, at (1024, 3277, -2112); the elevator at (1024, 3277,
/// -1663)); the elevator in its home blocks the way in. Standing on it starts it: down 20/frame for 132 frames, a
/// 110-frame wait, back up at 10/frame (express_elevator.inc.c), about 500 frames with the shaft open. Done: the
/// elevator 300 or more below its home and Mario alive, above it by 300 or more (not riding it down); then
/// `Star::Collect`, a few frames away, takes over. Three t1 trees (steps 211, 365, 366; 1.2 B frames) stood on the
/// cage roof 184 above the star ("ETA 6") and never got in.
pub struct WdwDrop {
    zone: (u16, u8),
}

fn express(s: &St) -> Option<&Obj> {
    s.objects().iter().find(|o| o.bhv_name() == "bhvWdwExpressElevator")
}

impl Objective for WdwDrop {
    fn kind(&self) -> &'static str {
        "WdwDrop::Gap"
    }
    fn key(&self) -> String {
        "WdwDrop".into()
    }
    /// A goal (over OBRRT_GOAL_MIN 20): the course's star depends on it.
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(11, 3) && express(s).is_some_and(|o| o.action == 0)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(11, 3) && express(s).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        let Some(o) = express(s) else { return Eta::Lost };
        let low = o.home[1] - o.pos[1];
        if low >= 300.0 {
            // Riding it down is no use: the way in is at the top.
            return if s.pos[1] >= o.pos[1] + 300.0 { Eta::Done } else { Eta::Lost };
        }
        if o.action == 0 {
            // Get onto it.
            return Eta::Frames(frames_for(dist(s.pos, o.pos)) + 20.0);
        }
        // Moving down with Mario on it: off and up again before it is 300 down.
        Eta::Frames(((300.0 - low) / 20.0).max(1.0) as f64)
    }
    fn success_value(&self) -> f64 {
        3.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        express(s).map(|o| if o.action == 0 { o.pos } else { o.home })
    }
}


/// The wire cage around WDW's express elevator (levels/wdw/areas/1/collision.inc.c, SURFACE_WALL_MISC): x 640..1178,
/// z -2175..-1279, roofed at 3523; its west, north and south sides end at y 2304, its east side (x 1178) goes down to
/// 404. The elevator (bhvWdwExpressElevator, collision x -357..512, z -64..384, top 51 from its origin at (1024,
/// 3277, -1663)) reaches through that east wall: x 667..1536.
const CAGE_X: [f32; 2] = [640.0, 1178.0];
const CAGE_Z: [f32; 2] = [-2175.0, -1279.0];
const LIFT_TOP: f32 = 51.0;
const CAGE_ROOF: f32 = 3500.0;
/// express_elevator.inc.c: down 20/frame for 133 frames, a 110-frame wait, up at 10/frame to its home.
const LIFT_DOWN: f32 = 20.0 * 133.0;
const LIFT_WAIT: f64 = 110.0;
/// Frames for a leg of the climb: flat at the reference pace, up at ~8 u/frame (wall kicks, swimming), down fast.
fn leg(p: [f32; 3], w: [f32; 3]) -> f64 {
    let h = ((p[0] - w[0]).powi(2) + (p[2] - w[2]).powi(2)).sqrt();
    frames_for(h) + ((w[1] - p[1]).max(0.0) / 8.0) as f64 + ((p[1] - w[1]).max(0.0) / 40.0) as f64
}

/// `WdwRide::Cage` (WDW's 4th star, "Express Elevator--Hurry Up!"): the star (890, 3400, -2040) is inside the cage
/// at the top of the shaft, beside the elevator's home. Stand on the elevator's part outside the cage (east of x
/// 1178) to send it down, ride it down, step off at the bottom, go round the end of the east wall (the other sides
/// are open below 2304) and be on its part inside the cage before it rises: it carries Mario back up into the cage.
/// Done inside the cage at the top; `Star::Collect` takes over. v2 (this one) from the collision of the tower round
/// the shaft (levels/wdw/areas/1/collision.inc.c): it is hollow, its floor at y 205 and its walls up to 2304 (the
/// wire cage above them, to 3523/3584, is closed too), and the only way in is a door at its east foot (the tunnel
/// x 1178..1280, z -1479..-1279, y 205..404). The elevator's lowest top (668) is ~460 above the floor, its north
/// half (z < -1727) is open to the platform above, so the trip is: ride down on the outside part, walk in by the
/// door (with the water under the floor: `WdwWater::Lower`; swimming doesn't fit, wall radius 110 > 100), get up
/// the shaft's north side (wall kicks) on to the elevator's top while it waits (110 frames) or rises, ride up. v1
/// pruned every state below the elevator's top - 250 as Lost, which is exactly the states the climb needs;
/// `swim_route` gives the waypoints and the ETA counts climbs at ~8 u/frame. WdwDrop (in from the top while the
/// elevator is down) /// failed: the cage is roofed.
pub struct WdwRide {
    zone: (u16, u8),
}

fn in_cage(p: [f32; 3]) -> bool {
    (CAGE_X[0]..CAGE_X[1]).contains(&p[0]) && (CAGE_Z[0]..CAGE_Z[1]).contains(&p[2])
}

impl WdwRide {
    /// Frames until the elevator is home again (0 at home).
    fn to_home(o: &Obj) -> f64 {
        let up = ((o.home[1] - o.pos[1]) / 10.0).max(0.0) as f64;
        match o.action {
            1 => ((o.pos[1] - (o.home[1] - LIFT_DOWN)) / 20.0).max(0.0) as f64 + LIFT_WAIT + (LIFT_DOWN / 10.0) as f64,
            2 => LIFT_WAIT + up,
            3 => up,
            _ => 0.0,
        }
    }
    /// Where to stand on it: its outside part (x 1400) at home, its inside part (x 900) while it is away.
    fn spot(o: &Obj, inside: bool) -> [f32; 3] {
        [if inside { 900.0 } else { 1400.0 }, o.pos[1] + LIFT_TOP, o.home[2] + 160.0]
    }
    /// The elevator down and Mario below its top or on its outside part: the waypoints from `s` to its inside part. The tower's walls (levels/wdw/areas/1/collision.inc.c) run from the shaft's
    /// floor (y 205) up to 2304 (the wire cage above, to 3523); the one way in is a door at its east side, the
    /// tunnel x 1178..1280, z -1479..-1279, y 205..404, at the foot of the elevator's east end (which hangs
    /// outside the wall at the top). Inside, the elevator (z -1727..-1279) covers the shaft's south half: up
    /// the north half (z < -1727, under the static platform), then over its top.
    fn swim_route(s: &St, o: &Obj) -> Vec<[f32; 3]> {
        let top = o.pos[1] + LIFT_TOP;
        let door_out = [1340.0, 250.0, -1380.0];
        let door_in = [1100.0, 250.0, -1380.0];
        let north = [900.0, 250.0, -1950.0];
        let up = [900.0, top + 60.0, -1950.0];
        if s.pos[0] > 1300.0 {
            vec![door_out, door_in, north, up]
        } else if s.pos[0] > CAGE_X[1] + 20.0 {
            vec![door_in, north, up]
        } else if s.pos[1] < top - 20.0 && s.pos[2] > o.home[2] - 64.0 {
            vec![north, up]
        } else if s.pos[1] < top - 20.0 {
            vec![up]
        } else {
            vec![]
        }
    }
    fn on_inside(s: &St, o: &Obj) -> bool {
        in_cage(s.pos) && s.pos[2] > o.home[2] - 64.0 && (s.pos[1] - (o.pos[1] + LIFT_TOP)).abs() < 40.0
    }
}

impl Objective for WdwRide {
    fn kind(&self) -> &'static str {
        "WdwRide::Cage"
    }
    fn key(&self) -> String {
        "WdwRide".into()
    }
    /// A goal (over OBRRT_GOAL_MIN 20): the course's star depends on it.
    fn obj_weight(&self, _b: &St) -> f64 {
        30.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && !s.has_star(11, 3)
            && express(s).is_some()
            && !(in_cage(s.pos) && s.pos[1] > 3200.0 && s.pos[1] < CAGE_ROOF)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(11, 3) && express(s).is_some()
    }
    fn ga_budget(&self) -> Option<obrrt_core::GaBudget> {
        Some(obrrt_core::GaBudget { population: 16, generations: 5, horizon: 300 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone || !s.alive() {
            return Eta::Lost;
        }
        let Some(o) = express(s) else { return Eta::Lost };
        // Under the cage's roof (3523; the trees often stood on it, 184 above the star).
        if in_cage(s.pos) && s.pos[1] > o.home[1] - 20.0 && s.pos[1] < CAGE_ROOF {
            return Eta::Done;
        }
        let home = o.action == 0 || o.action == 4 || o.home[1] - o.pos[1] < 1.0;
        if home {
            // Onto its outside part; then the whole trip.
            let t = Self::spot(o, false);
            // Walking distance when the mesh has it (OBRRT_GEO=1): the straight line led the water-raised trees to
            // the water surface right under the elevator (y 2736, 600 below its top: w3 test).
            let d = obrrt_core::geo::path_dist(s.zone(), s.pos, t).unwrap_or_else(|| dist(s.pos, t));
            return Eta::Frames(frames_for(d) + (LIFT_DOWN / 20.0) as f64 + LIFT_WAIT + (LIFT_DOWN / 10.0) as f64);
        }
        if Self::on_inside(s, o) {
            return Eta::Frames(Self::to_home(o).max(1.0));
        }
        // To its inside part, at its height round the south end of the east wall (which ends at the elevator's
        // south edge, z -1279) when outside it.
        let t = Self::spot(o, true);
        let top = o.pos[1] + LIFT_TOP;
        let low = s.pos[1] < top - 20.0 || (s.pos[0] > CAGE_X[1] && s.pos[1] < top + 100.0);
        let d = if low {
            // Below its top (or on its outside part): in by the door at the shaft's foot and up the shaft (see
            // [`Self::swim_route`]); climbs cost more than flat ground.
            let mut d = 0.0;
            let mut p = s.pos;
            for w in Self::swim_route(s, o).into_iter().chain([t]) {
                d += leg(p, w);
                p = w;
            }
            d
        } else {
            leg(s.pos, t)
        };
        Eta::Frames(d.max(0.0) + Self::to_home(o))
    }
    fn success_value(&self) -> f64 {
        5.0
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        let o = express(s)?;
        if o.action == 0 || o.action == 4 {
            let t = Self::spot(o, false);
            // A waypoint along the walk there (OBRRT_GEO=1), else the spot itself.
            return Some(obrrt_core::geo::waypoint(s.zone(), s.pos, t, 300.0).unwrap_or(t));
        }
        let t = Self::spot(o, true);
        let top = o.pos[1] + LIFT_TOP;
        if s.pos[1] < top - 20.0 || (s.pos[0] > CAGE_X[1] && s.pos[1] < top + 100.0) {
            return Some(Self::swim_route(s, o).first().copied().unwrap_or(t));
        }
        Some(t)
    }
}
