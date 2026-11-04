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
