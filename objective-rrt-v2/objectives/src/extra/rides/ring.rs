//! `WaterRing::SwimThrough(spawner)`: swim through one of the rings a
//! spawner makes (water_ring.inc.c). Five in a row, each the next index
//! after the last collected, give the spawner's star:
//! - DDD's manta ray (`bhvMantaRay`, behParam 0x04, manta_ray.inc.c) leaves
//!   `bhvMantaRayWaterRing`s behind it: star 5.
//! - DDD's jet stream (`bhvJetStreamRingSpawner`, behParam 0x03) sends
//!   `bhvJetStreamWaterRing`s up the stream every 50 frames: star 4.
//! A ring is collected (oAction 0 -> WATER_RING_ACT_COLLECTED = 1) when
//! Mario's position + 80 crosses its plane within (scale + 0.2) * 120 of its
//! centre. Rings are short-lived and have no home, so the objective belongs
//! to the spawner and rings are told apart by pool slot.
//!
//! Unseen: the spawner's count (oWaterRingSpawnerRingsCollected) and the
//! rings' indices, so a ring taken out of order (which resets the count)
//! still counts as a success here.

use std::sync::Arc;

use obrrt_core::{Eta, GaBudget, Mem, Obj, Objective, St, course_of_level, dist, frames_for};

use super::common::Ident;

/// No ring out: the jet stream sends one every 50 frames (half that on
/// average); the manta leaves one behind it about as often.
const RING_SPAWN_FRAMES: f64 = 25.0;

pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    let (ring, moving) = match o.bhv_name() {
        "bhvMantaRay" => ("bhvMantaRayWaterRing", true),
        "bhvJetStreamRingSpawner" => ("bhvJetStreamWaterRing", false),
        _ => return,
    };
    // The star the spawner gives is its behParam byte 0 (spawn_default_star
    // copies the spawner's oBehParams).
    let star = (course_of_level(b.level as u32), ((o.beh_params >> 24) & 0xFF) as u8);
    if b.has_star(star.0, star.1) {
        return;
    }
    out.push(Arc::new(SwimThrough { spawner: Ident::of(o, moving), ring, zone: b.zone(), star }));
}

pub struct SwimThrough {
    spawner: Ident,
    ring: &'static str,
    zone: (u16, u8),
    star: (i16, u8),
}

impl SwimThrough {
    fn rings<'a>(&self, s: &'a St) -> impl Iterator<Item = &'a Obj> + 'a {
        let ring = self.ring;
        let here = s.zone() == self.zone;
        s.objects().iter().filter(move |o| here && o.bhv_name() == ring)
    }

    /// Where the ring check measures Mario from (water_ring_calc_mario_dist).
    fn probe(s: &St) -> [f32; 3] {
        [s.pos[0], s.pos[1] + 80.0, s.pos[2]]
    }

    /// The uncollected ring nearest Mario, and its distance.
    fn nearest<'a>(&self, s: &'a St) -> Option<(&'a Obj, f32)> {
        let p = Self::probe(s);
        self.rings(s).filter(|o| o.action == 0).map(|o| (o, dist(o.pos, p))).min_by(|a, b| a.1.total_cmp(&b.1))
    }
}

impl Objective for SwimThrough {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "WaterRing::SwimThrough"
    }
    fn key(&self) -> String {
        format!("Rings {:?} {}", self.zone, self.spawner.key())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        8.0
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !s.has_star(self.star.0, self.star.1) && self.nearest(s).is_some()
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        // A ring open at the start now collected.
        let open: Vec<u16> = self.rings(start).filter(|o| o.action == 0).map(|o| o.slot).collect();
        if self.rings(s).any(|o| o.action == 1 && open.contains(&o.slot)) {
            return Eta::Done;
        }
        Eta::Frames(self.nearest(s).map_or(RING_SPAWN_FRAMES, |(_, d)| frames_for(d)))
    }
    fn success_value(&self) -> f64 {
        2.0
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 200 })
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.nearest(s).map(|(o, _)| [o.pos[0], o.pos[1] - 80.0, o.pos[2]])
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && self.spawner.find(s).is_some()
    }
}
