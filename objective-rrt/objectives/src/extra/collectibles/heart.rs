//! `Heart::Spin(id)`: heal at a spinning heart (bhvRecoveryHeart,
//! recovery_heart.inc.c). It has no interact type: while Mario touches it it
//! spins at 200 * forwardVel + 1000 per frame, and every full turn adds 4 to
//! his heal counter (one wedge). Running through it fast heals fastest.

use obrrt_core::{Budget, Obj, ObjId, Objective, Progress, St};

use super::common::{best, find};

/// Full health is 8 wedges (0x880).
const FULL: i32 = 8;

pub struct Spin {
    id: ObjId,
    zone: (u16, u8),
    pos: [f32; 3],
}

impl Spin {
    pub fn new(b: &St, o: &Obj) -> Spin {
        Spin { id: o.id, zone: b.zone(), pos: o.pos }
    }
}

impl Objective for Spin {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Heart::Spin"
    }
    fn key(&self) -> String {
        format!("Heart {:?} {}", self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        // Worth more the more hurt Mario is.
        0.5 * (FULL - b.wedges()).max(0) as f64
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.alive() && s.wedges() < FULL && find(s, self.zone, self.id).is_some()
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        // Close first, then the most hurt.
        s.dist(self.pos) as f64 / 400.0 - (FULL - s.wedges()) as f64 * 0.5
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        let w0 = traj[0].wedges();
        traj.iter().position(|s| s.zone() == self.zone && s.wedges() > w0 && s.dist(self.pos) < 600.0)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        // Closeness, and speed while close (the heart spins with it).
        best(traj, |s| {
            (s.zone() == self.zone).then(|| {
                let d = s.dist(self.pos) as f64;
                -d + if d < 120.0 { s.forward_vel.abs() as f64 * 5.0 } else { 0.0 }
            })
        })
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 12, generations: 4, horizon: 150 }
    }
    fn steer_target(&self, _s: &St) -> Option<[f32; 3]> {
        Some(self.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && s.objects.as_ref().is_none_or(|_| s.find(self.id).is_some())
    }
}
