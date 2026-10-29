//! `Mario::Settle` and `Mario::Wait(n)`: standing still, for clean start
//! states and for timing.

use obrrt_core::layout::ACT_FLAG_STATIONARY;
use obrrt_core::{Budget, GamePad, Objective, Progress, St};

fn still(s: &St) -> bool {
    s.action_has(ACT_FLAG_STATIONARY) && s.forward_vel.abs() < 0.1 && s.vel[1].abs() < 0.1
}

/// Come to a stop on the ground.
pub struct Settle {
    zone: (u16, u8),
    cell: (i32, i32, i32),
}

impl Settle {
    pub fn new(b: &St) -> Settle {
        let c = |v: f32| (v / 800.0).round() as i32;
        Settle { zone: b.zone(), cell: (c(b.pos[0]), c(b.pos[1]), c(b.pos[2])) }
    }
}

impl Objective for Settle {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): waiting in place: nothing new along the way.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Mario::Settle"
    }
    fn key(&self) -> String {
        format!("Settle {:?} {:?}", self.zone, self.cell)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.3
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && !still(s)
    }
    fn base_heuristic(&self, _s: &St) -> f64 {
        0.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().skip(1).position(still).map(|i| i + 1)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, v) = traj
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.forward_vel.abs() + s.vel[1].abs()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        Progress { score: -(v as f64), at }
    }
    fn budget(&self) -> Budget {
        Budget::Macro(vec![GamePad::new(0, 0, 0); 60])
    }
    fn kinship(&self) -> u32 {
        2
    }
}

/// Stand still for `n` frames: lets platforms, enemies and cycles line up.
pub struct Wait {
    zone: (u16, u8),
    n: usize,
}

impl Wait {
    pub fn new(b: &St, n: usize) -> Wait {
        Wait { zone: b.zone(), n }
    }
}

impl Objective for Wait {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): waiting in place: nothing new along the way.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::EndOnly
    }
    fn kind(&self) -> &'static str {
        "Mario::Wait"
    }
    fn key(&self) -> String {
        format!("Wait {:?} {}", self.zone, self.n)
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        0.15
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone && still(s)
    }
    fn base_heuristic(&self, _s: &St) -> f64 {
        0.0
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        (traj.len() > self.n).then_some(self.n)
    }
    fn progress(&self, traj: &[St]) -> Progress {
        Progress { score: traj.len() as f64, at: traj.len() - 1 }
    }
    fn budget(&self) -> Budget {
        Budget::Macro(vec![GamePad::new(0, 0, 0); self.n])
    }
    fn kinship(&self) -> u32 {
        2
    }
}
