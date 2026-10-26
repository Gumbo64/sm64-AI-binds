//! Helpers shared by the collectible objectives.

use obrrt_core::{Obj, ObjId, Progress, St, dist};

/// The object in `s`, if `s` is in `zone` and has it loaded.
pub fn find(s: &St, zone: (u16, u8), id: ObjId) -> Option<&Obj> {
    if s.zone() == zone { s.find(id) } else { None }
}

/// Whether the object was collected (or otherwise removed) in `s`: Mario is
/// alive in the same zone and the object is no longer in the pool.
pub fn gone(s: &St, zone: (u16, u8), id: ObjId) -> bool {
    s.zone() == zone && s.alive() && s.objects.is_some() && s.find(id).is_none()
}

/// −(closest Mario got to the point `target` gives per state), at that state.
pub fn closest(traj: &[St], target: impl Fn(&St) -> Option<[f32; 3]>) -> Progress {
    let (at, d) = traj
        .iter()
        .enumerate()
        .filter_map(|(i, s)| target(s).map(|p| (i, dist(s.pos, p))))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f32::MAX));
    Progress { score: -(d as f64), at }
}

/// The best `score` over a trajectory, at its first index.
pub fn best(traj: &[St], score: impl Fn(&St) -> Option<f64>) -> Progress {
    let mut out = Progress { score: f64::MIN, at: 0 };
    for (i, s) in traj.iter().enumerate() {
        if let Some(v) = score(s) {
            if v > out.score {
                out = Progress { score: v, at: i };
            }
        }
    }
    out
}
