//! `Shell::Ride(id)`: get on a Koopa shell (`bhvKoopaShell`,
//! `INTERACT_KOOPA_SHELL`). interact_koopa_shell (interaction.c) mounts Mario
//! (ACT_RIDING_SHELL_GROUND) when he lands on it from above or touches it
//! while in ACT_WALKING / ACT_HOLD_WALKING; otherwise it just pushes him.
//! Riding crosses lava (LLL, BitFS) and quicksand (SSL) safely. The
//! underwater shell (`bhvKoopaShellUnderwater`) is grabbable: `Object::Grab`.

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Budget, Obj, Objective, Progress, St, dist};

use super::common::{Ident, best};

pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    // oAction 0: lying free; 1: ridden (koopa_shell.inc.c).
    if o.bhv_name() == "bhvKoopaShell" && o.action == 0 {
        // No SET_HOME in bhvKoopaShell and it slides around: found by slot.
        out.push(Arc::new(Ride { ident: Ident::of(o, true), zone: b.zone(), level: b.level as u32 }));
    }
}

pub struct Ride {
    ident: Ident,
    zone: (u16, u8),
    level: u32,
}

impl Ride {
    fn shell<'a>(&self, s: &'a St) -> Option<&'a Obj> {
        if s.zone() != self.zone {
            return None;
        }
        self.ident.find(s)
    }
}

fn riding(s: &St) -> bool {
    s.action_has(ACT_FLAG_RIDING_SHELL)
}

impl Objective for Ride {
    /// Dense adds (`OBRRT_DENSE_EVERY=obj`): collecting and fighting: a state every 15 frames, none mid-arc.
    fn dense_keep(&self) -> obrrt_core::DenseKeep {
        obrrt_core::DenseKeep::Every { n: 15, air: false }
    }
    fn kind(&self) -> &'static str {
        "Shell::Ride"
    }
    fn key(&self) -> String {
        format!("Shell {:?} {}", self.zone, self.ident.key())
    }
    fn obj_weight(&self, _b: &St) -> f64 {
        // Where a shell gets Mario across something deadly.
        match self.level {
            LEVEL_LLL | LEVEL_SSL | LEVEL_BITFS => 4.0,
            _ => 2.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        !riding(s) && self.shell(s).is_some_and(|o| o.action == 0)
    }
    fn base_heuristic(&self, s: &St) -> f64 {
        self.shell(s).map_or(f64::MAX, |o| dist(s.pos, o.pos) as f64 / 400.0)
    }
    fn success(&self, traj: &[St]) -> Option<usize> {
        traj.iter().position(|s| s.zone() == self.zone && riding(s))
    }
    fn progress(&self, traj: &[St]) -> Progress {
        let (at, score) = best(traj, |s| self.shell(s).map(|o| -(dist(s.pos, o.pos) as f64)));
        Progress { score, at }
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn budget(&self) -> Budget {
        Budget::Ga { population: 16, generations: 5, horizon: 150 }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        self.shell(s).map(|o| o.pos)
    }
    fn exists_in(&self, s: &St) -> bool {
        self.shell(s).is_some_and(|o| o.action == 0)
    }
}
