//! `Shell::Ride(id)`: get on a Koopa shell (`bhvKoopaShell`,
//! `INTERACT_KOOPA_SHELL`). interact_koopa_shell (interaction.c) mounts Mario
//! (ACT_RIDING_SHELL_GROUND) when he lands on it from above or touches it
//! while in ACT_WALKING / ACT_HOLD_WALKING; otherwise it just pushes him.
//! Riding crosses lava (LLL, BitFS) and quicksand (SSL) safely. The
//! underwater shell (`bhvKoopaShellUnderwater`) is grabbable: `Object::Grab`.

use std::sync::Arc;

use obrrt_core::layout::*;
use obrrt_core::{Eta, GaBudget, Mem, Obj, Objective, St, dist, frames_for};

use super::common::Ident;

/// Touching the shell to riding it: landing on it from a jump, or walking
/// into it (~1/3 s).
const MOUNT_FRAMES: f64 = 10.0;

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
    fn eta(&self, s: &St, _start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        if riding(s) {
            return Eta::Done;
        }
        self.shell(s).map_or(Eta::Lost, |o| Eta::Frames(frames_for(dist(s.pos, o.pos)) + MOUNT_FRAMES))
    }
    fn success_value(&self) -> f64 {
        1.0
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 150 })
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
