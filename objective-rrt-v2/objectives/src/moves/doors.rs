//! Doors the library has seen, per zone, for door approaches
//! ([`super::arcs::plan_approach_with`]): rollout states usually come
//! without their object list, so doors are remembered from the states that
//! have one (a rollout's start). Entries are tagged with the publish epoch
//! they were added in and used only from the next one on
//! ([`obrrt_core::mover::epoch`]), so what a round's rollouts see doesn't
//! depend on thread timing.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use obrrt_core::St;
use obrrt_core::layout::{INTERACT_DOOR, INTERACT_WARP_DOOR};

type Zone = (u16, u8);

#[derive(Clone, Copy)]
struct Door {
    epoch: u64,
    pos: [f32; 3],
    yaw: i32,
}

#[derive(Default)]
struct Doors(RwLock<HashMap<Zone, Vec<Door>>>);

/// This thread's runtime's doors (the engine's own inside its threads).
fn doors() -> Arc<Doors> {
    obrrt_core::runtime::current().ext::<Doors>()
}

fn same(a: [f32; 3], b: [f32; 3]) -> bool {
    (a[0] - b[0]).abs() < 1.0 && (a[1] - b[1]).abs() < 1.0 && (a[2] - b[2]).abs() < 1.0
}

/// Remembers the doors in `st`'s object list (if it has one).
pub fn observe(st: &St) {
    let Some(objs) = st.objects.as_deref() else { return };
    let found: Vec<([f32; 3], i32)> = objs
        .iter()
        .filter(|o| o.interact_type & (INTERACT_DOOR | INTERACT_WARP_DOOR) != 0)
        .map(|o| (o.pos, o.yaw))
        .collect();
    if found.is_empty() {
        return;
    }
    let zone = st.zone();
    let known = |m: &HashMap<Zone, Vec<Door>>, p: [f32; 3]| m.get(&zone).is_some_and(|v| v.iter().any(|d| same(d.pos, p)));
    let doors = doors();
    if found.iter().all(|(p, _)| known(&doors.0.read().unwrap(), *p)) {
        return;
    }
    let epoch = obrrt_core::mover::epoch();
    let mut m = doors.0.write().unwrap();
    for (p, yaw) in found {
        if !known(&m, p) {
            m.entry(zone).or_default().push(Door { epoch, pos: p, yaw });
        }
    }
}

/// The published door within 50 (horizontally, 150 vertically) of
/// `target`, if any: (position, facing yaw). The nearest wins; equal
/// distances by position, so the answer doesn't depend on insertion order.
pub fn door_at(zone: Zone, target: [f32; 3]) -> Option<([f32; 3], i32)> {
    let now = obrrt_core::mover::epoch();
    let doors = doors();
    let m = doors.0.read().unwrap();
    let v = m.get(&zone)?;
    v.iter()
        .filter(|d| d.epoch < now)
        .map(|d| {
            let h = ((d.pos[0] - target[0]).powi(2) + (d.pos[2] - target[2]).powi(2)).sqrt();
            (h, d)
        })
        .filter(|(h, d)| *h <= 50.0 && (d.pos[1] - target[1]).abs() <= 150.0)
        .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.pos[0].total_cmp(&b.1.pos[0])).then(a.1.pos[2].total_cmp(&b.1.pos[2])))
        .map(|(_, d)| (d.pos, d.yaw))
}
