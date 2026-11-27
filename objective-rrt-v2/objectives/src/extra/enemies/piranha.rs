//! Piranha plants.
//!
//! - `PiranhaPlant::Sneak(id)` (bhvPiranhaPlant): it sleeps
//!   (`INTERACT_BOUNCE_TOP`, hitbox radius 250) until Mario comes within 400
//!   moving faster than 10 forward or 10 up, then wakes and bites
//!   (`INTERACT_DAMAGE`). Attacked while asleep (punch, kick, stomp, dive
//!   from outside its wake radius) it dies (`PIRANHA_PLANT_ACT_ATTACKED` 5,
//!   then 6, 7) and leaves a blue coin, then respawns when Mario leaves
//!   (piranha_plant.inc.c). It never disappears, so `Enemy::Defeat` (success
//!   = gone) can't see it die; this objective watches its action instead.
//! - `FirePiranha::Hit(id)` (bhvFirePiranhaPlant, big ones: 2nd param byte
//!   1): THI's five giant fire-spitting plants have 1 health; killing all
//!   five spawns "Pluck the Piranha Flower" (THI star 1;
//!   fire_piranha_plant_act_hide, fire_piranha_plant.inc.c). Each hit
//!   lowers oHealth.

use obrrt_core::buttons::B_BUTTON;
use obrrt_core::{Eta, GaBudget, Mem, Obj, ObjId, Objective, St, dist, frames_for};

use super::common::{bp2, find};

const PIRANHA_PLANT_ACT_ATTACKED: i32 = 5;
const PIRANHA_PLANT_ACT_WAIT_TO_RESPAWN: i32 = 7;
/// THI's piranha flower star: course 13, index 0.
const FLOWER_STAR: (i16, u8) = (13, 0);
/// Within reach to the hit: a dive (sleeping plant, from outside its wake
/// radius) or a punch (fire plant), ~1/3 s.
const HIT_FRAMES: f64 = 10.0;

#[derive(Clone, Copy, PartialEq)]
enum Plant {
    Sleeping,
    Fire,
}

pub struct Hit {
    id: ObjId,
    zone: (u16, u8),
    plant: Plant,
}

impl Hit {
    pub fn new(b: &St, o: &Obj) -> Option<Hit> {
        let plant = match o.bhv_name() {
            "bhvPiranhaPlant" => Plant::Sleeping,
            "bhvFirePiranhaPlant" if bp2(o) != 0 => Plant::Fire,
            _ => return None,
        };
        Some(Hit { id: o.id, zone: b.zone(), plant })
    }

    fn dead_sleeper(o: &Obj) -> bool {
        (PIRANHA_PLANT_ACT_ATTACKED..=PIRANHA_PLANT_ACT_WAIT_TO_RESPAWN).contains(&o.action)
    }
}

impl Objective for Hit {
    fn kind(&self) -> &'static str {
        match self.plant {
            Plant::Sleeping => "PiranhaPlant::Sneak",
            Plant::Fire => "FirePiranha::Hit",
        }
    }
    fn key(&self) -> String {
        format!("{} {:?} {}", self.kind(), self.zone, self.id)
    }
    fn obj_weight(&self, b: &St) -> f64 {
        match self.plant {
            // A blue coin (5).
            Plant::Sleeping => 2.0,
            Plant::Fire if !b.has_star(FLOWER_STAR.0, FLOWER_STAR.1) => 6.0,
            Plant::Fire => 1.0,
        }
    }
    fn base_criteria(&self, s: &St) -> bool {
        s.zone() == self.zone
            && find(s, self.id, false).is_some_and(|o| match self.plant {
                Plant::Sleeping => !Self::dead_sleeper(o),
                Plant::Fire => o.health > 0,
            })
    }
    fn ga_budget(&self) -> Option<GaBudget> {
        Some(GaBudget { population: 16, generations: 5, horizon: 150 })
    }
    fn eta(&self, s: &St, start: &St, _mem: &mut Mem) -> Eta {
        if s.zone() != self.zone {
            return Eta::Lost;
        }
        let o = find(s, self.id, false);
        let met = match self.plant {
            Plant::Sleeping => o.is_some_and(Self::dead_sleeper),
            Plant::Fire => {
                // Its health below the start's, or gone (a state whose
                // objects weren't read can't tell); never without it at the
                // start.
                let Some(hp0) = find(start, self.id, false).map(|o| o.health) else { return Eta::Lost };
                o.map_or(s.objects.is_some(), |o| o.health < hp0)
            }
        };
        if met {
            return Eta::Done;
        }
        o.map_or(Eta::Lost, |o| Eta::Frames(frames_for(dist(s.pos, o.pos)) + HIT_FRAMES))
    }
    fn success_value(&self) -> f64 {
        match self.plant {
            Plant::Sleeping => 2.0,
            Plant::Fire => 3.0,
        }
    }
    fn needs_objects(&self) -> bool {
        true
    }
    fn steer_target(&self, s: &St) -> Option<[f32; 3]> {
        find(s, self.id, false).map(|o| o.pos)
    }
    fn steer_button(&self, s: &St) -> u16 {
        // A sleeping plant: dive (B while running) from just outside its
        // wake radius. A fire plant: punch when close.
        let reach = match self.plant {
            Plant::Sleeping => 450.0,
            Plant::Fire => 200.0,
        };
        match find(s, self.id, false) {
            Some(o) if dist(s.pos, o.pos) < reach => B_BUTTON,
            _ => 0,
        }
    }
    fn exists_in(&self, s: &St) -> bool {
        s.zone() == self.zone && find(s, self.id, false).is_some()
    }
}
