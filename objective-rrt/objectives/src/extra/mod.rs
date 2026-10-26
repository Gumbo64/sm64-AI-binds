//! Objectives for the rest of the game, one file per category. Each file
//! offers `generate` (per object near `b`) and `generate_level` (per `b`).
//! Objectives here must not duplicate the hand-written ones in the parent
//! module (coins, stars, doors, paintings, grab, King Bob-omb hits, simple
//! stomp enemies, breakable boxes, ! boxes, NPC talk, dialogs and menus).

use std::sync::Arc;

use obrrt_core::{Obj, Objective, St};

pub mod bosses;
pub mod collectibles;
pub mod enemies;
pub mod missions;
pub mod npcs;
pub mod rides;
pub mod switches;
pub mod tech;
pub mod warps;

pub fn generate(b: &St, o: &Obj, out: &mut Vec<Arc<dyn Objective>>) {
    bosses::generate(b, o, out);
    collectibles::generate(b, o, out);
    enemies::generate(b, o, out);
    missions::generate(b, o, out);
    npcs::generate(b, o, out);
    rides::generate(b, o, out);
    switches::generate(b, o, out);
    tech::generate(b, o, out);
    warps::generate(b, o, out);
}

pub fn generate_level(b: &St, out: &mut Vec<Arc<dyn Objective>>) {
    bosses::generate_level(b, out);
    collectibles::generate_level(b, out);
    enemies::generate_level(b, out);
    missions::generate_level(b, out);
    npcs::generate_level(b, out);
    rides::generate_level(b, out);
    switches::generate_level(b, out);
    tech::generate_level(b, out);
    warps::generate_level(b, out);
}
