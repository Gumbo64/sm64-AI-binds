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
pub mod hmc_drive;
pub mod hmc_metal;
pub mod missions;
pub mod npcs;
pub mod pyramid;
pub mod rides;
pub mod slhead;
pub mod slides;
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
    hmc_drive::generate_level(b, out);
    hmc_metal::generate_level(b, out);
    missions::generate_level(b, out);
    npcs::generate_level(b, out);
    pyramid::generate_level(b, out);
    rides::generate_level(b, out);
    slhead::generate_level(b, out);
    switches::generate_level(b, out);
    tech::generate_level(b, out);
    warps::generate_level(b, out);
}

/// Synthetic states for the unit tests of this category's `eta`s.
#[cfg(test)]
pub(crate) mod testkit {
    use std::sync::Arc;

    use obrrt_core::layout::ACT_IDLE;
    use obrrt_core::{Obj, ObjId, St, bhv_addr};

    /// An object of behaviour `bhv` (its address, so `bhv_name` finds it)
    /// at `pos`, home there too.
    pub fn obj(bhv: &str, slot: u16, pos: [f32; 3], action: i32, health: i32) -> Obj {
        let bhv = bhv_addr(bhv).unwrap_or_else(|| panic!("{bhv} in the layout"));
        Obj {
            id: ObjId { bhv, home: [pos[0] as i32, pos[1] as i32, pos[2] as i32] },
            slot,
            bhv,
            pos,
            home: pos,
            interact_type: 0,
            interact_status: 0,
            action,
            health,
            value: 0,
            loot_coins: 0,
            beh_params: 0,
            held_state: 0,
            intangible_timer: 0,
            yaw: 0,
            raw_1ac: [0; 4],
        }
    }

    /// Mario idle at `pos` in (`level`, area 1) with `objs` loaded.
    pub fn st(level: u32, pos: [f32; 3], objs: Vec<Obj>) -> St {
        St {
            frame: 100,
            pos,
            vel: [0.0; 3],
            forward_vel: 0.0,
            yaw: 0,
            pitch: 0,
            angle_vel: [0; 3],
            water_level: -11000,
            cam_yaw: 0,
            action: ACT_IDLE,
            mario_flags: 0,
            holding: false,
            held_slot: None,
            red_coins: 0,
            level: level as u16,
            area: 1,
            act: 1,
            course: 0,
            stars: 0,
            keys: 0,
            coins: 0,
            lives: 4,
            health: 0x880,
            play_mode: 0,
            transition: 0,
            dialog: -1,
            dialog_choice: false,
            menu_line: 0,
            menu_mode: -1,
            act_select: false,
            act_init: 0,
            act_cursor: 0,
            dialog_state: 0,
            dialog_last_page: false,
            menu_timer: 0,
            floor_y: pos[1],
            ceil_y: 20000.0,
            floor_type: 0,
            floor_normal: [0.0, 1.0, 0.0],
            floor_obj: None,
            floor_obj_vy: None,
            floor_obj_y: None,
            action_timer: 0, cap_timer: 0,
            wall_normal: None,
            course_stars: [0; 25],
            save_flags: 0,
            objects: Some(Arc::new(objs)),
        }
    }

    /// The ETAs (in frames) of a fold must fall strictly.
    pub fn falls(f: &[f64]) -> bool {
        f.windows(2).all(|w| w[1] < w[0])
    }
}
