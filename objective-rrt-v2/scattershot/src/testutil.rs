//! Helpers for unit tests of objectives (shared; don't edit, copy what you need).

use std::sync::Arc;

use obrrt_core::{Obj, ObjId, St, bhv_addr};

use crate::objective::{Key, Tree};

/// A state in `level`/`area` with `keys`, Mario at `pos` walking at `fv`, no objects.
pub fn st(pos: [f32; 3], fv: f32, level: u16, area: u8, keys: u8) -> St {
    St {
        frame: 0, pos, vel: [0.0; 3], forward_vel: fv, yaw: 0, pitch: 0, angle_vel: [0; 3], water_level: -11000,
        cam_yaw: 0, action: 0x0400_0440, mario_flags: 0, holding: false, held_slot: None, red_coins: 0, level, area,
        act: 0, course: 0, stars: 0, keys, coins: 0, lives: 4, health: 0x880, play_mode: 0, transition: 0, dialog: -1,
        dialog_choice: false, menu_line: 0, menu_mode: -1, act_select: false, act_init: 0, act_cursor: 0,
        dialog_state: 0, dialog_last_page: false, menu_timer: 0, floor_y: 0.0, ceil_y: 20000.0, floor_type: 0,
        floor_normal: [0.0, 1.0, 0.0], floor_obj: None, floor_obj_vy: None, floor_obj_y: None, action_timer: 0, cap_timer: 0,
        wall_normal: None, course_stars: [0; 25], save_flags: 0, objects: None,
    }
}

/// An object with behaviour `bhv` (e.g. "bhvBowser") in pool slot `slot` at `pos`; everything else zero.
/// Panics if the behaviour name is unknown.
pub fn obj(bhv: &str, slot: u16, pos: [f32; 3]) -> Obj {
    let b = bhv_addr(bhv).unwrap_or_else(|| panic!("unknown behaviour {bhv}"));
    Obj {
        id: ObjId { bhv: b, home: [0; 3] }, slot, bhv: b, pos, home: pos, interact_type: 0, interact_status: 0, action: 0,
        health: 0, value: 0, loot_coins: 0, beh_params: 0, held_state: 0, intangible_timer: 0, yaw: 0, raw_1ac: [0; 4],
    }
}

/// `st` with these objects loaded.
pub fn with_objects(mut s: St, objects: Vec<Obj>) -> St {
    s.objects = Some(Arc::new(objects));
    s
}

/// The key of `s` under the schema `json`, as (objective id, field, value) of the objective with `id` only.
pub fn key_of(json: &str, id: u16, s: &St) -> Vec<(u8, i32)> {
    let k: Key = Tree::load(json).unwrap().key(s);
    k.iter().filter(|e| e.0 == id).map(|e| (e.1, e.2)).collect()
}
