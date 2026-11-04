//! Reading [`St`] and [`Obj`] out of the game's linear memory, at the
//! addresses and offsets in [`crate::layout`].

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Result, bail};
use sm64_binds::SM64Game;

use crate::layout::*;
use crate::{Obj, St};

struct Mem<'a>(&'a [u8]);

impl Mem<'_> {
    fn bytes<const N: usize>(&self, addr: usize) -> [u8; N] {
        self.0[addr..addr + N].try_into().unwrap()
    }
    fn u32(&self, addr: usize) -> u32 {
        u32::from_le_bytes(self.bytes(addr))
    }
    fn i32(&self, addr: usize) -> i32 {
        i32::from_le_bytes(self.bytes(addr))
    }
    fn i16(&self, addr: usize) -> i16 {
        i16::from_le_bytes(self.bytes(addr))
    }
    fn i8(&self, addr: usize) -> i8 {
        self.0[addr] as i8
    }
    fn f32(&self, addr: usize) -> f32 {
        f32::from_le_bytes(self.bytes(addr))
    }
    fn vec3(&self, addr: usize) -> [f32; 3] {
        [self.f32(addr), self.f32(addr + 4), self.f32(addr + 8)]
    }
    /// The `struct Surface *` stored at `addr`, if non-null and in memory.
    fn surface(&self, addr: usize) -> Option<usize> {
        let s = self.u32(addr) as usize;
        (s != 0 && s + SURFACE_SIZE <= self.0.len()).then_some(s)
    }
    /// The `struct Object *` stored at `addr`, if it points at a slot of
    /// the object pool.
    fn pool_obj(&self, addr: usize) -> Option<usize> {
        let o = self.u32(addr) as usize;
        (o >= POOL && o < POOL + OBJECT_SIZE * OBJECT_POOL_CAPACITY as usize && (o - POOL) % OBJECT_SIZE == 0).then_some(o)
    }
}

/// The behaviour script at `addr`, by name.
pub fn bhv_name(addr: u32) -> Option<&'static str> {
    BEHAVIOURS.binary_search_by_key(&addr, |&(a, _)| a).ok().map(|i| BEHAVIOURS[i].1)
}

/// The address of the behaviour script called `name`.
pub fn bhv_addr(name: &str) -> Option<u32> {
    static BY_NAME: OnceLock<HashMap<&'static str, u32>> = OnceLock::new();
    BY_NAME
        .get_or_init(|| BEHAVIOURS.iter().map(|&(a, n)| (n, a)).collect())
        .get(name)
        .copied()
}

const MARIO: usize = ADDR_gMarioStates as usize;
const POOL: usize = ADDR_gObjectPool as usize;

/// Fails unless the layout matches the game: Mario's object must be in the
/// object pool and run `bhvMario`. Call once per game build before searching.
pub fn check_layout(game: &SM64Game) -> Result<()> {
    game.with_memory(|m| {
        let m = Mem(m);
        let obj = m.u32(MARIO + MARIO_marioObj) as usize;
        let pool_end = POOL + OBJECT_SIZE * OBJECT_POOL_CAPACITY as usize;
        if obj < POOL || obj >= pool_end || (obj - POOL) % OBJECT_SIZE != 0 {
            bail!("gMarioState->marioObj ({obj:#x}) is not in gObjectPool: core/src/layout.rs doesn't match this game build (rerun tools/gen_layout.py)");
        }
        let bhv = m.u32(obj + OBJ_behavior);
        if bhv_addr("bhvMario") != Some(bhv) {
            bail!("Mario's object runs {bhv:#x}, not bhvMario: core/src/layout.rs doesn't match this game build");
        }
        let pos = m.vec3(MARIO + MARIO_pos);
        let opos = m.vec3(obj + OBJ_oPosX);
        if crate::dist(pos, opos) > 1.0 {
            bail!("Mario's position {pos:?} and his object's {opos:?} differ: object field offsets are wrong");
        }
        Ok(())
    })
}

/// Mario and progress after `frame` frames; with `objects`, also every active
/// object in the pool.
pub fn read_state(game: &SM64Game, frame: u32, objects: bool) -> St {
    game.with_memory(|m| {
        let m = Mem(m);
        let save_file = (m.i16(ADDR_gCurrSaveFileNum as usize).max(1) - 1) as usize;
        let flags = m.u32(ADDR_gSaveBuffer as usize + save_file * SAVE_FILE_STRIDE + SAVE_FILE_FLAGS);
        let keys = (flags & (SAVE_FLAG_HAVE_KEY_1 | SAVE_FLAG_UNLOCKED_BASEMENT_DOOR) != 0) as u8
            + (flags & (SAVE_FLAG_HAVE_KEY_2 | SAVE_FLAG_UNLOCKED_UPSTAIRS_DOOR) != 0) as u8;
        let floor = m.surface(MARIO + MARIO_floor);
        let floor_obj = floor.and_then(|f| m.pool_obj(f + SURFACE_object));
        let st = St {
            frame,
            pos: m.vec3(MARIO + MARIO_pos),
            vel: m.vec3(MARIO + MARIO_vel),
            forward_vel: m.f32(MARIO + MARIO_forwardVel),
            yaw: m.i16(MARIO + MARIO_faceAngle + 2),
            pitch: m.i16(MARIO + MARIO_faceAngle),
            angle_vel: [0, 2, 4].map(|k| m.i16(MARIO + MARIO_angleVel + k)),
            water_level: m.i16(MARIO + MARIO_waterLevel),
            cam_yaw: m.i16(ADDR_gLakituState as usize + LAKITU_yaw),
            action: m.u32(MARIO + MARIO_action),
            mario_flags: m.u32(MARIO + MARIO_flags),
            holding: m.u32(MARIO + MARIO_heldObj) != 0,
            held_slot: {
                let h = m.u32(MARIO + MARIO_heldObj) as usize;
                (h >= POOL && h < POOL + OBJECT_SIZE * OBJECT_POOL_CAPACITY as usize).then(|| ((h - POOL) / OBJECT_SIZE) as u16)
            },
            red_coins: m.i8(ADDR_gRedCoinsCollected as usize) as i16,
            level: m.i16(ADDR_gCurrLevelNum as usize) as u16,
            area: m.i16(ADDR_gCurrAreaIndex as usize) as u8,
            act: m.i16(ADDR_gCurrActNum as usize) as u8,
            course: m.i16(ADDR_gCurrCourseNum as usize),
            stars: m.i16(MARIO + MARIO_numStars),
            keys,
            coins: m.i16(MARIO + MARIO_numCoins),
            lives: m.i8(MARIO + MARIO_numLives),
            health: m.i16(MARIO + MARIO_health),
            play_mode: m.i16(ADDR_sCurrPlayMode as usize),
            transition: m.i16(ADDR_sTransitionTimer as usize),
            dialog: m.i16(ADDR_gDialogID as usize),
            dialog_choice: m.i8(ADDR_gLastDialogResponse as usize) == 1,
            menu_line: m.i8(ADDR_gDialogLineNum as usize),
            menu_mode: m.i16(ADDR_gMenuMode as usize),
            act_select: m.i8(ADDR_sLoadedActNum as usize) == 0 && m.i32(ADDR_sActSelectorMenuTimer as usize) > 0,
            act_init: m.i8(ADDR_sInitSelectedActNum as usize) as u8,
            act_cursor: m.i8(ADDR_sSelectableStarIndex as usize),
            dialog_state: m.i8(ADDR_gDialogBoxState as usize),
            dialog_last_page: m.i16(ADDR_gLastDialogPageStrPos as usize) == -1,
            menu_timer: if m.i16(ADDR_gMenuMode as usize) == MENU_MODE_RENDER_COURSE_COMPLETE_SCREEN as i16 {
                m.i32(ADDR_gCourseDoneMenuTimer as usize)
            } else if m.i8(ADDR_sLoadedActNum as usize) == 0 && m.i32(ADDR_sActSelectorMenuTimer as usize) > 0 {
                m.i32(ADDR_sActSelectorMenuTimer as usize)
            } else {
                0
            },
            floor_y: m.f32(MARIO + MARIO_floorHeight),
            ceil_y: m.f32(MARIO + MARIO_ceilHeight),
            floor_type: floor.map_or(-1, |f| m.i16(f + SURFACE_type)),
            floor_normal: floor.map_or([0.0; 3], |f| m.vec3(f + SURFACE_normal)),
            floor_obj: floor_obj.map(|o| ((o - POOL) / OBJECT_SIZE) as u16),
            floor_obj_vy: floor_obj.map(|o| m.f32(o + OBJ_oVelY)),
            floor_obj_y: floor_obj.map(|o| m.f32(o + OBJ_oPosY)),
            action_timer: m.i16(MARIO + MARIO_actionTimer) as u16,
            wall_normal: m.surface(MARIO + MARIO_wall).map(|w| m.vec3(w + SURFACE_normal)),
            course_stars: m.bytes(ADDR_gSaveBuffer as usize + save_file * SAVE_FILE_STRIDE + SAVE_FILE_COURSE_STARS),
            save_flags: flags,
            objects: objects.then(|| Arc::new(read_objects(&m))),
        };
        if objects {
            // Objectives' static-floor probes (off unless one asked).
            crate::geo::observe_floors(m.0, &st);
            // OBRRT_BLJ_SCAN_OBJECTS: the object collision near Mario.
            if crate::dyngeo::on() {
                crate::dyngeo::attach(&st, crate::dyngeo::read(m.0, st.pos));
            }
        }
        st
    })
}

fn read_objects(m: &Mem) -> Vec<Obj> {
    let mario_obj = m.u32(MARIO + MARIO_marioObj) as usize;
    (0..OBJECT_POOL_CAPACITY as usize)
        .filter_map(|i| {
            let o = POOL + i * OBJECT_SIZE;
            let active = m.i16(o + OBJ_activeFlags) as u16 as u32;
            if active & ACTIVE_FLAG_ACTIVE == 0 || o == mario_obj {
                return None;
            }
            let bhv = m.u32(o + OBJ_behavior);
            let home = m.vec3(o + OBJ_oHomeX);
            // oHome is only set by SET_HOME in a behaviour script; objects
            // without it have home 0,0,0, and every one of a behaviour would
            // share an id. Those are told apart by pool slot (stable while the
            // level stays loaded; objects spawn in the same order).
            let rounded = home.map(|v| v.round() as i32);
            let id_home = if rounded == [0, 0, 0] { [i32::MIN, i as i32, 0] } else { rounded };
            Some(Obj {
                id: crate::ObjId { bhv, home: id_home },
                slot: i as u16,
                bhv,
                pos: m.vec3(o + OBJ_oPosX),
                home,
                interact_type: m.u32(o + OBJ_oInteractType),
                interact_status: m.u32(o + OBJ_oInteractStatus),
                action: m.i32(o + OBJ_oAction),
                health: m.i32(o + OBJ_oHealth),
                value: m.i32(o + OBJ_oDamageOrCoinValue),
                loot_coins: m.i32(o + OBJ_oNumLootCoins),
                beh_params: m.u32(o + OBJ_oBehParams),
                held_state: m.u32(o + OBJ_oHeldState),
                intangible_timer: m.i32(o + OBJ_oIntangibleTimer),
                yaw: m.i32(o + OBJ_oFaceAngleYaw),
                raw_1ac: [0, 2, 4, 6].map(|k| m.i16(o + 0x1AC + k)),
            })
        })
        .collect()
}
