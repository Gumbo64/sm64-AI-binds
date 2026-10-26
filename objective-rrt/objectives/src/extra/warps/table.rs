//! Every INTERACT_WARP object the levels place (bhvWarp holes, bhvWarpPipe
//! pipes, bhvFadingWarp teleporters) and where it leads.
//!
//! Generated from sm64-port/levels/*/script.c: each OBJECT with one of these
//! behaviours, its area (from the enclosing AREA, following JUMP_LINKs), its
//! position, and the WARP_NODE of that area whose id is the object's 2nd
//! behaviour byte (level_update.c: `sSourceWarpNodeId = (oBehParams &
//! 0x00FF0000) >> 16` for WARP_OP_WARP_OBJECT / WARP_OP_TELEPORT). The 1st
//! behaviour byte is the hitbox radius / 10 (warp.inc.c, 0 = 50 or 85).

use obrrt_core::layout::*;

/// Level ids missing from the generated layout (levels/level_defines.h, in
/// order from LEVEL_NONE = 0).
pub const LEVEL_BOWSER_1: u32 = 30;
pub const LEVEL_BOWSER_2: u32 = 33;
pub const LEVEL_BOWSER_3: u32 = 34;

/// What else (besides being in the zone) entering needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Req {
    None,
    /// Total stars (the castle's star doors in front of it).
    Stars(i16),
    /// Total stars and a given star (course, index).
    StarsAndStar(i16, i16, u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// A hole / light / hidden warp to a secret level of the castle.
    Secret,
    /// The castle's warps into a Bowser course, and the course pipes to the
    /// Bowser fights.
    Bowser,
    /// A pipe or hole between two areas of one course.
    Area,
    /// A teleporter pad within one area.
    Teleport,
}

pub struct Warp {
    pub name: &'static str,
    pub level: u32,
    pub area: u8,
    pub bhv: &'static str,
    pub pos: [i32; 3],
    pub dest: (u32, u8),
    pub class: Class,
    pub req: Req,
}

const fn w(
    name: &'static str,
    level: u32,
    area: u8,
    bhv: &'static str,
    pos: [i32; 3],
    dest: (u32, u8),
    class: Class,
    req: Req,
) -> Warp {
    Warp { name, level, area, bhv, pos, dest, class, req }
}

use Class::*;
use Req::*;

pub const WARPS: &[Warp] = &[
    // Bowser courses' end pipes / hole to the fights.
    w("bowser_1", LEVEL_BITDW, 1, "bhvWarpPipe", [6816, 2860, -7], (LEVEL_BOWSER_1, 1), Bowser, None),
    w("bowser_2", LEVEL_BITFS, 1, "bhvWarp", [6735, 3681, 99], (LEVEL_BOWSER_2, 1), Bowser, None),
    w("bowser_3", LEVEL_BITS, 1, "bhvWarpPipe", [351, 6652, -6030], (LEVEL_BOWSER_3, 1), Bowser, None),
    // The castle's hidden warps. Star requirements are the star doors in
    // front of them (interaction.c get_door_save_file_flag: the 1-star door
    // to the PSS hallway, the 3-star JRB door, 8 / 30 / 50 / 70-star doors;
    // the 70 also stops the endless stairs, level_update.c check_instant_warp).
    w("pss", LEVEL_CASTLE, 1, "bhvWarp", [2013, 768, -2014], (LEVEL_PSS, 1), Secret, Stars(1)),
    // Under the trap floor (bhvCastleFloorTrap) behind the 8-star door.
    w("bitdw", LEVEL_CASTLE, 1, "bhvWarp", [-5513, 512, -4324], (LEVEL_BITDW, 1), Bowser, Stars(8)),
    // The aquarium wall in the JRB room.
    w("sa", LEVEL_CASTLE, 1, "bhvWarp", [1963, 819, 1280], (LEVEL_SA, 1), Secret, Stars(3)),
    // Third floor (behind the 50-star door), the light by the window.
    w("wmotr", LEVEL_CASTLE, 2, "bhvWarp", [3002, 2816, 5886], (LEVEL_WMOTR, 1), Secret, Stars(50)),
    // Top of the endless stairs (70-star door).
    w("bits", LEVEL_CASTLE, 2, "bhvWarp", [-230, 4813, -3352], (LEVEL_BITS, 1), Bowser, Stars(70)),
    // Behind the DDD painting, which only moves back once Bowser's sub is
    // gone (DDD star 1); the room is behind the 30-star door.
    w("bitfs", LEVEL_CASTLE, 3, "bhvWarp", [4147, -1280, 1997], (LEVEL_BITFS, 1), Bowser, StarsAndStar(30, 9, 0)),
    // The moat's bottom; reachable only once the moat is drained
    // (SAVE_FLAG_MOAT_DRAINED, moat_grill.inc.c).
    w("vcutm", LEVEL_CASTLE_GROUNDS, 1, "bhvWarp", [-3379, -815, -2025], (LEVEL_VCUTM, 1), Secret, None),
    // HMC's pool down to the Cavern of the Metal Cap.
    w("cotmc", LEVEL_HMC, 1, "bhvWarp", [3351, -4690, 4773], (LEVEL_COTMC, 1), Secret, None),
    // Area changes inside courses.
    w("ccm_slide", LEVEL_CCM, 1, "bhvWarp", [-181, 2918, -1486], (LEVEL_CCM, 2), Area, None),
    w("lll_volcano", LEVEL_LLL, 1, "bhvWarp", [0, 105, 0], (LEVEL_LLL, 2), Area, None),
    w("sl_igloo", LEVEL_SL, 1, "bhvWarp", [569, 2150, 1336], (LEVEL_SL, 2), Area, None),
    w("sl_igloo_exit", LEVEL_SL, 2, "bhvWarp", [0, 0, 3277], (LEVEL_SL, 1), Area, None),
    w("ssl_pyramid", LEVEL_SSL, 1, "bhvWarp", [-2048, 0, 56], (LEVEL_SSL, 2), Area, None),
    w("ssl_pyramid_top", LEVEL_SSL, 1, "bhvWarp", [-2048, 768, -1024], (LEVEL_SSL, 2), Area, None),
    w("thi_wiggler", LEVEL_THI, 1, "bhvWarp", [410, -512, 717], (LEVEL_THI, 3), Area, None),
    w("thi_wiggler_top", LEVEL_THI, 1, "bhvWarp", [0, 3170, -1570], (LEVEL_THI, 3), Area, None),
    w("thi_wiggler_exit", LEVEL_THI, 3, "bhvWarp", [512, 1024, 2355], (LEVEL_THI, 1), Area, None),
    w("thi_pipe_a", LEVEL_THI, 1, "bhvWarpPipe", [6656, -1536, -5632], (LEVEL_THI, 2), Area, None),
    w("thi_pipe_b", LEVEL_THI, 1, "bhvWarpPipe", [-5888, -2048, -5888], (LEVEL_THI, 2), Area, None),
    w("thi_pipe_c", LEVEL_THI, 1, "bhvWarpPipe", [-3072, 512, -3840], (LEVEL_THI, 2), Area, None),
    w("thi_pipe_a_back", LEVEL_THI, 2, "bhvWarpPipe", [1997, -461, -1690], (LEVEL_THI, 1), Area, None),
    w("thi_pipe_b_back", LEVEL_THI, 2, "bhvWarpPipe", [-1766, -614, -1766], (LEVEL_THI, 1), Area, None),
    w("thi_pipe_c_back", LEVEL_THI, 2, "bhvWarpPipe", [-922, 154, -1152], (LEVEL_THI, 1), Area, None),
    // Teleporter pads (bhvFadingWarp), in pairs within an area.
    w("bob_tp_a", LEVEL_BOB, 1, "bhvFadingWarp", [583, 2683, -5387], (LEVEL_BOB, 1), Teleport, None),
    w("bob_tp_b", LEVEL_BOB, 1, "bhvFadingWarp", [1680, 3835, -5523], (LEVEL_BOB, 1), Teleport, None),
    w("bob_tp_c", LEVEL_BOB, 1, "bhvFadingWarp", [-6612, 1024, -3351], (LEVEL_BOB, 1), Teleport, None),
    w("bob_tp_d", LEVEL_BOB, 1, "bhvFadingWarp", [1980, 768, 6618], (LEVEL_BOB, 1), Teleport, None),
    w("ccm_tp_a", LEVEL_CCM, 1, "bhvFadingWarp", [-1847, 2815, -321], (LEVEL_CCM, 1), Teleport, None),
    w("ccm_tp_b", LEVEL_CCM, 1, "bhvFadingWarp", [3349, -4694, -183], (LEVEL_CCM, 1), Teleport, None),
    w("lll_tp_a", LEVEL_LLL, 1, "bhvFadingWarp", [-3200, 11, 3456], (LEVEL_LLL, 1), Teleport, None),
    w("lll_tp_b", LEVEL_LLL, 1, "bhvFadingWarp", [-5888, 154, 6656], (LEVEL_LLL, 1), Teleport, None),
    w("rr_tp_a", LEVEL_RR, 1, "bhvFadingWarp", [-7092, 2364, -63], (LEVEL_RR, 1), Teleport, None),
    w("rr_tp_b", LEVEL_RR, 1, "bhvFadingWarp", [-4213, 3379, -2815], (LEVEL_RR, 1), Teleport, None),
    w("sl_tp_a", LEVEL_SL, 1, "bhvFadingWarp", [5468, 1056, -5400], (LEVEL_SL, 1), Teleport, None),
    w("sl_tp_b", LEVEL_SL, 1, "bhvFadingWarp", [-3698, 1024, -1237], (LEVEL_SL, 1), Teleport, None),
    w("ssl_tp_a", LEVEL_SSL, 1, "bhvFadingWarp", [6930, 0, -4871], (LEVEL_SSL, 1), Teleport, None),
    w("ssl_tp_b", LEVEL_SSL, 1, "bhvFadingWarp", [-5943, 0, -4903], (LEVEL_SSL, 1), Teleport, None),
    w("ssl_tp_c", LEVEL_SSL, 2, "bhvFadingWarp", [3070, 1280, 2900], (LEVEL_SSL, 2), Teleport, None),
    w("ssl_tp_d", LEVEL_SSL, 2, "bhvFadingWarp", [2546, 1150, -2647], (LEVEL_SSL, 2), Teleport, None),
    w("thi_tp_a", LEVEL_THI, 2, "bhvFadingWarp", [280, -767, -4180], (LEVEL_THI, 2), Teleport, None),
    w("thi_tp_b", LEVEL_THI, 2, "bhvFadingWarp", [-1638, 0, -1988], (LEVEL_THI, 2), Teleport, None),
    w("ttm_tp_a", LEVEL_TTM, 1, "bhvFadingWarp", [2267, -3006, -3788], (LEVEL_TTM, 1), Teleport, None),
    w("ttm_tp_b", LEVEL_TTM, 1, "bhvFadingWarp", [-557, -3448, -4146], (LEVEL_TTM, 1), Teleport, None),
    w("wdw_tp_a", LEVEL_WDW, 1, "bhvFadingWarp", [818, 0, 3634], (LEVEL_WDW, 1), Teleport, None),
    w("wdw_tp_b", LEVEL_WDW, 1, "bhvFadingWarp", [-2865, 3328, 3065], (LEVEL_WDW, 1), Teleport, None),
    w("wf_tp_a", LEVEL_WF, 1, "bhvFadingWarp", [-2925, 2560, -947], (LEVEL_WF, 1), Teleport, None),
    w("wf_tp_b", LEVEL_WF, 1, "bhvFadingWarp", [2548, 1075, -3962], (LEVEL_WF, 1), Teleport, None),
];

/// The table entry for a warp object in `(level, area)`, matched by
/// behaviour and spawn position (bhvWarp / bhvFadingWarp keep their script
/// position; pipes too).
pub fn find(level: u16, area: u8, bhv: &str, home: [f32; 3]) -> Option<&'static Warp> {
    WARPS.iter().find(|w| {
        w.level == level as u32
            && w.area == area
            && w.bhv == bhv
            && (0..3).all(|i| (w.pos[i] as f32 - home[i]).abs() < 40.0)
    })
}

/// Cannon targets worth a dedicated aimed shot: (level, area, name, point,
/// the star it leads to as (course, index)). Points are the star / star box
/// placements in the level data.
pub const CANNON_TARGETS: &[(u32, u8, &str, [f32; 3], (i16, u8))] = &[
    // levels/bob/areas/1/macro.inc.c: macro_box_star_3 on the floating island
    // ("Shoot to the Island in the Sky", star index 2).
    (LEVEL_BOB, 1, "bob_island", [5540.0, 3350.0, 1200.0], (1, 2)),
    // levels/wf/script.c: bhvStar 0x05 behind the wall ("Blast Away the
    // Wall", star index 5).
    (LEVEL_WF, 1, "wf_wall", [590.0, 2450.0, 2650.0], (2, 5)),
    // levels/rr/areas/1/macro.inc.c: macro_box_star_6 (star index 5).
    (LEVEL_RR, 1, "rr_box_6", [5000.0, 4100.0, 4440.0], (15, 5)),
];

/// The floor that warps to the Tower of the Wing Cap when Mario looks up at
/// the sun from it (SURFACE_LOOK_UP_WARP, levels/castle_inside/areas/1/
/// collision.inc.c: an octagon x -1279..-767, z 461..973 at y -50).
pub const SUN_SPOT: [f32; 3] = [-1023.0, -50.0, 717.0];
pub const SUN_SPOT_RADIUS: f32 = 200.0;
