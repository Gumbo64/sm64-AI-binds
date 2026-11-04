#!/usr/bin/env python3
"""Generates core/src/layout.rs: where things live in the headless game's memory.

    python3 tools/gen_layout.py [<sm64-port dir>]

Struct offsets and constants come from compiling a small C program against the
decomp headers with emcc (wasm32, the same target as the game) and running it
with node, so nothing is transcribed by hand. Global addresses and the
behaviour table come from the build's linker map (build/us_web_headless/sm64.map),
which only matches the WASM it was linked with: rerun this after rebuilding.
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "core", "src", "layout.rs")
PORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "sm64-binds", "sm64-port")
BUILD = os.path.join(PORT, "build", "us_web_headless")

GLOBALS = [
    "gMarioStates", "gObjectPool", "gCurrLevelNum", "gCurrAreaIndex", "gCurrActNum",
    "gCurrCourseNum", "gCurrSaveFileNum", "gSaveBuffer", "sCurrPlayMode", "gMenuMode",
    "gDialogID", "sTransitionTimer", "gHudDisplay", "gLakituState", "gRedCoinsCollected",
    "gDialogLineNum", "gLastDialogResponse", "gDialogBoxState", "gLastDialogPageStrPos", "gCourseDoneMenuTimer",
    "sLoadedActNum", "sActSelectorMenuTimer", "sInitSelectedActNum", "sSelectableStarIndex", "sObtainedStars",
    # Static floors for the walkable-distance graph (core/src/geo.rs).
    "gStaticSurfacePartition",
    # Water boxes for objectives' water-warp probes (core/src/geo.rs).
    "gEnvironmentRegions",
    # State features for the learned backbone (backbone/, run/src/dataset.rs).
    "gRandomSeed16", "gGlobalTimer", "gCamera", "sSelectionFlags", "gCameraMovementFlags",
    "sCSideButtonYaw", "gDynamicSurfacePartition",
]

OBJECT_FIELDS = [
    "oPosX", "oPosY", "oPosZ", "oVelY", "oForwardVel", "oHomeX", "oHomeY", "oHomeZ",
    "oInteractType", "oInteractStatus", "oHeldState", "oAction", "oTimer",
    "oDamageOrCoinValue", "oHealth", "oBehParams", "oFaceAngleYaw", "oIntangibleTimer", "oNumLootCoins",
    # Backbone features.
    "oVelX", "oVelZ", "oAngleVelYaw", "oMoveAngleYaw", "oSubAction",
    "hitboxRadius", "hitboxHeight", "hurtboxRadius", "hurtboxHeight",
]

MARIO_FIELDS = [
    "flags", "action", "faceAngle", "pos", "vel", "forwardVel", "heldObj", "marioObj",
    "numCoins", "numStars", "numKeys", "numLives", "health", "capTimer", "floorHeight",
    "angleVel", "waterLevel",
    # Read-only collision context (St::floor_y etc.).
    "ceilHeight", "floor", "ceil", "wall", "actionTimer",
    # Backbone features.
    "actionState", "actionArg", "intendedMag", "intendedYaw", "invincTimer", "framesSinceA",
    "framesSinceB", "wallKickTimer", "doubleJumpTimer", "slideYaw", "twirlYaw", "slideVelX",
    "slideVelZ", "hurtCounter", "healCounter", "squishTimer", "peakHeight", "quicksandDepth",
    "collidedObjInteractTypes", "prevAction",
]

SURFACE_FIELDS = ["type", "normal", "normal.y", "originOffset", "object"]

CONSTANTS = [
    "INTERACT_HOOT", "INTERACT_GRABBABLE", "INTERACT_DOOR", "INTERACT_DAMAGE", "INTERACT_COIN",
    "INTERACT_CAP", "INTERACT_POLE", "INTERACT_KOOPA", "INTERACT_UNKNOWN_08", "INTERACT_BREAKABLE",
    "INTERACT_STRONG_WIND", "INTERACT_WARP_DOOR", "INTERACT_STAR_OR_KEY", "INTERACT_WARP",
    "INTERACT_CANNON_BASE", "INTERACT_BOUNCE_TOP", "INTERACT_WATER_RING", "INTERACT_BULLY",
    "INTERACT_FLAME", "INTERACT_KOOPA_SHELL", "INTERACT_BOUNCE_TOP2", "INTERACT_MR_BLIZZARD",
    "INTERACT_HIT_FROM_BELOW", "INTERACT_TEXT", "INTERACT_TORNADO", "INTERACT_WHIRLPOOL",
    "INTERACT_CLAM_OR_BUBBA", "INTERACT_BBH_ENTRANCE", "INTERACT_SNUFIT_BULLET", "INTERACT_SHOCK",
    "INTERACT_IGLOO_BARRIER",
    "ACT_FLAG_STATIONARY", "ACT_FLAG_MOVING", "ACT_FLAG_AIR", "ACT_FLAG_INTANGIBLE",
    "ACT_FLAG_SWIMMING", "ACT_FLAG_HANGING", "ACT_FLAG_IDLE", "ACT_FLAG_PAUSE_EXIT",
    "ACT_FLAG_ON_POLE", "ACT_FLAG_RIDING_SHELL", "ACT_FLAG_INVULNERABLE",
    "ACT_GROUP_MASK", "ACT_GROUP_CUTSCENE", "ACT_GROUP_AUTOMATIC",
    "ACT_PULLING_DOOR", "ACT_PUSHING_DOOR", "ACT_UNLOCKING_KEY_DOOR", "ACT_UNLOCKING_STAR_DOOR",
    "ACT_ENTERING_STAR_DOOR", "ACT_WARP_DOOR_SPAWN", "ACT_STAR_DANCE_EXIT", "ACT_STAR_DANCE_NO_EXIT",
    "ACT_STAR_DANCE_WATER", "ACT_WAITING_FOR_DIALOG", "ACT_READING_NPC_DIALOG", "ACT_READING_AUTOMATIC_DIALOG", "ACT_FALL_AFTER_STAR_GRAB", "ACT_READING_SIGN", "ACT_IDLE",
    # Control modes the steering controller tells apart (search/src/steer.rs).
    "ACT_FLAG_METAL_WATER", "ACT_HOLDING_POLE", "ACT_GRAB_POLE_SLOW", "ACT_GRAB_POLE_FAST",
    "ACT_CLIMBING_POLE", "ACT_TOP_OF_POLE_TRANSITION", "ACT_TOP_OF_POLE", "ACT_START_HANGING",
    "ACT_HANGING", "ACT_HANG_MOVING", "ACT_IN_CANNON", "ACT_SHOT_FROM_CANNON", "ACT_FLYING",
    "ACT_HOLDING_BOWSER", "ACT_RIDING_HOOT", "ACT_WATER_PLUNGE", "ACT_WALL_KICK_AIR", "ACT_TOP_OF_POLE_JUMP",
    "SAVE_FLAG_HAVE_KEY_1", "SAVE_FLAG_HAVE_KEY_2", "SAVE_FLAG_UNLOCKED_BASEMENT_DOOR",
    "SAVE_FLAG_UNLOCKED_UPSTAIRS_DOOR",
    "MENU_MODE_RENDER_PAUSE_SCREEN", "MENU_MODE_RENDER_COURSE_COMPLETE_SCREEN",
    "MENU_OPT_SAVE_AND_CONTINUE", "MENU_OPT_SAVE_AND_QUIT", "MENU_OPT_CONTINUE_DONT_SAVE",
    "SPATIAL_PARTITION_FLOORS", "SPATIAL_PARTITION_CEILS", "SPATIAL_PARTITION_WALLS",
    "NUM_CELLS", "CELL_SIZE", "LEVEL_BOUNDARY_MAX", "SURFACE_INTANGIBLE", "SURFACE_CAMERA_BOUNDARY",
    "INT_STATUS_INTERACTED", "ACTIVE_FLAG_ACTIVE", "ACTIVE_FLAG_DEACTIVATED", "OBJECT_POOL_CAPACITY",
    "LEVEL_CASTLE", "LEVEL_CASTLE_GROUNDS", "LEVEL_CASTLE_COURTYARD", "LEVEL_BOB", "LEVEL_WF",
    "LEVEL_JRB", "LEVEL_CCM", "LEVEL_BBH", "LEVEL_HMC", "LEVEL_LLL", "LEVEL_SSL", "LEVEL_DDD",
    "LEVEL_SL", "LEVEL_WDW", "LEVEL_TTM", "LEVEL_THI", "LEVEL_TTC", "LEVEL_RR", "LEVEL_PSS",
    "LEVEL_SA", "LEVEL_WMOTR", "LEVEL_TOTWC", "LEVEL_COTMC", "LEVEL_VCUTM", "LEVEL_BITDW",
    "LEVEL_BITFS", "LEVEL_BITS", "LEVEL_BOWSER_1", "LEVEL_BOWSER_2", "LEVEL_BOWSER_3",
]

C_SOURCE = r"""
#include <stdio.h>
#include <stddef.h>
#include "types.h"
#include "sm64.h"
#include "level_table.h"
#include "object_fields.h"
#include "game/interaction.h"
#include "game/object_list_processor.h"
#include "game/save_file.h"
#include "game/camera.h"
#include "game/ingame_menu.h"
#include "engine/surface_collision.h"
#include "engine/surface_load.h"
#include "surface_terrains.h"
#define OBJ(name) printf("pub const OBJ_%s: usize = %zu;\n", #name, offsetof(struct Object, name));
#define MARIO(name) printf("pub const MARIO_%s: usize = %zu;\n", #name, offsetof(struct MarioState, name));
#define SURFACE(name, id) printf("pub const SURFACE_%s: usize = %zu;\n", #id, offsetof(struct Surface, name));
#define K(name) printf("pub const %s: u32 = %lu;\n", #name, (unsigned long) (name));
int main(void) {
    printf("pub const OBJECT_SIZE: usize = %zu;\n", sizeof(struct Object));
    printf("pub const OBJ_activeFlags: usize = %zu;\n", offsetof(struct Object, activeFlags));
    printf("pub const OBJ_behavior: usize = %zu;\n", offsetof(struct Object, behavior));
    printf("pub const MARIO_STATE_SIZE: usize = %zu;\n", sizeof(struct MarioState));
    printf("pub const SAVE_FILE_STRIDE: usize = %zu;\n", sizeof(((struct SaveBuffer *) 0)->files[0]));
    printf("pub const SAVE_FILE_FLAGS: usize = %zu;\n", offsetof(struct SaveBuffer, files[0][0].flags));
    printf("pub const SAVE_FILE_COURSE_STARS: usize = %zu;\n", offsetof(struct SaveBuffer, files[0][0].courseStars));
    printf("pub const SURFACE_SIZE: usize = %zu;\n", sizeof(struct Surface));
    printf("pub const SURFACE_NODE_SIZE: usize = %zu;\n", sizeof(struct SurfaceNode));
    printf("pub const SPATIAL_PARTITION_CELL_SIZE: usize = %zu;\n", sizeof(SpatialPartitionCell));
    printf("pub const LAKITU_yaw: usize = %zu;\n", offsetof(struct LakituState, yaw));
    printf("pub const CAMERA_mode: usize = %zu;\n", offsetof(struct Camera, mode));
    printf("pub const CAMERA_cutscene: usize = %zu;\n", offsetof(struct Camera, cutscene));
    @FIELDS@
    return 0;
}
"""


def run_c():
    lines = [f"OBJ({f})" for f in OBJECT_FIELDS] + [f"MARIO({f})" for f in MARIO_FIELDS] + \
        [f"SURFACE({f}, {f.replace('.', '_')})" for f in SURFACE_FIELDS] + [f"K({c})" for c in CONSTANTS]
    src = C_SOURCE.replace("@FIELDS@", "\n    ".join(lines))
    with tempfile.TemporaryDirectory() as tmp:
        c = os.path.join(tmp, "layout.c")
        js = os.path.join(tmp, "layout.js")
        with open(c, "w") as f:
            f.write(src)
        includes = [PORT, os.path.join(PORT, "include"), os.path.join(PORT, "src"), BUILD, os.path.join(BUILD, "include")]
        cmd = ["emcc", "-O0", "-DVERSION_US", "-D_LANGUAGE_C", "-DNON_MATCHING", "-DAVOID_UB", "-DTARGET_WEB",
               *[f"-I{i}" for i in includes], c, "-o", js]
        subprocess.run(cmd, check=True)
        return subprocess.run(["node", js], check=True, capture_output=True, text=True).stdout


def read_map():
    symbol = re.compile(r"^\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+(\S+)$")
    addresses, behaviours = {}, []
    with open(os.path.join(BUILD, "sm64.map")) as f:
        for line in f:
            m = symbol.match(line.rstrip("\n"))
            if not m:
                continue
            addr, size, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            if name in GLOBALS:
                addresses[name] = addr
            elif name.startswith("bhv") and "(" not in name and size > 0:
                behaviours.append((addr, name))
    missing = [g for g in GLOBALS if g not in addresses]
    if missing:
        sys.exit(f"not in the map: {missing}")
    return addresses, sorted(set(behaviours))


def main():
    consts = run_c()
    addresses, behaviours = read_map()
    out = ["// Generated by tools/gen_layout.py from the decomp headers and the build's",
           "// linker map. Do not edit; rerun the script after rebuilding the game.",
           "#![allow(non_upper_case_globals, dead_code)]", ""]
    out += [f"pub const ADDR_{name}: u32 = {addr:#x};" for name, addr in sorted(addresses.items())]
    out.append("")
    out += consts.strip().splitlines()
    out += ["", "/// Every behaviour script's address, sorted by address.",
            "pub const BEHAVIOURS: &[(u32, &str)] = &["]
    out += [f"    ({addr:#x}, \"{name}\")," for addr, name in behaviours]
    out += ["];", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}: {len(addresses)} globals, {len(behaviours)} behaviours")


if __name__ == "__main__":
    main()
