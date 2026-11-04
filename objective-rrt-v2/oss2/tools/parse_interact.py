#!/usr/bin/env python3
"""Each behaviour's interaction type, from the decomp:

    tools/parse_interact.py [<sm64-port dir>] [<out.json>]   (default tools/analysis/interact.json)

Two sources: SET_INT(oInteractType, INTERACT_X) in the behaviour's own script
(data/behavior_data.c), and `oInteractType = INTERACT_X` (or `|=`) in the C
functions the script calls with CALL_NATIVE (src/game/behaviors/*.inc.c and
src/game/*.c). Writes {"bhvName": <u32 mask>, ...}; a behaviour that sets
several types gets them ORed. Used by examples/gen_star120.rs to give probe
objects the interaction type the game would.
"""

import json
import os
import re
import sys

PORT = sys.argv[1] if len(sys.argv) > 1 else "/workspaces/sm64-combined/sm64-binds/sm64-port"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "analysis", "interact.json")


def read(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read().replace("\r\n", "\n")


consts = {}
for m in re.finditer(r"#define\s+(INTERACT_\w+)\s+(?:/\*.*?\*/\s*)?\(?\s*(?:1\s*<<\s*(\d+)|(0x[0-9a-fA-F]+|\d+))", read(f"{PORT}/src/game/interaction.h")):
    consts[m.group(1)] = 1 << int(m.group(2)) if m.group(2) else int(m.group(3), 0)
if not consts:
    sys.exit("no INTERACT_ constants found in src/game/interaction.h")


def mask(expr):
    v = 0
    for name in re.findall(r"INTERACT_\w+", expr):
        if name in consts:
            v |= consts[name]
    return v


# Functions that set an interaction type: directly, through a hitbox struct
# (`/* interactType: */ INTERACT_X` passed to obj_set_hitbox), or by calling
# such a function (followed until nothing changes).
func_types = {}
hitboxes = {}
bodies = {}
srcs = []
for root in (f"{PORT}/src/game/behaviors", f"{PORT}/src/game"):
    for name in os.listdir(root):
        if name.endswith((".c", ".inc.c")):
            srcs.append(os.path.join(root, name))
for path in srcs:
    text = read(path)
    for h in re.finditer(r"struct ObjectHitbox\s+(\w+)\s*=\s*\{\s*(?:/\*[^*]*\*/\s*)?([^,]+),", text):
        hitboxes[h.group(1)] = mask(h.group(2))
for path in srcs:
    text = read(path)
    for m in re.finditer(r"^\w[\w\s\*]*?\b(\w+)\s*\([^;{)]*\)\s*\{", text, re.M):
        start = m.end()
        depth, i = 1, start
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[start:i]
        bodies[m.group(1)] = bodies.get(m.group(1), "") + body
        v = 0
        for h in re.findall(r"set_hitbox\(\s*\w+\s*,\s*&(\w+)\s*\)", body):
            v |= hitboxes.get(h, 0)
        for a in re.finditer(r"oInteractType\s*\|?=\s*([^;]+);", body):
            v |= mask(a.group(1))
        if v:
            func_types[m.group(1)] = func_types.get(m.group(1), 0) | v

changed = True
while changed:
    changed = False
    for fn, body in bodies.items():
        v = func_types.get(fn, 0)
        for callee in re.findall(r"\b(\w+)\s*\(", body):
            v |= func_types.get(callee, 0) if callee != fn else 0
        if v != func_types.get(fn, 0):
            func_types[fn] = v
            changed = True

out = {}
data = read(f"{PORT}/data/behavior_data.c")
for m in re.finditer(r"const BehaviorScript (bhv\w+)\[\]\s*=\s*\{(.*?)\n\};", data, re.S):
    name, body = m.group(1), m.group(2)
    v = 0
    for a in re.finditer(r"SET_INT\(oInteractType,\s*([^)]+)\)", body):
        v |= mask(a.group(1))
    for f in re.findall(r"CALL_NATIVE\((\w+)\)", body):
        v |= func_types.get(f, 0)
    if v:
        out[name] = v

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(dict(sorted(out.items())), f, indent=1)
print(f"{len(out)} behaviours with an interaction type ({len(func_types)} functions set one) -> {OUT}")
for b in ("bhvDoor", "bhvStarDoor", "bhvDoorWarp", "bhvSmallBully", "bhvBigBully", "bhvPoleGrabbing", "bhvGiantPole", "bhvKoopaShell", "bhvMantaRayWaterRing"):
    print(f"  {b}: {out.get(b)}")
