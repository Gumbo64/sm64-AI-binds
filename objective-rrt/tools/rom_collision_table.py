#!/usr/bin/env python3
"""Finds each area's collision data in the US ROM, for the viewer to read it
from the user's own ROM instead of shipping Nintendo's level data:

    tools/rom_collision_table.py [baserom.us.z64]

Writes sm64-viewer/frontend/src/tree/collisionTable.js: per (level, area), the
ROM offset of the MIO0 block holding it and the offset inside the
decompressed block (numbers only), plus the surface type names
(include/surface_terrains.h). Each area is found by matching the decomp's
collision (levels/*/areas/*/collision.inc.c: vertex count + first vertices),
or, where the decomp has version #ifdefs, by an unclaimed blob with the US
vertex count in the same level's block.
"""

import glob
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DECOMP = os.path.normpath(os.path.join(HERE, "../../../sm64-binds/sm64-port"))
OUT = os.path.normpath(os.path.join(HERE, "../../../sm64-viewer/frontend/src/tree/collisionTable.js"))
FORCE = {0x04, 0x0E, 0x24, 0x25, 0x27, 0x2C, 0x2D}  # surface_load.c surface_has_force


def mio0(data, off):
    size, comp, uncomp = struct.unpack(">III", data[off + 4 : off + 16])
    out, layout, cp, up = bytearray(), off + 16, off + comp, off + uncomp
    while len(out) < size:
        word = struct.unpack(">I", data[layout : layout + 4])[0]
        layout += 4
        for b in range(32):
            if len(out) >= size:
                break
            if word & (0x80000000 >> b):
                out.append(data[up])
                up += 1
            else:
                v = struct.unpack(">H", data[cp : cp + 2])[0]
                cp += 2
                for _ in range((v >> 12) + 3):
                    out.append(out[-((v & 0xFFF) + 1)])
    return bytes(out)


def valid(d, p):
    """(vertices, triangles) of a well-formed collision list at p, else None."""
    H = lambda i: struct.unpack(">H", d[i : i + 2])[0]
    try:
        if H(p) != 0x40:
            return None
        nv = H(p + 2)
        if not 0 < nv < 6000:
            return None
        q, ntri = p + 4 + 6 * nv, 0
        while (t := H(q)) != 0x41:
            n = H(q + 2)
            q += 4
            if not 0 < n < 8000:
                return None
            for _ in range(n):
                if any(H(q + 2 * k) >= nv for k in range(3)):
                    return None
                q += 8 if t in FORCE else 6
            ntri += n
        return nv, ntri
    except (struct.error, IndexError):
        return None


def main():
    rom = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../../baserom.us.z64"), "rb").read()
    blocks = {}
    for m in re.finditer(b"MIO0", rom):
        try:
            d = mio0(rom, m.start())
            if len(d) > 64:
                blocks[m.start()] = d
        except (struct.error, IndexError):
            pass
    layout = open(os.path.join(HERE, "../core/src/layout.rs")).read()
    level_num = {m[1].lower(): int(m[2]) for m in re.finditer(r"pub const LEVEL_(\w+): u32 = (\d+);", layout)}
    level_num["castle_inside"] = level_num["castle"]

    table, unresolved = {}, []
    for path in sorted(glob.glob(f"{DECOMP}/levels/*/areas/*/collision.inc.c")):
        folder, area = path.split("/")[-4], int(path.split("/")[-2])
        text = open(path).read()
        counts = [int(c, 0) for c in re.findall(r"COL_VERTEX_INIT\(([^)]*)\)", text)]
        verts = [int(a, 0) for m in re.finditer(r"COL_VERTEX\(([^)]*)\)", text) for a in m[1].split(",")]
        key = f"{level_num[folder]}:{area}"
        if len(counts) == 1:
            pat = struct.pack(">HH", 0x40, counts[0]) + struct.pack(">12h", *verts[:12])
            hits = [(o, d.find(pat)) for o, d in blocks.items() if d.find(pat) >= 0]
            if len(hits) == 1:
                table[key] = [hits[0][0], hits[0][1], folder]
                continue
        unresolved.append((key, folder, counts))
    used = {(v[0], v[1]) for v in table.values()}
    for key, folder, counts in unresolved:
        # Version #ifdefs: an unclaimed list with one of the listed vertex
        # counts, preferring the block the level's other areas are in.
        same = {v[0] for v in table.values() if v[2] == folder}
        cands = []
        for o, d in blocks.items():
            for m in re.finditer(b"\x00\x40", d):
                r = valid(d, m.start())
                if r and r[0] in counts and (o, m.start()) not in used:
                    cands.append((o not in same, o, m.start()))
        cands.sort()
        if not cands:
            sys.exit(f"{key} ({folder}): not found")
        table[key] = [cands[0][1], cands[0][2], folder]
        used.add((cands[0][1], cands[0][2]))
    for key, (o, p, folder) in table.items():
        assert valid(blocks[o], p), key

    names = {}
    for m in re.finditer(r"#define (SURFACE_\w+)\s+(0x[0-9A-Fa-f]+)", open(f"{DECOMP}/include/surface_terrains.h").read()):
        names.setdefault(int(m[2], 16), m[1])
    js = (
        "// Where each area's collision list is in the US ROM (generated by\n"
        "// objective-rrt/tools/rom_collision_table.py; offsets only, no game data):\n"
        "// [ROM offset of the MIO0 block, offset in the decompressed block].\n"
        f"export const COLLISION = {json.dumps({k: v[:2] for k, v in sorted(table.items(), key=lambda kv: [int(x) for x in kv[0].split(':')])}, indent=0)};\n\n"
        "// Surface type names (include/surface_terrains.h), for colouring.\n"
        f"export const SURFACE_NAMES = {json.dumps({str(k): v for k, v in sorted(names.items())})};\n\n"
        "// Types whose triangles carry a fourth value (surface_load.c surface_has_force).\n"
        f"export const FORCE_TYPES = {json.dumps(sorted(FORCE))};\n"
    )
    open(OUT, "w").write(js)
    print(f"{len(table)} areas -> {OUT}")


if __name__ == "__main__":
    main()
