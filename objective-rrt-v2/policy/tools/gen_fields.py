#!/usr/bin/env python3
"""Generates policy/src/sense/fields.tsv and policy/FIELDS.md: every game field a policy can read by name.

    python3 policy/tools/gen_fields.py [<sm64-port dir>]

Sources (all read-only): struct definitions with their /*0xOFF*/ offset comments (wasm32 layout = the
N64 layout the comments document; checked against core/src/layout.rs), the Object raw-field macros of
include/object_fields.h, and the build's linker map (every global's address and size) joined with the
global's C declaration for its type. Rerun after rebuilding the game (addresses change).
"""
import glob, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "..", "sm64-binds", "sm64-port")
PORT = os.path.abspath(PORT)
BUILD = os.path.join(PORT, "build", "us_web_headless")
OUT_TSV = os.path.join(HERE, "..", "src", "sense", "fields.tsv")
OUT_MD = os.path.join(HERE, "..", "FIELDS.md")

SCALAR = {"u8": ("u8", 1), "s8": ("s8", 1), "u16": ("u16", 2), "s16": ("s16", 2), "u32": ("u32", 4), "s32": ("s32", 4),
          "f32": ("f32", 4), "float": ("f32", 4), "int": ("s32", 4), "uintptr_t": ("u32", 4), "u64": ("u64", 8), "s64": ("s64", 8),
          "bool": ("u8", 1), "char": ("s8", 1)}
VEC = {"Vec3f": ("f32", 3, 4), "Vec3s": ("s16", 3, 2), "Vec3i": ("s32", 3, 4), "Vec4f": ("f32", 4, 4), "Vec4s": ("s16", 4, 2), "Mat4": ("f32", 16, 4)}
STRUCTS = [("MarioState", "include/types.h"), ("Object", "include/types.h"), ("Camera", "src/game/camera.h"),
           ("LakituState", "src/game/camera.h"), ("Surface", "include/types.h"), ("PlayerCameraState", "src/game/camera.h"),
           ("Controller", "include/types.h"), ("Area", "src/game/area.h"), ("MarioBodyState", "include/types.h"),
           ("SurfaceNode", "include/types.h"), ("ObjectNode", "include/types.h"), ("GraphNodeObject", "include/types.h"),
           ("SpawnInfo", "include/types.h"), ("WarpTransition", "src/game/area.h"), ("SaveFile", "src/game/save_file.h"), ("HudDisplay", "src/game/level_update.h")]


def read(p):
    with open(os.path.join(PORT, p), errors="replace") as f:
        return f.read()


def struct_body(text, name):
    m = re.search(r"\bstruct\s+%s\s*\{" % name, text) or re.search(r"\bstruct\s+%s\s*\n\s*\{" % name, text)
    if not m:
        return None
    i = text.index("{", m.start())
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


FIELD = re.compile(r"/\*\s*0x([0-9A-Fa-f]+)\s*\*/\s*(?:const\s+)?(struct\s+\w+|union\s+\w+|[A-Za-z_]\w*)\s*(\*+)?\s*([A-Za-z_]\w*)\s*((?:\[[^\]]*\])*)\s*;")


MACROS = {"OBJECT_POOL_CAPACITY": 240}


def dims(s):
    n = 1
    for d in re.findall(r"\[([^\]]*)\]", s or ""):
        try:
            n *= int(d, 0)
        except ValueError:
            if d.strip() in MACROS:
                n *= MACROS[d.strip()]
            else:
                return None
    return n


def typeinfo(base, stars, arr):
    """-> (elem type, count, ptr struct or None)."""
    n = dims(arr)
    if stars:
        t = base.replace("struct ", "").replace("union ", "") if base.startswith("struct ") else "void"
        return "ptr", n or 1, t
    if base in SCALAR:
        return SCALAR[base][0], n or 1, None
    if base in VEC:
        e, c, _ = VEC[base]
        return e, c * (n or 1), None
    return "raw", n or 1, None


def parse_struct(sname, path):
    body = struct_body(read(path), sname)
    if body is None:
        return []
    out = []
    for m in FIELD.finditer(body):
        off, base, stars, name, arr = int(m.group(1), 16), m.group(2), m.group(3), m.group(4), m.group(5)
        if name == "rawData" or base.startswith("union"):
            continue
        t, c, ps = typeinfo(base, stars, arr)
        out.append((sname, name, t, off, c, ps or ""))
    # sizes of raw fields from the next offset
    offs = sorted(set(o for _, _, _, o, _, _ in out))
    fixed = []
    for s, n, t, o, c, ps in out:
        if t == "raw":
            nxt = [x for x in offs if x > o]
            c = (nxt[0] - o) if nxt else 4
            t = "u8"
        fixed.append((s, n, t, o, c, ps))
    return fixed


OBJ = re.compile(r"#define\s+/\*\s*0x([0-9A-Fa-f]+)\s*\*/\s*(o\w+)\s+OBJECT_FIELD_(\w+)\(([^)]*)\)")
OBJTY = {"U32": ("u32", ""), "S32": ("s32", ""), "S16": ("s16", ""), "F32": ("f32", ""), "S16P": ("ptr", "void"), "S32P": ("ptr", "void"),
         "ANIMS": ("ptr", "void"), "WAYPOINT": ("ptr", "void"), "CHAIN_SEGMENT": ("ptr", "void"), "OBJ": ("ptr", "Object"),
         "SURFACE": ("ptr", "Surface"), "VPTR": ("ptr", "void"), "CVPTR": ("ptr", "void")}


def parse_object_raw():
    out = []
    for m in OBJ.finditer(read("include/object_fields.h")):
        off, name, kind = int(m.group(1), 16), m.group(2), m.group(3)
        t, ps = OBJTY[kind]
        out.append(("Object", name, t, off, 1, ps))
    return out


def read_map():
    sym = re.compile(r"^\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)\s+(\S+)$")
    seen = collections.defaultdict(set)
    with open(os.path.join(BUILD, "sm64.map")) as f:
        for line in f:
            m = sym.match(line.rstrip("\n"))
            if not m:
                continue
            addr, size, name = int(m.group(1), 16), int(m.group(2), 16), m.group(3)
            if re.fullmatch(r"[gs][A-Z]\w*", name) and size > 0 and addr < 0x1000000:
                seen[name].add((addr, size))
    return {n: next(iter(v)) for n, v in seen.items() if len(v) == 1}


DECL = re.compile(r"^\s*(?:extern\s+|static\s+)*(?:const\s+|volatile\s+)*(struct\s+\w+|union\s+\w+|\w+)\s+(\*?)\s*([gs][A-Z]\w*)\s*((?:\[[^\]]*\])*)\s*(?:=|;)")


def find_decls(names):
    found = {}
    files = glob.glob(os.path.join(PORT, "src", "**", "*.[ch]"), recursive=True) + glob.glob(os.path.join(PORT, "include", "*.h"))
    for p in files:
        try:
            for line in open(p, errors="replace"):
                m = DECL.match(line)
                if m and m.group(3) in names:
                    base, star, name, arr = m.groups()
                    if name not in found or arr and not found[name][2]:
                        found[name] = (base, star, arr)
        except OSError:
            pass
    return found


def main():
    fields = []
    for s, p in STRUCTS:
        fields += parse_struct(s, p)
    fields += parse_object_raw()
    # fields the textual parse cannot see (anonymous nested structs); offsets checked against core/src/layout.rs
    fields.append(("Surface", "normal", "f32", 28, 3, ""))
    size_of = {"MarioState": 200, "Object": 608}
    RAWBYTES = ["gSaveBuffer", "gStaticSurfacePartition", "gDynamicSurfacePartition"]
    cmap = read_map()
    decl = find_decls(set(cmap))
    globs = []
    for name, (addr, size) in sorted(cmap.items()):
        if name not in decl:
            continue
        base, star, arr = decl[name]
        t, c, ps = typeinfo(base, star, arr)
        if t == "raw":
            sname = base.replace("struct ", "")
            if base.startswith("struct ") and any(sname == s for s, _ in STRUCTS):
                t, ps = "struct", sname
                esz = size_of.get(sname)
                c = dims(arr) or 1
                if esz and size != esz * c:
                    continue
            else:
                continue
        else:
            esz = {"u8": 1, "s8": 1, "u16": 2, "s16": 2, "u32": 4, "s32": 4, "f32": 4, "ptr": 4, "u64": 8, "s64": 8}[t]
            if size != esz * c and not (size % esz == 0 and dims(arr) is None):
                continue
            if size != esz * c:
                c = size // esz
        globs.append((name, t, addr, size, c, ps or ""))
    for n in RAWBYTES:
        if n in cmap and n not in {g[0] for g in globs}:
            a, sz = cmap[n]
            globs.append((n, "u8", a, sz, sz, ""))
    globs.sort()
    with open(OUT_TSV, "w") as f:
        f.write("# generated by policy/tools/gen_fields.py\n# G name type addr size count ptrstruct | F struct field type off count ptrstruct\n")
        for g in globs:
            f.write("G\t%s\t%s\t%d\t%d\t%d\t%s\n" % g)
        for s, n, t, o, c, ps in fields:
            f.write("F\t%s\t%s\t%s\t%d\t%d\t%s\n" % (s, n, t, o, c, ps))
    # markdown
    md = ["# Readable game fields", "",
          "Generated by `policy/tools/gen_fields.py` (rerun after rebuilding the game). Every field below can be named in a policy;",
          "see `src/sense/reader.rs` for the path grammar.", "",
          "## Paths", "",
          "- Globals: `gGlobalTimer`, `gRandomSeed16`, `gCurrLevelNum`, `gMarioStates[0].health` (`gMarioState` is an alias of `gMarioStates[0]`).",
          "- Struct fields: `.field`, array element `[i]`, pointer follow `->field` (`gCamera->mode`, `gMarioState.floor->type`, `gMarioState.heldObj->oAction`).",
          "- Objects: `obj[SLOT].oAction`, `nearest(bhv=bhvBobomb,k=0).oPosX`, `nearest(cat=enemy).oHealth`, `mario_obj.oPosX`, `held.oAction`.",
          "- Values are typed (`u8..s32`, `f32`, pointers as addresses; null pointers read as missing); `fetch` gives `f32`.", ""]
    md += ["## Globals (%d)" % len(globs), "", "| name | type | count | address |", "|---|---|---|---|"]
    for n, t, a, s, c, ps in globs:
        md.append("| `%s` | %s | %s | 0x%x |" % (n, (t if not ps else "%s %s" % (t, ps)), c if c > 1 else "", a))
    by = collections.defaultdict(list)
    for s, n, t, o, c, ps in fields:
        by[s].append((o, n, t, c, ps))
    for s in by:
        rows = sorted(by[s])
        md += ["", "## struct %s (%d fields)" % (s, len(rows)), "", "| offset | name | type | count |", "|---|---|---|---|"]
        for o, n, t, c, ps in rows:
            md.append("| 0x%03x | `%s` | %s | %s |" % (o, n, t if not ps else "%s %s" % (t, ps), c if c > 1 else ""))
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print("globals", len(globs), "fields", len(fields), "structs", {k: len(v) for k, v in by.items()})


main()
