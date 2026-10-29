#!/usr/bin/env python3
"""Parse SM64 decomp level scripts into one JSON data file.

usage: parse_levels.py [<sm64-port dir>] [<out.json>]

Reads levels/level_defines.h (level ids), levels/course_defines.h (course
numbers), levels/<dir>/script.c (areas, objects, warps, JUMP_LINK targets),
levels/<dir>/areas/<N>/macro.inc.c (macro objects) and include/macro_presets.h
(preset -> bhv/model/param).  Anything that cannot be resolved is listed under
"unresolved" in the output and on stderr; nothing is guessed.
Conditional compilation assumes the US version (no VERSION_* defined).
"""
import glob
import json
import os
import re
import sys

DEF_PORT = "/workspaces/sm64-combined/sm64-binds/sm64-port"
DEF_OUT = "/workspaces/sm64-combined/sm64-AI-binds/objective-scattershot/tools/analysis/levels.json"

unresolved = []


def read(path):
    with open(path, errors="replace") as f:
        return f.read().replace("\r\n", "\n").replace("\r", "\n")


def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.S)
    return re.sub(r"//[^\n]*", "", t)


def preprocess(text):
    """Strip comments, resolve #if/#ifdef assuming no VERSION_* macro is set."""
    out, stack = [], []  # stack of (active, taken, parent_active)
    active = True
    for line in strip_comments(text).split("\n"):
        s = line.strip()
        if s.startswith("#"):
            m = re.match(r"#\s*(\w+)\s*(.*)", s)
            d, arg = m.group(1), m.group(2)
            if d in ("ifdef", "ifndef", "if"):
                if d == "ifdef":
                    c = False
                elif d == "ifndef":
                    c = True
                else:
                    e = re.sub(r"defined\s*\(\s*\w+\s*\)", "0", arg)
                    e = re.sub(r"\bdefined\s+\w+", "0", e)
                    e = e.replace("!", " not ").replace("&&", " and ").replace("||", " or ")
                    try:
                        c = bool(eval(e, {}, {}))
                    except Exception:
                        c = False
                stack.append((active, c))
                active = active and c
            elif d == "else" and stack:
                parent, c = stack[-1]
                stack[-1] = (parent, not c)
                active = parent and not c
            elif d == "endif" and stack:
                active = stack.pop()[0]
            continue
        if active:
            out.append(line)
    return "\n".join(out)


def split_args(s):
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


def find_calls(text):
    """Yield (name, [args]) for every NAME(...) macro call, in source order."""
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]*)\s*\(", text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        yield m.group(1), split_args(text[m.end():i - 1]), m.start()


# ---------------------------------------------------------------- symbols
class Symbols:
    def __init__(self, port):
        self.v = {}

    def load_file_defines(self, text):
        for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)[ \t]+(.+)$", strip_comments(text), flags=re.M):
            self.v.setdefault(m.group(1), m.group(2).strip())

    def load_enums(self, text):
        for em in re.finditer(r"enum\s+\w*\s*\{(.*?)\}", strip_comments(text), flags=re.S):
            n = 0
            for item in em.group(1).split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    k, e = [x.strip() for x in item.split("=", 1)]
                    val = self.eval(e)
                    if val is None:
                        continue
                    n = val
                else:
                    k = item
                self.v[k] = str(n)
                n += 1

    def eval(self, expr, depth=0):
        """Evaluate a simple integer C expression; None if a symbol is unknown."""
        e = expr.strip()
        if depth > 20 or not e:
            return None
        e = re.sub(r"\(\s*(?:s|u)(?:8|16|32|64)\s*\)", "", e)
        e = re.sub(r"(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b", r"\1", e)
        bad = False

        def sub(m):
            nonlocal bad
            name = m.group(0)
            if name in self.v:
                r = self.eval(self.v[name], depth + 1)
                if r is not None:
                    return str(r)
            bad = True
            return "0"
        e2 = re.sub(r"\b[A-Za-z_]\w*\b", lambda m: m.group(0) if re.fullmatch(r"0[xX][0-9a-fA-F]+", m.group(0)) else sub(m), e)
        if bad or not re.fullmatch(r"[0-9a-fA-FxX\s()+\-*|&~<>^]*", e2):
            return None
        try:
            return int(eval(e2, {"__builtins__": {}}, {}))
        except Exception:
            return None


# ---------------------------------------------------------------- tables
def parse_levels(port, sym):
    """Level enum order from level_defines.h (LEVEL_NONE = 0, then each entry)."""
    text = strip_comments(read(os.path.join(port, "levels/level_defines.h")))
    course_ids = parse_courses(port)
    levels, n = [], 0
    for name, args, _ in find_calls(text):
        if name not in ("DEFINE_LEVEL", "STUB_LEVEL"):
            continue
        n += 1
        enum = args[1] if name == "DEFINE_LEVEL" else args[1]
        sym.v[enum] = str(n)
        if name == "DEFINE_LEVEL":
            levels.append({"id": n, "dir": args[3], "enum": enum, "course": course_ids.get(args[2])})
    sym.v["LEVEL_NONE"] = "0"
    return levels


def parse_courses(port):
    text = strip_comments(read(os.path.join(port, "levels/course_defines.h")))
    ids, n = {}, 0
    for name, args, _ in find_calls(text):
        if name in ("DEFINE_COURSE", "DEFINE_BONUS_COURSE"):
            ids[args[0]] = n
            n += 1
    return ids


def parse_presets(port, sym):
    names = re.search(r"enum\s+MacroPresets\s*\{(.*?)\}", read(os.path.join(port, "include/macro_preset_names.h")), flags=re.S)
    enum = [x.strip() for x in strip_comments(names.group(1)).split(",") if x.strip()]
    text = strip_comments(read(os.path.join(port, "include/macro_presets.h")))
    body = text[text.index("MacroObjectPresets[]"):]
    rows = re.findall(r"\{\s*(\w+)\s*,\s*(\w+)\s*,\s*([^}]*?)\s*\}", body)
    if len(rows) != len(enum):
        print("warning: preset enum has %d names but table has %d rows" % (len(enum), len(rows)), file=sys.stderr)
    presets = {}
    for nm, (bhv, model, param) in zip(enum, rows):
        p = sym.eval(param)
        presets[nm] = {"bhv": bhv, "model": None if model == "MODEL_NONE" else model, "param": p,
                       "param_raw": param}
    return presets


def parse_macro_files(port):
    """Map macro array name -> list of (preset, yaw, x, y, z, behParam|None)."""
    out = {}
    for p in glob.glob(os.path.join(port, "levels/*/areas/*/macro.inc.c")):
        text = preprocess(read(p))
        for m in re.finditer(r"MacroObject\s+(\w+)\s*\[\s*\]\s*=\s*\{(.*?)\};", text, flags=re.S):
            lst = []
            for name, a, _ in find_calls(m.group(2)):
                if name == "MACRO_OBJECT" and len(a) == 5:
                    lst.append(a + [None])
                elif name == "MACRO_OBJECT_WITH_BEH_PARAM" and len(a) == 6:
                    lst.append(a)
            out[m.group(1)] = lst
    return out


def parse_special_presets(port, sym):
    """Special object presets (doors, trees, ... in the areas' collision data):
    name -> {bhv, model, type, def_param}. Names come from the SpecialPresets
    enum (with its explicit values and aliases), rows from special_presets.h."""
    body = re.search(r"enum\s+SpecialPresets\s*\{(.*?)\}", strip_comments(read(os.path.join(port, "include/special_preset_names.h"))), flags=re.S).group(1)
    ids, n = {}, 0
    for item in [x.strip() for x in body.split(",") if x.strip()]:
        if "=" in item:
            name, val = [t.strip() for t in item.split("=", 1)]
            n = ids[val] if val in ids else sym.eval(val)
        else:
            name = item
        ids[name] = n
        n += 1
    types = {m.group(1): int(m.group(2)) for m in re.finditer(r"#define\s+(SPTYPE_\w+)\s+(\d+)", read(os.path.join(port, "include/special_presets.h")))}
    rows = {}
    text = strip_comments(read(os.path.join(port, "include/special_presets.h")))
    for m in re.finditer(r"\{\s*(0x[0-9a-fA-F]+|\d+)\s*,\s*(SPTYPE_\w+)\s*,\s*(0x[0-9a-fA-F]+|\d+)\s*,\s*(\w+)\s*,\s*(\w+)\s*\}", text):
        rows[int(m.group(1), 0)] = {"type": types.get(m.group(2)), "type_name": m.group(2), "def_param": int(m.group(3), 0), "model": m.group(4), "bhv": m.group(5)}
    return {name: rows[i] for name, i in ids.items() if i in rows}


def parse_special_objects(port, d, area_id, specials, sym):
    """Special objects of one area: its collision.inc.c's SPECIAL_OBJECT* list.
    oBehParams as the game sets it (macro_special_objects.c): params << 16 for
    SPTYPE_PARAMS_AND_YROT, the preset's default param << 24 for
    SPTYPE_DEF_PARAM_AND_YROT, else 0."""
    path = os.path.join(port, "levels", d, "areas", str(area_id), "collision.inc.c")
    if not os.path.exists(path):
        return []
    out = []
    for name, a, _ in find_calls(preprocess(read(path))):
        if not name.startswith("SPECIAL_OBJECT"):
            continue
        pr = specials.get(a[0])
        if pr is None:
            unresolved.append("%s area %s: special preset %r unknown" % (d, area_id, a[0]))
            continue
        if pr["bhv"] in ("NULL", "0"):
            continue
        vals = [sym.eval(t) for t in a[1:]]
        if None in vals:
            unresolved.append("%s area %s: special object %r" % (d, area_id, a))
            continue
        x, y, z = vals[:3]
        yaw = vals[3] if len(vals) > 3 else 0
        if pr["type_name"] == "SPTYPE_PARAMS_AND_YROT" and len(vals) > 4:
            bp = (vals[4] & 0xFFFF) << 16
        elif pr["type_name"] == "SPTYPE_DEF_PARAM_AND_YROT":
            bp = (pr["def_param"] & 0xFF) << 24
        else:
            bp = 0
        out.append({"bhv": pr["bhv"], "model": None if pr["model"] == "MODEL_NONE" else pr["model"], "pos": [x, y, z],
                    "angle": [0, yaw, 0], "beh_param": bp & 0xFFFFFFFF, "beh_param_raw": " ".join(a), "acts": [1, 2, 3, 4, 5, 6],
                    "source": "special"})
    return out


# ---------------------------------------------------------------- script
def acts_from(expr, sym, ctx):
    v = sym.eval(expr)
    if v is None:
        unresolved.append("%s: acts %r" % (ctx, expr))
        return None
    return [i + 1 for i in range(6) if v & (1 << i)]


def make_object(a, acts_expr, sym, ctx):
    model, x, y, z, rx, ry, rz, bp, bhv = a[:9]
    vals = [sym.eval(t) for t in (x, y, z, rx, ry, rz)]
    if any(v is None for v in vals):
        unresolved.append("%s: object %s coords %r" % (ctx, bhv, a))
        return None
    acts = [1, 2, 3, 4, 5, 6] if acts_expr is None else acts_from(acts_expr, sym, ctx)
    bpv = sym.eval(bp)
    if bpv is None:
        unresolved.append("%s: beh_param %r of %s" % (ctx, bp, bhv))
    return {"bhv": bhv, "model": None if model == "MODEL_NONE" else model, "pos": vals[:3],
            "angle": vals[3:], "beh_param": None if bpv is None else bpv & 0xFFFFFFFF,
            "beh_param_raw": bp, "acts": acts, "source": "script"}


def parse_blocks(text):
    """Return {name: body} for each `static const LevelScript name[] = { ... };`."""
    blocks = {}
    for m in re.finditer(r"LevelScript\s+(\w+)\s*\[\s*\]\s*=\s*\{", text):
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        blocks[m.group(1)] = text[m.end():i - 1]
    return blocks


def expand_calls(body, blocks, seen=()):
    """Flat list of (name, args) with JUMP_LINK/JUMP targets within the file inlined."""
    for name, args, _ in find_calls(body):
        if name in ("JUMP_LINK", "JUMP") and args and args[0] in blocks and args[0] not in seen:
            # only inline blocks that look like object lists (local/global script funcs)
            yield from expand_calls(blocks[args[0]], blocks, seen + (args[0],))
        else:
            yield name, args


def parse_level(lv, port, sym, presets, macros, specials):
    d = lv["dir"]
    path = os.path.join(port, "levels", d, "script.c")
    lv["areas"] = []
    if not os.path.exists(path):
        return
    text = preprocess(read(path))
    blocks = parse_blocks(text)
    entry = blocks.get("level_%s_entry" % d)
    if entry is None:
        return
    area = None
    mario_pos = {}
    for name, a in expand_calls(entry, blocks):
        ctx = "%s" % d
        if name == "AREA":
            area = {"id": sym.eval(a[0]), "mario_pos": None, "objects": [], "warps": []}
            lv["areas"].append(area)
        elif name == "END_AREA":
            area = None
        elif name in ("OBJECT", "OBJECT_WITH_ACTS") and area is not None:
            expect = 9 if name == "OBJECT" else 10
            if len(a) != expect:
                unresolved.append("%s: malformed %s %r" % (ctx, name, a))
                continue
            o = make_object(a, a[9] if name == "OBJECT_WITH_ACTS" else None, sym, "%s area %s" % (ctx, area["id"]))
            if o:
                area["objects"].append(o)
        elif name == "MACRO_OBJECTS" and area is not None:
            lst = macros.get(a[0])
            if lst is None:
                unresolved.append("%s: macro list %r not found" % (ctx, a[0]))
                continue
            for preset, yaw, x, y, z, bp in lst:
                o = make_macro_object(preset, yaw, x, y, z, bp, presets, sym, "%s area %s" % (ctx, area["id"]))
                if o:
                    area["objects"].append(o)
        elif name in ("WARP_NODE", "PAINTING_WARP_NODE") and area is not None:
            v = [sym.eval(t) for t in a[:4]]
            if None in v:
                unresolved.append("%s: warp %r" % (ctx, a))
                continue
            area["warps"].append({"id": v[0], "dest_level": v[1], "dest_area": v[2], "dest_node": v[3],
                                  "painting": name == "PAINTING_WARP_NODE"})
        elif name == "MARIO_POS":
            v = [sym.eval(t) for t in a[:5]]
            if None in v:
                unresolved.append("%s: MARIO_POS %r" % (ctx, a))
                continue
            mario_pos[v[0]] = {"yaw": v[1], "pos": v[2:5]}
    for ar in lv["areas"]:
        ar["mario_pos"] = mario_pos.get(ar["id"])
        ar["objects"].extend(parse_special_objects(port, d, ar["id"], specials, sym))


def make_macro_object(preset, yaw, x, y, z, bp, presets, sym, ctx):
    pr = presets.get(preset)
    if pr is None:
        unresolved.append("%s: macro preset %r unknown" % (ctx, preset))
        return None
    vals = [sym.eval(t) for t in (yaw, x, y, z)]
    if None in vals:
        unresolved.append("%s: macro object %s %r" % (ctx, preset, (yaw, x, y, z)))
        return None
    raw = "0" if bp is None else bp
    bpv = sym.eval(raw)
    if bpv is None or pr["param"] is None:
        unresolved.append("%s: macro beh_param %r preset %s" % (ctx, raw, preset))
        bpfinal = None
    else:
        # game: params = (params & 0xFF00) + (preset.param & 0xFF) if preset.param != 0;
        # oBehParams = (params & 0xFF) << 16
        params = bpv & 0xFFFF
        if pr["param"] != 0:
            params = (params & 0xFF00) + (pr["param"] & 0xFF)
        bpfinal = (params & 0xFF) << 16
    return {"bhv": pr["bhv"], "model": pr["model"], "pos": vals[1:], "angle": [0, vals[0], 0],
            "beh_param": bpfinal, "beh_param_raw": raw if pr["param"] in (0, None) else "%s | preset %s" % (raw, pr["param_raw"]),
            "acts": [1, 2, 3, 4, 5, 6], "source": "macro"}


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else DEF_PORT
    out = sys.argv[2] if len(sys.argv) > 2 else DEF_OUT
    sym = Symbols(port)
    for h in ("model_ids.h", "object_constants.h", "level_commands.h", "sm64.h", "types.h", "macros.h"):
        p = os.path.join(port, "include", h)
        if os.path.exists(p):
            sym.load_file_defines(read(p))
    for h in ("dialog_ids.h", "model_ids.h", "object_constants.h"):
        sym.load_enums(read(os.path.join(port, "include", h)))
    levels = parse_levels(port, sym)
    presets = parse_presets(port, sym)
    macros = parse_macro_files(port)
    specials = parse_special_presets(port, sym)
    for lv in levels:
        parse_level(lv, port, sym, presets, macros, specials)
    levels = [lv for lv in levels if lv["areas"]]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"levels": levels, "unresolved": unresolved}, f, indent=1)
    for lv in levels:
        print("%2d %-17s areas=%d objects=%d warps=%d" % (
            lv["id"], lv["dir"], len(lv["areas"]), sum(len(a["objects"]) for a in lv["areas"]),
            sum(len(a["warps"]) for a in lv["areas"])))
    for u in unresolved:
        print("UNRESOLVED:", u, file=sys.stderr)


if __name__ == "__main__":
    main()
