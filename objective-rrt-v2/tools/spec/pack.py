#!/usr/bin/env python3
"""Context packer: a markdown "brief" for an LLM writing spec objectives (or
a landmark route) for one route segment of an obrrt2 run.

  tools/spec/pack.py RUN_DIR --from MILESTONE --to MILESTONE [--node ID]
                     [--out brief.md] [--no-docs] [--max-objects N]

MILESTONE (searched along the run's best route, `--to` after `--from`):
  f<frame>        the first route node at or after that frame
  n<id>           that node (it must be on the route)
  stars=<n>       the first node with at least n stars
  zone=<l>[:<a>]  the first node in that level (and area); names work: bob, castle:1
  flag=<c>:<hex>  the first node whose course c star flags have those bits (1:20 = BoB star 6)
  start / end     the route's root / its last node

The route is the run's best (most stars, earliest), or the path to --node.
Sections: segment start/end states; the zone's objects from the decomp level
scripts (OBJECT / OBJECT_WITH_ACTS in script.c, MACRO_OBJECT in
areas/*/macro.inc.c, SPECIAL_OBJECT in collision.inc.c); what the route did
in the segment (each node's `via` objective and frame); the spec format and
the schema docs; ref/ notes mentioning the level.
"""
import argparse
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.normpath(os.path.join(HERE, "..", ".."))
DECOMP = os.path.normpath(os.path.join(V2, "..", "..", "sm64-binds", "sm64-port"))

LEVEL_NAMES = {
    "castle": 6, "inside": 6, "grounds": 16, "courtyard": 26, "bob": 9, "wf": 24, "jrb": 12, "ccm": 5, "bbh": 4,
    "hmc": 7, "lll": 22, "ssl": 8, "ddd": 23, "sl": 10, "wdw": 11, "ttm": 36, "thi": 13, "ttc": 14, "rr": 15,
    "pss": 27, "sa": 20, "wmotr": 31, "totwc": 29, "cotmc": 28, "vcutm": 18, "bitdw": 17, "bitfs": 19, "bits": 21,
}


def level_dirs():
    """Level number -> decomp dir name (levels/level_defines.h, in order)."""
    out, n = {}, 0
    for line in open(os.path.join(DECOMP, "levels/level_defines.h")):
        m = re.match(r"\s*(DEFINE_LEVEL|STUB_LEVEL)\((.*)\)", line)
        if not m:
            continue
        n += 1
        if m.group(1) == "DEFINE_LEVEL":
            out[n] = [a.strip() for a in m.group(2).split(",")][3]
    return out


def actions():
    out = {}
    for line in open(os.path.join(DECOMP, "include/sm64.h")):
        m = re.match(r"#define ACT_([A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+)", line)
        if m and not m.group(1).startswith(("FLAG_", "GROUP_", "ID_")):
            out.setdefault(int(m.group(2), 16), m.group(1))
    return out


def macro_presets():
    names = [l.strip().rstrip(",") for l in open(os.path.join(DECOMP, "include/macro_preset_names.h")) if l.strip().startswith("macro_")]
    bhvs = re.findall(r"^\s*\{(bhv\w+),", open(os.path.join(DECOMP, "include/macro_presets.h")).read(), re.M)
    return dict(zip(names, bhvs))


def special_presets():
    path = os.path.join(DECOMP, "include/special_presets.h")
    out = {}
    if os.path.exists(path):
        for m in re.finditer(r"\{\s*(special_\w+)\s*,[^}]*?(bhv\w+)", open(path).read()):
            out[m.group(1)] = m.group(2)
    return out


def parse_zone(s):
    l, _, a = s.partition(":")
    l = LEVEL_NAMES.get(l.lower(), None) if not l.isdigit() else int(l)
    return l, (int(a) if a else None)


def level_objects(level, area):
    """(bhv, x, y, z, note) from the level's script.c (the area's block), macro and special objects."""
    d = level_dirs().get(level)
    if not d:
        return []
    base = os.path.join(DECOMP, "levels", d)
    out = []
    script = os.path.join(base, "script.c")
    if os.path.exists(script):
        cur_area = None
        for line in open(script):
            m = re.search(r"AREA\(\s*/\*index\*/\s*(\d+)|AREA\((\d+)", line)
            if m:
                cur_area = int(m.group(1) or m.group(2))
            m = re.search(r"OBJECT(_WITH_ACTS)?\(.*?/\*pos\*/\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),.*?/\*beh\*/\s*(bhv\w+)(?:,\s*/\*acts\*/\s*([^)]*))?", line)
            if m and (area is None or cur_area in (None, area)):
                acts = (m.group(6) or "").strip().replace(" | ", "/")
                out.append((m.group(5), int(m.group(2)), int(m.group(3)), int(m.group(4)), f"script{' acts ' + acts if acts else ''}"))
    presets = macro_presets()
    specials = special_presets()
    for ad in sorted(os.listdir(os.path.join(base, "areas"))) if os.path.isdir(os.path.join(base, "areas")) else []:
        if area is not None and ad != str(area):
            continue
        mac = os.path.join(base, "areas", ad, "macro.inc.c")
        if os.path.exists(mac):
            for m in re.finditer(r"MACRO_OBJECT(?:_WITH_BEH_PARAM)?\(\s*/\*preset\*/\s*(\w+),\s*/\*yaw\*/\s*-?\d+,\s*/\*pos\*/\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)", open(mac).read()):
                out.append((presets.get(m.group(1), m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), f"macro {m.group(1)}"))
        col = os.path.join(base, "areas", ad, "collision.inc.c")
        if os.path.exists(col):
            for m in re.finditer(r"SPECIAL_OBJECT(?:_WITH_YAW(?:_AND_PARAM)?)?\(\s*/\*preset\*/\s*(\w+),\s*/\*pos\*/\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)", open(col).read()):
                out.append((specials.get(m.group(1), m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), f"special {m.group(1)}"))
    return out


def load_route(run, node=None):
    rows = list(csv.DictReader(open(os.path.join(run, "nodes.csv"))))
    by = {r["id"]: r for r in rows}
    if node is None:
        best = max(rows, key=lambda r: (int(r["stars"]), -int(r["frame"])))
        node = best["id"]
    chain, n = [], str(node)
    while n != "-1":
        chain.append(by[n])
        n = by[n]["parent"]
    return list(reversed(chain))


def flags_have(r, course, bits):
    for t in r.get("star_flags", "").split():
        c, _, h = t.partition(":")
        if int(c) == course and int(h, 16) & bits == bits:
            return True
    return False


def find(route, spec, after=0):
    if spec == "start":
        return 0
    if spec == "end":
        return len(route) - 1
    for i in range(after, len(route)):
        r = route[i]
        if spec.startswith("f") and spec[1:].isdigit() and int(r["frame"]) >= int(spec[1:]):
            return i
        if spec.startswith("n") and spec[1:].isdigit() and r["id"] == spec[1:]:
            return i
        if spec.startswith("stars=") and int(r["stars"]) >= int(spec[6:]):
            return i
        if spec.startswith("zone="):
            l, a = parse_zone(spec[5:])
            if int(r["level"]) == l and (a is None or int(r["area"]) == a):
                return i
        if spec.startswith("flag="):
            c, _, h = spec[5:].partition(":")
            if flags_have(r, int(c), int(h, 16)):
                return i
    sys.exit(f"milestone {spec!r} not found on the route after index {after}")


def state_line(r, acts):
    a = int(r["action"], 16)
    return (f"node {r['id']} frame {r['frame']} ({int(r['frame']) / 30:.1f} s) zone {r['level']}:{r['area']} "
            f"pos ({r['x']}, {r['y']}, {r['z']}) stars {r['stars']} keys {r['keys']} coins {r['coins']} "
            f"action {acts.get(a, hex(a))} ({r['action']}) speed {r.get('fwd_vel', '?')} star_flags [{r.get('star_flags', '')}] via {r['via']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--from", dest="frm", default="start")
    ap.add_argument("--to", default="end")
    ap.add_argument("--node", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-docs", action="store_true")
    ap.add_argument("--max-objects", type=int, default=200)
    ap.add_argument("--validate", default=None, help="a spec-validate JSON report to include")
    a = ap.parse_args()
    acts = actions()
    route = load_route(a.run, a.node)
    i0 = find(route, a.frm)
    i1 = find(route, a.to, i0)
    seg = route[i0:i1 + 1]
    s, e = seg[0], seg[-1]
    out = []
    w = out.append
    w(f"# Brief: {os.path.basename(os.path.normpath(a.run))}, {a.frm} -> {a.to}\n")
    w("You write spec objectives (JSON, format below) for this route segment, or a landmark route (a list of legs). "
      "Times are frames at 30/s from power-on. Positions are SM64 units (y up; yaw 0 faces +z, 0x4000 faces +x). "
      "A spec's ETA is honest frames: distance / pace (30 u/frame default) plus each phase's fixed frames.\n")
    w("## Segment\n")
    w(f"- start: {state_line(s, acts)}")
    w(f"- end: {state_line(e, acts)}")
    w(f"- length: {int(e['frame']) - int(s['frame'])} frames ({(int(e['frame']) - int(s['frame'])) / 30:.1f} s), {len(seg)} route nodes\n")
    w("## What the route did (route nodes: frame, zone, position, stars, action, the obrrt objective that made the node)\n")
    w("`~d` marks a dense node (a state along an attempt's winning rollout).\n")
    w("| node | frame | Δ | zone | x | y | z | stars | action | via |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    prev = int(s["frame"])
    for r in seg:
        a_ = int(r["action"], 16)
        w(f"| {r['id']} | {r['frame']} | {int(r['frame']) - prev} | {r['level']}:{r['area']} | {r['x']} | {r['y']} | {r['z']} | {r['stars']} | {acts.get(a_, hex(a_))} | {r['via']} |")
        prev = int(r["frame"])
    vias = {}
    for r in seg[1:]:
        k = r["via"].replace("~d", "")
        vias[k] = vias.get(k, 0) + 1
    w("\nObjectives along the segment: " + ", ".join(f"{k} ×{v}" for k, v in sorted(vias.items(), key=lambda x: -x[1])) + "\n")
    zones = []
    for r in seg:
        z = (int(r["level"]), int(r["area"]))
        if z not in zones:
            zones.append(z)
    names = {v: k for k, v in LEVEL_NAMES.items()}
    for (l, ar) in zones:
        objs = level_objects(l, ar)
        if not objs:
            continue
        w(f"## Objects in zone {l}:{ar} ({names.get(l, l)}) from the decomp level scripts\n")
        w("Spawn positions (objects may move). Behaviour names are what `bhv` selectors take.\n")
        w("| behaviour | x | y | z | source |")
        w("|---|---|---|---|---|")
        for (b, x, y, z, note) in objs[: a.max_objects]:
            w(f"| {b} | {x} | {y} | {z} | {note} |")
        if len(objs) > a.max_objects:
            w(f"\n({len(objs) - a.max_objects} more not shown)")
        w("")
    if a.validate and os.path.exists(a.validate):
        w("## Validator report (spec-validate --json)\n")
        w("```json\n" + open(a.validate).read()[:20000] + "\n```\n")
    if not a.no_docs:
        for title, path in [("Landmark tips", os.path.join(HERE, "LANDMARK_TIPS.md")),
                            ("Spec format", os.path.join(HERE, "FORMAT.md")),
                            ("Landmark routes and the loop", os.path.join(HERE, "README.md")),
                            ("Schema (bins) kinds and params: oss2", os.path.join(V2, "oss2", "README.md")),
                            ("Schema kinds: objective-scattershot PORTING.md", os.path.join(V2, "..", "objective-scattershot", "PORTING.md"))]:
            if os.path.exists(path):
                txt = open(path).read()
                if "oss2" in path:
                    m = re.search(r"## Key coarseness.*?(?=\n## )", txt, re.S)
                    txt = m.group(0) if m else txt
                w(f"## {title} ({os.path.relpath(path, V2)})\n")
                w(txt.strip() + "\n")
    # ref notes mentioning the level(s).
    terms = set()
    dirs = level_dirs()
    for (l, _) in zones:
        n = names.get(l)
        if n and n not in ("castle", "inside", "grounds", "courtyard"):
            terms.add(n)
        if dirs.get(l) and l not in (6, 16, 26):
            terms.add(dirs[l])
    hits = []
    for root, _, files in os.walk(os.path.join(V2, "ref")):
        for f in files:
            if not f.endswith((".md", ".txt")):
                continue
            p = os.path.join(root, f)
            try:
                lines = open(p, errors="replace").read().splitlines()
            except OSError:
                continue
            for k, line in enumerate(lines):
                if any(re.search(rf"\b{re.escape(t)}\b", line, re.I) for t in terms):
                    hits.append((os.path.relpath(p, V2), k + 1, line.strip()[:240]))
    if hits:
        w(f"## ref/ notes mentioning {', '.join(sorted(terms))} ({len(hits)} lines; first 80)\n")
        for (p, k, line) in hits[:80]:
            w(f"- {p}:{k}: {line}")
    text = "\n".join(out) + "\n"
    if a.out:
        open(a.out, "w").write(text)
        print(f"{a.out}: {len(text)} chars")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
