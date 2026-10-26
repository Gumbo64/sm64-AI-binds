#!/usr/bin/env python3
"""Per-star analysis of a replay dumped by the `tasplay` bin (one CSV row per
frame), for TAS movies and our own replays alike:

    tools/tas_analyse.py <label> <dump.csv> [--json out.json] [--stars N]

Per star segment (from the previous star, or power-on, to the frame the star
count goes up) it reports: the route (level/area entries with their times),
the time spent in the star's level, the movement actions entered (names from
the decomp's include/sm64.h), travel speed, and clips: frames where Mario's
centre (60 units above his feet) crosses a static wall triangle of the area
(collision from the decomp's levels/*/areas/*/collision.inc.c), jumps in
position that his velocity doesn't explain, crossings of Chain Chomp's gate,
deaths and lost lives.

Times are game frames from power-on at 30/s (one input sample per frame).
"""
import collections
import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DECOMP = os.path.normpath(os.path.join(HERE, "../../../sm64-binds/sm64-port"))

SHORT = {"bob": "BoB", "wf": "WF", "jrb": "JRB", "ccm": "CCM", "bbh": "BBH", "hmc": "HMC", "lll": "LLL", "ssl": "SSL",
         "ddd": "DDD", "sl": "SL", "wdw": "WDW", "ttm": "TTM", "thi": "THI", "ttc": "TTC", "rr": "RR", "pss": "PSS",
         "castle_inside": "Castle", "castle_grounds": "Grounds", "castle_courtyard": "Courtyard", "totwc": "TotWC",
         "cotmc": "CotMC", "vcutm": "VCutM", "wmotr": "WMotR", "sa": "SA", "bitdw": "BitDW", "bitfs": "BitFS",
         "bits": "BitS", "bowser_1": "Bowser1", "bowser_2": "Bowser2", "bowser_3": "Bowser3", "ending": "Ending"}
COURSE_LEVELS = {"BoB", "WF", "JRB", "CCM", "BBH", "HMC", "LLL", "SSL", "DDD", "SL", "WDW", "TTM", "THI", "TTC", "RR",
                 "PSS", "TotWC", "CotMC", "VCutM", "WMotR", "SA", "BitDW", "BitFS", "BitS"}


def levels():
    """Level number -> (dir, short name), from levels/level_defines.h."""
    out, n = {}, 0
    for line in open(os.path.join(DECOMP, "levels/level_defines.h")):
        m = re.match(r"\s*(DEFINE_LEVEL|STUB_LEVEL)\((.*)\)", line)
        if not m:
            continue
        n += 1
        if m.group(1) == "DEFINE_LEVEL":
            d = [a.strip() for a in m.group(2).split(",")][3]
            out[n] = (d, SHORT.get(d, d))
    return out


def actions():
    """Action value -> name, from include/sm64.h."""
    out = {}
    for line in open(os.path.join(DECOMP, "include/sm64.h")):
        m = re.match(r"#define (ACT_[A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+)", line)
        if m and not m.group(1).startswith(("ACT_FLAG", "ACT_GROUP", "ACT_ID")):
            out.setdefault(int(m.group(2), 16), m.group(1)[4:])
    return out


LEVELS = levels()
ACTIONS = actions()

# Movement moves we count by entries (the action names they start in).
MOVES = collections.OrderedDict([
    ("long jump", {"LONG_JUMP"}),
    ("dive", {"DIVE"}),
    ("rollout", {"FORWARD_ROLLOUT", "BACKWARD_ROLLOUT"}),
    ("wall kick", {"WALL_KICK_AIR"}),
    ("side flip", {"SIDE_FLIP"}),
    ("backflip", {"BACKFLIP"}),
    ("triple jump", {"TRIPLE_JUMP", "SPECIAL_TRIPLE_JUMP", "FLYING_TRIPLE_JUMP"}),
    ("double jump", {"DOUBLE_JUMP"}),
    ("single jump", {"JUMP"}),
    ("jump kick", {"JUMP_KICK"}),
    ("slide kick", {"SLIDE_KICK"}),
    ("ground pound", {"GROUND_POUND"}),
    ("punch", {"PUNCHING", "MOVE_PUNCHING"}),
    ("butt slide", {"BUTT_SLIDE", "STOMACH_SLIDE", "HOLD_BUTT_SLIDE", "HOLD_STOMACH_SLIDE"}),
    ("crouch slide", {"CROUCH_SLIDE"}),
    ("pick up", {"PICKING_UP", "DIVE_PICKING_UP"}),
    ("throw", {"THROWING", "AIR_THROW"}),
    ("pole", {"GRAB_POLE_SLOW", "GRAB_POLE_FAST"}),
    ("hang", {"START_HANGING", "LEDGE_GRAB"}),
    ("swim stroke", {"BREASTSTROKE", "FLUTTER_KICK"}),
    ("cannon", {"IN_CANNON"}),
    ("shot from cannon", {"SHOT_FROM_CANNON"}),
    ("flying", {"FLYING"}),
    ("knockback", {"BACKWARD_AIR_KB", "FORWARD_AIR_KB", "HARD_BACKWARD_AIR_KB", "HARD_FORWARD_AIR_KB",
                   "BACKWARD_GROUND_KB", "FORWARD_GROUND_KB", "HARD_BACKWARD_GROUND_KB", "HARD_FORWARD_GROUND_KB",
                   "SOFT_BACKWARD_GROUND_KB", "SOFT_FORWARD_GROUND_KB"}),
    ("bonk", {"BACKWARD_AIR_KB_BONK", "GROUND_BONK", "AIR_HIT_WALL"}),
    ("lava boost", {"LAVA_BOOST"}),
    ("star dance", {"STAR_DANCE_EXIT", "STAR_DANCE_NO_EXIT", "STAR_DANCE_WATER", "FALL_AFTER_STAR_GRAB"}),
])
DEATHS = {"STANDING_DEATH", "QUICKSAND_DEATH", "ELECTROCUTION", "SUFFOCATION", "DEATH_ON_STOMACH", "DEATH_ON_BACK",
          "EATEN_BY_BUBBA", "DROWNING", "WATER_DEATH", "DEATH_EXIT", "UNUSED_DEATH_EXIT", "FALLING_DEATH_EXIT",
          "SPECIAL_DEATH_EXIT", "WATER_DEATH"}
# Chain Chomp's gate (levels/bob/script.c, chain_chomp_gate/collision.inc.c):
# a 1023 x 640 plane at (1456, 768, 446), yaw 326 degrees.
GATE = (1456.0, 768.0, 446.0, math.radians(326.0))


# Wall surface types Mario passes through (include/surface_terrains.h).
PASSABLE = {"SURFACE_CAMERA_BOUNDARY", "SURFACE_INTANGIBLE", "SURFACE_VANISH_CAP_WALLS"}
# Actions in which Mario isn't really moving through the level.
NOT_MOVING = {"DISAPPEARED", "EMERGE_FROM_PIPE", "TELEPORT_FADE_OUT", "TELEPORT_FADE_IN", "SPAWN_SPIN_AIRBORNE",
              "SPAWN_SPIN_LANDING", "SPAWN_NO_SPIN_AIRBORNE", "SPAWN_NO_SPIN_LANDING", "IN_CANNON"}


def is_blj(prev, r):
    """A backwards long jump on this frame: in a long jump (or landed from one
    the same frame, as on stairs and slopes) with backwards speed grown by the
    long jump's x1.5 (act_long_jump, mario_actions_airborne.c)."""
    return (act_name(r["act"]) in ("LONG_JUMP", "LONG_JUMP_LAND") and r["fvel"] < -10
            and r["fvel"] < 1.4 * prev["fvel"] and r["vy"] > 0)


def act_name(a):
    return ACTIONS.get(a, f"{a:#x}")


# ---- static walls -----------------------------------------------------------

_walls = {}


def walls(level, area):
    """Static wall triangles of a level's area, in a 512-unit XZ grid."""
    key = (level, area)
    if key in _walls:
        return _walls[key]
    grid = collections.defaultdict(list)
    d = LEVELS.get(level, (None,))[0]
    path = os.path.join(DECOMP, "levels", str(d), "areas", str(area), "collision.inc.c") if d else ""
    if path and os.path.exists(path):
        text = open(path).read()
        verts = [tuple(map(int, m)) for m in re.findall(r"COL_VERTEX\((-?\d+),\s*(-?\d+),\s*(-?\d+)\)", text)]
        tris, kind = [], None
        for m in re.finditer(r"COL_TRI_INIT\((\w+)|COL_TRI(?:_SPECIAL)?\((\d+),\s*(\d+),\s*(\d+)", text):
            if m.group(1):
                kind = m.group(1)
            elif kind not in PASSABLE:
                tris.append(tuple(int(m.group(i)) for i in (2, 3, 4)))
        for t in tris:
            a, b, c = (verts[i] for i in t)
            n = cross(sub(b, a), sub(c, a))
            ln = math.sqrt(dot(n, n))
            if ln == 0:
                continue
            n = tuple(x / ln for x in n)
            if abs(n[1]) >= 0.01:  # floors and ceilings (surface_load.c)
                continue
            tri = (a, b, c, n)
            xs, zs = [p[0] for p in (a, b, c)], [p[2] for p in (a, b, c)]
            for gx in range(int(min(xs)) // 512, int(max(xs)) // 512 + 1):
                for gz in range(int(min(zs)) // 512, int(max(zs)) // 512 + 1):
                    grid[(gx, gz)].append(tri)
    _walls[key] = grid
    return grid


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def seg_hits_tri(p, q, tri):
    """Whether segment p->q crosses triangle tri; +1 front-to-back, -1 back-to-front."""
    a, b, c, n = tri
    dp, dq = dot(sub(p, a), n), dot(sub(q, a), n)
    if (dp > 0) == (dq > 0) or dp == dq:
        return 0
    t = dp / (dp - dq)
    x = (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2]))
    # Inside test by same-side edges.
    for u, v in ((a, b), (b, c), (c, a)):
        if dot(cross(sub(v, u), sub(x, u)), n) < 0:
            return 0
    return 1 if dp > 0 else -1


def wall_crossings(level, area, p, q):
    grid = walls(level, area)
    if not grid:
        return []
    out, seen = [], set()
    for gx in range(int(min(p[0], q[0])) // 512, int(max(p[0], q[0])) // 512 + 1):
        for gz in range(int(min(p[2], q[2])) // 512, int(max(p[2], q[2])) // 512 + 1):
            for tri in grid.get((gx, gz), ()):
                if id(tri) in seen:
                    continue
                seen.add(id(tri))
                s = seg_hits_tri(p, q, tri)
                if s:
                    out.append((s, tri))
    return out


# ---- the dump ---------------------------------------------------------------

def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({
            "f": int(r["frame"]), "level": int(r["level"]), "area": int(r["area"]), "course": int(r["course"]),
            "pos": (float(r["x"]), float(r["y"]), float(r["z"])), "vy": float(r["vy"]), "fvel": float(r["fvel"]),
            "hs": float(r["hspeed"]), "act": int(r["action"], 16), "stars": int(r["stars"]), "keys": int(r["keys"]),
            "health": int(r["health"]), "lives": int(r["lives"]), "dialog": int(r["dialog"]),
            "ctrl": r["controllable"] == "1", "held": r["held"] == "1",
        })
    return rows


def clock(f):
    t = round(f / 3)  # tenths of a second
    return f"{t // 600}:{t % 600 / 10:04.1f}"


def lname(level):
    return LEVELS.get(level, (None, f"L{level}"))[1]


def gate_local(pos):
    gx, gy, gz, th = GATE
    dx, dz = pos[0] - gx, pos[2] - gz
    return dx * math.cos(th) - dz * math.sin(th), pos[1] - gy, dx * math.sin(th) + dz * math.cos(th)


def segment_stats(rows, lo, hi):
    """Stats over rows[lo:hi] (the frames of one star segment)."""
    seg = rows[lo:hi]
    route, last = [], None
    for r in seg:
        z = (r["level"], r["area"])
        if r["level"] and z != last:
            route.append((lname(r["level"]), r["area"], r["f"]))
            last = z
    star_level = rows[hi - 1]["level"] if hi > 0 else 0
    # Last entry into the star's level within the segment.
    entry = next((r["f"] for r in reversed(seg) if r["level"] != star_level), seg[0]["f"] - 1) + 1 if seg else 0
    moves, acts, blj = collections.Counter(), collections.Counter(), 0
    hyper = 0
    in_level = [r for r in seg if r["f"] >= entry]
    prev = rows[lo - 1] if lo > 0 else None
    for r in seg:
        if prev is not None and r["act"] != prev["act"]:
            name = act_name(r["act"])
            acts[name] += 1
            for k, names in MOVES.items():
                if name in names:
                    moves[k] += 1
        if prev is not None and is_blj(prev, r):
            blj += 1
        if r["ctrl"] and abs(r["fvel"]) > 100:
            hyper += 1
        prev = r
    if blj:
        moves["BLJ"] = blj
    if hyper:
        moves["hyperspeed frames (|fvel| > 100)"] = hyper
    # Travel speed in the star's level while controllable.
    ctrl = [r for r in in_level if r["ctrl"]]
    path = 0.0
    for a, b in zip(in_level, in_level[1:]):
        if a["ctrl"] and b["ctrl"] and (a["level"], a["area"]) == (b["level"], b["area"]):
            d = math.dist(a["pos"], b["pos"])
            if d < 2000:
                path += d
    mean_hs = sum(r["hs"] for r in ctrl) / len(ctrl) if ctrl else 0.0
    # Clips and odd events.
    events = []
    for i in range(max(lo, 1), hi):
        a, b = rows[i - 1], rows[i]
        if (a["level"], a["area"]) != (b["level"], b["area"]) or not b["level"]:
            continue
        name = act_name(b["act"])
        if name in DEATHS and act_name(a["act"]) not in DEATHS:
            events.append((b["f"], "death", f"{name} at {fmt(b['pos'])}"))
        if b["lives"] < a["lives"]:
            events.append((b["f"], "life lost", f"{a['lives']} -> {b['lives']}"))
        d = math.dist(a["pos"], b["pos"])
        expect = math.sqrt(b["hs"] ** 2 + b["vy"] ** 2) + math.sqrt(a["hs"] ** 2 + a["vy"] ** 2)
        if d > expect + 150 and d > 300:
            events.append((b["f"], "teleport", f"{d:.0f} units in 1 frame ({fmt(a['pos'])} -> {fmt(b['pos'])}), {name}"))
            continue
        if d > 5000 or name in NOT_MOVING or act_name(a["act"]) in NOT_MOVING:
            continue
        p, q = tuple(a["pos"]), tuple(b["pos"])
        p, q = (p[0], p[1] + 60, p[2]), (q[0], q[1] + 60, q[2])
        for s, tri in wall_crossings(b["level"], b["area"], p, q):
            events.append((b["f"], "wall crossed" if s > 0 else "wall crossed from behind",
                           f"at {fmt(b['pos'])}, wall normal ({tri[3][0]:.2f}, {tri[3][2]:.2f}), "
                           f"{act_name(a['act'])} -> {name}, hspeed {b['hs']:.0f}"))
        if lname(b["level"]) == "BoB":
            la, lb = gate_local(a["pos"]), gate_local(b["pos"])
            if (la[2] > 0) != (lb[2] > 0) and abs(lb[0]) < 560 and -100 < lb[1] < 700:
                events.append((b["f"], "Chomp gate plane crossed",
                               f"local z {la[2]:.0f} -> {lb[2]:.0f}, x {lb[0]:.0f}, {act_name(a['act'])} -> {name}, "
                               f"holding {b['held']}"))
    return {
        "route": route,
        "entry": entry,
        "level": lname(star_level),
        "moves": dict(moves),
        "acts": dict(acts.most_common(25)),
        "ctrl_frames": len(ctrl),
        "mean_hspeed": round(mean_hs, 1),
        "path_per_frame": round(path / max(1, len(ctrl)), 1),
        "events": events,
    }


def fmt(p):
    return f"({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f})"


def analyse(label, path, max_stars=None):
    rows = load(path)
    marks = []  # (row index of the frame the count rose, stars, keys)
    for i in range(1, len(rows)):
        if (rows[i]["stars"], rows[i]["keys"]) > (rows[i - 1]["stars"], rows[i - 1]["keys"]):
            marks.append(i)
    segs, lo = [], 0
    for n, i in enumerate(marks):
        if max_stars is not None and rows[i]["stars"] > max_stars:
            break
        st = segment_stats(rows, lo, i + 1)
        st.update({"n": n + 1, "frame": rows[i]["f"], "stars": rows[i]["stars"], "keys": rows[i]["keys"],
                   "pos": rows[i]["pos"], "act_at_grab": act_name(rows[i]["act"]),
                   "act_before": act_name(rows[i - 1]["act"]), "level_time": rows[i]["f"] - st["entry"],
                   "segment_time": rows[i]["f"] - (rows[lo]["f"] if lo else 0)})
        segs.append(st)
        lo = i + 1
    last = rows[-1]
    return {"label": label, "frames": len(rows), "final": {"level": lname(last["level"]), "stars": last["stars"],
                                                           "keys": last["keys"], "act": act_name(last["act"])},
            "segments": segs}


def report(a):
    out = [f"# {a['label']}: {a['frames']} frames ({clock(a['frames'])}), ends in {a['final']['level']} with "
           f"{a['final']['stars']} stars, {a['final']['keys']} keys ({a['final']['act']})"]
    for s in a["segments"]:
        out.append(f"\n## {s['n']}. {s['level']} star -> {s['stars']} stars / {s['keys']} keys at {clock(s['frame'])} "
                   f"(frame {s['frame']}); in level {s['level_time'] / 30:.2f} s; segment {s['segment_time'] / 30:.2f} s")
        out.append(f"grab at {fmt(s['pos'])}: {s['act_before']} -> {s['act_at_grab']}")
        out.append("route: " + " > ".join(f"{l}{'/' + str(ar) if ar != 1 else ''}@{clock(f)}" for l, ar, f in s["route"]))
        out.append(f"in level: {s['ctrl_frames']} controllable frames, mean hspeed {s['mean_hspeed']}, "
                   f"path {s['path_per_frame']} u/frame")
        out.append("moves: " + ", ".join(f"{k} {v}" for k, v in s["moves"].items()))
        if s["events"]:
            out.append("events:")
            for f, kind, txt in s["events"][:40]:
                out.append(f"  {clock(f)} (f{f}) {kind}: {txt}")
            if len(s["events"]) > 40:
                out.append(f"  ... {len(s['events']) - 40} more")
    return "\n".join(out)


def main():
    args = sys.argv[1:]
    js = None
    max_stars = None
    if "--json" in args:
        i = args.index("--json")
        js = args[i + 1]
        del args[i:i + 2]
    if "--stars" in args:
        i = args.index("--stars")
        max_stars = int(args[i + 1])
        del args[i:i + 2]
    label, path = args
    a = analyse(label, path, max_stars)
    print(report(a))
    if js:
        json.dump(a, open(js, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
