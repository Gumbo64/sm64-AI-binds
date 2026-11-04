#!/usr/bin/env python3
"""Tune the BLJ spot detectors' thresholds from missed spots.

    python3 tools/analysis/blj_tune.py DIR [DIR ...] [--params ref/blj_detector.json]
        [--out ref/blj_detector.json] [--cap 0.25] [--unexplained blj_misses_unexplained.md]

Each DIR (a run dir with OBRRT_BLJ_SPOTS=1, or a lab dir) may hold:
- blj_misses.jsonl: observed BLJ chains no detector had a spot for, with
  their context and the triangles within 300 (spots.rs `miss_record`);
- blj_zones.json: the static floors / ceilings / walls of those zones;
- blj_spots.jsonl: the registry (attempted and never succeeded spots are
  the false-positive evidence reported per clause).

The detectors are re-implemented here on the dumps (same rules as
objectives/src/extra/tech/bljany/scan.rs and riser.rs: slope per triangle,
low-ceiling floor points, short risers with a lower floor in front and a
higher one behind). For each miss and clause it tries single-threshold
relaxations, smallest first, until one puts a candidate within
`--tol` (100, the registry's miss tolerance) of the miss start; a change is accepted only if the
extra candidates it adds over all the zones' geometry stay within `cap`
(+25%) for that clause. The accepted changes per parameter are combined
(the most relaxed value), re-checked against the cap, and written as a new
detector file (OBRRT_BLJ_DETECTOR) with a change log: which misses each
change covers and how many candidates it adds. Misses no single change
explains go to the unexplained file for review.
"""
import argparse
import json
import math
import os

DEFAULTS = {
    "slope_ny_min": 0.2924, "slope_ny_max": 0.6247, "slope_area": 20000.0, "obj_slope_area": 5000.0, "slope_spacing": 0.0,
    "riser_h_min": 15.0, "riser_h_max": 60.0, "riser_min_len": 100.0, "riser_front_lo": -60.0, "riser_front_hi": 8.0,
    "riser_behind_extra": 60.0, "ceil_head_lo": 0.0, "ceil_head_hi": 50.0, "corner_dist": 25.0, "corner_tall": 150.0,
    "pin_h": 37.5, "scan_r": 2500.0, "scan_n": 3.0, "match_dist": 300.0,
}

# Relaxations tried per clause, in order (parameter, values from the
# default outwards).
TRIES = {
    "Slope": [("slope_spacing", [400, 300, 250, 200, 150, 100]), ("slope_area", [15000, 10000, 7500, 5000, 2500, 1000]),
              ("slope_ny_max", [0.65, 0.7, 0.75, 0.8, 0.85]), ("slope_ny_min", [0.25, 0.2])],
    "Ceiling": [("ceil_head_hi", [60, 75, 100, 150]), ("ceil_head_lo", [-10, -20])],
    "Riser": [("riser_h_max", [70, 80, 100, 110]), ("riser_h_min", [10, 7.5]), ("riser_min_len", [75, 50, 25]),
              ("riser_front_hi", [12, 20]), ("riser_front_lo", [-80, -120]), ("riser_behind_extra", [80, 120])],
}

NOT_SLIPPERY = {0x15, 0x37, 0x7A}
SLIPPERY = {0x14, 0x2A, 0x35, 0x79}
VERY = {0x13, 0x2E, 0x36, 0x73, 0x74, 0x75, 0x78}
EXCLUDED = {0x01, 0x0A, 0x12, 0x22, 0x23, 0x24, 0x2D, 0x38}


def slide_ny(k):
    if k in NOT_SLIPPERY:
        return 0.0
    if k in SLIPPERY:
        return 0.9397
    if k in VERY:
        return 0.9848
    return 0.7880


def inside(t, x, z):
    a, b, c = t["v"]
    def side(p, q):
        return (q[0] - p[0]) * (z - p[2]) - (q[2] - p[2]) * (x - p[0])
    d1, d2, d3 = side(a, b), side(b, c), side(c, a)
    return (d1 >= 0 and d2 >= 0 and d3 >= 0) or (d1 <= 0 and d2 <= 0 and d3 <= 0)


def plane_y(t, x, z):
    a = t["v"][0]
    n = t["n"]
    return a[1] - (n[0] * (x - a[0]) + n[2] * (z - a[2])) / n[1]


def area(t):
    a, b, c = t["v"]
    return abs((b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])) / 2


class Grid:
    """Triangles by 256-unit column, for point queries."""
    def __init__(self, tris):
        self.tris, self.cells = tris, {}
        for i, t in enumerate(tris):
            xs = [p[0] for p in t["v"]]
            zs = [p[2] for p in t["v"]]
            for cx in range(math.floor(min(xs) / 256), math.floor(max(xs) / 256) + 1):
                for cz in range(math.floor(min(zs) / 256), math.floor(max(zs) / 256) + 1):
                    self.cells.setdefault((cx, cz), []).append(i)

    def at(self, x, z):
        return [self.tris[i] for i in self.cells.get((math.floor(x / 256), math.floor(z / 256)), []) if inside(self.tris[i], x, z)]


def slopes(floors, P):
    # One per (400-cell, facing quadrant): the largest (scan.rs slopes).
    best = {}
    for t in floors:
        k, ny = t["k"], t["n"][1]
        if k in EXCLUDED or k not in NOT_SLIPPERY or not (P["slope_ny_min"] < ny <= P["slope_ny_max"]) or area(t) < P["slope_area"]:
            continue
        yaw = int(math.atan2(t["n"][0], t["n"][2]) * 32768 / math.pi)
        pts = [[sum(p[i] for p in t["v"]) / 3 for i in range(3)]]
        sp = P["slope_spacing"]
        if sp > 0:
            xs = [p[0] for p in t["v"]]
            zs = [p[2] for p in t["v"]]
            x = math.ceil(min(xs) / sp) * sp
            while x <= max(xs):
                z = math.ceil(min(zs) / sp) * sp
                while z <= max(zs):
                    if inside(t, x, z):
                        pts.append([x, plane_y(t, x, z), z])
                    z += sp
                x += sp
        for c in pts:
            key = (round(c[0] / 400), round(c[2] / 400), ((yaw + 0x2000) % 0x10000) // 0x4000)
            if key not in best or area(t) > best[key][0]:
                best[key] = (area(t), c)
    return [v[1] for v in best.values()]


def ceilings(floors, ceils, P):
    cg = Grid([c for c in ceils if c["n"][1] < -0.01])
    best = {}
    for t in floors:
        if t["k"] in EXCLUDED or t["n"][1] <= max(slide_ny(t["k"]), 0.01):
            continue
        xs = [p[0] for p in t["v"]]
        zs = [p[2] for p in t["v"]]
        pts = [[sum(p[0] for p in t["v"]) / 3, sum(p[2] for p in t["v"]) / 3]]
        x = math.ceil(min(xs) / 100) * 100
        while x <= max(xs):
            z = math.ceil(min(zs) / 100) * 100
            while z <= max(zs):
                if inside(t, x, z):
                    pts.append([x, z])
                z += 100
            x += 100
        for x, z in pts:
            fy = plane_y(t, x, z)
            hs = [plane_y(c, x, z) for c in cg.at(x, z)]
            hs = [h for h in hs if h >= fy + 2]
            if not hs:
                continue
            head = min(hs) - fy - 160
            if P["ceil_head_lo"] <= head < P["ceil_head_hi"]:
                cell = (round(x / 400), round(z / 400))
                if cell not in best or head < best[cell][0]:
                    best[cell] = (head, [x, fy, z])
    return [v[1] for v in best.values()]


def risers(floors, walls, P):
    fg = Grid([f for f in floors if f["k"] not in EXCLUDED and f["n"][1] >= 0.01])
    def any_in(x, z, lo, hi):
        return any(lo <= plane_y(t, x, z) <= hi for t in fg.at(x, z))
    out = []
    for t in walls:
        ys = [p[1] for p in t["v"]]
        ylo, yhi = min(ys), max(ys)
        h = yhi - ylo
        if not (P["riser_h_min"] <= h <= P["riser_h_max"]):
            continue
        pts = sorted([[p[0], p[2]] for p in t["v"]])
        a, b = pts[0], pts[2]
        if math.dist(a, b) < P["riser_min_len"]:
            continue
        nl = math.hypot(t["n"][0], t["n"][2])
        if nl < 0.5:
            continue
        n = [t["n"][0] / nl, t["n"][2] / nl]
        c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        lower = lambda x, z: any_in(x, z, ylo + P["riser_front_lo"], ylo + P["riser_front_hi"])
        higher = lambda x, z: any_in(x, z, ylo + P["riser_front_hi"] + 0.001, yhi + P["riser_behind_extra"])
        probe = lambda s, f: any(f(c[0] + s * n[0] * e, c[1] + s * n[1] * e) for e in (15, 25, 40))
        if not (probe(1, lower) and probe(-1, higher)):
            if probe(-1, lower) and probe(1, higher):
                n = [-n[0], -n[1]]
            else:
                continue
        out.append([c[0] + n[0] * 60, ylo, c[1] + n[1] * 60])
    return out


def detect(clause, geo, P):
    if clause == "Slope":
        return slopes(geo["floors"], P)
    if clause == "Ceiling":
        return ceilings(geo["floors"], geo["ceils"], P)
    return risers(geo["floors"], geo["walls"], P)


TOL = 100.0


def caught(cands, pos, P):
    # Within the registry's miss tolerance (spots.rs MERGE_DIST, --tol).
    return any(math.dist(c, pos) <= TOL for c in cands)


def miss_geo(m):
    g = {"floors": [], "ceils": [], "walls": []}
    for t in m["tris"]:
        g[["floors", "ceils", "walls"][t["l"]]].append({"k": t["k"], "v": t["v"], "n": t["n"]})
    return g


def summary(m):
    n = m["floor_normal"]
    return (f"zone {m['zone']} node {m['node']} pos ({m['pos'][0]:.0f}, {m['pos'][1]:.0f}, {m['pos'][2]:.0f}) owner {m['owner']} "
            f"clause guess {m['clause']}; floor ny {n[1]:.3f} type {m['floor_type']:#x}, headroom {m['headroom']:.0f}, "
            f"wall {m['wall_normal']}, facing {m['yaw'] & 0xFFFF:#06x}, fv {m['fv']:.1f}, {len(m['tris'])} triangles dumped")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--params")
    ap.add_argument("--out", default="blj_detector.json")
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--tol", type=float, default=100.0, help="a candidate this close (3D) catches a miss (the registry's miss rule)")
    ap.add_argument("--unexplained", default="blj_misses_unexplained.md")
    a = ap.parse_args()
    global TOL
    TOL = a.tol
    P = dict(DEFAULTS)
    if a.params and os.path.exists(a.params):
        P.update({k: v for k, v in json.load(open(a.params)).items() if k in DEFAULTS})
    misses, zones, fp = [], {}, {}
    for d in a.dirs:
        p = os.path.join(d, "blj_misses.jsonl")
        if os.path.exists(p):
            misses += [json.loads(l) for l in open(p)]
        p = os.path.join(d, "blj_zones.json")
        if os.path.exists(p):
            zones.update(json.load(open(p)))
        p = os.path.join(d, "blj_spots.jsonl")
        if os.path.exists(p):
            for l in open(p):
                s = json.loads(l)
                if s.get("attempted", 0) >= 3 and s.get("succeeded", 0) == 0:
                    fp[s["clause"]] = fp.get(s["clause"], 0) + 1
    print(f"{len(misses)} misses, {len(zones)} zones of geometry; false-positive evidence (attempted >= 3, never succeeded): {fp or 'none'}")
    base = {c: sum(len(detect(c, g, P)) for g in zones.values()) for c in TRIES}
    print("candidates over the zones now: " + ", ".join(f"{c} {n}" for c, n in base.items()))

    def cost(c, Q):
        return sum(len(detect(c, g, Q)) for g in zones.values()) - base[c]

    proposals, unexplained = [], []
    for i, m in enumerate(misses):
        geo = miss_geo(m)
        found = None
        if any(caught(detect(c, geo, P), m["pos"], P) for c in TRIES):
            print(f"miss {i}: already caught by the current thresholds on its dump")
            continue
        # Clauses: the guessed one first.
        order = sorted(TRIES, key=lambda c: c != m["clause"])
        for c in order:
            for name, vals in TRIES[c]:
                for v in vals:
                    Q = dict(P, **{name: v})
                    if caught(detect(c, geo, Q), m["pos"], Q):
                        extra = cost(c, Q)
                        ok = extra <= a.cap * max(base[c], 1)
                        print(f"miss {i}: {c} {name} {P[name]} -> {v} catches it; +{extra} candidates ({'accepted' if ok else 'over the cap'})")
                        if ok:
                            found = (c, name, v, extra)
                        break
                if found:
                    break
            if found:
                break
        if found:
            proposals.append((i, *found))
        else:
            unexplained.append(m)
    # Combine per parameter: the most relaxed accepted value.
    new = dict(P)
    log = []
    for (c, name) in sorted({(p[1], p[2]) for p in proposals}):
        vals = [p[3] for p in proposals if p[2] == name]
        v = max(vals, key=lambda x: abs(x - P[name]))
        Q = dict(new, **{name: v})
        extra = cost(c, Q)
        if extra <= a.cap * max(base[c], 1):
            new[name] = v
            log.append({"clause": c, "param": name, "from": P[name], "to": v, "misses": [p[0] for p in proposals if p[2] == name], "extra_candidates": extra, "base_candidates": base[c]})
    out = dict(new)
    out["changes"] = log
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(log)} changes")
    for l in log:
        print(f"  {l['clause']}: {l['param']} {l['from']} -> {l['to']} covers misses {l['misses']}, +{l['extra_candidates']} candidates (of {l['base_candidates']})")
    with open(a.unexplained, "w") as f:
        f.write("# BLJ misses no single threshold change explains\n\n")
        f.write("Observed BLJ chains (blj_misses.jsonl) that no slope / ceiling / riser threshold change within the cap catches: "
                "likely a new clause type. Geometry summary per miss:\n\n")
        for m in unexplained:
            f.write(f"- {summary(m)}\n")
    print(f"{len(unexplained)} unexplained -> {a.unexplained}")


if __name__ == "__main__":
    main()
