#!/usr/bin/env python3
"""Compare runs on speed, GA effect and movement: tools/compare.py run [run ...]

Per run:
- milestones: game time first reaching the castle, BoB, the BoB summit, a
  King Bob-omb hit, a star (from nodes.csv);
- travel: speed of states made by movement objectives (PosDist, PosTime,
  Coin), as their end speed and as parent-to-state distance per frame;
- GA: where winners come from (generation, origin), how often evolution
  beat generation 0, and frames it saved on successes;
- movement: frames of long jumps etc. in GA winners.
"""

import csv
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
MOVES = ["long_jump", "dive", "wall_kick", "triple_jump", "side_flip", "backflip", "slide_kick"]
TRAVEL = ("Mario::PosDist", "Mario::PosTime", "Coin::Collect")


def clock(f):
    return "—" if f is None else f"{int(f / 30 // 60)}:{f / 30 % 60:04.1f}"


def load(run):
    d = os.path.join(ROOT, run)
    nodes = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(d, "nodes.csv")))}
    events = []
    for line in open(os.path.join(d, "events.jsonl")):
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    summary = json.load(open(os.path.join(d, "summary.json")))
    return nodes, events, summary


def first(nodes, pred):
    fs = [int(n["frame"]) for n in nodes.values() if pred(n)]
    return min(fs) if fs else None


def stats(run):
    nodes, events, summary = load(run)
    z = lambda n: (int(n["level"]), int(n["area"]))
    out = {"run": run, "states": len(nodes), "frames/s": round(summary["frames"] / summary["secs"])}
    out["castle"] = first(nodes, lambda n: z(n) == (6, 1))
    out["bob"] = first(nodes, lambda n: z(n) == (9, 1))
    out["summit"] = first(nodes, lambda n: z(n) == (9, 1) and float(n["y"]) > 4000)
    hit = {e["added"] for e in events if e["kind"] == "Boss::Hit" and e["success"] and e.get("added") is not None}
    out["king hit"] = first(nodes, lambda n: int(n["id"]) in hit)
    out["star"] = first(nodes, lambda n: int(n["stars"]) > 0)

    # Travel speed.
    ends, rates = [], []
    for n in nodes.values():
        if n["via"] in TRAVEL and "hspeed" in n:
            ends.append(float(n["hspeed"]))
        p = nodes.get(int(n["parent"]))
        if n["via"] in TRAVEL and p and z(p) == z(n):
            f = int(n["frame"]) - int(p["frame"])
            if f > 0:
                rates.append(math.dist((float(p["x"]), float(p["z"])), (float(n["x"]), float(n["z"]))) / f)
    out["end speed"] = round(sum(ends) / len(ends), 1) if ends else None
    out["u/frame"] = round(sum(rates) / len(rates), 1) if rates else None
    out["fast (>25 u/f)"] = f"{100 * sum(r > 25 for r in rates) / max(1, len(rates)):.0f}%"

    # GA.
    ga = [e["ga"] for e in events if e.get("ga")]
    if ga:
        out["GA attempts"] = len(ga)
        out["winner gen 0"] = f"{100 * sum(g['winner_gen'] == 0 for g in ga) / len(ga):.0f}%"
        origins = Counter(g["origin"] for g in ga)
        out["origins"] = ", ".join(f"{k} {100 * v / len(ga):.0f}%" for k, v in origins.most_common())
        out["evolution met more"] = f"{100 * sum(g['best_met'] > g['gen0_met'] for g in ga) / len(ga):.0f}%"
        # All goals met already in generation 0: frames evolution saved.
        full = [g for g in ga if g["gen0_met"] == g["best_met"] and g["best_met"] > 0 and -1e6 < g["gen0_tie"] < 0 and -1e6 < g["best_tie"] <= 0]
        saved = [g["best_tie"] - g["gen0_tie"] for g in full]
        out["frames saved (met in gen 0)"] = f"{sum(saved) / max(1, len(saved)):.1f} of {-sum(g['gen0_tie'] for g in full) / max(1, len(full)):.0f} (n={len(full)})"
        mv = [sum(g["moves"][i] for g in ga) for i in range(len(MOVES))]
        out["move frames"] = ", ".join(f"{MOVES[i]} {mv[i]}" for i in range(len(MOVES)) if mv[i])
        out["winners with a long jump"] = f"{100 * sum(g['moves'][0] > 0 for g in ga) / len(ga):.1f}%"
    # Inputs along the run's best route (most stars, then the latest
    # frontier state): how jerky are they?
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import struct, viz
    best = max(nodes.values(), key=lambda n: (int(n["stars"]), int(n["keys"]), int(n["depth"])))
    raw = viz.m64(run, int(best["id"]))[4 * 1690 :]
    pads = [struct.unpack(">Hbb", raw[i : i + 4]) for i in range(0, len(raw) - 3, 4)]
    if len(pads) > 1:
        ang = lambda q: math.atan2(q[1], q[2])
        tilted = lambda q: abs(q[1]) + abs(q[2]) > 40
        n = len(pads) - 1
        out["route stick changes"] = f"{100 * sum((a[1], a[2]) != (b[1], b[2]) for a, b in zip(pads, pads[1:])) / n:.0f}%"
        out["route turns >45°"] = f"{100 * sum(tilted(a) and tilted(b) and abs((ang(a) - ang(b) + math.pi) % (2 * math.pi) - math.pi) > math.radians(45) for a, b in zip(pads, pads[1:])) / n:.1f}%"
        out["route full tilt"] = f"{100 * sum(math.hypot(q[1], q[2]) > 110 for q in pads) / len(pads):.0f}%"
    return out


def main():
    runs = sys.argv[1:]
    rows = [stats(r) for r in runs]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    w = max(len(k) for k in keys)
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            vals.append(clock(v) if k in ("castle", "bob", "summit", "king hit", "star") else ("—" if v is None else str(v)))
        print(f"{k:<{w}}  " + "  |  ".join(f"{v:<14}" if len(v) < 30 else v for v in vals))


if __name__ == "__main__":
    main()
