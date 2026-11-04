#!/usr/bin/env python3
"""Movement statistics of tasplay dumps, per 1000 controllable frames, split
into course levels and the hub (castle, grounds, courtyard), up to a star
count: tools/tas_movestats.py <max stars> <label=dump.csv> ...

Rates are action entries (names from include/sm64.h, grouped as in
tas_analyse.MOVES); `air` is the share of controllable frames in an airborne
action (ACT_FLAG_AIR); speeds are mean |horizontal velocity| and path length
per controllable frame."""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tas_analyse as T

HUB = {"Castle", "Grounds", "Courtyard"}
KEYS = ["long jump", "BLJ", "dive", "rollout", "wall kick", "side flip", "backflip", "triple jump", "double jump",
        "single jump", "jump kick", "slide kick", "ground pound", "punch", "butt slide", "crouch slide", "bonk"]


def stats(rows, max_stars):
    out = {}
    for part in ("course", "hub"):
        out[part] = {"frames": 0, "hs": 0.0, "path": 0.0, "air": 0, "n": {k: 0 for k in KEYS}}
    prev = None
    for r in rows:
        if r["stars"] >= max_stars and prev is not None and prev["stars"] >= max_stars:
            break
        ln = T.lname(r["level"])
        part = "hub" if ln in HUB else "course" if ln in T.COURSE_LEVELS else None
        if part and r["ctrl"]:
            o = out[part]
            o["frames"] += 1
            o["hs"] += r["hs"]
            o["air"] += bool(r["act"] & 0x800)
            if prev is not None and prev["ctrl"] and (prev["level"], prev["area"]) == (r["level"], r["area"]):
                d = math.dist(prev["pos"], r["pos"])
                if d < 2000:
                    o["path"] += d
            if prev is not None and r["act"] != prev["act"]:
                name = T.act_name(r["act"])
                for k, names in T.MOVES.items():
                    if name in names and k in o["n"]:
                        o["n"][k] += 1
            if prev is not None and T.is_blj(prev, r):
                o["n"]["BLJ"] += 1
        prev = r
    return out


def main():
    max_stars = int(sys.argv[1])
    res = []
    for a in sys.argv[2:]:
        label, path = a.split("=", 1)
        res.append((label, stats(T.load(path), max_stars)))
    for part in ("course", "hub"):
        print(f"\n### {part} levels (rates per 1000 controllable frames)\n")
        print("| run | frames | mean hspeed | path u/f | air % | " + " | ".join(KEYS) + " |")
        print("|---|" + "---|" * (4 + len(KEYS)))
        for label, s in res:
            o = s[part]
            f = max(1, o["frames"])
            print(f"| {label} | {o['frames']} | {o['hs'] / f:.1f} | {o['path'] / f:.1f} | {100 * o['air'] / f:.0f} | "
                  + " | ".join(f"{1000 * o['n'][k] / f:.1f}" for k in KEYS) + " |")


if __name__ == "__main__":
    main()
