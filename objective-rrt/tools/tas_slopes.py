#!/usr/bin/env python3
"""Slope-speed analysis of a `tas_trace` dump (run/examples/tas_trace.rs).

    tools/tas_slopes.py phases <trace.csv> <from> <to>
        one line per action change: frame range, action, start/end position,
        speed, floor slope/type, peak height, Lakitu box flags
    tools/tas_slopes.py scan <trace.csv> [--md out.md]
        every slope-speed event in the trace:
          SLIDE   a slide action (dive/butt/stomach/crouch slide) that gains
                  >= 20 u/f on slopes, with the air action it ends in
          SURF    >= 3 landing-action frames (FREEFALL_LAND, JUMP_LAND, ...)
                  at >= 60 u/f on a slope: speed kept on the ground uncapped
          BIGJUMP a jump/double/triple/steep jump taken at >= 60 u/f (vy0 >= 57)
        BLJ / hyperspeed (|fvel| > 150) frames are excluded.

Ground frames are the trace's `grounded` column (action without ACT_FLAG_AIR).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import tas_analyse as T
    LEVELS = T.levels()
except Exception:  # decomp missing: numbers only
    LEVELS = {}

SLIDES = {"DIVE_SLIDE", "BUTT_SLIDE", "STOMACH_SLIDE", "CROUCH_SLIDE", "HOLD_BUTT_SLIDE", "HOLD_STOMACH_SLIDE",
          "SLIDE_KICK_SLIDE", "BEGIN_SLIDING", "HOLD_BEGIN_SLIDING"}
LANDINGS = {"FREEFALL_LAND", "JUMP_LAND", "DOUBLE_JUMP_LAND", "TRIPLE_JUMP_LAND", "SIDE_FLIP_LAND", "LONG_JUMP_LAND",
            "HOLD_FREEFALL_LAND", "HOLD_JUMP_LAND", "BACKFLIP_LAND"}
JUMPS = {"JUMP", "DOUBLE_JUMP", "TRIPLE_JUMP", "STEEP_JUMP", "HOLD_JUMP", "SPECIAL_TRIPLE_JUMP", "FLYING_TRIPLE_JUMP"}


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("x", "y", "z", "vy", "fvel", "hspeed", "floor_y", "ny", "slope_deg", "air_peak", "h_floor"):
                r[k] = float(r[k])
            for k in ("frame", "level", "area", "grounded", "in_box", "caught", "lakitu", "stars", "act_timer"):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def lname(n):
    return LEVELS.get(n, (None, str(n)))[1]


def pos(r):
    return f"({r['x']:.0f}, {r['y']:.0f}, {r['z']:.0f})"


def phases(rows, lo, hi):
    seg = [r for r in rows if lo <= r["frame"] <= hi]
    out = []
    i = 0
    while i < len(seg):
        j = i
        while j + 1 < len(seg) and seg[j + 1]["act_name"] == seg[i]["act_name"]:
            j += 1
        a, b = seg[i], seg[j]
        peak = max(r["y"] for r in seg[i:j + 1])
        slope = max(r["slope_deg"] for r in seg[i:j + 1])
        box = sum(r["in_box"] for r in seg[i:j + 1])
        caught = sum(r["caught"] for r in seg[i:j + 1])
        out.append(f"{a['frame']:6}-{b['frame']:<6} {lname(a['level'])}/{a['area']} {a['act_name']:<20} {pos(a):>22} -> {pos(b):<22} "
                   f"fv {a['fvel']:6.1f}->{b['fvel']:6.1f} vy0 {a['vy']:6.1f} peak {peak:6.0f} slope<={slope:4.1f} "
                   f"floor {a['floor_type']}{'  box' + str(box) if box else ''}{' CAUGHT' + str(caught) if caught else ''}"
                   f"{'' if a['lakitu'] < 0 else ' lakitu' + str(a['lakitu'])}")
        i = j + 1
    return out


def scan(rows):
    ev = []
    n = len(rows)
    i = 0
    while i < n:
        r = rows[i]
        if abs(r["fvel"]) > 150:
            i += 1
            continue
        name = r["act_name"]
        # SLIDE: a run of slide frames gaining speed on slopes.
        if name in SLIDES and r["grounded"]:
            j = i
            while j + 1 < n and rows[j + 1]["act_name"] in SLIDES and rows[j + 1]["grounded"]:
                j += 1
            seg = rows[i:j + 1]
            v0 = min(abs(s["fvel"]) for s in seg[:3])
            v1 = max(abs(s["fvel"]) for s in seg)
            sl = max(s["slope_deg"] for s in seg)
            if v1 - v0 >= 20 and v1 <= 150 and sl >= 5:
                nxt = rows[j + 1] if j + 1 < n else seg[-1]
                ev.append(("SLIDE", r, f"{'+'.join(sorted({s['act_name'] for s in seg}))} {v0:.0f} -> {v1:.0f} u/f in "
                           f"{len(seg)} f, max slope {sl:.0f} deg (floor {seg[len(seg)//2]['floor_type']}), "
                           f"{pos(r)} -> {pos(seg[-1])}, exit {nxt['act_name']} at {abs(nxt['fvel']):.0f}"))
            i = j + 1
            continue
        # SURF: landing actions kept at speed on a slope.
        if name in LANDINGS and r["grounded"] and abs(r["fvel"]) >= 60 and r["slope_deg"] >= 10:
            j = i
            while j + 1 < n and rows[j + 1]["grounded"] and rows[j + 1]["act_name"] in LANDINGS | {"FREEFALL_LAND_STOP"} \
                    and abs(rows[j + 1]["fvel"]) >= 40:
                j += 1
            seg = rows[i:j + 1]
            if len(seg) >= 3 and max(abs(s["fvel"]) for s in seg) <= 150:
                dy = seg[-1]["y"] - seg[0]["y"]
                ev.append(("SURF", r, f"{len(seg)} landing frames at {abs(seg[0]['fvel']):.0f} -> {abs(seg[-1]['fvel']):.0f}, "
                           f"slope {max(s['slope_deg'] for s in seg):.0f} deg, dy {dy:+.0f} ({'uphill' if dy > 0 else 'downhill'}), "
                           f"{pos(seg[0])} -> {pos(seg[-1])}"))
            i = j + 1
            continue
        # BIGJUMP: jump actions entered at high speed.
        if name in JUMPS and i > 0 and rows[i - 1]["act_name"] != name and rows[i - 1]["grounded"]:
            p = rows[i - 1]
            if 60 <= abs(p["fvel"]) <= 150 and r["vy"] + 4 >= 50:
                j = i
                while j + 1 < n and not rows[j + 1]["grounded"]:
                    j += 1
                seg = rows[i:j + 1]
                peak = max(s["y"] for s in seg)
                land = rows[j + 1] if j + 1 < n else seg[-1]
                d = ((land["x"] - p["x"]) ** 2 + (land["z"] - p["z"]) ** 2) ** 0.5
                ev.append(("BIGJUMP", r, f"{name} from {p['act_name']} at {abs(p['fvel']):.0f} (vy0 {r['vy'] + 4:.0f}), "
                           f"{pos(p)} -> {pos(land)}, peak +{peak - p['y']:.0f}, dist {d:.0f} in {len(seg)} f"))
        i += 1
    return ev


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    rows = load(sys.argv[2])
    if sys.argv[1] == "phases":
        for l in phases(rows, int(sys.argv[3]), int(sys.argv[4])):
            print(l)
        return
    ev = scan(rows)
    md = None
    if "--md" in sys.argv:
        md = open(sys.argv[sys.argv.index("--md") + 1], "w")
        md.write("# Slope-speed events in the 2026 120-star TAS\n\nGenerated by `tools/tas_slopes.py scan "
                 f"{os.path.basename(sys.argv[2])}`. Kinds: SLIDE (slide gaining >= 20 u/f on a slope), SURF (>= 3 "
                 "landing-action frames at >= 60 u/f on a slope), BIGJUMP (jump taken at >= 60 u/f). BLJ/hyperspeed "
                 "frames excluded.\n\n| Frame | Time | Level | Kind | Detail |\n|---|---|---|---|---|\n")
    for kind, r, d in ev:
        t = f"{r['frame'] // 1800}:{r['frame'] % 1800 / 30:04.1f}"
        print(f"{r['frame']:6} {t:>7} {lname(r['level'])}/{r['area']:<3} {kind:<7} {d}")
        if md:
            md.write(f"| {r['frame']} | {t} | {lname(r['level'])}/{r['area']} | {kind} | {d} |\n")
    counts = {}
    for kind, r, _ in ev:
        counts[(kind, lname(r["level"]))] = counts.get((kind, lname(r["level"])), 0) + 1
    print("counts:", sorted(counts.items(), key=lambda kv: -kv[1]))
    if md:
        md.write("\nCounts by kind and level: " + ", ".join(f"{k} {l} {c}" for (k, l), c in sorted(counts.items(), key=lambda kv: -kv[1])) + "\n")


if __name__ == "__main__":
    main()
