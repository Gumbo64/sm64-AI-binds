#!/usr/bin/env python3
"""Where a route wastes time: tools/route_waste.py <run> [node]
Replays nothing; uses the route's per-frame positions from the bench's
trace file (obrrt-route writes trace.csv) or, if missing, the route states.
Reports loops: frames t' < t where Mario is back within R of where he was,
in the same area, with more than MIN frames between."""
import csv, math, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
P = [(int(r['f']), r['zone'], float(r['x']), float(r['y']), float(r['z']), r['ctl'] == '1') for r in rows]
R, MIN = 120.0, 45
loops = []
i = 0
n = len(P)
# Greedy: for each t, the earliest t' (>= last loop end) within R in the same zone.
t = 0
last_end = 0
while t < n:
    f, z, x, y, zz, c = P[t]
    best = None
    for s in range(last_end, max(last_end, t - MIN)):
        f2, z2, x2, y2, zz2, c2 = P[s]
        if z2 == z and math.dist((x, y, zz), (x2, y2, zz2)) < R:
            best = s
            break
    if best is not None:
        loops.append((P[best][0], f, f - P[best][0], z))
        last_end = t
    t += 1
loops.sort(key=lambda l: -l[2])
tot = sum(l[2] for l in loops)
print(f"route frames {P[-1][0] - P[0][0]}, loop frames {tot} ({100 * tot / max(1, P[-1][0] - P[0][0]):.0f}%) in {len(loops)} loops")
for a, b, d, z in loops[:15]:
    print(f"  frames {a}..{b}: {d} frames ({d / 30:.1f} s) back to the same spot, zone {z}")
