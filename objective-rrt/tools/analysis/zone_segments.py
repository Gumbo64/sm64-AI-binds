#!/usr/bin/env python3
"""Segments between zone entries on a run's best route.

    tools/analysis/zone_segments.py runs/<run> [runs/<run2> ...] [--stars N]

The route is the soonest node with the most stars (or N), walked up to the
root. Each route node whose (level, area, stars) differs from its parent's
starts a segment (start -> castle grounds -> lobby 6:1 -> BoB 9:1 ...).
Printed per segment: the zone entered, its frame on the route, the frames
the segment took, and the soonest tree node with the same (level, area,
stars) (the first arrival anywhere, what grafting could reach), so the
route's slack per section shows. Several runs print side by side.
"""
import argparse, csv, os


def load(run):
    nodes = {}
    with open(os.path.join(run, "nodes.csv")) as f:
        for r in csv.DictReader(f):
            nodes[int(r["id"])] = (int(r["parent"]), int(r["level"]), int(r["area"]), int(r["stars"]), int(r["frame"]), r["via"])
    return nodes


def segments(nodes, stars=None):
    top = max(n[3] for n in nodes.values()) if stars is None else stars
    cand = [(n[4], i) for i, n in nodes.items() if n[3] >= top]
    if not cand:
        return top, []
    _, end = min(cand)
    route = []
    at = end
    while at >= 0:
        route.append(at)
        at = nodes[at][0]
    route.reverse()
    first = {}
    for n in nodes.values():
        k = (n[1], n[2], n[3])
        first[k] = min(first.get(k, 1 << 31), n[4])
    out = []
    prev_frame = nodes[route[0]][4]
    prev = None
    for i in route:
        n = nodes[i]
        k = (n[1], n[2], n[3])
        if k != prev:
            out.append((k, n[4], n[4] - prev_frame, first[k], n[5]))
            prev_frame, prev = n[4], k
    out.append((("end",), nodes[end][4], nodes[end][4] - prev_frame, nodes[end][4], nodes[end][5]))
    return top, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--stars", type=int)
    a = ap.parse_args()
    for run in a.runs:
        top, segs = segments(load(run), a.stars)
        print(f"== {run}: route to {top} stars, {len(segs) - 1} zone entries")
        print(f"   {'zone (lvl:area st)':<20} {'route frame':>11} {'segment':>8} {'first in tree':>13}  via")
        for k, fr, seg, first, via in segs:
            z = "end" if k == ("end",) else f"{k[0]}:{k[1]} {k[2]}*"
            print(f"   {z:<20} {fr:>11} {seg:>8} {first:>13}  {via}")


if __name__ == "__main__":
    main()
