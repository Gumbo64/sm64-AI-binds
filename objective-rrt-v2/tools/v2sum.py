#!/usr/bin/env python3
"""Star summary per run: tools/v2sum.py run [run ...] (names under runs/, or paths).

Per run: nodes, stars at the end, and the game time each star count was first
reached (the earliest node with at least that many stars), plus the mean over
the runs given of the star count and of the first-star time.
"""

import csv
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


def clock(f):
    if f is None:
        return "—"
    t = round(f / 30, 1)
    return f"{int(t // 60)}:{t % 60:04.1f}"


def load(run):
    d = run if os.path.isdir(run) else os.path.join(ROOT, run)
    best = {}
    n = 0
    with open(os.path.join(d, "nodes.csv")) as fh:
        for r in csv.DictReader(fh):
            n += 1
            s, f = int(r["stars"]), int(r["frame"])
            for k in range(1, s + 1):
                if k not in best or f < best[k]:
                    best[k] = f
    return n, best


def main():
    stars, firsts = [], []
    for run in sys.argv[1:]:
        n, best = load(run)
        top = max(best, default=0)
        stars.append(top)
        if 1 in best:
            firsts.append(best[1])
        times = "  ".join(f"#{k} {clock(best[k])}" for k in sorted(best))
        print(f"{os.path.basename(run.rstrip('/')):28} nodes {n:7} stars {top}  {times}")
    if len(sys.argv) > 2:
        first = clock(sum(firsts) / len(firsts)) if firsts else "—"
        print(f"{'mean':28} stars {sum(stars) / len(stars):.2f}  first star {first} ({len(firsts)}/{len(stars)} runs)")


if __name__ == "__main__":
    main()
