#!/usr/bin/env python3
"""ETA calibration: how honest each objective kind's ETA is.

    python3 tools/analysis/eta_calib.py <run dir> [min successes=3]

Reads <run dir>/events.jsonl (v2 runs write, per attempted stage, the start
ETA `start_eta` and the committed frames `seg_frames`) and prints per kind:
successes, and the median, p10 and p90 of actual frames / start ETA over
its successes. A ratio well above 1 means the kind's ETA is optimistic (its
phase constants or pace too small); well below 1, pessimistic. Successes
with no or a zero start ETA are skipped (counted in `skipped`).
"""

import json
import sys
from collections import defaultdict


def pct(xs, p):
    """The p-th percentile of sorted xs (nearest rank)."""
    i = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[i]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run = sys.argv[1].rstrip("/")
    min_n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ratios = defaultdict(list)
    skipped = defaultdict(int)
    attempts = defaultdict(int)
    with open(f"{run}/events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            kind = e.get("kind")
            if kind is None:
                continue
            attempts[kind] += 1
            if not e.get("success"):
                continue
            eta, frames = e.get("start_eta"), e.get("seg_frames")
            if eta is None or frames is None or eta <= 0:
                skipped[kind] += 1
                continue
            ratios[kind].append(frames / eta)
    kinds = sorted(set(ratios) | set(skipped), key=lambda k: -len(ratios[k]))
    print(f"{'kind':<34} {'att':>6} {'succ':>5} {'skip':>5} {'median':>7} {'p10':>7} {'p90':>7}")
    for k in kinds:
        xs = sorted(ratios[k])
        if len(xs) < min_n:
            print(f"{k:<34} {attempts[k]:>6} {len(xs):>5} {skipped[k]:>5} {'-':>7} {'-':>7} {'-':>7}")
            continue
        print(f"{k:<34} {attempts[k]:>6} {len(xs):>5} {skipped[k]:>5} {pct(xs, 50):>7.2f} {pct(xs, 10):>7.2f} {pct(xs, 90):>7.2f}")


if __name__ == "__main__":
    main()
