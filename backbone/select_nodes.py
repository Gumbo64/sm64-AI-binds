#!/usr/bin/env python3
"""Picks which tree edges the dataset extractor records.

    select_nodes.py <run_dir> <out select.bin> [target samples=1000000] [seed=0]

Level-balanced (sm64_AI_backbone.md 5.1: castle and BoB would dominate): each
level gets an equal share of the sample budget, filled with random edges of
that level; a level with fewer frames than its share gives all it has. The
route to the run's best node (most stars, soonest) is always recorded whole:
it is the search's winner, used for policy targets. Each selected edge's
parent edge also records its last 64 frames (mode 2) so k=64 histories exist.
Output is sorted in DFS preorder so the extractor's shards overlap little.
"""

import sys

import numpy as np
import pandas as pd

TAIL = 64


def preorder(parent):
    n = len(parent)
    kids = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    order = np.empty(n, np.int64)
    stack, k = [0], 0
    while stack:
        v = stack.pop()
        order[v] = k
        k += 1
        stack.extend(reversed(kids[v]))
    return order


def main():
    run, out = sys.argv[1], sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 1_000_000
    rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    d = pd.read_csv(f"{run}/nodes.csv", usecols=["id", "parent", "level", "frame", "stars"])
    par, fr = d.parent.values, d.frame.values
    length = np.where(par >= 0, fr - fr[np.maximum(par, 0)], 0)
    mode = np.zeros(len(d), np.uint8)

    best = d[d.stars == d.stars.max()].sort_values("frame").id.iloc[0]
    v = best
    while v > 0:
        mode[v] = 1
        v = par[v]
    route = int((mode == 1).sum())

    levels = sorted(d.level.unique())
    share = target / len(levels)
    for lv in levels:
        ids = np.flatnonzero((d.level.values == lv) & (length >= 8) & (mode == 0))
        rng.shuffle(ids)
        # Samples per edge: len/4, plus up to TAIL/4 from the parent's tail.
        cost = length[ids] // 4 + TAIL // 4
        take = ids[: int(np.searchsorted(np.cumsum(cost), share)) + 1]
        mode[take] = 1
        print(f"level {lv:>2}: {len(take):>6} of {len(ids):>6} edges, ~{int(cost[: len(take)].sum()):>7} samples")
    full = np.flatnonzero(mode == 1)
    tails = np.setdiff1d(par[full][par[full] >= 0], full)
    mode[tails] = 2
    sel = np.flatnonzero(mode > 0)
    sel = sel[np.argsort(preorder(par)[sel])]
    (sel.astype(np.uint32) << 2 | mode[sel].astype(np.uint32)).tofile(out)
    print(f"{len(full)} full edges ({route} on the route to node {best}), {len(tails)} tails -> {out}")


if __name__ == "__main__":
    main()
