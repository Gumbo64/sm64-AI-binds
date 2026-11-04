#!/usr/bin/env python3
"""Time-to-star quality on a tree the model never trained on:

    .venv/bin/python eval_star.py checkpoints/last.pt [tree=pf-mgel-lg-pf-s1] [n=4000]

For states of that tree's extracted data, the labels are the frames until
the earliest descendant collecting each star (Store.star_labels). Reports
the model's rank correlation and |log ratio| by horizon and star count gap.
"""

import os
import sys

import numpy as np
import torch

from sm64bb.data import Store
from sm64bb.runtime import Runtime
from train import RUNS_DIR, spearman

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ck = sys.argv[1]
    tree = sys.argv[2] if len(sys.argv) > 2 else "pf-mgel-lg-pf-s1"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    st = Store([(os.path.join(HERE, "data", tree), os.path.join(RUNS_DIR, tree, "nodes.csv")),
                (os.path.join(HERE, "data", "tas120"), None)]).star_labels()
    tree_rows = np.flatnonzero((st.node[np.repeat(np.arange(st.N), np.diff(st.lab_off))] >= 0))
    rng = np.random.default_rng(0)
    pick = rng.choice(tree_rows, min(n, len(tree_rows)), replace=False)
    samp = np.searchsorted(st.lab_off, pick, side="right") - 1
    rt = Runtime(ck)
    z = rt.encode(st.state(samp))
    with torch.no_grad():
        d = rt.m.distance_star(z, torch.as_tensor(st.lab_x[pick], device=rt.dev)).cpu()
    t = torch.as_tensor(st.lab_t[pick])
    print(f"{ck} (step {rt.step}) on held-out tree {tree}: {len(pick)} (state, star) labels")
    print(f"  all: spearman {spearman(d, t):.3f}, |log ratio| {(torch.log1p(d) - torch.log1p(t)).abs().mean():.2f}")
    for lo, hi in [(0, 300), (300, 1500), (1500, 1e9)]:
        m = (t >= lo) & (t < hi)
        if m.sum() > 20:
            print(f"  {int(lo):>5}-{'' if hi > 1e8 else int(hi):<5} frames: spearman {spearman(d[m], t[m]):.3f}, "
                  f"median pred/true {float((d[m] / t[m]).median()):.2f}  (n={int(m.sum())})")


if __name__ == "__main__":
    main()
