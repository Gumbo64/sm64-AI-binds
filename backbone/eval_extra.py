#!/usr/bin/env python3
"""Detailed checks of the goal-conditioned heads on the validation split:

    .venv/bin/python eval_extra.py [checkpoints/best_tas.pt] [batches=40] [device=cpu]

Uses the goal tokens as training does from v3 on (GOAL_REGION and
GOAL_STATE; GOAL_OBJECTIVE off, see train.py).
- Policy: accuracy against the majority-class baseline, split by whether the
  goal lies in the anchor's own edge or in a later edge.
- Reachability: AUC of positives against each negative type (fewer stars,
  TAS budget < 50 %, tree budget < 1/8), and against all budget negatives.
- Distance: rank correlation and log-ratio error per source and horizon.
"""

import os
import sys

import numpy as np
import torch

from sm64bb.data import Batcher, Store
from sm64bb.model import Backbone
from sm64bb.schema import Schema
from train import RUN, SOURCES, auc, cat_states, spearman, to_dev


@torch.no_grad()
def main():
    ck_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_tas.pt"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    dev = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    ck = torch.load(ck_path, map_location="cpu")
    cfg, sch = ck["cfg"], Schema()
    m = Backbone(sch, len(ck["kinds"]), cfg["d"], cfg["layers"], cfg["heads"]).to(dev).eval()
    m.load_state_dict(ck["model"])
    va = Store(SOURCES, os.path.join(RUN, "nodes.csv"), sch).setup_pairs("val")
    gen = Batcher(va, 64, seed=99)
    P = {k: [] for k in ["hit", "lab", "same"]}
    R = {k: [] for k in ["logit", "lab", "type"]}
    D = {k: [] for k in ["pred", "true", "tight"]}
    for _ in range(n):
        bt = to_dev(gen.batch(), dev)
        B = bt["k"].shape[0]
        z, h, pm = m.enc(cat_states(bt["a"], bt["g"]))
        za, zg, ha, pma = z[:B], z[B:], h[:B], pm[:B]
        keep = torch.ones(B, 3, dtype=torch.bool, device=dev)
        keep[:, 1] = False
        gt, gm = m.goal_tokens(bt["a"], bt["g"], zg, bt["g_kind"], keep)
        rl, pol = m.goal_tail(ha, pma, gt, gm, bt["budget"])
        pmask = bt["pmask_pol"] & (bt["pol"] > 0)
        P["hit"].append((pol[:, : m.n_kinds].argmax(-1) == bt["pol"])[pmask].cpu())
        P["lab"].append(bt["pol"][pmask].cpu())
        P["same"].append(bt["g_same_node"][pmask].cpu())
        rm = bt["rmask"]
        R["logit"].append(rl[rm].cpu())
        R["lab"].append(bt["reach"][rm].cpu())
        R["type"].append(bt["rtype"][rm].cpu())
        dm = bt["dist_mask"]
        D["pred"].append(m.distance(za, zg)[dm].cpu())
        D["true"].append(bt["dist"][dm].cpu())
        D["tight"].append(bt["tight"][dm].cpu())
    P, R, D = ({k: torch.cat(v) for k, v in X.items()} for X in (P, R, D))
    print(f"checkpoint {ck_path} (step {ck['step']}), {n} validation batches of 64")
    lab = P["lab"]
    print(f"\npolicy (n={len(lab)}, {len(ck['kinds'])} kinds)")
    print(f"  majority-class baseline   {torch.bincount(lab).max().item() / len(lab):.3f}")
    print(f"  accuracy, all             {P['hit'].float().mean():.3f}")
    for name, sel in [("goal in own edge", P["same"]), ("goal in a later edge", ~P["same"])]:
        if sel.any():
            print(f"  accuracy, {name:<16} {P['hit'][sel].float().mean():.3f}  (n={int(sel.sum())})")
    print("\nreachability AUC (positives vs ...)")
    rt, rl, rlab = R["type"], R["logit"], R["lab"]
    for t, name in [(1, "fewer stars (trivial)"), (2, "TAS budget < 50%"), (3, "tree budget < 1/8"), (None, "all budget negatives")]:
        sel = (rt == 0) | ((rt != 1) if t is None else (rt == t))
        print(f"  {name:<24} {auc(rl[sel], rlab[sel]):.3f}  (negatives {int((sel & (rlab == 0)).sum())})")
    print("\ndistance")
    for name, sel in [("TAS (held-out levels)", D["tight"]), ("tree (held-out subtrees)", ~D["tight"])]:
        p, t = D["pred"][sel], D["true"][sel]
        for lo, hi in [(0, 64), (64, 1000), (1000, 1e9)]:
            b = (t >= lo) & (t < hi)
            if b.sum() >= 3:
                err = (torch.log1p(p[b]) - torch.log1p(t[b])).abs().mean()
                print(f"  {name:<24} {int(lo):>5}-{'' if hi > 1e8 else int(hi):<5} frames: spearman {spearman(p[b], t[b]):.3f}, "
                      f"mean |log ratio| {err:.2f}  (n={int(b.sum())})")
        print(f"  {name:<24} all: spearman {spearman(p, t):.3f}  (n={len(t)})")


if __name__ == "__main__":
    main()
