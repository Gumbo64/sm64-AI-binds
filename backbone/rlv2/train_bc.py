#!/usr/bin/env python3
"""Hindsight BC of the follower, GPU only (no emulator once the traces are
cached): waypoints from each frame's own future. Saves a checkpoint for
train_follow.py --init.

    .venv/bin/python rlv2/train_bc.py --name bc-1024 --steps 40000 --batch 16384
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import follow_lib as L
import mllog
import routes as R
from train_follow import CK, RADII, SPACINGS, load_traces

CACHE = os.path.join(R.BB, "data", "rlv2")


def cached(a):
    os.makedirs(CACHE, exist_ok=True)
    pt, pv = os.path.join(CACHE, f"train{a.tree_nodes}.npz"), os.path.join(CACHE, "held.npz")
    if not os.path.exists(pt):
        sm = R.setup()
        pool = R.make_pool(sm, 1, 2)
        a.every, a.tree_nodes = 0, a.tree_nodes
        tr, va = load_traces(pool, 0, a)
        tr.save_arrays(pt), va.save_arrays(pv)
        del pool
    return L.TracePool.from_arrays(pt), L.TracePool.from_arrays(pv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--fast-weight", type=float, default=8.0)
    ap.add_argument("--every", type=int, default=240)
    ap.add_argument("--tree-nodes", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    if a.fast:
        os.makedirs(CACHE, exist_ok=True)
        pt, pv = os.path.join(CACHE, f"fast-train{a.tree_nodes}.npz"), os.path.join(CACHE, "fast-held.npz")
        if not os.path.exists(pt):
            sm = R.setup()
            pool = R.make_pool(sm, 1, 2)
            tr, va = L.fast_sets(pool, 0, 0, a.tree_nodes)
            tr.save_arrays(pt), va.save_arrays(pv)
            del pool
        tr, va = L.TracePool.from_arrays(pt), L.TracePool.from_arrays(pv)
    else:
        tr, va = cached(a)
    print(f"train frames {len(tr.tid)}, held-out {len(va.tid)}", flush=True)
    tt, tv = L.TraceT(tr, dev, fast_weight=a.fast_weight, copy=a.copy), L.TraceT(va, dev, fast_weight=a.fast_weight, copy=a.copy)
    sp, rad = torch.as_tensor(SPACINGS, device=dev), torch.as_tensor(RADII, device=dev)
    pol = L.FollowNet(h=a.h, layers=a.layers, ref=a.ref, copy=a.copy).to(dev)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    t0 = time.time()
    best = 1e9
    mllog.start(a.name, vars(a))

    def loss_of(bt):
        x, ra, ac, fl, y = bt
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = pol(x, ra, ac, fl)
        return [F.cross_entropy(lg.float(), y[:, k]) for k, lg in enumerate(logits)], logits, y

    for i in range(a.steps):
        ls, _, _ = loss_of(tt.batch(a.batch, sp, rad, ref=a.ref))
        loss = sum(ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sch.step()
        if i % 2000 == 0 or i == a.steps - 1:
            with torch.no_grad():
                lv, lg, y = loss_of(tv.batch(a.batch, sp, rad, ref=a.ref))
            acc = [round((l.argmax(1) == y[:, k]).float().mean().item(), 3) for k, l in enumerate(lg)]
            vl = sum(x.item() for x in lv)
            print(json.dumps({"step": i, "train": round(loss.item(), 3), "val": round(vl, 3), "val_each": [round(x.item(), 3) for x in lv], "val_acc": acc, "it_s": round((i + 1) / (time.time() - t0), 1)}), flush=True)
            mllog.log({"train": loss.item(), "val": vl, "val_dir": lv[0].item(), "val_mag": lv[1].item(), "val_btn": lv[2].item(), "val_hold": lv[3].item(), "val_acc_dir": acc[0], "val_acc_btn": acc[2], "it_s": (i + 1) / (time.time() - t0)}, i)
            if vl < best:
                best = vl
                torch.save({"policy": pol.state_dict(), "args": vars(a), "val": vl, "step": i}, os.path.join(CK, f"{a.name}.pt"))
    torch.save({"policy": pol.state_dict(), "args": vars(a), "step": a.steps}, os.path.join(CK, f"{a.name}.last.pt"))


if __name__ == "__main__":
    main()
