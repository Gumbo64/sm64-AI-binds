#!/usr/bin/env python3
"""A follower along whole held-out routes, deterministically: from the first
control state of a stretch, reach the end of it through waypoints at each
spacing. Stretches: the TAS's BoB speed chain + BLJ (never trained on), a
walking route, a tree's first star.

    .venv/bin/python rlv2/eval_whole.py checkpoints/rlv2/fol-r.best.pt
"""

import argparse
import json
import os

import numpy as np
import torch

import follow_lib as L
import routes as R
from eval_follow import Args, batched, rollout
from train_follow import Games

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
ca = ck["args"]
Args.rt = None
if ca.get("z_ckpt"):
    from sm64bb.runtime import Runtime

    Args.rt = Runtime(ca["z_ckpt"])
Args.ref = bool(ca.get("ref"))
pol = L.FollowNet(d_z=Args.rt.m.d if Args.rt else 0, h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref).to(dev)
pol.load_state_dict(ck["policy"])
pol.eval()
sm = R.setup()
E = 8
pool = R.make_pool(sm, E + 1, a.threads)
cases = [("tas-bob-speed-blj", os.path.join(R.V2, L.TAS), (22397, 22640), 22397, 243),
         ("walk-route-bob", os.path.join(R.V2, "oss2/runs/legs-fast-king-walk-s3/route.m64"), None, None, 800),
         ("q15-s3-first-star", os.path.join(R.V2, "runs/q15-pw1-s3/frontier-k0-s1.m64"), None, None, 1600)]
for name, path, win, st, H in cases:
    tp = L.TracePool(pool, E, [path], every=8, skip_common=False, tas_window=win, name=name)
    if win is None:
        # The last H frames of the route, in control.
        last = len(tp.tid) - 1
        s0 = [s for s in tp.saves if s[0] >= last - H]
    else:
        s0 = tp.saves
    t0, h0 = s0[0]
    g = Games(pool, list(range(E)), tp, Args())
    for S in (4, 8, 16, 32, 64, 128):
        if H / S > L.MAX_WP - 1:
            continue
        out = {}
        for arm in ("policy", "steer"):
            r = batched(g, tp, pol, [(t0, h0, False)], S, H, 100, arm)[0]
            out[arm] = dict(ok=r["ok"], prog=round(r["i"] / max(r["n"] - 1, 1), 3), frames=r["frames"], ref=r["ref"], term=r["term"])
        print(json.dumps({"case": name, "S": S, **out}), flush=True)
