#!/usr/bin/env python3
"""What the mender's `Follower` must deliver, measured without the mender: from
a route's own state at a, pass the route's position at b (radius 60, one
waypoint per 16 frames) strictly before the route does. Argmax, one rollout per
window, on held-out routes; fast and slow windows apart.

    .venv/bin/python rlv2/eval_gate.py checkpoints/rlv2/fol-g.best.pt --out logs/rlv2/gate.json
"""

import argparse
import json
import os

import numpy as np
import torch

import follow_lib as L
import routes as R
from eval_follow import Args, is_fast, rollout, usable
from train_follow import Games

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--n", type=int, default=128)
a = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
ca = ck["args"]
Args.rt, Args.ref, Args.copy, Args.hybrid = None, bool(ca.get("ref")), bool(ca.get("copy")), 0
Args.full, Args.score, Args.matching = bool(ca.get("full")), True, float(ca.get("matching", 0.0))
pol = L.FollowNet(d_z=L.N_FULL if Args.full else 0, h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref, copy=Args.copy).to(dev)
pol.load_state_dict(ck["policy"])
pol.eval()
sm = R.setup()
E = 64
pool = R.make_pool(sm, E + 1, a.threads)
tas = R.read_movie(os.path.join(R.V2, L.TAS))
tp = L.TracePool(pool, E, L.movies(L.FAST_HELD) + [("TAS-bob", tas[: L.TAS_HELD_WINDOW[1]], L.TAS_HELD_WINDOW[0])], every=8, name="gate")
g = Games(pool, list(range(E)), tp, Args())
rng = np.random.default_rng(3)
res = {}
for W in (32, 64, 128, 256):
    cand = [s for s in tp.saves if usable(tp, s[0], W)]
    idx = rng.choice(len(cand), min(a.n, len(cand)), replace=False)
    rows = []
    for b in range(0, len(idx), E):
        ch = [cand[i] for i in idx[b : b + E]]
        js = list(range(len(ch)))
        g.bind(js, [c[0] for c in ch], [16] * len(ch), [W] * len(ch), [60] * len(ch), [c[1] for c in ch])
        out = rollout(g, pol, js)
        rows += [dict(ok=out[j]["ok"], lead=out[j]["ref"] - out[j]["frames"], fast=is_fast(tp, c[0], W)) for j, c in zip(js, ch)]
    res[W] = {}
    for tag, q in (("all", rows), ("fast", [r for r in rows if r["fast"]]), ("slow", [r for r in rows if not r["fast"]])):
        if q:
            ok = [r for r in q if r["ok"]]
            ah = [r for r in ok if r["lead"] > 0]
            res[W][tag] = dict(n=len(q), reached=round(len(ok) / len(q), 3), ahead=round(len(ah) / len(q), 3), med_lead=float(np.median([r["lead"] for r in ah])) if ah else None)
    print(json.dumps({"window": W, **res[W]}), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
