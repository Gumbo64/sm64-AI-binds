#!/usr/bin/env python3
"""Is a fast stretch recoverable after a rewire at all? From K noise frames off a
held-out trace's fast or slow state, reach the trace's position D frames on:
the follower (one argmax rollout) against a seeded GA (a far larger budget).

    .venv/bin/python rlv2/eval_fast_ceiling.py checkpoints/rlv2/fol-r3.s.pt
"""
import argparse, json, os
import numpy as np, torch
import follow_lib as L, routes as R
from eval_follow import Args, _held_paths, is_fast, rollout, stratify, usable
from train_follow import Games

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--n", type=int, default=48); ap.add_argument("--scale", type=float, default=6.0)
a = ap.parse_args()
dev = torch.device("cuda"); ck = torch.load(a.ckpt, map_location=dev, weights_only=False); ca = ck["args"]
Args.rt, Args.ref, Args.copy, Args.hybrid = None, bool(ca.get("ref")), bool(ca.get("copy")), 0
pol = L.FollowNet(h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref, copy=Args.copy).to(dev); pol.load_state_dict(ck["policy"]); pol.eval()
sm = R.setup(); E = 48; pool = R.make_pool(sm, E + 1, a.threads)
tp = L.TracePool(pool, E, _held_paths(), every=32, name="held-out")
g = Games(pool, list(range(E)), tp, Args()); rng = np.random.default_rng(9)
for K, D in ((8, 64), (8, 128), (30, 128)):
    cand = [s for s in tp.saves if usable(tp, s[0], D + K)]
    picks = stratify(tp, cand, a.n, D + K, rng, True)
    rows = []
    for b in range(0, len(picks), E):
        ch = picks[b : b + E]; js = list(range(len(ch))); ids = [g.ids[j] for j in js]
        pr = np.random.default_rng(K * 13 + D + b)
        yw = pr.uniform(0, 2 * np.pi, len(ids)); x = (64 * np.sin(yw)).astype(np.int64) & 0xFF; y = (64 * np.cos(yw)).astype(np.int64) & 0xFF
        pool.load(ids, [c[1] for c in ch])
        pads = np.repeat(((x << 8) | y).astype(np.uint32)[:, None], K, 1)
        pool.step(ids, pads)
        hs = pool.save(ids)
        ts = [c[0] + K for c in ch]
        g.bind(js, ts, [16] * len(ch), [D] * len(ch), [100] * len(ch), None)
        res = rollout(g, pol, js)
        goals = np.stack([tp.pos[min(t + D, int(tp.end[t]))] for t in ts]).astype(np.float32)
        seeds = [tp.pads[t : t + D].astype(np.uint32) for t in ts]
        ga = pool.attempt_to(hs, goals, seeds, a.scale, 7)
        for j, c, q in zip(js, ch, ga):
            r = res[j]
            rows.append(dict(fast=c[2], pol=r["ok"], pol_f=r["frames"], ga=bool(q["success"]) and q["handle"] is not None, ga_len=len(q["inputs"]), ga_frames=int(q["frames"])))
        pool.free(hs)
    for tag, q in (("fast", [r for r in rows if r["fast"]]), ("slow", [r for r in rows if not r["fast"]])):
        print(json.dumps(dict(K=K, D=D, tag=tag, n=len(q), follower=round(np.mean([r["pol"] for r in q]), 3), ga=round(np.mean([r["ga"] for r in q]), 3), ga_frames=int(np.mean([r["ga_frames"] for r in q])), follower_frames=int(np.median([r["pol_f"] for r in q if r["pol"]] or [0])), ga_len=int(np.median([r["ga_len"] for r in q if r["ga"]] or [0])))), flush=True)
