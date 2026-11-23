#!/usr/bin/env python3
"""The wall: BLJ and hyperspeed recovery on held-out routes. Starts in the BLJ
stretch of the TAS (f22548-22640) and of King 3557's route (BoB's Blj and Redirect
legs, found by speed > 100); K frames off the trace (0 = none), then follow D
frames. Follower (argmax), blind replay, seeded GA. Also the share of the
reference's speed kept at the end.

    .venv/bin/python rlv2/eval_blj.py checkpoints/rlv2/fol-r3.s.pt
"""
import argparse, json, os
import numpy as np, torch
import follow_lib as L, routes as R
from eval_follow import Args, replay_arm, rollout
from train_follow import Games

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--out", default=None); ap.add_argument("--ga", type=int, default=1)
a = ap.parse_args()
dev = torch.device("cuda"); ck = torch.load(a.ckpt, map_location=dev, weights_only=False); ca = ck["args"]
Args.rt, Args.ref, Args.copy, Args.hybrid = None, bool(ca.get("ref")), bool(ca.get("copy")), 0
Args.full, Args.score, Args.matching = bool(ca.get("full")), True, float(ca.get("matching", 0.0))
pol = L.FollowNet(d_z=L.N_FULL if Args.full else 0, h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref, copy=Args.copy).to(dev); pol.load_state_dict(ck["policy"]); pol.eval()
sm = R.setup(); E = 48; pool = R.make_pool(sm, E + 1, a.threads)
tas = R.read_movie(os.path.join(R.V2, L.TAS))
tp = L.TracePool(pool, E, [("TAS-bob", tas[: L.TAS_HELD_WINDOW[1]], L.TAS_HELD_WINDOW[0])] + L.movies(L.FAST_HELD), every=2, name="blj")
g = Games(pool, list(range(E)), tp, Args())
# BLJ state: hyperspeed in control, on these held-out routes.
hyper = [s for s in tp.saves if abs(tp.speed[s[0]]) > 100 and tp.control[s[0]] and tp.end[s[0]] - s[0] > 80]
# And the setup just before it: within 60 frames ahead of a first hyper frame.
hset = np.array([s[0] for s in hyper])
near = [s for s in tp.saves if tp.control[s[0]] and abs(tp.speed[s[0]]) <= 100 and len(hset) and ((hset > s[0]) & (hset - s[0] <= 40)).any() and tp.end[s[0]] - s[0] > 80]
print("hyper starts", len(hyper), "setup starts", len(near), flush=True)
rng = np.random.default_rng(4)
res = {}
for name, pool_s in (("hyperspeed", hyper), ("blj-setup", near)):
    idx = rng.choice(len(pool_s), min(48, len(pool_s)), replace=False)
    base = [pool_s[i] for i in idx]
    for K in (0, 1, 3, 8):
        for D in (32, 64):
            cell = {}
            for arm in ("policy", "replay") + (("ga",) if a.ga and K else ()):
                ch = base[:E]
                js = list(range(len(ch))); ids = [g.ids[j] for j in js]
                pool.load(ids, [c[1] for c in ch])
                if K:
                    pr = np.random.default_rng(K + D)
                    yw = pr.uniform(0, 2 * np.pi, len(ids)); x = (64 * np.sin(yw)).astype(np.int64) & 0xFF; y = (64 * np.cos(yw)).astype(np.int64) & 0xFF
                    pool.step(ids, np.repeat(((x << 8) | y).astype(np.uint32)[:, None], K, 1))
                ts = [int(min(c[0] + K, tp.end[c[0]] - 1)) for c in ch]
                if arm == "ga":
                    hs = pool.save(ids)
                    goals = np.stack([tp.pos[min(t + D, int(tp.end[t]))] for t in ts]).astype(np.float32)
                    rr = pool.attempt_to(hs, goals, [tp.pads[t : t + D].astype(np.uint32) for t in ts], 6.0, 11)
                    cell[arm] = dict(end=round(float(np.mean([bool(q["success"]) and q["handle"] is not None for q in rr])), 3), ga_frames=int(np.mean([int(q["frames"]) for q in rr])))
                    pool.free(hs)
                    continue
                g.bind(js, ts, [16] * len(ch), [D] * len(ch), [100] * len(ch), None)
                out = replay_arm(g, tp, js, ts) if arm == "replay" else rollout(g, pol, js)
                rows = [out[j] for j in js]
                cell[arm] = dict(end=round(float(np.mean([r["ok"] for r in rows])), 3), prog=round(float(np.mean([1.0 if r["ok"] else r["i"] / max(r["n"] - 1, 1) for r in rows])), 3), speed=round(float(np.mean([r["speed"] for r in rows if r["scored"]] or [0])), 3), n=len(rows))
            res[f"{name}/K{K}/D{D}"] = cell
            print(json.dumps({"set": name, "K": K, "D": D, **cell}), flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
