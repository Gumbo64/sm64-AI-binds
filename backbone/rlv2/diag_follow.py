#!/usr/bin/env python3
"""Why following fails: last state vs the waypoint it was after, by arm."""
import argparse, json, collections
import numpy as np, torch
import follow_lib as L, routes as R, policy as P
from eval_follow import Args, usable, rollout, batched
from train_follow import Games

ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--S", type=int, default=64); ap.add_argument("--H", type=int, default=256); ap.add_argument("--n", type=int, default=128)
a = ap.parse_args()
dev = torch.device("cuda")
ck = torch.load(a.ckpt, map_location=dev, weights_only=False); ca = ck["args"]
Args.rt = None; Args.ref = bool(ca.get("ref"))
pol = L.FollowNet(h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref).to(dev); pol.load_state_dict(ck["policy"]); pol.eval()
sm = R.setup(); E = 64; pool = R.make_pool(sm, E + 1, 3)
held = [m for r in L.HELD_TREES for m in L.tree_movies(r, 60, np.random.default_rng(11))]
tp = L.TracePool(pool, E, L.movies(L.HELD_OUT) + held, every=32, name="held-out")
rng = np.random.default_rng(5); g = Games(pool, list(range(E)), tp, Args())
cand = [s for s in tp.saves if usable(tp, s[0], a.H)]
idx = rng.choice(len(cand), a.n, replace=False)
tot = collections.Counter(); rows = []
for b in range(0, a.n, E):
    ch = [cand[i] for i in idx[b:b+E]]; js = list(range(len(ch)))
    g.bind(js, [c[0] for c in ch], [a.S]*len(ch), [a.H]*len(ch), [100]*len(ch), [c[1] for c in ch])
    res = rollout(g, pol, js)
    o = pool.tr_obs(g.ids[:len(ch)])
    for j, c in zip(js, ch):
        r = res[j]; f = o["f"][j]; wp = g.W[j, min(r["i"], g.n_wp[j]-1)]
        d = np.linalg.norm(f[:3]-wp[:3]); dy = wp[1]-f[1]
        e = min(int(tp.end[c[0]]), c[0]+a.H); seg = tp.action[c[0]:e+1]
        air = ((seg & 0x800) != 0).mean(); sp = np.percentile(np.abs(tp.speed[c[0]:e+1]), 75); inter = tp.f[c[0]:e+1, 23].max()
        lv = int(tp.level[c[0]])
        tag = "end" if r["ok"] else ("died" if r["term"] in (2,3) else ("up" if dy > 150 else "down" if dy < -150 else "flat"))
        tot[tag] += 1; tot[f"lv{lv}/{'ok' if r['ok'] else 'no'}"] += 1
        rows.append(dict(tag=tag, lv=lv, d=float(d), dy=float(dy), air=float(air), sp=float(sp), hold=float(inter), prog=r["i"]/max(r["n"]-1,1)))
print(json.dumps(dict(tot)))
for t in ("end","died","up","down","flat"):
    q=[x for x in rows if x["tag"]==t]
    if q: print(t,len(q),{k:round(float(np.median([x[k] for x in q])),2) for k in ("d","dy","air","sp","hold","prog")})
