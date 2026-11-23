#!/usr/bin/env python3
"""The mender's loop with the learned follower instead of the GA, in Python: one
argmax rollout per window [a, b] of a route (spec/src/bin/mend.rs's Follower
contract: pass the route's position at b strictly sooner), then the rest of the
route's inputs carried from there, stick turned for the camera. Accepted only
if the goal star comes sooner in a full replay from power-on. No GA.

    .venv/bin/python rlv2/mend_follow.py checkpoints/rlv2/fol-r3.s.pt logs/mend/king-3557.m64 --goal-stars 1 --out logs/rlv2/mf-king
"""

import argparse
import json
import os

import numpy as np
import torch

import follow_lib as L
import routes as R
from eval_follow import Args, rollout
from train_follow import Games


def turned_batch(p, d):
    """Packed pads, sticks rotated by d (int16 units), as mend.rs's `turned`."""
    d = ((d.astype(np.int64) + 32768) % 65536 - 32768).astype(np.float64)
    x = ((p >> 8) & 0xFF).astype(np.int64)
    y = (p & 0xFF).astype(np.int64)
    x = (x - ((x & 0x80) << 1)).astype(np.float64)
    y = (y - ((y & 0x80) << 1)).astype(np.float64)
    a = np.arctan2(x, -y) + d * 2 * np.pi / 65536
    r = np.hypot(x, y)
    xn = np.clip(np.round(r * np.sin(a)), -127, 127).astype(np.int64)
    yn = np.clip(np.round(-r * np.cos(a)), -127, 127).astype(np.int64)
    same = (d == 0) | (r == 0)
    xn, yn = np.where(same, x, xn).astype(np.int64), np.where(same, y, yn).astype(np.int64)
    return ((p >> 16).astype(np.int64) << 16 | (xn & 0xFF) << 8 | (yn & 0xFF)).astype(np.uint32)


def goal_frame(pool, gid, pads, stars):
    """The first frame with `stars` stars."""
    pool.reset([gid])
    for f in range(len(pads)):
        pool.step([gid], pads[f : f + 1][None, :])
        if int(pool.status([gid])["stars"][0]) >= stars:
            return f + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("route")
    ap.add_argument("--out", required=True)
    ap.add_argument("--goal-stars", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--windows", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--radius", type=int, nargs="+", default=[30, 60])
    ap.add_argument("--from-level", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    Args.rt, Args.ref, Args.copy, Args.hybrid = None, bool(ca.get("ref")), bool(ca.get("copy")), 0
    pol = L.FollowNet(h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref, copy=Args.copy).to(dev)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    sm = R.setup()
    E = 64
    pool = R.make_pool(sm, E + 2, a.threads)
    gid = E
    pads = R.read_movie(a.route if a.route.startswith("/") else os.path.join(R.V2, a.route))
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    rng = np.random.default_rng(1)
    total_emu = 0
    for rnd in range(a.rounds):
        goal = goal_frame(pool, gid, pads, a.goal_stars)
        pads = pads[:goal]
        tp = L.TracePool(pool, gid, [("route", pads, 1646)], every=a.stride, skip_common=False, name=f"round{rnd}")
        g = Games(pool, list(range(E)), tp, Args())
        jobs = [(s, W, Rd) for s in tp.saves for W in a.windows for Rd in a.radius if tp.end[s[0]] - s[0] > W]
        order = rng.permutation(len(jobs))
        stats = dict(round=rnd, goal=goal, windows=0, reached=0, ahead=0, carried=0, survived=0, emu=0)
        best = None
        for b in range(0, len(order), E):
            ch = [jobs[i] for i in order[b : b + E]]
            js = list(range(len(ch)))
            g.bind(js, [c[0][0] for c in ch], [16] * len(ch), [c[1] for c in ch], [c[2] for c in ch], [c[0][1] for c in ch])
            out = rollout(g, pol, js)
            stats["windows"] += len(ch)
            stats["emu"] += sum(out[j]["frames"] for j in js)
            cands = []
            for j, c in zip(js, ch):
                r = out[j]
                if r["ok"]:
                    stats["reached"] += 1
                    if r["frames"] < r["ref"]:
                        stats["ahead"] += 1
                        cands.append((j, c, r))
            if not cands:
                continue
            # Carry: the rest of the route's inputs from there.
            ids = [g.ids[c[0]] for c in cands]
            at = np.array([c[1][0][0] + c[2]["ref"] for c in cands], np.int64)  # trace index of b
            alive = np.ones(len(ids), bool)
            lag = np.zeros(len(ids), np.int64)
            carried = [[] for _ in ids]
            t = 0
            n_tr = len(tp.tid)
            stats["carried"] += len(ids)
            while alive.any():
                live = np.flatnonzero(alive)
                st = pool.status([ids[i] for i in live])
                ix = at[live] + t
                over = ix >= n_tr
                # Past the route's end with no star: dropped.
                alive[live[over]] = False
                live, keep = live[~over], ~over
                if not len(live):
                    break
                ix = ix[keep]
                pos, cam, stars = st["pos"][keep], st["cam_yaw"][keep], st["stars"][keep]
                done = stars >= a.goal_stars
                for i, d in zip(live, done):
                    if d:
                        stats["survived"] += 1
                        k = int(cands[i][1][0][0]) + 1646 + len(pads[1646:]) * 0
                        ca_ = dict(c=cands[i], t=t)
                        total = int(tp.frame[cands[i][1][0][0]]) + cands[i][2]["frames"] + t
                        if total < goal and (best is None or total < best[0]):
                            pp, _ = pool.tr_played(ids[i])
                            best = (total, np.concatenate([pads[: int(tp.frame[cands[i][1][0][0]])], pp, np.asarray(carried[i], np.uint32)]), cands[i][1][1:], int(tp.frame[cands[i][1][0][0]]))
                        alive[i] = False
                live2 = live[~done]
                ix2 = ix[~done]
                if not len(live2):
                    break
                far = np.linalg.norm(pos[~done] - tp.pos[ix2], axis=1) > 500
                lag[live2] = np.where(far, lag[live2] + 1, 0)
                gone = lag[live2] > 40
                alive[live2[gone]] = False
                live3, ix3, cam3 = live2[~gone], ix2[~gone], cam[~done][~gone]
                if not len(live3):
                    break
                # The stick's camera, from the trace.
                d = tp.f[ix3, 9].astype(np.int64) - cam3
                p = turned_batch(tp.pads[ix3].astype(np.uint32), d)
                pool.step([ids[i] for i in live3], p[:, None].astype(np.uint32))
                for i, q in zip(live3, p):
                    carried[i].append(int(q))
                stats["emu"] += len(live3)
                t += 1
            if best is not None:
                # Accept the first improvement of a batch, after a full replay.
                total, new, what, frm = best
                g2 = goal_frame(pool, gid, new, a.goal_stars)
                stats["verified"] = g2
                if g2 is not None and g2 < goal:
                    stats["improved"] = dict(frm=goal, to=g2, window=what, at=frm)
                    R.write_m64(os.path.join(a.out, f"route-{g2}.m64"), new[:g2])
                    pads = new[:g2]
                    break
                best = None
            if stats["windows"] % (E * 8) == 0:
                print(json.dumps(stats), flush=True)
        total_emu += stats["emu"]
        stats["total_emu"] = total_emu
        print(json.dumps(stats), flush=True)
        log.write(json.dumps(stats) + "\n"), log.flush()
        if "improved" not in stats:
            break


if __name__ == "__main__":
    main()
