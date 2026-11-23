#!/usr/bin/env python3
"""Whole-route following, with the GA repairing where the follower stalls. From
a route's first control state to its end, through waypoints: argmax follower
until no waypoint has been passed for a while or Mario dies; then back to the
last waypoint's savestate and GA attempts (objective-rrt's Mario::PosDist, the
pool's `attempt`) at the next waypoint; on success, follow on from there.
The result's whole input list is replayed through a fresh game as a check.

    .venv/bin/python rlv2/eval_repair.py checkpoints/rlv2/fol-f.best.pt --out logs/rlv2/repair.jsonl
"""

import argparse
import json
import os

import numpy as np
import torch

import follow_lib as L
import policy as P
import routes as R
from eval_follow import Args
from train_follow import Games, choose, do_act

ROUTES = [("king-3557 (held out)", "logs/mend/king-3557.m64", None), ("ramp0-3610 (held out)", "logs/kingloop/best-fast-king-blj-tas.ramp0.m64", None),
          ("tas-bob (held out)", L.TAS, L.TAS_HELD_WINDOW), ("walk-s3 (held out)", "oss2/runs/legs-fast-king-walk-s3/route.m64", None),
          ("q15-s3-star1 (held out)", "runs/q15-pw1-s3/frontier-k0-s1.m64", None), ("q15-s2-8562 (trained on)", "logs/mend/q15-s2-star4-8562.m64", None)]


def run(pool, g, pol, tp, S, a, repair, K=0):
    """-> a result dict; g has one game."""
    gid = g.ids[0]
    cands = [s for s in tp.saves if tp.control[s[0]] and (not a.start_level or tp.level[s[0]] == a.start_level)]
    first = cands[0] if cands else next(s for s in tp.saves if tp.control[s[0]])
    t0, h = first
    end = int(tp.end[t0])
    H = end - t0
    S = max(S, H // (L.MAX_WP - 2) + 1)
    pads, repairs, ga_frames, frames = [], 0, 0, 0
    start_frame = int(tp.frame[t0])
    g.bind([0], [t0], [S], [H], [a.radius], [h])
    g.limit[0] = 10**9
    if K:
        # Rewire: K frames of a fixed off-trace input, then the rest of the trace.
        ang = 2.4
        x, y = int(64 * np.sin(ang)) & 0xFF, int(64 * np.cos(ang)) & 0xFF
        np_ = np.full((1, K), (x << 8) | y, np.uint32)
        pool.step([gid], np_)
        pads.append(np_[0].copy())
        frames += K
        h = pool.save([gid])[0]
        t0 = min(t0 + K, end - 1)
        H = end - t0
        g.bind([0], [t0], [S], [H], [a.radius], None)
        g.limit[0] = 10**9
    last_h, last_t, i_abs = h, t0, 0
    n_total = int(g.n_wp[0])
    wp_t = g.W[0, : n_total, 5].astype(int).tolist()
    since = 0
    fails = 0
    stall = max(a.stall, 3 * S)
    kept = np.zeros(0, np.uint32)
    best_prog = 0
    while True:
        ac, _, _ = choose(pol, g.obs([0]), greedy=True)
        ac = ac.cpu().numpy()
        if a.hybrid > 0:
            ac[g.al_dist < a.hybrid, 0] = P.N_DIR
        r = do_act(g, [0], ac)
        f, term, i = int(r["frames"][0]), int(r["term"][0]), int(r["i"][0])
        frames += f
        since += f
        done_now = term == 1
        if i > 0 and (int(g.W[0, i - 1, 5]) > last_t or done_now):
            # A new waypoint passed: its state and pads are the fallback. The
            # binding stays (a rebind would move the alignment ahead of Mario).
            kept, _ = pool.tr_played(gid)
            kept = kept.copy()
            last_h, last_t = pool.save([gid])[0], int(g.W[0, i - 1, 5])
            since = 0
            fails = 0
            best_prog = max(best_prog, last_t)
        if done_now:
            pads.append(kept)
            return dict(ok=True, frames=frames, ref=H, repairs=repairs, ga=ga_frames, S=S, pads=pads, start=start_frame)
        if term or since >= stall:
            if not repair or fails >= a.tries or repairs >= a.max_repairs:
                pads.append(kept)
                return dict(ok=False, frames=frames, ref=H, repairs=repairs, ga=ga_frames, S=S, prog=round((best_prog - t0) / H, 3), reason="dead" if term else "stall", pads=pads, start=start_frame)
            # Waypoints after the last save; each try aims further.
            ws = tp.waypoints(last_t, S, end - last_t)
            goal = ws[min(fails, len(ws) - 1)]
            handles = [last_h] * a.ga_batch
            goals = np.tile(goal[:3][None], (a.ga_batch, 1)).astype(np.float32)
            # Seeded with the trace's own inputs to the goal, as the mender's GA.
            sd = tp.pads[last_t : int(goal[5])].astype(np.uint32)
            res = pool.attempt_to(handles, goals, [sd] * len(handles), a.ga_scale, int(a.seed + repairs * 97 + fails))
            ga_frames += sum(int(x["frames"]) for x in res)
            ok = [x for x in res if x["success"] and x["handle"] is not None]
            repairs += 1
            if ok:
                best = min(ok, key=lambda x: len(x["inputs"]))
                pads.append(kept)
                kept = np.zeros(0, np.uint32)
                pads.append(np.asarray(best["inputs"], np.uint32))
                t_new = int(goal[5])
                frames += len(best["inputs"])
                last_h, last_t = best["handle"], t_new
                best_prog = max(best_prog, t_new)
                g.bind([0], [t_new], [S], [max(end - t_new, 1)], [a.radius], [best["handle"]])
                g.limit[0] = 10**9
                since, fails = 0, 0
                if t_new >= end:
                    return dict(ok=True, frames=frames, ref=H, repairs=repairs, ga=ga_frames, S=S, pads=pads, start=start_frame)
            else:
                fails += 1
                # Back to the fallback; what's already kept stays kept.
                pads.append(kept)
                kept = np.zeros(0, np.uint32)
                g.bind([0], [last_t], [S], [max(end - last_t, 1)], [a.radius], [last_h])
                g.limit[0] = 10**9
                since = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--radius", type=int, default=100)
    ap.add_argument("--stall", type=int, default=120)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--max-repairs", type=int, default=60)
    ap.add_argument("--ga-batch", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spacings", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--hybrid", type=float, default=0.0)
    ap.add_argument("--ga-scale", type=float, default=6.0)
    ap.add_argument("--start-level", type=int, default=9, help="start at the route's first control state in this level (0: its first)")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 15])
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    ca = ck["args"]
    Args.rt = None
    if ca.get("z_ckpt"):
        from sm64bb.runtime import Runtime

        Args.rt = Runtime(ca["z_ckpt"])
    net_copy = bool(ca.get("copy"))
    Args.ref, Args.copy = bool(ca.get("ref")), net_copy or a.hybrid > 0
    pol = L.FollowNet(d_z=Args.rt.m.d if Args.rt else 0, h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref, copy=net_copy).to(dev)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    sm = R.setup()
    pool = R.make_pool(sm, 2, a.threads)
    out = open(a.out, "a")
    for name, path, win in ROUTES:
        full = os.path.join(R.V2, path)
        pads = R.read_movie(full)
        tp = L.TracePool(pool, 1, [(name, pads[: win[1]] if win else pads, win[0] if win else 1646)], every=8, skip_common=False, name=name)
        tp.saves = sorted(tp.saves)
        g = Games(pool, [0], tp, Args())
        for S in a.spacings:
            for K, rep in [(k, b) for k in a.ks for b in (False, True)]:
                r = run(pool, g, pol, tp, S, a, rep, K)
                check = None
                if r["ok"]:
                    allp = np.concatenate(r["pads"])
                    r["route_frames"] = len(allp)
                    pool.reset([1])
                    pool.step([1], pads[: r["start"]][None, :])
                    pool.step([1], allp[None, :])
                    st = pool.status([1])
                    # Waypoints are positions only: a boss fight's clustered points all
                    # pass at once, so the stars say whether the route's work got done.
                    check = dict(end_pos=[round(float(x)) for x in st["pos"][0]], last_wp=[round(float(x)) for x in tp.waypoints(int(tp.saves[0][0]), r["S"], len(tp.tid))[-1, :3]], inputs=len(allp), stars=int(st["stars"][0]))
                pool.reset([1])
                pool.step([1], pads[None, :])
                ref_stars = int(pool.status([1])["stars"][0])
                rec = dict(route=name, ref_stars=ref_stars, K=K, repair=rep, **{k: v for k, v in r.items() if k != "pads"}, check=check)
                print(json.dumps(rec), flush=True)
                out.write(json.dumps(rec) + "\n"), out.flush()


if __name__ == "__main__":
    main()
