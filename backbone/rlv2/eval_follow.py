#!/usr/bin/env python3
"""Evaluates a follower on traces it never trained on, deterministically
(argmax; same inputs, same output), through the real emulator.

  Grid: from saves along held-out traces, waypoints sampled at each spacing;
  the rate of reaching the end, the progress through the waypoints, the share
  hit, frames against the route's own. Split by whether the reference moves
  fast there (technique) or walks.

  Rewire: the rest of a trace from a state it never passed through:
    perturb K: K frames of noise from the trace's own state, then follow;
    graft: another held-out trace's state near this trace, then follow
    (timing, objects and speed all differ, as after a tree rewire).
  Baselines on the same starts: blind replay of the trace's remaining inputs
  (what a graft does today), and scripted steering straight at the waypoint.

  Determinism: a repeat of the grid must match; the best rollouts' pads are
  replayed through a fresh game and must reach the same waypoints.

    .venv/bin/python rlv2/eval_follow.py checkpoints/rlv2/fol-a.best.pt --out logs/rlv2/fol-a.eval.json
"""

import argparse
import json
import os

import numpy as np
import torch

import follow_lib as L
import policy as P
import routes as R
from train_follow import Games, choose, do_act

SPACINGS = [8, 16, 32, 64, 128, 256]


class Args:
    gamma, death, clip_r, wp_bonus, finish, limit_mult = 0.99, 100.0, 50.0, 1.0, 30.0, 2.0


def scripted(n):
    """Run straight at the waypoint, no buttons."""
    return (np.ones(n, np.int32), np.zeros(n, np.int32), np.ones(n, np.float32), np.zeros(n, np.uint32), np.ones(n, np.int32))


def rollout(g, pol, js, arm="policy"):
    """Plays games js to their ends. -> dict per game."""
    alive, out = list(js), {}
    while alive:
        if arm == "steer":
            base, rel, mag, btn, hold = scripted(len(alive))
            r = g.pool.tr_act([g.ids[j] for j in alive], base, rel, mag, btn, hold, Args.gamma, Args.death, Args.clip_r)
        else:
            ac, _, _ = choose(pol, g.obs(alive), greedy=True)
            ac = ac.cpu().numpy()
            if getattr(Args, "hybrid", 0) > 0:
                # On the trace: exactly its input. Off it: the policy.
                ac[g.al_dist < Args.hybrid, 0] = P.N_DIR
            r = do_act(g, alive, ac)
        nxt = []
        for k, j in enumerate(alive):
            g.t[j] += int(r["frames"][k])
            term = int(r["term"][k])
            if term or g.t[j] >= g.limit[j]:
                out[j] = finish(g, j, term, int(r["i"][k]))
            else:
                nxt.append(j)
        alive = nxt
    return out


def finish(g, j, term, i):
    pads, hits = g.pool.tr_played(g.ids[j])
    m = g.pool.tr_match(g.ids[j])
    return dict(ok=term == 1, i=i, n=int(g.n_wp[j]), frames=int(g.t[j]), term=term, hits=len(hits), ref=int(g.ref[j]), pads=pads.copy(), hitlist=list(hits), match=m[0], speed=m[1], scored=m[2])


def replay_arm(g, tp, js, t0s):
    """Blind replay: the trace's own remaining inputs, from wherever the game is."""
    out = {}
    ids = [g.ids[j] for j in js]
    Tmax = max(int(g.ref[j]) for j in js)
    pads = np.zeros((len(js), Tmax), np.uint32)
    for k, j in enumerate(js):
        t = t0s[k]
        n = min(Tmax, int(tp.end[t]) - t + 1)
        pads[k, :n] = tp.pads[t : t + n]
    r = g.pool.tr_raw(ids, pads)
    for k, j in enumerate(js):
        g.t[j] = int(r["frames"][k])
        out[j] = finish(g, j, int(r["term"][k]), int(r["i"][k]))
    return out


def summary(rows):
    if not rows:
        return dict(n=0)
    ok = [r for r in rows if r["ok"]]
    return dict(n=len(rows), end=round(float(np.mean([r["ok"] for r in rows])), 3),
                prog=round(float(np.mean([1.0 if r["ok"] else r["i"] / max(r["n"] - 1, 1) for r in rows])), 3),
                hit=round(float(np.mean([min(r["hits"] / max(r["n"], 1), 1.0) for r in rows])), 3),
                ratio=round(float(np.median([r["frames"] / max(r["ref"], 1) for r in ok])), 3) if ok else None,
                died=round(float(np.mean([r["term"] in (2, 3) for r in rows])), 3),
                match=round(float(np.mean([r["match"] for r in rows if r.get("scored")] or [0])), 3), speed=round(float(np.mean([r["speed"] for r in rows if r.get("scored")] or [0])), 3))


def usable(tp, t, horizon):
    """A segment worth following: Mario moves, in control most of the time."""
    e = min(int(tp.end[t]), t + horizon)
    if e - t < horizon // 2 or not tp.control[t : e + 1].mean() > 0.7:
        return False
    return np.linalg.norm(np.diff(tp.pos[t : e + 1, [0, 2]], axis=0), axis=1).sum() > 200


def batched(g, tp, pol, picks, spacing, horizon, radius, arm, off=None):
    """picks: (t0, handle), for each batch bind and play."""
    rows = []
    E = len(g.ids)
    for b in range(0, len(picks), E):
        ch = picks[b : b + E]
        js = list(range(len(ch)))
        g.bind(js, [c[0] for c in ch], [spacing] * len(ch), [horizon] * len(ch), [radius] * len(ch), [c[1] for c in ch])
        if off:
            off(g, js, ch)
        if arm == "replay":
            res = replay_arm(g, tp, js, [c[0] for c in ch])
        else:
            res = rollout(g, pol, js, arm)
        rows += [res[j] | dict(fast=c[2]) for j, c in zip(js, ch)]
    return rows


def _held_paths():
    tas = R.read_movie(os.path.join(R.V2, L.TAS))
    # The level runs from a quick scan of the TAS's levels (cached).
    cache = os.path.join(R.BB, "data", "rlv2", "tas_levels.npy")
    lv = np.load(cache)
    idx = np.flatnonzero(np.isin(lv, L.TAS_HELD_LEVELS))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1) if len(idx) else []
    wins = [("TAS-bob", tas[: L.TAS_HELD_WINDOW[1]], L.TAS_HELD_WINDOW[0])]
    for r in runs:
        if len(r) > 300:
            s = int(r[0]) + 1646
            wins.append((f"TAS-lv{lv[r[0]]}-{s}", tas[: s + len(r)], s))
    tree = L.tree_movies("runs/q15-pw1-s3", 40, np.random.default_rng(11))
    return L.movies(L.FAST_HELD) + wins + L.movies(["oss2/runs/legs-fast-king-walk-s3/route.m64"]) + tree


def is_fast(tp, t, H):
    e = min(int(tp.end[t]), t + H)
    return bool(np.percentile(np.abs(tp.speed[t : e + 1]), 75) > 45)


def stratify(tp, cand, n, H, rng, balance):
    """n starts from saves `cand`: half fast, half slow when balancing."""
    fl = np.array([is_fast(tp, s[0], H) for s in cand])
    idx = np.arange(len(cand))
    if balance:
        f, s = idx[fl], idx[~fl]
        p = list(rng.choice(f, min(n // 2, len(f)), replace=False)) + list(rng.choice(s, min(n // 2, len(s)), replace=False))
    else:
        p = list(rng.choice(idx, min(n, len(idx)), replace=False))
    return [(cand[i][0], cand[i][1], bool(fl[i])) for i in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--radius", type=int, default=100)
    ap.add_argument("--horizons", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--grid-only", action="store_true")
    ap.add_argument("--hybrid", type=float, default=0.0, help="within this (units + speed) of the aligned reference state, play its input exactly")
    ap.add_argument("--fast", action="store_true", help="the fast round's held-out routes; starts half fast, half slow")
    ap.add_argument("--spacings", type=int, nargs="+", default=SPACINGS)
    ap.add_argument("--pert", type=int, nargs="+", default=[15, 30, 60])
    a = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    ca = ck["args"]
    Args.rt = None
    if ca.get("z_ckpt"):
        from sm64bb.runtime import Runtime

        Args.rt = Runtime(ca["z_ckpt"])
    Args.ref = bool(ca.get("ref"))
    Args.full = bool(ca.get("full"))
    Args.score = True
    Args.matching = float(ca.get("matching", 0.0))
    Args.hybrid = a.hybrid
    net_copy = bool(ca.get("copy"))
    Args.copy = net_copy or a.hybrid > 0
    pol = L.FollowNet(d_z=Args.rt.m.d if Args.rt else (L.N_FULL if Args.full else 0), h=ca.get("h", 1024), layers=ca.get("layers", 4), ref=Args.ref, copy=net_copy).to(device)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    sm = R.setup()
    E = 64
    pool = R.make_pool(sm, E + 1, a.threads)
    if a.fast:
            tp = L.TracePool(pool, E, _held_paths(), every=32, name="held-out")
    else:
        held = [m for r in L.HELD_TREES for m in L.tree_movies(r, 60, np.random.default_rng(11))]
        tp = L.TracePool(pool, E, L.movies(L.HELD_OUT) + held, every=32, name="held-out")
    rng = np.random.default_rng(a.seed)
    g = Games(pool, list(range(E)), tp, Args())
    res = dict(ckpt=a.ckpt, frames=ck.get("frames"), radius=a.radius, grid={}, rewire={})
    # Grid.
    for H in a.horizons:
        cand = [s for s in tp.saves if usable(tp, s[0], H)]
        picks = stratify(tp, cand, a.starts, H, rng, a.fast)
        res["grid"][H] = {"n_fast": int(sum(p[2] for p in picks))}
        for S in [s for s in a.spacings if s <= H]:
            cell = {}
            for arm in ("policy", "steer"):
                rows = batched(g, tp, pol, picks, S, H, a.radius, arm)
                cell[arm] = summary(rows)
                cell[arm + "_walk"] = summary([r for r in rows if not r["fast"]])
                cell[arm + "_fast"] = summary([r for r in rows if r["fast"]])
                if arm == "policy" and S == 32 and H == a.horizons[0]:
                    rep = batched(g, tp, pol, picks, S, H, a.radius, arm)
                    same = all(np.array_equal(x["pads"], y["pads"]) for x, y in zip(rows, rep))
                    # Pads replayed raw: the same waypoints must come out.
                    ok = 0
                    for x, p in zip(rows[:E], picks[:E]):
                        g.bind([0], [p[0]], [S], [H], [a.radius], [p[1]])
                        rr = pool.tr_raw([g.ids[0]], x["pads"][None, :])
                        ok += int(rr["i"][0]) == x["i"] and int(rr["frames"][0]) == x["frames"]
                    res["determinism"] = dict(repeat_identical=same, replay_match=f"{ok}/{min(E, len(rows))}")
                    print(json.dumps(res["determinism"]), flush=True)
            res["grid"][H][S] = cell
            print(json.dumps({"H": H, "S": S, "policy": cell["policy"], "walk": cell["policy_walk"], "fast": cell["policy_fast"], "steer": cell["steer"]}), flush=True)
    if a.grid_only:
        json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        return
    # Rewire: perturb, then follow the rest.
    H = 512
    cand = [s for s in tp.saves if usable(tp, s[0], H + 64)]
    base = stratify(tp, cand, a.starts, H, rng, a.fast)
    pert = {}
    for K in a.pert:
        for S in [s for s in a.spacings if s >= 16]:
            cell = {}
            for arm in ("policy", "steer", "replay"):
                def off(g, js, ch):
                    ids = [g.ids[j] for j in js]
                    pr = np.random.default_rng(K * 1000 + S)
                    yaw = pr.uniform(-180, 180, len(ids))
                    pads = np.zeros((len(ids), K), np.uint32)
                    x = (64 * np.sin(np.radians(yaw))).astype(np.int64) & 0xFF
                    y = (64 * np.cos(np.radians(yaw))).astype(np.int64) & 0xFF
                    pads[:] = ((x << 8) | y)[:, None].astype(np.uint32)
                    pool.step(ids, pads)
                    # Rebind here: waypoints after K, the leg's start is this state.
                    rows = [g.W[j, : g.n_wp[j], :6] for j in js]
                    pool.tr_bind(ids, rows, [0] * len(ids), [float(a.radius)] * len(ids), 4, 1.0, 30.0)
                picks = []
                for t, h, fl in base:
                    picks.append((t + K, h, fl))
                # The save is at t; the trace from t + K is the rest to follow.
                rows = batched(g, tp, pol, picks, S, H, a.radius, arm, off=off)
                cell[arm] = summary(rows)
                cell[arm + "_fast"] = summary([r for r in rows if r["fast"]])
                cell[arm + "_slow"] = summary([r for r in rows if not r["fast"]])
            pert[f"K{K}_S{S}"] = cell
            print(json.dumps({"perturb": K, "S": S, **cell}), flush=True)
    res["rewire"]["perturb"] = pert
    # Rewire: graft another trace's state near this trace.
    tid = tp.tid
    by = {}
    for t, h in tp.saves:
        by.setdefault((int(tp.level[t]), int(tp.area[t])), []).append((t, h))
    pairs = []
    keys = rng.permutation(len(tp.saves))
    for k in keys:
        tb, hb = tp.saves[k]
        z = (int(tp.level[tb]), int(tp.area[tb]))
        cands = [s for s in by[z] if tp.tid[s[0]] != tp.tid[tb] and usable(tp, s[0], H)]
        if not cands:
            continue
        ts = np.array([c[0] for c in cands])
        d = np.linalg.norm(tp.pos[ts] - tp.pos[tb], axis=1)
        # A different route's state close to this one, not a shared prefix.
        ok = np.flatnonzero((d < 300) & (d > 20) & (np.abs(tp.frame[ts] - tp.frame[tb]) > 8))
        if len(ok):
            j = ok[np.argmin(d[ok])]
            pairs.append((int(ts[j]), hb, is_fast(tp, int(ts[j]), H), float(d[j])))
        if len(pairs) >= a.starts:
            break
    res["rewire"]["graft_n"] = len(pairs)
    res["rewire"]["graft_median_gap"] = float(np.median([p[3] for p in pairs])) if pairs else None
    graft = {}
    for S in [s for s in a.spacings if s >= 16]:
        cell = {}
        for arm in ("policy", "steer", "replay"):
            rows = batched(g, tp, pol, pairs, S, H, a.radius, arm)
            cell[arm] = summary(rows)
            cell[arm + "_fast"] = summary([r for r in rows if r["fast"]])
            cell[arm + "_slow"] = summary([r for r in rows if not r["fast"]])
        graft[S] = cell
        print(json.dumps({"graft": S, **cell}), flush=True)
    res["rewire"]["graft"] = graft
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
