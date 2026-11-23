#!/usr/bin/env python3
"""Goal-conditioned PPO over a route's legs, on v2's objectives.

Reward = the GA's: r_t = eta_{t-1} - eta_t, +300 on a leg's Done, -600 on death
or Lost, discounted per frame (a decision holds its pad for 1, 2, 4 or 8
frames: SMDP). Resets from savestates along demonstrations, the leg's end
first, moving back. Behaviour cloning of the demos before and during PPO.

    .venv/bin/python rlv2/train_rl.py --route fast-king-walk.json --demo oss2/runs/legs-fast-king-walk-s1/route.m64 \
        --legs ChompSide --frames 3000000 --name walk-chomp
"""

import argparse
import collections
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mllog
import routes as R
import policy as P

CK = os.path.join(R.BB, "checkpoints", "rlv2")
LOGS = os.path.join(R.BB, "logs", "rlv2")


def bind_demo(pool, rid, names, path, first, last, gid, start=None, name=None, overlay=False):
    """Folds a movie over legs first..last and records BC samples per control frame."""
    pads = R.read_movie(path)
    if start is None:
        start = R.first_frame(pool, pads, R.in_castle, gid=gid)
        fl = names.index("BobPainting")
    else:
        fl = first
    d = R.Demo(pool, rid, names, pads, fl, last, start, gid=gid, every=2, name=name or os.path.basename(path))
    return d


def record_bc(pool, d, rid, names, gid, legs, rt=None):
    """Replays demo legs recording (obs, action, reward-to-go inputs)."""
    rec = []
    for L in d.legs:
        if L["leg"] not in legs or L["done"] is None or L["control"] is None or not L["saves"]:
            continue
        pool.load([gid], [L["saves"][0][1]])
        pool.rl_set_overlay([gid], False)
        f = L["saves"][0][0]
        obs, rd, pp, zs = [], [], [], []
        while f < L["done"]:
            o = pool.rl_obs([gid])
            if not bool(o["control"][0]):
                r = pool.rl_raw([gid], d.pads[f : f + 1][None, :])
                f += max(int(r["frames"][0]), 1)
                continue
            obs.append({k: v.copy() for k, v in o.items()})
            pp.append(d.pads[f])
            if rt is not None:
                zs.append(rt.encode(pool.read([gid])).cpu())
            r = pool.rl_raw([gid], d.pads[f : f + 1][None, :])
            rd.append(float(r["ret"][0]))
            f += max(int(r["frames"][0]), 1)
            if int(r["term"][0]) or int(r["legs_done"][0]):
                break
        rec.append(dict(leg=L["leg"], obs=obs, ret=rd, pads=pp, z=zs))
    return rec


def run_lengths(acts):
    """Hold class at each sample: how long the same action continues."""
    n = len(acts)
    run = np.ones(n, np.int64)
    for i in range(n - 2, -1, -1):
        if acts[i] == acts[i + 1]:
            run[i] = run[i + 1] + 1
    return np.searchsorted(P.HOLDS, np.minimum(run, 8), side="right") - 1


def bc_data(rec, landmarks, gamma, scale, device):
    xs, ac, bi, fl, lg, ys, gs, zz = [], [], [], [], [], [], [], []
    for r in rec:
        if not r["obs"]:
            continue
        lab = []
        for o, p in zip(r["obs"], r["pads"]):
            lab.append(P.encode_pad(int(p), o["f"][0], o["target"][0]))
        hold = run_lengths(lab)
        rtg, g = 0.0, np.zeros(len(lab), np.float32)
        for i in range(len(lab) - 1, -1, -1):
            rtg = r["ret"][i] + gamma * rtg
            g[i] = rtg * scale
        for i, o in enumerate(r["obs"]):
            x, a, b, f, l = P.featurize(o, landmarks)
            xs.append(x), ac.append(a), bi.append(b), fl.append(f), lg.append(l)
            ys.append((lab[i][0], lab[i][1], lab[i][2], hold[i]))
            if r["z"]:
                zz.append(r["z"][i])
        gs.append(g)
    t = lambda v, ty=None: torch.as_tensor(np.concatenate(v) if isinstance(v[0], np.ndarray) else np.array(v), device=device, dtype=ty)
    d = dict(x=t(xs), act=t(ac), bits=t(bi), floor=t(fl), leg=t(lg), y=torch.as_tensor(np.array(ys), device=device), g=t(gs))
    d["z"] = torch.cat(zz).to(device).float() if zz else None
    return d


def bc_loss(pol, d, idx):
    z = d["z"][idx] if d["z"] is not None else None
    logits, v = pol(d["x"][idx], d["act"][idx], d["bits"][idx], d["floor"][idx], d["leg"][idx], z)
    y = d["y"][idx]
    l = sum(F.cross_entropy(lg, y[:, i]) for i, lg in enumerate(logits))
    # Only moves count for the magnitude.
    return l, F.mse_loss(v, d["g"][idx]), (logits[0].argmax(1) == y[:, 0]).float().mean(), (logits[2].argmax(1) == y[:, 2]).float().mean()


class Curriculum:
    """Per leg: starts are drawn from the demo saves within `back` frames of
    the leg's Done. `back` grows when the episodes from its hardest band
    succeed often enough and not much slower than the demo."""

    def __init__(self, legs_info, a):
        self.a = a
        self.legs = legs_info  # leg -> list of (demo index, list of (frame, handle), done)
        self.back = {L: a.back0 for L in legs_info}
        self.win = {L: collections.deque(maxlen=a.cur_window) for L in legs_info}
        self.full = {L: False for L in legs_info}

    def length(self, L):
        return max(s[-1][0] - s[0][0] for _, s, _ in self.legs[L]) + 2

    def sample(self, L, rng):
        """-> (handle, frames the demo took from there, band hardest?, frame)."""
        k = rng.integers(len(self.legs[L]))
        _, saves, done = self.legs[L][k]
        back = self.back[L]
        ok = [s for s in saves if done - s[0] <= back]
        if not ok:
            ok = saves[-1:]
        hardest = rng.random() < 0.5
        if self.full[L] and rng.random() < self.a.p_start:
            s = saves[0]
        elif hardest:
            n = max(1, len(ok) // 3)
            s = ok[rng.integers(n)]
        else:
            s = ok[rng.integers(len(ok))]
        return s[1], done - s[0], hardest, s[0]

    def report(self, L, hard, ok, ratio):
        if hard:
            self.win[L].append((ok, ratio))
        w = self.win[L]
        if len(w) < self.a.cur_min:
            return False
        sr = np.mean([x[0] for x in w])
        if sr >= self.a.cur_success and not self.full[L]:
            med = np.median([x[1] for x in w if x[0]])
            if med <= self.a.cur_ratio:
                self.back[L] += self.a.back_step
                w.clear()
                if self.back[L] >= self.length(L):
                    self.full[L] = True
                return True
        return False


@torch.no_grad()
def act(pol, x, a, b, f, l, z, device, greedy=False):
    t = lambda v, ty=None: torch.as_tensor(v, device=device, dtype=ty)
    logits, v = pol(t(x), t(a), t(b), t(f), t(l), z)
    ds = P.dists(logits)
    if greedy:
        ac = torch.stack([d.probs.argmax(1) for d in ds], 1)
    else:
        ac = torch.stack([d.sample() for d in ds], 1)
    return ac, P.log_prob(ds, ac), v, (logits, ds)


def evaluate(pool, pol, rt, starts, landmarks, a, ids, device, greedy=True, samples=1):
    """Plays from each start (start saves' first frames) until finished or a
    limit. greedy: one rollout; else `samples` rollouts per start. Returns per
    start: successes, frames to Done list, played pads of the best."""
    out = []
    for st in starts:
        n = 1 if greedy else samples
        res, best = [], None
        for chunk in range(0, n, len(ids)):
            js = ids[: min(len(ids), n - chunk)]
            pool.load(js, [st["handle"]] * len(js))
            pool.rl_set_overlay(js, a.overlay)
            alive = list(js)
            t = {j: 0 for j in js}
            while alive:
                o = pool.rl_obs(alive)
                x, ac_, bi, fl, lg = P.featurize(o, landmarks)
                z = rt.encode(pool.read(alive)) if rt is not None else None
                ac, _, _, _ = act(pol, x, ac_, bi, fl, lg, z, device, greedy=greedy)
                ac = ac.cpu().numpy()
                base, rel, mag, btn, hold = P.decode(ac[:, 0], ac[:, 1], ac[:, 2], ac[:, 3])
                r = pool.rl_act(alive, base, rel, mag, btn, hold, a.gamma, a.bonus, a.death, 1000.0, True)
                nxt = []
                for k, j in enumerate(alive):
                    t[j] += int(r["frames"][k])
                    term = int(r["term"][k])
                    pads, dones = pool.rl_played(j)
                    if term == 1:
                        done_frame = dones[-1][1]
                        fr = int(done_frame - st["frame"])
                        res.append(fr)
                        if best is None or fr < best[0]:
                            best = (fr, pads[:fr].copy() if False else pads.copy(), list(dones))
                    elif term:
                        res.append(None)
                    elif t[j] >= st["limit"]:
                        res.append(None)
                    else:
                        nxt.append(j)
                alive = nxt
        out.append(dict(start=st["name"], ok=sum(x is not None for x in res), n=len(res), frames=[x for x in res if x is not None], best=best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="fast-king-walk.json")
    ap.add_argument("--demo", nargs="+", required=True, help="movies (from power-on), each optionally path@startframe")
    ap.add_argument("--legs", nargs="+", required=True, help="leg names, consecutive")
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=float, default=3e6, help="emulated frames")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=64, help="decisions per env per update")
    ap.add_argument("--gamma", type=float, default=0.99, help="per frame")
    ap.add_argument("--lam", type=float, default=0.95, help="per decision")
    ap.add_argument("--bonus", type=float, default=300.0)
    ap.add_argument("--death", type=float, default=600.0)
    ap.add_argument("--scale", type=float, default=0.02)
    ap.add_argument("--clip-r", type=float, default=200.0)
    ap.add_argument("--time-cost", type=float, default=0.0, help="reward per frame off, in ETA frames (time pressure on top of gamma)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--mb", type=int, default=2048)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--overlay", type=int, default=1)
    ap.add_argument("--ckpt", default=None, help="backbone checkpoint: z added to the observation")
    ap.add_argument("--init", default=None, help="policy checkpoint to start from")
    ap.add_argument("--bc-steps", type=int, default=2000)
    ap.add_argument("--bc-coef", type=float, default=0.3, help="BC loss weight during PPO, decays linearly to 0 over --bc-decay of the run")
    ap.add_argument("--bc-decay", type=float, default=0.5)
    ap.add_argument("--back0", type=int, default=30)
    ap.add_argument("--back-step", type=int, default=16)
    ap.add_argument("--cur-success", type=float, default=0.6)
    ap.add_argument("--cur-ratio", type=float, default=1.5)
    ap.add_argument("--cur-window", type=int, default=128)
    ap.add_argument("--cur-min", type=int, default=64)
    ap.add_argument("--p-start", type=float, default=0.3)
    ap.add_argument("--limit-mult", type=float, default=2.0)
    ap.add_argument("--seq", action="store_true", help="the legs are one sequence: resets anywhere along them, an episode ends after the last, eval from the first leg's start")
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--eval-samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.overlay = bool(a.overlay)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sm = R.setup()
    E, M = a.envs, 32
    pool = R.make_pool(sm, E + M + 2, a.threads)
    rid, names, doc = R.load_route(pool, os.path.join(R.V2, "tools", "spec", "routes", a.route))
    train_legs = [names.index(n) for n in a.legs]
    last_leg = max(train_legs)
    gid = E + M
    os.makedirs(CK, exist_ok=True), os.makedirs(LOGS, exist_ok=True)
    rt = None
    if a.ckpt:
        from sm64bb.runtime import Runtime

        rt = Runtime(a.ckpt)
    demos = []
    for spec in a.demo:
        path, _, st = spec.partition("@")
        path = path if path.startswith("/") else os.path.join(R.V2, path)
        d = bind_demo(pool, rid, names, path, train_legs[0], last_leg, gid, start=int(st) if st else None)
        demos.append(d)
        print(os.path.basename(path), [(l["name"], l["control"], l["done"]) for l in d.legs if l["leg"] in train_legs], flush=True)
    landmarks = np.full((len(names) + 2, 3), np.nan, np.float32)
    # Each leg's end, on the first demo that reached it.
    for d in demos[::-1]:
        for L in d.legs:
            if L["done"] is not None:
                pool.reset([gid])
                pool.step([gid], d.pads[: L["done"]][None, :])
                landmarks[L["leg"]] = pool.status([gid])["pos"][0]
    all_legs = list(train_legs)
    legs_info = {}
    starts = []
    if a.seq:
        # One curriculum over the whole sequence: back from the last leg's Done.
        L0 = train_legs[0]
        legs_info[L0] = []
        for k, d in enumerate(demos):
            ls = [l for l in d.legs if l["leg"] in train_legs]
            if len(ls) == len(train_legs) and all(l["done"] is not None and l["saves"] for l in ls):
                saves = [s for l in ls for s in l["saves"]]
                done = ls[-1]["done"]
                legs_info[L0].append((k, saves, done))
                starts.append(dict(name=f"seq@{k}", handle=saves[0][1], frame=saves[0][0], limit=int(a.limit_mult * (done - saves[0][0])) + 60, leg=L0, demo_frames=done - saves[0][0]))
        train_legs = [L0]
    else:
        for L in train_legs:
            legs_info[L] = []
            for k, d in enumerate(demos):
                for l in d.legs:
                    if l["leg"] == L and l["done"] is not None and l["saves"]:
                        legs_info[L].append((k, l["saves"], l["done"]))
                        starts.append(dict(name=f"{names[L]}@{k}", handle=l["saves"][0][1], frame=l["saves"][0][0], limit=int(a.limit_mult * (l["done"] - l["saves"][0][0])) + 60, leg=L, demo_frames=l["done"] - l["saves"][0][0]))
    cur = Curriculum(legs_info, a)
    # Behaviour cloning.
    d_z = rt.m.d if rt else 0
    pol = P.Policy(len(names), d_z=d_z).to(device)
    if a.init:
        pol.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["policy"])
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr, eps=1e-5)
    bc = None
    if a.bc_steps or a.bc_coef:
        rec = []
        for d in demos:
            rec += record_bc(pool, d, rid, names, gid, set(all_legs), rt)
        bc = bc_data(rec, landmarks, a.gamma, a.scale, device)
        n = len(bc["x"])
        t0 = time.time()
        for i in range(a.bc_steps):
            idx = torch.randint(n, (min(1024, n),), device=device)
            lp, lv, acc_d, acc_b = bc_loss(pol, bc, idx)
            loss = lp + lv
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            if i % 500 == 0 or i == a.bc_steps - 1:
                print(json.dumps({"bc": i, "n": n, "loss_pi": round(lp.item(), 3), "loss_v": round(lv.item(), 3), "acc_dir": round(acc_d.item(), 3), "acc_btn": round(acc_b.item(), 3)}), flush=True)
        print(f"bc {time.time() - t0:.0f} s", flush=True)
        ev = evaluate(pool, pol, rt, starts, landmarks, a, list(range(E, E + M)), device, greedy=True)
        print(json.dumps({"bc_eval": [{k: e[k] for k in ("start", "ok", "n", "frames")} for e in ev]}), flush=True)

    ids = list(range(E))
    ep_leg = np.zeros(E, int)
    ep_hard = np.zeros(E, bool)
    ep_ref = np.zeros(E, int)
    ep_limit = np.zeros(E, int)
    t_ep = np.zeros(E, int)

    def reset(js):
        hs = []
        for j in js:
            L = train_legs[rng.integers(len(train_legs))]
            h, rem, hard, _ = cur.sample(L, rng)
            hs.append(h)
            ep_leg[j], ep_hard[j], ep_ref[j] = L, hard, rem
            ep_limit[j] = int(a.limit_mult * rem) + 40
            t_ep[j] = 0
        pool.load(js, hs)
        pool.rl_set_overlay(js, a.overlay)

    reset(ids)
    frames, upd = 0, 0
    log = open(os.path.join(LOGS, f"{a.name}.jsonl"), "a")
    win = collections.deque(maxlen=400)
    mllog.start(a.name, vars(a))
    best_score = 1e9
    t0 = time.time()
    while frames < a.frames:
        buf = collections.defaultdict(list)
        for _ in range(a.rollout):
            o = pool.rl_obs(ids)
            x, ac_, bi, fl, lg = P.featurize(o, landmarks)
            z = rt.encode(pool.read(ids)) if rt else None
            ac, lp, v, _ = act(pol, x, ac_, bi, fl, lg, z, device)
            acn = ac.cpu().numpy()
            base, rel, mag, btn, hold = P.decode(acn[:, 0], acn[:, 1], acn[:, 2], acn[:, 3])
            r = pool.rl_act(ids, base, rel, mag, btn, hold, a.gamma, a.bonus, a.death, a.clip_r, True)
            k = r["frames"].astype(np.float64)
            frames += int(k.sum())
            t_ep += r["frames"]
            disc = a.gamma**k
            # Time pressure on top of γ: a per-frame cost, discounted.
            cost = a.time_cost * (1 - disc) / (1 - a.gamma)
            rew = (r["ret"] - cost) * a.scale
            term = r["term"]
            timeout = (t_ep >= ep_limit) & (term == 0)
            done = (term != 0) | timeout
            # A time limit isn't an end: bootstrap from the state it stopped in.
            if timeout.any():
                js = np.flatnonzero(timeout).tolist()
                o2 = pool.rl_obs(js)
                x2, a2, b2, f2, l2 = P.featurize(o2, landmarks)
                z2 = rt.encode(pool.read(js)) if rt else None
                _, _, v2, _ = act(pol, x2, a2, b2, f2, l2, z2, device)
                rew[js] += disc[js] * v2.cpu().numpy()
            buf["x"].append(x), buf["act"].append(ac_), buf["bits"].append(bi), buf["floor"].append(fl), buf["leg"].append(lg)
            if z is not None:
                buf["z"].append(z)
            buf["ac"].append(ac), buf["lp"].append(lp), buf["v"].append(v)
            buf["rew"].append(rew.astype(np.float32)), buf["done"].append(done), buf["disc"].append(disc.astype(np.float32))
            if done.any():
                js = np.flatnonzero(done).tolist()
                for j in js:
                    ok = int(term[j]) == 1
                    ratio = t_ep[j] / max(ep_ref[j], 1)
                    L = ep_leg[j]
                    win.append((L, ok, int(term[j]), ratio, ep_ref[j]))
                    if cur.report(L, ep_hard[j], ok, ratio):
                        print(json.dumps({"curriculum": names[L], "back": cur.back[L], "frames": frames}), flush=True)
                reset(js)
        # GAE over decisions; each step's discount is γ^frames.
        o = pool.rl_obs(ids)
        x, ac_, bi, fl, lg = P.featurize(o, landmarks)
        z = rt.encode(pool.read(ids)) if rt else None
        _, _, vlast, _ = act(pol, x, ac_, bi, fl, lg, z, device)
        T = len(buf["rew"])
        val = torch.stack(buf["v"]).cpu().numpy()
        vl = vlast.cpu().numpy()
        adv = np.zeros((T, E), np.float32)
        last = np.zeros(E, np.float32)
        for t in reversed(range(T)):
            nv = vl if t == T - 1 else val[t + 1]
            nd = 1 - buf["done"][t].astype(np.float32)
            delta = buf["rew"][t] + buf["disc"][t] * nv * nd - val[t]
            last = delta + buf["disc"][t] * a.lam * nd * last
            adv[t] = last
        ret = adv + val
        dv = lambda v, ty=None: torch.as_tensor(np.concatenate(v), device=device, dtype=ty)
        X, A_, B_, F_, L_ = dv(buf["x"]), dv(buf["act"]), dv(buf["bits"]), dv(buf["floor"]), dv(buf["leg"])
        Z_ = torch.cat(buf["z"]) if buf["z"] else None
        AC, LP = torch.cat(buf["ac"]), torch.cat(buf["lp"])
        ADV = torch.as_tensor(adv.reshape(-1), device=device)
        RET = torch.as_tensor(ret.reshape(-1), device=device)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = T * E
        coef = a.bc_coef * max(0.0, 1 - frames / (a.bc_decay * a.frames)) if bc is not None else 0.0
        stats = collections.defaultdict(list)
        for _ in range(a.epochs):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, a.mb):
                mb = perm[i : i + a.mb]
                logits, v = pol(X[mb], A_[mb], B_[mb], F_[mb], L_[mb], Z_[mb] if Z_ is not None else None)
                ds = P.dists(logits)
                ratio = torch.exp(P.log_prob(ds, AC[mb]) - LP[mb])
                l_pi = -torch.min(ratio * ADV[mb], ratio.clamp(0.8, 1.2) * ADV[mb]).mean()
                l_v = F.mse_loss(v, RET[mb])
                en = P.entropy(ds).mean()
                loss = l_pi + a.vf * l_v - a.ent * en
                if coef > 0:
                    idx = torch.randint(len(bc["x"]), (min(512, len(bc["x"])),), device=device)
                    lb, lbv, _, _ = bc_loss(pol, bc, idx)
                    loss = loss + coef * lb
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
                stats["pi"].append(l_pi.item()), stats["v"].append(l_v.item()), stats["ent"].append(en.item())
                stats["clipfrac"].append(((ratio - 1).abs() > 0.2).float().mean().item())
        upd += 1
        sr = np.mean([w[1] for w in win]) if win else 0.0
        rec = {"upd": upd, "frames": frames, "fps": int(frames / (time.time() - t0)), "sr": round(float(sr), 3),
               "ratio": round(float(np.median([w[3] for w in win if w[1]])), 3) if any(w[1] for w in win) else None,
               "died": round(float(np.mean([w[2] in (2, 3) for w in win])), 3) if win else 0, "rew": round(float(np.mean(buf["rew"])), 4),
               "v_loss": round(np.mean(stats["v"]), 4), "ent": round(np.mean(stats["ent"]), 3), "clip": round(np.mean(stats["clipfrac"]), 3),
               "mean_hold": round(float(N and np.mean([P.HOLDS[int(i)] for i in AC[:, 3].cpu().numpy()[:2000]])), 2),
               "back": {names[L]: cur.back[L] for L in train_legs}}
        if upd % a.eval_every == 0:
            ev = evaluate(pool, pol, rt, starts, landmarks, a, list(range(E, E + M)), device, greedy=True)
            evs = evaluate(pool, pol, rt, starts, landmarks, a, list(range(E, E + M)), device, greedy=False, samples=a.eval_samples)
            rec["eval"] = {e["start"]: e["frames"][0] if e["frames"] else None for e in ev}
            rec["eval_s"] = {e["start"]: dict(sr=round(e["ok"] / max(e["n"], 1), 2), best=min(e["frames"]) if e["frames"] else None, med=int(np.median(e["frames"])) if e["frames"] else None) for e in evs}
            score = sum((e["frames"][0] if e["frames"] else 10000) for e in ev) + 0.1 * sum(min(e["frames"]) if e["frames"] else 10000 for e in evs)
            if score < best_score:
                best_score = score
                torch.save({"policy": pol.state_dict(), "args": vars(a), "frames": frames, "score": score, "d_z": d_z, "names": names, "landmarks": landmarks}, os.path.join(CK, f"{a.name}.best.pt"))
            torch.save({"policy": pol.state_dict(), "args": vars(a), "frames": frames, "d_z": d_z, "names": names, "landmarks": landmarks}, os.path.join(CK, f"{a.name}.last.pt"))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"), log.flush()
        mllog.log(rec, frames)


if __name__ == "__main__":
    main()
