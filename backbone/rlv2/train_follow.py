#!/usr/bin/env python3
"""A deterministic follower of arbitrary traces: PPO + hindsight BC, on
waypoints sampled from any movie. Reward: the decrease of the trace's ETA
(reference frames left + distance to the current waypoint at its segment's
pace), per frame discounted, death -100, +30 at the end.

    .venv/bin/python rlv2/train_follow.py --name fol-a --frames 2e7
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import follow_lib as L
import policy as P
import mllog
import routes as R

CK = os.path.join(R.BB, "checkpoints", "rlv2")
LOGS = os.path.join(R.BB, "logs", "rlv2")
SPACINGS = np.array([8, 16, 32, 64, 128, 256])
RADII = np.array([60, 100, 200])


class Games:
    """A batch of games bound to waypoint arrays, with the padded W."""

    def __init__(self, pool, ids, tp, a):
        self.pool, self.ids, self.tp, self.a = pool, list(ids), tp, a
        self.dev = torch.device("cuda")
        n = len(self.ids)
        self.W = np.zeros((n, L.MAX_WP, L.FIELDS), np.float32)
        self.n_wp = np.ones(n, np.int64)
        self.rad, self.sp = np.zeros(n), np.zeros(n)
        self.ref = np.zeros(n, np.int64)
        self.limit = np.zeros(n, np.int64)
        self.t = np.zeros(n, np.int64)
        self.t0 = np.zeros(n, np.int64)
        self.i = np.zeros(n, np.int64)
        self.align = np.zeros(n, np.int64)

    def bind(self, js, t0s, spacing, horizon, radius, handles=None):
        """Loads saves `handles` and binds the games to their traces."""
        ids = [self.ids[j] for j in js]
        if handles is not None:
            self.pool.load(ids, handles)
        rows = []
        for k, j in enumerate(js):
            w = self.tp.waypoints(int(t0s[k]), int(spacing[k]), int(horizon[k]))
            n = len(w)
            self.W[j, :n] = w
            self.n_wp[j] = n
            self.sp[j], self.rad[j] = spacing[k], radius[k]
            self.ref[j] = int(w[-1, 5]) - int(t0s[k])
            self.limit[j] = int(self.a.limit_mult * self.ref[j]) + 40
            self.t[j] = 0
            self.t0[j] = t0s[k]
            self.align[j] = t0s[k]
            self.i[j] = 0
            # The Rust side needs reference frames for ETAs: flat indices work,
            # since differences within a trace are frames.
            if getattr(self.a, "full", False) or getattr(self.a, "score", False):
                # Full targets: pos, zone, frame, forward speed, facing, vy, action.
                rows.append(np.concatenate([w[:, :6], w[:, [9, 11, 7, 10]]], 1))
            else:
                rows.append(w[:, :6])
        self.pool.tr_bind(ids, rows, [0] * len(js), [float(radius[k]) for k in range(len(js))], 4, self.a.wp_bonus, self.a.finish, getattr(self.a, "matching", 0.0) if getattr(self.a, "full", False) else 0.0)

    def _zone_changed(self, j, cur_t):
        """Whether the game has warped into the zone of waypoint frame cur_t."""
        st = self.pool.status([self.ids[j]])
        return int(st["level"][0]) == int(self.tp.level[cur_t]) and int(st["area"][0]) == int(self.tp.area[cur_t])

    def obs(self, sub=None):
        """-> net inputs (x, ract, act, floor) on the device."""
        js = np.arange(len(self.ids)) if sub is None else np.asarray(sub)
        ids = [self.ids[j] for j in js]
        o = self.pool.tr_obs(ids)
        I = o["i"]
        last = self.n_wp[js] - 1
        W = np.stack([self.W[js, np.minimum(I + k, last)] for k in range(L.N_AHEAD)], 1)
        valid = np.stack([I + k <= last for k in range(L.N_AHEAD)], 1)
        left = self.W[js, last, 5] - self.W[js, np.minimum(I, last), 5]
        d = self.dev
        tt = lambda v, ty=torch.float32: torch.as_tensor(v, device=d, dtype=ty)
        x, ract = L.features(tt(o["f"]), tt(W), tt(valid, torch.bool), tt(left), tt(self.rad[js]), tt(self.sp[js]))
        out = (x, ract, tt(o["action"], torch.int64), tt(np.clip(o["floor_type"] + 1, 0, 255), torch.int64))
        if getattr(self.a, "ref", False):
            if not hasattr(self.tp, "tt"):
                self.tp.tt = L.TraceT(self.tp, d, copy=getattr(self.a, "copy", False))
            # Align: the nearest reference frame ahead (monotonic, bounded window).
            for k, j in enumerate(js):
                al = int(self.align[j])
                # Past the waypoint just passed; across a warp, into the new zone.
                ii = int(I[k])
                if ii > 0:
                    prev_t, cur_t = int(self.W[j, ii - 1, 5]), int(self.W[j, min(ii, int(last[k])), 5])
                    # A waypoint is passed up to `radius` early: exactly prev_t
                    # would be frames ahead of Mario, and a copy then wrong.
                    al = max(al, prev_t - 24)
                    tp = self.tp
                    if tp.level[prev_t] != tp.level[cur_t] or tp.area[prev_t] != tp.area[cur_t]:
                        if al <= prev_t and self._zone_changed(j, cur_t):
                            n = np.flatnonzero(tp.control[prev_t + 1 : cur_t + 1] & (tp.level[prev_t + 1 : cur_t + 1] == tp.level[cur_t]))
                            al = prev_t + 1 + int(n[0]) if len(n) else cur_t
                e = min(al + 48, int(self.tp.end[al])) + 1
                al2 = al + int(np.argmin(np.sum((self.tp.pos[al:e] - o["f"][k, :3]) ** 2, 1)))
                self.align[j] = al2
            af = self.tp.f[self.align[js]]
            self.al_dist = np.linalg.norm(af[:, :3] - o["f"][:, :3], axis=1) + np.abs(af[:, 6] - o["f"][:, 6])
            self.ref_pad = self.tp.pads[self.align[js]].astype(np.uint32)
            self.ref_cam = self.tp.f[self.align[js], 9].astype(np.int32)
            rx, ra = L.ref_feat(tt(o["f"]), self.tp.tt, tt(self.align[js], torch.int64))
            out = (torch.cat([out[0], rx], 1), torch.cat([out[1], ra], 1)) + out[2:]
        if getattr(self.a, "full", False):
            out = out + (L.full_feat(tt(o["f"]), tt(W), tt(valid, torch.bool)),)
        rt = getattr(self.a, "rt", None)
        if rt is not None:
            out = out + (rt.encode(self.pool.read(ids)),)
        return out


def do_act(g, js, ac):
    """Plays actions (numpy [n, 4]) in games js, right after g.obs(js)."""
    a = g.a
    base, rel, mag, btn, hold = P.decode(ac[:, 0], ac[:, 1], ac[:, 2], ac[:, 3])
    ids = [g.ids[j] for j in js]
    if getattr(a, "copy", False):
        cp = ac[:, 0] == P.N_DIR
        raw = np.where(cp, g.ref_pad, np.uint32(0xFFFFFFFF)).astype(np.uint32)
        hold = np.where(cp, 1, hold).astype(np.int32)
        r = g.pool.tr_act(ids, base, rel, mag, btn, hold, a.gamma, a.death, a.clip_r, raw, g.ref_cam)
        g.align[np.asarray(js)[cp]] += 1
        g.align[np.asarray(js)] = np.minimum(g.align[np.asarray(js)], g.tp.end[g.align[np.asarray(js)]])
        return r
    return g.pool.tr_act(ids, base, rel, mag, btn, hold, a.gamma, a.death, a.clip_r)


@torch.no_grad()
def choose(pol, inp, greedy=False):
    logits, v = pol(*inp)
    ds = P.dists(logits)
    ac = torch.stack([d.probs.argmax(1) if greedy else d.sample() for d in ds], 1)
    return ac, P.log_prob(ds, ac), v


def play(g, pol, device, js, a, greedy=True):
    """Runs games js to their end. -> per game (reached end, last index,
    number of waypoints, frames, term)."""
    alive = list(js)
    out = {}
    while alive:
        ac, _, _ = choose(pol, g.obs(alive), greedy=greedy)
        ac = ac.cpu().numpy()
        r = do_act(g, alive, ac)
        nxt = []
        for k, j in enumerate(alive):
            g.t[j] += int(r["frames"][k])
            term = int(r["term"][k])
            if term or g.t[j] >= g.limit[j]:
                _, hits = g.pool.tr_played(g.ids[j])
                out[j] = dict(ok=term == 1, i=int(r["i"][k]), n=int(g.n_wp[j]), frames=int(g.t[j]), term=term, hits=len(hits), ref=int(g.ref[j]))
            else:
                nxt.append(j)
        alive = nxt
    return out


def eval_grid(pool, ids, tp, pol, device, a, spacings, horizon, n_starts, rng, radius=100):
    """Deterministic following on `tp`, per spacing, from the same n starts."""
    g = Games(pool, ids, tp, a)
    saves = [s for s in tp.saves if tp.end[s[0]] - s[0] >= horizon // 2]
    picks = [saves[k] for k in rng.choice(len(saves), min(n_starts, len(saves)), replace=False)]
    res = {}
    for S in spacings:
        rows = []
        for b in range(0, len(picks), len(ids)):
            ch = picks[b : b + len(ids)]
            js = list(range(len(ch)))
            g.bind(js, [c[0] for c in ch], [S] * len(ch), [horizon] * len(ch), [radius] * len(ch), [c[1] for c in ch])
            rows += list(play(g, pol, device, js, a).values())
        ok = [r for r in rows if r["ok"]]
        res[int(S)] = dict(end=round(np.mean([r["ok"] for r in rows]), 3), prog=round(np.mean([r["i"] / max(r["n"] - 1, 1) if not r["ok"] else 1.0 for r in rows]), 3),
                           hit=round(np.mean([r["hits"] / max(r["n"], 1) for r in rows]), 3), ratio=round(float(np.median([r["frames"] / max(r["ref"], 1) for r in ok])), 3) if ok else None,
                           died=round(np.mean([r["term"] in (2, 3) for r in rows]), 3), n=len(rows))
    return res


def load_traces(pool, gid, a):
    """Train and held-out traces with savestates (every a.every frames)."""
    rng = np.random.default_rng(7)
    extra = [m for r in L.TRAIN_TREES for m in L.tree_movies(r, a.tree_nodes, rng)]
    tr = L.TracePool(pool, gid, L.movies(L.TRAIN) + [os.path.join(R.V2, L.TAS)] + extra, every=a.every, tas_window=(1646, 22390))
    held = [m for r in L.HELD_TREES for m in L.tree_movies(r, 60, np.random.default_rng(11))]
    va = L.TracePool(pool, gid, L.movies(L.HELD_OUT) + held, every=256, name="held-out")
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--frames", type=float, default=2e7)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--rollout", type=int, default=48)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--death", type=float, default=100.0)
    ap.add_argument("--finish", type=float, default=30.0)
    ap.add_argument("--wp-bonus", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--clip-r", type=float, default=50.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--bc-coef", type=float, default=0.3)
    ap.add_argument("--bc-batch", type=int, default=4096)
    ap.add_argument("--bc-decay", type=float, default=0.8)
    ap.add_argument("--limit-mult", type=float, default=2.0)
    ap.add_argument("--h0", type=int, default=128)
    ap.add_argument("--h-max", type=int, default=1024)
    ap.add_argument("--every", type=int, default=96)
    ap.add_argument("--tree-nodes", type=int, default=150)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--init", default=None, help="a BC checkpoint (train_bc.py)")
    ap.add_argument("--h", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ref", action="store_true", help="the reference's aligned inputs and state join the observation")
    ap.add_argument("--copy", action="store_true", help="an action playing the reference's aligned input, frame exact (needs --ref)")
    ap.add_argument("--dagger", default=None, help="GA recoveries off the trace (dagger_ga.py) as extra BC data")
    ap.add_argument("--dagger-coef", type=float, default=0.3)
    ap.add_argument("--full", action="store_true", help="full-target waypoints: speed, facing, vy and action group in the observation and reward")
    ap.add_argument("--matching", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true", help="the fast round's data (follow_lib.fast_sets) and starts")
    ap.add_argument("--p-noise", type=float, default=0.0, help="share of episodes starting with 1-60 frames of noise, off the trace")
    ap.add_argument("--fast-weight", type=float, default=3.0)
    ap.add_argument("--z-ckpt", default=None, help="backbone checkpoint: its z joins the observation (init weights from --init, z columns zero)")
    ap.add_argument("--v-warm", type=int, default=0, help="updates training only the value, policy frozen (a BC init has no value)")
    ap.add_argument("--kl", type=float, default=0.0, help="stop an update's epochs past this approximate KL")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    sm = R.setup()
    E, M = a.envs, 32
    pool = R.make_pool(sm, E + M + 1, a.threads)
    gid = E + M
    os.makedirs(CK, exist_ok=True), os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    tr, va = L.fast_sets(pool, gid, a.every, a.tree_nodes) if a.fast else load_traces(pool, gid, a)
    print(f"traces {time.time() - t0:.0f} s", flush=True)
    tt = L.TraceT(tr, device, fast_weight=a.fast_weight, copy=a.copy)
    tr.tt = tt
    sp_t, rad_t = torch.as_tensor(SPACINGS, device=device), torch.as_tensor(RADII, device=device)
    a.rt = None
    if a.z_ckpt:
        from sm64bb.runtime import Runtime

        a.rt = Runtime(a.z_ckpt)
    pol = L.FollowNet(d_z=a.rt.m.d if a.rt else (L.N_FULL if a.full else 0), h=a.h, layers=a.layers, ref=a.ref, copy=a.copy).to(device)
    if a.init:
        sd = torch.load(a.init, map_location=device, weights_only=False)["policy"]
        own = pol.state_dict()
        for k, v in sd.items():
            if own[k].shape == v.shape:
                own[k] = v
            elif v.dim() == 2 and own[k].shape[1] > v.shape[1]:  # A first layer, now with z columns: those start at zero.
                w = torch.zeros_like(own[k])
                w[:, : v.shape[1]] = v
                own[k] = w
            else:  # The direction head, now with the copy action: it starts small.
                w = torch.zeros_like(own[k])
                w[: v.shape[0]] = v
                own[k] = w
        pol.load_state_dict(own)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr, eps=1e-5)
    evals = list(range(E, E + M))
    print(json.dumps({"start_val": eval_grid(pool, evals, va, pol, device, a, [16, 64, 256], 256, 64, np.random.default_rng(1))}), flush=True)

    dag = None
    if a.dagger:
        dag = {k: v.to(device) for k, v in torch.load(a.dagger, map_location="cpu", weights_only=False).items()}
        print("dagger samples", len(dag["x"]), flush=True)
    g = Games(pool, list(range(E)), tr, a)
    hmax = a.h0
    saves = tr.saves

    fast_saves = [s for s in saves if abs(tr.speed[s[0]]) > 35]
    print(f"starts: {len(saves)}, fast {len(fast_saves)}", flush=True)

    def reset(js):
        picks, sp, hz, rad, hs = [], [], [], [], []
        for _ in js:
            while True:
                src = fast_saves if a.fast and fast_saves and rng.random() < 0.6 else saves
                t, h = src[rng.integers(len(src))]
                if tr.end[t] - t >= 32:
                    break
            H = int(2 ** rng.uniform(5, np.log2(hmax)))
            S = int(rng.choice(SPACINGS[SPACINGS <= max(16, H)]))
            picks.append(t), sp.append(S), hz.append(max(H, S)), rad.append(int(rng.choice(RADII))), hs.append(h)
        if a.p_noise > 0:
            # Off the trace: noise from the save, then the trace from K frames on.
            ids = [g.ids[j] for j in js]
            pool.load(ids, hs)
            K = np.where(rng.random(len(js)) < a.p_noise, (2 ** rng.uniform(0, np.log2(60), len(js))).astype(int), 0)
            Kmax = int(K.max())
            if Kmax:
                yw = rng.uniform(0, 2 * np.pi, len(js))
                x = (64 * np.sin(yw)).astype(np.int64) & 0xFF
                y = (64 * np.cos(yw)).astype(np.int64) & 0xFF
                btn = np.where(rng.random(len(js)) < 0.2, 0x8000, 0).astype(np.int64)
                base = (btn << 16) | (x << 8) | y
                for b in range(Kmax):
                    live = [i for i, k in zip(ids, K) if k > b]
                    pool.step(live, base[K > b].astype(np.uint32)[:, None])
            picks = [int(min(t + k, tr.end[t] - 1)) for t, k in zip(picks, K)]
            g.bind(js, picks, sp, hz, rad, None)
        else:
            g.bind(js, picks, sp, hz, rad, hs)

    reset(list(range(E)))
    frames, upd = 0, 0
    win = collections.deque(maxlen=1024)
    mllog.start(a.name, {k: v for k, v in vars(a).items() if k != "rt"})
    log = open(os.path.join(LOGS, f"{a.name}.jsonl"), "a")
    best = -1.0
    t0 = time.time()
    while frames < a.frames:
        buf = collections.defaultdict(list)
        for _ in range(a.rollout):
            inp = g.obs()
            ac, lp, v = choose(pol, inp)
            r = do_act(g, list(range(E)), ac.cpu().numpy())
            k = r["frames"].astype(np.float64)
            frames += int(k.sum())
            g.t += r["frames"]
            disc = a.gamma**k
            rew = r["ret"] * a.scale
            term = r["term"]
            timeout = (g.t >= g.limit) & (term == 0)
            done = (term != 0) | timeout
            if timeout.any():
                js = np.flatnonzero(timeout)
                _, _, v2 = choose(pol, g.obs(js))
                rew[js] += disc[js] * v2.cpu().numpy()
            buf["inp"].append(inp)
            buf["ac"].append(ac), buf["lp"].append(lp), buf["v"].append(v)
            buf["rew"].append(rew.astype(np.float32)), buf["done"].append(done), buf["disc"].append(disc.astype(np.float32))
            if done.any():
                js = np.flatnonzero(done).tolist()
                for j in js:
                    win.append((term[j] == 1, term[j] in (2, 3), g.t[j] / max(g.ref[j], 1), g.sp[j], g.ref[j]))
                reset(js)
        _, _, vlast = choose(pol, g.obs())
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
        IN = [torch.cat([i[k] for i in buf["inp"]]) for k in range(len(buf["inp"][0]))]
        AC, LP = torch.cat(buf["ac"]), torch.cat(buf["lp"])
        ADV = torch.as_tensor(adv.reshape(-1), device=device)
        RET = torch.as_tensor(ret.reshape(-1), device=device)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        N = T * E
        coef = a.bc_coef * max(0.0, 1 - frames / (a.bc_decay * a.frames))
        st = collections.defaultdict(list)
        warm = upd < a.v_warm
        for _ in range(a.epochs):
            kls = []
            perm = torch.randperm(N, device=device)
            for b in range(0, N, a.mb):
                mb = perm[b : b + a.mb]
                logits, v = pol(*[w[mb] for w in IN])
                if warm:
                    l_v = F.mse_loss(v, RET[mb])
                    opt.zero_grad()
                    l_v.backward()
                    # Only the value's parameters move.
                    for n_, p_ in pol.named_parameters():
                        if not n_.startswith("v") and p_.grad is not None:
                            p_.grad = None
                    nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                    opt.step()
                    st["v"].append(l_v.item()), st["ent"].append(0.0), st["clip"].append(0.0)
                    continue
                ds = P.dists(logits)
                new_lp = P.log_prob(ds, AC[mb])
                kls.append((LP[mb] - new_lp).mean().item())
                ratio = torch.exp(new_lp - LP[mb])
                l_pi = -torch.min(ratio * ADV[mb], ratio.clamp(0.8, 1.2) * ADV[mb]).mean()
                l_v = F.mse_loss(v, RET[mb])
                en = P.entropy(ds).mean()
                loss = l_pi + a.vf * l_v - a.ent * en
                if dag is not None:
                    ix = torch.randint(len(dag["x"]), (a.bc_batch,), device=device)
                    ld, _ = pol(dag["x"][ix], dag["ract"][ix], dag["act"][ix], dag["floor"][ix])
                    # The GA's holds mean nothing: its direction, strength and buttons.
                    loss = loss + a.dagger_coef * sum(F.cross_entropy(ld[k], dag["y"][ix, k]) for k in range(3))
                if coef > 0:
                    xb, rb, ab, fb, yt = tt.batch(a.bc_batch, sp_t, rad_t, ref=a.ref)
                    lb, _ = pol(xb, rb, ab, fb)
                    loss = loss + coef * sum(F.cross_entropy(lg, yt[:, k]) for k, lg in enumerate(lb))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                opt.step()
                st["v"].append(l_v.item()), st["ent"].append(en.item()), st["clip"].append(((ratio - 1).abs() > 0.2).float().mean().item())
            if a.kl and kls and np.mean(kls) > a.kl:
                break
        upd += 1
        sr = float(np.mean([w[0] for w in win])) if win else 0.0
        if sr > 0.7 and len(win) >= 512 and hmax < a.h_max:
            hmax = min(a.h_max, int(hmax * 1.25))
            win.clear()
        rec = {"upd": upd, "frames": frames, "fps": int(frames / (time.time() - t0)), "sr": round(sr, 3), "died": round(float(np.mean([w[1] for w in win])), 3) if win else None,
               "ratio": round(float(np.median([w[2] for w in win if w[0]])), 3) if any(w[0] for w in win) else None, "hmax": hmax, "rew": round(float(np.mean(buf["rew"])), 4),
               "v": round(float(np.mean(st["v"])), 4), "ent": round(float(np.mean(st["ent"])), 3), "clip": round(float(np.mean(st["clip"])), 3), "rss_gb": round(int(open('/proc/self/statm').read().split()[1]) * 4096 / 1e9, 2)}
        if upd % a.eval_every == 0:
            ev = eval_grid(pool, evals, va, pol, device, a, [16, 64, 256], 256, 64, np.random.default_rng(1))
            ev2 = eval_grid(pool, evals, tr, pol, device, a, [64], 256, 32, np.random.default_rng(1))
            rec["val"], rec["train64"] = ev, ev2[64]
            score = np.mean([v["prog"] for v in ev.values()])
            torch.save({"policy": pol.state_dict(), "args": {k: v for k, v in vars(a).items() if k != "rt"}, "frames": frames, "score": score}, os.path.join(CK, f"{a.name}.last.pt"))
            if score > best:
                best = score
                torch.save({"policy": pol.state_dict(), "args": {k: v for k, v in vars(a).items() if k != "rt"}, "frames": frames, "score": score}, os.path.join(CK, f"{a.name}.best.pt"))
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n"), log.flush()
        mllog.log(rec, frames)


if __name__ == "__main__":
    main()
