#!/usr/bin/env python3
"""Goal-conditioned RL policy for objective-rrt's objectives (PPO), from
start states where objective-rrt already succeeded at that objective.

    .venv/bin/python rl_objective.py [--tree pf-mgel-lg-pf-s3] [--steps 20000000] [--envs 128]

- Tasks: (start node, objective key) pairs of successful attempts in the
  tree's events.jsonl, at most --per-kind per objective kind (dialogs and
  menus left out). Start states are rebuilt from the tree's inputs
  (sm64env Pool.replay); the episode limit is 3x the frames objective-rrt's
  attempt took (+60, at most 900).
- Observation: the frozen backbone encoder's z of the state, the
  objective's kind, its steer target relative to Mario (if any), and the
  time left.
- Action every 4 frames (18 x 6): stick neutral, one of 8 directions
  relative to Mario's facing at 2 strengths, or straight at the objective's
  steer target (camera-relative on the stick, as the game reads it); and
  buttons: none, A, B, Z, A then B (dive kick), Z then A (long jump).
  The target is observed in Mario's frame (forward, right, up): a scripted
  "steer at the target" policy beats random (26 % vs 19 % success) while RL
  v1, with the target in world axes, learned nothing.
- Reward: the objective's own progress score's change (objective-rrt's
  `progress`, normalised per kind), +5 on success (`success`), -1 on death,
  -0.002 per step. Episodes end at success, death or the limit.
Logged to MLflow (experiment sm64-rl); policy saved to checkpoints/rl/.
"""

import argparse
import collections
import json
import math
import os
import struct
import time

import mlflow

from sm64bb import mlog
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sm64bb.refs import make_pool, pack
from sm64bb.runtime import Runtime

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "objective-rrt", "runs")
DIRS = [i * 2 * math.pi / 8 for i in range(8)]
A, B, Z = 0x8000, 0x4000, 0x2000
BUTTONS = ["none", "A", "B", "Z", "A>B", "Z>A"]
# 17 stick choices relative to Mario's facing, plus one "steer toward the
# objective's target" (stick straight at it), each with every button pattern.
N_STICK = 18
N_ACT = N_STICK * len(BUTTONS)
FRAMES = 4


def read_tree(path):
    data = open(path, "rb").read()
    n = struct.unpack_from("<I", data, 0)[0]
    pos, out = 4, []
    for _ in range(n):
        par, k = struct.unpack_from("<iI", data, pos)
        pos += 8
        a = np.frombuffer(data, np.uint8, 4 * k, pos).reshape(-1, 4)
        pads = ((a[:, 0].astype(np.uint32) << 24) | (a[:, 1].astype(np.uint32) << 16) | (a[:, 2].astype(np.uint32) << 8) | a[:, 3])
        out.append((par, pads))
        pos += 4 * k
    return out


def path_pads(records, node):
    chain, v = [], node
    while v >= 0:
        chain.append(records[v + 1][1])
        v = records[v + 1][0]
    return np.concatenate([records[0][1]] + chain[::-1])


def load_tasks(tree, per_kind, rng):
    import pandas as pd
    frame = pd.read_csv(os.path.join(RUNS, tree, "nodes.csv"), usecols=["id", "frame"]).frame.values
    by = collections.defaultdict(list)
    for line in open(os.path.join(RUNS, tree, "events.jsonl")):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not e.get("success") or e.get("added") is None or e.get("stage", 0) != 0 or e.get("from") is None:
            continue
        k = e["kind"]
        if k.startswith(("Dialog", "Menu", "Pause", "Save")):
            continue
        frames = int(frame[e["added"]] - frame[e["from"]])
        if 4 <= frames <= 600:
            by[k].append((int(e["from"]), e["key"], k, frames))
    tasks = []
    for k, v in by.items():
        rng.shuffle(v)
        tasks.extend(v[:per_kind])
    return tasks


def action_pads(acts, st, target):
    """Action ids [n] -> packed pads [n, FRAMES] for states st (status dict)
    and the objective's steer targets [n, 3] (NaN: none)."""
    n = len(acts)
    stick_i, btn_i = acts // len(BUTTONS), acts % len(BUTTONS)
    yaw = st["yaw"].astype(np.float64) * 2 * math.pi / 65536
    cam = st["cam_yaw"].astype(np.float64) * 2 * math.pi / 65536
    steer = stick_i == N_STICK - 1
    d = np.nan_to_num(target - st["pos"])
    to_tgt = np.arctan2(d[:, 0], d[:, 2])
    has = np.isfinite(target).all(1)
    mag = np.where(stick_i == 0, 0, np.where(stick_i <= 8, 64, 32))
    mag = np.where(steer, np.where(has, 64, 0), mag)
    off = np.array(DIRS)[(np.maximum(stick_i, 1) - 1) % 8]
    a = np.where(steer, to_tgt, yaw + off) - cam
    sx = np.round(mag * np.sin(a)).astype(np.int64)
    sy = np.round(-mag * np.cos(a)).astype(np.int64)
    btn = np.zeros((n, FRAMES), np.int64)
    for j, b in enumerate(BUTTONS):
        m = btn_i == j
        if b == "A":
            btn[m] = A
        elif b == "B":
            btn[m] = B
        elif b == "Z":
            btn[m] = Z
        elif b == "A>B":
            btn[m, :2], btn[m, 2:] = A, B
        elif b == "Z>A":
            btn[m, :1], btn[m, 1:] = Z, A
    return pack(btn, np.repeat(sx[:, None], FRAMES, 1), np.repeat(sy[:, None], FRAMES, 1))


class Policy(nn.Module):
    def __init__(self, d, n_kinds, h=512):
        super().__init__()
        self.kind = nn.Embedding(n_kinds, 32)
        inp = d + 32 + 4 + 1
        self.body = nn.Sequential(nn.Linear(inp, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.pi = nn.Linear(h, N_ACT)
        self.v = nn.Linear(h, 1)

    def forward(self, z, kind, tgt, tleft):
        x = self.body(torch.cat([z, self.kind(kind), tgt, tleft[:, None]], -1))
        return self.pi(x), self.v(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="pf-mgel-lg-pf-s3")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "v3_last.pt"))
    ap.add_argument("--steps", type=int, default=20_000_000, help="environment frames")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--per-kind", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default="rl-objectives")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    for k_, v_ in {"OBRRT_MOVES": "1", "OBRRT_MOVE_GA": "1", "OBRRT_TECH": "all", "OBRRT_PAINTING_FAST": "1"}.items():
        os.environ.setdefault(k_, v_)
    import sm64env
    sm64env.Pool.configure()
    rt = Runtime(a.ckpt)
    E = a.envs
    pool = make_pool(E + 1)
    tasks = load_tasks(a.tree, a.per_kind, rng)
    kinds = sorted({t[2] for t in tasks})
    kid = {k: i for i, k in enumerate(kinds)}
    print(f"{len(tasks)} tasks over {len(kinds)} kinds from {a.tree}", flush=True)
    # Start states, rebuilt once per start node.
    records = read_tree(os.path.join(RUNS, a.tree, "tree.bin"))
    t0 = time.time()
    starts = {}
    for node in sorted({t[0] for t in tasks}):
        starts[node] = pool.replay(E, path_pads(records, node))
    print(f"{len(starts)} start states rebuilt in {time.time() - t0:.0f} s", flush=True)

    pol = Policy(rt.m.d, len(kinds)).to(rt.dev)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    prog_scale = collections.defaultdict(lambda: 1.0)

    ids = list(range(E))
    task = [None] * E
    limit = np.zeros(E, np.int64)
    t_ep = np.zeros(E, np.int64)
    last_prog = np.zeros(E)

    def reset(js):
        for j in js:
            while True:
                t = tasks[rng.integers(len(tasks))]
                pool.load([j], [starts[t[0]]])
                if pool.bind([j], [t[1]], int(rng.integers(1 << 60)))[0]:
                    break
            task[j] = t
            limit[j] = min(900, 3 * t[3] + 60)
            t_ep[j] = 0
        r = pool.rl_step(list(js), np.zeros((len(js), 0), np.uint32))
        last_prog[list(js)] = r["progress"]

    reset(ids)

    def observe():
        st = pool.status(ids)
        z = rt.encode(pool.read(ids))
        r = pool.rl_step(ids, np.zeros((E, 0), np.uint32))
        d = r["target"] - st["pos"].astype(np.float32)
        has = np.isfinite(d).all(1)
        d = np.nan_to_num(d)
        # In Mario's frame: forward (his facing), right, up.
        yaw = st["yaw"].astype(np.float64) * 2 * math.pi / 65536
        fwd = d[:, 0] * np.sin(yaw) + d[:, 2] * np.cos(yaw)
        right = d[:, 0] * np.cos(yaw) - d[:, 2] * np.sin(yaw)
        tgt = np.stack([fwd, right, d[:, 1]], 1)
        tgt = np.where(has[:, None], np.sign(tgt) * np.log1p(np.abs(tgt) / 1000), 0)
        tg = torch.as_tensor(np.concatenate([tgt, has[:, None]], 1), dtype=torch.float32, device=rt.dev)
        tl = torch.as_tensor(1 - t_ep / limit, dtype=torch.float32, device=rt.dev)
        kk = torch.as_tensor([kid[t[2]] for t in task], device=rt.dev)
        return st, z, kk, tg, tl, r["target"]

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
    if mlflow.get_experiment_by_name("sm64-rl") is None:
        mlflow.create_experiment("sm64-rl", artifact_location=os.path.join(HERE, "mlartifacts"))
    mlflow.set_experiment("sm64-rl")
    os.makedirs(os.path.join(HERE, "checkpoints", "rl"), exist_ok=True)
    frames, upd = 0, 0
    done_log = collections.deque(maxlen=2000)
    with mlflow.start_run(run_name=a.run_name):
        mlflow.log_params({**vars(a), "tasks": len(tasks), "kinds": len(kinds), "encoder_step": rt.step})
        while frames < a.steps:
            buf = collections.defaultdict(list)
            for _ in range(a.rollout):
                st, z, kk, tg, tl, target = observe()
                with torch.no_grad():
                    logits, v = pol(z, kk, tg, tl)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                pads = action_pads(act.cpu().numpy(), st, target)
                r = pool.rl_step(ids, pads)
                frames += E * FRAMES
                t_ep += FRAMES
                dp = r["progress"] - last_prog
                rew = np.zeros(E)
                for j in range(E):
                    k = task[j][2]
                    if np.isfinite(dp[j]):
                        prog_scale[k] = 0.99 * prog_scale[k] + 0.01 * max(abs(dp[j]), 1e-3)
                        rew[j] = np.clip(dp[j] / (prog_scale[k] * 4), -1, 1)
                rew -= 0.002
                succ = r["success"] >= 0
                died = r["died"] & ~succ
                rew += 5 * succ - 1 * died
                timeout = t_ep >= limit
                done = succ | died | timeout
                last_prog[:] = r["progress"]
                for k_, v_ in (("z", z), ("kind", kk), ("tgt", tg), ("tl", tl), ("act", act), ("logp", dist.log_prob(act)),
                               ("v", v)):
                    buf[k_].append(v_)
                buf["rew"].append(torch.as_tensor(rew, dtype=torch.float32, device=rt.dev))
                buf["done"].append(torch.as_tensor(done, dtype=torch.float32, device=rt.dev))
                for j in np.flatnonzero(done):
                    done_log.append((task[j][2], bool(succ[j]), int(t_ep[j]), task[j][3]))
                if done.any():
                    reset(np.flatnonzero(done).tolist())
            # GAE.
            with torch.no_grad():
                _, z, kk, tg, tl, _ = observe()
                _, v_last = pol(z, kk, tg, tl)
            T = len(buf["rew"])
            adv = torch.zeros(T, E, device=rt.dev)
            last = torch.zeros(E, device=rt.dev)
            vals = torch.stack(buf["v"])
            for t in reversed(range(T)):
                nv = v_last if t == T - 1 else vals[t + 1]
                nd = 1 - buf["done"][t]
                delta = buf["rew"][t] + 0.99 * nv * nd - vals[t]
                last = delta + 0.99 * 0.95 * nd * last
                adv[t] = last
            ret = adv + vals
            flat = lambda k: torch.stack(buf[k]).reshape(T * E, *buf[k][0].shape[1:])
            zf, kf, tf, lf, af, pf = map(flat, ("z", "kind", "tgt", "tl", "act", "logp"))
            advf, retf = adv.reshape(-1), ret.reshape(-1)
            advf = (advf - advf.mean()) / (advf.std() + 1e-8)
            for _ in range(4):
                perm = torch.randperm(T * E, device=rt.dev)
                for i in range(0, T * E, 1024):
                    mb = perm[i : i + 1024]
                    logits, v = pol(zf[mb], kf[mb], tf[mb], lf[mb])
                    d_ = torch.distributions.Categorical(logits=logits)
                    ratio = torch.exp(d_.log_prob(af[mb]) - pf[mb])
                    l_pi = -torch.min(ratio * advf[mb], ratio.clamp(0.8, 1.2) * advf[mb]).mean()
                    l_v = F.mse_loss(v, retf[mb])
                    loss = l_pi + 0.5 * l_v - 0.01 * d_.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol.parameters(), 0.5)
                    opt.step()
            upd += 1
            if done_log:
                sr = np.mean([d[1] for d in done_log])
                ratio = [d[2] / max(d[3], 1) for d in done_log if d[1]]
                m = {"success_rate": sr, "frames_vs_tree": float(np.median(ratio)) if ratio else 0.0,
                     "reward_mean": float(torch.stack(buf["rew"]).mean()), "entropy": float(d_.entropy().mean()),
                     "frames": frames}
                mlog.log_metrics(m, step=upd)
                if upd % 10 == 0:
                    per = collections.defaultdict(list)
                    for k, s_, _, _ in done_log:
                        per[k].append(s_)
                    top = sorted(per.items(), key=lambda kv: -len(kv[1]))[:8]
                    print(json.dumps({"update": upd, "frames": frames, **{k: round(v, 3) for k, v in m.items()},
                                      "by_kind": {k: round(float(np.mean(v)), 2) for k, v in top}}), flush=True)
                    torch.save({"policy": pol.state_dict(), "kinds": kinds, "encoder": a.ckpt, "frames": frames},
                               os.path.join(HERE, "checkpoints", "rl", "policy.pt"))


if __name__ == "__main__":
    main()
