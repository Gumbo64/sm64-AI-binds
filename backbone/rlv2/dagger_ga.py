#!/usr/bin/env python3
"""Expert iteration for the follower: the GA as the expert off the trace. From
states knocked off a *training* trace (fast ones mostly), a seeded GA reaches the
trace again D frames on; each success's frames, seen through the follower's own
observation, become labelled data where hindsight BC has none. GPU-free but for
the features. Output: data/rlv2/dagger.pt (appended on each save).

    .venv/bin/python rlv2/dagger_ga.py --minutes 40 --threads 3
"""
import argparse, json, os, time
import numpy as np, torch
import follow_lib as L, policy as P, routes as R
from train_follow import Games

ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=40); ap.add_argument("--threads", type=int, default=3); ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--scale", type=float, default=4.0); ap.add_argument("--p-fast", type=float, default=0.75); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
class Args:
    gamma, death, clip_r, wp_bonus, finish, limit_mult, ref, copy, rt = 0.99, 100.0, 50.0, 1.0, 30.0, 2.0, True, False, None
dev = torch.device("cuda"); sm = R.setup(); E = a.batch; pool = R.make_pool(sm, E + 1, a.threads)
rng = np.random.default_rng(a.seed)
tr, _ = L.fast_sets(pool, E, 96, 100)
tt = L.TraceT(tr, dev); tr.tt = tt
fast = [s for s in tr.saves if abs(tr.speed[s[0]]) > 35 and tr.end[s[0]] - s[0] > 200]
slow = [s for s in tr.saves if abs(tr.speed[s[0]]) > 5 and tr.end[s[0]] - s[0] > 200]
print(f"starts fast {len(fast)}, others {len(slow)}", flush=True)
g = Games(pool, list(range(E)), tr, Args())
store = {k: [] for k in ("x", "ract", "act", "floor", "y")}
out = os.path.join(R.BB, "data", "rlv2", "dagger.pt")
t0, tries, wins, frames, n = time.time(), 0, 0, 0, 0
while time.time() - t0 < a.minutes * 60:
    picks = [(fast if rng.random() < a.p_fast else slow) for _ in range(E)]
    picks = [p[rng.integers(len(p))] for p in picks]
    ids = g.ids
    pool.load(ids, [p[1] for p in picks])
    K = (2 ** rng.uniform(1, np.log2(40), E)).astype(int)
    yw = rng.uniform(0, 2 * np.pi, E); xs = (64 * np.sin(yw)).astype(np.int64) & 0xFF; ys = (64 * np.cos(yw)).astype(np.int64) & 0xFF
    for b in range(int(K.max())):
        live = [i for i in range(E) if K[i] > b]
        pool.step([ids[i] for i in live], ((xs[live] << 8) | ys[live]).astype(np.uint32)[:, None])
    hs = pool.save(ids)
    D = rng.choice([32, 64, 128], E)
    ts = [int(min(p[0] + k, tr.end[p[0]] - 1)) for p, k in zip(picks, K)]
    goals = np.stack([tr.pos[min(t + d, int(tr.end[t]))] for t, d in zip(ts, D)]).astype(np.float32)
    seeds = [tr.pads[t : t + d].astype(np.uint32) for t, d in zip(ts, D)]
    res = pool.attempt_to(hs, goals, seeds, a.scale, int(rng.integers(1 << 40)))
    tries += E; frames += sum(int(r["frames"]) for r in res)
    good = [i for i, r in enumerate(res) if r["success"] and r["handle"] is not None and 0 < len(r["inputs"]) <= 2 * int(D[i])]
    wins += len(good)
    for r in res:
        if r["handle"] is not None:
            pool.free([r["handle"]])
    # Each success, seen by the follower: its obs on the way, labelled by the GA's pads.
    for i in good:
        S = int(rng.choice([8, 16, 32, 64, 128])); Rd = int(rng.choice([60, 100, 200]))
        g.bind([0], [ts[i]], [S], [int(D[i])], [Rd], [hs[i]])
        inp = res[i]["inputs"]
        for pad in inp:
            o = g.obs([0])
            if bool(pool.tr_obs([0])["control"][0]):
                f = torch.as_tensor(pool.tr_obs([0])["f"], device=dev, dtype=torch.float32)
                wp = o[0].new_tensor(g.W[0, int(pool.tr_obs([0])["i"][0]), :3])[None]
                y = tt.label_of(torch.as_tensor([int(pad)], device=dev), f, wp)
                for k, v in zip(("x", "ract", "act", "floor"), o):
                    store[k].append(v[0].cpu())
                store["y"].append(y[0].cpu())
                n += 1
            r = pool.tr_raw([0], np.array([[pad]], np.uint32))
            if int(r["term"][0]):
                break
    pool.free(hs)
    if tries % (E * 20) == 0:
        torch.save({k: torch.stack(v) for k, v in store.items() if v}, out)
        print(json.dumps(dict(mins=round((time.time() - t0) / 60, 1), tries=tries, wins=wins, samples=n, ga_frames=frames)), flush=True)
torch.save({k: torch.stack(v) for k, v in store.items()}, out)
print("saved", n, flush=True)
