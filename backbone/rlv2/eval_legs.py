#!/usr/bin/env python3
"""A route policy's deterministic result on its legs, replayed from power-on
through a fresh game; .m64 written. Also best-of-N stochastic rollouts.

    .venv/bin/python rlv2/eval_legs.py checkpoints/rlv2/speedchain-a.best.pt --samples 256
"""

import argparse
import json
import os

import numpy as np
import torch

import policy as P
import routes as R
import train_rl as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--threads", type=int, default=4)
    x = ap.parse_args()
    device = torch.device("cuda")
    ck = torch.load(x.ckpt, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    sm = R.setup()
    E = 64
    pool = R.make_pool(sm, E + 2, x.threads)
    rid, names, _ = R.load_route(pool, os.path.join(R.V2, "tools", "spec", "routes", a.route))
    legs = [names.index(n) for n in a.legs]
    gid = E
    demos = []
    for spec in a.demo:
        path, _, st = spec.partition("@")
        path = path if path.startswith("/") else os.path.join(R.V2, path)
        demos.append(T.bind_demo(pool, rid, names, path, legs[0], max(legs), gid, start=int(st) if st else None))
    starts = []
    for k, d in enumerate(demos):
        ls = [l for l in d.legs if l["leg"] in legs]
        if getattr(a, "seq", False):
            L0, Le = ls[0], ls[-1]
            starts.append(dict(name=f"seq@{k}", handle=L0["saves"][0][1], frame=L0["saves"][0][0], limit=int(2 * (Le["done"] - L0["saves"][0][0])) + 60, demo=d, demo_frames=Le["done"] - L0["saves"][0][0], leg=L0["leg"]))
        else:
            for l in ls:
                starts.append(dict(name=f"{l['name']}@{k}", handle=l["saves"][0][1], frame=l["saves"][0][0], limit=int(2 * (l["done"] - l["saves"][0][0])) + 60, demo=d, demo_frames=l["done"] - l["saves"][0][0], leg=l["leg"]))
    pol = P.Policy(len(names)).to(device)
    pol.load_state_dict(ck["policy"])
    pol.eval()
    lm = ck["landmarks"]
    ids = list(range(E))
    greedy = T.evaluate(pool, pol, None, starts, lm, a, ids, device, greedy=True)
    greedy2 = T.evaluate(pool, pol, None, starts, lm, a, ids, device, greedy=True)
    stoch = T.evaluate(pool, pol, None, starts, lm, a, ids, device, greedy=False, samples=x.samples)
    for st, g, g2, s in zip(starts, greedy, greedy2, stoch):
        rec = dict(start=st["name"], demo_frames=st["demo_frames"], greedy=g["frames"], repeat_same=g["frames"] == g2["frames"], sampled=dict(sr=round(s["ok"] / max(s["n"], 1), 3), best=min(s["frames"]) if s["frames"] else None, med=int(np.median(s["frames"])) if s["frames"] else None), frames_trained=ck.get("frames"))
        for tag, res in (("greedy", g), ("best", s)):
            if res["best"] is None:
                continue
            fr, pads, dones = res["best"]
            full = np.concatenate([st["demo"].pads[: st["frame"]], pads])
            # Fresh replay: power-on, the prefix, then the policy's pads, overlay already in them.
            pool.reset([gid])
            pool.step([gid], full[: st["frame"]][None, :])
            pool.rl_bind([gid], rid, st["leg"], max(legs), False)
            r = pool.rl_raw([gid], full[st["frame"] :][None, :], stop_at_leg=False)
            _, d2 = pool.rl_played(gid)
            rec[f"{tag}_replay"] = dict(term=int(r["term"][0]), done_at=(d2[-1][1] - st["frame"]) if d2 else None, per_leg=[(names[l], int(f - st["frame"])) for l, f in d2])
            out = os.path.join(R.BB, "logs", "rlv2", f"{os.path.basename(x.ckpt)[:-3]}.{st['name']}.{tag}.m64")
            R.write_m64(out, full)
            rec[f"{tag}_m64"] = out
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
