#!/usr/bin/env python3
"""Run-time checks of the model against a reference (what race.py and
follow.py rely on):

    .venv/bin/python diag_runtime.py tree14 [milestone=1] [n=128]

1. Distance calibration: along the reference segment up to the milestone,
   model frames-to-goal vs the true remaining frames.
2. Closed-loop inverse dynamics: from reference states, apply the inverse
   head's inputs toward the reference state k frames later, and measure how
   far Mario ends up from it (and from where k frames of doing nothing or
   of the reference's own inputs would leave him).
3. Timing of the pieces of one racer iteration (load, step, read, encode,
   distance, save) at the racer's batch size.
"""

import os
import sys
import time

import numpy as np
import torch

from sm64bb.refs import build_reference, make_pool
from sm64bb.runtime import Runtime
from train import spearman

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "tree14"
    k_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    ref = build_reference(name)
    rt = Runtime(os.path.join(HERE, "checkpoints", "v3_last.pt"))
    pool = make_pool(n + 1)
    R = n
    m = ref.milestones[k_ms - 1]
    r0 = ref.milestones[k_ms - 2]["frame"] if k_ms > 1 else 1700
    r1 = m["frame"]
    # Reference states every 16 frames from r0 to r1.
    frames = list(range(r0, r1, 16))
    pool.step([R], ref.pads[:r0][None, :])
    hs, xs, pos = [], [], []
    for f in frames:
        cur = int(pool.status([R])["frame"][0])
        if f > cur:
            pool.step([R], ref.pads[cur:f][None, :])
        hs.append(pool.save([R])[0])
        xs.append(pool.read([R]))
        pos.append(pool.status([R])["pos"][0])
    cur = int(pool.status([R])["frame"][0])
    pool.step([R], ref.pads[cur:r1][None, :])
    zg = rt.encode(pool.read([R]))
    x = {k: np.concatenate([d[k] for d in xs]) for k in xs[0]}
    z = rt.encode(x)
    pos = np.array(pos)
    d = rt.distance(z, zg.expand(len(z), -1)).cpu().numpy()
    true = r1 - np.array(frames)
    print(f"{name} milestone {k_ms} (frames {r0}..{r1}, {len(frames)} states)")
    print(f"1. distance to goal: spearman {spearman(torch.tensor(d), torch.tensor(true.astype(np.float32))):.3f}, "
          f"median pred/true {np.median(d / np.maximum(true, 1)):.2f}, "
          f"|log ratio| {np.mean(np.abs(np.log1p(d) - np.log1p(true))):.2f}")
    for q in np.linspace(0, len(frames) - 1, 6).astype(int):
        print(f"     {true[q]:>6} frames left -> model {d[q]:8.0f}")
    # 2. Closed-loop inverse dynamics.
    for k in (16, 64):
        sel = [i for i in range(len(frames)) if i + k // 16 < len(frames)][:n]
        if not sel:
            continue
        ids = list(range(len(sel)))
        za = z[sel]
        zb = z[[i + k // 16 for i in sel]]
        target = pos[[i + k // 16 for i in sel]]
        res = {}
        for label, pads in [("inverse", rt.inverse(za, zb, k)),
                            ("reference", np.stack([ref.pads[frames[i] : frames[i] + k] for i in sel])),
                            ("idle", np.zeros((len(sel), k), np.uint32))]:
            pool.load(ids, [hs[i] for i in sel])
            pool.step(ids, pads)
            p = pool.status(ids)["pos"]
            res[label] = np.linalg.norm(p - target, axis=1)
        moved = np.linalg.norm(target - pos[sel], axis=1)
        print(f"2. k={k}: distance to the target reference state after {k} frames (median units): "
              + ", ".join(f"{l} {np.median(v):.0f}" for l, v in res.items()) + f"; reference moves {np.median(moved):.0f}")
    # 3. Timing at the racer's batch.
    ids = list(range(n))
    src = [hs[i % len(hs)] for i in ids]
    pads = np.stack([ref.pads[frames[i % len(frames)] : frames[i % len(frames)] + 16] for i in ids])
    t = {}
    for rep in range(3):
        t0 = time.time(); pool.load(ids, src); t1 = time.time()
        pool.step(ids, pads, stop_mask=0); t2 = time.time()
        xb = pool.read(ids); t3 = time.time()
        zb = rt.encode(xb); torch.cuda.synchronize(); t4 = time.time()
        rt.distance(zb, zg.expand(n, -1)); torch.cuda.synchronize(); t5 = time.time()
        pool.save(ids); t6 = time.time()
        pool.status(ids); t7 = time.time()
        t = dict(load=t1 - t0, step=t2 - t1, read=t3 - t2, encode=t4 - t3, distance=t5 - t4, save=t6 - t5, status=t7 - t6)
    print(f"3. one batch of {n} children x 16 frames: " + ", ".join(f"{k} {1e3 * v:.0f} ms" for k, v in t.items()))


if __name__ == "__main__":
    main()
