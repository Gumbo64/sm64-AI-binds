#!/usr/bin/env python3
"""Checks that sm64env (objective-rrt/pyenv) gives the model exactly the
inputs it was trained on: replays the 2026 120-star TAS through one game
and compares `Pool.read` with the dataset's samples (data/tas120) at the same
frames. Also times stepping, saving and loading.

    .venv/bin/python check_env.py [n_frames=20000] [every=97]
"""

import os
import sys
import time

import numpy as np

from sm64bb.data import Store
from sm64bb.refs import make_pool, read_m64
from train import RUN

HERE = os.path.dirname(os.path.abspath(__file__))
ROM = os.path.join(HERE, "..", "baserom.us.z64")
TAS = os.path.join(HERE, "..", "objective-rrt", "ref", "tas", "full-120star-2026-U.m64")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 97
    st = Store([os.path.join(HERE, "data", "tas120")], os.path.join(RUN, "nodes.csv"))
    by_frame = {int(f): i for i, f in enumerate(st.frame)}
    pads = read_m64(TAS)[:n]
    pool = make_pool(1)
    bad, checked = {}, 0
    t_step = 0.0
    for f in range(0, n, every):
        chunk = pads[f : f + every]
        t0 = time.time()
        pool.step([0], chunk[None, :])
        t_step += time.time() - t0
        frame = f + len(chunk)
        i = by_frame.get(frame)
        if i is None:
            continue
        x = pool.read([0])
        ref = st.state(np.array([i]))
        checked += 1
        for k in ("gi", "pad_prev", "oi", "om", "tm"):
            if not np.array_equal(x[k], ref[k]):
                bad.setdefault(k, []).append(frame)
        for k in ("gf", "of"):
            if not np.allclose(x[k], ref[k], atol=1e-4, rtol=1e-5, equal_nan=True):
                bad.setdefault(k, []).append(frame)
        # Triangles: same set (the encoder ignores order).
        a = np.sort(x["tri"][0][x["tm"][0]].round(3), axis=0)
        b = np.sort(ref["tri"][0][ref["tm"][0]].round(3), axis=0)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-3):
            bad.setdefault("tri", []).append(frame)
    print(f"compared {checked} TAS frames up to {n}: "
          + ("all inputs identical" if not bad else "; ".join(f"{k} differs at {len(v)} frames, first {v[:5]}" for k, v in bad.items())))
    print(f"stepping: {n / t_step:.0f} frames/s (one game, including the per-frame state reads)")
    # Savestate cost.
    t0 = time.time()
    h = pool.save([0] * 1)
    for _ in range(200):
        pool.load([0], h)
    t1 = time.time()
    hs = [pool.save([0])[0] for _ in range(200)]
    t2 = time.time()
    print(f"load {1e3 * (t1 - t0) / 200:.2f} ms, save {1e3 * (t2 - t1) / 200:.2f} ms, "
          f"{pool.save_bytes() / 1e6:.1f} MB for {pool.n_saves} savestates")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
