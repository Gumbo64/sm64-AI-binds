#!/usr/bin/env python3
"""Follows a reference run (sm64bb/refs.py) without savestates: one game,
played forward only, with the backbone on the GPU choosing its inputs.
Shown live in sm64-viewer.

    .venv/bin/python follow.py tas16 [--replan 8] [--ahead 16] [--name follow-tas16]

From power-on the reference's own boot inputs are played until Mario first
has control (menus, identical for every run). Then every `replan` frames:
- align: the reference state nearest by model distance (both directions),
  among those in the same level and area with Mario within 200 units, in a
  window from the current alignment onwards;
- act: inverse dynamics toward the reference state `ahead` frames past the
  alignment, and play its first `replan` inputs. In dialogs, menus and
  cutscenes (no control), the reference's own inputs at the alignment are
  played instead (the inverse head is not trained to press through text).
Stops at the reference's end, or when the alignment has not moved for
`--stuck` frames. Reports which milestones (stars, keys) were matched and
how many frames behind or ahead of the reference.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from sm64bb.livewrite import LiveRun
from sm64bb.refs import KEY_BIT, OBRRT, STAR_BIT, build_reference, make_pool
from sm64bb.runtime import Runtime

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN_R = 200.0


def encode_reference(ref, rt, pool, game, every=4):
    """z of the reference's state after every `every`-th frame (cached per
    checkpoint step): [T // every, d] on the GPU, and those frames."""
    path = os.path.join(HERE, "refs", f"{ref.name}.z{every}.step{rt.step}.pt")
    frames = np.arange(every, ref.T + 1, every)
    if os.path.exists(path):
        return torch.load(path).to(rt.dev), frames
    pool.reset([game])
    zs, xs = [], []
    f = 0
    for fr in frames:
        pool.step([game], ref.pads[f:fr][None, :])
        f = fr
        xs.append(pool.read([game]))
        if len(xs) == 512:
            zs.append(rt.encode({k: np.concatenate([d[k] for d in xs]) for k in xs[0]}).half())
            xs = []
    if xs:
        zs.append(rt.encode({k: np.concatenate([d[k] for d in xs]) for k in xs[0]}).half())
    z = torch.cat(zs)
    torch.save(z.cpu(), path)
    return z, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "v3_last.pt"))
    ap.add_argument("--replan", type=int, default=8)
    ap.add_argument("--ahead", type=int, default=16)
    ap.add_argument("--window", type=int, default=48, help="reference states (x4 frames) searched ahead when aligning")
    ap.add_argument("--stuck", type=int, default=900)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.name = a.name or f"follow-{a.ref}-{time.strftime('%m%d-%H%M')}"
    ref = build_reference(a.ref)
    rt = Runtime(a.ckpt)
    pool = make_pool(2)
    PLAY, REFG = 0, 1
    t0 = time.time()
    zr, rframes = encode_reference(ref, rt, pool, REFG)
    zr = zr.float()
    print(f"{a.ref}: {ref.T} frames, {len(ref.milestones)} milestones, encoded in {time.time() - t0:.0f} s", flush=True)
    # Boot: the reference's inputs until Mario has control.
    pool.reset([PLAY])
    f = 0
    while f < ref.T:
        pool.step([PLAY], ref.pads[f : f + 1][None, :])
        f += 1
        if pool.status([PLAY])["controllable"][0]:
            break
    live = LiveRun(os.path.join(OBRRT, "runs", a.name), ref.pads[:f])
    st = pool.status([PLAY])
    row = lambda s: {k: (v[0].tolist() if v.ndim > 1 else v[0].item()) for k, v in s.items()}
    node = live.add(-1, np.zeros(0, np.uint32), row(st), "start", f)
    j = f // 4 - 1  # alignment: index into rframes
    last_move, last_j = f, j
    chunk, matched, missed = [], [], []
    next_ms = 0
    ms = ref.milestones
    t_loop = time.time()
    state = "following"
    while True:
        x = pool.read([PLAY])
        z = rt.encode(x)
        s = pool.status([PLAY])
        # Align: the nearest reference state by position (same level and
        # area) in a window from the current alignment, with the model's
        # distance breaking near-ties; with none near enough, move on at the
        # reference's pace so the target keeps advancing along the route.
        lo, hi = max(0, j - 1), min(len(rframes), j + a.window)
        cand = np.arange(lo, hi)
        rf = rframes[cand] - 1
        dpos = np.linalg.norm(ref.pos[rf] - s["pos"][0], axis=1)
        ok = (dpos < ALIGN_R) & (ref.level[rf] == s["level"][0]) & (ref.area[rf] == s["area"][0])
        if ok.any():
            zc = zr[torch.as_tensor(cand, device=rt.dev)]
            zq = z.expand(len(zc), -1)
            dm = (rt.distance(zq, zc) + rt.distance(zc, zq)).cpu().numpy()
            score = np.where(ok, dpos + 2.0 * dm, np.inf)
            j = int(cand[int(np.argmin(score))])
        else:
            j = min(len(rframes) - 1, j + a.replan // 4)
        if j > last_j and ok.any():
            last_move, last_j = f, j
        # Act.
        if s["controllable"][0]:
            tgt = min(len(rframes) - 1, j + a.ahead // 4)
            pads = rt.inverse(z, zr[tgt : tgt + 1], a.ahead)[0, : a.replan]
            via = "inverse"
        else:
            r = int(rframes[j])
            pads = ref.pads[r : r + a.replan]
            if len(pads) < a.replan:
                pads = np.concatenate([pads, np.zeros(a.replan - len(pads), np.uint32)])
            j = min(len(rframes) - 1, j + a.replan // 4)  # cutscenes run on time
            via = "reference"
        ev, taken = pool.step([PLAY], pads[None, :])
        f += int(taken[0])
        chunk.append(pads)
        e = int(np.bitwise_or.reduce(ev[0]))
        if e & ((1 << STAR_BIT) | (1 << KEY_BIT)):
            s2 = pool.status([PLAY])
            # Which milestone: the reference's next one if it matches.
            while next_ms < len(ms) and ms[next_ms]["frame"] < rframes[j] - 600:
                missed.append(next_ms)
                next_ms += 1
            m = ms[next_ms] if next_ms < len(ms) else None
            good = m is not None and ((m["kind"] == "key" and s2["keys"][0] > s["keys"][0]) or
                                      (m["kind"] == "star" and m["course"] > 0 and s2["course_stars"][0][m["course"] - 1] >> m["bit"] & 1))
            rec = {"k": next_ms + 1, "kind": m["kind"] if m else "?", "ref_frame": m["frame"] if m else None, "frame": f,
                   "behind": f - m["frame"] if m else None, "matched": bool(good), "stars": int(s2["stars"][0]),
                   "keys": int(s2["keys"][0]), "level": int(s2["level"][0])}
            matched.append(rec)
            print(json.dumps(rec), flush=True)
            if good:
                next_ms += 1
        # One viewer node every 16 frames.
        if sum(len(c) for c in chunk) >= 16:
            node = live.add(node, np.concatenate(chunk), row(pool.status([PLAY])), via, f)
            chunk = []
        done = rframes[j] >= ref.T - 4
        stuck = f - last_move > a.stuck
        if done or stuck or f > 2 * ref.T:
            state = "reached the end" if done else f"stuck at reference frame {int(rframes[last_j])}"
        live.flush(force=bool(done or stuck), frames=f, frontier={"keys": int(s["keys"][0]), "stars": int(s["stars"][0])},
                   follow={"ref": a.ref, "milestones": sum(r["matched"] for r in matched), "of": len(ms), "frame": f,
                           "ref_frame": int(rframes[j]), "behind": int(f - rframes[j]), "state": state,
                           "events": matched, "fps": round(f / max(1e-6, time.time() - t_loop))},
                   best=node)
        if done or stuck or f > 2 * ref.T:
            break
    if chunk:
        node = live.add(node, np.concatenate(chunk), row(pool.status([PLAY])), via, f)
        live.flush(force=True)
    from sm64bb.refs import write_raw
    write_raw(os.path.join(live.path, "best.m64"), live.path_pads(node))
    print(json.dumps({"done": state, "frame": f, "ref_frame": int(rframes[last_j]),
                      "milestones_matched": sum(r["matched"] for r in matched), "of": len(ms),
                      "secs": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    main()
