#!/usr/bin/env python3
"""Races a reference run (sm64bb/refs.py) milestone by milestone with the
backbone on the GPU and savestates in sm64env, shown live in sm64-viewer.

    .venv/bin/python race.py tree14 [--seg-secs 120] [--from 1] [--to 14] [--name race-tree14]

Per segment (reference milestone k-1 -> k, a star or key):
- Start: our best state at milestone k-1 (at the start, the reference's first
  controllable state). Goal: the reference's state at milestone k, and the
  same star (course and bit) or key.
- Best-first search over savestates. A node is a state t frames into the
  segment, aligned to the reference at j frames (the reference state it is
  closest to by model distance). Children are 16-frame chunks: the
  reference's own next inputs from j; inverse dynamics toward the reference
  state 16, 24, 32 or 64 frames past j (aiming past what 16 frames reach
  is how it looks for shortcuts); a sampled inverse; and noisy copies.
  Priority = t + model frames to the goal. Children are stepped in parallel in
  sm64env and scored on the GPU in one batch.
- While our start is the reference's own state, the reference's chain is
  always expanded first, so each segment ends no later than the reference.
  If a segment started from our own (faster) state finds no way to the
  milestone, the race falls back to the reference's state there (keeping
  the route continuous from power-on; earlier savings are dropped).
Run directory: objective-rrt/runs/<name>/ (viewer: ?tree=<name>).
"""

import argparse
import heapq
import json
import os
import time

import numpy as np
import torch

from sm64bb.livewrite import LiveRun
from sm64bb.refs import KEY_BIT, OBRRT, STAR_BIT, build_reference, make_pool
from sm64bb.runtime import Runtime

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK = 16
DEAD = (1 << 0) | (1 << 14)  # death, lost life
MILESTONE = (1 << STAR_BIT) | (1 << KEY_BIT)


def status_row(s, i):
    return {k: (v[i].tolist() if v.ndim > 1 else v[i].item()) for k, v in s.items()}


def has_milestone(st, m, start_st):
    """Whether status `st` has milestone `m` that `start_st` didn't."""
    if m["kind"] == "key":
        return st["keys"] > start_st["keys"]
    c, b = m["course"], m["bit"]
    if c > 0:
        return bool(st["course_stars"][c - 1] >> b & 1) and not bool(start_st["course_stars"][c - 1] >> b & 1)
    if c == 0 and b >= 0:
        f = 1 << (24 + b)
        return bool(st["save_flags"] & f) and not bool(start_st["save_flags"] & f)
    return st["stars"] > start_st["stars"]


class Node:
    __slots__ = ("parent", "pads", "t", "j", "handle", "st", "via", "on_ref", "wid", "z")

    def __init__(self, parent, pads, t, j, handle, st, via, on_ref, z):
        self.parent, self.pads, self.t, self.j, self.handle = parent, pads, t, j, handle
        self.st, self.via, self.on_ref, self.wid, self.z = st, via, on_ref, None, z


class Racer:
    def __init__(self, a):
        self.a = a
        self.rng = np.random.default_rng(a.seed)
        self.ref = build_reference(a.ref)
        self.rt = Runtime(a.ckpt)
        self.P, self.C = a.parents, 8
        # Per popped parent: C child games plus one that rebuilds the parent
        # itself (children aren't saved until they are expanded).
        self.N = self.P * (self.C + 1)
        self.REF = self.N  # the game that plays the reference
        self.pool = make_pool(self.N + 1)
        self.nodes = []
        self.frames = 0
        self.t0 = time.time()

    # -- reference ----------------------------------------------------------
    def ref_play(self, upto):
        """Plays the reference game up to frame `upto` (absolute)."""
        f = int(self.pool.status([self.REF])["frame"][0])
        if upto > f:
            self.pool.step([self.REF], self.ref.pads[f:upto][None, :])
            self.frames += upto - f

    def first_control(self):
        self.pool.reset([self.REF])
        f = 0
        while f < self.ref.T:
            self.pool.step([self.REF], self.ref.pads[f : f + 1][None, :])
            f += 1
            if self.pool.status([self.REF])["controllable"][0]:
                return f
        raise RuntimeError("reference never gives control")

    def ref_segment(self, r0, r1, seeds):
        """Plays the reference game from r0 (where it is) to r1 and encodes
        its states every 4 frames. With `seeds`, also saves its state every
        CHUNK frames. Returns z [m, d], their frames, the seeds as
        (frame, handle, status), the handle of the state at r1 and its status."""
        frames = list(range(r0, r1, 4)) + [r1]
        xs, seed = [], []
        for f in frames:
            self.ref_play(f)
            xs.append(self.pool.read([self.REF]))
            if seeds and f < r1 and (f - r0) % CHUNK == 0 and f > r0:
                seed.append((f, self.pool.save([self.REF])[0], status_row(self.pool.status([self.REF]), 0)))
        x = {k: np.concatenate([d[k] for d in xs]) for k in xs[0]}
        z = self.rt.encode(x)
        h = self.pool.save([self.REF])[0]
        return z, np.array(frames), seed, h, status_row(self.pool.status([self.REF]), 0)

    # -- nodes ----------------------------------------------------------------
    def materialize(self, i):
        n = self.nodes[i]
        if n.wid is not None:
            return n.wid
        pw = -1 if n.parent < 0 else self.materialize(n.parent)
        n.wid = self.live.add(pw, n.pads, n.st, n.via, n.st["frame"])
        return n.wid

    def proposals(self, parents, zr, rframes, r0):
        """C chunks of CHUNK pads per parent and their labels."""
        P = len(parents)
        L = rframes[-1] - r0
        jidx = lambda j: np.clip((np.asarray(j) + 3) // 4, 0, len(rframes) - 1)
        zp = torch.stack([self.nodes[i].z for i in parents])
        js = np.array([self.nodes[i].j for i in parents])
        out = np.zeros((P, self.C, CHUNK), np.uint32)
        labels = [[""] * self.C for _ in range(P)]
        # (0) the reference's own next inputs from j.
        for p, j in enumerate(js):
            a = r0 + j
            seg = self.ref.pads[a : a + CHUNK]
            out[p, 0, : len(seg)] = seg
            out[p, 0, len(seg) :] = seg[-1] if len(seg) else 0
            labels[p][0] = "replay_ref"
        # (1-4) inverse dynamics toward the reference 16 / 24 / 32 / 64 frames past j.
        col = 1
        for ahead, k in [(16, 16), (24, 16), (32, 16), (64, 64)]:
            tgt = zr[torch.as_tensor(jidx(js + ahead), device=zp.device)]
            pads = self.rt.inverse(zp, tgt, k)
            out[:, col] = pads[:, :CHUNK]
            for p in range(P):
                labels[p][col] = f"inv{ahead}"
            col += 1
        # (5) sampled inverse toward +24.
        tgt = zr[torch.as_tensor(jidx(js + 24), device=zp.device)]
        out[:, 5] = self.rt.inverse(zp, tgt, 16, temperature=1.0, rng=self.rng)
        for p in range(P):
            labels[p][5] = "inv_s"
        # (6, 7) noisy copies of the reference chunk and of inv24.
        for c, src in [(6, 0), (7, 2)]:
            out[:, c] = self.jitter(out[:, src])
            for p in range(P):
                labels[p][c] = "noise_" + labels[p][src].replace("replay_", "")
        return out, labels

    def jitter(self, pads):
        p = pads.astype(np.int64)
        btn = (p >> 16) & 0xFFFF
        sx = ((p >> 8) & 0xFF).astype(np.int8).astype(np.float64)
        sy = (p & 0xFF).astype(np.int8).astype(np.float64)
        n = pads.shape[0]
        ang = self.rng.normal(0, 0.35, (n, 1))
        mag = self.rng.uniform(0.85, 1.15, (n, 1))
        c, s = np.cos(ang), np.sin(ang)
        nx = np.clip(np.round((sx * c - sy * s) * mag), -128, 127).astype(np.int64)
        ny = np.clip(np.round((sx * s + sy * c) * mag), -128, 127).astype(np.int64)
        flip = self.rng.random(btn.shape) < 0.03
        bits = 1 << self.rng.choice([15, 14, 13], btn.shape)  # A, B, Z
        btn = np.where(flip, btn ^ bits, btn)
        return ((btn << 16) | ((nx & 0xFF) << 8) | (ny & 0xFF)).astype(np.uint32)

    # -- one segment ------------------------------------------------------------
    BUCKET = 128  # reference frames per frontier bucket
    ALIGN_R = 200.0  # units: how far Mario may be from a reference state he is aligned to

    def push(self, node, prio=None):
        """Adds a node to the frontier. Priority: its lag behind the reference
        (frames used minus reference frames covered; negative = ahead), plus a
        small share of the model's rescaled frames-to-goal to break ties."""
        self.nodes.append(node)
        self.cnt += 1
        i = len(self.nodes) - 1
        heapq.heappush(self.buckets.setdefault(node.j // self.BUCKET, []), (prio, self.cnt, i))
        return i

    def pop_parents(self, best_t):
        """Up to P nodes, taking the best of each bucket in turn, so every
        stretch of the segment keeps being searched."""
        out = []
        while len(out) < self.P and any(self.buckets.values()):
            for b in sorted(self.buckets):
                h = self.buckets[b]
                while h:
                    _, _, i = heapq.heappop(h)
                    n = self.nodes[i]
                    if n.t + 1 >= best_t or (n.handle is None and self.nodes[n.parent].handle is None):
                        continue
                    out.append(i)
                    break
                if len(out) >= self.P:
                    break
        return out

    def frontier_size(self):
        return sum(len(h) for h in self.buckets.values())

    def segment(self, k, start, r0):
        """Searches from node `start` (aligned with reference frame r0) to
        milestone k. Returns (success node or None, its frames, handle of the
        reference's state at the milestone)."""
        a = self.a
        m = self.ref.milestones[k]
        r1 = m["frame"]
        L = r1 - r0
        sn = self.nodes[start]
        on_ref = sn.on_ref
        zr, rframes, seeds, ref_goal_h, _ = self.ref_segment(r0, r1, on_ref)
        zg = zr[-1:]
        start_st = sn.st
        sn.z = self.rt.encode(self.pool_read_node(start))[0]
        self.stats = {"dead": 0, "wrong_milestone": 0, "touched": 0, "min_lag": 10**9, "max_j_off_ref": 0}
        self.buckets, self.cnt = {}, 0
        self.seen = set()
        self.seg_first = len(self.nodes)
        self.buckets[sn.j // self.BUCKET] = [(0.0, 0, start)]
        # The reference's own states along the segment: all on a valid route
        # (while we are still on it), so the search can leave it anywhere.
        prev = start
        dg = np.zeros(0)
        if seeds:
            zs = zr[torch.as_tensor([(f - r0) // 4 for f, _, _ in seeds], device=zr.device)]
            dg = self.rt.distance(zs, zg.expand(len(seeds), -1)).cpu().numpy()
            for (f, h, st), z, d in zip(seeds, zs, dg):
                j = f - r0
                pads = self.ref.pads[r0 + self.nodes[prev].j : r0 + j]
                prev = self.push(Node(prev, pads, j, j, h, st, "reference", True, z), 0.0)
                self.seen.add(self.key(st, j))
        # Model frames-to-goal, rescaled to this segment: along the reference
        # the true remaining frames are known (the tree's labels are upper
        # bounds, so on slow routes the raw estimate is far too low).
        if seeds:
            true = np.array([L - (f - r0) for f, _, _ in seeds], np.float64)
            self.scale = float(np.clip(np.median(true / np.maximum(dg, 1.0)), 0.5, 20.0))
        else:
            self.scale = getattr(self, "scale", 1.0)
        best, best_t = None, (L if on_ref else 10**9)
        t_end = time.time() + a.seg_secs
        iters = 0
        t_it = time.time()
        while time.time() < t_end:
            iters += 1
            parents = self.pop_parents(best_t)
            if not parents:
                break
            props, labels = self.proposals(parents, zr, rframes, r0)
            # Games: for each parent, C children and (if not saved yet) one to
            # rebuild and save the parent. A parent without a savestate is
            # rebuilt from its own parent's plus its CHUNK inputs.
            ids, src, pads, meta = [], [], [], []
            rebuild = []
            for p, i in enumerate(parents):
                n = self.nodes[i]
                pre = None if n.handle is not None else n.pads
                h = n.handle if n.handle is not None else self.nodes[n.parent].handle
                for c in range(self.C):
                    ids.append(len(ids))
                    src.append(h)
                    pads.append(props[p, c] if pre is None else np.concatenate([pre, props[p, c]]))
                    meta.append((p, c, 0 if pre is None else len(pre)))
                if pre is not None:
                    rebuild.append((len(ids), i, h, pre))
                    ids.append(len(ids))
                    src.append(h)
                    pads.append(None)
                    meta.append((p, -1, len(pre)))
            T = max(len(x) for x in pads if x is not None)
            arr = np.zeros((len(ids), T), np.uint32)
            lens = np.zeros(len(ids), np.int64)
            for g, x in enumerate(pads):
                x = self.nodes[parents[meta[g][0]]].pads if x is None else x
                arr[g, : len(x)] = x
                lens[g] = len(x)
            self.pool.load(ids, src)
            # Pad shorter rows by repeating their last input; their result is
            # read at their own length (all rows of one call share T, so rows
            # are grouped by length).
            ev = np.zeros((len(ids), T), np.uint32)
            taken = np.zeros(len(ids), np.int64)
            for Lr in sorted(set(lens.tolist())):
                grp = [g for g in ids if lens[g] == Lr]
                e, tk = self.pool.step(grp, arr[grp, :Lr], stop_mask=MILESTONE | DEAD)
                ev[grp, :Lr] = e
                taken[grp] = tk
            self.frames += int(taken.sum())
            for g, i, h, pre in rebuild:
                self.nodes[i].handle = self.pool.save([g])[0]
            st = self.pool.status(ids)
            alive = []
            for g in ids:
                p, c, off = meta[g]
                if c < 0:
                    continue
                e = int(np.bitwise_or.reduce(ev[g, : taken[g]]))
                if e & DEAD:
                    self.stats["dead"] += 1
                    continue
                if e & MILESTONE:
                    row = status_row(st, g)
                    if not has_milestone(row, m, start_st):
                        self.stats["wrong_milestone"] += 1
                        continue  # another star or key: off the route
                    self.stats["touched"] += 1
                    par = self.nodes[parents[p]]
                    t = par.t + int(taken[g]) - off
                    if t < best_t:
                        h = self.pool.save([g])[0]
                        if best is not None:
                            self.pool.free([self.nodes[best].handle])
                        self.nodes.append(Node(parents[p], props[p, c][: int(taken[g]) - off], t, L, h, row, labels[p][c], False, None))
                        best, best_t = len(self.nodes) - 1, t
                        self.materialize(best)
                    continue
                alive.append(g)
            if alive:
                goal_pos = np.repeat(np.array([m["pos"]], np.float32), len(alive), 0)
                z = self.rt.encode(self.pool.read(alive, goal_pos))
                dg = self.rt.distance(z, zg.expand(len(alive), -1)).cpu().numpy()
                # Alignment: among the reference states in a window ahead of the
                # parent's (same level and area, Mario within ALIGN_R units),
                # the one nearest by model distance in both directions; with
                # none, the parent's. Grounding it in position keeps a
                # mistaken model distance from claiming a big lead.
                W = 80
                pj = np.array([self.nodes[parents[meta[g][0]]].j for g in alive])
                lo = np.clip(pj // 4 - 2, 0, max(0, len(rframes) - 1))
                idx_np = np.minimum(lo[:, None] + np.arange(W)[None], len(rframes) - 1)
                rf = rframes[idx_np] - 1
                cpos = st["pos"][alive][:, None, :]
                ok = (np.linalg.norm(self.ref.pos[rf] - cpos, axis=2) < self.ALIGN_R) \
                    & (self.ref.level[rf] == st["level"][alive][:, None]) & (self.ref.area[rf] == st["area"][alive][:, None])
                idx = torch.as_tensor(idx_np, device=z.device)
                zw = zr[idx]  # [A, W, d]
                zq = z[:, None].expand(-1, W, -1)
                d = (self.rt.distance(zq.reshape(-1, z.shape[1]), zw.reshape(-1, z.shape[1])) +
                     self.rt.distance(zw.reshape(-1, z.shape[1]), zq.reshape(-1, z.shape[1]))).view(len(alive), W)
                d = d.masked_fill(~torch.as_tensor(ok, device=d.device), float("inf"))
                jb = d.argmin(1).cpu().numpy()
                has = ok.any(1)
                jbest = np.where(has, idx_np[np.arange(len(alive)), jb], -1)
                for q, g in enumerate(alive):
                    p, c, _ = meta[g]
                    par = self.nodes[parents[p]]
                    t = par.t + CHUNK
                    row = status_row(st, g)
                    key = self.key(row, t)
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    j = int(rframes[jbest[q]] - r0) if jbest[q] >= 0 else par.j
                    self.stats["min_lag"] = min(self.stats["min_lag"], t - j)
                    self.stats["max_j_off_ref"] = max(self.stats["max_j_off_ref"], j)
                    rest = self.scale * float(dg[q])
                    nid = self.push(Node(parents[p], props[p, c], t, j, None, row, labels[p][c], False, z[q]),
                                    (t - j) + 0.05 * (rest - (L - j)))
                    if q < 2:
                        self.materialize(nid)
            for i in parents:
                self.materialize(i)
            cap = a.max_frontier // max(1, len(self.buckets))
            for b, h in self.buckets.items():
                if len(h) > 2 * cap:
                    h.sort()
                    del h[cap:]
            self.report(k, best_t, L, iters, self.frontier_size(), it_s=(time.time() - t_it) / iters)
        # Segment over: free every savestate but the winner's.
        for i in range(self.seg_first, len(self.nodes)):
            n = self.nodes[i]
            if n.handle is not None and i != best and i != start:
                self.pool.free([n.handle])
                n.handle = None
        return (best, best_t, ref_goal_h)

    def key(self, row, t):
        return (round(row["pos"][0] / 8), round(row["pos"][1] / 8), round(row["pos"][2] / 8), row["action"], t // 4)

    def pool_read_node(self, i):
        self.pool.load([0], [self.nodes[i].handle])
        return self.pool.read([0])

    def report(self, k, best_t, L, iters, frontier, force=False, it_s=None):
        m = self.ref.milestones[k]
        self.live.flush(force=force, frames=self.frames, frontier={"keys": m["keys"], "stars": m["stars"]},
                        race={"ref": self.a.ref, "milestones": self.results,
                              "current": {"milestone": k + 1, "ref_frames": L, "best_frames": int(best_t) if best_t < 10**8 else None,
                                          "iters": iters, "frontier": frontier, "secs_per_iter": it_s}},
                        best=self.nodes[self.best_node].wid if self.best_node is not None else None,
                        ref_frame=self.ref_total, best_frame=self.our_total)

    # -- the race -----------------------------------------------------------------
    def run(self):
        a = self.a
        ms = self.ref.milestones
        if a.from_ > 1:
            # Start on the reference's own state at the previous milestone.
            f0 = ms[a.from_ - 2]["frame"]
            self.pool.reset([self.REF])
            self.ref_play(f0)
        else:
            f0 = self.first_control()
        self.live = LiveRun(os.path.join(OBRRT, "runs", a.name), self.ref.pads[:f0])
        self.results = []
        self.best_node = None
        st0 = status_row(self.pool.status([self.REF]), 0)
        h0 = self.pool.save([self.REF])[0]
        self.nodes.append(Node(-1, np.zeros(0, np.uint32), 0, 0, h0, st0, "start", True, None))
        self.materialize(0)
        start, r0 = 0, f0
        self.ref_total, self.our_total = f0, f0
        last = min(len(ms), a.to)
        for k in range(a.from_ - 1, last):
            t_seg = time.time()
            best, t, ref_h = self.segment(k, start, r0)
            L = ms[k]["frame"] - r0
            if best is None:
                # Back to the reference's own route at this milestone.
                pads = self.ref.pads[f0 : ms[k]["frame"]]
                node = Node(0, pads, L, L, ref_h, status_row(self.pool.status([self.REF]), 0), "ref_fallback", True, None)
                self.nodes.append(node)
                best, t = len(self.nodes) - 1, L
                self.materialize(best)
                self.our_total = ms[k]["frame"]
                fell = True
            else:
                self.pool.free([ref_h])
                self.our_total += t
                fell = False
            self.ref_total = ms[k]["frame"]
            self.best_node = best
            self.results.append({"k": k + 1, "kind": ms[k]["kind"], "course": ms[k]["course"], "bit": ms[k]["bit"],
                                 "level": ms[k]["level"], "ref_frame": ms[k]["frame"], "frame": int(self.our_total),
                                 "segment_ref": int(L), "segment": int(t), "saved": int(L - t),
                                 "total_saved": int(self.ref_total - self.our_total), "fallback": fell,
                                 "secs": round(time.time() - t_seg, 1), **self.stats})
            print(json.dumps(self.results[-1]), flush=True)
            # The next segment starts on the reference's own state only if
            # nothing was saved so far (then its chain is the safe fallback).
            self.nodes[best].on_ref = self.our_total == self.ref_total
            start, r0 = best, ms[k]["frame"]
            self.report(k, t, L, 0, 0, force=True)
        # Final route as a replay next to the run.
        from sm64bb.refs import write_raw
        write_raw(os.path.join(OBRRT, "runs", a.name, "best.m64"), self.live.path_pads(self.nodes[self.best_node].wid))
        self.report(last - 1, 0, 0, 0, 0, force=True)
        print(f"done: {len(self.results)} milestones, reference {self.ref_total} frames, ours {self.our_total} "
              f"({self.ref_total - self.our_total:+d}), {self.frames} frames emulated, {time.time() - self.t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "v3_last.pt"))
    ap.add_argument("--seg-secs", type=float, default=120.0)
    ap.add_argument("--from", dest="from_", type=int, default=1)
    ap.add_argument("--to", type=int, default=10**6)
    ap.add_argument("--parents", type=int, default=32)
    ap.add_argument("--max-frontier", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.name = a.name or f"race-{a.ref}-{time.strftime('%m%d-%H%M')}"
    Racer(a).run()


if __name__ == "__main__":
    main()
